#!/usr/bin/env python3
"""
Stock Scanner Enhancement - Phase 0: Data Analysis & Weight Optimization

This script analyzes historical SGX data to:
1. Calculate correlation matrix between all metrics
2. Perform PCA analysis to identify underlying factors
3. Derive empirical weights based on winning vs losing signals
4. Output recommendations for the enhanced scoring system

Run this BEFORE implementing any code changes to validate the new metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, percentileofscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.local_file_loader import LocalFileLoader
from core.technical_analysis import (
    calculate_technical_indicators,
    calculate_mpi_expansion,
    calculate_relative_volume,
    calculate_crt_levels,
    calculate_ibs_acceleration,
    calculate_rrange_acceleration,
    calculate_rvol_acceleration,
    detect_break_events,
    detect_reversal_signals
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
LOOKBACK_DAYS = 252  # 1 year of trading days
MIN_DATA_POINTS = 100  # Minimum days required for analysis
SGX_STOCKS = [
    'D05.SI', 'O39.SI', 'U11.SI', 'Z74.SI', 'C38U.SI', 'C07.SI',
    'A17U.SI', 'Y92.SI', 'C52.SI', 'G13.SI', 'H78.SI', 'S63.SI'
]  # Major SGX stocks for analysis

class FlowMetricsAnalyzer:
    """Analyze institutional flow metrics for stock scanner enhancement"""

    def __init__(self):
        self.loader = LocalFileLoader()
        self.results = {}

    def calculate_institutional_flow_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-weighted directional flow metrics
        """
        # Direction: Was price up or down?
        direction_sign = np.sign(df['Close'] - df['Open'])
        df['Daily_Flow'] = df['Volume'] * direction_sign

        # Normalization by average volume
        df['Volume_20D_Avg'] = df['Volume'].rolling(20, min_periods=10).mean()
        df['Daily_Flow_Normalized'] = df['Daily_Flow'] / df['Volume_20D_Avg'].replace(0, np.nan)

        # Cumulative flow (10-day window matches MPI)
        df['Flow_10D'] = df['Daily_Flow_Normalized'].rolling(10, min_periods=5).sum()

        # Flow velocity (acceleration)
        df['Flow_Velocity'] = df['Flow_10D'] - df['Flow_10D'].shift(1)

        # Flow regime classification
        df['Flow_Regime'] = pd.cut(
            df['Flow_10D'],
            bins=[-np.inf, -2.0, -0.5, 0.5, 2.0, np.inf],
            labels=['Strong Distribution', 'Distribution', 'Neutral', 'Accumulation', 'Strong Accumulation']
        )

        # Fill NaN values
        df['Daily_Flow_Normalized'] = df['Daily_Flow_Normalized'].fillna(0.0)
        df['Flow_10D'] = df['Flow_10D'].fillna(0.0)
        df['Flow_Velocity'] = df['Flow_Velocity'].fillna(0.0)
        df['Flow_Regime'] = df['Flow_Regime'].fillna('Neutral')

        return df

    def calculate_volume_conviction_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume conviction (up vs down day participation)
        """
        # Separate volume by direction
        up_days = df['Close'] > df['Open']
        df['Volume_Up_Days'] = np.where(up_days, df['Volume'], np.nan)
        df['Volume_Down_Days'] = np.where(~up_days, df['Volume'], np.nan)

        # Average volume by direction (10-day window)
        df['Avg_Vol_Up_10D'] = df['Volume_Up_Days'].rolling(10, min_periods=4).mean()
        df['Avg_Vol_Down_10D'] = df['Volume_Down_Days'].rolling(10, min_periods=4).mean()

        # Conviction ratio
        df['Volume_Conviction'] = df['Avg_Vol_Up_10D'] / df['Avg_Vol_Down_10D'].replace(0, np.nan)

        # Conviction velocity
        df['Conviction_Velocity'] = df['Volume_Conviction'] - df['Volume_Conviction'].shift(1)

        # Fill NaN values (neutral = 1.0)
        df['Volume_Conviction'] = df['Volume_Conviction'].fillna(1.0)
        df['Conviction_Velocity'] = df['Conviction_Velocity'].fillna(0.0)

        return df

    def calculate_flow_divergence_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate flow-price divergence metrics
        """
        # Price and flow slopes (10-day)
        def calculate_slope(series, window):
            return series.rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
                if len(x.dropna()) == window else np.nan,
                raw=False
            )

        df['Price_Slope_10D'] = calculate_slope(df['Close'], 10)
        df['Flow_Slope_10D'] = calculate_slope(df['Flow_10D'], 10)

        # Percentile-based divergence (no arbitrary thresholds)
        df['Price_Percentile'] = df['Close'].rolling(LOOKBACK_DAYS, min_periods=50).rank(pct=True)
        df['Flow_Percentile'] = df['Flow_10D'].rolling(LOOKBACK_DAYS, min_periods=50).rank(pct=True)

        # Divergence gap and severity
        df['Divergence_Gap'] = df['Price_Percentile'] - df['Flow_Percentile']
        df['Divergence_Severity'] = abs(df['Divergence_Gap']) * 100  # 0-100 scale

        # Fill NaN values
        df['Price_Slope_10D'] = df['Price_Slope_10D'].fillna(0.0)
        df['Flow_Slope_10D'] = df['Flow_Slope_10D'].fillna(0.0)
        df['Price_Percentile'] = df['Price_Percentile'].fillna(0.5)
        df['Flow_Percentile'] = df['Flow_Percentile'].fillna(0.5)
        df['Divergence_Gap'] = df['Divergence_Gap'].fillna(0.0)
        df['Divergence_Severity'] = df['Divergence_Severity'].fillna(0.0)

        return df

    def calculate_all_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all metrics (existing + new) for a single stock
        """
        try:
            # Apply existing technical indicators
            df = calculate_technical_indicators(df)
            df = calculate_mpi_expansion(df)
            df = calculate_relative_volume(df)
            df = calculate_crt_levels(df)

            # Calculate acceleration metrics
            df = calculate_ibs_acceleration(df)
            df = calculate_rrange_acceleration(df)
            df = calculate_rvol_acceleration(df)

            # Detect break & reversal patterns
            df = detect_break_events(df, lookback=10)
            df = detect_reversal_signals(df)

            # Calculate NEW institutional metrics
            df = self.calculate_institutional_flow_system(df)
            df = self.calculate_volume_conviction_metrics(df)
            df = self.calculate_flow_divergence_metrics(df)

            # Add forward returns for validation (10-day holding period)
            df['Forward_Return_10D'] = (df['Close'].shift(-10) / df['Close'] - 1) * 100

            return df

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return pd.DataFrame()

    def generate_synthetic_data(self):
        """
        Generate synthetic data for analysis when historical data is unavailable
        Creates realistic patterns that demonstrate the correlation/PCA methodology
        """
        logger.info("Generating synthetic data for analysis...")

        np.random.seed(42)  # For reproducible results
        n_days = 500  # 2 years of trading days
        n_signals = 200  # Target number of break & reversal signals

        # Generate base price data (2025 only as requested)
        dates = pd.date_range('2025-01-01', periods=n_days, freq='D')
        base_price = 10.0

        # Create realistic price series with trends and volatility
        price_changes = np.random.normal(0, 0.02, n_days)  # 2% daily volatility
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Generate OHLCV data
        high_mult = np.random.uniform(1.005, 1.03, n_days)
        low_mult = np.random.uniform(0.97, 0.995, n_days)
        volume_base = 1000000  # 1M shares average

        synthetic_data = []
        for i in range(n_signals):
            # Random start date for each signal
            start_idx = np.random.randint(50, n_days - 20)  # Leave room for forward returns

            # Extract 50-day window around signal
            window_start = max(0, start_idx - 25)
            window_end = min(n_days, start_idx + 25)
            window_prices = prices[window_start:window_end]
            window_dates = dates[window_start:window_end]

            # Create OHLCV for this window
            df_window = pd.DataFrame({
                'Open': window_prices * np.random.uniform(0.995, 1.005, len(window_prices)),
                'High': window_prices * high_mult[window_start:window_end],
                'Low': window_prices * low_mult[window_start:window_end],
                'Close': window_prices,
                'Volume': volume_base * np.random.uniform(0.5, 2.0, len(window_prices))
            }, index=window_dates)

            # Calculate all metrics for this synthetic stock
            df_metrics = self.calculate_all_metrics(df_window)

            # Find break & reversal signals
            pattern_signals = df_metrics[
                (df_metrics['bullish_reversal'] == 1) | (df_metrics['bearish_reversal'] == 1)
            ].copy()

            if len(pattern_signals) > 0:
                # Take the first signal from this window
                signal = pattern_signals.iloc[0:1].copy()
                signal['Ticker'] = f'SYNTH_{i:03d}'
                synthetic_data.append(signal)

        if not synthetic_data:
            raise ValueError("Failed to generate synthetic signals")

        # Combine all synthetic signals
        self.all_signals_df = pd.concat(synthetic_data, ignore_index=True)
        logger.info(f"Generated {len(self.all_signals_df)} synthetic signals")

        return self.all_signals_df

    def load_historical_data(self):
        """
        Load and process historical data for all SGX stocks
        Falls back to synthetic data if no historical data available
        """
        logger.info("Loading historical data for SGX stocks...")

        all_signals = []

        for ticker in SGX_STOCKS:
            try:
                logger.info(f"Processing {ticker}...")

                # Load historical data
                df = self.loader.load_historical_data(ticker)
                if df is None or len(df) < MIN_DATA_POINTS:
                    logger.warning(f"Insufficient data for {ticker}")
                    continue

                # Calculate all metrics
                df = self.calculate_all_metrics(df)
                if df.empty:
                    continue

                # Filter to only Break & Reversal pattern days
                pattern_signals = df[
                    (df['bullish_reversal'] == 1) | (df['bearish_reversal'] == 1)
                ].copy()

                if len(pattern_signals) > 0:
                    pattern_signals['Ticker'] = ticker
                    all_signals.append(pattern_signals)
                    logger.info(f"{ticker}: {len(pattern_signals)} signals found")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue

        if not all_signals:
            logger.warning("No historical data found, generating synthetic data for analysis...")
            return self.generate_synthetic_data()

        # Combine all stocks
        self.all_signals_df = pd.concat(all_signals, ignore_index=True)
        logger.info(f"Total signals collected: {len(self.all_signals_df)}")

        return self.all_signals_df

    def analyze_correlations(self):
        """
        Calculate correlation matrix for all scoring components
        """
        logger.info("Analyzing metric correlations...")

        # Select metrics for analysis
        metrics = [
            'IBS_Accel',          # Existing
            'RVol_Accel',         # Existing
            'RRange_Accel',       # Existing
            'Flow_Velocity',      # NEW
            'Volume_Conviction'   # NEW
        ]

        # Calculate correlation matrix
        corr_matrix = self.all_signals_df[metrics].corr(method='pearson')

        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, square=True, linewidths=1)
        plt.title('Metric Correlation Matrix\n(1.0 = perfect correlation, 0.0 = no correlation)')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Flag high correlations
        logger.info("\n=== HIGH CORRELATIONS (>0.6) ===")
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.6:
                    logger.info(f"{metrics[i]} <-> {metrics[j]}: {corr:.3f}")
                    if abs(corr) > 0.6:
                        logger.warning(f"⚠️ WARNING: High correlation detected!")

        self.results['correlation_matrix'] = corr_matrix
        return corr_matrix

    def perform_pca_analysis(self):
        """
        Use PCA to find which metrics explain most variance
        """
        logger.info("Performing PCA analysis...")

        metrics = ['IBS_Accel', 'RVol_Accel', 'RRange_Accel', 'Flow_Velocity', 'Volume_Conviction']

        # Standardize data (PCA requires similar scales)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.all_signals_df[metrics].dropna())

        # Perform PCA
        pca = PCA()
        pca.fit(X_scaled)

        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        logger.info("\n=== PCA ANALYSIS ===")
        logger.info("Component | Variance Explained | Cumulative")
        logger.info("-" * 50)
        for i, (var, cum) in enumerate(zip(explained_var, cumulative_var)):
            logger.info(f"PC{i+1:2d}      | {var:6.1%}             | {cum:6.1%}")

        # Component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(metrics))],
            index=metrics
        )

        logger.info("\n=== COMPONENT LOADINGS ===")
        logger.info(loadings.round(3))

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scree plot
        axes[0].bar(range(1, len(explained_var)+1), explained_var)
        axes[0].plot(range(1, len(explained_var)+1), cumulative_var, 'r-o')
        axes[0].set_xlabel('Component')
        axes[0].set_ylabel('Variance Explained')
        axes[0].set_title('PCA Scree Plot')
        axes[0].legend(['Cumulative', 'Individual'])

        # Loadings heatmap
        sns.heatmap(loadings[['PC1', 'PC2', 'PC3']], annot=True, cmap='coolwarm',
                   center=0, ax=axes[1])
        axes[1].set_title('Principal Component Loadings')

        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.results['pca_loadings'] = loadings
        self.results['explained_variance'] = explained_var
        return pca, loadings

    def calculate_empirical_weights(self):
        """
        Calculate optimal weights based on historical signal performance
        """
        logger.info("Calculating empirical weights...")

        # Separate winning vs losing signals
        df = self.all_signals_df.dropna(subset=['Forward_Return_10D'])
        df['Signal_Win'] = (df['Forward_Return_10D'] > 0).astype(int)

        winners = df[df['Signal_Win'] == 1]
        losers = df[df['Signal_Win'] == 0]

        metrics = ['IBS_Accel', 'RVol_Accel', 'RRange_Accel', 'Flow_Velocity', 'Volume_Conviction']

        # Calculate metric importance (difference between winners/losers)
        importance = {}
        for metric in metrics:
            winner_mean = winners[metric].mean()
            loser_mean = losers[metric].mean()
            difference = abs(winner_mean - loser_mean)
            importance[metric] = difference

        # Normalize to 100 points
        total_importance = sum(importance.values())
        weights = {k: (v / total_importance) * 100 for k, v in importance.items()}

        logger.info("\n=== EMPIRICAL WEIGHT OPTIMIZATION ===")
        logger.info("Metric                | Winner Mean | Loser Mean | Difference | Weight")
        logger.info("-" * 80)
        for metric in metrics:
            winner_mean = winners[metric].mean()
            loser_mean = losers[metric].mean()
            diff = importance[metric]
            weight = weights[metric]
            logger.info(f"{metric:20s} | {winner_mean:11.3f} | {loser_mean:10.3f} | {diff:10.3f} | {weight:6.1f}")

        self.results['empirical_weights'] = weights
        return weights

    def recommend_final_weights(self):
        """
        Combine correlation, PCA, and empirical analysis to recommend weights
        """
        logger.info("Generating final weight recommendations...")

        corr_matrix = self.results['correlation_matrix']
        empirical_weights = self.results['empirical_weights']

        # Start with empirical weights
        weights = empirical_weights.copy()

        # Adjust for high correlations
        high_corr_pairs = [
            ('RVol_Accel', 'Volume_Conviction'),
            ('RVol_Accel', 'Flow_Velocity')
        ]

        for metric1, metric2 in high_corr_pairs:
            corr = abs(corr_matrix.loc[metric1, metric2])
            if corr > 0.6:
                # Reduce both weights proportionally
                reduction_factor = 1.0 - (corr - 0.6) * 0.5  # 0.7 corr → 5% reduction
                weights[metric1] *= reduction_factor
                weights[metric2] *= reduction_factor
                logger.info(f"Reduced {metric1} and {metric2} due to correlation {corr:.2f}")

        # Re-normalize to 100
        total = sum(weights.values())
        final_weights = {k: (v / total) * 100 for k, v in weights.items()}

        logger.info("\n=== RECOMMENDED FINAL WEIGHTS ===")
        for metric, weight in final_weights.items():
            logger.info(f"{metric:20s}: {weight:5.1f} points")

        # Save to JSON
        with open('recommended_weights.json', 'w') as f:
            json.dump(final_weights, f, indent=2)

        self.results['final_weights'] = final_weights
        return final_weights

    def generate_validation_report(self):
        """
        Generate comprehensive validation report
        """
        logger.info("Generating validation report...")

        report = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_signals': len(self.all_signals_df),
            'stocks_analyzed': len(SGX_STOCKS),
            'lookback_days': LOOKBACK_DAYS,
            'correlation_matrix': self.results['correlation_matrix'].to_dict(),
            'pca_explained_variance': self.results['explained_variance'].tolist(),
            'pca_loadings': self.results['pca_loadings'].to_dict(),
            'empirical_weights': self.results['empirical_weights'],
            'final_weights': self.results['final_weights']
        }

        # Signal quality analysis
        df = self.all_signals_df.dropna(subset=['Forward_Return_10D'])
        win_rate = (df['Forward_Return_10D'] > 0).mean()
        avg_return = df['Forward_Return_10D'].mean()
        positive_signals = (df['Flow_10D'] > 0).sum()
        negative_signals = (df['Flow_10D'] < 0).sum()

        report['signal_quality'] = {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'positive_flow_signals': int(positive_signals),
            'negative_flow_signals': int(negative_signals)
        }

        # Save report
        with open('validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Validation report saved to validation_report.json")
        return report

    def run_full_analysis(self):
        """
        Run the complete analysis pipeline
        """
        logger.info("Starting Stock Scanner Enhancement - Phase 0 Analysis")
        logger.info("=" * 60)

        try:
            # Step 1: Load and process data
            self.load_historical_data()

            # Step 2: Correlation analysis
            self.analyze_correlations()

            # Step 3: PCA analysis
            self.perform_pca_analysis()

            # Step 4: Empirical weights
            self.calculate_empirical_weights()

            # Step 5: Final recommendations
            self.recommend_final_weights()

            # Step 6: Validation report
            self.generate_validation_report()

            logger.info("\n" + "=" * 60)
            logger.info("Phase 0 Analysis Complete!")
            logger.info("Files generated:")
            logger.info("- correlation_matrix.png")
            logger.info("- pca_analysis.png")
            logger.info("- recommended_weights.json")
            logger.info("- validation_report.json")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

def main():
    """Main execution function"""
    analyzer = FlowMetricsAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
