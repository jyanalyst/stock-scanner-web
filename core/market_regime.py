# File: core/market_regime.py
"""
Market Regime Detection using Gaussian Mixture Models
Simple and robust implementation for identifying market volatility regimes
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    GMM-based market regime detection focusing on volatility clustering
    Identifies low and high volatility regimes in stock price data
    """
    
    def __init__(self, n_regimes: int = 2, lookback_days: int = 20):
        """
        Initialize regime detector
        
        Args:
            n_regimes: Number of market regimes (default=2 for low/high volatility)
            lookback_days: Days for rolling volatility calculation
        """
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self.gmm = None
        self.regime_params = {}
        self.scaler = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for regime detection with better error handling
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime detection features
        """
        features = pd.DataFrame(index=df.index)
        
        # Calculate returns
        features['returns'] = df['Close'].pct_change()
        
        # Rolling volatility (standard deviation of returns)
        features['volatility'] = features['returns'].rolling(
            window=self.lookback_days,
            min_periods=5  # Allow calculation with at least 5 days
        ).std()
        
        # Also calculate a shorter-term volatility for better regime detection
        features['volatility_short'] = features['returns'].rolling(
            window=5,
            min_periods=3
        ).std()
        
        # Volume-weighted volatility (optional)
        if 'Volume' in df.columns:
            volume_norm = df['Volume'] / df['Volume'].rolling(
                window=self.lookback_days,
                min_periods=5
            ).mean()
            features['volume_weighted_vol'] = features['volatility'] * volume_norm
        
        # Clean up NaN values
        features = features.dropna()
        
        # Log data quality
        logger.info(f"Prepared {len(features)} feature rows from {len(df)} price rows")
        
        return features
    
    def fit(self, df: pd.DataFrame) -> 'MarketRegimeDetector':
        """
        Fit GMM model to identify market regimes with better error handling
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Self for method chaining
        """
        try:
            # Prepare features
            features = self.prepare_features(df)
            
            if len(features) < 30:  # Need at least 30 days
                logger.warning(f"Insufficient data for regime detection: {len(features)} rows")
                self._set_default_regimes()
                return self
            
            # Select features for GMM - use both returns and volatility
            X = features[['returns', 'volatility']].values
            
            # Remove any remaining NaN or infinite values
            mask = np.isfinite(X).all(axis=1)
            X = X[mask]
            
            if len(X) < 30:
                logger.warning(f"Insufficient valid data after cleaning: {len(X)} rows")
                self._set_default_regimes()
                return self
            
            # Standardize features for better GMM performance
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit GMM with better initialization
            self.gmm = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                n_init=10,
                init_params='k-means++'  # Better initialization
            )
            
            self.gmm.fit(X_scaled)
            
            # Get regime parameters in original scale
            means_original = self.scaler.inverse_transform(self.gmm.means_)
            
            # Store regime parameters and ensure proper labeling
            volatilities = []
            for i in range(self.n_regimes):
                self.regime_params[i] = {
                    'mean_return': means_original[i][0],
                    'mean_volatility': means_original[i][1],
                    'label': ''  # Will be set below
                }
                volatilities.append(means_original[i][1])
            
            # Label regimes based on volatility
            sorted_indices = np.argsort(volatilities)
            if self.n_regimes == 2:
                self.regime_params[sorted_indices[0]]['label'] = 'Low Volatility'
                self.regime_params[sorted_indices[1]]['label'] = 'High Volatility'
            else:
                for idx, regime_idx in enumerate(sorted_indices):
                    self.regime_params[regime_idx]['label'] = f'Regime {idx}'
            
            logger.info(f"Successfully fitted GMM with {self.n_regimes} regimes")
            for i, params in self.regime_params.items():
                logger.info(f"Regime {i} ({params['label']}): vol={params['mean_volatility']:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting GMM: {e}")
            self._set_default_regimes()
            return self
    
    def _set_default_regimes(self):
        """Set default regime parameters when GMM fails"""
        self.regime_params = {
            0: {'mean_return': 0.0, 'mean_volatility': 0.01, 'label': 'Low Volatility'},
            1: {'mean_return': 0.0, 'mean_volatility': 0.02, 'label': 'High Volatility'}
        }
    
    def _classify_regime(self, volatility: float) -> str:
        """Classify regime based on volatility level"""
        if self.n_regimes == 2:
            # Simple low/high classification
            all_vols = [params['mean_volatility'] for params in self.regime_params.values()]
            if not all_vols or volatility <= np.median(all_vols + [volatility]):
                return "Low Volatility"
            else:
                return "High Volatility"
        else:
            # For more regimes, use percentiles
            return f"Regime {len(self.regime_params)}"
    
    def predict_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict market regime with fallback for edge cases
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime predictions
        """
        try:
            if self.gmm is None:
                logger.warning("Model not fitted, using simple volatility-based classification")
                return self._simple_regime_classification(df)
            
            # Prepare features
            features = self.prepare_features(df)
            
            if len(features) == 0:
                return pd.DataFrame()
            
            X = features[['returns', 'volatility']].values
            
            # Remove invalid values
            mask = np.isfinite(X).all(axis=1)
            X_clean = X[mask]
            features_clean = features[mask]
            
            if len(X_clean) == 0:
                return self._simple_regime_classification(df)
            
            # Scale features if scaler exists
            if hasattr(self, 'scaler') and self.scaler is not None:
                X_scaled = self.scaler.transform(X_clean)
            else:
                X_scaled = X_clean
            
            # Predict regimes
            regimes = self.gmm.predict(X_scaled)
            regime_probs = self.gmm.predict_proba(X_scaled)
            
            # Create results DataFrame
            results = pd.DataFrame(index=features_clean.index)
            results['regime'] = regimes
            results['regime_label'] = [
                self.regime_params.get(r, {}).get('label', 'Unknown') for r in regimes
            ]
            
            # Add probability columns
            for i in range(self.n_regimes):
                results[f'prob_regime_{i}'] = regime_probs[:, i]
            
            # Add the original features
            results['returns'] = features_clean['returns']
            results['volatility'] = features_clean['volatility']
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting regimes: {e}")
            return self._simple_regime_classification(df)
    
    def _simple_regime_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple volatility-based regime classification as fallback"""
        features = self.prepare_features(df)
        
        if len(features) == 0:
            return pd.DataFrame()
        
        # Use median volatility as threshold
        median_vol = features['volatility'].median()
        
        results = pd.DataFrame(index=features.index)
        results['regime'] = (features['volatility'] > median_vol).astype(int)
        results['regime_label'] = results['regime'].map({0: 'Low Volatility', 1: 'High Volatility'})
        results['prob_regime_0'] = 1 - results['regime']
        results['prob_regime_1'] = results['regime']
        results['returns'] = features['returns']
        results['volatility'] = features['volatility']
        
        return results
    
    def get_current_regime(self, df: pd.DataFrame) -> Dict:
        """
        Get the current market regime
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with current regime info
        """
        regimes = self.predict_regime(df)
        
        if len(regimes) == 0:
            return {'regime': 'Unknown', 'probability': 0.0}
        
        current = regimes.iloc[-1]
        current_regime_idx = int(current['regime'])
        
        return {
            'regime': current['regime_label'],
            'regime_index': current_regime_idx,
            'probability': current[f'prob_regime_{current_regime_idx}'],
            'volatility': current['volatility'],
            'returns': current['returns']
        }

def calculate_regime_statistics(df: pd.DataFrame, regime_data: pd.DataFrame) -> Dict:
    """
    Calculate statistics for each regime
    
    Args:
        df: Original price data
        regime_data: DataFrame with regime predictions
        
    Returns:
        Dict with regime statistics
    """
    stats = {}
    
    for regime_idx in regime_data['regime'].unique():
        regime_mask = regime_data['regime'] == regime_idx
        regime_dates = regime_data[regime_mask].index
        
        # Get price data for this regime
        regime_prices = df.loc[regime_dates]
        
        if len(regime_prices) > 0:
            returns = regime_prices['Close'].pct_change().dropna()
            
            stats[int(regime_idx)] = {
                'label': regime_data[regime_mask]['regime_label'].iloc[0],
                'count': len(regime_dates),
                'percentage': len(regime_dates) / len(regime_data) * 100,
                'avg_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'max_return': returns.max(),
                'min_return': returns.min()
            }
    
    return stats

@st.cache_data(ttl=300)
def detect_market_regimes_cached(ticker: str, df: pd.DataFrame, n_regimes: int = 2) -> Tuple[pd.DataFrame, Dict]:
    """
    Cached function for market regime detection
    
    Args:
        ticker: Stock ticker
        df: DataFrame with OHLCV data
        n_regimes: Number of regimes to detect
        
    Returns:
        Tuple of (regime_data, regime_stats)
    """
    try:
        detector = MarketRegimeDetector(n_regimes=n_regimes)
        detector.fit(df)
        regime_data = detector.predict_regime(df)
        regime_stats = calculate_regime_statistics(df, regime_data)
        
        return regime_data, regime_stats
        
    except Exception as e:
        logger.error(f"Error detecting regimes for {ticker}: {e}")
        return pd.DataFrame(), {}

def plot_regime_visualization(df: pd.DataFrame, regime_data: pd.DataFrame, ticker: str):
    """
    Create visualization of market regimes
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{ticker} Price with Market Regimes', 'Rolling Volatility'),
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Define colors for regimes
    regime_colors = {
        'Low Volatility': 'lightgreen',
        'High Volatility': 'lightcoral'
    }
    
    # Plot price with regime backgrounds
    unique_regimes = regime_data['regime_label'].unique()
    
    for regime in unique_regimes:
        regime_mask = regime_data['regime_label'] == regime
        regime_periods = regime_data[regime_mask]
        
        # Add shaded regions for each continuous regime period
        if len(regime_periods) > 0:
            # Find continuous periods
            regime_groups = []
            current_group = [regime_periods.index[0]]
            
            for i in range(1, len(regime_periods)):
                if (regime_periods.index[i] - regime_periods.index[i-1]).days <= 5:
                    current_group.append(regime_periods.index[i])
                else:
                    regime_groups.append(current_group)
                    current_group = [regime_periods.index[i]]
            regime_groups.append(current_group)
            
            # Add shaded regions
            for group in regime_groups:
                if len(group) > 0:
                    fig.add_vrect(
                        x0=group[0], x1=group[-1],
                        fillcolor=regime_colors.get(regime, 'gray'),
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        row=1, col=1
                    )
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add volatility
    fig.add_trace(
        go.Scatter(
            x=regime_data.index,
            y=regime_data['volatility'],
            mode='lines',
            name='Volatility',
            line=dict(color='orange', width=1)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Market Regime Analysis",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title="Date", row=2, col=1)
    fig.update_yaxes(title="Price", row=1, col=1)
    fig.update_yaxes(title="Volatility", row=2, col=1)
    
    return fig

# Integration helpers for the scanner
def add_regime_to_scanner_results(results_df: pd.DataFrame, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Add market regime information to scanner results
    
    Args:
        results_df: Current scanner results
        stock_data: Dictionary of stock DataFrames
        
    Returns:
        Enhanced results DataFrame with regime info
    """
    if results_df.empty:
        return results_df
    
    # Add regime columns
    results_df['Market_Regime'] = 'Unknown'
    results_df['Regime_Probability'] = 0.0
    results_df['Regime_Volatility'] = 0.0
    
    for ticker in results_df['Ticker'].unique():
        if ticker in stock_data:
            df = stock_data[ticker]
            
            try:
                # Detect regimes
                detector = MarketRegimeDetector(n_regimes=2)
                detector.fit(df)
                current_regime = detector.get_current_regime(df)
                
                # Update results
                mask = results_df['Ticker'] == ticker
                results_df.loc[mask, 'Market_Regime'] = current_regime['regime']
                results_df.loc[mask, 'Regime_Probability'] = current_regime['probability']
                results_df.loc[mask, 'Regime_Volatility'] = current_regime['volatility']
                
            except Exception as e:
                logger.warning(f"Could not detect regime for {ticker}: {e}")
    
    return results_df