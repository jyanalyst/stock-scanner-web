"""
ML Factor Analyzer - Phase 2
Analyzes which features predict forward returns using:
- Information Coefficient (IC)
- Correlation analysis
- Feature selection
- Optimal weight calculation
- Optional PCA
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class MLFactorAnalyzer:
    """
    Analyzes predictive power of features for forward returns
    
    Key Methods:
    - calculate_information_coefficient(): IC for each feature
    - analyze_correlations(): Find redundant features
    - select_features(): Remove weak/redundant features
    - calculate_optimal_weights(): Weight features by IC
    - run_pca_analysis(): Optional dimensionality reduction
    """
    
    def __init__(self, 
                 data_path: str = "data/ml_training/raw/training_data_complete.parquet",
                 target: str = "return_3d"):
        """
        Initialize factor analyzer
        
        Args:
            data_path: Path to training data parquet file
            target: Target variable (return_2d, return_3d, or return_4d)
        """
        self.data_path = data_path
        self.target = target
        self.df = None
        self.features = None
        self.ic_results = None
        self.correlation_matrix = None
        self.selected_features = None
        self.optimal_weights = None
        self.pca_results = None
        
        logger.info(f"Initialized MLFactorAnalyzer with target={target}")
    
    def load_data(self) -> pd.DataFrame:
        """Load training data and identify features"""
        logger.info(f"Loading training data from {self.data_path}")
        
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.df):,} samples with {len(self.df.columns)} columns")
        
        # Identify feature columns (exclude metadata and target columns)
        exclude_cols = [
            'index', 'Ticker', 'ticker', 'ticker_sgx', 'ticker_earnings', 'ticker_sgx_earnings',
            'Name', 'Date', 'Analysis_Date', 'entry_date', 'entry_price',
            'return_2d', 'return_3d', 'return_4d', 'win_2d', 'win_3d', 'win_4d',
            'max_drawdown', 'analyst_firm', 'analyst_name', 'report_title',
            'pdf_filename', 'pdf_filename_earnings', 'upload_date', 'upload_date_earnings',
            'sentiment_reasoning', 'key_catalysts', 'key_risks', 'executive_summary',
            'key_highlights', 'concerns', 'mgmt_commentary_summary',
            'Sentiment_Display', 'Report_Date_Display', 'Report_Count_Display',
            'Earnings_Period', 'Guidance_Display', 'Rev_YoY_Display', 'EPS_DPU_Display',
            'earnings_reaction_stats', 'Earnings_Reaction', 'report_date_earnings',
            'company_type', 'report_type', 'fiscal_year', 'report_time',
            'analysis_method', 'sentiment_method', 'sentiment_label'
        ]
        
        # Get numeric columns that are not in exclude list
        self.features = [col for col in self.df.columns 
                        if col not in exclude_cols 
                        and pd.api.types.is_numeric_dtype(self.df[col])]
        
        logger.info(f"Identified {len(self.features)} numeric features for analysis")
        
        # Check target exists
        if self.target not in self.df.columns:
            raise ValueError(f"Target '{self.target}' not found in data")
        
        # Remove rows with missing target
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[self.target])
        final_count = len(self.df)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} rows with missing {self.target}")
        
        return self.df
    
    def calculate_information_coefficient(self, 
                                         rolling_window: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate Information Coefficient (IC) for each feature
        
        IC = Spearman correlation between feature ranks and return ranks
        
        Args:
            rolling_window: If provided, calculate IC over rolling windows for stability
        
        Returns:
            DataFrame with IC statistics for each feature
        """
        logger.info(f"Calculating Information Coefficient for {len(self.features)} features")
        
        ic_results = []
        
        for feature in self.features:
            try:
                # Get feature and target values, drop NaN
                feature_data = self.df[[feature, self.target]].dropna()
                
                if len(feature_data) < 100:  # Minimum sample size
                    logger.warning(f"Skipping {feature}: insufficient data ({len(feature_data)} samples)")
                    continue
                
                # Calculate Spearman correlation (rank-based)
                ic, p_value = spearmanr(feature_data[feature], feature_data[self.target])
                
                # Calculate IC over rolling windows for stability
                if rolling_window and len(feature_data) > rolling_window:
                    rolling_ics = []
                    for i in range(0, len(feature_data) - rolling_window, rolling_window // 2):
                        window_data = feature_data.iloc[i:i+rolling_window]
                        if len(window_data) >= 50:
                            window_ic, _ = spearmanr(window_data[feature], window_data[self.target])
                            if not np.isnan(window_ic):
                                rolling_ics.append(window_ic)
                    
                    ic_std = np.std(rolling_ics) if rolling_ics else np.nan
                    ic_ir = ic / ic_std if ic_std > 0 else np.nan  # Information Ratio
                else:
                    ic_std = np.nan
                    ic_ir = np.nan
                
                ic_results.append({
                    'feature': feature,
                    'IC_mean': ic,
                    'IC_std': ic_std,
                    'IC_IR': ic_ir,
                    'p_value': p_value,
                    'abs_IC': abs(ic),
                    'sample_size': len(feature_data),
                    'significant': p_value < 0.05
                })
                
            except Exception as e:
                logger.error(f"Error calculating IC for {feature}: {e}")
                continue
        
        self.ic_results = pd.DataFrame(ic_results)
        
        # Sort by absolute IC (predictive power regardless of direction)
        self.ic_results = self.ic_results.sort_values('abs_IC', ascending=False)
        
        logger.info(f"Calculated IC for {len(self.ic_results)} features")
        logger.info(f"Features with |IC| > 0.05: {(self.ic_results['abs_IC'] > 0.05).sum()}")
        logger.info(f"Features with |IC| > 0.10: {(self.ic_results['abs_IC'] > 0.10).sum()}")
        
        return self.ic_results
    
    def analyze_correlations(self, threshold: float = 0.85) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Analyze feature correlations to identify redundant features
        
        Args:
            threshold: Correlation threshold above which features are considered redundant
        
        Returns:
            Tuple of (correlation_matrix, redundant_pairs)
        """
        logger.info(f"Analyzing feature correlations (threshold={threshold})")
        
        # Get features with valid IC
        if self.ic_results is None:
            raise ValueError("Must run calculate_information_coefficient() first")
        
        valid_features = self.ic_results['feature'].tolist()
        
        # Calculate correlation matrix
        feature_data = self.df[valid_features].dropna(axis=1, how='all')
        self.correlation_matrix = feature_data.corr(method='spearman')
        
        # Find highly correlated pairs
        redundant_pairs = []
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                
                if corr_value > threshold:
                    feature1 = self.correlation_matrix.columns[i]
                    feature2 = self.correlation_matrix.columns[j]
                    
                    # Get IC for both features
                    ic1 = self.ic_results[self.ic_results['feature'] == feature1]['abs_IC'].values[0]
                    ic2 = self.ic_results[self.ic_results['feature'] == feature2]['abs_IC'].values[0]
                    
                    # Recommend keeping the one with higher IC
                    keep = feature1 if ic1 >= ic2 else feature2
                    remove = feature2 if ic1 >= ic2 else feature1
                    
                    redundant_pairs.append({
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation': corr_value,
                        'ic1': ic1,
                        'ic2': ic2,
                        'keep': keep,
                        'remove': remove
                    })
        
        logger.info(f"Found {len(redundant_pairs)} redundant feature pairs (correlation > {threshold})")
        
        return self.correlation_matrix, redundant_pairs
    
    def select_features(self, 
                       ic_threshold: float = 0.03,
                       correlation_threshold: float = 0.85) -> List[str]:
        """
        Select optimal feature set by removing weak and redundant features
        
        Args:
            ic_threshold: Minimum absolute IC to keep feature
            correlation_threshold: Maximum correlation for redundancy
        
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features (IC threshold={ic_threshold}, corr threshold={correlation_threshold})")
        
        if self.ic_results is None:
            raise ValueError("Must run calculate_information_coefficient() first")
        
        # Step 1: Remove weak features (low IC)
        strong_features = self.ic_results[self.ic_results['abs_IC'] >= ic_threshold]['feature'].tolist()
        logger.info(f"Step 1: {len(strong_features)} features with |IC| >= {ic_threshold}")
        
        # Step 2: Remove redundant features
        _, redundant_pairs = self.analyze_correlations(correlation_threshold)
        
        features_to_remove = set([pair['remove'] for pair in redundant_pairs])
        logger.info(f"Step 2: Removing {len(features_to_remove)} redundant features")
        
        # Final selected features
        self.selected_features = [f for f in strong_features if f not in features_to_remove]
        
        logger.info(f"Final selection: {len(self.selected_features)} features")
        logger.info(f"Reduction: {len(self.features)} → {len(self.selected_features)} ({len(self.selected_features)/len(self.features)*100:.1f}%)")
        
        return self.selected_features
    
    def calculate_optimal_weights(self, method: str = 'ic_squared') -> Dict[str, float]:
        """
        Calculate optimal feature weights based on IC
        
        Args:
            method: Weighting method
                - 'ic': Weight by IC
                - 'ic_squared': Weight by IC^2 (emphasize strong features)
                - 'ic_ir': Weight by Information Ratio (IC/IC_std)
        
        Returns:
            Dictionary of feature: weight
        """
        logger.info(f"Calculating optimal weights using method='{method}'")
        
        if self.selected_features is None:
            raise ValueError("Must run select_features() first")
        
        # Get IC for selected features
        selected_ic = self.ic_results[self.ic_results['feature'].isin(self.selected_features)].copy()
        
        # Calculate weights based on method
        if method == 'ic':
            selected_ic['weight'] = selected_ic['abs_IC']
        elif method == 'ic_squared':
            selected_ic['weight'] = selected_ic['abs_IC'] ** 2
        elif method == 'ic_ir':
            # Use IC/IC_std, fallback to IC if IC_std not available
            selected_ic['weight'] = selected_ic.apply(
                lambda row: abs(row['IC_IR']) if not pd.isna(row['IC_IR']) else row['abs_IC'],
                axis=1
            )
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Normalize weights to sum to 1
        total_weight = selected_ic['weight'].sum()
        selected_ic['weight'] = selected_ic['weight'] / total_weight
        
        # Convert to dictionary
        self.optimal_weights = dict(zip(selected_ic['feature'], selected_ic['weight']))
        
        logger.info(f"Calculated weights for {len(self.optimal_weights)} features")
        
        # Log top 10 weights
        top_weights = sorted(self.optimal_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 feature weights:")
        for feature, weight in top_weights:
            logger.info(f"  {feature}: {weight:.4f}")
        
        return self.optimal_weights
    
    def run_pca_analysis(self, n_components: Optional[int] = None) -> Dict:
        """
        Run PCA for dimensionality reduction analysis (optional)
        
        Args:
            n_components: Number of components (None = auto-determine)
        
        Returns:
            Dictionary with PCA results
        """
        logger.info("Running PCA analysis")
        
        if self.selected_features is None:
            raise ValueError("Must run select_features() first")
        
        # Prepare data
        feature_data = self.df[self.selected_features].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Run PCA
        if n_components is None:
            # Use all components to see explained variance
            n_components = min(len(self.selected_features), len(feature_data))
        
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(scaled_data)
        
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Get component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=self.selected_features
        )
        
        self.pca_results = {
            'n_components': pca.n_components_,
            'n_components_95': n_components_95,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'loadings': loadings,
            'pca_model': pca,
            'scaler': scaler
        }
        
        logger.info(f"PCA complete: {n_components_95} components explain 95% variance")
        logger.info(f"Original features: {len(self.selected_features)} → PCA components: {n_components_95}")
        
        return self.pca_results
    
    def generate_report(self, output_dir: str = "data/ml_training/analysis/") -> str:
        """
        Generate comprehensive factor analysis report
        
        Args:
            output_dir: Directory to save report files
        
        Returns:
            Path to HTML report
        """
        logger.info("Generating factor analysis report")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save IC results
        if self.ic_results is not None:
            ic_path = os.path.join(output_dir, 'ic_results.csv')
            self.ic_results.to_csv(ic_path, index=False)
            logger.info(f"Saved IC results to {ic_path}")
        
        # Save correlation matrix
        if self.correlation_matrix is not None:
            corr_path = os.path.join(output_dir, 'correlation_matrix.csv')
            self.correlation_matrix.to_csv(corr_path)
            logger.info(f"Saved correlation matrix to {corr_path}")
        
        # Save optimal weights
        if self.optimal_weights is not None:
            weights_path = os.path.join(output_dir, 'optimal_weights.json')
            with open(weights_path, 'w') as f:
                json.dump(self.optimal_weights, f, indent=2)
            logger.info(f"Saved optimal weights to {weights_path}")
        
        # Save selected features
        if self.selected_features is not None:
            features_path = os.path.join(output_dir, 'selected_features.json')
            with open(features_path, 'w') as f:
                json.dump(self.selected_features, f, indent=2)
            logger.info(f"Saved selected features to {features_path}")
        
        # Generate HTML report
        html_path = os.path.join(output_dir, 'factor_analysis_report.html')
        html_content = self._generate_html_report()
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {html_path}")
        
        return html_path
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Factor Analysis Report - {self.target}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .good {{ color: #27ae60; font-weight: bold; }}
                .bad {{ color: #e74c3c; font-weight: bold; }}
                .moderate {{ color: #f39c12; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🔬 Factor Analysis Report</h1>
            <p><strong>Target Variable:</strong> {self.target}</p>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Training Samples:</strong> {len(self.df):,}</p>
            
            <h2>📊 Summary Statistics</h2>
            <div class="metric">
                <p><strong>Total Features Analyzed:</strong> {len(self.features)}</p>
                <p><strong>Features with |IC| > 0.05:</strong> {(self.ic_results['abs_IC'] > 0.05).sum() if self.ic_results is not None else 'N/A'}</p>
                <p><strong>Features with |IC| > 0.10:</strong> {(self.ic_results['abs_IC'] > 0.10).sum() if self.ic_results is not None else 'N/A'}</p>
                <p><strong>Selected Features:</strong> {len(self.selected_features) if self.selected_features else 'N/A'}</p>
                <p><strong>Feature Reduction:</strong> {len(self.features)} → {len(self.selected_features) if self.selected_features else 'N/A'} ({len(self.selected_features)/len(self.features)*100:.1f}% retained)</p>
            </div>
            
            <h2>🏆 Top 20 Features by Information Coefficient</h2>
            {self._generate_ic_table()}
            
            <h2>🎯 Optimal Feature Weights</h2>
            {self._generate_weights_table()}
            
            <h2>💡 Recommendations</h2>
            {self._generate_recommendations()}
            
        </body>
        </html>
        """
        
        return html
    
    def _generate_ic_table(self) -> str:
        """Generate HTML table for IC results"""
        if self.ic_results is None or len(self.ic_results) == 0:
            return "<p>No IC results available</p>"
        
        top_20 = self.ic_results.head(20)
        
        html = "<table><tr><th>Rank</th><th>Feature</th><th>IC</th><th>|IC|</th><th>p-value</th><th>Significant</th></tr>"
        
        for i, row in enumerate(top_20.itertuples(), 1):
            sig_class = 'good' if row.significant else 'bad'
            ic_class = 'good' if row.abs_IC > 0.10 else 'moderate' if row.abs_IC > 0.05 else ''
            
            html += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{row.feature}</strong></td>
                <td>{row.IC_mean:.4f}</td>
                <td class="{ic_class}">{row.abs_IC:.4f}</td>
                <td>{row.p_value:.4f}</td>
                <td class="{sig_class}">{'✅ Yes' if row.significant else '❌ No'}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_weights_table(self) -> str:
        """Generate HTML table for optimal weights"""
        if self.optimal_weights is None:
            return "<p>No weights calculated</p>"
        
        sorted_weights = sorted(self.optimal_weights.items(), key=lambda x: x[1], reverse=True)[:20]
        
        html = "<table><tr><th>Rank</th><th>Feature</th><th>Weight</th><th>Weight %</th></tr>"
        
        for i, (feature, weight) in enumerate(sorted_weights, 1):
            html += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{feature}</strong></td>
                <td>{weight:.6f}</td>
                <td>{weight*100:.2f}%</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on analysis"""
        if self.ic_results is None:
            return "<p>Run analysis first</p>"
        
        strong_features = (self.ic_results['abs_IC'] > 0.10).sum()
        moderate_features = ((self.ic_results['abs_IC'] > 0.05) & (self.ic_results['abs_IC'] <= 0.10)).sum()
        weak_features = (self.ic_results['abs_IC'] <= 0.05).sum()
        
        html = "<div class='metric'>"
        
        if strong_features > 10:
            html += "<p class='good'>✅ <strong>Excellent:</strong> You have {} features with strong predictive power (|IC| > 0.10)</p>".format(strong_features)
        elif strong_features > 5:
            html += "<p class='moderate'>📊 <strong>Good:</strong> You have {} features with strong predictive power (|IC| > 0.10)</p>".format(strong_features)
        else:
            html += "<p class='bad'>⚠️ <strong>Limited:</strong> Only {} features with strong predictive power (|IC| > 0.10)</p>".format(strong_features)
        
        html += f"<p>📈 <strong>Moderate features:</strong> {moderate_features} features with |IC| between 0.05-0.10</p>"
        html += f"<p>📉 <strong>Weak features:</strong> {weak_features} features with |IC| < 0.05 (consider removing)</p>"
        
        if self.selected_features:
            html += f"<p class='good'>✅ <strong>Recommended feature set:</strong> Use the {len(self.selected_features)} selected features for Phase 3 model training</p>"
        
        html += "</div>"
        
        return html
    
    def run_full_analysis(self,
                         ic_threshold: float = 0.03,
                         correlation_threshold: float = 0.85,
                         run_pca: bool = False) -> Dict:
        """
        Run complete factor analysis pipeline
        
        Args:
            ic_threshold: Minimum IC to keep feature
            correlation_threshold: Maximum correlation for redundancy
            run_pca: Whether to run PCA analysis
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("=" * 80)
        logger.info("STARTING FULL FACTOR ANALYSIS")
        logger.info("=" * 80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Calculate IC
        self.calculate_information_coefficient(rolling_window=60)
        
        # Step 3: Analyze correlations
        self.analyze_correlations(correlation_threshold)
        
        # Step 4: Select features
        self.select_features(ic_threshold, correlation_threshold)
        
        # Step 5: Calculate optimal weights
        self.calculate_optimal_weights(method='ic_squared')
        
        # Step 6: Optional PCA
        if run_pca:
            self.run_pca_analysis()
        
        # Step 7: Generate report
        report_path = self.generate_report()
        
        logger.info("=" * 80)
        logger.info("FACTOR ANALYSIS COMPLETE")
        logger.info("=" * 80)
        
        return {
            'ic_results': self.ic_results,
            'correlation_matrix': self.correlation_matrix,
            'selected_features': self.selected_features,
            'optimal_weights': self.optimal_weights,
            'pca_results': self.pca_results,
            'report_path': report_path
        }
