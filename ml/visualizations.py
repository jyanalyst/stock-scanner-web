"""
ML Visualizations - Phase 2
Create interactive charts for factor analysis results
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MLVisualizer:
    """
    Create visualizations for factor analysis results
    
    Charts:
    - IC bar chart
    - Correlation heatmap
    - Feature importance ranking
    - IC stability over time
    - PCA scree plot
    - PCA biplot
    """
    
    def __init__(self):
        """Initialize visualizer"""
        self.default_height = 500
        self.default_width = None  # Auto-width
    
    def plot_ic_bar_chart(self, 
                         ic_results: pd.DataFrame,
                         top_n: int = 20,
                         title: str = "Information Coefficient by Feature") -> go.Figure:
        """
        Create bar chart of IC values
        
        Args:
            ic_results: DataFrame with IC results
            top_n: Number of top features to show
            title: Chart title
        
        Returns:
            Plotly figure
        """
        logger.info(f"Creating IC bar chart (top {top_n})")
        
        # Get top N features
        top_features = ic_results.head(top_n).copy()
        
        # Create bar chart
        fig = px.bar(
            top_features,
            x='abs_IC',
            y='feature',
            orientation='h',
            color='IC_mean',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title=title,
            labels={'abs_IC': 'Absolute IC', 'feature': 'Feature', 'IC_mean': 'IC'},
            hover_data=['IC_mean', 'abs_IC', 'p_value', 'sample_size']
        )
        
        # Add reference lines
        fig.add_vline(x=0.05, line_dash="dash", line_color="orange", 
                     annotation_text="IC = 0.05 (Moderate)")
        fig.add_vline(x=0.10, line_dash="dash", line_color="green", 
                     annotation_text="IC = 0.10 (Strong)")
        
        fig.update_layout(
            height=max(400, top_n * 25),
            yaxis={'categoryorder': 'total ascending'},
            showlegend=True
        )
        
        return fig
    
    def plot_correlation_heatmap(self,
                                correlation_matrix: pd.DataFrame,
                                title: str = "Feature Correlation Matrix") -> go.Figure:
        """
        Create correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix
            title: Chart title
        
        Returns:
            Plotly figure
        """
        logger.info("Creating correlation heatmap")
        
        # Limit to top 50 features for readability
        if len(correlation_matrix) > 50:
            logger.warning(f"Correlation matrix has {len(correlation_matrix)} features, showing top 50")
            correlation_matrix = correlation_matrix.iloc[:50, :50]
        
        fig = px.imshow(
            correlation_matrix,
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0,
            zmin=-1,
            zmax=1,
            title=title,
            labels={'color': 'Correlation'},
            aspect='auto'
        )
        
        fig.update_layout(
            height=max(600, len(correlation_matrix) * 12),
            width=max(600, len(correlation_matrix) * 12)
        )
        
        return fig
    
    def plot_feature_importance(self,
                               optimal_weights: Dict[str, float],
                               top_n: int = 20,
                               title: str = "Feature Importance (Optimal Weights)") -> go.Figure:
        """
        Create feature importance bar chart
        
        Args:
            optimal_weights: Dictionary of feature weights
            top_n: Number of top features to show
            title: Chart title
        
        Returns:
            Plotly figure
        """
        logger.info(f"Creating feature importance chart (top {top_n})")
        
        # Convert to DataFrame and sort
        weights_df = pd.DataFrame([
            {'feature': k, 'weight': v, 'weight_pct': v * 100}
            for k, v in optimal_weights.items()
        ]).sort_values('weight', ascending=False).head(top_n)
        
        fig = px.bar(
            weights_df,
            x='weight_pct',
            y='feature',
            orientation='h',
            title=title,
            labels={'weight_pct': 'Weight (%)', 'feature': 'Feature'},
            color='weight_pct',
            color_continuous_scale='Blues',
            hover_data=['weight']
        )
        
        fig.update_layout(
            height=max(400, top_n * 25),
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        return fig
    
    def plot_ic_distribution(self,
                            ic_results: pd.DataFrame,
                            title: str = "Distribution of Information Coefficients") -> go.Figure:
        """
        Create histogram of IC values
        
        Args:
            ic_results: DataFrame with IC results
            title: Chart title
        
        Returns:
            Plotly figure
        """
        logger.info("Creating IC distribution histogram")
        
        fig = px.histogram(
            ic_results,
            x='IC_mean',
            nbins=50,
            title=title,
            labels={'IC_mean': 'Information Coefficient', 'count': 'Frequency'},
            color_discrete_sequence=['skyblue']
        )
        
        # Add reference lines
        fig.add_vline(x=0, line_dash="solid", line_color="black", 
                     annotation_text="IC = 0")
        fig.add_vline(x=0.05, line_dash="dash", line_color="orange", 
                     annotation_text="IC = 0.05")
        fig.add_vline(x=-0.05, line_dash="dash", line_color="orange")
        fig.add_vline(x=0.10, line_dash="dash", line_color="green", 
                     annotation_text="IC = 0.10")
        fig.add_vline(x=-0.10, line_dash="dash", line_color="green")
        
        fig.update_layout(height=400)
        
        return fig
    
    def plot_ic_vs_sample_size(self,
                               ic_results: pd.DataFrame,
                               title: str = "IC vs Sample Size") -> go.Figure:
        """
        Create scatter plot of IC vs sample size
        
        Args:
            ic_results: DataFrame with IC results
            title: Chart title
        
        Returns:
            Plotly figure
        """
        logger.info("Creating IC vs sample size scatter plot")
        
        fig = px.scatter(
            ic_results,
            x='sample_size',
            y='abs_IC',
            color='significant',
            title=title,
            labels={'sample_size': 'Sample Size', 'abs_IC': 'Absolute IC', 'significant': 'Significant (p<0.05)'},
            hover_data=['feature', 'IC_mean', 'p_value'],
            color_discrete_map={True: 'green', False: 'red'}
        )
        
        # Add reference line
        fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                     annotation_text="IC = 0.05")
        
        fig.update_layout(height=500)
        
        return fig
    
    def plot_pca_scree(self,
                      pca_results: Dict,
                      title: str = "PCA Scree Plot - Explained Variance") -> go.Figure:
        """
        Create PCA scree plot
        
        Args:
            pca_results: Dictionary with PCA results
            title: Chart title
        
        Returns:
            Plotly figure
        """
        logger.info("Creating PCA scree plot")
        
        n_components = len(pca_results['explained_variance_ratio'])
        
        # Create DataFrame for plotting
        scree_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(n_components)],
            'component_num': list(range(1, n_components + 1)),
            'explained_variance': pca_results['explained_variance_ratio'] * 100,
            'cumulative_variance': pca_results['cumulative_variance'] * 100
        })
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for individual variance
        fig.add_trace(
            go.Bar(
                x=scree_df['component'],
                y=scree_df['explained_variance'],
                name='Individual Variance',
                marker_color='lightblue'
            ),
            secondary_y=False
        )
        
        # Add line chart for cumulative variance
        fig.add_trace(
            go.Scatter(
                x=scree_df['component'],
                y=scree_df['cumulative_variance'],
                name='Cumulative Variance',
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        # Add 95% reference line
        fig.add_hline(y=95, line_dash="dash", line_color="green", 
                     annotation_text="95% Variance", secondary_y=True)
        
        # Update layout
        fig.update_xaxes(title_text="Principal Component")
        fig.update_yaxes(title_text="Explained Variance (%)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True)
        
        fig.update_layout(
            title=title,
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_pca_loadings(self,
                         pca_results: Dict,
                         component_x: int = 1,
                         component_y: int = 2,
                         top_n: int = 15,
                         title: Optional[str] = None) -> go.Figure:
        """
        Create PCA loadings plot (biplot)
        
        Args:
            pca_results: Dictionary with PCA results
            component_x: X-axis component number
            component_y: Y-axis component number
            top_n: Number of top features to label
            title: Chart title
        
        Returns:
            Plotly figure
        """
        logger.info(f"Creating PCA loadings plot (PC{component_x} vs PC{component_y})")
        
        if title is None:
            title = f"PCA Loadings: PC{component_x} vs PC{component_y}"
        
        loadings = pca_results['loadings']
        
        pc_x = f'PC{component_x}'
        pc_y = f'PC{component_y}'
        
        # Calculate magnitude for each feature
        loadings_df = pd.DataFrame({
            'feature': loadings.index,
            'pc_x': loadings[pc_x],
            'pc_y': loadings[pc_y],
            'magnitude': np.sqrt(loadings[pc_x]**2 + loadings[pc_y]**2)
        }).sort_values('magnitude', ascending=False)
        
        # Create scatter plot
        fig = px.scatter(
            loadings_df,
            x='pc_x',
            y='pc_y',
            hover_data=['feature', 'magnitude'],
            title=title,
            labels={'pc_x': f'{pc_x} Loading', 'pc_y': f'{pc_y} Loading'},
            color='magnitude',
            color_continuous_scale='Viridis'
        )
        
        # Add arrows for top N features
        top_features = loadings_df.head(top_n)
        
        for _, row in top_features.iterrows():
            fig.add_annotation(
                x=row['pc_x'],
                y=row['pc_y'],
                ax=0,
                ay=0,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                text=row['feature'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='red'
            )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(height=600, width=800)
        
        return fig
    
    def plot_redundant_pairs(self,
                            redundant_pairs: list,
                            top_n: int = 20,
                            title: str = "Highly Correlated Feature Pairs") -> go.Figure:
        """
        Create bar chart of redundant feature pairs
        
        Args:
            redundant_pairs: List of redundant pair dictionaries
            top_n: Number of pairs to show
            title: Chart title
        
        Returns:
            Plotly figure
        """
        logger.info(f"Creating redundant pairs chart (top {top_n})")
        
        if not redundant_pairs:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No highly correlated pairs found",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title=title, height=400)
            return fig
        
        # Convert to DataFrame
        pairs_df = pd.DataFrame(redundant_pairs).head(top_n)
        pairs_df['pair_label'] = pairs_df['feature1'] + ' ↔ ' + pairs_df['feature2']
        
        fig = px.bar(
            pairs_df,
            x='correlation',
            y='pair_label',
            orientation='h',
            title=title,
            labels={'correlation': 'Correlation', 'pair_label': 'Feature Pair'},
            color='correlation',
            color_continuous_scale='Reds',
            hover_data=['ic1', 'ic2', 'keep', 'remove']
        )
        
        fig.update_layout(
            height=max(400, top_n * 30),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_summary_dashboard(self,
                                ic_results: pd.DataFrame,
                                correlation_matrix: pd.DataFrame,
                                optimal_weights: Dict[str, float],
                                redundant_pairs: list) -> Dict[str, go.Figure]:
        """
        Create complete dashboard of visualizations
        
        Args:
            ic_results: IC results DataFrame
            correlation_matrix: Correlation matrix
            optimal_weights: Optimal weights dictionary
            redundant_pairs: List of redundant pairs
        
        Returns:
            Dictionary of figure name: figure
        """
        logger.info("Creating complete visualization dashboard")
        
        figures = {}
        
        try:
            figures['ic_bar_chart'] = self.plot_ic_bar_chart(ic_results, top_n=20)
        except Exception as e:
            logger.error(f"Error creating IC bar chart: {e}")
        
        try:
            figures['ic_distribution'] = self.plot_ic_distribution(ic_results)
        except Exception as e:
            logger.error(f"Error creating IC distribution: {e}")
        
        try:
            figures['ic_vs_sample_size'] = self.plot_ic_vs_sample_size(ic_results)
        except Exception as e:
            logger.error(f"Error creating IC vs sample size: {e}")
        
        try:
            figures['correlation_heatmap'] = self.plot_correlation_heatmap(correlation_matrix)
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
        
        try:
            figures['feature_importance'] = self.plot_feature_importance(optimal_weights, top_n=20)
        except Exception as e:
            logger.error(f"Error creating feature importance: {e}")
        
        try:
            figures['redundant_pairs'] = self.plot_redundant_pairs(redundant_pairs, top_n=15)
        except Exception as e:
            logger.error(f"Error creating redundant pairs: {e}")
        
        logger.info(f"Created {len(figures)} visualizations")
        
        return figures
