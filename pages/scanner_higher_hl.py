# File: pages/scanner_higher_hl.py
# Part 1 of 3
"""
CRT Higher H/L Scanner - COMPLETE PART 1
REDESIGNED WITH PURE MPI EXPANSION SYSTEM
Enhanced with MPI (Market Positivity Index) using velocity-based expansion filtering
Focus on momentum direction rather than absolute levels
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import time
import traceback
import logging
import sys
from io import StringIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ErrorLogger:
    """Centralized error logging and display for debugging"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.debug_info = []
    
    def log_error(self, component: str, error: Exception, context: dict = None):
        """Log an error with full context"""
        error_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        self.errors.append(error_info)
        logger.error(f"{component}: {error_info['error_type']} - {error_info['error_message']}")
        return error_info
    
    def log_warning(self, component: str, message: str, context: dict = None):
        """Log a warning"""
        warning_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'component': component,
            'message': message,
            'context': context or {}
        }
        self.warnings.append(warning_info)
        logger.warning(f"{component}: {message}")
        return warning_info
    
    def log_debug(self, component: str, message: str, data: any = None):
        """Log debug information"""
        debug_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'component': component,
            'message': message,
            'data': str(data) if data is not None else None
        }
        self.debug_info.append(debug_info)
        logger.info(f"DEBUG {component}: {message}")
        return debug_info
    
    def display_errors_in_streamlit(self):
        """Display all logged errors in Streamlit interface"""
        if self.errors:
            st.error(f"❌ {len(self.errors)} error(s) occurred during execution")
            
            with st.expander("🔍 View Error Details", expanded=True):
                for i, error in enumerate(self.errors, 1):
                    st.markdown(f"**Error {i}: {error['component']}**")
                    st.code(f"Type: {error['error_type']}\nMessage: {error['error_message']}")
                    
                    if error['context']:
                        st.json(error['context'])
                    
                    with st.expander(f"Full Traceback - Error {i}"):
                        st.code(error['traceback'], language='python')
                    
                    st.markdown("---")
        
        if self.warnings:
            st.warning(f"⚠️ {len(self.warnings)} warning(s) occurred")
            
            with st.expander("📋 View Warnings"):
                for warning in self.warnings:
                    st.write(f"**{warning['component']}**: {warning['message']}")
                    if warning['context']:
                        st.json(warning['context'])
    
    def get_summary(self):
        """Get a summary of logged issues"""
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'debug_entries': len(self.debug_info)
        }

# Initialize global error logger
if 'error_logger' not in st.session_state:
    st.session_state.error_logger = ErrorLogger()

def get_price_format(price: float) -> str:
    """
    Get the appropriate price format string based on price level
    Returns format string for st.column_config.NumberColumn
    """
    if price < 1.00:
        return "$%.3f"  # 3 decimal places for stocks under $1.00
    else:
        return "$%.2f"  # 2 decimal places for regular stocks

def get_dynamic_column_config(df: pd.DataFrame, display_cols: list, base_config: dict) -> dict:
    """
    Create dynamic column configuration with price-based decimal formatting
    """
    config = base_config.copy()
    
    # Identify price columns that need dynamic formatting
    price_columns = ['Close', 'High', 'Low', 'CRT_High', 'CRT_Low', 'Weekly_Open']
    
    for col in price_columns:
        if col in display_cols and col in df.columns:
            # Check if we have price data to determine format
            if len(df) > 0:
                # Use the minimum price in the column to determine format
                min_price = df[col].min()
                if min_price < 1.00:
                    # At least one stock is under $1, use 3 decimals for all
                    config[col] = st.column_config.NumberColumn(
                        config[col].title if col in config and hasattr(config[col], 'title') else col.replace('_', ' ').title(),
                        format="$%.3f"
                    )
                else:
                    # All stocks are $1.00 or above, use 2 decimals
                    config[col] = st.column_config.NumberColumn(
                        config[col].title if col in config and hasattr(config[col], 'title') else col.replace('_', ' ').title(),
                        format="$%.2f"
                    )
    
    return config

def format_mpi_visual(mpi_value: float) -> str:
    """
    Convert MPI to visual blocks for intuitive display
    """
    if pd.isna(mpi_value):
        return "░░░░░░░░░░"
    
    blocks = max(0, min(10, int(mpi_value * 10)))  # Ensure 0-10 range
    return "█" * blocks + "░" * (10 - blocks)

def get_trend_description(trend: str) -> str:
    """
    Get trading description for MPI trends
    """
    descriptions = {
        'Strong Expansion': 'Buy - Strong momentum',
        'Expanding': 'Buy - Building momentum',
        'Flat': 'Hold - No change',
        'Mild Contraction': 'Caution - Weakening',
        'Strong Contraction': 'Sell/Short - Momentum lost'
    }
    return descriptions.get(trend, 'Unknown')

def apply_mpi_expansion_filter(base_stocks: pd.DataFrame, selected_trends: list) -> pd.DataFrame:
    """
    Apply pure MPI expansion-based filtering with intelligent sorting
    
    Args:
        base_stocks: Input DataFrame with MPI_Trend column
        selected_trends: List of selected MPI trends
    
    Returns:
        Filtered DataFrame sorted by velocity
    """
    if base_stocks.empty:
        return base_stocks
    
    if 'MPI_Trend' not in base_stocks.columns:
        st.warning("MPI_Trend column not found - using all stocks")
        return base_stocks
    
    # If no trends selected, return all stocks
    if not selected_trends:
        filtered = base_stocks.copy()
        return filtered.sort_values('MPI_Velocity', ascending=False)
    
    # Filter by selected MPI trends
    filtered = base_stocks[base_stocks['MPI_Trend'].isin(selected_trends)].copy()
    
    # Smart sorting based on selected trends
    expansion_trends = ['Strong Expansion', 'Expanding']
    contraction_trends = ['Mild Contraction', 'Strong Contraction']
    
    selected_categories = []
    if any(trend in expansion_trends for trend in selected_trends):
        selected_categories.append('expansion')
    if any(trend in contraction_trends for trend in selected_trends):
        selected_categories.append('contraction')
    if 'Flat' in selected_trends:
        selected_categories.append('flat')
    
    # Sort based on predominant selection
    if selected_categories == ['contraction']:
        # For shorting - sort weakest first (most negative velocity)
        filtered = filtered.sort_values('MPI_Velocity', ascending=True)
    elif selected_categories == ['expansion']:
        # For long trades - sort strongest first (highest positive velocity)
        filtered = filtered.sort_values('MPI_Velocity', ascending=False)
    else:
        # Mixed selection - sort by MPI level, then velocity
        filtered = filtered.sort_values(['MPI', 'MPI_Velocity'], ascending=[False, False])
    
    return filtered

# File: pages/scanner_higher_hl.py
# Part 2 of 3

def apply_dynamic_filters(base_stocks, results_df):
    """
    Apply dynamic filtering with VCRE Velocity, IBS, Higher H/L, and PURE MPI EXPANSION filters
    UPDATED: Column 4 now shows MPI Expansion Filter with velocity-based trends
    Returns filtered stocks
    """
    
    if base_stocks.empty:
        st.warning("No stocks available for filtering")
        return base_stocks
    
    st.subheader("🎯 Dynamic Filtering")
    
    # Create four columns for filters
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize filtered stocks
    filtered_stocks = base_stocks.copy()
    
    # COL 1: VCRE VELOCITY PERCENTILE FILTER (UNCHANGED)
    with col1:
        st.markdown("**VCRE Velocity Filter:**")
        
        # Get velocity statistics
        if 'VCRE_Velocity' in base_stocks.columns:
            velocities = base_stocks['VCRE_Velocity']
        else:
            velocities = base_stocks.index.map(lambda idx: results_df.loc[idx, 'VCRE_Velocity'] if idx in results_df.index else 0)
            velocities = pd.Series(velocities, index=base_stocks.index)
        
        # Remove any zero velocities for percentile calculation
        non_zero_velocities = velocities[velocities != 0]
        
        if len(non_zero_velocities) > 0:
            percentile_options = {
                "Top 25%": 75,
                "Top 50%": 50, 
                "Top 75%": 25,
                "No Filter": None
            }
            
            selected_percentile = st.radio(
                "Select velocity filter:",
                list(percentile_options.keys()),
                key="vcre_velocity_percentile_radio"
            )
            
            # Apply percentile filtering
            if selected_percentile != "No Filter":
                percentile_val = percentile_options[selected_percentile]
                threshold_value = np.percentile(non_zero_velocities, percentile_val)
                filtered_stocks = filtered_stocks[velocities >= threshold_value]
                st.info(f"VCRE Velocity ≥ {threshold_value:+.4f} pp")
            else:
                st.info("All velocities included")
            
            # Show velocity statistics
            with st.expander("VCRE Velocity Statistics", expanded=False):
                stats_data = {
                    "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                    "Value": [
                        f"{len(non_zero_velocities)}",
                        f"{non_zero_velocities.min():+.4f}",
                        f"{np.percentile(non_zero_velocities, 25):+.4f}",
                        f"{non_zero_velocities.median():+.4f}", 
                        f"{np.percentile(non_zero_velocities, 75):+.4f}",
                        f"{non_zero_velocities.max():+.4f}"
                    ],
                    "Unit": ["stocks", "pp", "pp", "pp", "pp", "pp"]
                }
                st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
        else:
            st.info("No VCRE Velocity data available for filtering")
    
    # COL 2: IBS FILTER (UNCHANGED)
    with col2:
        st.markdown("**IBS Filter:**")
        
        # Get IBS values
        ibs_values = filtered_stocks['IBS']
        
        # Percentile options (standardized)
        ibs_percentile_options = {
            "Top 25%": 75,
            "Top 50%": 50,
            "Top 75%": 25,
            "Custom": "custom",
            "No Filter": None
        }
        
        selected_ibs_option = st.radio(
            "Select IBS filter:",
            list(ibs_percentile_options.keys()),
            key="ibs_percentile_radio"
        )
        
        # Handle custom input
        if selected_ibs_option == "Custom":
            custom_ibs_value = st.number_input(
                "Enter minimum IBS value:",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                format="%.2f",
                key="custom_ibs_input"
            )
            filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= custom_ibs_value]
            st.info(f"IBS ≥ {custom_ibs_value:.2f}")
        elif selected_ibs_option != "No Filter":
            # Percentile-based filtering
            percentile_val = ibs_percentile_options[selected_ibs_option]
            threshold_value = np.percentile(ibs_values, percentile_val)
            filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= threshold_value]
            st.info(f"IBS ≥ {threshold_value:.3f} ({selected_ibs_option})")
        else:
            st.info("All IBS values included")
        
        # Show IBS statistics
        with st.expander("IBS Statistics", expanded=False):
            ibs_stats = {
                "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                "Value": [
                    f"{len(ibs_values)}",
                    f"{ibs_values.min():.3f}",
                    f"{np.percentile(ibs_values, 25):.3f}",
                    f"{ibs_values.median():.3f}",
                    f"{np.percentile(ibs_values, 75):.3f}",
                    f"{ibs_values.max():.3f}"
                ],
                "Unit": ["stocks", "", "", "", "", ""]
            }
            st.dataframe(pd.DataFrame(ibs_stats), hide_index=True, use_container_width=True)
    
    # COL 3: HIGHER H/L FILTER (UNCHANGED)
    with col3:
        # Check if we need this filter
        base_filter_type = st.session_state.get('base_filter_type', 'All Stocks')
        
        if base_filter_type not in ["Valid CRT + Higher H/L", "Higher H/L Only"]:
            st.markdown("**Higher H/L Filter:**")
            
            higher_hl_options = {
                "Higher H/L Only": 1,
                "No Filter": None
            }
            
            selected_higher_hl = st.radio(
                "Select Higher H/L filter:",
                list(higher_hl_options.keys()),
                key="higher_hl_pattern_radio"
            )
            
            # Apply Higher_HL filtering
            if selected_higher_hl != "No Filter":
                filtered_stocks = filtered_stocks[filtered_stocks['Higher_HL'] == 1]
                st.info("Only showing Higher High/Low patterns")
            else:
                st.info("All patterns included")
            
            # Show Higher_HL statistics
            with st.expander("Higher H/L Statistics", expanded=False):
                higher_hl_count = (base_stocks['Higher_HL'] == 1).sum()
                total_count = len(base_stocks)
                higher_hl_pct = (higher_hl_count / total_count * 100) if total_count > 0 else 0
                
                hl_stats = {
                    "Pattern": ["Higher H/L", "Not Higher H/L", "Total"],
                    "Count": [
                        f"{higher_hl_count}",
                        f"{total_count - higher_hl_count}",
                        f"{total_count}"
                    ],
                    "Percentage": [
                        f"{higher_hl_pct:.1f}%",
                        f"{100 - higher_hl_pct:.1f}%",
                        "100.0%"
                    ]
                }
                st.dataframe(pd.DataFrame(hl_stats), hide_index=True, use_container_width=True)
        else:
            st.markdown("**Additional Filters:**")
            st.info("Higher H/L filter not needed - already filtered by base selection")
            selected_higher_hl = "No Filter"
    
    # COL 4: MULTI-SELECT MPI EXPANSION FILTER (PURE VELOCITY-BASED)
    with col4:
        st.markdown("**MPI Expansion Filter:**")
        
        # Add toggle to enable/disable MPI filtering
        use_mpi_filter = st.checkbox("Enable MPI Filtering", value=True, key="mpi_filter_toggle")
        
        if use_mpi_filter and 'MPI_Trend' in filtered_stocks.columns:
            # MPI Expansion trend options (velocity-based, not threshold-based)
            trend_options = [
                "🚀 Strong Expansion",      # MPI Velocity ≥ 5%
                "📈 Expanding",            # MPI Velocity > 0%  
                "➖ Flat",                 # MPI Velocity = 0%
                "⚠️ Mild Contraction",     # MPI Velocity < 0% but > -5%
                "📉 Strong Contraction"    # MPI Velocity ≤ -5%
            ]
            
            # Multi-select checkboxes
            st.markdown("Select momentum trends:")
            selected_trends = []
            
            for i, trend in enumerate(trend_options):
                if st.checkbox(trend, key=f"mpi_trend_checkbox_{i}"):
                    # Remove emoji from trend name for internal use
                    trend_name = trend.split(' ', 1)[1] if ' ' in trend else trend
                    selected_trends.append(trend_name)
            
            # Apply MPI expansion filtering
            if selected_trends:
                filtered_stocks = apply_mpi_expansion_filter(filtered_stocks, selected_trends)
                
                # Show info about selected trends
                trend_emojis = [trend.split(' ')[0] for trend in trend_options if trend.split(' ', 1)[1] in selected_trends]
                st.info(f"Selected: {', '.join(trend_emojis)} ({len(filtered_stocks)} stocks)")
                
                # Show detailed explanations for selected trends
                with st.expander("Selected Trend Details", expanded=False):
                    for trend in selected_trends:
                        if trend == "Strong Expansion":
                            st.write("🚀 **Strong Expansion:** MPI Velocity ≥ 5% (Strong momentum building)")
                        elif trend == "Expanding":
                            st.write("📈 **Expanding:** MPI Velocity > 0% (Momentum improving)")
                        elif trend == "Flat":
                            st.write("➖ **Flat:** MPI Velocity = 0% (No momentum change)")
                        elif trend == "Mild Contraction":
                            st.write("⚠️ **Mild Contraction:** MPI Velocity > -5% (Slight weakening)")
                        elif trend == "Strong Contraction":
                            st.write("📉 **Strong Contraction:** MPI Velocity ≤ -5% (Momentum lost)")
            else:
                st.info("No trends selected - showing all MPI levels")
            
            # Show MPI Trend statistics and distribution
            with st.expander("MPI Trend Statistics", expanded=False):
                if 'MPI_Trend' in base_stocks.columns:
                    trend_counts = base_stocks['MPI_Trend'].value_counts()
                    
                    # Create comprehensive trend statistics
                    trend_stats_data = []
                    trend_order = [
                        'Strong Expansion', 'Expanding', 'Flat',
                        'Mild Contraction', 'Strong Contraction'
                    ]
                    
                    for trend in trend_order:
                        count = trend_counts.get(trend, 0)
                        if count > 0:
                            trend_stocks = base_stocks[base_stocks['MPI_Trend'] == trend]
                            avg_mpi = trend_stocks['MPI'].mean()
                            avg_velocity = trend_stocks['MPI_Velocity'].mean()
                            
                            trend_stats_data.append({
                                "MPI Trend": trend,
                                "Count": f"{count}",
                                "Avg MPI": f"{avg_mpi:.1%}",
                                "Avg Velocity": f"{avg_velocity:+.1%}",
                                "Action": get_trend_description(trend)
                            })
                    
                    if trend_stats_data:
                        st.dataframe(pd.DataFrame(trend_stats_data), hide_index=True, use_container_width=True)
                else:
                    st.warning("MPI_Trend column not available")
        elif not use_mpi_filter:
            st.info("MPI filtering disabled - showing all stocks")
            selected_trends = []
        else:
            st.warning("MPI trend filtering not available - MPI_Trend data missing")
            selected_trends = []
    
    # Show combined filter summary
    filter_summary = []
    if 'selected_percentile' in locals() and selected_percentile != "No Filter" and len(non_zero_velocities) > 0:
        filter_summary.append(f"VCRE Velocity {selected_percentile}")
    if selected_ibs_option == "Custom":
        filter_summary.append(f"IBS ≥ {custom_ibs_value:.2f}")
    elif selected_ibs_option != "No Filter":
        filter_summary.append(f"IBS {selected_ibs_option}")
    if 'selected_higher_hl' in locals() and selected_higher_hl != "No Filter":
        filter_summary.append("Higher H/L Only")
    if use_mpi_filter and selected_trends:
        filter_summary.append(f"MPI: {len(selected_trends)} trends")
    elif not use_mpi_filter:
        filter_summary.append("MPI: Disabled")
    
    if filter_summary:
        st.success(f"Active filters: {' + '.join(filter_summary)} → {len(filtered_stocks)} stocks")
    else:
        st.info(f"No additional filters applied → {len(filtered_stocks)} stocks")
    
    # Show distribution charts
    with st.expander("📊 Distribution Analysis", expanded=False):
        charts_to_show = []
        
        # Velocity chart (if available)
        if 'non_zero_velocities' in locals() and len(non_zero_velocities) > 0:
            charts_to_show.append(('velocity', non_zero_velocities, selected_percentile if 'selected_percentile' in locals() else None))
        
        # IBS chart
        charts_to_show.append(('ibs', filtered_stocks['IBS'] if not filtered_stocks.empty else pd.Series(), selected_ibs_option))
        
        # MPI Velocity chart (if available)
        if 'MPI_Velocity' in base_stocks.columns:
            charts_to_show.append(('mpi_velocity', base_stocks['MPI_Velocity'], selected_trends))
        
        # Create charts in rows of 2
        num_charts = len(charts_to_show)
        for i in range(0, num_charts, 2):
            cols = st.columns(2)
            for j, (chart_type, data, filter_selection) in enumerate(charts_to_show[i:i+2]):
                with cols[j]:
                    if chart_type == 'velocity' and len(data) > 0:
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="VCRE Velocity Distribution",
                            labels={'x': 'VCRE Velocity (pp)', 'y': 'Count'}
                        )
                        if filter_selection and filter_selection != "No Filter":
                            percentile_val = percentile_options[filter_selection]
                            threshold_value = np.percentile(non_zero_velocities, percentile_val)
                            fig.add_vline(x=threshold_value, line_dash="dash", line_color="red", 
                                         annotation_text=f"{filter_selection} threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'ibs' and len(data) > 0:
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="IBS Distribution",
                            labels={'x': 'IBS', 'y': 'Count'}
                        )
                        if filter_selection == "Custom":
                            if 'custom_ibs_value' in locals():
                                fig.add_vline(x=custom_ibs_value, line_dash="dash", line_color="red",
                                            annotation_text=f"Custom ≥ {custom_ibs_value:.2f}")
                        elif filter_selection != "No Filter" and filter_selection in ibs_percentile_options:
                            percentile_val = ibs_percentile_options[filter_selection]
                            if percentile_val is not None:
                                threshold_value = np.percentile(data, percentile_val)
                                fig.add_vline(x=threshold_value, line_dash="dash", line_color="red",
                                            annotation_text=f"{filter_selection} threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'mpi_velocity' and len(data) > 0:
                        # Create histogram colored by MPI_Trend
                        df_for_plot = base_stocks[['MPI_Velocity', 'MPI_Trend']].copy()
                        
                        fig = px.histogram(
                            df_for_plot,
                            x='MPI_Velocity',
                            color='MPI_Trend',
                            nbins=30,
                            title="MPI Velocity Distribution - Pure Expansion Focus",
                            labels={'MPI_Velocity': 'MPI Velocity (Day-over-Day Change)', 'count': 'Count'},
                            color_discrete_map={
                                'Strong Expansion': 'darkgreen',
                                'Expanding': 'green',
                                'Flat': 'gray',
                                'Mild Contraction': 'orange',
                                'Strong Contraction': 'red'
                            }
                        )
                        
                        # Add threshold lines
                        fig.add_vline(x=0.05, line_dash="dash", line_color="darkgreen", 
                                     annotation_text="Strong Expansion (≥5%)")
                        fig.add_vline(x=0, line_dash="solid", line_color="black", 
                                     annotation_text="Zero Velocity")
                        fig.add_vline(x=-0.05, line_dash="dash", line_color="red", 
                                     annotation_text="Strong Contraction (≤-5%)")
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    return filtered_stocks

# File: pages/scanner_higher_hl.py
# Part 3 of 3

def show():
    """Main scanner page display for Higher H/L patterns with Pure MPI Expansion filtering"""
    
    st.title("📈 CRT Higher H/L Scanner")
    st.markdown("Enhanced with **Pure MPI Expansion System** - Focus on momentum velocity, not absolute levels")
    
    # Clear previous errors for new scan
    if st.button("🗑️ Clear Error Log"):
        st.session_state.error_logger = ErrorLogger()
        st.success("Error log cleared!")
        st.rerun()
    
    # Check if core modules are available
    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher, get_company_name
        from core.technical_analysis import add_enhanced_columns
        from utils.watchlist import get_active_watchlist
        modules_available = True
        st.session_state.error_logger.log_debug("Module Check", "All core modules imported successfully")
    except ImportError as e:
        st.session_state.error_logger.log_error("Module Import", e, {
            'missing_modules': ['core.data_fetcher', 'core.technical_analysis', 'utils.watchlist'],
            'python_path': sys.path
        })
        st.error(f"❌ Import error: {e}")
        modules_available = False
    
    # Display any existing errors
    st.session_state.error_logger.display_errors_in_streamlit()
    
    # Scanning Configuration Panel
    st.subheader("🎯 Scanning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Scan Scope**")
        scan_scope = st.radio(
            "Choose scan scope:",
            ["Single Stock", "Full Watchlist"],
            help="Select whether to scan one stock or the entire watchlist"
        )
        
        if scan_scope == "Single Stock":
            try:
                if modules_available:
                    watchlist = get_active_watchlist()
                    st.session_state.error_logger.log_debug("Watchlist", f"Loaded {len(watchlist)} stocks", watchlist)
                else:
                    watchlist = ['A17U.SI', 'BN2.SI', 'C52.SI', 'E28.SI', 'G13.SI']
                    st.session_state.error_logger.log_warning("Watchlist", "Using fallback watchlist due to missing modules")
                
                selected_stock = st.selectbox(
                    "Select Stock:",
                    options=watchlist,
                    help="Choose which stock to analyze"
                )
                st.session_state.error_logger.log_debug("Stock Selection", f"Selected: {selected_stock}")
                
            except Exception as e:
                st.session_state.error_logger.log_error("Watchlist Loading", e)
                selected_stock = "A17U.SI"
        
    with col2:
        st.markdown("**📅 Analysis Date**")
        scan_date_type = st.radio(
            "Choose analysis date:",
            ["Current Date", "Historical Date"],
            help="Scan as of current date or specify a historical date"
        )
        
        if scan_date_type == "Historical Date":
            try:
                default_date = date.today() - timedelta(days=7)
                historical_date = st.date_input(
                    "Analysis Date:",
                    value=default_date,
                    max_value=date.today() - timedelta(days=1),
                    help="Choose the historical date for analysis (must be a past trading day)"
                )
                st.session_state.error_logger.log_debug("Date Selection", f"Historical date: {historical_date}")
            except Exception as e:
                st.session_state.error_logger.log_error("Date Selection", e)
                historical_date = date.today() - timedelta(days=7)
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days_back = st.number_input(
                "Days of Historical Data", 
                min_value=30, 
                max_value=250, 
                value=59,
                help="How many days of data to download for analysis"
            )
        
        with col2:
            rolling_window = st.number_input(
                "Rolling Window", 
                min_value=10, 
                max_value=50, 
                value=20,
                help="Rolling window for technical calculations"
            )
        
        with col3:
            debug_mode = st.checkbox(
                "Enable Debug Mode",
                value=False,
                help="Show detailed debug information during scanning"
            )
    
    # Scan Execution
    st.subheader("🚀 Execute Scan")
    
    if st.button("🚀 Execute Scan", type="primary", use_container_width=True):
        st.session_state.error_logger = ErrorLogger()
        
        if modules_available:
            try:
                if scan_scope == "Single Stock":
                    stocks_to_scan = [selected_stock]
                else:
                    stocks_to_scan = get_active_watchlist()
                
                if scan_date_type == "Historical Date":
                    analysis_date = historical_date
                else:
                    analysis_date = None
                
                run_enhanced_stock_scan(
                    stocks_to_scan=stocks_to_scan,
                    analysis_date=analysis_date,
                    days_back=days_back,
                    rolling_window=rolling_window,
                    debug_mode=debug_mode
                )
                
            except Exception as e:
                st.session_state.error_logger.log_error("Scan Execution", e, {
                    "scan_scope": scan_scope,
                    "scan_date_type": scan_date_type,
                    "selected_stock": selected_stock if scan_scope == "Single Stock" else None
                })
                st.error("❌ Failed to execute scan - check error details above")
        else:
            st.warning("Cannot execute scan - required modules are not available")
    
    # Display last scan info if available
    if 'last_scan_time' in st.session_state:
        st.info(f"📊 Last scan completed: {st.session_state.last_scan_time}")
        
        if 'last_scan_config' in st.session_state:
            config = st.session_state.last_scan_config
            st.caption(f"Scope: {config['scope']} | Date: {config['date']} | Stocks: {config['stock_count']}")
    
    # Display results if available
    if 'scan_results' in st.session_state:
        display_scan_results(st.session_state.scan_results)

def run_enhanced_stock_scan(stocks_to_scan, analysis_date=None, days_back=59, rolling_window=20, debug_mode=False):
    """Execute the enhanced stock scanning process with Pure MPI Expansion system integration"""
    
    error_logger = st.session_state.error_logger
    
    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher
        from core.technical_analysis import add_enhanced_columns  # Remove get_mpi_trend_emoji
        
        error_logger.log_debug("Scan Start", "Starting enhanced stock scan with Pure MPI Expansion", {
            "stocks_count": len(stocks_to_scan),
            "stocks": stocks_to_scan,
            "analysis_date": str(analysis_date) if analysis_date else "Current",
            "days_back": days_back,
            "rolling_window": rolling_window
        })
        
        is_historical = analysis_date is not None
        is_single_stock = len(stocks_to_scan) == 1
        
        if is_single_stock:
            scope_text = f"single stock ({stocks_to_scan[0]})"
        else:
            scope_text = f"{len(stocks_to_scan)} stocks"
        
        if is_historical:
            date_text = f"historical analysis (as of {analysis_date})"
        else:
            date_text = "current data analysis"
        
        st.info(f"🔄 Scanning {scope_text} with {date_text}... Calculating Pure MPI Expansion...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔧 Initializing data fetcher...")
        try:
            fetcher = DataFetcher(days_back=days_back)
            error_logger.log_debug("Data Fetcher", f"Initialized with {days_back} days back")
        except Exception as e:
            error_logger.log_error("Data Fetcher Initialization", e, {"days_back": days_back})
            raise
        
        progress_bar.progress(0.1)
        
        status_text.text("📥 Downloading stock data and company names...")
        try:
            stock_data = fetcher.download_stock_data(stocks_to_scan)
            error_logger.log_debug("Data Download", f"Downloaded data for {len(stock_data)} stocks", {
                "requested_stocks": len(stocks_to_scan),
                "successful_downloads": len(stock_data),
                "success_rate": f"{len(stock_data)/len(stocks_to_scan)*100:.1f}%"
            })
            
            missing_stocks = [stock for stock in stocks_to_scan if stock not in stock_data]
            if missing_stocks:
                error_logger.log_warning("Data Download", f"Failed to download {len(missing_stocks)} stocks", {
                    "missing_stocks": missing_stocks
                })
                
        except Exception as e:
            error_logger.log_error("Data Download", e, {
                "stocks_to_scan": stocks_to_scan,
                "fetcher_days_back": days_back
            })
            raise
        
        set_global_data_fetcher(fetcher)
        progress_bar.progress(0.3)
        
        if not stock_data:
            error_msg = "No stock data was downloaded successfully"
            error_logger.log_error("Data Validation", Exception(error_msg), {
                "stocks_requested": stocks_to_scan,
                "data_received": stock_data
            })
            st.error("❌ Failed to download stock data. Check error log for details.")
            return
        
        status_text.text("🔄 Calculating Pure MPI Expansion and technical analysis...")
        progress_bar.progress(0.4)
        
        results = []
        processing_errors = []
        
        for i, (ticker, df_raw) in enumerate(stock_data.items()):
            try:
                error_logger.log_debug("Stock Processing", f"Processing {ticker} with Pure MPI Expansion", {
                    "data_shape": df_raw.shape,
                    "date_range": f"{df_raw.index[0]} to {df_raw.index[-1]}" if len(df_raw) > 0 else "No data",
                    "columns": list(df_raw.columns)
                })
                
                if df_raw.empty:
                    error_logger.log_warning("Stock Processing", f"Empty dataframe for {ticker}")
                    continue
                
                try:
                    df_enhanced = add_enhanced_columns(df_raw, ticker, rolling_window)
                    error_logger.log_debug("Technical Analysis", f"Enhanced columns with Pure MPI Expansion added for {ticker}", {
                        "enhanced_shape": df_enhanced.shape,
                        "mpi_columns": [col for col in df_enhanced.columns if 'MPI' in col],
                        "has_mpi_trend": 'MPI_Trend' in df_enhanced.columns
                    })
                except Exception as e:
                    error_logger.log_error("Technical Analysis", e, {
                        "ticker": ticker,
                        "raw_data_shape": df_raw.shape,
                        "rolling_window": rolling_window
                    })
                    processing_errors.append(f"{ticker}: Technical analysis failed")
                    continue
                
                if is_historical:
                    try:
                        target_date = pd.to_datetime(analysis_date)
                        available_dates = df_enhanced.index
                        
                        if hasattr(available_dates, 'tz') and available_dates.tz is not None:
                            if target_date.tz is None:
                                target_date = target_date.tz_localize('Asia/Singapore')
                            else:
                                target_date = target_date.tz_convert(available_dates.tz)
                        else:
                            if target_date.tz is not None:
                                target_date = target_date.tz_localize(None)
                        
                        valid_dates = available_dates[available_dates <= target_date]
                        
                        error_logger.log_debug("Historical Date Processing", f"Date comparison for {ticker}", {
                            "target_date": str(target_date),
                            "available_dates_count": len(available_dates),
                            "valid_dates_count": len(valid_dates)
                        })
                        
                        if len(valid_dates) == 0:
                            error_logger.log_warning("Historical Analysis", f"No data for {ticker} on or before {analysis_date}", {
                                "target_date": str(target_date),
                                "available_date_range": f"{available_dates[0]} to {available_dates[-1]}" if len(available_dates) > 0 else "No dates"
                            })
                            continue
                        
                        analysis_row = df_enhanced.loc[valid_dates[-1]]
                        actual_date = valid_dates[-1]
                        
                    except Exception as e:
                        error_logger.log_error("Historical Date Processing", e, {
                            "ticker": ticker,
                            "target_date": str(analysis_date)
                        })
                        continue
                else:
                    analysis_row = df_enhanced.iloc[-1]
                    actual_date = analysis_row.name
                
                try:
                    company_name = fetcher.get_company_name(ticker)
                    error_logger.log_debug("Company Name", f"Retrieved name for {ticker}: {company_name}")
                except Exception as e:
                    error_logger.log_warning("Company Name", f"Failed to get name for {ticker}: {e}")
                    company_name = ticker.replace('.SI', '')
                
                # Dynamic decimal formatting based on price level
                close_price = float(analysis_row['Close'])
                high_price = float(analysis_row['High'])
                low_price = float(analysis_row['Low'])
                
                if close_price < 1.00:
                    price_decimals = 3
                else:
                    price_decimals = 2
                
                def safe_round(value, decimals):
                    try:
                        return round(float(value), decimals) if not pd.isna(value) else 0
                    except:
                        return 0
                
                # Get MPI values
                mpi_value = float(analysis_row.get('MPI', 0.5)) if not pd.isna(analysis_row.get('MPI', 0.5)) else 0.5
                mpi_velocity = float(analysis_row.get('MPI_Velocity', 0.0)) if not pd.isna(analysis_row.get('MPI_Velocity', 0.0)) else 0.0
                mpi_trend = str(analysis_row.get('MPI_Trend', 'Unknown'))
                
                # Collect results with PURE MPI EXPANSION data
                try:
                    result = {
                        'Ticker': ticker,
                        'Name': company_name,
                        'Analysis_Date': actual_date.strftime('%Y-%m-%d') if hasattr(actual_date, 'strftime') else str(actual_date),
                        'Close': round(close_price, price_decimals),
                        'High': round(high_price, price_decimals),
                        'Low': round(low_price, price_decimals),
                        'IBS': round(float(analysis_row['IBS']), 3) if not pd.isna(analysis_row['IBS']) else 0,
                        'Valid_CRT': int(analysis_row.get('Valid_CRT', 0)),
                        'Higher_HL': int(analysis_row.get('Higher_HL', 0)) if not pd.isna(analysis_row.get('Higher_HL', 0)) else 0,
                        'VCRE_Velocity': round(float(analysis_row.get('VCRE_Velocity', 0)), 4) if not pd.isna(analysis_row.get('VCRE_Velocity', 0)) else 0,
                        'Weekly_Open': safe_round(analysis_row.get('Weekly_Open', 0), price_decimals),
                        'CRT_High': safe_round(analysis_row.get('CRT_High', 0), price_decimals),
                        'CRT_Low': safe_round(analysis_row.get('CRT_Low', 0), price_decimals),
                        'VW_Range_Percentile': round(float(analysis_row.get('VW_Range_Percentile', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Percentile', 0)) else 0,
                        'Rel_Range_Signal': int(analysis_row.get('Rel_Range_Signal', 0)),
                        
                        # PURE MPI EXPANSION COLUMNS
                        'MPI': round(mpi_value, 4),
                        'MPI_Velocity': round(mpi_velocity, 4),
                        'MPI_Trend': mpi_trend,
                        'MPI_Trend_Emoji': str(analysis_row.get('MPI_Trend_Emoji', '❓')),
                        'MPI_Visual': format_mpi_visual(mpi_value),
                        
                        # Store price level for display formatting
                        'Price_Decimals': price_decimals
                    }
                    
                    results.append(result)
                    
                    if debug_mode:
                        error_logger.log_debug("Result Collection", f"Collected Pure MPI Expansion result for {ticker}", {
                            "mpi": mpi_value,
                            "mpi_velocity": mpi_velocity,
                            "mpi_trend": mpi_trend,
                            "mpi_visual": format_mpi_visual(mpi_value),
                            "vcre_velocity": analysis_row.get('VCRE_Velocity', 0.0),
                            "price_decimals": price_decimals
                        })
                        
                except Exception as e:
                    error_logger.log_error("Result Collection", e, {
                        "ticker": ticker,
                        "analysis_row_index": str(actual_date)
                    })
                    processing_errors.append(f"{ticker}: Result collection failed")
                    continue
                
                progress = 0.4 + (0.5 * (i + 1) / len(stock_data))
                progress_bar.progress(progress)
                
            except Exception as e:
                error_logger.log_error("Stock Processing", e, {
                    "ticker": ticker,
                    "processing_step": "overall"
                })
                processing_errors.append(f"{ticker}: {str(e)}")
                continue
        
        if processing_errors:
            error_logger.log_warning("Processing Summary", f"{len(processing_errors)} stocks had processing errors", {
                "errors": processing_errors,
                "success_rate": f"{len(results)}/{len(stock_data)} ({len(results)/len(stock_data)*100:.1f}%)"
            })
        
        status_text.text("📊 Preparing Pure MPI Expansion results...")
        progress_bar.progress(0.9)
        
        try:
            results_df = pd.DataFrame(results)
            error_logger.log_debug("Results Finalization", f"Created Pure MPI Expansion results dataframe", {
                "results_count": len(results_df),
                "mpi_columns": [col for col in results_df.columns if 'MPI' in col] if not results_df.empty else []
            })
        except Exception as e:
            error_logger.log_error("Results DataFrame Creation", e, {
                "results_data_sample": results[:3] if results else []
            })
            raise
        
        st.session_state.scan_results = results_df
        st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.last_scan_config = {
            'scope': 'Single Stock' if is_single_stock else 'Full Watchlist',
            'date': f'Historical ({analysis_date})' if is_historical else 'Current',
            'stock_count': len(results_df)
        }
        
        progress_bar.progress(1.0)
        status_text.text("✅ Pure MPI Expansion scan completed!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        success_message = f"🎉 Pure MPI Expansion Scan completed! Analyzed {len(results_df)} stocks successfully"
        if processing_errors:
            success_message += f" ({len(processing_errors)} errors - check log for details)"
        if is_historical:
            success_message += f" as of {analysis_date}"
        
        st.success(success_message)
        error_logger.log_debug("Scan Completion", "Pure MPI Expansion scan completed successfully", {
            "successful_stocks": len(results_df),
            "total_requested": len(stocks_to_scan),
            "processing_errors": len(processing_errors)
        })
        
        st.rerun()
        
    except Exception as e:
        error_logger.log_error("Scan Execution", e, {
            "scan_parameters": {
                "stocks_to_scan": stocks_to_scan,
                "analysis_date": str(analysis_date) if analysis_date else None,
                "days_back": days_back,
                "rolling_window": rolling_window
            }
        })
        st.error("❌ Pure MPI Expansion scan failed with critical error - check error log for full details")
        
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def display_scan_results(results_df: pd.DataFrame):
    """Display scanning results with Pure MPI Expansion filtering and velocity-based display"""
    
    error_logger = st.session_state.error_logger
    
    try:
        if results_df.empty:
            st.warning("No results to display.")
            error_logger.log_warning("Results Display", "Empty results dataframe")
            return
        
        # Summary metrics with PURE MPI EXPANSION STATISTICS
        st.subheader("📊 Scan Summary with Pure MPI Expansion Analysis")
        
        total_stocks = len(results_df)
        higher_hl_count = len(results_df[results_df['Higher_HL'] == 1])
        valid_crt_count = len(results_df[results_df['Valid_CRT'] == 1])
        higher_hl_with_crt = len(results_df[(results_df['Higher_HL'] == 1) & (results_df['Valid_CRT'] == 1)])
        
        # Pure MPI Expansion statistics
        if 'MPI_Trend' in results_df.columns:
            trend_counts = results_df['MPI_Trend'].value_counts()
            strong_expansion = trend_counts.get('Strong Expansion', 0)
            expanding = trend_counts.get('Expanding', 0)
            contracting = trend_counts.get('Strong Contraction', 0) + trend_counts.get('Mild Contraction', 0)
            avg_mpi = results_df['MPI'].mean()
            avg_velocity = results_df['MPI_Velocity'].mean()
        else:
            strong_expansion = expanding = contracting = 0
            avg_mpi = 0.5
            avg_velocity = 0.0
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Analyzed", total_stocks)
        with col2:
            st.metric("Higher H/L", higher_hl_count, delta=f"{higher_hl_count/total_stocks*100:.1f}%")
        with col3:
            st.metric("Valid CRT", valid_crt_count)
        with col4:
            st.metric("Both Patterns", higher_hl_with_crt)
        with col5:
            st.metric("🚀 Strong Exp", strong_expansion, delta="Velocity ≥5%")
        with col6:
            st.metric("📈 Expanding", expanding, delta="Velocity >0%")
        
        # Analysis Date Info with Pure MPI Expansion summary
        if len(results_df) > 0:
            analysis_dates = results_df['Analysis_Date'].unique()
            if len(analysis_dates) == 1:
                st.info(f"📅 Analysis date: **{analysis_dates[0]}** | Avg MPI: **{avg_mpi:.1%}** | Avg Velocity: **{avg_velocity:+.1%}** | Expanding: **{strong_expansion + expanding}** | Contracting: **{contracting}**")
            else:
                st.info(f"📅 Analysis dates: **{', '.join(analysis_dates)}** | Average MPI: **{avg_mpi:.1%}**")
        
        # Base Filter Selection (UNCHANGED)
        st.subheader("🎯 Pattern Analysis")
        
        base_filter_options = {
            "Valid CRT + Higher H/L": "Stocks with both Valid CRT AND Higher H/L patterns",
            "Valid CRT Only": "All stocks with Valid CRT (Monday range expansion)",
            "Higher H/L Only": "All stocks with Higher High AND Higher Low patterns",
            "All Stocks": "Complete scan results without pattern filtering"
        }
        
        selected_base_filter = st.radio(
            "Select base pattern filter:",
            list(base_filter_options.keys()),
            help="Choose which pattern combination to analyze",
            key="base_filter_radio"
        )
        
        # Store in session state for filter logic
        st.session_state.base_filter_type = selected_base_filter
        
        # Apply base filter
        if selected_base_filter == "Valid CRT + Higher H/L":
            base_stocks = results_df[(results_df['Valid_CRT'] == 1) & (results_df['Higher_HL'] == 1)].copy()
            st.info(f"Showing {len(base_stocks)} stocks with both Valid CRT and Higher H/L patterns")
        
        elif selected_base_filter == "Valid CRT Only":
            base_stocks = results_df[results_df['Valid_CRT'] == 1].copy()
            st.info(f"Showing {len(base_stocks)} stocks with Valid CRT (Monday range expansion)")
        
        elif selected_base_filter == "Higher H/L Only":
            base_stocks = results_df[results_df['Higher_HL'] == 1].copy()
            st.info(f"Showing {len(base_stocks)} stocks with Higher High AND Higher Low patterns")
        
        else:  # All Stocks
            base_stocks = results_df.copy()
            st.info(f"Showing all {len(base_stocks)} scanned stocks")
        
        if len(base_stocks) > 0:
            # Apply dynamic filters (now includes Pure MPI Expansion Filter)
            filtered_stocks = apply_dynamic_filters(base_stocks, results_df)
            
            # Display filtered results
            st.subheader(f"📋 Pure MPI Expansion Results ({len(filtered_stocks)} stocks)")
            
            if len(filtered_stocks) > 0:
                # Display columns in FILTER SEQUENCE ORDER with MPI Expansion focus
                display_cols = ['Ticker', 'Name', 'Close', 'CRT_High', 'CRT_Low',
                               'VCRE_Velocity', 'IBS', 'Higher_HL', 
                               'MPI_Trend_Emoji', 'MPI', 'MPI_Velocity', 'MPI_Visual']
                
                base_column_config = {
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Name': st.column_config.TextColumn('Company Name', width='medium'),
                    'VCRE_Velocity': st.column_config.NumberColumn('VCRE Vel', format='%+.4f'),
                    'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                    'Higher_HL': st.column_config.NumberColumn('H/L', width='small'),
                    'MPI': st.column_config.NumberColumn('MPI', format='%.0%%', help='Market Positivity Index'),
                    'MPI': st.column_config.NumberColumn('MPI', format='%.0%%', help='Market Positivity Index'),
                    'MPI_Velocity': st.column_config.NumberColumn('MPI Vel', format='%+.0%%', help='MPI Velocity (Day-over-Day Change)'),
                    'MPI_Visual': st.column_config.TextColumn('MPI Visual', width='medium', help='Visual MPI representation')
                }
                
                # Apply dynamic price formatting
                column_config = get_dynamic_column_config(filtered_stocks, display_cols, base_column_config)
                
                # Filter display_cols to only include existing columns
                display_cols = [col for col in display_cols if col in filtered_stocks.columns]
                
                # Verify we have all required columns before displaying
                missing_cols = [col for col in display_cols if col not in filtered_stocks.columns]
                if missing_cols:
                    st.error(f"❌ Missing columns in filtered results: {missing_cols}")
                    st.write("Available columns:", list(filtered_stocks.columns))
                else:
                    try:
                        st.dataframe(
                            filtered_stocks[display_cols],
                            column_config=column_config,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # TradingView Export (ENHANCED with Pure MPI Expansion info)
                        st.subheader("📋 TradingView Export (Pure MPI Expansion Filtered)")
                        tv_tickers = [f"SGX:{ticker.replace('.SI', '')}" for ticker in filtered_stocks['Ticker'].tolist()]
                        tv_string = ','.join(tv_tickers)
                        
                        # Calculate Pure MPI Expansion summary for export description
                        if 'MPI_Trend' in filtered_stocks.columns:
                            trend_summary = filtered_stocks['MPI_Trend'].value_counts()
                            top_trend = trend_summary.index[0] if len(trend_summary) > 0 else "Mixed"
                            mpi_summary = f"Top Trend: {top_trend}"
                            if 'MPI_Velocity' in filtered_stocks.columns:
                                mpi_summary += f" | Avg Velocity: {filtered_stocks['MPI_Velocity'].mean():+.1%}"
                        else:
                            mpi_summary = ""
                        
                        st.text_area(
                            f"Singapore Exchange (SGX) - {selected_base_filter} ({len(tv_tickers)} stocks) {mpi_summary}:",
                            value=tv_string,
                            height=100,
                            help="Copy and paste into TradingView watchlist. Sorted by MPI Expansion trends."
                        )
                        
                        # Export filtered data with Pure MPI Expansion
                        csv_data = filtered_stocks.to_csv(index=False)
                        filename_prefix = selected_base_filter.lower().replace(' ', '_').replace('+', 'and')
                        st.download_button(
                            label="📥 Download Pure MPI Expansion Data (CSV)",
                            data=csv_data,
                            file_name=f"pure_mpi_{filename_prefix}_{len(filtered_stocks)}_stocks.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error displaying filtered results: {str(e)}")
                        error_logger.log_error("Filtered Results Display", e, {
                            "filtered_stocks_shape": filtered_stocks.shape,
                            "display_cols": display_cols,
                            "missing_cols": missing_cols if 'missing_cols' in locals() else []
                        })
            
            else:
                st.warning("No stocks match the current filter criteria")
        
        else:
            st.warning(f"No stocks found for pattern: {selected_base_filter}")
        
        # Full Results Table with PURE MPI EXPANSION SYSTEM
        with st.expander("📋 Full Pure MPI Expansion Analysis Results", expanded=False):
            try:
                full_results_cols = [
                    'Analysis_Date', 'Ticker', 'Name', 'Close', 'CRT_High', 'CRT_Low',
                    'VCRE_Velocity', 'IBS', 'Higher_HL', 'Valid_CRT', 
                    'MPI_Trend_Emoji', 'MPI_Trend', 'MPI', 'MPI_Velocity', 'MPI_Visual'
                ]
                
                base_full_results_config = {
                    'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Name': st.column_config.TextColumn('Company Name', width='medium'),
                    'VCRE_Velocity': st.column_config.NumberColumn('VCRE Vel', format='%+.4f'),
                    'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                    'Higher_HL': st.column_config.NumberColumn('H/L', width='small'),
                    'Valid_CRT': st.column_config.NumberColumn('CRT', width='small'),
                    'MPI_Trend_Emoji': st.column_config.TextColumn('📊', width='small'),
                    'MPI_Trend': st.column_config.TextColumn('MPI Trend', width='medium'),
                    'MPI': st.column_config.NumberColumn('MPI', format='%.1%'),
                    'MPI_Velocity': st.column_config.NumberColumn('MPI Velocity', format='%+.1%'),
                    'MPI_Visual': st.column_config.TextColumn('MPI Visual', width='medium')
                }
                
                # Apply dynamic price formatting for full results
                full_results_column_config = get_dynamic_column_config(results_df, full_results_cols, base_full_results_config)
                
                # Filter to only existing columns
                full_results_cols = [col for col in full_results_cols if col in results_df.columns]
                
                # Check for missing columns
                missing_cols = [col for col in ['Ticker', 'Name', 'Close', 'MPI'] if col not in results_df.columns]
                if missing_cols:
                    st.error(f"❌ Missing essential columns in results: {missing_cols}")
                    st.write("Available columns:", list(results_df.columns))
                else:
                    st.dataframe(
                        results_df[full_results_cols],
                        column_config=full_results_column_config,
                        use_container_width=True,
                        hide_index=True
                    )
                    
            except Exception as e:
                st.error(f"❌ Error displaying full results: {str(e)}")
                error_logger.log_error("Full Results Display", e, {
                    "results_df_shape": results_df.shape if not results_df.empty else "Empty",
                    "attempted_columns": full_results_cols if 'full_results_cols' in locals() else "Unknown"
                })
        
        # Pure MPI Expansion Insights
        if 'MPI_Trend' in results_df.columns:
            with st.expander("📈 Pure MPI Expansion Insights", expanded=False):
                
                # Create comprehensive trend summary
                trend_summary = []
                trend_counts = results_df['MPI_Trend'].value_counts()
                
                for trend in ['Strong Expansion', 'Expanding', 'Flat', 'Mild Contraction', 'Strong Contraction']:
                    if trend in trend_counts.index:
                        count = trend_counts[trend]
                        trend_stocks = results_df[results_df['MPI_Trend'] == trend]
                        avg_mpi = trend_stocks['MPI'].mean()
                        avg_velocity = trend_stocks['MPI_Velocity'].mean()
                        
                        # Get top stock by velocity in this trend
                        if len(trend_stocks) > 0:
                            if trend in ['Strong Expansion', 'Expanding']:
                                top_stock_idx = trend_stocks['MPI_Velocity'].idxmax()
                            else:
                                top_stock_idx = trend_stocks['MPI'].idxmax()
                            top_stock = trend_stocks.loc[top_stock_idx, 'Name']
                        else:
                            top_stock = 'N/A'
                        
                        trend_summary.append({
                            'Trend': trend,
                            'Count': count,
                            'Avg MPI': f"{avg_mpi:.1%}",
                            'Avg Velocity': f"{avg_velocity:+.1%}",
                            'Top Stock': top_stock,
                            'Trading Action': get_trend_description(trend)
                        })
                
                if trend_summary:
                    st.dataframe(pd.DataFrame(trend_summary), hide_index=True, use_container_width=True)
                
                # MPI Velocity distribution chart
                if len(results_df) > 1:
                    fig = px.histogram(
                        results_df, 
                        x='MPI_Velocity',
                        color='MPI_Trend',
                        nbins=30,
                        title="Pure MPI Velocity Distribution - Focus on Momentum Direction",
                        labels={'MPI_Velocity': 'MPI Velocity (Day-over-Day Change)', 'count': 'Number of Stocks'},
                        color_discrete_map={
                            'Strong Expansion': 'darkgreen',
                            'Expanding': 'green',
                            'Flat': 'gray',
                            'Mild Contraction': 'orange',
                            'Strong Contraction': 'red'
                        }
                    )
                    
                    # Add velocity threshold lines
                    fig.add_vline(x=0.05, line_dash="dash", line_color="darkgreen", annotation_text="Strong Expansion (≥5%)")
                    fig.add_vline(x=0, line_dash="solid", line_color="black", annotation_text="Zero Velocity")
                    fig.add_vline(x=-0.05, line_dash="dash", line_color="red", annotation_text="Strong Contraction (≤-5%)")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Real example explanation
                with st.expander("💡 Understanding Pure MPI Expansion", expanded=False):
                    st.markdown("""
                    **Why Pure Expansion?**
                    
                    Traditional systems wait for MPI > 50% before buying. This misses huge moves:
                    
                    **Example: Stock recovering from oversold**
                    - Day 1: MPI = 20% (2 of 10 days green)
                    - Day 2: MPI = 30%, Velocity = +10% → **Strong Expansion Signal** 🚀
                    - Day 3: MPI = 50%, Velocity = +20% → Still expanding
                    - Day 4: MPI = 70%, Velocity = +20% → Strong trend continues
                    
                    Traditional system: Waits until Day 3 (50% threshold)
                    Pure MPI: Enters on Day 2 (catches 250% more of the move!)
                    
                    **The key insight:** It's not about WHERE momentum is, but WHERE IT'S GOING!
                    """)
        
        error_logger.log_debug("Results Display", "Successfully displayed Pure MPI Expansion results", {
            "total_displayed": len(results_df),
            "base_filter": selected_base_filter,
            "mpi_trend_available": 'MPI_Trend' in results_df.columns,
            "strong_expansion_count": strong_expansion
        })
        
    except Exception as e:
        error_logger.log_error("Results Display", e, {
            "results_shape": results_df.shape if not results_df.empty else "Empty DataFrame",
            "has_mpi_trend_columns": 'MPI_Trend' in results_df.columns if not results_df.empty else False
        })
        st.error("❌ Error displaying Pure MPI Expansion results - check error log for details")

if __name__ == "__main__":
    show()