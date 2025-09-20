# File: pages/scanner_higher_hl.py
# Part 1 of 4
"""
CRT Higher H/L Scanner - Optimized with Pure MPI Expansion Filtering
Enhanced error handling and streamlined UI components
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
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
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
            
            with st.expander("🔍 View Error Details", expanded=False):
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
    """Get the appropriate price format string based on price level"""
    return "$%.3f" if price < 1.00 else "$%.2f"

def get_dynamic_column_config(df: pd.DataFrame, display_cols: List[str], base_config: dict) -> dict:
    """Create dynamic column configuration with price-based decimal formatting"""
    config = base_config.copy()
    
    price_columns = ['Close', 'High', 'Low', 'CRT_High', 'CRT_Low', 'Weekly_Open']
    
    for col in price_columns:
        if col in display_cols and col in df.columns and len(df) > 0:
            min_price = df[col].min()
            format_str = "$%.3f" if min_price < 1.00 else "$%.2f"
            
            config[col] = st.column_config.NumberColumn(
                config[col].title if col in config and hasattr(config[col], 'title') else col.replace('_', ' ').title(),
                format=format_str
            )
    
    return config

def format_mpi_visual(mpi_value: float) -> str:
    """Convert MPI to visual blocks for intuitive display"""
    if pd.isna(mpi_value):
        return "░░░░░░░░░░"
    
    blocks = max(0, min(10, int(mpi_value * 10)))
    return "█" * blocks + "░" * (10 - blocks)

def get_mpi_expansion_description(trend: str) -> str:
    """Get trading description for MPI expansion trends"""
    descriptions = {
        'Strong Expansion': 'Strong momentum building',
        'Expanding': 'Positive momentum',
        'Flat': 'No momentum change',
        'Mild Contraction': 'Momentum weakening', 
        'Strong Contraction': 'Momentum declining'
    }
    return descriptions.get(trend, 'Unknown')

def apply_mpi_expansion_filter(base_stocks: pd.DataFrame, selected_trends: List[str]) -> pd.DataFrame:
    """Apply pure MPI expansion-based filtering with smart sorting"""
    if base_stocks.empty or 'MPI_Trend' not in base_stocks.columns:
        logger.warning("MPI_Trend column not found - returning all stocks")
        return base_stocks
    
    if not selected_trends:
        return base_stocks.sort_values('MPI', ascending=False)
    
    # Filter by selected MPI trends
    filtered = base_stocks[base_stocks['MPI_Trend'].isin(selected_trends)].copy()
    
    # Smart sorting based on selected trends
    expansion_trends = ['Strong Expansion', 'Expanding']
    contraction_trends = ['Mild Contraction', 'Strong Contraction']
    
    selected_categories = []
    if any(trend in expansion_trends for trend in selected_trends):
        selected_categories.append('expansion')
    if 'Flat' in selected_trends:
        selected_categories.append('flat')
    if any(trend in contraction_trends for trend in selected_trends):
        selected_categories.append('contraction')
    
    # Sort based on predominant selection
    if selected_categories == ['contraction']:
        filtered = filtered.sort_values('MPI_Velocity', ascending=True)
    elif selected_categories == ['expansion']:
        filtered = filtered.sort_values('MPI_Velocity', ascending=False)
    else:
        filtered = filtered.sort_values('MPI', ascending=False)
    
    return filtered

def create_filter_statistics_dataframe(data: pd.Series, metric_name: str) -> pd.DataFrame:
    """Create standardized statistics dataframe for filters"""
    if len(data) == 0:
        return pd.DataFrame()
    
    stats_data = {
        "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
        "Value": [
            f"{len(data)}",
            f"{data.min():.4f}",
            f"{np.percentile(data, 25):.4f}",
            f"{data.median():.4f}", 
            f"{np.percentile(data, 75):.4f}",
            f"{data.max():.4f}"
        ],
        "Unit": ["stocks", "pp", "pp", "pp", "pp", "pp"] if "Velocity" in metric_name else ["stocks", "", "", "", "", ""]
    }
    return pd.DataFrame(stats_data)

def create_distribution_chart(data: pd.Series, title: str, x_label: str, threshold_lines: dict = None):
    """Create standardized distribution charts"""
    if len(data) == 0:
        return None
    
    fig = px.histogram(
        x=data,
        nbins=20,
        title=title,
        labels={'x': x_label, 'y': 'Count'}
    )
    
    if threshold_lines:
        for label, (value, color) in threshold_lines.items():
            fig.add_vline(x=value, line_dash="dash", line_color=color, annotation_text=label)
    
    return fig

# File: pages/scanner_higher_hl.py
# Part 2 of 4
"""
Dynamic filtering functions - optimized and streamlined
"""

def apply_velocity_filter(filtered_stocks: pd.DataFrame, results_df: pd.DataFrame) -> tuple:
    """Apply CRT velocity percentile filter"""
    if filtered_stocks.empty:
        return filtered_stocks, None, "No stocks available"
    
    # Get velocity data
    if 'CRT_Velocity' in filtered_stocks.columns:
        velocities = filtered_stocks['CRT_Velocity']
    else:
        velocities = filtered_stocks.index.map(
            lambda idx: results_df.loc[idx, 'CRT_Velocity'] if idx in results_df.index else 0
        )
        velocities = pd.Series(velocities, index=filtered_stocks.index)
    
    # Remove zero velocities for percentile calculation
    non_zero_velocities = velocities[velocities != 0]
    
    if len(non_zero_velocities) == 0:
        return filtered_stocks, None, "No CRT Velocity data available"
    
    # Percentile options
    percentile_options = {
        "Top 25%": 75,
        "Top 50%": 50, 
        "Top 75%": 25,
        "No Filter": None
    }
    
    selected_percentile = st.radio(
        "Select velocity filter:",
        list(percentile_options.keys()),
        key="crt_velocity_percentile_radio"
    )
    
    # Apply filtering
    if selected_percentile != "No Filter":
        percentile_val = percentile_options[selected_percentile]
        threshold_value = np.percentile(non_zero_velocities, percentile_val)
        filtered_stocks = filtered_stocks[velocities >= threshold_value]
        info_message = f"CRT Velocity ≥ {threshold_value:+.4f} pp"
    else:
        info_message = "All velocities included"
    
    return filtered_stocks, non_zero_velocities, info_message

def apply_ibs_filter(filtered_stocks: pd.DataFrame) -> tuple:
    """Apply IBS percentile filter"""
    if filtered_stocks.empty:
        return filtered_stocks, "No stocks available"
    
    ibs_values = filtered_stocks['IBS']
    
    # Percentile options
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
    
    # Apply filtering
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
        info_message = f"IBS ≥ {custom_ibs_value:.2f}"
        
    elif selected_ibs_option != "No Filter":
        percentile_val = ibs_percentile_options[selected_ibs_option]
        threshold_value = np.percentile(ibs_values, percentile_val)
        filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= threshold_value]
        info_message = f"IBS ≥ {threshold_value:.3f} ({selected_ibs_option})"
        
    else:
        info_message = "All IBS values included"
    
    return filtered_stocks, info_message

def apply_higher_hl_filter(filtered_stocks: pd.DataFrame, base_filter_type: str) -> tuple:
    """Apply Higher H/L pattern filter if needed"""
    if base_filter_type in ["Valid CRT + Higher H/L", "Higher H/L Only"]:
        return filtered_stocks, "No Filter", "Filter not needed - already applied"
    
    higher_hl_options = {
        "Higher H/L Only": 1,
        "No Filter": None
    }
    
    selected_higher_hl = st.radio(
        "Select Higher H/L filter:",
        list(higher_hl_options.keys()),
        key="higher_hl_pattern_radio"
    )
    
    if selected_higher_hl != "No Filter":
        filtered_stocks = filtered_stocks[filtered_stocks['Higher_HL'] == 1]
        info_message = "Only showing Higher High/Low patterns"
    else:
        info_message = "All patterns included"
    
    return filtered_stocks, selected_higher_hl, info_message

def apply_mpi_filter(filtered_stocks: pd.DataFrame) -> tuple:
    """Apply MPI expansion trend filter"""
    if 'MPI_Trend' not in filtered_stocks.columns:
        st.warning("MPI expansion filtering not available - MPI_Trend data missing")
        return filtered_stocks, [], "MPI data unavailable"
    
    # Toggle to enable/disable MPI filtering
    use_mpi_filter = st.checkbox("Enable MPI Filtering", value=True, key="mpi_filter_toggle")
    
    if not use_mpi_filter:
        return filtered_stocks, [], "MPI filter disabled"
    
    # MPI Expansion trend options
    trend_options = [
        "📈 Expanding",
        "➖ Flat",
        "📉 Contracting"
    ]
    
    # Multi-select checkboxes for expansion trends
    st.markdown("Select momentum trends:")
    selected_trends = []
    
    trend_mapping = {
        "📈 Expanding": "Expanding",
        "➖ Flat": "Flat",
        "📉 Contracting": "Contracting"
    }
    
    for i, trend in enumerate(trend_options):
        if st.checkbox(trend, key=f"mpi_trend_checkbox_{i}"):
            selected_trends.append(trend_mapping[trend])
    
    # Apply MPI trend filtering
    if selected_trends:
        filtered_stocks = apply_mpi_expansion_filter(filtered_stocks, selected_trends)
        display_names = [k for k, v in trend_mapping.items() if v in selected_trends]
        info_message = f"Selected: {', '.join([name.split(' ')[0] for name in display_names])} ({len(filtered_stocks)} stocks)"
    else:
        info_message = "No trends selected - showing all MPI levels"
    
    return filtered_stocks, selected_trends, info_message

def show_filter_statistics(component_name: str, data: pd.Series, base_stocks: pd.DataFrame = None):
    """Show statistics for a filter component"""
    with st.expander(f"{component_name} Statistics", expanded=False):
        if component_name == "CRT Velocity" and len(data) > 0:
            stats_df = create_filter_statistics_dataframe(data, component_name)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
        elif component_name == "IBS" and len(data) > 0:
            stats_df = create_filter_statistics_dataframe(data, component_name)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
        elif component_name == "Higher H/L" and base_stocks is not None:
            higher_hl_count = (base_stocks['Higher_HL'] == 1).sum()
            total_count = len(base_stocks)
            higher_hl_pct = (higher_hl_count / total_count * 100) if total_count > 0 else 0
            
            hl_stats = pd.DataFrame({
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
            })
            st.dataframe(hl_stats, hide_index=True, use_container_width=True)
            
        elif component_name == "MPI Trend" and base_stocks is not None and 'MPI_Trend' in base_stocks.columns:
            trend_counts = base_stocks['MPI_Trend'].value_counts()
            trend_order = ['Expanding', 'Flat', 'Contracting']
            
            trend_stats_data = []
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
                        "Avg Velocity": f"{avg_velocity:+.3f}",
                        "Description": get_mpi_expansion_description(trend)
                    })
            
            if trend_stats_data:
                st.dataframe(pd.DataFrame(trend_stats_data), hide_index=True, use_container_width=True)

def apply_dynamic_filters(base_stocks: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dynamic filtering with optimized component structure
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
    filter_summary = []
    
    # COL 1: CRT VELOCITY FILTER
    with col1:
        st.markdown("**CRT Velocity Filter:**")
        filtered_stocks, velocity_data, velocity_info = apply_velocity_filter(filtered_stocks, results_df)
        st.info(velocity_info)
        
        if velocity_data is not None and len(velocity_data) > 0:
            show_filter_statistics("CRT Velocity", velocity_data)
            if "≥" in velocity_info:
                filter_summary.append("CRT Velocity filtered")
    
    # COL 2: IBS FILTER
    with col2:
        st.markdown("**IBS Filter:**")
        filtered_stocks, ibs_info = apply_ibs_filter(filtered_stocks)
        st.info(ibs_info)
        
        if len(filtered_stocks) > 0:
            show_filter_statistics("IBS", filtered_stocks['IBS'])
            if "≥" in ibs_info:
                filter_summary.append("IBS filtered")
    
    # COL 3: HIGHER H/L FILTER
    with col3:
        base_filter_type = st.session_state.get('base_filter_type', 'All Stocks')
        
        if base_filter_type not in ["Valid CRT + Higher H/L", "Higher H/L Only"]:
            st.markdown("**Higher H/L Filter:**")
            filtered_stocks, selected_higher_hl, hl_info = apply_higher_hl_filter(filtered_stocks, base_filter_type)
            st.info(hl_info)
            
            show_filter_statistics("Higher H/L", None, base_stocks)
            if selected_higher_hl != "No Filter":
                filter_summary.append("Higher H/L only")
        else:
            st.markdown("**Additional Filters:**")
            st.info("Higher H/L filter not needed - already filtered by base selection")
    
    # COL 4: MPI EXPANSION FILTER
    with col4:
        st.markdown("**MPI Expansion Filter:**")
        filtered_stocks, selected_trends, mpi_info = apply_mpi_filter(filtered_stocks)
        st.info(mpi_info)
        
        show_filter_statistics("MPI Trend", None, base_stocks)
        if selected_trends:
            filter_summary.append(f"MPI: {len(selected_trends)} trends")
        elif "disabled" in mpi_info:
            filter_summary.append("MPI: Disabled")
    
    # Show combined filter summary
    if filter_summary:
        st.success(f"Active filters: {' + '.join(filter_summary)} → {len(filtered_stocks)} stocks")
    else:
        st.info(f"No additional filters applied → {len(filtered_stocks)} stocks")
    
    return filtered_stocks

# File: pages/scanner_higher_hl.py
# Part 3 of 4
"""
Main scanning and result display functions - optimized
"""

def show_scanning_configuration():
    """Display the scanning configuration panel"""
    st.subheader("🎯 Scanning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Scan Scope**")
        scan_scope = st.radio(
            "Choose scan scope:",
            ["Single Stock", "Full Watchlist"],
            help="Select whether to scan one stock or the entire watchlist"
        )
        
        selected_stock = None
        if scan_scope == "Single Stock":
            try:
                from utils.watchlist import get_active_watchlist
                watchlist = get_active_watchlist()
                selected_stock = st.selectbox(
                    "Select Stock:",
                    options=watchlist,
                    help="Choose which stock to analyze"
                )
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
        
        historical_date = None
        if scan_date_type == "Historical Date":
            try:
                default_date = date.today() - timedelta(days=7)
                historical_date = st.date_input(
                    "Analysis Date:",
                    value=default_date,
                    max_value=date.today() - timedelta(days=1),
                    help="Choose the historical date for analysis (must be a past trading day)"
                )
            except Exception as e:
                st.session_state.error_logger.log_error("Date Selection", e)
                historical_date = date.today() - timedelta(days=7)
    
    return scan_scope, selected_stock, scan_date_type, historical_date

def show_advanced_settings():
    """Display advanced settings panel"""
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
    
    return days_back, rolling_window, debug_mode

def execute_scan_button(scan_scope, selected_stock, scan_date_type, historical_date, 
                       days_back, rolling_window, debug_mode):
    """Handle scan execution button and logic"""
    if st.button("🚀 Execute Scan", type="primary", use_container_width=True):
        st.session_state.error_logger = ErrorLogger()
        
        try:
            # Import required modules
            from core.data_fetcher import DataFetcher
            from utils.watchlist import get_active_watchlist
            
            # Determine stocks to scan
            if scan_scope == "Single Stock":
                stocks_to_scan = [selected_stock]
            else:
                stocks_to_scan = get_active_watchlist()
            
            # Determine analysis date
            analysis_date = historical_date if scan_date_type == "Historical Date" else None
            
            # Execute scan
            run_enhanced_stock_scan(
                stocks_to_scan=stocks_to_scan,
                analysis_date=analysis_date,
                days_back=days_back,
                rolling_window=rolling_window,
                debug_mode=debug_mode
            )
            
        except ImportError as e:
            st.session_state.error_logger.log_error("Module Import", e)
            st.error("❌ Required modules not available - check error details above")
        except Exception as e:
            st.session_state.error_logger.log_error("Scan Execution", e)
            st.error("❌ Failed to execute scan - check error details above")

def run_enhanced_stock_scan(stocks_to_scan, analysis_date=None, days_back=59, rolling_window=20, debug_mode=False):
    """Execute the enhanced stock scanning process with optimized error handling"""
    
    error_logger = st.session_state.error_logger
    
    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher
        from core.technical_analysis import add_enhanced_columns, get_mpi_trend_info
        
        # Log scan start
        is_historical = analysis_date is not None
        is_single_stock = len(stocks_to_scan) == 1
        
        scope_text = f"single stock ({stocks_to_scan[0]})" if is_single_stock else f"{len(stocks_to_scan)} stocks"
        date_text = f"historical analysis (as of {analysis_date})" if is_historical else "current data analysis"
        
        logger.info(f"Starting scan: {scope_text} with {date_text}")
        st.info(f"🔄 Scanning {scope_text} with {date_text}... Calculating Pure MPI Expansion...")
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize data fetcher
        status_text.text("🔧 Initializing data fetcher...")
        fetcher = DataFetcher(days_back=days_back)
        progress_bar.progress(0.1)
        
        # Download stock data
        status_text.text("📥 Downloading stock data and company names...")
        stock_data = fetcher.download_stock_data(stocks_to_scan)
        set_global_data_fetcher(fetcher)
        progress_bar.progress(0.3)
        
        if not stock_data:
            error_logger.log_error("Data Validation", Exception("No stock data downloaded"))
            st.error("❌ Failed to download stock data. Check error log for details.")
            return
        
        # Process stocks
        status_text.text("🔄 Calculating Pure MPI Expansion and technical analysis...")
        progress_bar.progress(0.4)
        
        results = []
        processing_errors = []
        
        for i, (ticker, df_raw) in enumerate(stock_data.items()):
            try:
                if df_raw.empty:
                    error_logger.log_warning("Stock Processing", f"Empty dataframe for {ticker}")
                    continue
                
                # Apply technical analysis
                df_enhanced = add_enhanced_columns(df_raw, ticker, rolling_window)
                
                # Handle historical analysis
                if is_historical:
                    analysis_row, actual_date = _get_historical_analysis_row(df_enhanced, analysis_date, ticker)
                    if analysis_row is None:
                        continue
                else:
                    analysis_row = df_enhanced.iloc[-1]
                    actual_date = analysis_row.name
                
                # Collect result
                result = _create_result_dict(analysis_row, actual_date, ticker, fetcher)
                results.append(result)
                
                # Update progress
                progress = 0.4 + (0.5 * (i + 1) / len(stock_data))
                progress_bar.progress(progress)
                
            except Exception as e:
                error_logger.log_error("Stock Processing", e, {"ticker": ticker})
                processing_errors.append(f"{ticker}: {str(e)}")
                continue
        
        # Finalize results
        status_text.text("📊 Preparing Pure MPI Expansion results...")
        progress_bar.progress(0.9)
        
        results_df = pd.DataFrame(results)
        
        # Store results in session state
        st.session_state.scan_results = results_df
        st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.last_scan_config = {
            'scope': 'Single Stock' if is_single_stock else 'Full Watchlist',
            'date': f'Historical ({analysis_date})' if is_historical else 'Current',
            'stock_count': len(results_df)
        }
        
        # Complete scan
        progress_bar.progress(1.0)
        status_text.text("✅ Pure MPI Expansion scan completed!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Show success message
        success_message = f"🎉 Pure MPI Expansion Scan completed! Analyzed {len(results_df)} stocks successfully"
        if processing_errors:
            success_message += f" ({len(processing_errors)} errors - check log for details)"
        if is_historical:
            success_message += f" as of {analysis_date}"
        
        st.success(success_message)
        logger.info(f"Scan completed: {len(results_df)} stocks processed successfully")
        
        st.rerun()
        
    except Exception as e:
        error_logger.log_error("Scan Execution", e)
        st.error("❌ Pure MPI Expansion scan failed with critical error - check error log for details")
        
        # Clean up progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def _get_historical_analysis_row(df_enhanced: pd.DataFrame, analysis_date: date, ticker: str) -> tuple:
    """Get the analysis row for historical date"""
    try:
        target_date = pd.to_datetime(analysis_date)
        available_dates = df_enhanced.index
        
        # Handle timezone consistency
        if hasattr(available_dates, 'tz') and available_dates.tz is not None:
            if target_date.tz is None:
                target_date = target_date.tz_localize('Asia/Singapore')
            else:
                target_date = target_date.tz_convert(available_dates.tz)
        else:
            if target_date.tz is not None:
                target_date = target_date.tz_localize(None)
        
        valid_dates = available_dates[available_dates <= target_date]
        
        if len(valid_dates) == 0:
            logger.warning(f"No data for {ticker} on or before {analysis_date}")
            return None, None
        
        analysis_row = df_enhanced.loc[valid_dates[-1]]
        actual_date = valid_dates[-1]
        
        return analysis_row, actual_date
        
    except Exception as e:
        logger.error(f"Historical date processing failed for {ticker}: {e}")
        return None, None

def _create_result_dict(analysis_row: pd.Series, actual_date, ticker: str, fetcher) -> dict:
    """Create result dictionary from analysis row"""
    # Get company name
    try:
        company_name = fetcher.get_company_name(ticker)
    except Exception:
        company_name = ticker.replace('.SI', '')
    
    # Determine price formatting
    close_price = float(analysis_row['Close'])
    price_decimals = 3 if close_price < 1.00 else 2
    
    def safe_round(value, decimals):
        try:
            return round(float(value), decimals) if not pd.isna(value) else 0
        except:
            return 0
    
    # Get MPI trend info
    mpi_trend = str(analysis_row.get('MPI_Trend', 'Unknown'))
    mpi_velocity = float(analysis_row.get('MPI_Velocity', 0.0)) if not pd.isna(analysis_row.get('MPI_Velocity', 0.0)) else 0.0
    
    try:
        from core.technical_analysis import get_mpi_trend_info
        mpi_trend_info = get_mpi_trend_info(mpi_trend, mpi_velocity)
    except:
        mpi_trend_info = {'emoji': '❓', 'description': 'Unknown'}
    
    # Create result dictionary
    result = {
        'Ticker': ticker,
        'Name': company_name,
        'Analysis_Date': actual_date.strftime('%Y-%m-%d') if hasattr(actual_date, 'strftime') else str(actual_date),
        'Close': round(close_price, price_decimals),
        'High': safe_round(analysis_row['High'], price_decimals),
        'Low': safe_round(analysis_row['Low'], price_decimals),
        'IBS': round(float(analysis_row['IBS']), 3) if not pd.isna(analysis_row['IBS']) else 0,
        'Valid_CRT': int(analysis_row.get('Valid_CRT', 0)),
        'Higher_HL': int(analysis_row.get('Higher_HL', 0)) if not pd.isna(analysis_row.get('Higher_HL', 0)) else 0,
        'CRT_Velocity': round(float(analysis_row.get('CRT_Qualifying_Velocity', 0)), 4) if not pd.isna(analysis_row.get('CRT_Qualifying_Velocity', 0)) else 0,
        'Weekly_Open': safe_round(analysis_row.get('Weekly_Open', 0), price_decimals),
        'CRT_High': safe_round(analysis_row.get('CRT_High', 0), price_decimals),
        'CRT_Low': safe_round(analysis_row.get('CRT_Low', 0), price_decimals),
        'VW_Range_Percentile': round(float(analysis_row.get('VW_Range_Percentile', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Percentile', 0)) else 0,
        'Rel_Range_Signal': int(analysis_row.get('Rel_Range_Signal', 0)),
        
        # Pure MPI Expansion columns
        'MPI': round(float(analysis_row.get('MPI', 0.5)), 4) if not pd.isna(analysis_row.get('MPI', 0.5)) else 0.5,
        'MPI_Velocity': round(mpi_velocity, 4),
        'MPI_Trend': mpi_trend,
        'MPI_Trend_Emoji': mpi_trend_info.get('emoji', '❓'),
        'MPI_Description': mpi_trend_info.get('description', 'Unknown'),
        'MPI_Visual': format_mpi_visual(analysis_row.get('MPI', 0.5)),
        'Price_Decimals': price_decimals
    }
    
    return result

# File: pages/scanner_higher_hl.py
# Part 4 of 4
"""
Results display and main show function - optimized
"""

def display_scan_summary(results_df: pd.DataFrame):
    """Display scan summary with Pure MPI Expansion statistics"""
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
        contracting = trend_counts.get('Mild Contraction', 0) + trend_counts.get('Strong Contraction', 0)
        avg_mpi = results_df['MPI'].mean()
        avg_velocity = results_df['MPI_Velocity'].mean()
    else:
        strong_expansion = expanding = contracting = 0
        avg_mpi = avg_velocity = 0
    
    # Display metrics
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
        st.metric("🚀 Strong Exp", strong_expansion, delta=f"≥5% velocity")
    with col6:
        st.metric("📈 Expanding", strong_expansion + expanding, delta=f">0% velocity")
    
    # Analysis date info
    if len(results_df) > 0:
        analysis_dates = results_df['Analysis_Date'].unique()
        if len(analysis_dates) == 1:
            st.info(f"📅 Analysis date: **{analysis_dates[0]}** | Avg MPI: **{avg_mpi:.1%}** | Avg Velocity: **{avg_velocity:+.1%}** | Expanding: **{strong_expansion + expanding}** | Contracting: **{contracting}**")
        else:
            st.info(f"📅 Analysis dates: **{', '.join(analysis_dates)}** | Avg MPI: **{avg_mpi:.1%}** | Avg Velocity: **{avg_velocity:+.1%}**")

def show_base_pattern_filter(results_df: pd.DataFrame) -> pd.DataFrame:
    """Show base pattern filter and return filtered stocks"""
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
    
    return base_stocks

def display_filtered_results(filtered_stocks: pd.DataFrame, selected_base_filter: str):
    """Display the filtered results table and export options"""
    st.subheader(f"📋 Pure MPI Expansion Results ({len(filtered_stocks)} stocks)")
    
    if len(filtered_stocks) == 0:
        st.warning("No stocks match the current filter criteria")
        return
    
    # Define display columns
    display_cols = ['Ticker', 'Name', 'Close', 'CRT_High', 'CRT_Low',
                   'CRT_Velocity', 'IBS', 'Higher_HL', 'MPI_Trend_Emoji', 'MPI', 'MPI_Velocity', 'MPI_Visual']
    
    # Create column configuration
    base_column_config = {
        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
        'Name': st.column_config.TextColumn('Company Name', width='medium'),
        'CRT_Velocity': st.column_config.NumberColumn('CRT Vel', format='%+.4f'),
        'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
        'Higher_HL': st.column_config.NumberColumn('H/L', width='small'),
        'MPI_Trend_Emoji': st.column_config.TextColumn('📊', width='small', help='MPI Expansion Trend'),
        'MPI': st.column_config.NumberColumn('MPI', format='%.1f', help='Market Positivity Index'),
        'MPI_Velocity': st.column_config.NumberColumn('MPI Vel', format='%.1f', help='MPI Expansion Rate'),
        'MPI_Visual': st.column_config.TextColumn('MPI Visual', width='medium', help='Visual MPI representation')
    }
    
    # Apply dynamic price formatting
    column_config = get_dynamic_column_config(filtered_stocks, display_cols, base_column_config)
    
    # Filter display columns to only existing ones
    display_cols = [col for col in display_cols if col in filtered_stocks.columns]
    
    # Display the results table
    try:
        st.dataframe(
            filtered_stocks[display_cols],
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        
        # TradingView Export
        show_tradingview_export(filtered_stocks, selected_base_filter)
        
    except Exception as e:
        st.error(f"❌ Error displaying filtered results: {str(e)}")
        st.session_state.error_logger.log_error("Filtered Results Display", e)

def show_tradingview_export(filtered_stocks: pd.DataFrame, selected_base_filter: str):
    """Show TradingView export section"""
    st.subheader("📋 TradingView Export (Pure MPI Expansion Filtered)")
    
    tv_tickers = [f"SGX:{ticker.replace('.SI', '')}" for ticker in filtered_stocks['Ticker'].tolist()]
    tv_string = ','.join(tv_tickers)
    
    # Calculate Pure MPI Expansion summary for export description
    mpi_summary = ""
    if 'MPI_Trend' in filtered_stocks.columns:
        trend_summary = filtered_stocks['MPI_Trend'].value_counts()
        top_trend = trend_summary.index[0] if len(trend_summary) > 0 else "Mixed"
        mpi_summary = f"Top Trend: {top_trend}"
        if 'MPI_Velocity' in filtered_stocks.columns:
            mpi_summary += f" | Avg Velocity: {filtered_stocks['MPI_Velocity'].mean():+.1%}"
    
    st.text_area(
        f"Singapore Exchange (SGX) - {selected_base_filter} ({len(tv_tickers)} stocks) {mpi_summary}:",
        value=tv_string,
        height=100,
        help="Copy and paste into TradingView watchlist. Sorted by MPI Expansion trends."
    )
    
    # CSV Export
    csv_data = filtered_stocks.to_csv(index=False)
    filename_prefix = selected_base_filter.lower().replace(' ', '_').replace('+', 'and')
    st.download_button(
        label="📥 Download MPI Expansion Data (CSV)",
        data=csv_data,
        file_name=f"mpi_expansion_{filename_prefix}_{len(filtered_stocks)}_stocks.csv",
        mime="text/csv"
    )

def show_full_results_table(results_df: pd.DataFrame):
    """Show the full results table in an expander"""
    with st.expander("📋 Full Pure MPI Expansion Analysis Results", expanded=False):
        try:
            full_results_cols = [
                'Analysis_Date', 'Ticker', 'Name', 'Close', 'CRT_High', 'CRT_Low',
                'CRT_Velocity', 'IBS', 'Higher_HL', 'Valid_CRT', 
                'MPI_Trend_Emoji', 'MPI_Trend', 'MPI', 'MPI_Velocity', 'MPI_Visual'
            ]
            
            base_full_results_config = {
                'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Name': st.column_config.TextColumn('Company Name', width='medium'),
                'CRT_Velocity': st.column_config.NumberColumn('CRT Vel', format='%+.4f'),
                'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                'Higher_HL': st.column_config.NumberColumn('H/L', width='small'),
                'Valid_CRT': st.column_config.NumberColumn('CRT', width='small'),
                'MPI_Trend_Emoji': st.column_config.TextColumn('📊', width='small'),
                'MPI_Trend': st.column_config.TextColumn('MPI Trend', width='medium'),
                'MPI': st.column_config.NumberColumn('MPI', format='%.1%'),
                'MPI_Velocity': st.column_config.NumberColumn('MPI Velocity', format='%+.1%'),
                'MPI_Visual': st.column_config.TextColumn('MPI Visual', width='medium')
            }
            
            # Apply dynamic formatting and filter columns
            full_results_column_config = get_dynamic_column_config(results_df, full_results_cols, base_full_results_config)
            full_results_cols = [col for col in full_results_cols if col in results_df.columns]
            
            st.dataframe(
                results_df[full_results_cols],
                column_config=full_results_column_config,
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"❌ Error displaying full results: {str(e)}")
            st.session_state.error_logger.log_error("Full Results Display", e)

def show_mpi_insights(results_df: pd.DataFrame):
    """Show Pure MPI Expansion insights"""
    if 'MPI_Trend' not in results_df.columns:
        return
        
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
                        top_stock_idx = trend_stocks['MPI_Velocity'].idxmin()
                    top_stock = trend_stocks.loc[top_stock_idx, 'Name']
                    top_velocity = trend_stocks.loc[top_stock_idx, 'MPI_Velocity']
                else:
                    top_stock = 'N/A'
                    top_velocity = 0
                
                trend_summary.append({
                    'Trend': trend,
                    'Count': count,
                    'Avg MPI': f"{avg_mpi:.1%}",
                    'Avg Velocity': f"{avg_velocity:+.1%}",
                    'Top Stock': top_stock,
                    'Top Velocity': f"{top_velocity:+.1%}",
                    'Trading Action': get_mpi_expansion_description(trend)
                })
        
        if trend_summary:
            st.dataframe(pd.DataFrame(trend_summary), hide_index=True, use_container_width=True)
        
        # MPI Velocity distribution chart
        if len(results_df) > 1:
            fig = create_mpi_velocity_chart(results_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def create_mpi_velocity_chart(results_df: pd.DataFrame):
    """Create MPI velocity distribution chart"""
    try:
        df_for_plot = results_df[['MPI_Velocity', 'MPI_Trend']].copy()
        
        color_map = {
            'Strong Expansion': 'darkgreen',
            'Expanding': 'green',
            'Flat': 'gray',
            'Mild Contraction': 'orange',
            'Strong Contraction': 'red'
        }
        
        fig = px.histogram(
            df_for_plot,
            x='MPI_Velocity',
            color='MPI_Trend',
            nbins=20,
            title="MPI Velocity Distribution - Pure Expansion Focus",
            labels={'MPI_Velocity': 'MPI Velocity (Expansion Rate)', 'count': 'Number of Stocks'},
            color_discrete_map=color_map,
            category_orders={'MPI_Trend': ['Strong Expansion', 'Expanding', 'Flat', 
                                          'Mild Contraction', 'Strong Contraction']}
        )
        
        # Add threshold lines
        fig.add_vline(x=0.05, line_dash="dash", line_color="darkgreen", annotation_text="Strong Expansion")
        fig.add_vline(x=0, line_dash="solid", line_color="black", annotation_text="Flat")
        fig.add_vline(x=-0.05, line_dash="dash", line_color="red", annotation_text="Strong Contraction")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating MPI velocity chart: {e}")
        return None

def display_scan_results(results_df: pd.DataFrame):
    """Main function to display scanning results with Pure MPI Expansion filtering"""
    
    if results_df.empty:
        st.warning("No results to display.")
        return
    
    try:
        # Display scan summary
        display_scan_summary(results_df)
        
        # Show base pattern filter and get filtered stocks
        base_stocks = show_base_pattern_filter(results_df)
        
        if len(base_stocks) > 0:
            # Apply dynamic filters
            filtered_stocks = apply_dynamic_filters(base_stocks, results_df)
            
            # Display filtered results
            display_filtered_results(filtered_stocks, st.session_state.base_filter_type)
        else:
            st.warning(f"No stocks found for pattern: {st.session_state.base_filter_type}")
        
        # Show full results table
        show_full_results_table(results_df)
        
        # Show MPI insights
        show_mpi_insights(results_df)
        
        logger.info(f"Successfully displayed results for {len(results_df)} stocks")
        
    except Exception as e:
        st.session_state.error_logger.log_error("Results Display", e)
        st.error("❌ Error displaying Pure MPI Expansion results - check error log for details")

def show():
    """Main scanner page display for Higher H/L patterns with Pure MPI Expansion filtering"""
    
    st.title("📈 CRT Higher H/L Scanner")
    st.markdown("Enhanced with **Pure MPI Expansion System** - Focus on momentum velocity, not absolute levels")
    
    # Clear error log button
    if st.button("🗑️ Clear Error Log"):
        st.session_state.error_logger = ErrorLogger()
        st.success("Error log cleared!")
        st.rerun()
    
    # Check module availability
    try:
        from core.data_fetcher import DataFetcher
        from core.technical_analysis import add_enhanced_columns
        from utils.watchlist import get_active_watchlist
        modules_available = True
        logger.info("All core modules imported successfully")
    except ImportError as e:
        st.session_state.error_logger.log_error("Module Import", e)
        st.error(f"❌ Import error: {e}")
        modules_available = False
    
    # Display any existing errors
    st.session_state.error_logger.display_errors_in_streamlit()
    
    if modules_available:
        # Show scanning configuration
        scan_scope, selected_stock, scan_date_type, historical_date = show_scanning_configuration()
        
        # Show advanced settings
        days_back, rolling_window, debug_mode = show_advanced_settings()
        
        # Scan execution
        st.subheader("🚀 Execute Scan")
        execute_scan_button(scan_scope, selected_stock, scan_date_type, historical_date, 
                           days_back, rolling_window, debug_mode)
        
        # Display last scan info if available
        if 'last_scan_time' in st.session_state:
            st.info(f"📊 Last scan completed: {st.session_state.last_scan_time}")
            
            if 'last_scan_config' in st.session_state:
                config = st.session_state.last_scan_config
                st.caption(f"Scope: {config['scope']} | Date: {config['date']} | Stocks: {config['stock_count']}")
        
        # Display results if available
        if 'scan_results' in st.session_state:
            display_scan_results(st.session_state.scan_results)
    
    else:
        st.warning("Cannot execute scan - required modules are not available")

if __name__ == "__main__":
    show()