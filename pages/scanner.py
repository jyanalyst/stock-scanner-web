"""
Enhanced Live Scanner Page with Buy Signals Dynamic Filtering
Real-time stock scanning functionality with buy signals filtering capability
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

def show_buy_signals_filter(buy_signals_stocks):
    """
    Display dynamic filtering for buy signals - consistent with Valid CRT filtering
    """
    
    if buy_signals_stocks.empty:
        st.warning("No buy signals available for filtering")
        return buy_signals_stocks
    
    st.subheader("🎯 Dynamic Filtering")
    
    # Create two columns for the two filters
    col1, col2 = st.columns(2)
    
    # Initialize filtered stocks
    filtered_stocks = buy_signals_stocks.copy()
    
    # CRT VELOCITY PERCENTILE FILTER
    with col1:
        st.markdown("**CRT Velocity Filter:**")
        
        # Get velocity statistics - now using CRT_Velocity
        velocities = buy_signals_stocks['CRT_Velocity']
        
        percentile_options = {
            "Top 25%": 75,
            "Top 50%": 50, 
            "Top 75%": 25,
            "No Filter": None
        }
        
        selected_percentile = st.radio(
            "Select velocity filter:",
            list(percentile_options.keys()),
            key="buy_signals_percentile_radio"
        )
        
        # Apply percentile filtering
        if selected_percentile != "No Filter":
            percentile_val = percentile_options[selected_percentile]
            threshold_value = np.percentile(velocities, percentile_val)
            filtered_stocks = filtered_stocks[filtered_stocks['CRT_Velocity'] >= threshold_value]
            st.info(f"CRT Velocity ≥ {threshold_value:+.4f} pp")
        else:
            st.info("All velocities included")
        
        # Show velocity statistics
        with st.expander("CRT Velocity Statistics", expanded=False):
            stats_data = {
                "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                "Value": [
                    len(velocities),
                    f"{velocities.min():+.4f} pp",
                    f"{np.percentile(velocities, 25):+.4f} pp",
                    f"{velocities.median():+.4f} pp", 
                    f"{np.percentile(velocities, 75):+.4f} pp",
                    f"{velocities.max():+.4f} pp"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
    
    # IBS FILTER
    with col2:
        st.markdown("**IBS Filter:**")
        
        ibs_options = {
            "IBS ≥ 0.75": 0.75,
            "IBS ≥ 0.5": 0.5,
            "No Filter": None
        }
        
        selected_ibs = st.radio(
            "Select IBS filter:",
            list(ibs_options.keys()),
            key="buy_signals_ibs_radio"
        )
        
        # Apply IBS filtering
        if selected_ibs != "No Filter":
            ibs_threshold = ibs_options[selected_ibs]
            filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= ibs_threshold]
            st.info(f"IBS ≥ {ibs_threshold:.2f}")
        else:
            st.info("All IBS values included")
        
        # Show IBS statistics for the velocity-filtered stocks
        with st.expander("IBS Statistics", expanded=False):
            # Use the stocks that passed velocity filter for IBS stats
            ibs_values = filtered_stocks['IBS'] if selected_percentile != "No Filter" else buy_signals_stocks['IBS']
            ibs_stats = {
                "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                "Value": [
                    len(ibs_values),
                    f"{ibs_values.min():.3f}",
                    f"{np.percentile(ibs_values, 25):.3f}",
                    f"{ibs_values.median():.3f}",
                    f"{np.percentile(ibs_values, 75):.3f}",
                    f"{ibs_values.max():.3f}"
                ]
            }
            st.dataframe(pd.DataFrame(ibs_stats), hide_index=True, use_container_width=True)
    
    # Show combined filter summary
    filter_summary = []
    if selected_percentile != "No Filter":
        filter_summary.append(f"CRT Velocity {selected_percentile}")
    if selected_ibs != "No Filter":
        filter_summary.append(f"{selected_ibs}")
    
    if filter_summary:
        st.success(f"Active filters: {' + '.join(filter_summary)} → {len(filtered_stocks)} signals")
    else:
        st.info(f"No filters applied → {len(filtered_stocks)} signals")
    
    # Show distribution charts
    with st.expander("📊 Distribution Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                x=velocities,
                nbins=20,
                title="CRT Velocity Distribution",
                labels={'x': 'CRT Velocity (pp)', 'y': 'Count'}
            )
            # Add threshold lines if filter is active
            if selected_percentile != "No Filter":
                percentile_val = percentile_options[selected_percentile]
                threshold_value = np.percentile(velocities, percentile_val)
                fig_hist.add_vline(x=threshold_value, line_dash="dash", line_color="red", 
                                 annotation_text=f"{selected_percentile} threshold")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # IBS distribution for buy signals
            fig_ibs = px.histogram(
                x=buy_signals_stocks['IBS'],
                nbins=20,
                title="IBS Distribution",
                labels={'x': 'IBS', 'y': 'Count'}
            )
            # Add threshold line if filter is active
            if selected_ibs != "No Filter":
                ibs_threshold = ibs_options[selected_ibs]
                fig_ibs.add_vline(x=ibs_threshold, line_dash="dash", line_color="red",
                                annotation_text=f"{selected_ibs} threshold")
            st.plotly_chart(fig_ibs, use_container_width=True)
    
    # Display filtered results
    st.subheader(f"📋 Filtered Results ({len(filtered_stocks)} signals)")
    
    if len(filtered_stocks) > 0:
        # Sort by CRT Velocity (descending)
        filtered_stocks_sorted = filtered_stocks.sort_values('CRT_Velocity', ascending=False)
        
        # Use consistent display columns with Valid CRT
        display_cols = ['Ticker', 'Name', 'Close', 'CRT_Velocity', 'IBS', 'Signal_Type']
        
        # Add Signal_Type column for better readability
        filtered_stocks_sorted['Signal_Type'] = filtered_stocks_sorted.apply(
            lambda row: get_signal_description(row), axis=1
        )
        
        column_config = {
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Name': st.column_config.TextColumn('Company Name', width='large'),
            'Close': st.column_config.NumberColumn('Close Price', format='$%.2f'),
            'CRT_Velocity': st.column_config.NumberColumn('CRT Velocity', format='%+.4f pp'),
            'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
            'Signal_Type': st.column_config.TextColumn('Signal Type', width='medium')
        }
        
        st.dataframe(
            filtered_stocks_sorted[display_cols],
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        
        # TradingView Export for filtered buy signals
        st.subheader("📋 TradingView Export (Filtered)")
        tv_tickers = [f"SGX:{ticker.replace('.SI', '')}" for ticker in filtered_stocks_sorted['Ticker'].tolist()]
        tv_string = ','.join(tv_tickers)
        
        st.text_area(
            f"Singapore Exchange (SGX) Buy Signals ({len(tv_tickers)} stocks):",
            value=tv_string,
            height=100,
            help="Copy and paste into TradingView watchlist. SGX: prefix ensures Singapore Exchange stocks."
        )
        
        # Export filtered buy signals data
        csv_data = filtered_stocks_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"filtered_buy_signals_{len(filtered_stocks)}_stocks.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No buy signals match the current filter criteria")
    
    return filtered_stocks

def show_velocity_filter(valid_crt_stocks):
    """
    Display dynamic velocity filtering with simplified percentile options and IBS filter
    """
    
    if valid_crt_stocks.empty or 'CRT_Velocity' not in valid_crt_stocks.columns:
        st.warning("No Valid CRT stocks available for filtering")
        return valid_crt_stocks
    
    st.subheader("🎯 Dynamic Filtering")
    
    # Create two columns for the two filters
    col1, col2 = st.columns(2)
    
    # Initialize filtered stocks
    filtered_stocks = valid_crt_stocks.copy()
    
    # CRT VELOCITY PERCENTILE FILTER
    with col1:
        st.markdown("**CRT Velocity Filter:**")
        
        # Get velocity statistics
        velocities = valid_crt_stocks['CRT_Velocity']
        
        percentile_options = {
            "Top 25%": 75,
            "Top 50%": 50, 
            "Top 75%": 25,
            "No Filter": None
        }
        
        selected_percentile = st.radio(
            "Select velocity filter:",
            list(percentile_options.keys()),
            key="percentile_filter_radio"
        )
        
        # Apply percentile filtering
        if selected_percentile != "No Filter":
            percentile_val = percentile_options[selected_percentile]
            threshold_value = np.percentile(velocities, percentile_val)
            filtered_stocks = filtered_stocks[filtered_stocks['CRT_Velocity'] >= threshold_value]
            st.info(f"CRT Velocity ≥ {threshold_value:+.4f} pp")
        else:
            st.info("All velocities included")
        
        # Show velocity statistics
        with st.expander("CRT Velocity Statistics", expanded=False):
            stats_data = {
                "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                "Value": [
                    len(velocities),
                    f"{velocities.min():+.4f} pp",
                    f"{np.percentile(velocities, 25):+.4f} pp",
                    f"{velocities.median():+.4f} pp", 
                    f"{np.percentile(velocities, 75):+.4f} pp",
                    f"{velocities.max():+.4f} pp"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
    
    # IBS FILTER
    with col2:
        st.markdown("**IBS Filter:**")
        
        # Check if IBS column exists
        if 'IBS' not in filtered_stocks.columns:
            st.warning("IBS data not available")
        else:
            ibs_options = {
                "IBS ≥ 0.75": 0.75,
                "IBS ≥ 0.5": 0.5,
                "No Filter": None
            }
            
            selected_ibs = st.radio(
                "Select IBS filter:",
                list(ibs_options.keys()),
                key="ibs_filter_radio"
            )
            
            # Apply IBS filtering
            if selected_ibs != "No Filter":
                ibs_threshold = ibs_options[selected_ibs]
                filtered_stocks = filtered_stocks[filtered_stocks['IBS'] >= ibs_threshold]
                st.info(f"IBS ≥ {ibs_threshold:.2f}")
            else:
                st.info("All IBS values included")
            
            # Show IBS statistics for the velocity-filtered stocks
            with st.expander("IBS Statistics", expanded=False):
                # Use the stocks that passed velocity filter for IBS stats
                ibs_values = filtered_stocks['IBS'] if selected_percentile != "No Filter" else valid_crt_stocks['IBS']
                ibs_stats = {
                    "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                    "Value": [
                        len(ibs_values),
                        f"{ibs_values.min():.3f}",
                        f"{np.percentile(ibs_values, 25):.3f}",
                        f"{ibs_values.median():.3f}",
                        f"{np.percentile(ibs_values, 75):.3f}",
                        f"{ibs_values.max():.3f}"
                    ]
                }
                st.dataframe(pd.DataFrame(ibs_stats), hide_index=True, use_container_width=True)
    
    # Show combined filter summary
    filter_summary = []
    if selected_percentile != "No Filter":
        filter_summary.append(f"CRT Velocity {selected_percentile}")
    if selected_ibs != "No Filter":
        filter_summary.append(f"{selected_ibs}")
    
    if filter_summary:
        st.success(f"Active filters: {' + '.join(filter_summary)} → {len(filtered_stocks)} stocks")
    else:
        st.info(f"No filters applied → {len(filtered_stocks)} stocks")
    
    # Show distribution charts
    with st.expander("📊 Distribution Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                x=velocities,
                nbins=20,
                title="CRT Velocity Distribution",
                labels={'x': 'CRT Velocity (pp)', 'y': 'Count'}
            )
            # Add threshold lines if filter is active
            if selected_percentile != "No Filter":
                percentile_val = percentile_options[selected_percentile]
                threshold_value = np.percentile(velocities, percentile_val)
                fig_hist.add_vline(x=threshold_value, line_dash="dash", line_color="red", 
                                 annotation_text=f"{selected_percentile} threshold")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # IBS distribution for valid CRT stocks
            if 'IBS' in valid_crt_stocks.columns:
                fig_ibs = px.histogram(
                    x=valid_crt_stocks['IBS'],
                    nbins=20,
                    title="IBS Distribution",
                    labels={'x': 'IBS', 'y': 'Count'}
                )
                # Add threshold line if filter is active
                if selected_ibs != "No Filter":
                    ibs_threshold = ibs_options[selected_ibs]
                    fig_ibs.add_vline(x=ibs_threshold, line_dash="dash", line_color="red",
                                    annotation_text=f"{selected_ibs} threshold")
                st.plotly_chart(fig_ibs, use_container_width=True)
    
    # Display filtered results
    st.subheader(f"📋 Filtered Results ({len(filtered_stocks)} stocks)")
    
    if len(filtered_stocks) > 0:
        # Sort by CRT Velocity (descending)
        filtered_stocks_sorted = filtered_stocks.sort_values('CRT_Velocity', ascending=False)
        
        # Add IBS to display columns
        display_cols = ['Ticker', 'Name', 'Weekly_Open', 'CRT_Velocity', 'IBS', 'CRT_High', 'CRT_Low']
        column_config = {
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Name': st.column_config.TextColumn('Company Name', width='large'),
            'Weekly_Open': st.column_config.NumberColumn('Weekly Open', format='$%.2f'),
            'CRT_Velocity': st.column_config.NumberColumn('CRT Velocity', format='%+.4f pp'),
            'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
            'CRT_High': st.column_config.NumberColumn('CRT High', format='$%.2f'),
            'CRT_Low': st.column_config.NumberColumn('CRT Low', format='$%.2f')
        }
        
        st.dataframe(
            filtered_stocks_sorted[display_cols],
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        
        # TradingView Export for filtered list
        st.subheader("📋 TradingView Export (Filtered)")
        tv_tickers = [f"SGX:{ticker.replace('.SI', '')}" for ticker in filtered_stocks_sorted['Ticker'].tolist()]
        tv_string = ','.join(tv_tickers)
        
        st.text_area(
            f"Singapore Exchange (SGX) TradingView list ({len(tv_tickers)} stocks):",
            value=tv_string,
            height=100,
            help="Copy and paste into TradingView watchlist. SGX: prefix ensures Singapore Exchange stocks."
        )
        
        # Export filtered data
        csv_data = filtered_stocks_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"filtered_valid_crt_{len(filtered_stocks)}_stocks.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No stocks match the current filter criteria")
    
    return filtered_stocks

def show():
    """Main scanner page display with enhanced error logging"""
    
    st.title("🔍 Live Stock Scanner")
    st.markdown("Flexible stock scanning with comprehensive error logging and debugging")
    
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
        st.info("""
        **Missing modules detected!** Please create the following files:
        
        1. `core/data_fetcher.py` - Copy the enhanced data fetcher code
        2. `core/technical_analysis.py` - Copy the technical analysis code  
        3. `utils/watchlist.py` - Copy the watchlist code
        """)
        modules_available = False
    
    # Display any existing errors
    st.session_state.error_logger.display_errors_in_streamlit()
    
    # Scanning Configuration Panel
    st.subheader("🎯 Scanning Configuration")
    
    # Create two main columns for scan configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Scan Scope**")
        scan_scope = st.radio(
            "Choose scan scope:",
            ["Single Stock", "Full Watchlist"],
            help="Select whether to scan one stock or the entire watchlist"
        )
        
        # Single stock selection
        if scan_scope == "Single Stock":
            try:
                if modules_available:
                    watchlist = get_active_watchlist()
                    st.session_state.error_logger.log_debug("Watchlist", f"Loaded {len(watchlist)} stocks", watchlist)
                else:
                    # Fallback watchlist for testing
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
                selected_stock = "A17U.SI"  # Fallback
        
    with col2:
        st.markdown("**📅 Analysis Date**")
        scan_date_type = st.radio(
            "Choose analysis date:",
            ["Current Date", "Historical Date"],
            help="Scan as of current date or specify a historical date"
        )
        
        # Historical date selection
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
    
    # Debug Information Display
    if debug_mode:
        with st.expander("🔧 Debug Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "modules_available": modules_available,
                    "scan_scope": scan_scope,
                    "scan_date_type": scan_date_type,
                    "days_back": days_back,
                    "rolling_window": rolling_window
                })
            
            with col2:
                summary = st.session_state.error_logger.get_summary()
                st.json(summary)
            
            # Show recent debug entries
            if st.session_state.error_logger.debug_info:
                st.subheader("Recent Debug Log")
                for entry in st.session_state.error_logger.debug_info[-5:]:  # Last 5 entries
                    st.text(f"{entry['timestamp']} - {entry['component']}: {entry['message']}")
    
    # Scan Execution
    st.subheader("🚀 Execute Scan")
    
    # Generate scan description
    try:
        if scan_scope == "Single Stock":
            scope_desc = f"**{selected_stock}**"
        else:
            if modules_available:
                scope_desc = f"**Full Watchlist** ({len(get_active_watchlist())} stocks)"
            else:
                scope_desc = "**Full Watchlist** (unknown count - modules not available)"
        
        if scan_date_type == "Historical Date":
            date_desc = f"as of **{historical_date}**"
        else:
            date_desc = f"with **current data**"
        
        st.info(f"📋 Ready to scan: {scope_desc} {date_desc}")
        
    except Exception as e:
        st.session_state.error_logger.log_error("Scan Description", e)
        st.warning("Error generating scan description - see debug log for details")
    
    # Main scan button
    if st.button("🚀 Execute Scan", type="primary", use_container_width=True):
        # Clear previous errors for this scan
        st.session_state.error_logger = ErrorLogger()
        
        if modules_available:
            try:
                # Determine scan parameters
                if scan_scope == "Single Stock":
                    stocks_to_scan = [selected_stock]
                else:
                    stocks_to_scan = get_active_watchlist()
                
                if scan_date_type == "Historical Date":
                    analysis_date = historical_date
                else:
                    analysis_date = None  # Use current date
                
                st.session_state.error_logger.log_debug("Scan Parameters", "Parameters determined", {
                    "stocks_to_scan": stocks_to_scan,
                    "analysis_date": str(analysis_date) if analysis_date else "Current",
                    "days_back": days_back,
                    "rolling_window": rolling_window
                })
                
                # Execute the scan
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
    """Execute the enhanced stock scanning process with comprehensive error logging"""
    
    error_logger = st.session_state.error_logger
    
    try:
        # Import modules
        from core.data_fetcher import DataFetcher, set_global_data_fetcher
        from core.technical_analysis import add_enhanced_columns
        
        error_logger.log_debug("Scan Start", "Starting enhanced stock scan", {
            "stocks_count": len(stocks_to_scan),
            "stocks": stocks_to_scan,
            "analysis_date": str(analysis_date) if analysis_date else "Current",
            "days_back": days_back,
            "rolling_window": rolling_window
        })
        
        # Determine scan type for display
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
        
        st.info(f"🔄 Scanning {scope_text} with {date_text}... This may take a moment.")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize data fetcher
        status_text.text("🔧 Initializing data fetcher...")
        try:
            fetcher = DataFetcher(days_back=days_back)
            error_logger.log_debug("Data Fetcher", f"Initialized with {days_back} days back")
        except Exception as e:
            error_logger.log_error("Data Fetcher Initialization", e, {"days_back": days_back})
            raise
        
        progress_bar.progress(0.1)
        
        # Download stock data
        status_text.text("📥 Downloading stock data and company names...")
        try:
            stock_data = fetcher.download_stock_data(stocks_to_scan)
            error_logger.log_debug("Data Download", f"Downloaded data for {len(stock_data)} stocks", {
                "requested_stocks": len(stocks_to_scan),
                "successful_downloads": len(stock_data),
                "success_rate": f"{len(stock_data)/len(stocks_to_scan)*100:.1f}%"
            })
            
            # Log any missing stocks
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
        
        # Set global fetcher
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
        
        # Process each stock
        status_text.text("🔄 Analyzing technical indicators...")
        progress_bar.progress(0.4)
        
        results = []
        processing_errors = []
        
        for i, (ticker, df_raw) in enumerate(stock_data.items()):
            try:
                error_logger.log_debug("Stock Processing", f"Processing {ticker}", {
                    "data_shape": df_raw.shape,
                    "date_range": f"{df_raw.index[0]} to {df_raw.index[-1]}" if len(df_raw) > 0 else "No data",
                    "columns": list(df_raw.columns)
                })
                
                if df_raw.empty:
                    error_logger.log_warning("Stock Processing", f"Empty dataframe for {ticker}")
                    continue
                
                # Apply technical analysis
                try:
                    df_enhanced = add_enhanced_columns(df_raw, ticker, rolling_window)
                    error_logger.log_debug("Technical Analysis", f"Enhanced columns added for {ticker}", {
                        "enhanced_shape": df_enhanced.shape,
                        "new_columns": [col for col in df_enhanced.columns if col not in df_raw.columns]
                    })
                except Exception as e:
                    error_logger.log_error("Technical Analysis", e, {
                        "ticker": ticker,
                        "raw_data_shape": df_raw.shape,
                        "rolling_window": rolling_window
                    })
                    processing_errors.append(f"{ticker}: Technical analysis failed")
                    continue
                
                # Determine which row to analyze
                if is_historical:
                    try:
                        # Handle timezone-aware datetime comparison properly
                        target_date = pd.to_datetime(analysis_date)
                        available_dates = df_enhanced.index
                        
                        # Convert target_date to match the timezone of available_dates if needed
                        if hasattr(available_dates, 'tz') and available_dates.tz is not None:
                            # Data has timezone info (Singapore timezone)
                            if target_date.tz is None:
                                # Target date is naive, localize it to Singapore timezone
                                target_date = target_date.tz_localize('Asia/Singapore')
                            else:
                                # Convert target date to same timezone as data
                                target_date = target_date.tz_convert(available_dates.tz)
                        else:
                            # Data is timezone-naive, ensure target_date is also naive
                            if target_date.tz is not None:
                                target_date = target_date.tz_localize(None)
                        
                        # Now we can safely compare
                        valid_dates = available_dates[available_dates <= target_date]
                        
                        error_logger.log_debug("Historical Date Processing", f"Date comparison for {ticker}", {
                            "target_date": str(target_date),
                            "target_date_tz": str(target_date.tz) if hasattr(target_date, 'tz') else "None",
                            "available_dates_tz": str(available_dates.tz) if hasattr(available_dates, 'tz') else "None",
                            "available_dates_count": len(available_dates),
                            "valid_dates_count": len(valid_dates)
                        })
                        
                        if len(valid_dates) == 0:
                            error_logger.log_warning("Historical Analysis", f"No data for {ticker} on or before {analysis_date}", {
                                "target_date": str(target_date),
                                "available_date_range": f"{available_dates[0]} to {available_dates[-1]}" if len(available_dates) > 0 else "No dates",
                                "total_available_days": len(available_dates)
                            })
                            continue
                        
                        analysis_row = df_enhanced.loc[valid_dates[-1]]
                        actual_date = valid_dates[-1]
                        
                        # Calculate days difference safely
                        try:
                            if hasattr(actual_date, 'date'):
                                actual_date_simple = actual_date.date()
                            else:
                                actual_date_simple = actual_date
                            
                            if hasattr(target_date, 'date'):
                                target_date_simple = target_date.date()
                            else:
                                target_date_simple = target_date
                            
                            days_diff = (target_date_simple - actual_date_simple).days
                        except:
                            days_diff = "Unable to calculate"
                        
                        error_logger.log_debug("Historical Analysis", f"Using data for {ticker}", {
                            "requested_date": str(analysis_date),
                            "actual_date": str(actual_date),
                            "days_difference": days_diff
                        })
                        
                    except Exception as e:
                        error_logger.log_error("Historical Date Processing", e, {
                            "ticker": ticker,
                            "target_date": str(analysis_date),
                            "available_dates_count": len(df_enhanced),
                            "index_dtype": str(df_enhanced.index.dtype),
                            "index_tz": str(df_enhanced.index.tz) if hasattr(df_enhanced.index, 'tz') else "None",
                            "sample_dates": [str(d) for d in df_enhanced.index[:3].tolist()] if len(df_enhanced) > 0 else []
                        })
                        continue
                else:
                    analysis_row = df_enhanced.iloc[-1]
                    actual_date = analysis_row.name
                
                # Get company name
                try:
                    company_name = fetcher.get_company_name(ticker)
                    error_logger.log_debug("Company Name", f"Retrieved name for {ticker}: {company_name}")
                except Exception as e:
                    error_logger.log_warning("Company Name", f"Failed to get name for {ticker}: {e}")
                    company_name = ticker.replace('.SI', '')
                
                # Collect results
                try:
                    result = {
                        'Ticker': ticker,
                        'Name': company_name,
                        'Analysis_Date': actual_date.strftime('%Y-%m-%d') if hasattr(actual_date, 'strftime') else str(actual_date),
                        'Close': round(float(analysis_row['Close']), 2),
                        'IBS': round(float(analysis_row['IBS']), 3) if not pd.isna(analysis_row['IBS']) else 0,
                        'Valid_CRT': int(analysis_row.get('Valid_CRT', 0)),
                        'Wick_Below': int(analysis_row.get('Wick_Below', 0)),
                        'Close_Above': int(analysis_row.get('Close_Above', 0)),
                        'Buy_Signal': int(analysis_row.get('Buy_Signal', 0)),
                        'Rel_Range_Signal': int(analysis_row.get('Rel_Range_Signal', 0)),
                        'VW_Range_Percentile': round(float(analysis_row.get('VW_Range_Percentile', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Percentile', 0)) else 0,
                        'VW_Range_Velocity': round(float(analysis_row.get('VW_Range_Velocity', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Velocity', 0)) else 0,
                        'CRT_Velocity': round(float(analysis_row.get('CRT_Qualifying_Velocity', 0)), 4) if not pd.isna(analysis_row.get('CRT_Qualifying_Velocity', 0)) else 0,  # Renamed here
                        'Weekly_Open': round(float(analysis_row.get('Weekly_Open', 0)), 2) if not pd.isna(analysis_row.get('Weekly_Open', 0)) else 0,
                        'CRT_High': round(float(analysis_row.get('CRT_High', 0)), 2) if not pd.isna(analysis_row.get('CRT_High', 0)) else 0,
                        'CRT_Low': round(float(analysis_row.get('CRT_Low', 0)), 2) if not pd.isna(analysis_row.get('CRT_Low', 0)) else 0
                    }
                    results.append(result)
                    
                    if debug_mode:
                        error_logger.log_debug("Result Collection", f"Collected result for {ticker}", result)
                        
                except Exception as e:
                    error_logger.log_error("Result Collection", e, {
                        "ticker": ticker,
                        "analysis_row_index": str(actual_date),
                        "analysis_row_data": str(analysis_row.to_dict()) if hasattr(analysis_row, 'to_dict') else str(analysis_row)
                    })
                    processing_errors.append(f"{ticker}: Result collection failed")
                    continue
                
                # Update progress
                progress = 0.4 + (0.5 * (i + 1) / len(stock_data))
                progress_bar.progress(progress)
                
            except Exception as e:
                error_logger.log_error("Stock Processing", e, {
                    "ticker": ticker,
                    "processing_step": "overall",
                    "data_available": ticker in stock_data
                })
                processing_errors.append(f"{ticker}: {str(e)}")
                continue
        
        # Log processing summary
        if processing_errors:
            error_logger.log_warning("Processing Summary", f"{len(processing_errors)} stocks had processing errors", {
                "errors": processing_errors,
                "success_rate": f"{len(results)}/{len(stock_data)} ({len(results)/len(stock_data)*100:.1f}%)"
            })
        
        # Finalize results
        status_text.text("📊 Preparing results...")
        progress_bar.progress(0.9)
        
        try:
            results_df = pd.DataFrame(results)
            error_logger.log_debug("Results Finalization", f"Created results dataframe", {
                "results_count": len(results_df),
                "columns": list(results_df.columns) if not results_df.empty else []
            })
        except Exception as e:
            error_logger.log_error("Results DataFrame Creation", e, {
                "results_data": results
            })
            raise
        
        # Store results and configuration in session state
        st.session_state.scan_results = results_df
        st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.last_scan_config = {
            'scope': 'Single Stock' if is_single_stock else 'Full Watchlist',
            'date': f'Historical ({analysis_date})' if is_historical else 'Current',
            'stock_count': len(results_df)
        }
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text("✅ Scan completed!")
        
        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Success message
        success_message = f"🎉 Scan completed! Analyzed {len(results_df)} stocks successfully"
        if processing_errors:
            success_message += f" ({len(processing_errors)} errors - check log for details)"
        if is_historical:
            success_message += f" as of {analysis_date}"
        
        st.success(success_message)
        error_logger.log_debug("Scan Completion", "Scan completed successfully", {
            "successful_stocks": len(results_df),
            "total_requested": len(stocks_to_scan),
            "processing_errors": len(processing_errors)
        })
        
        # Auto-refresh to show results
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
        st.error("❌ Scan failed with critical error - check error log for full details")
        
        # Clear progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def display_scan_results(results_df: pd.DataFrame):
    """Display scanning results with error context and dynamic filtering for both buy signals and valid CRT"""
    
    error_logger = st.session_state.error_logger
    
    try:
        if results_df.empty:
            st.warning("No results to display.")
            error_logger.log_warning("Results Display", "Empty results dataframe")
            return
        
        # Summary metrics
        st.subheader("📊 Scan Summary")
        
        total_stocks = len(results_df)
        buy_signals = len(results_df[results_df['Buy_Signal'] == 1])
        expansion_signals = len(results_df[results_df['Rel_Range_Signal'] == 1])
        high_ibs = len(results_df[results_df['IBS'] >= 0.5])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyzed", total_stocks)
        with col2:
            st.metric("Buy Signals", buy_signals, delta=f"{buy_signals/total_stocks*100:.1f}%")
        with col3:
            st.metric("Range Expansion", expansion_signals)
        with col4:
            st.metric("High IBS (≥0.5)", high_ibs)
        
        # Analysis Date Info
        if len(results_df) > 0:
            analysis_dates = results_df['Analysis_Date'].unique()
            if len(analysis_dates) == 1:
                st.info(f"📅 Analysis performed for trading date: **{analysis_dates[0]}**")
            else:
                st.info(f"📅 Analysis performed for dates: **{', '.join(analysis_dates)}**")
        
        # Buy Signals Section with Dynamic Filtering
        st.subheader("🎯 Buy Signals Detected")
        
        buy_signals_df = results_df[results_df['Buy_Signal'] == 1].copy()
        
        if len(buy_signals_df) > 0:
            # Dynamic Filtering for Buy Signals
            show_buy_signals_filter(buy_signals_df)
            
        else:
            st.info("🔍 No buy signals detected in this analysis.")
        
        # Valid CRT Watch List Section with Dynamic Filtering
        st.subheader("📋 Valid CRT Watch List with Dynamic Filtering")
        
        valid_crt_stocks = results_df[results_df['Valid_CRT'] == 1].copy()
        
        if len(valid_crt_stocks) > 0:
            # Sort by CRT qualifying velocity (highest to lowest)
            valid_crt_stocks = valid_crt_stocks.sort_values('CRT_Velocity', ascending=False)
            
            st.info("📅 Stocks that qualified for Valid CRT - Use filters below to refine your selection")
            
            # Show basic table first
            with st.expander("📊 Full Valid CRT List (Click to expand)", expanded=False):
                display_cols = ['Ticker', 'Name', 'Analysis_Date', 'Weekly_Open', 'CRT_Velocity', 'CRT_High', 'CRT_Low']
                column_config = {
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Name': st.column_config.TextColumn('Company Name', width='large'),
                    'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                    'Weekly_Open': st.column_config.NumberColumn('Weekly_Open', format='$%.2f'),
                    'CRT_Velocity': st.column_config.NumberColumn('CRT Velocity', format='%+.4f pp'),
                    'CRT_High': st.column_config.NumberColumn('CRT_High', format='$%.2f'),
                    'CRT_Low': st.column_config.NumberColumn('CRT_Low', format='$%.2f')
                }
                
                st.dataframe(
                    valid_crt_stocks[display_cols],
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Dynamic Velocity Filtering - RADIO BUTTON VERSION
            show_velocity_filter(valid_crt_stocks)
            
        else:
            st.info("No Valid CRT stocks detected in this analysis.")
        
        # Full Results Table with Custom Column Order
        with st.expander("📋 Full Analysis Results", expanded=False):
            # Reorder columns for better analysis flow
            full_results_cols = [
                'Analysis_Date', 'Ticker', 'Name', 'Weekly_Open', 'CRT_High', 'CRT_Low', 
                'Close', 'VW_Range_Percentile', 'VW_Range_Velocity', 'CRT_Velocity',
                'Rel_Range_Signal', 'Valid_CRT', 'Wick_Below', 'Close_Above', 'IBS', 'Buy_Signal'
            ]
            
            # Configure column display
            full_results_column_config = {
                'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Name': st.column_config.TextColumn('Company Name', width='medium'),
                'Weekly_Open': st.column_config.NumberColumn('Weekly Open', format='$%.2f'),
                'CRT_High': st.column_config.NumberColumn('CRT High', format='$%.2f'),
                'CRT_Low': st.column_config.NumberColumn('CRT Low', format='$%.2f'),
                'Close': st.column_config.NumberColumn('Close', format='$%.2f'),
                'VW_Range_Percentile': st.column_config.NumberColumn('VW Range %ile', format='%.4f'),
                'VW_Range_Velocity': st.column_config.NumberColumn('VW Velocity', format='%+.4f pp'),
                'CRT_Velocity': st.column_config.NumberColumn('CRT Velocity', format='%+.4f pp'),
                'Rel_Range_Signal': st.column_config.NumberColumn('Range Signal', width='small'),
                'Valid_CRT': st.column_config.NumberColumn('Valid CRT', width='small'),
                'Wick_Below': st.column_config.NumberColumn('Wick Below', width='small'),
                'Close_Above': st.column_config.NumberColumn('Close Above', width='small'),
                'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                'Buy_Signal': st.column_config.NumberColumn('Buy Signal', width='small')
            }
            
            st.dataframe(
                results_df[full_results_cols],
                column_config=full_results_column_config,
                use_container_width=True,
                hide_index=True
            )
        
        error_logger.log_debug("Results Display", "Successfully displayed all results", {
            "total_displayed": len(results_df),
            "buy_signals_displayed": len(buy_signals_df) if 'buy_signals_df' in locals() else 0
        })
        
    except Exception as e:
        error_logger.log_error("Results Display", e, {
            "results_shape": results_df.shape if not results_df.empty else "Empty DataFrame"
        })
        st.error("❌ Error displaying results - check error log for details")

def get_signal_description(row):
    """Get human-readable signal description"""
    signals = []
    if row['Wick_Below']:
        signals.append('Wick Below')
    if row['Close_Above']:
        signals.append('Close Above')
    
    return ' + '.join(signals) if signals else 'None'

if __name__ == "__main__":
    show()