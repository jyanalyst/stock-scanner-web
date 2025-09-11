"""
CRT Close Above Scanner
Focus on stocks that close above CRT High (breakout patterns)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import time
import traceback
import logging
import sys
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

def show_close_above_filter(close_above_stocks):
    """
    Display dynamic filtering for Close Above stocks
    Focus on breakout strength and momentum
    """
    
    if close_above_stocks.empty:
        st.warning("No Close Above patterns available for filtering")
        return close_above_stocks
    
    st.subheader("🎯 Dynamic Filtering")
    
    # Create two columns for filters
    col1, col2 = st.columns(2)
    
    # Initialize filtered stocks
    filtered_stocks = close_above_stocks.copy()
    
    # CRT VELOCITY PERCENTILE FILTER
    with col1:
        st.markdown("**CRT Velocity Filter:**")
        
        # Get velocity statistics
        velocities = close_above_stocks['CRT_Velocity']
        
        percentile_options = {
            "Top 25%": 75,
            "Top 50%": 50, 
            "Top 75%": 25,
            "No Filter": None
        }
        
        selected_percentile = st.radio(
            "Select velocity filter:",
            list(percentile_options.keys()),
            key="close_above_percentile_radio"
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
    
    # BREAKOUT STRENGTH FILTER
    with col2:
        st.markdown("**Breakout Strength Filter:**")
        
        # Calculate breakout percentage (how far above CRT High)
        breakout_pct = ((close_above_stocks['Close'] - close_above_stocks['CRT_High']) / 
                       close_above_stocks['CRT_High'] * 100)
        
        breakout_options = {
            "Strong Breakout (>2%)": 2.0,
            "Moderate Breakout (>1%)": 1.0,
            "Any Breakout": 0,
            "No Filter": None
        }
        
        selected_breakout = st.radio(
            "Select breakout strength:",
            list(breakout_options.keys()),
            key="close_above_breakout_radio"
        )
        
        # Apply breakout filtering
        if selected_breakout != "No Filter":
            breakout_threshold = breakout_options[selected_breakout]
            mask = breakout_pct >= breakout_threshold
            filtered_stocks = filtered_stocks[mask.loc[filtered_stocks.index]]
            st.info(f"Breakout strength ≥ {breakout_threshold}%")
        else:
            st.info("All breakout strengths included")
        
        # Show breakout statistics
        with st.expander("Breakout Strength Statistics", expanded=False):
            breakout_stats = {
                "Metric": ["Count", "Min", "25th %ile", "Median", "75th %ile", "Max"],
                "Value": [
                    len(breakout_pct),
                    f"{breakout_pct.min():.2f}%",
                    f"{np.percentile(breakout_pct, 25):.2f}%",
                    f"{breakout_pct.median():.2f}%",
                    f"{np.percentile(breakout_pct, 75):.2f}%",
                    f"{breakout_pct.max():.2f}%"
                ]
            }
            st.dataframe(pd.DataFrame(breakout_stats), hide_index=True, use_container_width=True)
    
    # Show combined filter summary
    filter_summary = []
    if selected_percentile != "No Filter":
        filter_summary.append(f"CRT Velocity {selected_percentile}")
    if selected_breakout != "No Filter":
        filter_summary.append(f"{selected_breakout}")
    
    if filter_summary:
        st.success(f"Active filters: {' + '.join(filter_summary)} → {len(filtered_stocks)} patterns")
    else:
        st.info(f"No filters applied → {len(filtered_stocks)} patterns")
    
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
            if selected_percentile != "No Filter":
                percentile_val = percentile_options[selected_percentile]
                threshold_value = np.percentile(velocities, percentile_val)
                fig_hist.add_vline(x=threshold_value, line_dash="dash", line_color="red", 
                                 annotation_text=f"{selected_percentile} threshold")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Breakout strength distribution
            fig_breakout = px.histogram(
                x=breakout_pct,
                nbins=20,
                title="Breakout Strength Distribution (%)",
                labels={'x': 'Breakout Strength (%)', 'y': 'Count'}
            )
            if selected_breakout != "No Filter":
                breakout_threshold = breakout_options[selected_breakout]
                fig_breakout.add_vline(x=breakout_threshold, line_dash="dash", line_color="red",
                                     annotation_text=f"{selected_breakout} threshold")
            st.plotly_chart(fig_breakout, use_container_width=True)
    
    # Display filtered results
    st.subheader(f"📋 Filtered Results ({len(filtered_stocks)} patterns)")
    
    if len(filtered_stocks) > 0:
        # Sort by CRT Velocity (descending)
        filtered_stocks_sorted = filtered_stocks.sort_values('CRT_Velocity', ascending=False)
        
        # Add breakout strength column
        filtered_stocks_sorted['Breakout_%'] = ((filtered_stocks_sorted['Close'] - filtered_stocks_sorted['CRT_High']) / 
                                               filtered_stocks_sorted['CRT_High'] * 100)
        
        # Display columns focused on breakout analysis
        display_cols = ['Ticker', 'Name', 'Close', 'CRT_High', 'High', 'Breakout_%', 
                       'Volume', 'CRT_Velocity', 'IBS']
        
        # Add volume change if possible
        filtered_stocks_sorted['Volume'] = filtered_stocks_sorted.get('Volume', 0)
        
        column_config = {
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Name': st.column_config.TextColumn('Company Name', width='large'),
            'Close': st.column_config.NumberColumn('Close', format='$%.2f'),
            'CRT_High': st.column_config.NumberColumn('CRT High', format='$%.2f'),
            'High': st.column_config.NumberColumn('Day High', format='$%.2f'),
            'Breakout_%': st.column_config.NumberColumn('Breakout %', format='%+.2f'),
            'Volume': st.column_config.NumberColumn('Volume', format='%,.0f'),
            'CRT_Velocity': st.column_config.NumberColumn('CRT Velocity', format='%+.4f pp'),
            'IBS': st.column_config.NumberColumn('IBS', format='%.3f')
        }
        
        st.dataframe(
            filtered_stocks_sorted[display_cols],
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        
        # TradingView Export
        st.subheader("📋 TradingView Export (Filtered)")
        tv_tickers = [f"SGX:{ticker.replace('.SI', '')}" for ticker in filtered_stocks_sorted['Ticker'].tolist()]
        tv_string = ','.join(tv_tickers)
        
        st.text_area(
            f"Singapore Exchange (SGX) Close Above Patterns ({len(tv_tickers)} stocks):",
            value=tv_string,
            height=100,
            help="Copy and paste into TradingView watchlist. SGX: prefix ensures Singapore Exchange stocks."
        )
        
        # Export filtered data
        csv_data = filtered_stocks_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"close_above_patterns_{len(filtered_stocks)}_stocks.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No patterns match the current filter criteria")
    
    return filtered_stocks

def show():
    """Main scanner page display for Close Above patterns"""
    
    st.title("🚀 CRT Close Above Scanner")
    st.markdown("Focus on stocks breaking above CRT High levels (momentum breakouts)")
    
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
    """Execute the enhanced stock scanning process with comprehensive error logging"""
    
    error_logger = st.session_state.error_logger
    
    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher
        from core.technical_analysis import add_enhanced_columns
        
        error_logger.log_debug("Scan Start", "Starting enhanced stock scan", {
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
        
        st.info(f"🔄 Scanning {scope_text} with {date_text}... This may take a moment.")
        
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
                
                try:
                    company_name = fetcher.get_company_name(ticker)
                    error_logger.log_debug("Company Name", f"Retrieved name for {ticker}: {company_name}")
                except Exception as e:
                    error_logger.log_warning("Company Name", f"Failed to get name for {ticker}: {e}")
                    company_name = ticker.replace('.SI', '')
                
                # Collect results (focused on Close Above)
                try:
                    # Get volume from raw data
                    volume = float(df_raw.loc[actual_date, 'Volume']) if 'Volume' in df_raw.columns and actual_date in df_raw.index else 0
                    
                    result = {
                        'Ticker': ticker,
                        'Name': company_name,
                        'Analysis_Date': actual_date.strftime('%Y-%m-%d') if hasattr(actual_date, 'strftime') else str(actual_date),
                        'Close': round(float(analysis_row['Close']), 2),
                        'High': round(float(analysis_row['High']), 2),
                        'Low': round(float(analysis_row['Low']), 2),
                        'Volume': int(volume),
                        'IBS': round(float(analysis_row['IBS']), 3) if not pd.isna(analysis_row['IBS']) else 0,
                        'Valid_CRT': int(analysis_row.get('Valid_CRT', 0)),
                        'Close_Above': int(analysis_row.get('Close_Above', 0)),
                        'CRT_Velocity': round(float(analysis_row.get('CRT_Qualifying_Velocity', 0)), 4) if not pd.isna(analysis_row.get('CRT_Qualifying_Velocity', 0)) else 0,
                        'Weekly_Open': round(float(analysis_row.get('Weekly_Open', 0)), 2) if not pd.isna(analysis_row.get('Weekly_Open', 0)) else 0,
                        'CRT_High': round(float(analysis_row.get('CRT_High', 0)), 2) if not pd.isna(analysis_row.get('CRT_High', 0)) else 0,
                        'CRT_Low': round(float(analysis_row.get('CRT_Low', 0)), 2) if not pd.isna(analysis_row.get('CRT_Low', 0)) else 0,
                        'VW_Range_Percentile': round(float(analysis_row.get('VW_Range_Percentile', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Percentile', 0)) else 0,
                        'Rel_Range_Signal': int(analysis_row.get('Rel_Range_Signal', 0))
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
        
        if processing_errors:
            error_logger.log_warning("Processing Summary", f"{len(processing_errors)} stocks had processing errors", {
                "errors": processing_errors,
                "success_rate": f"{len(results)}/{len(stock_data)} ({len(results)/len(stock_data)*100:.1f}%)"
            })
        
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
        
        st.session_state.scan_results = results_df
        st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.last_scan_config = {
            'scope': 'Single Stock' if is_single_stock else 'Full Watchlist',
            'date': f'Historical ({analysis_date})' if is_historical else 'Current',
            'stock_count': len(results_df)
        }
        
        progress_bar.progress(1.0)
        status_text.text("✅ Scan completed!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
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
        
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

def display_scan_results(results_df: pd.DataFrame):
    """Display scanning results focused on Close Above patterns"""
    
    error_logger = st.session_state.error_logger
    
    try:
        if results_df.empty:
            st.warning("No results to display.")
            error_logger.log_warning("Results Display", "Empty results dataframe")
            return
        
        # Summary metrics
        st.subheader("📊 Scan Summary")
        
        total_stocks = len(results_df)
        close_above_count = len(results_df[results_df['Close_Above'] == 1])
        valid_crt_count = len(results_df[results_df['Valid_CRT'] == 1])
        close_above_with_crt = len(results_df[(results_df['Close_Above'] == 1) & (results_df['Valid_CRT'] == 1)])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyzed", total_stocks)
        with col2:
            st.metric("Close Above Patterns", close_above_count, delta=f"{close_above_count/total_stocks*100:.1f}%")
        with col3:
            st.metric("Valid CRT", valid_crt_count)
        with col4:
            st.metric("Close Above + CRT", close_above_with_crt)
        
        # Analysis Date Info
        if len(results_df) > 0:
            analysis_dates = results_df['Analysis_Date'].unique()
            if len(analysis_dates) == 1:
                st.info(f"📅 Analysis performed for trading date: **{analysis_dates[0]}**")
            else:
                st.info(f"📅 Analysis performed for dates: **{', '.join(analysis_dates)}**")
        
        # Close Above Patterns Section
        st.subheader("🚀 Close Above Patterns Detected")
        
        close_above_df = results_df[results_df['Close_Above'] == 1].copy()
        
        if len(close_above_df) > 0:
            # Show stocks with both Close Above AND Valid CRT first
            close_above_with_crt_df = close_above_df[close_above_df['Valid_CRT'] == 1]
            
            if len(close_above_with_crt_df) > 0:
                st.success(f"🎯 {len(close_above_with_crt_df)} stocks with Close Above + Valid CRT")
                show_close_above_filter(close_above_with_crt_df)
            else:
                st.info("📊 Close Above patterns found but none with Valid CRT")
                # Still show the filtering interface for all Close Above patterns
                show_close_above_filter(close_above_df)
        else:
            st.info("🔍 No Close Above patterns detected in this analysis.")
        
        # Valid CRT Watch List Section
        st.subheader("📋 Valid CRT Watch List")
        
        valid_crt_stocks = results_df[results_df['Valid_CRT'] == 1].copy()
        
        if len(valid_crt_stocks) > 0:
            valid_crt_stocks = valid_crt_stocks.sort_values('CRT_Velocity', ascending=False)
            
            with st.expander("📊 Full Valid CRT List (Click to expand)", expanded=False):
                display_cols = ['Ticker', 'Name', 'Analysis_Date', 'Weekly_Open', 'CRT_Velocity', 
                              'CRT_High', 'CRT_Low', 'Close_Above', 'IBS']
                
                column_config = {
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Name': st.column_config.TextColumn('Company Name', width='large'),
                    'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                    'Weekly_Open': st.column_config.NumberColumn('Weekly_Open', format='$%.2f'),
                    'CRT_Velocity': st.column_config.NumberColumn('CRT Velocity', format='%+.4f pp'),
                    'CRT_High': st.column_config.NumberColumn('CRT_High', format='$%.2f'),
                    'CRT_Low': st.column_config.NumberColumn('CRT_Low', format='$%.2f'),
                    'Close_Above': st.column_config.NumberColumn('Close Above', width='small'),
                    'IBS': st.column_config.NumberColumn('IBS', format='%.3f')
                }
                
                st.dataframe(
                    valid_crt_stocks[display_cols],
                    column_config=column_config,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No Valid CRT stocks detected in this analysis.")
        
        # Full Results Table
        with st.expander("📋 Full Analysis Results", expanded=False):
            full_results_cols = [
                'Analysis_Date', 'Ticker', 'Name', 'High', 'CRT_High', 'Close', 
                'Close_Above', 'Valid_CRT', 'CRT_Velocity', 'Volume', 
                'Rel_Range_Signal', 'IBS'
            ]
            
            full_results_column_config = {
                'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Name': st.column_config.TextColumn('Company Name', width='medium'),
                'High': st.column_config.NumberColumn('Day High', format='$%.2f'),
                'CRT_High': st.column_config.NumberColumn('CRT High', format='$%.2f'),
                'Close': st.column_config.NumberColumn('Close', format='$%.2f'),
                'Close_Above': st.column_config.NumberColumn('Close Above', width='small'),
                'Valid_CRT': st.column_config.NumberColumn('Valid CRT', width='small'),
                'CRT_Velocity': st.column_config.NumberColumn('CRT Velocity', format='%+.4f pp'),
                'Volume': st.column_config.NumberColumn('Volume', format='%,.0f'),
                'Rel_Range_Signal': st.column_config.NumberColumn('Range Signal', width='small'),
                'IBS': st.column_config.NumberColumn('IBS', format='%.3f')
            }
            
            st.dataframe(
                results_df[full_results_cols],
                column_config=full_results_column_config,
                use_container_width=True,
                hide_index=True
            )
        
        error_logger.log_debug("Results Display", "Successfully displayed all results", {
            "total_displayed": len(results_df),
            "close_above_displayed": len(close_above_df) if 'close_above_df' in locals() else 0
        })
        
    except Exception as e:
        error_logger.log_error("Results Display", e, {
            "results_shape": results_df.shape if not results_df.empty else "Empty DataFrame"
        })
        st.error("❌ Error displaying results - check error log for details")

if __name__ == "__main__":
    show()