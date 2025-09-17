"""
CRT Higher H/L Scanner - PART 1
Enhanced with flexible base filtering for Valid CRT and/or Higher H/L patterns
PHASE 2 ENHANCED: Added strategy filtering with momentum + autocorr analysis
Updated for dual timeframe momentum (5-day and 20-day) display
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

def classify_trading_strategy(momentum_1d, autocorr):
    """
    DEPRECATED: Use Strategy_Type column directly from technical analysis
    Kept for backward compatibility
    """
    # This function is no longer needed as classification happens in technical_analysis.py
    # But kept for any legacy code that might call it
    if momentum_1d > 0.60 and autocorr > 0.15:
        return "Pure Momentum"
    elif momentum_1d < 0.45 and autocorr < -0.15:
        return "Pure Mean Reversion"
    elif momentum_1d > 0.60 and autocorr < -0.15:
        return "Momentum + Daily Reversals"
    elif momentum_1d < 0.50 and autocorr > 0.15:
        return "Weak Momentum + Persistence"
    else:
        return "Neutral/Mixed"

def apply_strategy_filter(filtered_stocks, selected_strategy):
    """Apply strategy-based filtering using the new Strategy_Type column"""
    
    if selected_strategy == "Strong Momentum Continuation 📈":
        result = filtered_stocks[filtered_stocks['Strategy_Type'] == "Strong Momentum Continuation"]
        st.info(f"Found {len(result)} stocks with strong momentum in both timeframes - Hold for trend")
        
    elif selected_strategy == "Momentum Acceleration ⚡":
        result = filtered_stocks[filtered_stocks['Strategy_Type'] == "Momentum Acceleration"]
        st.info(f"Found {len(result)} stocks with accelerating momentum (5d > 20d by 15%+) - Enter on strength")
        
    elif selected_strategy == "Fresh Momentum Surge 🚀":
        result = filtered_stocks[filtered_stocks['Strategy_Type'] == "Fresh Momentum Surge"]
        st.info(f"Found {len(result)} stocks with new bullish crossovers - Best entry timing")
        
    elif selected_strategy == "Oversold Bounce Setup 🎯":
        result = filtered_stocks[filtered_stocks['Strategy_Type'] == "Oversold Bounce Setup"]
        st.info(f"Found {len(result)} oversold stocks in uptrends - Buy dip for 1-3 day bounce")
        
    elif selected_strategy == "Strong Mean Reversion 🔁":
        result = filtered_stocks[filtered_stocks['Strategy_Type'] == "Strong Mean Reversion"]
        st.info(f"Found {len(result)} stocks with mean reversion in both timeframes - Buy weakness")
        
    elif selected_strategy == "Momentum + Intraday Reversals 🔄":
        result = filtered_stocks[filtered_stocks['Strategy_Type'] == "Momentum with Intraday Reversals"]
        st.info(f"Found {len(result)} momentum stocks with daily reversals - Enter on intraday dips")
        
    elif selected_strategy == "Short-term Weakness 📉":
        result = filtered_stocks[filtered_stocks['Strategy_Type'] == "Short-term Weakness"]
        st.info(f"Found {len(result)} stocks with temporary weakness - Monitor for reversal")
        
    elif selected_strategy == "Regime Change - Caution ⚠️":
        result = filtered_stocks[filtered_stocks['Strategy_Type'] == "Regime Change - Caution"]
        st.warning(f"Found {len(result)} stocks with unstable patterns - Use caution or avoid")
        
    else:  # "All Strategies"
        result = filtered_stocks
        st.info(f"Showing all {len(result)} stocks across all strategies")
    
    return result

def apply_dynamic_filters(base_stocks, results_df):
    """
    Apply dynamic filtering with CRT Velocity, IBS, Higher H/L, and Strategy filters
    UPDATED: New strategy filter options for dual timeframe system
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
    
    # COL 1: CRT VELOCITY PERCENTILE FILTER
    with col1:
        st.markdown("**CRT Velocity Filter:**")
        
        # Get velocity statistics - handle both Valid CRT and non-Valid CRT stocks
        if 'CRT_Velocity' in base_stocks.columns:
            # For stocks with Valid CRT, use CRT_Velocity
            velocities = base_stocks['CRT_Velocity']
        else:
            # If CRT_Velocity not in base_stocks, get from original results
            velocities = base_stocks.index.map(lambda idx: results_df.loc[idx, 'CRT_Velocity'] if idx in results_df.index else 0)
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
                key="crt_velocity_percentile_radio"
            )
            
            # Apply percentile filtering
            if selected_percentile != "No Filter":
                percentile_val = percentile_options[selected_percentile]
                threshold_value = np.percentile(non_zero_velocities, percentile_val)
                filtered_stocks = filtered_stocks[velocities >= threshold_value]
                st.info(f"CRT Velocity ≥ {threshold_value:+.4f} pp")
            else:
                st.info("All velocities included")
            
            # Show velocity statistics
            with st.expander("CRT Velocity Statistics", expanded=False):
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
                st.dataframe(pd.DataFrame(stats_data), hide_index=True, width="stretch")
        else:
            st.info("No CRT Velocity data available for filtering")
    
    # COL 2: IBS FILTER
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
            st.dataframe(pd.DataFrame(ibs_stats), hide_index=True, width="stretch")
    
    # COL 3: HIGHER H/L FILTER
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
                st.dataframe(pd.DataFrame(hl_stats), hide_index=True, width="stretch")
        else:
            st.markdown("**Additional Filters:**")
            st.info("Higher H/L filter not needed - already filtered by base selection")
            selected_higher_hl = "No Filter"
    
    # COL 4: STRATEGY FILTER (UPDATED FOR DUAL TIMEFRAME)
    with col4:
        st.markdown("**Strategy Filter:**")
        
        # Check if momentum data is available
        if 'Strategy_Type' in filtered_stocks.columns:
            
            strategy_filter_options = {
                "Strong Momentum Continuation 📈": "strong_momentum",
                "Momentum Acceleration ⚡": "momentum_accel",
                "Fresh Momentum Surge 🚀": "fresh_surge",
                "Oversold Bounce Setup 🎯": "oversold_bounce",
                "Strong Mean Reversion 🔁": "mean_reversion",
                "Momentum + Intraday Reversals 🔄": "momentum_reversals",
                "Short-term Weakness 📉": "short_weakness",
                "Regime Change - Caution ⚠️": "regime_change",
                "All Strategies": None
            }
            
            selected_strategy = st.radio(
                "Select strategy filter:",
                list(strategy_filter_options.keys()),
                key="strategy_filter_radio"
            )
            
            # Apply strategy filtering
            filtered_stocks = apply_strategy_filter(filtered_stocks, selected_strategy)
            
            # Show strategy statistics
            with st.expander("Strategy Statistics", expanded=False):
                if len(base_stocks) > 0 and 'Strategy_Type' in base_stocks.columns:
                    # Calculate strategy distribution for base stocks
                    strategy_counts = base_stocks['Strategy_Type'].value_counts()
                    
                    strategy_stats = {
                        "Strategy": strategy_counts.index.tolist(),
                        "Count": strategy_counts.values.tolist(),
                        "Percentage": [f"{(count/len(base_stocks)*100):.1f}%" for count in strategy_counts.values]
                    }
                    st.dataframe(pd.DataFrame(strategy_stats), hide_index=True, width="stretch")
        else:
            st.warning("Strategy filtering not available - momentum data missing")
            selected_strategy = "All Strategies"
    
    # Show combined filter summary
    filter_summary = []
    if 'selected_percentile' in locals() and selected_percentile != "No Filter" and len(non_zero_velocities) > 0:
        filter_summary.append(f"CRT Velocity {selected_percentile}")
    if selected_ibs_option == "Custom":
        filter_summary.append(f"IBS ≥ {custom_ibs_value:.2f}")
    elif selected_ibs_option != "No Filter":
        filter_summary.append(f"IBS {selected_ibs_option}")
    if 'selected_higher_hl' in locals() and selected_higher_hl != "No Filter":
        filter_summary.append("Higher H/L Only")
    if selected_strategy != "All Strategies":
        filter_summary.append(f"Strategy: {selected_strategy.split(' ')[0]}")
    
    if filter_summary:
        st.success(f"Active filters: {' + '.join(filter_summary)} → {len(filtered_stocks)} stocks")
    else:
        st.info(f"No additional filters applied → {len(filtered_stocks)} stocks")
    
    # Show distribution charts
    with st.expander("📊 Distribution Analysis", expanded=False):
        charts_to_show = []
        
        # Velocity chart (if available)
        if 'non_zero_velocities' in locals() and len(non_zero_velocities) > 0:
            charts_to_show.append(('velocity', non_zero_velocities, selected_percentile))
        
        # IBS chart
        charts_to_show.append(('ibs', ibs_values, selected_ibs_option))
        
        # Momentum chart (if available)
        if 'Momentum_5Day' in base_stocks.columns:
            charts_to_show.append(('momentum', base_stocks['Momentum_5Day'], selected_strategy))
            charts_to_show.append(('spread', base_stocks['Momentum_Spread'], selected_strategy))
        
        # Create charts in rows of 2
        num_charts = len(charts_to_show)
        for i in range(0, num_charts, 2):
            cols = st.columns(2)
            for j, (chart_type, data, filter_selection) in enumerate(charts_to_show[i:i+2]):
                with cols[j]:
                    if chart_type == 'velocity':
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="CRT Velocity Distribution",
                            labels={'x': 'CRT Velocity (pp)', 'y': 'Count'}
                        )
                        if selected_percentile != "No Filter":
                            percentile_val = percentile_options[selected_percentile]
                            threshold_value = np.percentile(non_zero_velocities, percentile_val)
                            fig.add_vline(x=threshold_value, line_dash="dash", line_color="red", 
                                         annotation_text=f"{selected_percentile} threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'ibs':
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="IBS Distribution",
                            labels={'x': 'IBS', 'y': 'Count'}
                        )
                        if selected_ibs_option == "Custom":
                            fig.add_vline(x=custom_ibs_value, line_dash="dash", line_color="red",
                                        annotation_text=f"Custom ≥ {custom_ibs_value:.2f}")
                        elif selected_ibs_option != "No Filter":
                            percentile_val = ibs_percentile_options[selected_ibs_option]
                            threshold_value = np.percentile(ibs_values, percentile_val)
                            fig.add_vline(x=threshold_value, line_dash="dash", line_color="red",
                                        annotation_text=f"{selected_ibs_option} threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'momentum':
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="5-Day Momentum Distribution",
                            labels={'x': '5-Day Momentum', 'y': 'Count'}
                        )
                        # Add strategy thresholds
                        fig.add_vline(x=0.60, line_dash="dash", line_color="green", 
                                     annotation_text="High Momentum")
                        fig.add_vline(x=0.40, line_dash="dash", line_color="red",
                                     annotation_text="Low Momentum")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == 'spread':
                        fig = px.histogram(
                            x=data,
                            nbins=20,
                            title="Momentum Spread (5d - 20d)",
                            labels={'x': 'Momentum Spread', 'y': 'Count'}
                        )
                        # Add spread thresholds
                        fig.add_vline(x=0.15, line_dash="dash", line_color="green",
                                     annotation_text="Acceleration")
                        fig.add_vline(x=-0.15, line_dash="dash", line_color="red",
                                     annotation_text="Exhaustion")
                        st.plotly_chart(fig, use_container_width=True)
    
    return filtered_stocks

"""
CRT Higher H/L Scanner - PART 2 (COMPLETE)
Main application functions: show(), run_enhanced_stock_scan(), display_scan_results()
Updated with dual timeframe momentum display and enhanced column configurations
"""

def show():
    """Main scanner page display for Higher H/L patterns with flexible base filtering and strategy analysis"""
    
    st.title("📈 CRT Higher H/L Scanner")
    st.markdown("Comprehensive analysis with dual timeframe momentum (5-day & 20-day) and advanced strategy classification")
    
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
    """Execute the enhanced stock scanning process with comprehensive error logging and dynamic decimal formatting"""
    
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
        
        status_text.text("🔄 Analyzing technical indicators and dual timeframe momentum...")
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
                
                # DYNAMIC DECIMAL FORMATTING BASED ON PRICE
                close_price = float(analysis_row['Close'])
                high_price = float(analysis_row['High'])
                low_price = float(analysis_row['Low'])
                
                # Determine decimal places based on price level
                if close_price < 1.00:
                    price_decimals = 3  # For all stocks under $1.00
                else:
                    price_decimals = 2  # For regular stocks $1.00 and above
                
                # Helper function to safely round values
                def safe_round(value, decimals):
                    try:
                        return round(float(value), decimals) if not pd.isna(value) else 0
                    except:
                        return 0
                
                # Collect results with DYNAMIC DECIMAL FORMATTING
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
                        'CRT_Velocity': round(float(analysis_row.get('CRT_Qualifying_Velocity', 0)), 4) if not pd.isna(analysis_row.get('CRT_Qualifying_Velocity', 0)) else 0,
                        'Weekly_Open': safe_round(analysis_row.get('Weekly_Open', 0), price_decimals),
                        'CRT_High': safe_round(analysis_row.get('CRT_High', 0), price_decimals),
                        'CRT_Low': safe_round(analysis_row.get('CRT_Low', 0), price_decimals),
                        'VW_Range_Percentile': round(float(analysis_row.get('VW_Range_Percentile', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Percentile', 0)) else 0,
                        'Rel_Range_Signal': int(analysis_row.get('Rel_Range_Signal', 0)),
                        # NEW DUAL TIMEFRAME COLUMNS
                        'Momentum_5Day': round(float(analysis_row.get('Momentum_5Day', 0.5)), 4) if not pd.isna(analysis_row.get('Momentum_5Day', 0.5)) else 0.5,
                        'Momentum_20Day': round(float(analysis_row.get('Momentum_20Day', 0.5)), 4) if not pd.isna(analysis_row.get('Momentum_20Day', 0.5)) else 0.5,
                        'Momentum_Spread': round(float(analysis_row.get('Momentum_Spread', 0.0)), 4) if not pd.isna(analysis_row.get('Momentum_Spread', 0.0)) else 0.0,
                        'Momentum_Crossover': int(analysis_row.get('Momentum_Crossover', 0)) if not pd.isna(analysis_row.get('Momentum_Crossover', 0)) else 0,
                        'Autocorr_5Day': round(float(analysis_row.get('Autocorr_5Day', 0.0)), 4) if not pd.isna(analysis_row.get('Autocorr_5Day', 0.0)) else 0.0,
                        'Autocorr_20Day': round(float(analysis_row.get('Autocorr_20Day', 0.0)), 4) if not pd.isna(analysis_row.get('Autocorr_20Day', 0.0)) else 0.0,
                        'Strategy_Type': str(analysis_row.get('Strategy_Type', 'Unknown')),
                        'Strategy_Signal': str(analysis_row.get('Strategy_Signal', '❓')),
                        # Keep backward compatibility columns
                        'Momentum_1Day_Prob': round(float(analysis_row.get('Momentum_5Day', 0.5)), 4),
                        'Momentum_3Day_Prob': round(float(analysis_row.get('Momentum_5Day', 0.5)), 4),
                        'Autocorr_1Day': round(float(analysis_row.get('Autocorr_5Day', 0.0)), 4),
                        # Store price level for display formatting
                        'Price_Decimals': price_decimals
                    }
                    
                    results.append(result)
                    
                    if debug_mode:
                        error_logger.log_debug("Result Collection", f"Collected result for {ticker}", {
                            "close_price": close_price,
                            "price_decimals": price_decimals,
                            "formatted_close": result['Close']
                        })
                        
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
    """Display scanning results with dual timeframe momentum columns"""
    
    error_logger = st.session_state.error_logger
    
    try:
        if results_df.empty:
            st.warning("No results to display.")
            error_logger.log_warning("Results Display", "Empty results dataframe")
            return
        
        # Summary metrics with MOMENTUM STATISTICS
        st.subheader("📊 Scan Summary")
        
        total_stocks = len(results_df)
        higher_hl_count = len(results_df[results_df['Higher_HL'] == 1])
        valid_crt_count = len(results_df[results_df['Valid_CRT'] == 1])
        higher_hl_with_crt = len(results_df[(results_df['Higher_HL'] == 1) & (results_df['Valid_CRT'] == 1)])
        
        # Momentum statistics
        if 'Momentum_5Day' in results_df.columns:
            high_momentum_count = len(results_df[results_df['Momentum_5Day'] > 0.60])
            momentum_accel_count = len(results_df[results_df['Momentum_Spread'] > 0.15])
            avg_momentum_5d = results_df['Momentum_5Day'].mean()
            avg_momentum_20d = results_df['Momentum_20Day'].mean()
        else:
            high_momentum_count = momentum_accel_count = 0
            avg_momentum_5d = avg_momentum_20d = 0.5
        
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
            st.metric("High Momentum", high_momentum_count, delta="5d > 60%")
        with col6:
            st.metric("Accelerating", momentum_accel_count, delta="Spread > 15%")
        
        # Analysis Date Info with momentum averages
        if len(results_df) > 0:
            analysis_dates = results_df['Analysis_Date'].unique()
            if len(analysis_dates) == 1:
                st.info(f"📅 Analysis date: **{analysis_dates[0]}** | Avg Momentum: **{avg_momentum_5d:.1%}** (5d) / **{avg_momentum_20d:.1%}** (20d)")
            else:
                st.info(f"📅 Analysis dates: **{', '.join(analysis_dates)}** | Avg Momentum: **{avg_momentum_5d:.1%}** (5d) / **{avg_momentum_20d:.1%}** (20d)")
        
        # Base Filter Selection
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
            # Apply dynamic filters (includes new strategy filter)
            filtered_stocks = apply_dynamic_filters(base_stocks, results_df)
            
            # Display filtered results
            st.subheader(f"📋 Filtered Results ({len(filtered_stocks)} stocks)")
            
            if len(filtered_stocks) > 0:
                # Sort by Strategy Type first, then by momentum spread
                if 'Strategy_Type' in filtered_stocks.columns:
                    # Define strategy sort order
                    strategy_order = ['Fresh Momentum Surge', 'Momentum Acceleration', 'Strong Momentum Continuation',
                                    'Oversold Bounce Setup', 'Momentum with Intraday Reversals', 
                                    'Short-term Weakness', 'Strong Mean Reversion', 
                                    'Regime Change - Caution', 'Neutral - No Edge', 'Mixed Signals']
                    strategy_map = {strategy: i for i, strategy in enumerate(strategy_order)}
                    filtered_stocks['strategy_sort'] = filtered_stocks['Strategy_Type'].map(lambda x: strategy_map.get(x, 999))
                    filtered_stocks_sorted = filtered_stocks.sort_values(['strategy_sort', 'Momentum_Spread'], ascending=[True, False])
                    filtered_stocks_sorted = filtered_stocks_sorted.drop('strategy_sort', axis=1)
                else:
                    filtered_stocks_sorted = filtered_stocks.sort_values('CRT_Velocity', ascending=False)
                
                # Display columns based on selected filter with DUAL TIMEFRAME
                if selected_base_filter in ["Valid CRT + Higher H/L", "Valid CRT Only"]:
                    # Show CRT-focused columns with dual momentum
                    display_cols = ['Ticker', 'Name', 'Strategy_Signal', 'Strategy_Type', 'Close', 
                                   'CRT_High', 'CRT_Low', 'CRT_Velocity', 'IBS', 
                                   'Momentum_5Day', 'Momentum_20Day', 'Momentum_Spread']
                    
                    column_config = {
                        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                        'Name': st.column_config.TextColumn('Company Name', width='medium'),
                        'Strategy_Signal': st.column_config.TextColumn('', width='small'),
                        'Strategy_Type': st.column_config.TextColumn('Strategy', width='medium'),
                        'Close': st.column_config.NumberColumn('Close', format='$%.2f'),
                        'CRT_High': st.column_config.NumberColumn('CRT High', format='$%.2f'),
                        'CRT_Low': st.column_config.NumberColumn('CRT Low', format='$%.2f'),
                        'CRT_Velocity': st.column_config.NumberColumn('CRT Vel', format='%+.4f'),
                        'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                        'Momentum_5Day': st.column_config.NumberColumn('Mom 5D', format='%.1%'),
                        'Momentum_20Day': st.column_config.NumberColumn('Mom 20D', format='%.1%'),
                        'Momentum_Spread': st.column_config.NumberColumn('Spread', format='%+.1%')
                    }
                
                elif selected_base_filter == "Higher H/L Only":
                    # Show H/L pattern-focused columns with momentum
                    display_cols = ['Ticker', 'Name', 'Strategy_Signal', 'Strategy_Type', 'Close', 
                                   'High', 'Low', 'IBS', 'Momentum_5Day', 'Momentum_20Day', 
                                   'Momentum_Crossover']
                    
                    column_config = {
                        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                        'Name': st.column_config.TextColumn('Company Name', width='medium'),
                        'Strategy_Signal': st.column_config.TextColumn('', width='small'),
                        'Strategy_Type': st.column_config.TextColumn('Strategy', width='medium'),
                        'Close': st.column_config.NumberColumn('Close', format='$%.2f'),
                        'High': st.column_config.NumberColumn('Day High', format='$%.2f'),
                        'Low': st.column_config.NumberColumn('Day Low', format='$%.2f'),
                        'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                        'Momentum_5Day': st.column_config.NumberColumn('Mom 5D', format='%.1%'),
                        'Momentum_20Day': st.column_config.NumberColumn('Mom 20D', format='%.1%'),
                        'Momentum_Crossover': st.column_config.NumberColumn('Cross', help='1=Bullish, -1=Bearish, 0=None')
                    }
                
                else:  # All Stocks
                    # Show comprehensive columns including dual momentum
                    display_cols = ['Ticker', 'Name', 'Strategy_Signal', 'Strategy_Type', 'Close', 
                                   'IBS', 'Valid_CRT', 'Higher_HL', 'Momentum_5Day', 
                                   'Momentum_20Day', 'Momentum_Spread']
                    
                    column_config = {
                        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                        'Name': st.column_config.TextColumn('Company Name', width='medium'),
                        'Strategy_Signal': st.column_config.TextColumn('', width='small'),
                        'Strategy_Type': st.column_config.TextColumn('Strategy', width='medium'),
                        'Close': st.column_config.NumberColumn('Close', format='$%.2f'),
                        'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                        'Valid_CRT': st.column_config.NumberColumn('CRT', width='small'),
                        'Higher_HL': st.column_config.NumberColumn('H/L', width='small'),
                        'Momentum_5Day': st.column_config.NumberColumn('Mom 5D', format='%.1%'),
                        'Momentum_20Day': st.column_config.NumberColumn('Mom 20D', format='%.1%'),
                        'Momentum_Spread': st.column_config.NumberColumn('Spread', format='%+.1%')
                    }
                
                st.dataframe(
                    filtered_stocks_sorted[display_cols],
                    column_config=column_config,
                    width='stretch',
                    hide_index=True
                )
                
                # Enhanced Strategy Summary with emojis
                if 'Strategy_Type' in filtered_stocks_sorted.columns:
                    st.subheader("📈 Strategy Summary")
                    strategy_summary = filtered_stocks_sorted.groupby(['Strategy_Type', 'Strategy_Signal']).size().reset_index(name='Count')
                    
                    if len(strategy_summary) > 0:
                        # Create a more visual display
                        summary_cols = st.columns(min(len(strategy_summary), 4))
                        for i, row in enumerate(strategy_summary.itertuples()):
                            with summary_cols[i % 4]:
                                st.metric(
                                    f"{row.Strategy_Signal} {row.Strategy_Type}", 
                                    row.Count,
                                    delta=f"{row.Count/len(filtered_stocks_sorted)*100:.1f}%"
                                )
                
                # TradingView Export
                st.subheader("📋 TradingView Export (Filtered)")
                tv_tickers = [f"SGX:{ticker.replace('.SI', '')}" for ticker in filtered_stocks_sorted['Ticker'].tolist()]
                tv_string = ','.join(tv_tickers)
                
                st.text_area(
                    f"Singapore Exchange (SGX) - {selected_base_filter} ({len(tv_tickers)} stocks):",
                    value=tv_string,
                    height=100,
                    help="Copy and paste into TradingView watchlist. SGX: prefix ensures Singapore Exchange stocks."
                )
                
                # Export filtered data
                csv_data = filtered_stocks_sorted.to_csv(index=False)
                filename_prefix = selected_base_filter.lower().replace(' ', '_').replace('+', 'and')
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name=f"{filename_prefix}_{len(filtered_stocks)}_stocks.csv",
                    mime="text/csv"
                )
            
            else:
                st.warning("No stocks match the current filter criteria")
        
        else:
            st.warning(f"No stocks found for pattern: {selected_base_filter}")
        
        # Full Results Table with DUAL TIMEFRAME
        with st.expander("📋 Full Analysis Results", expanded=False):
            full_results_cols = [
                'Analysis_Date', 'Ticker', 'Name', 'Strategy_Signal', 'Strategy_Type', 
                'High', 'Low', 'Close', 'Higher_HL', 'Valid_CRT', 'CRT_Velocity', 'IBS', 
                'Momentum_5Day', 'Momentum_20Day', 'Momentum_Spread', 'Momentum_Crossover',
                'Autocorr_5Day', 'Autocorr_20Day'
            ]
            
            full_results_column_config = {
                'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Name': st.column_config.TextColumn('Company Name', width='medium'),
                'Strategy_Signal': st.column_config.TextColumn('', width='small'),
                'Strategy_Type': st.column_config.TextColumn('Strategy', width='medium'),
                'High': st.column_config.NumberColumn('High', format='$%.2f'),
                'Low': st.column_config.NumberColumn('Low', format='$%.2f'),
                'Close': st.column_config.NumberColumn('Close', format='$%.2f'),
                'Higher_HL': st.column_config.NumberColumn('H/L', width='small'),
                'Valid_CRT': st.column_config.NumberColumn('CRT', width='small'),
                'CRT_Velocity': st.column_config.NumberColumn('CRT Vel', format='%+.4f'),
                'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                'Momentum_5Day': st.column_config.NumberColumn('Mom 5D', format='%.4f'),
                'Momentum_20Day': st.column_config.NumberColumn('Mom 20D', format='%.4f'),
                'Momentum_Spread': st.column_config.NumberColumn('Spread', format='%+.4f'),
                'Momentum_Crossover': st.column_config.NumberColumn('Cross', width='small'),
                'Autocorr_5Day': st.column_config.NumberColumn('Auto 5D', format='%+.4f'),
                'Autocorr_20Day': st.column_config.NumberColumn('Auto 20D', format='%+.4f')
            }
            
            st.dataframe(
                results_df[full_results_cols],
                column_config=full_results_column_config,
                width='stretch',
                hide_index=True
            )
        
        error_logger.log_debug("Results Display", "Successfully displayed all results", {
            "total_displayed": len(results_df),
            "base_filter": selected_base_filter,
            "base_stocks_count": len(base_stocks) if 'base_stocks' in locals() else 0,
            "filtered_stocks_count": len(filtered_stocks) if 'filtered_stocks' in locals() else 0
        })
        
    except Exception as e:
        error_logger.log_error("Results Display", e, {
            "results_shape": results_df.shape if not results_df.empty else "Empty DataFrame"
        })
        st.error("❌ Error displaying results - check error log for details")

if __name__ == "__main__":
    show()