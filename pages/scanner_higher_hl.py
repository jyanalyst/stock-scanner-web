# File: pages/scanner_higher_hl.py
# Part 1 of 4
"""
CRT Higher H/L Scanner - Local File System
Enhanced with Pure MPI Expansion Filtering
NOW USING VW_RANGE_VELOCITY FOR DAILY MOMENTUM FILTERING
Automatically checks for Historical_Data updates on startup
ENHANCED: Force update capability to re-process latest EOD file
UPDATED: Simplified base filter, enhanced H/L filter with Higher_H support
UPDATED: Removed Market Regime filter, moved H/L filter to first position
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

def show_update_prompt():
    """
    Check for updates and prompt user if new data available
    ENHANCED: Added Force Update option
    Returns True if update was performed or skipped, False if check failed
    """
    try:
        from core.local_file_loader import get_local_loader
        
        loader = get_local_loader()
        
        # Check for updates
        with st.spinner("Checking for Historical_Data updates..."):
            needs_update, eod_filename, eod_date = loader.check_for_updates()
        
        if needs_update:
            # Show update prompt
            st.info(f"📥 **New EOD data available!**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latest EOD File", eod_filename)
            with col2:
                st.metric("Date", eod_date.strftime('%d/%m/%Y') if eod_date else 'Unknown')
            
            st.markdown("Historical_Data can be updated with this new data.")
            
            col_update, col_skip = st.columns(2)
            
            with col_update:
                if st.button("🔄 Update Now", type="primary", use_container_width=True):
                    return perform_update(loader, force=False)
            
            with col_skip:
                if st.button("⏭️ Skip This Time", use_container_width=True):
                    st.info("Skipped update. You can update later using the buttons below.")
                    return True
            
            return False  # Waiting for user action
        
        else:
            # No new updates - show current status and force update option
            st.success("✅ Historical_Data is up to date!")
            
            # Get latest EOD file info for force update
            latest_eod = loader.get_latest_eod_file()
            if latest_eod:
                eod_date_str = latest_eod.replace('.csv', '')
                eod_date = datetime.strptime(eod_date_str, '%d_%b_%Y')
                
                st.info(f"📄 Latest EOD file: **{latest_eod}** ({eod_date.strftime('%d/%m/%Y')})")
                
                # Show force update option
                with st.expander("⚙️ Advanced: Force Re-process Latest File", expanded=False):
                    st.warning("⚠️ **Force Update** will re-process the latest EOD file even though it's already been imported.")
                    st.markdown("""
                    **Use this when:**
                    - You've manually updated the contents of the latest EOD file
                    - You need to fix data quality issues in the current date
                    - You want to re-import today's data after corrections
                    
                    **This will:**
                    - Remove existing entries for this date from Historical_Data
                    - Re-import all data from the latest EOD file
                    - Preserve all other historical data
                    """)
                    
                    if st.button("🔧 Force Update Latest File", type="secondary", use_container_width=True, key="force_update_button"):
                        result = perform_update(loader, force=True)
                        if result:
                            st.success("Force update completed!")
                            time.sleep(2)
                        return result
                
                return False
            
            return True
            
    except Exception as e:
        st.session_state.error_logger.log_error("Update Check", e)
        st.error("❌ Could not check for updates")
        return False

def perform_update(loader, force: bool = False) -> bool:
    """
    Perform the Historical_Data update
    ENHANCED: Added force parameter to bypass date checks
    
    Args:
        loader: LocalFileLoader instance
        force: If True, force re-processing of latest EOD file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        update_type = "FORCE UPDATE" if force else "UPDATE"
        status_text.text(f"🔄 Starting Historical_Data {update_type.lower()}...")
        
        # Show force update warning
        if force:
            st.warning(f"⚠️ **FORCE UPDATE MODE** - Re-processing latest EOD file regardless of dates")
        
        # Perform update
        stats = loader.update_historical_from_eod(force=force)
        
        # Show progress during update
        if stats['total_stocks'] > 0:
            for i, detail in enumerate(stats['details']):
                progress = (i + 1) / stats['total_stocks']
                progress_bar.progress(progress)
                
                ticker = detail['ticker']
                status = detail['status']
                
                if status == 'updated':
                    status_text.text(f"✅ {ticker} - {detail['message']}")
                elif status == 'created':
                    status_text.text(f"🆕 {ticker} - {detail['message']}")
                elif status == 'skipped':
                    status_text.text(f"⏭️ {ticker} - {detail['message']}")
                elif status == 'error':
                    status_text.text(f"❌ {ticker} - {detail['message']}")
                
                time.sleep(0.1)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show summary
        if force:
            st.success(f"✅ **Force Update Complete!**")
        else:
            st.success(f"✅ **Update Complete!**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stocks", stats['total_stocks'])
        with col2:
            st.metric("Updated", stats['updated'], delta=f"+{stats['created']} created")
        with col3:
            st.metric("Skipped", stats['skipped'])
        with col4:
            st.metric("Errors", stats['errors'])
        
        if stats['eod_date']:
            date_msg = f"📅 Historical_Data now current through: **{stats['eod_date']}**"
            if force:
                date_msg += " **(FORCED)**"
            st.info(date_msg)
        
        # Show errors if any
        if stats['errors'] > 0:
            with st.expander("⚠️ View Errors", expanded=False):
                for detail in stats['details']:
                    if detail['status'] == 'error':
                        st.write(f"**{detail['ticker']}**: {detail['message']}")
        
        time.sleep(2)
        return True
        
    except Exception as e:
        st.session_state.error_logger.log_error("Update Execution", e)
        st.error("❌ Update failed - check error log for details")
        return False

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
# COMPLETE FIXED VERSION of apply_velocity_filter

def apply_velocity_filter(filtered_stocks: pd.DataFrame, results_df: pd.DataFrame) -> tuple:
    """
    Apply VW Range Velocity percentile filter for daily momentum
    UPDATED: Added Custom filter option similar to IBS filter
    FIXED: Robust error handling and guaranteed return
    """
    # Early return if no stocks
    if filtered_stocks.empty:
        return filtered_stocks, None, "No stocks available"
    
    # Get velocity data
    if 'VW_Range_Velocity' in filtered_stocks.columns:
        velocities = filtered_stocks['VW_Range_Velocity']
    else:
        velocities = filtered_stocks.index.map(
            lambda idx: results_df.loc[idx, 'VW_Range_Velocity'] if idx in results_df.index else 0
        )
        velocities = pd.Series(velocities, index=filtered_stocks.index)
    
    # Remove zero velocities for percentile calculation
    non_zero_velocities = velocities[velocities != 0]
    
    # Early return if no velocity data
    if len(non_zero_velocities) == 0:
        return filtered_stocks, None, "No VW Range Velocity data available"
    
    # Percentile options with Custom added
    percentile_options = {
        "Top 25%": 75,
        "Top 50%": 50, 
        "Top 75%": 25,
        "Custom": "custom",
        "No Filter": None
    }
    
    # Radio button with default index
    selected_percentile = st.radio(
        "Select velocity filter:",
        list(percentile_options.keys()),
        index=4,  # Default to "No Filter" (5th option, index 4)
        key="vw_range_velocity_percentile_radio"
    )
    
    # Initialize info_message
    info_message = "All velocities included"
    
    # Apply filtering based on selection
    if selected_percentile == "Custom":
        # Add custom number input for velocity threshold
        custom_velocity_value = st.number_input(
            "Enter minimum VW Range Velocity:",
            min_value=-1.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.4f",
            key="custom_velocity_input",
            help="Filter stocks with velocity >= this value. Positive = expanding, Negative = contracting"
        )
        filtered_stocks = filtered_stocks[velocities >= custom_velocity_value]
        info_message = f"VW Range Velocity ≥ {custom_velocity_value:+.4f} pp"
        
    elif selected_percentile in ["Top 25%", "Top 50%", "Top 75%"]:
        percentile_val = percentile_options[selected_percentile]
        threshold_value = np.percentile(non_zero_velocities, percentile_val)
        filtered_stocks = filtered_stocks[velocities >= threshold_value]
        info_message = f"VW Range Velocity ≥ {threshold_value:+.4f} pp"
    
    # else: No Filter - keep info_message as "All velocities included"
    
    # Always return the tuple
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

def apply_relative_volume_filter(filtered_stocks: pd.DataFrame) -> tuple:
    """Apply Relative Volume percentile filter"""
    if filtered_stocks.empty:
        return filtered_stocks, "No stocks available"
    
    rel_volume_values = filtered_stocks['Relative_Volume']
    
    # Percentile options
    rel_volume_percentile_options = {
        "Top 25%": 75,
        "Top 50%": 50,
        "Top 75%": 25,
        "Custom": "custom",
        "No Filter": None
    }
    
    selected_rel_volume_option = st.radio(
        "Select Relative Volume filter:",
        list(rel_volume_percentile_options.keys()),
        key="rel_volume_percentile_radio"
    )
    
    # Apply filtering
    if selected_rel_volume_option == "Custom":
        custom_rel_volume_value = st.number_input(
            "Enter minimum Relative Volume %:",
            min_value=50.0,
            max_value=500.0,
            value=100.0,
            step=10.0,
            format="%.1f",
            key="custom_rel_volume_input",
            help="100% = average volume, 200% = double average volume"
        )
        filtered_stocks = filtered_stocks[filtered_stocks['Relative_Volume'] >= custom_rel_volume_value]
        info_message = f"Rel Volume ≥ {custom_rel_volume_value:.1f}%"
        
    elif selected_rel_volume_option != "No Filter":
        percentile_val = rel_volume_percentile_options[selected_rel_volume_option]
        threshold_value = np.percentile(rel_volume_values, percentile_val)
        filtered_stocks = filtered_stocks[filtered_stocks['Relative_Volume'] >= threshold_value]
        info_message = f"Rel Volume ≥ {threshold_value:.1f}% ({selected_rel_volume_option})"
        
    else:
        info_message = "All Relative Volume levels included"
    
    return filtered_stocks, info_message

# File: pages/scanner_higher_hl.py
# Part 2 of 4

def apply_higher_hl_filter(filtered_stocks: pd.DataFrame, base_filter_type: str) -> tuple:
    """
    Apply Higher H/L pattern filter - UPDATED with Higher_H support
    Now offers three options: Higher H/L Only, Higher H Only, No Filter
    """
    # Filter options with descriptions
    hl_filter_options = {
        "Higher H/L Only": "Both higher high AND higher low (HHL)",
        "Higher H Only": "Any higher high (HH or HHL)",
        "No Filter": "All patterns"
    }
    
    st.markdown("**Higher H/L Filter:**")
    
    selected_hl_filter = st.radio(
        "Select Higher H/L filter:",
        list(hl_filter_options.keys()),
        key="higher_hl_pattern_radio",
        help="Filter by higher high and/or higher low patterns"
    )
    
    # Apply filtering based on selection
    if selected_hl_filter == "Higher H/L Only":
        # Only stocks with both higher high AND higher low
        filtered_stocks = filtered_stocks[filtered_stocks['Higher_HL'] == 1]
        info_message = f"Higher H/L Only (HHL) - {len(filtered_stocks)} stocks"
        
    elif selected_hl_filter == "Higher H Only":
        # Stocks with higher high (includes both HH and HHL)
        filtered_stocks = filtered_stocks[filtered_stocks['Higher_H'] == 1]
        info_message = f"Higher H Only (HH + HHL) - {len(filtered_stocks)} stocks"
        
    else:  # No Filter
        info_message = f"All patterns - {len(filtered_stocks)} stocks"
    
    return filtered_stocks, selected_hl_filter, info_message

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
    """Show statistics for a filter component - UPDATED: Removed Market Regime section"""
    with st.expander(f"{component_name} Statistics", expanded=False):
        if component_name == "VW Range Velocity" and len(data) > 0:
            stats_df = create_filter_statistics_dataframe(data, component_name)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
        elif component_name == "IBS" and len(data) > 0:
            stats_df = create_filter_statistics_dataframe(data, component_name)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
        elif component_name == "Relative Volume" and len(data) > 0:
            stats_df = create_filter_statistics_dataframe(data, component_name)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
        elif component_name == "Higher H/L" and base_stocks is not None:
            # Show statistics for both Higher_H and Higher_HL
            higher_h_count = (base_stocks['Higher_H'] == 1).sum()
            higher_hl_count = (base_stocks['Higher_HL'] == 1).sum()
            hh_only_count = ((base_stocks['Higher_H'] == 1) & (base_stocks['Higher_HL'] == 0)).sum()
            total_count = len(base_stocks)
            
            hl_stats = pd.DataFrame({
                "Pattern": ["HHL (H/L)", "HH Only", "Total Higher H", "Neither", "Total"],
                "Count": [
                    f"{higher_hl_count}",
                    f"{hh_only_count}",
                    f"{higher_h_count}",
                    f"{total_count - higher_h_count}",
                    f"{total_count}"
                ],
                "Percentage": [
                    f"{higher_hl_count/total_count*100:.1f}%",
                    f"{hh_only_count/total_count*100:.1f}%",
                    f"{higher_h_count/total_count*100:.1f}%",
                    f"{(total_count - higher_h_count)/total_count*100:.1f}%",
                    "100.0%"
                ],
                "Description": [
                    "Both higher high AND higher low",
                    "Higher high only (not higher low)",
                    "Any higher high (HH + HHL)",
                    "No higher high pattern",
                    "All stocks"
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
    UPDATED: Removed Market Regime filter, moved H/L filter to first position
    """
    if base_stocks.empty:
        st.warning("No stocks available for filtering")
        return base_stocks
    
    st.subheader("🎯 Dynamic Filtering")
    
    # UPDATED: Create FIVE columns for filters (removed regime, reordered)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Initialize filtered stocks
    filtered_stocks = base_stocks.copy()
    filter_summary = []
    
    # COL 1: HIGHER H/L FILTER - MOVED TO FIRST POSITION
    with col1:
        filtered_stocks, selected_hl_filter, hl_info = apply_higher_hl_filter(filtered_stocks, st.session_state.get('base_filter_type', 'All Stocks'))
        st.info(hl_info)
        
        show_filter_statistics("Higher H/L", None, base_stocks)
        if selected_hl_filter != "No Filter":
            filter_summary.append(f"H/L: {selected_hl_filter}")
    
    # COL 2: VW RANGE VELOCITY FILTER
    with col2:
        st.markdown("**VW Range Velocity Filter:**")
        filtered_stocks, velocity_data, velocity_info = apply_velocity_filter(filtered_stocks, results_df)
        st.info(velocity_info)
        
        if velocity_data is not None and len(velocity_data) > 0:
            show_filter_statistics("VW Range Velocity", velocity_data)
            if "≥" in velocity_info:
                filter_summary.append("VW Range Velocity filtered")
    
    # COL 3: IBS FILTER
    with col3:
        st.markdown("**IBS Filter:**")
        filtered_stocks, ibs_info = apply_ibs_filter(filtered_stocks)
        st.info(ibs_info)
        
        if len(filtered_stocks) > 0:
            show_filter_statistics("IBS", filtered_stocks['IBS'])
            if "≥" in ibs_info:
                filter_summary.append("IBS filtered")
    
    # COL 4: RELATIVE VOLUME FILTER
    with col4:
        st.markdown("**Relative Volume Filter:**")
        filtered_stocks, rel_volume_info = apply_relative_volume_filter(filtered_stocks)
        st.info(rel_volume_info)
        
        if len(filtered_stocks) > 0:
            show_filter_statistics("Relative Volume", filtered_stocks['Relative_Volume'])
            if "≥" in rel_volume_info:
                filter_summary.append("Relative Volume filtered")
    
    # COL 5: MPI EXPANSION FILTER
    with col5:
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
                selected_stock = "A17U.SG"
        
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
                    help="Choose the historical date for analysis"
                )
                
                st.caption("ℹ️ Scanner will filter Historical_Data to this date")
                
            except Exception as e:
                st.session_state.error_logger.log_error("Date Selection", e)
                historical_date = date.today() - timedelta(days=7)
    
    return scan_scope, selected_stock, scan_date_type, historical_date

def show_advanced_settings():
    """Display advanced settings panel"""
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            days_back = st.number_input(
                "Days of Historical Data", 
                min_value=30, 
                max_value=250, 
                value=100,
                help="Number of days to load (loads all available from local files)"
            )
        
        with col2:
            rolling_window = st.number_input(
                "Rolling Window", 
                min_value=10, 
                max_value=50, 
                value=20,
                help="Rolling window for technical calculations"
            )
    
    return days_back, rolling_window

def execute_scan_button(scan_scope, selected_stock, scan_date_type, historical_date, days_back, rolling_window):
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
                rolling_window=rolling_window
            )
            
        except ImportError as e:
            st.session_state.error_logger.log_error("Module Import", e)
            st.error("❌ Required modules not available - check error details above")
        except Exception as e:
            st.session_state.error_logger.log_error("Scan Execution", e)
            st.error("❌ Failed to execute scan - check error details above")

def run_enhanced_stock_scan(stocks_to_scan, analysis_date=None, days_back=59, rolling_window=20):
    """Execute the enhanced stock scanning process from local files"""
    
    error_logger = st.session_state.error_logger
    
    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher
        from core.technical_analysis import add_enhanced_columns, get_mpi_trend_info
        
        # Log scan start
        is_historical = analysis_date is not None
        is_single_stock = len(stocks_to_scan) == 1
        
        scope_text = f"single stock ({stocks_to_scan[0]})" if is_single_stock else f"{len(stocks_to_scan)} stocks"
        date_text = f"historical analysis (as of {analysis_date.strftime('%d/%m/%Y')})" if is_historical else "current data analysis"
        
        logger.info(f"Starting scan: {scope_text} with {date_text}")
        st.info(f"🔄 Scanning {scope_text} with {date_text}... Loading from local files...")
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize data fetcher
        status_text.text("🔧 Initializing data fetcher...")
        fetcher = DataFetcher(days_back=days_back)
        progress_bar.progress(0.1)
        
        # Download stock data from local files
        status_text.text("📥 Loading stock data from local files...")
        stock_data = fetcher.download_stock_data(stocks_to_scan, target_date=analysis_date)
        set_global_data_fetcher(fetcher)
        progress_bar.progress(0.3)
        
        if not stock_data:
            error_logger.log_error("Data Validation", Exception("No stock data loaded from local files"))
            st.error("❌ Failed to load stock data from local files. Check error log for details.")
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
                
                # Get analysis row (most recent)
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
            'date': f'Historical ({analysis_date.strftime("%d/%m/%Y")})' if is_historical else 'Current',
            'stock_count': len(results_df)
        }
        
        # Complete scan
        progress_bar.progress(1.0)
        status_text.text("✅ Pure MPI Expansion scan completed!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Show success message
        success_message = f"🎉 Pure MPI Expansion Scan completed! Analyzed {len(results_df)} stocks successfully from local files"
        if processing_errors:
            success_message += f" ({len(processing_errors)} errors - check log for details)"
        
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

# File: pages/scanner_higher_hl.py
# Part 3 of 4

def _create_result_dict(analysis_row: pd.Series, actual_date, ticker: str, fetcher) -> dict:
    """Create result dictionary from analysis row - UPDATED with Higher_H and HL_Pattern"""
    # Get company name
    try:
        company_name = fetcher.get_company_name(ticker)
    except Exception:
        company_name = ticker.replace('.SG', '')
    
    # Determine price formatting
    close_price = float(analysis_row['Close'])
    price_decimals = 3 if close_price < 1.00 else 2
    
    def safe_round(value, decimals):
        try:
            return round(float(value), decimals) if not pd.isna(value) else 0
        except:
            return 0
    
    def safe_int(value, default=0):
        """Safely convert value to int, handling NaN and other edge cases"""
        try:
            if pd.isna(value):
                return default
            return int(value)
        except:
            return default
    
    # Get MPI trend info
    mpi_trend = str(analysis_row.get('MPI_Trend', 'Unknown'))
    mpi_velocity = float(analysis_row.get('MPI_Velocity', 0.0)) if not pd.isna(analysis_row.get('MPI_Velocity', 0.0)) else 0.0
    
    try:
        from core.technical_analysis import get_mpi_trend_info
        mpi_trend_info = get_mpi_trend_info(mpi_trend, mpi_velocity)
    except:
        mpi_trend_info = {'emoji': '❓', 'description': 'Unknown'}
    
    # Get Higher_H and Higher_HL values and create HL_Pattern
    higher_h = safe_int(analysis_row.get('Higher_H', 0))
    higher_hl = safe_int(analysis_row.get('Higher_HL', 0))
    
    # Create HL_Pattern display value
    if higher_hl == 1:
        hl_pattern = "HHL"
    elif higher_h == 1:
        hl_pattern = "HH"
    else:
        hl_pattern = "-"
    
    # Create result dictionary
    result = {
        'Ticker': ticker,
        'Name': company_name,
        'Analysis_Date': actual_date.strftime('%d/%m/%Y') if hasattr(actual_date, 'strftime') else str(actual_date),
        'Close': round(close_price, price_decimals),
        'High': safe_round(analysis_row['High'], price_decimals),
        'Low': safe_round(analysis_row['Low'], price_decimals),
        'IBS': round(float(analysis_row['IBS']), 3) if not pd.isna(analysis_row['IBS']) else 0,
        'Valid_CRT': safe_int(analysis_row.get('Valid_CRT', 0)),
        
        # Add Higher_H and HL_Pattern
        'Higher_H': higher_h,
        'Higher_HL': higher_hl,
        'HL_Pattern': hl_pattern,
        
        'CRT_Velocity': round(float(analysis_row.get('CRT_Qualifying_Velocity', 0)), 4) if not pd.isna(analysis_row.get('CRT_Qualifying_Velocity', 0)) else 0,
        'VW_Range_Velocity': round(float(analysis_row.get('VW_Range_Velocity', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Velocity', 0)) else 0,
        'Weekly_Open': safe_round(analysis_row.get('Weekly_Open', 0), price_decimals),
        'CRT_High': safe_round(analysis_row.get('CRT_High', 0), price_decimals),
        'CRT_Low': safe_round(analysis_row.get('CRT_Low', 0), price_decimals),
        'VW_Range_Percentile': round(float(analysis_row.get('VW_Range_Percentile', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Percentile', 0)) else 0,
        'Rel_Range_Signal': safe_int(analysis_row.get('Rel_Range_Signal', 0)),
        
        # Pure MPI Expansion columns
        'MPI': round(float(analysis_row.get('MPI', 0.5)), 4) if not pd.isna(analysis_row.get('MPI', 0.5)) else 0.5,
        'MPI_Velocity': round(mpi_velocity, 4),
        'MPI_Trend': mpi_trend,
        'MPI_Trend_Emoji': mpi_trend_info.get('emoji', '❓'),
        'MPI_Description': mpi_trend_info.get('description', 'Unknown'),
        'MPI_Visual': format_mpi_visual(analysis_row.get('MPI', 0.5)),
        
        # Relative Volume columns
        'Relative_Volume': round(float(analysis_row.get('Relative_Volume', 100.0)), 1) if not pd.isna(analysis_row.get('Relative_Volume', 100.0)) else 100.0,
        'High_Rel_Volume_150': safe_int(analysis_row.get('High_Rel_Volume_150', 0)),
        'High_Rel_Volume_200': safe_int(analysis_row.get('High_Rel_Volume_200', 0)),
        
        # Market Regime columns - KEEP IN DATA but won't display
        'Market_Regime': str(analysis_row.get('Market_Regime', 'Unknown')),
        'Regime_Probability': round(float(analysis_row.get('Regime_Probability', 0.5)), 3) if not pd.isna(analysis_row.get('Regime_Probability', 0.5)) else 0.5,
        
        'Price_Decimals': price_decimals
    }
    
    return result

def display_scan_summary(results_df: pd.DataFrame):
    """Display scan summary with Pure MPI Expansion statistics - UPDATED with Higher_H"""
    st.subheader("📊 Scan Summary with Pure MPI Expansion Analysis")
    
    total_stocks = len(results_df)
    
    # Calculate Higher_H and Higher_HL counts
    higher_h_count = len(results_df[results_df['Higher_H'] == 1])
    higher_hl_count = len(results_df[results_df['Higher_HL'] == 1])
    hh_only_count = len(results_df[(results_df['Higher_H'] == 1) & (results_df['Higher_HL'] == 0)])
    
    valid_crt_count = len(results_df[results_df['Valid_CRT'] == 1])
    
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
        st.metric("Higher H", higher_h_count, delta=f"{higher_h_count/total_stocks*100:.1f}%")
    with col3:
        st.metric("HHL", higher_hl_count, delta=f"Both H & L")
    with col4:
        st.metric("Valid CRT", valid_crt_count)
    with col5:
        st.metric("🚀 Strong Exp", strong_expansion, delta=f"≥5% velocity")
    with col6:
        st.metric("📈 Expanding", strong_expansion + expanding, delta=f">0% velocity")
    
    # Analysis date info with H/L pattern breakdown
    if len(results_df) > 0:
        analysis_dates = results_df['Analysis_Date'].unique()
        if len(analysis_dates) == 1:
            st.info(f"📅 Analysis date: **{analysis_dates[0]}** | Higher H: **{higher_h_count}** (HHL: **{higher_hl_count}**, HH only: **{hh_only_count}**) | Avg MPI: **{avg_mpi:.1%}** | Avg Velocity: **{avg_velocity:+.1%}**")
        else:
            st.info(f"📅 Analysis dates: **{', '.join(analysis_dates)}** | Higher H: **{higher_h_count}** | Avg MPI: **{avg_mpi:.1%}** | Avg Velocity: **{avg_velocity:+.1%}**")

def show_base_pattern_filter(results_df: pd.DataFrame) -> pd.DataFrame:
    """Show base pattern filter - SIMPLIFIED to only Valid CRT and All Stocks"""
    st.subheader("🎯 Pattern Analysis")
    
    # Only 2 base filter options
    base_filter_options = {
        "Valid CRT Only": "All stocks with Valid CRT (Monday range expansion)",
        "All Stocks": "Complete scan results without pattern filtering"
    }
    
    selected_base_filter = st.radio(
        "Select base pattern filter:",
        list(base_filter_options.keys()),
        help="Choose base filtering criteria",
        key="base_filter_radio"
    )
    
    # Store in session state for filter logic
    st.session_state.base_filter_type = selected_base_filter
    
    # Apply base filter
    if selected_base_filter == "Valid CRT Only":
        base_stocks = results_df[results_df['Valid_CRT'] == 1].copy()
        st.info(f"Showing {len(base_stocks)} stocks with Valid CRT (Monday range expansion)")
    
    else:  # All Stocks
        base_stocks = results_df.copy()
        st.info(f"Showing all {len(base_stocks)} scanned stocks")
    
    return base_stocks

def display_filtered_results(filtered_stocks: pd.DataFrame, selected_base_filter: str):
    """
    Display the filtered results table and export options
    UPDATED: Removed Market_Regime and Regime_Probability, moved HL_Pattern to prominent position
    """
    st.subheader(f"📋 Pure MPI Expansion Results ({len(filtered_stocks)} stocks)")
    
    if len(filtered_stocks) == 0:
        st.warning("No stocks match the current filter criteria")
        return
    
    # UPDATED: Define display columns - HL_Pattern moved after Close, regime columns removed
    display_cols = ['Analysis_Date', 'Ticker', 'Name', 'Close',
                    'CRT_High', 'CRT_Low', 'HL_Pattern',
                    'VW_Range_Velocity', 'IBS', 'Relative_Volume',
                    'MPI_Trend_Emoji', 'MPI_Visual']
    
    # UPDATED: Create column configuration without regime columns
    base_column_config = {
        'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
        'Name': st.column_config.TextColumn('Company Name', width='medium'),
        'HL_Pattern': st.column_config.TextColumn('H/L', width='small', help='HHL=Both H&L, HH=Higher H only, -=Neither'),
        'VW_Range_Velocity': st.column_config.NumberColumn('Range Vel', format='%+.4f', help='Daily range expansion velocity'),
        'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
        'Relative_Volume': st.column_config.NumberColumn('Rel Vol', format='%.1f%%', help='Relative Volume vs 14-day average'),
        'MPI_Trend_Emoji': st.column_config.TextColumn('📊', width='small', help='MPI Expansion Trend'),
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
    
    # Remove .SG suffix for TradingView format
    tv_tickers = [f"SGX:{ticker.replace('.SG', '')}" for ticker in filtered_stocks['Ticker'].tolist()]
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
    """
    Show the full results table in an expander
    UPDATED: Removed regime columns, reordered with HL_Pattern prominent
    """
    with st.expander("📋 Full Pure MPI Expansion Analysis Results", expanded=False):
        try:
            # UPDATED: Include HL_Pattern early, remove regime columns
            full_results_cols = [
                'Analysis_Date', 'Ticker', 'Name', 'Close',
                'CRT_High', 'CRT_Low', 'HL_Pattern',
                'VW_Range_Velocity', 'IBS', 'Valid_CRT', 
                'MPI_Trend_Emoji', 'MPI_Trend', 'MPI', 'MPI_Velocity', 'MPI_Visual'
            ]
            
            base_full_results_config = {
                'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Name': st.column_config.TextColumn('Company Name', width='medium'),
                'HL_Pattern': st.column_config.TextColumn('H/L', width='small', help='HHL=Both, HH=High only, -=Neither'),
                'VW_Range_Velocity': st.column_config.NumberColumn('Range Vel', format='%+.4f'),
                'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
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

# File: pages/scanner_higher_hl.py
# Part 4 of 4

def show():
    """Main scanner page display for Higher H/L patterns with Pure MPI Expansion filtering"""
    
    st.title("📈 CRT Higher H/L Scanner")
    st.markdown("Enhanced with **Pure MPI Expansion System** - Focus on momentum velocity")
    st.markdown("**Data Source: Local File System** (./data/Historical_Data + ./data/EOD_Data)")
    
    # Show update check/prompt
    st.subheader("📥 Data Management")
    
    if 'update_check_done' not in st.session_state:
        st.session_state.update_check_done = False
    
    if not st.session_state.update_check_done:
        update_result = show_update_prompt()
        if update_result:
            st.session_state.update_check_done = True
            st.rerun()
    else:
        # Show manual update button
        if st.button("🔄 Check for Updates Again"):
            st.session_state.update_check_done = False
            st.rerun()
    
    st.markdown("---")
    
    # Clear error log button
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🗑️ Clear Error Log"):
            st.session_state.error_logger = ErrorLogger()
            st.success("Error log cleared!")
            st.rerun()
    with col2:
        if st.button("💾 Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    # Display any existing errors
    st.session_state.error_logger.display_errors_in_streamlit()
    
    # Show scanning configuration
    scan_scope, selected_stock, scan_date_type, historical_date = show_scanning_configuration()
    
    # Show advanced settings
    days_back, rolling_window = show_advanced_settings()
    
    # Scan execution
    st.subheader("🚀 Execute Scan")
    
    execute_scan_button(scan_scope, selected_stock, scan_date_type, historical_date, 
                       days_back, rolling_window)
    
    # Display last scan info if available
    if 'last_scan_time' in st.session_state:
        st.info(f"📊 Last scan completed: {st.session_state.last_scan_time}")
        
        if 'last_scan_config' in st.session_state:
            config = st.session_state.last_scan_config
            st.caption(f"Scope: {config['scope']} | Date: {config['date']} | Stocks: {config['stock_count']}")
    
    # Display results if available
    if 'scan_results' in st.session_state:
        display_scan_results(st.session_state.scan_results)

if __name__ == "__main__":
    show()

