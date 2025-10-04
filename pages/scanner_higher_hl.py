# File: pages/scanner_higher_hl.py
# Part 1 of 3
"""
CRT Higher H/L Scanner - Local File System
Enhanced with Pure MPI Expansion Filtering
NOW USING VW_RANGE_VELOCITY FOR DAILY MOMENTUM FILTERING
Automatically checks for Historical_Data updates on startup
ENHANCED: Force update capability to re-process latest EOD file
UPDATED: Simplified base filter, enhanced H/L filter with Higher_H support
UPDATED: Removed Market Regime filter, moved H/L filter to first position
UPDATED: Custom filters now support both Minimum and Maximum filtering
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

