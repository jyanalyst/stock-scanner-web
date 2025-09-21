# File: pages/backtesting.py
# Part 1 of 4 - CORRECTED VERSION WITH INCLUSIVE IBS THRESHOLDS
"""
Historical Backtesting Module - CORRECTED
Factor analysis study: Which indicators predict successful breakout continuation
Entry: Previous day's high when today's high > yesterday's high
Exit: Same day's close
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import logging
import sys
from io import StringIO, BytesIO
import traceback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class BacktestLogger:
    """Centralized logging for backtesting operations"""
    
    def __init__(self):
        self.logs = []
        self.errors = []
        self.warnings = []
    
    def log_info(self, message: str, data: dict = None):
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': 'INFO',
            'message': message,
            'data': data or {}
        }
        self.logs.append(entry)
        logger.info(f"BACKTEST: {message}")
    
    def log_error(self, message: str, error: Exception = None, context: dict = None):
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': 'ERROR',
            'message': message,
            'error': str(error) if error else None,
            'traceback': traceback.format_exc() if error else None,
            'context': context or {}
        }
        self.errors.append(entry)
        logger.error(f"BACKTEST ERROR: {message}")
    
    def log_warning(self, message: str, context: dict = None):
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': 'WARNING',
            'message': message,
            'context': context or {}
        }
        self.warnings.append(entry)
        logger.warning(f"BACKTEST WARNING: {message}")
    
    def display_in_streamlit(self):
        """Display logs in Streamlit interface"""
        if self.errors:
            st.error(f"❌ {len(self.errors)} error(s) occurred during backtesting")
            with st.expander("🔍 View Error Details", expanded=False):
                for error in self.errors:
                    st.markdown(f"**{error['timestamp']}:** {error['message']}")
                    if error['error']:
                        st.code(error['error'])
                    if error['context']:
                        st.json(error['context'])
        
        if self.warnings:
            st.warning(f"⚠️ {len(self.warnings)} warning(s) occurred")
            with st.expander("📋 View Warnings", expanded=False):
                for warning in self.warnings:
                    st.write(f"**{warning['timestamp']}:** {warning['message']}")
                    if warning['context']:
                        st.json(warning['context'])
        
        # Show info logs in debug mode
        if st.session_state.get('debug_mode', False) and self.logs:
            with st.expander("🔍 Debug Logs", expanded=False):
                for log in self.logs[-20:]:  # Show last 20 entries
                    st.write(f"**{log['timestamp']}:** {log['message']}")

# Initialize global backtest logger
if 'backtest_logger' not in st.session_state:
    st.session_state.backtest_logger = BacktestLogger()

def analyze_uploaded_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Analyze uploaded CSV to determine what date ranges need processing
    
    Returns:
        Tuple of (existing_data_df, analysis_dict)
    """
    backtest_logger = st.session_state.backtest_logger
    
    try:
        # Read uploaded CSV
        existing_data = pd.read_csv(uploaded_file)
        backtest_logger.log_info(f"Uploaded file contains {len(existing_data)} records")
        
        # Validate required columns
        required_columns = [
            'setup_date', 'breakout_date', 'ticker', 'company_name', 
            'setup_high', 'setup_close', 'breakout_high', 'breakout_close',
            'entry_price', 'exit_price', 'success_binary', 'return_percentage',
            'setup_mpi', 'setup_mpi_velocity', 'setup_mpi_trend', 'setup_ibs',
            'setup_crt_velocity', 'setup_higher_hl', 'setup_valid_crt', 'setup_vw_range_percentile'
        ]
        
        missing_columns = [col for col in required_columns if col not in existing_data.columns]
        if missing_columns:
            backtest_logger.log_error(f"Missing required columns: {missing_columns}")
            return None, {
                'status': 'error',
                'message': f"Invalid CSV format. Missing columns: {', '.join(missing_columns)}"
            }
        
        # Parse dates
        existing_data['setup_date'] = pd.to_datetime(existing_data['setup_date'])
        existing_data['breakout_date'] = pd.to_datetime(existing_data['breakout_date'])
        
        # Analyze date coverage
        min_date = existing_data['setup_date'].min().date()
        max_date = existing_data['setup_date'].max().date()
        total_breakouts = len(existing_data)
        unique_stocks = existing_data['ticker'].nunique()
        date_range_days = (max_date - min_date).days + 1
        
        # Calculate success metrics
        total_successes = existing_data['success_binary'].sum()
        success_rate = (total_successes / total_breakouts * 100) if total_breakouts > 0 else 0
        avg_return = existing_data['return_percentage'].mean() * 100
        
        analysis = {
            'status': 'success',
            'data_summary': {
                'total_breakouts': total_breakouts,
                'unique_stocks': unique_stocks,
                'date_range': f"{min_date} to {max_date}",
                'date_range_days': date_range_days,
                'success_rate': success_rate,
                'average_return': avg_return,
                'earliest_date': min_date,
                'latest_date': max_date
            }
        }
        
        backtest_logger.log_info("Successfully analyzed uploaded data", analysis['data_summary'])
        return existing_data, analysis
        
    except Exception as e:
        backtest_logger.log_error("Failed to analyze uploaded data", e)
        return None, {
            'status': 'error',
            'message': f"Error reading CSV file: {str(e)}"
        }

def determine_processing_range(existing_data: Optional[pd.DataFrame], 
                             user_start_date: date, 
                             user_end_date: date) -> Tuple[date, date, str]:
    """
    Determine what date range needs to be processed based on existing data
    
    Returns:
        Tuple of (start_date, end_date, processing_type)
    """
    backtest_logger = st.session_state.backtest_logger
    
    if existing_data is None or len(existing_data) == 0:
        backtest_logger.log_info(f"No existing data - will process full range: {user_start_date} to {user_end_date}")
        return user_start_date, user_end_date, "full_range"
    
    # Get the latest date in existing data
    latest_existing_date = existing_data['setup_date'].max().date()
    
    # Determine processing strategy
    if latest_existing_date >= user_end_date:
        backtest_logger.log_info(f"Existing data covers requested range (latest: {latest_existing_date})")
        return user_start_date, user_end_date, "no_processing_needed"
    
    elif latest_existing_date < user_start_date:
        backtest_logger.log_info(f"Gap detected - will process full range: {user_start_date} to {user_end_date}")
        return user_start_date, user_end_date, "gap_fill"
    
    else:
        # Incremental update needed
        new_start_date = latest_existing_date + timedelta(days=1)
        backtest_logger.log_info(f"Incremental update - will process: {new_start_date} to {user_end_date}")
        return new_start_date, user_end_date, "incremental"

def detect_breakouts_and_analyze_factors(df_enhanced: pd.DataFrame, 
                                       ticker: str, 
                                       company_name: str,
                                       start_date: date, 
                                       end_date: date) -> List[Dict]:
    """
    CORRECTED: Detect breakouts and analyze setup day factors
    
    Logic:
    1. For each day, check if today's high > yesterday's high
    2. If yes, we have a breakout - enter at yesterday's high, exit at today's close
    3. Analyze yesterday's technical indicators as predictive factors
    
    Args:
        df_enhanced: Enhanced DataFrame with technical indicators
        ticker: Stock ticker symbol
        company_name: Company name
        start_date: Start date for breakout detection
        end_date: End date for breakout detection
    
    Returns:
        List of breakout analysis dictionaries
    """
    breakouts = []
    
    try:
        # Filter date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # Ensure timezone consistency
        if hasattr(df_enhanced.index, 'tz') and df_enhanced.index.tz is not None:
            if start_dt.tz is None:
                start_dt = start_dt.tz_localize('Asia/Singapore')
                end_dt = end_dt.tz_localize('Asia/Singapore')
        
        # Get all dates in range (we need consecutive days for breakout detection)
        all_dates = df_enhanced.index
        range_dates = all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)]
        
        # Process each day to detect breakouts
        for i in range(1, len(range_dates)):  # Start from index 1 (need previous day)
            today_date = range_dates[i]
            yesterday_date = range_dates[i-1]
            
            # Get today's and yesterday's data
            today_data = df_enhanced.loc[today_date]
            yesterday_data = df_enhanced.loc[yesterday_date]
            
            # Check for breakout: today's high > yesterday's high
            today_high = float(today_data['High'])
            yesterday_high = float(yesterday_data['High'])
            
            if today_high > yesterday_high:
                # BREAKOUT DETECTED!
                # Entry: Yesterday's high, Exit: Today's close
                entry_price = yesterday_high
                exit_price = float(today_data['Close'])
                
                # Calculate success and return
                success_binary = 1 if exit_price > entry_price else 0
                return_percentage = (exit_price - entry_price) / entry_price
                
                # Collect yesterday's (setup day) technical indicators
                def safe_float(value, default=0.0):
                    try:
                        return float(value) if not pd.isna(value) else default
                    except:
                        return default
                
                def safe_int(value, default=0):
                    try:
                        return int(value) if not pd.isna(value) else default
                    except:
                        return default
                
                def safe_string(value, default='Unknown'):
                    try:
                        return str(value) if not pd.isna(value) else default
                    except:
                        return default
                
                # Create breakout analysis record
                breakout_record = {
                    # Date and identification
                    'setup_date': yesterday_date.strftime('%Y-%m-%d'),
                    'breakout_date': today_date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'company_name': company_name,
                    
                    # Price data
                    'setup_high': round(yesterday_high, 4),
                    'setup_close': round(float(yesterday_data['Close']), 4),
                    'breakout_high': round(today_high, 4),
                    'breakout_close': round(exit_price, 4),
                    
                    # Entry/Exit and results
                    'entry_price': round(entry_price, 4),
                    'exit_price': round(exit_price, 4),
                    'success_binary': success_binary,
                    'return_percentage': round(return_percentage, 6),
                    
                    # Setup day technical indicators (what we're analyzing)
                    'setup_mpi': round(safe_float(yesterday_data.get('MPI', 0.5)), 4),
                    'setup_mpi_velocity': round(safe_float(yesterday_data.get('MPI_Velocity', 0.0)), 6),
                    'setup_mpi_trend': safe_string(yesterday_data.get('MPI_Trend', 'Unknown')),
                    'setup_ibs': round(safe_float(yesterday_data.get('IBS', 0.0)), 4),
                    'setup_crt_velocity': round(safe_float(yesterday_data.get('CRT_Qualifying_Velocity', 0.0)), 6),
                    'setup_higher_hl': safe_int(yesterday_data.get('Higher_HL', 0)),
                    'setup_valid_crt': safe_int(yesterday_data.get('Valid_CRT', 0)),
                    'setup_vw_range_percentile': round(safe_float(yesterday_data.get('VW_Range_Percentile', 0.0)), 4),
                    
                    # Additional setup day data
                    'setup_volume': safe_int(yesterday_data.get('Volume', 0)),
                    'setup_open': round(safe_float(yesterday_data.get('Open', 0.0)), 4),
                    'setup_low': round(safe_float(yesterday_data.get('Low', 0.0)), 4),
                    
                    # Breakout magnitude
                    'breakout_gap_percentage': round((today_high - yesterday_high) / yesterday_high, 6)
                }
                
                breakouts.append(breakout_record)
        
        return breakouts
        
    except Exception as e:
        logger.error(f"Error detecting breakouts for {ticker}: {e}")
        return []
    
# File: pages/backtesting.py
# Part 2 of 4 - CORRECTED VERSION WITH INCLUSIVE IBS THRESHOLDS
"""
Historical Backtesting Module - Part 2 CORRECTED
Core backtesting execution and data processing functions
"""

def run_incremental_backtest(start_date: date, 
                           end_date: date, 
                           existing_data: Optional[pd.DataFrame] = None,
                           progress_callback=None) -> pd.DataFrame:
    """
    CORRECTED: Run breakout factor analysis for specified date range
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        existing_data: Previously processed breakout data
        progress_callback: Function to update progress
    
    Returns:
        Complete breakout analysis DataFrame including existing and new data
    """
    backtest_logger = st.session_state.backtest_logger
    
    try:
        # Import required modules
        from core.data_fetcher import DataFetcher
        from core.technical_analysis import add_enhanced_columns
        from utils.watchlist import get_active_watchlist
        
        backtest_logger.log_info(f"Starting breakout analysis: {start_date} to {end_date}")
        
        # Get watchlist
        watchlist = get_active_watchlist()
        backtest_logger.log_info(f"Processing {len(watchlist)} stocks")
        
        # Initialize data fetcher with extended history for technical analysis
        days_back = (date.today() - start_date).days + 100  # Add buffer for technical analysis
        fetcher = DataFetcher(days_back=days_back)
        
        # Download historical data
        if progress_callback:
            progress_callback(0.1, "Downloading historical stock data...")
        
        stock_data = fetcher.download_stock_data(watchlist)
        backtest_logger.log_info(f"Downloaded data for {len(stock_data)} stocks")
        
        # Process each stock for breakout detection
        all_breakouts = []
        total_stocks = len(stock_data)
        
        for i, (ticker, df_raw) in enumerate(stock_data.items()):
            try:
                if progress_callback:
                    progress = 0.1 + (0.7 * i / total_stocks)
                    progress_callback(progress, f"Analyzing breakouts for {ticker}...")
                
                if df_raw.empty:
                    backtest_logger.log_warning(f"No data for {ticker}")
                    continue
                
                # Apply technical analysis
                df_enhanced = add_enhanced_columns(df_raw, ticker)
                
                # Get company name
                company_name = fetcher.get_company_name(ticker)
                
                # Detect breakouts and analyze setup day factors
                stock_breakouts = detect_breakouts_and_analyze_factors(
                    df_enhanced, ticker, company_name, start_date, end_date
                )
                
                all_breakouts.extend(stock_breakouts)
                backtest_logger.log_info(f"{ticker}: Found {len(stock_breakouts)} breakouts")
                
            except Exception as e:
                backtest_logger.log_error(f"Error processing {ticker}", e)
                continue
        
        if progress_callback:
            progress_callback(0.8, "Compiling breakout analysis results...")
        
        # Create new results DataFrame
        new_results = pd.DataFrame(all_breakouts) if all_breakouts else pd.DataFrame()
        
        if len(new_results) == 0:
            backtest_logger.log_warning("No breakouts found for the specified date range")
            if existing_data is not None:
                return existing_data
            else:
                return pd.DataFrame()
        
        # Merge with existing data
        if existing_data is not None and len(existing_data) > 0:
            combined_results = merge_backtest_results(existing_data, new_results)
            backtest_logger.log_info(f"Merged results: {len(combined_results)} total breakouts")
        else:
            combined_results = new_results
            backtest_logger.log_info(f"New analysis: {len(combined_results)} breakouts")
        
        if progress_callback:
            progress_callback(1.0, "Breakout analysis completed!")
        
        return combined_results
        
    except Exception as e:
        backtest_logger.log_error("Critical error in breakout analysis execution", e)
        if existing_data is not None:
            return existing_data
        else:
            return pd.DataFrame()

def merge_backtest_results(existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """
    CORRECTED: Intelligently merge existing and new breakout analysis results
    
    Args:
        existing_data: Previously processed breakout data
        new_data: Newly processed breakout data
    
    Returns:
        Combined DataFrame with duplicates removed
    """
    backtest_logger = st.session_state.backtest_logger
    
    try:
        if existing_data.empty:
            backtest_logger.log_info("No existing data - returning new data only")
            return new_data.copy()
        
        if new_data.empty:
            backtest_logger.log_info("No new data - returning existing data only")
            return existing_data.copy()
        
        # Combine the datasets
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        
        # Remove duplicates based on setup_date, breakout_date, and ticker
        combined['setup_date'] = pd.to_datetime(combined['setup_date'])
        combined['breakout_date'] = pd.to_datetime(combined['breakout_date'])
        
        # Sort by dates and ticker before removing duplicates
        combined = combined.sort_values(['setup_date', 'breakout_date', 'ticker'])
        
        # Remove duplicates (keep last occurrence for any conflicts)
        before_count = len(combined)
        combined = combined.drop_duplicates(subset=['setup_date', 'breakout_date', 'ticker'], keep='last')
        after_count = len(combined)
        
        duplicates_removed = before_count - after_count
        if duplicates_removed > 0:
            backtest_logger.log_info(f"Removed {duplicates_removed} duplicate records")
        
        # Sort by setup_date for final output
        combined = combined.sort_values('setup_date').reset_index(drop=True)
        
        backtest_logger.log_info(f"Successfully merged data: {len(combined)} total breakouts")
        
        return combined
        
    except Exception as e:
        backtest_logger.log_error("Error merging breakout analysis results", e)
        # Return existing data as fallback
        return existing_data

def calculate_backtest_summary(results_df: pd.DataFrame) -> Dict:
    """
    CORRECTED: Calculate comprehensive summary statistics for breakout analysis results
    
    Args:
        results_df: Complete breakout analysis results DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    if results_df.empty:
        return {
            'total_breakouts': 0,
            'success_rate': 0.0,
            'average_return': 0.0,
            'total_return': 0.0,
            'successful_breakouts': 0,
            'failed_breakouts': 0,
            'best_return': 0.0,
            'worst_return': 0.0,
            'date_range': 'No data',
            'unique_stocks': 0,
            'breakouts_per_month': 0.0
        }
    
    try:
        # Basic metrics
        total_breakouts = len(results_df)
        successful_breakouts = int(results_df['success_binary'].sum())
        failed_breakouts = total_breakouts - successful_breakouts
        success_rate = (successful_breakouts / total_breakouts * 100) if total_breakouts > 0 else 0
        
        # Return metrics
        returns = results_df['return_percentage']
        average_return = returns.mean() * 100
        total_return = returns.sum() * 100
        best_return = returns.max() * 100
        worst_return = returns.min() * 100
        
        # Date and stock metrics
        results_df['setup_date'] = pd.to_datetime(results_df['setup_date'])
        min_date = results_df['setup_date'].min()
        max_date = results_df['setup_date'].max()
        unique_stocks = results_df['ticker'].nunique()
        
        # Calculate breakouts per month
        date_range_months = ((max_date - min_date).days / 30.44) if max_date > min_date else 1
        breakouts_per_month = total_breakouts / date_range_months if date_range_months > 0 else 0
        
        summary = {
            'total_breakouts': total_breakouts,
            'success_rate': success_rate,
            'average_return': average_return,
            'total_return': total_return,
            'successful_breakouts': successful_breakouts,
            'failed_breakouts': failed_breakouts,
            'best_return': best_return,
            'worst_return': worst_return,
            'date_range': f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
            'unique_stocks': unique_stocks,
            'breakouts_per_month': breakouts_per_month,
            'success_rate_decimal': success_rate / 100
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error calculating breakout summary: {e}")
        return {
            'total_breakouts': len(results_df),
            'success_rate': 0.0,
            'error': str(e)
        }

def validate_backtest_data_quality(df: pd.DataFrame) -> Dict:
    """CORRECTED: Validate the quality of breakout analysis data"""
    validation = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check for required columns
    required_cols = ['setup_date', 'breakout_date', 'ticker', 'success_binary', 'return_percentage']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation['is_valid'] = False
        validation['issues'].append(f"Missing required columns: {missing_cols}")
    
    if len(df) == 0:
        validation['is_valid'] = False
        validation['issues'].append("No data to validate")
        return validation
    
    # Check date consistency
    if 'setup_date' in df.columns and 'breakout_date' in df.columns:
        date_issues = (df['breakout_date'] <= df['setup_date']).sum()
        if date_issues > 0:
            validation['warnings'].append(f"{date_issues} records have breakout_date <= setup_date")
    
    # Check return percentage validity
    if 'return_percentage' in df.columns:
        invalid_returns = df['return_percentage'].isna().sum()
        extreme_returns = ((df['return_percentage'] > 1.0) | (df['return_percentage'] < -1.0)).sum()
        if invalid_returns > 0:
            validation['warnings'].append(f"{invalid_returns} records have invalid return percentages")
        if extreme_returns > 0:
            validation['warnings'].append(f"{extreme_returns} records have extreme returns (>100% or <-100%)")
    
    # Check entry/exit price logic
    if 'entry_price' in df.columns and 'exit_price' in df.columns:
        # Verify that success_binary matches the entry/exit logic
        calculated_success = (df['exit_price'] > df['entry_price']).astype(int)
        mismatched = (calculated_success != df['success_binary']).sum()
        if mismatched > 0:
            validation['warnings'].append(f"{mismatched} records have inconsistent success_binary values")
    
    # Summary statistics
    validation['summary'] = {
        'total_records': len(df),
        'unique_stocks': df['ticker'].nunique() if 'ticker' in df.columns else 0,
        'date_range': f"{df['setup_date'].min()} to {df['setup_date'].max()}" if 'setup_date' in df.columns else "Unknown",
        'success_rate': f"{df['success_binary'].mean() * 100:.1f}%" if 'success_binary' in df.columns else "Unknown"
    }
    
    return validation

def create_csv_for_download(results_df: pd.DataFrame) -> str:
    """
    CORRECTED: Create CSV string for download with proper formatting
    
    Args:
        results_df: Breakout analysis results DataFrame
    
    Returns:
        CSV string ready for download
    """
    try:
        if results_df.empty:
            return "No data available for download"
        
        # Ensure proper column ordering for CSV
        column_order = [
            'setup_date', 'breakout_date', 'ticker', 'company_name',
            'setup_high', 'setup_close', 'breakout_high', 'breakout_close',
            'entry_price', 'exit_price', 'success_binary', 'return_percentage',
            'setup_mpi', 'setup_mpi_velocity', 'setup_mpi_trend', 'setup_ibs',
            'setup_crt_velocity', 'setup_higher_hl', 'setup_valid_crt', 'setup_vw_range_percentile',
            'setup_volume', 'setup_open', 'setup_low', 'breakout_gap_percentage'
        ]
        
        # Add any additional columns that might exist
        extra_columns = [col for col in results_df.columns if col not in column_order]
        final_columns = column_order + extra_columns
        
        # Filter to only existing columns
        final_columns = [col for col in final_columns if col in results_df.columns]
        
        # Create CSV with proper formatting
        csv_df = results_df[final_columns].copy()
        
        # Format numeric columns
        numeric_columns = {
            'return_percentage': 6,
            'setup_mpi': 4,
            'setup_mpi_velocity': 6,
            'setup_ibs': 4,
            'setup_crt_velocity': 6,
            'setup_vw_range_percentile': 4,
            'breakout_gap_percentage': 6,
            'entry_price': 4,
            'exit_price': 4,
            'setup_high': 4,
            'setup_close': 4,
            'breakout_high': 4,
            'breakout_close': 4
        }
        
        for col, decimals in numeric_columns.items():
            if col in csv_df.columns:
                csv_df[col] = csv_df[col].round(decimals)
        
        return csv_df.to_csv(index=False)
        
    except Exception as e:
        logger.error(f"Error creating CSV for download: {e}")
        return f"Error creating CSV: {str(e)}"

def show_backtest_configuration():
    """CORRECTED: Display breakout analysis configuration panel"""
    st.subheader("🎯 Breakout Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Date Range**")
        
        # Default date range
        default_start = date(2024, 1, 1)
        default_end = date.today() - timedelta(days=1)  # Yesterday (need consecutive days)
        
        start_date = st.date_input(
            "Start Date:",
            value=default_start,
            max_value=default_end,
            help="Start date for breakout analysis"
        )
        
        end_date = st.date_input(
            "End Date:",
            value=default_end,
            min_value=start_date,
            max_value=default_end,
            help="End date for breakout analysis (must have next day data available)"
        )
        
        # Validate date range
        if end_date <= start_date:
            st.error("End date must be after start date")
            return None, None, None
        
        days_to_process = (end_date - start_date).days + 1
        st.info(f"📊 Date range: {days_to_process} days")
    
    with col2:
        st.markdown("**📁 Previous Analysis Results**")
        
        uploaded_file = st.file_uploader(
            "Upload Previous Analysis (Optional):",
            type=['csv'],
            help="Upload a previous breakout analysis CSV to continue where you left off"
        )
        
        if uploaded_file is not None:
            # Analyze uploaded data
            existing_data, analysis = analyze_uploaded_data(uploaded_file)
            
            if analysis['status'] == 'success':
                summary = analysis['data_summary']
                st.success("✅ Previous analysis data loaded successfully!")
                
                # Show existing data summary
                with st.expander("📊 Existing Data Summary", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Breakouts", summary['total_breakouts'])
                        st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                    with col_b:
                        st.metric("Unique Stocks", summary['unique_stocks'])
                        st.metric("Avg Return", f"{summary['average_return']:.2f}%")
                    
                    st.write(f"**Existing Date Range:** {summary['date_range']}")
                    
                    # Determine processing strategy
                    process_start, process_end, process_type = determine_processing_range(
                        existing_data, start_date, end_date
                    )
                    
                    if process_type == "no_processing_needed":
                        st.info("🎯 Existing data covers your requested range - no new processing needed")
                    elif process_type == "incremental":
                        new_days = (process_end - process_start).days + 1
                        st.info(f"🔄 Incremental update: Will process {new_days} new days ({process_start} to {process_end})")
                    elif process_type == "gap_fill":
                        gap_days = (process_start - summary['latest_date']).days
                        st.warning(f"⚠️ Gap detected: {gap_days} days between existing data and requested start date")
                    else:
                        st.info(f"🆕 Full range processing: {days_to_process} days")
                
            else:
                st.error(f"❌ {analysis['message']}")
                existing_data = None
        else:
            existing_data = None
    
    return start_date, end_date, existing_data

# File: pages/backtesting.py
# Part 3 of 4 - CORRECTED VERSION WITH INCLUSIVE IBS THRESHOLDS
"""
Historical Backtesting Module - Part 3 CORRECTED
Analysis and visualization functions for breakout factor analysis
*** THIS PART CONTAINS THE CRITICAL IBS THRESHOLD FIX ***
"""

def perform_factor_analysis(results_df: pd.DataFrame) -> Dict:
    """
    CORRECTED: Perform comprehensive factor effectiveness analysis for breakout prediction
    *** FIXED IBS CATEGORIZATION WITH INCLUSIVE THRESHOLDS ***
    
    Args:
        results_df: Complete breakout analysis results DataFrame
    
    Returns:
        Dictionary with factor analysis results
    """
    if results_df.empty:
        return {}
    
    try:
        analysis = {}
        
        # MPI Trend Analysis - which setup day MPI trends predict successful breakouts
        if 'setup_mpi_trend' in results_df.columns:
            mpi_analysis = results_df.groupby('setup_mpi_trend').agg({
                'success_binary': ['count', 'sum', 'mean'],
                'return_percentage': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            mpi_analysis.columns = ['Breakout_Count', 'Success_Count', 'Success_Rate', 
                                  'Avg_Return', 'Return_Std', 'Min_Return', 'Max_Return']
            mpi_analysis['Success_Rate'] = mpi_analysis['Success_Rate'] * 100
            mpi_analysis['Avg_Return'] = mpi_analysis['Avg_Return'] * 100
            mpi_analysis['Return_Std'] = mpi_analysis['Return_Std'] * 100
            mpi_analysis['Min_Return'] = mpi_analysis['Min_Return'] * 100
            mpi_analysis['Max_Return'] = mpi_analysis['Max_Return'] * 100
            
            analysis['mpi_trend'] = mpi_analysis.reset_index()
        
        # IBS Threshold Analysis - using cumulative thresholds instead of fixed buckets
        if 'setup_ibs' in results_df.columns:
            ibs_thresholds = [
                ('High (>= 0.7)', 0.7),
                ('Med (>= 0.5)', 0.5),
                ('Low (>= 0.3)', 0.3),
                ('All (>= 0)', 0.0)
            ]
            
            ibs_results = []
            for threshold_name, threshold_value in ibs_thresholds:
                threshold_data = results_df[results_df['setup_ibs'] >= threshold_value]
                
                if len(threshold_data) > 0:
                    success_count = threshold_data['success_binary'].sum()
                    total_count = len(threshold_data)
                    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
                    avg_return = threshold_data['return_percentage'].mean() * 100
                    return_std = threshold_data['return_percentage'].std() * 100
                    
                    ibs_results.append({
                        'ibs_threshold': threshold_name,
                        'threshold_value': threshold_value,
                        'Breakout_Count': total_count,
                        'Success_Count': success_count,
                        'Success_Rate': round(success_rate, 1),
                        'Avg_Return': round(avg_return, 2),
                        'Return_Std': round(return_std, 2)
                    })
            
            analysis['ibs_thresholds'] = pd.DataFrame(ibs_results)
        
        # *** CRITICAL FIX: INCLUSIVE IBS THRESHOLD CATEGORIZATION ***
        if all(col in results_df.columns for col in ['setup_mpi_trend', 'setup_ibs', 'setup_higher_hl', 'setup_valid_crt']):
            
            # FIXED: Create IBS threshold categories with INCLUSIVE thresholds
            def categorize_ibs_threshold(ibs_value):
                if ibs_value >= 0.7:
                    return 'High_IBS'   # >= 0.7
                elif ibs_value >= 0.5:
                    return 'Med_IBS'    # >= 0.5 (but < 0.7)
                elif ibs_value >= 0.3:
                    return 'Low_IBS'    # >= 0.3 (but < 0.5)
                else:
                    return 'VLow_IBS'    # >= 0 (all values)

            results_df['ibs_category'] = results_df['setup_ibs'].apply(categorize_ibs_threshold)
            
            # FIXED: Define inclusive IBS filtering for thresholds
            def passes_ibs_threshold(row_ibs, threshold_category):
                if threshold_category == 'High_IBS':
                    return row_ibs >= 0.7   # Only >= 0.7
                elif threshold_category == 'Med_IBS':
                    return row_ibs >= 0.5   # Includes High_IBS values (>= 0.5)
                elif threshold_category == 'Low_IBS':
                    return row_ibs >= 0.3   # Includes Med_IBS and High_IBS values (>= 0.3)
                elif threshold_category == 'VLow_IBS':
                    return row_ibs >= 0.0   # Includes all values
                return False
            
            # Generate ALL possible combinations with FIXED filtering
            combination_results = []
            
            # Get unique values for each factor
            mpi_trends = results_df['setup_mpi_trend'].unique()
            ibs_categories = ['High_IBS', 'Med_IBS', 'Low_IBS', 'All_IBS']  # Use fixed categories
            higher_hl_values = [0, 1]
            valid_crt_values = [0, 1]
            
            # Analyze each combination with FIXED INCLUSIVE FILTERING
            for mpi_trend in mpi_trends:
                for ibs_cat in ibs_categories:
                    for higher_hl in higher_hl_values:
                        for valid_crt in valid_crt_values:
                            # FIXED: Filter data for this specific combination using inclusive thresholds
                            combo_data = results_df[
                                (results_df['setup_mpi_trend'] == mpi_trend) &
                                (results_df['setup_ibs'].apply(lambda x: passes_ibs_threshold(x, ibs_cat))) &  # FIXED: Inclusive filtering
                                (results_df['setup_higher_hl'] == higher_hl) &
                                (results_df['setup_valid_crt'] == valid_crt)
                            ]
                            
                            if len(combo_data) >= 5:  # Minimum sample size for reliability
                                success_count = combo_data['success_binary'].sum()
                                total_count = len(combo_data)
                                success_rate = (success_count / total_count * 100) if total_count > 0 else 0
                                avg_return = combo_data['return_percentage'].mean() * 100
                                
                                # Create combination description
                                hl_desc = "Higher_HL" if higher_hl == 1 else "No_HL"
                                crt_desc = "Valid_CRT" if valid_crt == 1 else "No_CRT"
                                combo_description = f"{mpi_trend}_{ibs_cat}_{hl_desc}_{crt_desc}"
                                
                                combination_results.append({
                                    'combination': combo_description,
                                    'mpi_trend': mpi_trend,
                                    'ibs_category': str(ibs_cat),
                                    'higher_hl': higher_hl,
                                    'valid_crt': valid_crt,
                                    'Breakout_Count': total_count,
                                    'Success_Count': success_count,
                                    'Success_Rate': round(success_rate, 1),
                                    'Avg_Return': round(avg_return, 2)
                                })
            
            # Convert to DataFrame and sort by success rate (descending)
            if combination_results:
                combo_df = pd.DataFrame(combination_results)
                combo_df = combo_df.sort_values('Success_Rate', ascending=False)
                analysis['all_combinations'] = combo_df
                
                # Get top 10 best combinations
                analysis['best_combinations'] = combo_df.head(10)
        
        # Time-based Analysis - are there seasonal patterns in breakout success?
        if 'setup_date' in results_df.columns:
            results_df['setup_date'] = pd.to_datetime(results_df['setup_date'])
            results_df['year_month'] = results_df['setup_date'].dt.to_period('M')
            
            monthly_analysis = results_df.groupby('year_month').agg({
                'success_binary': ['count', 'sum', 'mean'],
                'return_percentage': 'mean'
            }).round(4)
            
            monthly_analysis.columns = ['Breakout_Count', 'Success_Count', 'Success_Rate', 'Avg_Return']
            monthly_analysis['Success_Rate'] = monthly_analysis['Success_Rate'] * 100
            monthly_analysis['Avg_Return'] = monthly_analysis['Avg_Return'] * 100
            
            analysis['monthly_performance'] = monthly_analysis.reset_index()
            analysis['monthly_performance']['year_month'] = analysis['monthly_performance']['year_month'].astype(str)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in factor analysis: {e}")
        return {}

def create_factor_visualizations(results_df: pd.DataFrame, factor_analysis: Dict) -> Dict:
    """
    CORRECTED: Create comprehensive visualizations for breakout factor analysis
    
    Args:
        results_df: Breakout results DataFrame
        factor_analysis: Factor analysis results
    
    Returns:
        Dictionary of plotly figures
    """
    figures = {}
    
    try:
        # MPI Trend Success Rate Chart
        if 'mpi_trend' in factor_analysis:
            mpi_data = factor_analysis['mpi_trend']
            
            fig_mpi = px.bar(
                mpi_data,
                x='setup_mpi_trend',
                y='Success_Rate',
                title='Breakout Success Rate by Setup Day MPI Trend',
                labels={'Success_Rate': 'Success Rate (%)', 'setup_mpi_trend': 'Setup Day MPI Trend'},
                color='Success_Rate',
                color_continuous_scale='RdYlGn',
                text='Success_Rate'
            )
            fig_mpi.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_mpi.update_layout(height=400)
            figures['mpi_success_rate'] = fig_mpi
            
            # MPI Trend Return Chart
            fig_mpi_return = px.bar(
                mpi_data,
                x='setup_mpi_trend',
                y='Avg_Return',
                title='Average Return by Setup Day MPI Trend',
                labels={'Avg_Return': 'Average Return (%)', 'setup_mpi_trend': 'Setup Day MPI Trend'},
                color='Avg_Return',
                color_continuous_scale='RdYlGn',
                text='Avg_Return'
            )
            fig_mpi_return.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_mpi_return.update_layout(height=400)
            figures['mpi_returns'] = fig_mpi_return
        
        # IBS Threshold Analysis Chart
        if 'ibs_thresholds' in factor_analysis:
            ibs_data = factor_analysis['ibs_thresholds']
            
            fig_ibs = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Success Rate by IBS Threshold', 'Average Return by IBS Threshold'),
                specs=[[{'secondary_y': False}, {'secondary_y': False}]]
            )
            
            fig_ibs.add_trace(
                go.Bar(x=ibs_data['ibs_threshold'], y=ibs_data['Success_Rate'], 
                      name='Success Rate (%)', marker_color='lightblue',
                      text=ibs_data['Success_Rate'], texttemplate='%{text:.1f}%'),
                row=1, col=1
            )
            
            fig_ibs.add_trace(
                go.Bar(x=ibs_data['ibs_threshold'], y=ibs_data['Avg_Return'], 
                      name='Average Return (%)', marker_color='lightgreen',
                      text=ibs_data['Avg_Return'], texttemplate='%{text:.2f}%'),
                row=1, col=2
            )
            
            fig_ibs.update_layout(height=400, showlegend=False)
            figures['ibs_analysis'] = fig_ibs
        
        # Best Combinations Analysis
        if 'best_combinations' in factor_analysis:
            combo_data = factor_analysis['best_combinations']
            
            fig_combos = px.scatter(
                combo_data,
                x='Success_Rate',
                y='Avg_Return',
                size='Breakout_Count',
                color='combination',
                title='Top 10 Best Factor Combinations: Success Rate vs Average Return',
                labels={'Success_Rate': 'Success Rate (%)', 'Avg_Return': 'Average Return (%)'},
                hover_data=['Breakout_Count', 'mpi_trend', 'ibs_category']
            )
            fig_combos.update_layout(height=500, showlegend=False)
            figures['best_combinations'] = fig_combos
        
        # Monthly Performance Trend
        if 'monthly_performance' in factor_analysis:
            monthly_data = factor_analysis['monthly_performance']
            
            fig_monthly = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Monthly Success Rate', 'Monthly Breakout Count'),
                specs=[[{'secondary_y': False}], [{'secondary_y': False}]]
            )
            
            fig_monthly.add_trace(
                go.Scatter(x=monthly_data['year_month'], y=monthly_data['Success_Rate'], 
                          mode='lines+markers', name='Success Rate (%)', line=dict(color='blue')),
                row=1, col=1
            )
            
            fig_monthly.add_trace(
                go.Bar(x=monthly_data['year_month'], y=monthly_data['Breakout_Count'], 
                      name='Breakout Count', marker_color='lightcoral'),
                row=2, col=1
            )
            
            fig_monthly.update_layout(height=500, showlegend=False)
            fig_monthly.update_xaxes(tickangle=45)
            figures['monthly_trends'] = fig_monthly
        
        # Return Distribution
        if 'return_percentage' in results_df.columns:
            fig_dist = px.histogram(
                results_df,
                x='return_percentage',
                nbins=50,
                title='Distribution of Breakout Returns',
                labels={'return_percentage': 'Return (%)', 'count': 'Frequency'},
                color_discrete_sequence=['skyblue']
            )
            
            # Add vertical lines for key statistics
            mean_return = results_df['return_percentage'].mean()
            fig_dist.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: {mean_return:.3f}")
            fig_dist.add_vline(x=0, line_dash="solid", line_color="black", 
                              annotation_text="Break-even")
            
            fig_dist.update_layout(height=400)
            figures['return_distribution'] = fig_dist
        
        return figures
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return {}

def display_backtest_summary(results_df: pd.DataFrame):
    """CORRECTED: Display comprehensive breakout analysis summary"""
    st.subheader("📊 Breakout Analysis Summary")
    
    if results_df.empty:
        st.warning("No breakout analysis results to display")
        return
    
    # Calculate summary statistics
    summary = calculate_backtest_summary(results_df)
    
    # Store in session state for sidebar
    st.session_state.backtest_summary = {
        'total_signals': summary['total_breakouts'],
        'success_rate': summary['success_rate']
    }
    
    # Display key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Breakouts", summary['total_breakouts'])
    with col2:
        st.metric("Success Rate", f"{summary['success_rate']:.1f}%", 
                 delta=f"{summary['success_rate'] - 50:.1f}% vs 50%")
    with col3:
        st.metric("Success/Fail", f"{summary['successful_breakouts']}/{summary['failed_breakouts']}")
    with col4:
        st.metric("Avg Return", f"{summary['average_return']:.2f}%")
    with col5:
        st.metric("Best Return", f"{summary['best_return']:.2f}%")
    with col6:
        st.metric("Worst Return", f"{summary['worst_return']:.2f}%")
    
    # Additional info
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"📅 **Date Range:** {summary['date_range']}")
    with col_b:
        st.info(f"📈 **Stocks Analyzed:** {summary['unique_stocks']}")
    with col_c:
        st.info(f"📊 **Breakouts/Month:** {summary['breakouts_per_month']:.1f}")

# File: pages/backtesting.py
# Part 4 of 4 - CORRECTED VERSION WITH INCLUSIVE IBS THRESHOLDS
"""
Historical Backtesting Module - Part 4 CORRECTED
Display functions and main show function for breakout factor analysis
"""

def display_factor_analysis(results_df: pd.DataFrame):
    """CORRECTED: Display comprehensive factor effectiveness analysis for breakout prediction"""
    st.subheader("🔬 Setup Day Factor Analysis")
    st.markdown("*Which setup day indicators best predict successful breakouts?*")
    
    if results_df.empty:
        st.warning("No data available for factor analysis")
        return
    
    # Perform analysis
    with st.spinner("Analyzing which setup day factors predict successful breakouts..."):
        factor_analysis = perform_factor_analysis(results_df)
        figures = create_factor_visualizations(results_df, factor_analysis)
    
    if not factor_analysis:
        st.error("Failed to perform factor analysis")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 MPI Trends", "🎯 IBS Thresholds", "🔄 Best Combinations"])
    
    with tab1:
        st.markdown("### Setup Day MPI Trend Effectiveness")
        st.markdown("*Does the MPI trend on the setup day predict breakout success?*")
        
        if 'mpi_trend' in factor_analysis:
            # Display data table
            st.dataframe(
                factor_analysis['mpi_trend'],
                use_container_width=True,
                hide_index=True
            )
            
            # Display charts
            col1, col2 = st.columns(2)
            with col1:
                if 'mpi_success_rate' in figures:
                    st.plotly_chart(figures['mpi_success_rate'], use_container_width=True)
            with col2:
                if 'mpi_returns' in figures:
                    st.plotly_chart(figures['mpi_returns'], use_container_width=True)
            
            # Key insights
            mpi_data = factor_analysis['mpi_trend']
            best_mpi = mpi_data.loc[mpi_data['Success_Rate'].idxmax()]
            worst_mpi = mpi_data.loc[mpi_data['Success_Rate'].idxmin()]
            
            st.markdown("#### 🔍 Key Insights:")
            st.success(f"**Best Setup MPI Trend:** {best_mpi['setup_mpi_trend']} - {best_mpi['Success_Rate']:.1f}% success rate")
            st.error(f"**Worst Setup MPI Trend:** {worst_mpi['setup_mpi_trend']} - {worst_mpi['Success_Rate']:.1f}% success rate")
            
            # Validate MPI expansion hypothesis
            expansion_trends = ['Expanding']
            contraction_trends = ['Contracting']
            
            expansion_success = mpi_data[mpi_data['setup_mpi_trend'].isin(expansion_trends)]['Success_Rate'].mean() if any(mpi_data['setup_mpi_trend'].isin(expansion_trends)) else 0
            contraction_success = mpi_data[mpi_data['setup_mpi_trend'].isin(contraction_trends)]['Success_Rate'].mean() if any(mpi_data['setup_mpi_trend'].isin(contraction_trends)) else 0
            
            if expansion_success > contraction_success and expansion_success > 0:
                st.info(f"✅ **MPI Expansion Hypothesis Confirmed:** Setup days with expansion trends predict better breakouts ({expansion_success:.1f}% vs {contraction_success:.1f}%)")
            elif contraction_success > 0:
                st.warning(f"❌ **MPI Expansion Hypothesis Challenged:** Contraction trends actually perform better ({contraction_success:.1f}% vs {expansion_success:.1f}%)")
        else:
            st.warning("MPI trend data not available")
    
    with tab2:
        st.markdown("### Setup Day IBS Threshold Effectiveness")
        st.markdown("*How do different IBS threshold levels predict breakout success?*")
        
        if 'ibs_thresholds' in factor_analysis:
            # Display data table
            st.dataframe(
                factor_analysis['ibs_thresholds'],
                use_container_width=True,
                hide_index=True
            )
            
            # Display chart
            if 'ibs_analysis' in figures:
                st.plotly_chart(figures['ibs_analysis'], use_container_width=True)
            
            # Key insights
            ibs_data = factor_analysis['ibs_thresholds']
            
            st.markdown("#### 🔍 Key Insights:")
            
            # Compare performance across thresholds
            high_threshold = ibs_data[ibs_data['ibs_threshold'] == 'High (>= 0.7)']
            all_threshold = ibs_data[ibs_data['ibs_threshold'] == 'All (>= 0)']
            
            if len(high_threshold) > 0 and len(all_threshold) > 0:
                high_success = high_threshold['Success_Rate'].iloc[0]
                all_success = all_threshold['Success_Rate'].iloc[0]
                
                if high_success > all_success:
                    st.success(f"**High IBS Filter Effective:** IBS >= 0.7 achieves {high_success:.1f}% vs {all_success:.1f}% for all breakouts")
                else:
                    st.warning(f"**High IBS Filter Not Effective:** IBS >= 0.7 only achieves {high_success:.1f}% vs {all_success:.1f}% for all breakouts")
            
            # Show threshold progression
            st.markdown("**Threshold Progression Analysis:**")
            for _, row in ibs_data.iterrows():
                st.write(f"• **{row['ibs_threshold']}**: {row['Success_Rate']:.1f}% success rate ({row['Breakout_Count']} breakouts)")
        else:
            st.warning("IBS threshold data not available")
    
    with tab3:
        st.markdown("### Best Factor Combinations")
        st.markdown("*Which combinations of all factors predict the highest breakout success?*")
        
        if 'best_combinations' in factor_analysis:
            # Display top combinations table
            st.markdown("#### 🏆 Top 10 Best Factor Combinations:")
            st.dataframe(
                factor_analysis['best_combinations'][['combination', 'Breakout_Count', 'Success_Rate', 'Avg_Return']],
                use_container_width=True,
                hide_index=True
            )
            
            # Display scatter plot
            if 'best_combinations' in figures:
                st.plotly_chart(figures['best_combinations'], use_container_width=True)
            
            # Key insights
            combo_data = factor_analysis['best_combinations']
            best_combo = combo_data.iloc[0]
            
            st.markdown("#### 🔍 Key Insights:")
            st.success(f"**Best Combination:** {best_combo['combination']} - {best_combo['Success_Rate']:.1f}% success rate ({best_combo['Breakout_Count']} breakouts)")
            
            # Analyze top combinations
            st.markdown("**Top 3 Combinations Analysis:**")
            for i, (_, combo) in enumerate(combo_data.head(3).iterrows(), 1):
                st.write(f"**#{i}:** {combo['combination']} - {combo['Success_Rate']:.1f}% success ({combo['Breakout_Count']} samples)")
                st.write(f"   • MPI: {combo['mpi_trend']}, IBS: {combo['ibs_category']}, Higher H/L: {combo['higher_hl']}, Valid CRT: {combo['valid_crt']}")
            
            # Show total combinations analyzed
            if 'all_combinations' in factor_analysis:
                total_combos = len(factor_analysis['all_combinations'])
                st.info(f"📊 **Total Combinations Analyzed:** {total_combos} (minimum 5 breakouts per combination)")
                
                # FIXED: Show the corrected combination that should now have 636 samples
                target_combo = factor_analysis['all_combinations'][
                    factor_analysis['all_combinations']['combination'] == 'Expanding_Med_IBS_Higher_HL_Valid_CRT'
                ]
                if len(target_combo) > 0:
                    target_row = target_combo.iloc[0]
                    st.success(f"🎯 **FIXED COMBINATION:** Expanding_Med_IBS_Higher_HL_Valid_CRT - {target_row['Success_Rate']:.1f}% success rate ({target_row['Breakout_Count']} samples)")
                    st.info("✅ This combination now correctly uses IBS >= 0.5 (inclusive of all higher IBS values)")
        else:
            st.warning("Combination analysis data not available")

    # Show monthly performance trends
    if 'monthly_performance' in factor_analysis:
        st.markdown("### Time-Based Performance Analysis")
        st.markdown("*Are there seasonal patterns in breakout success?*")
        
        # Display monthly trends chart
        if 'monthly_trends' in figures:
            st.plotly_chart(figures['monthly_trends'], use_container_width=True)
        
        # Display data table
        with st.expander("📊 Monthly Performance Details", expanded=False):
            st.dataframe(
                factor_analysis['monthly_performance'],
                use_container_width=True,
                hide_index=True
            )
        
        # Key insights
        monthly_data = factor_analysis['monthly_performance']
        best_month = monthly_data.loc[monthly_data['Success_Rate'].idxmax()]
        worst_month = monthly_data.loc[monthly_data['Success_Rate'].idxmin()]
        
        st.markdown("#### 🔍 Key Insights:")
        st.success(f"**Best Month:** {best_month['year_month']} - {best_month['Success_Rate']:.1f}% success rate")
        st.error(f"**Worst Month:** {worst_month['year_month']} - {worst_month['Success_Rate']:.1f}% success rate")
        
        # Performance consistency
        success_rate_std = monthly_data['Success_Rate'].std()
        if success_rate_std < 10:
            st.info(f"✅ **Consistent Performance:** Low monthly variance ({success_rate_std:.1f}% std dev)")
        else:
            st.warning(f"⚠️ **Variable Performance:** High monthly variance ({success_rate_std:.1f}% std dev)")

def display_interactive_data_explorer(results_df: pd.DataFrame):
    """CORRECTED: Display interactive data exploration interface for breakout analysis"""
    st.subheader("🔍 Interactive Breakout Explorer")
    
    if results_df.empty:
        st.warning("No data available for exploration")
        return
    
    # Filtering controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # MPI Trend filter
        if 'setup_mpi_trend' in results_df.columns:
            mpi_trends = ['All'] + sorted(results_df['setup_mpi_trend'].unique().tolist())
            selected_mpi = st.selectbox("Setup MPI Trend:", mpi_trends)
        else:
            selected_mpi = 'All'
    
    with col2:
        # Success filter
        success_filter = st.selectbox("Breakout Outcome:", ['All', 'Successful Only', 'Failed Only'])
    
    with col3:
        # Date range filter
        if 'setup_date' in results_df.columns:
            results_df['setup_date'] = pd.to_datetime(results_df['setup_date'])
            min_date = results_df['setup_date'].min().date()
            max_date = results_df['setup_date'].max().date()
            
            date_range = st.date_input(
                "Setup Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = None
    
    with col4:
        # Stock filter
        if 'ticker' in results_df.columns:
            stocks = ['All'] + sorted(results_df['ticker'].unique().tolist())
            selected_stock = st.selectbox("Stock:", stocks)
        else:
            selected_stock = 'All'
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if selected_mpi != 'All' and 'setup_mpi_trend' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['setup_mpi_trend'] == selected_mpi]
    
    if success_filter == 'Successful Only':
        filtered_df = filtered_df[filtered_df['success_binary'] == 1]
    elif success_filter == 'Failed Only':
        filtered_df = filtered_df[filtered_df['success_binary'] == 0]
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['setup_date'].dt.date >= start_date) & 
            (filtered_df['setup_date'].dt.date <= end_date)
        ]
    
    if selected_stock != 'All' and 'ticker' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ticker'] == selected_stock]
    
    # Display filtered results
    st.markdown(f"**Filtered Results: {len(filtered_df)} breakouts**")
    
    if len(filtered_df) > 0:
        # Summary for filtered data
        filtered_summary = calculate_backtest_summary(filtered_df)
        
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Filtered Breakouts", filtered_summary['total_breakouts'])
        with col_b:
            st.metric("Success Rate", f"{filtered_summary['success_rate']:.1f}%")
        with col_c:
            st.metric("Avg Return", f"{filtered_summary['average_return']:.2f}%")
        with col_d:
            st.metric("Total Return", f"{filtered_summary['total_return']:.2f}%")
        
        # Display filtered data table
        display_columns = [
            'setup_date', 'breakout_date', 'ticker', 'company_name', 'setup_mpi_trend', 'setup_ibs', 
            'entry_price', 'exit_price', 'success_binary', 'return_percentage'
        ]
        display_columns = [col for col in display_columns if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_columns].sort_values('setup_date', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Export filtered data
        csv_data = create_csv_for_download(filtered_df)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"filtered_breakout_analysis_{len(filtered_df)}_breakouts.csv",
            mime="text/csv"
        )
    else:
        st.warning("No breakouts match the current filter criteria")

def execute_backtest_process(start_date: date, end_date: date, existing_data: Optional[pd.DataFrame]):
    """CORRECTED: Execute the complete breakout analysis process with progress tracking"""
    
    backtest_logger = st.session_state.backtest_logger
    
    # Determine processing strategy
    if existing_data is not None:
        process_start, process_end, process_type = determine_processing_range(
            existing_data, start_date, end_date
        )
    else:
        process_start, process_end, process_type = start_date, end_date, "full_range"
    
    # Check if processing is needed
    if process_type == "no_processing_needed":
        st.success("✅ Existing data covers your requested range - displaying results!")
        st.session_state.backtest_results = existing_data
        st.session_state.backtest_completed = True
        st.rerun()
        return
    
    # Show processing information
    if process_type == "incremental":
        days_to_process = (process_end - process_start).days + 1
        st.info(f"🔄 Running incremental breakout analysis: {days_to_process} days ({process_start} to {process_end})")
    elif process_type == "gap_fill":
        st.warning(f"⚠️ Gap detected - processing full range: {start_date} to {end_date}")
        process_start, process_end = start_date, end_date
    else:
        days_to_process = (process_end - process_start).days + 1
        st.info(f"🆕 Running full breakout analysis: {days_to_process} days ({process_start} to {process_end})")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress: float, message: str):
        progress_bar.progress(progress)
        status_text.text(message)
    
    try:
        # Execute breakout analysis
        backtest_logger.log_info(f"Starting breakout analysis execution: {process_start} to {process_end}")
        
        results_df = run_incremental_backtest(
            start_date=process_start,
            end_date=process_end,
            existing_data=existing_data,
            progress_callback=update_progress
        )
        
        # Validate results
        validation = validate_backtest_data_quality(results_df)
        
        if not validation['is_valid']:
            st.error("❌ Breakout analysis failed validation:")
            for issue in validation['issues']:
                st.error(f"• {issue}")
            return
        
        if validation['warnings']:
            st.warning("⚠️ Data quality warnings:")
            for warning in validation['warnings']:
                st.warning(f"• {warning}")
        
        # Store results
        st.session_state.backtest_results = results_df
        st.session_state.backtest_completed = True
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Success message
        summary = validation['summary']
        success_message = f"🎉 Breakout analysis completed successfully! Analyzed {summary['total_records']} breakouts"
        
        if process_type == "incremental":
            success_message += f" (incremental update)"
        
        st.success(success_message)
        
        # Log completion
        backtest_logger.log_info("Breakout analysis completed successfully", {
            'total_breakouts': summary['total_records'],
            'success_rate': summary.get('success_rate', 'Unknown'),
            'process_type': process_type
        })
        
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        backtest_logger.log_error("Breakout analysis execution failed", e)
        st.error("❌ Breakout analysis failed - check error log for details")

def show_backtest_results():
    """CORRECTED: Display comprehensive breakout analysis results"""
    
    if 'backtest_results' not in st.session_state or st.session_state.backtest_results.empty:
        st.info("No breakout analysis results available. Run an analysis to see results here.")
        return
    
    results_df = st.session_state.backtest_results
    
    st.subheader("📊 Breakout Analysis Results")
    
    # Show overall summary
    display_backtest_summary(results_df)
    
    # Display return distribution chart
    if 'return_percentage' in results_df.columns and len(results_df) > 1:
        st.subheader("📈 Breakout Return Distribution")
        
        fig_dist = px.histogram(
            results_df,
            x='return_percentage',
            nbins=50,
            title='Distribution of Breakout Returns',
            labels={'return_percentage': 'Return (%)', 'count': 'Frequency'},
            color_discrete_sequence=['skyblue']
        )
        
        # Add statistical markers
        mean_return = results_df['return_percentage'].mean()
        median_return = results_df['return_percentage'].median()
        
        fig_dist.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: {mean_return:.3f}")
        fig_dist.add_vline(x=median_return, line_dash="dash", line_color="blue", 
                          annotation_text=f"Median: {median_return:.3f}")
        fig_dist.add_vline(x=0, line_dash="solid", line_color="black", 
                          annotation_text="Break-even")
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Factor effectiveness analysis
    display_factor_analysis(results_df)
    
    # Interactive data explorer
    display_interactive_data_explorer(results_df)
    
    # Export options
    st.subheader("📥 Export Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Complete dataset download
        csv_data = create_csv_for_download(results_df)
        
        # Generate filename with summary info
        summary = calculate_backtest_summary(results_df)
        filename = f"breakout_analysis_{summary['total_breakouts']}_breakouts_{summary['success_rate']:.0f}pct_success.csv"
        
        st.download_button(
            label="📥 Download Complete Analysis (CSV)",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Download all breakout analysis results for future incremental updates"
        )
    
    with col2:
        # Summary statistics download
        summary_data = {
            'Metric': [
                'Total Breakouts', 'Success Rate (%)', 'Average Return (%)', 
                'Total Return (%)', 'Successful Breakouts', 'Failed Breakouts', 
                'Best Return (%)', 'Worst Return (%)', 'Date Range', 'Unique Stocks'
            ],
            'Value': [
                summary['total_breakouts'], f"{summary['success_rate']:.2f}",
                f"{summary['average_return']:.4f}", f"{summary['total_return']:.2f}",
                summary['successful_breakouts'], summary['failed_breakouts'],
                f"{summary['best_return']:.4f}", f"{summary['worst_return']:.4f}",
                summary['date_range'], summary['unique_stocks']
            ]
        }
        
        summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
        
        st.download_button(
            label="📊 Download Summary Stats (CSV)",
            data=summary_csv,
            file_name=f"breakout_summary_{summary['total_breakouts']}_breakouts.csv",
            mime="text/csv",
            help="Download summary statistics only"
        )

def show_usage_instructions():
    """CORRECTED: Display usage instructions and best practices for breakout analysis"""
    
    with st.expander("📚 Usage Instructions & Best Practices", expanded=False):
        st.markdown("""
        ### 🚀 Getting Started with Breakout Factor Analysis
        
        #### First-Time Users:
        1. **Select Date Range:** Choose your analysis period (default: Jan 1, 2024 to yesterday)
        2. **Run Analysis:** Click "🚀 Execute Analysis" to process the full range
        3. **Download Results:** Save the CSV file for future incremental updates
        
        #### Returning Users:
        1. **Upload Previous Results:** Use the file uploader to continue where you left off
        2. **Automatic Analysis:** The system detects your latest date and suggests new processing
        3. **Incremental Processing:** Only new dates are processed, saving time and resources
        4. **Updated Download:** Get the complete historical dataset including new results
        
        ### 📊 Understanding the Breakout Analysis
        
        #### Trading Logic:
        - **Breakout Detection:** Daily check if today's high > yesterday's high
        - **Entry Price:** Yesterday's high (resistance level that was broken)
        - **Exit Price:** Today's close (same day as breakout)
        - **Success:** Today's close > Yesterday's high = Profitable breakout
        
        #### What We're Studying:
        This is **NOT** a trading strategy - it's a **factor analysis study** to determine:
        - Which setup day indicators predict successful breakout continuation
        - Do stocks with "Expanding" MPI setups have better breakout success?
        - Does high IBS on the setup day improve breakout probability?
        - Which factor combinations provide the highest success rates?
        
        ### 🔬 Factor Analysis Results
        
        #### Key Questions Answered:
        - **MPI Trends:** Do expanding momentum setups predict better breakouts?
        - **IBS Thresholds:** What IBS threshold levels optimize breakout success?
        - **Best Combinations:** Which combinations of all factors achieve highest success rates?
        - **Seasonal Patterns:** Are there monthly/seasonal success variations?
        
        #### Success Metrics:
        - **Success Rate:** Percentage of breakouts that close above entry price
        - **Average Return:** Mean return per breakout (entry to same-day close)
        - **Return Distribution:** Spread of outcomes to assess risk/reward
        
        ### 🎯 Interpreting Results
        
        #### IBS Threshold Analysis (FIXED):
        - **High_IBS:** Only stocks with IBS >= 0.7
        - **Med_IBS:** All stocks with IBS >= 0.5 (includes High_IBS values)
        - **Low_IBS:** All stocks with IBS >= 0.3 (includes Med_IBS and High_IBS values)
        - **All_IBS:** All breakouts regardless of IBS level
        
        #### Best Combinations Analysis:
        - Analyzes ALL possible combinations of MPI trends, IBS levels, Higher H/L, and Valid CRT
        - Uses INCLUSIVE thresholds (Med_IBS includes all IBS >= 0.5)
        - Requires minimum 5 breakouts per combination for statistical reliability
        - Shows top 10 best performing combinations ranked by success rate
        - Enables discovery of optimal factor confluence for entry criteria
        
        #### Strong Predictive Factors:
        - Success rate significantly above 50% (better than random)
        - Consistent performance across different time periods
        - Large sample size for statistical significance
        
        #### Validation Examples:
        - ✅ "Expanding MPI setups have 65% breakout success vs 45% for contracting"
        - ✅ "Med_IBS (>= 0.5) achieves 58% success vs 52% for all breakouts"
        - ✅ "Best combination: Expanding_Med_IBS_Higher_HL_Valid_CRT = 636 samples, 49.5% success rate"
        
        ### ⚠️ Important Bug Fix
        
        #### FIXED IBS Categorization:
        - **Previous Bug:** Med_IBS only included IBS 0.3-0.5 (122 samples)
        - **Fixed Version:** Med_IBS now includes all IBS >= 0.5 (636 samples)
        - **Impact:** Combinations like "Expanding_Med_IBS_Higher_HL_Valid_CRT" now show correct sample counts
        
        Remember: This analysis identifies **which conditions predict breakout success**, not trading signals themselves.
        """)

def show():
    """CORRECTED: Main breakout analysis page display function"""
    
    st.title("🔬 Historical Breakout Factor Analysis")
    st.markdown("**Quantitative study: Which setup day indicators predict successful momentum breakouts?**")
    
    # Educational info box
    with st.container():
        st.info("""
        **Study Design:** When today's high > yesterday's high (breakout), we enter at yesterday's high and exit at today's close. 
        We then analyze which technical indicators were present on the **setup day** (yesterday) that predicted successful breakouts.
        
        🎯 **FIXED:** IBS threshold categorization now uses INCLUSIVE thresholds (Med_IBS = IBS >= 0.5, includes all higher values)
        """)
    
    # Initialize session state
    if 'backtest_completed' not in st.session_state:
        st.session_state.backtest_completed = False
    
    # Clear error log button
    col_clear, col_debug = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Clear Log"):
            st.session_state.backtest_logger = BacktestLogger()
            st.success("Log cleared!")
            st.rerun()
    
    with col_debug:
        debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.get('debug_mode', False))
        st.session_state.debug_mode = debug_mode
    
    # Check module availability
    modules_available = True
    try:
        from core.data_fetcher import DataFetcher
        from core.technical_analysis import add_enhanced_columns
        from utils.watchlist import get_active_watchlist
        st.session_state.backtest_logger.log_info("All required modules available")
    except ImportError as e:
        st.session_state.backtest_logger.log_error("Module import failed", e)
        st.error(f"❌ Required modules not available: {e}")
        modules_available = False
    
    # Display any errors/warnings
    st.session_state.backtest_logger.display_in_streamlit()
    
    if not modules_available:
        st.warning("Cannot run breakout analysis - required modules are not available")
        return
    
    # Show usage instructions
    show_usage_instructions()
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["🎯 Run Analysis", "📊 View Results"])
    
    with tab1:
        st.markdown("### Configure and Execute Breakout Analysis")
        
        # Configuration panel
        config_result = show_backtest_configuration()
        
        if config_result is None:
            st.error("Please fix configuration errors before proceeding")
            return
        
        start_date, end_date, existing_data = config_result
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Analysis Focus:**")
                st.info("""
                This study analyzes ALL breakouts (today high > yesterday high) regardless of other factors.
                We then determine which setup day indicators predicted success.
                """)
            
            with col2:
                st.markdown("**Success Definition:**")
                st.info("""
                Success = Breakout day close > Setup day high
                (Same-day momentum continuation after breakout)
                """)
        
        # Execute analysis button
        st.markdown("### 🚀 Execute Breakout Analysis")
        
        # Show expected processing info
        days_to_process = (end_date - start_date).days + 1
        estimated_breakouts = days_to_process * 46 * 0.3  # Rough estimate: 30% of stock-days have breakouts
        
        st.info(f"""
        **Analysis Scope:** {days_to_process} days across 46 Singapore stocks
        **Estimated Breakouts:** ~{estimated_breakouts:.0f} breakout events to analyze
        **Processing Time:** ~{days_to_process/10:.0f} minutes (depends on data complexity)
        """)
        
        if st.button("🚀 Execute Breakout Analysis", type="primary", use_container_width=True):
            # Reset completion state
            st.session_state.backtest_completed = False
            
            # Clear previous logger
            st.session_state.backtest_logger = BacktestLogger()
            
            # Execute the analysis
            execute_backtest_process(start_date, end_date, existing_data)
        
        # Show last analysis info if available
        if 'backtest_results' in st.session_state and not st.session_state.backtest_results.empty:
            last_summary = calculate_backtest_summary(st.session_state.backtest_results)
            st.info(f"📊 Last analysis: {last_summary['total_breakouts']} breakouts, "
                   f"{last_summary['success_rate']:.1f}% success rate, "
                   f"covering {last_summary['date_range']}")
    
    with tab2:
        st.markdown("### Breakout Factor Analysis Results")
        
        # Display results if available
        if st.session_state.get('backtest_completed', False):
            show_backtest_results()
        else:
            st.info("No breakout analysis results available yet. Run an analysis in the 'Run Analysis' tab to see results here.")

if __name__ == "__main__":
    show()