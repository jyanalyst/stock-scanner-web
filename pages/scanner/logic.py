"""
Scanner Business Logic - Core calculations and processing
Extracted from scanner.py for better organization
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import logging
import streamlit as st

from pages.common.data_utils import safe_float, safe_int, safe_string
from pages.common.error_handler import (handle_error, ErrorLogger, structured_logger,
                                       safe_execute, with_error_handling, display_user_friendly_error,
                                       DataLoadError, DataProcessingError, DataValidationError)
from pages.common.ui_components import create_progress_container, update_progress, clear_progress
from pages.common.constants import PROGRESS_UPDATE_INTERVAL
from pages.common.performance import (performance_monitor, cached_data_operation,
                                     cached_computation, optimize_memory_usage,
                                     _memory_manager, _data_cache, _computation_cache)
from pages.common.data_validation import (validate_data_quality, clean_data,
                                        get_validation_summary, DataQualityMetrics)
from pages.scanner.constants import ScanProgress, ScanScope, ScanDateType
from core.technical_analysis import (add_enhanced_columns, get_mpi_trend_info, 
                                     calculate_bullish_signal_quality, calculate_bearish_signal_quality,
                                     calculate_signal_direction, get_entry_signal_label)

logger = logging.getLogger(__name__)


@performance_monitor("stock_scan")
def run_enhanced_stock_scan(stocks_to_scan: List[str], analysis_date: Optional[date] = None,
                           days_back: int = 59, rolling_window: int = 20) -> None:
    """
    Execute the enhanced stock scanning process from local files
    CORRECTED: Creates display columns immediately after merging reports
    PERFORMANCE OPTIMIZED: Added caching and memory management
    """

    error_logger = st.session_state.error_logger

    try:
        from core.data_fetcher import DataFetcher, set_global_data_fetcher

        from pages.common.error_handler import (safe_execute, with_error_handling,
                                               DataLoadError, DataProcessingError,
                                               validate_data_quality, handle_error, display_user_friendly_error)

        is_historical = analysis_date is not None
        is_single_stock = len(stocks_to_scan) == 1

        scope_text = f"single stock ({stocks_to_scan[0]})" if is_single_stock else f"{len(stocks_to_scan)} stocks"
        date_text = f"historical analysis (as of {analysis_date.strftime('%d/%m/%Y')})" if is_historical else "current data analysis"

        structured_logger.log('INFO', 'Scanner', f"Starting scan: {scope_text} with {date_text}")
        st.info(f"🔄 Scanning {scope_text} with {date_text}... Loading from local files...")

        progress_bar, status_text = create_progress_container()

        update_progress(progress_bar, status_text, 0, "🔧 Initializing data fetcher...")
        fetcher = safe_execute(DataFetcher, days_back=days_back,
                             component="DataFetcher", fallback_value=None)
        if fetcher is None:
            raise DataLoadError("Failed to initialize data fetcher",
                              recovery_action="Check configuration and try again")
        update_progress(progress_bar, status_text, ScanProgress.INIT, "🔧 Data fetcher initialized")

        # Check memory before loading data
        if _memory_manager.check_memory_pressure():
            optimize_memory_usage()

        update_progress(progress_bar, status_text, ScanProgress.INIT, "📥 Loading stock data from local files...")
        stock_data = safe_execute(fetcher.download_stock_data, stocks_to_scan, target_date=analysis_date,
                                component="DataLoading", fallback_value={})
        set_global_data_fetcher(fetcher)
        update_progress(progress_bar, status_text, ScanProgress.LOAD_DATA, "📥 Stock data loaded successfully")

        if not stock_data:
            raise DataLoadError("No stock data loaded from local files",
                              recovery_action="Check that data files exist and are accessible")

        # Skip all data validation and cleaning - use original data as-is
        update_progress(progress_bar, status_text, 0.35, "📊 Analyzing original data (no validation or cleaning applied)...")

        update_progress(progress_bar, status_text, ScanProgress.CALCULATE, "🔄 Calculating Pure MPI Expansion and technical analysis...")

        results = []
        processing_errors = []

        for i, (ticker, df_raw) in enumerate(stock_data.items()):
            try:
                # Check for empty dataframes only
                if df_raw is None or df_raw.empty:
                    structured_logger.log('WARNING', 'StockProcessing',
                                        f"Empty dataframe for {ticker}", ticker=ticker)
                    continue

                # Process all stocks regardless of data quality - no validation checks

                # Process stock data with error handling
                df_enhanced = safe_execute(add_enhanced_columns, df_raw, ticker, rolling_window,
                                          component="TechnicalAnalysis", fallback_value=None)
                if df_enhanced is None or df_enhanced.empty:
                    processing_errors.append(f"{ticker}: Technical analysis failed")
                    continue

                # Skip technical indicator validation - process all data

                analysis_row = df_enhanced.iloc[-1]
                actual_date = analysis_row.name

                result = _create_result_dict(analysis_row, actual_date, ticker, fetcher)
                results.append(result)

                progress_range = ScanProgress.PROCESS_STOCKS_END - ScanProgress.PROCESS_STOCKS_START
                progress = ScanProgress.PROCESS_STOCKS_START + (progress_range * (i + 1) / len(stock_data))
                update_progress(progress_bar, status_text, progress, f"🔄 Processing {ticker}...")

                time.sleep(PROGRESS_UPDATE_INTERVAL)

            except Exception as e:
                error_info = handle_error(e, "StockProcessing", {"ticker": ticker}, show_user_message=False)
                processing_errors.append(f"{ticker}: {str(e)} (ID: {error_info.get('correlation_id', 'N/A')})")
                continue

        update_progress(progress_bar, status_text, ScanProgress.PREPARE_RESULTS, "📊 Preparing filtered results...")

        if not results:
            raise DataProcessingError("No stocks were successfully processed",
                                    recovery_action="Check data quality and error logs")

        results_df = pd.DataFrame(results)

        # Optimize memory usage after creating results
        results_df = _memory_manager.optimize_dataframe(results_df)

        # Load and merge analyst reports with enhanced error handling
        update_progress(progress_bar, status_text, ScanProgress.ANALYST_REPORTS, "📊 Loading analyst reports...")

        try:
            from utils.analyst_reports import get_cached_reports, merge_reports_with_scan_results

            all_reports, latest_reports = safe_execute(get_cached_reports,
                                                     component="AnalystReports", fallback_value=(pd.DataFrame(), pd.DataFrame()))

            if latest_reports is not None and not latest_reports.empty:
                results_df = safe_execute(merge_reports_with_scan_results, results_df, latest_reports,
                                        component="ReportMerging", fallback_value=results_df)

                matches = results_df['sentiment_score'].notna().sum() if 'sentiment_score' in results_df.columns else 0
                structured_logger.log('INFO', 'AnalystReports',
                                    f"Merged analyst reports: {matches} matches out of {len(results_df)} stocks")
                update_progress(progress_bar, status_text, ScanProgress.ANALYST_REPORTS, f"📊 Analyst reports: {matches} matches found")

                # Create analyst display columns with error handling
                if 'sentiment_score' in results_df.columns:
                    try:
                        results_df['Sentiment_Display'] = results_df.apply(
                            lambda row: f"{format_sentiment_emoji(row['sentiment_label'])} {row['sentiment_score']:.2f}"
                            if pd.notna(row['sentiment_score']) else '—',
                            axis=1
                        )
                        results_df['Report_Date_Display'] = results_df.apply(
                            lambda row: row['report_date'].strftime('%b %d')
                            if pd.notna(row['report_date']) else '—',
                            axis=1
                        )
                        results_df['Report_Count_Display'] = results_df.apply(
                            lambda row: f"{int(row['report_count'])} 📊"
                            if pd.notna(row['report_count']) and row['report_count'] > 1
                            else ('1 📊' if pd.notna(row['report_count']) else '—'),
                            axis=1
                        )
                        structured_logger.log('INFO', 'AnalystReports', "Created analyst display columns")
                    except Exception as e:
                        handle_error(e, "AnalystDisplay", {"operation": "create_display_columns"}, show_user_message=False)
            else:
                structured_logger.log('INFO', 'AnalystReports', "No analyst reports available")

        except Exception as e:
            handle_error(e, "AnalystReports", {"operation": "load_and_merge"}, show_user_message=False)

        # Load and merge earnings reports with enhanced error handling
        update_progress(progress_bar, status_text, ScanProgress.EARNINGS_REPORTS, "💰 Loading earnings reports...")

        try:
            from utils.earnings_reports import get_cached_earnings, merge_earnings_with_scan_results

            all_earnings, latest_earnings = safe_execute(get_cached_earnings,
                                                       component="EarningsReports", fallback_value=(pd.DataFrame(), pd.DataFrame()))

            if latest_earnings is not None and not latest_earnings.empty:
                results_df = safe_execute(merge_earnings_with_scan_results, results_df, latest_earnings,
                                        component="EarningsMerging", fallback_value=results_df)

                matches = results_df['revenue'].notna().sum() if 'revenue' in results_df.columns else 0
                structured_logger.log('INFO', 'EarningsReports',
                                    f"Merged earnings reports: {matches} matches out of {len(results_df)} stocks")
                update_progress(progress_bar, status_text, ScanProgress.EARNINGS_REPORTS, f"💰 Earnings reports: {matches} matches found")

                # Create earnings display columns with error handling
                if 'revenue' in results_df.columns:
                    try:
                        results_df['Earnings_Period'] = results_df.apply(
                            lambda row: row['report_type'] if pd.notna(row.get('report_type')) else '—',
                            axis=1
                        )

                        results_df['Guidance_Display'] = results_df.apply(
                            lambda row: f"{format_guidance_emoji(row['guidance_tone'])} {row['guidance_tone'].title()}"
                            if pd.notna(row.get('guidance_tone')) else '—',
                            axis=1
                        )

                        results_df['Rev_YoY_Display'] = results_df.apply(
                            lambda row: format_percentage_change(row['revenue_yoy_change'])
                            if pd.notna(row.get('revenue_yoy_change')) else '—',
                            axis=1
                        )

                        def get_eps_dpu_display(row):
                            if pd.isna(row.get('company_type')):
                                return '—'

                            company_type = row['company_type']

                            if company_type in ['reit', 'business_trust']:
                                if pd.notna(row.get('dpu_yoy_change')):
                                    return f"DPU: {format_percentage_change(row['dpu_yoy_change'])}"
                                else:
                                    return '—'
                            else:
                                if pd.notna(row.get('eps_yoy_change')):
                                    return f"EPS: {format_percentage_change(row['eps_yoy_change'])}"
                                else:
                                    return '—'

                        results_df['EPS_DPU_Display'] = results_df.apply(get_eps_dpu_display, axis=1)
                        structured_logger.log('INFO', 'EarningsReports', "Created earnings display columns")

                        # Calculate earnings reaction analysis for intraday traders
                        try:
                            from utils.earnings_reports import calculate_earnings_reaction_analysis, format_earnings_reaction_display

                            results_df['earnings_reaction_stats'] = results_df['ticker_sgx'].apply(
                                lambda ticker: calculate_earnings_reaction_analysis(ticker)
                            )

                            results_df['Earnings_Reaction'] = results_df['earnings_reaction_stats'].apply(
                                format_earnings_reaction_display
                            )

                            reaction_count = results_df['Earnings_Reaction'].notna().sum()
                            structured_logger.log('INFO', 'EarningsReaction',
                                                f"Calculated earnings reaction for {reaction_count} stocks")
                        except Exception as e:
                            handle_error(e, "EarningsReaction", {"operation": "calculate_reaction"}, show_user_message=False)

                    except Exception as e:
                        handle_error(e, "EarningsDisplay", {"operation": "create_display_columns"}, show_user_message=False)
            else:
                structured_logger.log('INFO', 'EarningsReports', "No earnings reports available")

        except Exception as e:
            handle_error(e, "EarningsReports", {"operation": "load_and_merge"}, show_user_message=False)

        # Store results in session state
        st.session_state.scan_results = results_df
        st.session_state.last_scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.last_scan_config = {
            'scope': ScanScope.SINGLE_STOCK if is_single_stock else ScanScope.FULL_WATCHLIST,
            'date': f'Historical ({analysis_date.strftime("%d/%m/%Y")})' if is_historical else 'Current',
            'stock_count': len(results_df)
        }

        update_progress(progress_bar, status_text, ScanProgress.COMPLETE, "✅ Scan completed!")
        clear_progress(progress_bar, status_text)

        time.sleep(1)

        success_message = f"🎉 Scan completed! Analyzed {len(results_df)} stocks successfully"
        if processing_errors:
            success_message += f" ({len(processing_errors)} errors - check log for details)"

        st.success(success_message)
        structured_logger.log('INFO', 'Scanner', f"Scan completed: {len(results_df)} stocks processed")

        st.rerun()

    except (DataLoadError, DataProcessingError, DataValidationError) as e:
        # Handle known application errors with user-friendly messages
        display_user_friendly_error(e, show_detailed=True)
        structured_logger.log('ERROR', 'Scanner', f"Scan failed with known error: {e.error_code}")

    except Exception as e:
        # Handle unexpected errors
        error_info = handle_error(e, "ScanExecution")
        st.error("❌ Pure MPI Expansion scan failed with critical error - check error log for details")
        structured_logger.log('CRITICAL', 'Scanner', f"Scan failed with unexpected error: {str(e)}")
        raise  # Re-raise to trigger the outer exception handler

    finally:
        # Clean up progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()


def _create_result_dict(analysis_row: pd.Series, actual_date, ticker: str, fetcher) -> dict:
    """Create result dictionary from analysis row"""
    try:
        company_name = fetcher.get_company_name(ticker)
    except Exception:
        company_name = ticker.replace('.SG', '')

    close_price = float(analysis_row['Close'])
    price_decimals = 3 if close_price < 1.00 else 2

    def safe_round(value, decimals):
        try:
            return round(float(value), decimals) if not pd.isna(value) else 0
        except:
            return 0

    mpi_trend = str(analysis_row.get('MPI_Trend', 'Unknown'))
    mpi_velocity = float(analysis_row.get('MPI_Velocity', 0.0)) if not pd.isna(analysis_row.get('MPI_Velocity', 0.0)) else 0.0

    try:
        from core.technical_analysis import get_mpi_trend_info
        mpi_trend_info = get_mpi_trend_info(mpi_trend, mpi_velocity)
    except:
        mpi_trend_info = {'emoji': '❓', 'description': 'Unknown'}

    higher_h = safe_int(analysis_row.get('Higher_H', 0))
    higher_hl = safe_int(analysis_row.get('Higher_HL', 0))
    lower_l = safe_int(analysis_row.get('Lower_L', 0))
    lower_hl = safe_int(analysis_row.get('Lower_HL', 0))

    if higher_hl == 1:
        hl_pattern = "HHL"  # Higher High & Low
    elif higher_h == 1:
        hl_pattern = "HH"   # Higher High only
    elif lower_hl == 1:
        hl_pattern = "LHL"  # Lower High & Low
    elif lower_l == 1:
        hl_pattern = "LL"   # Lower Low only
    else:
        hl_pattern = "-"

    # Calculate both bullish and bearish signal quality scores
    mpi_value = float(analysis_row.get('MPI', 0.5))
    
    bullish_quality = calculate_bullish_signal_quality(
        mpi=mpi_value,
        velocity=mpi_velocity,
        higher_h=higher_h
    )
    
    bearish_quality = calculate_bearish_signal_quality(
        mpi=mpi_value,
        velocity=mpi_velocity,
        lower_l=lower_l
    )
    
    # Determine signal direction based on velocity
    signal_direction = calculate_signal_direction(mpi_velocity)
    
    # Select the active signal quality based on direction
    active_quality = bullish_quality if signal_direction == 'bullish' else bearish_quality
    
    # Get the appropriate entry signal label
    entry_signal = get_entry_signal_label(bullish_quality, bearish_quality, signal_direction)
    
    # Determine Signal Bias: MPI primary, IBS fallback
    ibs_value = round(float(analysis_row['IBS']), 3) if not pd.isna(analysis_row['IBS']) else 0.5
    
    def determine_signal_bias(mpi_direction, ibs):
        """Determine signal bias: MPI primary, IBS fallback"""
        if mpi_direction == 'bullish':
            return '🟢 BULLISH'
        elif mpi_direction == 'bearish':
            return '🔴 BEARISH'
        else:
            # Fallback to IBS
            if ibs > 0.5:
                return '🟢 BULLISH'
            elif ibs < 0.5:
                return '🔴 BEARISH'
            else:
                return '⚪ NEUTRAL'
    
    signal_bias = determine_signal_bias(signal_direction, ibs_value)
    
    result = {
        'Signal_Bias': signal_bias,
        'Ticker': ticker,
        'Name': company_name,
        'Analysis_Date': actual_date.strftime('%d/%m/%Y') if hasattr(actual_date, 'strftime') else str(actual_date),
        'Close': round(close_price, price_decimals),
        'High': safe_round(analysis_row['High'], price_decimals),
        'Low': safe_round(analysis_row['Low'], price_decimals),
        'IBS': round(float(analysis_row['IBS']), 3) if not pd.isna(analysis_row['IBS']) else 0,
        'Higher_H': higher_h,
        'Higher_HL': higher_hl,
        'Lower_L': lower_l,
        'Lower_HL': lower_hl,
        'HL_Pattern': hl_pattern,
        'VW_Range_Velocity': round(float(analysis_row.get('VW_Range_Velocity', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Velocity', 0)) else 0,
        'VW_Range_Percentile': round(float(analysis_row.get('VW_Range_Percentile', 0)), 4) if not pd.isna(analysis_row.get('VW_Range_Percentile', 0)) else 0,
        'Rel_Range_Signal': safe_int(analysis_row.get('Rel_Range_Signal', 0)),
        'MPI': round(mpi_value, 2),
        'MPI_Velocity': round(mpi_velocity, 2),
        'MPI_Trend': mpi_trend,
        'MPI_Bullish_Quality': bullish_quality,
        'MPI_Bearish_Quality': bearish_quality,
        'MPI_Signal_Direction': signal_direction,
        'MPI_Signal_Quality': active_quality,
        'MPI_Entry_Signal': entry_signal,
        'Relative_Volume': round(float(analysis_row.get('Relative_Volume', 100.0)), 1) if not pd.isna(analysis_row.get('Relative_Volume', 100.0)) else 100.0,
        'RelVol_Velocity': round(float(analysis_row.get('RelVol_Velocity', 0.0)), 1) if not pd.isna(analysis_row.get('RelVol_Velocity', 0.0)) else 0.0,
        'RelVol_Trend': str(analysis_row.get('RelVol_Trend', 'Stable')),
        'RelVol_Percentile': round(float(analysis_row.get('RelVol_Percentile', 0.5)), 4) if not pd.isna(analysis_row.get('RelVol_Percentile', 0.5)) else 0.5,
        'RelVol_Percentile_Velocity': round(float(analysis_row.get('RelVol_Percentile_Velocity', 0.0)), 4) if not pd.isna(analysis_row.get('RelVol_Percentile_Velocity', 0.0)) else 0.0,
        'High_Rel_Volume_150': safe_int(analysis_row.get('High_Rel_Volume_150', 0)),
        'High_Rel_Volume_200': safe_int(analysis_row.get('High_Rel_Volume_200', 0)),
        'Price_Decimals': price_decimals
    }

    return result


def format_mpi_visual(mpi_value: float) -> str:
    """Convert MPI to visual blocks for intuitive display"""
    if pd.isna(mpi_value):
        return "░░░░░░░░░░"

    blocks = max(0, min(10, int(mpi_value * 10)))
    return "█" * blocks + "░" * (10 - blocks)


def format_sentiment_emoji(sentiment_label: str) -> str:
    """Get emoji for sentiment label"""
    emoji_map = {
        'positive': '📈',
        'neutral': '➖',
        'negative': '📉'
    }
    return emoji_map.get(sentiment_label, '❓')


def format_guidance_emoji(guidance_tone: str) -> str:
    """Get emoji for guidance tone"""
    emoji_map = {
        'positive': '📈',
        'neutral': '➖',
        'negative': '📉'
    }
    return emoji_map.get(guidance_tone, '❓')


def format_percentage_change(value: float) -> str:
    """
    Format percentage change with color indication

    Returns:
        String like "+15.3%" or "-5.2%"
    """
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:+.1f}%"
