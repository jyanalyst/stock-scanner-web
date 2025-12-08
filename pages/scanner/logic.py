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
                                     get_mpi_zone)

logger = logging.getLogger(__name__)


def calculate_signal_score(row):
    """
    Calculate composite signal score (0-100 scale)

    Components:
    - Flow_Velocity_Rank: 35% (acceleration)
    - Flow_Rank: 25% (current strength)
    - Flow_Percentile: 20% (sweet spot positioning)
    - Volume_Conviction: 15% (commitment)
    - Three_Indicator: 5% (confirmation)
    """

    # Component 1: Flow_Velocity_Rank (35 points max)
    vel_score = row['Flow_Velocity_Rank'] * 0.35

    # Component 2: Flow_Rank (25 points max)
    rank_score = row['Flow_Rank'] * 0.25

    # Component 3: Flow_Percentile (20 points max) - Sweet spot scoring
    perc = row['Flow_Percentile']
    if 40 <= perc <= 70:
        perc_score = 20  # IDEAL sweet spot
    elif 71 <= perc <= 80:
        perc_score = 15  # Good
    elif 81 <= perc <= 85:
        perc_score = 10  # Late stage
    elif 86 <= perc <= 95:
        perc_score = 5   # Exhaustion risk
    elif perc > 95:
        perc_score = 0   # PEAK - avoid
    else:  # < 40
        perc_score = 10  # Too weak

    # Component 4: Volume_Conviction (15 points max)
    conv = row['Volume_Conviction']
    if conv >= 1.5:
        conv_score = 15  # Excellent
    elif conv >= 1.25:
        conv_score = 12  # Good
    elif conv >= 1.0:
        conv_score = 8   # Marginal
    else:
        conv_score = 3   # Weak

    # Component 5: Three_Indicator confirmation (5 points max)
    indicators_above_55 = sum([
        row['MPI_Percentile'] > 55,
        row['IBS_Percentile'] > 55,
        row['VPI_Percentile'] > 55
    ])
    indicator_score = indicators_above_55 * 1.67  # 3 indicators = 5 points

    total_score = vel_score + rank_score + perc_score + conv_score + indicator_score

    return round(total_score, 1)


def calculate_suggested_risk(row):
    """
    Calculate suggested position size based on Trade_Rank and Signal_Score

    Returns: Risk percentage (0.0 to 1.0)
    """
    rank = row['Trade_Rank']
    score = row['Signal_Score']

    # Rank 1 - Best idea
    if rank == 1:
        if score >= 75:
            return 1.0   # Full allocation
        elif score >= 65:
            return 0.75  # Reduced
        else:
            return 0.5   # Marginal

    # Rank 2 - Second best
    elif rank == 2:
        if score >= 75:
            return 0.75
        elif score >= 65:
            return 0.5
        else:
            return 0.25

    # Rank 3 - Third best
    elif rank == 3:
        if score >= 75:
            return 0.5
        elif score >= 65:
            return 0.25
        else:
            return 0.25  # Minimum or skip

    # Rank 4+ - Don't trade
    else:
        return 0.0


def get_quality_flag(score):
    """Convert numeric score to quality flag"""
    if score >= 80:
        return '⭐⭐⭐ EXCELLENT'
    elif score >= 70:
        return '⭐⭐ GOOD'
    elif score >= 60:
        return '⭐ MARGINAL'
    else:
        return '⚠️ RISKY'


def add_ranking_columns(df):
    """
    Add Signal_Score, Trade_Rank, Suggested_Risk, Quality_Flag to dataframe

    This should be called AFTER filtering for bullish/bearish signals
    but BEFORE displaying results
    """
    if df.empty:
        return df

    # Create explicit copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Calculate signal score for each stock
    df['Signal_Score'] = df.apply(calculate_signal_score, axis=1)

    # Rank by score (descending) - best score gets rank 1
    df['Trade_Rank'] = df['Signal_Score'].rank(method='dense', ascending=False).astype(int)

    # Calculate suggested risk based on rank and score
    df['Suggested_Risk'] = df.apply(calculate_suggested_risk, axis=1)

    # Add quality flag for visual reference
    df['Quality_Flag'] = df['Signal_Score'].apply(get_quality_flag)

    return df


@performance_monitor("stock_scan")
def run_enhanced_stock_scan(stocks_to_scan: List[str], analysis_date: Optional[date] = None,
                           days_back: int = 59, rolling_window: int = 20,
                           # NEW: Confirmation filter parameters
                           use_ibs: bool = False, use_rvol: bool = False, use_rrange: bool = False,
                           confirmation_logic: str = "OR",
                           ibs_threshold: float = 0.10, rvol_threshold: float = 0.20, rrange_threshold: float = 0.30) -> None:
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
                                          # NEW: Pass confirmation parameters
                                          use_ibs=use_ibs, use_rvol=use_rvol, use_rrange=use_rrange,
                                          confirmation_logic=confirmation_logic,
                                          ibs_threshold=ibs_threshold, rvol_threshold=rvol_threshold,
                                          rrange_threshold=rrange_threshold,
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

        # ===== CALCULATE CROSS-STOCK RANKINGS =====
        # These rankings compare stocks against each other in the watchlist (not time-series)
        update_progress(progress_bar, status_text, 0.75, "📊 Calculating cross-stock flow rankings...")

        try:
            # Calculate cross-stock rankings for key flow metrics
            # These show relative strength: "Which stocks are strongest TODAY?"
            if 'Flow_10D' in results_df.columns:
                results_df['Flow_Rank'] = results_df['Flow_10D'].rank(pct=True) * 100
                results_df['Flow_Rank'] = results_df['Flow_Rank'].round(1)

            if 'Flow_Velocity' in results_df.columns:
                results_df['Flow_Velocity_Rank'] = results_df['Flow_Velocity'].rank(pct=True) * 100
                results_df['Flow_Velocity_Rank'] = results_df['Flow_Velocity_Rank'].round(1)

            if 'Volume_Conviction' in results_df.columns:
                results_df['Volume_Conviction_Rank'] = results_df['Volume_Conviction'].rank(pct=True) * 100
                results_df['Volume_Conviction_Rank'] = results_df['Volume_Conviction_Rank'].round(1)

            structured_logger.log('INFO', 'CrossStockRankings',
                                f"Calculated cross-stock rankings for {len(results_df)} stocks")
        except Exception as e:
            handle_error(e, "CrossStockRankings", {"operation": "calculate_rankings"}, show_user_message=False)

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
    """Create result dictionary from analysis row with Three-Indicator System"""
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

    # ===== PRIMARY SIGNAL =====
    signal_bias = str(analysis_row.get('Signal_Bias', '⚪ NEUTRAL'))
    signal_state = str(analysis_row.get('Signal_State', '⚪ Neutral'))
    conviction_level = str(analysis_row.get('Conviction_Level', 'Low'))
    mpi_position = str(analysis_row.get('MPI_Position', '❓ UNKNOWN'))
    
    # Boolean flags
    is_triple = bool(analysis_row.get('Is_Triple_Aligned', False))
    is_divergent = bool(analysis_row.get('Is_Divergent', False))
    is_accumulation = bool(analysis_row.get('Is_Accumulation', False))

    # ===== PATTERN DETAILS =====
    ref_high = safe_round(analysis_row.get('Ref_High', np.nan), price_decimals)
    ref_low = safe_round(analysis_row.get('Ref_Low', np.nan), price_decimals)
    purge_level = safe_round(analysis_row.get('Purge_Level', np.nan), price_decimals)
    entry_level = safe_round(analysis_row.get('Entry_Level', np.nan), price_decimals)
    bars_since_break = safe_int(analysis_row.get('Bars_Since_Break', np.nan))

    # ===== THREE-INDICATOR METRICS (PERCENTILES) =====
    mpi_percentile = round(float(analysis_row.get('MPI_Percentile', 50.0)), 1)
    ibs_percentile = round(float(analysis_row.get('IBS_Percentile', 50.0)), 1)
    vpi_percentile = round(float(analysis_row.get('VPI_Percentile', 50.0)), 1)

    # ===== ACCELERATION METRICS (RAW) =====
    ibs_accel = round(float(analysis_row.get('IBS_Accel', 0)), 3)
    rvol_accel = round(float(analysis_row.get('RVol_Accel', 0)), 3)
    rrange_accel = round(float(analysis_row.get('RRange_Accel', 0)), 3)
    vpi_accel = round(float(analysis_row.get('VPI_Accel', 0)), 3)

    # ===== CONTEXT METRICS =====
    mpi_value = float(analysis_row.get('MPI', 0.5))
    mpi_velocity = float(analysis_row.get('MPI_Velocity', 0.0))
    mpi_trend = str(analysis_row.get('MPI_Trend', 'Unknown'))
    mpi_zone = int(analysis_row.get('MPI_Zone', 3))

    # Price action patterns
    higher_h = safe_int(analysis_row.get('Higher_H', 0))
    higher_hl = safe_int(analysis_row.get('Higher_HL', 0))
    lower_l = safe_int(analysis_row.get('Lower_L', 0))
    lower_hl = safe_int(analysis_row.get('Lower_HL', 0))

    if higher_hl == 1:
        hl_pattern = "HHL"
    elif higher_h == 1:
        hl_pattern = "HH"
    elif lower_hl == 1:
        hl_pattern = "LHL"
    elif lower_l == 1:
        hl_pattern = "LL"
    else:
        hl_pattern = "-"

    # IBS and volume metrics
    ibs_value = round(float(analysis_row.get('IBS', 0.5)), 3)
    relative_volume = round(float(analysis_row.get('Relative_Volume', 100.0)), 1)
    relvol_velocity = round(float(analysis_row.get('RelVol_Velocity', 0.0)), 1)
    relvol_trend = str(analysis_row.get('RelVol_Trend', 'Stable'))
    vpi_velocity = round(float(analysis_row.get('VPI_Velocity', 0.0)), 2)

    # Range and volatility metrics
    vw_range_velocity = round(float(analysis_row.get('VW_Range_Velocity', 0)), 4)
    vw_range_percentile = round(float(analysis_row.get('VW_Range_Percentile', 0)), 4)
    rel_range_signal = safe_int(analysis_row.get('Rel_Range_Signal', 0))

    # Build comprehensive result dictionary
    result = {
        # ===== PRIMARY SIGNAL =====
        'Signal_Bias': signal_bias,
        'Signal_State': signal_state,
        'Conviction_Level': conviction_level,
        'MPI_Position': mpi_position,
        'Is_Triple_Aligned': is_triple,
        'Is_Divergent': is_divergent,
        'Is_Accumulation': is_accumulation,

        # ===== PATTERN DETAILS =====
        'Ref_High': ref_high,
        'Ref_Low': ref_low,
        'Purge_Level': purge_level,
        'Entry_Level': entry_level,
        'Bars_Since_Break': bars_since_break,

        # ===== THREE-INDICATOR METRICS =====
        'MPI_Percentile': mpi_percentile,
        'IBS_Percentile': ibs_percentile,
        'VPI_Percentile': vpi_percentile,

        # ===== ACCELERATION METRICS =====
        'IBS_Accel': ibs_accel,
        'RVol_Accel': rvol_accel,
        'RRange_Accel': rrange_accel,
        'VPI_Accel': vpi_accel,

        # ===== BASIC IDENTIFICATION =====
        'Ticker': ticker,
        'Name': company_name,
        'Analysis_Date': actual_date.strftime('%d/%m/%Y') if hasattr(actual_date, 'strftime') else str(actual_date),
        'Close': round(close_price, price_decimals),
        'High': safe_round(analysis_row['High'], price_decimals),
        'Low': safe_round(analysis_row['Low'], price_decimals),

        # ===== CONTEXT: PRICE ACTION =====
        'IBS': ibs_value,
        'Higher_H': higher_h,
        'Higher_HL': higher_hl,
        'Lower_L': lower_l,
        'Lower_HL': lower_hl,
        'HL_Pattern': hl_pattern,

        # ===== CONTEXT: VOLATILITY/RANGE =====
        'VW_Range_Velocity': vw_range_velocity,
        'VW_Range_Percentile': vw_range_percentile,
        'Rel_Range_Signal': rel_range_signal,

        # ===== CONTEXT: VOLUME =====
        'Relative_Volume': relative_volume,
        'RelVol_Velocity': relvol_velocity,
        'RelVol_Trend': relvol_trend,
        'VPI_Velocity': vpi_velocity,

        # ===== CONTEXT: MPI INDICATORS =====
        'MPI': round(mpi_value, 2),
        'MPI_Velocity': round(mpi_velocity, 2),
        'MPI_Trend': mpi_trend,
        'MPI_Zone': mpi_zone,

        # ===== PHASE 1: INSTITUTIONAL FLOW METRICS =====
        # Raw flow metrics
        'Daily_Flow': round(float(analysis_row.get('Daily_Flow', 0.0)), 1),
        'Flow_10D': round(float(analysis_row.get('Flow_10D', 0.0)), 1),
        'Flow_Velocity': round(float(analysis_row.get('Flow_Velocity', 0.0)), 2),
        'Flow_Regime': str(analysis_row.get('Flow_Regime', 'Neutral')),

        # Individual stock percentiles (historical, 100-day rolling)
        'Flow_Percentile': round(float(analysis_row.get('Flow_Percentile', 50.0)), 1),
        'Flow_Velocity_Percentile': round(float(analysis_row.get('Flow_Velocity_Percentile', 50.0)), 1),

        # Cross-stock rankings (watchlist relative strength) - calculated in run_enhanced_stock_scan
        'Flow_Rank': round(float(analysis_row.get('Flow_Rank', 50.0)), 1),
        'Flow_Velocity_Rank': round(float(analysis_row.get('Flow_Velocity_Rank', 50.0)), 1),

        # Volume conviction metrics
        'Volume_Conviction': round(float(analysis_row.get('Volume_Conviction', 1.0)), 2),
        'Volume_Conviction_Rank': round(float(analysis_row.get('Volume_Conviction_Rank', 50.0)), 1),
        'Conviction_Velocity': round(float(analysis_row.get('Conviction_Velocity', 0.0)), 3),
        'Avg_Vol_Up_10D': round(float(analysis_row.get('Avg_Vol_Up_10D', 0.0)), 0),
        
        'Divergence_Gap': round(float(analysis_row.get('Divergence_Gap', 0.0)), 2),
        'Divergence_Severity': round(float(analysis_row.get('Divergence_Severity', 0.0)), 1),
        'Price_Percentile': round(float(analysis_row.get('Price_Percentile', 0.5)), 2),

        # ===== UTILITY =====
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
