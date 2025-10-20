"""
Scanner UI Components - Streamlit interface elements
Extracted from scanner.py for better organization
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from io import StringIO
import numpy as np
import time

# Import common utilities
from pages.common.ui_components import (
    create_progress_container, update_progress, clear_progress,
    display_metric_grid, create_dynamic_column_config, format_mpi_visual,
    create_download_button, create_info_box, create_success_box,
    create_warning_box, create_error_box, create_section_header,
    display_data_table, validate_dataframe
)
from pages.common.constants import (
    DEFAULT_DAYS_BACK, DEFAULT_ROLLING_WINDOW, PROGRESS_UPDATE_INTERVAL
)
from pages.scanner.constants import ScanScope, ScanDateType
from utils.date_utils import format_singapore_date
from pages.common.error_handler import (handle_error, safe_execute, with_error_handling,
                                       FileNotFoundError as AppFileNotFoundError,
                                       NetworkTimeoutError, NetworkConnectionError,
                                       display_user_friendly_error)
from pages.common.performance import get_performance_stats
from pages.common.data_validation import validate_data_quality, get_validation_summary
from utils import analyst_reports


def show_update_prompt() -> Optional[bool]:
    """
    Check for updates and prompt user with options for EOD update and/or yfinance download
    ENHANCED: Now shows both EOD and yfinance options when appropriate
    Returns True if update was performed or skipped, False if check failed
    """
    try:
        from core.local_file_loader import get_local_loader

        loader = get_local_loader()

        # Check for updates
        with st.spinner("Checking for Historical_Data updates and date gaps..."):
            (eod_available, eod_filename, eod_date,
             gap_exists, last_historical_date, current_working_day) = loader.check_for_updates()

        # Scenario A: EOD available AND gap exists - show BOTH options
        if eod_available and gap_exists:
            st.info("📥 **Two update options available!**")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Option 1: EOD File Update")
                st.metric("Latest EOD File", eod_filename)
                st.metric("EOD Date", eod_date.strftime('%d/%m/%Y') if eod_date else 'Unknown')

                if st.button("🔄 Update from EOD File", type="primary", use_container_width=True, key="eod_update_btn"):
                    result = perform_eod_update(loader, force=False)
                    if result:
                        st.rerun()
                    return result

            with col2:
                st.markdown("### Option 2: External Download")
                st.metric("Last Historical", last_historical_date.strftime('%d/%m/%Y') if last_historical_date else 'None')
                st.metric("Current Working Day", current_working_day.strftime('%d/%m/%Y') if current_working_day else 'Unknown')

                if gap_exists and last_historical_date and current_working_day:
                    gap_days = (current_working_day - last_historical_date).days
                    st.caption(f"Gap: {gap_days} day(s)")

                if st.button("📥 Download Missing Data", type="secondary", use_container_width=True, key="download_missing_btn"):
                    if last_historical_date and current_working_day:
                        start_date = last_historical_date + timedelta(days=1)
                        result = perform_yfinance_download(loader, start_date, current_working_day)
                        if result:
                            st.rerun()
                        return result
                    else:
                        st.error("Cannot determine date range for download")
                        return False

            # Skip button
            if st.button("⏭️ Skip Updates", use_container_width=True, key="skip_both_btn"):
                st.info("Skipped updates. You can update later.")
                return True

            return None  # Waiting for user action

        # Scenario B: Only EOD available (no gap)
        elif eod_available and not gap_exists:
            st.info("📥 **New EOD data available!**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latest EOD File", eod_filename)
            with col2:
                st.metric("Date", eod_date.strftime('%d/%m/%Y') if eod_date else 'Unknown')

            st.markdown("Historical_Data can be updated with this new data.")

            col_update, col_skip = st.columns(2)

            with col_update:
                if st.button("🔄 Update Now", type="primary", use_container_width=True, key="eod_only_update_btn"):
                    result = perform_eod_update(loader, force=False)
                    if result:
                        st.rerun()
                    return result

            with col_skip:
                if st.button("⏭️ Skip This Time", use_container_width=True, key="eod_only_skip_btn"):
                    st.info("Skipped update. You can update later using the buttons below.")
                    return True

            return None  # Waiting for user action

        # Scenario C: Only gap exists (no EOD)
        elif not eod_available and gap_exists:
            st.warning("📊 **Date gap detected!**")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Last Historical", last_historical_date.strftime('%d/%m/%Y') if last_historical_date else 'None')
            with col2:
                st.metric("Current Working Day", current_working_day.strftime('%d/%m/%Y') if current_working_day else 'Unknown')
            with col3:
                if last_historical_date and current_working_day:
                    gap_days = (current_working_day - last_historical_date).days
                    st.metric("Gap", f"{gap_days} day(s)")

            st.markdown("No EOD file available, but you can download missing dates.")

            col_download, col_skip = st.columns(2)

            with col_download:
                if st.button("📥 Download Missing Data", type="primary", use_container_width=True, key="gap_only_download_btn"):
                    if last_historical_date and current_working_day:
                        start_date = last_historical_date + timedelta(days=1)
                        result = perform_yfinance_download(loader, start_date, current_working_day)
                        if result:
                            st.rerun()
                        return result
                    else:
                        st.error("Cannot determine date range for download")
                        return False

            with col_skip:
                if st.button("⏭️ Skip for Now", use_container_width=True, key="gap_only_skip_btn"):
                    st.info("Skipped download. Data gap remains.")
                    return True

            return None  # Waiting for user action

        # Scenario D: No updates needed
        else:
            st.success("✅ Historical_Data is up to date!")

            # Get latest EOD file info for force update option
            latest_eod = loader.get_latest_eod_file()
            if latest_eod:
                eod_date_str = latest_eod.replace('.csv', '')
                eod_date_display = datetime.strptime(eod_date_str, '%d_%b_%Y')

                st.info(f"📄 Latest data: **{last_historical_date.strftime('%d/%m/%Y') if last_historical_date else 'Unknown'}**")
                st.caption(f"Current working day: {current_working_day.strftime('%d/%m/%Y') if current_working_day else 'Unknown'}")

                # Show force update options
                with st.expander("⚙️ Advanced: Force Update Options", expanded=False):
                    st.markdown("### 🔄 Force Update from EOD File")
                    st.warning("⚠️ **Force EOD Update** will re-process the latest EOD file even though it's already been imported.")
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

                    if st.button("🔄 Force Update from EOD", type="secondary", use_container_width=True, key="force_eod_button"):
                        result = perform_eod_update(loader, force=True)
                        if result:
                            st.success("Force EOD update completed!")
                            time.sleep(2)
                            st.rerun()
                        return result

                    # Add force EOD update even when no new EOD file exists
                    st.markdown("---")
                    st.markdown("### 🔄 Force Update Any EOD File")
                    st.info("Choose any EOD file to force re-process, even if it's already been imported.")
                    eod_files = loader.list_eod_files()
                    if eod_files:
                        selected_eod = st.selectbox(
                            "Select EOD file to force update:",
                            options=eod_files,
                            key="force_eod_select"
                        )
                        if st.button("🔄 Force Update Selected EOD", type="secondary", use_container_width=True, key="force_selected_eod_button"):
                            # Import the function that handles specific file updates
                            from core.local_file_loader import LocalFileLoader
                            # Create a temporary loader instance to handle specific file
                            temp_loader = LocalFileLoader()
                            # For now, we'll use the force update on latest file
                            # TODO: Implement specific file force update in LocalFileLoader
                            result = perform_eod_update(loader, force=True)
                            if result:
                                st.success(f"Force update completed!")
                                time.sleep(2)
                                st.rerun()
                            return result
                    else:
                        st.info("No EOD files available")

                    st.markdown("---")
                    st.markdown("### 📥 Force Download from External Source")
                    st.warning("⚠️ **Force Download** will download data regardless of current state.")
                    st.markdown("""
                    **Use this when:**
                    - You want to update to today's data without waiting for EOD file
                    - You need to re-download data to fix issues
                    - You want the absolute latest available data

                    **This will:**
                    - Calculate date range: (last EOD date + 1 day) to (current working day)
                    - Download data from all stocks
                    - Overwrite any existing dates in this range
                    - Use abbreviated volume format (÷1000)
                    """)

                    # Calculate download range based on EOD file date (not Historical_Data date)
                    if latest_eod and current_working_day:
                        # Parse date from EOD filename (e.g., "10_Oct_2025.csv" → 10/10/2025)
                        try:
                            eod_date_str = latest_eod.replace('.csv', '')
                            eod_date_obj = datetime.strptime(eod_date_str, '%d_%b_%Y').date()

                            # Force download range: (EOD date + 1) to (current working day)
                            force_start_date = eod_date_obj + timedelta(days=1)
                            force_end_date = current_working_day
                        except Exception as e:
                            st.error(f"Could not parse EOD date: {e}")
                            force_start_date = None
                            force_end_date = None

                        # Check if there's actually a range to download
                        if force_start_date and force_end_date and force_start_date <= force_end_date:
                            gap_days = (force_end_date - force_start_date).days + 1

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Last EOD Date", eod_date_obj.strftime('%d/%m/%Y'))
                            with col_b:
                                st.metric("Download Range", f"{gap_days} day(s)")
                            with col_c:
                                st.metric("End Date", force_end_date.strftime('%d/%m/%Y'))

                            st.info(f"📥 Will download: **{force_start_date.strftime('%d/%m/%Y')}** to **{force_end_date.strftime('%d/%m/%Y')}**")

                            if st.button("📥 Force Download Latest Data", type="secondary", use_container_width=True, key="force_download_button"):
                                result = perform_yfinance_download(loader, force_start_date, force_end_date, force_mode=True)
                                if result:
                                    st.success("Force download completed!")
                                    time.sleep(2)
                                    st.rerun()
                                return result
                        elif force_start_date and force_end_date:
                            st.info("✅ No date range to download - Historical data is current to working day")
                        else:
                            st.error("Cannot determine date range for force download")

            return True

    except (AppFileNotFoundError, NetworkTimeoutError, NetworkConnectionError) as e:
        display_user_friendly_error(e)
        return False
    except Exception as e:
        handle_error(e, "UpdateCheck")
        return False


def perform_eod_update(loader, force: bool = False) -> Optional[bool]:
    """
    Perform the Historical_Data update from EOD file
    ENHANCED: Added force parameter to bypass date checks
    """
    try:
        # Create progress containers
        progress_bar, status_text = create_progress_container()

        update_type = "FORCE UPDATE" if force else "UPDATE"
        update_progress(progress_bar, status_text, 0, f"🔄 Starting Historical_Data {update_type.lower()}...")

        # Show force update warning
        if force:
            st.warning(f"⚠️ **FORCE UPDATE MODE** - Re-processing latest EOD file regardless of dates")

        # Perform update
        stats = loader.update_historical_from_eod(force=force)

        # Show progress during update
        if stats['total_stocks'] > 0:
            for i, detail in enumerate(stats['details']):
                progress = (i + 1) / stats['total_stocks']
                status = detail['status']

                if status == 'updated':
                    update_progress(progress_bar, status_text, progress, f"✅ {detail['ticker']} - {detail['message']}")
                elif status == 'created':
                    update_progress(progress_bar, status_text, progress, f"🆕 {detail['ticker']} - {detail['message']}")
                elif status == 'skipped':
                    update_progress(progress_bar, status_text, progress, f"⏭️ {detail['ticker']} - {detail['message']}")
                elif status == 'error':
                    update_progress(progress_bar, status_text, progress, f"❌ {detail['ticker']} - {detail['message']}")

                time.sleep(PROGRESS_UPDATE_INTERVAL)

        # Clear progress indicators
        clear_progress(progress_bar, status_text)

        # Show summary
        if force:
            create_success_box("**Force Update Complete!**")
        else:
            create_success_box("**Update Complete!**")

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
            create_info_box(date_msg)

        # Show errors if any
        if stats['errors'] > 0:
            with st.expander("⚠️ View Errors", expanded=False):
                for detail in stats['details']:
                    if detail['status'] == 'error':
                        st.write(f"**{detail['ticker']}**: {detail['message']}")

        time.sleep(2)
        return True

    except (AppFileNotFoundError, NetworkTimeoutError, NetworkConnectionError) as e:
        display_user_friendly_error(e)
        return False
    except Exception as e:
        handle_error(e, "EODUpdateExecution")
        return False


def perform_yfinance_download(loader, start_date: date, end_date: date, force_mode: bool = False) -> Optional[bool]:
    """
    NEW: Perform yfinance download to fill date gaps
    """
    try:
        # Create progress containers
        progress_bar, status_text = create_progress_container()

        download_type = "FORCE DOWNLOAD" if force_mode else "download"
        update_progress(progress_bar, status_text, 0, f"📥 Starting {download_type}: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}...")

        # Show download info
        gap_days = (end_date - start_date).days + 1
        if force_mode:
            create_warning_box(f"⚠️ FORCE MODE: Downloading {gap_days} day(s) - will overwrite existing dates")
        else:
            create_info_box(f"📊 Downloading {gap_days} day(s) of data for all stocks...")

        # Perform download
        stats = loader.download_missing_dates_from_yfinance(start_date, end_date, force_mode=force_mode)

        # Show progress during download
        if stats['total_stocks'] > 0:
            for i, detail in enumerate(stats['details']):
                progress = (i + 1) / stats['total_stocks']
                status = detail['status']

                if status == 'updated':
                    update_progress(progress_bar, status_text, progress, f"✅ {detail['ticker']} - {detail['message']}")
                elif status == 'skipped':
                    update_progress(progress_bar, status_text, progress, f"⏭️ {detail['ticker']} - {detail['message']}")
                elif status == 'failed':
                    update_progress(progress_bar, status_text, progress, f"❌ {detail['ticker']} - {detail['message']}")

                time.sleep(PROGRESS_UPDATE_INTERVAL)

        # Clear progress indicators
        clear_progress(progress_bar, status_text)

        # Show summary
        if force_mode:
            create_success_box("**Force Download Complete!**")
        else:
            create_success_box("**Download Complete!**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stocks", stats['total_stocks'])
        with col2:
            st.metric("Updated", stats['updated'])
        with col3:
            st.metric("Skipped", stats['skipped'])
        with col4:
            st.metric("Failed", stats['failed'])

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Dates Added", stats['total_dates_added'])
        with col_b:
            create_info_box(f"📅 Date range: **{stats['start_date']}** to **{stats['end_date']}**")

        # Show failures if any
        if stats['failed'] > 0:
            with st.expander("⚠️ View Failed Downloads", expanded=False):
                for detail in stats['details']:
                    if detail['status'] == 'failed':
                        st.write(f"**{detail['ticker']}**: {detail['message']}")

        time.sleep(2)
        return True

    except (AppFileNotFoundError, NetworkTimeoutError, NetworkConnectionError) as e:
        display_user_friendly_error(e)
        return False
    except Exception as e:
        handle_error(e, "YfinanceDownloadExecution")
        return False


def show_scanning_configuration() -> Tuple[str, Optional[str], str, Optional[date]]:
    """Display the scanning configuration panel"""
    create_section_header("🎯 Scanning Configuration", "")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Scan Scope**")
        scan_scope = st.radio(
            "Choose scan scope:",
            [ScanScope.SINGLE_STOCK, ScanScope.FULL_WATCHLIST],
            help="Select whether to scan one stock or the entire watchlist"
        )

        selected_stock = None
        if scan_scope == ScanScope.SINGLE_STOCK:
            try:
                from utils.watchlist import get_active_watchlist
                watchlist = get_active_watchlist()
                selected_stock = st.selectbox(
                    "Select Stock:",
                    options=watchlist,
                    help="Choose which stock to analyze"
                )
            except Exception as e:
                handle_error(e, "WatchlistLoading")
                selected_stock = "A17U.SG"

    with col2:
        st.markdown("**📅 Analysis Date**")
        scan_date_type = st.radio(
            "Choose analysis date:",
            [ScanDateType.CURRENT, ScanDateType.HISTORICAL],
            help="Scan as of current date or specify a historical date"
        )

        historical_date = None
        if scan_date_type == ScanDateType.HISTORICAL:
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
                handle_error(e, "DateSelection")
                historical_date = date.today() - timedelta(days=7)

    return scan_scope, selected_stock, scan_date_type, historical_date


def show_advanced_settings() -> Tuple[int, int]:
    """Display advanced settings panel"""
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)

        with col1:
            days_back = st.number_input(
                "Days of Historical Data",
                min_value=30,
                max_value=250,
                value=DEFAULT_DAYS_BACK,
                help="Number of days to load (loads all available from local files)"
            )

        with col2:
            rolling_window = st.number_input(
                "Rolling Window",
                min_value=10,
                max_value=50,
                value=DEFAULT_ROLLING_WINDOW,
                help="Rolling window for technical calculations"
            )

    return days_back, rolling_window


def execute_scan_button(scan_scope: str, selected_stock: Optional[str],
                       scan_date_type: str, historical_date: Optional[date],
                       days_back: int, rolling_window: int) -> None:
    """Handle scan execution button and logic"""
    if st.button("🚀 Execute Scan", type="primary", use_container_width=True):
        # Initialize error logger if not exists
        if 'error_logger' not in st.session_state:
            from pages.common.error_handler import ErrorLogger
            st.session_state.error_logger = ErrorLogger()

        try:
            from pages.scanner.logic import run_enhanced_stock_scan

            if scan_scope == ScanScope.SINGLE_STOCK:
                stocks_to_scan = [selected_stock]
            else:
                from utils.watchlist import get_active_watchlist
                stocks_to_scan = get_active_watchlist()

            analysis_date = historical_date if scan_date_type == ScanDateType.HISTORICAL else None

            run_enhanced_stock_scan(
                stocks_to_scan=stocks_to_scan,
                analysis_date=analysis_date,
                days_back=days_back,
                rolling_window=rolling_window
            )

        except ImportError as e:
            handle_error(e, "ModuleImport")
            st.error("❌ Required modules not available - check error details above")
        except Exception as e:
            handle_error(e, "ScanExecution")
            st.error("❌ Failed to execute scan - check error details above")


def display_scan_summary(results_df: pd.DataFrame) -> None:
    """Display scan summary with Pure MPI Expansion statistics"""
    create_section_header("📊 Scan Summary with Pure MPI Expansion Analysis", "")

    total_stocks = len(results_df)

    higher_h_count = len(results_df[results_df['Higher_H'] == 1])
    higher_hl_count = len(results_df[results_df['Higher_HL'] == 1])
    hh_only_count = len(results_df[(results_df['Higher_H'] == 1) & (results_df['Higher_HL'] == 0)])

    valid_crt_count = len(results_df[results_df['Valid_CRT'] == 1])

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

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Analyzed", total_stocks)
    with col2:
        st.metric("Higher H", higher_h_count, delta=f"{higher_h_count/total_stocks*100:.1f}%")
    with col3:
        st.metric("HHL", higher_hl_count, delta="Both H & L")
    with col4:
        st.metric("Valid CRT", valid_crt_count)
    with col5:
        st.metric("🚀 Strong Exp", strong_expansion, delta="≥5% velocity")
    with col6:
        st.metric("📈 Expanding", strong_expansion + expanding, delta=">0% velocity")

    if len(results_df) > 0:
        analysis_dates = results_df['Analysis_Date'].unique()
        if len(analysis_dates) == 1:
            create_info_box(f"📅 Analysis date: **{analysis_dates[0]}** | Higher H: **{higher_h_count}** (HHL: **{higher_hl_count}**, HH only: **{hh_only_count}**) | Avg MPI: **{avg_mpi:.1%}** | Avg Velocity: **{avg_velocity:+.1%}**")
        else:
            create_info_box(f"📅 Analysis dates: **{', '.join(analysis_dates)}** | Higher H: **{higher_h_count}** | Avg MPI: **{avg_mpi:.1%}** | Avg Velocity: **{avg_velocity:+.1%}**")

    # Analyst report statistics
    if 'sentiment_score' in results_df.columns:
        with_reports = results_df['sentiment_score'].notna().sum()
        if with_reports > 0:
            avg_sentiment = results_df[results_df['sentiment_score'].notna()]['sentiment_score'].mean()
            positive = (results_df['sentiment_label'] == 'positive').sum()
            negative = (results_df['sentiment_label'] == 'negative').sum()
            neutral = (results_df['sentiment_label'] == 'neutral').sum()

            st.markdown("#### 📊 Analyst Reports Coverage")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Stocks with Reports", with_reports, delta=f"{with_reports/total_stocks*100:.0f}% coverage")
            with col2:
                st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", delta="Range: -1 to +1")
            with col3:
                st.metric("📈 Positive", positive, delta="Bullish")
            with col4:
                st.metric("➖ Neutral", neutral, delta="Neutral")
            with col5:
                st.metric("📉 Negative", negative, delta="Bearish")

    # Earnings report statistics
    if 'revenue' in results_df.columns:
        with_earnings = results_df['revenue'].notna().sum()
        if with_earnings > 0:
            positive_guidance = (results_df['guidance_tone'] == 'positive').sum()
            negative_guidance = (results_df['guidance_tone'] == 'negative').sum()
            neutral_guidance = (results_df['guidance_tone'] == 'neutral').sum()

            avg_age = results_df[results_df['report_age_days'].notna()]['report_age_days'].mean()

            st.markdown("#### 💰 Earnings Reports Coverage")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Stocks with Earnings", with_earnings, delta=f"{with_earnings/total_stocks*100:.0f}% coverage")
            with col2:
                st.metric("Avg Report Age", f"{avg_age:.0f} days" if not pd.isna(avg_age) else "N/A")
            with col3:
                st.metric("📈 Positive", positive_guidance, delta="Bullish")
            with col4:
                st.metric("➖ Neutral", neutral_guidance, delta="Neutral")
            with col5:
                st.metric("📉 Negative", negative_guidance, delta="Bearish")


def show_base_pattern_filter(results_df: pd.DataFrame) -> pd.DataFrame:
    """Show base pattern filter"""
    create_section_header("🎯 Pattern Analysis", "")

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

    st.session_state.base_filter_type = selected_base_filter

    if selected_base_filter == "Valid CRT Only":
        base_stocks = results_df[results_df['Valid_CRT'] == 1].copy()
        create_info_box(f"Showing {len(base_stocks)} stocks with Valid CRT (Monday range expansion)")
    else:
        base_stocks = results_df.copy()
        create_info_box(f"Showing all {len(base_stocks)} scanned stocks")

    return base_stocks


def display_filtered_results(filtered_stocks: pd.DataFrame, selected_base_filter: str) -> None:
    """Display the filtered results table and export options"""
    create_section_header(f"📋 Filtered Results ({len(filtered_stocks)} stocks)", "")

    if len(filtered_stocks) == 0:
        create_warning_box("No stocks match the current filter criteria")
        return

    # Display columns already exist from scan - just use them
    display_cols = ['Analysis_Date', 'Ticker', 'Name', 'HL_Pattern',
                    'VW_Range_Velocity', 'IBS', 'Relative_Volume',
                    'MPI_Trend_Emoji', 'MPI_Visual']

    # Add analyst columns if available
    if 'Sentiment_Display' in filtered_stocks.columns:
        display_cols.extend(['Sentiment_Display', 'Report_Date_Display', 'Report_Count_Display'])

    # Add earnings columns if available
    if 'Earnings_Period' in filtered_stocks.columns:
        display_cols.extend(['Earnings_Period', 'Guidance_Display', 'Rev_YoY_Display', 'EPS_DPU_Display'])

    base_column_config = {
        'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
        'Name': st.column_config.TextColumn('Company Name', width='medium'),
        'HL_Pattern': st.column_config.TextColumn('H/L', width='small', help='HHL=Both H&L, HH=Higher H only, -=Neither'),
        'VW_Range_Velocity': st.column_config.NumberColumn('Range Vel', format='%+.4f', help='Daily range expansion velocity'),
        'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
        'Relative_Volume': st.column_config.NumberColumn('Rel Vol', format='%.1f%%', help='Relative Volume vs 14-day average'),
        'MPI_Trend_Emoji': st.column_config.TextColumn('📊', width='small', help='MPI Expansion Trend'),
        'MPI_Visual': st.column_config.TextColumn('MPI Visual', width='medium', help='Visual MPI representation'),
        'Sentiment_Display': st.column_config.TextColumn('Sentiment', width='small', help='Analyst sentiment score'),
        'Report_Date_Display': st.column_config.TextColumn('Report', width='small', help='Report date'),
        'Report_Count_Display': st.column_config.TextColumn('Reports', width='small', help='Number of reports'),
        'Earnings_Period': st.column_config.TextColumn('Period', width='small', help='Earnings period (Q1/Q2/FY etc.)'),
        'Guidance_Display': st.column_config.TextColumn('Guidance', width='small', help='Management guidance tone'),
        'Rev_YoY_Display': st.column_config.TextColumn('Rev YoY', width='small', help='Revenue year-over-year change'),
        'EPS_DPU_Display': st.column_config.TextColumn('EPS/DPU', width='small', help='EPS or DPU year-over-year change')
    }

    column_config = create_dynamic_column_config(filtered_stocks, display_cols, base_column_config)
    display_cols = [col for col in display_cols if col in filtered_stocks.columns]

    try:
        st.dataframe(
            filtered_stocks[display_cols],
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )

        show_tradingview_export(filtered_stocks, selected_base_filter)

    except Exception as e:
        create_error_box(f"❌ Error displaying filtered results: {str(e)}")
        handle_error(e, "FilteredResultsDisplay")


def show_tradingview_export(filtered_stocks: pd.DataFrame, selected_base_filter: str) -> None:
    """Show TradingView export section"""
    create_section_header("📋 TradingView Export (Filtered)", "")

    tv_tickers = [f"SGX:{ticker.replace('.SG', '')}" for ticker in filtered_stocks['Ticker'].tolist()]
    tv_string = ','.join(tv_tickers)

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

    csv_data = filtered_stocks.to_csv(index=False)
    filename_prefix = selected_base_filter.lower().replace(' ', '_').replace('+', 'and')
    create_download_button(
        csv_data,
        f"mpi_expansion_{filename_prefix}_{len(filtered_stocks)}_stocks.csv",
        "📥 Download MPI Expansion Data (CSV)",
        "text/csv",
        "Download complete breakout analysis results with MPI expansion data"
    )


def show_full_results_table(results_df: pd.DataFrame) -> None:
    """Show the full results table in an expander"""
    with st.expander("📋 Full Analysis Results", expanded=False):
        try:
            full_results_cols = [
                'Analysis_Date', 'Ticker', 'Name', 'Close',
                'CRT_High', 'CRT_Low', 'HL_Pattern',
                'VW_Range_Velocity', 'IBS', 'Valid_CRT',
                'MPI_Trend_Emoji', 'MPI_Trend', 'MPI', 'MPI_Velocity', 'MPI_Visual'
            ]

            if 'Sentiment_Display' in results_df.columns:
                full_results_cols.extend(['Sentiment_Display', 'Report_Date_Display', 'Report_Count_Display'])

            if 'Earnings_Period' in results_df.columns:
                full_results_cols.extend(['Earnings_Period', 'Guidance_Display', 'Rev_YoY_Display', 'EPS_DPU_Display'])

            base_full_results_config = {
                'Analysis_Date': st.column_config.TextColumn('Date', width='small'),
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Name': st.column_config.TextColumn('Company Name', width='medium'),
                'HL_Pattern': st.column_config.TextColumn('H/L', width='small'),
                'VW_Range_Velocity': st.column_config.NumberColumn('Range Vel', format='%+.4f'),
                'IBS': st.column_config.NumberColumn('IBS', format='%.3f'),
                'Valid_CRT': st.column_config.NumberColumn('CRT', width='small'),
                'MPI_Trend_Emoji': st.column_config.TextColumn('📊', width='small'),
                'MPI_Trend': st.column_config.TextColumn('MPI Trend', width='medium'),
                'MPI': st.column_config.NumberColumn('MPI', format='%.1%'),
                'MPI_Velocity': st.column_config.NumberColumn('MPI Velocity', format='%+.1%'),
                'MPI_Visual': st.column_config.TextColumn('MPI Visual', width='medium'),
                'Sentiment_Display': st.column_config.TextColumn('Sentiment', width='small'),
                'Report_Date_Display': st.column_config.TextColumn('Report', width='small'),
                'Report_Count_Display': st.column_config.TextColumn('Reports', width='small'),
                'Earnings_Period': st.column_config.TextColumn('Period', width='small'),
                'Guidance_Display': st.column_config.TextColumn('Guidance', width='small'),
                'Rev_YoY_Display': st.column_config.TextColumn('Rev YoY', width='small'),
                'EPS_DPU_Display': st.column_config.TextColumn('EPS/DPU', width='small')
            }

            full_results_column_config = create_dynamic_column_config(results_df, full_results_cols, base_full_results_config)
            full_results_cols = [col for col in full_results_cols if col in results_df.columns]

            st.dataframe(
                results_df[full_results_cols],
                column_config=full_results_column_config,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            create_error_box(f"❌ Error displaying full results: {str(e)}")
            handle_error(e, "FullResultsDisplay")


def show_mpi_insights(results_df: pd.DataFrame) -> None:
    """Show MPI Insights insights"""
    if 'MPI_Trend' not in results_df.columns:
        return

    with st.expander("📈 MPI Insights Insights", expanded=False):
        trend_summary = []
        trend_counts = results_df['MPI_Trend'].value_counts()

        for trend in ['Strong Expansion', 'Expanding', 'Flat', 'Mild Contraction', 'Strong Contraction']:
            if trend in trend_counts.index:
                count = trend_counts[trend]
                trend_stocks = results_df[results_df['MPI_Trend'] == trend]
                avg_mpi = trend_stocks['MPI'].mean()
                avg_velocity = trend_stocks['MPI_Velocity'].mean()

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


def display_detailed_analyst_reports(results_df: pd.DataFrame) -> None:
    """Display detailed analyst reports with dropdown selection"""
    if results_df.empty:
        return

    # Check if we have analyst report data
    if 'sentiment_score' not in results_df.columns:
        st.info("📊 No analyst reports available in current scan results")
        return

    # Get stocks with analyst reports
    stocks_with_reports = results_df[results_df['sentiment_score'].notna()].copy()

    if stocks_with_reports.empty:
        st.info("📊 No stocks with analyst reports in current scan results")
        return

    create_section_header("📊 Analyst Reports Analysis", "")

    # Create dropdown options
    report_options = []
    for _, row in stocks_with_reports.iterrows():
        ticker = row['Ticker']
        name = row.get('Name', ticker)
        sentiment = row.get('sentiment_label', 'unknown')
        sentiment_emoji = analyst_reports.format_sentiment_emoji(sentiment)
        report_date = row.get('report_date', 'Unknown')
        if isinstance(report_date, str):
            report_date = report_date[:10]  # YYYY-MM-DD format

        option_text = f"{ticker} - {name} {sentiment_emoji} ({report_date})"
        report_options.append((option_text, ticker))

    # Sort by ticker for consistency
    report_options.sort(key=lambda x: x[1])

    # Dropdown selection
    selected_option = st.selectbox(
        "📋 Select stock to view detailed analyst report:",
        options=[opt[0] for opt in report_options],
        help="Choose a stock to view its detailed analyst report information"
    )

    # Find selected stock data
    selected_ticker = next(opt[1] for opt in report_options if opt[0] == selected_option)
    stock_data = stocks_with_reports[stocks_with_reports['Ticker'] == selected_ticker].iloc[0]

    # Display detailed analyst report
    display_analyst_report_details(stock_data)


def display_analyst_report_details(stock_data: pd.Series) -> None:
    """Display detailed analyst report information for a selected stock"""
    ticker = stock_data['Ticker']
    name = stock_data.get('Name', ticker)

    st.markdown(f"### 📊 {ticker} - {name}")

    # Basic report info
    col1, col2, col3 = st.columns(3)
    with col1:
        sentiment = stock_data.get('sentiment_label', 'unknown')
        sentiment_emoji = analyst_reports.format_sentiment_emoji(sentiment)
        st.metric("Sentiment", f"{sentiment_emoji} {sentiment.title()}")
    with col2:
        score = stock_data.get('sentiment_score', 0)
        st.metric("Sentiment Score", f"{score:.2f}", help="Range: -1 (bearish) to +1 (bullish)")
    with col3:
        recommendation = stock_data.get('recommendation', 'N/A')
        st.metric("Recommendation", recommendation)

    # Report details
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Report Information:**")
        report_date = stock_data.get('report_date')
        if report_date:
            if isinstance(report_date, str):
                report_date = report_date[:10]
            st.write(f"📅 **Date:** {report_date}")
        else:
            st.write("📅 **Date:** Unknown")

        analyst_firm = stock_data.get('analyst_firm', 'Unknown')
        st.write(f"🏢 **Analyst Firm:** {analyst_firm}")

        report_count = stock_data.get('report_count', 1)
        st.write(f"📄 **Total Reports:** {report_count}")

    with col2:
        st.markdown("**Price Information:**")
        price_target = stock_data.get('price_target')
        if price_target:
            st.write(f"🎯 **Price Target:** ${price_target:.2f}")
        else:
            st.write("🎯 **Price Target:** N/A")

        price_at_report = stock_data.get('price_at_report')
        if price_at_report:
            st.write(f"💰 **Price at Report:** ${price_at_report:.2f}")
        else:
            st.write("💰 **Price at Report:** N/A")

        upside_pct = stock_data.get('upside_pct')
        if upside_pct is not None:
            color = "green" if upside_pct > 0 else "red"
            st.write(f"📈 **Upside:** {upside_pct:+.1f}%")

    # Catalysts and Risks
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Key Catalysts")
        catalysts = stock_data.get('key_catalysts', [])
        if catalysts:
            for catalyst in catalysts:
                st.write(f"• {catalyst}")
        else:
            st.info("No catalysts specified")

    with col2:
        st.markdown("### ⚠️ Key Risks")
        risks = stock_data.get('key_risks', [])
        if risks:
            for risk in risks:
                st.write(f"• {risk}")
        else:
            st.info("No risks specified")

    # Executive Summary
    st.markdown("---")
    st.markdown("### 📝 Executive Summary")
    summary = stock_data.get('executive_summary', '')
    if summary:
        st.write(summary)
    else:
        st.info("No executive summary available")

    # Report History (if multiple reports)
    if stock_data.get('report_count', 1) > 1:
        st.markdown("---")
        st.markdown("### 📚 Report History")

        try:
            from utils.analyst_reports import get_report_history
            all_reports, _ = analyst_reports.get_cached_reports()
            history_df = get_report_history(all_reports, stock_data['ticker_sgx'])

            if not history_df.empty and len(history_df) > 1:
                # Show sentiment trend
                trend = analyst_reports.get_sentiment_trend_description(history_df)
                st.info(f"📈 **Sentiment Trend:** {trend}")

                # Show historical reports table
                history_display = history_df[['report_date', 'sentiment_label', 'recommendation', 'price_target']].copy()
                history_display['sentiment_label'] = history_display['sentiment_label'].apply(
                    lambda x: f"{analyst_reports.format_sentiment_emoji(x)} {x.title()}"
                )
                history_display['report_date'] = history_display['report_date'].dt.strftime('%Y-%m-%d')

                st.dataframe(
                    history_display,
                    column_config={
                        'report_date': st.column_config.TextColumn('Date', width='small'),
                        'sentiment_label': st.column_config.TextColumn('Sentiment', width='medium'),
                        'recommendation': st.column_config.TextColumn('Recommendation', width='medium'),
                        'price_target': st.column_config.NumberColumn('Price Target', format='$%.2f')
                    },
                    hide_index=True,
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"Could not load report history: {e}")


def display_detailed_earnings_reports(results_df: pd.DataFrame) -> None:
    """Display detailed earnings reports with dropdown selection"""
    if results_df.empty:
        return

    # Check if we have earnings report data
    earnings_columns = ['revenue', 'net_profit', 'guidance_tone']
    has_earnings = any(col in results_df.columns for col in earnings_columns)

    if not has_earnings:
        st.info("💰 No earnings reports available in current scan results")
        return

    # Get stocks with earnings reports
    earnings_mask = results_df['revenue'].notna() | results_df['net_profit'].notna()
    stocks_with_earnings = results_df[earnings_mask].copy()

    if stocks_with_earnings.empty:
        st.info("💰 No stocks with earnings reports in current scan results")
        return

    create_section_header("💰 Earnings Reports Analysis", "")

    # Create dropdown options
    earnings_options = []
    for _, row in stocks_with_earnings.iterrows():
        ticker = row['Ticker']
        name = row.get('Name', ticker)
        period = row.get('Earnings_Period', 'Unknown')
        guidance = row.get('guidance_tone', 'neutral')

        guidance_emoji = {
            'positive': '📈',
            'neutral': '➖',
            'negative': '📉'
        }.get(guidance, '❓')

        option_text = f"{ticker} - {name} ({period}) {guidance_emoji}"
        earnings_options.append((option_text, ticker))

    # Sort by ticker for consistency
    earnings_options.sort(key=lambda x: x[1])

    # Dropdown selection
    selected_option = st.selectbox(
        "📋 Select stock to view detailed earnings report:",
        options=[opt[0] for opt in earnings_options],
        help="Choose a stock to view its detailed earnings report information"
    )

    # Find selected stock data
    selected_ticker = next(opt[1] for opt in earnings_options if opt[0] == selected_option)
    stock_data = stocks_with_earnings[stocks_with_earnings['Ticker'] == selected_ticker].iloc[0]

    # Display detailed earnings report
    display_earnings_report_details(stock_data)


def display_earnings_report_details(stock_data: pd.Series) -> None:
    """Display detailed earnings report information for a selected stock"""
    ticker = stock_data['Ticker']
    name = stock_data.get('Name', ticker)

    st.markdown(f"### 💰 {ticker} - {name}")

    # Basic earnings info
    col1, col2, col3 = st.columns(3)
    with col1:
        period = stock_data.get('Earnings_Period', 'Unknown')
        st.metric("Period", period)
    with col2:
        guidance = stock_data.get('guidance_tone', 'neutral')
        guidance_emoji = {
            'positive': '📈',
            'neutral': '➖',
            'negative': '📉'
        }.get(guidance, '❓')
        st.metric("Guidance", f"{guidance_emoji} {guidance.title()}")
    with col3:
        report_age = stock_data.get('report_age_days', None)
        if report_age is not None:
            age_str = "days" if report_age != 1 else "day"
            st.metric("Report Age", f"{report_age} {age_str}")
        else:
            st.metric("Report Age", "Unknown")

    # Financial Results
    st.markdown("### 💵 Financial Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        revenue = stock_data.get('revenue')
        if revenue is not None:
            st.metric("Revenue", f"${revenue:,.0f}")
        else:
            st.metric("Revenue", "N/A")

    with col2:
        net_profit = stock_data.get('net_profit')
        if net_profit is not None:
            st.metric("Net Profit", f"${net_profit:,.0f}")
        else:
            st.metric("Net Profit", "N/A")

    with col3:
        eps = stock_data.get('eps')
        if eps is not None:
            st.metric("EPS", f"${eps:.3f}")
        else:
            st.metric("EPS", "N/A")

    # Year-over-Year Changes
    st.markdown("### 📊 Year-over-Year Changes")

    col1, col2, col3 = st.columns(3)
    with col1:
        rev_yoy = stock_data.get('revenue_yoy_pct')
        if rev_yoy is not None:
            color = "green" if rev_yoy > 0 else "red"
            st.metric("Revenue YoY", f"{rev_yoy:+.1f}%")
        else:
            st.metric("Revenue YoY", "N/A")

    with col2:
        profit_yoy = stock_data.get('profit_yoy_pct')
        if profit_yoy is not None:
            color = "green" if profit_yoy > 0 else "red"
            st.metric("Profit YoY", f"{profit_yoy:+.1f}%")
        else:
            st.metric("Profit YoY", "N/A")

    with col3:
        eps_yoy = stock_data.get('eps_yoy_pct')
        if eps_yoy is not None:
            color = "green" if eps_yoy > 0 else "red"
            st.metric("EPS YoY", f"{eps_yoy:+.1f}%")
        else:
            st.metric("EPS YoY", "N/A")

    # Management Commentary
    st.markdown("---")
    st.markdown("### 💬 Management Commentary")

    commentary = stock_data.get('management_commentary', '')
    if commentary:
        st.write(commentary)
    else:
        st.info("No management commentary available")

    # Future Guidance
    st.markdown("---")
    st.markdown("### 🔮 Future Guidance")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Revenue Guidance:**")
        rev_guidance = stock_data.get('revenue_guidance', '')
        if rev_guidance:
            st.write(rev_guidance)
        else:
            st.info("No revenue guidance provided")

    with col2:
        st.markdown("**Profit Guidance:**")
        profit_guidance = stock_data.get('profit_guidance', '')
        if profit_guidance:
            st.write(profit_guidance)
        else:
            st.info("No profit guidance provided")

    # Key Points
    st.markdown("---")
    st.markdown("### 🎯 Key Points")

    key_points = stock_data.get('key_points', [])
    if key_points:
        for point in key_points:
            st.write(f"• {point}")
    else:
        st.info("No key points specified")


def show_force_update_options():
    """Show force update options when data is up to date"""
    try:
        from core.local_file_loader import get_local_loader
        from datetime import date

        loader = get_local_loader()
        latest_eod = loader.get_latest_eod_file()
        current_working_day = date.today()

        if latest_eod:
            eod_date_str = latest_eod.replace('.csv', '')
            eod_date_display = datetime.strptime(eod_date_str, '%d_%b_%Y')

            # Show force update options
            with st.expander("⚙️ Advanced: Force Update Options", expanded=False):
                st.markdown("### 🔄 Force Update from EOD File")
                st.warning("⚠️ **Force EOD Update** will re-process the latest EOD file even though it's already been imported.")
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

                if st.button("🔄 Force Update from EOD", type="secondary", use_container_width=True, key="force_eod_button"):
                    result = perform_eod_update(loader, force=True)
                    if result:
                        st.success("Force EOD update completed!")
                        time.sleep(2)
                        st.rerun()
                    return result

                # Add force EOD update even when no new EOD file exists
                st.markdown("---")
                st.markdown("### 🔄 Force Update Any EOD File")
                st.info("Choose any EOD file to force re-process, even if it's already been imported.")
                eod_files = loader.list_eod_files()
                if eod_files:
                    selected_eod = st.selectbox(
                        "Select EOD file to force update:",
                        options=eod_files,
                        key="force_eod_select"
                    )
                    if st.button("🔄 Force Update Selected EOD", type="secondary", use_container_width=True, key="force_selected_eod_button"):
                        # Import the function that handles specific file updates
                        from core.local_file_loader import LocalFileLoader
                        # Create a temporary loader instance to handle specific file
                        temp_loader = LocalFileLoader()
                        # For now, we'll use the force update on latest file
                        # TODO: Implement specific file force update in LocalFileLoader
                        result = perform_eod_update(loader, force=True)
                        if result:
                            st.success(f"Force update completed!")
                            time.sleep(2)
                            st.rerun()
                        return result
                else:
                    st.info("No EOD files available")

                st.markdown("---")
                st.markdown("### 📥 Force Download from External Source")
                st.warning("⚠️ **Force Download** will download data regardless of current state.")
                st.markdown("""
                **Use this when:**
                - You want to update to today's data without waiting for EOD file
                - You need to re-download data to fix issues
                - You want the absolute latest available data

                **This will:**
                - Calculate date range: (last EOD date + 1 day) to (current working day)
                - Download data from all stocks
                - Overwrite any existing dates in this range
                - Use abbreviated volume format (÷1000)
                """)

                # Calculate download range based on EOD file date (not Historical_Data date)
                if latest_eod and current_working_day:
                    # Parse date from EOD filename (e.g., "10_Oct_2025.csv" → 10/10/2025)
                    try:
                        eod_date_str = latest_eod.replace('.csv', '')
                        eod_date_obj = datetime.strptime(eod_date_str, '%d_%b_%Y').date()

                        # Force download range: (EOD date + 1) to (current working day)
                        force_start_date = eod_date_obj + timedelta(days=1)
                        force_end_date = current_working_day
                    except Exception as e:
                        st.error(f"Could not parse EOD date: {e}")
                        force_start_date = None
                        force_end_date = None

                    # Check if there's actually a range to download
                    if force_start_date and force_end_date and force_start_date <= force_end_date:
                        gap_days = (force_end_date - force_start_date).days + 1

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Last EOD Date", eod_date_obj.strftime('%d/%m/%Y'))
                        with col_b:
                            st.metric("Download Range", f"{gap_days} day(s)")
                        with col_c:
                            st.metric("End Date", force_end_date.strftime('%d/%m/%Y'))

                        st.info(f"📥 Will download: **{force_start_date.strftime('%d/%m/%Y')}** to **{force_end_date.strftime('%d/%m/%Y')}**")

                        if st.button("📥 Force Download Latest Data", type="secondary", use_container_width=True, key="force_download_button"):
                            result = perform_yfinance_download(loader, force_start_date, force_end_date, force_mode=True)
                            if result:
                                st.success("Force download completed!")
                                time.sleep(2)
                                st.rerun()
                            return result
                    elif force_start_date and force_end_date:
                        st.info("✅ No date range to download - Historical data is current to working day")
                    else:
                        st.error("Cannot determine date range for force download")

    except Exception as e:
        handle_error(e, "ForceUpdateOptions")
        st.error(f"❌ Error showing force update options: {str(e)}")
