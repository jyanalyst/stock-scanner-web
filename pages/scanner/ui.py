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
from pages.scanner.ui_flow_analysis import show_institutional_flow_analysis


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

                if st.button("🔄 Update from EOD File", type="primary", width="stretch", key="eod_update_btn"):
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

                if st.button("📥 Download Missing Data", type="secondary", width='stretch', key="download_missing_btn"):
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
            if st.button("⏭️ Skip Updates", width='stretch', key="skip_both_btn"):
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
                if st.button("🔄 Update Now", type="primary", width='stretch', key="eod_only_update_btn"):
                    result = perform_eod_update(loader, force=False)
                    if result:
                        st.rerun()
                    return result

            with col_skip:
                if st.button("⏭️ Skip This Time", width='stretch', key="eod_only_skip_btn"):
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
                if st.button("📥 Download Missing Data", type="primary", width='stretch', key="gap_only_download_btn"):
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
                if st.button("⏭️ Skip for Now", width='stretch', key="gap_only_skip_btn"):
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

                    if st.button("🔄 Force Update from EOD", type="secondary", width='stretch', key="force_eod_button"):
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
                        if st.button("🔄 Force Update Selected EOD", type="secondary", width='stretch', key="force_selected_eod_button"):
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

                            if st.button("📥 Force Download Latest Data", type="secondary", width='stretch', key="force_download_button"):
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


def show_confirmation_filters() -> Tuple[bool, bool, bool, str, float, float, float]:
    """Display confirmation filter settings (matching Pine Script)"""
    with st.expander("🎯 Confirmation Filters (Optional)", expanded=False):
        st.info("⚙️ By default, ALL reversal signals are shown. Enable filters below to require specific confirmations.")

        # Checkboxes for which metrics to use
        col1, col2, col3 = st.columns(3)
        with col1:
            use_ibs = st.checkbox("Use IBS Confirmation", value=False,
                help="Require IBS acceleration to meet threshold")
        with col2:
            use_rvol = st.checkbox("Use RVol Confirmation", value=False,
                help="Require RVol acceleration to meet threshold")
        with col3:
            use_rrange = st.checkbox("Use RRange Confirmation", value=False,
                help="Require RRange acceleration to meet threshold")

        # AND/OR logic (only show if at least one is checked)
        if use_ibs or use_rvol or use_rrange:
            confirmation_logic = st.radio(
                "Confirmation Logic:",
                options=["OR", "AND"],
                index=0,  # Default OR
                horizontal=True,
                help="OR = Any selected metric can pass | AND = ALL selected must pass"
            )
        else:
            confirmation_logic = "OR"  # Default when nothing selected

        # Threshold inputs (only show when relevant)
        if use_ibs or use_rvol or use_rrange:
            st.markdown("**Thresholds:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                ibs_threshold = st.number_input("IBS", value=0.10, step=0.01, format="%.2f", key="ibs_threshold_filter") if use_ibs else 0.10
            with col2:
                rvol_threshold = st.number_input("RVol", value=0.20, step=0.01, format="%.2f", key="rvol_threshold_filter") if use_rvol else 0.20
            with col3:
                rrange_threshold = st.number_input("RRange", value=0.30, step=0.01, format="%.2f", key="rrange_threshold_filter") if use_rrange else 0.30
        else:
            ibs_threshold = 0.10
            rvol_threshold = 0.20
            rrange_threshold = 0.30

    return (use_ibs, use_rvol, use_rrange, confirmation_logic,
            ibs_threshold, rvol_threshold, rrange_threshold)


def execute_scan_button(scan_scope: str, selected_stock: Optional[str],
                       scan_date_type: str, historical_date: Optional[date],
                       days_back: int, rolling_window: int,
                       # NEW: Confirmation filter parameters
                       use_ibs: bool = False, use_rvol: bool = False, use_rrange: bool = False,
                       confirmation_logic: str = "OR",
                       ibs_threshold: float = 0.10, rvol_threshold: float = 0.20, rrange_threshold: float = 0.30) -> None:
    """Handle scan execution button and logic"""
    if st.button("🚀 Execute Scan", type="primary", width='stretch'):
        # Initialize error logger if not exists
        if 'error_logger' not in st.session_state:
            from pages.common.error_handler import ErrorLogger
            st.session_state.error_logger = ErrorLogger()

        # Store confirmation filter settings in session state for later use
        st.session_state.use_ibs = use_ibs
        st.session_state.use_rvol = use_rvol
        st.session_state.use_rrange = use_rrange
        st.session_state.confirmation_logic = confirmation_logic
        st.session_state.ibs_threshold = ibs_threshold
        st.session_state.rvol_threshold = rvol_threshold
        st.session_state.rrange_threshold = rrange_threshold

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
                rolling_window=rolling_window,
                # NEW: Pass confirmation parameters
                use_ibs=use_ibs,
                use_rvol=use_rvol,
                use_rrange=use_rrange,
                confirmation_logic=confirmation_logic,
                ibs_threshold=ibs_threshold,
                rvol_threshold=rvol_threshold,
                rrange_threshold=rrange_threshold
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

    # Count Signal_Bias
    bullish_count = len(results_df[results_df['Signal_Bias'] == '🟢 BULLISH']) if 'Signal_Bias' in results_df.columns else 0
    bearish_count = len(results_df[results_df['Signal_Bias'] == '🔴 BEARISH']) if 'Signal_Bias' in results_df.columns else 0

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
        st.metric("🟢 Bullish", bullish_count)
    with col5:
        st.metric("🔴 Bearish", bearish_count)
    with col6:
        st.metric(" Strong Exp", strong_expansion, delta="≥5% velocity")
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

    # ===== PHASE 1: INSTITUTIONAL FLOW METRICS SUMMARY =====
    if 'Flow_10D' in results_df.columns:
        st.markdown("---")
        create_section_header("🏦 Institutional Flow Analysis", "Phase 1: Enhanced scanner with institutional order flow metrics")

        # Flow regime distribution
        if 'Flow_Regime' in results_df.columns:
            flow_regimes = results_df['Flow_Regime'].value_counts()
            st.markdown("#### 🌊 Flow Regime Distribution")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                strong_acc = flow_regimes.get('Strong Accumulation', 0)
                st.metric("🔥 Strong Acc", strong_acc, delta=f"{strong_acc/total_stocks*100:.1f}%")
            with col2:
                acc = flow_regimes.get('Accumulation', 0)
                st.metric("📈 Accumulation", acc, delta=f"{acc/total_stocks*100:.1f}%")
            with col3:
                neutral = flow_regimes.get('Neutral', 0)
                st.metric("➖ Neutral", neutral, delta=f"{neutral/total_stocks*100:.1f}%")
            with col4:
                dist = flow_regimes.get('Distribution', 0)
                st.metric("📉 Distribution", dist, delta=f"{dist/total_stocks*100:.1f}%")
            with col5:
                strong_dist = flow_regimes.get('Strong Distribution', 0)
                st.metric("💥 Strong Dist", strong_dist, delta=f"{strong_dist/total_stocks*100:.1f}%")

        # Conviction and divergence metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_conviction = results_df['Volume_Conviction'].mean()
            st.metric("🎯 Avg Conviction", f"{avg_conviction:.3f}", help="Volume ratio: up-day vs down-day volume (1.0=neutral)")
        with col2:
            avg_flow = results_df['Flow_10D'].mean()
            st.metric("🌊 Avg Flow 10D", f"{avg_flow:+.2f}", help="10-day cumulative institutional flow")
        with col3:
            avg_divergence = results_df['Divergence_Severity'].mean()
            st.metric("📊 Avg Divergence", f"{avg_divergence:.1f}", help="Price-flow divergence severity (0-100)")

        # Flow velocity analysis
        if 'Flow_Velocity' in results_df.columns:
            st.markdown("#### ⚡ Flow Velocity Analysis")
            flow_vel_positive = (results_df['Flow_Velocity'] > 0).sum()
            flow_vel_negative = (results_df['Flow_Velocity'] < 0).sum()
            flow_vel_neutral = (results_df['Flow_Velocity'] == 0).sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Accelerating", flow_vel_positive, delta=f"{flow_vel_positive/total_stocks*100:.1f}%")
            with col2:
                st.metric("📉 Decelerating", flow_vel_negative, delta=f"{flow_vel_negative/total_stocks*100:.1f}%")
            with col3:
                st.metric("➖ Stable", flow_vel_neutral, delta=f"{flow_vel_neutral/total_stocks*100:.1f}%")

        # Divergence signals
        if 'Divergence_Gap' in results_df.columns:
            st.markdown("#### 📈 Price-Flow Divergence Signals")
            bullish_divergence = (results_df['Divergence_Gap'] < -10).sum()  # Price weak, flow strong
            bearish_divergence = (results_df['Divergence_Gap'] > 10).sum()   # Price strong, flow weak
            neutral_divergence = ((results_df['Divergence_Gap'] >= -10) & (results_df['Divergence_Gap'] <= 10)).sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 Bullish Divergence", bullish_divergence, help="Price weak, flow strong (potential reversal up)")
            with col2:
                st.metric("🔴 Bearish Divergence", bearish_divergence, help="Price strong, flow weak (potential reversal down)")
            with col3:
                st.metric("⚪ Neutral", neutral_divergence, help="Price and flow aligned")

        # Key insights
        st.markdown("#### 🎯 Institutional Flow Insights")
        insights = []

        # Flow regime bias
        total_accumulation = flow_regimes.get('Strong Accumulation', 0) + flow_regimes.get('Accumulation', 0)
        total_distribution = flow_regimes.get('Strong Distribution', 0) + flow_regimes.get('Distribution', 0)

        if total_accumulation > total_distribution:
            insights.append(f"📈 **Accumulation Bias**: {total_accumulation} stocks showing institutional buying vs {total_distribution} distribution")
        elif total_distribution > total_accumulation:
            insights.append(f"📉 **Distribution Bias**: {total_distribution} stocks showing institutional selling vs {total_accumulation} accumulation")

        # Conviction analysis
        high_conviction = (results_df['Volume_Conviction'] > 1.2).sum()
        low_conviction = (results_df['Volume_Conviction'] < 0.8).sum()

        if high_conviction > 0:
            insights.append(f"🎯 **High Conviction**: {high_conviction} stocks with strong up-day volume participation")

        # Divergence warnings
        severe_divergence = (results_df['Divergence_Severity'] > 50).sum()
        if severe_divergence > 0:
            insights.append(f"⚠️ **Severe Divergence**: {severe_divergence} stocks showing extreme price-flow misalignment")

        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.info("📊 Flow analysis complete - no significant institutional patterns detected")


def show_base_pattern_filter(results_df: pd.DataFrame) -> pd.DataFrame:
    """Show base pattern filter with confirmation settings"""
    create_section_header("🎯 Pattern Analysis & Filters", "")

    # ===== THREE-INDICATOR SYSTEM: ALL SIGNALS ARE QUALIFIED =====
    # No Pattern_Quality filtering needed - all signals from Three-Indicator System are valid
    all_signals = results_df.copy()

    # ===== CONFIRMATION FILTERING CONTROLS =====
    # Optional confirmation filters for additional signal validation
    st.markdown("**Confirmation Settings (Optional):**")
    st.info("⚙️ Use these optional filters to require additional confirmation beyond the Three-Indicator System.")

    # Checkboxes for which metrics to use
    col1, col2, col3 = st.columns(3)
    with col1:
        use_ibs = st.checkbox("Use IBS Confirmation", value=False,
            help="Require IBS acceleration to meet threshold")
    with col2:
        use_rvol = st.checkbox("Use RVol Confirmation", value=False,
            help="Require RVol acceleration to meet threshold")
    with col3:
        use_rrange = st.checkbox("Use RRange Confirmation", value=False,
            help="Require RRange acceleration to meet threshold")

    # AND/OR logic (only show if at least one is checked)
    if use_ibs or use_rvol or use_rrange:
        confirmation_logic = st.radio(
            "Confirmation Logic:",
            options=["OR", "AND"],
            index=0,  # Default OR
            horizontal=True,
            help="OR = Any selected metric can pass | AND = ALL selected must pass"
        )
    else:
        confirmation_logic = "OR"  # Default when nothing selected

    # Threshold inputs (only show when relevant)
    if use_ibs or use_rvol or use_rrange:
        st.markdown("**Thresholds:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            ibs_threshold = st.number_input("IBS", value=0.10, step=0.01, format="%.2f", key="ibs_threshold_filter") if use_ibs else 0.10
        with col2:
            rvol_threshold = st.number_input("RVol", value=0.20, step=0.01, format="%.2f", key="rvol_threshold_filter") if use_rvol else 0.20
        with col3:
            rrange_threshold = st.number_input("RRange", value=0.30, step=0.01, format="%.2f", key="rrange_threshold_filter") if use_rrange else 0.30
    else:
        ibs_threshold = 0.10
        rvol_threshold = 0.20
        rrange_threshold = 0.30

    # Show confirmation filter toggle
    show_confirmed_only = st.checkbox(
        "✅ Show Confirmed Signals Only",
        value=False,  # Default to showing all signals
        help="When enabled, only shows signals that passed the confirmation filters above. When disabled, shows all Three-Indicator signals."
    )

    # Apply confirmation filtering if enabled
    if show_confirmed_only and (use_ibs or use_rvol or use_rrange):
        # Calculate confirmation for each signal based on current settings
        confirmed_signals = all_signals.copy()

        # Build confirmation conditions based on user selections
        confirmation_conditions = []
        if use_ibs:
            if confirmation_logic == "AND":
                # For AND logic, IBS must meet threshold for bullish (positive accel) or bearish (negative accel)
                confirmed_signals['ibs_confirmed'] = (
                    ((confirmed_signals['Signal_Bias'] == '🟢 BULLISH') & (confirmed_signals['IBS_Accel'] >= ibs_threshold)) |
                    ((confirmed_signals['Signal_Bias'] == '🔴 BEARISH') & (confirmed_signals['IBS_Accel'] <= -ibs_threshold))
                )
            else:  # OR logic
                confirmed_signals['ibs_confirmed'] = (
                    ((confirmed_signals['Signal_Bias'] == '🟢 BULLISH') & (confirmed_signals['IBS_Accel'] >= ibs_threshold)) |
                    ((confirmed_signals['Signal_Bias'] == '🔴 BEARISH') & (confirmed_signals['IBS_Accel'] <= -ibs_threshold))
                )
            confirmation_conditions.append('ibs_confirmed')

        if use_rvol:
            confirmed_signals['rvol_confirmed'] = confirmed_signals['RVol_Accel'] >= rvol_threshold
            confirmation_conditions.append('rvol_confirmed')

        if use_rrange:
            confirmed_signals['rrange_confirmed'] = confirmed_signals['RRange_Accel'] >= rrange_threshold
            confirmation_conditions.append('rrange_confirmed')

        # Apply logic
        if confirmation_logic == "AND":
            # All selected conditions must be true
            confirmed_signals['is_confirmed_dynamic'] = confirmed_signals[confirmation_conditions].all(axis=1)
        else:  # OR
            # Any selected condition must be true
            confirmed_signals['is_confirmed_dynamic'] = confirmed_signals[confirmation_conditions].any(axis=1)

        # Filter to confirmed signals
        filtered_signals = confirmed_signals[confirmed_signals['is_confirmed_dynamic'] == True]
        st.info(f"🔍 **Confirmation Filters Active** - Showing {len(filtered_signals)} confirmed signals out of {len(confirmed_signals)} total signals")
    else:
        filtered_signals = all_signals
        st.info(f"📊 **All Three-Indicator Signals** - Showing {len(filtered_signals)} signals (confirmation filters bypassed)")

    # ===== SIGNAL BIAS FILTERING =====
    # Simplified filter options - signal-based only
    filter_options = [
        "All Signals",
        "🟢 Bullish Signals",
        "🔴 Bearish Signals"
    ]

    # Radio buttons with default to "All Signals" (index=0)
    selected_filter = st.radio(
        "Signal Filter:",
        options=filter_options,
        index=0,  # Default to "All Signals"
        help="Filter signals by bias (bullish/bearish) - all signals are from Three-Indicator System",
        horizontal=True
    )

    # Store in session state
    st.session_state.base_filter_type = selected_filter

    # Apply signal bias filtering based on selection
    if selected_filter == "All Signals":
        final_filtered_df = filtered_signals
        create_info_box(f"Showing all {len(final_filtered_df)} signals from Three-Indicator System")
    elif selected_filter == "🟢 Bullish Signals":
        final_filtered_df = filtered_signals[filtered_signals['Signal_Bias'] == '🟢 BULLISH']
        create_info_box(f"Showing {len(final_filtered_df)} bullish signals")
    elif selected_filter == "🔴 Bearish Signals":
        final_filtered_df = filtered_signals[filtered_signals['Signal_Bias'] == '🔴 BEARISH']
        create_info_box(f"Showing {len(final_filtered_df)} bearish signals")
    else:
        final_filtered_df = filtered_signals
        create_info_box(f"Showing all {len(final_filtered_df)} signals")

    return final_filtered_df


def display_filtered_results(filtered_stocks: pd.DataFrame, selected_base_filter: str) -> None:
    """Display the filtered results table and export options"""
    create_section_header(f"📋 Filtered Results ({len(filtered_stocks)} stocks)", "")

    if len(filtered_stocks) == 0:
        create_warning_box("No stocks match the current filter criteria")
        return

    # Display columns - Institutional Flow Focus (16 columns)
    display_cols = [
        # Core Identity (4)
        'Analysis_Date', 'Ticker', 'Name', 'Signal_Bias',

        # Institutional Flow (5)
        'Daily_Flow', 'Flow_10D', 'Flow_Percentile', 'Flow_Velocity', 'Flow_Velocity_Percentile',

        # Technical Indicators (7)
        'VPI_Percentile', 'IBS_Percentile', 'MPI_Percentile', 'HL_Pattern', 'IBS', 'VW_Range_Velocity', 'Relative_Volume'
    ]

    base_column_config = {
        # Core Identity (4)
        'Analysis_Date': st.column_config.TextColumn('📅 Date', width='small', help='Analysis date for the scan'),
        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
        'Name': st.column_config.TextColumn('Company', width='medium'),
        'Signal_Bias': st.column_config.TextColumn('🎯 Signal', width='small', help='🟢 BULLISH / 🔴 BEARISH'),

        # Institutional Flow (5)
        'Daily_Flow': st.column_config.NumberColumn('Daily Flow', format='%+.2f', help='Institutional flow for current day'),
        'Flow_10D': st.column_config.NumberColumn('Flow 10D', format='%+.2f', help='10-day cumulative institutional flow'),
        'Flow_Percentile': st.column_config.NumberColumn('Flow %ile', format='%.1f', help='Flow percentile ranking (0-100)'),
        'Flow_Velocity': st.column_config.NumberColumn('Flow Vel', format='%+.2f', help='Rate of flow change'),
        'Flow_Velocity_Percentile': st.column_config.NumberColumn('Flow Vel %ile', format='%.1f', help='Flow velocity percentile ranking (0-100)'),

        # Three-Indicator System (6) - Percentiles + Velocities
        'MPI_Percentile': st.column_config.NumberColumn('MPI %ile', format='%.1f', help='Trend Strength (0-100)'),
        'MPI_Velocity': st.column_config.NumberColumn('MPI Vel', format='%+.2f', help='Trend acceleration (+expanding/-contracting)'),
        'IBS_Percentile': st.column_config.NumberColumn('IBS %ile', format='%.1f', help='Momentum Timing (0-100)'),
        'IBS_Accel': st.column_config.NumberColumn('IBS Accel', format='%+.3f', help='Momentum acceleration (+bullish/-bearish)'),
        'VPI_Percentile': st.column_config.NumberColumn('VPI %ile', format='%.1f', help='Volume Confirmation (0-100)'),
        'VPI_Velocity': st.column_config.NumberColumn('VPI Vel', format='%+.2f', help='Volume acceleration (+increasing/-decreasing)'),

        # Context: Reports (2)
        'Sentiment_Display': st.column_config.TextColumn('Sentiment', width='small', help='Analyst sentiment score'),
        'Guidance_Display': st.column_config.TextColumn('Guidance', width='small', help='Earnings guidance tone')
    }

    column_config = create_dynamic_column_config(filtered_stocks, display_cols, base_column_config)
    display_cols = [col for col in display_cols if col in filtered_stocks.columns]

    # Apply default sorting: Signal_Bias (Bullish first) - no scoring-based sorting
    if 'Signal_Bias' in filtered_stocks.columns:
        # Make explicit copy and convert Signal_Bias to string to avoid categorical issues
        filtered_stocks_sorted = filtered_stocks.copy()

        # Convert Signal_Bias to string if it's categorical
        if pd.api.types.is_categorical_dtype(filtered_stocks_sorted['Signal_Bias']):
            filtered_stocks_sorted['Signal_Bias'] = filtered_stocks_sorted['Signal_Bias'].astype(str)

        bias_order = {'🟢 BULLISH': 1, '⚪ NEUTRAL': 2, '🔴 BEARISH': 3}
        filtered_stocks_sorted['_bias_sort'] = filtered_stocks_sorted['Signal_Bias'].map(bias_order).fillna(4)

        filtered_stocks_sorted = filtered_stocks_sorted.sort_values(
            by=['_bias_sort'],
            ascending=[True]  # Bullish first
        ).drop('_bias_sort', axis=1)
    else:
        filtered_stocks_sorted = filtered_stocks

    try:
        st.dataframe(
            filtered_stocks_sorted[display_cols],
            column_config=column_config,
            width='stretch',
            hide_index=True
        )

        show_tradingview_export(filtered_stocks, selected_base_filter)

        # ===== PHASE 1: INSTITUTIONAL FLOW ANALYSIS (SEPARATE EXPANDER) =====
        show_institutional_flow_analysis(filtered_stocks)

    except Exception as e:
        create_error_box(f"❌ Error displaying filtered results: {str(e)}")
        handle_error(e, "FilteredResultsDisplay")


def show_tradingview_export(filtered_stocks: pd.DataFrame, selected_base_filter: str) -> None:
    """Show TradingView export section"""
    create_section_header("📋 TradingView Export (Filtered)", "")

    # Sort by Signal_Bias (Bullish first) for better trading priority - no scoring
    if 'Signal_Bias' in filtered_stocks.columns:
        # Make explicit copy and convert Signal_Bias to string to avoid categorical issues
        sorted_stocks = filtered_stocks.copy()

        # Convert Signal_Bias to string if it's categorical
        if pd.api.types.is_categorical_dtype(sorted_stocks['Signal_Bias']):
            sorted_stocks['Signal_Bias'] = sorted_stocks['Signal_Bias'].astype(str)

        bias_order = {'🟢 BULLISH': 1, '⚪ NEUTRAL': 2, '🔴 BEARISH': 3}
        sorted_stocks['_bias_sort'] = sorted_stocks['Signal_Bias'].map(bias_order).fillna(4)

        sorted_stocks = sorted_stocks.sort_values(
            by=['_bias_sort'],
            ascending=[True]  # Bullish first
        ).drop('_bias_sort', axis=1)
    else:
        sorted_stocks = filtered_stocks
    tv_tickers = [f"SGX:{ticker.replace('.SG', '')}" for ticker in sorted_stocks['Ticker'].tolist()]
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

    # ===== ENHANCED CSV EXPORT WITH NORMALIZED DATA =====
    # Create export dataframe with both raw and normalized institutional flow metrics
    export_df = filtered_stocks.copy()

    # Add percentile columns for cross-stock comparison (if available)
    percentile_columns = {
        'Flow_Percentile': 'Flow_Pct',
        'Flow_Velocity_Percentile': 'Flow_Vel_Pct',
        'Volume_Conviction_Percentile': 'Conviction_Pct',
        'Price_Percentile': 'Price_Pct'
    }

    # Rename percentile columns for clarity in export
    for original_col, export_col in percentile_columns.items():
        if original_col in export_df.columns:
            export_df[export_col] = export_df[original_col]

    # Create CSV data with enhanced institutional flow columns
    csv_data = export_df.to_csv(index=False)
    filename_prefix = selected_base_filter.lower().replace(' ', '_').replace('+', 'and')

    # Enhanced help text explaining the data
    help_text = """
    Download complete analysis with institutional flow metrics.

    RAW VALUES (stock-specific):
    - Daily_Flow, Flow_10D, Flow_Velocity: Absolute institutional pressure
    - Volume_Conviction: Up/down volume ratio
    - Divergence_Gap: Price vs flow misalignment

    NORMALIZED VALUES (cross-stock comparison):
    - Flow_Pct, Flow_Vel_Pct: Percentile rankings (0-1 scale)
    - Conviction_Pct: Volume conviction percentile
    - Price_Pct: Price position vs 252-day range

    Use percentiles for comparing across different stocks/prices.
    Use raw values for absolute magnitude within same stock.
    """

    create_download_button(
        csv_data,
        f"institutional_flow_{filename_prefix}_{len(filtered_stocks)}_stocks.csv",
        "📥 Download Institutional Flow Data (CSV)",
        "text/csv",
        help_text
    )


def show_full_results_table(results_df: pd.DataFrame) -> None:
    """Show the full results table in an expander with comprehensive Break & Reversal Pattern data"""
    with st.expander("📋 Full Analysis Results (Break & Reversal Pattern System)", expanded=False):
        try:
            # ===== COMPREHENSIVE COLUMN ORGANIZATION =====
            # Following the trading flow: Signal → Pattern → Score → Context
            full_results_cols = [
                # 1. Core Identification
                'Analysis_Date', 'Ticker', 'Name', 'Close',

                # 2. Primary Signal (Three-Indicator System)
                'Signal_Bias', 'Signal_State', 'Conviction_Level', 'MPI_Position',
                'Is_Triple_Aligned', 'Is_Divergent', 'Is_Accumulation',

                # 3. Pattern Details
                'Ref_High', 'Ref_Low', 'Purge_Level', 'Entry_Level', 'Bars_Since_Break',

                # 4. Three-Indicator Metrics (Percentiles)
                'MPI_Percentile', 'IBS_Percentile', 'VPI_Percentile',

                # 5. Acceleration Metrics (Raw)
                'IBS_Accel', 'RVol_Accel', 'RRange_Accel', 'VPI_Accel',

                # 6. Context: Price Action
                'High', 'Low', 'IBS',
                'Higher_H', 'Higher_HL', 'Lower_L', 'Lower_HL', 'HL_Pattern',

                # 7. Context: Volatility/Range
                'VW_Range_Velocity', 'VW_Range_Percentile', 'Rel_Range_Signal',

                # 8. Context: Volume
                'Relative_Volume', 'RelVol_Velocity', 'RelVol_Trend',
                'VPI_Velocity',

                # 9. Context: MPI Indicators
                'MPI', 'MPI_Velocity', 'MPI_Trend', 'MPI_Zone',

                # 10. Analyst Reports (if available)
                'Sentiment_Display', 'Report_Date_Display', 'Report_Count_Display',

                # 11. Earnings Reports (if available)
                'Earnings_Period', 'Guidance_Display', 'Rev_YoY_Display', 'EPS_DPU_Display',
                'Earnings_Reaction'
            ]

            # ===== COMPREHENSIVE COLUMN CONFIGURATION =====
            base_full_results_config = {
                # === CORE IDENTIFICATION ===
                'Analysis_Date': st.column_config.TextColumn('📅 Date', width='small', help='Analysis date for the scan'),
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Name': st.column_config.TextColumn('Company', width='medium'),
                'Close': st.column_config.NumberColumn('Close', width='small'),

                # === PRIMARY SIGNAL (THREE-INDICATOR SYSTEM) ===
                'Signal_Bias': st.column_config.TextColumn('🎯 Bias', width='small', help='🟢 BULLISH / 🔴 BEARISH / ⚪ NEUTRAL'),
                'Signal_State': st.column_config.TextColumn('📊 State', width='medium', help='🔥 Triple Bullish / ⚠️ Divergence / 👀 Accumulation'),
                'Conviction_Level': st.column_config.TextColumn('💪 Conviction', width='small', help='High / Moderate / Low / Warning'),
                'MPI_Position': st.column_config.TextColumn('📍 Stage', width='medium', help='📍 EARLY STAGE / ⚠️ MID STAGE / 🚨 LATE STAGE / ❌ TOO WEAK'),
                'Is_Triple_Aligned': st.column_config.CheckboxColumn('Triple', width='tiny', help='All 3 indicators aligned'),
                'Is_Divergent': st.column_config.CheckboxColumn('Div', width='tiny', help='Price/Volume Divergence'),
                'Is_Accumulation': st.column_config.CheckboxColumn('Acc', width='tiny', help='Accumulation Pattern'),

                # === PATTERN DETAILS ===
                'Ref_High': st.column_config.NumberColumn('Ref High', width='small', help='Reference high level that was broken'),
                'Ref_Low': st.column_config.NumberColumn('Ref Low', width='small', help='Reference low level that was broken'),
                'Purge_Level': st.column_config.NumberColumn('Purge', width='small', help='Extreme price of the break candle'),
                'Entry_Level': st.column_config.NumberColumn('Entry', width='small', help='Price where reversal signal triggered'),
                'Bars_Since_Break': st.column_config.NumberColumn('Bars', width='tiny', help='Bars since break occurred'),

                # === THREE-INDICATOR METRICS (PERCENTILES) ===
                'MPI_Percentile': st.column_config.NumberColumn('MPI %ile', format='%.1f', help='Trend Strength (0-100)'),
                'IBS_Percentile': st.column_config.NumberColumn('IBS %ile', format='%.1f', help='Momentum Timing (0-100)'),
                'VPI_Percentile': st.column_config.NumberColumn('VPI %ile', format='%.1f', help='Volume Confirmation (0-100)'),

                # === ACCELERATION METRICS (RAW) ===
                'IBS_Accel': st.column_config.NumberColumn('IBS Accel', format='%+.3f', help='IBS 3-bar acceleration'),
                'RVol_Accel': st.column_config.NumberColumn('RVol Accel', format='%+.3f', help='Relative Volume 3-bar acceleration'),
                'RRange_Accel': st.column_config.NumberColumn('RRng Accel', format='%+.3f', help='Relative Range 3-bar acceleration'),
                'VPI_Accel': st.column_config.NumberColumn('VPI Accel', format='%+.3f', help='VPI 2nd derivative'),

                # === CONTEXT: PRICE ACTION ===
                'High': st.column_config.NumberColumn('High', width='small'),
                'Low': st.column_config.NumberColumn('Low', width='small'),
                'IBS': st.column_config.NumberColumn('IBS', format='%.3f', help='Internal Bar Strength (0-1)'),
                'Higher_H': st.column_config.NumberColumn('HH', width='tiny', help='Higher High pattern'),
                'Higher_HL': st.column_config.NumberColumn('HHL', width='tiny', help='Higher High & Low pattern'),
                'Lower_L': st.column_config.NumberColumn('LL', width='tiny', help='Lower Low pattern'),
                'Lower_HL': st.column_config.NumberColumn('LHL', width='tiny', help='Lower High & Low pattern'),
                'HL_Pattern': st.column_config.TextColumn('H/L', width='small', help='HHL=Both H&L, HH=Higher H only, LHL=Both L&L, LL=Lower L only'),

                # === CONTEXT: VOLATILITY/RANGE ===
                'VW_Range_Velocity': st.column_config.NumberColumn('VW Range Vel', format='%+.4f', help='Volume-weighted range velocity'),
                'VW_Range_Percentile': st.column_config.NumberColumn('VW Range Pct', format='%.3f', help='Volume-weighted range percentile (0-1)'),
                'Rel_Range_Signal': st.column_config.NumberColumn('Range Signal', width='tiny', help='Range expansion signal'),

                # === CONTEXT: VOLUME ===
                'Relative_Volume': st.column_config.NumberColumn('Rel Vol', format='%.1f%%', help='Volume relative to 14-day average'),
                'RelVol_Velocity': st.column_config.NumberColumn('Vol Vel', format='%+.1f', help='Day-over-day volume change'),
                'RelVol_Trend': st.column_config.TextColumn('Vol Trend', width='small', help='Building/Stable/Fading'),
                'VPI_Velocity': st.column_config.NumberColumn('VPI Vel', format='%+.2f', help='VPI 1st derivative'),

                # === CONTEXT: MPI INDICATORS ===
                'MPI': st.column_config.NumberColumn('MPI', format='%.2f', help='Market Positivity Index (0-1 scale)'),
                'MPI_Velocity': st.column_config.NumberColumn('MPI Vel', format='%+.2f', help='Rate of MPI change'),
                'MPI_Trend': st.column_config.TextColumn('MPI Trend', width='small', help='Expanding/Flat/Contracting'),
                'MPI_Zone': st.column_config.NumberColumn('MPI Zone', width='tiny', help='1=Weak, 2=Transition, 3=Bullish, 4=Strong'),

                # === ANALYST REPORTS ===
                'Sentiment_Display': st.column_config.TextColumn('Sentiment', width='small', help='Analyst sentiment score'),
                'Report_Date_Display': st.column_config.TextColumn('Report Date', width='small'),
                'Report_Count_Display': st.column_config.TextColumn('Reports', width='small'),

                # === EARNINGS REPORTS ===
                'Earnings_Period': st.column_config.TextColumn('Period', width='small', help='Earnings period (Q1/Q2/FY etc.)'),
                'Guidance_Display': st.column_config.TextColumn('Guidance', width='small', help='Management guidance tone'),
                'Rev_YoY_Display': st.column_config.TextColumn('Rev YoY', width='small', help='Revenue year-over-year change'),
                'EPS_DPU_Display': st.column_config.TextColumn('EPS/DPU', width='small', help='EPS or DPU year-over-year change'),
                'Earnings_Reaction': st.column_config.TextColumn('Earn React', width='medium', help='Historical win rate after earnings releases')
            }

            # Create dynamic column config and filter to available columns
            full_results_column_config = create_dynamic_column_config(results_df, full_results_cols, base_full_results_config)
            available_cols = [col for col in full_results_cols if col in results_df.columns]

            # Add informational header
            st.markdown("""
            **📊 Break & Reversal Pattern Analysis Results**

            **Column Organization:**
            - 🎯 **Primary Signal**: Break & Reversal pattern detection
            - 📊 **Pattern Details**: Reference levels and timing
            - 🚀 **Acceleration Metrics**: Momentum scoring (primary ranking)
            - 📈 **Context**: Supporting technical indicators
            - 📊 **Reports**: Analyst and earnings data

            **Signal Quality Guide:**
            - 🔥 **EXCEPTIONAL**: Triple confirmation + strong acceleration
            - 🟢 **STRONG**: 2-3 accelerations met, good quality
            - 🟡 **GOOD**: 1-2 accelerations met
            - ⚪ **MODERATE**: Weak accelerations
            - 🟠 **WEAK**: Barely qualified
            - 🔴 **POOR**: Below thresholds
            """)

            # Display the comprehensive dataframe
            st.dataframe(
                results_df[available_cols],
                column_config=full_results_column_config,
                width='stretch',
                hide_index=True
            )

            # Show summary statistics
            if len(results_df) > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    bullish_count = (results_df['Signal_Bias'] == '🟢 BULLISH').sum()
                    st.metric("🟢 Bullish Signals", bullish_count)
                with col2:
                    bearish_count = (results_df['Signal_Bias'] == '🔴 BEARISH').sum()
                    st.metric("🔴 Bearish Signals", bearish_count)
                with col3:
                    triple_count = results_df['Is_Triple_Aligned'].sum() if 'Is_Triple_Aligned' in results_df.columns else 0
                    st.metric("🔥 Triple Aligned", triple_count)
                with col4:
                    div_count = results_df['Is_Divergent'].sum() if 'Is_Divergent' in results_df.columns else 0
                    st.metric("⚠️ Divergences", div_count)

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
            st.dataframe(pd.DataFrame(trend_summary), hide_index=True, width='stretch')


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
                    width='stretch'
                )
        except Exception as e:
            st.warning(f"Could not load report history: {e}")


def display_detailed_earnings_reports(results_df: pd.DataFrame) -> None:
    """Display detailed earnings reports with dropdown selection"""
    if results_df.empty:
        return

    # Check if earnings columns exist and have data
    has_revenue = 'revenue' in results_df.columns and results_df['revenue'].notna().any()
    has_profit = 'net_profit' in results_df.columns and results_df['net_profit'].notna().any()
    has_guidance = 'guidance_tone' in results_df.columns and results_df['guidance_tone'].notna().any()

    if not (has_revenue or has_profit or has_guidance):
        st.info("💰 No earnings reports available in current scan results")
        return

    # Create mask for stocks with earnings data (defensive approach)
    earnings_mask = pd.Series(False, index=results_df.index)
    if 'revenue' in results_df.columns:
        earnings_mask = earnings_mask | results_df['revenue'].notna()
    if 'net_profit' in results_df.columns:
        earnings_mask = earnings_mask | results_df['net_profit'].notna()
    if 'guidance_tone' in results_df.columns:
        earnings_mask = earnings_mask | results_df['guidance_tone'].notna()

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

                if st.button("🔄 Force Update from EOD", type="secondary", width='stretch', key="force_eod_button"):
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
                    if st.button("🔄 Force Update Selected EOD", type="secondary", width='stretch', key="force_selected_eod_button"):
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

                        if st.button("📥 Force Download Latest Data", type="secondary", width='stretch', key="force_download_button"):
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


def display_error_log_section():
    """Display error log section at bottom of scanner with session diagnostics"""
    # Only show if error_logger exists in session state
    if 'error_logger' not in st.session_state:
        return

    error_logger = st.session_state.error_logger
    summary = error_logger.get_summary()

    st.markdown("---")  # Separator
    create_section_header("📋 Error & Warning Log", "Session diagnostics and troubleshooting")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Errors", summary['errors'])
    with col2:
        st.metric("Warnings", summary['warnings'])
    with col3:
        st.metric("Debug Entries", summary['debug_entries'])
    with col4:
        st.metric("Performance Logs", summary['performance_metrics'])

    # Show success message if no issues
    if summary['errors'] == 0 and summary['warnings'] == 0 and summary['debug_entries'] == 0:
        create_success_box("✅ No errors or warnings - scan completed successfully!")
    else:
        # Display errors/warnings with existing method
        error_logger.display_errors_in_streamlit()

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Clear Log", help="Clear all logged errors and warnings for new scan"):
            error_logger.clear()
            st.success("Error log cleared!")
            time.sleep(1)
            st.rerun()

    with col2:
        # Export logs as JSON
        log_export = error_logger.export_logs(format="json")
        st.download_button(
            "📥 Download JSON Log",
            data=log_export,
            file_name=f"scanner_error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download detailed error log in JSON format for troubleshooting"
        )

    with col3:
        # Export logs as text
        log_export_text = error_logger.export_logs(format="text")
        st.download_button(
            "📥 Download Text Log",
            data=log_export_text,
            file_name=f"scanner_error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download simplified error log in text format"
        )
