# Scanner module package
"""
Stock Scanner - Main Entry Point
Refactored into modular components for better maintainability
"""

import streamlit as st

# Import the modular components
from pages.scanner.ui import (
    show_update_prompt, show_scanning_configuration, show_advanced_settings,
    show_confirmation_filters, execute_scan_button, display_scan_summary, show_base_pattern_filter,
    display_filtered_results, show_full_results_table, show_mpi_insights,
    show_force_update_options,
    display_detailed_analyst_reports, display_detailed_earnings_reports,
    display_error_log_section
)
from pages.scanner.logic import run_enhanced_stock_scan
from pages.scanner.data import apply_dynamic_filters
from pages.common.error_handler import ErrorLogger


def display_scan_results(results_df):
    """Main function to display scanning results with all features"""
    if results_df.empty:
        st.warning("No results to display.")
        return

    try:
        display_scan_summary(results_df)

        base_stocks = show_base_pattern_filter(results_df)

        if len(base_stocks) > 0:
            filtered_stocks = apply_dynamic_filters(base_stocks, results_df)
            display_filtered_results(filtered_stocks, st.session_state.base_filter_type)
        else:
            st.warning(f"No stocks found for pattern: {st.session_state.base_filter_type}")

        show_full_results_table(results_df)
        show_mpi_insights(results_df)

        # CRITICAL: These functions now display with dropdown selection
        # Note: These functions are now implemented and enabled
        display_detailed_analyst_reports(results_df)
        display_detailed_earnings_reports(results_df)

        # Display error log section at the bottom
        display_error_log_section()

    except Exception as e:
        st.session_state.error_logger.log_error("Results Display", e)
        st.error(f"❌ Error displaying results: {str(e)}")
        st.info("💡 Try refreshing the page or running the scan again")


def show():
    """Main scanner page display with all features"""

    # Initialize session state
    if 'error_logger' not in st.session_state:
        from pages.common.error_handler import ErrorLogger
        st.session_state.error_logger = ErrorLogger()

    st.title("📈 Stock Scanner")
    st.markdown("Enhanced with **Pure MPI Expansion System**, **Analyst Report Integration**, and **Earnings Report Integration**")
    st.markdown("**Data Source: Local File System** (./data/Historical_Data + ./data/EOD_Data + yfinance API)")

    st.subheader("📥 Data Management")

    if 'update_check_done' not in st.session_state:
        st.session_state.update_check_done = False

    if not st.session_state.update_check_done:
        update_result = show_update_prompt()
        if update_result:
            st.session_state.update_check_done = True
            st.rerun()
    else:
        if st.button("🔄 Check for Updates Again"):
            st.session_state.update_check_done = False
            st.rerun()

        # Show force update options when data is up to date
        show_force_update_options()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear Error Log"):
            st.session_state.error_logger = ErrorLogger()
            st.success("Error log cleared!")
            st.rerun()
    with col2:
        if st.button("🔄 Refresh Analyst Reports"):
            from utils.analyst_reports import clear_reports_cache
            clear_reports_cache()
            st.success("Analyst reports cache cleared - reports will reload on next scan")
            st.rerun()
    with col3:
        if st.button("🔄 Refresh Earnings"):
            from utils.earnings_reports import clear_earnings_cache
            clear_earnings_cache()
            st.success("Earnings reports cache cleared - earnings will reload on next scan")
            st.rerun()

    # Performance monitoring section
    with st.expander("📊 Performance Stats", expanded=False):
        try:
            from pages.common.performance import get_performance_stats
            perf_stats = get_performance_stats()

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Memory Usage", f"{perf_stats['memory_usage_mb']:.1f} MB")
            with col_b:
                st.metric("Cache Hit Rate", f"{perf_stats['cache_stats']['data_cache']['hit_rate']:.1%}")
            with col_c:
                st.metric("Active Operations", len(perf_stats['operation_stats']))

            if perf_stats['operation_stats']:
                st.subheader("⚡ Operation Performance")
                for op_name, stats in list(perf_stats['operation_stats'].items())[:5]:  # Show top 5
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**{op_name}**")
                    with col2:
                        st.write(f"Count: {stats['count']}")
                    with col3:
                        st.write(f"Avg: {stats['avg_time']:.3f}s")
                    with col4:
                        st.write(f"Last: {stats['last_time']:.3f}s")

        except Exception as e:
            st.warning(f"Performance stats unavailable: {e}")

    # Data Quality section
    with st.expander("🔍 Data Quality Stats", expanded=False):
        try:
            if 'scan_results' in st.session_state and st.session_state.scan_results is not None:
                from pages.common.data_validation import validate_data_quality, get_validation_summary
                results_df = st.session_state.scan_results

                # Overall data quality
                quality_metrics = validate_data_quality(results_df, 'stock_data', strict=False)
                quality_summary = get_validation_summary(quality_metrics)

                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Data Completeness", quality_summary['data_completeness'])
                with col_b:
                    st.metric("Validation Score", quality_summary['validation_score'])
                with col_c:
                    st.metric("Missing Values", quality_summary['missing_values'])
                with col_d:
                    st.metric("Duplicate Rows", quality_summary['duplicate_rows'])

                # Issues breakdown
                if quality_summary['total_issues'] > 0:
                    st.subheader("⚠️ Data Quality Issues")
                    issue_cols = st.columns(4)
                    with issue_cols[0]:
                        st.metric("Critical", quality_summary['critical_issues'])
                    with issue_cols[1]:
                        st.metric("Errors", quality_summary['error_issues'])
                    with issue_cols[2]:
                        st.metric("Warnings", quality_summary['warning_issues'])
                    with issue_cols[3]:
                        st.metric("Info", quality_summary['info_issues'])

                    # Show top issues
                    if quality_metrics.issues:
                        st.subheader("Top Issues")
                        for issue in quality_metrics.issues[:5]:  # Show top 5
                            severity_emoji = {
                                'critical': '🚨',
                                'error': '❌',
                                'warning': '⚠️',
                                'info': 'ℹ️'
                            }.get(issue.severity.value, '❓')

                            st.write(f"{severity_emoji} **{issue.check_name}**: {issue.message}")

        except Exception as e:
            st.warning(f"Data quality stats unavailable: {e}")

    st.session_state.error_logger.display_errors_in_streamlit()

    scan_scope, selected_stock, scan_date_type, historical_date = show_scanning_configuration()
    days_back, rolling_window = show_advanced_settings()

    st.subheader("🚀 Execute Scan")

    # Always calculate confirmation with default settings (matching Pine Script behavior)
    execute_scan_button(scan_scope, selected_stock, scan_date_type, historical_date,
                       days_back, rolling_window,
                       # Always calculate confirmation with default thresholds
                       use_ibs=True, use_rvol=True, use_rrange=True, confirmation_logic="OR",
                       ibs_threshold=0.10, rvol_threshold=0.20, rrange_threshold=0.30)

    if 'last_scan_time' in st.session_state:
        st.info(f"📊 Last scan completed: {st.session_state.last_scan_time}")

        if 'last_scan_config' in st.session_state:
            config = st.session_state.last_scan_config
            st.info(f"Scope: {config['scope']} | Date: {config['date']} | Stocks: {config['stock_count']}")

    if 'scan_results' in st.session_state:
        display_scan_results(st.session_state.scan_results)


if __name__ == "__main__":
    show()
