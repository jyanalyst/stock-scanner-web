# RVOL BackTest Module
"""
Monthly VWAP Mean Reversion Backtesting System
Provides optimal entry parameters for selected stocks using realistic retest fill logic
"""

import streamlit as st
from .ui import (
    show_configuration_panel,
    execute_optimization,
    show_optimization_results,
    show_trade_log,
    show_export_options
)

def show():
    """Main entry point for RVOL BackTest page"""
    st.title("📊 RVOL BackTest - Monthly VWAP Mean Reversion")
    st.markdown("**Optimize entry parameters for mean reversion trades using Monthly VWAP as anchor**")

    # Strategy overview
    with st.expander("🎯 Strategy Overview", expanded=False):
        st.markdown("""
        **Monthly VWAP Mean Reversion Strategy:**
        - **Entry Signal:** Price drops below Monthly VWAP by X% threshold
        - **Order Fill:** Realistic retest fill (limit order at signal day's Low, fills on retest)
        - **Exit Rules:**
          - Target: High reaches VWAP (mean reversion complete)
          - Stop Loss: Low drops 1.2% below entry price
        - **Goal:** Find optimal deviation threshold that maximizes Sharpe ratio

        **Key Innovation:** Realistic fill logic prevents over-optimization by accounting for orders that never fill.
        """)

    # Clear results button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ Clear Results"):
            if 'rvol_optimization_results' in st.session_state:
                del st.session_state.rvol_optimization_results
            if 'rvol_optimization_completed' in st.session_state:
                del st.session_state.rvol_optimization_completed
            st.rerun()

    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Main workflow
    try:
        # Step 1: Configuration
        config_result = show_configuration_panel()

        if config_result is None:
            st.error("Please fix configuration errors above")
            return

        ticker, start_date, end_date, backtest_params = config_result

        # Step 2: Execute optimization
        execute_optimization(ticker, start_date, end_date, backtest_params)

        # Step 3: Display results (if available)
        if st.session_state.get('rvol_optimization_completed', False):
            st.markdown("---")

            results_df = st.session_state.get('rvol_optimization_results')
            if results_df is not None and not results_df.empty:
                # Show optimization results
                show_optimization_results(results_df)

                # Show trade log
                st.markdown("---")
                show_trade_log(results_df)

                # Show export options
                st.markdown("---")
                show_export_options(results_df)
            else:
                st.warning("No optimization results available")

    except Exception as e:
        st.error(f"❌ RVOL BackTest error: {e}")
        st.info("💡 Try refreshing the page or check the configuration")

        # Show error details in expander
        with st.expander("🔍 Error Details", expanded=False):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())
