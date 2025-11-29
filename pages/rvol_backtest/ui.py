# UI Components - Streamlit Interface for RVOL BackTest
"""
Streamlit UI components for RVOL BackTest
Enhanced with fill rate displays and optimization results
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional

from .optimizer import run_parameter_optimization, get_optimal_parameters
from .vwap_engine import calculate_monthly_vwap, get_vwap_summary
from .backtest_engine import backtest_strategy_with_retest

def show_configuration_panel() -> Tuple[str, date, date, Dict]:
    """
    Display configuration panel for backtest parameters

    Returns:
        Tuple of (ticker, start_date, end_date, backtest_params)
    """
    st.subheader("🎯 Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Stock Selection**")

        # Get available stocks from watchlist
        try:
            from utils.watchlist import get_active_watchlist
            watchlist = get_active_watchlist()

            if watchlist:
                ticker_options = [(ticker, f"{ticker}") for ticker in watchlist]
                selected_ticker = st.selectbox(
                    "Select Stock:",
                    options=[t[0] for t in ticker_options],
                    format_func=lambda x: f"{x}",
                    help="Choose from your active watchlist"
                )
            else:
                st.warning("No stocks in watchlist. Please add stocks first.")
                selected_ticker = st.text_input("Enter Ticker:", value="A17U.SG")

        except Exception as e:
            st.warning(f"Could not load watchlist: {e}")
            selected_ticker = st.text_input("Enter Ticker:", value="A17U.SG")

        st.markdown("**📅 Date Range**")

        # Default to last 2 years
        default_end = date.today()
        default_start = default_end - timedelta(days=730)

        start_date = st.date_input(
            "Start Date:",
            value=default_start,
            max_value=default_end - timedelta(days=30),
            help="Start date for backtesting (minimum 30 days before end date)"
        )

        end_date = st.date_input(
            "End Date:",
            value=default_end,
            min_value=start_date + timedelta(days=30),
            max_value=default_end,
            help="End date for backtesting"
        )

        # Validate date range
        if end_date <= start_date + timedelta(days=30):
            st.error("Date range must be at least 30 days")
            return None, None, None, None

    with col2:
        st.markdown("**⚙️ Strategy Parameters**")

        # Threshold range
        col_a, col_b = st.columns(2)
        with col_a:
            threshold_min = st.slider(
                "Min Threshold (%):",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Minimum deviation below VWAP to trigger signal"
            ) / 100  # Convert to decimal

        with col_b:
            threshold_max = st.slider(
                "Max Threshold (%):",
                min_value=threshold_min * 100 + 0.1,
                max_value=3.0,
                value=2.0,
                step=0.1,
                help="Maximum deviation below VWAP to test"
            ) / 100  # Convert to decimal

        # Step size
        threshold_step = st.slider(
            "Step Size (%):",
            min_value=0.05,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Increment between threshold tests"
        ) / 100  # Convert to decimal

        # Stop loss
        stop_loss = st.slider(
            "Stop Loss (%):",
            min_value=0.5,
            max_value=3.0,
            value=1.2,
            step=0.1,
            help="Stop loss percentage below entry price"
        ) / 100  # Convert to decimal

        # Position size
        position_size = st.number_input(
            "Position Size (shares):",
            min_value=1000,
            max_value=100000,
            value=50000,
            step=5000,
            help="Number of shares per trade"
        )

        # Order expiry (optional)
        use_order_expiry = st.checkbox(
            "Enable Order Expiry",
            value=False,
            help="Cancel unfilled orders after specified days"
        )

        order_expiry_days = None
        if use_order_expiry:
            order_expiry_days = st.slider(
                "Order Expiry (days):",
                min_value=1,
                max_value=30,
                value=10,
                help="Cancel orders if not filled within this many days"
            )

    # Summary
    st.markdown("---")
    st.markdown("**📋 Configuration Summary:**")
    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.info(f"**Stock:** {selected_ticker}")
        st.info(f"**Date Range:** {(end_date - start_date).days} days")

    with summary_col2:
        st.info(f"**Thresholds:** {threshold_min:.1%} - {threshold_max:.1%}")
        st.info(f"**Step Size:** {threshold_step:.1%}")

    with summary_col3:
        st.info(f"**Stop Loss:** {stop_loss:.1%}")
        st.info(f"**Position:** {position_size:,} shares")

    backtest_params = {
        'threshold_min': threshold_min,
        'threshold_max': threshold_max,
        'threshold_step': threshold_step,
        'stop_loss': stop_loss,
        'position_size': position_size,
        'order_expiry_days': order_expiry_days
    }

    return selected_ticker, start_date, end_date, backtest_params

def show_data_validation(df: pd.DataFrame, ticker: str) -> bool:
    """
    Display data validation results

    Args:
        df: DataFrame with stock data
        ticker: Stock ticker

    Returns:
        bool: True if data is valid for backtesting
    """
    st.subheader("🔍 Data Validation")

    if df.empty:
        st.error(f"❌ No data available for {ticker}")
        return False

    # Basic data checks
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_days = len(df)
        st.metric("Total Days", total_days)

    with col2:
        date_range = f"{df.index.min().date()} to {df.index.max().date()}"
        st.metric("Date Range", f"{(df.index.max() - df.index.min()).days} days")

    with col3:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        has_required = all(col in df.columns for col in required_cols)
        st.metric("Data Quality", "✅ Complete" if has_required else "❌ Missing columns")

    with col4:
        # Check for VWAP calculation
        try:
            df_with_vwap = calculate_monthly_vwap(df.copy())
            vwap_summary = get_vwap_summary(df_with_vwap)
            months = vwap_summary.get('months_covered', 0)
            st.metric("VWAP Ready", f"✅ {months} months")
        except Exception as e:
            st.metric("VWAP Ready", "❌ Error")
            st.error(f"VWAP calculation failed: {e}")
            return False

    # VWAP summary if available
    try:
        df_with_vwap = calculate_monthly_vwap(df.copy())
        vwap_summary = get_vwap_summary(df_with_vwap)

        st.markdown("**VWAP Statistics:**")
        vwap_col1, vwap_col2, vwap_col3 = st.columns(3)

        with vwap_col1:
            st.info(f"**VWAP Range:** ${vwap_summary.get('vwap_min', 0):.2f} - ${vwap_summary.get('vwap_max', 0):.2f}")

        with vwap_col2:
            st.info(f"**Avg Distance:** {vwap_summary.get('avg_vwap_distance_pct', 0):.1f}%")

        with vwap_col3:
            st.info(f"**Months:** {vwap_summary.get('months_covered', 0)}")

    except Exception as e:
        st.warning(f"Could not calculate VWAP summary: {e}")

    return True

def execute_optimization(ticker: str, start_date: date, end_date: date, backtest_params: Dict):
    """
    Execute the parameter optimization

    Args:
        ticker: Stock ticker
        start_date: Start date for backtesting
        end_date: End date for backtesting
        backtest_params: Dictionary of backtest parameters
    """
    st.subheader("🚀 Running Optimization")

    # Load data
    with st.spinner(f"Loading data for {ticker}..."):
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            df = fetcher.get_single_stock_data(ticker)

            if df.empty:
                st.error(f"No data available for {ticker}")
                return

            # Filter date range
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]

            if len(df) < 30:
                st.error(f"Insufficient data: only {len(df)} days available")
                return

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    # Validate data
    if not show_data_validation(df, ticker):
        return

    # Show optimization parameters
    st.markdown("**Optimization Parameters:**")
    opt_col1, opt_col2, opt_col3 = st.columns(3)

    with opt_col1:
        thresholds_to_test = int((backtest_params['threshold_max'] - backtest_params['threshold_min']) / backtest_params['threshold_step']) + 1
        st.info(f"**Thresholds to Test:** {thresholds_to_test}")
        st.info(f"**Range:** {backtest_params['threshold_min']:.1%} - {backtest_params['threshold_max']:.1%}")

    with opt_col2:
        st.info(f"**Stop Loss:** {backtest_params['stop_loss']:.1%}")
        st.info(f"**Position Size:** {backtest_params['position_size']:,} shares")

    with opt_col3:
        estimated_time = thresholds_to_test * 0.5  # Rough estimate: 0.5 seconds per threshold
        st.info(f"**Est. Time:** {estimated_time:.0f} seconds")
        st.info(f"**Order Expiry:** {backtest_params.get('order_expiry_days', 'None')} days")

    # Execute optimization
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        run_optimization_process(df, backtest_params)

def run_optimization_process(df: pd.DataFrame, backtest_params: Dict):
    """
    Run the optimization process with progress tracking

    Args:
        df: DataFrame with stock data
        backtest_params: Optimization parameters
    """
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Run optimization
        with st.spinner("Running parameter optimization..."):
            results_df = run_parameter_optimization(
                df=df,
                threshold_min=backtest_params['threshold_min'],
                threshold_max=backtest_params['threshold_max'],
                threshold_step=backtest_params['threshold_step'],
                stop_loss=backtest_params['stop_loss'],
                position_size=backtest_params['position_size'],
                order_expiry_days=backtest_params.get('order_expiry_days')
            )

        progress_bar.progress(100)
        status_text.success("✅ Optimization completed!")

        # Store results in session state
        st.session_state.rvol_optimization_results = results_df
        st.session_state.rvol_optimization_completed = True

        # Get optimal parameters
        optimal_params = get_optimal_parameters(results_df)

        # Display results
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.error(f"❌ Optimization failed: {e}")
        st.error(f"Error details: {e}")

def show_optimization_results(results_df: pd.DataFrame):
    """
    Display optimization results with enhanced fill rate metrics

    Args:
        results_df: DataFrame from run_parameter_optimization
    """
    if results_df.empty:
        st.warning("No optimization results to display")
        return

    st.subheader("📊 Optimization Results")

    # Get optimal parameters
    optimal_params = get_optimal_parameters(results_df)

    if 'error' in optimal_params:
        st.error(f"Could not determine optimal parameters: {optimal_params['error']}")
        return

    # Key metrics dashboard
    st.markdown("### 🎯 Optimal Parameters")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Optimal Threshold",
            f"{optimal_params['optimal_threshold_pct']:.1f}%",
            help="Best deviation below VWAP"
        )

    with metric_col2:
        st.metric(
            "Expected Sharpe",
            f"{optimal_params['expected_sharpe_ratio']:.2f}",
            help="Risk-adjusted return measure"
        )

    with metric_col3:
        st.metric(
            "Expected Win Rate",
            f"{optimal_params['expected_win_rate_pct']:.1f}%",
            help="Percentage of profitable trades"
        )

    with metric_col4:
        fill_rate = optimal_params.get('expected_fill_rate_pct', 0)
        st.metric(
            "Fill Rate",
            f"{fill_rate:.1f}%",
            help="Percentage of signals that fill",
            delta="Realistic" if 30 <= fill_rate <= 80 else "Check" if fill_rate > 80 else "Low"
        )

    # Confidence and warnings
    confidence_col, warnings_col = st.columns(2)

    with confidence_col:
        confidence_level = optimal_params.get('confidence_level', 'UNKNOWN')
        if confidence_level == 'HIGH':
            st.success(f"🎉 **High Confidence** - Results are statistically robust")
        elif confidence_level == 'MODERATE':
            st.info(f"📊 **Moderate Confidence** - Results are reasonable")
        else:
            st.warning(f"⚠️ **Low Confidence** - Results may not be reliable")

    with warnings_col:
        risk_warnings = optimal_params.get('risk_warnings', [])
        if risk_warnings:
            st.warning("⚠️ **Risk Warnings:**")
            for warning in risk_warnings:
                st.write(f"• {warning}")
        else:
            st.success("✅ **No major risk warnings**")

    # Optimization table
    st.markdown("### 📋 Complete Results")

    # Prepare display columns
    display_columns = [
        'threshold_pct', 'signals_generated', 'fill_rate_pct',
        'total_trades', 'win_rate_pct', 'sharpe_ratio',
        'total_pnl_dollars', 'max_drawdown'
    ]

    display_names = {
        'threshold_pct': 'Threshold (%)',
        'signals_generated': 'Signals',
        'fill_rate_pct': 'Fill Rate (%)',
        'total_trades': 'Trades',
        'win_rate_pct': 'Win Rate (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'total_pnl_dollars': 'Total P&L ($)',
        'max_drawdown': 'Max Drawdown ($)'
    }

    # Filter and rename columns
    display_df = results_df[display_columns].copy()
    display_df.columns = [display_names.get(col, col) for col in display_columns]

    # Format numeric columns
    display_df['Total P&L ($)'] = display_df['Total P&L ($)'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    display_df['Max Drawdown ($)'] = display_df['Max Drawdown ($)'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

    st.dataframe(display_df, use_container_width=True)

    # Sharpe ratio chart
    show_sharpe_chart(results_df)

def show_sharpe_chart(results_df: pd.DataFrame):
    """
    Display Sharpe ratio vs threshold chart

    Args:
        results_df: Optimization results DataFrame
    """
    st.markdown("### 📈 Sharpe Ratio Analysis")

    if results_df.empty or 'sharpe_ratio' not in results_df.columns:
        st.warning("No Sharpe ratio data available")
        return

    # Create chart
    fig = go.Figure()

    # Sharpe ratio line
    fig.add_trace(go.Scatter(
        x=results_df['threshold_pct'],
        y=results_df['sharpe_ratio'],
        mode='lines+markers',
        name='Sharpe Ratio',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Highlight optimal point
    optimal_row = results_df.iloc[0]  # Already sorted by Sharpe
    fig.add_trace(go.Scatter(
        x=[optimal_row['threshold_pct']],
        y=[optimal_row['sharpe_ratio']],
        mode='markers',
        name='Optimal',
        marker=dict(size=12, color='red', symbol='star')
    ))

    # Add reference line at Sharpe = 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")

    fig.update_layout(
        title="Sharpe Ratio vs Deviation Threshold",
        xaxis_title="Deviation Threshold (%)",
        yaxis_title="Sharpe Ratio",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Chart insights
    st.markdown("**Chart Insights:**")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Optimal Threshold:** {optimal_row['threshold_pct']:.1f}% (Sharpe: {optimal_row['sharpe_ratio']:.2f})")

    with col2:
        positive_sharpe = (results_df['sharpe_ratio'] > 0).sum()
        total_thresholds = len(results_df)
        st.info(f"**Positive Sharpe:** {positive_sharpe}/{total_thresholds} thresholds")

def show_trade_log(results_df: pd.DataFrame):
    """
    Display detailed trade log for optimal threshold

    Args:
        results_df: Optimization results DataFrame
    """
    st.subheader("📋 Trade Log (Optimal Threshold)")

    if results_df.empty:
        st.warning("No trade data available")
        return

    # Get optimal threshold
    optimal_threshold = results_df.iloc[0]['threshold_decimal']

    # Note: In a full implementation, we'd store the actual trades
    # For now, show a placeholder
    st.info("Trade log will be displayed here after running optimization")

    # Placeholder trade log structure
    st.markdown("**Sample Trade Log Structure:**")
    sample_trades = pd.DataFrame({
        'Signal Date': ['2024-01-15', '2024-02-20', '2024-03-10'],
        'Entry Date': ['2024-01-16', '2024-02-22', '2024-03-12'],
        'Entry Price': [2.15, 2.32, 2.08],
        'Exit Price': [2.20, 2.28, 2.12],
        'P&L ($)': [2500, -2000, 2000],
        'Exit Reason': ['VWAP_Target', 'Stop_Loss', 'VWAP_Target']
    })

    st.dataframe(sample_trades, use_container_width=True)

    st.markdown("**Note:** Actual trade log will show all completed trades for the optimal threshold")

def show_export_options(results_df: pd.DataFrame):
    """
    Display export options for results

    Args:
        results_df: Optimization results DataFrame
    """
    st.subheader("📥 Export Results")

    if results_df.empty:
        st.warning("No results to export")
        return

    # Export optimization results
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Optimization Results (CSV)",
        data=csv_data,
        file_name=f"rvol_optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download complete optimization results"
    )

    # Export summary
    optimal_params = get_optimal_parameters(results_df)
    summary_text = f"""
RVOL BackTest Optimization Summary
==================================

Optimal Parameters:
- Threshold: {optimal_params.get('optimal_threshold_pct', 'N/A'):.1f}%
- Expected Sharpe: {optimal_params.get('expected_sharpe_ratio', 'N/A'):.2f}
- Expected Win Rate: {optimal_params.get('expected_win_rate_pct', 'N/A'):.1f}%
- Expected Fill Rate: {optimal_params.get('expected_fill_rate_pct', 'N/A'):.1f}%
- Expected Total Trades: {optimal_params.get('expected_total_trades', 'N/A')}
- Confidence Level: {optimal_params.get('confidence_level', 'N/A')}

Risk Warnings:
{chr(10).join(optimal_params.get('risk_warnings', ['None']))}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    st.download_button(
        label="📄 Download Summary Report (TXT)",
        data=summary_text,
        file_name=f"rvol_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        help="Download optimization summary report"
    )
