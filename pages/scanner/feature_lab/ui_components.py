"""
Feature Lab UI Components - Streamlit interface for style learning system

Provides the user interface for recording historical winner selections and
tracking progress toward building a personalized trading model.
"""

import streamlit as st
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from pages.common.ui_components import (
    create_section_header, create_info_box, create_success_box,
    create_warning_box, create_error_box, display_data_table
)
from pages.common.constants import DEFAULT_DAYS_BACK
from pages.scanner.constants import ScanScope
from utils.date_utils import format_singapore_date


def show_feature_lab_section():
    """
    Main Feature Lab section with 4 tabs for the complete style learning workflow
    """
    create_section_header("🧪 Feature Lab", "Learn and optimize your trading style")

    # Initialize session state for Feature Lab
    if 'feature_lab_selected_date' not in st.session_state:
        st.session_state.feature_lab_selected_date = date.today() - timedelta(days=7)

    if 'feature_lab_scan_results' not in st.session_state:
        st.session_state.feature_lab_scan_results = None

    if 'feature_lab_bullish_winners' not in st.session_state:
        st.session_state.feature_lab_bullish_winners = []

    if 'feature_lab_bearish_winners' not in st.session_state:
        st.session_state.feature_lab_bearish_winners = []

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Historical Backfill",
        "🔬 Features in Testing",
        "⚖️ Weight Optimization",
        "📊 History & Analytics"
    ])

    with tab1:
        show_historical_backfill_tab()

    with tab2:
        show_features_testing_tab()

    with tab3:
        show_weight_optimization_tab()

    with tab4:
        show_history_analytics_tab()


def show_historical_backfill_tab():
    """
    Tab 1: Historical Backfill - Primary workflow for labeling winners
    """
    st.markdown("### 📅 Historical Backfill")
    st.markdown("**Goal:** Label 60-90 historical dates to train your style")
    st.info("💡 **How it works:** Select a historical date, run the scan, review results, pick your top 3 winners, add notes.")

    # Show progress
    try:
        from .feature_tracker import FeatureTracker
        tracker = FeatureTracker()
        summary = tracker.get_summary_statistics()

        progress = summary.get('completion_progress', {})
        current = progress.get('current', 0)
        target = progress.get('target', 90)
        percentage = progress.get('percentage', 0.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dates Labeled", f"{current}/{target}")
        with col2:
            st.metric("Progress", f"{percentage:.1f}%")
        with col3:
            st.metric("Winners Selected", summary.get('total_winners_selected', 0))

        # Progress bar
        st.progress(min(percentage / 100.0, 1.0))

    except Exception as e:
        st.error(f"Failed to load progress: {e}")

    # Step 1: Date Selection
    st.markdown("---")
    st.markdown("### 📅 Step 1: Select Historical Date")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_date = st.date_input(
            "Analysis Date:",
            value=st.session_state.feature_lab_selected_date,
            max_value=date.today() - timedelta(days=1),
            min_value=date.today() - timedelta(days=180),
            help="Choose a historical date to scan and label winners"
        )
        st.session_state.feature_lab_selected_date = selected_date

    with col2:
        if st.button("◀️ Previous Day", help="Go back one trading day"):
            st.session_state.feature_lab_selected_date = selected_date - timedelta(days=1)
            st.rerun()

    with col3:
        if st.button("Next Day ▶️", help="Go forward one trading day"):
            st.session_state.feature_lab_selected_date = selected_date + timedelta(days=1)
            st.rerun()

    # Step 2: Run Historical Scan
    st.markdown("---")
    st.markdown("### 🔍 Step 2: Run Historical Scan")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**Selected Date:** {selected_date.strftime('%A, %d %B %Y')}")

        # Check if we already have scan results for this date
        has_existing_results = False
        try:
            tracker = FeatureTracker()
            history = tracker.get_selection_history()
            has_existing_results = selected_date.isoformat() in history.get('dates', {})
        except:
            pass

        if has_existing_results:
            st.info("✅ This date has already been labeled. You can re-scan or modify selections below.")
        else:
            st.info("🔄 Ready to scan this historical date.")

    with col2:
        if st.button("🚀 Run Historical Scan", type="primary", width="stretch"):
            run_historical_scan(selected_date)

    # Step 3: Review Results and Select Winners
    if st.session_state.feature_lab_scan_results is not None:
        show_scan_results_and_winner_selection(selected_date)


def run_historical_scan(scan_date: date):
    """
    Execute historical scan for the selected date
    """
    try:
        from pages.scanner.logic import run_enhanced_stock_scan
        from utils.watchlist import get_active_watchlist

        with st.spinner(f"🔄 Scanning {scan_date.strftime('%d/%m/%Y')}..."):
            # Get watchlist
            stocks_to_scan = get_active_watchlist()

            # Run historical scan
            results_df = run_enhanced_stock_scan(
                stocks_to_scan=stocks_to_scan,
                analysis_date=scan_date,
                days_back=DEFAULT_DAYS_BACK,
                rolling_window=20
            )

            if results_df is not None and not results_df.empty:
                # Add ranking columns that are missing from historical scans
                from pages.scanner.logic import add_ranking_columns
                results_df = add_ranking_columns(results_df)

                st.session_state.feature_lab_scan_results = results_df
                st.success(f"✅ Scan completed! Found {len(results_df)} signals.")
                st.rerun()
            else:
                st.error("❌ Scan failed or returned no results.")

    except Exception as e:
        st.error(f"❌ Failed to run historical scan: {e}")


def show_scan_results_and_winner_selection(scan_date: date):
    """
    Display scan results and allow winner selection
    """
    results_df = st.session_state.feature_lab_scan_results

    if results_df is None or results_df.empty:
        st.warning("No scan results available.")
        return

    st.markdown("---")
    st.markdown("### 📊 Step 3: Review Scan Results & Select Winners")

    # Summary stats
    total_signals = len(results_df)
    bullish_signals = len(results_df[results_df['Signal_Bias'] == '🟢 BULLISH'])
    bearish_signals = len(results_df[results_df['Signal_Bias'] == '🔴 BEARISH'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", total_signals)
    with col2:
        st.metric("🟢 Bullish", bullish_signals)
    with col3:
        st.metric("🔴 Bearish", bearish_signals)
    with col4:
        st.metric("⚪ Neutral", total_signals - bullish_signals - bearish_signals)

    # Winner Selection
    st.markdown("#### 🏆 Select Your Top Winners")
    st.info("💡 **Instructions:** Check the boxes next to stocks that actually performed well. Focus on stocks you would have traded based on the signals.")

    # Bullish Signals Selection
    if bullish_signals > 0:
        st.markdown("**🟢 Bullish Signals - Select up to 3 winners:**")

        bullish_df = results_df[results_df['Signal_Bias'] == '🟢 BULLISH'].copy()
        bullish_df = bullish_df.sort_values('Trade_Rank').head(10)  # Show top 10

        for _, row in bullish_df.iterrows():
            ticker = row['Ticker']
            name = row.get('Name', ticker)
            score = row.get('Signal_Score', 0)
            rank = row.get('Trade_Rank', 999)  # Use 999 for unranked

            option_text = f"{ticker} - {name} (Score: {score:.1f}, Rank: #{rank})"

            # Use callback for proper state management
            def toggle_bullish_winner(ticker=ticker):
                if ticker in st.session_state.feature_lab_bullish_winners:
                    st.session_state.feature_lab_bullish_winners.remove(ticker)
                else:
                    st.session_state.feature_lab_bullish_winners.append(ticker)

            st.checkbox(
                option_text,
                key=f"bullish_{ticker}",
                value=ticker in st.session_state.feature_lab_bullish_winners,
                on_change=toggle_bullish_winner
            )

    # Bearish Signals Selection
    if bearish_signals > 0:
        st.markdown("**🔴 Bearish Signals - Select up to 3 winners:**")

        bearish_df = results_df[results_df['Signal_Bias'] == '🔴 BEARISH'].copy()
        bearish_df = bearish_df.sort_values('Trade_Rank').head(10)  # Show top 10

        for _, row in bearish_df.iterrows():
            ticker = row['Ticker']
            name = row.get('Name', ticker)
            score = row.get('Signal_Score', 0)
            rank = row.get('Trade_Rank', 999)  # Use 999 for unranked

            option_text = f"{ticker} - {name} (Score: {score:.1f}, Rank: #{rank})"

            # Use callback for proper state management
            def toggle_bearish_winner(ticker=ticker):
                if ticker in st.session_state.feature_lab_bearish_winners:
                    st.session_state.feature_lab_bearish_winners.remove(ticker)
                else:
                    st.session_state.feature_lab_bearish_winners.append(ticker)

            st.checkbox(
                option_text,
                key=f"bearish_{ticker}",
                value=ticker in st.session_state.feature_lab_bearish_winners,
                on_change=toggle_bearish_winner
            )

    # Selection Notes
    st.markdown("#### 📝 Why Did These Work?")
    selection_notes = st.text_area(
        "Add your commentary on why these selections worked:",
        value=getattr(st.session_state, 'feature_lab_notes', ''),
        height=100,
        placeholder="Example: D05 had clean breakout above MA20 with volume confirmation. C52 earnings beat drove momentum continuation...",
        help="Your notes help the system learn your trading style patterns."
    )
    st.session_state.feature_lab_notes = selection_notes

    # Save Button
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        total_selected = len(st.session_state.feature_lab_bullish_winners) + len(st.session_state.feature_lab_bearish_winners)
        if total_selected == 0:
            st.error("❌ Please select at least one winner.")
        elif total_selected > 6:
            st.warning(f"⚠️ You selected {total_selected} winners. Consider focusing on your top 3-6 strongest convictions.")
        else:
            st.success(f"✅ Ready to save {total_selected} winners for {scan_date.strftime('%d/%m/%Y')}.")

    with col2:
        can_save = total_selected > 0 and selection_notes.strip()
        if st.button(
            "💾 Save Winners & Continue",
            type="primary",
            width="stretch",
            disabled=not can_save
        ):
            save_winners_and_continue(scan_date, selection_notes)

    with col3:
        if st.button("🔄 Clear Selections", width="stretch"):
            clear_selections()

    # Recent Selections
    show_recent_selections()


def save_winners_and_continue(scan_date: date, notes: str):
    """
    Save the winner selections to the feature tracker
    """
    try:
        from .feature_tracker import FeatureTracker

        tracker = FeatureTracker()

        # Save to tracker
        success = tracker.record_historical_winners(
            scan_date=scan_date,
            scan_results=st.session_state.feature_lab_scan_results,
            bullish_winners=st.session_state.feature_lab_bullish_winners,
            bearish_winners=st.session_state.feature_lab_bearish_winners,
            selection_notes=notes,
            scoring_system="production"
        )

        if success:
            st.success(f"✅ Successfully saved winners for {scan_date.strftime('%d/%m/%Y')}!")
            st.balloons()

            # Clear selections for next date
            clear_selections()

            # Move to next day automatically
            st.session_state.feature_lab_selected_date = scan_date + timedelta(days=1)

            # Clear scan results to force re-scan
            st.session_state.feature_lab_scan_results = None

            st.rerun()
        else:
            st.error("❌ Failed to save winners. Please try again.")

    except Exception as e:
        st.error(f"❌ Error saving winners: {e}")


def clear_selections():
    """Clear all winner selections and notes"""
    st.session_state.feature_lab_bullish_winners = []
    st.session_state.feature_lab_bearish_winners = []
    st.session_state.feature_lab_notes = ""


def show_recent_selections():
    """
    Display recent selections for reference
    """
    st.markdown("---")
    st.markdown("### 📚 Recent Selections")

    try:
        from .feature_tracker import FeatureTracker

        tracker = FeatureTracker()
        history = tracker.get_selection_history(days_back=30)  # Last 30 days

        dates = history.get('dates', {})

        if not dates:
            st.info("No recent selections. Start by scanning and labeling historical dates above.")
            return

        # Sort by date (most recent first)
        sorted_dates = sorted(dates.keys(), reverse=True)[:5]  # Show last 5

        for date_str in sorted_dates:
            date_data = dates[date_str]
            bullish_count = len(date_data.get('bullish_winners', []))
            bearish_count = len(date_data.get('bearish_winners', []))
            notes = date_data.get('selection_notes', '')[:100]  # Truncate long notes

            with st.expander(f"📅 {date_str}: {bullish_count + bearish_count} winners", expanded=False):
                st.write(f"**Bullish Winners:** {', '.join(date_data.get('bullish_winners', []))}")
                if date_data.get('bearish_winners'):
                    st.write(f"**Bearish Winners:** {', '.join(date_data.get('bearish_winners', []))}")
                if notes:
                    st.write(f"**Notes:** {notes}{'...' if len(date_data.get('selection_notes', '')) > 100 else ''}")

                # Delete button
                if st.button(f"🗑️ Delete {date_str}", key=f"delete_{date_str}"):
                    if st.session_state.get(f"confirm_delete_{date_str}", False):
                        # Actually delete
                        if tracker.delete_date_selection(datetime.fromisoformat(date_str).date()):
                            st.success(f"Deleted selection for {date_str}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {date_str}")
                    else:
                        st.session_state[f"confirm_delete_{date_str}"] = True
                        st.warning(f"⚠️ Click again to confirm deletion of {date_str}")

    except Exception as e:
        st.error(f"Failed to load recent selections: {e}")


def show_features_testing_tab():
    """Tab 2: Features in Testing - Placeholder for now"""
    st.markdown("### 🔬 Features in Testing")
    st.info("🚧 This tab will be implemented in Phase 2. For now, focus on collecting historical data in the Historical Backfill tab.")


def show_weight_optimization_tab():
    """Tab 3: Weight Optimization - Placeholder for now"""
    st.markdown("### ⚖️ Weight Optimization")
    st.info("🚧 This tab will be implemented in Phase 3. First, collect 60-90 days of labeled data.")


def show_history_analytics_tab():
    """Tab 4: History & Analytics - Placeholder for now"""
    st.markdown("### 📊 History & Analytics")
    st.info("🚧 This tab will be implemented in Phase 4. Continue building your historical dataset.")

    # Show basic stats for now
    try:
        from .feature_tracker import FeatureTracker
        tracker = FeatureTracker()
        summary = tracker.get_summary_statistics()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dates Labeled", summary.get('total_dates_labeled', 0))
        with col2:
            st.metric("Total Winners", summary.get('total_winners_selected', 0))
        with col3:
            progress = summary.get('completion_progress', {})
            st.metric("Progress", f"{progress.get('percentage', 0):.1f}%")

    except Exception as e:
        st.error(f"Failed to load analytics: {e}")
