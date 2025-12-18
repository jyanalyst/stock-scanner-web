"""
Feature Lab UI Components - Streamlit interface for style learning system

Provides the user interface for recording historical winner selections and
tracking progress toward building a personalized trading model.
"""

import streamlit as st
import json
import yaml
from pathlib import Path
import plotly.graph_objects as go
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

    if 'feature_lab_outcomes' not in st.session_state:
        st.session_state.feature_lab_outcomes = {}

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
        from .target_outcome import calculate_outcomes_for_scan

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
                
                # Calculate target outcomes
                with st.spinner("Calculating target outcomes..."):
                    outcomes = calculate_outcomes_for_scan(scan_date, results_df)
                    st.session_state.feature_lab_outcomes = outcomes

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
    outcomes = st.session_state.get('feature_lab_outcomes', {})

    if results_df is None or results_df.empty:
        st.warning("No scan results available.")
        return

    st.markdown("---")
    st.markdown("### 📊 Step 3: Review Scan Results & Select Winners")

    # Summary stats
    total_signals = len(results_df)
    bullish_signals = len(results_df[results_df['Signal_Bias'] == '🟢 BULLISH'])
    bearish_signals = len(results_df[results_df['Signal_Bias'] == '🔴 BEARISH'])

    # Outcome stats
    true_breaks = sum(1 for o in outcomes.values() if o.get('outcome') == 'TRUE_BREAK')
    invalidations = sum(1 for o in outcomes.values() if o.get('outcome') == 'INVALIDATION')
    timeouts = sum(1 for o in outcomes.values() if o.get('outcome') == 'TIMEOUT')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", total_signals)
    with col2:
        st.metric("🟢 Bullish", bullish_signals)
    with col3:
        st.metric("🔴 Bearish", bearish_signals)
    with col4:
        st.metric("✅ True Breaks", true_breaks)

    # Winner Selection
    st.markdown("#### 🏆 Select Your Top Winners")
    st.info("💡 **Instructions:** Check the boxes next to stocks that actually performed well. Focus on stocks you would have traded based on the signals.")

    def get_outcome_badge(ticker):
        if ticker not in outcomes:
            return "❓"
        
        outcome = outcomes[ticker].get('outcome')
        return_pct = outcomes[ticker].get('return_pct', 0.0)
        day = outcomes[ticker].get('day', 0)
        
        badge = "❓"
        if outcome == 'TRUE_BREAK':
            badge = "✅"
        elif outcome == 'INVALIDATION':
            badge = "❌"
        elif outcome == 'TIMEOUT':
            badge = "⏱️"
        elif outcome == 'INSUFFICIENT_DATA':
            badge = "⚠️"
            
        return f"{badge} {outcome} (Day {day}, {return_pct:+.1f}%)"

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
            
            outcome_text = get_outcome_badge(ticker)
            option_text = f"{ticker} - {name} (Score: {score:.1f}, Rank: #{rank}) | {outcome_text}"

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

            outcome_text = get_outcome_badge(ticker)
            option_text = f"{ticker} - {name} (Score: {score:.1f}, Rank: #{rank}) | {outcome_text}"

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
            scoring_system="production",
            target_outcomes=st.session_state.get('feature_lab_outcomes', {})
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
            st.session_state.feature_lab_outcomes = {}

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
    # Don't clear outcomes here as they might be needed if user clears just selections


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


def _calculate_feature_for_ui(feature_name: str):
    """Calculate feature for all historical data (UI helper)"""
    try:
        from pages.scanner.feature_lab.feature_tracker import FeatureTracker

        tracker = FeatureTracker()

        # Start tracking if not already
        if not tracker.start_feature_tracking(feature_name):
            st.error(f"Failed to start tracking for {feature_name}")
            return

        # Calculate feature
        with st.spinner(f"Calculating {feature_name} for all historical dates..."):
            success = tracker.calculate_feature_for_all_history(feature_name)

        if success:
            st.success(f"✅ Successfully calculated {feature_name} for all historical selections!")
            st.rerun()
        else:
            st.error(f"❌ Failed to calculate {feature_name}")

    except Exception as e:
        st.error(f"Error calculating feature: {e}")


def _analyze_feature_for_ui(feature_name: str):
    """Analyze feature significance (UI helper)"""
    try:
        from pages.scanner.feature_lab.feature_tracker import FeatureTracker

        tracker = FeatureTracker()

        with st.spinner(f"Analyzing statistical significance of {feature_name}..."):
            analysis = tracker.analyze_feature_significance(feature_name)

        if 'error' not in analysis:
            st.session_state.current_feature_analysis = analysis
            st.success("✅ Analysis completed!")
            st.rerun()
        else:
            st.error(f"❌ Analysis failed: {analysis['error']}")

    except Exception as e:
        st.error(f"Error analyzing feature: {e}")


def show_features_testing_tab():
    """Tab 2: Features in Testing - Analyze experimental features"""
    st.markdown("### 🔬 Features in Testing")
    st.markdown("**Goal:** Test experimental features to see if they distinguish winners from non-winners")

    try:
        # Load feature config
        config_path = Path("configs/feature_config.yaml")
        if not config_path.exists():
            st.error("Feature config not found")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        experimental_features = config.get('experimental_features', {})

        # Load features testing data
        testing_path = Path("data/feature_lab/features_testing.json")
        if testing_path.exists():
            with open(testing_path, 'r') as f:
                testing_data = json.load(f)
        else:
            testing_data = {"features": {}}

        # Section 1: Feature Selection and Calculation
        st.markdown("#### 1️⃣ Select Feature to Test")

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            available_features = list(experimental_features.keys())
            selected_feature = st.selectbox(
                "Choose experimental feature:",
                available_features,
                help="Select a feature to test against your historical winner selections"
            )

            if selected_feature:
                feature_info = experimental_features[selected_feature]
                st.info(f"**{feature_info['name']}**: {feature_info['description']}")

        with col2:
            if st.button("🚀 Calculate Feature", type="primary", use_container_width=True):
                _calculate_feature_for_ui(selected_feature)

        with col3:
            if st.button("📊 Analyze Results", use_container_width=True):
                _analyze_feature_for_ui(selected_feature)

        # Section 2: Features Status
        st.markdown("#### 2️⃣ Features Testing Status")

        if testing_data.get('features'):
            for feature_name, feature_data in testing_data['features'].items():
                status = feature_data.get('status', 'unknown')
                status_emoji = {
                    'initialized': '📝',
                    'calculated': '✅',
                    'analyzed': '📊',
                    'calculation_failed': '❌',
                    'analysis_failed': '❌'
                }.get(status, '❓')

                with st.expander(f"{status_emoji} {feature_name} - {status.replace('_', ' ').title()}", expanded=False):
                    st.write(f"**Started:** {feature_data.get('started_at', 'Unknown')[:19]}")

                    if status == 'analyzed':
                        analysis = feature_data.get('analysis_results', {})
                        if analysis:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Winner Mean", f"{analysis.get('winner_mean', 0):.2f}")
                            with col2:
                                st.metric("Non-Winner Mean", f"{analysis.get('non_winner_mean', 0):.2f}")
                            with col3:
                                recommendation = analysis.get('recommendation_text', 'Unknown')
                                if 'STRONG' in recommendation:
                                    st.success(recommendation)
                                elif 'NOT' in recommendation:
                                    st.error(recommendation)
                                else:
                                    st.warning(recommendation)
        else:
            st.info("No features have been tested yet. Select a feature above and click 'Calculate Feature' to begin.")

        # Section 3: Current Analysis Results
        st.markdown("#### 3️⃣ Analysis Results")

        # Get current analysis from session state or testing data
        current_analysis = getattr(st.session_state, 'current_feature_analysis', None)

        if current_analysis and 'error' not in current_analysis:
            feature_name = current_analysis.get('feature_name', 'Unknown')

            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Winner Mean", f"{current_analysis.get('winner_mean', 0):.2f}")
            with col2:
                st.metric("Non-Winner Mean", f"{current_analysis.get('non_winner_mean', 0):.2f}")
            with col3:
                p_value = current_analysis.get('mann_whitney_p', 1.0)
                st.metric("P-Value", f"{p_value:.4f}")
            with col4:
                cohens_d = current_analysis.get('cohens_d', 0.0)
                st.metric("Cohen's d", f"{cohens_d:.2f}")

            # Recommendation
            recommendation = current_analysis.get('recommendation_text', '')
            if recommendation:
                if 'STRONG' in recommendation:
                    st.success(f"🎯 **{feature_name}**: {recommendation}")
                elif 'NOT' in recommendation:
                    st.error(f"❌ **{feature_name}**: {recommendation}")
                elif 'REDUNDANT' in recommendation:
                    st.warning(f"⚠️ **{feature_name}**: {recommendation}")
                else:
                    st.info(f"🤔 **{feature_name}**: {recommendation}")

            # Box Plot
            st.markdown("**Distribution Comparison:**")
            winner_values = current_analysis.get('winners_values', [])
            non_winner_values = current_analysis.get('non_winners_values', [])

            if winner_values and non_winner_values:
                fig = go.Figure()

                fig.add_trace(go.Box(
                    y=winner_values,
                    name="Winners",
                    marker_color='green',
                    boxmean=True
                ))

                fig.add_trace(go.Box(
                    y=non_winner_values,
                    name="Non-Winners",
                    marker_color='red',
                    boxmean=True
                ))

                fig.update_layout(
                    title=f"{feature_name} Distribution: Winners vs Non-Winners",
                    yaxis_title=feature_name,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

        elif current_analysis and 'error' in current_analysis:
            st.error(f"Analysis failed: {current_analysis['error']}")
        else:
            st.info("Click 'Analyze Results' to see statistical analysis of the selected feature.")

        # Section 4: Action Buttons
        st.markdown("#### 4️⃣ Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Validate Feature", use_container_width=True,
                        disabled=not (current_analysis and current_analysis.get('recommendation') == 'STRONG_CANDIDATE')):
                st.success("Feature validated and ready for weight optimization!")
                # TODO: Move to optimization phase

        with col2:
            if st.button("🗑️ Discard Feature", use_container_width=True):
                if current_analysis:
                    # Remove from testing data
                    if selected_feature in testing_data.get('features', {}):
                        del testing_data['features'][selected_feature]
                        testing_data['last_modified'] = datetime.now().isoformat()

                        with open(testing_path, 'w') as f:
                            json.dump(testing_data, f, indent=2)

                        st.success(f"Discarded {selected_feature} from testing")
                        st.rerun()
                else:
                    st.warning("No feature selected for discarding")

        with col3:
            if st.button("📤 Export Results", use_container_width=True):
                if current_analysis:
                    # Create export data
                    export_data = {
                        "feature_analysis": current_analysis,
                        "exported_at": datetime.now().isoformat(),
                        "version": "1.0"
                    }

                    # Convert to JSON and offer download
                    json_str = json.dumps(export_data, indent=2)

                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"feature_analysis_{selected_feature}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No analysis results to export")

    except Exception as e:
        st.error(f"Failed to load features testing interface: {e}")
        st.info("Make sure all required files are present: configs/feature_config.yaml, data/feature_lab/features_testing.json")


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
