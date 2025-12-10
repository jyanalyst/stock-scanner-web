"""
ML Lab Phase 5: Standalone ML Scanner
Isolated ML-powered scanner for testing without affecting production
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Optional


def show_phase5():
    """Display Phase 5: Standalone ML Scanner"""
    
    with st.expander("🚀 PHASE 5: ML Scanner (Standalone)", expanded=False):
        st.markdown("""
        **Goal:** Test ML predictions in isolated environment without affecting production scanner
        
        **Features:**
        - Scan stocks with ML predictions
        - Compare ML vs Technical signals
        - Filter by ML confidence
        - Analyze agreement/divergence
        
        **Status:** Model validated with 52.5% accuracy, 53% win rate, 3.22 profit factor
        """)
        
        # Check if model is available
        from ml.scanner_integration import ScannerMLIntegration
        
        ml_integration = ScannerMLIntegration()
        
        if not ml_integration.is_ml_available():
            st.error("❌ ML model not available. Please complete Phase 3 (Model Training) first.")
            return
        
        # Show model status
        status = ml_integration.get_ml_status()
        
        col_status1, col_status2, col_status3, col_status4 = st.columns(4)
        with col_status1:
            st.metric("Model Status", "✅ Ready")
        with col_status2:
            st.metric("Accuracy", f"{status['accuracy']:.1%}")
        with col_status3:
            st.metric("Win Rate", f"{status['win_rate']:.1%}")
        with col_status4:
            st.metric("Threshold", f"{status['threshold']:.2f}")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "⚙️ Configuration & Run",
            "📊 ML Signals & Results",
            "🎯 Analysis & Insights"
        ])
        
        with tab1:
            show_configuration_tab(ml_integration)
        
        with tab2:
            show_results_tab(ml_integration)
        
        with tab3:
            show_analysis_tab(ml_integration)


def show_configuration_tab(ml_integration):
    """Configuration and Run Tab"""
    st.markdown("### ⚙️ ML Scanner Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Scan Scope**")
        scan_scope = st.radio(
            "Select stocks to scan:",
            ["📋 Full Watchlist", "🎯 Single Stock", "📝 Custom List"],
            help="Choose which stocks to analyze with ML"
        )
        
        if scan_scope == "🎯 Single Stock":
            from utils.watchlist import get_active_watchlist
            watchlist = get_active_watchlist()
            selected_stock = st.selectbox("Select Stock:", watchlist)
            stocks_to_scan = [selected_stock]
        elif scan_scope == "📝 Custom List":
            custom_stocks = st.text_area(
                "Enter tickers (one per line):",
                value="A17U.SG\nD05.SG\nO39.SG",
                help="Enter stock tickers, one per line"
            )
            stocks_to_scan = [s.strip() for s in custom_stocks.split('\n') if s.strip()]
        else:
            from utils.watchlist import get_active_watchlist
            stocks_to_scan = get_active_watchlist()
    
    with col2:
        st.markdown("**Analysis Settings**")
        
        analysis_date_type = st.radio(
            "Analysis Date:",
            ["📅 Current (Latest Data)", "📆 Historical Date"],
            help="Analyze current data or specific historical date"
        )
        
        if analysis_date_type == "📆 Historical Date":
            historical_date = st.date_input(
                "Select Date:",
                value=date.today() - timedelta(days=7),
                max_value=date.today() - timedelta(days=1)
            )
        else:
            historical_date = None
        
        ml_threshold = st.slider(
            "ML Confidence Threshold:",
            min_value=0.50,
            max_value=0.75,
            value=0.60,
            step=0.05,
            help="Minimum confidence for BUY signal (0.60 = optimal from validation)"
        )
    
    st.markdown("---")
    st.markdown("### 🚀 Run ML Scan")
    
    st.info(f"""
    **Scan Summary:**
    - Stocks: {len(stocks_to_scan)}
    - Date: {'Current' if historical_date is None else historical_date.strftime('%Y-%m-%d')}
    - ML Threshold: {ml_threshold:.2f}
    - Expected runtime: ~{len(stocks_to_scan) * 0.5:.0f} seconds
    """)
    
    if st.button("🚀 Run ML Scan", type="primary"):
        run_ml_scan(stocks_to_scan, historical_date, ml_threshold, ml_integration)


def run_ml_scan(stocks_to_scan: List[str], 
                analysis_date: Optional[date],
                ml_threshold: float,
                ml_integration):
    """Execute ML scan"""
    
    with st.spinner(f"🔄 Scanning {len(stocks_to_scan)} stocks with ML predictions..."):
        try:
            # Import scanner logic
            from pages.scanner.logic import run_enhanced_stock_scan
            
            # Run scanner (this will calculate all technical indicators)
            results_df = run_enhanced_stock_scan(
                stocks_to_scan=stocks_to_scan,
                analysis_date=analysis_date,
                days_back=59,
                rolling_window=20
            )
            
            if results_df is None or results_df.empty:
                st.error("❌ Scan failed - no results returned")
                return
            
            # Add ML predictions with custom threshold
            ml_integration.threshold = ml_threshold
            results_with_ml = ml_integration.add_ml_predictions(results_df)
            
            # Store in session state
            st.session_state.ml_scan_results = results_with_ml
            st.session_state.ml_scan_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.ml_scan_config = {
                'stocks_count': len(stocks_to_scan),
                'date': 'Current' if analysis_date is None else analysis_date.strftime('%Y-%m-%d'),
                'threshold': ml_threshold
            }
            
            # Get summary
            summary = ml_integration.get_ml_summary(results_with_ml)
            
            st.success(f"✅ ML Scan complete! Found {summary['ml_buy_count']} BUY signals out of {len(results_with_ml)} stocks")
            
            # Force refresh to show results
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ ML Scan failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def show_results_tab(ml_integration):
    """ML Signals & Results Tab"""
    st.markdown("### 📊 ML Signals & Results")
    
    if 'ml_scan_results' not in st.session_state:
        st.info("👈 Run ML scan in Configuration tab to see results")
        return
    
    results_df = st.session_state.ml_scan_results
    config = st.session_state.ml_scan_config
    timestamp = st.session_state.ml_scan_timestamp
    
    # Show scan info
    st.caption(f"Scan completed: {timestamp} | {config['stocks_count']} stocks | Threshold: {config['threshold']:.2f}")
    
    # Get summary
    summary = ml_integration.get_ml_summary(results_df)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", summary['total_stocks'])
    with col2:
        st.metric("🤖 ML Buy Signals", summary['ml_buy_count'])
    with col3:
        st.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}" if summary['ml_buy_count'] > 0 else "—")
    with col4:
        st.metric("High Confidence", summary['high_confidence_count'], help="Confidence ≥ 0.70")
    
    st.markdown("---")
    
    # Filters
    st.markdown("#### 🔍 Filters")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        signal_filter = st.selectbox(
            "Signal Type:",
            ["All Stocks", "🤖 ML Buy Only", "Tech + ML Agree", "Divergence Alerts"],
            help="Filter by signal type"
        )
    
    with col_f2:
        confidence_filter = st.selectbox(
            "Confidence:",
            ["All", "⭐⭐⭐ High (≥0.70)", "⭐⭐ Good (0.60-0.70)", "⭐ Marginal (0.55-0.60)"],
            help="Filter by confidence tier"
        )
    
    with col_f3:
        sort_by = st.selectbox(
            "Sort By:",
            ["ML Confidence (High→Low)", "Win Probability", "Ticker"],
            help="Sort results"
        )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if signal_filter == "🤖 ML Buy Only":
        filtered_df = filtered_df[filtered_df['ML_Signal'] == '🤖 BUY']
    elif signal_filter == "Tech + ML Agree":
        filtered_df = filtered_df[
            (filtered_df['Signal_Bias'] == '🟢 BULLISH') &
            (filtered_df['ML_Signal'] == '🤖 BUY')
        ]
    elif signal_filter == "Divergence Alerts":
        filtered_df = filtered_df[
            ((filtered_df['Signal_Bias'] == '🟢 BULLISH') & (filtered_df['ML_Signal'] != '🤖 BUY')) |
            ((filtered_df['Signal_Bias'] == '🔴 BEARISH') & (filtered_df['ML_Signal'] == '🤖 BUY'))
        ]
    
    if confidence_filter == "⭐⭐⭐ High (≥0.70)":
        filtered_df = filtered_df[filtered_df['ML_Confidence'] >= 0.70]
    elif confidence_filter == "⭐⭐ Good (0.60-0.70)":
        filtered_df = filtered_df[(filtered_df['ML_Confidence'] >= 0.60) & (filtered_df['ML_Confidence'] < 0.70)]
    elif confidence_filter == "⭐ Marginal (0.55-0.60)":
        filtered_df = filtered_df[(filtered_df['ML_Confidence'] >= 0.55) & (filtered_df['ML_Confidence'] < 0.60)]
    
    # Apply sorting
    if sort_by == "ML Confidence (High→Low)":
        filtered_df = filtered_df.sort_values('ML_Confidence', ascending=False)
    elif sort_by == "Win Probability":
        filtered_df = filtered_df.sort_values('ML_Win_Probability', ascending=False)
    else:
        filtered_df = filtered_df.sort_values('Ticker')
    
    st.info(f"📊 Showing {len(filtered_df)} stocks after filters")
    
    # Display results table
    if len(filtered_df) > 0:
        display_cols = [
            'Ticker', 'Name', 'Close',
            'ML_Signal', 'ML_Win_Probability', 'ML_Confidence', 'ML_Confidence_Tier',
            'Signal_Bias', 'Signal_Score',
            'Flow_Velocity_Rank', 'Flow_Rank'
        ]
        
        # Filter to available columns
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        column_config = {
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Name': st.column_config.TextColumn('Company', width='medium'),
            'Close': st.column_config.NumberColumn('Close', format='%.3f', width='small'),
            'ML_Signal': st.column_config.TextColumn('🤖 ML', width='small', help='ML Buy signal'),
            'ML_Win_Probability': st.column_config.NumberColumn('Win %', format='%.1f%%', width='small'),
            'ML_Confidence': st.column_config.NumberColumn('Confidence', format='%.2f', width='small'),
            'ML_Confidence_Tier': st.column_config.TextColumn('⭐ Tier', width='small'),
            'Signal_Bias': st.column_config.TextColumn('Tech Signal', width='small'),
            'Signal_Score': st.column_config.NumberColumn('Tech Score', format='%.1f', width='small'),
            'Flow_Velocity_Rank': st.column_config.NumberColumn('Flow Vel', format='%.1f', width='small'),
            'Flow_Rank': st.column_config.NumberColumn('Flow Rank', format='%.1f', width='small'),
        }
        
        st.dataframe(
            filtered_df[display_cols],
            column_config=column_config,
            width='stretch',
            hide_index=True
        )
        
        # Export options
        st.markdown("---")
        st.markdown("#### 📥 Export Results")
        
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📊 Download ML Signals (CSV)",
            data=csv_data,
            file_name=f"ml_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No stocks match the current filters")


def show_analysis_tab(ml_integration):
    """Analysis & Insights Tab"""
    st.markdown("### 🎯 Analysis & Insights")
    
    if 'ml_scan_results' not in st.session_state:
        st.info("👈 Run ML scan in Configuration tab to see analysis")
        return
    
    results_df = st.session_state.ml_scan_results
    summary = ml_integration.get_ml_summary(results_df)
    
    if not summary['available']:
        st.warning("ML predictions not available in results")
        return
    
    # Overall Summary
    st.markdown("#### 📊 ML Prediction Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Stocks", summary['total_stocks'])
    with col2:
        st.metric("🤖 ML Buy", summary['ml_buy_count'])
    with col3:
        st.metric("Avg Confidence", f"{summary['avg_confidence']:.1%}" if summary['ml_buy_count'] > 0 else "—")
    with col4:
        st.metric("Expected Win Rate", f"{summary['expected_win_rate']:.1%}")
    with col5:
        st.metric("Profit Factor", f"{summary['expected_profit_factor']:.2f}")
    
    st.markdown("---")
    
    # Agreement Analysis
    st.markdown("#### 🤝 Technical vs ML Agreement")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        st.metric(
            "✅ Agree (Both Bullish)",
            summary['tech_ml_agree'],
            help="Both technical and ML signals are bullish"
        )
    
    with col_a2:
        agreement_rate = summary['agreement_rate']
        st.metric(
            "Agreement Rate",
            f"{agreement_rate:.0f}%",
            help="Percentage of ML signals that agree with technical"
        )
    
    with col_a3:
        ml_only = summary['ml_buy_count'] - summary['tech_ml_agree']
        st.metric(
            "🤖 ML Only",
            ml_only,
            help="ML bullish but technical neutral/bearish"
        )
    
    # Show agreement breakdown
    if 'Signal_Bias' in results_df.columns and 'ML_Signal' in results_df.columns:
        st.markdown("##### Signal Breakdown")
        
        ml_buy_df = results_df[results_df['ML_Signal'] == '🤖 BUY']
        
        if len(ml_buy_df) > 0:
            tech_bullish = (ml_buy_df['Signal_Bias'] == '🟢 BULLISH').sum()
            tech_neutral = (ml_buy_df['Signal_Bias'] == '⚪ NEUTRAL').sum()
            tech_bearish = (ml_buy_df['Signal_Bias'] == '🔴 BEARISH').sum()
            
            breakdown_data = pd.DataFrame({
                'Category': ['✅ Tech + ML Bullish', '⚪ ML Bullish, Tech Neutral', '⚠️ ML Bullish, Tech Bearish'],
                'Count': [tech_bullish, tech_neutral, tech_bearish],
                'Percentage': [
                    f"{tech_bullish/len(ml_buy_df)*100:.0f}%",
                    f"{tech_neutral/len(ml_buy_df)*100:.0f}%",
                    f"{tech_bearish/len(ml_buy_df)*100:.0f}%"
                ]
            })
            
            st.dataframe(breakdown_data, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Confidence Distribution
    st.markdown("#### ⭐ Confidence Distribution")
    
    if summary['ml_buy_count'] > 0:
        ml_buy_df = results_df[results_df['ML_Signal'] == '🤖 BUY']
        
        high_conf = (ml_buy_df['ML_Confidence'] >= 0.70).sum()
        good_conf = ((ml_buy_df['ML_Confidence'] >= 0.60) & (ml_buy_df['ML_Confidence'] < 0.70)).sum()
        marginal_conf = ((ml_buy_df['ML_Confidence'] >= 0.55) & (ml_buy_df['ML_Confidence'] < 0.60)).sum()
        
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.metric("⭐⭐⭐ High", high_conf, help="Confidence ≥ 0.70")
        with col_c2:
            st.metric("⭐⭐ Good", good_conf, help="Confidence 0.60-0.70")
        with col_c3:
            st.metric("⭐ Marginal", marginal_conf, help="Confidence 0.55-0.60")
        
        # Show confidence chart
        conf_data = pd.DataFrame({
            'Tier': ['⭐⭐⭐ High', '⭐⭐ Good', '⭐ Marginal'],
            'Count': [high_conf, good_conf, marginal_conf]
        })
        
        st.bar_chart(conf_data.set_index('Tier'))
    
    st.markdown("---")
    
    # Top ML Picks
    st.markdown("#### 🏆 Top ML Picks")
    
    if summary['ml_buy_count'] > 0:
        ml_buy_df = results_df[results_df['ML_Signal'] == '🤖 BUY'].copy()
        top_picks = ml_buy_df.nlargest(5, 'ML_Confidence')
        
        for idx, row in top_picks.iterrows():
            ticker = row['Ticker']
            name = row.get('Name', ticker)
            confidence = row['ML_Confidence']
            win_prob = row['ML_Win_Probability']
            close = row['Close']
            target = close * 1.05  # +5% target
            
            tier_emoji = '⭐⭐⭐' if confidence >= 0.70 else '⭐⭐' if confidence >= 0.60 else '⭐'
            
            st.write(f"{tier_emoji} **{ticker}** - {name}")
            st.write(f"   → Confidence: {confidence:.1%} | Win Prob: {win_prob:.1f}% | Entry: ${close:.3f} | Target: ${target:.3f} (+5%)")
    else:
        st.info("No ML Buy signals found in current scan")
    
    st.markdown("---")
    
    # Trading Recommendations
    st.markdown("#### 💡 Trading Recommendations")
    
    if summary['ml_buy_count'] > 0:
        st.success(f"""
        **✅ {summary['ml_buy_count']} ML Buy signals detected**
        
        **Expected Performance:**
        - Win Rate: {summary['expected_win_rate']:.1%} (53% validated)
        - Profit Factor: {summary['expected_profit_factor']:.2f} (3.22 validated)
        - Average Win Probability: {summary['avg_win_prob']:.1f}%
        
        **Recommendation:**
        - Focus on ⭐⭐⭐ High confidence signals (≥0.70)
        - Prioritize stocks where Tech + ML agree
        - Use 3-day holding period (model trained on 3-day returns)
        - Position size based on confidence tier
        """)
    else:
        st.info("""
        **No ML Buy signals found**
        
        **Possible reasons:**
        - Market conditions not favorable
        - Threshold too high (try lowering to 0.55)
        - Stocks not meeting ML criteria
        
        **Recommendation:**
        - Wait for better setups
        - Review technical signals
        - Consider lowering ML threshold
        """)
