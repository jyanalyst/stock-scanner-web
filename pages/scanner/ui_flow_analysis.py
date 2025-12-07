"""
Institutional Flow Analysis UI Component
Separate expander for organized display of Phase 1 institutional flow metrics
"""

import streamlit as st
import pandas as pd
from pages.common.ui_components import create_section_header, create_dynamic_column_config


def show_institutional_flow_analysis(filtered_stocks: pd.DataFrame) -> None:
    """
    Display institutional flow analysis in a separate expander
    Organizes the 10 flow metrics into 3 logical sections
    """
    # Check if flow data exists
    if 'Flow_10D' not in filtered_stocks.columns:
        return
    
    with st.expander("🏦 Detailed Institutional Flow Analysis", expanded=False):
        st.markdown("""
        **Phase 1: Institutional Order Flow Metrics**
        
        This analysis tracks institutional buying/selling patterns through volume-weighted flow calculations,
        conviction metrics, and price-flow divergence signals.
        """)
        
        # ===== SECTION 1: FLOW ANALYSIS =====
        st.markdown("---")
        st.markdown("### 🌊 Flow Analysis")
        st.caption("Volume-weighted directional flow tracking institutional accumulation/distribution")
        
        # Include both raw, individual percentiles, and cross-stock rankings for comprehensive analysis
        flow_cols = ['Ticker', 'Name', 'Daily_Flow', 'Flow_10D', 'Flow_Percentile', 'Flow_Rank', 'Flow_Velocity', 'Flow_Velocity_Percentile', 'Flow_Velocity_Rank', 'Flow_Regime']
        flow_data = filtered_stocks[[col for col in flow_cols if col in filtered_stocks.columns]].copy()

        # Sort by Flow_Rank descending (strongest relative accumulation first)
        if 'Flow_Rank' in flow_data.columns:
            flow_data = flow_data.sort_values('Flow_Rank', ascending=False)
        elif 'Flow_10D' in flow_data.columns:
            flow_data = flow_data.sort_values('Flow_10D', ascending=False)

        flow_config = {
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Name': st.column_config.TextColumn('Company', width='medium'),
            'Daily_Flow': st.column_config.NumberColumn(
                'Daily Flow',
                format='%+.1f',
                help='Volume-weighted directional flow for current day (+buy pressure/-sell pressure)'
            ),
            'Flow_10D': st.column_config.NumberColumn(
                'Flow 10D',
                format='%+.1f',
                help='10-day cumulative institutional flow (positive=accumulation, negative=distribution)'
            ),
            'Flow_Percentile': st.column_config.NumberColumn(
                'Flow %ile',
                format='%.1f',
                help='Individual stock flow percentile (0-100): Where Flow_10D ranks in stock\'s 100-day history'
            ),
            'Flow_Rank': st.column_config.NumberColumn(
                'Flow Rank',
                format='%.1f',
                help='Cross-stock flow ranking (0-100): How this stock ranks vs others today (higher = stronger accumulation)'
            ),
            'Flow_Velocity': st.column_config.NumberColumn(
                'Flow Velocity',
                format='%+.2f',
                help='Day-over-day flow change (acceleration/deceleration)'
            ),
            'Flow_Velocity_Percentile': st.column_config.NumberColumn(
                'Vel %ile',
                format='%.1f',
                help='Individual stock flow velocity percentile (0-100): Where Flow_Velocity ranks in stock\'s 100-day history'
            ),
            'Flow_Velocity_Rank': st.column_config.NumberColumn(
                'Vel Rank',
                format='%.1f',
                help='Cross-stock flow velocity ranking (0-100): How this stock\'s flow change ranks vs others today'
            ),
            'Flow_Regime': st.column_config.TextColumn(
                'Flow Regime',
                width='medium',
                help='Classification: Strong Accumulation/Accumulation/Neutral/Distribution/Strong Distribution'
            )
        }
        
        st.dataframe(
            flow_data,
            column_config=flow_config,
            hide_index=True,
            use_container_width=True
        )
        
        # Flow insights
        if 'Flow_Regime' in flow_data.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                strong_acc = (flow_data['Flow_Regime'] == 'Strong Accumulation').sum()
                st.metric("🔥 Strong Accumulation", strong_acc)
            with col2:
                acc = (flow_data['Flow_Regime'] == 'Accumulation').sum()
                st.metric("📈 Accumulation", acc)
            with col3:
                dist = (flow_data['Flow_Regime'] == 'Distribution').sum()
                strong_dist = (flow_data['Flow_Regime'] == 'Strong Distribution').sum()
                st.metric("📉 Distribution", dist + strong_dist)
        
        # ===== SECTION 2: CONVICTION ANALYSIS =====
        st.markdown("---")
        st.markdown("### 🎯 Conviction Analysis")
        st.caption("Volume participation patterns showing institutional commitment")
        
        # Include both raw and percentile columns for comprehensive analysis
        conviction_cols = ['Ticker', 'Name', 'Volume_Conviction', 'Volume_Conviction_Percentile', 'Conviction_Velocity', 'Avg_Vol_Up_10D']
        conviction_data = filtered_stocks[[col for col in conviction_cols if col in filtered_stocks.columns]].copy()

        # Sort by Volume_Conviction descending (highest conviction first)
        if 'Volume_Conviction' in conviction_data.columns:
            conviction_data = conviction_data.sort_values('Volume_Conviction', ascending=False)

        conviction_config = {
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Name': st.column_config.TextColumn('Company', width='medium'),
            'Volume_Conviction': st.column_config.NumberColumn(
                'Conviction',
                format='%.3f',
                help='Ratio of up-day vs down-day volume (>1.0=bullish, <1.0=bearish, 1.0=neutral)'
            ),
            'Volume_Conviction_Percentile': st.column_config.NumberColumn(
                'Conv %ile',
                format='%.0f%%',
                help='Volume_Conviction percentile rank: 80% = top 20% of 50-day history (higher = stronger conviction)'
            ),
            'Conviction_Velocity': st.column_config.NumberColumn(
                'Conv Velocity',
                format='%+.3f',
                help='Day-over-day conviction change (positive=strengthening, negative=weakening)'
            ),
            'Avg_Vol_Up_10D': st.column_config.NumberColumn(
                'Avg Up Vol',
                format='%.0f',
                help='Average volume on up days over 10-day period'
            )
        }
        
        st.dataframe(
            conviction_data,
            column_config=conviction_config,
            hide_index=True,
            use_container_width=True
        )
        
        # Conviction insights
        if 'Volume_Conviction' in conviction_data.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                high_conv = (conviction_data['Volume_Conviction'] > 1.2).sum()
                st.metric("🎯 High Conviction", high_conv, help="Conviction > 1.2")
            with col2:
                neutral_conv = ((conviction_data['Volume_Conviction'] >= 0.8) & 
                               (conviction_data['Volume_Conviction'] <= 1.2)).sum()
                st.metric("➖ Neutral", neutral_conv, help="Conviction 0.8-1.2")
            with col3:
                low_conv = (conviction_data['Volume_Conviction'] < 0.8).sum()
                st.metric("⚠️ Low Conviction", low_conv, help="Conviction < 0.8")
        
        # ===== SECTION 3: DIVERGENCE ANALYSIS =====
        st.markdown("---")
        st.markdown("### 📊 Price-Flow Divergence Analysis")
        st.caption("Misalignment between price action and institutional flow (potential reversal signals)")
        
        divergence_cols = ['Ticker', 'Name', 'Divergence_Gap', 'Divergence_Severity', 'Price_Percentile']
        divergence_data = filtered_stocks[[col for col in divergence_cols if col in filtered_stocks.columns]].copy()
        
        # Sort by Divergence_Severity descending (most severe divergences first)
        if 'Divergence_Severity' in divergence_data.columns:
            divergence_data = divergence_data.sort_values('Divergence_Severity', ascending=False)
        
        divergence_config = {
            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
            'Name': st.column_config.TextColumn('Company', width='medium'),
            'Divergence_Gap': st.column_config.NumberColumn(
                'Div Gap',
                format='%+.2f',
                help='Price percentile - Flow percentile (positive=bearish divergence, negative=bullish divergence)'
            ),
            'Divergence_Severity': st.column_config.NumberColumn(
                'Severity',
                format='%.1f',
                help='Absolute divergence magnitude on 0-100 scale (higher=more extreme misalignment)'
            ),
            'Price_Percentile': st.column_config.NumberColumn(
                'Price Pct',
                format='%.2f',
                help='Price ranking vs 252-day history (0=lowest, 1=highest)'
            )
        }
        
        st.dataframe(
            divergence_data,
            column_config=divergence_config,
            hide_index=True,
            use_container_width=True
        )
        
        # Divergence insights
        if 'Divergence_Gap' in divergence_data.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                bullish_div = (divergence_data['Divergence_Gap'] < -10).sum()
                st.metric("🟢 Bullish Divergence", bullish_div, 
                         help="Price weak but flow strong (potential reversal up)")
            with col2:
                neutral_div = ((divergence_data['Divergence_Gap'] >= -10) & 
                              (divergence_data['Divergence_Gap'] <= 10)).sum()
                st.metric("⚪ Aligned", neutral_div, help="Price and flow aligned")
            with col3:
                bearish_div = (divergence_data['Divergence_Gap'] > 10).sum()
                st.metric("🔴 Bearish Divergence", bearish_div,
                         help="Price strong but flow weak (potential reversal down)")
        
        # ===== KEY INSIGHTS =====
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        insights = []
        
        # Flow regime summary
        if 'Flow_Regime' in filtered_stocks.columns:
            strong_acc = (filtered_stocks['Flow_Regime'] == 'Strong Accumulation').sum()
            acc = (filtered_stocks['Flow_Regime'] == 'Accumulation').sum()
            dist = (filtered_stocks['Flow_Regime'] == 'Distribution').sum()
            strong_dist = (filtered_stocks['Flow_Regime'] == 'Strong Distribution').sum()
            
            total_acc = strong_acc + acc
            total_dist = dist + strong_dist
            
            if total_acc > total_dist:
                insights.append(f"📈 **Accumulation Bias**: {total_acc} stocks showing institutional buying vs {total_dist} distribution")
            elif total_dist > total_acc:
                insights.append(f"📉 **Distribution Bias**: {total_dist} stocks showing institutional selling vs {total_acc} accumulation")
        
        # Conviction summary
        if 'Volume_Conviction' in filtered_stocks.columns:
            high_conv = (filtered_stocks['Volume_Conviction'] > 1.2).sum()
            if high_conv > 0:
                insights.append(f"🎯 **High Conviction**: {high_conv} stocks with strong up-day volume participation (>1.2 ratio)")
        
        # Divergence warnings
        if 'Divergence_Severity' in filtered_stocks.columns:
            severe_div = (filtered_stocks['Divergence_Severity'] > 50).sum()
            if severe_div > 0:
                insights.append(f"⚠️ **Severe Divergence**: {severe_div} stocks showing extreme price-flow misalignment (>50 severity)")
        
        # Flow velocity trends
        if 'Flow_Velocity' in filtered_stocks.columns:
            accelerating = (filtered_stocks['Flow_Velocity'] > 0).sum()
            decelerating = (filtered_stocks['Flow_Velocity'] < 0).sum()
            if accelerating > decelerating * 1.5:
                insights.append(f"⚡ **Flow Acceleration**: {accelerating} stocks with accelerating institutional flow")
            elif decelerating > accelerating * 1.5:
                insights.append(f"⚡ **Flow Deceleration**: {decelerating} stocks with decelerating institutional flow")
        
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.info("📊 No significant institutional flow patterns detected in filtered results")
        
        # ===== INTERPRETATION GUIDE =====
        with st.expander("📖 Interpretation Guide", expanded=False):
            st.markdown("""
            ### How to Read Institutional Flow Metrics
            
            **Flow Analysis:**
            - **Daily Flow**: Current day's volume-weighted directional pressure
            - **Flow 10D**: Cumulative 10-day flow (positive=accumulation, negative=distribution)
            - **Flow Velocity**: Rate of change in flow (acceleration/deceleration)
            - **Flow Regime**: Classification based on flow strength and direction
            
            **Conviction Analysis:**
            - **Volume Conviction**: Ratio of up-day to down-day volume
              - >1.2 = High conviction (strong buying)
              - 0.8-1.2 = Neutral
              - <0.8 = Low conviction (weak buying or distribution)
            - **Conviction Velocity**: Change in conviction over time
            - **Avg Up Vol**: Average volume on up days (institutional participation level)
            
            **Divergence Analysis:**
            - **Divergence Gap**: Price percentile minus flow percentile
              - Positive gap = Bearish divergence (price strong, flow weak)
              - Negative gap = Bullish divergence (price weak, flow strong)
            - **Divergence Severity**: Magnitude of misalignment (0-100)
              - >50 = Severe divergence (high reversal potential)
              - 20-50 = Moderate divergence
              - <20 = Minor divergence
            - **Price Percentile**: Where price sits in 252-day range (0=low, 1=high)
            
            ### Trading Applications
            
            **Bullish Signals:**
            - Strong Accumulation regime + High conviction + Bullish divergence
            - Accelerating flow velocity + Rising conviction
            
            **Bearish Signals:**
            - Strong Distribution regime + Low conviction + Bearish divergence
            - Decelerating flow velocity + Falling conviction
            
            **Reversal Signals:**
            - Severe divergence (>50) with opposite flow direction
            - Price at extremes (percentile >0.9 or <0.1) with divergent flow
            """)
