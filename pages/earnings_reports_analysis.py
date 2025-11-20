# File: pages/earnings_reports_analysis.py
"""
Analysis (Earnings Reports) - Dedicated page for earnings report analysis
Merged from Scanner page earnings section and Earnings Trend Analyzer for unified functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import json

# Import common utilities
from pages.common.ui_components import (
    create_section_header, create_info_box, create_warning_box,
    create_success_box, create_error_box
)
from pages.common.error_handler import handle_error


def load_earnings_data(ticker: str) -> pd.DataFrame:
    """Load all earnings reports for a specific ticker"""
    earnings_dir = Path("data/earnings_reports")

    if not earnings_dir.exists():
        return pd.DataFrame()

    # Find all JSON files for this ticker
    pattern = f"{ticker}_*.json"
    earnings_files = list(earnings_dir.glob(pattern))

    if not earnings_files:
        return pd.DataFrame()

    # Load all earnings data
    earnings_data = []
    for file_path in earnings_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                earnings_data.append(data)
        except Exception as e:
            st.warning(f"Error loading {file_path.name}: {e}")
            continue

    if not earnings_data:
        return pd.DataFrame()

    # Convert to DataFrame and sort by date
    df = pd.DataFrame(earnings_data)
    df['report_date'] = pd.to_datetime(df['report_date'])
    df = df.sort_values('report_date')

    return df


def analyze_earnings_trends(earnings_df: pd.DataFrame) -> Dict:
    """Analyze earnings trends and patterns"""
    if earnings_df.empty:
        return {}

    analysis = {
        'total_reports': len(earnings_df),
        'date_range': {
            'start': earnings_df['report_date'].min(),
            'end': earnings_df['report_date'].max()
        },
        'latest_quarter': earnings_df.iloc[-1] if len(earnings_df) > 0 else None,
        'trends': {},
        'predictions': {}
    }

    # Financial trends
    if 'revenue' in earnings_df.columns:
        analysis['trends']['revenue'] = calculate_trend_metrics(earnings_df['revenue'])

    if 'dpu' in earnings_df.columns:
        analysis['trends']['dpu'] = calculate_trend_metrics(earnings_df['dpu'])

    if 'gearing_ratio' in earnings_df.columns:
        analysis['trends']['gearing'] = calculate_trend_metrics(earnings_df['gearing_ratio'], inverse=True)

    if 'interest_coverage_ratio' in earnings_df.columns:
        analysis['trends']['interest_coverage'] = calculate_trend_metrics(earnings_df['interest_coverage_ratio'])

    # Qualitative trend
    if 'guidance_tone' in earnings_df.columns:
        analysis['trends']['guidance'] = analyze_guidance_trends(earnings_df['guidance_tone'])

    # Generate predictions
    analysis['predictions'] = generate_earnings_predictions(earnings_df, analysis['trends'])

    return analysis


def calculate_trend_metrics(series: pd.Series, inverse: bool = False) -> Dict:
    """
    Calculate trend metrics for a time series
    inverse=True means lower values are better (e.g., gearing ratio)
    """
    if len(series) < 2:
        return {'direction': 'insufficient_data', 'strength': 'none'}

    # Calculate linear regression slope
    x = np.arange(len(series))
    valid_mask = series.notna()

    if valid_mask.sum() < 2:
        return {'direction': 'insufficient_data', 'strength': 'none'}

    x_valid = x[valid_mask]
    y_valid = series[valid_mask].values

    slope, intercept = np.polyfit(x_valid, y_valid, 1)

    # Adjust slope interpretation for inverse metrics
    if inverse:
        slope = -slope

    # Calculate correlation coefficient for strength
    correlation = np.corrcoef(x_valid, y_valid)[0, 1]

    # Determine direction
    if abs(slope) < 0.01 * abs(y_valid.mean()):  # Less than 1% change is stable
        direction = 'stable'
    elif slope > 0:
        direction = 'improving'
    else:
        direction = 'declining'

    # Determine strength based on correlation
    if abs(correlation) > 0.7:
        strength = 'strong'
    elif abs(correlation) > 0.4:
        strength = 'moderate'
    else:
        strength = 'weak'

    return {
        'direction': direction,
        'strength': strength,
        'slope': slope,
        'correlation': correlation,
        'recent_value': y_valid[-1] if len(y_valid) > 0 else None,
        'change_pct': ((y_valid[-1] - y_valid[0]) / abs(y_valid[0]) * 100) if len(y_valid) > 0 and y_valid[0] != 0 else None
    }


def analyze_guidance_trends(guidance_series: pd.Series) -> Dict:
    """Analyze management guidance tone trends"""
    if len(guidance_series) < 2:
        return {'direction': 'insufficient_data', 'most_common_tone': None}

    # Map tones to scores
    tone_scores = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }

    scores = guidance_series.map(tone_scores)
    valid_scores = scores.dropna()

    if len(valid_scores) < 2:
        return {'direction': 'insufficient_data', 'most_common_tone': None}

    # Calculate trend
    trend_metrics = calculate_trend_metrics(valid_scores)

    # Most common tone
    tone_counts = guidance_series.value_counts()
    most_common = tone_counts.index[0] if len(tone_counts) > 0 else None

    # Check for recent deterioration
    if len(scores) >= 3:
        recent_tones = guidance_series.iloc[-3:]
        deterioration = all(recent_tones.isin(['neutral', 'negative'])) and not recent_tones.isin(['positive']).any()
    else:
        deterioration = False

    return {
        **trend_metrics,
        'most_common_tone': most_common,
        'tone_distribution': tone_counts.to_dict(),
        'recent_deterioration': deterioration
    }


def generate_earnings_predictions(earnings_df: pd.DataFrame, trends: Dict) -> Dict:
    """Generate predictions for next earnings report"""
    prediction_score = 0
    factors = []

    # Financial trend analysis (40% weight)
    if 'dpu' in trends:
        dpu_trend = trends['dpu']
        if dpu_trend['direction'] == 'declining':
            if dpu_trend['strength'] == 'strong':
                prediction_score -= 40
                factors.append("Strong DPU decline trend (-40 points)")
            else:
                prediction_score -= 20
                factors.append("Moderate DPU decline (-20 points)")
        elif dpu_trend['direction'] == 'improving':
            prediction_score += 20
            factors.append("DPU improving (+20 points)")

    if 'revenue' in trends:
        rev_trend = trends['revenue']
        if rev_trend['direction'] == 'declining':
            prediction_score -= 20
            factors.append("Revenue declining (-20 points)")
        elif rev_trend['direction'] == 'improving':
            prediction_score += 15
            factors.append("Revenue improving (+15 points)")

    # Balance sheet analysis (30% weight)
    if 'gearing' in trends:
        gearing_trend = trends['gearing']
        if gearing_trend['direction'] == 'improving':  # Lower gearing is better
            prediction_score += 15
            factors.append("Gearing improving (+15 points)")
        elif gearing_trend['direction'] == 'declining':  # Higher gearing is worse
            prediction_score -= 15
            factors.append("Gearing deteriorating (-15 points)")

    if 'interest_coverage' in trends:
        ic_trend = trends['interest_coverage']
        if ic_trend['direction'] == 'improving':
            prediction_score += 15
            factors.append("Interest coverage improving (+15 points)")
        elif ic_trend['direction'] == 'declining':
            prediction_score -= 15
            factors.append("Interest coverage weakening (-15 points)")

    # Qualitative analysis (30% weight)
    if 'guidance' in trends:
        guidance_trend = trends['guidance']
        if guidance_trend.get('recent_deterioration'):
            prediction_score -= 30
            factors.append("Recent guidance deterioration (-30 points)")
        elif guidance_trend.get('most_common_tone') == 'positive':
            prediction_score += 15
            factors.append("Generally positive guidance (+15 points)")
        elif guidance_trend.get('most_common_tone') == 'negative':
            prediction_score -= 15
            factors.append("Generally negative guidance (-15 points)")

    # Determine prediction category
    if prediction_score >= 30:
        prediction = "Strong Positive"
        emoji = "🟢"
        description = "Multiple positive indicators suggest strong next earnings"
    elif prediction_score >= 10:
        prediction = "Moderately Positive"
        emoji = "🟡"
        description = "Some positive trends, cautiously optimistic outlook"
    elif prediction_score >= -10:
        prediction = "Neutral"
        emoji = "⚪"
        description = "Mixed signals, no clear directional bias"
    elif prediction_score >= -30:
        prediction = "Moderately Negative"
        emoji = "🟠"
        description = "Some concerning trends, potential headwinds"
    else:
        prediction = "Strong Negative"
        emoji = "🔴"
        description = "Multiple negative indicators suggest challenging next earnings"

    return {
        'prediction': prediction,
        'emoji': emoji,
        'score': prediction_score,
        'description': description,
        'factors': factors
    }


def display_prediction_panel(predictions: Dict) -> None:
    """Display earnings prediction panel"""
    st.markdown("---")
    st.markdown("### 🔮 Next Earnings Prediction")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"## {predictions['emoji']}")
        st.markdown(f"### {predictions['prediction']}")
        st.metric("Prediction Score", f"{predictions['score']:+d}")

    with col2:
        st.markdown(f"**Analysis:** {predictions['description']}")
        st.markdown("**Key Factors:**")
        for factor in predictions['factors']:
            st.write(f"• {factor}")

    st.markdown("---")


def create_trend_chart(earnings_df: pd.DataFrame, metric: str, title: str) -> Optional[go.Figure]:
    """Create a trend chart with trendline"""
    if metric not in earnings_df.columns:
        return None

    valid_data = earnings_df[earnings_df[metric].notna()]

    if len(valid_data) == 0:
        return None

    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=valid_data['report_date'],
        y=valid_data[metric],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Add trendline
    if len(valid_data) >= 2:
        x_numeric = np.arange(len(valid_data))
        slope, intercept = np.polyfit(x_numeric, valid_data[metric].values, 1)
        trend_line = intercept + slope * x_numeric

        fig.add_trace(go.Scatter(
            x=valid_data['report_date'],
            y=trend_line,
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Report Date",
        yaxis_title=metric.replace('_', ' ').title(),
        height=400,
        showlegend=True
    )

    return fig


def create_comparison_chart(earnings_df: pd.DataFrame) -> go.Figure:
    """Create a multi-metric comparison chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue & DPU Trends', 'Balance Sheet Health',
                       'Margins & Coverage', 'YoY Changes'),
        specs=[[{"secondary_y": True}, {}],
               [{}, {"secondary_y": True}]]
    )

    # Revenue and DPU
    if 'revenue' in earnings_df.columns:
        fig.add_trace(go.Scatter(
            x=earnings_df['report_date'], y=earnings_df['revenue'],
            name='Revenue', line=dict(color='blue')
        ), row=1, col=1)

    if 'dpu' in earnings_df.columns:
        fig.add_trace(go.Scatter(
            x=earnings_df['report_date'], y=earnings_df['dpu'],
            name='DPU', line=dict(color='green')
        ), row=1, col=1, secondary_y=True)

    # Balance sheet
    if 'gearing_ratio' in earnings_df.columns:
        fig.add_trace(go.Scatter(
            x=earnings_df['report_date'], y=earnings_df['gearing_ratio'],
            name='Gearing %', line=dict(color='red')
        ), row=1, col=2)

    if 'interest_coverage_ratio' in earnings_df.columns:
        fig.add_trace(go.Scatter(
            x=earnings_df['report_date'], y=earnings_df['interest_coverage_ratio'],
            name='Interest Coverage', line=dict(color='orange')
        ), row=1, col=2)

    # Margins and coverage
    if 'property_operating_margin' in earnings_df.columns:
        fig.add_trace(go.Scatter(
            x=earnings_df['report_date'], y=earnings_df['property_operating_margin'],
            name='Operating Margin %', line=dict(color='purple')
        ), row=2, col=1)

    # YoY changes
    if 'revenue_yoy_change' in earnings_df.columns:
        fig.add_trace(go.Scatter(
            x=earnings_df['report_date'], y=earnings_df['revenue_yoy_change'],
            name='Revenue YoY %', line=dict(color='blue')
        ), row=2, col=2)

    if 'dpu_yoy_change' in earnings_df.columns:
        fig.add_trace(go.Scatter(
            x=earnings_df['report_date'], y=earnings_df['dpu_yoy_change'],
            name='DPU YoY %', line=dict(color='green')
        ), row=2, col=2, secondary_y=True)

    fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Financial Analysis")
    return fig


def get_available_tickers() -> List[str]:
    """Get list of tickers that have earnings data"""
    earnings_dir = Path("data/earnings_reports")

    if not earnings_dir.exists():
        return []

    # Extract unique tickers from filenames (format: TICKER_DATE_TYPE.json)
    json_files = list(earnings_dir.glob("*.json"))
    tickers = set()

    for file_path in json_files:
        # Extract ticker from filename (everything before first underscore)
        ticker = file_path.stem.split('_')[0]
        tickers.add(ticker)

    return sorted(list(tickers))


def load_all_earnings_reports() -> pd.DataFrame:
    """Load all earnings reports from the data source"""
    try:
        from utils.earnings_reports import get_cached_reports
        all_reports, latest_reports = get_cached_reports()

        if latest_reports.empty:
            return pd.DataFrame()

        return latest_reports
    except Exception as e:
        handle_error(e, "EarningsReportsLoading")
        return pd.DataFrame()


def get_stocks_with_earnings(reports_df: pd.DataFrame) -> List[str]:
    """Get list of unique tickers that have earnings reports"""
    if reports_df.empty:
        return []

    return sorted(reports_df['ticker_sgx'].unique().tolist())


def display_earnings_report_details(stock_data: pd.Series) -> None:
    """Display detailed earnings report information for a selected stock"""
    ticker = stock_data['ticker_sgx']
    name = stock_data.get('company_name', ticker)

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


def show():
    """Main Analysis (Earnings Reports) page"""
    st.title("📊 Analysis (Earnings Reports)")

    create_info_box(
        "💰 **About This Tool**\n\n"
        "Comprehensive earnings analysis combining detailed report viewing with advanced trend analysis "
        "and future prediction. Analyze historical earnings data to identify patterns and predict performance."
    )

    # Load all earnings reports
    with st.spinner("Loading earnings reports..."):
        reports_df = load_all_earnings_reports()

    if reports_df.empty:
        create_warning_box(
            "⚠️ No earnings reports available. Please upload earnings reports to begin analysis."
        )
        return

    # Get stocks with earnings
    stocks_with_earnings = get_stocks_with_earnings(reports_df)

    if not stocks_with_earnings:
        create_warning_box("⚠️ No stocks with earnings reports found.")
        return

    # Summary statistics
    create_section_header(f"💰 Earnings Reports Overview - {len(stocks_with_earnings)} stocks", "")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(stocks_with_earnings))
    with col2:
        total_reports = len(reports_df)
        st.metric("Total Reports", total_reports)
    with col3:
        avg_reports_per_stock = total_reports / len(stocks_with_earnings)
        st.metric("Avg Reports/Stock", f"{avg_reports_per_stock:.1f}")
    with col4:
        guidance_counts = reports_df['guidance_tone'].value_counts()
        positive_pct = (guidance_counts.get('positive', 0) / len(reports_df)) * 100
        st.metric("Positive Guidance", f"{positive_pct:.0f}%")

    # Stock selection dropdown
    selected_ticker = st.selectbox(
        "📋 Select stock to view detailed earnings report:",
        options=stocks_with_earnings,
        help="Choose a stock to view its detailed earnings report information and trend analysis"
    )

    if selected_ticker:
        # Find the latest report for this stock
        stock_reports = reports_df[reports_df['ticker_sgx'] == selected_ticker].copy()

        if not stock_reports.empty:
            # Sort by date and get the latest
            stock_reports['report_date'] = pd.to_datetime(stock_reports['report_date'])
            latest_report = stock_reports.sort_values('report_date', ascending=False).iloc[0]

            # Display detailed report
            display_earnings_report_details(latest_report)

            # Load full earnings data for trend analysis
            with st.spinner(f"Loading trend data for {selected_ticker}..."):
                earnings_df = load_earnings_data(selected_ticker)

            if not earnings_df.empty:
                # Analyze trends
                analysis = analyze_earnings_trends(earnings_df)

                # Prediction panel
                if analysis['predictions']:
                    display_prediction_panel(analysis['predictions'])

                # Trend analysis tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Financial Trends", "⚖️ Balance Sheet", "💬 Qualitative Analysis", "📈 Intraday Earnings Reaction"])

                with tab1:
                    st.markdown("### Financial Performance Trends")

                    # Revenue trend
                    if 'revenue' in earnings_df.columns:
                        fig_rev = create_trend_chart(earnings_df, 'revenue', 'Revenue Trend')
                        if fig_rev:
                            st.plotly_chart(fig_rev, width="stretch")

                    # DPU trend
                    if 'dpu' in earnings_df.columns:
                        fig_dpu = create_trend_chart(earnings_df, 'dpu', 'DPU Trend')
                        if fig_dpu:
                            st.plotly_chart(fig_dpu, width="stretch")

                    # YoY changes
                    st.markdown("### Year-over-Year Changes")
                    yoy_cols = ['revenue_yoy_change', 'dpu_yoy_change', 'eps_yoy_change']
                    yoy_data = earnings_df[['report_date', 'report_type'] + [col for col in yoy_cols if col in earnings_df.columns]].copy()
                    if len(yoy_data.columns) > 1:
                        st.dataframe(yoy_data, width="stretch", hide_index=True)

                with tab2:
                    st.markdown("### Balance Sheet Health Trends")

                    # Gearing and coverage
                    balance_cols = ['gearing_ratio', 'interest_coverage_ratio', 'nav_per_unit']
                    for col in balance_cols:
                        if col in earnings_df.columns:
                            fig = create_trend_chart(earnings_df, col, col.replace('_', ' ').title())
                            if fig:
                                st.plotly_chart(fig, width="stretch")

                with tab3:
                    st.markdown("### Qualitative Analysis")

                    # Guidance tone over time
                    if 'guidance_tone' in earnings_df.columns:
                        tone_counts = earnings_df['guidance_tone'].value_counts()
                        fig_tone = px.pie(values=tone_counts.values, names=tone_counts.index,
                                        title="Guidance Tone Distribution")
                        st.plotly_chart(fig_tone)

                    # Recent management commentary
                    if analysis['latest_quarter'] is not None:
                        latest = analysis['latest_quarter']
                        if 'mgmt_commentary_summary' in latest:
                            st.markdown("### 💬 Latest Management Commentary")
                            st.write(latest['mgmt_commentary_summary'])

                        if 'key_highlights' in latest:
                            st.markdown("### 🎯 Key Highlights")
                            for highlight in latest['key_highlights'][:3]:  # Show top 3
                                st.write(f"• {highlight}")

                        if 'concerns' in latest:
                            st.markdown("### ⚠️ Key Concerns")
                            for concern in latest['concerns'][:3]:  # Show top 3
                                st.write(f"• {concern}")

                with tab4:
                    st.markdown("### 📈 Intraday Earnings Reaction Analysis")

                    # Get earnings reaction analysis
                    from utils.earnings_reports import calculate_earnings_reaction_analysis
                    reaction_stats = calculate_earnings_reaction_analysis(selected_ticker)

                    if reaction_stats is None:
                        st.warning("⚠️ Insufficient earnings data for intraday reaction analysis (need at least 3 earnings events)")
                    else:
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Events", reaction_stats['total_events'])
                        with col2:
                            win_rate = reaction_stats['win_rate']
                            emoji = "🟢" if win_rate >= 60 else "🟡" if win_rate >= 40 else "🔴"
                            st.metric("Win Rate", f"{emoji} {win_rate:.0f}%")
                        with col3:
                            if win_rate >= 50:
                                avg_return = reaction_stats['avg_positive_return']
                                st.metric("Avg Positive", f"+{avg_return:.1f}%")
                            else:
                                avg_return = reaction_stats['avg_negative_return']
                                st.metric("Avg Negative", f"{avg_return:.1f}%")
                        with col4:
                            overall = reaction_stats['overall_avg_return']
                            st.metric("Overall Avg", f"{overall:+.1f}%")

                        # Interpretation
                        st.markdown("---")
                        if win_rate >= 60:
                            interpretation = f"🟢 **Strong Positive Pattern**: Stock tends to build momentum intraday on earnings days. {win_rate:.0f}% of earnings resulted in positive intraday moves with average gains of +{reaction_stats['avg_positive_return']:.1f}%."
                        elif win_rate >= 40:
                            interpretation = f"🟡 **Mixed Pattern**: Stock shows moderate intraday performance on earnings. {win_rate:.0f}% positive days with average {'positive' if win_rate >= 50 else 'negative'} moves of {reaction_stats['avg_positive_return'] if win_rate >= 50 else reaction_stats['avg_negative_return']:+.1f}%."
                        else:
                            interpretation = f"🔴 **Negative Pattern**: Stock tends to fade intraday on earnings days. Only {win_rate:.0f}% positive with average losses of {reaction_stats['avg_negative_return']:.1f}%."

                        create_info_box(f"**Trading Insight:** {interpretation}")

                        # Detailed event table
                        st.markdown("### 📋 Historical Earnings Reactions")

                        events_df = pd.DataFrame(reaction_stats['events'])
                        if not events_df.empty:
                            # Format for display
                            display_df = events_df.copy()
                            display_df['intraday_return'] = display_df['intraday_return'].apply(lambda x: f"{x:+.2f}%")
                            display_df['guidance_tone'] = display_df['guidance_tone'].str.title()

                            # Add status column
                            display_df['status'] = display_df['intraday_return'].apply(
                                lambda x: "✅ Positive" if "+" in x else "❌ Negative"
                            )

                            st.dataframe(
                                display_df[['report_date', 'report_type', 'report_time', 'target_date',
                                          'open', 'close', 'intraday_return', 'guidance_tone', 'status']],
                                column_config={
                                    'report_date': st.column_config.TextColumn('Report Date', width='small'),
                                    'report_type': st.column_config.TextColumn('Period', width='small'),
                                    'report_time': st.column_config.TextColumn('Time', width='small'),
                                    'target_date': st.column_config.TextColumn('Trade Date', width='small'),
                                    'open': st.column_config.NumberColumn('Open', format='%.3f'),
                                    'close': st.column_config.NumberColumn('Close', format='%.3f'),
                                    'intraday_return': st.column_config.TextColumn('Intraday %', width='small'),
                                    'guidance_tone': st.column_config.TextColumn('Guidance', width='small'),
                                    'status': st.column_config.TextColumn('Status', width='small')
                                },
                                hide_index=True,
                                width="stretch"
                            )

                            # Visualization
                            st.markdown("### 📊 Intraday Returns Over Time")
                            fig = go.Figure()

                            # Add bars for each earnings event
                            colors = ['green' if x > 0 else 'red' for x in events_df['intraday_return']]
                            fig.add_trace(go.Bar(
                                x=events_df['report_date'],
                                y=events_df['intraday_return'],
                                marker_color=colors,
                                name='Intraday Return'
                            ))

                            # Add zero line
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")

                            fig.update_layout(
                                title="Historical Intraday Earnings Reactions",
                                xaxis_title="Earnings Report Date",
                                yaxis_title="Intraday Return %",
                                height=400,
                                showlegend=False
                            )

                            st.plotly_chart(fig, width="stretch")

                            # Pattern insights
                            st.markdown("### 🔍 Pattern Insights")

                            # Guidance correlation
                            guidance_positive = events_df[
                                (events_df['guidance_tone'] == 'positive') & (events_df['intraday_return'] > 0)
                            ].shape[0]
                            guidance_total = events_df[events_df['guidance_tone'] == 'positive'].shape[0]

                            if guidance_total >= 2:
                                guidance_win_rate = (guidance_positive / guidance_total) * 100
                                st.write(f"• **Positive Guidance**: {guidance_win_rate:.0f}% win rate ({guidance_positive}/{guidance_total} events)")

                            # Recent performance (last 3)
                            if len(events_df) >= 3:
                                recent = events_df.tail(3)
                                recent_wins = (recent['intraday_return'] > 0).sum()
                                recent_win_rate = (recent_wins / 3) * 100
                                st.write(f"• **Recent Performance**: {recent_win_rate:.0f}% win rate in last 3 earnings ({recent_wins}/3)")

                # Multi-metric comparison
                st.markdown("---")
                st.markdown("### 📊 Multi-Metric Comparison")
                comparison_fig = create_comparison_chart(earnings_df)
                st.plotly_chart(comparison_fig, width="stretch")
            else:
                st.warning(f"No detailed trend data available for {selected_ticker}")
        else:
            st.error(f"No reports found for {selected_ticker}")

    # Show all reports table
    st.markdown("---")
    create_section_header("📋 All Earnings Reports", "")

    # Prepare display dataframe
    display_df = reports_df.copy()

    # Format dates
    if 'report_date' in display_df.columns:
        display_df['report_date'] = pd.to_datetime(display_df['report_date']).dt.strftime('%Y-%m-%d')

    st.dataframe(
        display_df[['ticker_sgx', 'company_name', 'report_date', 'Earnings_Period',
                   'guidance_tone', 'revenue', 'net_profit', 'eps']],
        column_config={
            'ticker_sgx': st.column_config.TextColumn('Ticker', width='small'),
            'company_name': st.column_config.TextColumn('Company', width='medium'),
            'report_date': st.column_config.TextColumn('Date', width='small'),
            'Earnings_Period': st.column_config.TextColumn('Period', width='small'),
            'guidance_tone': st.column_config.TextColumn('Guidance', width='small'),
            'revenue': st.column_config.NumberColumn('Revenue', format='$%.0f'),
            'net_profit': st.column_config.NumberColumn('Net Profit', format='$%.0f'),
            'eps': st.column_config.NumberColumn('EPS', format='%.3f')
        },
        hide_index=True,
        width="stretch"
    )


if __name__ == "__main__":
    show()
