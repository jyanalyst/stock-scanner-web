# File: pages/earnings_trend_analyzer.py
"""
Earnings Trend Analyzer - Advanced earnings analysis and prediction
Analyzes historical earnings data to identify trends and predict future performance
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

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


def show():
    """Main earnings trend analyzer page"""
    st.title("📈 Earnings Trend Analyzer")

    create_info_box(
        "📊 **About This Tool**\n\n"
        "Analyze historical earnings trends and predict future performance using advanced statistical analysis. "
        "This tool identifies patterns in revenue, DPU, balance sheet metrics, and management guidance to "
        "generate forward-looking predictions."
    )

    # Get available tickers
    tickers = get_available_tickers()

    if not tickers:
        create_warning_box(
            "⚠️ No earnings data available. Please upload earnings reports to begin analysis."
        )
        return

    # Ticker selection
    selected_ticker = st.selectbox(
        "Select stock to analyze:",
        options=sorted(tickers),
        help="Choose a stock with earnings data to analyze trends and predictions"
    )

    if selected_ticker:
        # Load and analyze data
        with st.spinner(f"Loading earnings data for {selected_ticker}..."):
            earnings_df = load_earnings_data(selected_ticker)

        if earnings_df.empty:
            st.error(f"No earnings data found for {selected_ticker}")
            return

        # Analyze trends
        analysis = analyze_earnings_trends(earnings_df)

        # Display summary
        create_section_header(f"📈 {selected_ticker} Earnings Analysis", f"{analysis['total_reports']} reports from {analysis['date_range']['start'].strftime('%Y-%m-%d')} to {analysis['date_range']['end'].strftime('%Y-%m-%d')}")

        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)

        latest = analysis['latest_quarter']
        if latest is not None:
            with col1:
                revenue = latest.get('revenue', 'N/A')
                st.metric("Latest Revenue", f"${revenue:,.0f}" if isinstance(revenue, (int, float)) else revenue)
            with col2:
                dpu = latest.get('dpu', 'N/A')
                st.metric("Latest DPU", f"{dpu:.2f}¢" if isinstance(dpu, (int, float)) else dpu)
            with col3:
                gearing = latest.get('gearing_ratio', 'N/A')
                st.metric("Gearing", f"{gearing:.1f}%" if isinstance(gearing, (int, float)) else gearing)
            with col4:
                guidance = latest.get('guidance_tone', 'N/A')
                st.metric("Guidance", guidance.title())

        # Prediction panel
        if analysis['predictions']:
            display_prediction_panel(analysis['predictions'])

        # Trend analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Financial Trends", "⚖️ Balance Sheet", "💬 Qualitative Analysis"])

        with tab1:
            st.markdown("### Financial Performance Trends")

            # Revenue trend
            if 'revenue' in earnings_df.columns:
                fig_rev = create_trend_chart(earnings_df, 'revenue', 'Revenue Trend')
                if fig_rev:
                    st.plotly_chart(fig_rev, use_container_width=True)

            # DPU trend
            if 'dpu' in earnings_df.columns:
                fig_dpu = create_trend_chart(earnings_df, 'dpu', 'DPU Trend')
                if fig_dpu:
                    st.plotly_chart(fig_dpu, use_container_width=True)

            # YoY changes
            st.markdown("### Year-over-Year Changes")
            yoy_cols = ['revenue_yoy_change', 'dpu_yoy_change', 'eps_yoy_change']
            yoy_data = earnings_df[['report_date', 'report_type'] + [col for col in yoy_cols if col in earnings_df.columns]].copy()
            if len(yoy_data.columns) > 1:
                st.dataframe(yoy_data, use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("### Balance Sheet Health Trends")

            # Gearing and coverage
            balance_cols = ['gearing_ratio', 'interest_coverage_ratio', 'nav_per_unit']
            for col in balance_cols:
                if col in earnings_df.columns:
                    fig = create_trend_chart(earnings_df, col, col.replace('_', ' ').title())
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

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

        # Multi-metric comparison
        st.markdown("---")
        st.markdown("### 📊 Multi-Metric Comparison")
        comparison_fig = create_comparison_chart(earnings_df)
        st.plotly_chart(comparison_fig, use_container_width=True)


if __name__ == "__main__":
    show()