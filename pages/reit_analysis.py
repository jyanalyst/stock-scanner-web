"""
Analysis (REITs) - Dedicated page for comprehensive REIT valuation analysis
Based on Singapore REIT Framework v2.1 with Property Yield Spread economics
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


def load_reit_data(ticker: str) -> pd.DataFrame:
    """Load all REIT earnings reports for a specific ticker"""
    earnings_dir = Path("data/earnings_reports")

    if not earnings_dir.exists():
        return pd.DataFrame()

    # Find all JSON files for this ticker
    pattern = f"{ticker}_*.json"
    earnings_files = list(earnings_dir.glob(pattern))

    if not earnings_files:
        return pd.DataFrame()

    # Load all REIT data
    reit_data = []
    for file_path in earnings_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Only include REIT data
                if data.get('company_type') == 'reit':
                    reit_data.append(data)
        except Exception as e:
            st.warning(f"Error loading {file_path.name}: {e}")
            continue

    if not reit_data:
        return pd.DataFrame()

    # Convert to DataFrame and sort by date
    df = pd.DataFrame(reit_data)
    df['report_date'] = pd.to_datetime(df['report_date'])
    df = df.sort_values('report_date')

    return df


def get_available_reits() -> List[str]:
    """Get list of tickers that have REIT data"""
    earnings_dir = Path("data/earnings_reports")

    if not earnings_dir.exists():
        return []

    # Find all JSON files and filter for REITs
    json_files = list(earnings_dir.glob("*.json"))
    reit_tickers = set()

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get('company_type') == 'reit':
                    ticker = file_path.stem.split('_')[0]
                    reit_tickers.add(ticker)
        except:
            continue

    return sorted(list(reit_tickers))


def calculate_property_yield_spread(reit_data: pd.Series) -> Dict:
    """
    Calculate Property Yield and Spread - THE critical metric
    Returns comprehensive analysis of the REIT's economic model
    """
    analysis = {
        'property_yield': None,
        'cost_of_debt': None,
        'spread': None,
        'status': 'unknown',
        'score': 0,
        'assessment': '',
        'impact': '',
        'stress_test': {},
        'sector_benchmark': {}
    }

    # Get required data
    npi_annual = reit_data.get('net_property_income_annual')
    gav = reit_data.get('gross_asset_value')
    cost_of_debt = reit_data.get('average_cost_of_debt')
    reit_sector = reit_data.get('reit_sector', 'unknown')

    # Calculate Property Yield
    if npi_annual and gav and gav > 0:
        property_yield = (npi_annual / gav) * 100
        analysis['property_yield'] = property_yield
    else:
        # Try to estimate from quarterly data
        npi_quarterly = reit_data.get('net_property_income')
        if npi_quarterly and gav and gav > 0:
            # Annualize quarterly NPI (rough estimate)
            npi_annual_est = npi_quarterly * 4
            property_yield = (npi_annual_est / gav) * 100
            analysis['property_yield'] = property_yield
            analysis['npi_estimated'] = True

    # Get cost of debt
    if cost_of_debt:
        analysis['cost_of_debt'] = cost_of_debt
    else:
        # Estimate from interest expense and total debt
        interest_expense = reit_data.get('interest_expense')
        total_debt = reit_data.get('total_debt')
        if interest_expense and total_debt and total_debt > 0:
            estimated_cost = (interest_expense / total_debt) * 100
            analysis['cost_of_debt'] = estimated_cost
            analysis['cost_estimated'] = True

    # Calculate spread
    if analysis['property_yield'] is not None and analysis['cost_of_debt'] is not None:
        spread = analysis['property_yield'] - analysis['cost_of_debt']
        analysis['spread'] = spread

        # Assessment based on spread
        if spread >= 3.0:
            analysis['status'] = 'excellent'
            analysis['score'] = 100
            analysis['assessment'] = '🟢 EXCELLENT - Strong value creation from leverage'
            analysis['impact'] = 'Major positive factor - can absorb rate rises'
        elif spread >= 2.5:
            analysis['status'] = 'good'
            analysis['score'] = 85
            analysis['assessment'] = '🟢 GOOD - Healthy value creation'
            analysis['impact'] = 'Positive factor - moderate rate rise buffer'
        elif spread >= 2.0:
            analysis['status'] = 'adequate'
            analysis['score'] = 70
            analysis['assessment'] = '🟡 ADEQUATE - Meets minimum target'
            analysis['impact'] = 'Acceptable - limited buffer for rate rises'
        elif spread >= 1.5:
            analysis['status'] = 'weak'
            analysis['score'] = 50
            analysis['assessment'] = '🟠 WEAK - Below target'
            analysis['impact'] = '⚠️ Requires compensation from other metrics'
        elif spread >= 1.0:
            analysis['status'] = 'poor'
            analysis['score'] = 30
            analysis['assessment'] = '🔴 POOR - Significant concern'
            analysis['impact'] = '⚠️ Needs deep discount to compensate'
        else:
            analysis['status'] = 'critical'
            analysis['score'] = 10
            analysis['assessment'] = '🔴 CRITICAL - Minimal value creation'
            analysis['impact'] = '⚠️⚠️ Major concern - high risk'

        # Stress testing
        analysis['stress_test'] = {
            'current_spread': spread,
            'rates_up_0_5pct': spread - 0.5,
            'rates_up_1_0pct': spread - 1.0,
            'rates_up_1_5pct': spread - 1.5
        }

    # Sector benchmarks
    sector_benchmarks = {
        'data_centers': {'yield': '6.5-8.5%', 'min_spread': 3.0, 'typical_pnav': '1.10-1.35x'},
        'industrial_logistics': {'yield': '5.5-7.0%', 'min_spread': 2.5, 'typical_pnav': '1.00-1.20x'},
        'healthcare': {'yield': '5.5-6.5%', 'min_spread': 2.0, 'typical_pnav': '1.00-1.20x'},
        'suburban_retail': {'yield': '5.5-6.5%', 'min_spread': 2.0, 'typical_pnav': '0.85-1.00x'},
        'prime_retail': {'yield': '4.5-5.5%', 'min_spread': 1.5, 'typical_pnav': '0.95-1.10x'},
        'hospitality': {'yield': '5.0-7.0%', 'min_spread': 2.5, 'typical_pnav': '0.90-1.10x'},
        'office_cbd': {'yield': '4.0-5.0%', 'min_spread': 1.5, 'typical_pnav': '0.75-0.95x'}
    }

    # Map sector to benchmark
    sector_key = reit_sector.lower().replace(' ', '_').replace('/', '_')
    if sector_key in sector_benchmarks:
        analysis['sector_benchmark'] = sector_benchmarks[sector_key]
    else:
        analysis['sector_benchmark'] = sector_benchmarks.get('suburban_retail', {})

    return analysis


def calculate_pnav_metrics(reit_data: pd.Series, current_price: float) -> Dict:
    """Calculate P/NAV and related valuation metrics"""
    nav_per_unit = reit_data.get('nav_per_unit')

    if not nav_per_unit or current_price <= 0:
        return {}

    p_nav_ratio = current_price / nav_per_unit

    # Determine valuation category
    if p_nav_ratio < 0.70:
        valuation_status = 'deep_value'
        valuation_desc = 'Deep Value - Significant discount'
    elif p_nav_ratio < 0.85:
        valuation_status = 'value'
        valuation_desc = 'Value - Attractive discount'
    elif p_nav_ratio < 0.95:
        valuation_status = 'fair_value'
        valuation_desc = 'Fair Value - Reasonable pricing'
    elif p_nav_ratio < 1.05:
        valuation_status = 'premium'
        valuation_desc = 'Premium - Above NAV'
    elif p_nav_ratio < 1.15:
        valuation_status = 'expensive'
        valuation_desc = 'Expensive - Notable premium'
    else:
        valuation_status = 'very_expensive'
        valuation_desc = 'Very Expensive - High premium'

    return {
        'p_nav_ratio': p_nav_ratio,
        'nav_per_unit': nav_per_unit,
        'current_price': current_price,
        'valuation_status': valuation_status,
        'valuation_desc': valuation_desc,
        'discount_premium': ((p_nav_ratio - 1) * 100)
    }


def calculate_distribution_yield(reit_data: pd.Series, current_price: float) -> Dict:
    """Calculate distribution yield and assess sustainability"""
    dpu_value = reit_data.get('dpu')
    distribution_frequency = reit_data.get('distribution_frequency', 'Semi-Annual')

    if not dpu_value or current_price <= 0:
        return {}

    # Convert DPU from cents to dollars
    dpu_in_dollars = dpu_value / 100

    # Calculate trailing 12-month DPU based on distribution frequency
    if distribution_frequency.lower() == 'annual':
        dpu_12m = dpu_in_dollars
    elif distribution_frequency.lower() == 'semi-annual':
        # For semi-annual, multiply by 2 to get annual equivalent
        dpu_12m = dpu_in_dollars * 2
    elif distribution_frequency.lower() == 'quarterly':
        # For quarterly, multiply by 4 to get annual equivalent
        dpu_12m = dpu_in_dollars * 4
    else:
        # Default to semi-annual
        dpu_12m = dpu_in_dollars * 2

    # Calculate yield percentage
    yield_pct = (dpu_12m / current_price) * 100

    # Assessment
    if yield_pct >= 7.0:
        yield_status = 'very_high'
        yield_desc = 'Very High - Check sustainability'
    elif yield_pct >= 6.0:
        yield_status = 'high'
        yield_desc = 'High - Attractive income'
    elif yield_pct >= 5.0:
        yield_status = 'target'
        yield_desc = 'Target Range - Balanced income'
    elif yield_pct >= 4.5:
        yield_status = 'below_target'
        yield_desc = 'Below Target - Limited income'
    else:
        yield_status = 'low'
        yield_desc = 'Low - Income focus elsewhere'

    return {
        'yield_pct': yield_pct,
        'dpu_12m': dpu_12m,
        'dpu_display': dpu_value,  # Keep original cents value for display
        'distribution_frequency': distribution_frequency,
        'yield_status': yield_status,
        'yield_desc': yield_desc
    }


def create_pnav_spectrum_chart(p_nav_ratio: float) -> go.Figure:
    """Create P/NAV spectrum visualization"""
    fig = go.Figure()

    # Define spectrum ranges
    ranges = [
        (0.70, 0.85, 'Deep Value', 'green'),
        (0.85, 1.00, 'Fair Value', 'lightgreen'),
        (1.00, 1.15, 'Premium', 'orange'),
        (1.15, 1.30, 'Expensive', 'red')
    ]

    # Add spectrum bars
    for i, (start, end, label, color) in enumerate(ranges):
        fig.add_trace(go.Bar(
            x=[end - start],
            y=[0],
            orientation='h',
            marker_color=color,
            opacity=0.7,
            showlegend=False,
            hovertemplate=f'{label}: {start:.2f}x - {end:.2f}x<extra></extra>'
        ))

    # Add current P/NAV marker
    fig.add_trace(go.Scatter(
        x=[p_nav_ratio],
        y=[0],
        mode='markers+text',
        marker=dict(size=15, color='black', symbol='diamond'),
        text=[f'{p_nav_ratio:.2f}x'],
        textposition='top center',
        showlegend=False,
        hovertemplate=f'Current P/NAV: {p_nav_ratio:.2f}x<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title='P/NAV Valuation Spectrum',
        xaxis=dict(
            title='P/NAV Ratio',
            tickvals=[0.7, 0.85, 1.0, 1.15, 1.3],
            ticktext=['0.70x<br>Deep Value', '0.85x<br>Fair Value', '1.00x<br>Premium', '1.15x<br>Expensive', '1.30x<br>Very Expensive']
        ),
        yaxis=dict(showticklabels=False, showgrid=False),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def display_section_1_quick_snapshot(reit_data: pd.Series, current_price: float) -> None:
    """Section 1: Quick Snapshot - 30-second health check"""
    create_section_header("⚡ Section 1: Quick Snapshot", "30-second REIT health check")

    if reit_data.empty:
        create_warning_box("No REIT data available for analysis")
        return

    # Get key metrics
    pnav_metrics = calculate_pnav_metrics(reit_data, current_price)
    yield_metrics = calculate_distribution_yield(reit_data, current_price)
    spread_analysis = calculate_property_yield_spread(reit_data)

    # REIT identification
    reit_name = reit_data.get('reit_name', 'Unknown REIT')
    ticker = reit_data.get('ticker', 'Unknown')
    reit_sector = reit_data.get('reit_sector', 'Unknown')

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"### {ticker} - {reit_name}")
        st.markdown(f"**Sector:** {reit_sector}")
        st.markdown(f"**Report Date:** {reit_data.get('report_date', 'Unknown')}")

    with col2:
        if pnav_metrics:
            st.metric("Current Price", f"S${current_price:.3f}")
            st.metric("NAV/Unit", f"S${pnav_metrics['nav_per_unit']:.3f}")
            st.metric("P/NAV", f"{pnav_metrics['p_nav_ratio']:.2f}x")

    with col3:
        if yield_metrics:
            st.metric("Distribution Yield", f"{yield_metrics['yield_pct']:.1f}%")
        if spread_analysis.get('spread'):
            st.metric("Property Spread", f"{spread_analysis['spread']:.1f}%")

    # P/NAV Spectrum Chart
    if pnav_metrics:
        st.markdown("#### P/NAV Valuation Spectrum")
        spectrum_chart = create_pnav_spectrum_chart(pnav_metrics['p_nav_ratio'])
        st.plotly_chart(spectrum_chart, width="stretch")

    # Investment Criteria Check
    st.markdown("#### Investment Criteria Check")

    criteria_checks = []

    # Yield check
    if yield_metrics:
        yield_ok = yield_metrics['yield_pct'] >= 5.0
        criteria_checks.append(('Yield ≥5.0%', yield_ok, f"{yield_metrics['yield_pct']:.1f}%"))

    # Property Spread check
    if spread_analysis.get('spread') is not None:
        spread_ok = spread_analysis['spread'] >= 2.0
        criteria_checks.append(('Property Spread ≥2.0%', spread_ok, f"{spread_analysis['spread']:.1f}%"))

    # Gearing check
    gearing = reit_data.get('gearing_ratio')
    if gearing:
        gearing_ok = gearing < 42
        criteria_checks.append(('Gearing <42%', gearing_ok, f"{gearing:.1f}%"))

    # ICR check
    icr = reit_data.get('interest_coverage_ratio')
    if icr:
        icr_ok = icr > 2.5
        criteria_checks.append(('ICR >2.5x', icr_ok, f"{icr:.1f}x"))

    # Display criteria table
    if criteria_checks:
        criteria_df = pd.DataFrame(criteria_checks, columns=['Criteria', 'Met', 'Value'])
        criteria_df['Status'] = criteria_df['Met'].apply(lambda x: '✅' if x else '❌')

        st.dataframe(
            criteria_df[['Criteria', 'Value', 'Status']],
            column_config={
                'Criteria': st.column_config.TextColumn('Investment Criteria', width='medium'),
                'Value': st.column_config.TextColumn('Current Value', width='small'),
                'Status': st.column_config.TextColumn('Status', width='small')
            },
            hide_index=True,
            width="stretch"
        )

        met_count = sum(1 for _, met, _ in criteria_checks if met)
        total_criteria = len(criteria_checks)

        if met_count == total_criteria:
            st.success(f"✅ **Strong Pass**: {met_count}/{total_criteria} criteria met")
        elif met_count >= total_criteria - 1:
            st.warning(f"⚠️ **Borderline**: {met_count}/{total_criteria} criteria met")
        else:
            st.error(f"❌ **Fail**: {met_count}/{total_criteria} criteria met")


def display_section_2_economic_model(reit_data: pd.Series) -> None:
    """Section 2: Economic Model Validation - Property Yield Spread Analysis"""
    create_section_header("🏗️ Section 2: Economic Model Validation", "Property Yield Spread - The Critical Metric")

    spread_analysis = calculate_property_yield_spread(reit_data)

    if not spread_analysis.get('property_yield'):
        create_warning_box("⚠️ Insufficient data to calculate Property Yield Spread. Need NPI and GAV values.")
        return

    # Main spread display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Property Yield", f"{spread_analysis['property_yield']:.1f}%")
        if spread_analysis.get('npi_estimated'):
            st.caption("Estimated from quarterly data")

    with col2:
        cost_display = f"{spread_analysis['cost_of_debt']:.1f}%"
        if spread_analysis.get('cost_estimated'):
            cost_display += " (est.)"
        st.metric("Cost of Debt", cost_display)

    with col3:
        spread_value = spread_analysis['spread']
        st.metric("PROPERTY YIELD SPREAD", f"{spread_value:.1f}%")

    # Spread assessment
    st.markdown("#### Spread Assessment")
    assessment_color = {
        'excellent': 'green',
        'good': 'green',
        'adequate': 'orange',
        'weak': 'orange',
        'poor': 'red',
        'critical': 'red'
    }.get(spread_analysis['status'], 'gray')

    st.markdown(f"<h4 style='color: {assessment_color};'>{spread_analysis['assessment']}</h4>",
                unsafe_allow_html=True)
    st.markdown(f"**Impact:** {spread_analysis['impact']}")

    # Sector comparison
    if spread_analysis.get('sector_benchmark'):
        benchmark = spread_analysis['sector_benchmark']
        st.markdown("#### Sector Benchmark Comparison")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Typical Property Yield", benchmark.get('yield', 'N/A'))
        with col2:
            st.metric("Minimum Spread Target", f"{benchmark.get('min_spread', 'N/A')}%")
        with col3:
            st.metric("Typical P/NAV", benchmark.get('typical_pnav', 'N/A'))

    # Stress testing
    if spread_analysis.get('stress_test'):
        st.markdown("#### Stress Test: Rate Rise Scenarios")

        stress_data = spread_analysis['stress_test']
        scenarios = [
            ('Current', stress_data['current_spread']),
            ('+0.5% Rates', stress_data['rates_up_0_5pct']),
            ('+1.0% Rates', stress_data['rates_up_1_0pct']),
            ('+1.5% Rates', stress_data['rates_up_1_5pct'])
        ]

        stress_df = pd.DataFrame(scenarios, columns=['Scenario', 'Spread'])
        stress_df['Status'] = stress_df['Spread'].apply(
            lambda x: '🟢 Safe' if x >= 2.0 else '🟡 Concern' if x >= 1.0 else '🔴 Critical'
        )

        st.dataframe(
            stress_df,
            column_config={
                'Scenario': st.column_config.TextColumn('Interest Rate Scenario'),
                'Spread': st.column_config.NumberColumn('Property Spread', format='%.1f%'),
                'Status': st.column_config.TextColumn('Risk Level')
            },
            hide_index=True,
            width="stretch"
        )

    # Key insights
    st.markdown("#### Key Economic Insights")

    spread_value = spread_analysis['spread']
    if spread_value >= 2.0:
        st.success("✅ **Leverage creates value** - REIT generates more from properties than debt costs")
    elif spread_value >= 1.0:
        st.warning("⚠️ **Limited value creation** - Leverage has minimal impact, requires other strengths")
    else:
        st.error("❌ **Value destruction risk** - Properties earn less than debt costs")

    # Recommendations for weak spreads
    if spread_value < 2.0:
        st.markdown("**Compensation Strategies for Weak Spread:**")
        st.markdown("• Seek deeper P/NAV discount (<0.85x)")
        st.markdown("• Prioritize strong balance sheet metrics")
        st.markdown("• Look for NAV growth potential")
        st.markdown("• Consider sector rotation to higher-spread REITs")


def check_data_completeness(reit_data: pd.Series) -> Dict:
    """Check which required fields are present"""
    critical_fields = {
        'nav_per_unit': 'NAV per Unit',
        'dpu': 'DPU (Current)',
        'net_property_income_annual': 'Net Property Income (Annual)',
        'gross_asset_value': 'Gross Asset Value',
        'gearing_ratio': 'Gearing Ratio',
        'interest_coverage_ratio': 'Interest Coverage',
        'average_cost_of_debt': 'Cost of Debt',
        'portfolio_occupancy': 'Portfolio Occupancy'
    }

    results = {}
    for field, label in critical_fields.items():
        value = reit_data.get(field)
        results[label] = {
            'present': value is not None and value != '',
            'value': value
        }

    return results


def display_completeness_grid(completeness: Dict) -> None:
    """Display data completeness in grid format"""
    data = []
    for field, info in completeness.items():
        status = '✅' if info['present'] else '❌'
        value = str(info['value']) if info['present'] else 'Missing'
        data.append([field, status, value])

    df = pd.DataFrame(data, columns=['Field', 'Status', 'Value'])

    st.dataframe(
        df,
        column_config={
            'Field': st.column_config.TextColumn('Required Field', width='medium'),
            'Status': st.column_config.TextColumn('✓', width='small'),
            'Value': st.column_config.TextColumn('Current Value', width='medium')
        },
        hide_index=True,
        width="stretch"
    )

    complete_count = sum(1 for info in completeness.values() if info['present'])
    total_count = len(completeness)

    if complete_count == total_count:
        st.success(f"✅ Data Complete: {complete_count}/{total_count} fields present")
    elif complete_count >= total_count * 0.75:
        st.warning(f"⚠️ Mostly Complete: {complete_count}/{total_count} fields present")
    else:
        st.error(f"❌ Incomplete: Only {complete_count}/{total_count} fields present")


def generate_sgs_yield_prompt() -> str:
    """Generate AI prompt for Singapore 10-year government bond yield"""
    return """What is the current Singapore 10-year government bond yield as of November 2025?

Please provide:
1. The current yield percentage
2. Whether it's trending up/down vs 6 months ago
3. Brief context (1-2 sentences)

Format: Yield: X.XX%, Trend: [Up/Down/Stable], Context: [...]"""


def generate_sector_peers_prompt(reit_sector: str) -> str:
    """Generate AI prompt for sector peer comparison"""
    return f"""For Singapore {reit_sector} REITs as of November 2025, provide:
1. Average P/NAV ratio across the sector
2. Average distribution yield
3. Average gearing ratio
4. List 3-5 comparable REITs with their tickers

Include recent data from SGX or REIT market reports.

Format:
- Avg P/NAV: X.XXx
- Avg Yield: X.X%
- Avg Gearing: XX%
- Peers: [Ticker1, Ticker2, ...]"""


def generate_news_check_prompt(reit_name: str, ticker: str, report_date: str) -> str:
    """Generate AI prompt for recent news check"""
    return f"""Search for recent SGX announcements and material news for {reit_name} ({ticker})
since {report_date}.

Focus on:
- Acquisitions or divestments
- Rights issues or placements
- Major tenant changes
- Guidance updates

Summarize key developments (if any) in 2-3 bullet points."""


def display_copy_prompt(prompt: str, key: str) -> None:
    """Display prompt in copyable format"""
    st.markdown("**AI Research Prompt:**")
    st.code(prompt, language=None)
    st.caption("� Click the copy icon or select and Ctrl+C")


def display_section_3_data_validation(reit_data: pd.Series) -> Dict:
    """Section 3: Data Validation & Market Research"""
    create_section_header("🔍 Section 3: Data Validation & Market Research", "Verify data quality and gather market context")

    # Part A: Data Completeness Checklist
    st.markdown("### Part A: Data Completeness")
    completeness = check_data_completeness(reit_data)
    display_completeness_grid(completeness)

    # Part B: AI-Assisted Market Research
    st.markdown("### Part B: Market Context Research")
    st.info("💡 Copy the prompts below and paste into ChatGPT/Claude/Perplexity")

    market_context = {}

    # Risk-free rate
    with st.expander("📋 Research Task 1: Risk-Free Rate", expanded=True):
        prompt_1 = generate_sgs_yield_prompt()
        display_copy_prompt(prompt_1, "sgs_prompt")
        market_context['sgs_yield'] = st.number_input(
            "Enter 10Y SGS Yield (%)",
            value=3.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            key="sgs_input"
        )

    # Sector peers
    reit_sector = reit_data.get('reit_sector', 'retail')
    with st.expander("📋 Research Task 2: Sector Peer Comparison", expanded=False):
        prompt_2 = generate_sector_peers_prompt(reit_sector)
        display_copy_prompt(prompt_2, "peers_prompt")

        col1, col2, col3 = st.columns(3)
        with col1:
            market_context['sector_avg_pnav'] = st.number_input(
                "Sector Avg P/NAV",
                value=0.92,
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                key="pnav_input"
            )
        with col2:
            market_context['sector_avg_yield'] = st.number_input(
                "Sector Avg Yield (%)",
                value=5.8,
                min_value=0.0,
                max_value=15.0,
                step=0.1,
                key="yield_input"
            )
        with col3:
            market_context['sector_avg_gearing'] = st.number_input(
                "Sector Avg Gearing (%)",
                value=38.5,
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key="gearing_input"
            )

    # Recent news (if data old)
    report_date = pd.to_datetime(reit_data.get('report_date'))
    data_age = (datetime.now() - report_date).days

    if data_age > 60:
        with st.expander("📋 Research Task 3: Recent News Check", expanded=False):
            ticker = reit_data.get('ticker')
            reit_name = reit_data.get('reit_name')
            prompt_3 = generate_news_check_prompt(reit_name, ticker, report_date.strftime('%Y-%m-%d'))
            display_copy_prompt(prompt_3, "news_prompt")

            market_context['recent_news'] = st.text_area(
                "Paste news summary (if any material changes)",
                height=100,
                key="news_input"
            )

    return market_context


def calculate_nav_trend(reit_df: pd.DataFrame) -> Dict:
    """Calculate NAV trend analysis"""
    if len(reit_df) < 2:
        return {
            'cagr': None,
            'status': 'insufficient_data',
            'message': 'Need at least 2 reports for trend analysis'
        }

    # Sort by date to ensure chronological order
    df_sorted = reit_df.sort_values('report_date')

    nav_oldest = df_sorted.iloc[0]['nav_per_unit']
    nav_latest = df_sorted.iloc[-1]['nav_per_unit']

    # Calculate time period in years
    oldest_date = pd.to_datetime(df_sorted.iloc[0]['report_date'])
    latest_date = pd.to_datetime(df_sorted.iloc[-1]['report_date'])
    years = (latest_date - oldest_date).days / 365.25

    if years <= 0 or nav_oldest <= 0:
        return {
            'cagr': None,
            'status': 'invalid_data',
            'message': 'Invalid NAV data for CAGR calculation'
        }

    # Calculate CAGR
    cagr = ((nav_latest / nav_oldest) ** (1/years) - 1) * 100

    # Determine status
    if cagr >= 3.0:
        status = 'growing'
        message = '🟢 GROWING - Strong NAV growth'
    elif cagr >= 0:
        status = 'stable'
        message = '✅ STABLE - NAV maintained'
    elif cagr >= -3.0:
        status = 'weak'
        message = '⚠️ WEAK - NAV declining slowly'
    else:
        status = 'declining'
        message = '🔴 DECLINING - Significant NAV erosion'

    return {
        'cagr': cagr,
        'status': status,
        'message': message,
        'nav_oldest': nav_oldest,
        'nav_latest': nav_latest,
        'years': years,
        'data_points': len(reit_df)
    }


def get_sector_fair_pnav(sector: str) -> Dict:
    """Get sector fair P/NAV ranges from framework"""
    sector_ranges = {
        'data_centers': {'low': 1.10, 'median': 1.20, 'high': 1.35},
        'industrial_logistics': {'low': 1.00, 'median': 1.10, 'high': 1.20},
        'healthcare': {'low': 1.00, 'median': 1.10, 'high': 1.20},
        'suburban_retail': {'low': 0.85, 'median': 0.90, 'high': 1.00},
        'prime_retail': {'low': 0.95, 'median': 1.00, 'high': 1.10},
        'hospitality': {'low': 0.90, 'median': 1.00, 'high': 1.10},
        'office_cbd': {'low': 0.75, 'median': 0.85, 'high': 0.95}
    }

    # Normalize sector key
    sector_key = sector.lower().replace(' ', '_').replace('/', '_')
    if sector_key in sector_ranges:
        return sector_ranges[sector_key]
    else:
        # Default to suburban retail
        return sector_ranges['suburban_retail']


def calculate_quality_adjustments(reit_data: pd.Series, spread_analysis: Dict) -> Dict:
    """Calculate quality adjustments based on spread and gearing"""
    adjustments = {}

    # Property Yield Spread adjustment
    spread = spread_analysis.get('spread', 0)
    if spread > 3.0:
        adjustments['property_spread'] = 0.10
    elif spread > 2.5:
        adjustments['property_spread'] = 0.05
    elif spread >= 2.0:
        adjustments['property_spread'] = 0.00
    elif spread >= 1.5:
        adjustments['property_spread'] = -0.05
    else:
        adjustments['property_spread'] = -0.10

    # Gearing adjustment
    gearing = reit_data.get('gearing_ratio', 0)
    if gearing < 35:
        adjustments['gearing'] = 0.05
    elif gearing <= 40:
        adjustments['gearing'] = 0.00
    elif gearing <= 45:
        adjustments['gearing'] = -0.05
    else:
        adjustments['gearing'] = -0.10

    return adjustments


def calculate_fair_value(nav_per_unit: float, fair_pnav: float) -> Dict:
    """Calculate fair value scenarios"""
    sector_ranges = get_sector_fair_pnav('suburban_retail')  # Default, will be overridden

    conservative_pnav = fair_pnav * 0.85  # Conservative scenario
    base_pnav = fair_pnav
    optimistic_pnav = fair_pnav * 1.15  # Optimistic scenario

    return {
        'conservative': nav_per_unit * conservative_pnav,
        'base': nav_per_unit * base_pnav,
        'optimistic': nav_per_unit * optimistic_pnav,
        'conservative_pnav': conservative_pnav,
        'base_pnav': base_pnav,
        'optimistic_pnav': optimistic_pnav
    }


def calculate_margin_of_safety(fair_value: float, current_price: float) -> Dict:
    """Calculate margin of safety"""
    if fair_value <= 0:
        return {'mos': None, 'status': 'invalid', 'message': 'Invalid fair value'}

    mos = ((fair_value - current_price) / fair_value) * 100

    if mos > 25:
        status = 'extreme_value'
        message = '🟢 EXTREME VALUE - Significant discount'
    elif mos > 20:
        status = 'deep_value'
        message = '🟢 DEEP VALUE - Attractive discount'
    elif mos > 15:
        status = 'strong_value'
        message = '🟢 STRONG VALUE - Good discount'
    elif mos > 10:
        status = 'good_value'
        message = '✅ GOOD VALUE - Reasonable discount'
    elif mos > 5:
        status = 'fair_entry'
        message = '⚠️ FAIR ENTRY - Minimal discount'
    elif mos > 0:
        status = 'minimal_mos'
        message = '⏳ MINIMAL MOS - Close to fair value'
    else:
        status = 'overvalued'
        message = '❌ OVERVALUED - Trading above fair value'

    return {
        'mos': mos,
        'status': status,
        'message': message,
        'fair_value': fair_value,
        'current_price': current_price,
        'discount_premium': mos
    }


def display_nav_trend_chart(reit_df: pd.DataFrame) -> None:
    """Display NAV trend chart"""
    if len(reit_df) < 2:
        st.info("📊 NAV trend chart requires at least 2 reports")
        return

    # Prepare data
    df_chart = reit_df.sort_values('report_date').copy()
    df_chart['report_date'] = pd.to_datetime(df_chart['report_date'])

    # Create chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_chart['report_date'],
        y=df_chart['nav_per_unit'],
        mode='lines+markers',
        name='NAV per Unit',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='NAV per Unit Trend',
        xaxis_title='Report Date',
        yaxis_title='NAV per Unit (S$)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, width="stretch")


def display_adjustment_breakdown(sector_median: float, adjustments: Dict) -> None:
    """Display fair value adjustment breakdown"""
    st.markdown("#### Fair P/NAV Calculation")

    # Create breakdown table
    breakdown_data = [
        ['Sector Median P/NAV', f"{sector_median:.2f}x", ''],
        ['Property Spread Adj', f"{adjustments.get('property_spread', 0):+.2f}x", get_adjustment_reason('property_spread', adjustments)],
        ['Gearing Adj', f"{adjustments.get('gearing', 0):+.2f}x", get_adjustment_reason('gearing', adjustments)],
    ]

    # Calculate total
    total_adjustment = sum(adjustments.values())
    fair_pnav = sector_median + total_adjustment

    breakdown_data.append(['─' * 20, '─' * 10, ''])
    breakdown_data.append(['Fair P/NAV', f"{fair_pnav:.2f}x", ''])

    # Display as dataframe
    df_breakdown = pd.DataFrame(breakdown_data, columns=['Component', 'Value', 'Reason'])
    st.dataframe(
        df_breakdown,
        column_config={
            'Component': st.column_config.TextColumn('Component', width='medium'),
            'Value': st.column_config.TextColumn('Adjustment', width='small'),
            'Reason': st.column_config.TextColumn('Reason', width='large')
        },
        hide_index=True,
        width="stretch"
    )


def get_adjustment_reason(adj_type: str, adjustments: Dict) -> str:
    """Get explanation for adjustment"""
    if adj_type == 'property_spread':
        value = adjustments.get('property_spread', 0)
        if value == 0.10:
            return 'Excellent spread (>3.0%)'
        elif value == 0.05:
            return 'Good spread (2.5-3.0%)'
        elif value == 0.00:
            return 'Adequate spread (2.0-2.5%)'
        elif value == -0.05:
            return 'Weak spread (1.5-2.0%)'
        else:
            return 'Poor spread (<1.5%)'
    elif adj_type == 'gearing':
        value = adjustments.get('gearing', 0)
        if value == 0.05:
            return 'Conservative gearing (<35%)'
        elif value == 0.00:
            return 'Healthy gearing (35-40%)'
        elif value == -0.05:
            return 'Elevated gearing (40-45%)'
        else:
            return 'High risk gearing (>45%)'
    return ''


def display_fair_value_range(fair_values: Dict, current_price: float) -> None:
    """Display fair value range chart"""
    st.markdown("#### Fair Value Range")

    # Prepare data for bar chart
    scenarios = ['Conservative', 'Base Case', 'Optimistic']
    values = [fair_values['conservative'], fair_values['base'], fair_values['optimistic']]
    colors = ['lightcoral', 'lightblue', 'lightgreen']

    # Create bar chart
    fig = go.Figure()

    # Add fair value bars
    for i, (scenario, value, color) in enumerate(zip(scenarios, values, colors)):
        fig.add_trace(go.Bar(
            x=[scenario],
            y=[value],
            name=scenario,
            marker_color=color,
            showlegend=False,
            hovertemplate=f'{scenario}: S${value:.2f}<extra></extra>'
        ))

    # Add current price line
    fig.add_trace(go.Scatter(
        x=scenarios,
        y=[current_price] * len(scenarios),
        mode='lines+markers',
        name='Current Price',
        line=dict(color='red', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        hovertemplate=f'Current Price: S${current_price:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Fair Value Scenarios vs Current Price',
        xaxis_title='Scenario',
        yaxis_title='Price (S$)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, width="stretch")

    # Summary
    base_fair = fair_values['base']
    diff = current_price - base_fair
    diff_pct = (diff / base_fair) * 100

    if diff > 0:
        st.metric("Premium to Fair Value", f"S${diff:.2f}", f"{diff_pct:+.1f}%")
    else:
        st.metric("Discount to Fair Value", f"S${abs(diff):.2f}", f"{diff_pct:+.1f}%")


def display_mos_gauge(mos_data: Dict) -> None:
    """Display margin of safety gauge"""
    st.markdown("#### Margin of Safety")

    mos = mos_data['mos']
    status = mos_data['status']
    message = mos_data['message']

    # Color mapping
    color_map = {
        'extreme_value': 'green',
        'deep_value': 'green',
        'strong_value': 'green',
        'good_value': 'blue',
        'fair_entry': 'orange',
        'minimal_mos': 'orange',
        'overvalued': 'red'
    }

    gauge_color = color_map.get(status, 'gray')

    # Display metric
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Margin of Safety",
            f"{mos:+.1f}%" if mos is not None else "N/A",
            delta=message.split(' - ')[0] if ' - ' in message else None
        )

    with col2:
        st.markdown(f"**Assessment:** {message}")

    # Visual gauge using progress bar
    if mos is not None:
        # Normalize MOS for progress bar (0-100 scale)
        # Map -50% to +50% MOS to 0-100 progress
        normalized_mos = max(0, min(100, 50 + mos))

        st.progress(normalized_mos / 100, text=f"MOS: {mos:+.1f}%")

        # MOS scale reference
        st.caption("MOS Scale: ← Overvalued | Fair Value | Undervalued →")


def display_section_4_valuation(reit_data: pd.Series, reit_df: pd.DataFrame, current_price: float) -> None:
    """Section 4: Valuation Framework"""
    create_section_header("💰 Section 4: Valuation Framework", "Determine intrinsic value and margin of safety")

    # Step 4A: NAV Trend Analysis
    st.markdown("### Step 4A: NAV Trend Analysis")

    nav_trend = calculate_nav_trend(reit_df)

    if nav_trend['cagr'] is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("NAV CAGR", f"{nav_trend['cagr']:+.1f}%")
        with col2:
            st.metric("Period", f"{nav_trend['years']:.1f} years")
        with col3:
            st.metric("Data Points", nav_trend['data_points'])

        st.markdown(f"**Status:** {nav_trend['message']}")

        # Show NAV trend chart
        display_nav_trend_chart(reit_df)

    else:
        st.warning(f"⚠️ {nav_trend['message']}")
        st.info("NAV trend analysis requires multiple reports over time")

    # Step 4B: Sector-Adjusted Fair P/NAV
    st.markdown("### Step 4B: Fair Value Calculation")

    sector = reit_data.get('reit_sector', 'suburban_retail')
    sector_ranges = get_sector_fair_pnav(sector)

    st.markdown(f"**Sector:** {sector.replace('_', ' ').title()}")
    st.markdown(f"**Sector P/NAV Range:** {sector_ranges['low']:.2f}x - {sector_ranges['median']:.2f}x - {sector_ranges['high']:.2f}x")

    # Step 4C: Quality Adjustments
    spread_analysis = calculate_property_yield_spread(reit_data)
    adjustments = calculate_quality_adjustments(reit_data, spread_analysis)

    # Display adjustment breakdown
    display_adjustment_breakdown(sector_ranges['median'], adjustments)

    # Step 4D: Fair Value Scenarios
    nav_per_unit = reit_data.get('nav_per_unit', 0)
    fair_pnav = sector_ranges['median'] + sum(adjustments.values())

    fair_values = calculate_fair_value(nav_per_unit, fair_pnav)
    display_fair_value_range(fair_values, current_price)

    # Step 4E: Margin of Safety
    st.markdown("### Step 4E: Margin of Safety Assessment")

    mos_data = calculate_margin_of_safety(fair_values['base'], current_price)
    display_mos_gauge(mos_data)

    # Summary
    st.markdown("#### Valuation Summary")

    summary_data = [
        ['Current Price', f"S${current_price:.3f}"],
        ['NAV per Unit', f"S${nav_per_unit:.3f}"],
        ['Current P/NAV', f"{current_price/nav_per_unit:.2f}x" if nav_per_unit > 0 else "N/A"],
        ['Fair P/NAV (Base)', f"{fair_values['base_pnav']:.2f}x"],
        ['Fair Value (Base)', f"S${fair_values['base']:.2f}"],
        ['Margin of Safety', f"{mos_data['mos']:+.1f}%" if mos_data['mos'] is not None else "N/A"]
    ]

    df_summary = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    st.dataframe(
        df_summary,
        column_config={
            'Metric': st.column_config.TextColumn('Metric', width='medium'),
            'Value': st.column_config.TextColumn('Value', width='medium')
        },
        hide_index=True,
        width="stretch"
    )


def calculate_dpu_trend(reit_df: pd.DataFrame) -> Dict:
    """Calculate DPU trend analysis"""
    if len(reit_df) < 2:
        return {
            'cagr': None,
            'status': 'insufficient_data',
            'message': 'Need at least 2 reports for DPU trend analysis'
        }

    # Sort by date
    df_sorted = reit_df.sort_values('report_date')

    dpu_oldest = df_sorted.iloc[0]['dpu']
    dpu_latest = df_sorted.iloc[-1]['dpu']

    # Calculate time period in years
    oldest_date = pd.to_datetime(df_sorted.iloc[0]['report_date'])
    latest_date = pd.to_datetime(df_sorted.iloc[-1]['report_date'])
    years = (latest_date - oldest_date).days / 365.25

    if years <= 0 or dpu_oldest <= 0:
        return {
            'cagr': None,
            'status': 'invalid_data',
            'message': 'Invalid DPU data for CAGR calculation'
        }

    # Calculate CAGR
    dpu_cagr = ((dpu_latest / dpu_oldest) ** (1/years) - 1) * 100

    # Determine status
    if dpu_cagr >= 3.0:
        status = 'strong_growth'
        message = '🟢 STRONG GROWTH - DPU increasing significantly'
    elif dpu_cagr >= 0:
        status = 'stable'
        message = '✅ STABLE - DPU maintained or slightly growing'
    elif dpu_cagr >= -5.0:
        status = 'declining'
        message = '⚠️ DECLINING - DPU decreasing moderately'
    else:
        status = 'severe_decline'
        message = '🔴 SEVERE DECLINE - Significant DPU erosion'

    return {
        'cagr': dpu_cagr,
        'status': status,
        'message': message,
        'dpu_oldest': dpu_oldest,
        'dpu_latest': dpu_latest,
        'years': years,
        'data_points': len(reit_df)
    }


def assess_distribution_sustainability(reit_data: pd.Series) -> Dict:
    """Assess distribution sustainability"""
    payout_ratio = reit_data.get('payout_ratio', 100)

    # Payout ratio assessment
    if payout_ratio < 90:
        payout_status = 'sustainable'
        payout_message = '🟢 SUSTAINABLE - Strong retention buffer'
    elif payout_ratio < 95:
        payout_status = 'balanced'
        payout_message = '✅ BALANCED - Standard payout level'
    elif payout_ratio <= 100:
        payout_status = 'full_payout'
        payout_message = '⚠️ FULL PAYOUT - No buffer for fluctuations'
    else:
        payout_status = 'unsustainable'
        payout_message = '🔴 UNSUSTAINABLE - Exceeds distributable income'

    # DPU coverage calculation
    income_available = reit_data.get('income_available_for_distribution', 0)
    units_outstanding = reit_data.get('units_outstanding', 0)
    dpu_cents = reit_data.get('dpu', 0)

    if units_outstanding > 0 and dpu_cents > 0:
        # Convert DPU to dollars and calculate annual total
        dpu_dollars = dpu_cents / 100
        distribution_frequency = reit_data.get('distribution_frequency', 'Semi-Annual')

        if distribution_frequency.lower() == 'annual':
            annual_dpu = dpu_dollars
        elif distribution_frequency.lower() == 'semi-annual':
            annual_dpu = dpu_dollars * 2
        else:
            annual_dpu = dpu_dollars * 4  # Quarterly

        total_annual_distributions = annual_dpu * units_outstanding
        coverage_ratio = income_available / total_annual_distributions if total_annual_distributions > 0 else 0

        if coverage_ratio >= 1.1:
            coverage_status = 'excellent'
            coverage_message = '🟢 EXCELLENT - Strong coverage buffer'
        elif coverage_ratio >= 1.0:
            coverage_status = 'adequate'
            coverage_message = '✅ ADEQUATE - Meets distribution requirements'
        elif coverage_ratio >= 0.9:
            coverage_status = 'tight'
            coverage_message = '⚠️ TIGHT - Limited buffer'
        else:
            coverage_status = 'insufficient'
            coverage_message = '🔴 INSUFFICIENT - Cannot fully cover distributions'
    else:
        coverage_ratio = None
        coverage_status = 'unknown'
        coverage_message = 'Unable to calculate coverage ratio'

    return {
        'payout_ratio': payout_ratio,
        'payout_status': payout_status,
        'payout_message': payout_message,
        'coverage_ratio': coverage_ratio,
        'coverage_status': coverage_status,
        'coverage_message': coverage_message,
        'income_available': income_available,
        'total_distributions': total_annual_distributions if 'total_annual_distributions' in locals() else None
    }


def assess_income_quality(reit_data: pd.Series, reit_df: pd.DataFrame) -> Dict:
    """Assess income quality metrics"""
    metrics = {}

    # NPI Margin
    revenue = reit_data.get('revenue', 0)
    npi = reit_data.get('net_property_income', 0)

    if revenue > 0:
        npi_margin = (npi / revenue) * 100

        if npi_margin >= 70:
            margin_status = 'excellent'
            margin_message = '🟢 EXCELLENT - High operating efficiency'
        elif npi_margin >= 60:
            margin_status = 'good'
            margin_message = '🟢 GOOD - Strong margins'
        elif npi_margin >= 50:
            margin_status = 'adequate'
            margin_message = '✅ ADEQUATE - Acceptable margins'
        elif npi_margin >= 40:
            margin_status = 'weak'
            margin_message = '⚠️ WEAK - Below target'
        else:
            margin_status = 'poor'
            margin_message = '🔴 POOR - Significant concern'

        metrics['npi_margin'] = {
            'value': npi_margin,
            'status': margin_status,
            'message': margin_message
        }

    # Rental Reversion
    rental_reversion = reit_data.get('rental_reversion', 0)

    if rental_reversion > 5.0:
        reversion_status = 'strong'
        reversion_message = '🟢 STRONG - Positive rental growth'
    elif rental_reversion > 0:
        reversion_status = 'positive'
        reversion_message = '✅ POSITIVE - Rental increases'
    elif rental_reversion > -5.0:
        reversion_status = 'weak'
        reversion_message = '⚠️ WEAK - Limited rental growth'
    else:
        reversion_status = 'negative'
        reversion_message = '🔴 NEGATIVE - Rental declines'

    metrics['rental_reversion'] = {
        'value': rental_reversion,
        'status': reversion_status,
        'message': reversion_message
    }

    # Portfolio Occupancy
    occupancy = reit_data.get('portfolio_occupancy', 0)

    if occupancy >= 95:
        occ_status = 'excellent'
        occ_message = '🟢 EXCELLENT - Near full occupancy'
    elif occupancy >= 90:
        occ_status = 'healthy'
        occ_message = '✅ HEALTHY - Strong occupancy'
    elif occupancy >= 85:
        occ_status = 'below_target'
        occ_message = '⚠️ BELOW TARGET - Room for improvement'
    else:
        occ_status = 'weak'
        occ_message = '🔴 WEAK - Occupancy concerns'

    metrics['occupancy'] = {
        'value': occupancy,
        'status': occ_status,
        'message': occ_message
    }

    # Shopper Traffic (if available)
    shopper_traffic_yoy = reit_data.get('shopper_traffic_yoy_change')
    if shopper_traffic_yoy is not None:
        if shopper_traffic_yoy > 5:
            traffic_status = 'strong'
            traffic_message = '🟢 STRONG - Significant traffic growth'
        elif shopper_traffic_yoy > 0:
            traffic_status = 'positive'
            traffic_message = '✅ POSITIVE - Traffic increasing'
        elif shopper_traffic_yoy > -5:
            traffic_status = 'stable'
            traffic_message = '⚠️ STABLE - Traffic maintained'
        else:
            traffic_status = 'declining'
            traffic_message = '🔴 DECLINING - Traffic decreasing'

        metrics['shopper_traffic'] = {
            'value': shopper_traffic_yoy,
            'status': traffic_status,
            'message': traffic_message
        }

    # Tenant Sales (if available)
    tenant_sales_yoy = reit_data.get('tenant_sales_yoy_change')
    if tenant_sales_yoy is not None:
        if tenant_sales_yoy > 5:
            sales_status = 'strong'
            sales_message = '🟢 STRONG - Robust sales growth'
        elif tenant_sales_yoy > 0:
            sales_status = 'positive'
            sales_message = '✅ POSITIVE - Sales increasing'
        elif tenant_sales_yoy > -5:
            sales_status = 'stable'
            sales_message = '⚠️ STABLE - Sales maintained'
        else:
            sales_status = 'declining'
            sales_message = '🔴 DECLINING - Sales decreasing'

        metrics['tenant_sales'] = {
            'value': tenant_sales_yoy,
            'status': sales_status,
            'message': sales_message
        }

    return metrics


def calculate_income_quality_score(dpu_trend: Dict, sustainability: Dict, quality: Dict) -> Dict:
    """Calculate overall income quality score (0-100)"""
    score = 0
    breakdown = {}

    # DPU Growth (30 points)
    if dpu_trend.get('cagr') is not None:
        cagr = dpu_trend['cagr']
        if cagr >= 3.0:
            dpu_score = 30
        elif cagr >= 0:
            dpu_score = 20
        elif cagr >= -3.0:
            dpu_score = 10
        else:
            dpu_score = 0
    else:
        # No historical data - neutral score
        dpu_score = 15

    score += dpu_score
    breakdown['dpu_growth'] = dpu_score

    # Payout Sustainability (25 points)
    payout_ratio = sustainability.get('payout_ratio', 100)
    if payout_ratio < 90:
        payout_score = 25
    elif payout_ratio < 95:
        payout_score = 20
    elif payout_ratio <= 100:
        payout_score = 15
    else:
        payout_score = 0

    score += payout_score
    breakdown['payout_sustainability'] = payout_score

    # NPI Margin (20 points)
    npi_metric = quality.get('npi_margin', {})
    npi_margin = npi_metric.get('value', 0)
    if npi_margin >= 70:
        npi_score = 20
    elif npi_margin >= 60:
        npi_score = 15
    elif npi_margin >= 50:
        npi_score = 10
    elif npi_margin >= 40:
        npi_score = 5
    else:
        npi_score = 0

    score += npi_score
    breakdown['npi_margin'] = npi_score

    # Occupancy (15 points)
    occ_metric = quality.get('occupancy', {})
    occupancy = occ_metric.get('value', 0)
    if occupancy >= 95:
        occ_score = 15
    elif occupancy >= 90:
        occ_score = 10
    elif occupancy >= 85:
        occ_score = 5
    else:
        occ_score = 0

    score += occ_score
    breakdown['occupancy'] = occ_score

    # Rental Reversion (10 points)
    rev_metric = quality.get('rental_reversion', {})
    reversion = rev_metric.get('value', 0)
    if reversion > 5:
        rev_score = 10
    elif reversion > 0:
        rev_score = 7
    elif reversion > -5:
        rev_score = 3
    else:
        rev_score = 0

    score += rev_score
    breakdown['rental_reversion'] = rev_score

    # Determine grade
    if score >= 90:
        grade = 'A'
        grade_desc = 'EXCELLENT - Superior income quality'
    elif score >= 80:
        grade = 'B'
        grade_desc = 'GOOD - Strong income fundamentals'
    elif score >= 70:
        grade = 'C'
        grade_desc = 'ADEQUATE - Acceptable income quality'
    elif score >= 60:
        grade = 'D'
        grade_desc = 'WEAK - Income concerns present'
    else:
        grade = 'F'
        grade_desc = 'POOR - Significant income risks'

    return {
        'score': score,
        'grade': grade,
        'description': grade_desc,
        'breakdown': breakdown
    }


def display_dpu_trend_chart(reit_df: pd.DataFrame) -> None:
    """Display DPU trend chart"""
    if len(reit_df) < 2:
        st.info("📊 DPU trend chart requires at least 2 reports")
        return

    # Prepare data
    df_chart = reit_df.sort_values('report_date').copy()
    df_chart['report_date'] = pd.to_datetime(df_chart['report_date'])

    # Create chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_chart['report_date'],
        y=df_chart['dpu'],
        mode='lines+markers',
        name='DPU (cents)',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='DPU Trend (cents per unit)',
        xaxis_title='Report Date',
        yaxis_title='DPU (cents)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, width="stretch")


def display_sustainability_dashboard(sustainability: Dict) -> None:
    """Display distribution sustainability metrics"""
    st.markdown("#### Distribution Sustainability")

    col1, col2, col3 = st.columns(3)

    with col1:
        payout_ratio = sustainability.get('payout_ratio', 100)
        st.metric("Payout Ratio", f"{payout_ratio:.1f}%")
        st.caption(sustainability.get('payout_message', ''))

    with col2:
        coverage = sustainability.get('coverage_ratio')
        if coverage is not None:
            st.metric("DPU Coverage", f"{coverage:.2f}x")
            st.caption(sustainability.get('coverage_message', ''))
        else:
            st.metric("DPU Coverage", "N/A")
            st.caption("Unable to calculate")

    with col3:
        income = sustainability.get('income_available', 0)
        st.metric("Income Available", f"S${income:,.0f}M")
        st.caption("For distribution")


def display_operating_metrics_grid(reit_data: pd.Series) -> None:
    """Display operating metrics in a grid"""
    st.markdown("#### Operating Metrics")

    metrics_data = []

    # NPI Margin
    revenue = reit_data.get('revenue', 0)
    npi = reit_data.get('net_property_income', 0)
    if revenue > 0:
        npi_margin = (npi / revenue) * 100
        metrics_data.append(['NPI Margin', f"{npi_margin:.1f}%", 'Operating efficiency'])

    # Occupancy
    occupancy = reit_data.get('portfolio_occupancy', 0)
    metrics_data.append(['Portfolio Occupancy', f"{occupancy:.1f}%", 'Income stability'])

    # Rental Reversion
    reversion = reit_data.get('rental_reversion', 0)
    metrics_data.append(['Rental Reversion', f"{reversion:+.1f}%", 'Rental growth'])

    # Shopper Traffic
    traffic = reit_data.get('shopper_traffic_yoy_change')
    if traffic is not None:
        metrics_data.append(['Shopper Traffic', f"{traffic:+.1f}% YoY", 'Footfall trends'])

    # Tenant Sales
    sales = reit_data.get('tenant_sales_yoy_change')
    if sales is not None:
        metrics_data.append(['Tenant Sales', f"{sales:+.1f}% YoY", 'Business performance'])

    # Tenant Retention
    retention = reit_data.get('tenant_sales_retention')
    if retention is not None:
        metrics_data.append(['Tenant Retention', f"{retention:.1f}%", 'Lease stability'])

    # Lease Expiry Profile
    expiry_1yr = reit_data.get('lease_expiry_1yr')
    if expiry_1yr is not None:
        metrics_data.append(['Lease Expiry <1yr', f"{expiry_1yr:.1f}%", 'Near-term rollover risk'])

    # Display as dataframe
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data, columns=['Metric', 'Value', 'Significance'])
        st.dataframe(
            df_metrics,
            column_config={
                'Metric': st.column_config.TextColumn('Operating Metric', width='medium'),
                'Value': st.column_config.TextColumn('Current Value', width='small'),
                'Significance': st.column_config.TextColumn('Business Impact', width='large')
            },
            hide_index=True,
            width="stretch"
        )


def display_income_quality_grade(score_data: Dict) -> None:
    """Display income quality score and grade"""
    st.markdown("#### Income Quality Assessment")

    score = score_data['score']
    grade = score_data['grade']
    description = score_data['description']

    # Main grade display
    col1, col2 = st.columns([1, 3])

    with col1:
        # Large grade display
        grade_color = {
            'A': 'green',
            'B': 'blue',
            'C': 'orange',
            'D': 'red',
            'F': 'red'
        }.get(grade, 'gray')

        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border: 2px solid {grade_color}; border-radius: 10px;'>
            <h1 style='color: {grade_color}; margin: 0; font-size: 3em;'>{grade}</h1>
            <p style='margin: 5px 0 0 0; font-size: 1.2em;'>{score}/100</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"**Grade: {grade}**")
        st.markdown(f"**Score: {score}/100**")
        st.markdown(f"**Assessment:** {description}")

        # Score breakdown
        st.markdown("**Score Breakdown:**")
        breakdown = score_data['breakdown']
        for component, points in breakdown.items():
            component_name = component.replace('_', ' ').title()
            st.markdown(f"- {component_name}: {points} points")

    # Key insights
    st.markdown("**Key Insights:**")

    insights = []

    # DPU trend insight
    if score_data['breakdown'].get('dpu_growth', 0) >= 20:
        insights.append("✅ Strong DPU growth supports income reliability")
    elif score_data['breakdown'].get('dpu_growth', 0) <= 10:
        insights.append("⚠️ DPU growth concerns require monitoring")

    # Payout insight
    if score_data['breakdown'].get('payout_sustainability', 0) >= 20:
        insights.append("✅ Conservative payout ratio provides safety buffer")
    elif score_data['breakdown'].get('payout_sustainability', 0) <= 10:
        insights.append("⚠️ High payout ratio increases sustainability risk")

    # Operating insights
    if score_data['breakdown'].get('npi_margin', 0) >= 15:
        insights.append("✅ Strong NPI margins indicate efficient operations")
    if score_data['breakdown'].get('occupancy', 0) >= 10:
        insights.append("✅ High occupancy provides stable income base")
    if score_data['breakdown'].get('rental_reversion', 0) >= 7:
        insights.append("✅ Positive rental reversions support future growth")

    for insight in insights:
        st.markdown(insight)


def display_section_5_income_sustainability(reit_data: pd.Series, reit_df: pd.DataFrame) -> None:
    """Section 5: Income & Sustainability Analysis"""
    create_section_header("💰 Section 5: Income & Sustainability Analysis", "Assess income quality and distribution sustainability")

    # Part A: DPU Trend Analysis
    st.markdown("### Part A: DPU Trend Analysis")

    dpu_trend = calculate_dpu_trend(reit_df)

    if dpu_trend['cagr'] is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("DPU CAGR", f"{dpu_trend['cagr']:+.1f}%")
        with col2:
            st.metric("Period", f"{dpu_trend['years']:.1f} years")
        with col3:
            st.metric("Data Points", dpu_trend['data_points'])

        st.markdown(f"**Status:** {dpu_trend['message']}")

        # Show DPU trend chart
        display_dpu_trend_chart(reit_df)

    else:
        st.warning(f"⚠️ {dpu_trend['message']}")
        st.info("DPU trend analysis requires multiple reports over time")

        # Show current DPU info
        current_dpu = reit_data.get('dpu', 0)
        frequency = reit_data.get('distribution_frequency', 'Semi-Annual')

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current DPU", f"{current_dpu:.3f} cents")
        with col2:
            st.metric("Distribution Frequency", frequency)

        if frequency.lower() == 'semi-annual':
            annualized = current_dpu * 2
        elif frequency.lower() == 'quarterly':
            annualized = current_dpu * 4
        else:
            annualized = current_dpu

        st.info(f"Annualized DPU: {annualized:.3f} cents per unit")

    # Part B: Distribution Sustainability
    st.markdown("### Part B: Distribution Sustainability")
    sustainability = assess_distribution_sustainability(reit_data)
    display_sustainability_dashboard(sustainability)

    # Part C: Income Quality Metrics
    st.markdown("### Part C: Income Quality Assessment")
    quality_metrics = assess_income_quality(reit_data, reit_df)
    display_operating_metrics_grid(reit_data)

    # Part D: Overall Assessment
    st.markdown("### Part D: Overall Income Quality Score")
    income_score = calculate_income_quality_score(dpu_trend, sustainability, quality_metrics)
    display_income_quality_grade(income_score)


def calculate_balance_sheet_scorecard(reit_data: pd.Series) -> Dict:
    """Calculate 7-metric balance sheet scorecard"""
    scorecard = {}

    # Metric 1: Gearing Ratio (0-15 points)
    gearing = reit_data.get('gearing_ratio', 0)
    if gearing < 30:
        scorecard['gearing'] = {'score': 15, 'status': 'excellent', 'value': gearing}
    elif gearing < 35:
        scorecard['gearing'] = {'score': 12, 'status': 'very_good', 'value': gearing}
    elif gearing < 40:
        scorecard['gearing'] = {'score': 10, 'status': 'good', 'value': gearing}
    elif gearing < 42:
        scorecard['gearing'] = {'score': 7, 'status': 'adequate', 'value': gearing}
    elif gearing < 45:
        scorecard['gearing'] = {'score': 4, 'status': 'elevated', 'value': gearing}
    else:
        scorecard['gearing'] = {'score': 0, 'status': 'high_risk', 'value': gearing}

    # Metric 2: Interest Coverage Ratio (0-15 points)
    icr = reit_data.get('interest_coverage_ratio', 0)
    if icr >= 5.0:
        scorecard['icr'] = {'score': 15, 'status': 'excellent', 'value': icr}
    elif icr >= 4.0:
        scorecard['icr'] = {'score': 12, 'status': 'very_good', 'value': icr}
    elif icr >= 3.0:
        scorecard['icr'] = {'score': 10, 'status': 'good', 'value': icr}
    elif icr >= 2.5:
        scorecard['icr'] = {'score': 7, 'status': 'adequate', 'value': icr}
    elif icr >= 2.0:
        scorecard['icr'] = {'score': 4, 'status': 'weak', 'value': icr}
    else:
        scorecard['icr'] = {'score': 0, 'status': 'critical', 'value': icr}

    # Metric 3: Cost of Debt (0-10 points)
    cost_of_debt = reit_data.get('average_cost_of_debt', 0)
    if cost_of_debt <= 2.5:
        scorecard['cost_of_debt'] = {'score': 10, 'status': 'excellent', 'value': cost_of_debt}
    elif cost_of_debt <= 3.0:
        scorecard['cost_of_debt'] = {'score': 8, 'status': 'very_good', 'value': cost_of_debt}
    elif cost_of_debt <= 3.5:
        scorecard['cost_of_debt'] = {'score': 6, 'status': 'good', 'value': cost_of_debt}
    elif cost_of_debt <= 4.0:
        scorecard['cost_of_debt'] = {'score': 4, 'status': 'adequate', 'value': cost_of_debt}
    elif cost_of_debt <= 4.5:
        scorecard['cost_of_debt'] = {'score': 2, 'status': 'high', 'value': cost_of_debt}
    else:
        scorecard['cost_of_debt'] = {'score': 0, 'status': 'very_high', 'value': cost_of_debt}

    # Metric 4: Unencumbered Assets (0-15 points)
    unencumbered_pct = reit_data.get('unencumbered_assets_percentage', 0)
    if unencumbered_pct >= 70:
        scorecard['unencumbered'] = {'score': 15, 'status': 'excellent', 'value': unencumbered_pct}
    elif unencumbered_pct >= 60:
        scorecard['unencumbered'] = {'score': 12, 'status': 'very_good', 'value': unencumbered_pct}
    elif unencumbered_pct >= 50:
        scorecard['unencumbered'] = {'score': 10, 'status': 'good', 'value': unencumbered_pct}
    elif unencumbered_pct >= 40:
        scorecard['unencumbered'] = {'score': 7, 'status': 'adequate', 'value': unencumbered_pct}
    elif unencumbered_pct >= 30:
        scorecard['unencumbered'] = {'score': 4, 'status': 'limited', 'value': unencumbered_pct}
    else:
        scorecard['unencumbered'] = {'score': 0, 'status': 'very_limited', 'value': unencumbered_pct}

    # Metric 5: Fixed Rate Debt (0-10 points)
    fixed_rate_pct = reit_data.get('fixed_rate_debt_percentage', 0)
    if fixed_rate_pct >= 80:
        scorecard['fixed_rate'] = {'score': 10, 'status': 'excellent', 'value': fixed_rate_pct}
    elif fixed_rate_pct >= 70:
        scorecard['fixed_rate'] = {'score': 8, 'status': 'very_good', 'value': fixed_rate_pct}
    elif fixed_rate_pct >= 60:
        scorecard['fixed_rate'] = {'score': 6, 'status': 'good', 'value': fixed_rate_pct}
    elif fixed_rate_pct >= 50:
        scorecard['fixed_rate'] = {'score': 4, 'status': 'adequate', 'value': fixed_rate_pct}
    elif fixed_rate_pct >= 40:
        scorecard['fixed_rate'] = {'score': 2, 'status': 'limited', 'value': fixed_rate_pct}
    else:
        scorecard['fixed_rate'] = {'score': 0, 'status': 'high_exposure', 'value': fixed_rate_pct}

    # Metric 6: Debt Maturity Profile (0-15 points)
    debt_1yr_pct = reit_data.get('debt_maturity_1yr_percentage', 0)
    if debt_1yr_pct <= 10:
        scorecard['debt_maturity'] = {'score': 15, 'status': 'excellent', 'value': debt_1yr_pct}
    elif debt_1yr_pct <= 15:
        scorecard['debt_maturity'] = {'score': 12, 'status': 'very_good', 'value': debt_1yr_pct}
    elif debt_1yr_pct <= 20:
        scorecard['debt_maturity'] = {'score': 10, 'status': 'good', 'value': debt_1yr_pct}
    elif debt_1yr_pct <= 25:
        scorecard['debt_maturity'] = {'score': 7, 'status': 'adequate', 'value': debt_1yr_pct}
    elif debt_1yr_pct <= 30:
        scorecard['debt_maturity'] = {'score': 4, 'status': 'elevated', 'value': debt_1yr_pct}
    else:
        scorecard['debt_maturity'] = {'score': 0, 'status': 'high_risk', 'value': debt_1yr_pct}

    # Metric 7: Debt-to-Equity (0-20 points)
    debt_to_equity = reit_data.get('debt_to_equity', 0)
    if debt_to_equity <= 0.40:
        scorecard['debt_to_equity'] = {'score': 20, 'status': 'excellent', 'value': debt_to_equity}
    elif debt_to_equity <= 0.50:
        scorecard['debt_to_equity'] = {'score': 16, 'status': 'very_good', 'value': debt_to_equity}
    elif debt_to_equity <= 0.60:
        scorecard['debt_to_equity'] = {'score': 12, 'status': 'good', 'value': debt_to_equity}
    elif debt_to_equity <= 0.70:
        scorecard['debt_to_equity'] = {'score': 8, 'status': 'adequate', 'value': debt_to_equity}
    elif debt_to_equity <= 0.80:
        scorecard['debt_to_equity'] = {'score': 4, 'status': 'elevated', 'value': debt_to_equity}
    else:
        scorecard['debt_to_equity'] = {'score': 0, 'status': 'high_leverage', 'value': debt_to_equity}

    return scorecard


def display_scorecard_table(scorecard: Dict) -> None:
    """Display the 7-metric scorecard table"""
    st.markdown("### 7-Metric Balance Sheet Scorecard")

    # Prepare data for table
    table_data = []

    # Metric 1: Gearing
    gearing = scorecard['gearing']
    status_emoji = {'excellent': '🟢', 'very_good': '🟢', 'good': '✅', 'adequate': '⚠️', 'elevated': '🔴', 'high_risk': '🔴'}[gearing['status']]
    table_data.append(['Gearing Ratio', f"{gearing['value']:.1f}%", f"{gearing['score']}/15", f"{status_emoji} {gearing['status'].replace('_', ' ').title()}"])

    # Metric 2: ICR
    icr = scorecard['icr']
    status_emoji = {'excellent': '🟢', 'very_good': '🟢', 'good': '✅', 'adequate': '⚠️', 'weak': '🔴', 'critical': '🔴'}[icr['status']]
    table_data.append(['Interest Coverage Ratio', f"{icr['value']:.2f}x", f"{icr['score']}/15", f"{status_emoji} {icr['status'].replace('_', ' ').title()}"])

    # Metric 3: Cost of Debt
    cost_debt = scorecard['cost_of_debt']
    status_emoji = {'excellent': '🟢', 'very_good': '🟢', 'good': '✅', 'adequate': '⚠️', 'high': '🔴', 'very_high': '🔴'}[cost_debt['status']]
    table_data.append(['Cost of Debt', f"{cost_debt['value']:.1f}%", f"{cost_debt['score']}/10", f"{status_emoji} {cost_debt['status'].replace('_', ' ').title()}"])

    # Metric 4: Unencumbered Assets
    unencumbered = scorecard['unencumbered']
    status_emoji = {'excellent': '🟢', 'very_good': '🟢', 'good': '✅', 'adequate': '⚠️', 'limited': '🔴', 'very_limited': '🔴'}[unencumbered['status']]
    table_data.append(['Unencumbered Assets', f"{unencumbered['value']:.1f}%", f"{unencumbered['score']}/15", f"{status_emoji} {unencumbered['status'].replace('_', ' ').title()}"])

    # Metric 5: Fixed Rate Debt
    fixed_rate = scorecard['fixed_rate']
    status_emoji = {'excellent': '🟢', 'very_good': '🟢', 'good': '✅', 'adequate': '⚠️', 'limited': '🔴', 'high_exposure': '🔴'}[fixed_rate['status']]
    table_data.append(['Fixed Rate Debt', f"{fixed_rate['value']:.1f}%", f"{fixed_rate['score']}/10", f"{status_emoji} {fixed_rate['status'].replace('_', ' ').title()}"])

    # Metric 6: Debt Maturity
    debt_maturity = scorecard['debt_maturity']
    status_emoji = {'excellent': '🟢', 'very_good': '🟢', 'good': '✅', 'adequate': '⚠️', 'elevated': '🔴', 'high_risk': '🔴'}[debt_maturity['status']]
    table_data.append(['Debt Maturity <1yr', f"{debt_maturity['value']:.1f}%", f"{debt_maturity['score']}/15", f"{status_emoji} {debt_maturity['status'].replace('_', ' ').title()}"])

    # Metric 7: Debt-to-Equity
    debt_equity = scorecard['debt_to_equity']
    status_emoji = {'excellent': '🟢', 'very_good': '🟢', 'good': '✅', 'adequate': '⚠️', 'elevated': '🔴', 'high_leverage': '🔴'}[debt_equity['status']]
    table_data.append(['Debt-to-Equity', f"{debt_equity['value']:.2f}", f"{debt_equity['score']}/20", f"{status_emoji} {debt_equity['status'].replace('_', ' ').title()}"])

    # Calculate total
    total_score = sum(metric['score'] for metric in scorecard.values())
    table_data.append(['─' * 20, '─' * 10, '─' * 10, ''])
    table_data.append(['**TOTAL SCORE**', '', f"**{total_score}/100**", ''])

    # Display table
    df_scorecard = pd.DataFrame(table_data, columns=['Metric', 'Current Value', 'Score', 'Status'])
    st.dataframe(
        df_scorecard,
        column_config={
            'Metric': st.column_config.TextColumn('Balance Sheet Metric', width='medium'),
            'Current Value': st.column_config.TextColumn('Value', width='small'),
            'Score': st.column_config.TextColumn('Score', width='small'),
            'Status': st.column_config.TextColumn('Assessment', width='large')
        },
        hide_index=True,
        width="stretch"
    )


def display_balance_sheet_radar(scorecard: Dict) -> None:
    """Display balance sheet radar chart"""
    st.markdown("#### Balance Sheet Strength Radar")

    # Prepare data for radar chart
    categories = ['Gearing\n(Lower Better)', 'ICR\n(Higher Better)', 'Cost of Debt\n(Lower Better)',
                  'Unencumbered\n(Higher Better)', 'Fixed Rate\n(Higher Better)', 'Debt Maturity\n(Lower Better)', 'D/E Ratio\n(Lower Better)']

    # Normalize scores to 0-100 scale for radar
    scores = [
        scorecard['gearing']['score'] / 15 * 100,  # 15 points max
        scorecard['icr']['score'] / 15 * 100,      # 15 points max
        scorecard['cost_of_debt']['score'] / 10 * 100,  # 10 points max
        scorecard['unencumbered']['score'] / 15 * 100,  # 15 points max
        scorecard['fixed_rate']['score'] / 10 * 100,    # 10 points max
        scorecard['debt_maturity']['score'] / 15 * 100, # 15 points max
        scorecard['debt_to_equity']['score'] / 20 * 100 # 20 points max
    ]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Balance Sheet Strength',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            )),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, width="stretch")


def determine_balance_sheet_grade(total_score: int) -> Dict:
    """Determine balance sheet grade based on total score"""
    if total_score >= 90:
        grade = 'A'
        description = 'FORTRESS - Exceptional financial strength'
        color = 'green'
    elif total_score >= 80:
        grade = 'B'
        description = 'STRONG - Solid balance sheet'
        color = 'blue'
    elif total_score >= 70:
        grade = 'C'
        description = 'ADEQUATE - Acceptable but limited buffer'
        color = 'orange'
    elif total_score >= 60:
        grade = 'D'
        description = 'WEAK - Balance sheet concerns'
        color = 'red'
    else:
        grade = 'F'
        description = 'CRITICAL - Significant financial risk'
        color = 'red'

    return {
        'grade': grade,
        'score': total_score,
        'description': description,
        'color': color
    }


def display_balance_sheet_grade(grade_info: Dict, scorecard: Dict) -> None:
    """Display balance sheet grade and breakdown"""
    st.markdown("### Overall Balance Sheet Assessment")

    grade = grade_info['grade']
    score = grade_info['score']
    description = grade_info['description']

    col1, col2 = st.columns([1, 3])

    with col1:
        # Large grade display
        grade_color = grade_info['color']

        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border: 2px solid {grade_color}; border-radius: 10px;'>
            <h1 style='color: {grade_color}; margin: 0; font-size: 3em;'>{grade}</h1>
            <p style='margin: 5px 0 0 0; font-size: 1.2em;'>{score}/100</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"**Grade: {grade}**")
        st.markdown(f"**Score: {score}/100**")
        st.markdown(f"**Assessment:** {description}")

        # Score breakdown
        st.markdown("**Score Breakdown:**")
        breakdown = {
            'Gearing': scorecard['gearing']['score'],
            'ICR': scorecard['icr']['score'],
            'Cost of Debt': scorecard['cost_of_debt']['score'],
            'Unencumbered': scorecard['unencumbered']['score'],
            'Fixed Rate': scorecard['fixed_rate']['score'],
            'Debt Maturity': scorecard['debt_maturity']['score'],
            'D/E Ratio': scorecard['debt_to_equity']['score']
        }

        for component, points in breakdown.items():
            st.markdown(f"- {component}: {points} points")


def display_debt_maturity_chart(reit_data: pd.Series) -> None:
    """Display debt maturity profile chart"""
    st.markdown("#### Debt Maturity Profile")

    # Get maturity data
    maturity_1yr = reit_data.get('debt_maturity_1yr_percentage', 0)
    maturity_2yr = reit_data.get('debt_maturity_2yr_percentage', 0) if reit_data.get('debt_maturity_2yr_percentage') else 0
    maturity_3yr = reit_data.get('debt_maturity_3yr_percentage', 0) if reit_data.get('debt_maturity_3yr_percentage') else 0
    maturity_beyond_3yr = reit_data.get('debt_maturity_beyond_3yr_percentage', 0) if reit_data.get('debt_maturity_beyond_3yr_percentage') else 0

    # Calculate beyond 3yr if not provided
    if maturity_beyond_3yr == 0:
        maturity_beyond_3yr = 100 - maturity_1yr - maturity_2yr - maturity_3yr

    # Prepare data
    categories = ['<1 Year', '1-2 Years', '2-3 Years', '>3 Years']
    values = [maturity_1yr, maturity_2yr, maturity_3yr, maturity_beyond_3yr]
    colors = ['red', 'orange', 'yellow', 'green']

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='auto',
        showlegend=False
    ))

    fig.update_layout(
        title='Debt Maturity Distribution',
        xaxis_title='Time to Maturity',
        yaxis_title='Percentage of Total Debt (%)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, width="stretch")


def display_debt_composition(reit_data: pd.Series) -> None:
    """Display fixed vs floating rate debt composition"""
    st.markdown("#### Debt Composition")

    fixed_rate = reit_data.get('fixed_rate_debt_percentage', 0)
    floating_rate = reit_data.get('floating_rate_debt_percentage', 0)

    # Calculate if not provided
    if floating_rate == 0:
        floating_rate = 100 - fixed_rate

    # Create pie chart
    labels = ['Fixed Rate', 'Floating Rate']
    values = [fixed_rate, floating_rate]
    colors = ['green', 'red']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo='label+percent',
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title='Fixed vs Floating Rate Debt',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, width="stretch")


def perform_stress_test(reit_data: pd.Series) -> Dict:
    """Perform stress testing on balance sheet"""
    # Current metrics
    current_icr = reit_data.get('interest_coverage_ratio', 0)
    current_gearing = reit_data.get('gearing_ratio', 0)
    current_cost_debt = reit_data.get('average_cost_of_debt', 0)

    # Stress scenario: Rates +1.0%, Property values -10%
    # Assumptions:
    # - Interest expense increases by ~15% (rate rise + property value decline impact)
    # - Gearing increases by ~11% (property values down 10%)

    stressed_icr = current_icr * (1 - 0.15) if current_icr > 0 else 0
    stressed_gearing = current_gearing * 1.11
    stressed_cost_debt = current_cost_debt + 1.0

    # Assess stress test results
    icr_pass = stressed_icr >= 2.5
    gearing_pass = stressed_gearing <= 45
    cost_debt_concern = stressed_cost_debt > 4.5

    if icr_pass and gearing_pass and not cost_debt_concern:
        overall_status = 'passes'
        message = '✅ PASSES stress test - Resilient balance sheet'
    elif icr_pass and gearing_pass:
        overall_status = 'marginal'
        message = '⚠️ MARGINAL under stress - Monitor closely'
    else:
        overall_status = 'fails'
        message = '🔴 FAILS stress test - Vulnerable to shocks'

    return {
        'current': {
            'icr': current_icr,
            'gearing': current_gearing,
            'cost_debt': current_cost_debt
        },
        'stressed': {
            'icr': stressed_icr,
            'gearing': stressed_gearing,
            'cost_debt': stressed_cost_debt
        },
        'status': overall_status,
        'message': message,
        'icr_pass': icr_pass,
        'gearing_pass': gearing_pass,
        'cost_debt_concern': cost_debt_concern
    }


def display_stress_test_results(stress_results: Dict) -> None:
    """Display stress test results"""
    st.markdown("#### Stress Test: Adverse Scenario")

    st.markdown("**Scenario:** Interest rates +1.0%, Property values -10%")

    col1, col2, col3 = st.columns(3)

    with col1:
        current_icr = stress_results['current']['icr']
        stressed_icr = stress_results['stressed']['icr']
        icr_status = '✅' if stress_results['icr_pass'] else '🔴'
        st.metric("Interest Coverage", f"{stressed_icr:.2f}x", f"{current_icr:.2f}x → {stressed_icr:.2f}x")
        st.caption(f"{icr_status} {'Pass' if stress_results['icr_pass'] else 'Fail'}")

    with col2:
        current_gearing = stress_results['current']['gearing']
        stressed_gearing = stress_results['stressed']['gearing']
        gearing_status = '✅' if stress_results['gearing_pass'] else '🔴'
        st.metric("Gearing Ratio", f"{stressed_gearing:.1f}%", f"{current_gearing:.1f}% → {stressed_gearing:.1f}%")
        st.caption(f"{gearing_status} {'Pass' if stress_results['gearing_pass'] else 'Fail'}")

    with col3:
        current_cost = stress_results['current']['cost_debt']
        stressed_cost = stress_results['stressed']['cost_debt']
        cost_status = '⚠️' if stress_results['cost_debt_concern'] else '✅'
        st.metric("Cost of Debt", f"{stressed_cost:.1f}%", f"{current_cost:.1f}% → {stressed_cost:.1f}%")
        st.caption(f"{cost_status} {'High' if stress_results['cost_debt_concern'] else 'Manageable'}")

    st.markdown(f"**Overall Assessment:** {stress_results['message']}")


def display_key_balance_sheet_risks(reit_data: pd.Series, scorecard: Dict) -> None:
    """Display key balance sheet risks and strengths"""
    st.markdown("### Key Balance Sheet Insights")

    strengths = []
    concerns = []

    # Analyze each metric
    gearing = scorecard['gearing']
    if gearing['status'] in ['excellent', 'very_good']:
        strengths.append(f"✅ Conservative gearing at {gearing['value']:.1f}%")
    elif gearing['status'] in ['elevated', 'high_risk']:
        concerns.append(f"⚠️ High gearing at {gearing['value']:.1f}% increases risk")

    icr = scorecard['icr']
    if icr['status'] in ['excellent', 'very_good']:
        strengths.append(f"✅ Strong interest coverage at {icr['value']:.2f}x")
    elif icr['status'] in ['weak', 'critical']:
        concerns.append(f"⚠️ Weak interest coverage at {icr['value']:.2f}x")

    fixed_rate = scorecard['fixed_rate']
    if fixed_rate['status'] in ['excellent', 'very_good']:
        strengths.append(f"✅ High fixed rate debt ({fixed_rate['value']:.1f}%) protects from rate rises")
    elif fixed_rate['status'] in ['limited', 'high_exposure']:
        concerns.append(f"⚠️ High floating rate exposure ({100-fixed_rate['value']:.1f}%)")

    unencumbered = scorecard['unencumbered']
    if unencumbered['status'] in ['excellent', 'very_good']:
        strengths.append(f"✅ Strong unencumbered assets ({unencumbered['value']:.1f}%) for refinancing")
    elif unencumbered['status'] in ['limited', 'very_limited']:
        concerns.append(f"⚠️ Limited unencumbered assets ({unencumbered['value']:.1f}%)")

    debt_maturity = scorecard['debt_maturity']
    if debt_maturity['status'] in ['excellent', 'very_good']:
        strengths.append(f"✅ Well-distributed debt maturity profile")
    elif debt_maturity['status'] in ['elevated', 'high_risk']:
        concerns.append(f"⚠️ High near-term refinancing risk ({debt_maturity['value']:.1f}% due <1yr)")

    # Display strengths
    if strengths:
        st.markdown("**Strengths:**")
        for strength in strengths:
            st.markdown(strength)

    # Display concerns
    if concerns:
        st.markdown("**Key Concerns:**")
        for concern in concerns:
            st.markdown(concern)

    # Overall risk assessment
    total_score = sum(metric['score'] for metric in scorecard.values())

    if total_score >= 80:
        st.success("🛡️ **Overall:** Strong balance sheet provides good downside protection")
    elif total_score >= 60:
        st.warning("⚠️ **Overall:** Balance sheet is adequate but monitor key metrics")
    else:
        st.error("🚨 **Overall:** Balance sheet vulnerabilities require careful monitoring")


def display_section_6_balance_sheet_risk(reit_data: pd.Series, reit_df: pd.DataFrame) -> None:
    """Section 6: Balance Sheet & Risk Assessment"""
    create_section_header("🛡️ Section 6: Balance Sheet & Risk Assessment", "Evaluate financial strength and risk profile with 7-metric scorecard")

    # Calculate 7-metric scorecard
    scorecard = calculate_balance_sheet_scorecard(reit_data)

    # Display scorecard table
    display_scorecard_table(scorecard)

    # Display radar chart
    display_balance_sheet_radar(scorecard)

    # Overall grade
    total_score = sum(metric['score'] for metric in scorecard.values())
    grade_info = determine_balance_sheet_grade(total_score)
    display_balance_sheet_grade(grade_info, scorecard)

    # Additional visualizations
    col1, col2 = st.columns(2)

    with col1:
        display_debt_maturity_chart(reit_data)

    with col2:
        display_debt_composition(reit_data)

    # Stress testing
    st.markdown("### Stress Testing")
    stress_results = perform_stress_test(reit_data)
    display_stress_test_results(stress_results)

    # Key insights
    display_key_balance_sheet_risks(reit_data, scorecard)


def check_value_trap_red_flags(reit_data: pd.Series, nav_trend: Dict, dpu_trend: Dict, spread_analysis: Dict, scorecard: Dict) -> Dict:
    """Check for 7 value trap red flags"""
    red_flags = {}

    # Red Flag 1: Declining NAV Trend
    nav_cagr = nav_trend.get('cagr')
    if nav_cagr is not None and nav_cagr < -3.0:
        red_flags['nav_decline'] = {
            'triggered': True,
            'description': f'NAV declining at {nav_cagr:.1f}% CAGR',
            'severity': 'high',
            'impact': 'Asset value erosion indicates fundamental deterioration'
        }
    else:
        red_flags['nav_decline'] = {'triggered': False}

    # Red Flag 2: Shrinking DPU
    dpu_cagr = dpu_trend.get('cagr')
    if dpu_cagr is not None and dpu_cagr < -5.0:
        red_flags['dpu_decline'] = {
            'triggered': True,
            'description': f'DPU declining at {dpu_cagr:.1f}% CAGR',
            'severity': 'high',
            'impact': 'Income deterioration threatens distribution sustainability'
        }
    else:
        red_flags['dpu_decline'] = {'triggered': False}

    # Red Flag 3: Poor Property Spread
    spread = spread_analysis.get('spread', 0)
    if spread < 1.0:
        red_flags['poor_spread'] = {
            'triggered': True,
            'description': f'Property spread only {spread:.1f}%',
            'severity': 'high',
            'impact': 'Uneconomic leverage destroys shareholder value'
        }
    else:
        red_flags['poor_spread'] = {'triggered': False}

    # Red Flag 4: High Gearing + Weak ICR
    gearing = scorecard['gearing']['value']
    icr = scorecard['icr']['value']
    if gearing > 42 and icr < 2.5:
        red_flags['high_gearing_weak_icr'] = {
            'triggered': True,
            'description': f'High gearing ({gearing:.1f}%) with weak ICR ({icr:.2f}x)',
            'severity': 'critical',
            'impact': 'Refinancing risk creates existential threat'
        }
    else:
        red_flags['high_gearing_weak_icr'] = {'triggered': False}

    # Red Flag 5: Deteriorating Occupancy
    occupancy = reit_data.get('portfolio_occupancy', 100)
    # For single report, we can't check trend, so use absolute level
    if occupancy < 90:
        red_flags['low_occupancy'] = {
            'triggered': True,
            'description': f'Low occupancy at {occupancy:.1f}%',
            'severity': 'medium',
            'impact': 'Weak demand signals operational issues'
        }
    else:
        red_flags['low_occupancy'] = {'triggered': False}

    # Red Flag 6: Negative Rental Reversions
    reversion = reit_data.get('rental_reversion', 0)
    if reversion < -5.0:
        red_flags['negative_reversion'] = {
            'triggered': True,
            'description': f'Negative rental reversion at {reversion:+.1f}%',
            'severity': 'high',
            'impact': 'Weak pricing power indicates competitive disadvantage'
        }
    else:
        red_flags['negative_reversion'] = {'triggered': False}

    # Red Flag 7: Unsustainable Distributions
    payout_ratio = reit_data.get('payout_ratio', 100)
    if payout_ratio > 100:
        red_flags['unsustainable_payout'] = {
            'triggered': True,
            'description': f'Unsustainable payout ratio at {payout_ratio:.1f}%',
            'severity': 'critical',
            'impact': 'Distributions exceed income - capital return, not yield'
        }
    else:
        red_flags['unsustainable_payout'] = {'triggered': False}

    return red_flags


def display_red_flags_table(red_flags: Dict) -> None:
    """Display the red flags table"""
    st.markdown("### Value Trap Red Flags Check")

    # Prepare data for table
    table_data = []

    flag_descriptions = {
        'nav_decline': 'Declining NAV Trend (CAGR < -3%)',
        'dpu_decline': 'Shrinking DPU (CAGR < -5%)',
        'poor_spread': 'Poor Property Spread (<1.0%)',
        'high_gearing_weak_icr': 'High Gearing + Weak ICR',
        'low_occupancy': 'Deteriorating Occupancy (<90%)',
        'negative_reversion': 'Negative Rental Reversions (<-5%)',
        'unsustainable_payout': 'Unsustainable Payout (>100%)'
    }

    severity_colors = {
        'low': '🟡',
        'medium': '🟠',
        'high': '🔴',
        'critical': '🚨'
    }

    for flag_key, flag_info in red_flags.items():
        flag_name = flag_descriptions[flag_key]

        if flag_info['triggered']:
            status = f"{severity_colors[flag_info['severity']]} TRIGGERED"
            description = flag_info['description']
            impact = flag_info['impact']
        else:
            status = "✅ CLEAR"
            description = "No concerns detected"
            impact = "No issues identified"

        table_data.append([flag_name, status, description, impact])

    # Display table
    df_flags = pd.DataFrame(table_data, columns=['Red Flag', 'Status', 'Details', 'Impact'])
    st.dataframe(
        df_flags,
        column_config={
            'Red Flag': st.column_config.TextColumn('Value Trap Red Flag', width='medium'),
            'Status': st.column_config.TextColumn('Status', width='small'),
            'Details': st.column_config.TextColumn('Details', width='medium'),
            'Impact': st.column_config.TextColumn('Business Impact', width='large')
        },
        hide_index=True,
        width="stretch"
    )


def display_value_trap_assessment(red_flags: Dict) -> None:
    """Display overall value trap assessment"""
    st.markdown("### Overall Value Trap Risk Assessment")

    # Count red flags
    triggered_flags = sum(1 for flag in red_flags.values() if flag['triggered'])
    total_flags = len(red_flags)

    # Determine risk level
    if triggered_flags == 0:
        risk_level = 'CLEAR'
        risk_color = 'green'
        risk_description = 'No value trap red flags detected'
        recommendation = '✅ Proceed with analysis - appears fundamentally sound'
    elif triggered_flags == 1:
        risk_level = 'LOW RISK'
        risk_color = 'blue'
        risk_description = 'One red flag detected'
        recommendation = '⚠️ Exercise caution - monitor the identified concern'
    elif triggered_flags == 2:
        risk_level = 'MODERATE RISK'
        risk_color = 'orange'
        risk_description = 'Multiple red flags suggest caution needed'
        recommendation = '⚠️⚠️ Significant concerns - deep due diligence required'
    elif triggered_flags <= 4:
        risk_level = 'HIGH RISK'
        risk_color = 'red'
        risk_description = 'Several red flags indicate potential value trap'
        recommendation = '❌ High risk - consider avoiding unless exceptional discount'
    else:
        risk_level = 'EXTREME RISK'
        risk_color = 'red'
        risk_description = 'Multiple critical red flags'
        recommendation = '🚨 Extreme caution - likely value trap, avoid investment'

    # Display assessment
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border: 2px solid {risk_color}; border-radius: 10px;'>
            <h2 style='color: {risk_color}; margin: 0;'>{risk_level}</h2>
            <p style='margin: 10px 0 0 0; font-size: 1.5em;'>{triggered_flags}/{total_flags}</p>
            <p style='margin: 5px 0 0 0; font-size: 0.9em;'>Red Flags</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"**Risk Level:** {risk_level}")
        st.markdown(f"**Red Flags Detected:** {triggered_flags} out of {total_flags}")
        st.markdown(f"**Assessment:** {risk_description}")
        st.markdown(f"**Recommendation:** {recommendation}")

        # Show breakdown by severity
        severity_count = {}
        for flag in red_flags.values():
            if flag['triggered']:
                severity = flag['severity']
                severity_count[severity] = severity_count.get(severity, 0) + 1

        if severity_count:
            st.markdown("**Severity Breakdown:**")
            for severity, count in severity_count.items():
                severity_emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴', 'critical': '🚨'}[severity]
                st.markdown(f"- {severity_emoji} {severity.title()}: {count} flag(s)")


def display_section_7_value_trap_assessment(reit_data: pd.Series, nav_trend: Dict, dpu_trend: Dict, spread_analysis: Dict, scorecard: Dict) -> None:
    """Section 7: Value Trap Assessment"""
    create_section_header("🚨 Section 7: Value Trap Assessment", "Identify red flags that could indicate a value trap")

    # Check red flags
    red_flags = check_value_trap_red_flags(reit_data, nav_trend, dpu_trend, spread_analysis, scorecard)

    # Display red flags table
    display_red_flags_table(red_flags)

    # Display overall assessment
    display_value_trap_assessment(red_flags)


def calculate_overall_score(spread_analysis: Dict, valuation_data: Dict, income_score: Dict, balance_sheet_score: int, value_trap_flags: Dict) -> Dict:
    """Calculate weighted overall investment score (0-100)"""

    # Property Spread Score (0-100, already calculated)
    property_spread_score = spread_analysis.get('score', 50)

    # Valuation Score (0-100, based on MOS)
    mos = valuation_data.get('mos', 0)
    if mos is not None:
        if mos > 25:
            valuation_score = 95
        elif mos > 20:
            valuation_score = 85
        elif mos > 15:
            valuation_score = 75
        elif mos > 10:
            valuation_score = 65
        elif mos > 5:
            valuation_score = 55
        elif mos > 0:
            valuation_score = 45
        else:
            valuation_score = 25
    else:
        valuation_score = 50

    # Income Quality Score (0-100, already calculated)
    income_quality_score = income_score.get('score', 50)

    # Balance Sheet Score (0-100, already calculated)
    balance_sheet_score = min(100, balance_sheet_score)

    # Value Trap Score (0-100, based on red flags)
    triggered_flags = sum(1 for flag in value_trap_flags.values() if flag['triggered'])
    if triggered_flags == 0:
        value_trap_score = 100
    elif triggered_flags == 1:
        value_trap_score = 80
    elif triggered_flags == 2:
        value_trap_score = 60
    elif triggered_flags == 3:
        value_trap_score = 40
    elif triggered_flags == 4:
        value_trap_score = 20
    else:
        value_trap_score = 0

    # Weighted Overall Score
    weights = {
        'property_spread': 0.25,    # 25% - Most important economic metric
        'valuation': 0.20,          # 20% - Price matters
        'income_quality': 0.20,     # 20% - Income sustainability
        'balance_sheet': 0.20,      # 20% - Financial strength
        'value_trap': 0.15          # 15% - Risk adjustment
    }

    overall_score = (
        property_spread_score * weights['property_spread'] +
        valuation_score * weights['valuation'] +
        income_quality_score * weights['income_quality'] +
        balance_sheet_score * weights['balance_sheet'] +
        value_trap_score * weights['value_trap']
    )

    return {
        'overall_score': overall_score,
        'component_scores': {
            'property_spread': property_spread_score,
            'valuation': valuation_score,
            'income_quality': income_quality_score,
            'balance_sheet': balance_sheet_score,
            'value_trap': value_trap_score
        },
        'weights': weights
    }


def determine_investment_rating(overall_score: float) -> Dict:
    """Determine investment rating based on overall score"""
    if overall_score >= 85:
        rating = 'STRONG BUY'
        color = 'green'
        description = 'Exceptional opportunity - all factors align'
        confidence = 'High'
    elif overall_score >= 70:
        rating = 'BUY'
        color = 'blue'
        description = 'Good investment case - positive fundamentals'
        confidence = 'Medium-High'
    elif overall_score >= 55:
        rating = 'HOLD'
        color = 'orange'
        description = 'Fair value - no strong conviction either way'
        confidence = 'Medium'
    elif overall_score >= 40:
        rating = 'AVOID'
        color = 'red'
        description = 'Better opportunities elsewhere'
        confidence = 'Medium'
    else:
        rating = 'SELL/AVOID'
        color = 'red'
        description = 'Significant concerns - avoid investment'
        confidence = 'High'

    return {
        'rating': rating,
        'color': color,
        'description': description,
        'confidence': confidence,
        'score': overall_score
    }


def display_investment_decision_framework(rating_info: Dict, reit_data: pd.Series, current_price: float) -> None:
    """Display the investment decision framework"""
    st.markdown("### Investment Decision Framework")

    rating = rating_info['rating']
    color = rating_info['color']
    description = rating_info['description']
    confidence = rating_info['confidence']
    score = rating_info['score']

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px;'>
            <h1 style='color: {color}; margin: 0; font-size: 2em;'>{rating}</h1>
            <p style='margin: 10px 0 0 0; font-size: 1.2em;'>{score:.0f}/100</p>
            <p style='margin: 5px 0 0 0; font-size: 0.9em;'>Confidence: {confidence}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"**Investment Rating:** {rating}")
        st.markdown(f"**Overall Score:** {score:.0f}/100")
        st.markdown(f"**Description:** {description}")
        st.markdown(f"**Confidence Level:** {confidence}")

        # Key decision factors
        st.markdown("**Key Decision Factors:**")

        # Current valuation
        pnav = current_price / reit_data.get('nav_per_unit', current_price)
        if pnav < 0.85:
            st.markdown("✅ Attractive valuation (P/NAV < 0.85x)")
        elif pnav < 0.95:
            st.markdown("⚪ Fair valuation (P/NAV 0.85-0.95x)")
        else:
            st.markdown("⚠️ Expensive valuation (P/NAV > 0.95x)")

        # Yield
        yield_pct = (reit_data.get('dpu', 0) / 100 / current_price) * 100 * 2  # Convert cents to dollars, then annualize
        if yield_pct >= 6.0:
            st.markdown(f"✅ Attractive yield ({yield_pct:.1f}%)")
        elif yield_pct >= 5.0:
            st.markdown(f"⚪ Reasonable yield ({yield_pct:.1f}%)")
        else:
            st.markdown(f"⚠️ Low yield ({yield_pct:.1f}%)")

        # Property spread
        spread = (reit_data.get('net_property_income_annual', 0) / reit_data.get('gross_asset_value', 1) * 100) - reit_data.get('average_cost_of_debt', 0)
        if spread >= 2.0:
            st.markdown(f"✅ Strong property spread ({spread:.1f}%)")
        elif spread >= 1.0:
            st.markdown(f"⚠️ Adequate property spread ({spread:.1f}%)")
        else:
            st.markdown(f"❌ Weak property spread ({spread:.1f}%)")


def display_action_plan(rating_info: Dict, reit_data: pd.Series, current_price: float) -> None:
    """Display specific action plan based on rating"""
    st.markdown("### Action Plan & Recommendations")

    rating = rating_info['rating']
    reit_name = reit_data.get('reit_name', 'REIT')

    if rating == 'STRONG BUY':
        st.success("🎯 **Aggressive Investment Recommended**")
        st.markdown(f"""
        **Immediate Actions:**
        • Establish position in {reit_name} at current price (S${current_price:.3f})
        • Consider increasing allocation if within risk limits
        • Set stop loss at 10% below current price

        **Position Sizing:**
        • Up to 5-8% of portfolio for conservative investors
        • Up to 10-15% of portfolio for aggressive investors

        **Monitoring:**
        • Track quarterly earnings for NAV and DPU stability
        • Monitor property spread trends
        • Review balance sheet metrics annually
        """)

    elif rating == 'BUY':
        st.info("✅ **Positive Investment Case**")
        st.markdown(f"""
        **Recommended Actions:**
        • Accumulate {reit_name} gradually at current levels
        • Consider dollar-cost averaging over 1-2 months
        • Target entry price near S${current_price:.3f}

        **Position Sizing:**
        • 3-5% of portfolio for balanced investors
        • Monitor valuation for better entry points

        **Monitoring:**
        • Watch for P/NAV improvements
        • Track rental reversion trends
        • Review occupancy rates quarterly
        """)

    elif rating == 'HOLD':
        st.warning("⏳ **Wait for Better Entry**")
        st.markdown(f"""
        **Recommended Actions:**
        • Hold existing positions if any
        • Wait for more attractive valuation (P/NAV < 0.90x)
        • Consider sector rotation to higher-conviction REITs

        **Entry Criteria:**
        • Target P/NAV ratio: < 0.85x
        • Target yield: > 6.0%
        • Improved property spread

        **Monitoring:**
        • Track sector performance vs {reit_name}
        • Monitor for NAV accretion opportunities
        """)

    elif rating == 'AVOID':
        st.error("❌ **Seek Better Opportunities**")
        st.markdown(f"""
        **Recommended Actions:**
        • Avoid new investment in {reit_name}
        • Consider reducing existing positions gradually
        • Focus on higher-quality REITs with better fundamentals

        **Alternative Focus:**
        • Look for REITs with property spread > 2.0%
        • Prioritize balance sheet grade A or B
        • Seek income quality score > 70

        **Exit Triggers:**
        • Valuation becomes more attractive (P/NAV < 0.80x)
        • Significant improvement in fundamentals
        """)

    else:  # SELL/AVOID
        st.error("🚨 **Strongly Avoid Investment**")
        st.markdown(f"""
        **Immediate Actions:**
        • Exit existing positions in {reit_name} if any
        • Avoid completely until major fundamental improvements
        • Consider short position if valuation expertise allows

        **Exit Strategy:**
        • Sell gradually to minimize market impact
        • Target complete exit within 1-2 months
        • Book losses if necessary to reallocate capital

        **Re-entry Criteria:**
        • Property spread > 1.5%
        • Balance sheet grade improves to C or better
        • Clear path to NAV and DPU stability
        """)


def display_summary_dashboard(all_scores: Dict, reit_data: pd.Series, current_price: float) -> None:
    """Display comprehensive summary dashboard"""
    st.markdown("### Comprehensive Analysis Summary")

    # Prepare summary data
    summary_data = [
        ['Property Spread Analysis', f"{all_scores['component_scores']['property_spread']:.0f}/100", 'Economic model quality'],
        ['Valuation Assessment', f"{all_scores['component_scores']['valuation']:.0f}/100", 'Price vs intrinsic value'],
        ['Income Quality Score', f"{all_scores['component_scores']['income_quality']:.0f}/100", 'Distribution sustainability'],
        ['Balance Sheet Grade', f"{all_scores['component_scores']['balance_sheet']:.0f}/100", 'Financial strength'],
        ['Value Trap Risk', f"{all_scores['component_scores']['value_trap']:.0f}/100", 'Red flag assessment'],
        ['─' * 20, '─' * 10, ''],
        ['**OVERALL SCORE**', f"**{all_scores['overall_score']:.0f}/100**", '**Final Investment Rating**']
    ]

    df_summary = pd.DataFrame(summary_data, columns=['Analysis Section', 'Score', 'Description'])
    st.dataframe(
        df_summary,
        column_config={
            'Analysis Section': st.column_config.TextColumn('Analysis Section', width='medium'),
            'Score': st.column_config.TextColumn('Score', width='small'),
            'Description': st.column_config.TextColumn('Focus Area', width='large')
        },
        hide_index=True,
        width="stretch"
    )

    # Key metrics summary
    st.markdown("#### Key Investment Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pnav = current_price / reit_data.get('nav_per_unit', current_price)
        st.metric("Current P/NAV", f"{pnav:.2f}x")

    with col2:
        yield_pct = (reit_data.get('dpu', 0) / 100 / current_price) * 100 * 2  # Convert cents to dollars, then annualize
        st.metric("Distribution Yield", f"{yield_pct:.1f}%")

    with col3:
        spread = (reit_data.get('net_property_income_annual', 0) / reit_data.get('gross_asset_value', 1) * 100) - reit_data.get('average_cost_of_debt', 0)
        st.metric("Property Spread", f"{spread:.1f}%")

    with col4:
        gearing = reit_data.get('gearing_ratio', 0)
        st.metric("Gearing Ratio", f"{gearing:.1f}%")


def display_section_8_investment_decision(all_scores: Dict, reit_data: pd.Series, current_price: float, red_flags: Dict) -> None:
    """Section 8: Investment Decision & Recommendation"""
    create_section_header("🎯 Section 8: Investment Decision & Recommendation", "Synthesize all analysis into actionable investment decision")

    # Calculate overall score
    overall_scores = calculate_overall_score(
        spread_analysis={'score': all_scores.get('property_spread', 50)},
        valuation_data={'mos': all_scores.get('mos', 0)},
        income_score={'score': all_scores.get('income_quality', 50)},
        balance_sheet_score=all_scores.get('balance_sheet', 50),
        value_trap_flags=red_flags
    )

    # Determine rating
    rating_info = determine_investment_rating(overall_scores['overall_score'])

    # Display decision framework
    display_investment_decision_framework(rating_info, reit_data, current_price)

    # Display action plan
    display_action_plan(rating_info, reit_data, current_price)

    # Display summary dashboard
    display_summary_dashboard(overall_scores, reit_data, current_price)


def show():
    """Main Analysis (REITs) page"""
    st.title("📊 Analysis (REITs)")

    create_info_box(
        "🏗️ **Singapore REIT Intrinsic Value Analyzer**\n\n"
        "Comprehensive REIT valuation following Framework v2.1 with Property Yield Spread economics. "
        "Analyze REITs using 8-section framework including critical economic model validation."
    )

    # Get available REITs
    available_reits = get_available_reits()

    if not available_reits:
        create_warning_box(
            "⚠️ No REIT data available. Please import REIT earnings reports first using the REIT importer."
        )
        return

    # REIT selection
    selected_ticker = st.selectbox(
        "🏢 Select REIT to analyze:",
        options=available_reits,
        help="Choose a REIT from your imported earnings data"
    )

    if selected_ticker:
        # Load REIT data
        with st.spinner(f"Loading data for {selected_ticker}..."):
            reit_df = load_reit_data(selected_ticker)

        if reit_df.empty:
            create_error_box(f"No REIT data found for {selected_ticker}")
            return

        # Get latest report
        latest_report = reit_df.iloc[-1]

        # Current price input (required for analysis)
        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            current_price = st.number_input(
                "💰 Current Stock Price (S$)",
                min_value=0.001,
                value=1.00,
                step=0.01,
                help="Required for P/NAV and yield calculations",
                key="current_price"
            )

        with col2:
            st.markdown("**Data Source:** Latest earnings report")
            st.markdown(f"**Report Date:** {latest_report.get('report_date', 'Unknown')}")
            st.markdown(f"**REIT Name:** {latest_report.get('reit_name', 'Unknown')}")

        if current_price <= 0:
            create_warning_box("⚠️ Please enter a valid current stock price to proceed with analysis")
            return

        # Analysis sections
        display_section_1_quick_snapshot(latest_report, current_price)
        st.markdown("---")
        display_section_2_economic_model(latest_report)
        st.markdown("---")
        market_context = display_section_3_data_validation(latest_report)
        st.markdown("---")
        display_section_4_valuation(latest_report, reit_df, current_price)
        st.markdown("---")
        display_section_5_income_sustainability(latest_report, reit_df)
        st.markdown("---")
        display_section_6_balance_sheet_risk(latest_report, reit_df)
        st.markdown("---")

        # Calculate scores for final sections
        nav_trend = calculate_nav_trend(reit_df)
        dpu_trend = calculate_dpu_trend(reit_df)
        spread_analysis = calculate_property_yield_spread(latest_report)
        scorecard = calculate_balance_sheet_scorecard(latest_report)

        display_section_7_value_trap_assessment(latest_report, nav_trend, dpu_trend, spread_analysis, scorecard)
        st.markdown("---")

        # Collect all scores for final decision
        all_scores = {
            'property_spread': spread_analysis.get('score', 50),
            'mos': calculate_margin_of_safety(
                calculate_fair_value(latest_report.get('nav_per_unit', 0),
                                   get_sector_fair_pnav(latest_report.get('reit_sector', 'suburban_retail'))['median'] +
                                   sum(calculate_quality_adjustments(latest_report, spread_analysis).values()))['base'],
                current_price
            ).get('mos', 0),
            'income_quality': calculate_income_quality_score(dpu_trend, assess_distribution_sustainability(latest_report), assess_income_quality(latest_report, reit_df))['score'],
            'balance_sheet': sum(metric['score'] for metric in scorecard.values()) if scorecard else 50
        }

        red_flags = check_value_trap_red_flags(latest_report, nav_trend, dpu_trend, spread_analysis, scorecard)
        display_section_8_investment_decision(all_scores, latest_report, current_price, red_flags)


if __name__ == "__main__":
    show()
