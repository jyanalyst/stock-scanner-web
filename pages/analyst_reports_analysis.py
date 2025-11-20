# File: pages/analyst_reports_analysis.py
"""
Analysis (Analyst Reports) - Dedicated page for analyst report analysis
Moved from Scanner page for better organization and standalone functionality
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
from pages.common.ui_components import (
    create_section_header, create_info_box, create_warning_box,
    create_success_box, create_error_box
)
from pages.common.error_handler import handle_error
from utils import analyst_reports


def load_all_analyst_reports() -> pd.DataFrame:
    """Load all analyst reports from the data source"""
    try:
        from utils.analyst_reports import get_cached_reports
        all_reports, latest_reports = get_cached_reports()

        if latest_reports.empty:
            return pd.DataFrame()

        return latest_reports
    except Exception as e:
        handle_error(e, "AnalystReportsLoading")
        return pd.DataFrame()


def get_stocks_with_reports(reports_df: pd.DataFrame) -> List[str]:
    """Get list of unique tickers that have analyst reports"""
    if reports_df.empty:
        return []

    return sorted(reports_df['ticker_sgx'].unique().tolist())


def display_analyst_report_details(stock_data: pd.Series) -> None:
    """Display detailed analyst report information for a selected stock"""
    ticker = stock_data['ticker_sgx']
    name = stock_data.get('company_name', ticker)

    st.markdown(f"### 📊 {ticker} - {name}")

    # Basic report info
    col1, col2, col3 = st.columns(3)
    with col1:
        sentiment = stock_data.get('sentiment_label', 'unknown')
        sentiment_emoji = analyst_reports.format_sentiment_emoji(sentiment)
        st.metric("Sentiment", f"{sentiment_emoji} {sentiment.title()}")
    with col2:
        score = stock_data.get('sentiment_score', 0)
        st.metric("Sentiment Score", f"{score:.2f}", help="Range: -1 (bearish) to +1 (bullish)")
    with col3:
        recommendation = stock_data.get('recommendation', 'N/A')
        st.metric("Recommendation", recommendation)

    # Report details
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Report Information:**")
        report_date = stock_data.get('report_date')
        if report_date:
            if isinstance(report_date, str):
                report_date = report_date[:10]
            st.write(f"📅 **Date:** {report_date}")
        else:
            st.write("📅 **Date:** Unknown")

        analyst_firm = stock_data.get('analyst_firm', 'Unknown')
        st.write(f"🏢 **Analyst Firm:** {analyst_firm}")

        report_count = stock_data.get('report_count', 1)
        st.write(f"📄 **Total Reports:** {report_count}")

    with col2:
        st.markdown("**Price Information:**")
        price_target = stock_data.get('price_target')
        if price_target:
            st.write(f"🎯 **Price Target:** ${price_target:.2f}")
        else:
            st.write("🎯 **Price Target:** N/A")

        price_at_report = stock_data.get('price_at_report')
        if price_at_report:
            st.write(f"💰 **Price at Report:** ${price_at_report:.2f}")
        else:
            st.write("💰 **Price at Report:** N/A")

        upside_pct = stock_data.get('upside_pct')
        if upside_pct is not None:
            color = "green" if upside_pct > 0 else "red"
            st.write(f"📈 **Upside:** {upside_pct:+.1f}%")

    # Catalysts and Risks
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Key Catalysts")
        catalysts = stock_data.get('key_catalysts', [])
        if catalysts:
            for catalyst in catalysts:
                st.write(f"• {catalyst}")
        else:
            st.info("No catalysts specified")

    with col2:
        st.markdown("### ⚠️ Key Risks")
        risks = stock_data.get('key_risks', [])
        if risks:
            for risk in risks:
                st.write(f"• {risk}")
        else:
            st.info("No risks specified")

    # Executive Summary
    st.markdown("---")
    st.markdown("### 📝 Executive Summary")
    summary = stock_data.get('executive_summary', '')
    if summary:
        st.write(summary)
    else:
        st.info("No executive summary available")

    # Report History (if multiple reports)
    if stock_data.get('report_count', 1) > 1:
        st.markdown("---")
        st.markdown("### 📚 Report History")

        try:
            from utils.analyst_reports import get_report_history
            all_reports, _ = analyst_reports.get_cached_reports()
            history_df = get_report_history(all_reports, stock_data['ticker_sgx'])

            if not history_df.empty and len(history_df) > 1:
                # Show sentiment trend
                trend = analyst_reports.get_sentiment_trend_description(history_df)
                st.info(f"📈 **Sentiment Trend:** {trend}")

                # Show historical reports table
                history_display = history_df[['report_date', 'sentiment_label', 'recommendation', 'price_target']].copy()
                history_display['sentiment_label'] = history_display['sentiment_label'].apply(
                    lambda x: f"{analyst_reports.format_sentiment_emoji(x)} {x.title()}"
                )
                history_display['report_date'] = history_display['report_date'].dt.strftime('%Y-%m-%d')

                st.dataframe(
                    history_display,
                    column_config={
                        'report_date': st.column_config.TextColumn('Date', width='small'),
                        'sentiment_label': st.column_config.TextColumn('Sentiment', width='medium'),
                        'recommendation': st.column_config.TextColumn('Recommendation', width='medium'),
                        'price_target': st.column_config.NumberColumn('Price Target', format='$%.2f')
                    },
                    hide_index=True,
                    width="stretch"
                )
        except Exception as e:
            st.warning(f"Could not load report history: {e}")


def show():
    """Main Analysis (Analyst Reports) page"""
    st.title("📊 Analysis (Analyst Reports)")

    create_info_box(
        "📊 **About This Tool**\n\n"
        "Comprehensive analysis of analyst reports with sentiment analysis, price targets, "
        "and historical tracking. View detailed reports for any stock with analyst coverage."
    )

    # Load all analyst reports
    with st.spinner("Loading analyst reports..."):
        reports_df = load_all_analyst_reports()

    if reports_df.empty:
        create_warning_box(
            "⚠️ No analyst reports available. Please upload analyst reports to begin analysis."
        )
        return

    # Get stocks with reports
    stocks_with_reports = get_stocks_with_reports(reports_df)

    if not stocks_with_reports:
        create_warning_box("⚠️ No stocks with analyst reports found.")
        return

    # Summary statistics
    create_section_header(f"📊 Analyst Reports Overview - {len(stocks_with_reports)} stocks", "")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(stocks_with_reports))
    with col2:
        total_reports = len(reports_df)
        st.metric("Total Reports", total_reports)
    with col3:
        avg_reports_per_stock = total_reports / len(stocks_with_reports)
        st.metric("Avg Reports/Stock", f"{avg_reports_per_stock:.1f}")
    with col4:
        sentiment_counts = reports_df['sentiment_label'].value_counts()
        positive_pct = (sentiment_counts.get('positive', 0) / len(reports_df)) * 100
        st.metric("Positive Sentiment", f"{positive_pct:.0f}%")

    # Stock selection dropdown
    selected_ticker = st.selectbox(
        "📋 Select stock to view detailed analyst report:",
        options=stocks_with_reports,
        help="Choose a stock to view its detailed analyst report information"
    )

    if selected_ticker:
        # Find the latest report for this stock
        stock_reports = reports_df[reports_df['ticker_sgx'] == selected_ticker].copy()

        if not stock_reports.empty:
            # Sort by date and get the latest
            stock_reports['report_date'] = pd.to_datetime(stock_reports['report_date'])
            latest_report = stock_reports.sort_values('report_date', ascending=False).iloc[0]

            # Display detailed report
            display_analyst_report_details(latest_report)
        else:
            st.error(f"No reports found for {selected_ticker}")

    # Show all reports table
    st.markdown("---")
    create_section_header("📋 All Analyst Reports", "")

    # Prepare display dataframe
    display_df = reports_df.copy()
    display_df['sentiment_label'] = display_df['sentiment_label'].apply(
        lambda x: f"{analyst_reports.format_sentiment_emoji(x)} {x.title()}"
    )

    # Format dates
    if 'report_date' in display_df.columns:
        display_df['report_date'] = pd.to_datetime(display_df['report_date']).dt.strftime('%Y-%m-%d')

    st.dataframe(
        display_df[['ticker_sgx', 'company_name', 'report_date', 'sentiment_label',
                   'recommendation', 'price_target', 'analyst_firm']],
        column_config={
            'ticker_sgx': st.column_config.TextColumn('Ticker', width='small'),
            'company_name': st.column_config.TextColumn('Company', width='medium'),
            'report_date': st.column_config.TextColumn('Date', width='small'),
            'sentiment_label': st.column_config.TextColumn('Sentiment', width='small'),
            'recommendation': st.column_config.TextColumn('Recommendation', width='medium'),
            'price_target': st.column_config.NumberColumn('Price Target', format='$%.2f'),
            'analyst_firm': st.column_config.TextColumn('Analyst Firm', width='medium')
        },
        hide_index=True,
        width="stretch"
    )


if __name__ == "__main__":
    show()
