# File: utils/earnings_reports.py
"""
Earnings Report Utilities
Load and manage earnings report data from JSON files
Similar structure to analyst_reports.py but for earnings data
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def get_earnings_reports_dir() -> Path:
    """Get the earnings reports directory"""
    from utils.paths import EARNINGS_REPORTS_DIR
    return EARNINGS_REPORTS_DIR


def load_all_earnings_reports() -> pd.DataFrame:
    """
    Load all earnings report JSONs into a DataFrame
    
    Returns:
        DataFrame with all earnings data including company_type field
    """
    reports_dir = get_earnings_reports_dir()
    
    if not reports_dir.exists():
        logger.warning(f"Earnings reports directory not found: {reports_dir}")
        return pd.DataFrame()
    
    # Find all JSON files
    json_files = list(reports_dir.glob("*.json"))
    
    if not json_files:
        logger.info("No earnings report JSON files found")
        return pd.DataFrame()
    
    # Load all JSONs
    reports = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
                # Calculate report age
                if 'report_date' in report_data:
                    report_date = datetime.strptime(report_data['report_date'], '%Y-%m-%d').date()
                    report_data['report_age_days'] = (date.today() - report_date).days
                else:
                    report_data['report_age_days'] = None
                
                reports.append(report_data)
                
        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {e}")
            continue
    
    if not reports:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(reports)
    
    # Ensure report_date is datetime
    if 'report_date' in df.columns:
        df['report_date'] = pd.to_datetime(df['report_date'])
    
    logger.info(f"Loaded {len(df)} earnings reports from {len(json_files)} files")
    
    return df


def get_latest_earnings_only(all_reports_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get only the latest earnings report for each ticker
    
    Args:
        all_reports_df: DataFrame with all reports
        
    Returns:
        DataFrame with one row per ticker (latest report only)
        Includes 'report_count' column showing total reports available
    """
    if all_reports_df.empty:
        return pd.DataFrame()
    
    # Sort by date (newest first)
    df_sorted = all_reports_df.sort_values('report_date', ascending=False)
    
    # Keep only latest report per ticker
    latest_df = df_sorted.drop_duplicates(subset='ticker_sgx', keep='first').copy()
    
    # Add report count
    report_counts = all_reports_df.groupby('ticker_sgx').size()
    latest_df['report_count'] = latest_df['ticker_sgx'].map(report_counts)
    
    logger.info(f"Latest earnings: {len(latest_df)} unique tickers")
    
    return latest_df


def get_earnings_history(all_reports_df: pd.DataFrame, ticker_sgx: str) -> pd.DataFrame:
    """
    Get all earnings reports for a specific ticker, sorted by date (newest first)
    
    Args:
        all_reports_df: DataFrame with all reports
        ticker_sgx: Ticker in SGX format (e.g., "N2IU.SG")
        
    Returns:
        DataFrame with all earnings reports for this ticker
    """
    if all_reports_df.empty:
        return pd.DataFrame()
    
    ticker_reports = all_reports_df[all_reports_df['ticker_sgx'] == ticker_sgx].copy()
    ticker_reports = ticker_reports.sort_values('report_date', ascending=False)
    
    return ticker_reports


def format_guidance_emoji(guidance_tone: str) -> str:
    """Get emoji for guidance tone"""
    emoji_map = {
        'positive': '📈',
        'neutral': '➖',
        'negative': '📉'
    }
    return emoji_map.get(guidance_tone, '❓')


def format_percentage_change(value: float) -> str:
    """
    Format percentage change with color indication
    
    Returns:
        String like "+15.3%" or "-5.2%"
    """
    if value is None or pd.isna(value):
        return "N/A"
    
    return f"{value:+.1f}%"


def format_currency(value: float, currency: str = "SGD") -> str:
    """
    Format currency values with appropriate units
    
    Returns:
        String like "SGD 125.3M" or "SGD 1.2B"
    """
    if value is None or pd.isna(value):
        return "N/A"
    
    # Convert to millions/billions
    if abs(value) >= 1_000_000_000:
        return f"{currency} {value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{currency} {value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{currency} {value/1_000:.1f}K"
    else:
        return f"{currency} {value:.0f}"


def format_report_age(age_days: int) -> str:
    """
    Format report age in human-readable form
    
    Returns:
        String like "7 days old" or "3 months old"
        Adds warning emoji if >120 days (4 months)
    """
    if age_days is None:
        return "Unknown age"
    
    if age_days == 0:
        return "Today"
    elif age_days == 1:
        return "1 day old"
    elif age_days < 30:
        return f"{age_days} days old"
    elif age_days < 120:
        months = age_days // 30
        return f"{months} month{'s' if months > 1 else ''} old"
    else:
        months = age_days // 30
        return f"⚠️ {months} months old"


def get_earnings_trend_description(history_df: pd.DataFrame) -> str:
    """
    Analyze earnings trend across multiple quarters
    
    Args:
        history_df: DataFrame with earnings history (sorted newest first)
        
    Returns:
        Description like "Revenue Growing" or "Margins Declining"
    """
    if len(history_df) < 2:
        return "Single report"
    
    # Look at revenue trend
    revenue_changes = history_df['revenue_yoy_change'].dropna()
    
    if len(revenue_changes) >= 2:
        recent_avg = revenue_changes.head(2).mean()
        
        if recent_avg > 10:
            return "📈 Strong Growth"
        elif recent_avg > 5:
            return "📈 Growing"
        elif recent_avg > -5:
            return "➖ Flat"
        else:
            return "📉 Declining"
    
    return "Insufficient data"


def merge_earnings_with_scan_results(scan_results_df: pd.DataFrame, 
                                      latest_earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge earnings reports with scan results
    
    Args:
        scan_results_df: Scanner results DataFrame (must have 'Ticker' column)
        latest_earnings_df: Latest earnings reports DataFrame
        
    Returns:
        Merged DataFrame with earnings columns added
    """
    if scan_results_df.empty:
        return scan_results_df
    
    if latest_earnings_df.empty:
        logger.info("No earnings reports to merge")
        return scan_results_df
    
    # Merge on ticker
    merged_df = scan_results_df.merge(
        latest_earnings_df,
        left_on='Ticker',
        right_on='ticker_sgx',
        how='left',
        suffixes=('', '_earnings')
    )
    
    # Count how many matches
    matches = merged_df['revenue'].notna().sum()
    logger.info(f"Merged earnings reports: {matches} matches out of {len(scan_results_df)} stocks")
    
    return merged_df


# Cache for all reports (load once per session)
_cached_all_earnings = None
_cached_latest_earnings = None


def get_cached_earnings() -> tuple:
    """
    Get cached earnings reports (all reports and latest only)
    Loads from disk only once per session
    
    Returns:
        Tuple of (all_earnings_df, latest_earnings_df)
    """
    global _cached_all_earnings, _cached_latest_earnings
    
    if _cached_all_earnings is None:
        _cached_all_earnings = load_all_earnings_reports()
        _cached_latest_earnings = get_latest_earnings_only(_cached_all_earnings)
    
    return _cached_all_earnings, _cached_latest_earnings


def clear_earnings_cache():
    """Clear cached earnings (call after processing new reports)"""
    global _cached_all_earnings, _cached_latest_earnings
    _cached_all_earnings = None
    _cached_latest_earnings = None
    logger.info("Earnings reports cache cleared")


def get_earnings_coverage_stats(scan_results_df: pd.DataFrame,
                                 latest_earnings_df: pd.DataFrame) -> Dict:
    """
    Get statistics about earnings coverage in scan results

    Returns:
        Dictionary with coverage statistics
    """
    if scan_results_df.empty or latest_earnings_df.empty:
        return {
            'total_stocks': len(scan_results_df),
            'stocks_with_earnings': 0,
            'coverage_pct': 0.0,
            'avg_report_age': 0,
            'positive_guidance': 0,
            'negative_guidance': 0
        }

    merged = merge_earnings_with_scan_results(scan_results_df, latest_earnings_df)

    with_earnings = merged['revenue'].notna().sum()

    stats = {
        'total_stocks': len(scan_results_df),
        'stocks_with_earnings': with_earnings,
        'coverage_pct': (with_earnings / len(scan_results_df) * 100) if len(scan_results_df) > 0 else 0,
        'avg_report_age': merged['report_age_days'].mean() if 'report_age_days' in merged.columns else 0,
        'positive_guidance': (merged['guidance_tone'] == 'positive').sum() if 'guidance_tone' in merged.columns else 0,
        'negative_guidance': (merged['guidance_tone'] == 'negative').sum() if 'guidance_tone' in merged.columns else 0
    }

    return stats


def get_stock_price_on_date(ticker_sgx: str, target_date: date) -> tuple:
    """
    Get open and close prices for a specific date from Historical_Data

    Args:
        ticker_sgx: Ticker in SGX format (e.g., "N2IU.SG")
        target_date: Date to get prices for

    Returns:
        Tuple of (open_price, close_price) or (None, None) if data unavailable
    """
    try:
        from utils.paths import HISTORICAL_DATA_DIR

        # Convert ticker to filename format (remove .SG)
        ticker_base = ticker_sgx.replace('.SG', '')
        csv_file = HISTORICAL_DATA_DIR / f"{ticker_base}.csv"

        if not csv_file.exists():
            logger.warning(f"Historical data file not found: {csv_file}")
            return None, None

        # Load CSV and filter to target date
        df = pd.read_csv(csv_file, parse_dates=['Date'])

        # Ensure Date column is datetime (fallback conversion if parse_dates failed)
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Look for exact date first
        price_row = df[df['Date'].dt.date == target_date]

        if price_row.empty:
            # If exact date not found, try next available trading day
            # (useful for weekends/holidays when earnings released)
            future_dates = df[df['Date'].dt.date >= target_date].head(1)
            if not future_dates.empty:
                price_row = future_dates
                logger.info(f"Using next available date for {ticker_sgx} on {target_date}: {price_row['Date'].iloc[0].date()}")
            else:
                logger.warning(f"No price data found for {ticker_sgx} on or after {target_date}")
                return None, None

        open_price = float(price_row['Open'].iloc[0])
        close_price = float(price_row['Close'].iloc[0])

        return open_price, close_price

    except Exception as e:
        logger.error(f"Error getting price data for {ticker_sgx} on {target_date}: {e}")
        return None, None


def calculate_earnings_reaction_analysis(ticker_sgx: str) -> Optional[Dict]:
    """
    Calculate historical intraday earnings reaction for a ticker
    Focuses on open-to-close move on earnings day (intraday trader perspective)

    Args:
        ticker_sgx: Ticker in SGX format (e.g., "N2IU.SG")

    Returns:
        Dictionary with reaction statistics, or None if insufficient data (< 3 events)
    """
    try:
        # Get all earnings reports for this ticker
        all_reports, _ = get_cached_earnings()
        ticker_reports = all_reports[all_reports['ticker_sgx'] == ticker_sgx].copy()

        if ticker_reports.empty or len(ticker_reports) < 3:
            return None  # Insufficient data

        # Sort by date (oldest first for chronological analysis)
        ticker_reports = ticker_reports.sort_values('report_date')

        events = []
        positive_returns = []
        negative_returns = []

        for _, report in ticker_reports.iterrows():
            report_date = report['report_date'].date()
            report_time = report.get('report_time', 'PM')  # Default to PM if missing

            # Calculate target trading date
            if report_time == 'AM':
                target_date = report_date
            else:  # PM release
                # Add one day for PM releases
                target_date = report_date + pd.Timedelta(days=1)

            # Get prices for target date
            open_price, close_price = get_stock_price_on_date(ticker_sgx, target_date)

            if open_price is None or close_price is None:
                logger.warning(f"No price data for {ticker_sgx} on {target_date} (earnings: {report_date})")
                continue

            # Calculate intraday return
            intraday_return = ((close_price - open_price) / open_price) * 100

            # Store event details
            event = {
                'report_date': report_date.strftime('%Y-%m-%d'),
                'report_type': report.get('report_type', 'Unknown'),
                'report_time': report_time,
                'target_date': target_date.strftime('%Y-%m-%d'),
                'open': round(open_price, 3),
                'close': round(close_price, 3),
                'intraday_return': round(intraday_return, 2),
                'guidance_tone': report.get('guidance_tone', 'unknown')
            }
            events.append(event)

            # Categorize returns
            if intraday_return > 0:
                positive_returns.append(intraday_return)
            else:
                negative_returns.append(intraday_return)

        if len(events) < 3:
            return None  # Still insufficient after filtering missing data

        # Calculate statistics
        total_events = len(events)
        positive_count = len(positive_returns)
        win_rate = (positive_count / total_events) * 100

        avg_positive_return = sum(positive_returns) / len(positive_returns) if positive_returns else 0
        avg_negative_return = sum(negative_returns) / len(negative_returns) if negative_returns else 0
        overall_avg_return = (sum(positive_returns) + sum(negative_returns)) / total_events

        return {
            'total_events': total_events,
            'positive_count': positive_count,
            'win_rate': round(win_rate, 1),
            'avg_positive_return': round(avg_positive_return, 2),
            'avg_negative_return': round(avg_negative_return, 2),
            'overall_avg_return': round(overall_avg_return, 2),
            'events': events
        }

    except Exception as e:
        logger.error(f"Error calculating earnings reaction for {ticker_sgx}: {e}")
        return None


def format_earnings_reaction_display(reaction_stats: Optional[Dict]) -> str:
    """
    Format earnings reaction statistics for display in scanner results

    Args:
        reaction_stats: Dictionary from calculate_earnings_reaction_analysis()

    Returns:
        Formatted string like "70% ↑ +1.8% (7/10)" or "N/A"
    """
    if reaction_stats is None or reaction_stats['total_events'] < 3:
        return 'N/A'

    win_rate = reaction_stats['win_rate']
    total = reaction_stats['total_events']
    positive = reaction_stats['positive_count']

    if win_rate >= 50:
        avg_return = reaction_stats['avg_positive_return']
        return f"{win_rate:.0f}% ↑ +{avg_return:.1f}% ({positive}/{total})"
    else:
        avg_return = reaction_stats['avg_negative_return']
        return f"{win_rate:.0f}% ↓ {avg_return:.1f}% ({positive}/{total})"
