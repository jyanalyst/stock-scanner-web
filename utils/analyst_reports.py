# File: utils/analyst_reports.py
"""
Analyst Report Utilities
Load and manage analyst report data from JSON files
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def get_analyst_reports_dir() -> Path:
    """Get the analyst reports directory"""
    from utils.paths import ANALYST_REPORTS_DIR
    return ANALYST_REPORTS_DIR


def load_all_analyst_reports() -> pd.DataFrame:
    """
    Load all analyst report JSONs into a DataFrame
    
    Returns:
        DataFrame with columns:
        - ticker: Display ticker (e.g., "N2IU")
        - ticker_sgx: SGX format (e.g., "N2IU.SG")
        - report_date: Date of report
        - sentiment_score: -1 to +1
        - sentiment_label: "positive", "neutral", "negative"
        - recommendation: "ADD", "HOLD", "REDUCE", etc.
        - price_target: Target price
        - price_at_report: Price when report written
        - upside_pct: Percentage upside to target
        - analyst_firm: Firm name
        - key_catalysts: List of catalysts
        - key_risks: List of risks
        - executive_summary: Summary text
        - pdf_filename: Original PDF name
        - report_age_days: Days since report date
    """
    reports_dir = get_analyst_reports_dir()
    
    if not reports_dir.exists():
        logger.warning(f"Analyst reports directory not found: {reports_dir}")
        return pd.DataFrame()
    
    # Find all JSON files
    json_files = list(reports_dir.glob("*.json"))
    
    if not json_files:
        logger.info("No analyst report JSON files found")
        return pd.DataFrame()
    
    # Load all JSONs
    reports = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
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
    
    logger.info(f"Loaded {len(df)} analyst reports from {len(json_files)} files")
    
    return df


def get_latest_reports_only(all_reports_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get only the latest report for each ticker
    
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
    
    logger.info(f"Latest reports: {len(latest_df)} unique tickers")
    
    return latest_df


def get_report_history(all_reports_df: pd.DataFrame, ticker_sgx: str) -> pd.DataFrame:
    """
    Get all reports for a specific ticker, sorted by date (newest first)
    
    Args:
        all_reports_df: DataFrame with all reports
        ticker_sgx: Ticker in SGX format (e.g., "N2IU.SG")
        
    Returns:
        DataFrame with all reports for this ticker
    """
    if all_reports_df.empty:
        return pd.DataFrame()
    
    ticker_reports = all_reports_df[all_reports_df['ticker_sgx'] == ticker_sgx].copy()
    ticker_reports = ticker_reports.sort_values('report_date', ascending=False)
    
    return ticker_reports


def format_sentiment_emoji(sentiment_label: str) -> str:
    """Get emoji for sentiment label"""
    emoji_map = {
        'positive': '📈',
        'neutral': '➖',
        'negative': '📉'
    }
    return emoji_map.get(sentiment_label, '❓')


def format_report_age(age_days: int) -> str:
    """
    Format report age in human-readable form
    
    Returns:
        String like "7 days old" or "3 months old"
        Adds warning emoji if >90 days
    """
    if age_days is None:
        return "Unknown age"
    
    if age_days == 0:
        return "Today"
    elif age_days == 1:
        return "1 day old"
    elif age_days < 30:
        return f"{age_days} days old"
    elif age_days < 90:
        months = age_days // 30
        return f"{months} month{'s' if months > 1 else ''} old"
    else:
        months = age_days // 30
        return f"⚠️ {months} months old"


def get_sentiment_trend_description(history_df: pd.DataFrame) -> str:
    """
    Analyze sentiment trend across multiple reports
    
    Args:
        history_df: DataFrame with report history (sorted newest first)
        
    Returns:
        Description like "Improving" or "Deteriorating"
    """
    if len(history_df) < 2:
        return "Single report"
    
    scores = history_df['sentiment_score'].tolist()
    
    # Compare latest vs oldest
    latest = scores[0]
    oldest = scores[-1]
    
    if latest > oldest + 0.2:
        return "📈 Improving"
    elif latest < oldest - 0.2:
        return "📉 Deteriorating"
    else:
        return "➖ Stable"


def merge_reports_with_scan_results(scan_results_df: pd.DataFrame, 
                                     latest_reports_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge analyst reports with scan results
    
    Args:
        scan_results_df: Scanner results DataFrame (must have 'Ticker' column)
        latest_reports_df: Latest analyst reports DataFrame
        
    Returns:
        Merged DataFrame with analyst report columns added
    """
    if scan_results_df.empty:
        return scan_results_df
    
    if latest_reports_df.empty:
        logger.info("No analyst reports to merge")
        return scan_results_df
    
    # Merge on ticker
    merged_df = scan_results_df.merge(
        latest_reports_df,
        left_on='Ticker',
        right_on='ticker_sgx',
        how='left',
        suffixes=('', '_report')
    )
    
    # Count how many matches
    matches = merged_df['sentiment_score'].notna().sum()
    logger.info(f"Merged analyst reports: {matches} matches out of {len(scan_results_df)} stocks")
    
    return merged_df


# Cache for all reports (load once per session)
_cached_all_reports = None
_cached_latest_reports = None


def get_cached_reports() -> tuple:
    """
    Get cached reports (all reports and latest only)
    Loads from disk only once per session
    
    Returns:
        Tuple of (all_reports_df, latest_reports_df)
    """
    global _cached_all_reports, _cached_latest_reports
    
    if _cached_all_reports is None:
        _cached_all_reports = load_all_analyst_reports()
        _cached_latest_reports = get_latest_reports_only(_cached_all_reports)
    
    return _cached_all_reports, _cached_latest_reports


def clear_reports_cache():
    """Clear cached reports (call after processing new PDFs)"""
    global _cached_all_reports, _cached_latest_reports
    _cached_all_reports = None
    _cached_latest_reports = None
    logger.info("Analyst reports cache cleared")