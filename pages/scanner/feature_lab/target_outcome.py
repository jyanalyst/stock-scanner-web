"""
Target Outcome Module - Calculates objective trading outcomes for historical scans.

This module determines if a trade signal would have been a winner, loser, or timeout
based on subsequent price action relative to the signal day's High/Low range.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_future_prices(
    ticker: str,
    start_date: date,
    days: int = 4,
    data_dir: str = "data/Historical_Data"
) -> List[float]:
    """
    Get closing prices for a ticker starting from start_date.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'D05')
        start_date: Starting date (Day 0 - the signal day)
        days: Number of days to retrieve (default 4 for day0-day3)
        data_dir: Directory containing stock CSV files
    
    Returns:
        List of closing prices [day0_close, day1_close, day2_close, day3_close]
        Returns empty list if insufficient data or file not found
    
    CSV Format Expected:
        - File path: {data_dir}/{ticker}.csv
        - Columns: Date, Open, High, Low, Close, Volume (at minimum)
        - Date format: YYYY-MM-DD or DD/MM/YYYY (handle both)
    """
    try:
        # Construct file path
        # Handle .SG suffix if present in ticker but not in filename, or vice versa
        # Based on file listing, files seem to be like 'D05.csv' or 'Z74.csv'
        clean_ticker = ticker.replace('.SG', '')
        file_path = Path(data_dir) / f"{clean_ticker}.csv"
        
        if not file_path.exists():
            # Try with .SG if not found
            file_path = Path(data_dir) / f"{clean_ticker}.SG.csv"
            if not file_path.exists():
                # Try original ticker just in case
                file_path = Path(data_dir) / f"{ticker}.csv"
                if not file_path.exists():
                    # logger.warning(f"CSV file not found for {ticker}: {file_path}")
                    return []
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Check required columns
        if 'Date' not in df.columns or 'Close' not in df.columns:
            logger.error(f"Missing required columns in {ticker}.csv")
            return []
        
        # Parse dates (handle multiple formats)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Filter to dates >= start_date
        # Ensure start_date is Timestamp for comparison
        start_ts = pd.Timestamp(start_date)
        df = df[df['Date'] >= start_ts]
        
        # Check if we have enough data
        # We need at least 1 day (Day 0) to return anything useful, 
        # but ideally 'days' amount. We return what we have.
        if len(df) == 0:
            return []
            
        # Extract Close prices for next {days} trading days
        prices = df['Close'].head(days).tolist()
        
        # Convert to float (handle any string values)
        prices = [float(p) for p in prices]
        
        return prices
        
    except Exception as e:
        logger.error(f"Error loading future prices for {ticker}: {e}")
        return []

def get_target_outcome(
    signal_high: float,
    signal_low: float,
    price_series: List[float],
    max_days: int = 3
) -> Tuple[str, int]:
    """
    Core function that determines outcome based on Day 0 range.
    
    Args:
        signal_high: Day 0 High (breakout target)
        signal_low: Day 0 Low (breakdown invalidation)
        price_series: List of closes [day0, day1, day2, day3]
                      Index 0 = signal day, Index 1+ = days after signal
        max_days: Days to monitor after signal (default: 3)
    
    Returns:
        tuple: (outcome, day) where outcome is:
            'TRUE_BREAK': Price closed above signal_high
            'INVALIDATION': Price closed below signal_low
            'TIMEOUT': Price stayed within range for max_days
            'INSUFFICIENT_DATA': Not enough price data
            'INVALID_RANGE': signal_high <= signal_low (data error)
            'AMBIGUOUS': Edge case (shouldn't happen)
    """
    # Validation: Check range validity
    if signal_high <= signal_low:
        return 'INVALID_RANGE', 0
    
    # Validation: Check sufficient data
    # We need at least day 0 + 1 day to check outcome
    # If we have less than max_days + 1, we can still check available days,
    # but if we run out of data before a result, it's INSUFFICIENT_DATA
    if len(price_series) < 2:
        return 'INSUFFICIENT_DATA', 0
    
    # Check each day after signal (days 1, 2, 3...)
    # Limit by available data and max_days
    check_days = min(len(price_series) - 1, max_days)
    
    for day in range(1, check_days + 1):
        close = price_series[day]
        
        # Breakout above Day 0 High → Winner
        if close > signal_high:
            return 'TRUE_BREAK', day
        
        # Breakdown below Day 0 Low → Loser
        elif close < signal_low:
            return 'INVALIDATION', day
    
    # If we checked all required days and found no break
    if check_days == max_days:
        final_close = price_series[max_days]
        # Still within range → No clear direction
        if signal_low <= final_close <= signal_high:
            return 'TIMEOUT', max_days
            
    # If we ran out of data before max_days and didn't find a break
    if len(price_series) < max_days + 1:
        return 'INSUFFICIENT_DATA', len(price_series) - 1
    
    # Edge case: Outside range but didn't trigger earlier (should be caught by loop)
    return 'AMBIGUOUS', max_days

def get_target_outcome_with_metrics(
    signal_high: float,
    signal_low: float,
    price_series: List[float],
    max_days: int = 3
) -> Dict[str, Any]:
    """
    Enhanced version with return calculations and metrics.
    
    Returns:
        dict: {
            'outcome': str,           # 'TRUE_BREAK', 'INVALIDATION', 'TIMEOUT', etc.
            'day': int,               # Day when outcome determined (1-3)
            'return_pct': float,      # Percentage return at outcome
            'signal_high': float,     # Copy of signal_high for reference
            'signal_low': float,      # Copy of signal_low for reference
            'max_return': float,      # Highest return during period
            'min_return': float,      # Lowest return during period
            'volatility': float,      # Std dev of returns
            'days_in_range': int      # Days price stayed within range
        }
    """
    outcome, day = get_target_outcome(signal_high, signal_low, price_series, max_days)
    
    result = {
        'outcome': outcome,
        'day': day,
        'signal_high': signal_high,
        'signal_low': signal_low,
        'return_pct': 0.0,
        'max_return': 0.0,
        'min_return': 0.0,
        'volatility': 0.0,
        'days_in_range': 0
    }
    
    if not price_series or len(price_series) < 2:
        return result
        
    # Calculate returns relative to signal_high (assumed entry for breakout)
    # Note: For a breakout strategy, we assume entry at signal_high (or close to it)
    # But for calculation consistency, let's use signal_high as the reference base
    base_price = signal_high
    
    # Calculate return at outcome day
    if day > 0 and day < len(price_series):
        outcome_close = price_series[day]
        result['return_pct'] = ((outcome_close - base_price) / base_price) * 100
    
    # Calculate metrics over the monitored period
    available_days = min(len(price_series), max_days + 1)
    period_prices = price_series[1:available_days] # Exclude day 0
    
    if period_prices:
        returns = [((p - base_price) / base_price) * 100 for p in period_prices]
        result['max_return'] = max(returns)
        result['min_return'] = min(returns)
        if len(returns) > 1:
            result['volatility'] = float(np.std(returns))
            
    # Days in range calculation
    days_in_range = 0
    for i in range(1, available_days):
        if signal_low <= price_series[i] <= signal_high:
            days_in_range += 1
        else:
            break
    result['days_in_range'] = days_in_range
    
    return result

def calculate_outcomes_for_scan(
    scan_date: date,
    scan_results: pd.DataFrame,
    price_data_loader: Callable = None
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate outcomes for all stocks in a historical scan.
    
    Args:
        scan_date: The date the scanner ran (Day 0)
        scan_results: DataFrame with columns including 'Ticker', 'High', 'Low'
        price_data_loader: Function that takes (ticker, start_date, days) 
                          and returns List[float] of closing prices.
                          Defaults to get_future_prices if None.
    
    Returns:
        dict: {ticker: outcome_data} for all stocks in scan_results
    """
    if price_data_loader is None:
        price_data_loader = get_future_prices
        
    outcomes = {}
    
    if scan_results is None or scan_results.empty:
        return outcomes
        
    # Ensure we have required columns
    required_cols = ['Ticker', 'High', 'Low']
    if not all(col in scan_results.columns for col in required_cols):
        logger.error(f"Scan results missing required columns: {required_cols}")
        return outcomes
        
    for _, row in scan_results.iterrows():
        ticker = row['Ticker']
        signal_high = float(row['High'])
        signal_low = float(row['Low'])
        
        # Get future prices
        # We need Day 0 + 3 days = 4 days total
        price_series = price_data_loader(ticker, scan_date, days=4)
        
        # Calculate outcome
        outcome_data = get_target_outcome_with_metrics(
            signal_high, 
            signal_low, 
            price_series, 
            max_days=3
        )
        
        outcomes[ticker] = outcome_data
        
    return outcomes
