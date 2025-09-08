"""
Technical Analysis Module
Contains the enhanced columns function migrated from notebook Cell 6
This is the complete implementation from your original Jupyter notebook
"""

import pandas as pd
import numpy as np
from typing import Optional

def add_enhanced_columns(df_daily: pd.DataFrame, ticker: str, rolling_window: int = 20) -> pd.DataFrame:
    """
    Add all enhanced columns for a single stock with simplified forward fill logic
    
    Simplified Logic:
    - Monday: Sets CRT levels and Valid_CRT
    - Tue-Fri: Forward fill from Monday
    
    Args:
        df_daily: Raw OHLCV data from yfinance
        ticker: Stock symbol
        rolling_window: Window for moving averages (default 20)
    
    Returns:
        DataFrame with enhanced technical analysis columns
    """
    
    df = df_daily.copy()
    
    # Handle multi-level columns from yfinance if present
    if df.columns.nlevels > 1:
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # 1. Calculate daily relative range
    df['Daily_Rel_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # 2. Add percentile rankings
    df['Daily_Range_Percentile'] = df['Daily_Rel_Range'].rolling(window=50, min_periods=20).rank(pct=True)
    
    # 3. Add volume normalization
    df['Volume_Normalized'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=10).mean()
    
    # 4. Create volume-weighted range
    df['Volume_Weighted_Range'] = df['Daily_Rel_Range'] * df['Volume_Normalized']
    
    # 5. Calculate volume-weighted range percentile
    df['VW_Range_Percentile'] = df['Volume_Weighted_Range'].rolling(window=50, min_periods=20).rank(pct=True)
    
    # 5a. Calculate velocity (absolute difference in percentage points)
    df['VW_Range_Velocity'] = df['VW_Range_Percentile'] - df['VW_Range_Percentile'].shift(1)
    
    # 6. Range Expansion Signal (Simplified - no threshold)
    range_expanding = (df['VW_Range_Percentile'] > df['VW_Range_Percentile'].shift(1))
    df['Rel_Range_Signal'] = np.where(range_expanding, 1, 0)
    
    # 7. Create Is_First_Trading_Day column
    df['Is_First_Trading_Day'] = np.where(df.index.weekday == 0, 1, 0)
    
    # 8. Initialize CRT columns
    df['Weekly_Open'] = np.nan
    df['CRT_High'] = np.nan
    df['CRT_Low'] = np.nan
    df['CRT_Close'] = np.nan
    
    # 9. Set CRT values on Mondays only (using Monday's own OHLC)
    monday_mask = df['Is_First_Trading_Day'] == 1
    df.loc[monday_mask, 'Weekly_Open'] = df.loc[monday_mask, 'Open']
    df.loc[monday_mask, 'CRT_High'] = df.loc[monday_mask, 'High']
    df.loc[monday_mask, 'CRT_Low'] = df.loc[monday_mask, 'Low']
    df.loc[monday_mask, 'CRT_Close'] = df.loc[monday_mask, 'Close']
    
    # 10. Forward fill CRT values from Monday through Friday
    df['Weekly_Open'] = df['Weekly_Open'].ffill()
    df['CRT_High'] = df['CRT_High'].ffill()
    df['CRT_Low'] = df['CRT_Low'].ffill()
    df['CRT_Close'] = df['CRT_Close'].ffill()
    
    # 11. Create Valid_CRT (only Mondays with range expansion)
    df['Valid_CRT'] = np.where(
        (df['Is_First_Trading_Day'] == 1) & (df['Rel_Range_Signal'] == 1), 1,
        np.where(df['Is_First_Trading_Day'] == 1, 0, np.nan)  # NaN for non-Mondays
    )
    
    # 12. Forward fill Valid_CRT from Monday through Friday
    df['Valid_CRT'] = df['Valid_CRT'].ffill()
    
    # 13. Calculate IBS for all days
    df['IBS'] = np.where(
        df['High'] != df['Low'],
        (df['Close'] - df['Low']) / (df['High'] - df['Low']),
        1.0
    )
    
    # 14. Initialize signal columns
    df['Wick_Below'] = 0
    df['Close_Above'] = 0
    
    # 15. Calculate signals using forward-filled CRT levels
    # Group by week to process signals within each trading week
    df['week_start'] = df.index - pd.to_timedelta(df.index.weekday, unit='D')
    df['week_start'] = df['week_start'].dt.normalize()
    
    unique_weeks = df['week_start'].unique()
    
    for week_start in unique_weeks:
        # Get all days in this week (Mon-Fri)
        week_mask = df['week_start'] == week_start
        week_data = df[week_mask].copy()
        
        if len(week_data) == 0:
            continue
        
        # Get CRT levels for this week (should be same for all days due to ffill)
        crt_high = week_data['CRT_High'].iloc[0]
        crt_low = week_data['CRT_Low'].iloc[0]
        
        if pd.isna(crt_high) or pd.isna(crt_low):
            continue
        
        # WICK_BELOW Analysis: Look for low < CRT_Low, then close >= CRT_Low
        # Exclude Monday from signal calculations (Monday is the reference day)
        signal_days = week_data[week_data.index.weekday > 0]  # Only Tue-Fri
        
        condition_1_triggered = False
        wick_below_trigger_date = None
        
        for day_date, day_row in signal_days.iterrows():
            if day_row['Low'] < crt_low:
                condition_1_triggered = True
            if condition_1_triggered and day_row['Close'] >= crt_low:
                wick_below_trigger_date = day_date
                break
        
        if wick_below_trigger_date is not None:
            # Signal propagates through rest of week (excluding Monday)
            subsequent_days = signal_days[signal_days.index >= wick_below_trigger_date].index
            df.loc[subsequent_days, 'Wick_Below'] = 1
        
        # CLOSE_ABOVE Analysis: Look for close >= CRT_High
        # Exclude Monday from signal calculations (Monday is the reference day)
        close_above_trigger_date = None
        
        for day_date, day_row in signal_days.iterrows():
            if day_row['Close'] >= crt_high:
                close_above_trigger_date = day_date
                break
        
        if close_above_trigger_date is not None:
            # Signal propagates through rest of week (excluding Monday)
            subsequent_days = signal_days[signal_days.index >= close_above_trigger_date].index
            df.loc[subsequent_days, 'Close_Above'] = 1
    
    # 16. Clean up temporary column
    df.drop('week_start', axis=1, inplace=True)
    
    # 17. Calculate Buy_Signal with Valid_CRT logic
    df['Buy_Signal'] = np.where(
        (df['Valid_CRT'] == 1) &
        (df['IBS'] >= 0.5) &
        ((df['Wick_Below'] == 1) | (df['Close_Above'] == 1)),
        1, 0
    )
    
    return df


def calculate_ibs(high: float, low: float, close: float) -> float:
    """
    Calculate Internal Bar Strength (IBS)
    
    Args:
        high: High price
        low: Low price  
        close: Close price
    
    Returns:
        IBS value between 0 and 1
    """
    if high == low:
        return 1.0
    return (close - low) / (high - low)


def detect_range_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect range expansion signals
    
    Args:
        df: DataFrame with VW_Range_Percentile column
    
    Returns:
        DataFrame with expansion signals
    """
    expansion_mask = (
        (df['VW_Range_Percentile'] > df['VW_Range_Percentile'].shift(1)) & 
        (df['VW_Range_Percentile'].shift(1) <= 0.5)
    )
    
    return df[expansion_mask]


def get_latest_signals(df: pd.DataFrame) -> dict:
    """
    Get the latest signal information for a stock
    
    Args:
        df: Enhanced DataFrame with all indicators
    
    Returns:
        Dictionary with latest signal information
    """
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    return {
        'ticker': getattr(latest, 'ticker', 'Unknown'),
        'date': latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name),
        'close': float(latest['Close']),
        'ibs': float(latest['IBS']) if not pd.isna(latest['IBS']) else 0.0,
        'valid_crt': bool(latest.get('Valid_CRT', 0)),
        'wick_below': bool(latest.get('Wick_Below', 0)),
        'close_above': bool(latest.get('Close_Above', 0)),
        'buy_signal': bool(latest.get('Buy_Signal', 0)),
        'rel_range_signal': bool(latest.get('Rel_Range_Signal', 0)),
        'vw_range_percentile': float(latest.get('VW_Range_Percentile', 0)) if not pd.isna(latest.get('VW_Range_Percentile', 0)) else 0.0,
        'crt_high': float(latest.get('CRT_High', 0)) if not pd.isna(latest.get('CRT_High', 0)) else 0.0,
        'crt_low': float(latest.get('CRT_Low', 0)) if not pd.isna(latest.get('CRT_Low', 0)) else 0.0
    }


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate the quality of the enhanced data
    
    Args:
        df: Enhanced DataFrame
    
    Returns:
        Dictionary with validation results
    """
    required_columns = ['Close', 'High', 'Low', 'Volume', 'IBS', 'Buy_Signal']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation_results = {
        'is_valid': len(missing_columns) == 0,
        'missing_columns': missing_columns,
        'row_count': len(df),
        'has_recent_data': len(df) > 0,
        'buy_signals_count': int(df['Buy_Signal'].sum()) if 'Buy_Signal' in df.columns else 0,
        'expansion_signals_count': int(df['Rel_Range_Signal'].sum()) if 'Rel_Range_Signal' in df.columns else 0
    }
    
    return validation_results


def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of all signals in the DataFrame
    
    Args:
        df: Enhanced DataFrame with signals
    
    Returns:
        Dictionary with signal summary statistics
    """
    if df.empty:
        return {
            'total_days': 0,
            'buy_signals': 0,
            'expansion_signals': 0,
            'high_ibs_days': 0,
            'valid_crt_days': 0
        }
    
    return {
        'total_days': len(df),
        'buy_signals': int(df['Buy_Signal'].sum()) if 'Buy_Signal' in df.columns else 0,
        'expansion_signals': int(df['Rel_Range_Signal'].sum()) if 'Rel_Range_Signal' in df.columns else 0,
        'high_ibs_days': int((df['IBS'] >= 0.5).sum()) if 'IBS' in df.columns else 0,
        'valid_crt_days': int(df['Valid_CRT'].sum()) if 'Valid_CRT' in df.columns else 0,
        'wick_below_signals': int(df['Wick_Below'].sum()) if 'Wick_Below' in df.columns else 0,
        'close_above_signals': int(df['Close_Above'].sum()) if 'Close_Above' in df.columns else 0
    }


# Backward compatibility functions for notebook migration
def calculate_technical_indicators(df: pd.DataFrame, ticker: str = 'Unknown') -> pd.DataFrame:
    """
    Backward compatibility wrapper for add_enhanced_columns
    """
    return add_enhanced_columns(df, ticker)


def get_buy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get only the rows where buy signals are active
    """
    if 'Buy_Signal' not in df.columns:
        return pd.DataFrame()
    
    return df[df['Buy_Signal'] == 1].copy()