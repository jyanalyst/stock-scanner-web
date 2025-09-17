"""
Technical Analysis Module
Contains the enhanced columns function migrated from notebook Cell 6
This is the complete implementation from your original Jupyter notebook
Updated with Higher_HL logic - daily check for higher high AND higher low
PHASE 1: Added momentum persistence calculations with independent signal assessment
"""

import pandas as pd
import numpy as np
from typing import Optional

def calculate_momentum_persistence(returns: pd.Series, periods: list = [1, 3, 5]) -> dict:
    """
    Calculate momentum persistence probabilities for multiple periods
    Pure statistical calculation - no CRT signal contamination
    
    Args:
        returns: Daily returns series
        periods: List of forward-looking periods to calculate
    
    Returns:
        Dictionary with momentum probabilities for each period
    """
    momentum_results = {}
    
    # Calculate up/down days
    up_days = returns > 0
    
    for period in periods:
        if len(returns) < period + 10:  # Need minimum data
            momentum_results[f'momentum_{period}day'] = 0.0
            continue
        
        # Calculate future returns for this period
        future_positive = returns.rolling(window=period).sum().shift(-period) > 0
        
        # Calculate conditional probability: P(future positive | today positive)
        up_days_mask = up_days & future_positive.notna()
        if up_days_mask.sum() > 0:
            up_followed_by_positive = (up_days & future_positive).sum()
            total_up_days = up_days_mask.sum()
            momentum_prob = up_followed_by_positive / total_up_days if total_up_days > 0 else 0.0
        else:
            momentum_prob = 0.0
        
        momentum_results[f'momentum_{period}day'] = momentum_prob
    
    return momentum_results


def calculate_simple_momentum_persistence(returns: pd.Series) -> dict:
    """
    Calculate simple next-day momentum persistence
    P(Up tomorrow | Up today)
    
    Args:
        returns: Daily returns series
    
    Returns:
        Dictionary with momentum statistics
    """
    if len(returns) < 20:  # Need minimum data
        return {
            'momentum_1day': 0.0,
            'momentum_3day': 0.0,
            'autocorr_1day': 0.0
        }
    
    # Calculate up days
    up_days = returns > 0
    
    # 1-day momentum persistence
    up_tomorrow = up_days.shift(-1)
    up_today_and_tomorrow = up_days & up_tomorrow
    momentum_1day = up_today_and_tomorrow.sum() / up_days.sum() if up_days.sum() > 0 else 0.0
    
    # 3-day momentum persistence (positive return in next 3 days)
    future_3day_returns = returns.rolling(window=3).sum().shift(-3)
    future_3day_positive = future_3day_returns > 0
    up_today_and_3day_positive = up_days & future_3day_positive
    momentum_3day = up_today_and_3day_positive.sum() / up_days.sum() if up_days.sum() > 0 else 0.0
    
    # Autocorrelation (1-day lag)
    try:
        autocorr_1day = returns.autocorr(lag=1)
        if pd.isna(autocorr_1day):
            autocorr_1day = 0.0
    except:
        autocorr_1day = 0.0
    
    return {
        'momentum_1day': momentum_1day,
        'momentum_3day': momentum_3day,
        'autocorr_1day': autocorr_1day
    }


def add_enhanced_columns(df_daily: pd.DataFrame, ticker: str, rolling_window: int = 20) -> pd.DataFrame:
    """
    Add all enhanced columns for a single stock with simplified forward fill logic
    UPDATED: Now includes momentum persistence calculations (independent of CRT signals)
    
    Simplified Logic:
    - Monday: Sets CRT levels and Valid_CRT
    - Tue-Fri: Forward fill from Monday
    - Momentum: Pure statistical calculation based on price returns only
    
    Args:
        df_daily: Raw OHLCV data from yfinance
        ticker: Stock symbol
        rolling_window: Window for moving averages (default 20)
    
    Returns:
        DataFrame with enhanced technical analysis columns including momentum persistence
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
    
    # 11. Calculate IBS for all days
    df['IBS'] = np.where(
        df['High'] != df['Low'],
        (df['Close'] - df['Low']) / (df['High'] - df['Low']),
        1.0
    )
    
    # 12. Create Valid_CRT (Mondays with range expansion - no threshold)
    df['Valid_CRT'] = np.where(
        (df['Is_First_Trading_Day'] == 1) & (df['Rel_Range_Signal'] == 1), 1,
        np.where(df['Is_First_Trading_Day'] == 1, 0, np.nan)  # NaN for non-Mondays
    )
    
    # 13. Capture the qualifying velocity on Mondays for Valid_CRT
    df['CRT_Qualifying_Velocity'] = np.where(
        (df['Is_First_Trading_Day'] == 1) & (df['Rel_Range_Signal'] == 1),
        df['VW_Range_Velocity'],
        np.nan
    )
    
    # 14. NEW: Calculate Higher_HL pattern for ALL days (higher high AND higher low)
    df['Higher_HL'] = np.where(
        (df['High'] > df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1)),
        1, 
        0
    )

    # 15. Forward fill Valid_CRT and CRT_Qualifying_Velocity from Monday through Friday
    df['Valid_CRT'] = df['Valid_CRT'].ffill()
    df['CRT_Qualifying_Velocity'] = df['CRT_Qualifying_Velocity'].ffill()
    
    # 16. NEW: MOMENTUM PERSISTENCE CALCULATIONS (Independent of CRT signals)
    # Calculate daily returns for momentum analysis
    df['Daily_Returns'] = df['Close'].pct_change()
    
    # Calculate momentum persistence probabilities
    try:
        momentum_stats = calculate_simple_momentum_persistence(df['Daily_Returns'])
        
        # Add momentum columns (constant values for all rows - represents stock's historical momentum characteristics)
        df['Momentum_1Day_Prob'] = momentum_stats['momentum_1day']
        df['Momentum_3Day_Prob'] = momentum_stats['momentum_3day'] 
        df['Autocorr_1Day'] = momentum_stats['autocorr_1day']
        
        # Debug print for momentum calculations
        print(f"DEBUG {ticker}: Momentum 1-day: {momentum_stats['momentum_1day']:.4f}")
        print(f"DEBUG {ticker}: Momentum 3-day: {momentum_stats['momentum_3day']:.4f}")
        print(f"DEBUG {ticker}: Autocorr 1-day: {momentum_stats['autocorr_1day']:.4f}")
        
    except Exception as e:
        print(f"WARNING {ticker}: Momentum calculation failed: {e}")
        # Fallback values if momentum calculation fails
        df['Momentum_1Day_Prob'] = 0.5  # 50% = no momentum edge
        df['Momentum_3Day_Prob'] = 0.5
        df['Autocorr_1Day'] = 0.0
    
    # 17. Initialize signal columns
    df['Wick_Below'] = 0
    df['Close_Above'] = 0
    
    # 18. Calculate signals using forward-filled CRT levels
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
    
    # Clean up temporary columns
    df.drop(['week_start', 'Daily_Returns'], axis=1, inplace=True)
    
    # 19. Calculate Buy_Signal with Valid_CRT logic
    df['Buy_Signal'] = np.where(
        (df['Valid_CRT'] == 1) &
        (df['IBS'] >= 0.5) &
        ((df['Wick_Below'] == 1) | (df['Close_Above'] == 1)),
        1, 0
    )
    
    # Debug print to verify column creation
    print(f"DEBUG {ticker}: Created columns: {list(df.columns)}")
    print(f"DEBUG {ticker}: CRT_Qualifying_Velocity sample: {df['CRT_Qualifying_Velocity'].tail().values}")
    print(f"DEBUG {ticker}: Higher_HL sample: {df['Higher_HL'].tail().values}")
    
    # Debug Higher_HL pattern checks
    higher_hl_days = df[df['Higher_HL'] == 1]
    print(f"DEBUG {ticker}: Total Higher_HL patterns: {len(higher_hl_days)} out of {len(df)} days")
    if len(higher_hl_days) > 0:
        print(f"DEBUG {ticker}: Recent Higher_HL dates: {higher_hl_days.index[-5:].tolist()}")
    
    # Debug momentum statistics
    momentum_1day_unique = df['Momentum_1Day_Prob'].nunique()
    momentum_3day_unique = df['Momentum_3Day_Prob'].nunique()
    print(f"DEBUG {ticker}: Momentum columns created successfully")
    print(f"DEBUG {ticker}: Momentum_1Day_Prob unique values: {momentum_1day_unique} (should be 1)")
    print(f"DEBUG {ticker}: Momentum_3Day_Prob unique values: {momentum_3day_unique} (should be 1)")
    
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
    UPDATED: Now includes momentum persistence signals
    
    Args:
        df: Enhanced DataFrame with all indicators
    
    Returns:
        Dictionary with latest signal information including momentum
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
        'crt_low': float(latest.get('CRT_Low', 0)) if not pd.isna(latest.get('CRT_Low', 0)) else 0.0,
        'crt_qualifying_velocity': float(latest.get('CRT_Qualifying_Velocity', 0)) if not pd.isna(latest.get('CRT_Qualifying_Velocity', 0)) else 0.0,
        'higher_hl': bool(latest.get('Higher_HL', 0)),
        # NEW: Momentum signals
        'momentum_1day_prob': float(latest.get('Momentum_1Day_Prob', 0.5)) if not pd.isna(latest.get('Momentum_1Day_Prob', 0.5)) else 0.5,
        'momentum_3day_prob': float(latest.get('Momentum_3Day_Prob', 0.5)) if not pd.isna(latest.get('Momentum_3Day_Prob', 0.5)) else 0.5,
        'autocorr_1day': float(latest.get('Autocorr_1Day', 0.0)) if not pd.isna(latest.get('Autocorr_1Day', 0.0)) else 0.0
    }


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate the quality of the enhanced data
    UPDATED: Now includes momentum column validation
    
    Args:
        df: Enhanced DataFrame
    
    Returns:
        Dictionary with validation results
    """
    required_columns = ['Close', 'High', 'Low', 'Volume', 'IBS', 'Buy_Signal', 'Momentum_1Day_Prob', 'Momentum_3Day_Prob', 'Autocorr_1Day']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation_results = {
        'is_valid': len(missing_columns) == 0,
        'missing_columns': missing_columns,
        'row_count': len(df),
        'has_recent_data': len(df) > 0,
        'buy_signals_count': int(df['Buy_Signal'].sum()) if 'Buy_Signal' in df.columns else 0,
        'expansion_signals_count': int(df['Rel_Range_Signal'].sum()) if 'Rel_Range_Signal' in df.columns else 0,
        'momentum_data_available': all(col in df.columns for col in ['Momentum_1Day_Prob', 'Momentum_3Day_Prob', 'Autocorr_1Day'])
    }
    
    return validation_results


def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of all signals in the DataFrame
    UPDATED: Now includes momentum signal statistics
    
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
            'valid_crt_days': 0,
            'momentum_1day_avg': 0.5,
            'momentum_3day_avg': 0.5,
            'autocorr_avg': 0.0
        }
    
    return {
        'total_days': len(df),
        'buy_signals': int(df['Buy_Signal'].sum()) if 'Buy_Signal' in df.columns else 0,
        'expansion_signals': int(df['Rel_Range_Signal'].sum()) if 'Rel_Range_Signal' in df.columns else 0,
        'high_ibs_days': int((df['IBS'] >= 0.5).sum()) if 'IBS' in df.columns else 0,
        'valid_crt_days': int(df['Valid_CRT'].sum()) if 'Valid_CRT' in df.columns else 0,
        'wick_below_signals': int(df['Wick_Below'].sum()) if 'Wick_Below' in df.columns else 0,
        'close_above_signals': int(df['Close_Above'].sum()) if 'Close_Above' in df.columns else 0,
        # NEW: Momentum statistics
        'momentum_1day_avg': float(df['Momentum_1Day_Prob'].iloc[0]) if 'Momentum_1Day_Prob' in df.columns and len(df) > 0 else 0.5,
        'momentum_3day_avg': float(df['Momentum_3Day_Prob'].iloc[0]) if 'Momentum_3Day_Prob' in df.columns and len(df) > 0 else 0.5,
        'autocorr_avg': float(df['Autocorr_1Day'].iloc[0]) if 'Autocorr_1Day' in df.columns and len(df) > 0 else 0.0
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


# NEW: Momentum-specific utility functions
def get_momentum_summary(df: pd.DataFrame) -> dict:
    """
    Get momentum-specific summary statistics
    
    Args:
        df: Enhanced DataFrame with momentum columns
    
    Returns:
        Dictionary with momentum summary
    """
    if df.empty or 'Momentum_1Day_Prob' not in df.columns:
        return {
            'momentum_available': False,
            'momentum_1day': 0.5,
            'momentum_3day': 0.5,
            'autocorr': 0.0,
            'momentum_strength': 'Unknown'
        }
    
    momentum_1day = df['Momentum_1Day_Prob'].iloc[0] if len(df) > 0 else 0.5
    momentum_3day = df['Momentum_3Day_Prob'].iloc[0] if len(df) > 0 else 0.5
    autocorr = df['Autocorr_1Day'].iloc[0] if len(df) > 0 else 0.0
    
    # Classify momentum strength
    if momentum_1day > 0.65:
        momentum_strength = 'Strong Positive'
    elif momentum_1day > 0.55:
        momentum_strength = 'Moderate Positive'
    elif momentum_1day < 0.35:
        momentum_strength = 'Strong Negative (Mean Reversion)'
    elif momentum_1day < 0.45:
        momentum_strength = 'Moderate Negative'
    else:
        momentum_strength = 'Neutral'
    
    return {
        'momentum_available': True,
        'momentum_1day': momentum_1day,
        'momentum_3day': momentum_3day,
        'autocorr': autocorr,
        'momentum_strength': momentum_strength
    }


def validate_momentum_calculations(df: pd.DataFrame, ticker: str) -> bool:
    """
    Validate that momentum calculations were successful
    
    Args:
        df: Enhanced DataFrame
        ticker: Stock symbol for logging
    
    Returns:
        Boolean indicating if momentum data is valid
    """
    required_momentum_cols = ['Momentum_1Day_Prob', 'Momentum_3Day_Prob', 'Autocorr_1Day']
    
    # Check if columns exist
    if not all(col in df.columns for col in required_momentum_cols):
        print(f"ERROR {ticker}: Missing momentum columns")
        return False
    
    # Check if values are reasonable (probabilities between 0 and 1)
    momentum_1day = df['Momentum_1Day_Prob'].iloc[0] if len(df) > 0 else None
    momentum_3day = df['Momentum_3Day_Prob'].iloc[0] if len(df) > 0 else None
    
    if momentum_1day is None or momentum_3day is None:
        print(f"ERROR {ticker}: No momentum data available")
        return False
    
    if not (0 <= momentum_1day <= 1) or not (0 <= momentum_3day <= 1):
        print(f"ERROR {ticker}: Invalid momentum probabilities - 1day: {momentum_1day}, 3day: {momentum_3day}")
        return False
    
    print(f"SUCCESS {ticker}: Momentum validation passed")
    return True