"""
Technical Analysis Module
Contains the enhanced columns function migrated from notebook Cell 6
This is the complete implementation from your original Jupyter notebook
Updated with Higher_HL logic - daily check for higher high AND higher low
ENHANCED: Dual timeframe momentum (5-day and 20-day) with crossover detection
Optimized for short-term trading (1-3 day holding periods)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

def calculate_dual_rolling_momentum(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate dual timeframe rolling momentum persistence and autocorrelation
    Uses 5-day (short-term) and 20-day (medium-term) windows
    
    Args:
        returns: Daily returns series
    
    Returns:
        DataFrame with rolling momentum probabilities and autocorrelation for both timeframes
    """
    # Initialize result dataframe
    result_df = pd.DataFrame(index=returns.index)
    
    # Initialize columns with default values
    result_df['momentum_5day'] = np.nan
    result_df['momentum_20day'] = np.nan
    result_df['autocorr_5day'] = np.nan
    result_df['autocorr_20day'] = np.nan
    result_df['momentum_crossover'] = 0  # 1 = bullish cross, -1 = bearish cross
    result_df['momentum_spread'] = np.nan
    result_df['autocorr_alignment'] = 'Mixed'  # Both Positive, Both Negative, Mixed
    
    # Calculate for both windows
    windows = {'5day': 5, '20day': 20}
    
    for window_name, window_size in windows.items():
        # Calculate rolling momentum for each date
        for i in range(window_size, len(returns)):
            # Get the window of returns
            window_returns = returns.iloc[i-window_size:i]
            
            # Calculate up days in the window
            up_days = window_returns > 0
            
            # Skip if no up days in window
            if up_days.sum() == 0:
                result_df.loc[returns.index[i], f'momentum_{window_name}'] = 0.5
                result_df.loc[returns.index[i], f'autocorr_{window_name}'] = 0.0
                continue
            
            # 1-day momentum persistence for this window
            up_tomorrow = up_days.shift(-1)
            up_today_and_tomorrow = up_days & up_tomorrow
            momentum_1day = up_today_and_tomorrow.sum() / up_days.sum()
            
            # Autocorrelation for this window
            try:
                autocorr = window_returns.autocorr(lag=1)
                if pd.isna(autocorr):
                    autocorr = 0.0
            except:
                autocorr = 0.0
            
            # Store results for this date
            result_df.loc[returns.index[i], f'momentum_{window_name}'] = momentum_1day
            result_df.loc[returns.index[i], f'autocorr_{window_name}'] = autocorr
    
    # Calculate momentum spread (5-day minus 20-day)
    result_df['momentum_spread'] = result_df['momentum_5day'] - result_df['momentum_20day']
    
    # Detect crossovers (need at least 2 days of data)
    for i in range(1, len(result_df)):
        if pd.notna(result_df['momentum_5day'].iloc[i]) and pd.notna(result_df['momentum_20day'].iloc[i]):
            # Current and previous spreads
            curr_spread = result_df['momentum_spread'].iloc[i]
            prev_spread = result_df['momentum_spread'].iloc[i-1]
            
            # Detect crossovers
            if pd.notna(prev_spread) and pd.notna(curr_spread):
                if prev_spread <= 0 and curr_spread > 0:
                    result_df.loc[result_df.index[i], 'momentum_crossover'] = 1  # Bullish
                elif prev_spread >= 0 and curr_spread < 0:
                    result_df.loc[result_df.index[i], 'momentum_crossover'] = -1  # Bearish
    
    # Classify autocorrelation alignment
    for i in range(len(result_df)):
        auto_5 = result_df['autocorr_5day'].iloc[i]
        auto_20 = result_df['autocorr_20day'].iloc[i]
        
        if pd.notna(auto_5) and pd.notna(auto_20):
            if auto_5 > 0.10 and auto_20 > 0:
                result_df.loc[result_df.index[i], 'autocorr_alignment'] = 'Both Positive'
            elif auto_5 < -0.10 and auto_20 < 0:
                result_df.loc[result_df.index[i], 'autocorr_alignment'] = 'Both Negative'
            elif abs(auto_5 - auto_20) > 0.40:
                result_df.loc[result_df.index[i], 'autocorr_alignment'] = 'Divergent'
            else:
                result_df.loc[result_df.index[i], 'autocorr_alignment'] = 'Mixed'
    
    # Fill initial NaN values with neutral values
    result_df['momentum_5day'] = result_df['momentum_5day'].fillna(0.5)
    result_df['momentum_20day'] = result_df['momentum_20day'].fillna(0.5)
    result_df['autocorr_5day'] = result_df['autocorr_5day'].fillna(0.0)
    result_df['autocorr_20day'] = result_df['autocorr_20day'].fillna(0.0)
    result_df['momentum_spread'] = result_df['momentum_spread'].fillna(0.0)
    
    return result_df


def classify_advanced_trading_strategy(mom_5d: float, mom_20d: float, auto_5d: float, auto_20d: float, 
                                     crossover: int, spread: float) -> Tuple[str, str]:
    """
    Advanced strategy classification using dual timeframe analysis
    Optimized for 1-3 day holding periods
    
    Returns:
        Tuple of (strategy_name, emoji_indicator)
    """
    
    # Strong momentum continuation - ideal for trend following
    if mom_5d > 0.60 and mom_20d > 0.55:
        if auto_5d > 0.15 and auto_20d > 0:
            if crossover == 1:
                return ("Fresh Momentum Surge", "🚀")
            return ("Strong Momentum Continuation", "📈")
        elif auto_5d < -0.15:
            return ("Momentum with Intraday Reversals", "🔄")
    
    # Mean reversion opportunity
    elif mom_5d < 0.40 and mom_20d > 0.55:
        if auto_5d < -0.20:
            return ("Oversold Bounce Setup", "🎯")
        return ("Short-term Weakness", "📉")
    
    # Momentum acceleration - catching the move
    elif spread > 0.10 and mom_5d > 0.55:
        if crossover == 1:
            return ("Bullish Momentum Cross", "✨")
        return ("Momentum Acceleration", "⚡")
    
    # Momentum exhaustion - warning sign
    elif spread < -0.15 and mom_20d > 0.60:
        if crossover == -1:
            return ("Bearish Momentum Cross", "⚠️")
        return ("Momentum Exhaustion", "🔻")
    
    # Pure mean reversion
    elif mom_5d < 0.45 and mom_20d < 0.45:
        if auto_5d < -0.15 and auto_20d < -0.10:
            return ("Strong Mean Reversion", "🔁")
    
    # Regime change detected
    elif abs(auto_5d - auto_20d) > 0.40:
        return ("Regime Change - Caution", "⚠️")
    
    # Stable but weak
    elif 0.45 <= mom_5d <= 0.55 and 0.45 <= mom_20d <= 0.55:
        return ("Neutral - No Edge", "➖")
    
    # Default case
    return ("Mixed Signals", "❓")


def add_enhanced_columns(df_daily: pd.DataFrame, ticker: str, rolling_window: int = 20) -> pd.DataFrame:
    """
    Add all enhanced columns for a single stock with dual timeframe analysis
    ENHANCED: Now includes 5-day and 20-day momentum/autocorrelation with crossover detection
    
    Simplified Logic:
    - Monday: Sets CRT levels and Valid_CRT
    - Tue-Fri: Forward fill from Monday
    - Momentum: Dual timeframe (5-day and 20-day) rolling calculations
    
    Args:
        df_daily: Raw OHLCV data from yfinance
        ticker: Stock symbol
        rolling_window: Window for moving averages (default 20)
    
    Returns:
        DataFrame with enhanced technical analysis columns including dual timeframe momentum
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
    
    # 16. ENHANCED: DUAL TIMEFRAME MOMENTUM CALCULATIONS
    # Calculate daily returns for momentum analysis
    df['Daily_Returns'] = df['Close'].pct_change()
    
    # Calculate dual timeframe momentum and autocorrelation
    try:
        momentum_df = calculate_dual_rolling_momentum(df['Daily_Returns'])
        
        # Add all momentum-related columns
        df['Momentum_5Day'] = momentum_df['momentum_5day']
        df['Momentum_20Day'] = momentum_df['momentum_20day']
        df['Autocorr_5Day'] = momentum_df['autocorr_5day']
        df['Autocorr_20Day'] = momentum_df['autocorr_20day']
        df['Momentum_Crossover'] = momentum_df['momentum_crossover']
        df['Momentum_Spread'] = momentum_df['momentum_spread']
        df['Autocorr_Alignment'] = momentum_df['autocorr_alignment']
        
        # Add advanced strategy classification
        strategy_classifications = []
        strategy_emojis = []
        
        for i in range(len(df)):
            if pd.notna(df['Momentum_5Day'].iloc[i]):
                strategy, emoji = classify_advanced_trading_strategy(
                    df['Momentum_5Day'].iloc[i],
                    df['Momentum_20Day'].iloc[i],
                    df['Autocorr_5Day'].iloc[i],
                    df['Autocorr_20Day'].iloc[i],
                    df['Momentum_Crossover'].iloc[i],
                    df['Momentum_Spread'].iloc[i]
                )
                strategy_classifications.append(strategy)
                strategy_emojis.append(emoji)
            else:
                strategy_classifications.append("Insufficient Data")
                strategy_emojis.append("❌")
        
        df['Strategy_Type'] = strategy_classifications
        df['Strategy_Signal'] = strategy_emojis
        
        # Debug print
        print(f"DEBUG {ticker}: Dual timeframe momentum calculated (5-day and 20-day)")
        print(f"DEBUG {ticker}: Latest Momentum 5-day: {df['Momentum_5Day'].iloc[-1]:.4f}")
        print(f"DEBUG {ticker}: Latest Momentum 20-day: {df['Momentum_20Day'].iloc[-1]:.4f}")
        print(f"DEBUG {ticker}: Latest Momentum Spread: {df['Momentum_Spread'].iloc[-1]:+.4f}")
        print(f"DEBUG {ticker}: Latest Strategy: {df['Strategy_Type'].iloc[-1]} {df['Strategy_Signal'].iloc[-1]}")
        
    except Exception as e:
        print(f"WARNING {ticker}: Dual momentum calculation failed: {e}")
        # Fallback values
        df['Momentum_5Day'] = 0.5
        df['Momentum_20Day'] = 0.5
        df['Autocorr_5Day'] = 0.0
        df['Autocorr_20Day'] = 0.0
        df['Momentum_Crossover'] = 0
        df['Momentum_Spread'] = 0.0
        df['Autocorr_Alignment'] = 'Unknown'
        df['Strategy_Type'] = 'Calculation Error'
        df['Strategy_Signal'] = '❌'
    
    # Keep backward compatibility columns (map from 5-day values)
    df['Momentum_1Day_Prob'] = df['Momentum_5Day']
    df['Momentum_3Day_Prob'] = df['Momentum_5Day']  # Simplified for compatibility
    df['Autocorr_1Day'] = df['Autocorr_5Day']
    
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
    print(f"DEBUG {ticker}: Created {len(df.columns)} columns including dual timeframe analysis")
    
    return df


def classify_trading_strategy(momentum_1d, autocorr):
    """
    DEPRECATED: Use classify_advanced_trading_strategy() instead
    Kept for backward compatibility with scanner
    """
    # Map 5-day values to old classification for compatibility
    if momentum_1d > 0.60 and autocorr > 0.15:
        return "Pure Momentum"
    elif momentum_1d < 0.45 and autocorr < -0.15:
        return "Pure Mean Reversion"
    elif momentum_1d > 0.60 and autocorr < -0.15:
        return "Momentum + Daily Reversals"
    elif momentum_1d < 0.50 and autocorr > 0.15:
        return "Weak Momentum + Persistence"
    else:
        return "Neutral/Mixed"


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
    ENHANCED: Now includes dual timeframe momentum signals
    
    Args:
        df: Enhanced DataFrame with all indicators
    
    Returns:
        Dictionary with latest signal information including dual timeframe momentum
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
        # Dual timeframe momentum signals
        'momentum_5day': float(latest.get('Momentum_5Day', 0.5)) if not pd.isna(latest.get('Momentum_5Day', 0.5)) else 0.5,
        'momentum_20day': float(latest.get('Momentum_20Day', 0.5)) if not pd.isna(latest.get('Momentum_20Day', 0.5)) else 0.5,
        'momentum_spread': float(latest.get('Momentum_Spread', 0.0)) if not pd.isna(latest.get('Momentum_Spread', 0.0)) else 0.0,
        'momentum_crossover': int(latest.get('Momentum_Crossover', 0)),
        'autocorr_5day': float(latest.get('Autocorr_5Day', 0.0)) if not pd.isna(latest.get('Autocorr_5Day', 0.0)) else 0.0,
        'autocorr_20day': float(latest.get('Autocorr_20Day', 0.0)) if not pd.isna(latest.get('Autocorr_20Day', 0.0)) else 0.0,
        'strategy_type': str(latest.get('Strategy_Type', 'Unknown')),
        'strategy_signal': str(latest.get('Strategy_Signal', '❓'))
    }


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate the quality of the enhanced data
    ENHANCED: Now includes dual timeframe momentum validation
    
    Args:
        df: Enhanced DataFrame
    
    Returns:
        Dictionary with validation results
    """
    required_columns = ['Close', 'High', 'Low', 'Volume', 'IBS', 'Buy_Signal', 
                       'Momentum_5Day', 'Momentum_20Day', 'Autocorr_5Day', 'Autocorr_20Day']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation_results = {
        'is_valid': len(missing_columns) == 0,
        'missing_columns': missing_columns,
        'row_count': len(df),
        'has_recent_data': len(df) > 0,
        'buy_signals_count': int(df['Buy_Signal'].sum()) if 'Buy_Signal' in df.columns else 0,
        'expansion_signals_count': int(df['Rel_Range_Signal'].sum()) if 'Rel_Range_Signal' in df.columns else 0,
        'momentum_data_available': all(col in df.columns for col in ['Momentum_5Day', 'Momentum_20Day']),
        'crossover_signals': int(df['Momentum_Crossover'].abs().sum()) if 'Momentum_Crossover' in df.columns else 0
    }
    
    return validation_results


def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of all signals in the DataFrame
    ENHANCED: Now includes dual timeframe momentum statistics
    
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
            'momentum_5day_latest': 0.5,
            'momentum_20day_latest': 0.5,
            'momentum_spread_latest': 0.0,
            'bullish_crossovers': 0,
            'bearish_crossovers': 0
        }
    
    return {
        'total_days': len(df),
        'buy_signals': int(df['Buy_Signal'].sum()) if 'Buy_Signal' in df.columns else 0,
        'expansion_signals': int(df['Rel_Range_Signal'].sum()) if 'Rel_Range_Signal' in df.columns else 0,
        'high_ibs_days': int((df['IBS'] >= 0.5).sum()) if 'IBS' in df.columns else 0,
        'valid_crt_days': int(df['Valid_CRT'].sum()) if 'Valid_CRT' in df.columns else 0,
        'wick_below_signals': int(df['Wick_Below'].sum()) if 'Wick_Below' in df.columns else 0,
        'close_above_signals': int(df['Close_Above'].sum()) if 'Close_Above' in df.columns else 0,
        # Dual timeframe momentum
        'momentum_5day_latest': float(df['Momentum_5Day'].iloc[-1]) if 'Momentum_5Day' in df.columns and len(df) > 0 else 0.5,
        'momentum_20day_latest': float(df['Momentum_20Day'].iloc[-1]) if 'Momentum_20Day' in df.columns and len(df) > 0 else 0.5,
        'momentum_spread_latest': float(df['Momentum_Spread'].iloc[-1]) if 'Momentum_Spread' in df.columns and len(df) > 0 else 0.0,
        'bullish_crossovers': int((df['Momentum_Crossover'] == 1).sum()) if 'Momentum_Crossover' in df.columns else 0,
        'bearish_crossovers': int((df['Momentum_Crossover'] == -1).sum()) if 'Momentum_Crossover' in df.columns else 0
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


# Enhanced momentum analysis functions
def get_momentum_analysis(df: pd.DataFrame) -> dict:
    """
    Get comprehensive momentum analysis using dual timeframe
    
    Args:
        df: Enhanced DataFrame with momentum columns
    
    Returns:
        Dictionary with detailed momentum analysis
    """
    if df.empty or 'Momentum_5Day' not in df.columns:
        return {
            'analysis_available': False,
            'error': 'Insufficient data for analysis'
        }
    
    # Get latest values
    latest = df.iloc[-1]
    
    # Recent crossover detection (last 3 days)
    recent_data = df.iloc[-3:] if len(df) >= 3 else df
    recent_bullish_cross = (recent_data['Momentum_Crossover'] == 1).any()
    recent_bearish_cross = (recent_data['Momentum_Crossover'] == -1).any()
    
    # Momentum trend analysis
    if len(df) >= 10:
        mom_5d_trend = df['Momentum_5Day'].iloc[-5:].mean() - df['Momentum_5Day'].iloc[-10:-5].mean()
        mom_20d_trend = df['Momentum_20Day'].iloc[-5:].mean() - df['Momentum_20Day'].iloc[-10:-5].mean()
    else:
        mom_5d_trend = 0
        mom_20d_trend = 0
    
    return {
        'analysis_available': True,
        'latest': {
            'momentum_5day': float(latest['Momentum_5Day']),
            'momentum_20day': float(latest['Momentum_20Day']),
            'momentum_spread': float(latest['Momentum_Spread']),
            'autocorr_5day': float(latest['Autocorr_5Day']),
            'autocorr_20day': float(latest['Autocorr_20Day']),
            'strategy': str(latest['Strategy_Type']),
            'signal': str(latest['Strategy_Signal'])
        },
        'crossovers': {
            'recent_bullish': recent_bullish_cross,
            'recent_bearish': recent_bearish_cross,
            'total_bullish': int((df['Momentum_Crossover'] == 1).sum()),
            'total_bearish': int((df['Momentum_Crossover'] == -1).sum())
        },
        'trends': {
            '5day_trend': 'Strengthening' if mom_5d_trend > 0.05 else 'Weakening' if mom_5d_trend < -0.05 else 'Stable',
            '20day_trend': 'Strengthening' if mom_20d_trend > 0.05 else 'Weakening' if mom_20d_trend < -0.05 else 'Stable',
            '5day_trend_value': float(mom_5d_trend),
            '20day_trend_value': float(mom_20d_trend)
        },
        'regime': {
            'current': 'Momentum' if latest['Momentum_5Day'] > 0.55 else 'Mean Reversion' if latest['Momentum_5Day'] < 0.45 else 'Neutral',
            'stability': 'Stable' if abs(latest['Autocorr_5Day'] - latest['Autocorr_20Day']) < 0.30 else 'Changing'
        }
    }


def get_trading_recommendation(analysis: dict) -> dict:
    """
    Generate trading recommendations based on momentum analysis
    
    Args:
        analysis: Result from get_momentum_analysis()
    
    Returns:
        Dictionary with trading recommendations
    """
    if not analysis.get('analysis_available', False):
        return {'recommendation': 'No Analysis Available', 'confidence': 'N/A'}
    
    latest = analysis['latest']
    crossovers = analysis['crossovers']
    
    # Strong buy conditions
    if crossovers['recent_bullish'] and latest['momentum_5day'] > 0.60:
        return {
            'recommendation': 'STRONG BUY',
            'confidence': 'High',
            'rationale': 'Recent bullish crossover with strong momentum',
            'entry': 'Buy at market or on any intraday dip',
            'exit': 'Trail stop or exit on bearish crossover'
        }
    
    # Buy conditions
    elif latest['strategy'] == "Oversold Bounce Setup":
        return {
            'recommendation': 'BUY',
            'confidence': 'Medium',
            'rationale': 'Oversold with mean reversion setup',
            'entry': 'Buy on confirmation of bounce',
            'exit': 'Take profit at 20-day momentum level'
        }
    
    # Hold conditions
    elif latest['strategy'] == "Strong Momentum Continuation":
        return {
            'recommendation': 'HOLD/ADD',
            'confidence': 'High',
            'rationale': 'Strong momentum in both timeframes',
            'entry': 'Add on pullbacks to support',
            'exit': 'Hold until momentum weakens'
        }
    
    # Caution conditions
    elif crossovers['recent_bearish'] or latest['strategy'] == "Momentum Exhaustion":
        return {
            'recommendation': 'SELL/AVOID',
            'confidence': 'High',
            'rationale': 'Bearish crossover or exhaustion detected',
            'entry': 'Do not enter new positions',
            'exit': 'Exit existing positions'
        }
    
    # Neutral
    else:
        return {
            'recommendation': 'NEUTRAL',
            'confidence': 'Low',
            'rationale': 'Mixed signals - wait for clearer setup',
            'entry': 'Wait for crossover or extreme readings',
            'exit': 'N/A'
        }