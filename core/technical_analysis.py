# File: core/technical_analysis.py
"""
Technical Analysis Module - CLEAN MPI SYSTEM
Complete replacement of complex dual timeframe momentum with simple MPI
NO backward compatibility - clean slate implementation
MPI = Market Positivity Index (percentage of positive days in rolling window)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

def calculate_enhanced_mpi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced MPI - A complete momentum system in one indicator
    
    Core Concept: Just count positive days and divide by window size
    - MPI_Fast (3-day): Current momentum pulse
    - MPI_Base (5-day): Core momentum reading  
    - MPI_Slow (10-day): Underlying trend strength
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with MPI columns added
    """
    # Calculate daily returns
    returns = df['Close'].pct_change()
    positive_days = (returns > 0).astype(int)
    
    # Core MPI calculations - just counting positive days!
    df['MPI_Fast'] = positive_days.rolling(3, min_periods=2).mean()    # 3-day: Current momentum
    df['MPI_Base'] = positive_days.rolling(5, min_periods=3).mean()    # 5-day: Base momentum  
    df['MPI_Slow'] = positive_days.rolling(10, min_periods=5).mean()   # 10-day: Trend strength
    
    # Momentum Quality Metrics
    df['MPI_Acceleration'] = df['MPI_Fast'] - df['MPI_Slow']           # Is momentum accelerating?
    df['MPI_Consistency'] = df['MPI_Base'].rolling(5, min_periods=3).std()  # Is momentum stable?
    
    # Convert to Market State (categorical - easy to understand!)
    conditions = [
        (df['MPI_Base'] >= 0.7) & (df['MPI_Acceleration'] > 0.1),     # Strong Bull + Accelerating
        (df['MPI_Base'] >= 0.7) & (df['MPI_Acceleration'] <= 0.1),    # Strong Bull + Slowing
        (df['MPI_Base'] >= 0.5) & (df['MPI_Acceleration'] > 0.1),     # Weak Bull + Accelerating
        (df['MPI_Base'] >= 0.5) & (df['MPI_Acceleration'] <= 0.1),    # Weak Bull + Slowing
        (df['MPI_Base'] < 0.5) & (df['MPI_Base'] >= 0.3),             # Neutral/Choppy
        (df['MPI_Base'] < 0.3)                                         # Bearish
    ]
    
    choices = [
        'Strong Bull Rising',
        'Strong Bull Slowing', 
        'Bull Acceleration',
        'Bull Deceleration',
        'Neutral Zone',
        'Bear Market'
    ]
    
    df['MPI_State'] = pd.Series(np.select(conditions, choices, default='Neutral Zone'), index=df.index)
    
    # Generate Trading Signals (3 clear strategies)
    df['Signal_Breakout'] = (
        (df['MPI_Base'] >= 0.7) & 
        (df['MPI_Acceleration'] > 0.1)
    ).astype(int)
    
    df['Signal_Pullback'] = (
        (df['MPI_Base'] >= 0.4) & 
        (df['MPI_Base'] <= 0.6) & 
        (df['MPI_Slow'] > 0.6)
    ).astype(int)
    
    df['Signal_Short'] = (
        (df['MPI_Base'] < 0.3) & 
        (df['MPI_Acceleration'] < -0.1)
    ).astype(int)
    
    # Fill NaN values with neutral defaults
    mpi_columns = ['MPI_Fast', 'MPI_Base', 'MPI_Slow']
    for col in mpi_columns:
        df[col] = df[col].fillna(0.5)  # 50% = neutral
    
    df['MPI_Acceleration'] = df['MPI_Acceleration'].fillna(0.0)
    df['MPI_Consistency'] = df['MPI_Consistency'].fillna(0.0)
    
    return df

def format_mpi_visual(mpi_value: float) -> str:
    """
    Convert MPI to visual blocks for intuitive display
    
    Args:
        mpi_value: MPI value between 0 and 1
    
    Returns:
        Visual representation using block characters
    """
    if pd.isna(mpi_value):
        return "░░░░░░░░░░"
    
    blocks = max(0, min(10, int(mpi_value * 10)))  # Ensure 0-10 range
    return "█" * blocks + "░" * (10 - blocks)

def get_mpi_strategy_zone(mpi_value: float) -> Dict[str, str]:
    """
    Determine MPI strategy zone and characteristics
    
    Args:
        mpi_value: MPI Base value (0.0 to 1.0)
    
    Returns:
        Dictionary with zone info, color, and interpretation
    """
    if pd.isna(mpi_value):
        return {
            'zone': 'Unknown',
            'emoji': '❓',
            'color': 'gray',
            'interpretation': 'Insufficient data',
            'action': 'Wait for more data'
        }
    
    if mpi_value >= 0.70:
        return {
            'zone': 'Strong Bull',
            'emoji': '🚀',
            'color': 'darkgreen',
            'interpretation': f'{mpi_value:.0%} green days - Strong upward momentum',
            'action': 'Buy on dips, ride the trend'
        }
    elif mpi_value >= 0.50:
        return {
            'zone': 'Bull Trend',
            'emoji': '📈',
            'color': 'green',
            'interpretation': f'{mpi_value:.0%} green days - Positive momentum',
            'action': 'Buy breakouts, hold positions'
        }
    elif mpi_value >= 0.30:
        return {
            'zone': 'Neutral',
            'emoji': '➖',
            'color': 'orange',
            'interpretation': f'{mpi_value:.0%} green days - Mixed signals',
            'action': 'Wait for clearer direction'
        }
    else:
        return {
            'zone': 'Bear Trend',
            'emoji': '📉',
            'color': 'red',
            'interpretation': f'{mpi_value:.0%} green days - Weak momentum',
            'action': 'Avoid longs, consider shorts'
        }

def add_enhanced_columns(df_daily: pd.DataFrame, ticker: str, rolling_window: int = 20) -> pd.DataFrame:
    """
    Add enhanced columns with CLEAN MPI system
    Removes ALL legacy momentum/autocorrelation complexity
    
    Args:
        df_daily: Raw OHLCV data from yfinance
        ticker: Stock symbol
        rolling_window: Window for moving averages (kept for backward compatibility)
    
    Returns:
        DataFrame with MPI-enhanced technical analysis columns
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
    
    # 5a. Calculate VCRE velocity (renamed from VW_Range_Velocity)
    df['VCRE_Velocity'] = df['VW_Range_Percentile'] - df['VW_Range_Percentile'].shift(1)
    
    # 6. Range Expansion Signal
    range_expanding = (df['VW_Range_Percentile'] > df['VW_Range_Percentile'].shift(1))
    df['Rel_Range_Signal'] = np.where(range_expanding, 1, 0)
    
    # 7. Create Is_First_Trading_Day column
    df['Is_First_Trading_Day'] = np.where(df.index.weekday == 0, 1, 0)
    
    # 8. Initialize CRT columns
    df['Weekly_Open'] = np.nan
    df['CRT_High'] = np.nan
    df['CRT_Low'] = np.nan
    df['CRT_Close'] = np.nan
    
    # 9. Set CRT values on Mondays only
    monday_mask = df['Is_First_Trading_Day'] == 1
    df.loc[monday_mask, 'Weekly_Open'] = df.loc[monday_mask, 'Open']
    df.loc[monday_mask, 'CRT_High'] = df.loc[monday_mask, 'High']
    df.loc[monday_mask, 'CRT_Low'] = df.loc[monday_mask, 'Low']
    df.loc[monday_mask, 'CRT_Close'] = df.loc[monday_mask, 'Close']
    
    # 10. Forward fill CRT values
    df['Weekly_Open'] = df['Weekly_Open'].ffill()
    df['CRT_High'] = df['CRT_High'].ffill()
    df['CRT_Low'] = df['CRT_Low'].ffill()
    df['CRT_Close'] = df['CRT_Close'].ffill()
    
    # 11. Calculate IBS
    df['IBS'] = np.where(
        df['High'] != df['Low'],
        (df['Close'] - df['Low']) / (df['High'] - df['Low']),
        1.0
    )
    
    # 12. Create Valid_CRT
    df['Valid_CRT'] = np.where(
        (df['Is_First_Trading_Day'] == 1) & (df['Rel_Range_Signal'] == 1), 1,
        np.where(df['Is_First_Trading_Day'] == 1, 0, np.nan)
    )
    
    # NOTE: CRT_Qualifying_Velocity removed - using VCRE_Velocity directly
    
    # 14. Higher_HL pattern
    df['Higher_HL'] = np.where(
        (df['High'] > df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1)),
        1, 
        0
    )

    # 15. Forward fill Valid_CRT
    df['Valid_CRT'] = df['Valid_CRT'].ffill()
    
    # 16. ✨ CLEAN MPI SYSTEM (replaces ALL complex momentum calculations)
    try:
        df = calculate_enhanced_mpi(df)
        
        # Debug print for MPI system
        print(f"DEBUG {ticker}: Clean MPI system calculated successfully")
        print(f"DEBUG {ticker}: Latest MPI Base: {df['MPI_Base'].iloc[-1]:.1%}")
        print(f"DEBUG {ticker}: Latest MPI State: {df['MPI_State'].iloc[-1]}")
        print(f"DEBUG {ticker}: Latest MPI Acceleration: {df['MPI_Acceleration'].iloc[-1]:+.3f}")
        
        # Create visual representation
        latest_mpi = df['MPI_Base'].iloc[-1]
        visual = format_mpi_visual(latest_mpi)
        print(f"DEBUG {ticker}: MPI Visual: {visual} ({latest_mpi:.1%})")
        
    except Exception as e:
        print(f"WARNING {ticker}: MPI calculation failed: {e}")
        # Fallback values
        df['MPI_Fast'] = 0.5
        df['MPI_Base'] = 0.5
        df['MPI_Slow'] = 0.5
        df['MPI_Acceleration'] = 0.0
        df['MPI_Consistency'] = 0.0
        df['MPI_State'] = 'Calculation Error'
        df['Signal_Breakout'] = 0
        df['Signal_Pullback'] = 0
        df['Signal_Short'] = 0
    
    # 17. CRT Signal calculations
    df['Wick_Below'] = 0
    df['Close_Above'] = 0
    
    # Calculate CRT signals using forward-filled levels
    df['week_start'] = df.index - pd.to_timedelta(df.index.weekday, unit='D')
    df['week_start'] = df['week_start'].dt.normalize()
    
    unique_weeks = df['week_start'].unique()
    
    for week_start in unique_weeks:
        week_mask = df['week_start'] == week_start
        week_data = df[week_mask].copy()
        
        if len(week_data) == 0:
            continue
        
        crt_high = week_data['CRT_High'].iloc[0]
        crt_low = week_data['CRT_Low'].iloc[0]
        
        if pd.isna(crt_high) or pd.isna(crt_low):
            continue
        
        # WICK_BELOW and CLOSE_ABOVE logic
        signal_days = week_data[week_data.index.weekday > 0]
        
        condition_1_triggered = False
        wick_below_trigger_date = None
        
        for day_date, day_row in signal_days.iterrows():
            if day_row['Low'] < crt_low:
                condition_1_triggered = True
            if condition_1_triggered and day_row['Close'] >= crt_low:
                wick_below_trigger_date = day_date
                break
        
        if wick_below_trigger_date is not None:
            subsequent_days = signal_days[signal_days.index >= wick_below_trigger_date].index
            df.loc[subsequent_days, 'Wick_Below'] = 1
        
        close_above_trigger_date = None
        
        for day_date, day_row in signal_days.iterrows():
            if day_row['Close'] >= crt_high:
                close_above_trigger_date = day_date
                break
        
        if close_above_trigger_date is not None:
            subsequent_days = signal_days[signal_days.index >= close_above_trigger_date].index
            df.loc[subsequent_days, 'Close_Above'] = 1
    
    # Clean up temporary columns
    df.drop(['week_start'], axis=1, inplace=True)
    
    # 18. Calculate Buy_Signal
    df['Buy_Signal'] = np.where(
        (df['Valid_CRT'] == 1) &
        (df['IBS'] >= 0.5) &
        ((df['Wick_Below'] == 1) | (df['Close_Above'] == 1)),
        1, 0
    )
    
    # Debug print to verify column creation
    print(f"DEBUG {ticker}: Created {len(df.columns)} columns with CLEAN MPI system")
    
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

def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Validate the quality of the MPI-enhanced data
    
    Args:
        df: Enhanced DataFrame
    
    Returns:
        Dictionary with validation results
    """
    required_columns = ['Close', 'High', 'Low', 'Volume', 'IBS', 'Buy_Signal', 
                       'MPI_Base', 'MPI_Fast', 'MPI_Slow']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation_results = {
        'is_valid': len(missing_columns) == 0,
        'missing_columns': missing_columns,
        'row_count': len(df),
        'has_recent_data': len(df) > 0,
        'buy_signals_count': int(df['Buy_Signal'].sum()) if 'Buy_Signal' in df.columns else 0,
        'expansion_signals_count': int(df['Rel_Range_Signal'].sum()) if 'Rel_Range_Signal' in df.columns else 0,
        'mpi_data_available': 'MPI_Base' in df.columns,
        'strong_bull_signals': int((df['MPI_Base'] >= 0.7).sum()) if 'MPI_Base' in df.columns else 0,
        'bear_signals': int((df['MPI_Base'] < 0.3).sum()) if 'MPI_Base' in df.columns else 0
    }
    
    return validation_results

# Utility functions for the scanner (simplified, no legacy baggage)
def get_buy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get only the rows where buy signals are active
    """
    if 'Buy_Signal' not in df.columns:
        return pd.DataFrame()
    
    return df[df['Buy_Signal'] == 1].copy()

def calculate_technical_indicators(df: pd.DataFrame, ticker: str = 'Unknown') -> pd.DataFrame:
    """
    Simple wrapper for add_enhanced_columns (for any external compatibility needs)
    """
    return add_enhanced_columns(df, ticker)

# MPI-specific utility functions
def get_mpi_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of MPI signals in the DataFrame
    """
    if df.empty or 'MPI_Base' not in df.columns:
        return {
            'total_days': 0,
            'avg_mpi': 0.5,
            'strong_bull_days': 0,
            'bull_trend_days': 0,
            'neutral_days': 0,
            'bear_trend_days': 0
        }
    
    mpi_base = df['MPI_Base']
    
    return {
        'total_days': len(df),
        'avg_mpi': float(mpi_base.mean()),
        'strong_bull_days': int((mpi_base >= 0.7).sum()),
        'bull_trend_days': int(((mpi_base >= 0.5) & (mpi_base < 0.7)).sum()),
        'neutral_days': int(((mpi_base >= 0.3) & (mpi_base < 0.5)).sum()),
        'bear_trend_days': int((mpi_base < 0.3).sum()),
        'breakout_signals': int(df['Signal_Breakout'].sum()) if 'Signal_Breakout' in df.columns else 0,
        'pullback_signals': int(df['Signal_Pullback'].sum()) if 'Signal_Pullback' in df.columns else 0,
        'short_signals': int(df['Signal_Short'].sum()) if 'Signal_Short' in df.columns else 0
    }

def get_mpi_zones_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get distribution of stocks across MPI strategy zones
    """
    if df.empty or 'MPI_Base' not in df.columns:
        return pd.DataFrame()
    
    zones = []
    for _, row in df.iterrows():
        mpi_base = row['MPI_Base']
        zone_info = get_mpi_strategy_zone(mpi_base)
        zones.append({
            'Ticker': row.get('Ticker', 'Unknown'),
            'MPI_Base': mpi_base,
            'Zone': zone_info['zone'],
            'Zone_Emoji': zone_info['emoji'],
            'Interpretation': zone_info['interpretation']
        })
    
    return pd.DataFrame(zones)

# Remove ALL legacy functions - clean slate!
# ❌ calculate_dual_rolling_momentum() - REMOVED
# ❌ classify_advanced_trading_strategy() - REMOVED  
# ❌ get_momentum_analysis() - REMOVED
# ❌ get_trading_recommendation() - REMOVED
# ❌ get_latest_signals() - REMOVED
# ❌ get_signal_summary() - REMOVED
# ❌ All complex momentum/autocorrelation functions - REMOVED

print("✅ Technical Analysis Module loaded with CLEAN MPI system - no legacy code!")