"""
Technical Analysis Module - PURE MPI EXPANSION SYSTEM
Simple MPI with pure expansion/contraction detection
No baseline threshold - expansion at any level is opportunity
MPI = Market Positivity Index (percentage of positive days)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

def calculate_mpi_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MPI with pure expansion/contraction focus
    No baseline threshold - all expansion is opportunity
    
    Core Concept: Count positive days and track expansion velocity
    - MPI: 10-day percentage of positive days (0-1 scale)
    - MPI_Velocity: Day-over-day change in MPI
    - MPI_Trend: Categorized by velocity alone
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with MPI columns added
    """
    # Calculate daily returns
    returns = df['Close'].pct_change()
    positive_days = (returns > 0).astype(int)
    
    # Core MPI calculation - 10-day rolling percentage
    df['MPI'] = positive_days.rolling(10, min_periods=5).mean()
    
    # Pure velocity calculation (expansion/contraction)
    df['MPI_Velocity'] = df['MPI'] - df['MPI'].shift(1)
    
    # Velocity-based classification (NO BASELINE)
    conditions = [
        df['MPI_Velocity'] >= 0.05,     # Strong expansion (5%+ improvement)
        df['MPI_Velocity'] > 0,         # Any expansion
        df['MPI_Velocity'] == 0,        # Flat/unchanged
        df['MPI_Velocity'] > -0.05,     # Mild contraction
        df['MPI_Velocity'] <= -0.05,    # Strong contraction
    ]
    
    choices = [
        'Strong Expansion',      # 🚀 Best entry signal
        'Expanding',            # 📈 Good entry signal
        'Flat',                 # ➖ Hold/neutral
        'Mild Contraction',     # ⚠️ Warning signal
        'Strong Contraction'    # 📉 Exit/Short signal
    ]
    
    df['MPI_Trend'] = pd.Series(np.select(conditions, choices, default='Flat'), index=df.index)
    
    # Trading signals based on pure expansion
    df['Signal_Expansion_Buy'] = (
        df['MPI_Velocity'] > 0  # Any expansion is a buy signal
    ).astype(int)
    
    df['Signal_Strong_Buy'] = (
        df['MPI_Velocity'] >= 0.05  # Strong expansion
    ).astype(int)
    
    df['Signal_Exit'] = (
        df['MPI_Velocity'] < 0  # Any contraction is exit signal
    ).astype(int)
    
    # Fill NaN values
    df['MPI'] = df['MPI'].fillna(0.5)  # Neutral default
    df['MPI_Velocity'] = df['MPI_Velocity'].fillna(0.0)
    
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

def get_mpi_trend_info(trend: str, mpi_value: float = None) -> Dict[str, str]:
    """
    Get trading guidance for MPI trend (pure expansion focus)
    
    Args:
        trend: MPI trend classification
        mpi_value: Optional MPI value for context
    
    Returns:
        Dictionary with trend info and trading guidance
    """
    trend_info = {
        'Strong Expansion': {
            'emoji': '🚀',
            'color': 'darkgreen',
            'interpretation': 'Strong momentum building (≥5% MPI improvement)',
            'action': 'Strong buy signal - ride the momentum',
            'risk': 'Low - momentum strongly favors upside'
        },
        'Expanding': {
            'emoji': '📈',
            'color': 'green',
            'interpretation': 'Positive momentum developing',
            'action': 'Buy signal - enter or add to positions',
            'risk': 'Low to moderate - positive momentum'
        },
        'Flat': {
            'emoji': '➖',
            'color': 'gray',
            'interpretation': 'No momentum change',
            'action': 'Hold - wait for directional signal',
            'risk': 'Moderate - no clear direction'
        },
        'Mild Contraction': {
            'emoji': '⚠️',
            'color': 'orange',
            'interpretation': 'Momentum weakening slightly',
            'action': 'Caution - consider reducing position',
            'risk': 'Moderate to high - momentum fading'
        },
        'Strong Contraction': {
            'emoji': '📉',
            'color': 'red',
            'interpretation': 'Significant momentum loss (≥5% MPI decline)',
            'action': 'Exit long positions - consider shorts',
            'risk': 'High - strong negative momentum'
        }
    }
    
    info = trend_info.get(trend, {
        'emoji': '❓',
        'color': 'gray',
        'interpretation': 'Unknown trend',
        'action': 'No action - invalid data',
        'risk': 'Unknown'
    })
    
    # Add MPI context if provided
    if mpi_value is not None:
        info['mpi_level'] = f"{mpi_value:.0%}"
        
    return info

def add_enhanced_columns(df_daily: pd.DataFrame, ticker: str, rolling_window: int = 20) -> pd.DataFrame:
    """
    Add enhanced columns with PURE MPI EXPANSION system
    
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
    
    # 5a. Calculate velocity
    df['VW_Range_Velocity'] = df['VW_Range_Percentile'] - df['VW_Range_Percentile'].shift(1)
    
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
    
    # 13. Capture qualifying velocity
    df['CRT_Qualifying_Velocity'] = np.where(
        (df['Is_First_Trading_Day'] == 1) & (df['Rel_Range_Signal'] == 1),
        df['VW_Range_Velocity'],
        np.nan
    )
    
    # 14. Higher_HL pattern
    df['Higher_HL'] = np.where(
        (df['High'] > df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1)),
        1, 
        0
    )

    # 15. Forward fill Valid_CRT and CRT_Qualifying_Velocity
    df['Valid_CRT'] = df['Valid_CRT'].ffill()
    df['CRT_Qualifying_Velocity'] = df['CRT_Qualifying_Velocity'].ffill()
    
    # 16. ✨ PURE MPI EXPANSION SYSTEM
    try:
        df = calculate_mpi_expansion(df)
        
        # Debug print for MPI system
        print(f"DEBUG {ticker}: Pure MPI expansion calculated successfully")
        print(f"DEBUG {ticker}: Latest MPI: {df['MPI'].iloc[-1]:.1%}")
        print(f"DEBUG {ticker}: Latest MPI Velocity: {df['MPI_Velocity'].iloc[-1]:+.3f}")
        print(f"DEBUG {ticker}: Latest MPI Trend: {df['MPI_Trend'].iloc[-1]}")
        
        # Create visual representation
        latest_mpi = df['MPI'].iloc[-1]
        visual = format_mpi_visual(latest_mpi)
        print(f"DEBUG {ticker}: MPI Visual: {visual} ({latest_mpi:.1%})")
        
    except Exception as e:
        print(f"WARNING {ticker}: MPI calculation failed: {e}")
        # Fallback values
        df['MPI'] = 0.5
        df['MPI_Velocity'] = 0.0
        df['MPI_Trend'] = 'Calculation Error'
        df['Signal_Expansion_Buy'] = 0
        df['Signal_Strong_Buy'] = 0
        df['Signal_Exit'] = 0
    
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
    print(f"DEBUG {ticker}: Created {len(df.columns)} columns with PURE MPI EXPANSION system")
    
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
                       'MPI', 'MPI_Velocity', 'MPI_Trend']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation_results = {
        'is_valid': len(missing_columns) == 0,
        'missing_columns': missing_columns,
        'row_count': len(df),
        'has_recent_data': len(df) > 0,
        'buy_signals_count': int(df['Buy_Signal'].sum()) if 'Buy_Signal' in df.columns else 0,
        'expansion_signals_count': int(df['Signal_Expansion_Buy'].sum()) if 'Signal_Expansion_Buy' in df.columns else 0,
        'mpi_data_available': 'MPI' in df.columns,
        'strong_expansion_count': int((df['MPI_Trend'] == 'Strong Expansion').sum()) if 'MPI_Trend' in df.columns else 0,
        'contraction_count': int(df['MPI_Trend'].isin(['Mild Contraction', 'Strong Contraction']).sum()) if 'MPI_Trend' in df.columns else 0
    }
    
    return validation_results

# Utility functions for the scanner
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
def get_mpi_expansion_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of MPI expansion signals in the DataFrame
    """
    if df.empty or 'MPI' not in df.columns:
        return {
            'total_days': 0,
            'avg_mpi': 0.5,
            'avg_velocity': 0.0,
            'strong_expansion_days': 0,
            'expanding_days': 0,
            'flat_days': 0,
            'contracting_days': 0
        }
    
    return {
        'total_days': len(df),
        'avg_mpi': float(df['MPI'].mean()),
        'avg_velocity': float(df['MPI_Velocity'].mean()),
        'strong_expansion_days': int((df['MPI_Trend'] == 'Strong Expansion').sum()) if 'MPI_Trend' in df.columns else 0,
        'expanding_days': int((df['MPI_Trend'] == 'Expanding').sum()) if 'MPI_Trend' in df.columns else 0,
        'flat_days': int((df['MPI_Trend'] == 'Flat').sum()) if 'MPI_Trend' in df.columns else 0,
        'mild_contraction_days': int((df['MPI_Trend'] == 'Mild Contraction').sum()) if 'MPI_Trend' in df.columns else 0,
        'strong_contraction_days': int((df['MPI_Trend'] == 'Strong Contraction').sum()) if 'MPI_Trend' in df.columns else 0,
        'expansion_buy_signals': int(df['Signal_Expansion_Buy'].sum()) if 'Signal_Expansion_Buy' in df.columns else 0,
        'strong_buy_signals': int(df['Signal_Strong_Buy'].sum()) if 'Signal_Strong_Buy' in df.columns else 0,
        'exit_signals': int(df['Signal_Exit'].sum()) if 'Signal_Exit' in df.columns else 0
    }

def get_mpi_trend_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get distribution of stocks across MPI expansion trends
    """
    if df.empty or 'MPI_Trend' not in df.columns:
        return pd.DataFrame()
    
    trends = []
    for _, row in df.iterrows():
        mpi_value = row['MPI']
        mpi_velocity = row['MPI_Velocity']
        trend = row['MPI_Trend']
        trend_info = get_mpi_trend_info(trend, mpi_value)
        
        trends.append({
            'Ticker': row.get('Ticker', 'Unknown'),
            'MPI': mpi_value,
            'Velocity': mpi_velocity,
            'Trend': trend,
            'Trend_Emoji': trend_info['emoji'],
            'Action': trend_info['action']
        })
    
    return pd.DataFrame(trends)

print("✅ Technical Analysis Module loaded with PURE MPI EXPANSION system - no baseline threshold!")