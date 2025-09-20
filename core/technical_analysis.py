# File: core/technical_analysis.py
# Part 1 of 2
"""
Technical Analysis Module - PURE MPI EXPANSION SYSTEM
Optimized MPI with pure expansion/contraction detection
MPI = Market Positivity Index (percentage of positive days)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

def calculate_mpi_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MPI with pure expansion/contraction focus
    
    Core Concept: Count positive days and track expansion velocity
    - MPI: 10-day percentage of positive days (0-1 scale)
    - MPI_Velocity: Day-over-day change in MPI
    - MPI_Trend: Categorized by velocity alone
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with MPI columns added
    """
    # Calculate daily returns and positive days
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
    df['Signal_Expansion_Buy'] = (df['MPI_Velocity'] > 0).astype(int)
    df['Signal_Strong_Buy'] = (df['MPI_Velocity'] >= 0.05).astype(int)
    df['Signal_Exit'] = (df['MPI_Velocity'] < 0).astype(int)
    
    # Fill NaN values
    df['MPI'] = df['MPI'].fillna(0.5)  # Neutral default
    df['MPI_Velocity'] = df['MPI_Velocity'].fillna(0.0)
    
    return df

def format_mpi_visual(mpi_value: float) -> str:
    """Convert MPI to visual blocks for intuitive display"""
    if pd.isna(mpi_value):
        return "░░░░░░░░░░"
    
    blocks = max(0, min(10, int(mpi_value * 10)))
    return "█" * blocks + "░" * (10 - blocks)

def get_mpi_trend_info(trend: str, mpi_value: float = None) -> Dict[str, str]:
    """Get trading guidance for MPI trend (pure expansion focus)"""
    trend_info = {
        'Strong Expansion': {
            'emoji': '🚀',
            'color': 'darkgreen',
            'description': 'Strong momentum building',
            'action': 'Strong buy signal - ride the momentum',
            'risk': 'Low - momentum strongly favors upside'
        },
        'Expanding': {
            'emoji': '📈',
            'color': 'green',
            'description': 'Positive momentum developing',
            'action': 'Buy signal - enter or add to positions',
            'risk': 'Low to moderate - positive momentum'
        },
        'Flat': {
            'emoji': '➖',
            'color': 'gray',
            'description': 'No momentum change',
            'action': 'Hold - wait for directional signal',
            'risk': 'Moderate - no clear direction'
        },
        'Mild Contraction': {
            'emoji': '⚠️',
            'color': 'orange',
            'description': 'Momentum weakening slightly',
            'action': 'Caution - consider reducing position',
            'risk': 'Moderate to high - momentum fading'
        },
        'Strong Contraction': {
            'emoji': '📉',
            'color': 'red',
            'description': 'Significant momentum loss',
            'action': 'Exit long positions - consider shorts',
            'risk': 'High - strong negative momentum'
        }
    }
    
    info = trend_info.get(trend, {
        'emoji': '❓',
        'color': 'gray',
        'description': 'Unknown trend',
        'action': 'No action - invalid data',
        'risk': 'Unknown'
    })
    
    if mpi_value is not None:
        info['mpi_level'] = f"{mpi_value:.0%}"
        
    return info

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate core technical indicators efficiently"""
    # Daily relative range
    df['Daily_Rel_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Percentile rankings
    df['Daily_Range_Percentile'] = df['Daily_Rel_Range'].rolling(
        window=50, min_periods=20
    ).rank(pct=True)
    
    # Volume normalization
    df['Volume_Normalized'] = df['Volume'] / df['Volume'].rolling(
        window=20, min_periods=10
    ).mean()
    
    # Volume-weighted range
    df['Volume_Weighted_Range'] = df['Daily_Rel_Range'] * df['Volume_Normalized']
    
    # Volume-weighted range percentile and velocity
    df['VW_Range_Percentile'] = df['Volume_Weighted_Range'].rolling(
        window=50, min_periods=20
    ).rank(pct=True)
    df['VW_Range_Velocity'] = df['VW_Range_Percentile'] - df['VW_Range_Percentile'].shift(1)
    
    # Range expansion signal
    df['Rel_Range_Signal'] = (
        df['VW_Range_Percentile'] > df['VW_Range_Percentile'].shift(1)
    ).astype(int)
    
    # IBS calculation
    df['IBS'] = np.where(
        df['High'] != df['Low'],
        (df['Close'] - df['Low']) / (df['High'] - df['Low']),
        1.0
    )
    
    # Higher H/L pattern
    df['Higher_HL'] = (
        (df['High'] > df['High'].shift(1)) & 
        (df['Low'] > df['Low'].shift(1))
    ).astype(int)
    
    return df

def calculate_crt_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate CRT (Candle Range Theory) levels and signals"""
    # Trading day identification
    df['Is_First_Trading_Day'] = (df.index.weekday == 0).astype(int)
    
    # Initialize CRT columns
    crt_columns = ['Weekly_Open', 'CRT_High', 'CRT_Low', 'CRT_Close']
    for col in crt_columns:
        df[col] = np.nan
    
    # Set CRT values on Mondays
    monday_mask = df['Is_First_Trading_Day'] == 1
    df.loc[monday_mask, 'Weekly_Open'] = df.loc[monday_mask, 'Open']
    df.loc[monday_mask, 'CRT_High'] = df.loc[monday_mask, 'High']
    df.loc[monday_mask, 'CRT_Low'] = df.loc[monday_mask, 'Low']
    df.loc[monday_mask, 'CRT_Close'] = df.loc[monday_mask, 'Close']
    
    # Forward fill CRT values
    for col in crt_columns:
        df[col] = df[col].ffill()
    
    # Valid CRT and qualifying velocity
    df['Valid_CRT'] = np.where(
        (df['Is_First_Trading_Day'] == 1) & (df['Rel_Range_Signal'] == 1), 1,
        np.where(df['Is_First_Trading_Day'] == 1, 0, np.nan)
    )
    
    df['CRT_Qualifying_Velocity'] = np.where(
        (df['Is_First_Trading_Day'] == 1) & (df['Rel_Range_Signal'] == 1),
        df['VW_Range_Velocity'],
        np.nan
    )
    
    # Forward fill
    df['Valid_CRT'] = df['Valid_CRT'].ffill()
    df['CRT_Qualifying_Velocity'] = df['CRT_Qualifying_Velocity'].ffill()
    
    return df

# File: core/technical_analysis.py
# Part 2 of 2
"""
Technical Analysis Module - Part 2
CRT signal calculations and main enhancement function
"""

def calculate_crt_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate CRT trading signals efficiently"""
    # Initialize signal columns
    df['Wick_Below'] = 0
    df['Close_Above'] = 0
    
    # Create week grouping for efficient processing
    df['week_start'] = df.index - pd.to_timedelta(df.index.weekday, unit='D')
    df['week_start'] = df['week_start'].dt.normalize()
    
    # Process each week's signals
    for week_start in df['week_start'].unique():
        week_mask = df['week_start'] == week_start
        week_data = df[week_mask].copy()
        
        if len(week_data) == 0:
            continue
        
        # Get CRT levels for the week
        crt_high = week_data['CRT_High'].iloc[0]
        crt_low = week_data['CRT_Low'].iloc[0]
        
        if pd.isna(crt_high) or pd.isna(crt_low):
            continue
        
        # Process signal days (Tuesday-Friday)
        signal_days = week_data[week_data.index.weekday > 0]
        if len(signal_days) == 0:
            continue
        
        # WICK_BELOW signal logic
        wick_below_triggered = _process_wick_below_signal(signal_days, crt_low)
        if wick_below_triggered is not None:
            subsequent_days = signal_days[signal_days.index >= wick_below_triggered].index
            df.loc[subsequent_days, 'Wick_Below'] = 1
        
        # CLOSE_ABOVE signal logic  
        close_above_triggered = _process_close_above_signal(signal_days, crt_high)
        if close_above_triggered is not None:
            subsequent_days = signal_days[signal_days.index >= close_above_triggered].index
            df.loc[subsequent_days, 'Close_Above'] = 1
    
    # Clean up temporary column
    df.drop(['week_start'], axis=1, inplace=True)
    
    # Calculate final buy signal
    df['Buy_Signal'] = (
        (df['Valid_CRT'] == 1) &
        (df['IBS'] >= 0.5) &
        ((df['Wick_Below'] == 1) | (df['Close_Above'] == 1))
    ).astype(int)
    
    return df

def _process_wick_below_signal(signal_days: pd.DataFrame, crt_low: float) -> Optional[pd.Timestamp]:
    """Process wick below signal for a week"""
    condition_1_triggered = False
    
    for day_date, day_row in signal_days.iterrows():
        # Check if low breached CRT low
        if day_row['Low'] < crt_low:
            condition_1_triggered = True
        
        # Check if close recovered above CRT low after breach
        if condition_1_triggered and day_row['Close'] >= crt_low:
            return day_date
    
    return None

def _process_close_above_signal(signal_days: pd.DataFrame, crt_high: float) -> Optional[pd.Timestamp]:
    """Process close above signal for a week"""
    for day_date, day_row in signal_days.iterrows():
        if day_row['Close'] >= crt_high:
            return day_date
    
    return None

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
    
    try:
        # Apply technical indicators in logical sequence
        df = calculate_technical_indicators(df)
        df = calculate_crt_levels(df)
        df = calculate_mpi_expansion(df)
        df = calculate_crt_signals(df)
        
        # Log successful calculation
        logger.info(f"{ticker}: Enhanced analysis completed successfully")
        logger.info(f"{ticker}: Latest MPI: {df['MPI'].iloc[-1]:.1%}, "
                   f"Velocity: {df['MPI_Velocity'].iloc[-1]:+.3f}, "
                   f"Trend: {df['MPI_Trend'].iloc[-1]}")
        
    except Exception as e:
        logger.error(f"{ticker}: Technical analysis failed: {e}")
        # Add fallback MPI values
        df['MPI'] = 0.5
        df['MPI_Velocity'] = 0.0
        df['MPI_Trend'] = 'Calculation Error'
        df['Signal_Expansion_Buy'] = 0
        df['Signal_Strong_Buy'] = 0
        df['Signal_Exit'] = 0
    
    return df

# Utility functions
def calculate_ibs(high: float, low: float, close: float) -> float:
    """Calculate Internal Bar Strength (IBS)"""
    if high == low:
        return 1.0
    return (close - low) / (high - low)

def detect_range_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """Detect range expansion signals"""
    expansion_mask = (
        (df['VW_Range_Percentile'] > df['VW_Range_Percentile'].shift(1)) & 
        (df['VW_Range_Percentile'].shift(1) <= 0.5)
    )
    return df[expansion_mask]

def validate_data_quality(df: pd.DataFrame) -> dict:
    """Validate the quality of the MPI-enhanced data"""
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

def get_buy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Get only the rows where buy signals are active"""
    if 'Buy_Signal' not in df.columns:
        return pd.DataFrame()
    return df[df['Buy_Signal'] == 1].copy()

def calculate_technical_indicators_wrapper(df: pd.DataFrame, ticker: str = 'Unknown') -> pd.DataFrame:
    """Simple wrapper for add_enhanced_columns (backward compatibility)"""
    return add_enhanced_columns(df, ticker)

def get_mpi_expansion_summary(df: pd.DataFrame) -> dict:
    """Get a summary of MPI expansion signals in the DataFrame"""
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
    """Get distribution of stocks across MPI expansion trends"""
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

logger.info("Technical Analysis Module loaded with optimized PURE MPI EXPANSION system")