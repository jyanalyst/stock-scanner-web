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

# Import performance optimizations
try:
    from pages.common.performance import cached_computation
except ImportError:
    # Fallback if performance module not available
    def cached_computation(name):
        def decorator(func):
            return func
        return decorator

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
    
    # FIXED: Velocity-based classification with correct logic
    conditions = [
        df['MPI_Velocity'] > 0,    # Expanding - any positive momentum
        df['MPI_Velocity'] == 0,   # Flat - no change
        df['MPI_Velocity'] < 0,    # Contracting - any negative momentum
    ]

    choices = [
        'Expanding',     # 📈 Positive momentum
        'Flat',          # ➖ No change  
        'Contracting'    # 📉 Negative momentum
    ]
    
    df['MPI_Trend'] = pd.Series(np.select(conditions, choices, default='Flat'), index=df.index)
    
    # Log MPI trend distribution for monitoring
    if len(df) > 0:
        trend_counts = df['MPI_Trend'].value_counts()
        logger.debug(f"MPI Trends: {dict(trend_counts)}")

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
        'Expanding': {
            'emoji': '📈',
            'color': 'green',
            'description': 'Positive momentum',
            'action': 'Buy signal - enter positions',
            'risk': 'Low to moderate - positive momentum'
        },
        'Flat': {
            'emoji': '➖',
            'color': 'gray',
            'description': 'No momentum change',
            'action': 'Hold - wait for directional signal',
            'risk': 'Moderate - no clear direction'
        },
        'Contracting': {
            'emoji': '📉',
            'color': 'red',
            'description': 'Negative momentum',
            'action': 'Exit positions - consider shorts',
            'risk': 'High - negative momentum'
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
    
    # Higher H pattern - NEW: Only requires higher high
    df['Higher_H'] = (df['High'] > df['High'].shift(1)).astype(int)

    # Higher H/L pattern - Existing: Requires both higher high AND higher low
    df['Higher_HL'] = (
        (df['High'] > df['High'].shift(1)) &
        (df['Low'] > df['Low'].shift(1))
    ).astype(int)

    # Lower L pattern - Lower low only
    df['Lower_L'] = (df['Low'] < df['Low'].shift(1)).astype(int)

    # Lower H/L pattern - Both lower high AND lower low
    df['Lower_HL'] = (
        (df['High'] < df['High'].shift(1)) &
        (df['Low'] < df['Low'].shift(1))
    ).astype(int)
    
    return df

def calculate_relative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Relative Volume with Velocity tracking for filtering
    Enhanced to match VW Range Velocity and MPI Velocity patterns
    
    Relative Volume = (Current Day Volume) / (14-day Average Volume) × 100
    RelVol Velocity = Day-over-day change in Relative Volume
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with Relative Volume columns and velocity metrics added
    """
    # Calculate 14-day rolling average volume
    df['Volume_14D_Avg'] = df['Volume'].rolling(14, min_periods=7).mean()
    
    # Calculate relative volume as percentage
    df['Relative_Volume'] = (df['Volume'] / df['Volume_14D_Avg']) * 100
    
    # NEW: Direct velocity (day-over-day change)
    df['RelVol_Velocity'] = df['Relative_Volume'] - df['Relative_Volume'].shift(1)
    
    # NEW: Percentile-based velocity (normalized, comparable across stocks)
    df['RelVol_Percentile'] = df['Relative_Volume'].rolling(
        window=50, min_periods=20
    ).rank(pct=True)
    df['RelVol_Percentile_Velocity'] = df['RelVol_Percentile'] - df['RelVol_Percentile'].shift(1)
    
    # NEW: Volume trend classification (for display and filtering)
    conditions = [
        df['RelVol_Velocity'] > 0,    # Building volume
        df['RelVol_Velocity'] == 0,   # Stable volume
        df['RelVol_Velocity'] < 0,    # Fading volume
    ]
    choices = ['Building', 'Stable', 'Fading']
    df['RelVol_Trend'] = pd.Series(np.select(conditions, choices, default='Stable'), index=df.index)
    
    # Fill NaN values
    df['Relative_Volume'] = df['Relative_Volume'].fillna(100.0)
    df['RelVol_Velocity'] = df['RelVol_Velocity'].fillna(0.0)
    df['RelVol_Percentile'] = df['RelVol_Percentile'].fillna(0.5)
    df['RelVol_Percentile_Velocity'] = df['RelVol_Percentile_Velocity'].fillna(0.0)
    
    # High activity flags for reference (existing)
    df['High_Rel_Volume_150'] = (df['Relative_Volume'] >= 150).astype(int)  # 1.5x average
    df['High_Rel_Volume_200'] = (df['Relative_Volume'] >= 200).astype(int)  # 2x average
    
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

@cached_computation("technical_analysis")
def add_enhanced_columns(df_daily: pd.DataFrame, ticker: str, rolling_window: int = 20) -> pd.DataFrame:
    """
    Add enhanced columns with PURE MPI EXPANSION system and Relative Volume
    PERFORMANCE OPTIMIZED: Added caching for expensive computations
    REMOVED: Market Regime analysis

    Args:
        df_daily: Raw OHLCV data from yfinance
        ticker: Stock symbol
        rolling_window: Window for moving averages (kept for backward compatibility)

    Returns:
        DataFrame with MPI-enhanced technical analysis columns and Relative Volume
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
        df = calculate_relative_volume(df)
        
        # Log successful calculation
        logger.debug(f"{ticker}: MPI={df['MPI'].iloc[-1]:.1%}, "
                    f"Velocity={df['MPI_Velocity'].iloc[-1]:+.3f}, "
                    f"Trend={df['MPI_Trend'].iloc[-1]}, "
                    f"RelVol={df['Relative_Volume'].iloc[-1]:.0f}%")
        
    except Exception as e:
        logger.error(f"{ticker}: Technical analysis failed: {e}")
        # Add fallback values
        df['MPI'] = 0.5
        df['MPI_Velocity'] = 0.0
        df['MPI_Trend'] = 'Calculation Error'
        df['Signal_Expansion_Buy'] = 0
        df['Signal_Strong_Buy'] = 0
        df['Signal_Exit'] = 0
        df['Relative_Volume'] = 100.0
        df['High_Rel_Volume_150'] = 0
        df['High_Rel_Volume_200'] = 0
        df['Higher_H'] = 0
        df['Higher_HL'] = 0
    
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
    """Validate the quality of the MPI-enhanced data (Buy Signal references removed)"""
    required_columns = ['Close', 'High', 'Low', 'Volume', 'IBS', 
                       'MPI', 'MPI_Velocity', 'MPI_Trend']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation_results = {
        'is_valid': len(missing_columns) == 0,
        'missing_columns': missing_columns,
        'row_count': len(df),
        'has_recent_data': len(df) > 0,
        'expansion_signals_count': int(df['Signal_Expansion_Buy'].sum()) if 'Signal_Expansion_Buy' in df.columns else 0,
        'mpi_data_available': 'MPI' in df.columns,
        'expansion_count': int((df['MPI_Trend'] == 'Expanding').sum()) if 'MPI_Trend' in df.columns else 0,
        'contraction_count': int((df['MPI_Trend'] == 'Contracting').sum()) if 'MPI_Trend' in df.columns else 0
    }
    
    return validation_results



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
        'expanding_days': int((df['MPI_Trend'] == 'Expanding').sum()) if 'MPI_Trend' in df.columns else 0,
        'flat_days': int((df['MPI_Trend'] == 'Flat').sum()) if 'MPI_Trend' in df.columns else 0,
        'contracting_days': int((df['MPI_Trend'] == 'Contracting').sum()) if 'MPI_Trend' in df.columns else 0,
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

def calculate_bullish_signal_quality(mpi: float, velocity: float, higher_h: int = 0) -> int:
    """
    Calculate BULLISH signal quality score (0-100) for long entry timing

    Args:
        mpi: MPI value (0.0 to 1.0)
        velocity: MPI velocity (-0.1 to +0.1 typically)
        higher_h: Higher high pattern flag (0 or 1)

    Returns:
        Score from 0-100 (100 = perfect early long entry)
    """
    score = 50  # Start neutral

    # PRIMARY: Velocity (most important for timing)
    if velocity > 0:       # Expanding momentum = bullish
        score += 30
    elif velocity < 0:     # Contracting momentum = bearish warning
        score -= 40
    # velocity == 0: No change to score

    # SECONDARY: MPI Level (graduated scoring for timing)
    if 0.3 <= mpi <= 0.5:      # Sweet spot - early bullish trend
        score += 20
    elif 0.5 < mpi <= 0.7:     # Good - maturing bullish trend
        score += 10
    elif mpi > 0.7:            # Caution - overbought
        score -= 20
    elif mpi < 0.3:            # Weak - risky
        score -= 10

    # CONFIRMATION: Pattern bonus
    if higher_h == 1:
        score += 10

    # Cap between 0-100
    return max(0, min(100, score))


def calculate_bearish_signal_quality(mpi: float, velocity: float, lower_l: int = 0) -> int:
    """
    Calculate BEARISH signal quality score (0-100) for short entry timing

    Args:
        mpi: MPI value (0.0 to 1.0)
        velocity: MPI velocity (-0.1 to +0.1 typically)
        lower_l: Lower low pattern flag (0 or 1)

    Returns:
        Score from 0-100 (100 = perfect early short entry)
    """
    score = 50  # Start neutral

    # PRIMARY: Velocity (most important for timing)
    if velocity < 0:       # Contracting momentum = bearish
        score += 30
    elif velocity > 0:     # Expanding momentum = bad for shorts
        score -= 40
    # velocity == 0: No change to score

    # SECONDARY: MPI Level (INVERTED for bearish - high MPI = early downtrend opportunity)
    if 0.5 <= mpi <= 0.7:      # Sweet spot - early bearish trend from high
        score += 20
    elif 0.3 <= mpi < 0.5:     # Good - maturing bearish trend
        score += 10
    elif mpi < 0.3:            # Caution - oversold, bounce risk
        score -= 20
    elif mpi > 0.7:            # Risky - still too strong for shorts
        score -= 10

    # CONFIRMATION: Pattern bonus
    if lower_l == 1:
        score += 10

    # Cap between 0-100
    return max(0, min(100, score))


def calculate_signal_direction(velocity: float) -> str:
    """
    Determine if bullish or bearish signal is relevant based on velocity

    Args:
        velocity: MPI velocity

    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    if velocity > 0:
        return 'bullish'
    elif velocity < 0:
        return 'bearish'
    else:
        return 'neutral'


def get_entry_signal_label(bullish_score: int, bearish_score: int, direction: str) -> str:
    """
    Get appropriate signal label based on direction and score

    Args:
        bullish_score: Bullish quality score (0-100)
        bearish_score: Bearish quality score (0-100)
        direction: 'bullish', 'bearish', or 'neutral'

    Returns:
        Signal label with emoji
    """
    if direction == 'bullish':
        if bullish_score >= 80:
            return "🟢 EARLY BUY"
        elif bullish_score >= 60:
            return "🟡 GOOD ENTRY"
        elif bullish_score >= 40:
            return "⚪ NEUTRAL"
        elif bullish_score >= 20:
            return "🟠 LATE/RISKY"
        else:
            return "🔴 AVOID"
    
    elif direction == 'bearish':
        if bearish_score >= 80:
            return "🔴 EARLY SHORT"
        elif bearish_score >= 60:
            return "🟠 GOOD SHORT"
        elif bearish_score >= 40:
            return "⚪ NEUTRAL"
        elif bearish_score >= 20:
            return "🟡 LATE SHORT"
        else:
            return "🟢 AVOID SHORT"
    
    else:  # neutral
        return "⚪ NEUTRAL"


# Backward compatibility alias
def calculate_mpi_signal_quality(mpi: float, velocity: float, higher_h: int = 0) -> int:
    """
    DEPRECATED: Use calculate_bullish_signal_quality() instead
    Kept for backward compatibility
    """
    return calculate_bullish_signal_quality(mpi, velocity, higher_h)

logger.info("Technical Analysis Module loaded with optimized PURE MPI EXPANSION system (Market Regime removed)")
