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
    
    # Calculate relative volume as decimal (not percentage)
    df['Relative_Volume'] = df['Volume'] / df['Volume_14D_Avg']
    
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
    
    # High activity flags for reference (existing) - decimal scale
    df['High_Rel_Volume_150'] = (df['Relative_Volume'] >= 1.5).astype(int)  # 1.5x average
    df['High_Rel_Volume_200'] = (df['Relative_Volume'] >= 2.0).astype(int)  # 2x average
    
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
    Add enhanced columns with BREAK & REVERSAL PATTERN system
    Includes MPI expansion, Relative Volume, and acceleration metrics
    PERFORMANCE OPTIMIZED: Added caching for expensive computations

    Args:
        df_daily: Raw OHLCV data from yfinance
        ticker: Stock symbol
        rolling_window: Window for moving averages (kept for backward compatibility)

    Returns:
        DataFrame with break & reversal pattern analysis columns
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

        # ===== NEW: Break & Reversal Pattern Calculations =====
        # Calculate acceleration metrics (core scoring components)
        df = calculate_ibs_acceleration(df)
        df = calculate_rrange_acceleration(df)
        df = calculate_rvol_acceleration(df)

        # Detect break events and reversal signals (with error handling)
        try:
            df = detect_break_events(df, lookback=10)
            df = detect_reversal_signals(df)
        except Exception as break_error:
            logger.warning(f"{ticker}: Break detection failed: {break_error}")
            # Add fallback columns
            df['bullish_reversal'] = 0
            df['bearish_reversal'] = 0
            df['break_high'] = 0
            df['break_low'] = 0
            df['ref_high'] = np.nan
            df['ref_low'] = np.nan
            df['purge_high'] = np.nan
            df['purge_low'] = np.nan

        # Calculate acceleration scores for latest signals
        latest_row = df.iloc[-1]
        if latest_row['bullish_reversal'] == 1:
            direction = 'bullish'
            total_score, component_scores, has_triple = calculate_acceleration_score(
                latest_row['IBS_Accel'],
                latest_row['RVol_Accel'],
                latest_row['RRange_Accel'],
                direction
            )
        elif latest_row['bearish_reversal'] == 1:
            direction = 'bearish'
            total_score, component_scores, has_triple = calculate_acceleration_score(
                latest_row['IBS_Accel'],
                latest_row['RVol_Accel'],
                latest_row['RRange_Accel'],
                direction
            )
        else:
            # No signal today - use neutral values
            direction = 'neutral'
            total_score = 0
            component_scores = {'ibs': 0, 'rvol': 0, 'rrange': 0}
            has_triple = False

        # Add signal summary columns for scanner
        df['Signal_Bias'] = '🟢 BULLISH' if latest_row['bullish_reversal'] == 1 else (
            '🔴 BEARISH' if latest_row['bearish_reversal'] == 1 else '⚪ NEUTRAL'
        )
        df['Pattern_Quality'] = get_pattern_quality_label(total_score, has_triple)
        df['Total_Score'] = total_score
        df['Triple_Confirm'] = '🔥 YES' if has_triple else '—'
        df['IBS_Score'] = component_scores['ibs']
        df['RVol_Score'] = component_scores['rvol']
        df['RRange_Score'] = component_scores['rrange']

        # Add MPI position context
        mpi_zone = get_mpi_zone(latest_row['MPI'])
        df['MPI_Zone'] = mpi_zone
        df['MPI_Position'] = get_mpi_position_label(mpi_zone, direction)

        # Add pattern details for display
        df['Ref_High'] = latest_row.get('ref_high', np.nan)
        df['Ref_Low'] = latest_row.get('ref_low', np.nan)
        df['Purge_Level'] = (
            latest_row.get('purge_high', np.nan) if latest_row['bullish_reversal'] == 1
            else latest_row.get('purge_low', np.nan)
        )
        df['Entry_Level'] = latest_row['Close'] if (
            latest_row['bullish_reversal'] == 1 or latest_row['bearish_reversal'] == 1
        ) else np.nan
        df['Bars_Since_Break'] = latest_row.get('ref_offset', np.nan)

        # Log successful calculation
        logger.debug(f"{ticker}: Break&Reversal={df['Signal_Bias'].iloc[-1]}, "
                    f"Score={df['Total_Score'].iloc[-1]}, "
                    f"MPI={df['MPI'].iloc[-1]:.1%}, "
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

        # Break & Reversal fallbacks
        df['Signal_Bias'] = '⚪ NEUTRAL'
        df['Pattern_Quality'] = '🔴 POOR'
        df['Total_Score'] = 0
        df['Triple_Confirm'] = '—'
        df['IBS_Score'] = 0
        df['RVol_Score'] = 0
        df['RRange_Score'] = 0
        df['MPI_Position'] = '❓ UNKNOWN'

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


# ===== NEW SIMPLIFIED SCORING SYSTEM (v2) =====
# Uses discrete MPI velocity as direction gate + quality scoring
#
# DESIGN PRINCIPLES:
# - MPI velocity is discrete (±0.1 or 0) due to 10-day rolling window
# - Direction determined by velocity: +0.1=bullish, -0.1=bearish, 0=IBS fallback
# - Quality score (0-100) from 3 components: Position (50pts), IBS (25pts), RelVol (25pts)
# - No complex velocity magnitude scoring - keeps it simple and robust
#
# ZONES: 1=(0.0-0.3), 2=(0.3-0.5), 3=(0.5-0.7), 4=(0.7-1.0)
# SGX VOLUME THRESHOLDS: Exceptional=1.8x, Strong=1.4x, Above Avg=1.15x

def get_mpi_zone(mpi: float) -> int:
    """
    Classify MPI into zones for position quality scoring.

    Zones represent different market conditions:
    - Zone 1 (0.0-0.3): Strong bearish territory
    - Zone 2 (0.3-0.5): Weak bearish / transition zone
    - Zone 3 (0.5-0.7): Weak bullish territory
    - Zone 4 (0.7-1.0): Strong bullish territory

    Args:
        mpi: MPI value (0.0 to 1.0) - percentage of positive days over 10-day window

    Returns:
        Zone number: 1, 2, 3, or 4
    """
    if mpi < 0.3:
        return 1
    elif mpi < 0.5:
        return 2
    elif mpi < 0.7:
        return 3
    else:
        return 4


def calculate_mpi_position_score(mpi: float, direction: str) -> int:
    """
    Calculate position quality based on MPI zone (0-50 points)

    Args:
        mpi: MPI value (0.0 to 1.0)
        direction: 'bullish' or 'bearish'

    Returns:
        Score from 0-50 points
    """
    zone = get_mpi_zone(mpi)

    if direction == 'bullish':
        # Bullish scoring: favor breakout zone 0.3-0.5
        zone_scores = {1: 10, 2: 40, 3: 30, 4: 5}
    else:  # bearish
        # Bearish scoring: favor topping zone 0.5-0.7
        zone_scores = {1: 5, 2: 30, 3: 40, 4: 10}

    return zone_scores[zone]


def calculate_ibs_confirmation_score(ibs: float, direction: str) -> int:
    """
    Calculate IBS confirmation quality (0-25 points)

    Args:
        ibs: Internal Bar Strength (0.0 to 1.0)
        direction: 'bullish' or 'bearish'

    Returns:
        Score from 0-25 points
    """
    if direction == 'bullish':
        if ibs >= 0.8:
            return 25
        elif ibs >= 0.7:
            return 20
        elif ibs >= 0.6:
            return 12
        elif ibs >= 0.5:
            return 6
        else:
            return 0
    else:  # bearish
        if ibs <= 0.2:
            return 25
        elif ibs <= 0.3:
            return 20
        elif ibs <= 0.4:
            return 12
        elif ibs <= 0.5:
            return 6
        else:
            return 0


def calculate_relvol_confirmation_score(
    relative_volume: float,
    relvol_trend: str,
    relvol_velocity: float,
    direction: str
) -> int:
    """
    Calculate RelVol confirmation quality (0-25 points)

    Args:
        relative_volume: Relative volume as percentage (e.g., 150.0 for 1.5x)
        relvol_trend: 'Building', 'Stable', or 'Fading'
        relvol_velocity: Day-over-day change in relative volume
        direction: 'bullish' or 'bearish'

    Returns:
        Score from 0-25 points (magnitude 0-10, trend 0-10, velocity 0-5)
    """
    # Component 1: Magnitude (0-10 points) - decimal scale
    if relative_volume >= 1.8:
        magnitude_score = 10
    elif relative_volume >= 1.4:
        magnitude_score = 7
    elif relative_volume >= 1.15:
        magnitude_score = 4
    else:
        magnitude_score = 0

    # Component 2: Trend (0-10 points)
    trend_scores = {'Building': 10, 'Stable': 5, 'Fading': 0}
    trend_score = trend_scores.get(relvol_trend, 0)

    # Component 3: Velocity alignment (0-5 points)
    if direction == 'bullish':
        velocity_score = 5 if relvol_velocity > 0 else 0
    else:  # bearish
        velocity_score = 5 if relvol_velocity < 0 else 0

    return magnitude_score + trend_score + velocity_score


def calculate_signal_quality_v2(
    mpi: float,
    velocity: float,
    ibs: float,
    relative_volume: float,
    relvol_trend: str,
    relvol_velocity: float
) -> Tuple[str, int, str]:
    """
    Calculate signal direction and quality score using simplified system

    Args:
        mpi: MPI value (0.0 to 1.0)
        velocity: MPI velocity (-0.1, 0, or +0.1)
        ibs: Internal Bar Strength (0.0 to 1.0)
        relative_volume: Relative volume as percentage
        relvol_trend: 'Building', 'Stable', or 'Fading'
        relvol_velocity: Day-over-day change in relative volume

    Returns:
        Tuple of (direction, quality_score, signal_label)
        - direction: 'bullish', 'bearish', or 'neutral'
        - quality_score: 0-100 points
        - signal_label: Description like "🟢 STRONG BUY"
    """
    # Determine direction from velocity (discrete: -0.1, 0, +0.1)
    if velocity > 0:
        direction = 'bullish'
    elif velocity < 0:
        direction = 'bearish'
    else:
        # Neutral velocity - use IBS as fallback
        if ibs > 0.5:
            direction = 'bullish'
        elif ibs < 0.5:
            direction = 'bearish'
        else:
            direction = 'neutral'

    if direction == 'neutral':
        return ('neutral', 50, '⚪ NEUTRAL')

    # Calculate quality components
    position_score = calculate_mpi_position_score(mpi, direction)
    ibs_score = calculate_ibs_confirmation_score(ibs, direction)
    relvol_score = calculate_relvol_confirmation_score(
        relative_volume, relvol_trend, relvol_velocity, direction
    )

    # Total quality score
    quality = position_score + ibs_score + relvol_score

    # Determine signal label
    signal_label = get_signal_label_v2(quality, direction)

    return (direction, quality, signal_label)


def get_signal_label_v2(quality: int, direction: str) -> str:
    """
    Get signal label based on quality score and direction

    Args:
        quality: Quality score (0-100)
        direction: 'bullish' or 'bearish'

    Returns:
        Signal label with emoji
    """
    if direction == 'bullish':
        if quality >= 80:
            return "🟢 STRONG BUY"
        elif quality >= 60:
            return "🟡 GOOD ENTRY"
        elif quality >= 40:
            return "⚪ NEUTRAL"
        elif quality >= 20:
            return "🟠 WEAK SETUP"
        else:
            return "🔴 AVOID"
    else:  # bearish
        if quality >= 80:
            return "🔴 STRONG SHORT"
        elif quality >= 60:
            return "🟠 GOOD SHORT"
        elif quality >= 40:
            return "⚪ NEUTRAL"
        elif quality >= 20:
            return "🟡 WEAK SHORT"
        else:
            return "🟢 AVOID SHORT"

# ===== BREAK & REVERSAL PATTERN FUNCTIONS =====
# Implementation of JY_OB_IBS_V3 logic for scanner integration

def calculate_ibs_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate IBS (Internal Bar Strength) and its 3-bar acceleration

    IBS = (close - low) / (high - low)
    IBS_Accel = IBS[t] - 2×IBS[t-1] + IBS[t-2]  (2nd derivative)

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with IBS and IBS_Accel columns added
    """
    # IBS calculation with flat candle handling
    df['IBS'] = np.where(
        df['High'] == df['Low'],
        0.5,  # Flat candle = neutral
        (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    )

    # 3-bar acceleration (2nd derivative)
    df['IBS_Accel'] = (
        df['IBS'] -
        2 * df['IBS'].shift(1) +
        df['IBS'].shift(2)
    )

    # Fill NaN values
    df['IBS_Accel'] = df['IBS_Accel'].fillna(0.0)

    return df


def calculate_rrange_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Relative Range (RRange) and its 3-bar acceleration

    RRange = ((high - low) / close) × 100  (percentage of price)
    RRange_Accel = RRange[t] - 2×RRange[t-1] + RRange[t-2]  (2nd derivative)

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with RRange and RRange_Accel columns added
    """
    # Relative Range calculation (percentage of close price)
    df['RRange'] = np.where(
        df['High'] == df['Low'],
        0.0,  # Flat candle = 0% range
        ((df['High'] - df['Low']) / df['Close']) * 100
    )

    # 3-bar acceleration (2nd derivative)
    df['RRange_Accel'] = (
        df['RRange'] -
        2 * df['RRange'].shift(1) +
        df['RRange'].shift(2)
    )

    # Fill NaN values
    df['RRange_Accel'] = df['RRange_Accel'].fillna(0.0)

    return df


def calculate_rvol_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Relative Volume (RVol) and its 3-bar acceleration

    RVol = (volume / 14-day_avg_volume) × 100
    RVol_Accel = RVol[t] - 2×RVol[t-1] + RVol[t-2]  (2nd derivative)

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with RVol_Accel column added (Relative_Volume already exists)
    """
    # 3-bar acceleration (2nd derivative) on existing Relative_Volume
    df['RVol_Accel'] = (
        df['Relative_Volume'] -
        2 * df['Relative_Volume'].shift(1) +
        df['Relative_Volume'].shift(2)
    )

    # Fill NaN values
    df['RVol_Accel'] = df['RVol_Accel'].fillna(0.0)

    return df


def find_reference_candle(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Find the first non-flat reference candle within lookback period for each row

    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to look back (default 10)

    Returns:
        DataFrame with reference data columns: [ref_offset, is_ref_flat, ref_high, ref_low]
    """
    # Initialize result columns
    df = df.copy()
    df['ref_offset'] = lookback  # Default to max lookback
    df['is_ref_flat'] = True
    df['ref_high'] = np.nan
    df['ref_low'] = np.nan

    # For each row, find the first non-flat candle within lookback
    for i, idx in enumerate(df.index):
        # Look back from current position
        for offset in range(1, lookback + 1):
            try:
                # Get the candle at this offset using integer position
                ref_pos = i - offset
                if ref_pos >= 0:
                    ref_row = df.iloc[ref_pos]
                    if ref_row['High'] != ref_row['Low']:  # Non-flat candle
                        df.at[idx, 'ref_offset'] = offset
                        df.at[idx, 'is_ref_flat'] = False
                        df.at[idx, 'ref_high'] = ref_row['High']
                        df.at[idx, 'ref_low'] = ref_row['Low']
                        break
            except (KeyError, IndexError):
                continue

    return df


def detect_break_events(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Detect break events (price breaking above/below reference levels)

    Args:
        df: DataFrame with OHLC data
        lookback: Reference candle lookback period

    Returns:
        DataFrame with break detection columns added
    """
    # Use the rewritten find_reference_candle function
    df = find_reference_candle(df, lookback)

    # Check if current candle is flat
    df['is_current_flat'] = (df['High'] == df['Low'])

    # Detect breaks
    df['break_high'] = (
        (df['High'] > df['ref_high']) &
        (~df['is_current_flat']) &
        (~df['is_ref_flat'])
    ).astype(int)

    df['break_low'] = (
        (df['Low'] < df['ref_low']) &
        (~df['is_current_flat']) &
        (~df['is_ref_flat'])
    ).astype(int)

    # Store break levels
    df['break_high_level'] = np.where(df['break_high'] == 1, df['ref_high'], np.nan)
    df['break_low_level'] = np.where(df['break_low'] == 1, df['ref_low'], np.nan)

    # Store purge candle data (the break candle)
    df['purge_high'] = np.where(df['break_high'] == 1, df['High'], np.nan)
    df['purge_low'] = np.where(df['break_low'] == 1, df['Low'], np.nan)
    # Store integer position instead of datetime index to avoid type conflicts
    df['purge_bar_pos'] = np.where(
        (df['break_high'] == 1) | (df['break_low'] == 1),
        range(len(df)),  # Integer position 0, 1, 2, ...
        np.nan
    )

    return df


def detect_reversal_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect reversal signals after break events

    Args:
        df: DataFrame with break detection data

    Returns:
        DataFrame with reversal signal columns added
    """
    # Initialize signal columns
    df['bullish_reversal'] = 0
    df['bearish_reversal'] = 0

    # Initialize active break levels (will be updated dynamically)
    df['active_break_high'] = np.nan
    df['active_break_low'] = np.nan
    df['active_purge_high_idx'] = np.nan
    df['active_purge_low_idx'] = np.nan

    # For each row, check if it's a valid reversal signal
    for i, idx in enumerate(df.index):
        # Skip if this is a flat candle
        if df.at[idx, 'is_current_flat']:
            continue

        # CRITICAL FIX: Update active levels when NEW breaks occur
        # This mimics Pine Script's array.set() behavior
        if df.at[idx, 'break_high'] == 1:
            # New high break - update active level immediately
            new_level = df.at[idx, 'break_high_level']
            df.loc[idx:, 'active_break_high'] = new_level
            df.loc[idx:, 'active_purge_high_idx'] = df.at[idx, 'purge_bar_pos']

        if df.at[idx, 'break_low'] == 1:
            # New low break - update active level immediately
            new_level = df.at[idx, 'break_low_level']
            df.loc[idx:, 'active_break_low'] = new_level
            df.loc[idx:, 'active_purge_low_idx'] = df.at[idx, 'purge_bar_pos']

        # Check bullish reversal: Close above active low break level
        if (not pd.isna(df.at[idx, 'active_break_low']) and
            not pd.isna(df.at[idx, 'active_purge_low_idx']) and
            df.at[idx, 'Close'] > df.at[idx, 'active_break_low'] and
            i != df.at[idx, 'active_purge_low_idx']):  # Not the purge candle

            df.at[idx, 'bullish_reversal'] = 1
            # Clear the active break after signal generation (use integer position slicing)
            df.iloc[i:, df.columns.get_loc('active_break_low')] = np.nan
            df.iloc[i:, df.columns.get_loc('active_purge_low_idx')] = np.nan
            # Continue loop - allow multiple break→reversal sequences

        # Check bearish reversal: Close below active high break level
        elif (not pd.isna(df.at[idx, 'active_break_high']) and
              not pd.isna(df.at[idx, 'active_purge_high_idx']) and
              df.at[idx, 'Close'] < df.at[idx, 'active_break_high'] and
              i != df.at[idx, 'active_purge_high_idx']):  # Not the purge candle

            df.at[idx, 'bearish_reversal'] = 1
            # Clear the active break after signal generation (use integer position slicing)
            df.iloc[i:, df.columns.get_loc('active_break_high')] = np.nan
            df.iloc[i:, df.columns.get_loc('active_purge_high_idx')] = np.nan
            # Continue loop - allow multiple break→reversal sequences

    # Clean up temporary columns
    df = df.drop(columns=['active_break_high', 'active_break_low',
                         'active_purge_high_idx', 'active_purge_low_idx'])

    return df


def calculate_acceleration_score(
    ibs_accel: float,
    rvol_accel: float,
    rrange_accel: float,
    direction: str,
    thresholds: dict = None
) -> Tuple[int, dict, str]:
    """
    Calculate acceleration-based score for break & reversal signals with refined gating

    NEW SYSTEM:
    - Qualification: RVol > 0.30 AND (IBS > 0.10 OR RRange > 0.10)
    - Scoring: Weighted system (RVol=45, RRange=30, IBS=25)
    - Classification: Score tiers (no arbitrary "triple confirm")

    Args:
        ibs_accel: IBS 3-bar acceleration
        rvol_accel: RVol 3-bar acceleration
        rrange_accel: RRange 3-bar acceleration
        direction: 'bullish' or 'bearish'
        thresholds: Qualification thresholds (not used for scoring)

    Returns:
        Tuple of (total_score, component_scores, pattern_quality)
        - total_score: 0-100 points (0 if not qualified)
        - component_scores: dict with individual scores
        - pattern_quality: quality tier string
    """
    # QUALIFICATION GATES (Pass/Fail)
    rvol_gate = abs(rvol_accel) >= 0.0030  # Mandatory volume threshold (decimal scale)
    momentum_gate = (abs(ibs_accel) >= 0.10) or (abs(rrange_accel) >= 0.10)  # At least one momentum type

    is_qualified = rvol_gate and momentum_gate

    if not is_qualified:
        # Not qualified - return zero score
        return 0, {'ibs': 0, 'rvol': 0, 'rrange': 0}, '🟠 NOT QUALIFIED'

    # SCORING WEIGHTS (for qualified signals only)
    RVOL_WEIGHT = 45    # Most important (volume validation)
    RRANGE_WEIGHT = 30  # Medium importance (volatility opportunity)
    IBS_WEIGHT = 25     # Confirmation role (positioning)

    # Calculate component scores (linear scaling within qualified range)
    component_scores = {}

    # RVol Score (0-45 pts) - Most important
    # NEW: Much lower multiplier for proper differentiation (decimal scale)
    rvol_score = min(RVOL_WEIGHT, max(0, (abs(rvol_accel) - 0.0030) * 10))  # 0.0030 = 0pts, 0.0480 = 45pts
    component_scores['rvol'] = int(rvol_score)

    # RRange Score (0-30 pts) - Medium importance
    # NEW: Even lower multiplier for much better differentiation
    rrange_score = min(RRANGE_WEIGHT, max(0, (abs(rrange_accel) - 0.10) * 6))  # 0.10 = 0pts, 5.10 = 30pts
    component_scores['rrange'] = int(rrange_score)

    # IBS Score (0-25 pts) - Confirmation role
    # Keep current multiplier - working well for differentiation
    ibs_score = min(IBS_WEIGHT, max(0, (abs(ibs_accel) - 0.10) * 25))  # 0.10 = 0pts, 1.10 = 25pts
    component_scores['ibs'] = int(ibs_score)

    # Total score (0-100)
    total_score = component_scores['rvol'] + component_scores['rrange'] + component_scores['ibs']

    # SCORE TIERS (Non-arbitrary classification)
    if total_score >= 90:
        pattern_quality = '🔥 EXCEPTIONAL'
    elif total_score >= 75:
        pattern_quality = '🟢 STRONG'
    elif total_score >= 60:
        pattern_quality = '🟡 GOOD'
    elif total_score >= 45:
        pattern_quality = '⚪ MODERATE'
    else:
        pattern_quality = '🟠 WEAK'

    return total_score, component_scores, pattern_quality


def get_pattern_quality_label(score: int, has_triple: bool = False) -> str:
    """
    Get pattern quality label based on acceleration score

    Args:
        score: Total acceleration score (0-100)
        has_triple: Whether all 3 accelerations met thresholds

    Returns:
        Quality label with emoji
    """
    if has_triple:
        return "🔥 EXCEPTIONAL"
    elif score >= 70:
        return "🟢 STRONG"
    elif score >= 55:
        return "🟡 GOOD"
    elif score >= 40:
        return "⚪ MODERATE"
    elif score >= 25:
        return "🟠 WEAK"
    else:
        return "🔴 POOR"


def get_mpi_position_label(mpi_zone: int, direction: str) -> str:
    """
    Get MPI position context label

    Args:
        mpi_zone: MPI zone (1-4)
        direction: 'bullish' or 'bearish'

    Returns:
        Position label with emoji
    """
    if direction == 'bullish':
        zone_labels = {
            1: "❌ TOO WEAK",
            2: "📍 EARLY STAGE",
            3: "⚠️ MID STAGE",
            4: "🚨 LATE STAGE"
        }
    else:  # bearish
        zone_labels = {
            1: "🚨 LATE SHORT",
            2: "⚠️ MID SHORT",
            3: "📍 EARLY SHORT",
            4: "❌ TOO STRONG"
        }

    return zone_labels.get(mpi_zone, "❓ UNKNOWN")


logger.info("Technical Analysis Module loaded with optimized PURE MPI EXPANSION system (Market Regime removed)")
