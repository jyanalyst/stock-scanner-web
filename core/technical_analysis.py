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
def add_enhanced_columns(df_daily: pd.DataFrame, ticker: str, rolling_window: int = 20,
                        # NEW: Confirmation filter parameters
                        use_ibs: bool = False, use_rvol: bool = False, use_rrange: bool = False,
                        confirmation_logic: str = "OR",
                        ibs_threshold: float = 0.10, rvol_threshold: float = 0.20, rrange_threshold: float = 0.30) -> pd.DataFrame:
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

        # Add placeholder metrics for Phase 0 compatibility (Flow_Velocity=0 weight, Volume_Conviction=minimal weight)
        df['Flow_Velocity'] = 0.0  # Neutral flow (zero weight in Phase 0)
        df['Volume_Conviction'] = 1.0  # Neutral conviction (minimal weight in Phase 0)

        # Calculate percentile ranks for scoring (Phase 2 integration)
        df = calculate_percentile_ranks(df)

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
            # Use percentile-based scoring with Phase 0 weights
            phase0_weights = {
                'IBS_Accel': 1.5,
                'RVol_Accel': 86.5,
                'RRange_Accel': 11.3,
                'Flow_Velocity': 0.0,  # Monitor this
                'Volume_Conviction': 0.7
            }
            total_score, component_scores, pattern_quality = calculate_acceleration_score_v3(
                direction=direction,
                ibs_pct=latest_row['IBS_Bullish_Pct'],
                rvol_pct=latest_row['RVol_Accel_Percentile'],
                rrange_pct=latest_row['RRange_Accel_Percentile'],
                flow_pct=latest_row['Flow_Bullish_Pct'],
                conviction_pct=latest_row['Volume_Conviction_Percentile'],
                weights=phase0_weights
            )
            # LEGACY: Keep old confirmation logic for backward compatibility
            is_confirmed = True  # Percentile system doesn't use gates
        elif latest_row['bearish_reversal'] == 1:
            direction = 'bearish'
            # Use percentile-based scoring with Phase 0 weights
            phase0_weights = {
                'IBS_Accel': 1.5,
                'RVol_Accel': 86.5,
                'RRange_Accel': 11.3,
                'Flow_Velocity': 0.0,  # Monitor this
                'Volume_Conviction': 0.7
            }
            total_score, component_scores, pattern_quality = calculate_acceleration_score_v3(
                direction=direction,
                ibs_pct=latest_row['IBS_Bearish_Pct'],
                rvol_pct=latest_row['RVol_Accel_Percentile'],
                rrange_pct=latest_row['RRange_Accel_Percentile'],
                flow_pct=latest_row['Flow_Bearish_Pct'],
                conviction_pct=latest_row['Volume_Conviction_Percentile'],
                weights=phase0_weights
            )
            # LEGACY: Keep old confirmation logic for backward compatibility
            is_confirmed = True  # Percentile system doesn't use gates
        else:
            # No signal today - use neutral values
            direction = 'neutral'
            total_score = 0
            is_confirmed = True  # No signal = no filtering needed
            component_scores = {'ibs': 0, 'rvol': 0, 'rrange': 0, 'flow': 0, 'conviction': 0}
            pattern_quality = '⚪ NEUTRAL'

        # Add signal summary columns for scanner
        df['Signal_Bias'] = '🟢 BULLISH' if latest_row['bullish_reversal'] == 1 else (
            '🔴 BEARISH' if latest_row['bearish_reversal'] == 1 else '⚪ NEUTRAL'
        )
        df['Pattern_Quality'] = pattern_quality  # Use the pattern_quality from calculate_acceleration_score tuple
        df['Total_Score'] = total_score
        df['Triple_Confirm'] = '🔥 YES' if total_score >= 90 else '—'  # Triple confirm based on exceptional score
        df['IBS_Score'] = component_scores['ibs']
        df['RVol_Score'] = component_scores['rvol']
        df['RRange_Score'] = component_scores['rrange']

        # ===== CONFIRMATION FILTERING =====
        # Store confirmation status for filtering
        df['Is_Confirmed'] = is_confirmed

        # Add MPI position context
        mpi_zone = get_mpi_zone(latest_row['MPI'])
        df['MPI_Zone'] = mpi_zone
        df['MPI_Position'] = get_mpi_position_label(mpi_zone, direction)

        # Add pattern details for display
        df['Ref_High'] = latest_row.get('ref_high', np.nan)
        df['Ref_Low'] = latest_row.get('ref_low', np.nan)
        df['Purge_Level'] = latest_row.get('reversal_purge_level', np.nan)
        df['Entry_Level'] = (
            latest_row['High'] if latest_row['bullish_reversal'] == 1
            else latest_row['Low'] if latest_row['bearish_reversal'] == 1
            else np.nan
        )
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


def calculate_percentile_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentile ranks for acceleration metrics used in scoring
    Creates direction-specific percentiles for IBS and Flow metrics

    Args:
        df: DataFrame with acceleration metrics (IBS_Accel, RVol_Accel, etc.)

    Returns:
        DataFrame with percentile rank columns added
    """
    df = df.copy()

    # Calculate basic percentile ranks (0-1 scale)
    df['IBS_Accel_Percentile'] = df['IBS_Accel'].rolling(window=50, min_periods=20).rank(pct=True)
    df['RVol_Accel_Percentile'] = df['RVol_Accel'].rolling(window=50, min_periods=20).rank(pct=True)
    df['RRange_Accel_Percentile'] = df['RRange_Accel'].rolling(window=50, min_periods=20).rank(pct=True)
    df['Flow_Velocity_Percentile'] = df['Flow_Velocity'].rolling(window=50, min_periods=20).rank(pct=True)
    df['Volume_Conviction_Percentile'] = df['Volume_Conviction'].rolling(window=50, min_periods=20).rank(pct=True)

    # Direction-specific IBS percentiles
    # When IBS_Accel > 0: bullish_pct = percentile, bearish_pct = 0
    # When IBS_Accel < 0: bearish_pct = percentile, bullish_pct = 0
    # When IBS_Accel = 0: both = 0.5 (neutral)
    df['IBS_Bullish_Pct'] = np.where(
        df['IBS_Accel'] > 0,
        df['IBS_Accel_Percentile'],
        np.where(df['IBS_Accel'] == 0, 0.5, 0.0)
    )
    df['IBS_Bearish_Pct'] = np.where(
        df['IBS_Accel'] < 0,
        df['IBS_Accel_Percentile'],
        np.where(df['IBS_Accel'] == 0, 0.5, 0.0)
    )

    # Direction-specific Flow percentiles
    # Same logic as IBS but for Flow_Velocity
    df['Flow_Bullish_Pct'] = np.where(
        df['Flow_Velocity'] > 0,
        df['Flow_Velocity_Percentile'],
        np.where(df['Flow_Velocity'] == 0, 0.5, 0.0)
    )
    df['Flow_Bearish_Pct'] = np.where(
        df['Flow_Velocity'] < 0,
        df['Flow_Velocity_Percentile'],
        np.where(df['Flow_Velocity'] == 0, 0.5, 0.0)
    )

    # Fill NaN values with neutral values
    percentile_cols = [
        'IBS_Accel_Percentile', 'IBS_Bullish_Pct', 'IBS_Bearish_Pct',
        'RVol_Accel_Percentile', 'RRange_Accel_Percentile',
        'Flow_Velocity_Percentile', 'Flow_Bullish_Pct', 'Flow_Bearish_Pct',
        'Volume_Conviction_Percentile'
    ]

    for col in percentile_cols:
        df[col] = df[col].fillna(0.5)  # Neutral percentile

    return df


def calculate_acceleration_score_v3(
    direction: str,
    ibs_pct: float,
    rvol_pct: float,
    rrange_pct: float,
    flow_pct: float,
    conviction_pct: float,
    weights: dict = None
) -> Tuple[int, dict, str]:
    """
    Pure percentile-based scoring with data-driven weights
    NO GATES, NO THRESHOLDS - Every signal gets scored

    Args:
        direction: 'bullish' or 'bearish'
        ibs_pct: IBS percentile rank (0.0 to 1.0)
        rvol_pct: RVol percentile rank (0.0 to 1.0)
        rrange_pct: RRange percentile rank (0.0 to 1.0)
        flow_pct: Flow percentile rank (0.0 to 1.0)
        conviction_pct: Conviction percentile rank (0.0 to 1.0)
        weights: Data-driven weights from Phase 0 analysis

    Returns:
        Tuple of (total_score, component_scores, quality_label)
        - total_score: 0-100 points
        - component_scores: dict with individual component scores
        - quality_label: descriptive label
    """
    # Use data-driven weights from Phase 0 analysis
    if weights is None:
        # Fallback to balanced weights if not provided
        weights = {
            'IBS_Accel': 1.5,
            'RVol_Accel': 86.5,
            'RRange_Accel': 11.3,
            'Flow_Velocity': 0.0,  # Monitor this
            'Volume_Conviction': 0.7
        }

    # Direct percentile to points conversion
    component_scores = {
        'ibs': int(ibs_pct * weights['IBS_Accel']),
        'rvol': int(rvol_pct * weights['RVol_Accel']),
        'rrange': int(rrange_pct * weights['RRange_Accel']),
        'flow': int(flow_pct * weights['Flow_Velocity']),
        'conviction': int(conviction_pct * weights['Volume_Conviction'])
    }

    # Total score (0-100)
    total_score = sum(component_scores.values())

    # Quality label based on percentile ranking (not arbitrary tiers)
    if total_score >= 80:
        quality_label = '🔥 TOP 20%'
    elif total_score >= 60:
        quality_label = '🟢 TOP 40%'
    elif total_score >= 40:
        quality_label = '🟡 AVERAGE'
    else:
        quality_label = '🟠 BELOW AVERAGE'

    return total_score, component_scores, quality_label


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
            df.at[idx, 'reversal_purge_level'] = df.at[idx, 'active_break_low']  # Store purge level
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
            df.at[idx, 'reversal_purge_level'] = df.at[idx, 'active_break_high']  # Store purge level
            # Clear the active break after signal generation (use integer position slicing)
            df.iloc[i:, df.columns.get_loc('active_break_high')] = np.nan
            df.iloc[i:, df.columns.get_loc('active_purge_high_idx')] = np.nan
            # Continue loop - allow multiple break→reversal sequences

    # Clean up temporary columns
    df = df.drop(columns=['active_break_high', 'active_break_low',
                         'active_purge_high_idx', 'active_purge_low_idx'])

    return df


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
