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

# ===== FEATURE CONFIGURATION SYSTEM =====
# Controls which features are calculated in production vs archived for Phase 2 testing
# Set 'calculate_unused_features' = True to re-enable all archived features for testing

FEATURE_CONFIG = {
    # Main toggle for Phase 2 testing
    'calculate_unused_features': False,  # Set True to re-enable all archived features

    # Feature categorization for easy reference
    'features_in_production': [
        # Scoring Features (Category A)
        'Flow_Velocity_Rank', 'Flow_Rank', 'Flow_Percentile', 'Volume_Conviction',
        'MPI_Percentile', 'IBS_Percentile', 'VPI_Percentile', 'Signal_Bias',

        # Display Features (Category B) - Core UI columns
        'Signal_Bias', 'Signal_State', 'Conviction_Level', 'MPI_Position',
        'Is_Triple_Aligned', 'Is_Divergent', 'Is_Accumulation',
        'Ref_High', 'Ref_Low', 'Purge_Level', 'Entry_Level', 'Bars_Since_Break',
        'MPI_Percentile', 'IBS_Percentile', 'VPI_Percentile',
        'IBS_Accel', 'VPI_Accel',
        'Ticker', 'Name', 'Analysis_Date', 'Close', 'High', 'Low',
        'IBS', 'Higher_H', 'Higher_HL', 'Lower_L', 'Lower_HL', 'HL_Pattern',
        'Relative_Volume', 'VPI_Velocity',
        'MPI', 'MPI_Velocity',
        'Daily_Flow', 'Flow_10D', 'Flow_Velocity',
        'Flow_Percentile', 'Flow_Rank', 'Flow_Velocity_Rank',
        'Volume_Conviction', 'Avg_Vol_Up_10D',
        'Price_Decimals',

        # Ranking columns (added in logic.py)
        'Signal_Score', 'Trade_Rank', 'Suggested_Risk', 'Quality_Flag',

        # Analyst/Earnings reports (added in logic.py)
        'Sentiment_Display', 'Report_Date_Display', 'Report_Count_Display',
        'Earnings_Period', 'Guidance_Display', 'Rev_YoY_Display', 'EPS_DPU_Display', 'Earnings_Reaction',

        # ML predictions (added in logic.py)
        'ML_Signal', 'ML_Confidence'
    ],

    'features_archived': [
        # Archived Features (Category D) - Removed from production
        'RVol_Accel', 'RRange_Accel', 'VW_Range_Velocity', 'VW_Range_Percentile', 'Rel_Range_Signal',
        'RelVol_Velocity', 'RelVol_Trend', 'VPI_Velocity', 'Volume_Conviction_Rank',
        'Flow_Velocity_Percentile', 'Conviction_Velocity', 'Divergence_Gap', 'Divergence_Severity',
        'Price_Percentile', 'Flow_Regime', 'MPI_Trend', 'MPI_Zone'
    ],

    # Dependency tracking for safe removal
    'calculation_dependencies': {
        'MPI_Percentile': ['MPI', 'MPI_Velocity'],
        'IBS_Percentile': ['IBS', 'IBS_Velocity', 'IBS_Accel'],
        'VPI_Percentile': ['VPI_Accel'],
        'Flow_Percentile': ['Flow_10D'],
        'Flow_Velocity_Rank': ['Flow_Velocity'],
        'Flow_Rank': ['Flow_10D'],
        'Volume_Conviction': ['Avg_Vol_Up_10D'],
        'HL_Pattern': ['Higher_H', 'Higher_HL', 'Lower_L', 'Lower_HL']
    }
}

def calculate_mpi_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MPI (Market Positivity Index) with Exponential Weighting

    Core Concept: Multi-day trend strength through weighted positive day participation
    - Weighted_MPI: Exponentially weighted sum of positive days (0-1 scale)
    - MPI_Velocity: Day-over-day change in Weighted MPI
    - MPI_Percentile: 100-day rolling percentile of velocity

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with MPI columns added
    """
    # Step 1: Identify positive days
    returns = df['Close'].pct_change()
    positive_days = (returns > 0).astype(float)

    # Step 2: Apply exponential weighting (10-day window)
    # Weights: 1, 2, 3, ..., 10 (Sum = 55)
    weights = np.arange(1, 11)
    sum_weights = weights.sum() # 55

    def weighted_sum(x):
        if len(x) < 10:
            return np.nan
        return np.dot(x, weights) / sum_weights

    # Step 3: Calculate Weighted MPI
    # Use rolling apply with raw=True for performance
    df['MPI'] = positive_days.rolling(10).apply(weighted_sum, raw=True)

    # Step 4: Calculate Velocity (1st Derivative)
    df['MPI_Velocity'] = df['MPI'] - df['MPI'].shift(1)

    # Step 5: Percentile Classification (100-day rolling of Velocity)
    # CRITICAL FIX: Use .shift(1) to avoid lookahead bias (don't include current day in ranking)
    df['MPI_Percentile'] = df['MPI_Velocity'].shift(1).rolling(100, min_periods=20).rank(pct=True) * 100

    # Fill NaN values
    df['MPI'] = df['MPI'].fillna(0.5)  # Neutral default
    df['MPI_Velocity'] = df['MPI_Velocity'].fillna(0.0)
    df['MPI_Percentile'] = df['MPI_Percentile'].fillna(50.0) # Neutral default

    # ===== ARCHIVED FEATURES: MPI_Trend and MPI_Zone =====
    # Removed: 2025-12-19 - Trend/zone classifications never used in scoring, display, or dependencies
    # Archive: data/feature_library/unused_features_library.json
    # To re-enable: Set FEATURE_CONFIG['calculate_unused_features'] = True
    if FEATURE_CONFIG['calculate_unused_features']:
        # Calculate trend classification
        df['MPI_Trend'] = np.where(
            df['MPI_Velocity'] > 0.1, 'Expanding',
            np.where(df['MPI_Velocity'] < -0.1, 'Contracting', 'Flat')
        )

        # Calculate zone classification
        df['MPI_Zone'] = df['MPI_Percentile'].apply(get_mpi_zone)
    else:
        # PRODUCTION MODE: Add placeholder columns
        df['MPI_Trend'] = 'Flat'
        df['MPI_Zone'] = 3

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

    # Volume-weighted range (use Relative_Volume instead of Volume_Normalized)
    df['Volume_Weighted_Range'] = df['Daily_Rel_Range'] * df['Relative_Volume']

    # ===== ARCHIVED FEATURES: VW_Range_Percentile, VW_Range_Velocity, Rel_Range_Signal =====
    # Removed: 2025-12-19 - Volatility metrics never used in scoring, display, or dependencies
    # Archive: data/feature_library/unused_features_library.json
    # To re-enable: Set FEATURE_CONFIG['calculate_unused_features'] = True
    if FEATURE_CONFIG['calculate_unused_features']:
        # Volume-weighted range percentile (for display/filtering only) - RAW method
        df['VW_Range_Percentile'] = df['Volume_Weighted_Range']

        # VW_Range_Velocity calculated from RAW Volume_Weighted_Range (not percentile)
        df['VW_Range_Velocity'] = df['Volume_Weighted_Range'] - df['Volume_Weighted_Range'].shift(1)

        # Range expansion signal based on percentile thresholds
        df['Rel_Range_Signal'] = np.where(
            df['VW_Range_Percentile'] > df['VW_Range_Percentile'].shift(1), 1, 0
        )
    else:
        # PRODUCTION MODE: Add placeholder columns
        df['VW_Range_Percentile'] = 50.0
        df['VW_Range_Velocity'] = 0.0
        df['Rel_Range_Signal'] = 0

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

    # Fill NaN values
    df['Relative_Volume'] = df['Relative_Volume'].fillna(100.0)

    # High activity flags for reference (existing) - decimal scale
    df['High_Rel_Volume_150'] = (df['Relative_Volume'] >= 1.5).astype(int)  # 1.5x average
    df['High_Rel_Volume_200'] = (df['Relative_Volume'] >= 2.0).astype(int)  # 2x average

    # ===== ARCHIVED FEATURES: RelVol_Velocity and RelVol_Trend =====
    # Removed: 2025-12-19 - Volume metrics never used in scoring, display, or dependencies
    # Archive: data/feature_library/unused_features_library.json
    # To re-enable: Set FEATURE_CONFIG['calculate_unused_features'] = True
    if FEATURE_CONFIG['calculate_unused_features']:
        # RelVol_Velocity: Day-over-day change in Relative Volume
        df['RelVol_Velocity'] = df['Relative_Volume'] - df['Relative_Volume'].shift(1)
        df['RelVol_Velocity'] = df['RelVol_Velocity'].fillna(0.0)

        # RelVol_Trend: Trend classification (Building/Stable/Fading)
        df['RelVol_Trend'] = np.where(
            df['Relative_Volume'] > 1.2, 'Building',
            np.where(df['Relative_Volume'] < 0.8, 'Fading', 'Stable')
        )
    else:
        # PRODUCTION MODE: Add placeholder columns
        df['RelVol_Velocity'] = 0.0
        df['RelVol_Trend'] = 'Stable'

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
        # IMPORTANT: calculate_relative_volume MUST come before calculate_technical_indicators
        # because calculate_technical_indicators uses Relative_Volume
        df = calculate_relative_volume(df)
        df = calculate_technical_indicators(df)
        df = calculate_mpi_expansion(df)

        # ===== NEW: Break & Reversal Pattern Calculations =====
        # Calculate acceleration metrics (core scoring components)
        df = calculate_ibs_acceleration(df)

        # Calculate VPI System (New Phase 1)
        df = calculate_vpi_system(df)

        # ===== PHASE 1: INSTITUTIONAL FLOW ANALYSIS =====
        # Calculate real institutional flow metrics (replacing Phase 0 placeholders)
        df = calculate_institutional_flow(df)
        df = classify_flow_regime(df)
        df = calculate_volume_conviction(df)

        # ===== CONDITIONAL: ARCHIVED FEATURES (PHASE 2 TESTING) =====
        # Only calculate archived features if FEATURE_CONFIG allows it
        if FEATURE_CONFIG['calculate_unused_features']:
            logger.debug(f"{ticker}: Calculating archived features for Phase 2 testing")

            # Archived acceleration metrics
            df = calculate_rrange_acceleration(df)  # ARCHIVED: RRange_Accel
            df = calculate_rvol_acceleration(df)    # ARCHIVED: RVol_Accel

            # Archived divergence detection
            from core.divergence_analysis import DivergenceDetector
            divergence_detector = DivergenceDetector(
                lookback_percentile=252,
                lookback_slope=20,
                min_slope_threshold=0.05,
                misalignment_threshold=20.0
            )
            df = divergence_detector.calculate_all_divergences(df)  # ARCHIVED: Divergence_Gap, Divergence_Severity, Price_Percentile
        else:
            # PRODUCTION MODE: Skip archived calculations and add placeholder columns
            logger.debug(f"{ticker}: Skipping archived features (production mode)")

            # Add placeholder columns for archived features to prevent KeyErrors
            archived_features = FEATURE_CONFIG['features_archived']

            for feature in archived_features:
                if feature not in df.columns:
                    # Set appropriate default values based on feature type
                    if 'Percentile' in feature or 'Pct' in feature:
                        df[feature] = 50.0  # Neutral percentile
                    elif 'Velocity' in feature or 'Accel' in feature:
                        df[feature] = 0.0   # Neutral velocity/acceleration
                    elif 'Rank' in feature:
                        df[feature] = 50.0  # Neutral rank
                    elif 'Regime' in feature:
                        df[feature] = 'Neutral'  # Neutral regime
                    elif 'Trend' in feature:
                        df[feature] = 'Flat'  # Neutral trend
                    elif 'Zone' in feature:
                        df[feature] = 3      # Neutral zone
                    elif 'Gap' in feature or 'Severity' in feature:
                        df[feature] = 0.0   # Neutral divergence
                    elif 'Signal' in feature:
                        df[feature] = 0     # No signal
                    else:
                        df[feature] = 0.0   # Default numeric

            # Ensure divergence columns exist (they're added by DivergenceDetector)
            divergence_features = ['Divergence_Gap', 'Divergence_Severity', 'Price_Percentile']
            for feature in divergence_features:
                if feature not in df.columns:
                    df[feature] = 0.0 if 'Gap' in feature or 'Severity' in feature else 50.0

        # Note: calculate_percentile_ranks is deprecated as percentiles are now
        # calculated within each indicator function (MPI, IBS, VPI)
        # df = calculate_percentile_ranks(df)

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

        # Determine signal direction from break & reversal patterns
        latest_row = df.iloc[-1]
        if latest_row['bullish_reversal'] == 1:
            direction = 'bullish'
        elif latest_row['bearish_reversal'] == 1:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Add signal summary columns for scanner
        df['Signal_Bias'] = '🟢 BULLISH' if latest_row['bullish_reversal'] == 1 else (
            '🔴 BEARISH' if latest_row['bearish_reversal'] == 1 else '⚪ NEUTRAL'
        )

        # ===== PHASE 2: CONFLUENCE & SIGNAL LOGIC =====
        # Implement the Three-Indicator System Logic
        
        # Get current percentile values
        mpi_pct = latest_row.get('MPI_Percentile', 50.0)
        ibs_pct = latest_row.get('IBS_Percentile', 50.0)
        vpi_pct = latest_row.get('VPI_Percentile', 50.0)
        
        # 1. Triple Alignment (High Conviction)
        # MPI > 60 (Trend) + IBS > 80 (Momentum) + VPI > 70 (Volume)
        is_triple_bullish = (mpi_pct > 60) and (ibs_pct > 80) and (vpi_pct > 70)
        
        # 2. Divergence Detection (Warning)
        # Price Strong (MPI>60 or IBS>70) BUT Volume Weak (VPI<40)
        is_divergent = (mpi_pct > 60 or ibs_pct > 70) and (vpi_pct < 40)
        
        # 3. Accumulation (Early Entry)
        # Price Weak/Neutral (MPI<50) BUT Volume Strong (VPI>80)
        is_accumulation = (mpi_pct < 50) and (vpi_pct > 80)
        
        # Determine Signal State and Conviction
        signal_state = "⚪ Neutral"
        conviction_level = "Low"
        
        if is_triple_bullish:
            signal_state = "🔥 Triple Bullish"
            conviction_level = "High"
        elif is_divergent:
            signal_state = "⚠️ Divergence"
            conviction_level = "Warning"
        elif is_accumulation:
            signal_state = "👀 Accumulation"
            conviction_level = "Watch"
        elif direction == 'bullish':
            # Check for moderate conviction (2 of 3 aligned)
            aligned_count = sum([mpi_pct > 55, ibs_pct > 70, vpi_pct > 65])
            if aligned_count >= 2:
                signal_state = "✅ Bullish"
                conviction_level = "Moderate"
            else:
                signal_state = "⚪ Weak Bullish"
                conviction_level = "Low"
        elif direction == 'bearish':
            # Check for moderate conviction (2 of 3 aligned)
            aligned_count = sum([mpi_pct < 45, ibs_pct < 30, vpi_pct < 35])
            if aligned_count >= 2:
                signal_state = "🔴 Bearish"
                conviction_level = "Moderate"
            else:
                signal_state = "⚪ Weak Bearish"
                conviction_level = "Low"

        # Add new columns
        df['Signal_State'] = signal_state
        df['Conviction_Level'] = conviction_level
        df['Is_Triple_Aligned'] = is_triple_bullish
        df['Is_Divergent'] = is_divergent
        df['Is_Accumulation'] = is_accumulation

        # ⭐ NEW: Add time-decay features for signal freshness
        df = add_time_decay_features(df)

        # Log successful calculation
        logger.debug(f"{ticker}: Signal={df['Signal_Bias'].iloc[-1]}, "
                    f"State={df['Signal_State'].iloc[-1]}, "
                    f"MPI={df['MPI'].iloc[-1]:.1%}, "
                    f"RelVol={df['Relative_Volume'].iloc[-1]:.0f}%")

    except Exception as e:
        logger.error(f"{ticker}: Technical analysis failed: {e}")
        # Add fallback values
        df['MPI'] = 0.5
        df['MPI_Velocity'] = 0.0
        df['MPI_Percentile'] = 50.0
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
        df['Signal_State'] = '⚪ Neutral'
        df['Conviction_Level'] = 'Low'
        df['Is_Triple_Aligned'] = False
        df['Is_Divergent'] = False
        df['Is_Accumulation'] = False
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

def get_mpi_zone(mpi_percentile: float) -> int:
    """
    Classify MPI into zones based on Percentile (New System).

    Zones:
    - Zone 1 (<20): Strong Contraction
    - Zone 2 (20-40): Contraction
    - Zone 3 (40-60): Neutral
    - Zone 4 (60-80): Expansion
    - Zone 5 (>80): Strong Expansion

    Args:
        mpi_percentile: MPI Velocity Percentile (0-100)

    Returns:
        Zone number: 1-5
    """
    if mpi_percentile < 20:
        return 1
    elif mpi_percentile < 40:
        return 2
    elif mpi_percentile < 60:
        return 3
    elif mpi_percentile < 80:
        return 4
    else:
        return 5


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

    # Calculate basic percentile ranks (0-1 scale) - removed time-series percentiles

    # Direction-specific percentiles calculated directly from raw values (no time-series)
    # IBS direction percentiles
    df['IBS_Bullish_Pct'] = np.where(
        df['IBS_Accel'] > 0,
        0.7,  # High percentile for positive acceleration
        np.where(df['IBS_Accel'] == 0, 0.5, 0.3)  # Neutral for zero, low for negative
    )
    df['IBS_Bearish_Pct'] = np.where(
        df['IBS_Accel'] < 0,
        0.7,  # High percentile for negative acceleration
        np.where(df['IBS_Accel'] == 0, 0.5, 0.3)  # Neutral for zero, low for positive
    )

    # Flow direction percentiles
    df['Flow_Bullish_Pct'] = np.where(
        df['Flow_Velocity'] > 0,
        0.7,  # High percentile for positive flow
        np.where(df['Flow_Velocity'] == 0, 0.5, 0.3)  # Neutral for zero, low for negative
    )
    df['Flow_Bearish_Pct'] = np.where(
        df['Flow_Velocity'] < 0,
        0.7,  # High percentile for negative flow
        np.where(df['Flow_Velocity'] == 0, 0.5, 0.3)  # Neutral for zero, low for positive
    )

    # Fill NaN values with neutral values (only for columns that exist)
    percentile_cols = ['IBS_Bullish_Pct', 'IBS_Bearish_Pct', 'Flow_Bullish_Pct', 'Flow_Bearish_Pct']

    for col in percentile_cols:
        if col in df.columns:
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
    Calculate IBS (Internal Bar Strength) with EMA Smoothing and Acceleration
    
    Core Concept: Intra-day price position and momentum acceleration
    - Raw_IBS: Position of close within high-low range
    - IBS_EMA: 14-period EMA of Raw IBS
    - IBS_Velocity: 1st derivative of EMA
    - IBS_Accel: 2nd derivative of EMA
    - IBS_Percentile: 100-day rolling percentile of Acceleration
    
    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with IBS columns added
    """
    # Step 1: Calculate Raw IBS
    df['Raw_IBS'] = np.where(
        df['High'] == df['Low'],
        0.5,  # Flat candle = neutral
        (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    )
    
    # Step 2: Smooth with EMA (14 periods)
    df['IBS'] = df['Raw_IBS'].ewm(span=14, adjust=False).mean()
    
    # Step 3: Calculate Velocity (1st Derivative)
    df['IBS_Velocity'] = df['IBS'] - df['IBS'].shift(1)
    
    # Step 4: Calculate Acceleration (2nd Derivative)
    # Acceleration = Velocity[today] - Velocity[yesterday]
    df['IBS_Accel'] = df['IBS_Velocity'] - df['IBS_Velocity'].shift(1)
    
    # Step 5: Percentile Classification (100-day rolling of Acceleration)
    # CRITICAL FIX: Use .shift(1) to avoid lookahead bias (don't include current day in ranking)
    df['IBS_Percentile'] = df['IBS_Accel'].shift(1).rolling(100, min_periods=20).rank(pct=True) * 100
    
    # Fill NaN values
    df['IBS'] = df['IBS'].fillna(0.5)
    df['IBS_Velocity'] = df['IBS_Velocity'].fillna(0.0)
    df['IBS_Accel'] = df['IBS_Accel'].fillna(0.0)
    df['IBS_Percentile'] = df['IBS_Percentile'].fillna(50.0)

    return df

def calculate_vpi_system(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VPI (Volume Positivity Index)
    
    Core Concept: Pure volume momentum and acceleration
    - Volume_Ratio: Volume / 20-day SMA
    - Volume_Percentile: 100-day rank of Volume Ratio
    - VPI_Velocity: 1st derivative of Volume Percentile
    - VPI_Accel: 2nd derivative of Volume Percentile
    - VPI_Percentile: 100-day rolling percentile of VPI_Accel
    
    Args:
        df: DataFrame with Volume data
        
    Returns:
        DataFrame with VPI columns added
    """
    # Step 1: Normalize Volume (Volume Ratio)
    vol_sma = df['Volume'].rolling(20).mean()
    # Avoid division by zero
    df['Volume_Ratio'] = np.where(vol_sma > 0, df['Volume'] / vol_sma, 0)
    
    # Step 2: Calculate Volume Percentile (Rank of Ratio over 100 days)
    # This normalizes across different stocks and periods
    df['VPI_Raw_Pct'] = df['Volume_Ratio'].rolling(100, min_periods=20).rank(pct=True) * 100
    df['VPI_Raw_Pct'] = df['VPI_Raw_Pct'].fillna(50.0)
    
    # Step 3: Calculate Velocity (1st Derivative)
    df['VPI_Velocity'] = df['VPI_Raw_Pct'] - df['VPI_Raw_Pct'].shift(1)
    
    # Step 4: Calculate Acceleration (2nd Derivative)
    df['VPI_Accel'] = df['VPI_Velocity'] - df['VPI_Velocity'].shift(1)
    
    # Step 5: Percentile Classification (100-day rolling of Acceleration)
    # This is the final "VPI" metric used for scoring/signals
    # CRITICAL FIX: Use .shift(1) to avoid lookahead bias (don't include current day in ranking)
    df['VPI_Percentile'] = df['VPI_Accel'].shift(1).rolling(100, min_periods=20).rank(pct=True) * 100
    
    # Fill NaN values
    df['VPI_Velocity'] = df['VPI_Velocity'].fillna(0.0)
    df['VPI_Accel'] = df['VPI_Accel'].fillna(0.0)
    df['VPI_Percentile'] = df['VPI_Percentile'].fillna(50.0)
    
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
        is_current_flat = df.at[idx, 'is_current_flat']
        if is_current_flat:
            continue

        # CRITICAL FIX: Update active levels when NEW breaks occur
        # This mimics Pine Script's array.set() behavior
        break_high = df.at[idx, 'break_high']
        break_low = df.at[idx, 'break_low']

        if break_high == 1:
            # New high break - update active level immediately
            new_level = df.at[idx, 'break_high_level']
            df.loc[idx:, 'active_break_high'] = new_level
            df.loc[idx:, 'active_purge_high_idx'] = df.at[idx, 'purge_bar_pos']

        if break_low == 1:
            # New low break - update active level immediately
            new_level = df.at[idx, 'break_low_level']
            df.loc[idx:, 'active_break_low'] = new_level
            df.loc[idx:, 'active_purge_low_idx'] = df.at[idx, 'purge_bar_pos']

        # Get values for reversal checks (ensure scalar values)
        active_break_low = df.at[idx, 'active_break_low']
        active_purge_low_idx = df.at[idx, 'active_purge_low_idx']
        active_break_high = df.at[idx, 'active_break_high']
        active_purge_high_idx = df.at[idx, 'active_purge_high_idx']
        close_price = df.at[idx, 'Close']

        # Check bullish reversal: Close above active low break level
        if (not pd.isna(active_break_low) and
            not pd.isna(active_purge_low_idx) and
            close_price > active_break_low and
            i != active_purge_low_idx):  # Not the purge candle

            df.at[idx, 'bullish_reversal'] = 1
            df.at[idx, 'reversal_purge_level'] = active_break_low  # Store purge level
            # Clear the active break after signal generation (use integer position slicing)
            df.iloc[i:, df.columns.get_loc('active_break_low')] = np.nan
            df.iloc[i:, df.columns.get_loc('active_purge_low_idx')] = np.nan
            # Continue loop - allow multiple break→reversal sequences

        # Check bearish reversal: Close below active high break level
        elif (not pd.isna(active_break_high) and
              not pd.isna(active_purge_high_idx) and
              close_price < active_break_high and
              i != active_purge_high_idx):  # Not the purge candle

            df.at[idx, 'bearish_reversal'] = 1
            df.at[idx, 'reversal_purge_level'] = active_break_high  # Store purge level
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
    Get MPI position context label based on Percentile Zones (1-5)

    Args:
        mpi_zone: MPI zone (1-5)
        direction: 'bullish', 'bearish', or 'neutral'

    Returns:
        Position label with emoji
    """
    # Zone 1: Strong Contraction (<20)
    # Zone 2: Contraction (20-40)
    # Zone 3: Neutral (40-60)
    # Zone 4: Expansion (60-80)
    # Zone 5: Strong Expansion (>80)

    if direction == 'bullish':
        zone_labels = {
            1: "❌ TOO WEAK (<20%)",
            2: "⚠️ WEAK (20-40%)",
            3: "📍 EARLY (40-60%)",
            4: "🚀 STRONG (60-80%)",
            5: "🔥 EXTENDED (>80%)"
        }
    elif direction == 'bearish':
        zone_labels = {
            1: "🔥 EXTENDED (<20%)", # Strong contraction is good for shorts
            2: "🚀 STRONG (20-40%)",
            3: "📍 EARLY (40-60%)",
            4: "⚠️ WEAK (60-80%)",
            5: "❌ TOO STRONG (>80%)"
        }
    else: # Neutral
        zone_labels = {
            1: "📉 Strong Downtrend",
            2: "↘️ Downtrend",
            3: "➡️ Range/Neutral",
            4: "↗️ Uptrend",
            5: "📈 Strong Uptrend"
        }

    return zone_labels.get(mpi_zone, "❓ UNKNOWN")


# ===== INSTITUTIONAL FLOW ANALYSIS FUNCTIONS =====
# Phase 1: Core institutional flow metrics implementation

def calculate_institutional_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume-weighted directional institutional flow metrics

    Core Concept: Track institutional buying/selling pressure through volume-weighted flow
    - Daily_Flow: Volume-weighted directional pressure (+buy/-sell)
    - Flow_10D: 10-day cumulative institutional flow
    - Flow_Velocity: Day-over-day flow acceleration/deceleration

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with institutional flow columns added
    """
    # Calculate price change and direction
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Direction'] = np.sign(df['Price_Change'])

    # Handle flat candles (no price change)
    df['Price_Direction'] = np.where(
        df['Price_Change'] == 0,
        0,  # Neutral for flat candles
        df['Price_Direction']
    )

    # Volume-weighted directional flow
    # Positive flow = buying pressure, negative flow = selling pressure
    df['Daily_Flow'] = (
        df['Volume'] *
        df['Price_Direction'] *
        (abs(df['Price_Change']) / df['Close'])  # Normalize by price
    )

    # 10-day cumulative flow (rolling sum)
    df['Flow_10D'] = df['Daily_Flow'].rolling(10, min_periods=5).sum()

    # Flow velocity (acceleration/deceleration)
    df['Flow_Velocity'] = df['Flow_10D'] - df['Flow_10D'].shift(1)

    # Fill NaN values
    df['Daily_Flow'] = df['Daily_Flow'].fillna(0.0)
    df['Flow_10D'] = df['Flow_10D'].fillna(0.0)
    df['Flow_Velocity'] = df['Flow_Velocity'].fillna(0.0)

    return df


def classify_flow_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify institutional flow into regime categories based on strength and direction

    Regimes:
    - Strong Accumulation: Top 20% of positive flow
    - Accumulation: Top 40% of positive flow
    - Neutral: Middle 20% around zero
    - Distribution: Bottom 40% of negative flow
    - Strong Distribution: Bottom 20% of negative flow

    Args:
        df: DataFrame with Flow_10D column

    Returns:
        DataFrame with Flow_Regime column added
    """
    # Calculate INDIVIDUAL STOCK historical percentiles (100-day rolling)
    # These compare current values against the stock's own history
    # CRITICAL FIX: Use .shift(1) to avoid lookahead bias (don't include current day in ranking)
    df['Flow_Percentile'] = df['Flow_10D'].shift(1).rolling(window=100, min_periods=20).rank(pct=True) * 100

    # Fill NaN values
    df['Flow_Percentile'] = df['Flow_Percentile'].fillna(50.0)

    # ===== ARCHIVED FEATURE: Flow_Velocity_Percentile =====
    # Removed: 2025-12-19 - Individual percentile never used (Flow_Velocity_Rank used instead)
    # Archive: data/feature_library/unused_features_library.json
    # To re-enable: Set FEATURE_CONFIG['calculate_unused_features'] = True
    if FEATURE_CONFIG['calculate_unused_features']:
        df['Flow_Velocity_Percentile'] = df['Flow_Velocity'].shift(1).rolling(window=100, min_periods=20).rank(pct=True) * 100
        df['Flow_Velocity_Percentile'] = df['Flow_Velocity_Percentile'].fillna(50.0)
    else:
        # PRODUCTION MODE: Add placeholder column
        df['Flow_Velocity_Percentile'] = 50.0

    # ===== ARCHIVED FEATURE: Flow_Regime =====
    # Removed: 2025-12-19 - Regime classification never used in scoring, display, or dependencies
    # Archive: data/feature_library/unused_features_library.json
    # To re-enable: Set FEATURE_CONFIG['calculate_unused_features'] = True
    if FEATURE_CONFIG['calculate_unused_features']:
        # Classify into regimes based on percentile (now 0-100 scale)
        df['Flow_Regime'] = 'Neutral'  # Default
        df.loc[df['Flow_Percentile'] >= 80, 'Flow_Regime'] = 'Strong Accumulation'
        df.loc[(df['Flow_Percentile'] >= 60) & (df['Flow_Percentile'] < 80), 'Flow_Regime'] = 'Accumulation'
        df.loc[(df['Flow_Percentile'] <= 40) & (df['Flow_Percentile'] > 20), 'Flow_Regime'] = 'Distribution'
        df.loc[df['Flow_Percentile'] <= 20, 'Flow_Regime'] = 'Strong Distribution'
    else:
        # PRODUCTION MODE: Add placeholder column
        df['Flow_Regime'] = 'Neutral'

    return df


def calculate_volume_conviction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume conviction metrics showing institutional commitment

    Core Concept: Track ratio of up-day vs down-day volume participation
    - Volume_Conviction: Ratio of up-day to down-day volume (>1.0 = bullish, <1.0 = bearish)
    - Avg_Vol_Up_10D: Average volume on up days over 10-day period
    - Conviction_Velocity: Day-over-day conviction change

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with volume conviction columns added
    """
    # Identify up and down days
    df['Is_Up_Day'] = (df['Close'] > df['Open']).astype(int)
    df['Is_Down_Day'] = (df['Close'] < df['Open']).astype(int)

    # Separate volume by direction
    df['Volume_Up'] = np.where(df['Is_Up_Day'] == 1, df['Volume'], 0)
    df['Volume_Down'] = np.where(df['Is_Down_Day'] == 1, df['Volume'], 0)

    # 10-day average volumes by direction
    df['Avg_Vol_Up_10D'] = df['Volume_Up'].rolling(10, min_periods=5).mean()
    df['Avg_Vol_Down_10D'] = df['Volume_Down'].rolling(10, min_periods=5).mean()

    # Volume conviction ratio (up-day vs down-day volume)
    # >1.0 = bullish conviction, <1.0 = bearish conviction, =1.0 = neutral
    df['Volume_Conviction'] = np.where(
        df['Avg_Vol_Down_10D'] > 0,
        df['Avg_Vol_Up_10D'] / df['Avg_Vol_Down_10D'],
        1.0  # Neutral if no down-day volume
    )

    # ===== ARCHIVED FEATURE: Conviction_Velocity =====
    # Removed: 2025-12-19 - Velocity metric never used in scoring, display, or dependencies
    # Archive: data/feature_library/unused_features_library.json
    # To re-enable: Set FEATURE_CONFIG['calculate_unused_features'] = True
    if FEATURE_CONFIG['calculate_unused_features']:
        # Conviction velocity (day-over-day change)
        df['Conviction_Velocity'] = df['Volume_Conviction'] - df['Volume_Conviction'].shift(1)
        df['Conviction_Velocity'] = df['Conviction_Velocity'].fillna(0.0)
    else:
        # PRODUCTION MODE: Add placeholder column
        df['Conviction_Velocity'] = 0.0

    # Fill NaN values
    df['Volume_Conviction'] = df['Volume_Conviction'].fillna(1.0)  # Neutral default
    df['Avg_Vol_Up_10D'] = df['Avg_Vol_Up_10D'].fillna(0.0)

    return df


# OLD FUNCTION REMOVED: calculate_price_flow_divergence()
# This function had a critical math error (scale mismatch: 0-1 vs 0-100)
# Replaced with comprehensive DivergenceDetector from core/divergence_analysis.py
# The new module provides:
#   - Type 1: Price-Flow Misalignment (mean reversion)
#   - Type 2: Classical Divergence (momentum exhaustion)
#   - Type 3: Volume-Price Divergence (quality signals)


def add_time_decay_features(df):
    """
    Add features measuring time since key events
    
    Critical for distinguishing fresh vs stale signals
    
    Args:
        df: DataFrame with OHLCV data (must be sorted by date)
        
    Returns:
        DataFrame with time-decay features added
    """
    import pandas as pd
    import numpy as np
    
    df = df.copy()
    df = df.sort_index()  # Ensure sorted by date
    
    logger.debug("⏱️  Adding time-decay features...")
    
    # 1. Days Since High (252-day rolling high)
    rolling_high = df['High'].rolling(window=252, min_periods=20).max()
    df['Days_Since_High'] = 0
    
    counter = 0
    for i in range(len(df)):
        if i == 0:
            df.iloc[i, df.columns.get_loc('Days_Since_High')] = 0
            continue
            
        if df['High'].iloc[i] >= rolling_high.iloc[i]:
            counter = 0  # New high, reset counter
        else:
            counter += 1
        
        df.iloc[i, df.columns.get_loc('Days_Since_High')] = counter
    
    logger.debug(f"  ✅ Added Days_Since_High (max: {df['Days_Since_High'].max():.0f} days)")
    
    # 2. Days Since Low (CRITICAL - 252-day rolling low)
    rolling_low = df['Low'].rolling(window=252, min_periods=20).min()
    df['Days_Since_Low'] = 0
    
    counter = 0
    for i in range(len(df)):
        if i == 0:
            df.iloc[i, df.columns.get_loc('Days_Since_Low')] = 0
            continue
            
        if df['Low'].iloc[i] <= rolling_low.iloc[i]:
            counter = 0  # New low, reset counter
        else:
            counter += 1
        
        df.iloc[i, df.columns.get_loc('Days_Since_Low')] = counter
    
    logger.debug(f"  ✅ Added Days_Since_Low (max: {df['Days_Since_Low'].max():.0f} days) - CRITICAL")
    
    # 3. Days Since Triple Aligned (FIXED LOGIC)
    if 'Is_Triple_Aligned' in df.columns:
        df['Days_Since_Triple_Aligned'] = 0
        days_since_last_alignment = 0

        for i in range(len(df)):
            if df['Is_Triple_Aligned'].iloc[i]:
                # Currently aligned → Reset counter to 0
                days_since_last_alignment = 0
            else:
                # Not aligned → Increment "days since last alignment"
                days_since_last_alignment += 1

            df.iloc[i, df.columns.get_loc('Days_Since_Triple_Aligned')] = days_since_last_alignment

        non_zero_count = (df['Days_Since_Triple_Aligned'] > 0).sum()
        logger.debug(f"  ✅ Days_Since_Triple_Aligned (max: {df['Days_Since_Triple_Aligned'].max():.0f} days, non-zero: {non_zero_count})")
    else:
        logger.debug(f"  ⚠️  Is_Triple_Aligned not found, skipping Days_Since_Triple_Aligned")
    
    # 4. Days Since Flow Regime Change (FIXED LOGIC)
    if 'Flow_10D' in df.columns:
        df['Days_Since_Flow_Regime_Change'] = 0
        days_since_change = 0
        previous_sign = None

        for i in range(len(df)):
            current_sign = np.sign(df['Flow_10D'].iloc[i])

            if i == 0:
                # First row - no previous regime to compare
                previous_sign = current_sign
                days_since_change = 0
            elif current_sign != previous_sign and previous_sign != 0 and current_sign != 0:
                # Regime changed (positive ↔ negative)
                days_since_change = 0  # Reset counter
                previous_sign = current_sign
            else:
                # Regime continues (same sign)
                days_since_change += 1

            df.iloc[i, df.columns.get_loc('Days_Since_Flow_Regime_Change')] = days_since_change

        non_zero_count = (df['Days_Since_Flow_Regime_Change'] > 0).sum()
        logger.debug(f"  ✅ Days_Since_Flow_Regime_Change (max: {df['Days_Since_Flow_Regime_Change'].max():.0f} days, non-zero: {non_zero_count})")
    else:
        logger.debug(f"  ⚠️  Flow_10D not found, skipping Days_Since_Flow_Regime_Change")
    
    logger.debug("✅ Time-decay features complete")
    
    return df


logger.info("Technical Analysis Module loaded with optimized PURE MPI EXPANSION system (Market Regime removed)")
