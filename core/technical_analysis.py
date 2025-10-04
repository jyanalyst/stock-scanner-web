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
    
    # TEMPORARY DEBUG: Check MPI trend distribution
    if len(df) > 0:
        trend_counts = df['MPI_Trend'].value_counts()
        logger.info(f"DEBUG MPI Trends: {dict(trend_counts)}")
        
        # Sample some velocity values
        sample_velocities = df['MPI_Velocity'].dropna().tail(10)
        logger.info(f"DEBUG Sample velocities: {sample_velocities.tolist()}")

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
    
    return df

def calculate_relative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Relative Volume similar to the website's approach
    
    Relative Volume = (Current Day Volume) / (14-day Average Volume) × 100
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with Relative Volume columns added
    """
    # Calculate 14-day rolling average volume
    df['Volume_14D_Avg'] = df['Volume'].rolling(14, min_periods=7).mean()
    
    # Calculate relative volume as percentage
    df['Relative_Volume'] = (df['Volume'] / df['Volume_14D_Avg']) * 100
    
    # Fill NaN values with 100% (neutral)
    df['Relative_Volume'] = df['Relative_Volume'].fillna(100.0)
    
    # High activity flags for reference
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

def add_enhanced_columns(df_daily: pd.DataFrame, ticker: str, rolling_window: int = 20) -> pd.DataFrame:
    """
    Add enhanced columns with PURE MPI EXPANSION system, Relative Volume, and Market Regime
    
    Args:
        df_daily: Raw OHLCV data from yfinance
        ticker: Stock symbol
        rolling_window: Window for moving averages (kept for backward compatibility)
    
    Returns:
        DataFrame with MPI-enhanced technical analysis columns, Relative Volume, and Market Regime
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
        
        # NEW: Add market regime analysis
        df = add_market_regime_analysis(df, ticker)
        
        # Log successful calculation
        logger.info(f"{ticker}: Enhanced analysis completed successfully")
        logger.info(f"{ticker}: Latest MPI: {df['MPI'].iloc[-1]:.1%}, "
                   f"Velocity: {df['MPI_Velocity'].iloc[-1]:+.3f}, "
                   f"Trend: {df['MPI_Trend'].iloc[-1]}, "
                   f"Regime: {df['Market_Regime'].iloc[-1]}")
        logger.info(f"{ticker}: Latest Relative Volume: {df['Relative_Volume'].iloc[-1]:.0f}%")
        
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
        df['Market_Regime'] = 'Unknown'
        df['Regime_Probability'] = 0.5
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
        'strong_expansion_count': int((df['MPI_Trend'] == 'Strong Expansion').sum()) if 'MPI_Trend' in df.columns else 0,
        'contraction_count': int(df['MPI_Trend'].isin(['Mild Contraction', 'Strong Contraction']).sum()) if 'MPI_Trend' in df.columns else 0
    }
    
    return validation_results

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

logger.info("Technical Analysis Module loaded with optimized PURE MPI EXPANSION system (Buy Signal logic removed)")

def add_market_regime_analysis(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add market regime analysis to enhanced dataframe
    
    Args:
        df: DataFrame with technical analysis
        ticker: Stock ticker
        
    Returns:
        DataFrame with regime analysis added
    """
    try:
        from core.market_regime import MarketRegimeDetector
        
        # Initialize detector
        detector = MarketRegimeDetector(n_regimes=2)
        
        # Fit on historical data
        detector.fit(df)
        
        # Get regime predictions
        regime_data = detector.predict_regime(df)
        
        # Add regime data to main dataframe
        df['Market_Regime'] = 'Unknown'
        df['Regime_Probability'] = 0.0
        
        # Align indices and add regime data
        for idx in regime_data.index:
            if idx in df.index:
                regime_idx = int(regime_data.loc[idx, 'regime'])
                df.loc[idx, 'Market_Regime'] = regime_data.loc[idx, 'regime_label']
                df.loc[idx, 'Regime_Probability'] = regime_data.loc[idx, f'prob_regime_{regime_idx}']
        
        # Forward fill for any missing values
        df['Market_Regime'] = df['Market_Regime'].ffill()
        df['Regime_Probability'] = df['Regime_Probability'].ffill()
        
        logger.info(f"{ticker}: Market regime analysis completed")
        
    except Exception as e:
        logger.warning(f"{ticker}: Market regime analysis failed: {e}")
        # Add default values
        df['Market_Regime'] = 'Unknown'
        df['Regime_Probability'] = 0.5
    
    return df