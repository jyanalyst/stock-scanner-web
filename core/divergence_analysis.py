"""
Divergence Analysis Module
Comprehensive divergence detection for ML feature engineering

This module provides three types of divergence detection:
1. Price-Flow Misalignment: Mean-reversion signals (overbought/oversold)
2. Classical Divergence: Momentum exhaustion signals (reversals)
3. Volume-Price Divergence: Quality signals (weak moves, fake breakouts)

Author: Stock Scanner ML Team
Date: 2025-12-11
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class DivergenceDetector:
    """
    Multi-dimensional divergence detection system
    Combines mean-reversion, momentum exhaustion, and volume quality signals
    
    Features Generated:
    - Flow_Price_Gap: Price percentile - Flow percentile (-100 to +100)
    - Gap_Severity: Absolute magnitude of gap (0 to 100)
    - Is_Flow_Misaligned: Binary flag for significant misalignment
    - Bullish_Divergence: Price down + Flow up (reversal signal)
    - Bearish_Divergence: Price up + Flow down (reversal signal)
    - Divergence_Strength: Magnitude of slope difference (0-100)
    - Divergence_Duration: Days since divergence started
    - Is_Volume_Weak: Strong price + weak volume (quality warning)
    - Volume_Price_Alignment: Correlation score (0-100)
    """
    
    def __init__(self, 
                 lookback_percentile: int = 252,
                 lookback_slope: int = 20,
                 min_slope_threshold: float = 0.05,
                 misalignment_threshold: float = 20.0,
                 correlation_window: int = 20):
        """
        Initialize divergence detector with configurable parameters
        
        Args:
            lookback_percentile: Window for percentile calculations (default 252 = 1 year)
            lookback_slope: Window for slope/trend calculations (default 20 days)
            min_slope_threshold: Minimum slope for divergence detection (default 5%)
            misalignment_threshold: Minimum gap for misalignment flag (default 20 percentile points)
            correlation_window: Window for volume-price correlation (default 20 days)
        """
        self.lookback_percentile = lookback_percentile
        self.lookback_slope = lookback_slope
        self.min_slope_threshold = min_slope_threshold
        self.misalignment_threshold = misalignment_threshold
        self.correlation_window = correlation_window
        
        logger.info(f"DivergenceDetector initialized: lookback={lookback_percentile}, "
                   f"slope_window={lookback_slope}, min_slope={min_slope_threshold}")
    
    def calculate_all_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master function: Calculate all divergence types
        
        Args:
            df: DataFrame with required columns (Close, Flow_10D, MPI_Percentile, IBS_Percentile, VPI_Percentile)
        
        Returns:
            DataFrame with all divergence features added
        """
        try:
            # Validate required columns
            self._validate_required_columns(df)
            
            # Type 1: Price-Flow Misalignment (Mean Reversion)
            df = self._calculate_price_flow_misalignment(df)
            
            # Type 2: Classical Divergence (Momentum Exhaustion)
            df = self._calculate_classical_divergence(df)
            
            # Type 3: Volume-Price Divergence (Quality Signal)
            df = self._calculate_volume_price_divergence(df)
            
            logger.debug(f"Calculated all divergences: {len(df)} rows processed")
            
        except Exception as e:
            logger.error(f"Divergence calculation failed: {e}")
            # Add fallback columns with neutral values
            df = self._add_fallback_columns(df)
        
        return df
    
    def _calculate_price_flow_misalignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate misalignment between price position and institutional flow
        
        Type 1: Mean-Reversion Signal
        Concept: Compare where price sits vs where flow sits in historical range
        
        Positive gap = Bearish (price ahead of flow, overbought)
        Negative gap = Bullish (flow ahead of price, accumulation)
        
        Args:
            df: DataFrame with Close and Flow_10D
        
        Returns:
            DataFrame with misalignment features added
        """
        # BOTH on 0-100 scale (FIXED from original bug)
        df['Price_Percentile'] = df['Close'].rolling(
            self.lookback_percentile, 
            min_periods=60
        ).rank(pct=True) * 100
        
        df['Flow_Percentile'] = df['Flow_10D'].rolling(
            self.lookback_percentile, 
            min_periods=60
        ).rank(pct=True) * 100
        
        # Gap: Positive = Price ahead of flow (bearish), Negative = Flow ahead of price (bullish)
        df['Flow_Price_Gap'] = df['Price_Percentile'] - df['Flow_Percentile']
        
        # Severity: 0-100 scale
        df['Gap_Severity'] = abs(df['Flow_Price_Gap'])
        
        # Binary flag: Significant misalignment (>threshold percentile points)
        df['Is_Flow_Misaligned'] = (df['Gap_Severity'] > self.misalignment_threshold).astype(int)
        
        # Fill NaN values
        df['Price_Percentile'] = df['Price_Percentile'].fillna(50.0)
        df['Flow_Percentile'] = df['Flow_Percentile'].fillna(50.0)
        df['Flow_Price_Gap'] = df['Flow_Price_Gap'].fillna(0.0)
        df['Gap_Severity'] = df['Gap_Severity'].fillna(0.0)
        df['Is_Flow_Misaligned'] = df['Is_Flow_Misaligned'].fillna(0)
        
        return df
    
    def _calculate_classical_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect classical divergence (opposite trends in price vs flow)
        
        Type 2: Momentum Exhaustion Signal
        Concept: Price making lower lows while flow makes higher lows = Bullish divergence
                 Price making higher highs while flow makes lower highs = Bearish divergence
        
        Args:
            df: DataFrame with Close and Flow_10D
        
        Returns:
            DataFrame with divergence features added
        """
        # Calculate trends (percentage change over slope window)
        df['Price_Slope'] = df['Close'].pct_change(self.lookback_slope)
        df['Flow_Slope'] = df['Flow_10D'].pct_change(self.lookback_slope)
        
        # Normalize slopes to -100 to +100 scale for interpretability
        df['Price_Slope_Pct'] = df['Price_Slope'] * 100
        df['Flow_Slope_Pct'] = df['Flow_Slope'] * 100
        
        # Bullish divergence: Price down, Flow up, Price near lows
        df['Bullish_Divergence'] = (
            (df['Price_Slope'] < -self.min_slope_threshold) &  # Price declining >5%
            (df['Flow_Slope'] > self.min_slope_threshold) &    # Flow rising >5%
            (df['Price_Percentile'] < 30)                      # Price near lows
        ).astype(int)
        
        # Bearish divergence: Price up, Flow down, Price near highs
        df['Bearish_Divergence'] = (
            (df['Price_Slope'] > self.min_slope_threshold) &   # Price rising >5%
            (df['Flow_Slope'] < -self.min_slope_threshold) &   # Flow declining >5%
            (df['Price_Percentile'] > 70)                      # Price near highs
        ).astype(int)
        
        # Divergence strength: How far apart are the slopes?
        df['Slope_Difference'] = abs(df['Price_Slope_Pct'] - df['Flow_Slope_Pct'])
        df['Divergence_Strength'] = df['Slope_Difference'].clip(0, 100)  # Cap at 100
        
        # Divergence duration: Count consecutive days since divergence started
        df['Div_Active'] = df['Bullish_Divergence'] | df['Bearish_Divergence']
        
        # Calculate duration using cumsum trick
        df['Div_Group'] = (df['Div_Active'] != df['Div_Active'].shift()).cumsum()
        df['Divergence_Duration'] = df.groupby('Div_Group').cumcount().where(df['Div_Active'], 0)
        
        # Clean up temporary columns
        df = df.drop(columns=['Div_Active', 'Div_Group', 'Price_Slope', 'Flow_Slope', 
                              'Price_Slope_Pct', 'Flow_Slope_Pct', 'Slope_Difference'])
        
        # Fill NaN values
        df['Bullish_Divergence'] = df['Bullish_Divergence'].fillna(0)
        df['Bearish_Divergence'] = df['Bearish_Divergence'].fillna(0)
        df['Divergence_Strength'] = df['Divergence_Strength'].fillna(0.0)
        df['Divergence_Duration'] = df['Divergence_Duration'].fillna(0)
        
        return df
    
    def _calculate_volume_price_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect volume-price divergence (weak volume on strong price moves)
        
        Type 3: Quality Signal
        Concept: Strong price move should have confirming volume
                 Weak volume on strong price = fake move, likely to reverse
        
        Args:
            df: DataFrame with MPI_Percentile, IBS_Percentile, VPI_Percentile
        
        Returns:
            DataFrame with volume-price features added
        """
        # Get price momentum strength (average of MPI and IBS percentiles)
        df['Price_Momentum_Avg'] = (df['MPI_Percentile'] + df['IBS_Percentile']) / 2
        
        # Volume strength from VPI
        volume_strength = df['VPI_Percentile']
        
        # Flag: Strong price momentum (>65) BUT weak volume (<40)
        df['Is_Volume_Weak'] = (
            (df['Price_Momentum_Avg'] > 65) &
            (volume_strength < 40)
        ).astype(int)
        
        # Alignment score: Rolling correlation between price momentum and volume
        df['Volume_Price_Correlation'] = df['Price_Momentum_Avg'].rolling(
            self.correlation_window
        ).corr(volume_strength)
        
        # Convert correlation (-1 to 1) to alignment score (0 to 100)
        # +1 correlation = 100 (perfect alignment), -1 = 0 (divergent), 0 = 50 (no relationship)
        df['Volume_Price_Alignment'] = ((df['Volume_Price_Correlation'] + 1) / 2) * 100
        
        # Clean up temporary columns
        df = df.drop(columns=['Price_Momentum_Avg', 'Volume_Price_Correlation'])
        
        # Fill NaN values
        df['Is_Volume_Weak'] = df['Is_Volume_Weak'].fillna(0)
        df['Volume_Price_Alignment'] = df['Volume_Price_Alignment'].fillna(50.0)  # Neutral
        
        return df
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist in DataFrame
        
        Args:
            df: DataFrame to validate
        
        Raises:
            ValueError: If required columns are missing
        """
        required = ['Close', 'Flow_10D', 'MPI_Percentile', 'IBS_Percentile', 'VPI_Percentile']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for divergence calculation: {missing}")
    
    def _add_fallback_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fallback columns with neutral values if calculation fails
        
        Args:
            df: DataFrame to add fallback columns to
        
        Returns:
            DataFrame with fallback columns added
        """
        fallback_cols = {
            'Price_Percentile': 50.0,
            'Flow_Percentile': 50.0,
            'Flow_Price_Gap': 0.0,
            'Gap_Severity': 0.0,
            'Is_Flow_Misaligned': 0,
            'Bullish_Divergence': 0,
            'Bearish_Divergence': 0,
            'Divergence_Strength': 0.0,
            'Divergence_Duration': 0,
            'Is_Volume_Weak': 0,
            'Volume_Price_Alignment': 50.0  # Neutral
        }
        for col, default_val in fallback_cols.items():
            if col not in df.columns:
                df[col] = default_val
        return df
    
    def get_divergence_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary statistics of divergence signals
        
        Args:
            df: DataFrame with divergence features
        
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'total_rows': len(df),
            'misalignment_count': int(df['Is_Flow_Misaligned'].sum()),
            'bullish_div_count': int(df['Bullish_Divergence'].sum()),
            'bearish_div_count': int(df['Bearish_Divergence'].sum()),
            'volume_weak_count': int(df['Is_Volume_Weak'].sum()),
            'avg_gap_severity': float(df['Gap_Severity'].mean()),
            'avg_divergence_strength': float(df['Divergence_Strength'].mean()),
            'avg_volume_alignment': float(df['Volume_Price_Alignment'].mean())
        }
        
        return summary


# Utility functions for external use
def get_divergence_interpretation(row: pd.Series) -> str:
    """
    Get human-readable interpretation of divergence signals for a single row
    
    Args:
        row: Single row from DataFrame with divergence features
    
    Returns:
        String interpretation of divergence state
    """
    interpretations = []
    
    # Check misalignment
    if row.get('Is_Flow_Misaligned', 0) == 1:
        gap = row.get('Flow_Price_Gap', 0)
        if gap > 20:
            interpretations.append("⚠️ BEARISH: Price ahead of flow (overbought)")
        elif gap < -20:
            interpretations.append("✅ BULLISH: Flow ahead of price (accumulation)")
    
    # Check classical divergence
    if row.get('Bullish_Divergence', 0) == 1:
        interpretations.append("🔄 BULLISH DIVERGENCE: Reversal up likely")
    if row.get('Bearish_Divergence', 0) == 1:
        interpretations.append("🔄 BEARISH DIVERGENCE: Reversal down likely")
    
    # Check volume quality
    if row.get('Is_Volume_Weak', 0) == 1:
        interpretations.append("⚠️ WEAK VOLUME: Move lacks conviction")
    
    if not interpretations:
        return "✅ ALIGNED: No divergence detected"
    
    return " | ".join(interpretations)


logger.info("Divergence Analysis Module loaded successfully")
