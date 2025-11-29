# VWAP Engine - Monthly VWAP Calculation
"""
Monthly VWAP calculation engine for RVOL BackTest
Calculates running VWAP that resets each month
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def calculate_monthly_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Monthly VWAP with daily updates and monthly resets

    Core Algorithm:
    1. Group data by Year-Month combination
    2. For each month, calculate running cumulative:
       - Cumulative Price × Volume
       - Cumulative Volume
       - VWAP = Cumulative(P×V) / Cumulative(V)
    3. Reset calculations at start of new month

    Args:
        df: DataFrame with Date, Close, Volume columns

    Returns:
        DataFrame with added 'Monthly_VWAP' column
    """
    df = df.copy()

    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    # Create Year-Month identifier for grouping
    df['Year_Month'] = df.index.to_period('M')

    # Initialize VWAP column
    df['Monthly_VWAP'] = np.nan

    # Calculate VWAP for each month separately
    for year_month, month_data in df.groupby('Year_Month'):
        # Sort by date to ensure chronological order
        month_data = month_data.sort_index()

        # Calculate cumulative Price × Volume and cumulative Volume
        month_data['Cum_Price_Vol'] = (month_data['Close'] * month_data['Volume']).cumsum()
        month_data['Cum_Volume'] = month_data['Volume'].cumsum()

        # Calculate running VWAP for this month
        month_data['Monthly_VWAP'] = month_data['Cum_Price_Vol'] / month_data['Cum_Volume']

        # Update the main dataframe
        df.loc[month_data.index, 'Monthly_VWAP'] = month_data['Monthly_VWAP']

    # Clean up temporary columns
    df = df.drop(columns=['Year_Month'], errors='ignore')

    # Fill any remaining NaN values (shouldn't happen with proper data)
    df['Monthly_VWAP'] = df['Monthly_VWAP'].fillna(method='ffill')

    logger.info(f"Calculated Monthly VWAP for {len(df)} days across {df.index.to_period('M').nunique()} months")

    return df

def validate_vwap_calculation(df: pd.DataFrame) -> dict:
    """
    Validate Monthly VWAP calculation quality

    Args:
        df: DataFrame with Monthly_VWAP column

    Returns:
        dict: Validation results
    """
    validation = {
        'total_days': len(df),
        'months_covered': df.index.to_period('M').nunique(),
        'vwap_values': df['Monthly_VWAP'].notna().sum(),
        'vwap_completeness': df['Monthly_VWAP'].notna().mean(),
        'vwap_range': {
            'min': df['Monthly_VWAP'].min(),
            'max': df['Monthly_VWAP'].max(),
            'mean': df['Monthly_VWAP'].mean()
        }
    }

    # Check for monthly resets (VWAP should reset or change significantly at month boundaries)
    df_temp = df.copy()
    df_temp['Month_Change'] = df_temp.index.to_period('M') != df_temp.index.to_period('M').shift(1)
    df_temp['VWAP_Change'] = abs(df_temp['Monthly_VWAP'] - df_temp['Monthly_VWAP'].shift(1))

    # Large VWAP changes at month boundaries indicate proper resets
    month_boundary_changes = df_temp[df_temp['Month_Change']]['VWAP_Change']
    validation['month_boundary_changes'] = {
        'count': len(month_boundary_changes),
        'avg_change': month_boundary_changes.mean(),
        'large_changes': (month_boundary_changes > df_temp['Monthly_VWAP'].mean() * 0.1).sum()
    }

    return validation

def get_vwap_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for Monthly VWAP

    Args:
        df: DataFrame with Monthly_VWAP column

    Returns:
        dict: Summary statistics
    """
    if 'Monthly_VWAP' not in df.columns:
        return {'error': 'Monthly_VWAP column not found'}

    vwap_series = df['Monthly_VWAP']

    # Basic statistics
    summary = {
        'total_days': len(df),
        'vwap_mean': vwap_series.mean(),
        'vwap_std': vwap_series.std(),
        'vwap_min': vwap_series.min(),
        'vwap_max': vwap_series.max(),
        'months_covered': df.index.to_period('M').nunique()
    }

    # Monthly VWAP ranges
    monthly_ranges = df.groupby(df.index.to_period('M'))['Monthly_VWAP'].agg(['min', 'max'])
    summary['monthly_range_avg'] = (monthly_ranges['max'] - monthly_ranges['min']).mean()

    # VWAP vs Price relationship
    if 'Close' in df.columns:
        df_temp = df.copy()
        df_temp['VWAP_Distance'] = (df_temp['Close'] - df_temp['Monthly_VWAP']) / df_temp['Monthly_VWAP']
        summary['avg_vwap_distance_pct'] = df_temp['VWAP_Distance'].mean() * 100
        summary['vwap_distance_std_pct'] = df_temp['VWAP_Distance'].std() * 100

    return summary
