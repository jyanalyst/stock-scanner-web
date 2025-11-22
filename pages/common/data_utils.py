"""
Common data processing utilities used across modules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        if pd.isna(value) or value == '' or str(value).lower() == 'nan':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        if pd.isna(value) or value == '' or str(value).lower() == 'nan':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_string(value: Any, default: str = 'Unknown') -> str:
    """Safely convert value to string"""
    try:
        if pd.isna(value) or value is None:
            return default
        return str(value)
    except:
        return default


def validate_ticker(ticker: str) -> bool:
    """Validate ticker format"""
    if not ticker or pd.isna(ticker):
        return False

    ticker_str = str(ticker).strip()
    if not ticker_str or ticker_str.lower() == 'nan':
        return False

    # Basic SGX ticker validation (should end with .SG and have reasonable length)
    if not ticker_str.endswith('.SG'):
        return False

    if len(ticker_str) < 4 or len(ticker_str) > 10:
        return False

    return True


def clean_ticker_list(tickers: List[str]) -> List[str]:
    """Clean and validate a list of tickers"""
    cleaned = []
    for ticker in tickers:
        if validate_ticker(ticker):
            cleaned.append(ticker)
        else:
            logger.warning(f"Invalid ticker skipped: {ticker}")

    return cleaned


def standardize_columns(df: pd.DataFrame, column_mapping: Dict[str, str] = None) -> pd.DataFrame:
    """
    Standardize column names in DataFrame

    Args:
        df: DataFrame to standardize
        column_mapping: Optional custom mapping, otherwise uses defaults
    """
    if column_mapping is None:
        column_mapping = {
            'Last': 'Close',
            'Vol': 'Volume',
            'open': 'Open',
            'high': 'High',
            'low': 'Low'
        }

    df = df.rename(columns=column_mapping)

    # Ensure Volume is integer if present
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].astype(float).astype(int)

    return df


def parse_date_column(df: pd.DataFrame, date_col: str = 'Date',
                     date_format: str = 'mixed', dayfirst: bool = True) -> pd.DataFrame:
    """
    Parse date column with proper error handling

    Args:
        df: DataFrame with date column
        date_col: Name of date column
        date_format: Date format ('mixed', 'iso', etc.)
        dayfirst: Whether day comes first in format
    """
    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame")
        return df

    try:
        if date_format == 'mixed':
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, format='mixed')
        else:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)

        # Set as index if it's the date column
        if date_col == 'Date':
            df.set_index('Date', inplace=True)

        logger.info(f"Successfully parsed {len(df)} date entries")

    except Exception as e:
        logger.error(f"Error parsing dates: {e}")

    return df


def filter_date_range(df: pd.DataFrame, start_date: date = None,
                     end_date: date = None) -> pd.DataFrame:
    """
    Filter DataFrame by date range

    Args:
        df: DataFrame with datetime index
        start_date: Start date filter
        end_date: End date filter
    """
    if df.empty or df.index.empty:
        return df

    # Ensure we have datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame does not have DatetimeIndex, skipping date filter")
        return df

    mask = pd.Series(True, index=df.index)

    if start_date:
        start_dt = pd.Timestamp(start_date)
        mask &= (df.index >= start_dt)

    if end_date:
        end_dt = pd.Timestamp(end_date)
        mask &= (df.index <= end_dt)

    filtered_df = df[mask].copy()
    logger.info(f"Date filter: {len(df)} → {len(filtered_df)} rows")

    return filtered_df


def calculate_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """Calculate returns from price data"""
    if price_col not in df.columns:
        logger.warning(f"Price column '{price_col}' not found")
        return df

    df = df.copy()
    df['Returns'] = df[price_col].pct_change()
    df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))

    return df


def calculate_moving_averages(df: pd.DataFrame, price_col: str = 'Close',
                            windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
    """Calculate moving averages for different windows"""
    if price_col not in df.columns:
        logger.warning(f"Price column '{price_col}' not found")
        return df

    df = df.copy()

    for window in windows:
        df[f'MA_{window}'] = df[price_col].rolling(window=window, min_periods=1).mean()
        df[f'MA_{window}_Slope'] = df[f'MA_{window}'].pct_change(periods=window)

    return df


def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a column using specified method

    Args:
        df: DataFrame containing the column
        column: Column name to check for outliers
        method: Method to use ('iqr', 'zscore', 'percentile')
        threshold: Threshold for outlier detection

    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found")
        return pd.Series(False, index=df.index)

    data = df[column].dropna()

    if len(data) < 10:
        return pd.Series(False, index=df.index)

    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (data < lower_bound) | (data > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > threshold

    elif method == 'percentile':
        lower_percentile = data.quantile(threshold/100)
        upper_percentile = data.quantile(1 - threshold/100)
        outliers = (data < lower_percentile) | (data > upper_percentile)

    else:
        logger.warning(f"Unknown outlier detection method: {method}")
        return pd.Series(False, index=df.index)

    # Return series aligned with original index
    result = pd.Series(False, index=df.index)
    result.loc[outliers.index] = outliers

    outlier_count = outliers.sum()
    logger.info(f"Detected {outlier_count} outliers in {column} using {method} method")

    return result


def resample_data(df: pd.DataFrame, freq: str = 'D',
                 aggregation: Dict[str, str] = None) -> pd.DataFrame:
    """
    Resample time series data

    Args:
        df: DataFrame with datetime index
        freq: Frequency string ('D', 'W', 'M', etc.)
        aggregation: Aggregation methods for columns
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame does not have DatetimeIndex, skipping resampling")
        return df

    if aggregation is None:
        # Default aggregation
        aggregation = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

    # Filter aggregation to only existing columns
    valid_agg = {col: method for col, method in aggregation.items() if col in df.columns}

    resampled = df.resample(freq).agg(valid_agg)
    logger.info(f"Resampled data: {len(df)} → {len(resampled)} rows at {freq} frequency")

    return resampled


def merge_dataframes(dfs: List[pd.DataFrame], on: str = None,
                    how: str = 'outer') -> pd.DataFrame:
    """
    Merge multiple DataFrames

    Args:
        dfs: List of DataFrames to merge
        on: Column/index to merge on
        how: Merge method ('outer', 'inner', 'left', 'right')
    """
    if not dfs:
        return pd.DataFrame()

    if len(dfs) == 1:
        return dfs[0]

    result = dfs[0]
    for df in dfs[1:]:
        if on:
            result = result.merge(df, on=on, how=how, suffixes=('', '_right'))
        else:
            result = result.join(df, how=how, rsuffix='_right')

    logger.info(f"Merged {len(dfs)} DataFrames into {len(result)} rows")
    return result


def validate_data_quality(df: pd.DataFrame, checks: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Perform data quality checks on DataFrame

    Args:
        checks: Dictionary of quality checks to perform

    Returns:
        Dictionary with quality check results
    """
    if checks is None:
        checks = {
            'min_rows': 10,
            'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'date_index': True,
            'no_null_prices': True,
            'reasonable_price_range': (0.01, 10000.0)
        }

    results = {
        'passed': True,
        'issues': [],
        'stats': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_counts': df.isnull().sum().to_dict()
        }
    }

    # Check minimum rows
    if len(df) < checks.get('min_rows', 10):
        results['issues'].append(f"Insufficient data: {len(df)} rows (minimum {checks['min_rows']})")
        results['passed'] = False

    # Check required columns
    required_cols = checks.get('required_columns', [])
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results['issues'].append(f"Missing required columns: {missing_cols}")
        results['passed'] = False

    # Check date index
    if checks.get('date_index', True):
        if not isinstance(df.index, pd.DatetimeIndex):
            results['issues'].append("DataFrame does not have DatetimeIndex")
            results['passed'] = False

    # Check for null prices
    if checks.get('no_null_prices', True):
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    results['issues'].append(f"Null values in {col}: {null_count} rows")
                    results['passed'] = False

    # Check reasonable price range
    price_range = checks.get('reasonable_price_range')
    if price_range and 'Close' in df.columns:
        prices = df['Close'].dropna()
        if len(prices) > 0:
            min_price, max_price = prices.min(), prices.max()
            if min_price < price_range[0] or max_price > price_range[1]:
                results['issues'].append(f"Prices outside reasonable range: {min_price:.2f} - {max_price:.2f}")
                results['passed'] = False

    return results
