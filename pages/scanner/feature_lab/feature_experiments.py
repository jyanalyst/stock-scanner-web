"""
Feature Experiments - Technical Indicator Calculations

Provides functions to calculate experimental technical indicators for feature testing.
These features are calculated for historical dates to analyze their predictive power.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from utils.paths import HISTORICAL_DATA_DIR

logger = logging.getLogger(__name__)


def calculate_dist_from_ma20(ticker: str, date: date) -> float:
    """
    Calculate the percentage distance from 20-day moving average.

    Args:
        ticker: Stock ticker symbol
        date: Analysis date

    Returns:
        Percentage distance from MA20 (positive = above MA, negative = below MA)
        Returns NaN if insufficient data
    """
    try:
        # Load historical data
        # Handle .SG suffix if present
        clean_ticker = ticker.replace('.SG', '')
        csv_path = HISTORICAL_DATA_DIR / f"{clean_ticker}.csv"
        
        if not csv_path.exists():
            # Try with .SG just in case
            csv_path = HISTORICAL_DATA_DIR / f"{ticker}.csv"
            if not csv_path.exists():
                return np.nan

        df = pd.read_csv(csv_path)

        # Ensure Date column exists and convert to datetime
        if 'Date' not in df.columns:
            return np.nan

        # Parse dates with dayfirst=True to handle DD/MM/YYYY format
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')

        # Find the row for the given date
        target_date = pd.Timestamp(date)
        date_rows = df[df['Date'] == target_date]

        if date_rows.empty:
            return np.nan

        # Get close price for the target date
        close_price = date_rows['Close'].iloc[0]

        # Calculate 20-day MA ending on the target date
        # Need at least 20 days of data including the target date
        end_idx = df[df['Date'] <= target_date].index[-1]
        start_idx = max(0, end_idx - 19)  # 20 days including current

        if end_idx - start_idx + 1 < 20:
            return np.nan

        ma20 = df.loc[start_idx:end_idx, 'Close'].mean()

        # Calculate percentage distance
        if ma20 == 0:
            return np.nan

        distance_pct = ((close_price - ma20) / ma20) * 100

        return float(distance_pct)

    except Exception as e:
        logger.warning(f"Error calculating DistFromMA20 for {ticker} on {date}: {e}")
        return np.nan


def calculate_rsi_14(ticker: str, date: date) -> float:
    """
    Calculate 14-period RSI (Relative Strength Index).

    Args:
        ticker: Stock ticker symbol
        date: Analysis date

    Returns:
        RSI value (0-100 scale)
        Returns NaN if insufficient data
    """
    try:
        # Load historical data
        # Handle .SG suffix if present
        clean_ticker = ticker.replace('.SG', '')
        csv_path = HISTORICAL_DATA_DIR / f"{clean_ticker}.csv"
        
        if not csv_path.exists():
            # Try with .SG just in case
            csv_path = HISTORICAL_DATA_DIR / f"{ticker}.csv"
            if not csv_path.exists():
                return np.nan

        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if 'Date' not in df.columns or 'Close' not in df.columns:
            return np.nan

        # Parse dates with dayfirst=True to handle DD/MM/YYYY format
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values('Date')

        # Find data up to and including the target date
        target_date = pd.Timestamp(date)
        mask = df['Date'] <= target_date
        historical_data = df[mask].copy()

        if len(historical_data) < 15:  # Need at least 15 days for 14-period RSI
            return np.nan

        # Calculate daily price changes
        historical_data['Price_Change'] = historical_data['Close'].diff()

        # Calculate gains and losses
        historical_data['Gain'] = historical_data['Price_Change'].apply(lambda x: x if x > 0 else 0)
        historical_data['Loss'] = historical_data['Price_Change'].apply(lambda x: abs(x) if x < 0 else 0)

        # Calculate average gains and losses using Wilder's smoothing
        # First average (simple average of first 14 periods)
        avg_gain = historical_data['Gain'].iloc[1:15].mean()  # Skip first NaN
        avg_loss = historical_data['Loss'].iloc[1:15].mean()

        # Apply Wilder's smoothing for remaining periods
        for i in range(15, len(historical_data)):
            avg_gain = (avg_gain * 13 + historical_data['Gain'].iloc[i]) / 14
            avg_loss = (avg_loss * 13 + historical_data['Loss'].iloc[i]) / 14

        # Calculate RS and RSI
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    except Exception as e:
        logger.warning(f"Error calculating RSI_14 for {ticker} on {date}: {e}")
        return np.nan


def calculate_volume_rank(ticker: str, date: date, all_stocks: pd.DataFrame) -> float:
    """
    Calculate volume percentile rank across all stocks for the given date.

    Args:
        ticker: Stock ticker symbol
        date: Analysis date
        all_stocks: DataFrame with all stocks' data for the date

    Returns:
        Volume percentile rank (0-100)
        Returns NaN if data unavailable
    """
    try:
        # Find the row for this ticker and date
        ticker_row = all_stocks[
            (all_stocks['Ticker'] == ticker) &
            (all_stocks['Date'] == date)
        ]

        if ticker_row.empty:
            return np.nan

        # Get relative volume for this ticker
        relative_volume = ticker_row['Relative_Volume'].iloc[0]

        if pd.isna(relative_volume):
            return np.nan

        # Get all relative volumes for this date
        date_data = all_stocks[all_stocks['Date'] == date]
        all_relative_volumes = date_data['Relative_Volume'].dropna()

        if len(all_relative_volumes) == 0:
            return np.nan

        # Calculate percentile rank
        rank = (all_relative_volumes <= relative_volume).sum() / len(all_relative_volumes) * 100

        return float(rank)

    except Exception as e:
        logger.warning(f"Error calculating VolumeRank for {ticker} on {date}: {e}")
        return np.nan


def add_feature_to_history(feature_name: str, selection_history_path: str) -> bool:
    """
    Calculate and add a feature to all historical selections.

    Args:
        feature_name: Name of the feature to calculate
        selection_history_path: Path to selection_history.json

    Returns:
        True if successful, False otherwise
    """
    try:
        import json
        from pathlib import Path

        # Load selection history
        with open(selection_history_path, 'r') as f:
            selection_history = json.load(f)

        # Create backup
        backup_path = f"{selection_history_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w') as f:
            json.dump(selection_history, f, indent=2)

        logger.info(f"Created backup: {backup_path}")

        # Collect all unique dates and tickers for volume rank calculation
        all_dates_data = []
        unique_dates = set()

        for date_str, date_data in selection_history.get('dates', {}).items():
            unique_dates.add(date_str)
            scan_results = date_data.get('scan_results', {})

            for ticker in scan_results.keys():
                all_dates_data.append({
                    'Date': date_str,
                    'Ticker': ticker,
                    'Relative_Volume': scan_results[ticker].get('Relative_Volume', np.nan)
                })

        # Convert to DataFrame for volume rank calculation
        all_stocks_df = pd.DataFrame(all_dates_data)
        all_stocks_df['Date'] = pd.to_datetime(all_stocks_df['Date']).dt.date

        # Process each date
        total_dates = len(selection_history.get('dates', {}))
        processed_dates = 0

        for date_str, date_data in selection_history.get('dates', {}).items():
            try:
                analysis_date = datetime.fromisoformat(date_str).date()
                scan_results = date_data.get('scan_results', {})

                # Calculate feature for each ticker
                for ticker in scan_results.keys():
                    if feature_name == 'DistFromMA20':
                        value = calculate_dist_from_ma20(ticker, analysis_date)
                    elif feature_name == 'RSI_14':
                        value = calculate_rsi_14(ticker, analysis_date)
                    elif feature_name == 'VolumeRank':
                        value = calculate_volume_rank(ticker, analysis_date, all_stocks_df)
                    else:
                        logger.warning(f"Unknown feature: {feature_name}")
                        continue

                    # Add feature to scan results
                    scan_results[ticker][feature_name] = value if not np.isnan(value) else None

                processed_dates += 1
                logger.info(f"Processed {processed_dates}/{total_dates} dates for {feature_name}")

            except Exception as e:
                logger.error(f"Error processing date {date_str}: {e}")
                continue

        # Update last modified timestamp
        selection_history['last_modified'] = datetime.now().isoformat()

        # Save updated selection history
        with open(selection_history_path, 'w') as f:
            json.dump(selection_history, f, indent=2)

        logger.info(f"Successfully added {feature_name} to all historical selections")
        return True

    except Exception as e:
        logger.error(f"Error adding feature {feature_name} to history: {e}")
        return False
