"""
Historical Data Collection for ML Training
- Runs scanner on past dates
- Calculates forward returns
- Labels outcomes
- Saves to training dataset
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os
import streamlit as st

logger = logging.getLogger(__name__)


class MLDataCollector:
    def __init__(self,
                 start_date: str = "2023-01-01",
                 end_date: str = "2024-12-31",
                 forward_days: List[int] = [2, 4]):
        """
        Initialize ML data collector

        Args:
            start_date: Start of training period
            end_date: End of training period
            forward_days: Days ahead to calculate returns (e.g., [2, 4])
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.forward_days = forward_days
        self.logger = logging.getLogger(__name__)

    def collect_training_data(self,
                             save_path: str = "data/ml_training/raw/",
                             resume_from: Optional[str] = None):
        """
        Main collection loop

        Process:
        1. Get all trading dates in range
        2. For each date:
           a. Run scanner as of that date
           b. Calculate forward returns
           c. Label outcomes
           d. Save to disk
        3. Combine into single dataset
        """
        # Get trading dates
        trading_dates = self._get_trading_dates()

        if resume_from:
            # Resume from checkpoint
            trading_dates = trading_dates[trading_dates > resume_from]
            self.logger.info(f"Resuming from {resume_from}")

        all_samples = []

        for i, date in enumerate(trading_dates):
            try:
                # Run historical scan
                scan_results = self._run_historical_scan(date)

                # Calculate labels
                labeled_data = self._calculate_forward_returns(
                    scan_results,
                    date
                )

                all_samples.extend(labeled_data)

                # Save checkpoint every 20 days
                if i % 20 == 0:
                    self._save_checkpoint(all_samples, date)
                    self.logger.info(f"Processed {i}/{len(trading_dates)} dates")

                # Update progress in session state for UI
                if hasattr(st, 'session_state'):
                    st.session_state.collection_progress = (i + 1) / len(trading_dates) * 100
                    st.session_state.current_date = date.strftime('%Y-%m-%d')

            except Exception as e:
                self.logger.error(f"Error on {date}: {e}")
                continue

        # Final save
        final_df = pd.DataFrame(all_samples)
        final_df.to_parquet(f"{save_path}/training_data_complete.parquet")

        return final_df

    def _run_historical_scan(self, target_date: datetime) -> pd.DataFrame:
        """
        Run scanner as of historical date

        This uses your EXISTING scanner logic but with a past date
        """
        from pages.scanner.logic import run_enhanced_stock_scan
        from utils.watchlist import get_active_watchlist

        # Get watchlist
        stocks = get_active_watchlist()

        # Run scan as of this date (NO LOOKAHEAD)
        # This will use Historical_Data filtered to target_date
        scan_results = run_enhanced_stock_scan(
            stocks_to_scan=stocks,
            analysis_date=target_date.date(),
            days_back=59,
            rolling_window=20
        )

        return scan_results

    def _calculate_forward_returns(self,
                                   scan_results: pd.DataFrame,
                                   entry_date: datetime) -> List[Dict]:
        """
        Calculate forward returns for each stock in scan

        CONVENTION: Signal Day = Day 0
        - entry_date = Day 0 (signal fires at close)
        - entry_price = Close price on Day 0
        - exit_date = Day 0 + N trading days
        - return_Nd = (Price_DayN - Price_Day0) / Price_Day0

        Example:
            Signal: Monday Nov 20 @ $3.39 (Day 0)
            return_2d: Wednesday Nov 22 @ $3.45 = +1.77%
            return_3d: Thursday Nov 23 @ $3.48 = +2.65%
            return_4d: Friday Nov 24 @ $3.50 = +3.24%

        Returns list of labeled samples
        """
        from core.local_file_loader import get_local_loader

        loader = get_local_loader()
        labeled_samples = []

        for _, stock in scan_results.iterrows():
            ticker = stock['Ticker']
            entry_price = stock['Close']

            # Calculate returns for each forward period
            forward_returns = {}

            for days in self.forward_days:
                exit_date = entry_date + timedelta(days=days)

                # Get future price (NO LOOKAHEAD - this is OK for training)
                exit_price = self._get_price_on_date(
                    ticker,
                    exit_date,
                    loader
                )

                if exit_price is not None:
                    ret = (exit_price - entry_price) / entry_price
                    forward_returns[f'return_{days}d'] = ret
                    forward_returns[f'win_{days}d'] = ret > 0
                else:
                    forward_returns[f'return_{days}d'] = None
                    forward_returns[f'win_{days}d'] = None

            # Calculate max drawdown during holding period
            max_dd = self._calculate_max_drawdown(
                ticker,
                entry_date,
                entry_date + timedelta(days=max(self.forward_days)),
                entry_price,
                loader
            )

            # Combine features + labels
            sample = {
                **stock.to_dict(),  # All scanner features
                **forward_returns,   # Forward returns
                'max_drawdown': max_dd,
                'entry_date': entry_date,
                'entry_price': entry_price
            }

            labeled_samples.append(sample)

        return labeled_samples

    def _get_price_on_date(self, ticker: str, date: datetime, loader) -> float:
        """Get closing price on specific date"""
        try:
            df = loader.load_historical_data(ticker)
            df['Date'] = pd.to_datetime(df['Date'])

            # Find closest trading day
            df = df[df['Date'] <= date]
            if len(df) == 0:
                return None

            return df.iloc[-1]['Close']
        except:
            return None

    def _calculate_max_drawdown(self, ticker, start_date, end_date,
                                entry_price, loader) -> float:
        """Calculate max drawdown during holding period"""
        try:
            df = loader.load_historical_data(ticker)
            df['Date'] = pd.to_datetime(df['Date'])

            # Filter to holding period
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            period_df = df[mask]

            if len(period_df) == 0:
                return 0

            # Calculate drawdown from entry
            returns = (period_df['Close'] - entry_price) / entry_price
            max_dd = returns.min()

            return max_dd
        except:
            return 0

    def _get_trading_dates(self) -> pd.DatetimeIndex:
        """Get all trading dates in range"""
        from core.local_file_loader import get_local_loader

        loader = get_local_loader()

        # Get reference stock to find trading dates
        reference_df = loader.load_historical_data("A17U.SG")
        reference_df['Date'] = pd.to_datetime(reference_df['Date'])

        # Filter to date range
        mask = (reference_df['Date'] >= self.start_date) & \
               (reference_df['Date'] <= self.end_date)

        trading_dates = reference_df[mask]['Date']

        return trading_dates

    def _save_checkpoint(self, samples: List[Dict], current_date: datetime):
        """Save progress checkpoint"""
        df = pd.DataFrame(samples)
        checkpoint_path = f"data/ml_training/checkpoints/checkpoint_{current_date.strftime('%Y%m%d')}.parquet"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        df.to_parquet(checkpoint_path)
