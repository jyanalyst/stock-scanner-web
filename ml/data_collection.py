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


class MockErrorLogger:
    """Mock error logger for scanner compatibility"""
    def __init__(self):
        self.errors = []
    
    def log_error(self, *args, **kwargs):
        """Silent error logging - scanner expects this method"""
        pass
    
    def log_performance(self, *args, **kwargs):
        """Silent performance logging - scanner expects this method too"""
        pass


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
                             resume_from: Optional[str] = None,
                             use_validation: bool = True):
        """
        Main collection loop

        Process:
        1. Get all trading dates in range
        2. Filter stocks by validation status (Ready + Partial only)
        3. For each date:
           a. Run scanner as of that date
           b. Calculate forward returns
           c. Label outcomes
           d. Save to disk
        4. Combine into single dataset

        Args:
            save_path: Directory to save training data
            resume_from: Date to resume from (for interrupted runs)
            use_validation: If True, filter stocks by validation status
        """
        # Initialize error_logger if needed (prevents scanner errors)
        # Only do this if we're actually in Streamlit mode
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and not hasattr(st.session_state, 'error_logger'):
                st.session_state.error_logger = MockErrorLogger()
        except ImportError:
            # Not in Streamlit environment, skip session state setup
            pass
        
        # Get usable stocks from validation (Ready + Partial, exclude Failed)
        usable_stocks = None
        if use_validation:
            usable_stocks = self._get_usable_stocks()
            if usable_stocks:
                self.logger.info(f"✅ Using {len(usable_stocks['all'])} validated stocks for training:")
                self.logger.info(f"   - Ready: {len(usable_stocks['ready'])} stocks (95%+ coverage)")
                self.logger.info(f"   - Partial: {len(usable_stocks['partial'])} stocks (80-95% coverage)")
                self.logger.info(f"   - Excluded: {len(usable_stocks['failed'])} failed stocks (<80% coverage)")
            else:
                self.logger.warning("⚠️ Validation not available - using all stocks from watchlist")
        
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
                
                # Filter to usable stocks only (if validation was used)
                if usable_stocks and 'all' in usable_stocks:
                    original_count = len(scan_results)
                    scan_results = scan_results[scan_results['Ticker'].isin(usable_stocks['all'])]
                    filtered_count = len(scan_results)
                    
                    if i == 0:  # Log on first iteration
                        self.logger.info(f"Filtered scan results: {original_count} → {filtered_count} stocks")

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

                # Update progress in session state for UI (only if in Streamlit mode)
                try:
                    import streamlit as st
                    if hasattr(st, 'session_state'):
                        st.session_state.collection_progress = (i + 1) / len(trading_dates) * 100
                        st.session_state.current_date = date.strftime('%Y-%m-%d')
                except ImportError:
                    # Not in Streamlit environment, skip progress updates
                    pass

            except Exception as e:
                self.logger.error(f"Error on {date}: {e}")
                continue

        # Final save - ensure directory exists
        os.makedirs(save_path, exist_ok=True)
        final_df = pd.DataFrame(all_samples)
        final_df.to_parquet(f"{save_path}/training_data_complete.parquet")
        
        self.logger.info(f"✅ Collection complete! Saved {len(final_df)} samples to {save_path}")

        return final_df

    def _run_historical_scan(self, target_date: datetime) -> pd.DataFrame:
        """
        Run scanner as of historical date

        This uses your EXISTING scanner logic but with a past date
        """
        # Temporarily suppress ALL output during scan by redirecting stderr
        import logging
        import warnings
        import sys
        import os
        from contextlib import redirect_stderr, redirect_stdout

        # Save current logging levels for ALL loggers
        original_levels = {}

        # Get ALL active loggers and silence them
        all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        all_loggers.append(logging.getLogger())  # Root logger

        for logger in all_loggers:
            if hasattr(logger, 'level'):
                original_levels[logger.name or 'root'] = logger.level
                logger.setLevel(logging.CRITICAL)

        # Also silence specific known noisy loggers
        noisy_loggers = [
            'streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner_utils',
            'streamlit.runtime.state', 'streamlit.runtime.caching',
            'pages.scanner.logic', 'pages.scanner.data', 'core.local_file_loader',
            'core.technical_analysis', 'CrossStockRankings', 'AnalystReports',
            'EarningsReports', 'EarningsReaction', 'Scanner', 'Unknown'
        ]

        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.CRITICAL)

        # Suppress warnings during scan - more comprehensive
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
            warnings.filterwarnings('ignore', category=DeprecationWarning, module='streamlit')
            warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
            warnings.filterwarnings('ignore', message='.*Session state does not function.*')
            warnings.filterwarnings('ignore', message='.*No runtime found.*')
            warnings.filterwarnings('ignore', message='.*Thread.*missing ScriptRunContext.*')
            warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')

            # Redirect stderr to /dev/null (or equivalent) to suppress warnings
            with open(os.devnull, 'w') as devnull:
                with redirect_stderr(devnull):
                    # Also redirect stdout to suppress any remaining output
                    with redirect_stdout(devnull):
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

                        # Ensure scan_results is a DataFrame
                        if scan_results is None:
                            self.logger.error(f"Scanner returned None for {target_date.strftime('%Y-%m-%d')}")
                            return None

        # Restore original logging levels
        for logger_name, level in original_levels.items():
            if logger_name == 'root':
                logging.getLogger().setLevel(level)
            else:
                logging.getLogger(logger_name).setLevel(level)
        
        # CRITICAL FIX: Ensure Date column exists (scanner may set it as index)
        if scan_results is not None and not scan_results.empty:
            if 'Date' not in scan_results.columns:
                # Reset index to get Date back as column
                scan_results = scan_results.reset_index()

                # Find datetime column and rename to 'Date' if needed
                for col in scan_results.columns:
                    if col != 'Date' and pd.api.types.is_datetime64_any_dtype(scan_results[col]):
                        scan_results = scan_results.rename(columns={col: 'Date'})
                        break

                # Final check
                if 'Date' not in scan_results.columns:
                    self.logger.error(f"FAILED to create Date column! Final columns: {list(scan_results.columns)}")

        return scan_results

    def _get_future_trading_date(self, base_date: datetime, trading_days: int, 
                                stock_df: pd.DataFrame) -> Optional[datetime]:
        """
        Get future date that is N TRADING days ahead (not calendar days)
        
        CRITICAL FIX: This replaces calendar day arithmetic which caused
        weekend signals to have compressed holding periods.
        
        Args:
            base_date: Starting date (signal date)
            trading_days: Number of trading days ahead (e.g., 2, 3, 4)
            stock_df: Stock historical data with Date column
        
        Returns:
            Future trading date or None if insufficient data
            
        Example:
            Signal on Friday Nov 22, 2024
            trading_days=2 → Returns Tuesday Nov 26, 2024 (not Sunday Nov 24)
        """
        try:
            # CRITICAL FIX: Make a copy to avoid mutating cached data!
            stock_df = stock_df.copy()
            
            # Check if Date is index BEFORE trying to access it
            if 'Date' not in stock_df.columns:
                stock_df = stock_df.reset_index()
            
            # Ensure Date column is datetime
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            
            # Get all trading dates AFTER base_date (strictly greater than)
            future_dates = stock_df[stock_df['Date'] > base_date]['Date'].sort_values()
            
            # Check if we have enough future data
            if len(future_dates) < trading_days:
                return None
            
            # Return the Nth trading day (0-indexed, so subtract 1)
            return future_dates.iloc[trading_days - 1]
            
        except Exception as e:
            self.logger.error(f"Error getting future trading date: {e}")
            return None

    def _calculate_forward_returns(self,
                                   scan_results: pd.DataFrame,
                                   entry_date: datetime) -> List[Dict]:
        """
        Calculate forward returns for each stock in scan

        OPTIMIZED: Cache stock data to avoid 4x file loading per stock

        CONVENTION: Signal Day = Day 0
        - entry_date = Day 0 (signal fires at close)
        - entry_price = Close price on Day 0
        - exit_date = Day 0 + N TRADING days (FIXED: was calendar days)
        - return_Nd = (Price_DayN - Price_Day0) / Price_Day0

        Example:
            Signal: Friday Nov 22 @ $3.39 (Day 0)
            return_2d: Tuesday Nov 26 @ $3.45 = +1.77% (2 TRADING days)
            return_3d: Wednesday Nov 27 @ $3.48 = +2.65% (3 TRADING days)
            return_4d: Thursday Nov 28 @ $3.50 = +3.24% (4 TRADING days)

        Returns list of labeled samples
        """
        # Handle None or empty scan results
        if scan_results is None or len(scan_results) == 0:
            self.logger.warning(f"No scan results for {entry_date.strftime('%Y-%m-%d')}")
            return []

        from core.local_file_loader import get_local_loader

        loader = get_local_loader()
        labeled_samples = []

        # OPTIMIZATION: Cache stock data to avoid 4x loading per stock
        stock_data_cache = {}

        def get_cached_stock_data(ticker: str):
            """Get stock data with caching to avoid repeated file I/O"""
            if ticker not in stock_data_cache:
                try:
                    stock_data_cache[ticker] = loader.load_historical_data(ticker)
                except Exception as e:
                    self.logger.warning(f"Failed to load {ticker}: {e}")
                    stock_data_cache[ticker] = None
            return stock_data_cache[ticker]

        for _, stock in scan_results.iterrows():
            ticker = stock['Ticker']
            entry_price = stock['Close']

            # Get cached stock data (loaded once per stock per date)
            stock_df = get_cached_stock_data(ticker)
            if stock_df is None:
                continue

            # Calculate returns for each forward period
            forward_returns = {}

            for days in self.forward_days:
                # CRITICAL FIX: Use actual trading days, not calendar days
                exit_date = self._get_future_trading_date(entry_date, days, stock_df)
                
                if exit_date is None:
                    # Insufficient future data (e.g., signal too close to present)
                    forward_returns[f'return_{days}d'] = None
                    forward_returns[f'win_{days}d'] = None
                    continue

                # Get future price using cached data
                exit_price = self._get_price_on_date_cached(
                    ticker,
                    exit_date,
                    stock_df
                )

                if exit_price is not None:
                    ret = (exit_price - entry_price) / entry_price
                    forward_returns[f'return_{days}d'] = ret
                    forward_returns[f'win_{days}d'] = ret > 0
                else:
                    forward_returns[f'return_{days}d'] = None
                    forward_returns[f'win_{days}d'] = None

            # Calculate max drawdown during holding period using cached data
            max_dd = self._calculate_max_drawdown_cached(
                ticker,
                entry_date,
                entry_date + timedelta(days=max(self.forward_days)),
                entry_price,
                stock_df
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

        # ⭐ NEW: Convert to DataFrame for cross-sectional ranking
        labeled_samples_df = pd.DataFrame(labeled_samples)
        
        # ⭐ NEW: Add cross-sectional ranks (peer comparison)
        if len(labeled_samples_df) > 1:
            self.logger.info(f"📊 Adding CS ranks for {len(labeled_samples_df)} stocks on {entry_date.strftime('%Y-%m-%d')}")
            labeled_samples_df = add_cross_sectional_percentiles(labeled_samples_df)
        else:
            self.logger.info(f"⚠️  Only 1 stock on {entry_date.strftime('%Y-%m-%d')}, skipping CS ranks")
        
        # Convert back to list of dicts
        return labeled_samples_df.to_dict('records')

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

    def _get_price_on_date_cached(self, ticker: str, date: datetime, stock_df: pd.DataFrame) -> float:
        """Get closing price on specific date using cached stock data"""
        try:
            if stock_df is None or stock_df.empty:
                return None

            # CRITICAL FIX: Make a copy to avoid mutating cached data!
            stock_df = stock_df.copy()
            
            # Ensure Date column exists and is datetime
            if 'Date' not in stock_df.columns:
                stock_df = stock_df.reset_index()
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])

            # Find closest trading day
            df = stock_df[stock_df['Date'] <= date]
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

    def _calculate_max_drawdown_cached(self, ticker, start_date, end_date,
                                       entry_price, stock_df: pd.DataFrame) -> float:
        """Calculate max drawdown during holding period using cached stock data"""
        try:
            if stock_df is None or stock_df.empty:
                return 0

            # CRITICAL FIX: Make a copy to avoid mutating cached data!
            stock_df = stock_df.copy()
            
            # Ensure Date column exists and is datetime
            if 'Date' not in stock_df.columns:
                stock_df = stock_df.reset_index()
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])

            # Filter to holding period
            mask = (stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)
            period_df = stock_df[mask]

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
        
        # CRITICAL FIX: Ensure Date is a column, not index
        if 'Date' not in reference_df.columns:
            reference_df = reference_df.reset_index()
        
        reference_df['Date'] = pd.to_datetime(reference_df['Date'])

        # Filter to date range
        mask = (reference_df['Date'] >= self.start_date) & \
               (reference_df['Date'] <= self.end_date)

        trading_dates = reference_df[mask]['Date']

        return trading_dates

    def _get_usable_stocks(self) -> Optional[Dict[str, List[str]]]:
        """
        Get usable stocks from validation results
        Returns Ready + Partial stocks, excludes Failed stocks
        
        Returns:
            Dictionary with 'ready', 'partial', 'failed', and 'all' stock lists
            or None if validation not available
        """
        try:
            # Check if validation results exist in session state
            if hasattr(st, 'session_state') and 'validation_results' in st.session_state:
                validation = st.session_state.validation_results
                
                ready_stocks = validation.get('ready', [])
                partial_stocks = validation.get('partial', [])
                failed_stocks = validation.get('failed', [])
                
                # Combine Ready + Partial for training
                usable_stocks = ready_stocks + partial_stocks
                
                return {
                    'ready': ready_stocks,
                    'partial': partial_stocks,
                    'failed': failed_stocks,
                    'all': usable_stocks
                }
            else:
                # Try to run validation if not available
                self.logger.info("Running validation to determine usable stocks...")
                from ml.data_validator import MLDataValidator
                
                validator = MLDataValidator(
                    start_date=self.start_date.strftime('%Y-%m-%d'),
                    end_date=self.end_date.strftime('%Y-%m-%d'),
                    min_days=100
                )
                
                validation = validator.validate_all_stocks()
                
                ready_stocks = validation.get('ready', [])
                partial_stocks = validation.get('partial', [])
                failed_stocks = validation.get('failed', [])
                
                usable_stocks = ready_stocks + partial_stocks
                
                return {
                    'ready': ready_stocks,
                    'partial': partial_stocks,
                    'failed': failed_stocks,
                    'all': usable_stocks
                }
                
        except Exception as e:
            self.logger.error(f"Could not get validation results: {e}")
            return None

    def _save_checkpoint(self, samples: List[Dict], current_date: datetime):
        """Save progress checkpoint"""
        df = pd.DataFrame(samples)
        checkpoint_path = f"data/ml_training/checkpoints/checkpoint_{current_date.strftime('%Y%m%d')}.parquet"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        df.to_parquet(checkpoint_path)


# Global flag to track if we've already printed the CS rank summary
_cs_rank_summary_printed = False

def add_cross_sectional_percentiles(df):
    """
    Add cross-sectional percentile ranks for peer comparison
    
    CRITICAL: Compares stocks to PEERS on SAME date, not own history
    
    Args:
        df: DataFrame with entry_date, Ticker, and feature columns
        
    Returns:
        DataFrame with added *_CS_Rank columns (0-100 scale)
    """
    import pandas as pd
    global _cs_rank_summary_printed
    
    if 'entry_date' not in df.columns:
        raise ValueError("entry_date required for cross-sectional ranks")
    
    # PHASE 1: Three-Indicator System (CRITICAL - Must Add)
    three_indicator = [
        'MPI_Percentile',  # Core indicator #1
        'IBS_Percentile',  # Core indicator #2
        'VPI_Percentile',  # Core indicator #3
    ]
    
    # PHASE 1: Acceleration Metrics (HIGH - Should Add)
    acceleration = [
        'IBS_Accel',       # Most important acceleration
        'RVol_Accel',      # Volume acceleration
        'RRange_Accel',    # Range acceleration
        'VPI_Accel',       # VPI acceleration
    ]
    
    # PHASE 2: Divergence (MEDIUM - Nice to Add)
    divergence = [
        'Flow_Price_Gap',  # Divergence magnitude
    ]
    
    # Combine features to rank (Phase 1 + Phase 2)
    features_to_rank = three_indicator + acceleration + divergence
    
    # Silent operation - only print summary once
    missing_features = []
    added_features = []
    
    for feature in features_to_rank:
        if feature not in df.columns:
            missing_features.append(feature)
            continue
        
        # Rank within each date (percentile rank 0-100)
        cs_rank_name = f'{feature}_CS_Rank'
        df[cs_rank_name] = df.groupby('entry_date')[feature].rank(pct=True) * 100
        
        # Round for readability
        df[cs_rank_name] = df[cs_rank_name].round(1)
        added_features.append(cs_rank_name)
    
    # Fill NaN with neutral value (50.0 = median)
    cs_rank_cols = [col for col in df.columns if col.endswith('_CS_Rank')]
    for col in cs_rank_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            df[col] = df[col].fillna(50.0)
    
    # Print summary only once (first call)
    if not _cs_rank_summary_printed:
        print(f"\n📊 Cross-Sectional Ranks: {len(added_features)}/{len(features_to_rank)} features")
        if missing_features:
            print(f"   ⚠️  Missing: {', '.join(missing_features)}")
        _cs_rank_summary_printed = True
    
    return df
