# File: core/local_file_loader.py
# Part 1 of 2
"""
Local File System Data Loader
Handles reading and updating stock data from local CSV files
All dates use Singapore format: D/M/YYYY (dayfirst=True)
FIXED: Proper date validation, cleanup of erroneous dates, and metadata column handling
ENHANCED: Force update capability to re-process latest EOD file
NEW: yfinance download capability for filling date gaps
FIXED: Volume scaling - yfinance volumes divided by 1000 to match abbreviated EOD format
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import glob

logger = logging.getLogger(__name__)

class LocalFileLoader:
    """Load and manage stock data from local file system"""
    
    def __init__(self, historical_path: str = None, eod_path: str = None):
        """
        Initialize local file loader
        
        Args:
            historical_path: Path to Historical_Data folder (defaults to config)
            eod_path: Path to EOD_Data folder (defaults to config)
        """
        from config import HISTORICAL_DATA_PATH, EOD_DATA_PATH
        
        self.historical_path = historical_path or HISTORICAL_DATA_PATH
        self.eod_path = eod_path or EOD_DATA_PATH
        
        # Verify folders exist
        self._verify_folders()
    
    def _verify_folders(self):
        """Verify that data folders exist, create if needed"""
        for path in [self.historical_path, self.eod_path]:
            if not os.path.exists(path):
                logger.warning(f"Creating folder: {path}")
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as e:
                    logger.error(f"Could not create folder {path}: {e}")
    
    def list_historical_files(self) -> List[str]:
        """
        List all CSV files in Historical_Data folder
        
        Returns:
            List of filenames (e.g., ['A17U.csv', 'C38U.csv'])
        """
        try:
            pattern = os.path.join(self.historical_path, '*.csv')
            files = glob.glob(pattern)
            filenames = [os.path.basename(f) for f in files]
            logger.info(f"Found {len(filenames)} historical files")
            return sorted(filenames)
        except Exception as e:
            logger.error(f"Error listing historical files: {e}")
            return []
    
    def list_eod_files(self) -> List[str]:
        """
        List all CSV files in EOD_Data folder
        
        Returns:
            List of filenames (e.g., ['01_Oct_2025.csv', '02_Oct_2025.csv'])
        """
        try:
            pattern = os.path.join(self.eod_path, '*.csv')
            files = glob.glob(pattern)
            filenames = [os.path.basename(f) for f in files]
            logger.info(f"Found {len(filenames)} EOD files")
            return sorted(filenames)
        except Exception as e:
            logger.error(f"Error listing EOD files: {e}")
            return []
    
    def load_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific ticker
        FIXED: Properly handles files with or without Code/Shortname columns
        
        Args:
            ticker: Stock ticker (e.g., 'A17U.SG')
            
        Returns:
            DataFrame with historical data or None
        """
        try:
            # Remove .SG suffix to get filename
            ticker_clean = ticker.replace('.SG', '')
            filename = f"{ticker_clean}.csv"
            filepath = os.path.join(self.historical_path, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"No historical file found for {ticker} at {filepath}")
                return None
            
            # Read CSV with Singapore date format
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Parse dates with STRICT Singapore format (dayfirst=True)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
                df.set_index('Date', inplace=True)
            
            # Standardize column names (Last→Close, Vol→Volume)
            df = self._standardize_columns(df)
            
            logger.info(f"✅ Loaded {ticker} from {filename}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading historical data for {ticker}: {e}")
            return None
    
    def get_latest_eod_file(self) -> Optional[str]:
        """
        Get the most recent EOD file by parsing dates from filenames
        
        Returns:
            Filename or None
        """
        try:
            files = self.list_eod_files()
            
            if not files:
                logger.warning("No EOD files found")
                return None
            
            # Parse dates from filenames (format: DD_MMM_YYYY.csv)
            dated_files = []
            for filename in files:
                try:
                    date_str = filename.replace('.csv', '')
                    file_date = datetime.strptime(date_str, '%d_%b_%Y')
                    dated_files.append((file_date, filename))
                except ValueError:
                    logger.warning(f"Could not parse date from filename: {filename}")
                    continue
            
            if not dated_files:
                return None
            
            # Sort by date and get latest
            dated_files.sort(key=lambda x: x[0], reverse=True)
            latest_file = dated_files[0][1]
            
            logger.info(f"Latest EOD file: {latest_file}")
            return latest_file
            
        except Exception as e:
            logger.error(f"Error getting latest EOD file: {e}")
            return None
    
    def load_eod_data(self, target_date: Optional[date] = None) -> Dict[str, pd.DataFrame]:
        """
        Load EOD data for all stocks
        
        Args:
            target_date: Specific date to load (default: most recent)
            
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        try:
            # Find target file
            if target_date:
                date_str = target_date.strftime('%d_%b_%Y')
                filename = f"{date_str}.csv"
            else:
                filename = self.get_latest_eod_file()
                if filename is None:
                    return {}
            
            filepath = os.path.join(self.eod_path, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"EOD file not found: {filepath}")
                return {}
            
            logger.info(f"Loading EOD data from {filename}")
            
            # Read EOD file
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Add Date column from filename
            date_str = self.parse_eod_filename_to_date(filename)
            if date_str:
                df['Date'] = date_str
            
            # Split into separate DataFrames per ticker
            stock_data = self._split_eod_data(df)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error loading EOD data: {e}")
            return {}
    
    def parse_eod_filename_to_date(self, filename: str) -> str:
        """
        Parse EOD filename to Singapore date format (D/M/YYYY)
        
        Args:
            filename: e.g., "01_Oct_2025.csv"
            
        Returns:
            Date string in D/M/YYYY format, e.g., "1/10/2025"
        """
        try:
            date_str = filename.replace('.csv', '')
            date_obj = datetime.strptime(date_str, '%d_%b_%Y')
            # Format as D/M/YYYY (no leading zeros)
            return date_obj.strftime('%-d/%-m/%Y') if os.name != 'nt' else date_obj.strftime('%#d/%#m/%Y')
        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")
            return None
    
    def _split_eod_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split EOD DataFrame into separate DataFrames per ticker
        
        Args:
            df: Combined DataFrame with all stocks
            
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        stock_data = {}
        
        try:
            if 'Code' not in df.columns:
                logger.error("No 'Code' column found in EOD data")
                return {}
            
            # Group by ticker
            for ticker in df['Code'].unique():
                ticker_df = df[df['Code'] == ticker].copy()
                
                # Process dates
                if 'Date' in ticker_df.columns:
                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], dayfirst=True)
                    ticker_df.set_index('Date', inplace=True)
                
                # Standardize columns
                ticker_df = self._standardize_columns(ticker_df)
                
                stock_data[ticker] = ticker_df
            
            logger.info(f"Split EOD data into {len(stock_data)} stocks")
            
        except Exception as e:
            logger.error(f"Error splitting EOD data: {e}")
        
        return stock_data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names
        
        EOD: Code, Shortname, Open, High, Low, Last, Volume
        Historical: Date, Code, Shortname, Open, High, Low, Close, Vol
        Target: Open, High, Low, Close, Volume
        """
        column_mapping = {
            'Last': 'Close',
            'Vol': 'Volume',
            'volume': 'Volume',
            'close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure Volume is integer
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(float).astype(int)
        
        return df
    
    def get_last_date_in_historical(self, ticker: str) -> Optional[datetime]:
        """
        Get the last date in a historical CSV file
        
        Args:
            ticker: Stock ticker (e.g., 'A17U.SG')
            
        Returns:
            Last date as datetime or None
        """
        try:
            df = self.load_historical_data(ticker)
            
            if df is None or df.empty:
                return None
            
            return df.index[-1]
            
        except Exception as e:
            logger.error(f"Error getting last date for {ticker}: {e}")
            return None
    
    def get_current_working_day(self) -> date:
        """
        Get current working day (today if weekday, else last Friday)
        
        Returns:
            date object representing current working day
        """
        today = date.today()
        weekday = today.weekday()  # Monday=0, Sunday=6
        
        if weekday <= 4:  # Monday to Friday
            return today
        elif weekday == 5:  # Saturday
            return today - timedelta(days=1)  # Friday
        else:  # Sunday
            return today - timedelta(days=2)  # Friday
    
    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[datetime], bool, Optional[date], Optional[date]]:
        """
        Check if Historical_Data needs updating from EOD_Data OR if gap exists for yfinance download
        
        Returns:
            Tuple of (eod_available, eod_filename, eod_date, gap_exists, last_historical_date, current_working_day)
        """
        try:
            # Get current working day
            current_working_day = self.get_current_working_day()
            
            # Get latest EOD file
            latest_eod = self.get_latest_eod_file()
            
            eod_date = None
            eod_available = False
            
            if latest_eod:
                # Parse EOD date
                eod_date_str = latest_eod.replace('.csv', '')
                eod_date = datetime.strptime(eod_date_str, '%d_%b_%Y')
            
            # Load EOD file to get first ticker
            if latest_eod:
                filepath = os.path.join(self.eod_path, latest_eod)
                eod_df = pd.read_csv(filepath, encoding='utf-8')
                
                if 'Code' in eod_df.columns:
                    sample_ticker = eod_df['Code'].iloc[0]
                else:
                    sample_ticker = None
            else:
                sample_ticker = None
            
            # Get last date in historical
            last_hist_date = None
            if sample_ticker:
                last_hist_date = self.get_last_date_in_historical(sample_ticker)
            
            # Check if EOD update is available
            if eod_date and last_hist_date:
                eod_available = eod_date.date() > last_hist_date.date()
            elif eod_date and not last_hist_date:
                eod_available = True  # No historical data, so EOD is "new"
            
            # Check if gap exists (current working day > last historical date)
            gap_exists = False
            if last_hist_date:
                gap_exists = current_working_day > last_hist_date.date()
            else:
                gap_exists = True  # No historical data means gap exists
            
            logger.info(f"Update check: EOD available={eod_available}, Gap exists={gap_exists}")
            logger.info(f"  Current working day: {current_working_day}")
            logger.info(f"  Last historical: {last_hist_date.date() if last_hist_date else 'None'}")
            logger.info(f"  EOD date: {eod_date.date() if eod_date else 'None'}")
            
            return (
                eod_available,
                latest_eod,
                eod_date,
                gap_exists,
                last_hist_date.date() if last_hist_date else None,
                current_working_day
            )
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return False, None, None, False, None, None
        
# File: core/local_file_loader.py
# Part 2 of 2

    def download_missing_dates_from_yfinance(self, start_date: date, end_date: date, force_mode: bool = False) -> Dict[str, any]:
        """
        Download missing dates from yfinance and append to Historical_Data
        FIXED: Divides yfinance volumes by 1000 to match abbreviated EOD format
        
        Args:
            start_date: Start of missing date range
            end_date: End of missing date range (inclusive)
            
        Returns:
            Statistics dictionary with success/failure counts
        """
        stats = {
            'total_stocks': 0,
            'updated': 0,
            'failed': 0,
            'skipped': 0,
            'total_dates_added': 0,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'force_mode': force_mode,
            'details': []
        }
        
        try:
            import yfinance as yf
            
            # Get watchlist
            from utils.watchlist import get_active_watchlist
            tickers = get_active_watchlist()
            
            stats['total_stocks'] = len(tickers)
            
            logger.info(f"Starting yfinance download: {start_date} to {end_date} for {len(tickers)} stocks")
            
            # Download for each ticker
            for ticker in tickers:
                try:
                    result = self._download_and_append_single_stock(ticker, start_date, end_date, force_mode=force_mode)
                    
                    stats['details'].append(result)
                    
                    if result['status'] == 'updated':
                        stats['updated'] += 1
                        stats['total_dates_added'] += result.get('dates_added', 0)
                    elif result['status'] == 'skipped':
                        stats['skipped'] += 1
                    elif result['status'] == 'failed':
                        stats['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error downloading {ticker}: {e}")
                    stats['failed'] += 1
                    stats['details'].append({
                        'ticker': ticker,
                        'status': 'failed',
                        'message': str(e)
                    })
            
            logger.info(f"yfinance download complete: {stats['updated']} updated, {stats['failed']} failed, {stats['skipped']} skipped")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in download_missing_dates_from_yfinance: {e}")
            stats['error'] = str(e)
            return stats
    
    def _download_and_append_single_stock(self, ticker: str, start_date: date, end_date: date, force_mode: bool = False) -> Dict:
        """
        Download data for a single stock and append to its Historical_Data CSV
        FIXED: Divides yfinance volumes by 1000 to match abbreviated EOD format
        
        Args:
            ticker: Stock ticker (e.g., 'A17U.SG')
            start_date: Start date for download
            end_date: End date for download
            
        Returns:
            Dictionary with download result
        """
        import yfinance as yf
        
        result = {
            'ticker': ticker,
            'status': 'unknown',
            'message': '',
            'dates_added': 0
        }
        
        try:
            # Skip invalid tickers
            if pd.isna(ticker) or ticker == '' or str(ticker).lower() == 'nan':
                result['status'] = 'skipped'
                result['message'] = 'Invalid ticker'
                return result
            
            # Load existing historical data
            ticker_clean = ticker.replace('.SG', '')
            filename = f"{ticker_clean}.csv"
            filepath = os.path.join(self.historical_path, filename)
            
            # Check if file exists and get last date
            if os.path.exists(filepath):
                existing_df = pd.read_csv(filepath, encoding='utf-8')
                
                # Parse dates
                existing_df['Date'] = pd.to_datetime(existing_df['Date'], dayfirst=True, format='mixed')
                
                # Get actual last date in this file
                last_date_in_file = existing_df['Date'].max().date()
                
                # In force mode, use the requested start_date regardless
                # In normal mode, adjust start_date to be day after last date in file
                if force_mode:
                    actual_start_date = start_date
                    logger.info(f"FORCE MODE: Downloading from {actual_start_date} (may overwrite existing)")
                else:
                    actual_start_date = last_date_in_file + timedelta(days=1)
                    
                    if actual_start_date > end_date:
                        result['status'] = 'skipped'
                        result['message'] = f'No gap - file current to {last_date_in_file}'
                        return result
                
            else:
                # File doesn't exist - will create new one
                existing_df = pd.DataFrame()
                actual_start_date = start_date
                logger.info(f"Creating new historical file for {ticker}")
            
            # Download from yfinance
            # Convert to yfinance format (A17U.SG → A17U.SI for Singapore)
            yf_ticker = ticker.replace('.SG', '.SI')
            
            logger.info(f"Downloading {yf_ticker} from {actual_start_date} to {end_date}")
            
            # Download data
            stock = yf.Ticker(yf_ticker)
            downloaded_df = stock.history(
                start=actual_start_date,
                end=end_date + timedelta(days=1),  # yfinance end is exclusive
                auto_adjust=False
            )
            
            if downloaded_df.empty:
                result['status'] = 'skipped'
                result['message'] = f'No data available from yfinance for {actual_start_date} to {end_date}'
                return result
            
            # Convert downloaded data to our format
            new_df = pd.DataFrame()
            
            # Reset index to get Date as column
            downloaded_df = downloaded_df.reset_index()
            
            # Convert Date to Singapore format (D/M/YYYY)
            downloaded_df['Date'] = downloaded_df['Date'].dt.strftime('%-d/%-m/%Y' if os.name != 'nt' else '%#d/%#m/%Y')
            
            # Build dataframe with correct column order
            new_df['Date'] = downloaded_df['Date']
            new_df['Code'] = ticker
            
            # Get company name (shortname)
            if not existing_df.empty and 'Shortname' in existing_df.columns:
                shortname = existing_df['Shortname'].iloc[0]
            else:
                shortname = ticker_clean
            
            new_df['Shortname'] = shortname
            new_df['Open'] = downloaded_df['Open'].round(4)
            new_df['High'] = downloaded_df['High'].round(4)
            new_df['Low'] = downloaded_df['Low'].round(4)
            new_df['Close'] = downloaded_df['Close'].round(4)
            
            # CRITICAL FIX: Divide yfinance volumes by 1000 to match abbreviated EOD format
            new_df['Vol'] = (downloaded_df['Volume'] / 1000).round(0).astype(int)
            
            logger.info(f"✅ VOLUME FIX APPLIED: Divided yfinance volumes by 1000 for {ticker}")
            
            # Filter: Remove any dates that might overlap (safety check)
            # In force mode, we REMOVE existing overlapping dates from historical, not from new data
            if not existing_df.empty:
                if force_mode:
                    # Force mode: Remove dates from existing data that overlap with new download
                    new_df['Date_dt'] = pd.to_datetime(new_df['Date'], dayfirst=True)
                    new_dates = new_df['Date_dt'].dt.date
                    
                    # Keep only existing dates that don't overlap with new download
                    existing_df = existing_df[~existing_df['Date'].isin(new_dates)]
                    
                    new_df = new_df.drop(columns=['Date_dt'])
                    logger.info(f"FORCE MODE: Removed overlapping dates from existing data for {ticker}")
                else:
                    # Normal mode: Skip dates that already exist
                    new_df['Date_dt'] = pd.to_datetime(new_df['Date'], dayfirst=True)
                    new_df = new_df[new_df['Date_dt'] > last_date_in_file]
                    new_df = new_df.drop(columns=['Date_dt'])
                    
                    if new_df.empty:
                        result['status'] = 'skipped'
                        result['message'] = 'No new dates after filtering'
                        return result
            
            # Sort new data by date
            new_df['Date_dt'] = pd.to_datetime(new_df['Date'], dayfirst=True)
            new_df = new_df.sort_values('Date_dt')
            new_df = new_df.drop(columns=['Date_dt'])
            
            # Append to existing data
            if not existing_df.empty:
                # Convert existing dates back to string format for consistency
                existing_df['Date'] = existing_df['Date'].dt.strftime('%-d/%-m/%Y' if os.name != 'nt' else '%#d/%#m/%Y')
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Ensure column order
            column_order = ['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol']
            combined_df = combined_df[column_order]
            
            # Save to file
            combined_df.to_csv(filepath, index=False, encoding='utf-8')
            
            result['status'] = 'updated'
            result['dates_added'] = len(new_df)
            if force_mode:
                result['message'] = f'FORCE: Added {len(new_df)} date(s) from yfinance (volumes ÷1000, overwrote existing)'
            else:
                result['message'] = f'Added {len(new_df)} date(s) from yfinance (volumes ÷1000)'
            
            logger.info(f"✅ {ticker}: Added {len(new_df)} dates from yfinance with volume scaling")
            
            return result
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            result['status'] = 'failed'
            result['message'] = str(e)
            return result
    
    def update_historical_from_eod(self, force: bool = False) -> Dict[str, any]:
        """
        Update all Historical_Data files from latest EOD_Data file
        ENHANCED: Added force parameter to bypass date checks
        
        Args:
            force: If True, update regardless of date comparisons
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_stocks': 0,
            'updated': 0,
            'skipped': 0,
            'created': 0,
            'errors': 0,
            'cleaned': 0,
            'forced': force,
            'eod_date': None,
            'details': []
        }
        
        try:
            # Get latest EOD file
            latest_eod = self.get_latest_eod_file()
            if not latest_eod:
                return stats
            
            filepath = os.path.join(self.eod_path, latest_eod)
            eod_df = pd.read_csv(filepath, encoding='utf-8')
            
            if 'Code' not in eod_df.columns:
                return stats
            
            # Parse EOD date
            eod_date_str = self.parse_eod_filename_to_date(latest_eod)
            eod_date_obj = datetime.strptime(latest_eod.replace('.csv', ''), '%d_%b_%Y')
            stats['eod_date'] = eod_date_str
            
            log_msg = f"Starting {'FORCED' if force else 'normal'} update from {latest_eod} ({eod_date_str})"
            logger.info(log_msg)
            
            # Process each stock
            tickers = eod_df['Code'].unique()
            stats['total_stocks'] = len(tickers)
            
            for ticker in tickers:
                try:
                    result = self._update_single_stock(ticker, eod_df, eod_date_str, eod_date_obj, force=force)
                    
                    stats['details'].append(result)
                    
                    if result['status'] == 'updated':
                        stats['updated'] += 1
                    elif result['status'] == 'skipped':
                        stats['skipped'] += 1
                    elif result['status'] == 'created':
                        stats['created'] += 1
                    elif result['status'] == 'error':
                        stats['errors'] += 1
                    
                    # Track cleaned rows
                    if 'cleaned' in result.get('message', ''):
                        stats['cleaned'] += 1
                        
                except Exception as e:
                    logger.error(f"Error updating {ticker}: {e}")
                    stats['errors'] += 1
                    stats['details'].append({
                        'ticker': ticker,
                        'status': 'error',
                        'message': str(e)
                    })
            
            logger.info(f"Update complete: {stats['updated']} updated, {stats['created']} created, "
                       f"{stats['skipped']} skipped, {stats['errors']} errors, {stats['cleaned']} cleaned")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in update_historical_from_eod: {e}")
            return stats
    
    def _update_single_stock(self, ticker: str, eod_df: pd.DataFrame, 
                            eod_date_str: str, eod_date_obj: datetime, force: bool = False) -> Dict:
        """
        Update historical data for a single stock
        ENHANCED: Added force parameter to bypass date validation
        FIXED: Handle NaN ticker values
        
        Args:
            ticker: Stock ticker (e.g., 'A17U.SG')
            eod_df: EOD DataFrame containing all stocks
            eod_date_str: Date string in D/M/YYYY format
            eod_date_obj: Date as datetime object
            force: If True, bypass date comparison checks
            
        Returns:
            Dictionary with update result
        """
        result = {
            'ticker': ticker,
            'status': 'unknown',
            'message': ''
        }
        
        try:
            # FIXED: Skip if ticker is NaN or empty
            if pd.isna(ticker) or ticker == '' or str(ticker).lower() == 'nan':
                result['status'] = 'skipped'
                result['message'] = 'Invalid ticker (NaN or empty)'
                logger.warning(f"Skipping invalid ticker: {ticker}")
                return result
            
            # Get ticker's row from EOD data
            ticker_rows = eod_df[eod_df['Code'] == ticker]
            
            if len(ticker_rows) == 0:
                result['status'] = 'error'
                result['message'] = 'Ticker not found in EOD data'
                logger.warning(f"Ticker {ticker} not found in EOD data")
                return result
            
            eod_row = ticker_rows.iloc[0]
            
            # Load existing historical data
            ticker_clean = ticker.replace('.SG', '')
            filename = f"{ticker_clean}.csv"
            filepath = os.path.join(self.historical_path, filename)
            
            file_existed = os.path.exists(filepath)
            
            if file_existed:
                # Load existing file
                hist_df = pd.read_csv(filepath, encoding='utf-8')
                
                # Check if Code and Shortname columns exist
                needs_metadata_columns = 'Code' not in hist_df.columns or 'Shortname' not in hist_df.columns
                
                if needs_metadata_columns:
                    logger.info(f"Adding Code and Shortname columns to {ticker}")
                    if 'Code' not in hist_df.columns:
                        hist_df['Code'] = ticker
                    if 'Shortname' not in hist_df.columns:
                        hist_df['Shortname'] = eod_row.get('Shortname', ticker)
                
                # Parse dates with STRICT Singapore format (dayfirst=True)
                hist_df['Date'] = pd.to_datetime(hist_df['Date'], dayfirst=True, format='mixed')
                
                # CRITICAL FIX: Remove any rows with dates >= EOD date (cleanup bad data)
                rows_before = len(hist_df)
                hist_df = hist_df[hist_df['Date'].dt.date < eod_date_obj.date()]
                rows_after = len(hist_df)
                
                removed_rows = rows_before - rows_after
                if removed_rows > 0:
                    logger.warning(f"Removed {removed_rows} row(s) with dates >= {eod_date_obj.date()} from {ticker}")
                
                # Check if EOD date already exists
                if eod_date_obj.date() in hist_df['Date'].dt.date.values:
                    if not force:
                        result['status'] = 'skipped'
                        result['message'] = 'Date already exists'
                        return result
                    else:
                        # Force mode: Remove existing date entry
                        hist_df = hist_df[hist_df['Date'].dt.date != eod_date_obj.date()]
                        logger.info(f"FORCE: Removed existing {eod_date_obj.date()} entry from {ticker}")
                
                # Validate: Check if we're actually adding a newer date (skip in force mode)
                if not force and len(hist_df) > 0:
                    last_hist_date = hist_df['Date'].max().date()
                    if eod_date_obj.date() <= last_hist_date:
                        result['status'] = 'skipped'
                        result['message'] = f'EOD date {eod_date_obj.date()} not newer than last historical date {last_hist_date}'
                        return result
                
            else:
                # Create new DataFrame (auto-create missing historical files)
                hist_df = pd.DataFrame()
                needs_metadata_columns = False
                removed_rows = 0
                logger.info(f"Creating new historical file for {ticker}")
            
            # Create new row from EOD data with FULL column set
            new_row = pd.DataFrame([{
                'Date': eod_date_obj,  # Use datetime object for consistency
                'Code': ticker,
                'Shortname': eod_row.get('Shortname', ticker),
                'Open': float(eod_row['Open']),
                'High': float(eod_row['High']),
                'Low': float(eod_row['Low']),
                'Close': float(eod_row.get('Last', eod_row.get('Close', 0))),
                'Vol': int(float(eod_row['Volume']))
            }])
            
            # Append new row
            hist_df = pd.concat([hist_df, new_row], ignore_index=True)
            
            # Sort by date to ensure chronological order
            hist_df = hist_df.sort_values('Date')
            
            # Validate: Check for date gaps (warn if gap > 5 days)
            if len(hist_df) > 1:
                hist_df_sorted = hist_df.sort_values('Date')
                date_diffs = hist_df_sorted['Date'].diff()
                max_gap = date_diffs.max()
                if pd.notna(max_gap) and max_gap.days > 5:
                    logger.warning(f"{ticker}: Largest date gap is {max_gap.days} days")
            
            # Format dates back to D/M/YYYY (Singapore format, no leading zeros)
            hist_df['Date'] = hist_df['Date'].dt.strftime('%-d/%-m/%Y') if os.name != 'nt' else hist_df['Date'].dt.strftime('%#d/%#m/%Y')
            
            # Ensure column order: Date, Code, Shortname, Open, High, Low, Close, Vol
            column_order = ['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol']
            hist_df = hist_df[column_order]
            
            # Save to file
            hist_df.to_csv(filepath, index=False, encoding='utf-8')
            
            # Build status message
            result['status'] = 'updated' if file_existed else 'created'
            result['message'] = 'Added 1 row' if not force else 'FORCED: Added 1 row'
            
            if file_existed:
                if needs_metadata_columns:
                    result['message'] += ' + added Code/Shortname'
                if removed_rows > 0:
                    result['message'] += f' + cleaned {removed_rows} bad row(s)'
            else:
                result['message'] = 'Created with 1 row'
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
            result['status'] = 'error'
            result['message'] = str(e)
            return result
    
    def get_watchlist_from_eod(self) -> List[str]:
        """
        Extract watchlist from latest EOD file
        
        Returns:
            List of ticker strings (e.g., ['A17U.SG', 'C38U.SG'])
        """
        try:
            latest_eod = self.get_latest_eod_file()
            
            if not latest_eod:
                logger.warning("No EOD file found for watchlist extraction")
                return []
            
            filepath = os.path.join(self.eod_path, latest_eod)
            eod_df = pd.read_csv(filepath, encoding='utf-8')
            
            if 'Code' not in eod_df.columns:
                logger.error("Could not extract watchlist from EOD file")
                return []
            
            watchlist = eod_df['Code'].unique().tolist()
            logger.info(f"Extracted watchlist: {len(watchlist)} stocks")
            
            return watchlist
            
        except Exception as e:
            logger.error(f"Error getting watchlist from EOD: {e}")
            return []


def get_local_loader() -> LocalFileLoader:
    """
    Get LocalFileLoader instance
    
    Returns:
        LocalFileLoader instance
    """
    return LocalFileLoader()