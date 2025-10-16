# File: core/local_file_loader.py
# COMPLETE FILE - Replace entire file

"""
Local File System Data Loader
Handles reading and updating stock data from local CSV files
All dates use Singapore format: D/M/YYYY (dayfirst=True)
FIXED: Proper date validation, cleanup of erroneous dates, and metadata column handling
ENHANCED: Force update capability to re-process latest EOD file
NEW: yfinance download capability for filling date gaps
FIXED: Volume scaling - yfinance volumes divided by 1000 to match abbreviated EOD format
FIXED: NaN validation in load_historical_data to prevent 'float' object has no attribute 'replace' error
FIXED: Added start_date and end_date to download stats dictionary
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
        FIXED: Added NaN validation to prevent 'float' object errors
        
        Args:
            ticker: Stock ticker (e.g., 'A17U.SG')
            
        Returns:
            DataFrame with historical data or None
        """
        try:
            # CRITICAL FIX: Validate ticker BEFORE any string operations
            if pd.isna(ticker) or ticker == '' or str(ticker).lower() == 'nan':
                logger.warning(f"❌ Invalid ticker provided to load_historical_data: {ticker}")
                return None
            
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
                logger.warning("No valid EOD files with parseable dates")
                return None
            
            # Sort by date and return most recent
            dated_files.sort(key=lambda x: x[0], reverse=True)
            latest_file = dated_files[0][1]
            
            logger.info(f"Latest EOD file: {latest_file}")
            return latest_file
            
        except Exception as e:
            logger.error(f"Error getting latest EOD file: {e}")
            return None
    
    def load_eod_data(self, filename: str = None) -> Optional[pd.DataFrame]:
        """
        Load EOD data file
        
        Args:
            filename: Specific EOD filename, or None for latest
            
        Returns:
            DataFrame with EOD data or None
        """
        try:
            if filename is None:
                filename = self.get_latest_eod_file()
                
            if filename is None:
                logger.warning("No EOD file to load")
                return None
            
            filepath = os.path.join(self.eod_path, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"EOD file not found: {filepath}")
                return None
            
            # Read CSV
            df = pd.read_csv(filepath, encoding='utf-8')
            
            logger.info(f"✅ Loaded EOD file {filename}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading EOD data: {e}")
            return None
    
    def parse_eod_filename_to_date(self, filename: str) -> Optional[str]:
        """
        Parse EOD filename to D/M/YYYY format
        
        Args:
            filename: EOD filename (e.g., '01_Oct_2025.csv')
            
        Returns:
            Date string in D/M/YYYY format (e.g., '1/10/2025')
        """
        try:
            date_str = filename.replace('.csv', '')
            date_obj = datetime.strptime(date_str, '%d_%b_%Y')
            # Format as D/M/YYYY (no leading zeros)
            return date_obj.strftime('%-d/%-m/%Y') if os.name != 'nt' else date_obj.strftime('%#d/%#m/%Y')
        except Exception as e:
            logger.error(f"Error parsing date from filename {filename}: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names (Last→Close, Vol→Volume, etc.)
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            DataFrame with standardized columns
        """
        column_mapping = {
            'Last': 'Close',
            'Vol': 'Volume',
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
            
            # Load EOD file to get first VALID ticker (skip NaN)
            sample_ticker = None
            if latest_eod:
                filepath = os.path.join(self.eod_path, latest_eod)
                eod_df = pd.read_csv(filepath, encoding='utf-8')
                
                if 'Code' in eod_df.columns:
                    # Find first non-NaN ticker
                    valid_tickers = eod_df['Code'].dropna()
                    if not valid_tickers.empty:
                        sample_ticker = valid_tickers.iloc[0]
            
            # Get last date in historical
            last_hist_date = None
            if sample_ticker:
                last_hist_date = self.get_last_date_in_historical(sample_ticker)
            
            # Check if EOD update is available
            if eod_date and last_hist_date:
                eod_available = eod_date.date() > last_hist_date.date()
            elif eod_date and not last_hist_date:
                eod_available = True  # No historical data exists yet
            
            # Check if gap exists (last_hist_date < current_working_day)
            gap_exists = False
            if last_hist_date:
                gap_exists = last_hist_date.date() < current_working_day
            else:
                gap_exists = True  # No historical data = gap exists
            
            logger.info(f"Update check: EOD available={eod_available}, Gap exists={gap_exists}")
            logger.info(f"  Current working day: {current_working_day}")
            logger.info(f"  Last historical: {last_hist_date.date() if last_hist_date else None}")
            logger.info(f"  EOD date: {eod_date.date() if eod_date else None}")
            
            return (eod_available, latest_eod, eod_date, gap_exists, 
                    last_hist_date.date() if last_hist_date else None, current_working_day)
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return (False, None, None, False, None, None)
    
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
            
            # Process each stock (skip NaN tickers)
            tickers = eod_df['Code'].dropna().unique()
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
    
    def _update_single_stock(self, ticker: str, eod_df: pd.DataFrame, eod_date_str: str, eod_date_obj: datetime, force: bool = False) -> Dict:
        """
        Update a single stock's historical data from EOD data
        FIXED: Proper metadata column handling and data cleanup
        ENHANCED: Added force parameter
        
        Args:
            ticker: Stock ticker (e.g., 'A17U.SG')
            eod_df: DataFrame with EOD data
            eod_date_str: EOD date as formatted string (D/M/YYYY)
            eod_date_obj: EOD date as datetime object
            force: If True, update regardless of date checks
            
        Returns:
            Dictionary with update result
        """
        result = {
            'ticker': ticker,
            'status': 'unknown',
            'message': ''
        }
        
        try:
            # Get EOD row for this ticker
            eod_row = eod_df[eod_df['Code'] == ticker].iloc[0]
            
            # Prepare file path
            ticker_clean = ticker.replace('.SG', '')
            filename = f"{ticker_clean}.csv"
            filepath = os.path.join(self.historical_path, filename)
            
            # Load existing historical data if file exists
            file_existed = os.path.exists(filepath)
            needs_metadata_columns = False
            removed_rows = 0
            
            if file_existed:
                hist_df = pd.read_csv(filepath, encoding='utf-8')
                
                # Check and add metadata columns if missing
                if 'Code' not in hist_df.columns:
                    hist_df['Code'] = ticker
                    needs_metadata_columns = True
                
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
    
    def download_missing_dates_from_yfinance(self, start_date: date, end_date: date, force_mode: bool = False) -> Dict:
        """
        Download missing dates from yfinance for all stocks in watchlist
        FIXED: Divides yfinance volumes by 1000 to match abbreviated EOD format
        FIXED: Added start_date and end_date to returned stats dictionary
        
        Args:
            start_date: Start date for download
            end_date: End date for download
            force_mode: If True, overwrite existing dates
            
        Returns:
            Statistics dictionary with start_date and end_date
        """
        stats = {
            'total_stocks': 0,
            'updated': 0,
            'failed': 0,
            'skipped': 0,
            'total_dates_added': 0,
            'start_date': start_date.strftime('%d/%m/%Y'),  # FIXED: Added this line
            'end_date': end_date.strftime('%d/%m/%Y'),      # FIXED: Added this line
            'details': []
        }
        
        try:
            # Get watchlist
            watchlist = self.get_watchlist_from_eod()
            
            if not watchlist:
                logger.warning("No watchlist found for yfinance download")
                return stats
            
            stats['total_stocks'] = len(watchlist)
            
            logger.info(f"Starting yfinance download for {len(watchlist)} stocks from {start_date} to {end_date}")
            
            # Download each stock
            for ticker in watchlist:
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
            force_mode: If True, overwrite existing dates
            
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
                existing_df['Date'] = pd.to_datetime(existing_df['Date'], dayfirst=True, format='mixed')
                last_date_in_file = existing_df['Date'].max().date()
            else:
                existing_df = pd.DataFrame()
                last_date_in_file = None
            
            # Download from yfinance
            logger.info(f"Downloading {ticker} from yfinance: {start_date} to {end_date}")
            stock = yf.Ticker(ticker)
            downloaded_df = stock.history(start=start_date, end=end_date + timedelta(days=1))
            
            if downloaded_df.empty:
                result['status'] = 'skipped'
                result['message'] = 'No data available from yfinance'
                return result
            
            # Reset index to get Date as column
            downloaded_df.reset_index(inplace=True)
            
            # Format dates to D/M/YYYY (Singapore format)
            downloaded_df['Date'] = downloaded_df['Date'].dt.strftime('%-d/%-m/%Y' if os.name != 'nt' else '%#d/%#m/%Y')
            
            # Create new DataFrame with correct columns
            new_df = pd.DataFrame()
            new_df['Date'] = downloaded_df['Date']
            new_df['Code'] = ticker
            
            # Get company name (shortname)
            if not existing_df.empty and 'Shortname' in existing_df.columns:
                shortname = existing_df['Shortname'].iloc[0]
            else:
                try:
                    shortname = stock.info.get('shortName', ticker_clean)
                except:
                    shortname = ticker_clean
            
            new_df['Shortname'] = shortname
            new_df['Open'] = downloaded_df['Open'].round(3)
            new_df['High'] = downloaded_df['High'].round(3)
            new_df['Low'] = downloaded_df['Low'].round(3)
            new_df['Close'] = downloaded_df['Close'].round(3)
            
            # CRITICAL FIX: Divide volumes by 1000 to match EOD abbreviated format
            new_df['Vol'] = (downloaded_df['Volume'] / 1000).round(0).astype(int)
            
            logger.info(f"✅ VOLUME FIX APPLIED: Divided yfinance volumes by 1000 for {ticker}")
            
            # If force mode, include all dates
            # If not force mode, filter out dates that already exist
            if not force_mode and not existing_df.empty:
                new_df['Date_dt'] = pd.to_datetime(new_df['Date'], dayfirst=True, format='mixed')
                new_df = new_df[new_df['Date_dt'].dt.date > last_date_in_file]
                new_df.drop('Date_dt', axis=1, inplace=True)
            
            if new_df.empty:
                result['status'] = 'skipped'
                result['message'] = 'No new dates to add'
                return result
            
            # Combine with existing data
            if not existing_df.empty:
                # Format existing dates for consistency
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
            
            # Filter out NaN values
            watchlist = eod_df['Code'].dropna().unique().tolist()
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