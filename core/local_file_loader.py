# File: core/local_file_loader.py
"""
Local File System Data Loader
Handles reading and updating stock data from local CSV files
All dates use Singapore format: D/M/YYYY (dayfirst=True)
FIXED: Proper date validation, cleanup of erroneous dates, and metadata column handling
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
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
    
    def check_for_updates(self) -> Tuple[bool, Optional[str], Optional[datetime]]:
        """
        Check if Historical_Data needs updating from EOD_Data
        
        Returns:
            Tuple of (needs_update, eod_filename, eod_date)
        """
        try:
            # Get latest EOD file
            latest_eod = self.get_latest_eod_file()
            
            if not latest_eod:
                logger.warning("No EOD files found")
                return False, None, None
            
            # Parse EOD date
            eod_date_str = latest_eod.replace('.csv', '')
            eod_date = datetime.strptime(eod_date_str, '%d_%b_%Y')
            
            # Load EOD file to get first ticker
            filepath = os.path.join(self.eod_path, latest_eod)
            eod_df = pd.read_csv(filepath, encoding='utf-8')
            
            if 'Code' not in eod_df.columns:
                return False, None, None
            
            sample_ticker = eod_df['Code'].iloc[0]
            
            # Get last date in historical
            last_hist_date = self.get_last_date_in_historical(sample_ticker)
            
            if last_hist_date is None:
                logger.info(f"Historical file for {sample_ticker} is missing or empty")
                return True, latest_eod, eod_date
            
            # Compare dates
            needs_update = eod_date.date() > last_hist_date.date()
            
            if needs_update:
                logger.info(f"Update available: EOD {eod_date.date()} > Historical {last_hist_date.date()}")
            else:
                logger.info(f"Historical data is current (last: {last_hist_date.date()})")
            
            return needs_update, latest_eod, eod_date
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return False, None, None
    
    def update_historical_from_eod(self) -> Dict[str, any]:
        """
        Update all Historical_Data files from latest EOD_Data file
        FIXED: Validates dates, cleans up erroneous data, handles metadata columns
        
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
            
            logger.info(f"Starting update from {latest_eod} ({eod_date_str})")
            
            # Process each stock
            tickers = eod_df['Code'].unique()
            stats['total_stocks'] = len(tickers)
            
            for ticker in tickers:
                try:
                    result = self._update_single_stock(ticker, eod_df, eod_date_str, eod_date_obj)
                    
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
                            eod_date_str: str, eod_date_obj: datetime) -> Dict:
        """
        Update historical data for a single stock
        FIXED: 
        - Handles missing Code/Shortname columns
        - Validates and cleans up date sequences
        - Removes erroneous future/duplicate dates
        
        Args:
            ticker: Stock ticker (e.g., 'A17U.SG')
            eod_df: EOD DataFrame containing all stocks
            eod_date_str: Date string in D/M/YYYY format
            eod_date_obj: Date as datetime object
            
        Returns:
            Dictionary with update result
        """
        result = {
            'ticker': ticker,
            'status': 'unknown',
            'message': ''
        }
        
        try:
            # Get ticker's row from EOD data
            eod_row = eod_df[eod_df['Code'] == ticker].iloc[0]
            
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
                
                # Check if EOD date already exists (should not happen after cleanup, but double-check)
                if eod_date_obj.date() in hist_df['Date'].dt.date.values:
                    result['status'] = 'skipped'
                    result['message'] = 'Date already exists'
                    return result
                
                # Validate: Check if we're actually adding a newer date
                if len(hist_df) > 0:
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
            result['message'] = 'Added 1 row'
            
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