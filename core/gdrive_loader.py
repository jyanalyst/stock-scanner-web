# File: core/gdrive_loader.py
# Part 1 of 2
"""
Google Drive Data Loader
Handles downloading and loading EOD data from Google Drive folders
Supports updating Historical_Data from EOD_Data automatically
All dates use Singapore format: D/M/YYYY (dayfirst=True)
"""

import os
import io
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import streamlit as st

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

logger = logging.getLogger(__name__)

# Google Drive API scopes - WRITE permission needed for updates
SCOPES = ['https://www.googleapis.com/auth/drive']

class GoogleDriveLoader:
    """Load EOD data from Google Drive folders and update Historical_Data"""
    
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'token.json'):
        """
        Initialize Google Drive loader
        
        Args:
            credentials_file: Path to credentials.json file
            token_file: Path to store/load token
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API - Manual URL/Code Method with redirect_uri"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
                logger.info("Loaded existing credentials")
            except Exception as e:
                logger.warning(f"Could not load existing token: {e}")
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing expired credentials...")
                    creds.refresh(Request())
                    logger.info("Credentials refreshed successfully")
                except Exception as e:
                    logger.error(f"Could not refresh token: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"❌ Credentials file not found: {self.credentials_file}\n"
                        "Please download credentials.json from Google Cloud Console\n"
                        "See README_GDRIVE_SETUP.md for instructions"
                    )
                
                print("\n" + "="*70)
                print("🔐 GOOGLE DRIVE AUTHENTICATION REQUIRED")
                print("="*70)
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file,
                    SCOPES,
                    redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # Use OOB (out-of-band) flow
                )
                
                # Generate the authorization URL
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    prompt='consent'
                )
                
                print("\n📋 STEP 1: Visit this URL in your browser:\n")
                print(auth_url)
                print("\n📋 STEP 2: Sign in and authorize the app")
                print("\n📋 STEP 3: Google will show you an authorization code")
                print("           Copy that code (it will be displayed on the page)")
                print("\n📋 STEP 4: Enter the authorization code below:\n")
                
                # Get code from console input
                auth_code = input("Enter authorization code: ").strip()
                
                if auth_code:
                    try:
                        # Exchange authorization code for credentials
                        flow.fetch_token(code=auth_code)
                        creds = flow.credentials
                        
                        # Save credentials
                        with open(self.token_file, 'w') as token:
                            token.write(creds.to_json())
                        
                        print("\n✅ Authentication successful!")
                        print(f"✅ Credentials saved to {self.token_file}")
                        logger.info("✅ Authentication successful")
                    except Exception as e:
                        error_msg = f"❌ Authentication failed: {e}"
                        print(error_msg)
                        logger.error(error_msg)
                        raise
                else:
                    raise ValueError("No authorization code provided")
        
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("✅ Successfully authenticated with Google Drive")
    
    def list_files_in_folder(self, folder_id: str) -> List[Dict]:
        """
        List all CSV files in a Google Drive folder
        """
        try:
            # List all non-trashed files in the folder
            query = f"'{folder_id}' in parents and trashed=false"
            
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, modifiedTime, size, mimeType)',
                orderBy='name'
            ).execute()
            
            all_files = results.get('files', [])
            
            # Filter for CSV files manually (by name ending)
            csv_files = [f for f in all_files if f['name'].endswith('.csv')]
            
            logger.info(f"Found {len(csv_files)} CSV files out of {len(all_files)} total files in folder")
            
            # Debug: log what we found
            for f in csv_files[:5]:  # Show first 5
                logger.info(f"  - {f['name']} (MIME: {f.get('mimeType', 'unknown')})")
            
            return csv_files
            
        except Exception as e:
            logger.error(f"Error listing files in folder {folder_id}: {e}")
            return []
    
    def download_file_content(self, file_id: str, file_name: str) -> Optional[pd.DataFrame]:
        """
        Download a CSV file from Google Drive and return as DataFrame
        
        Args:
            file_id: Google Drive file ID
            file_name: File name (for logging)
            
        Returns:
            DataFrame with file content or None if error
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            file_buffer.seek(0)
            
            # Read CSV with UTF-8 encoding
            df = pd.read_csv(file_buffer, encoding='utf-8')
            
            logger.info(f"✅ Downloaded {file_name}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error downloading file {file_name}: {e}")
            return None
    
    def upload_csv_to_drive(self, folder_id: str, filename: str, df: pd.DataFrame, 
                           existing_file_id: Optional[str] = None) -> bool:
        """
        Upload or update a CSV file to Google Drive
        
        Args:
            folder_id: Google Drive folder ID
            filename: Name for the file (e.g., 'A17U.csv')
            df: DataFrame to upload
            existing_file_id: If provided, updates existing file instead of creating new
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert DataFrame to CSV in memory
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_buffer.seek(0)
            
            media = MediaIoBaseUpload(csv_buffer, mimetype='text/csv', resumable=True)
            
            if existing_file_id:
                # Update existing file
                self.service.files().update(
                    fileId=existing_file_id,
                    media_body=media
                ).execute()
                logger.info(f"✅ Updated {filename} in Google Drive")
            else:
                # Create new file
                file_metadata = {
                    'name': filename,
                    'parents': [folder_id]
                }
                self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"✅ Created {filename} in Google Drive")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error uploading {filename}: {e}")
            return False
    
    def get_file_id_by_name(self, folder_id: str, filename: str) -> Optional[str]:
        """
        Get file ID for a file in a folder by name
        
        Args:
            folder_id: Google Drive folder ID
            filename: Name of file to find
            
        Returns:
            File ID or None if not found
        """
        try:
            files = self.list_files_in_folder(folder_id)
            for file in files:
                if file['name'] == filename:
                    return file['id']
            return None
        except Exception as e:
            logger.error(f"Error finding file {filename}: {e}")
            return None
    
    def get_latest_eod_file(self, folder_id: str) -> Optional[Dict]:
        """
        Get the most recent EOD file from EOD_Data folder
        
        Args:
            folder_id: EOD_Data folder ID
            
        Returns:
            File metadata dict or None
        """
        try:
            files = self.list_files_in_folder(folder_id)
            
            if not files:
                logger.warning("No EOD files found")
                return None
            
            # Parse dates from filenames (format: DD_MMM_YYYY.csv)
            dated_files = []
            for file in files:
                try:
                    # Remove .csv extension
                    date_str = file['name'].replace('.csv', '')
                    # Parse: "01_Oct_2025" → datetime
                    file_date = datetime.strptime(date_str, '%d_%b_%Y')
                    dated_files.append((file_date, file))
                except ValueError:
                    logger.warning(f"Could not parse date from filename: {file['name']}")
                    continue
            
            if not dated_files:
                return None
            
            # Sort by date and get latest
            dated_files.sort(key=lambda x: x[0], reverse=True)
            latest_file = dated_files[0][1]
            
            logger.info(f"Latest EOD file: {latest_file['name']}")
            return latest_file
            
        except Exception as e:
            logger.error(f"Error getting latest EOD file: {e}")
            return None

# Continue to Part 2...
# File: core/gdrive_loader.py
# Part 2 of 2

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
            # Format as D/M/YYYY (Singapore format, no leading zeros for day)
            return date_obj.strftime('%-d/%-m/%Y') if os.name != 'nt' else date_obj.strftime('%#d/%#m/%Y')
        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")
            return None
    
    def load_historical_data(self, folder_id: str, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load historical data for a specific ticker from Historical_Data folder
        
        Args:
            folder_id: Google Drive folder ID for Historical_Data
            ticker: Stock ticker (e.g., 'A17U.SG')
            
        Returns:
            DataFrame with historical data or None
        """
        try:
            # Remove .SG suffix to get filename
            ticker_clean = ticker.replace('.SG', '')
            filename = f"{ticker_clean}.csv"
            
            # List files in folder
            files = self.list_files_in_folder(folder_id)
            
            # Find matching file
            matching_file = None
            for f in files:
                if f['name'] == filename:
                    matching_file = f
                    break
            
            if not matching_file:
                logger.warning(f"No historical file found for ticker {ticker} (looking for {filename})")
                return None
            
            logger.info(f"Loading {ticker} from {matching_file['name']}")
            
            df = self.download_file_content(matching_file['id'], matching_file['name'])
            
            if df is not None:
                # Parse dates with Singapore format (D/M/YYYY)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                    df.set_index('Date', inplace=True)
                
                # Standardize column names
                df = self._standardize_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data for {ticker}: {e}")
            return None
    
    def load_eod_data(self, folder_id: str, target_date: Optional[date] = None) -> Dict[str, pd.DataFrame]:
        """
        Load EOD data for all stocks from EOD_Data folder
        
        Args:
            folder_id: Google Drive folder ID for EOD_Data
            target_date: Specific date to load (default: most recent)
            
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        try:
            # List files in folder
            files = self.list_files_in_folder(folder_id)
            
            if not files:
                logger.warning("No EOD files found")
                return {}
            
            # If target_date specified, find matching file
            if target_date:
                date_str = target_date.strftime('%d_%b_%Y')
                matching_files = [f for f in files if date_str in f['name']]
                
                if not matching_files:
                    logger.warning(f"No EOD file found for date {target_date}")
                    return {}
                
                file = matching_files[0]
            else:
                # Use most recent file
                file = self.get_latest_eod_file(folder_id)
                if file is None:
                    return {}
            
            logger.info(f"Loading EOD data from {file['name']}")
            
            # Download and parse file
            df = self.download_file_content(file['id'], file['name'])
            
            if df is None:
                return {}
            
            # Add Date column from filename
            date_str = self.parse_eod_filename_to_date(file['name'])
            if date_str:
                df['Date'] = date_str
            
            # Split into separate DataFrames per ticker
            stock_data = self._split_eod_data(df)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error loading EOD data: {e}")
            return {}
    
    def _split_eod_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split EOD DataFrame containing all stocks into separate DataFrames per ticker
        
        Args:
            df: Combined DataFrame with all stocks
            
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        stock_data = {}
        
        try:
            # Check for Code column
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
        Standardize column names to match scanner expectations
        
        EOD columns: Code, Shortname, Open, High, Low, Last, Volume
        Historical columns: Date, Code, Shortname, Open, High, Low, Close, Vol
        Scanner expects: Open, High, Low, Close, Volume
        """
        # Column mapping
        column_mapping = {
            'Last': 'Close',      # EOD uses 'Last' instead of 'Close'
            'Vol': 'Volume',       # Historical uses 'Vol' instead of 'Volume'
            'volume': 'Volume',    # Lowercase variants
            'close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low'
        }
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure Volume is integer (not float with decimals like 10800.7)
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(float).astype(int)
        
        return df
    
    def get_last_date_in_historical(self, folder_id: str, ticker: str) -> Optional[datetime]:
        """
        Get the last date in a historical CSV file
        
        Args:
            folder_id: Historical_Data folder ID
            ticker: Stock ticker (e.g., 'A17U.SG')
            
        Returns:
            Last date as datetime or None
        """
        try:
            df = self.load_historical_data(folder_id, ticker)
            
            if df is None or df.empty:
                return None
            
            # Get last date from index
            return df.index[-1]
            
        except Exception as e:
            logger.error(f"Error getting last date for {ticker}: {e}")
            return None
    
    def check_for_updates(self, historical_folder_id: str, eod_folder_id: str) -> Tuple[bool, Optional[str], Optional[datetime]]:
        """
        Check if Historical_Data needs updating from EOD_Data
        
        Args:
            historical_folder_id: Historical_Data folder ID
            eod_folder_id: EOD_Data folder ID
            
        Returns:
            Tuple of (needs_update, eod_filename, eod_date)
        """
        try:
            # Get latest EOD file
            latest_eod = self.get_latest_eod_file(eod_folder_id)
            
            if not latest_eod:
                logger.warning("No EOD files found")
                return False, None, None
            
            # Parse EOD date
            eod_date_str = latest_eod['name'].replace('.csv', '')
            eod_date = datetime.strptime(eod_date_str, '%d_%b_%Y')
            
            # Get watchlist from EOD file
            eod_df = self.download_file_content(latest_eod['id'], latest_eod['name'])
            if eod_df is None or 'Code' not in eod_df.columns:
                return False, None, None
            
            # Get first ticker to check
            sample_ticker = eod_df['Code'].iloc[0]
            
            # Get last date in historical for this ticker
            last_hist_date = self.get_last_date_in_historical(historical_folder_id, sample_ticker)
            
            if last_hist_date is None:
                # Historical file doesn't exist or is empty - needs update
                logger.info(f"Historical file for {sample_ticker} is missing or empty")
                return True, latest_eod['name'], eod_date
            
            # Compare dates
            needs_update = eod_date.date() > last_hist_date.date()
            
            if needs_update:
                logger.info(f"Update available: EOD {eod_date.date()} > Historical {last_hist_date.date()}")
            else:
                logger.info(f"Historical data is current (last: {last_hist_date.date()})")
            
            return needs_update, latest_eod['name'], eod_date
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return False, None, None
    
    def update_historical_from_eod(self, historical_folder_id: str, eod_folder_id: str) -> Dict[str, any]:
        """Update all Historical_Data files from latest EOD_Data file"""
        stats = {
            'total_stocks': 0,
            'updated': 0,
            'skipped': 0,
            'created': 0,
            'errors': 0,
            'eod_date': None,
            'details': []
        }
        
        try:
            # Load latest EOD file
            latest_eod = self.get_latest_eod_file(eod_folder_id)
            if not latest_eod:
                return stats
            
            eod_df = self.download_file_content(latest_eod['id'], latest_eod['name'])
            if eod_df is None:
                return stats
            
            # Parse EOD date
            eod_date_str = self.parse_eod_filename_to_date(latest_eod['name'])
            eod_date_obj = datetime.strptime(latest_eod['name'].replace('.csv', ''), '%d_%b_%Y')
            stats['eod_date'] = eod_date_str
            
            logger.info(f"Starting update from {latest_eod['name']} ({eod_date_str})")
            
            # PRE-LOAD all historical files ONCE (optimization)
            all_hist_files = self.list_files_in_folder(historical_folder_id)
            file_map = {f['name']: f['id'] for f in all_hist_files}
            logger.info(f"Pre-loaded {len(file_map)} historical files")
            
            # Process each stock
            tickers = eod_df['Code'].unique()
            stats['total_stocks'] = len(tickers)
            
            for ticker in tickers:
                try:
                    result = self._update_single_stock(
                        historical_folder_id, 
                        ticker, 
                        eod_df, 
                        eod_date_str,
                        eod_date_obj,
                        file_map  # Pass pre-loaded map
                    )
                    
                    stats['details'].append(result)
                    
                    if result['status'] == 'updated':
                        stats['updated'] += 1
                    elif result['status'] == 'skipped':
                        stats['skipped'] += 1
                    elif result['status'] == 'created':
                        stats['created'] += 1
                    elif result['status'] == 'error':
                        stats['errors'] += 1
                        
                except Exception as e:
                    logger.error(f"Error updating {ticker}: {e}")
                    stats['errors'] += 1
                    stats['details'].append({
                        'ticker': ticker,
                        'status': 'error',
                        'message': str(e)
                    })
            
            logger.info(f"Update complete: {stats['updated']} updated, {stats['created']} created, "
                    f"{stats['skipped']} skipped, {stats['errors']} errors")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in update_historical_from_eod: {e}")
            return stats
    
    def _update_single_stock(self, historical_folder_id: str, ticker: str, 
                            eod_df: pd.DataFrame, eod_date_str: str,
                            eod_date_obj: datetime, file_map: Dict[str, str]) -> Dict:
        """
        Update historical data for a single stock
        
        Args:
            historical_folder_id: Historical_Data folder ID
            ticker: Stock ticker (e.g., 'A17U.SG')
            eod_df: EOD DataFrame containing all stocks
            eod_date_str: Date string in D/M/YYYY format
            eod_date_obj: Date as datetime object
            file_map: Pre-loaded map of {filename: file_id}
            
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
            file_id = file_map.get(filename)  # Use pre-loaded map instead of API call
            
            if file_id:
                # File exists - load it
                hist_df = self.download_file_content(file_id, filename)
                
                if hist_df is None or hist_df.empty:
                    result['status'] = 'error'
                    result['message'] = 'Could not load historical file'
                    return result
                
                # Parse dates with Singapore format
                hist_df['Date'] = pd.to_datetime(hist_df['Date'], dayfirst=True)
                
                # Check if date already exists
                if eod_date_obj.date() in hist_df['Date'].dt.date.values:
                    result['status'] = 'skipped'
                    result['message'] = 'Date already exists'
                    return result
                
                # Add Code and Shortname columns if missing
                if 'Code' not in hist_df.columns:
                    hist_df['Code'] = ticker
                if 'Shortname' not in hist_df.columns:
                    hist_df['Shortname'] = eod_row.get('Shortname', ticker)
                
                file_existed = True
                
            else:
                # File doesn't exist - create new DataFrame
                hist_df = pd.DataFrame()
                file_existed = False
            
            # Create new row from EOD data
            new_row = pd.DataFrame([{
                'Date': eod_date_str,
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
            
            # Sort by date
            hist_df['Date'] = pd.to_datetime(hist_df['Date'], dayfirst=True)
            hist_df = hist_df.sort_values('Date')
            
            # Format dates back to D/M/YYYY for saving
            hist_df['Date'] = hist_df['Date'].dt.strftime('%-d/%-m/%Y') if os.name != 'nt' else hist_df['Date'].dt.strftime('%#d/%#m/%Y')
            
            # Upload updated file
            success = self.upload_csv_to_drive(
                historical_folder_id, 
                filename, 
                hist_df, 
                existing_file_id=file_id
            )
            
            if success:
                result['status'] = 'updated' if file_existed else 'created'
                result['message'] = 'Added 1 row' if file_existed else 'Created with 1 row'
            else:
                result['status'] = 'error'
                result['message'] = 'Upload failed'
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
            result['status'] = 'error'
            result['message'] = str(e)
            return result
    
    def get_watchlist_from_eod(self, eod_folder_id: str) -> List[str]:
        """
        Extract watchlist (list of tickers) from latest EOD file
        
        Args:
            eod_folder_id: EOD_Data folder ID
            
        Returns:
            List of ticker strings (e.g., ['A17U.SG', 'C38U.SG', ...])
        """
        try:
            latest_eod = self.get_latest_eod_file(eod_folder_id)
            
            if not latest_eod:
                logger.warning("No EOD file found for watchlist extraction")
                return []
            
            eod_df = self.download_file_content(latest_eod['id'], latest_eod['name'])
            
            if eod_df is None or 'Code' not in eod_df.columns:
                logger.error("Could not extract watchlist from EOD file")
                return []
            
            watchlist = eod_df['Code'].unique().tolist()
            logger.info(f"Extracted watchlist: {len(watchlist)} stocks")
            
            return watchlist
            
        except Exception as e:
            logger.error(f"Error getting watchlist from EOD: {e}")
            return []


@st.cache_resource
def get_gdrive_loader() -> Optional[GoogleDriveLoader]:
    """
    Get cached Google Drive loader instance
    
    Returns:
        GoogleDriveLoader instance or None if setup incomplete
    """
    try:
        loader = GoogleDriveLoader()
        return loader
    except FileNotFoundError as e:
        logger.warning(f"Google Drive not configured: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing Google Drive loader: {e}")
        return None


def get_folder_ids() -> Dict[str, Optional[str]]:
    """
    Get Google Drive folder IDs from environment
    
    Returns:
        Dictionary with 'historical' and 'eod' folder IDs
    """
    return {
        'historical': os.getenv('GDRIVE_HISTORICAL_FOLDER_ID'),
        'eod': os.getenv('GDRIVE_EOD_FOLDER_ID')
    }