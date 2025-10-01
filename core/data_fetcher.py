# File: core/data_fetcher.py
"""
Data Fetcher Module - Google Drive Only
Handles stock data loading from Google Drive CSV files
All dates use Singapore format: D/M/YYYY (dayfirst=True)
"""

import pandas as pd
import datetime as dt
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    """Data fetcher class for Google Drive CSV files"""
    
    def __init__(self, days_back: int = 59):
        """
        Initialize data fetcher
        
        Args:
            days_back: Number of days of historical data (used for compatibility, not strictly needed for Google Drive)
        """
        self.days_back = days_back
        self.company_names = {}
    
    def download_stock_data(self, tickers: List[str], target_date: Optional[dt.date] = None) -> Dict[str, pd.DataFrame]:
        """
        Load stock data for multiple tickers from Google Drive Historical_Data folder
        Also extracts and stores company names from Shortname column
        
        Args:
            tickers: List of stock tickers (e.g., ['A17U.SG', 'C38U.SG'])
            target_date: Optional date to filter historical data up to
            
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        from core.gdrive_loader import get_gdrive_loader, get_folder_ids
        
        loader = get_gdrive_loader()
        if loader is None:
            logger.error("❌ Google Drive loader not available")
            return {}
        
        folder_ids = get_folder_ids()
        historical_folder_id = folder_ids['historical']
        
        if not historical_folder_id:
            logger.error("❌ Historical folder ID not configured in .env file")
            return {}
        
        all_data = {}
        
        for ticker in tickers:
            try:
                # Load historical data
                df = loader.load_historical_data(historical_folder_id, ticker)
                
                if df is not None and not df.empty:
                    # Filter by target_date if specified
                    if target_date:
                        df = df[df.index.date <= target_date]
                    
                    # Extract company name from Shortname column if available
                    if 'Shortname' in df.columns:
                        self.company_names[ticker] = df['Shortname'].iloc[0]
                    else:
                        self.company_names[ticker] = ticker.replace('.SG', '')
                    
                    # Remove metadata columns (Code, Shortname) before returning
                    # Keep only OHLCV data
                    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[col for col in cols_to_keep if col in df.columns]]
                    
                    all_data[ticker] = df
                else:
                    logger.warning(f"No data loaded for {ticker}")
                    self.company_names[ticker] = ticker.replace('.SG', '')
                    
            except Exception as e:
                logger.error(f"Error loading {ticker}: {e}")
                self.company_names[ticker] = ticker.replace('.SG', '')
        
        logger.info(f"Loaded data for {len(all_data)} stocks from Google Drive")
        return all_data
    
    def get_company_name(self, ticker: str) -> str:
        """
        Get company name for a ticker
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Company name string
        """
        if ticker in self.company_names:
            return self.company_names[ticker]
        else:
            return ticker.replace('.SG', '')
    
    def get_all_company_names(self) -> Dict[str, str]:
        """Get all stored company names"""
        return self.company_names.copy()
    
    def get_single_stock_data(self, ticker: str, target_date: Optional[dt.date] = None) -> pd.DataFrame:
        """
        Load data for a single stock
        
        Args:
            ticker: Stock ticker
            target_date: Optional date to filter up to
            
        Returns:
            DataFrame with stock data
        """
        stock_data = self.download_stock_data([ticker], target_date)
        return stock_data.get(ticker, pd.DataFrame())
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """
        Validate the quality of loaded data
        
        Args:
            data: Dictionary of {ticker: DataFrame}
            
        Returns:
            Dictionary of {ticker: is_valid}
        """
        validation_results = {}
        min_days_required = 30
        
        for ticker, df in data.items():
            if df.empty:
                validation_results[ticker] = False
                continue
                
            has_required_columns = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            has_sufficient_data = len(df) >= min_days_required
            has_recent_data = (dt.date.today() - df.index[-1].date()).days <= 7  # Allow up to 7 days old
            
            validation_results[ticker] = has_required_columns and has_sufficient_data and has_recent_data
        
        return validation_results


# Global instance management
_global_data_fetcher = None

def get_company_name(ticker: str) -> str:
    """Global function to get company name"""
    global _global_data_fetcher
    
    if _global_data_fetcher is None:
        _global_data_fetcher = DataFetcher()
    
    return _global_data_fetcher.get_company_name(ticker)

def set_global_data_fetcher(fetcher: DataFetcher):
    """Set the global data fetcher instance"""
    global _global_data_fetcher
    _global_data_fetcher = fetcher