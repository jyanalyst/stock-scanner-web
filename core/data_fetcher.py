# File: core/data_fetcher.py
"""
Data Fetcher Module
Handles stock data download from Yahoo Finance with optimized caching and error handling
"""

import yfinance as yf
import pandas as pd
import datetime as dt
from typing import List, Dict, Optional, Tuple
import streamlit as st
import logging

logger = logging.getLogger(__name__)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def _download_stock_data_cached(tickers: List[str], days_back: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Cached function to download stock data and company names
    Returns: (stock_data_dict, company_names_dict)
    """
    logger.info(f"Downloading data for {len(tickers)} stocks...")
    
    start_date = dt.date.today() - dt.timedelta(days=days_back)
    logger.info(f"Data range: {start_date} to {dt.date.today()}")
    
    all_data = {}
    company_names = {}
    successful_downloads = []
    failed_downloads = []
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"Downloading {ticker}...")
            progress_bar.progress((i + 1) / len(tickers))
            
            # Create ticker object and download data
            ticker_obj = yf.Ticker(ticker)
            df_daily = ticker_obj.history(
                start=start_date,
                end=dt.date.today() + dt.timedelta(days=1)
            )
            
            if len(df_daily) > 0:
                # Fix multi-level columns
                if df_daily.columns.nlevels > 1:
                    df_daily.columns = [col[0] if isinstance(col, tuple) else col for col in df_daily.columns]
                
                # Validate required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df_daily.columns for col in required_cols):
                    all_data[ticker] = df_daily
                    successful_downloads.append(ticker)
                else:
                    logger.warning(f"{ticker}: Missing required columns")
                    failed_downloads.append(ticker)
                
                # Get company name
                company_names[ticker] = _get_company_name(ticker_obj, ticker)
                
            else:
                failed_downloads.append(ticker)
                company_names[ticker] = ticker.replace('.SI', '')
                
        except Exception as e:
            failed_downloads.append(ticker)
            company_names[ticker] = ticker.replace('.SI', '')
            logger.error(f"Error downloading {ticker}: {str(e)}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    logger.info(f"Successfully downloaded: {len(successful_downloads)} stocks")
    if failed_downloads:
        logger.warning(f"Failed downloads: {failed_downloads}")
    
    return all_data, company_names

@st.cache_data(ttl=3600)  # Cache company names for 1 hour
def _get_company_name_cached(ticker: str) -> str:
    """Get company name for a single ticker (cached separately)"""
    try:
        ticker_obj = yf.Ticker(ticker)
        return _get_company_name(ticker_obj, ticker)
    except Exception as e:
        logger.warning(f"Could not get company name for {ticker}: {e}")
        return ticker.replace('.SI', '')

def _get_company_name(ticker_obj, ticker: str) -> str:
    """Extract company name from ticker object"""
    try:
        info = ticker_obj.info
        return (
            info.get('shortName') or 
            info.get('displayName') or
            info.get('longName') or
            ticker.replace('.SI', '')
        )
    except Exception:
        return ticker.replace('.SI', '')

class DataFetcher:
    """Main data fetcher class with optimized caching and error handling"""
    
    def __init__(self, days_back: int = 59):
        self.days_back = days_back
        self.company_names = {}
    
    def download_stock_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Download stock data for multiple tickers
        Also fetches and stores company names
        """
        stock_data, company_names = _download_stock_data_cached(tickers, self.days_back)
        self.company_names = company_names
        return stock_data
    
    def get_company_name(self, ticker: str) -> str:
        """Get company name for a ticker"""
        if ticker in self.company_names:
            return self.company_names[ticker]
        else:
            company_name = _get_company_name_cached(ticker)
            self.company_names[ticker] = company_name
            return company_name
    
    def get_all_company_names(self) -> Dict[str, str]:
        """Get all stored company names"""
        return self.company_names.copy()
    
    def get_single_stock_data(self, ticker: str, days: Optional[int] = None) -> pd.DataFrame:
        """Download data for a single stock with improved error handling"""
        days = days or self.days_back
        start_date = dt.date.today() - dt.timedelta(days=days)
        
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date,
                end=dt.date.today() + dt.timedelta(days=1)
            )
            
            # Fix multi-level columns
            if df.columns.nlevels > 1:
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Store company name
            self.company_names[ticker] = _get_company_name(ticker_obj, ticker)
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Validate the quality of downloaded data"""
        validation_results = {}
        min_days_required = 30
        
        for ticker, df in data.items():
            if df.empty:
                validation_results[ticker] = False
                continue
                
            has_required_columns = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            has_sufficient_data = len(df) >= min_days_required
            has_recent_data = (dt.date.today() - df.index[-1].date()).days <= 3
            
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

def download_stock_data(tickers: List[str], days_back: int = 59) -> Dict:
    """Wrapper function for backward compatibility"""
    stock_data, _ = _download_stock_data_cached(tickers, days_back)
    return stock_data

def test_single_download(ticker: str = "A17U.SI") -> bool:
    """Test downloading a single stock for debugging"""
    try:
        logger.info(f"Testing download for {ticker}...")
        
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="5d")
        
        if len(df) > 0:
            logger.info(f"Success: Downloaded {len(df)} days of data")
            logger.info(f"Latest close: {df['Close'].iloc[-1]:.2f}")
            return True
        else:
            logger.error("No data returned")
            return False
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return False