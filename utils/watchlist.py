# File: utils/watchlist.py
"""
Watchlist Management - Dynamic from Google Drive EOD Data
Automatically loads watchlist from latest EOD file
"""

import streamlit as st
import logging
from typing import List

logger = logging.getLogger(__name__)

# Minimal fallback watchlist in case Google Drive is unavailable
FALLBACK_WATCHLIST = [
    'A17U.SG', 'C38U.SG', 'M44U.SG', 'ME8U.SG', 'AJBU.SG', 'J69U.SG', 
    'N2IU.SG', 'BUOU.SG', 'K71U.SG', 'JYEU.SG'
]

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_active_watchlist() -> List[str]:
    """
    Get the list of active stocks to scan from latest EOD file
    
    Returns:
        List of ticker symbols (e.g., ['A17U.SG', 'C38U.SG', ...])
    """
    try:
        from core.gdrive_loader import get_gdrive_loader, get_folder_ids
        
        # Get Google Drive loader
        loader = get_gdrive_loader()
        if loader is None:
            logger.warning("Google Drive not configured, using fallback watchlist")
            return FALLBACK_WATCHLIST.copy()
        
        # Get folder IDs
        folder_ids = get_folder_ids()
        eod_folder_id = folder_ids['eod']
        
        if not eod_folder_id:
            logger.warning("EOD folder ID not configured, using fallback watchlist")
            return FALLBACK_WATCHLIST.copy()
        
        # Load watchlist from latest EOD file
        watchlist = loader.get_watchlist_from_eod(eod_folder_id)
        
        if not watchlist:
            logger.warning("Could not load watchlist from EOD file, using fallback")
            return FALLBACK_WATCHLIST.copy()
        
        logger.info(f"✅ Loaded watchlist from Google Drive: {len(watchlist)} stocks")
        return sorted(watchlist)
        
    except Exception as e:
        logger.error(f"Error loading watchlist: {e}")
        logger.info("Using fallback watchlist")
        return FALLBACK_WATCHLIST.copy()

def get_stock_name(symbol: str) -> str:
    """
    Get friendly name for a stock symbol
    
    Args:
        symbol: Stock ticker (e.g., 'A17U.SG')
        
    Returns:
        Friendly name (e.g., 'A17U')
    """
    return symbol.replace('.SG', '')

def get_watchlist_info() -> dict:
    """Get information about the current watchlist"""
    watchlist = get_active_watchlist()
    
    return {
        'total_stocks': len(watchlist),
        'exchange': 'Singapore Exchange (SGX)',
        'source': 'Google Drive EOD_Data (dynamic)',
        'note': 'Watchlist automatically updated from latest EOD file'
    }

def clear_watchlist_cache():
    """Clear the watchlist cache to force reload"""
    get_active_watchlist.clear()
    logger.info("Watchlist cache cleared")