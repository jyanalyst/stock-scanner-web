# File: utils/watchlist.py
"""
Watchlist Management
Contains the Singapore stock symbols with company name fetching from yfinance
"""

# Singapore Exchange stocks
DEFAULT_WATCHLIST = [
    'A17U.SI', 'C38U.SI', 'M44U.SI', 'ME8U.SI', 'AJBU.SI', 
    'J69U.SI', 'N2IU.SI', 'BUOU.SI', 'K71U.SI', 'JYEU.SI',
    'HMN.SI','AU8U.SI','9CI.SI', 'AWX.SI', 'E28.SI', '558.SI', 'MZH.SI',
    'BN2.SI', 'BS6.SI', '5E2.SI', 'Z74.SI', 
    'C52.SI', 'YF8.SI', '5LY.SI', 'G13.SI', 'OV8.SI',
    'S56.SI', 'AP4.SI', '544.SI', 'G92.SI', 'Z25.SI',
    'RXS.SI', 'S58.SI', 'EB5.SI', '1MZ.SI', '41O.SI',
    'OYY.SI', 'E3B.SI', 'OU8.SI', 'LCC.SI',
    'A04.SI', 'ER0.SI', '5WH.SI', '1E3.SI', '5TP.SI', 'P52.SI',
    'QES.SI', 'NTDU.SI', 'CLN.SI', 'QC7.SI', 'P8Z.SI', 'RE4.SI', 'DU4.SI'
]

def get_active_watchlist():
    """Get the list of active stocks to scan"""
    return DEFAULT_WATCHLIST.copy()

def get_stock_name(symbol):
    """
    Get friendly name for a stock symbol
    Use DataFetcher.get_company_name() for real company names from yfinance
    """
    try:
        from core.data_fetcher import get_company_name
        return get_company_name(symbol)
    except ImportError:
        return symbol.replace('.SI', '')

def add_stock_to_watchlist(symbol):
    """Add a new stock to the watchlist"""
    if symbol not in DEFAULT_WATCHLIST:
        DEFAULT_WATCHLIST.append(symbol)
        return True
    return False

def remove_stock_from_watchlist(symbol):
    """Remove a stock from the watchlist"""
    if symbol in DEFAULT_WATCHLIST:
        DEFAULT_WATCHLIST.remove(symbol)
        return True
    return False

def get_watchlist_info():
    """Get information about the current watchlist"""
    return {
        'total_stocks': len(DEFAULT_WATCHLIST),
        'exchange': 'Singapore Exchange (SGX)',
        'sectors': 'Mixed (REITs, Financial, Industrial, etc.)',
        'note': 'Company names are fetched dynamically from yfinance'
    }