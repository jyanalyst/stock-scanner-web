"""
Date Utility Functions
Centralized date formatting and parsing for Singapore format (D/M/YYYY)
"""

import os
from datetime import datetime, date
from typing import Union
import pandas as pd


def format_singapore_date(date_obj: Union[datetime, date, pd.Timestamp]) -> str:
    """
    Format date as D/M/YYYY (Singapore format, no leading zeros)
    
    Args:
        date_obj: Date to format (datetime, date, or pd.Timestamp)
        
    Returns:
        Date string in D/M/YYYY format (e.g., '1/10/2025', '15/3/2024')
        
    Examples:
        >>> format_singapore_date(datetime(2025, 10, 1))
        '1/10/2025'
        >>> format_singapore_date(datetime(2024, 3, 15))
        '15/3/2024'
    """
    if pd.isna(date_obj):
        return ''
    
    # Convert pd.Timestamp to datetime if needed
    if isinstance(date_obj, pd.Timestamp):
        date_obj = date_obj.to_pydatetime()
    
    # Platform-specific formatting (no leading zeros)
    if os.name == 'nt':  # Windows
        return date_obj.strftime('%#d/%#m/%Y')
    else:  # Unix/Linux/Mac
        return date_obj.strftime('%-d/%-m/%Y')


def parse_singapore_date(date_str: str) -> pd.Timestamp:
    """
    Parse Singapore format date string (D/M/YYYY or DD/MM/YYYY)
    
    Args:
        date_str: Date string in Singapore format
        
    Returns:
        pd.Timestamp object
        
    Examples:
        >>> parse_singapore_date('1/10/2025')
        Timestamp('2025-10-01 00:00:00')
        >>> parse_singapore_date('15/3/2024')
        Timestamp('2024-03-15 00:00:00')
    """
    return pd.to_datetime(date_str, dayfirst=True, format='mixed')


def format_eod_filename_to_date(filename: str) -> str:
    """
    Parse EOD filename to D/M/YYYY format
    
    Args:
        filename: EOD filename (e.g., '01_Oct_2025.csv')
        
    Returns:
        Date string in D/M/YYYY format (e.g., '1/10/2025')
        
    Examples:
        >>> format_eod_filename_to_date('01_Oct_2025.csv')
        '1/10/2025'
        >>> format_eod_filename_to_date('15_Mar_2024.csv')
        '15/3/2024'
    """
    date_str = filename.replace('.csv', '')
    date_obj = datetime.strptime(date_str, '%d_%b_%Y')
    return format_singapore_date(date_obj)


def parse_eod_filename(filename: str) -> datetime:
    """
    Parse EOD filename to datetime object
    
    Args:
        filename: EOD filename (e.g., '01_Oct_2025.csv')
        
    Returns:
        datetime object
        
    Examples:
        >>> parse_eod_filename('01_Oct_2025.csv')
        datetime.datetime(2025, 10, 1, 0, 0)
    """
    date_str = filename.replace('.csv', '')
    return datetime.strptime(date_str, '%d_%b_%Y')
