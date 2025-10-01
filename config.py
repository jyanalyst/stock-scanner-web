# File: config.py
"""
Configuration for Stock Scanner Application
Centralized configuration for data paths and application settings
"""
import os

# Data paths - relative to application root
# Override with environment variables if needed
HISTORICAL_DATA_PATH = os.getenv('HISTORICAL_DATA_PATH', './data/Historical_Data')
EOD_DATA_PATH = os.getenv('EOD_DATA_PATH', './data/EOD_Data')

# Application settings
DEFAULT_DAYS_BACK = 100
WATCHLIST_CACHE_TTL = 3600  # 1 hour in seconds

# Date format settings
SINGAPORE_DATE_FORMAT = 'D/M/YYYY'  # e.g., 1/10/2025
EOD_FILENAME_FORMAT = '%d_%b_%Y'     # e.g., 01_Oct_2025.csv

# Data validation settings
MIN_DAYS_REQUIRED = 30
MAX_DAYS_OLD = 7  # Maximum age of latest data in days