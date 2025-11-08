"""
Common constants used across the application
"""

# MPI Trend categories
MPI_TRENDS = ['Expanding', 'Flat', 'Contracting']

# Break H/L patterns
HL_PATTERNS = {
    'BHL': 'Both break high AND break low',
    'BH': 'Break high only (not break low)',
    '-': 'Neither break high nor break low'
}

# Sentiment labels
SENTIMENT_LABELS = ['positive', 'neutral', 'negative']

# Guidance tones
GUIDANCE_TONES = ['positive', 'neutral', 'negative']

# Default configuration values
DEFAULT_DAYS_BACK = 100
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_MPI_THRESHOLD = 0.3

# File extensions
CSV_EXTENSION = '.csv'
JSON_EXTENSION = '.json'

# Date formats
SINGAPORE_DATE_FORMAT = '%d/%m/%Y'
DISPLAY_DATE_FORMAT = '%d/%m/%Y'

# Volume scaling factor for yfinance data
YFINANCE_VOLUME_SCALE = 1000

# Progress bar update intervals
PROGRESS_UPDATE_INTERVAL = 0.1

# Cache TTL values (seconds)
DEFAULT_CACHE_TTL = 3600  # 1 hour
SHORT_CACHE_TTL = 300     # 5 minutes
LONG_CACHE_TTL = 86400    # 24 hours

# Data validation thresholds
MIN_DAYS_REQUIRED = 30
MAX_DAYS_OLD = 7
MIN_DATA_POINTS = 20

# UI constants
MAX_ROWS_DISPLAY = 1000
DEFAULT_PAGE_SIZE = 50

# Error messages
ERROR_DATA_LOAD = "Failed to load data"
ERROR_PROCESSING = "Processing failed"
ERROR_VALIDATION = "Data validation failed"