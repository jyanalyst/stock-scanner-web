"""
Scanner Configuration - Scanner-specific settings and constants
"""

from pages.common.constants import DEFAULT_DAYS_BACK, DEFAULT_ROLLING_WINDOW

# Scanner-specific configuration
SCANNER_CONFIG = {
    'default_days_back': DEFAULT_DAYS_BACK,
    'default_rolling_window': DEFAULT_ROLLING_WINDOW,
    'max_single_stock_limit': 1,
    'progress_update_interval': 0.1,
    'max_display_rows': 1000,
    'default_page_size': 50
}

# Filter configuration
FILTER_CONFIG = {
    'velocity_percentiles': {
        "Top 25%": 75,
        "Top 50%": 50,
        "Top 75%": 25
    },
    'ibs_percentiles': {
        "Top 25%": 75,
        "Top 50%": 50,
        "Top 75%": 25
    },
    'rel_volume_percentiles': {
        "Top 25%": 75,
        "Top 50%": 50,
        "Top 75%": 25
    },
    'mpi_trends': ['Expanding', 'Flat', 'Contracting'],
    'hl_patterns': ['Higher H/L Only', 'Higher H Only', 'No Filter']
}

# Display configuration
DISPLAY_CONFIG = {
    'price_decimals_under_1': 3,
    'price_decimals_normal': 2,
    'mpi_visual_blocks': 10,
    'default_relative_volume': 100.0,
    'velocity_rounding': 4,
    'ibs_rounding': 3,
    'rel_volume_rounding': 1
}

# Column configurations for different views
FILTERED_RESULTS_COLUMNS = [
    'Analysis_Date', 'Ticker', 'Name', 'HL_Pattern',
    'VW_Range_Velocity', 'IBS', 'Relative_Volume',
    'MPI_Trend_Emoji', 'MPI_Visual'
]

FULL_RESULTS_COLUMNS = [
    'Analysis_Date', 'Ticker', 'Name', 'Close',
    'CRT_High', 'CRT_Low', 'HL_Pattern',
    'VW_Range_Velocity', 'IBS', 'Valid_CRT',
    'MPI_Trend_Emoji', 'MPI_Trend', 'MPI', 'MPI_Velocity', 'MPI_Visual'
]

# Base column configurations (shared with UI)
BASE_COLUMN_CONFIG = {
    'Analysis_Date': {'title': 'Date', 'width': 'small'},
    'Ticker': {'title': 'Ticker', 'width': 'small'},
    'Name': {'title': 'Company Name', 'width': 'medium'},
    'HL_Pattern': {'title': 'H/L', 'width': 'small', 'help': 'HHL=Both H&L, HH=Higher H only, -=Neither'},
    'VW_Range_Velocity': {'title': 'Range Vel', 'format': '%+.4f', 'help': 'Daily range expansion velocity'},
    'IBS': {'title': 'IBS', 'format': '%.3f'},
    'Relative_Volume': {'title': 'Rel Vol', 'format': '%.1f%%', 'help': 'Relative Volume vs 14-day average'},
    'MPI_Trend_Emoji': {'title': '📊', 'width': 'small', 'help': 'MPI Expansion Trend'},
    'MPI_Visual': {'title': 'MPI Visual', 'width': 'medium', 'help': 'Visual MPI representation'},
    'Sentiment_Display': {'title': 'Sentiment', 'width': 'small', 'help': 'Analyst sentiment score'},
    'Report_Date_Display': {'title': 'Report', 'width': 'small', 'help': 'Report date'},
    'Report_Count_Display': {'title': 'Reports', 'width': 'small', 'help': 'Number of reports'},
    'Earnings_Period': {'title': 'Period', 'width': 'small', 'help': 'Earnings period (Q1/Q2/FY etc.)'},
    'Guidance_Display': {'title': 'Guidance', 'width': 'small', 'help': 'Management guidance tone'},
    'Rev_YoY_Display': {'title': 'Rev YoY', 'width': 'small', 'help': 'Revenue year-over-year change'},
    'EPS_DPU_Display': {'title': 'EPS/DPU', 'width': 'small', 'help': 'EPS or DPU year-over-year change'}
}

# Export configuration
EXPORT_CONFIG = {
    'csv_filename_prefix': 'mpi_expansion',
    'csv_separator': ',',
    'tradingview_exchange': 'SGX',
    'max_export_rows': 1000
}