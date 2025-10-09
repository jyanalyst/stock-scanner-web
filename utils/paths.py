# File: utils/paths.py
"""
Path utilities for data directories in Codespace environment
Integrates with existing config.py structure
"""
from pathlib import Path
import os


def get_data_dir():
    """Get data directory (uses existing config.py structure)"""
    # Import from your existing config
    from config import HISTORICAL_DATA_PATH, EOD_DATA_PATH
    
    # Get parent of Historical_Data (which is ./data)
    data_dir = Path(HISTORICAL_DATA_PATH).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_stock_data_dir():
    """
    Get stock data directory (Historical_Data)
    Uses your existing HISTORICAL_DATA_PATH from config.py
    """
    from config import HISTORICAL_DATA_PATH
    stock_dir = Path(HISTORICAL_DATA_PATH)
    stock_dir.mkdir(parents=True, exist_ok=True)
    return stock_dir


def get_eod_data_dir():
    """
    Get EOD data directory
    Uses your existing EOD_DATA_PATH from config.py
    """
    from config import EOD_DATA_PATH
    eod_dir = Path(EOD_DATA_PATH)
    eod_dir.mkdir(parents=True, exist_ok=True)
    return eod_dir


def get_analyst_reports_dir():
    """Get analyst reports JSON directory"""
    reports_dir = get_data_dir() / "analyst_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def get_analyst_pdf_dir():
    """Get analyst reports PDF directory"""
    pdf_dir = get_data_dir() / "analyst_reports_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return pdf_dir


def get_earnings_reports_dir():
    """Get earnings reports JSON directory (NEW)"""
    reports_dir = get_data_dir() / "earnings_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def get_earnings_pdf_dir():
    """Get earnings reports PDF directory (NEW)"""
    pdf_dir = get_data_dir() / "earnings_reports_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return pdf_dir


def get_backup_dir():
    """Get backup directory"""
    backup_dir = Path("backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


# Convenience exports
DATA_DIR = get_data_dir()
STOCK_DATA_DIR = get_stock_data_dir()  # Your existing Historical_Data
EOD_DATA_DIR = get_eod_data_dir()      # Your existing EOD_Data
ANALYST_REPORTS_DIR = get_analyst_reports_dir()
ANALYST_PDF_DIR = get_analyst_pdf_dir()
EARNINGS_REPORTS_DIR = get_earnings_reports_dir()  # NEW
EARNINGS_PDF_DIR = get_earnings_pdf_dir()          # NEW
BACKUP_DIR = get_backup_dir()


if __name__ == "__main__":
    # Test the paths
    print("=" * 60)
    print("STOCK SCANNER - PATH CONFIGURATION")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR.absolute()}")
    print(f"Historical_Data: {STOCK_DATA_DIR.absolute()}")
    print(f"EOD_Data: {EOD_DATA_DIR.absolute()}")
    print(f"Analyst reports (JSON): {ANALYST_REPORTS_DIR.absolute()}")
    print(f"Analyst reports (PDF): {ANALYST_PDF_DIR.absolute()}")
    print(f"Earnings reports (JSON): {EARNINGS_REPORTS_DIR.absolute()}")
    print(f"Earnings reports (PDF): {EARNINGS_PDF_DIR.absolute()}")
    print(f"Backups: {BACKUP_DIR.absolute()}")
    print("\n✓ All paths configured correctly")
    print("=" * 60)