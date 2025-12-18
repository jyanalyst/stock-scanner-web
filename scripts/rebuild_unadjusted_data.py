"""
Rebuild Historical Data with UNADJUSTED (Pristine) Prices
==========================================================

PURPOSE: Create clean, raw OHLCV data that reflects actual traded prices.
This is critical for price action strategies where signals are based on
real support/resistance levels and breakout points.

KEY DIFFERENCE FROM ADJUSTED SCRIPTS:
- auto_adjust=False → Returns actual traded prices (not dividend-adjusted)
- No derived features → Keep data pristine for signal generation
- 8 columns only → Date, Code, Shortname, Open, High, Low, Close, Vol

WHY UNADJUSTED:
- Buy-stop orders at previous day's high use REAL prices
- Support/resistance levels reflect actual market prices
- CRT patterns based on real candle ranges
- No artificial gaps from dividend adjustments

Usage:
    python scripts/rebuild_unadjusted_data.py
"""

import os
import sys
import shutil
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.local_file_loader import LocalFileLoader


def remove_old_backups(data_dir: str):
    """Remove all old backup folders"""
    print("\n🗑️  Removing old backup folders...")
    
    removed_count = 0
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and 'BACKUP' in item:
            try:
                shutil.rmtree(item_path)
                print(f"   ✓ Removed: {item}")
                removed_count += 1
            except Exception as e:
                print(f"   ✗ Failed to remove {item}: {e}")
    
    if removed_count > 0:
        print(f"✅ Removed {removed_count} old backup folder(s)")
    else:
        print("   No old backups found")


def clear_historical_data(historical_path: str) -> int:
    """Delete all CSV files in Historical_Data folder"""
    print("\n🧹 Clearing existing historical data...")
    
    if not os.path.exists(historical_path):
        print(f"   Creating folder: {historical_path}")
        os.makedirs(historical_path, exist_ok=True)
        return 0
    
    deleted_count = 0
    for filename in os.listdir(historical_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(historical_path, filename)
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"   ✗ Failed to delete {filename}: {e}")
    
    print(f"✅ Deleted {deleted_count} existing file(s)")
    return deleted_count


def download_unadjusted_data(ticker: str, start_date: date, end_date: date) -> dict:
    """
    Download UNADJUSTED historical data from yfinance
    
    CRITICAL: auto_adjust=False returns actual traded prices
    
    Args:
        ticker: Stock ticker (e.g., 'A17U.SG')
        start_date: Start date
        end_date: End date
    
    Returns:
        Dictionary with status, data, and metadata
    """
    result = {
        'ticker': ticker,
        'status': 'pending',
        'days': 0,
        'data': None,
        'message': ''
    }
    
    try:
        # Convert .SG to .SI for yfinance (SGX uses .SI suffix in yfinance)
        ticker_yf = ticker.replace('.SG', '.SI')
        ticker_clean = ticker.replace('.SG', '')
        
        # Download from yfinance with auto_adjust=FALSE (CRITICAL!)
        # This gives us actual traded prices, not dividend-adjusted
        stock = yf.Ticker(ticker_yf)
        df = stock.history(
            start=start_date,
            end=end_date,
            auto_adjust=False,  # <-- CRITICAL: Unadjusted prices
            actions=False       # Don't need dividend/split columns
        )
        
        if df.empty:
            result['status'] = 'failed'
            result['message'] = 'No data returned'
            return result
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Get short name for the ticker
        try:
            shortname = stock.info.get('shortName', ticker_clean)[:20]
        except:
            shortname = ticker_clean
        
        # Create formatted DataFrame with 8 columns only (pristine)
        formatted_df = pd.DataFrame()
        
        # Format date to D/M/YYYY
        formatted_df['Date'] = df['Date'].dt.strftime('%-d/%-m/%Y' if os.name != 'nt' else '%#d/%#m/%Y')
        
        # Add identifiers (use original .SG ticker, not .SI)
        formatted_df['Code'] = ticker  # Original ticker with .SG
        formatted_df['Shortname'] = shortname
        
        # UNADJUSTED OHLC prices (rounded to 3 decimals)
        formatted_df['Open'] = df['Open'].round(3)
        formatted_df['High'] = df['High'].round(3)
        formatted_df['Low'] = df['Low'].round(3)
        formatted_df['Close'] = df['Close'].round(3)
        
        # Volume divided by 1000 to match EOD abbreviated format
        formatted_df['Vol'] = (df['Volume'] / 1000).round(0).astype(int)
        
        result['status'] = 'success'
        result['days'] = len(formatted_df)
        result['data'] = formatted_df
        result['message'] = f'{len(formatted_df)} days'
        
        return result
        
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = str(e)
        return result


def save_to_csv(ticker: str, df: pd.DataFrame, historical_path: str) -> bool:
    """Save DataFrame to CSV file"""
    try:
        ticker_clean = ticker.replace('.SG', '')
        filename = f"{ticker_clean}.csv"
        filepath = os.path.join(historical_path, filename)
        
        # Ensure column order (8 columns - pristine)
        column_order = ['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol']
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error saving {ticker}: {e}")
        return False


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("🔄 UNADJUSTED (PRISTINE) DATA REBUILD")
    print("=" * 70)
    print("\n📋 Configuration:")
    print("   • Backup: NO (clean rebuild)")
    print("   • Date Range: 2020-01-01 to today (FULL)")
    print("   • Remove Old Backups: YES")
    print("   • Dividend Adjustment: DISABLED (auto_adjust=False)")
    print("   • Columns: 8 (Date, Code, Shortname, Open, High, Low, Close, Vol)")
    print("   • Volume Scaling: ÷1000 (abbreviated format)")
    print("\n📌 Why Unadjusted?")
    print("   • Real prices for price action signals")
    print("   • Accurate support/resistance levels")
    print("   • True buy-stop entry points")
    print("   • No artificial dividend gaps")
    
    # Confirm before proceeding
    print("\n⚠️  WARNING: This will DELETE all existing historical data!")
    response = input("\n   Type 'YES' to proceed: ")
    
    if response != 'YES':
        print("\n❌ Rebuild cancelled")
        return
    
    # Initialize
    start_time = datetime.now()
    loader = LocalFileLoader()
    historical_path = loader.historical_path
    data_dir = os.path.dirname(historical_path)
    
    # Date range
    start_date = date(2020, 1, 1)
    end_date = date.today()
    
    print(f"\n📅 Date Range: {start_date} to {end_date}")
    
    # Step 1: Remove old backups
    remove_old_backups(data_dir)
    
    # Step 2: Clear existing data
    deleted_count = clear_historical_data(historical_path)
    
    # Step 3: Get watchlist from EOD file
    print("\n📋 Loading watchlist from latest EOD file...")
    watchlist = loader.get_watchlist_from_eod()
    
    if not watchlist:
        print("❌ Could not load watchlist from EOD file. Exiting.")
        return
    
    print(f"✅ Found {len(watchlist)} stocks in watchlist")
    
    # Step 4: Download unadjusted data
    print(f"\n📥 Downloading UNADJUSTED data for {len(watchlist)} stocks...")
    print("   (This may take a few minutes)\n")
    
    stats = {
        'total': len(watchlist),
        'success': 0,
        'failed': 0,
        'total_days': 0,
        'failed_tickers': []
    }
    
    # Progress bar
    with tqdm(total=len(watchlist), desc="Downloading", unit="stock") as pbar:
        for ticker in watchlist:
            # Download unadjusted data
            result = download_unadjusted_data(ticker, start_date, end_date)
            
            if result['status'] == 'success':
                # Save to CSV
                if save_to_csv(ticker, result['data'], historical_path):
                    stats['success'] += 1
                    stats['total_days'] += result['days']
                    pbar.set_postfix_str(f"✓ {ticker}: {result['days']} days")
                else:
                    stats['failed'] += 1
                    stats['failed_tickers'].append((ticker, 'Save failed'))
                    pbar.set_postfix_str(f"✗ {ticker}: Save failed")
            else:
                stats['failed'] += 1
                stats['failed_tickers'].append((ticker, result['message']))
                pbar.set_postfix_str(f"✗ {ticker}: {result['message']}")
            
            pbar.update(1)
    
    # Calculate summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    # Calculate disk space
    total_size = 0
    for filename in os.listdir(historical_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(historical_path, filename)
            total_size += os.path.getsize(filepath)
    disk_space_mb = total_size / (1024 * 1024)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 REBUILD SUMMARY")
    print("=" * 70)
    print(f"✅ Success: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📈 Total days: {stats['total_days']:,}")
    print(f"💾 Disk space: {disk_space_mb:.1f} MB")
    print(f"⏱️  Duration: {duration/60:.1f} minutes")
    
    if stats['failed_tickers']:
        print(f"\n⚠️  Failed Downloads ({len(stats['failed_tickers'])}):")
        for ticker, reason in stats['failed_tickers']:
            print(f"   • {ticker}: {reason}")
    
    # Save report
    report_path = os.path.join('scripts', f"rebuild_unadjusted_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs('scripts', exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("UNADJUSTED (PRISTINE) DATA REBUILD REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Date Range: {start_date} to {end_date}\n")
        f.write(f"Dividend Adjustment: DISABLED (auto_adjust=False)\n")
        f.write(f"Columns: 8 (Date, Code, Shortname, Open, High, Low, Close, Vol)\n\n")
        f.write(f"Success: {stats['success']}/{stats['total']} ({success_rate:.1f}%)\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"Total Days: {stats['total_days']:,}\n")
        f.write(f"Disk Space: {disk_space_mb:.1f} MB\n")
        f.write(f"Duration: {duration/60:.1f} minutes\n\n")
        
        if stats['failed_tickers']:
            f.write("Failed Downloads:\n")
            for ticker, reason in stats['failed_tickers']:
                f.write(f"  • {ticker}: {reason}\n")
    
    print(f"\n📄 Report saved: {report_path}")
    
    print("\n" + "=" * 70)
    print("🎉 REBUILD COMPLETE!")
    print("=" * 70)
    print("\n✅ Your historical data is now UNADJUSTED (pristine)!")
    print("✅ Prices reflect actual traded values.")
    print("\n📌 Key Points:")
    print("   • Use this data for signal generation (price action)")
    print("   • For P&L calculations, you may need adjusted close separately")
    print("   • Features should be computed separately in EOD files")


if __name__ == "__main__":
    main()