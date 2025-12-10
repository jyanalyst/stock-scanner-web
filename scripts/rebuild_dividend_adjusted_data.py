"""
Rebuild Historical Data with Dividend-Adjusted Prices
======================================================

This script completely rebuilds the Historical_Data folder with dividend-adjusted
prices from yfinance. This is CRITICAL for ML model accuracy.

User Preferences:
- Backup: NO (clean rebuild)
- Date Range: 2020-01-01 to today (FULL)
- Remove Old Backups: YES

Usage:
    python scripts/rebuild_dividend_adjusted_data.py
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
from utils.watchlist import get_active_watchlist


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


def clear_historical_data(historical_path: str):
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
    
    print(f"✅ Deleted {deleted_count} existing CSV file(s)")
    return deleted_count


def download_dividend_adjusted_data(ticker: str, start_date: date, end_date: date) -> dict:
    """
    Download dividend-adjusted data for a single stock
    
    Args:
        ticker: Stock ticker (e.g., 'A17U.SG')
        start_date: Start date
        end_date: End date
    
    Returns:
        Dictionary with status and data
    """
    result = {
        'ticker': ticker,
        'status': 'unknown',
        'message': '',
        'days': 0,
        'data': None
    }
    
    try:
        # Convert .SG to .SI for yfinance
        yf_ticker = ticker.replace('.SG', '.SI')
        
        # Download with auto_adjust=True for dividend adjustment
        stock = yf.Ticker(yf_ticker)
        df = stock.history(
            start=start_date,
            end=end_date,
            auto_adjust=True  # ← CRITICAL: Dividend-adjusted prices
        )
        
        if df.empty:
            result['status'] = 'failed'
            result['message'] = 'No data available'
            return result
        
        # Reset index to get Date as column
        df.reset_index(inplace=True)
        
        # Get company name
        try:
            shortname = stock.info.get('shortName', ticker.replace('.SG', ''))
        except:
            shortname = ticker.replace('.SG', '')
        
        # Create formatted DataFrame
        formatted_df = pd.DataFrame()
        
        # Format dates to D/M/YYYY (Singapore format)
        formatted_df['Date'] = df['Date'].dt.strftime('%-d/%-m/%Y' if os.name != 'nt' else '%#d/%#m/%Y')
        formatted_df['Code'] = ticker
        formatted_df['Shortname'] = shortname
        formatted_df['Open'] = df['Open'].round(3)
        formatted_df['High'] = df['High'].round(3)
        formatted_df['Low'] = df['Low'].round(3)
        formatted_df['Close'] = df['Close'].round(3)
        
        # Divide volumes by 1000 to match EOD abbreviated format
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
        
        # Ensure column order
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
    print("🔄 DIVIDEND-ADJUSTED DATA REBUILD")
    print("=" * 70)
    print("\n📋 Configuration:")
    print("   • Backup: NO (clean rebuild)")
    print("   • Date Range: 2020-01-01 to today (FULL)")
    print("   • Remove Old Backups: YES")
    print("   • Dividend Adjustment: ENABLED (auto_adjust=True)")
    print("   • Volume Scaling: ÷1000 (abbreviated format)")
    
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
    
    # Step 3: Get watchlist
    print("\n📋 Loading watchlist...")
    try:
        watchlist = get_active_watchlist()
        print(f"✅ Found {len(watchlist)} stocks in watchlist")
    except Exception as e:
        print(f"❌ Error loading watchlist: {e}")
        print("   Trying to extract from EOD file...")
        watchlist = loader.get_watchlist_from_eod()
        if not watchlist:
            print("❌ Could not load watchlist. Exiting.")
            return
        print(f"✅ Extracted {len(watchlist)} stocks from EOD file")
    
    # Step 4: Download dividend-adjusted data
    print(f"\n📥 Downloading dividend-adjusted data for {len(watchlist)} stocks...")
    print("   (This may take 25-30 minutes due to API rate limits)\n")
    
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
            # Download data
            result = download_dividend_adjusted_data(ticker, start_date, end_date)
            
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
    
    # Calculate statistics
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
    print(f"\n✅ Success: {stats['success']}/{stats['total']} stocks ({success_rate:.1f}%)")
    print(f"✗  Failed: {stats['failed']} stocks")
    print(f"📈 Total data points: {stats['total_days']:,} days")
    print(f"💾 Disk space: {disk_space_mb:.1f} MB")
    print(f"⏱️  Duration: {duration/60:.1f} minutes")
    
    if stats['failed_tickers']:
        print(f"\n⚠️  Failed Downloads ({len(stats['failed_tickers'])}):")
        for ticker, reason in stats['failed_tickers']:
            print(f"   • {ticker}: {reason}")
    
    # Save report
    report_path = os.path.join('scripts', f"rebuild_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, 'w') as f:
        f.write("DIVIDEND-ADJUSTED DATA REBUILD REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Date Range: {start_date} to {end_date}\n")
        f.write(f"Dividend Adjustment: ENABLED\n\n")
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
    print("\n✅ Your historical data is now dividend-adjusted!")
    print("✅ This will significantly improve ML model accuracy.")
    print("\n📌 Next Steps:")
    print("   1. Verify data quality (check a few CSV files)")
    print("   2. Re-run ML data collection (Phase 1)")
    print("   3. Retrain models (Phase 3)")
    print("   4. Validate new models (Phase 4)")


if __name__ == "__main__":
    main()
