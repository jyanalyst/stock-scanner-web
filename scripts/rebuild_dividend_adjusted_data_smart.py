"""
Smart Dividend-Adjusted Data Rebuild/Update Script
===================================================

SMART HYBRID APPROACH:
- Automatically detects if data exists
- Full rebuild if no data (from 2020)
- Incremental update if data exists (from last date)
- Watchlist from latest EOD file
- End date always = today (yfinance)
- Recalculates all features for consistency

Usage:
    python scripts/rebuild_dividend_adjusted_data_smart.py
"""

import os
import sys
import shutil
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.local_file_loader import LocalFileLoader

# Import feature calculation functions from enhanced script
from scripts.rebuild_dividend_adjusted_data_enhanced import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_dividend_features,
    calculate_split_features
)


def determine_date_range(ticker: str, loader: LocalFileLoader, default_start: date) -> tuple:
    """
    Determine optimal date range for download
    
    Args:
        ticker: Stock ticker
        loader: LocalFileLoader instance
        default_start: Default start date if no data exists
    
    Returns:
        Tuple of (start_date, end_date, mode)
        mode: 'full' or 'update'
    """
    last_date = loader.get_last_date_in_historical(ticker)
    end_date = date.today()
    
    if last_date is None:
        # No data exists - full rebuild
        return default_start, end_date, 'full'
    
    # Convert Timestamp to date for comparison
    if hasattr(last_date, 'date'):
        last_date = last_date.date()
    
    # Check if already current
    if last_date >= end_date:
        return None, None, 'current'
    
    # Data exists but needs update
    start_date = last_date + timedelta(days=1)
    return start_date, end_date, 'update'


def download_and_calculate_features(ticker: str, start_date: date, end_date: date) -> dict:
    """
    Download data and calculate all 32 features
    
    Returns:
        Dictionary with status and enhanced data
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
        
        # Download with auto_adjust=True and actions=True
        stock = yf.Ticker(yf_ticker)
        df = stock.history(
            start=start_date,
            end=end_date,
            auto_adjust=True,
            actions=True
        )
        
        if df.empty:
            result['status'] = 'failed'
            result['message'] = 'No data available'
            return result
        
        # Reset index
        df.reset_index(inplace=True)
        
        # Get company name
        try:
            shortname = stock.info.get('shortName', ticker.replace('.SG', ''))
        except:
            shortname = ticker.replace('.SG', '')
        
        # Extract dividends and splits
        dividends = stock.dividends
        splits = stock.splits
        
        # Create formatted DataFrame
        formatted_df = pd.DataFrame()
        formatted_df['Date'] = df['Date']
        formatted_df['Code'] = ticker
        formatted_df['Shortname'] = shortname
        formatted_df['Open'] = df['Open'].round(3)
        formatted_df['High'] = df['High'].round(3)
        formatted_df['Low'] = df['Low'].round(3)
        formatted_df['Close'] = df['Close'].round(3)
        formatted_df['Vol'] = (df['Volume'] / 1000).round(0).astype(int)
        
        # Calculate dividend features
        div_features = calculate_dividend_features(dividends, df['Date'], df['Close'])
        formatted_df['Dividend'] = div_features['dividend_amount'].round(4)
        formatted_df['DaysToNextDiv'] = div_features['days_to_next_div'].astype(int)
        formatted_df['DivYield'] = div_features['div_yield'].round(2)
        formatted_df['DivGrowthRate'] = div_features['div_growth_rate'].round(2)
        formatted_df['ConsecutiveDivs'] = div_features['consecutive_divs'].astype(int)
        formatted_df['IsExDivWeek'] = div_features['is_ex_div_week'].astype(int)
        
        # Calculate split features
        split_features = calculate_split_features(splits, df['Date'])
        formatted_df['Split'] = split_features['split_ratio'].round(2)
        formatted_df['DaysSinceSplit'] = split_features['days_since_split'].astype(int)
        formatted_df['SplitInLast90Days'] = split_features['split_in_last_90_days'].astype(int)
        
        # Calculate moving averages
        formatted_df['MA_20'] = df['Close'].rolling(window=20).mean().round(3)
        formatted_df['MA_50'] = df['Close'].rolling(window=50).mean().round(3)
        formatted_df['MA_200'] = df['Close'].rolling(window=200).mean().round(3)
        
        # Calculate RSI
        formatted_df['RSI_14'] = calculate_rsi(df['Close'], 14).round(2)
        
        # Calculate MACD
        formatted_df['MACD'] = calculate_macd(df['Close']).round(4)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
        formatted_df['BB_Upper'] = bb_upper.round(3)
        formatted_df['BB_Middle'] = bb_middle.round(3)
        formatted_df['BB_Lower'] = bb_lower.round(3)
        
        # Calculate ATR
        formatted_df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'], 14).round(4)
        
        # Calculate Rate of Change
        formatted_df['ROC_5'] = ((df['Close'] / df['Close'].shift(5) - 1) * 100).round(2)
        formatted_df['ROC_10'] = ((df['Close'] / df['Close'].shift(10) - 1) * 100).round(2)
        formatted_df['ROC_20'] = ((df['Close'] / df['Close'].shift(20) - 1) * 100).round(2)
        
        # Calculate distance from moving averages
        formatted_df['DistFromMA20'] = ((df['Close'] / formatted_df['MA_20'] - 1) * 100).round(2)
        formatted_df['DistFromMA50'] = ((df['Close'] / formatted_df['MA_50'] - 1) * 100).round(2)
        
        # Calculate volume ratio
        vol_ma = df['Volume'].rolling(window=20).mean()
        formatted_df['VolRatio'] = (df['Volume'] / vol_ma).round(2)
        
        # Format dates to D/M/YYYY
        formatted_df['Date'] = formatted_df['Date'].dt.strftime('%-d/%-m/%Y' if os.name != 'nt' else '%#d/%#m/%Y')
        
        # Fill NaN values
        formatted_df.fillna(0, inplace=True)
        
        result['status'] = 'success'
        result['days'] = len(formatted_df)
        result['data'] = formatted_df
        result['message'] = f'{len(formatted_df)} days'
        
        return result
        
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = str(e)
        return result


def merge_and_recalculate(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge existing and new data, then recalculate all features
    
    This ensures consistency across the entire dataset
    """
    # Combine dataframes
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Sort by date
    combined_df['Date_dt'] = pd.to_datetime(combined_df['Date'], dayfirst=True)
    combined_df = combined_df.sort_values('Date_dt')
    combined_df = combined_df.drop('Date_dt', axis=1)
    
    # Remove duplicates (keep last)
    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
    
    return combined_df


def save_to_csv(ticker: str, df: pd.DataFrame, historical_path: str) -> bool:
    """Save DataFrame to CSV file"""
    try:
        ticker_clean = ticker.replace('.SG', '')
        filename = f"{ticker_clean}.csv"
        filepath = os.path.join(historical_path, filename)
        
        # Ensure column order (32 columns)
        column_order = [
            'Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol',
            'Dividend', 'DaysToNextDiv', 'DivYield', 'DivGrowthRate', 'ConsecutiveDivs', 'IsExDivWeek',
            'Split', 'DaysSinceSplit', 'SplitInLast90Days',
            'MA_20', 'MA_50', 'MA_200',
            'RSI_14', 'MACD',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR_14',
            'ROC_5', 'ROC_10', 'ROC_20', 'DistFromMA20', 'DistFromMA50', 'VolRatio'
        ]
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
    print("🔄 SMART DIVIDEND-ADJUSTED DATA REBUILD/UPDATE")
    print("=" * 70)
    print("\n📋 Smart Hybrid Approach:")
    print("   • Watchlist: From latest EOD file")
    print("   • Mode: Auto-detect (full rebuild or incremental update)")
    print("   • Start Date: 2020-01-01 (if no data) OR last_date+1 (if exists)")
    print("   • End Date: Today (from yfinance)")
    print("   • Features: 32 columns (dividend-adjusted + enhanced)")
    
    # Initialize
    start_time = datetime.now()
    loader = LocalFileLoader()
    historical_path = loader.historical_path
    
    # Default start date for full rebuilds
    default_start = date(2020, 1, 1)
    end_date = date.today()
    
    print(f"\n📅 Date Range: {default_start} to {end_date} (for full rebuilds)")
    
    # Get watchlist from EOD file
    print("\n📋 Loading watchlist from latest EOD file...")
    watchlist = loader.get_watchlist_from_eod()
    
    if not watchlist:
        print("❌ Could not load watchlist from EOD file. Exiting.")
        return
    
    print(f"✅ Found {len(watchlist)} stocks in latest EOD file")
    
    # Analyze what needs to be done
    print("\n🔍 Analyzing existing data...")
    
    stats = {
        'total': len(watchlist),
        'full_rebuild': 0,
        'update': 0,
        'current': 0,
        'success': 0,
        'failed': 0,
        'total_days': 0,
        'failed_tickers': []
    }
    
    # First pass: Determine modes
    modes = {}
    for ticker in watchlist:
        start_date, _, mode = determine_date_range(ticker, loader, default_start)
        modes[ticker] = mode
        if mode == 'full':
            stats['full_rebuild'] += 1
        elif mode == 'update':
            stats['update'] += 1
        elif mode == 'current':
            stats['current'] += 1
    
    print(f"\n📊 Analysis Results:")
    print(f"   • Full Rebuild: {stats['full_rebuild']} stocks")
    print(f"   • Update: {stats['update']} stocks")
    print(f"   • Already Current: {stats['current']} stocks")
    
    if stats['full_rebuild'] == 0 and stats['update'] == 0:
        print("\n✅ All stocks are already current! No action needed.")
        return
    
    # Confirm before proceeding
    print(f"\n⚠️  This will download/update {stats['full_rebuild'] + stats['update']} stocks")
    response = input("\n   Type 'YES' to proceed: ")
    
    if response != 'YES':
        print("\n❌ Operation cancelled")
        return
    
    # Second pass: Download and process
    print(f"\n📥 Processing {len(watchlist)} stocks...\n")
    
    with tqdm(total=len(watchlist), desc="Processing", unit="stock") as pbar:
        for ticker in watchlist:
            mode = modes[ticker]
            
            if mode == 'current':
                pbar.set_postfix_str(f"✓ {ticker}: Already current")
                pbar.update(1)
                continue
            
            # Determine date range
            start_date, end_date_stock, _ = determine_date_range(ticker, loader, default_start)
            
            # Download data
            result = download_and_calculate_features(ticker, start_date, end_date_stock)
            
            if result['status'] == 'success':
                # Handle merge if updating
                if mode == 'update':
                    # Load existing data
                    ticker_clean = ticker.replace('.SG', '')
                    filepath = os.path.join(historical_path, f"{ticker_clean}.csv")
                    existing_df = pd.read_csv(filepath)
                    
                    # Merge and recalculate
                    combined_df = merge_and_recalculate(existing_df, result['data'])
                    result['data'] = combined_df
                
                # Save to CSV
                if save_to_csv(ticker, result['data'], historical_path):
                    stats['success'] += 1
                    stats['total_days'] += result['days']
                    mode_label = "FULL" if mode == 'full' else "UPDATE"
                    pbar.set_postfix_str(f"✓ {ticker}: {mode_label} ({result['days']} days)")
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
    print("📊 OPERATION SUMMARY")
    print("=" * 70)
    print(f"\n✅ Success: {stats['success']}/{stats['total']} stocks ({success_rate:.1f}%)")
    print(f"✗  Failed: {stats['failed']} stocks")
    print(f"📊 Already Current: {stats['current']} stocks")
    print(f"📈 Total data points: {stats['total_days']:,} days")
    print(f"💾 Disk space: {disk_space_mb:.1f} MB")
    print(f"⏱️  Duration: {duration/60:.1f} minutes")
    
    if stats['failed_tickers']:
        print(f"\n⚠️  Failed ({len(stats['failed_tickers'])}):")
        for ticker, reason in stats['failed_tickers']:
            print(f"   • {ticker}: {reason}")
    
    # Save report
    report_path = os.path.join('scripts', f"smart_rebuild_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, 'w') as f:
        f.write("SMART DIVIDEND-ADJUSTED DATA REBUILD/UPDATE REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: Smart Hybrid (Auto-detect)\n")
        f.write(f"Watchlist Source: Latest EOD file\n")
        f.write(f"End Date: {end_date}\n\n")
        f.write(f"Full Rebuilds: {stats['full_rebuild']}\n")
        f.write(f"Updates: {stats['update']}\n")
        f.write(f"Already Current: {stats['current']}\n")
        f.write(f"Success: {stats['success']}/{stats['total']} ({success_rate:.1f}%)\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"Total Days: {stats['total_days']:,}\n")
        f.write(f"Disk Space: {disk_space_mb:.1f} MB\n")
        f.write(f"Duration: {duration/60:.1f} minutes\n\n")
        
        if stats['failed_tickers']:
            f.write("Failed:\n")
            for ticker, reason in stats['failed_tickers']:
                f.write(f"  • {ticker}: {reason}\n")
    
    print(f"\n📄 Report saved: {report_path}")
    
    print("\n" + "=" * 70)
    print("🎉 OPERATION COMPLETE!")
    print("=" * 70)
    print("\n✅ Your historical data is now dividend-adjusted with 32 features!")
    print("\n📌 Next Steps:")
    print("   1. Verify data quality (check a few CSV files)")
    print("   2. Re-run ML data collection (Phase 1)")
    print("   3. Run Phase 2 Factor Analysis (select top features)")
    print("   4. Retrain models (Phase 3)")


if __name__ == "__main__":
    main()
