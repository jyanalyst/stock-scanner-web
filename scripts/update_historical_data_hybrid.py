"""
Hybrid Historical Data Updater - UNADJUSTED VERSION
=====================================================

PURPOSE: Update Historical_Data with pristine, unadjusted prices.
This version does NOT apply dividend adjustments, keeping prices
exactly as they traded in the market.

DATA SOURCES:
- Priority 1: EOD_Data (manual CSV files - used as-is, no adjustment)
- Priority 2: yfinance (with auto_adjust=False for real prices)

COLUMN FORMAT (8 columns - pristine):
- Date, Code, Shortname, Open, High, Low, Close, Vol

WHY UNADJUSTED:
- Real prices for price action signals
- Accurate support/resistance levels
- True buy-stop entry points
- No artificial dividend gaps

Usage:
    python scripts/update_historical_data_hybrid.py
"""

import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.local_file_loader import LocalFileLoader


def parse_eod_date_from_filename(filename: str) -> date:
    """
    Parse date from EOD filename (e.g., '18_Dec_2025.csv' -> date(2025, 12, 18))
    
    Args:
        filename: EOD filename
    
    Returns:
        Parsed date
    """
    try:
        # Remove .csv extension
        name = filename.replace('.csv', '')
        # Parse date (format: DD_MMM_YYYY)
        return datetime.strptime(name, '%d_%b_%Y').date()
    except Exception as e:
        print(f"   ⚠️  Could not parse date from filename: {filename}")
        return None


def process_eod_file(eod_filename: str, loader: LocalFileLoader) -> dict:
    """
    Process EOD file and extract data for each ticker (NO dividend adjustment)
    
    New EOD format columns:
    - (empty), Code, Shortname, Open, High, Low, Last, Volume, (empty)
    
    Args:
        eod_filename: EOD filename
        loader: LocalFileLoader instance
    
    Returns:
        Dict[ticker, row_data] with unadjusted prices
    """
    eod_data = {}
    
    try:
        filepath = os.path.join(loader.eod_path, eod_filename)
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # Parse date from filename
        eod_date = parse_eod_date_from_filename(eod_filename)
        if eod_date is None:
            print(f"   ❌ Could not parse date from {eod_filename}")
            return eod_data
        
        print(f"\n📊 Processing EOD file: {eod_filename}")
        print(f"   Date: {eod_date}")
        print(f"   Stocks: {len(df)}")
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                ticker = row['Code']
                
                # Skip invalid tickers
                if pd.isna(ticker) or str(ticker).strip() == '':
                    continue
                
                # Skip rows with missing price data
                # Use 'Last' column for close price (new format)
                close_col = 'Last' if 'Last' in df.columns else 'Close'
                if pd.isna(row[close_col]) or row[close_col] == 0:
                    continue
                
                # Get shortname (remove .SG suffix if present in shortname)
                shortname = row['Shortname'] if 'Shortname' in df.columns else ticker.replace('.SG', '')
                if isinstance(shortname, str) and shortname.endswith('.SG'):
                    shortname = shortname.replace('.SG', '')
                
                # Extract UNADJUSTED prices (use as-is from EOD)
                eod_data[ticker] = {
                    'Date': eod_date,
                    'Code': ticker,
                    'Shortname': shortname[:20] if isinstance(shortname, str) else str(shortname)[:20],
                    'Open': round(float(row['Open']), 3) if not pd.isna(row['Open']) else 0,
                    'High': round(float(row['High']), 3) if not pd.isna(row['High']) else 0,
                    'Low': round(float(row['Low']), 3) if not pd.isna(row['Low']) else 0,
                    'Close': round(float(row[close_col]), 3),
                    'Vol': int(round(float(row['Volume']), 0)) if not pd.isna(row['Volume']) else 0
                }
                
            except Exception as e:
                print(f"   ⚠️  Error processing row {idx}: {e}")
                continue
        
        print(f"✅ Processed {len(eod_data)} stocks (unadjusted prices)")
        
        return eod_data
        
    except Exception as e:
        print(f"❌ Error processing EOD file: {e}")
        return eod_data


def fill_gaps_with_yfinance(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fill gaps with yfinance data (UNADJUSTED prices)
    
    CRITICAL: Uses auto_adjust=False for real traded prices
    
    Args:
        ticker: Stock ticker (e.g., 'A17U.SG')
        start_date: Start date for gap
        end_date: End date for gap
    
    Returns:
        DataFrame with gap data
    """
    try:
        # Convert .SG to .SI for yfinance
        ticker_yf = ticker.replace('.SG', '.SI')
        ticker_clean = ticker.replace('.SG', '')
        
        stock = yf.Ticker(ticker_yf)
        df = stock.history(
            start=start_date,
            end=end_date + timedelta(days=1),  # Include end date
            auto_adjust=False,  # <-- CRITICAL: Unadjusted prices
            actions=False
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Reset index
        df = df.reset_index()
        
        # Get shortname
        try:
            shortname = stock.info.get('shortName', ticker_clean)[:20]
        except:
            shortname = ticker_clean
        
        # Create formatted DataFrame
        result_df = pd.DataFrame()
        result_df['Date'] = df['Date']
        result_df['Code'] = ticker
        result_df['Shortname'] = shortname
        result_df['Open'] = df['Open'].round(3)
        result_df['High'] = df['High'].round(3)
        result_df['Low'] = df['Low'].round(3)
        result_df['Close'] = df['Close'].round(3)
        result_df['Vol'] = (df['Volume'] / 1000).round(0).astype(int)
        
        return result_df
        
    except Exception as e:
        return pd.DataFrame()


def update_single_ticker(
    ticker: str,
    eod_data: dict,
    loader: LocalFileLoader,
    current_working_day: date
) -> dict:
    """
    Update a single ticker with hybrid approach (UNADJUSTED)
    
    Args:
        ticker: Stock ticker
        eod_data: EOD data (if available)
        loader: LocalFileLoader instance
        current_working_day: Current working day
    
    Returns:
        Result dictionary
    """
    result = {
        'ticker': ticker,
        'status': 'unknown',
        'message': '',
        'eod_days': 0,
        'yfinance_days': 0,
        'total_days': 0
    }
    
    try:
        # Load existing historical data
        historical_df = loader.load_historical_data(ticker)
        
        if historical_df is None or historical_df.empty:
            historical_df = pd.DataFrame()
            last_hist_date = None
        else:
            historical_df = historical_df.reset_index()
            # Get last date
            last_date_value = historical_df['Date'].iloc[-1]
            if hasattr(last_date_value, 'date'):
                last_hist_date = last_date_value.date()
            else:
                last_hist_date = pd.to_datetime(last_date_value, dayfirst=True).date()
        
        # Prepare list of new data to add
        new_rows = []
        
        # Add EOD data if available
        if ticker in eod_data:
            eod_row = eod_data[ticker]
            eod_date = eod_row['Date']
            
            # Check if EOD date is newer than last historical date
            if last_hist_date is None or eod_date > last_hist_date:
                new_rows.append(eod_row)
                result['eod_days'] = 1
                last_hist_date = eod_date
        
        # Check for gaps and fill with yfinance
        if last_hist_date is not None:
            # Ensure it's a plain date object
            if hasattr(last_hist_date, 'date'):
                last_hist_date = last_hist_date.date()
            
            if last_hist_date < current_working_day:
                gap_start = last_hist_date + timedelta(days=1)
                gap_end = current_working_day
                
                # Download gap data from yfinance (UNADJUSTED)
                yf_df = fill_gaps_with_yfinance(ticker, gap_start, gap_end)
                
                if not yf_df.empty:
                    for idx, row in yf_df.iterrows():
                        new_rows.append({
                            'Date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                            'Code': row['Code'],
                            'Shortname': row['Shortname'],
                            'Open': row['Open'],
                            'High': row['High'],
                            'Low': row['Low'],
                            'Close': row['Close'],
                            'Vol': row['Vol']
                        })
                    
                    result['yfinance_days'] = len(yf_df)
        
        # If no new data, skip
        if not new_rows:
            result['status'] = 'current'
            result['message'] = 'Already current'
            return result
        
        # Merge historical + new data
        if not historical_df.empty:
            # Convert historical dates to datetime for merging
            historical_df['Date'] = pd.to_datetime(historical_df['Date'], dayfirst=True)
            
            # Keep only 8 columns
            if 'Volume' in historical_df.columns:
                hist_basic = historical_df[['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                hist_basic = hist_basic.rename(columns={'Volume': 'Vol'})
            elif 'Vol' in historical_df.columns:
                hist_basic = historical_df[['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol']].copy()
            else:
                # Fallback - take first 8 columns
                hist_basic = historical_df.iloc[:, :8].copy()
                hist_basic.columns = ['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol']
        else:
            hist_basic = pd.DataFrame()
        
        # Create DataFrame from new rows
        new_df = pd.DataFrame(new_rows)
        
        # Ensure Date is datetime
        if 'Date' in new_df.columns:
            if new_df['Date'].dtype == 'object':
                new_df['Date'] = pd.to_datetime(new_df['Date'])
            elif not pd.api.types.is_datetime64_any_dtype(new_df['Date']):
                new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # Combine
        if not hist_basic.empty:
            combined_df = pd.concat([hist_basic, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Sort by date
        combined_df = combined_df.sort_values('Date')
        
        # Remove duplicates (keep last)
        combined_df['Date_str'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        combined_df = combined_df.drop_duplicates(subset=['Date_str'], keep='last')
        combined_df = combined_df.drop('Date_str', axis=1)
        
        # Format dates to D/M/YYYY
        combined_df['Date'] = combined_df['Date'].dt.strftime('%-d/%-m/%Y' if os.name != 'nt' else '%#d/%#m/%Y')
        
        # Ensure column order (8 columns - pristine)
        column_order = ['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol']
        combined_df = combined_df[column_order]
        
        # Save to CSV
        ticker_clean = ticker.replace('.SG', '')
        filename = f"{ticker_clean}.csv"
        filepath = os.path.join(loader.historical_path, filename)
        
        combined_df.to_csv(filepath, index=False, encoding='utf-8')
        
        result['status'] = 'success'
        result['total_days'] = len(new_rows)
        result['message'] = f"Added {result['total_days']} days (EOD: {result['eod_days']}, yfinance: {result['yfinance_days']})"
        
        return result
        
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = str(e)
        return result


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("🔄 HYBRID HISTORICAL DATA UPDATER (UNADJUSTED)")
    print("=" * 70)
    print("\n📋 Strategy:")
    print("   1. Process latest EOD file (use as-is, no dividend adjustment)")
    print("   2. Fill gaps with yfinance (auto_adjust=False)")
    print("   3. Keep pristine 8-column format")
    print("   4. Save updated data")
    print("\n📌 Key Settings:")
    print("   • Dividend Adjustment: DISABLED")
    print("   • Columns: 8 (Date, Code, Shortname, Open, High, Low, Close, Vol)")
    print("   • yfinance: auto_adjust=False")
    
    # Initialize
    start_time = datetime.now()
    loader = LocalFileLoader()
    
    # Step 1: Get latest EOD file
    print("\n" + "=" * 70)
    print("📊 STEP 1: Check for New EOD Data")
    print("=" * 70)
    
    latest_eod = loader.get_latest_eod_file()
    
    if not latest_eod:
        print("   ⚠️  No EOD files found")
        print("   Will use yfinance only...")
        eod_data = {}
        eod_date = None
    else:
        print(f"   Latest EOD file: {latest_eod}")
        eod_date = parse_eod_date_from_filename(latest_eod)
    
    # Get watchlist
    watchlist = loader.get_watchlist_from_eod()
    
    if not watchlist:
        print("❌ Could not load watchlist. Exiting.")
        return
    
    print(f"   Watchlist: {len(watchlist)} stocks")
    
    # Check last historical date
    sample_ticker = watchlist[0]
    last_hist_date = loader.get_last_date_in_historical(sample_ticker)
    
    if last_hist_date:
        if hasattr(last_hist_date, 'date'):
            last_hist_date = last_hist_date.date()
        print(f"   Last historical date: {last_hist_date}")
    else:
        print(f"   No historical data found")
    
    # Process EOD file if needed
    eod_data = {}
    if latest_eod and eod_date:
        if last_hist_date is None or eod_date > last_hist_date:
            print(f"   ✅ New EOD data available: {eod_date}")
            eod_data = process_eod_file(latest_eod, loader)
        else:
            print(f"   ℹ️  EOD file already processed (date: {eod_date})")
    
    # Step 2: Determine current working day and check for gaps
    print("\n" + "=" * 70)
    print("📊 STEP 2: Check for Gaps")
    print("=" * 70)
    
    current_working_day = loader.get_current_working_day()
    print(f"   Current working day: {current_working_day}")
    
    if last_hist_date:
        if last_hist_date < current_working_day:
            gap_days = (current_working_day - last_hist_date).days
            print(f"   ⚠️  Gap detected: {gap_days} day(s)")
            print(f"   Will fill with yfinance (unadjusted)...")
        else:
            print(f"   ✅ No gaps detected")
    
    # Step 3: Update all tickers
    print("\n" + "=" * 70)
    print("📊 STEP 3: Update Historical Data")
    print("=" * 70)
    print(f"\n   Processing {len(watchlist)} stocks...\n")
    
    stats = {
        'total': len(watchlist),
        'success': 0,
        'current': 0,
        'failed': 0,
        'total_eod_days': 0,
        'total_yfinance_days': 0,
        'total_days_added': 0,
        'details': []
    }
    
    with tqdm(total=len(watchlist), desc="Updating", unit="stock") as pbar:
        for ticker in watchlist:
            result = update_single_ticker(
                ticker,
                eod_data,
                loader,
                current_working_day
            )
            
            stats['details'].append(result)
            
            if result['status'] == 'success':
                stats['success'] += 1
                stats['total_eod_days'] += result['eod_days']
                stats['total_yfinance_days'] += result['yfinance_days']
                stats['total_days_added'] += result['total_days']
                pbar.set_postfix_str(f"✓ {ticker}: {result['message']}")
            elif result['status'] == 'current':
                stats['current'] += 1
                pbar.set_postfix_str(f"○ {ticker}: Current")
            else:
                stats['failed'] += 1
                pbar.set_postfix_str(f"✗ {ticker}: {result['message']}")
            
            pbar.update(1)
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 UPDATE SUMMARY")
    print("=" * 70)
    print(f"\n✅ Updated: {stats['success']}/{stats['total']} stocks")
    print(f"○  Already Current: {stats['current']} stocks")
    print(f"✗  Failed: {stats['failed']} stocks")
    print(f"\n📈 Data Added:")
    print(f"   • EOD days: {stats['total_eod_days']}")
    print(f"   • yfinance days: {stats['total_yfinance_days']}")
    print(f"   • Total data points: {stats['total_days_added']}")
    print(f"\n⏱️  Duration: {duration:.1f} seconds")
    
    # Save report
    report_path = os.path.join('scripts', f"hybrid_update_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs('scripts', exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("HYBRID HISTORICAL DATA UPDATE REPORT (UNADJUSTED)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"EOD File: {latest_eod}\n")
        f.write(f"Current Working Day: {current_working_day}\n")
        f.write(f"Dividend Adjustment: DISABLED\n")
        f.write(f"Columns: 8 (pristine)\n\n")
        f.write(f"Updated: {stats['success']}/{stats['total']}\n")
        f.write(f"Already Current: {stats['current']}\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"EOD Days Added: {stats['total_eod_days']}\n")
        f.write(f"yfinance Days Added: {stats['total_yfinance_days']}\n")
        f.write(f"Total Data Points: {stats['total_days_added']}\n")
        f.write(f"Duration: {duration:.1f} seconds\n\n")
        
        f.write("Details:\n")
        for detail in stats['details']:
            f.write(f"  {detail['ticker']}: {detail['status']} - {detail['message']}\n")
    
    print(f"\n📄 Report saved: {report_path}")
    
    print("\n" + "=" * 70)
    print("🎉 HYBRID UPDATE COMPLETE!")
    print("=" * 70)
    print("\n✅ Your historical data is now current with UNADJUSTED prices!")
    print("✅ 8-column pristine format preserved.")
    print("\n📌 Next Steps:")
    print("   • Run this script daily to keep data current")
    print("   • Features are computed separately in EOD scanner")


if __name__ == "__main__":
    main()