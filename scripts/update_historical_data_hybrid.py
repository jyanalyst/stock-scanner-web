"""
Hybrid Historical Data Updater
===============================

HYBRID APPROACH:
- Priority 1: EOD_Data (manual CSV files with dividend adjustment)
- Priority 2: yfinance (automated API, already dividend-adjusted)
- Auto-fills gaps between data sources

Features:
- Loads dividend calendar for proper EOD price adjustment
- Applies backward dividend adjustment (current prices stay real)
- Processes latest EOD file only
- Fills gaps with yfinance
- Recalculates all 32 features for entire dataset (consistency)
- Comprehensive validation and reporting

Usage:
    python scripts/update_historical_data_hybrid.py
"""

import os
import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.local_file_loader import LocalFileLoader

# Import feature calculation functions
from scripts.rebuild_dividend_adjusted_data_enhanced import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_dividend_features,
    calculate_split_features
)


def load_dividend_calendar() -> dict:
    """
    Load dividend calendar from JSON file
    
    Returns:
        Dict[ticker, Dict[date_str, amount]]
    """
    calendar_path = os.path.join('data', 'dividend_calendar', 'dividend_calendar.json')
    
    if not os.path.exists(calendar_path):
        print(f"⚠️  Dividend calendar not found at {calendar_path}")
        print("   Run: python scripts/build_dividend_calendar.py")
        return {}
    
    try:
        with open(calendar_path, 'r') as f:
            calendar = json.load(f)
        
        print(f"✅ Loaded dividend calendar: {sum(len(v) for v in calendar.values())} events")
        return calendar
        
    except Exception as e:
        print(f"❌ Error loading dividend calendar: {e}")
        return {}


def calculate_dividend_adjustment_factor(dividend_amount: float, close_price: float) -> float:
    """
    Calculate dividend adjustment factor
    
    Formula: factor = 1 - (dividend / close_price)
    
    Args:
        dividend_amount: Dividend amount (e.g., 0.02)
        close_price: Closing price before ex-div date (e.g., 1.00)
    
    Returns:
        Adjustment factor (e.g., 0.98)
    """
    if close_price <= 0:
        return 1.0
    
    factor = 1.0 - (dividend_amount / close_price)
    return max(factor, 0.0)  # Prevent negative factors


def apply_dividend_adjustments_to_eod_row(
    ticker: str,
    eod_row: pd.Series,
    eod_date: date,
    dividend_calendar: dict,
    historical_df: pd.DataFrame
) -> pd.Series:
    """
    Apply dividend adjustments to a single EOD row
    
    BACKWARD ADJUSTMENT:
    - EOD prices are RAW (unadjusted)
    - We need to calculate what adjustment factor to apply
    - This factor accounts for ALL dividends that occurred AFTER the last historical date
    
    Args:
        ticker: Stock ticker
        eod_row: Single row from EOD file
        eod_date: Date of EOD file
        dividend_calendar: Full dividend calendar
        historical_df: Existing historical data (for reference prices)
    
    Returns:
        Adjusted EOD row
    """
    adjusted_row = eod_row.copy()
    
    # Get dividends for this ticker
    if ticker not in dividend_calendar:
        return adjusted_row  # No dividends, return as-is
    
    ticker_dividends = dividend_calendar[ticker]
    
    if not ticker_dividends:
        return adjusted_row  # No dividends, return as-is
    
    # Find dividends that occurred AFTER the last historical date but ON OR BEFORE EOD date
    # These are the dividends we need to account for
    if historical_df.empty:
        last_hist_date = date(2000, 1, 1)  # Very old date if no history
    else:
        last_hist_date = pd.to_datetime(historical_df['Date'].iloc[-1], dayfirst=True).date()
    
    # Calculate compound adjustment factor
    compound_factor = 1.0
    adjustments_applied = []
    
    for div_date_str, div_amount in ticker_dividends.items():
        div_date = datetime.strptime(div_date_str, '%Y-%m-%d').date()
        
        # Only apply if dividend is AFTER last historical date and ON OR BEFORE EOD date
        if last_hist_date < div_date <= eod_date:
            # Get closing price before dividend (from historical data or use EOD price as fallback)
            if not historical_df.empty:
                # Find closest date before dividend
                hist_dates = pd.to_datetime(historical_df['Date'], dayfirst=True).dt.date
                before_div = historical_df[hist_dates < div_date]
                
                if not before_div.empty:
                    close_before = before_div['Close'].iloc[-1]
                else:
                    close_before = float(eod_row['Last'])  # Fallback
            else:
                close_before = float(eod_row['Last'])  # Fallback
            
            # Calculate adjustment factor
            factor = calculate_dividend_adjustment_factor(div_amount, close_before)
            compound_factor *= factor
            
            adjustments_applied.append({
                'date': div_date_str,
                'amount': div_amount,
                'factor': factor
            })
    
    # Apply compound factor to OHLC prices
    if compound_factor < 1.0:  # Only if adjustments needed
        adjusted_row['Open'] = float(eod_row['Open']) * compound_factor
        adjusted_row['High'] = float(eod_row['High']) * compound_factor
        adjusted_row['Low'] = float(eod_row['Low']) * compound_factor
        adjusted_row['Last'] = float(eod_row['Last']) * compound_factor
    
    return adjusted_row


def process_eod_file(
    eod_filename: str,
    dividend_calendar: dict,
    loader: LocalFileLoader
) -> dict:
    """
    Process EOD file with dividend adjustments
    
    Args:
        eod_filename: EOD filename (e.g., '09_Dec_2025.csv')
        dividend_calendar: Dividend calendar
        loader: LocalFileLoader instance
    
    Returns:
        Dict[ticker, adjusted_row_dict]
    """
    print(f"\n📊 Processing EOD file: {eod_filename}")
    
    # Load EOD file
    eod_df = loader.load_eod_data(eod_filename)
    
    if eod_df is None or eod_df.empty:
        print("❌ Failed to load EOD file")
        return {}
    
    # Parse EOD date
    eod_date_str = eod_filename.replace('.csv', '')
    eod_date = datetime.strptime(eod_date_str, '%d_%b_%Y').date()
    
    print(f"   Date: {eod_date}")
    print(f"   Stocks: {len(eod_df)}")
    
    # Process each stock
    adjusted_data = {}
    adjustments_count = 0
    
    for idx, eod_row in eod_df.iterrows():
        ticker = eod_row['Code']
        
        if pd.isna(ticker):
            continue
        
        # Load historical data for this ticker
        historical_df = loader.load_historical_data(ticker)
        
        if historical_df is None:
            historical_df = pd.DataFrame()
        else:
            # Reset index to get Date as column
            historical_df = historical_df.reset_index()
        
        # Apply dividend adjustments
        adjusted_row = apply_dividend_adjustments_to_eod_row(
            ticker,
            eod_row,
            eod_date,
            dividend_calendar,
            historical_df
        )
        
        # Check if adjustments were applied
        if adjusted_row['Last'] != eod_row['Last']:
            adjustments_count += 1
        
        # Store adjusted data
        adjusted_data[ticker] = {
            'Date': eod_date,
            'Code': ticker,
            'Shortname': eod_row.get('Shortname', ticker.replace('.SG', '')),
            'Open': round(float(adjusted_row['Open']), 3),
            'High': round(float(adjusted_row['High']), 3),
            'Low': round(float(adjusted_row['Low']), 3),
            'Close': round(float(adjusted_row['Last']), 3),
            'Vol': int(float(eod_row['Volume']))
        }
    
    print(f"✅ Processed {len(adjusted_data)} stocks")
    print(f"   Dividend adjustments applied: {adjustments_count} stocks")
    
    return adjusted_data


def fill_gaps_with_yfinance(
    ticker: str,
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Download missing dates from yfinance
    
    Args:
        ticker: Stock ticker
        start_date: Start date for gap
        end_date: End date for gap
    
    Returns:
        DataFrame with gap data (already dividend-adjusted)
    """
    try:
        # Convert .SG to .SI for yfinance
        yf_ticker = ticker.replace('.SG', '.SI')
        
        # Download with auto_adjust=True (already dividend-adjusted)
        stock = yf.Ticker(yf_ticker)
        df = stock.history(
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=True
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Reset index
        df.reset_index(inplace=True)
        
        # Get company name
        try:
            shortname = stock.info.get('shortName', ticker.replace('.SG', ''))
        except:
            shortname = ticker.replace('.SG', '')
        
        # Format data
        formatted_df = pd.DataFrame()
        formatted_df['Date'] = df['Date']
        formatted_df['Code'] = ticker
        formatted_df['Shortname'] = shortname
        formatted_df['Open'] = df['Open'].round(3)
        formatted_df['High'] = df['High'].round(3)
        formatted_df['Low'] = df['Low'].round(3)
        formatted_df['Close'] = df['Close'].round(3)
        formatted_df['Vol'] = (df['Volume'] / 1000).round(0).astype(int)
        
        return formatted_df
        
    except Exception as e:
        print(f"   ✗ yfinance error for {ticker}: {e}")
        return pd.DataFrame()


def recalculate_all_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Recalculate all 32 features for entire dataset
    
    Args:
        df: DataFrame with basic OHLCV data
        ticker: Stock ticker (for dividend/split features)
    
    Returns:
        DataFrame with all 32 features
    """
    # Download dividends and splits for feature calculation
    try:
        yf_ticker = ticker.replace('.SG', '.SI')
        stock = yf.Ticker(yf_ticker)
        dividends = stock.dividends
        splits = stock.splits
    except:
        dividends = pd.Series()
        splits = pd.Series()
    
    # Ensure Date is datetime
    if df['Date'].dtype == 'object':
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Calculate dividend features
    div_features = calculate_dividend_features(dividends, df['Date'], df['Close'])
    df['Dividend'] = div_features['dividend_amount'].round(4)
    df['DaysToNextDiv'] = div_features['days_to_next_div'].astype(int)
    df['DivYield'] = div_features['div_yield'].round(2)
    df['DivGrowthRate'] = div_features['div_growth_rate'].round(2)
    df['ConsecutiveDivs'] = div_features['consecutive_divs'].astype(int)
    df['IsExDivWeek'] = div_features['is_ex_div_week'].astype(int)
    
    # Calculate split features
    split_features = calculate_split_features(splits, df['Date'])
    df['Split'] = split_features['split_ratio'].round(2)
    df['DaysSinceSplit'] = split_features['days_since_split'].astype(int)
    df['SplitInLast90Days'] = split_features['split_in_last_90_days'].astype(int)
    
    # Calculate moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean().round(3)
    df['MA_50'] = df['Close'].rolling(window=50).mean().round(3)
    df['MA_200'] = df['Close'].rolling(window=200).mean().round(3)
    
    # Calculate RSI
    df['RSI_14'] = calculate_rsi(df['Close'], 14).round(2)
    
    # Calculate MACD
    df['MACD'] = calculate_macd(df['Close']).round(4)
    
    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper.round(3)
    df['BB_Middle'] = bb_middle.round(3)
    df['BB_Lower'] = bb_lower.round(3)
    
    # Calculate ATR
    df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'], 14).round(4)
    
    # Calculate Rate of Change
    df['ROC_5'] = ((df['Close'] / df['Close'].shift(5) - 1) * 100).round(2)
    df['ROC_10'] = ((df['Close'] / df['Close'].shift(10) - 1) * 100).round(2)
    df['ROC_20'] = ((df['Close'] / df['Close'].shift(20) - 1) * 100).round(2)
    
    # Calculate distance from moving averages
    df['DistFromMA20'] = ((df['Close'] / df['MA_20'] - 1) * 100).round(2)
    df['DistFromMA50'] = ((df['Close'] / df['MA_50'] - 1) * 100).round(2)
    
    # Calculate volume ratio
    vol_ma = (df['Vol'] * 1000).rolling(window=20).mean()  # Convert back to full volume
    df['VolRatio'] = ((df['Vol'] * 1000) / vol_ma).round(2)
    
    # Format dates to D/M/YYYY
    df['Date'] = df['Date'].dt.strftime('%-d/%-m/%Y' if os.name != 'nt' else '%#d/%#m/%Y')
    
    # Fill NaN values
    df.fillna(0, inplace=True)
    
    return df


def update_single_ticker(
    ticker: str,
    eod_data: dict,
    dividend_calendar: dict,
    loader: LocalFileLoader,
    current_working_day: date
) -> dict:
    """
    Update a single ticker with hybrid approach
    
    Args:
        ticker: Stock ticker
        eod_data: Adjusted EOD data (if available)
        dividend_calendar: Dividend calendar
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
            # FIXED: Ensure we get a plain date object, not a Timestamp
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
            eod_date = eod_row['Date']  # This is already a plain date object
            
            # Check if EOD date is newer than last historical date
            if last_hist_date is None or eod_date > last_hist_date:
                new_rows.append(eod_row)
                result['eod_days'] = 1
                last_hist_date = eod_date  # Update to plain date object
        
        # Check for gaps and fill with yfinance
        # FIXED: Ensure last_hist_date is a plain date object before timedelta arithmetic
        if last_hist_date is not None:
            # Ensure it's a plain date object (not Timestamp)
            if hasattr(last_hist_date, 'date'):
                last_hist_date = last_hist_date.date()
            
            if last_hist_date < current_working_day:
                gap_start = last_hist_date + timedelta(days=1)
                gap_end = current_working_day
                
                # Download gap data from yfinance
                yf_df = fill_gaps_with_yfinance(ticker, gap_start, gap_end)
                
                if not yf_df.empty:
                    # Convert to list of dicts
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
        
        # Create DataFrame from new rows
        new_df = pd.DataFrame(new_rows)
        
        # Ensure Date is datetime
        if 'Date' in new_df.columns:
            if new_df['Date'].dtype == 'object':
                new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # Combine
        if not historical_df.empty:
            # Keep only basic columns from historical for merging
            # Note: LocalFileLoader standardizes 'Vol' to 'Volume', so we need to handle both
            if 'Volume' in historical_df.columns:
                hist_basic = historical_df[['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                hist_basic = hist_basic.rename(columns={'Volume': 'Vol'})
            else:
                hist_basic = historical_df[['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol']].copy()
            combined_df = pd.concat([hist_basic, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Sort by date
        combined_df = combined_df.sort_values('Date')
        
        # Remove duplicates (keep last)
        combined_df['Date_str'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        combined_df = combined_df.drop_duplicates(subset=['Date_str'], keep='last')
        combined_df = combined_df.drop('Date_str', axis=1)
        
        # Recalculate all 32 features
        enhanced_df = recalculate_all_features(combined_df, ticker)
        
        # Ensure column order
        column_order = [
            'Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol',
            'Dividend', 'DaysToNextDiv', 'DivYield', 'DivGrowthRate', 'ConsecutiveDivs', 'IsExDivWeek',
            'Split', 'DaysSinceSplit', 'SplitInLast90Days',
            'MA_20', 'MA_50', 'MA_200',
            'RSI_14', 'MACD',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR_14',
            'ROC_5', 'ROC_10', 'ROC_20', 'DistFromMA20', 'DistFromMA50', 'VolRatio'
        ]
        enhanced_df = enhanced_df[column_order]
        
        # Save to CSV
        ticker_clean = ticker.replace('.SG', '')
        filename = f"{ticker_clean}.csv"
        filepath = os.path.join(loader.historical_path, filename)
        
        enhanced_df.to_csv(filepath, index=False, encoding='utf-8')
        
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
    print("🔄 HYBRID HISTORICAL DATA UPDATER")
    print("=" * 70)
    print("\n📋 Strategy:")
    print("   1. Load dividend calendar")
    print("   2. Process latest EOD file (with dividend adjustment)")
    print("   3. Fill gaps with yfinance (already adjusted)")
    print("   4. Recalculate all 32 features")
    print("   5. Validate and save")
    
    # Initialize
    start_time = datetime.now()
    loader = LocalFileLoader()
    
    # Step 1: Load dividend calendar
    print("\n" + "=" * 70)
    print("📅 STEP 1: Load Dividend Calendar")
    print("=" * 70)
    
    dividend_calendar = load_dividend_calendar()
    
    if not dividend_calendar:
        print("\n⚠️  WARNING: No dividend calendar found!")
        print("   EOD prices will NOT be adjusted for dividends.")
        print("   Run: python scripts/build_dividend_calendar.py")
        response = input("\n   Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("\n❌ Update cancelled")
            return
    
    # Step 2: Check for new EOD file
    print("\n" + "=" * 70)
    print("📊 STEP 2: Check for New EOD Data")
    print("=" * 70)
    
    latest_eod = loader.get_latest_eod_file()
    
    if not latest_eod:
        print("❌ No EOD files found")
        print("   Place EOD CSV files in: data/EOD_Data/")
        return
    
    print(f"   Latest EOD file: {latest_eod}")
    
    # Parse EOD date
    eod_date_str = latest_eod.replace('.csv', '')
    eod_date = datetime.strptime(eod_date_str, '%d_%b_%Y').date()
    
    # Get watchlist
    watchlist = loader.get_watchlist_from_eod()
    
    if not watchlist:
        print("❌ Could not extract watchlist from EOD file")
        return
    
    print(f"   Watchlist: {len(watchlist)} stocks")
    
    # Check if EOD file is new
    sample_ticker = watchlist[0]
    last_hist_date = loader.get_last_date_in_historical(sample_ticker)
    
    # FIXED: last_hist_date is already a plain date object from get_last_date_in_historical()
    if last_hist_date:
        print(f"   Last historical date: {last_hist_date}")
        print(f"   EOD date: {eod_date}")
        
        if eod_date <= last_hist_date:
            print(f"\n   ℹ️  EOD file already processed (date: {eod_date})")
            eod_data = {}  # No new EOD data
        else:
            print(f"\n   ✅ New EOD data available!")
            # Process EOD file
            eod_data = process_eod_file(latest_eod, dividend_calendar, loader)
    else:
        print(f"   No historical data found - will process EOD file")
        eod_data = process_eod_file(latest_eod, dividend_calendar, loader)
    
    # Step 3: Determine current working day and check for gaps
    print("\n" + "=" * 70)
    print("📊 STEP 3: Check for Gaps")
    print("=" * 70)
    
    current_working_day = loader.get_current_working_day()
    print(f"   Current working day: {current_working_day}")
    
    if last_hist_date:
        if last_hist_date < current_working_day:
            gap_days = (current_working_day - last_hist_date).days
            print(f"   ⚠️  Gap detected: {gap_days} day(s)")
            print(f"   Will fill with yfinance...")
        else:
            print(f"   ✅ No gaps detected")
    
    # Step 4: Update all tickers
    print("\n" + "=" * 70)
    print("📊 STEP 4: Update Historical Data")
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
                dividend_calendar,
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
    with open(report_path, 'w') as f:
        f.write("HYBRID HISTORICAL DATA UPDATE REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"EOD File: {latest_eod}\n")
        f.write(f"Current Working Day: {current_working_day}\n\n")
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
    print("\n✅ Your historical data is now current with 32 features!")
    print("\n📌 Next Steps:")
    print("   • Run this script daily to keep data current")
    print("   • Update dividend calendar monthly")
    print("   • Verify data quality periodically")


if __name__ == "__main__":
    main()
