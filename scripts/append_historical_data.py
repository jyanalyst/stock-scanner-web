# File: /workspaces/stock-scanner-web/scripts/append_historical_data.py
"""
Historical Data Append Script - BATCH PROCESSING with SMART DEDUPLICATION
Appends data from Historical_Data_Add/{TICKER}_YYYY.csv to Historical_Data/{TICKER}.csv

Features:
- SMART DEDUPLICATION: Automatically removes existing dates before appending
- Allows reprocessing files to fix errors without creating duplicates
- Processes ALL files in Historical_Data_Add automatically
- Creates timestamped backups ONCE per ticker (e.g., C38U.csv.backup_20251028_143000)
- Automatically moves processed files to Historical_Data_Add/processed/
- Suppresses harmless validation warnings (Date_diff, date gaps)
- Generic script for any ticker
- Extracts ticker from filename pattern: {TICKER}_YYYY.csv
- Converts date format: YYYY-MM-DD → D/M/YYYY (no leading zeros)
- Converts volume format: float → integer
- Adds Code and Shortname columns from main file
- Removes extra calculated columns
- Skips duplicate end-of-file marker (last line)
- Sorts data chronologically
- Validates data integrity and reports critical irregularities only
- Comprehensive final summary

Usage:
    python scripts/append_historical_data.py
    
To reprocess a file (fix errors):
    1. Move corrected file from 'processed/' back to 'Historical_Data_Add/'
    2. Run the script
    3. Old data for those dates will be automatically removed and replaced
"""

import pandas as pd
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import sys

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
HISTORICAL_DATA_DIR = PROJECT_ROOT / "data" / "Historical_Data"
HISTORICAL_DATA_ADD_DIR = PROJECT_ROOT / "data" / "Historical_Data_Add"
PROCESSED_DIR = HISTORICAL_DATA_ADD_DIR / "processed"

# Configuration
DATE_GAP_THRESHOLD = 5  # Days - report gaps longer than this (but not as errors)
PRICE_JUMP_THRESHOLD = 0.20  # 20% price change threshold


def ensure_processed_folder():
    """Create processed folder if it doesn't exist"""
    if not PROCESSED_DIR.exists():
        PROCESSED_DIR.mkdir(parents=True)
        print(f"✅ Created processed folder: {PROCESSED_DIR}")


def extract_ticker_from_filename(filename: str) -> tuple:
    """
    Extract ticker and year from filename
    
    Args:
        filename: e.g., "C38U_2024.csv"
        
    Returns:
        (ticker, year) e.g., ("C38U", "2024") or (None, None) if invalid
    """
    try:
        # Remove .csv extension
        basename = filename.replace('.csv', '')
        
        # Split by underscore
        parts = basename.split('_')
        
        if len(parts) != 2:
            return None, None
        
        ticker = parts[0]
        year = parts[1]
        
        # Validate year is 4 digits
        if not year.isdigit() or len(year) != 4:
            return None, None
        
        return ticker, year
    
    except Exception as e:
        print(f"❌ Error extracting ticker from filename '{filename}': {e}")
        return None, None


def convert_date_format(date_str: str) -> str:
    """
    Convert date from YYYY-MM-DD to D/M/YYYY (no leading zeros)
    
    Args:
        date_str: Date in YYYY-MM-DD format (e.g., "2024-01-02")
        
    Returns:
        Date in D/M/YYYY format (e.g., "2/1/2024")
    """
    try:
        # Parse date
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Format without leading zeros
        # Use %-d and %-m on Unix, %#d and %#m on Windows
        if os.name == 'nt':  # Windows
            formatted = date_obj.strftime('%#d/%#m/%Y')
        else:  # Unix/Linux/Mac
            formatted = date_obj.strftime('%-d/%-m/%Y')
        
        return formatted
    
    except Exception as e:
        print(f"❌ Error converting date '{date_str}': {e}")
        return None


def read_main_file(ticker: str) -> tuple:
    """
    Read main historical data file
    
    Args:
        ticker: Stock ticker (e.g., "C38U")
        
    Returns:
        (df, code, shortname, filepath) or (None, None, None, None) if error
    """
    try:
        filepath = HISTORICAL_DATA_DIR / f"{ticker}.csv"
        
        if not filepath.exists():
            print(f"   ❌ Main file not found: {filepath}")
            return None, None, None, None
        
        # Read CSV
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # Extract Code and Shortname from first row
        if len(df) == 0:
            print(f"   ❌ Main file is empty: {filepath}")
            return None, None, None, None
        
        code = df['Code'].iloc[0] if 'Code' in df.columns else f"{ticker}.SG"
        shortname = df['Shortname'].iloc[0] if 'Shortname' in df.columns else ticker
        
        print(f"   ✅ Read main file: {len(df)} rows")
        
        return df, code, shortname, filepath
    
    except Exception as e:
        print(f"   ❌ Error reading main file for {ticker}: {e}")
        return None, None, None, None


def read_addition_file(filepath: Path) -> pd.DataFrame:
    """
    Read addition file and process it
    
    Args:
        filepath: Path to addition file
        
    Returns:
        DataFrame or None if error
    """
    try:
        # Read CSV
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # Skip last row (duplicate end-of-file marker)
        if len(df) > 1:
            df = df.iloc[:-1]
        
        return df
    
    except Exception as e:
        print(f"   ❌ Error reading addition file: {e}")
        return None


def process_addition_data(df_add: pd.DataFrame, code: str, shortname: str, year: str) -> pd.DataFrame:
    """
    Process addition data to match main file format
    
    Args:
        df_add: Addition DataFrame
        code: Stock code (e.g., "C38U.SG")
        shortname: Stock shortname
        year: Year being processed
        
    Returns:
        Processed DataFrame or None if error
    """
    try:
        df_processed = df_add.copy()
        
        # Convert date format
        df_processed['Date'] = df_processed['Date'].apply(convert_date_format)
        
        # Check for conversion errors
        if df_processed['Date'].isnull().any():
            failed_count = df_processed['Date'].isnull().sum()
            print(f"   ⚠️  WARNING: {failed_count} dates failed to convert in {year}")
        
        # Convert volume to integer
        df_processed['Vol'] = df_processed['Vol'].astype(float).astype(int)
        
        # Add Code and Shortname columns
        df_processed['Code'] = code
        df_processed['Shortname'] = shortname
        
        # Keep only required columns in correct order
        required_columns = ['Date', 'Code', 'Shortname', 'Open', 'High', 'Low', 'Close', 'Vol']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_columns if col not in df_processed.columns]
        if missing_cols:
            print(f"   ❌ Missing required columns: {missing_cols}")
            return None
        
        # Select and reorder columns
        df_processed = df_processed[required_columns]
        
        return df_processed
    
    except Exception as e:
        print(f"   ❌ Error processing addition data: {e}")
        return None


def create_timestamped_backup(filepath: Path) -> bool:
    """
    Create timestamped backup of main file
    
    Args:
        filepath: Path to main file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = Path(str(filepath) + f'.backup_{timestamp}')
        shutil.copy2(filepath, backup_path)
        print(f"   ✅ Created backup: {backup_path.name}")
        return True
    
    except Exception as e:
        print(f"   ❌ Error creating backup: {e}")
        return False


def deduplicate_and_merge(df_main: pd.DataFrame, df_add_list: list) -> tuple:
    """
    Smart deduplication: Remove dates from main file that exist in addition files,
    then append the new data
    
    Args:
        df_main: Main DataFrame
        df_add_list: List of addition DataFrames
        
    Returns:
        (combined_df, stats_dict) - Combined DataFrame and deduplication statistics
    """
    try:
        stats = {
            'original_rows': len(df_main),
            'rows_to_add': sum(len(df) for df in df_add_list),
            'duplicates_removed': 0,
            'final_rows': 0
        }
        
        # Collect all dates from addition files
        all_addition_dates = set()
        for df_add in df_add_list:
            all_addition_dates.update(df_add['Date'].tolist())
        
        # Check for overlapping dates
        overlapping_dates = df_main[df_main['Date'].isin(all_addition_dates)]
        
        if len(overlapping_dates) > 0:
            stats['duplicates_removed'] = len(overlapping_dates)
            print(f"\n   🔄 SMART DEDUPLICATION:")
            print(f"      Found {len(overlapping_dates)} existing dates that will be replaced")
            print(f"      Removing old data for these dates...")
            
            # Remove overlapping dates from main file
            df_main_cleaned = df_main[~df_main['Date'].isin(all_addition_dates)].copy()
            print(f"      ✅ Removed {len(overlapping_dates)} rows from main file")
        else:
            print(f"\n   ✅ No overlapping dates found - pure append")
            df_main_cleaned = df_main.copy()
        
        # Combine all dataframes
        all_dfs = [df_main_cleaned] + df_add_list
        df_combined = pd.concat(all_dfs, ignore_index=True)
        
        # Convert Date to datetime for sorting
        df_combined['Date_dt'] = pd.to_datetime(df_combined['Date'], dayfirst=True, format='mixed')
        
        # Sort by date
        df_combined = df_combined.sort_values('Date_dt')
        
        # Drop temporary datetime column
        df_combined = df_combined.drop('Date_dt', axis=1)
        
        # Reset index
        df_combined = df_combined.reset_index(drop=True)
        
        stats['final_rows'] = len(df_combined)
        
        return df_combined, stats
    
    except Exception as e:
        print(f"   ❌ Error in deduplication and merge: {e}")
        return None, None


def validate_data(df: pd.DataFrame, ticker: str) -> dict:
    """
    Validate data integrity and identify CRITICAL irregularities only
    Suppresses harmless warnings like Date_diff missing and normal date gaps
    
    Args:
        df: DataFrame to validate
        ticker: Stock ticker
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'total_rows': len(df),
        'issues': []
    }
    
    # Convert dates for validation
    df_val = df.copy()
    df_val['Date_dt'] = pd.to_datetime(df_val['Date'], dayfirst=True, format='mixed')
    
    # 1. Check for duplicate dates (CRITICAL)
    duplicates = df_val[df_val.duplicated(subset=['Date'], keep=False)]
    if len(duplicates) > 0:
        results['valid'] = False
        dup_dates = duplicates['Date'].unique().tolist()
        results['issues'].append({
            'type': 'DUPLICATE_DATES',
            'severity': 'HIGH',
            'count': len(duplicates),
            'dates': dup_dates[:10]
        })
    
    # 2. Check chronological order (CRITICAL)
    is_sorted = df_val['Date_dt'].is_monotonic_increasing
    if not is_sorted:
        results['valid'] = False
        results['issues'].append({
            'type': 'NOT_CHRONOLOGICAL',
            'severity': 'HIGH',
            'message': 'Dates are not in chronological order'
        })
    
    # 3. Check for date gaps > 5 days (INFO ONLY - not counted as error)
    df_val['Date_diff'] = df_val['Date_dt'].diff()
    large_gaps = df_val[df_val['Date_diff'] > timedelta(days=DATE_GAP_THRESHOLD)]
    
    if len(large_gaps) > 0:
        gap_info = []
        for idx, row in large_gaps.iterrows():
            if idx > 0:
                prev_date = df_val.loc[idx-1, 'Date']
                curr_date = row['Date']
                gap_days = row['Date_diff'].days
                gap_info.append({
                    'from': prev_date,
                    'to': curr_date,
                    'gap_days': gap_days
                })
        
        # Store as INFO, not as an issue
        results['date_gaps'] = {
            'count': len(large_gaps),
            'gaps': gap_info[:10]
        }
    
    # 4. Check for missing values (CRITICAL - but exclude Date_diff which is expected)
    missing_counts = df_val.drop('Date_diff', axis=1).isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        results['valid'] = False
        results['issues'].append({
            'type': 'MISSING_VALUES',
            'severity': 'HIGH',
            'columns': missing_cols.to_dict()
        })
    
    # 5. Check for price jumps > 20% (WARNING)
    df_val['Price_change'] = df_val['Close'].pct_change().abs()
    large_jumps = df_val[df_val['Price_change'] > PRICE_JUMP_THRESHOLD]
    
    if len(large_jumps) > 0:
        jump_info = []
        for idx, row in large_jumps.iterrows():
            if idx > 0:
                prev_price = df_val.loc[idx-1, 'Close']
                curr_price = row['Close']
                change_pct = row['Price_change'] * 100
                jump_info.append({
                    'date': row['Date'],
                    'prev_price': prev_price,
                    'curr_price': curr_price,
                    'change_pct': round(change_pct, 2)
                })
        
        results['issues'].append({
            'type': 'LARGE_PRICE_JUMPS',
            'severity': 'MEDIUM',
            'count': len(large_jumps),
            'jumps': jump_info[:10]
        })
    
    # 6. Check for zero volumes (INFO)
    zero_vols = df_val[df_val['Vol'] == 0]
    
    if len(zero_vols) > 0:
        results['issues'].append({
            'type': 'ZERO_VOLUMES',
            'severity': 'LOW',
            'count': len(zero_vols),
            'dates': zero_vols['Date'].tolist()[:10]
        })
    
    # Store date range info
    results['first_date'] = df_val['Date'].iloc[0]
    results['last_date'] = df_val['Date'].iloc[-1]
    
    return results


def save_combined_data(df: pd.DataFrame, filepath: Path) -> bool:
    """
    Save combined data to main file
    
    Args:
        df: DataFrame to save
        filepath: Path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        df.to_csv(filepath, index=False, encoding='utf-8')
        return True
    
    except Exception as e:
        print(f"   ❌ Error saving data: {e}")
        return False


def move_to_processed(filepath: Path) -> bool:
    """
    Move processed file to processed subfolder
    
    Args:
        filepath: Path to file to move
        
    Returns:
        True if successful, False otherwise
    """
    try:
        destination = PROCESSED_DIR / filepath.name
        
        # If file already exists in processed folder, remove it first
        if destination.exists():
            destination.unlink()
            print(f"      ℹ️  Replaced existing file in processed folder")
        
        shutil.move(str(filepath), str(destination))
        return True
    
    except Exception as e:
        print(f"   ⚠️  Could not move file to processed folder: {e}")
        return False


def process_ticker_files(ticker: str, year_files: list) -> dict:
    """
    Process all year files for a single ticker with smart deduplication
    
    Args:
        ticker: Stock ticker (e.g., "C38U")
        year_files: List of (filepath, year) tuples
        
    Returns:
        Dictionary with processing results
    """
    print(f"\n{'='*60}")
    print(f"📂 PROCESSING TICKER: {ticker}")
    print(f"{'='*60}")
    print(f"   Files to process: {len(year_files)}")
    for filepath, year in year_files:
        print(f"      - {filepath.name} ({year})")
    
    result = {
        'ticker': ticker,
        'success': False,
        'files_processed': 0,
        'total_files': len(year_files),
        'rows_added': 0,
        'duplicates_removed': 0,
        'validation_results': None,
        'error': None,
        'files_moved': []
    }
    
    # Read main file
    print(f"\n📖 Reading main file...")
    df_main, code, shortname, main_filepath = read_main_file(ticker)
    if df_main is None:
        result['error'] = "Failed to read main file"
        return result
    
    original_rows = len(df_main)
    
    # Create timestamped backup
    print(f"\n💾 Creating timestamped backup...")
    if not create_timestamped_backup(main_filepath):
        result['error'] = "Failed to create backup"
        return result
    
    # Process each year file
    print(f"\n🔄 Processing addition files...")
    df_add_list = []
    files_to_move = []
    
    for filepath, year in sorted(year_files, key=lambda x: x[1]):  # Sort by year
        print(f"\n   📅 Processing {year}...")
        
        # Read addition file
        df_add = read_addition_file(filepath)
        if df_add is None:
            print(f"      ⚠️  Skipped {year} - read error")
            continue
        
        print(f"      ✅ Read {len(df_add)} rows (skipped last line)")
        
        # Process addition data
        df_add_processed = process_addition_data(df_add, code, shortname, year)
        if df_add_processed is None:
            print(f"      ⚠️  Skipped {year} - processing error")
            continue
        
        print(f"      ✅ Processed {len(df_add_processed)} rows")
        
        df_add_list.append(df_add_processed)
        files_to_move.append(filepath)
        result['files_processed'] += 1
        result['rows_added'] += len(df_add_processed)
    
    if not df_add_list:
        result['error'] = "No files successfully processed"
        return result
    
    # Smart deduplication and merge
    print(f"\n🔗 Smart deduplication and merge...")
    df_combined, dedup_stats = deduplicate_and_merge(df_main, df_add_list)
    if df_combined is None:
        result['error'] = "Failed to combine data"
        return result
    
    result['duplicates_removed'] = dedup_stats['duplicates_removed']
    
    print(f"\n   📊 Merge Statistics:")
    print(f"      Original rows: {dedup_stats['original_rows']}")
    print(f"      Duplicates removed: {dedup_stats['duplicates_removed']}")
    print(f"      New rows added: {dedup_stats['rows_to_add']}")
    print(f"      Final total: {dedup_stats['final_rows']}")
    print(f"      Net change: {dedup_stats['final_rows'] - dedup_stats['original_rows']:+d}")
    
    # Validate
    print(f"\n🔍 Validating combined data...")
    validation_results = validate_data(df_combined, ticker)
    result['validation_results'] = validation_results
    
    # Display validation summary (only critical issues)
    critical_issues = [i for i in validation_results['issues'] if i['severity'] == 'HIGH']
    
    if validation_results['valid'] and not critical_issues:
        print(f"   ✅ Validation PASSED - No critical issues found")
    else:
        print(f"   ⚠️  Validation found {len(critical_issues)} CRITICAL issue(s)")
        for issue in critical_issues:
            print(f"      🚨 {issue['type']} ({issue['severity']})")
    
    # Show date gaps as info (not error)
    if 'date_gaps' in validation_results:
        gap_count = validation_results['date_gaps']['count']
        print(f"   ℹ️  Note: {gap_count} date gap(s) > {DATE_GAP_THRESHOLD} days (holidays/weekends)")
    
    # Save combined data
    print(f"\n💾 Saving combined data...")
    if save_combined_data(df_combined, main_filepath):
        print(f"   ✅ Successfully saved to {main_filepath.name}")
        result['success'] = True
        
        # Move processed files to processed folder
        print(f"\n📦 Moving processed files to 'processed' folder...")
        for filepath in files_to_move:
            if move_to_processed(filepath):
                print(f"   ✅ Moved: {filepath.name}")
                result['files_moved'].append(filepath.name)
            else:
                print(f"   ⚠️  Could not move: {filepath.name}")
    else:
        result['error'] = "Failed to save combined data"
    
    return result


def print_detailed_validation_report(ticker: str, validation_results: dict):
    """Print detailed validation report for a ticker (CRITICAL issues only)"""
    print(f"\n   📊 Detailed Validation Report:")
    print(f"      Date range: {validation_results['first_date']} to {validation_results['last_date']}")
    print(f"      Total rows: {validation_results['total_rows']}")
    
    # Only show critical issues (HIGH severity)
    critical_issues = [i for i in validation_results['issues'] if i['severity'] == 'HIGH']
    
    if not critical_issues:
        print(f"      ✅ No critical issues found")
        return
    
    for issue in critical_issues:
        print(f"\n      🚨 {issue['type']} (CRITICAL):")
        
        if issue['type'] == 'DUPLICATE_DATES':
            print(f"         Found {issue['count']} duplicate entries")
            print(f"         Duplicate dates: {', '.join(issue['dates'][:5])}")
            if len(issue['dates']) > 5:
                print(f"         ... and {len(issue['dates']) - 5} more")
        
        elif issue['type'] == 'NOT_CHRONOLOGICAL':
            print(f"         {issue['message']}")
        
        elif issue['type'] == 'MISSING_VALUES':
            print(f"         Missing values found:")
            for col, count in issue['columns'].items():
                print(f"         - {col}: {count} missing")


def main():
    """Main function - Batch processing with smart deduplication"""
    print("="*60)
    print("📊 HISTORICAL DATA APPEND SCRIPT - SMART DEDUPLICATION")
    print("="*60)
    print(f"\nScanning: {HISTORICAL_DATA_ADD_DIR}")
    
    # Check if directories exist
    if not HISTORICAL_DATA_DIR.exists():
        print(f"❌ Historical_Data directory not found: {HISTORICAL_DATA_DIR}")
        sys.exit(1)
    
    if not HISTORICAL_DATA_ADD_DIR.exists():
        print(f"❌ Historical_Data_Add directory not found: {HISTORICAL_DATA_ADD_DIR}")
        sys.exit(1)
    
    # Ensure processed folder exists
    ensure_processed_folder()
    
    # Find all addition files
    addition_files = list(HISTORICAL_DATA_ADD_DIR.glob("*_*.csv"))
    
    if not addition_files:
        print(f"\n⚠️  No addition files found in {HISTORICAL_DATA_ADD_DIR}")
        print(f"   All files may have been processed already.")
        print(f"   Check: {PROCESSED_DIR}")
        print(f"\n💡 To reprocess a file (fix errors):")
        print(f"   1. Move corrected file from 'processed/' back to 'Historical_Data_Add/'")
        print(f"   2. Run this script again")
        print(f"   3. Old data will be automatically replaced")
        sys.exit(0)
    
    # Group files by ticker
    ticker_files = defaultdict(list)
    invalid_files = []
    
    for filepath in addition_files:
        ticker, year = extract_ticker_from_filename(filepath.name)
        if ticker and year:
            ticker_files[ticker].append((filepath, year))
        else:
            invalid_files.append(filepath.name)
    
    # Display scan results
    print(f"\n✅ Found {len(addition_files)} addition file(s)")
    print(f"   Valid tickers: {len(ticker_files)}")
    
    for ticker in sorted(ticker_files.keys()):
        years = sorted([year for _, year in ticker_files[ticker]])
        print(f"      - {ticker}: {len(ticker_files[ticker])} files ({', '.join(years)})")
    
    if invalid_files:
        print(f"\n   ⚠️  Invalid filenames (will be skipped): {len(invalid_files)}")
        for fname in invalid_files:
            print(f"      - {fname}")
    
    # Confirm processing
    print(f"\n{'='*60}")
    print(f"⚠️  READY TO PROCESS {len(ticker_files)} TICKER(S)")
    print(f"{'='*60}")
    print(f"\nThis will:")
    print(f"   1. Create timestamped backups for each ticker")
    print(f"   2. Remove any existing dates that overlap with addition files")
    print(f"   3. Append new/corrected data")
    print(f"   4. Validate data integrity")
    print(f"   5. Save combined data")
    print(f"   6. Move processed files to 'processed' subfolder")
    print(f"\n💡 SMART DEDUPLICATION: If reprocessing, old data will be replaced automatically")
    print(f"\nProceed with batch processing? (y/n): ", end='')
    
    response = input().strip().lower()
    if response != 'y':
        print("\n❌ Processing cancelled by user")
        sys.exit(0)
    
    # Process each ticker
    print(f"\n{'='*60}")
    print(f"🚀 STARTING BATCH PROCESSING")
    print(f"{'='*60}")
    
    all_results = []
    
    for ticker in sorted(ticker_files.keys()):
        result = process_ticker_files(ticker, ticker_files[ticker])
        all_results.append(result)
    
    # Final summary report
    print(f"\n{'='*60}")
    print(f"📊 FINAL SUMMARY REPORT")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\n✅ Successfully processed: {len(successful)}/{len(all_results)} tickers")
    print(f"❌ Failed: {len(failed)}/{len(all_results)} tickers")
    
    if successful:
        print(f"\n📈 SUCCESSFUL TICKERS:")
        for result in successful:
            print(f"\n   🎯 {result['ticker']}:")
            print(f"      Files processed: {result['files_processed']}/{result['total_files']}")
            print(f"      Duplicates removed: {result['duplicates_removed']}")
            print(f"      Rows added: {result['rows_added']}")
            print(f"      Total rows: {result['validation_results']['total_rows']}")
            print(f"      Files moved: {len(result['files_moved'])}")
            
            # Show validation status (only critical issues)
            val_results = result['validation_results']
            critical_issues = [i for i in val_results['issues'] if i['severity'] == 'HIGH']
            
            if val_results['valid'] and not critical_issues:
                print(f"      Validation: ✅ PASSED")
            else:
                print(f"      Validation: ⚠️  {len(critical_issues)} CRITICAL issue(s) found")
                
                # Print detailed validation report (only critical)
                print_detailed_validation_report(result['ticker'], val_results)
    
    if failed:
        print(f"\n❌ FAILED TICKERS:")
        for result in failed:
            print(f"   - {result['ticker']}: {result['error']}")
    
    print(f"\n{'='*60}")
    print(f"✅ BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    
    # Summary stats
    total_files = sum(r['files_processed'] for r in all_results)
    total_duplicates = sum(r['duplicates_removed'] for r in all_results)
    total_rows = sum(r['rows_added'] for r in all_results)
    total_moved = sum(len(r['files_moved']) for r in all_results)
    
    print(f"\n📊 Statistics:")
    print(f"   Total files processed: {total_files}")
    print(f"   Duplicates removed: {total_duplicates}")
    print(f"   Total rows added: {total_rows}")
    print(f"   Files moved to 'processed': {total_moved}")
    print(f"   Success rate: {len(successful)}/{len(all_results)} ({len(successful)/len(all_results)*100:.1f}%)")
    
    print(f"\n💡 Processed files moved to:")
    print(f"   {PROCESSED_DIR}")
    print(f"\n   You can safely run this script again for new files!")
    print(f"   To fix errors: move corrected file back and rerun script.")


if __name__ == "__main__":
    main()
