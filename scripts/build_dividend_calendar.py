"""
Dividend Calendar Builder
=========================

Downloads complete dividend history from yfinance for all stocks in watchlist.
This calendar is used to properly adjust prices when appending EOD data.

Run this:
- Once initially
- Monthly to update
- When new dividends are announced

Usage:
    python scripts/build_dividend_calendar.py
"""

import os
import sys
import json
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.local_file_loader import LocalFileLoader


def download_dividend_history(ticker: str) -> dict:
    """
    Download complete dividend history for a ticker
    
    Returns:
        Dictionary with dividend dates and amounts
    """
    try:
        # Convert .SG to .SI for yfinance
        yf_ticker = ticker.replace('.SG', '.SI')
        
        stock = yf.Ticker(yf_ticker)
        dividends = stock.dividends
        
        if dividends.empty:
            return {}
        
        # Convert to dictionary with string dates
        div_dict = {}
        for div_date, div_amount in dividends.items():
            date_str = div_date.strftime('%Y-%m-%d')
            div_dict[date_str] = float(div_amount)
        
        return div_dict
        
    except Exception as e:
        print(f"   ✗ Error downloading {ticker}: {e}")
        return {}


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("📅 DIVIDEND CALENDAR BUILDER")
    print("=" * 70)
    print("\nThis tool downloads complete dividend history from yfinance")
    print("for all stocks in your watchlist.\n")
    
    # Initialize
    start_time = datetime.now()
    loader = LocalFileLoader()
    
    # Get watchlist from EOD file
    print("📋 Loading watchlist from latest EOD file...")
    watchlist = loader.get_watchlist_from_eod()
    
    if not watchlist:
        print("❌ Could not load watchlist from EOD file. Exiting.")
        return
    
    print(f"✅ Found {len(watchlist)} stocks in watchlist\n")
    
    # Download dividend history
    print(f"📥 Downloading dividend history for {len(watchlist)} stocks...\n")
    
    dividend_calendar = {}
    stats = {
        'total': len(watchlist),
        'success': 0,
        'no_dividends': 0,
        'failed': 0,
        'total_dividends': 0
    }
    
    with tqdm(total=len(watchlist), desc="Downloading", unit="stock") as pbar:
        for ticker in watchlist:
            div_history = download_dividend_history(ticker)
            
            if div_history:
                dividend_calendar[ticker] = div_history
                stats['success'] += 1
                stats['total_dividends'] += len(div_history)
                pbar.set_postfix_str(f"✓ {ticker}: {len(div_history)} dividends")
            elif div_history == {}:
                # Successfully downloaded but no dividends
                dividend_calendar[ticker] = {}
                stats['no_dividends'] += 1
                pbar.set_postfix_str(f"○ {ticker}: No dividends")
            else:
                stats['failed'] += 1
                pbar.set_postfix_str(f"✗ {ticker}: Failed")
            
            pbar.update(1)
    
    # Save to file
    output_dir = os.path.join('data', 'dividend_calendar')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'dividend_calendar.json')
    
    with open(output_file, 'w') as f:
        json.dump(dividend_calendar, f, indent=2)
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\n✅ Success: {stats['success']}/{stats['total']} stocks with dividends")
    print(f"○  No Dividends: {stats['no_dividends']} stocks")
    print(f"✗  Failed: {stats['failed']} stocks")
    print(f"📈 Total Dividends: {stats['total_dividends']:,} dividend events")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"\n💾 Saved to: {output_file}")
    
    # Show sample
    if dividend_calendar:
        print("\n📋 Sample Dividend History:")
        sample_ticker = list(dividend_calendar.keys())[0]
        sample_divs = dividend_calendar[sample_ticker]
        
        if sample_divs:
            print(f"\n   {sample_ticker}:")
            # Show last 5 dividends
            sorted_dates = sorted(sample_divs.keys(), reverse=True)[:5]
            for div_date in sorted_dates:
                print(f"   • {div_date}: ${sample_divs[div_date]:.4f}")
    
    # Save metadata
    metadata = {
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_stocks': stats['total'],
        'stocks_with_dividends': stats['success'],
        'stocks_without_dividends': stats['no_dividends'],
        'total_dividend_events': stats['total_dividends'],
        'watchlist': watchlist
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Metadata saved: {metadata_file}")
    
    print("\n" + "=" * 70)
    print("🎉 DIVIDEND CALENDAR BUILD COMPLETE!")
    print("=" * 70)
    print("\n✅ You can now use this calendar to adjust EOD prices")
    print("✅ Re-run this script monthly to update dividend history")
    print("\n📌 Next Step:")
    print("   Run: python scripts/update_historical_data_hybrid.py")


if __name__ == "__main__":
    main()
