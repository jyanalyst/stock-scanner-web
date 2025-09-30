# File: test_alphavantage.py
"""
Alpha Vantage API Test Script for Singapore Exchange Stocks
Tests API connectivity and data availability for SGX stocks
"""

import requests
import json
from datetime import datetime

# Alpha Vantage Configuration
API_KEY = "NMWQL9IF5SRBA33N"
BASE_URL = "https://www.alphavantage.co/query"

def test_api_status():
    """Test if API key is valid and working"""
    print("=" * 60)
    print("TEST 1: API Key Validation")
    print("=" * 60)
    
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'IBM',  # Use a known US stock for validation
        'apikey': API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if 'Error Message' in data:
            print(f"❌ API Error: {data['Error Message']}")
            return False
        elif 'Note' in data:
            print(f"⚠️  API Limit: {data['Note']}")
            return False
        elif 'Time Series (Daily)' in data:
            print("✅ API key is valid and working!")
            print(f"   Successfully retrieved data for IBM")
            return True
        else:
            print(f"❓ Unexpected response: {list(data.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_symbol_search(keyword):
    """Test symbol search functionality for Singapore stocks"""
    print("\n" + "=" * 60)
    print(f"TEST 2: Symbol Search for '{keyword}'")
    print("=" * 60)
    
    params = {
        'function': 'SYMBOL_SEARCH',
        'keywords': keyword,
        'apikey': API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if 'bestMatches' in data:
            matches = data['bestMatches']
            print(f"✅ Found {len(matches)} matches:")
            
            singapore_matches = []
            for match in matches[:10]:  # Show top 10
                symbol = match.get('1. symbol', 'N/A')
                name = match.get('2. name', 'N/A')
                region = match.get('4. region', 'N/A')
                
                is_singapore = 'Singapore' in region or 'SGX' in region
                marker = "🇸🇬" if is_singapore else "  "
                
                print(f"   {marker} {symbol:12} | {name:40} | {region}")
                
                if is_singapore:
                    singapore_matches.append(symbol)
            
            return singapore_matches
        else:
            print(f"❌ No matches found or error: {data}")
            return []
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

def test_time_series_data(symbol, output_size='compact'):
    """Test downloading historical time series data"""
    print("\n" + "=" * 60)
    print(f"TEST 3: Time Series Data for '{symbol}'")
    print("=" * 60)
    
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': output_size,  # 'compact' = 100 days, 'full' = 20+ years
        'apikey': API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            dates = sorted(time_series.keys(), reverse=True)
            
            print(f"✅ Successfully retrieved {len(dates)} days of data")
            print(f"   Date range: {dates[-1]} to {dates[0]}")
            
            # Show latest 5 days
            print(f"\n   Latest 5 trading days:")
            print(f"   {'Date':12} | {'Open':8} | {'High':8} | {'Low':8} | {'Close':8} | {'Volume':12}")
            print(f"   {'-'*75}")
            
            for date in dates[:5]:
                day_data = time_series[date]
                open_price = day_data.get('1. open', 'N/A')
                high = day_data.get('2. high', 'N/A')
                low = day_data.get('3. low', 'N/A')
                close = day_data.get('4. close', 'N/A')
                volume = day_data.get('5. volume', 'N/A')
                
                print(f"   {date:12} | {open_price:8} | {high:8} | {low:8} | {close:8} | {volume:12}")
            
            return True
            
        elif 'Error Message' in data:
            print(f"❌ Error: {data['Error Message']}")
            return False
        elif 'Note' in data:
            print(f"⚠️  Rate limit hit: {data['Note']}")
            print(f"   Alpha Vantage free tier: 5 calls/minute, 500 calls/day")
            return False
        else:
            print(f"❓ Unexpected response structure:")
            print(f"   Keys: {list(data.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

def test_singapore_stocks():
    """Test with actual Singapore Exchange stocks from your watchlist"""
    print("\n" + "=" * 60)
    print("TEST 4: Your Singapore Watchlist Stocks")
    print("=" * 60)
    
    # Test stocks from your watchlist
    test_stocks = [
        ('A17U.SI', 'Ascendas REIT'),
        ('C38U.SI', 'CapitaLand Integrated Commercial Trust'),
        ('D05.SI', 'DBS Group Holdings'),
    ]
    
    print("\n📋 Testing stock symbol formats:")
    print("   Alpha Vantage may use different formats than Yahoo Finance")
    print("   Common formats: A17U.SI, A17U, A17U.SGX, SGX:A17U\n")
    
    successful_symbols = []
    
    for yahoo_symbol, name in test_stocks:
        print(f"\n🔍 Testing: {name} ({yahoo_symbol})")
        
        # Try multiple symbol formats
        formats_to_try = [
            yahoo_symbol,                    # A17U.SI
            yahoo_symbol.replace('.SI', ''), # A17U
            yahoo_symbol.replace('.SI', '.SGX'), # A17U.SGX
            f"SGX:{yahoo_symbol.replace('.SI', '')}", # SGX:A17U
        ]
        
        for symbol_format in formats_to_try:
            print(f"   Trying format: {symbol_format}...", end=" ")
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol_format,
                'outputsize': 'compact',
                'apikey': API_KEY
            }
            
            try:
                response = requests.get(BASE_URL, params=params, timeout=10)
                data = response.json()
                
                if 'Time Series (Daily)' in data:
                    time_series = data['Time Series (Daily)']
                    print(f"✅ SUCCESS! ({len(time_series)} days)")
                    successful_symbols.append((yahoo_symbol, symbol_format, name))
                    break
                elif 'Note' in data:
                    print(f"⏸️  Rate limited")
                    break
                else:
                    print(f"❌ No data")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print(f"   ⚠️  Could not find working format for {yahoo_symbol}")
    
    return successful_symbols

def main():
    """Run all tests"""
    print("\n" + "🇸🇬" * 30)
    print(" ALPHA VANTAGE API TEST - SINGAPORE EXCHANGE STOCKS")
    print("🇸🇬" * 30)
    print(f"\nAPI Key: {API_KEY}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 60)
    
    # Test 1: Validate API key
    if not test_api_status():
        print("\n❌ API key validation failed. Cannot proceed with further tests.")
        return
    
    # Test 2: Search for Singapore stocks
    print("\n⏳ Waiting 15 seconds (rate limit compliance)...")
    import time
    time.sleep(15)
    
    sg_symbols = test_symbol_search("Ascendas REIT")
    
    # Test 3: Try downloading time series for a found symbol
    if sg_symbols:
        print("\n⏳ Waiting 15 seconds (rate limit compliance)...")
        time.sleep(15)
        test_time_series_data(sg_symbols[0])
    
    # Test 4: Test your actual watchlist stocks
    print("\n⏳ Waiting 15 seconds (rate limit compliance)...")
    time.sleep(15)
    
    successful = test_singapore_stocks()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if successful:
        print(f"✅ Successfully downloaded data for {len(successful)} stocks:")
        for yahoo_sym, alpha_sym, name in successful:
            print(f"   {yahoo_sym:12} → {alpha_sym:15} | {name}")
        
        print("\n💡 RECOMMENDATION:")
        print("   Alpha Vantage CAN download Singapore Exchange data!")
        print("   However, you may need to map Yahoo Finance symbols to Alpha Vantage symbols.")
        print("\n⚠️  IMPORTANT LIMITATIONS:")
        print("   • Free tier: 5 API calls/minute, 500 calls/day")
        print("   • Your scanner has 46 stocks - would take ~10 minutes to scan all")
        print("   • Consider caching strategy or paid tier for production use")
    else:
        print("❌ Could not retrieve data for Singapore stocks")
        print("   This could be due to:")
        print("   • Rate limiting (wait and try again)")
        print("   • Symbol format issues")
        print("   • Limited SGX coverage in Alpha Vantage")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()