
import pandas as pd
from datetime import date
from pages.scanner.feature_lab.feature_experiments import calculate_dist_from_ma20
from utils.paths import HISTORICAL_DATA_DIR

# Test parameters
ticker = "5E2.SG"
test_date = date(2025, 7, 1)

print(f"Testing DistFromMA20 for {ticker} on {test_date}")
print(f"Historical Data Dir: {HISTORICAL_DATA_DIR}")

# Check if file exists
clean_ticker = ticker.replace('.SG', '')
csv_path = HISTORICAL_DATA_DIR / f"{clean_ticker}.csv"
print(f"CSV Path: {csv_path}")
print(f"File exists: {csv_path.exists()}")

if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"Loaded DataFrame with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    if 'Date' in df.columns:
        # Check specific date format
        print(f"Sample date format: {df['Date'].iloc[0]}")
        
        # Try to find the date
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        target_date = pd.Timestamp(test_date)
        print(f"Target date timestamp: {target_date}")
        
        match = df[df['Date'] == target_date]
        print(f"Matches for date: {len(match)}")
        if not match.empty:
            print(f"Close price: {match['Close'].iloc[0]}")

# Run calculation
try:
    result = calculate_dist_from_ma20(ticker, test_date)
    print(f"Calculation Result: {result}")
except Exception as e:
    print(f"Calculation Error: {e}")
