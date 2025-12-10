# Hybrid Historical Data Updater - Complete Guide

## 📋 Overview

The Hybrid Historical Data Updater is a sophisticated script that maintains your historical stock data by intelligently combining multiple data sources with proper dividend adjustments.

**File:** `scripts/update_historical_data_hybrid.py`

---

## 🎯 Purpose

Automatically update Historical_Data with:
1. **EOD files** (manual CSV files with dividend adjustment)
2. **yfinance API** (automated downloads, already dividend-adjusted)
3. **Gap filling** (automatically fills missing dates)
4. **Feature calculation** (all 32 features recalculated for consistency)

---

## 🔧 How It Works

### **Three-Tier Data Strategy**

```
Priority 1: EOD_Data (manual, reliable, needs dividend adjustment)
     ↓
Priority 2: yfinance (automated, has delays, already adjusted)
     ↓
Priority 3: User Alert (manual intervention needed)
```

### **Dividend Adjustment Method**

**BACKWARD ADJUSTMENT** (Current prices stay real)

```
Example:
- Stock closes at $1.00 on 08/12/2025
- Dividend of $0.02 announced
- Adjustment factor = 1 - (0.02 / 1.00) = 0.98

Result:
- Prices BEFORE 08/12/2025: Multiplied by 0.98
- Prices ON/AFTER 08/12/2025: Stay at real market prices
```

**Why backward adjustment?**
- ✅ Current prices = REAL market prices
- ✅ Matches yfinance methodology
- ✅ Industry standard approach
- ✅ Easier to understand and validate

---

## 📊 Process Flow

```
┌─────────────────────────────────────────┐
│ 1. Load Dividend Calendar               │
│    (1,627 dividend events)              │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. Check for New EOD File               │
│    - Compare dates                      │
│    - Identify if processing needed      │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Process EOD File                     │
│    - Apply dividend adjustments         │
│    - Calculate adjustment factors       │
│    - Adjust OHLC prices                 │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. Check for Gaps                       │
│    - Compare last_date vs today         │
│    - Identify missing dates             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 5. Fill Gaps with yfinance              │
│    - Download missing dates             │
│    - Already dividend-adjusted          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 6. Merge & Recalculate Features         │
│    - Combine all data sources           │
│    - Recalculate all 32 features        │
│    - Sort and remove duplicates         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 7. Validate & Save                      │
│    - Check data consistency             │
│    - Save to Historical_Data/*.csv      │
│    - Generate report                    │
└─────────────────────────────────────────┘
```

---

## 🚀 Usage

### **Basic Usage**

```bash
python scripts/update_historical_data_hybrid.py
```

### **Prerequisites**

1. **Dividend Calendar** (required for EOD adjustment)
   ```bash
   python scripts/build_dividend_calendar.py
   ```

2. **EOD Files** (optional, but recommended)
   - Place in: `data/EOD_Data/`
   - Format: `DD_MMM_YYYY.csv` (e.g., `09_Dec_2025.csv`)

3. **Internet Connection** (for yfinance gap filling)

---

## 📈 Usage Scenarios

### **Scenario A: New EOD File Available**

```bash
$ python scripts/update_historical_data_hybrid.py
```

**Output:**
```
======================================================================
🔄 HYBRID HISTORICAL DATA UPDATER
======================================================================

📅 STEP 1: Load Dividend Calendar
✅ Loaded dividend calendar: 1,627 events

📊 STEP 2: Check for New EOD Data
   Latest EOD file: 09_Dec_2025.csv
   Watchlist: 54 stocks
   Last historical date: 2025-12-08
   EOD date: 2025-12-09
   ✅ New EOD data available!

📊 Processing EOD file: 09_Dec_2025.csv
   Date: 2025-12-09
   Stocks: 54
✅ Processed 54 stocks
   Dividend adjustments applied: 3 stocks

📊 STEP 3: Check for Gaps
   Current working day: 2025-12-09
   ✅ No gaps detected

📊 STEP 4: Update Historical Data
   Processing 54 stocks...
   [████████████████████████████████] 54/54 100%

======================================================================
📊 UPDATE SUMMARY
======================================================================

✅ Updated: 54/54 stocks
○  Already Current: 0 stocks
✗  Failed: 0 stocks

📈 Data Added:
   • EOD days: 54
   • yfinance days: 0
   • Total data points: 54

⏱️  Duration: 45.2 seconds

📄 Report saved: scripts/hybrid_update_report_20251209_201530.txt

======================================================================
🎉 HYBRID UPDATE COMPLETE!
======================================================================
```

---

### **Scenario B: EOD Delayed, yfinance Available**

```bash
$ python scripts/update_historical_data_hybrid.py
```

**Output:**
```
📊 STEP 2: Check for New EOD Data
   Latest EOD file: 06_Dec_2025.csv
   ℹ️  EOD file already processed (date: 2025-12-06)

📊 STEP 3: Check for Gaps
   Current working day: 2025-12-09
   ⚠️  Gap detected: 3 day(s)
   Will fill with yfinance...

📊 STEP 4: Update Historical Data
   Processing 54 stocks...
   [████████████████████████████████] 54/54 100%

======================================================================
📊 UPDATE SUMMARY
======================================================================

✅ Updated: 54/54 stocks
○  Already Current: 0 stocks
✗  Failed: 0 stocks

📈 Data Added:
   • EOD days: 0
   • yfinance days: 162 (3 days × 54 stocks)
   • Total data points: 162

⏱️  Duration: 3.5 minutes
```

---

### **Scenario C: Already Current**

```bash
$ python scripts/update_historical_data_hybrid.py
```

**Output:**
```
📊 STEP 2: Check for New EOD Data
   Latest EOD file: 09_Dec_2025.csv
   ℹ️  EOD file already processed (date: 2025-12-09)

📊 STEP 3: Check for Gaps
   Current working day: 2025-12-09
   ✅ No gaps detected

📊 STEP 4: Update Historical Data
   Processing 54 stocks...
   [████████████████████████████████] 54/54 100%

======================================================================
📊 UPDATE SUMMARY
======================================================================

✅ Updated: 0/54 stocks
○  Already Current: 54 stocks
✗  Failed: 0 stocks

📈 Data Added:
   • EOD days: 0
   • yfinance days: 0
   • Total data points: 0

⏱️  Duration: 5.1 seconds
```

---

## 🔍 Technical Details

### **Dividend Adjustment Calculation**

```python
# Single dividend
adjustment_factor = 1 - (dividend_amount / close_price_before_ex_div)

# Example:
# Dividend: $0.02
# Close before: $1.00
# Factor: 1 - (0.02 / 1.00) = 0.98

# Apply to OHLC:
adjusted_open = raw_open × 0.98
adjusted_high = raw_high × 0.98
adjusted_low = raw_low × 0.98
adjusted_close = raw_close × 0.98
```

### **Multiple Dividends (Compound)**

```python
# Stock has 3 dividends:
# - 01/06/2025: $0.01 (factor: 0.99)
# - 01/09/2025: $0.015 (factor: 0.985)
# - 01/12/2025: $0.02 (factor: 0.98)

# Compound factor for prices before 01/06/2025:
compound_factor = 0.99 × 0.985 × 0.98 = 0.955

# All prices before 01/06/2025 multiplied by 0.955
```

### **Feature Recalculation**

**Why recalculate ALL features?**

1. **Moving Averages** need historical context
   ```python
   MA_50 = mean(prices[-50:])  # Needs 50 days
   ```

2. **Technical Indicators** have dependencies
   ```python
   RSI_14 = calculate_rsi(prices, 14)  # Needs 14+ days
   MACD = calculate_macd(prices, 12, 26)  # Needs 26+ days
   ```

3. **Consistency** across entire dataset
   - Prevents drift between old and new data
   - Ensures all features use same methodology
   - Makes debugging easier

**Performance:** ~5-10 seconds for 54 stocks (acceptable trade-off)

---

## 📁 File Structure

### **Input Files**

```
data/
├── EOD_Data/
│   ├── 06_Dec_2025.csv          # Manual EOD files
│   ├── 09_Dec_2025.csv
│   └── ...
├── dividend_calendar/
│   ├── dividend_calendar.json   # Dividend history
│   └── metadata.json
└── Historical_Data/
    ├── A17U.csv                 # Existing historical data
    ├── C38U.csv
    └── ...
```

### **Output Files**

```
data/
└── Historical_Data/
    ├── A17U.csv                 # Updated with 32 features
    ├── C38U.csv
    └── ...

scripts/
└── hybrid_update_report_YYYYMMDD_HHMMSS.txt  # Detailed report
```

---

## 📊 Output Format

### **CSV Structure (32 columns)**

```csv
Date,Code,Shortname,Open,High,Low,Close,Vol,
Dividend,DaysToNextDiv,DivYield,DivGrowthRate,ConsecutiveDivs,IsExDivWeek,
Split,DaysSinceSplit,SplitInLast90Days,
MA_20,MA_50,MA_200,
RSI_14,MACD,
BB_Upper,BB_Middle,BB_Lower,ATR_14,
ROC_5,ROC_10,ROC_20,DistFromMA20,DistFromMA50,VolRatio
```

**Column Groups:**
- **Basic (8):** Date, Code, Shortname, OHLC, Vol
- **Dividends (6):** Amount, Days, Yield, Growth, Count, Flag
- **Splits (3):** Ratio, Days, Flag
- **Moving Averages (3):** MA_20, MA_50, MA_200
- **Oscillators (2):** RSI_14, MACD
- **Volatility (4):** BB_Upper/Middle/Lower, ATR_14
- **Momentum (6):** ROC_5/10/20, Dist, VolRatio

---

## ⚙️ Configuration

### **Data Source Priority**

```python
# Priority order (highest to lowest):
1. EOD_Data (manual, reliable, dividend-adjusted by script)
2. yfinance (automated, may have delays, already adjusted)
3. Existing Historical_Data (fallback)
```

### **Date Handling**

```python
# Current working day calculation:
- Monday-Friday: Today
- Saturday: Friday
- Sunday: Friday
```

### **Gap Detection**

```python
# Gaps are filled if:
last_historical_date < current_working_day

# Example:
# Last historical: 06/12/2025
# Current working day: 09/12/2025
# Gap: 07/12, 08/12, 09/12 (3 days)
```

---

## 🔧 Maintenance

### **Daily Workflow**

```bash
# Run hybrid updater daily
python scripts/update_historical_data_hybrid.py
```

**Expected duration:**
- With new EOD file: 45-60 seconds
- With yfinance only: 2-4 minutes
- Already current: 5-10 seconds

### **Monthly Workflow**

```bash
# Update dividend calendar monthly
python scripts/build_dividend_calendar.py

# Then run hybrid updater
python scripts/update_historical_data_hybrid.py
```

### **Troubleshooting**

**Problem: "Dividend calendar not found"**
```bash
# Solution: Build dividend calendar
python scripts/build_dividend_calendar.py
```

**Problem: "No EOD files found"**
```bash
# Solution: Place EOD CSV files in data/EOD_Data/
# Or: Script will use yfinance automatically
```

**Problem: "yfinance download failed"**
```bash
# Possible causes:
# 1. Internet connection issue
# 2. yfinance API rate limit
# 3. Stock ticker not found

# Solution: Wait and retry, or manually download EOD file
```

---

## 📈 Performance Metrics

### **Typical Performance**

| Scenario | Stocks | Days | Duration |
|----------|--------|------|----------|
| EOD only | 54 | 1 | 45-60 sec |
| yfinance 1 day | 54 | 1 | 2-3 min |
| yfinance 3 days | 54 | 3 | 3-5 min |
| Already current | 54 | 0 | 5-10 sec |

### **Bottlenecks**

1. **yfinance API calls** (rate limited)
2. **Feature calculation** (acceptable ~10 sec)
3. **Disk I/O** (minimal impact)

---

## ✅ Validation

### **Automatic Checks**

1. **Date consistency** - Chronological order
2. **Duplicate detection** - Removed automatically
3. **Gap detection** - Filled with yfinance
4. **Feature completeness** - All 32 columns present

### **Manual Verification**

```bash
# Check a sample file
head -20 data/Historical_Data/A17U.csv

# Verify last date
tail -5 data/Historical_Data/A17U.csv

# Check report
cat scripts/hybrid_update_report_*.txt
```

---

## 🎯 Best Practices

### **DO:**
✅ Run daily to keep data current
✅ Update dividend calendar monthly
✅ Keep EOD files organized by date
✅ Review reports for errors
✅ Verify data quality periodically

### **DON'T:**
❌ Run multiple times simultaneously
❌ Manually edit Historical_Data files
❌ Delete dividend calendar
❌ Skip dividend calendar updates
❌ Ignore error messages in reports

---

## 🔄 Integration with Other Scripts

### **Workflow Integration**

```bash
# 1. Monthly: Update dividend calendar
python scripts/build_dividend_calendar.py

# 2. Daily: Update historical data
python scripts/update_historical_data_hybrid.py

# 3. As needed: Full rebuild (if data corrupted)
python scripts/rebuild_dividend_adjusted_data_smart.py

# 4. After updates: Re-run ML collection
python scripts/run_ml_collection_clean.py
```

---

## 📚 Related Scripts

| Script | Purpose | Frequency |
|--------|---------|-----------|
| `build_dividend_calendar.py` | Build dividend calendar | Monthly |
| `update_historical_data_hybrid.py` | Daily updates | Daily |
| `rebuild_dividend_adjusted_data_smart.py` | Full rebuild | As needed |
| `rebuild_dividend_adjusted_data_enhanced.py` | Complete rebuild | Rarely |

---

## 🎉 Summary

The Hybrid Historical Data Updater provides:

✅ **Automated daily updates** with minimal manual intervention
✅ **Proper dividend adjustments** for EOD data
✅ **Gap filling** with yfinance
✅ **Full feature calculation** (32 features)
✅ **Comprehensive reporting** and validation
✅ **Flexible data sources** (EOD + yfinance)

**Result:** Always-current, dividend-adjusted historical data with complete technical features for ML model training and stock analysis.

---

## 📞 Support

For issues or questions:
1. Check the generated report file
2. Review error messages in console output
3. Verify dividend calendar exists
4. Ensure EOD files are properly formatted
5. Check internet connection for yfinance

---

**Last Updated:** December 10, 2025
**Version:** 1.0
**Author:** Stock Scanner Web Project
