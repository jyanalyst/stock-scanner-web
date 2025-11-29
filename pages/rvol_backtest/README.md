# RVOL BackTest - Monthly VWAP Mean Reversion Strategy

## 🎯 Overview

RVOL BackTest is a sophisticated backtesting system that optimizes entry parameters for a Monthly VWAP mean reversion trading strategy. Unlike traditional backtests that assume 100% fill rates, this system implements **realistic retest fill logic** to provide actionable, tradeable results.

## 📊 Strategy Description

### Core Concept
The strategy identifies mean reversion opportunities when price drops significantly below the Monthly VWAP (Volume Weighted Average Price), expecting price to revert back to the VWAP level.

### Key Innovation: Realistic Fill Logic
- **Traditional Backtests**: Assume orders fill immediately (100% fill rate)
- **RVOL BackTest**: Orders only fill when price **retests** the entry level
- **Result**: Fill rates of 50-70% (realistic) vs 100% (unrealistic)

### Strategy Rules

#### Entry Conditions
1. **Signal Generation**: Current Low ≤ Monthly VWAP - X% (X = deviation threshold)
2. **Order Placement**: Limit order placed at signal day's Low
3. **Order Fill**: Order fills only when price retests the limit level on subsequent days

#### Exit Conditions (First triggered wins)
1. **Target Exit**: High ≥ Entry VWAP (mean reversion achieved)
2. **Stop Loss**: Low ≤ Entry Price × (1 - 1.2%) (risk management)

#### Position Sizing
- Fixed: 50,000 shares per trade (configurable)

## 🏗️ System Architecture

```
pages/rvol_backtest/
├── __init__.py           # Main page entry point
├── vwap_engine.py        # Monthly VWAP calculation
├── backtest_engine.py    # Realistic retest fill logic
├── optimizer.py          # Parameter sweep optimization
├── ui.py                 # Streamlit user interface
└── test_rvol_backtest.py # Validation tests
```

### Core Components

#### 1. VWAP Engine (`vwap_engine.py`)
**Purpose**: Calculate Monthly VWAP with proper resets
- Groups data by Year-Month
- Calculates running cumulative VWAP within each month
- Resets calculations at month boundaries

#### 2. Backtest Engine (`backtest_engine.py`)
**Purpose**: Implement realistic trading simulation
- **State Machine**: NO_POSITION → PENDING_ORDER → IN_POSITION
- **Realistic Fills**: Orders fill on retest, not immediately
- **Complete Trade Tracking**: Signal date, fill date, exit date, P&L

#### 3. Optimizer (`optimizer.py`)
**Purpose**: Find optimal deviation threshold
- Tests multiple thresholds (e.g., 0.5% to 2.0%)
- Maximizes Sharpe ratio (risk-adjusted returns)
- Provides confidence levels and risk warnings

#### 4. UI Components (`ui.py`)
**Purpose**: Professional Streamlit interface
- Stock selection from watchlist
- Date range and parameter configuration
- Real-time optimization progress
- Results dashboard with fill rate metrics

## 📈 Key Metrics & Outputs

### Fill Rate Analysis
- **Signals Generated**: Total entry signals detected
- **Orders Filled**: Signals that resulted in actual trades
- **Fill Rate**: (Filled Orders / Total Signals) × 100
- **Average Days to Fill**: How long orders take to execute

### Performance Metrics
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return measure (PRIMARY optimization target)
- **Total P&L**: Cumulative profit/loss in dollars
- **Max Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profit / Gross loss

### Optimization Results
- **Optimal Threshold**: Best deviation percentage below VWAP
- **Confidence Level**: HIGH/MODERATE/LOW based on sample size and metrics
- **Risk Warnings**: Alerts for unrealistic results or high risk

## 🚀 Usage Guide

### 1. Access the System
Navigate to "📊 RVOL BackTest" in the sidebar

### 2. Configure Parameters
- **Stock Selection**: Choose from your watchlist
- **Date Range**: Select backtesting period (minimum 30 days)
- **Threshold Range**: Min/Max deviation below VWAP (e.g., 0.5% to 2.0%)
- **Step Size**: Increment between threshold tests (e.g., 0.1%)
- **Stop Loss**: Risk management level (default 1.2%)

### 3. Run Optimization
Click "🚀 Start Optimization" to begin parameter sweep

### 4. Analyze Results
- **Optimal Threshold**: Recommended deviation for live trading
- **Fill Rate**: How tradeable the strategy is (50-70% is good)
- **Performance Metrics**: Expected win rate, Sharpe ratio, etc.
- **Trade Log**: Detailed history of all completed trades

## 🎯 Interpretation Guide

### Fill Rate Guidelines
- **< 30%**: Too restrictive, hard to execute
- **30-70%**: Realistic, good balance of selectivity and tradeability
- **> 80%**: May be over-optimized, potentially curve-fitted

### Sharpe Ratio Guidelines
- **> 1.0**: Excellent risk-adjusted returns
- **0.5-1.0**: Good risk-adjusted returns
- **< 0.0**: Negative risk-adjusted returns (avoid)

### Confidence Level
- **HIGH**: Large sample size, realistic metrics
- **MODERATE**: Reasonable sample size and metrics
- **LOW**: Small sample or concerning metrics

## 🔧 Technical Implementation

### State Machine Logic
```python
# Three states with realistic order handling
NO_POSITION    → Scan for signals
PENDING_ORDER  → Limit order placed, waiting for retest
IN_POSITION    → Trade active, monitor exits
```

### VWAP Calculation Algorithm
```python
# Monthly VWAP with running resets
For each month:
    cumulative_price_volume = Σ(price × volume)
    cumulative_volume = Σ(volume)
    monthly_vwap = cumulative_price_volume / cumulative_volume
```

### Optimization Process
```python
# Parameter sweep maximizing Sharpe ratio
For threshold in [0.5%, 0.6%, ..., 2.0%]:
    Run backtest with realistic fills
    Calculate Sharpe ratio
    Track fill rate and other metrics

Return threshold with highest Sharpe ratio
```

## ⚠️ Important Considerations

### Known Limitations
1. **No Slippage**: Assumes exact limit order execution
2. **No Commissions**: Zero transaction costs assumed
3. **Daily Data**: Uses daily OHLCV bars (not intraday)
4. **Fixed Position Size**: No dynamic position sizing

### Risk Warnings
- Backtest results ≠ Live trading performance
- Past performance ≠ Future results
- Fill rates may vary in live markets
- Consider transaction costs in live implementation

### Best Practices
- Use at least 1 year of historical data
- Validate results across different market conditions
- Consider fill rate when selecting optimal threshold
- Test with out-of-sample data when possible

## 🧪 Validation & Testing

### Automated Tests
Run validation tests with:
```bash
cd pages/rvol_backtest
python test_rvol_backtest.py
```

### Test Coverage
- ✅ VWAP calculation accuracy
- ✅ State machine logic
- ✅ Fill rate realism
- ✅ Performance metric calculations
- ✅ Optimization parameter sweep

## 📊 Sample Output

### Optimization Results Table
| Threshold | Signals | Fill Rate | Trades | Win Rate | Sharpe | Total P&L |
|-----------|---------|-----------|--------|----------|--------|-----------|
| 1.2%     | 45      | 62.2%    | 28     | 67.9%   | 1.45  | $142,500 |
| 1.3%     | 38      | 55.3%    | 21     | 71.4%   | 1.32  | $98,700  |
| 1.1%     | 52      | 69.2%    | 36     | 63.9%   | 1.28  | $156,200 |

### Optimal Recommendation
```
🎯 Optimal Threshold: 1.2%
📊 Expected Sharpe: 1.45
✅ Expected Win Rate: 67.9%
🎯 Fill Rate: 62.2% (Realistic)
💰 Expected P&L: $142,500
🎚️ Confidence Level: HIGH
```

## 🔄 Future Enhancements

### Phase 2 Features (Planned)
- Multi-stock portfolio optimization
- Walk-forward testing
- Transaction cost modeling
- Intraday simulation (if data available)
- Advanced risk management
- Performance attribution analysis

### Integration Opportunities
- Connect to live trading platforms
- Automated parameter re-optimization
- Real-time signal generation
- Performance monitoring dashboard

## 📞 Support & Documentation

### Getting Help
- Check this README for detailed explanations
- Review inline code documentation
- Run validation tests for troubleshooting
- Check Streamlit error logs for runtime issues

### Code Structure
- All functions have comprehensive docstrings
- Key algorithms include detailed comments
- Validation functions ensure data integrity
- Error handling prevents crashes

---

**RVOL BackTest**: Where sophisticated backtesting meets realistic execution assumptions.
