# ML Pipeline Known Limitations

**Document Version:** 1.0  
**Last Updated:** December 11, 2025  
**Status:** Active - Production Awareness Required

---

## Executive Summary

This document outlines known limitations in the ML training pipeline that affect model performance expectations. These limitations are **accepted trade-offs** due to data availability constraints and practical implementation considerations.

**Key Impact:** Expected 3-5% inflation in win rates and ~0.5-1.0% underestimation of risk metrics.

---

## 1. Survivorship Bias

### Description
Training data only includes stocks that **survived until 2025** (current date). Stocks that were delisted, acquired, or failed during the 2023-2024 training period are **excluded** from the dataset.

### Impact
- **Win Rate Inflation:** 3-5% higher than realistic
- **Return Inflation:** Approximately +0.3-0.5% per trade
- **Risk Underestimation:** Missing tail risk from failed stocks

### Example
```
Stock XYZ Timeline:
- 2023 Q1-Q3: Generated 50 signals (mixed performance)
- 2023 Q4: Company announced financial difficulties
- 2024 Q1: Stock delisted after bankruptcy

Training Data Impact:
- All 50 XYZ signals are MISSING from training data
- Model never learns from pre-failure patterns
- Overestimates strategy profitability
```

### Why Not Fixed
- **Data Unavailable:** No historical watchlist snapshots exist for 2023-2024
- **Effort vs Benefit:** Would require manual reconstruction of historical universe
- **Acceptable Trade-off:** 3-5% bias is within acceptable tolerance for strategy validation

### Mitigation Strategy
1. **Awareness:** Document this bias in all model reports
2. **Conservative Sizing:** Reduce position sizes by 10% to account for bias
3. **Live Monitoring:** Track actual vs predicted performance to quantify real-world impact
4. **Future Improvement:** Maintain historical watchlist snapshots going forward

### Validation Check
```python
# Compare 2023 vs 2025 stock universe
stocks_2023 = set(get_stocks_from_training_data())  # ~40 stocks
stocks_2025 = set(get_active_watchlist())           # ~42 stocks

missing_stocks = stocks_2023 - stocks_2025
new_stocks = stocks_2025 - stocks_2023

print(f"Stocks that disappeared: {len(missing_stocks)}")
print(f"New stocks added: {len(new_stocks)}")
# Expected: 2-5 stocks disappeared (delisted/acquired)
```

---

## 2. Entry Price Convention

### Description
The model uses **Day 0 close price** as the entry price, assuming zero slippage and perfect execution at the signal close.

### Reality vs Convention

| Aspect | Model Convention | Real Trading |
|--------|-----------------|--------------|
| Signal Time | 5:00 PM close | 5:00 PM close |
| Entry Time | 5:00 PM (same day) | 9:00 AM next day |
| Entry Price | Day 0 close | Day 1 open |
| Slippage | 0% | ~0.15% average |

### Impact
- **Return Overestimation:** +0.15% per trade on average
- **Overnight Gap Risk:** Not captured in model
- **Execution Assumptions:** Unrealistic for live trading

### Example
```
Signal: Friday Nov 22, 2024
- Day 0 Close: $3.39 (model entry price)
- Day 1 Open: $3.42 (realistic entry price)
- Overnight Gap: +0.88% (not in model)

Model Prediction: +2.00% return
Realistic Expectation: +1.85% return (after slippage)
```

### Why This Convention
1. **Consistency:** Matches backtest assumptions across all historical data
2. **Simplicity:** Easier to calculate and validate
3. **Conservative Adjustment:** Can apply slippage factor in production

### Mitigation Strategy
1. **Production Adjustment:** Subtract 0.15% from all model predictions
2. **Slippage Tracking:** Monitor actual entry prices vs Day 0 close
3. **Model Metadata:** Document this assumption clearly
4. **Future Enhancement:** Consider adding Day 1 open as alternative entry price

### Production Formula
```python
# Model prediction
model_return = 2.00%

# Adjust for realistic entry
expected_slippage = 0.15%
realistic_return = model_return - expected_slippage
# = 1.85%

# Use realistic_return for position sizing and risk management
```

---

## 3. Corporate Actions (Accepted Risk)

### Description
The model does **not adjust** for stock splits, dividends, or bonus issues during the holding period.

### Impact
- **Rare Occurrence:** SGX REITs rarely split (last major split: 2019)
- **Dividend Impact:** Minimal for 2-4 day holding periods
- **Potential Error:** If split occurs, returns will be grossly incorrect

### Example of Split Impact
```
Stock A17U:
- Nov 15: Signal at $3.50
- Nov 16: 2-for-1 split → $1.75
- Nov 18: Exit at $1.80

Without Adjustment:
  return = ($1.80 - $3.50) / $3.50 = -48.6%  ❌ WRONG

With Adjustment:
  adjusted_entry = $3.50 / 2 = $1.75
  return = ($1.80 - $1.75) / $1.75 = +2.86%  ✅ Correct
```

### Why Not Implemented
- **Low Frequency:** No major splits in SGX REITs during 2023-2024
- **Validation:** Manual check confirmed no splits in training period
- **Effort vs Benefit:** Implementation cost > expected benefit

### Mitigation Strategy
1. **Manual Monitoring:** Check for announced splits before trading
2. **Data Validation:** Verify no splits occurred in training period
3. **Future Enhancement:** Implement if splits become more common

### Validation Check
```python
# Check for suspicious return spikes (potential splits)
suspicious_returns = df[abs(df['return_3d']) > 0.30]  # >30% moves
print(f"Suspicious returns: {len(suspicious_returns)}")
# Expected: 0-2 (should be rare for REITs)
```

---

## 4. Max Drawdown Calculation (Minor Issue)

### Description
Max drawdown currently uses **Close prices** instead of **Low prices**, underestimating intraday risk.

### Impact
- **Risk Underestimation:** 0.5-1.0% per trade
- **Stop Loss Placement:** May be too tight for actual volatility
- **Backtesting:** Overly optimistic risk-adjusted returns

### Example
```
Holding Period: Nov 22-26
Entry: $3.39

Using Close Prices:
  Worst close: $3.30
  Max DD: -2.65%

Using Low Prices (Realistic):
  Worst intraday low: $3.25
  Max DD: -4.13%  (1.48% worse)
```

### Status
**FIXED in ml/data_collection.py** - Now uses Low prices for max drawdown calculation.

---

## 5. Weekend and Holiday Effects

### Description
The model does **not explicitly account** for weekend decay or holiday effects on forward returns.

### Impact
- **Friday Signals:** May have different risk/return profile
- **Holiday Periods:** Reduced liquidity not captured
- **Seasonal Patterns:** Not explicitly modeled

### Mitigation Strategy
1. **Weekday Features:** Added in Fix #8 (one-hot encoding of entry day)
2. **Live Monitoring:** Track Friday vs Monday signal performance
3. **Future Enhancement:** Add holiday calendar features

---

## Summary Table

| Limitation | Impact | Status | Mitigation |
|-----------|--------|--------|------------|
| Survivorship Bias | +3-5% win rate | Accepted | Conservative sizing |
| Entry Price Convention | +0.15% return | Accepted | Production adjustment |
| Corporate Actions | Rare, high impact | Accepted | Manual monitoring |
| Max Drawdown | +0.5-1.0% risk | **FIXED** | Using Low prices |
| Weekend Effects | Unknown | **FIXING** | Adding weekday features |

---

## Production Checklist

Before deploying model to live trading:

- [ ] Apply -0.15% slippage adjustment to all predictions
- [ ] Reduce position sizes by 10% to account for survivorship bias
- [ ] Check for announced corporate actions before trading
- [ ] Monitor Friday signals separately (may need different thresholds)
- [ ] Track actual vs predicted returns to quantify real-world bias
- [ ] Document these limitations in trading plan

---

## Model Performance Expectations

### Backtested (Inflated)
- IC: 0.084
- Win Rate: 58%
- Avg Return: 2.0%

### Realistic (After Adjustments)
- IC: 0.045-0.055 (↓40% due to trading days fix)
- Win Rate: 52-54% (↓4-6% due to survivorship bias)
- Avg Return: 1.85% (↓0.15% due to slippage)

### Still Tradeable?
**YES** - IC > 0.04 with 20+ stocks is viable for systematic trading.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-11 | Initial documentation of known limitations |

---

**Document Owner:** ML Pipeline Team  
**Review Frequency:** Quarterly  
**Next Review:** March 2026
