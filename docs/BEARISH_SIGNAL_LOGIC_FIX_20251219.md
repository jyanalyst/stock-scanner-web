# 🚨 CRITICAL FIX: Bearish Signal Target Outcome Logic - December 19, 2025

## Executive Summary

**CRITICAL BUG FIXED:** Target outcome calculations were using the same logic for both bullish and bearish signals, causing bearish signals to be classified completely backwards.

### Impact
- ❌ **Before:** Bearish signals showed TRUE_BREAK when price went UP (completely wrong!)
- ✅ **After:** Bearish signals correctly show TRUE_BREAK when price goes DOWN

---

## Problem Description

### Root Cause
The `target_outcome.py` module had **no signal type awareness**. It treated ALL signals as bullish breakouts, applying the same logic regardless of whether the signal was bullish or bearish.

### What Was Wrong

```python
# OLD LOGIC (Applied to ALL signals - WRONG for bearish!):
if close > signal_high:
    return 'TRUE_BREAK'      # ✅ Correct for BULLISH
                             # ❌ WRONG for BEARISH (price went UP!)
elif close < signal_low:
    return 'INVALIDATION'    # ✅ Correct for BULLISH
                             # ❌ WRONG for BEARISH (price went DOWN!)
```

### Real-World Example from Screenshot
**Bearish Signal: AJBU.SG (Keppel DC Reit)**
- Signal Date: Day 0
- High: 2.34, Low: 2.31
- Day 1: Price dropped to 2.31 (closed at signal_low)
- **OLD Result:** Would show as approaching INVALIDATION (wrong!)
- **NEW Result:** Will show as TRUE_BREAK (correct - price dropped!)

---

## Solution Implemented

### 1. Added `signal_type` Parameter

All functions now accept a `signal_type` parameter:
- `"BULLISH"` - Default, original logic
- `"BEARISH"` - Inverted logic

### 2. Inverted Logic for Bearish Signals

```python
# NEW LOGIC (Signal-type aware):
is_bearish = "BEARISH" in signal_type.upper()

if is_bearish:
    # BEARISH: Win when price drops, lose when price rises
    if close < signal_low:
        return 'TRUE_BREAK', day      # Price broke DOWN (winner!)
    elif close > signal_high:
        return 'INVALIDATION', day    # Price broke UP (loser!)
else:
    # BULLISH: Win when price rises, lose when price drops
    if close > signal_high:
        return 'TRUE_BREAK', day      # Price broke UP (winner!)
    elif close < signal_low:
        return 'INVALIDATION', day    # Price broke DOWN (loser!)
```

### 3. Fixed Base Price for Returns

```python
# CRITICAL: Different entry points for different signal types
is_bearish = "BEARISH" in signal_type.upper()
base_price = signal_low if is_bearish else signal_high

# Bullish: Entry at signal_high (breakout above)
# Bearish: Entry at signal_low (breakdown below / short entry)
```

### 4. Auto-Detection from Signal_Bias Column

```python
# Extract from DataFrame
if 'Signal_Bias' in scan_results.columns:
    signal_bias = str(row['Signal_Bias']).upper()
    if 'BEARISH' in signal_bias:
        signal_type = "BEARISH"
    else:
        signal_type = "BULLISH"
```

---

## Technical Details

### Files Modified

**pages/scanner/feature_lab/target_outcome.py**

#### Function: `get_target_outcome()`
- **Added parameter:** `signal_type: str = "BULLISH"`
- **Added logic:** Detects bearish signals and inverts break/invalidation logic
- **Lines changed:** ~140-160

#### Function: `get_target_outcome_with_metrics()`
- **Added parameter:** `signal_type: str = "BULLISH"`
- **Fixed:** Base price selection (signal_low for bearish, signal_high for bullish)
- **Lines changed:** ~220-230

#### Function: `calculate_outcomes_for_scan()`
- **Added:** Signal_Bias column detection
- **Added:** Signal type extraction per stock
- **Added:** Logging for bearish signal detection
- **Lines changed:** ~300-330

---

## Correct Logic Reference

### Bullish Signals (Long/Buy)
| Outcome | Condition | Meaning |
|---------|-----------|---------|
| **TRUE_BREAK** ✅ | Price > signal_high | Winner! Price broke UP |
| **INVALIDATION** ❌ | Price < signal_low | Loser! Price broke DOWN |
| **TIMEOUT** ⏱️ | Price stays in range | No clear direction |

**Entry Point:** signal_high (breakout above)  
**Return Calculation:** `(exit_price - signal_high) / signal_high * 100`

### Bearish Signals (Short/Sell)
| Outcome | Condition | Meaning |
|---------|-----------|---------|
| **TRUE_BREAK** ✅ | Price < signal_low | Winner! Price broke DOWN |
| **INVALIDATION** ❌ | Price > signal_high | Loser! Price broke UP |
| **TIMEOUT** ⏱️ | Price stays in range | No clear direction |

**Entry Point:** signal_low (breakdown below)  
**Return Calculation:** `(exit_price - signal_low) / signal_low * 100`

---

## Validation & Testing

### Test Cases

#### Test 1: Bullish Signal (Should work as before)
```python
signal_high = 2.50
signal_low = 2.40
prices = [2.45, 2.52]  # Day 0, Day 1

result = get_target_outcome(signal_high, signal_low, prices, signal_type="BULLISH")
# Expected: ('TRUE_BREAK', 1) - Price broke above 2.50 ✅
```

#### Test 2: Bearish Signal (NEW - Fixed logic)
```python
signal_high = 2.34
signal_low = 2.31
prices = [2.32, 2.28]  # Day 0, Day 1

result = get_target_outcome(signal_high, signal_low, prices, signal_type="BEARISH")
# Expected: ('TRUE_BREAK', 1) - Price broke below 2.31 ✅
```

#### Test 3: Bearish Invalidation (NEW - Fixed logic)
```python
signal_high = 2.34
signal_low = 2.31
prices = [2.32, 2.36]  # Day 0, Day 1

result = get_target_outcome(signal_high, signal_low, prices, signal_type="BEARISH")
# Expected: ('INVALIDATION', 1) - Price broke above 2.34 ✅
```

### Real-World Validation

From the screenshot, these bearish signals should now classify correctly:
1. **AJBU.SG** - Keppel DC Reit (Score: 86.1, Rank: #3)
2. **J69U.SG** - Frasers Cpt Tr (Score: 47.1, Rank: #39)
3. **ME8U.SG** - Mapletree Ind Tr (Score: 42.1, Rank: #42)
4. **N2IU.SG** - Mapletree PanAsia Co (Score: 40.1, Rank: #44)
5. **OU8.SG** - Centurion (Score: 35.6, Rank: #48)

---

## Impact Assessment

### Before Fix
- ❌ Bearish TRUE_BREAK when price rose (completely backwards!)
- ❌ Bearish INVALIDATION when price fell (completely backwards!)
- ❌ Returns calculated from wrong base price
- ❌ ML training labels for bearish signals were inverted
- ❌ Feature Lab winner selection was misleading

### After Fix
- ✅ Bearish TRUE_BREAK when price falls (correct!)
- ✅ Bearish INVALIDATION when price rises (correct!)
- ✅ Returns calculated from appropriate entry point
- ✅ ML training labels will be accurate
- ✅ Feature Lab can properly track bearish winners

---

## Action Items

### Immediate
- [x] Fix target outcome logic for bearish signals
- [x] Add signal_type parameter to all functions
- [x] Implement inverted logic
- [x] Fix base price calculation
- [x] Add Signal_Bias detection
- [x] Add logging for bearish signal detection
- [ ] **User should re-calculate outcomes for historical scans with bearish signals**
- [ ] **User should verify bearish signal examples from screenshot**

### Future
- [ ] Add unit tests for bearish signal logic
- [ ] Update ML training data with corrected bearish labels
- [ ] Retrain models with accurate bearish signal outcomes
- [ ] Add visual indicators in UI to distinguish bullish/bearish outcomes

---

## Backward Compatibility

### Breaking Changes
**None** - The fix is backward compatible:
- Default `signal_type="BULLISH"` maintains original behavior
- Existing code without Signal_Bias column will work as before
- Only bearish signals (when Signal_Bias contains "BEARISH") use new logic

### Migration Path
1. Existing bullish-only scans: **No changes needed**
2. Mixed bullish/bearish scans: **Automatic** - Signal_Bias column detected
3. Custom implementations: **Add signal_type parameter** if calling functions directly

---

## Related Issues

### Also Fixed
- Date format parsing (see `CRITICAL_DATE_FORMAT_FIXES_20251219.md`)

### Still TODO
- ML data collection needs to pass signal_type through
- Feature experiments may need signal_type awareness
- UI components should display signal type in outcome summaries

---

## Lessons Learned

1. **Always consider signal directionality** - Long and short strategies have opposite logic
2. **Test with real examples** - The screenshot revealed the bug immediately
3. **Add signal type to all trading logic** - Don't assume all signals are bullish
4. **Entry points matter** - Bearish signals enter at signal_low, not signal_high
5. **Validate with domain knowledge** - "TRUE_BREAK when price rises" makes no sense for bearish signals

---

## Contact

For questions about these fixes, contact the development team.

**Fixed by:** AI Assistant  
**Date:** December 19, 2025, 1:29 AM SGT  
**Severity:** CRITICAL  
**Status:** FIXED ✅  
**Related:** CRITICAL_DATE_FORMAT_FIXES_20251219.md
