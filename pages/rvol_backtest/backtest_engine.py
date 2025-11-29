# Backtest Engine - Realistic Retest Fill Logic
"""
Backtesting engine with state machine for realistic order fills
Implements pending orders that fill on price retest
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BacktestState:
    """State machine for backtesting with pending orders"""

    NO_POSITION = "NO_POSITION"
    PENDING_ORDER = "PENDING_ORDER"
    IN_POSITION = "IN_POSITION"

def backtest_strategy_with_retest(
    df: pd.DataFrame,
    deviation_threshold: float,
    stop_loss: float = 0.012,
    position_size: int = 50000,
    order_expiry_days: Optional[int] = None
) -> Tuple[List[Dict], Dict, Dict]:
    """
    Run Monthly VWAP Mean Reversion backtest with REALISTIC RETEST FILL LOGIC

    KEY INNOVATION: Unlike most backtests that assume 100% fill rates, this system
    implements realistic order fills that only execute when price RETESTS the entry level.

    Strategy Rules:
    - ENTRY: Place limit order at signal day's Low when price drops X% below Monthly VWAP
    - FILL: Order fills only when subsequent price action retests the limit level
    - EXIT: Target at VWAP (mean reversion complete) OR 1.2% stop loss

    State Machine Logic:
    1. NO_POSITION: Scan for signals (Low < VWAP - threshold)
    2. PENDING_ORDER: Limit order placed at signal Low, waiting for retest
    3. IN_POSITION: Order filled, monitor VWAP target and stop loss

    Why This Matters:
    - Prevents over-optimization from unrealistic fill assumptions
    - Fill rate becomes a key metric (50-70% is realistic)
    - Only tradeable setups get included in results
    - Better represents actual trading conditions

    Args:
        df: DataFrame with OHLCV data and Monthly_VWAP column
        deviation_threshold: Entry threshold below VWAP (decimal, e.g., 0.05 = 5%)
        stop_loss: Stop loss percentage below entry (decimal, e.g., 0.012 = 1.2%)
        position_size: Shares per trade (integer)
        order_expiry_days: Days before unfilled orders expire (None = no expiry)

    Returns:
        Tuple of (trades, metrics, fill_stats)
        - trades: List of completed trade dictionaries with full trade details
        - metrics: Performance metrics (win rate, Sharpe, P&L, etc.)
        - fill_stats: Fill rate statistics (signals vs fills, avg days to fill)

    Example:
        # Test 5% deviation threshold
        trades, metrics, fill_stats = backtest_strategy_with_retest(
            df=df_with_vwap,
            deviation_threshold=0.05,
            stop_loss=0.012,
            position_size=50000
        )

        print(f"Fill Rate: {fill_stats['fill_rate']:.1f}%")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    """
    if 'Monthly_VWAP' not in df.columns:
        raise ValueError("DataFrame must contain 'Monthly_VWAP' column")

    # Initialize state machine
    state = BacktestState.NO_POSITION
    pending_order = None
    trades = []
    signals_generated = 0

    # Process each day chronologically
    for current_date, row in df.iterrows():
        # Extract current day's data
        current_low = row['Low']
        current_high = row['High']
        current_close = row['Close']
        monthly_vwap = row['Monthly_VWAP']

        # STATE 1: NO_POSITION - Scan for entry signals
        if state == BacktestState.NO_POSITION:
            # Check entry condition: Low below VWAP by threshold
            if current_low <= monthly_vwap * (1 - deviation_threshold):
                # Signal generated!
                signals_generated += 1

                # Place limit order at signal day's low
                pending_order = {
                    'limit_price': current_low,
                    'signal_date': current_date,
                    'vwap_at_signal': monthly_vwap,
                    'days_waiting': 0
                }

                state = BacktestState.PENDING_ORDER
                logger.debug(f"Signal generated on {current_date}: Low={current_low:.4f}, VWAP={monthly_vwap:.4f}")

        # STATE 2: PENDING_ORDER - Wait for retest fill
        elif state == BacktestState.PENDING_ORDER:
            # Increment days waiting
            pending_order['days_waiting'] += 1

            # Check if order fills (price retests limit level)
            if current_low <= pending_order['limit_price']:
                # ORDER FILLED! Enter position
                entry_price = pending_order['limit_price']
                entry_date = current_date
                entry_vwap = pending_order['vwap_at_signal']

                state = BacktestState.IN_POSITION
                pending_order = None  # Clear pending order

                logger.debug(f"Order filled on {current_date}: Entry={entry_price:.4f}")

            # Check for order expiry (if enabled)
            elif order_expiry_days and pending_order['days_waiting'] >= order_expiry_days:
                # Order expired, cancel
                logger.debug(f"Order expired on {current_date} after {pending_order['days_waiting']} days")
                state = BacktestState.NO_POSITION
                pending_order = None

        # STATE 3: IN_POSITION - Monitor exits
        elif state == BacktestState.IN_POSITION:
            # Check exit conditions (priority order)

            # Exit 1: Target hit (mean reversion complete)
            if current_high >= entry_vwap:
                exit_price = entry_vwap
                exit_date = current_date
                exit_reason = "VWAP_Target"
                pnl_dollars = (exit_price - entry_price) * position_size
                pnl_percent = (exit_price - entry_price) / entry_price

                # Record completed trade
                trade = {
                    'signal_date': pending_order['signal_date'] if pending_order else entry_date,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_vwap': entry_vwap,
                    'hold_days': (exit_date - entry_date).days,
                    'pnl_dollars': pnl_dollars,
                    'pnl_percent': pnl_percent,
                    'exit_reason': exit_reason,
                    'position_size': position_size,
                    'deviation_threshold': deviation_threshold
                }
                trades.append(trade)

                state = BacktestState.NO_POSITION
                logger.debug(f"Target exit on {current_date}: P&L={pnl_dollars:.2f}")

            # Exit 2: Stop loss triggered
            elif current_low <= entry_price * (1 - stop_loss):
                exit_price = entry_price * (1 - stop_loss)
                exit_date = current_date
                exit_reason = "Stop_Loss"
                pnl_dollars = (exit_price - entry_price) * position_size
                pnl_percent = (exit_price - entry_price) / entry_price

                # Record completed trade
                trade = {
                    'signal_date': pending_order['signal_date'] if pending_order else entry_date,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_vwap': entry_vwap,
                    'hold_days': (exit_date - entry_date).days,
                    'pnl_dollars': pnl_dollars,
                    'pnl_percent': pnl_percent,
                    'exit_reason': exit_reason,
                    'position_size': position_size,
                    'deviation_threshold': deviation_threshold
                }
                trades.append(trade)

                state = BacktestState.NO_POSITION
                logger.debug(f"Stop loss exit on {current_date}: P&L={pnl_dollars:.2f}")

    # Handle any remaining pending order at end of data
    if state == BacktestState.PENDING_ORDER and pending_order:
        logger.debug(f"Pending order unfilled at end of data: {pending_order['days_waiting']} days waiting")

    # Calculate fill statistics
    fill_stats = {
        'signals_generated': signals_generated,
        'orders_filled': len(trades),
        'fill_rate': (len(trades) / signals_generated * 100) if signals_generated > 0 else 0,
        'orders_expired': signals_generated - len(trades),
        'avg_days_to_fill': calculate_avg_days_to_fill(trades) if trades else 0
    }

    # Calculate performance metrics
    metrics = calculate_performance_metrics(trades)

    logger.info(f"Backtest completed: {len(trades)} trades from {signals_generated} signals ({fill_stats['fill_rate']:.1f}% fill rate)")

    return trades, metrics, fill_stats

def calculate_avg_days_to_fill(trades: List[Dict]) -> float:
    """Calculate average days between signal and fill"""
    if not trades:
        return 0

    days_to_fill = []
    for trade in trades:
        if 'signal_date' in trade and 'entry_date' in trade:
            days = (trade['entry_date'] - trade['signal_date']).days
            days_to_fill.append(days)

    return np.mean(days_to_fill) if days_to_fill else 0

def calculate_performance_metrics(trades: List[Dict]) -> Dict:
    """
    Calculate comprehensive performance metrics

    Args:
        trades: List of completed trade dictionaries

    Returns:
        dict: Performance metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_trade_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'avg_hold_days': 0,
            'target_exits': 0,
            'stop_losses': 0
        }

    # Basic trade counts
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl_dollars'] > 0)
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # P&L metrics
    pnl_values = [t['pnl_dollars'] for t in trades]
    total_pnl = sum(pnl_values)
    avg_trade_pnl = total_pnl / total_trades
    best_trade = max(pnl_values) if pnl_values else 0
    worst_trade = min(pnl_values) if pnl_values else 0

    # Risk-adjusted metrics
    if len(pnl_values) > 1:
        returns = np.array(pnl_values)
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # Profit factor
        gross_profit = sum(p for p in pnl_values if p > 0)
        gross_loss = abs(sum(p for p in pnl_values if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        sharpe_ratio = 0
        profit_factor = 0

    # Max drawdown calculation
    cumulative_pnl = np.cumsum(pnl_values)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

    # Holding period
    hold_days = [t['hold_days'] for t in trades]
    avg_hold_days = np.mean(hold_days) if hold_days else 0

    # Exit analysis
    target_exits = sum(1 for t in trades if t['exit_reason'] == 'VWAP_Target')
    stop_losses = sum(1 for t in trades if t['exit_reason'] == 'Stop_Loss')

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_trade_pnl': avg_trade_pnl,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_hold_days': avg_hold_days,
        'target_exits': target_exits,
        'stop_losses': stop_losses
    }

def validate_backtest_results(trades: List[Dict], fill_stats: Dict) -> Dict:
    """
    Validate backtest results for consistency and realism

    Args:
        trades: List of completed trades
        fill_stats: Fill rate statistics

    Returns:
        dict: Validation results
    """
    validation = {
        'total_trades': len(trades),
        'fill_rate': fill_stats['fill_rate'],
        'avg_days_to_fill': fill_stats['avg_days_to_fill'],
        'issues': []
    }

    # Check for unrealistic results
    if fill_stats['fill_rate'] > 95:
        validation['issues'].append("⚠️ Fill rate >95% - may indicate over-optimization")

    if fill_stats['avg_days_to_fill'] > 10:
        validation['issues'].append("⚠️ Average fill time >10 days - orders may be too tight")

    if fill_stats['fill_rate'] < 10:
        validation['issues'].append("⚠️ Fill rate <10% - threshold may be too restrictive")

    # Check trade logic
    for i, trade in enumerate(trades):
        # Verify entry logic
        if trade['entry_price'] > trade['entry_vwap'] * (1 - trade['deviation_threshold']):
            validation['issues'].append(f"❌ Trade {i}: Entry price above threshold")

        # Verify exit logic
        if trade['exit_reason'] == 'VWAP_Target' and trade['exit_price'] != trade['entry_vwap']:
            validation['issues'].append(f"❌ Trade {i}: Target exit not at VWAP")

        if trade['exit_reason'] == 'Stop_Loss':
            expected_stop = trade['entry_price'] * (1 - 0.012)
            if abs(trade['exit_price'] - expected_stop) > 0.01:
                validation['issues'].append(f"❌ Trade {i}: Stop loss price incorrect")

    validation['is_valid'] = len(validation['issues']) == 0

    return validation
