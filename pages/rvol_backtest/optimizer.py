# Optimizer - Parameter Sweep with Fill Rate Tracking
"""
Parameter optimization engine for RVOL BackTest
Tests multiple deviation thresholds and identifies optimal parameters
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .vwap_engine import calculate_monthly_vwap
from .backtest_engine import backtest_strategy_with_retest

logger = logging.getLogger(__name__)

def run_parameter_optimization(
    df: pd.DataFrame,
    threshold_min: float = 0.005,  # 0.5%
    threshold_max: float = 0.020,  # 2.0%
    threshold_step: float = 0.001, # 0.1% steps
    stop_loss: float = 0.012,      # 1.2%
    position_size: int = 50000,
    order_expiry_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Run parameter sweep across deviation thresholds

    Args:
        df: DataFrame with OHLCV data
        threshold_min: Minimum deviation threshold (decimal)
        threshold_max: Maximum deviation threshold (decimal)
        threshold_step: Step size for threshold sweep (decimal)
        stop_loss: Stop loss percentage (decimal)
        position_size: Shares per trade
        order_expiry_days: Days before orders expire (None = no expiry)

    Returns:
        DataFrame with optimization results sorted by Sharpe ratio
    """
    # Calculate Monthly VWAP once for all tests
    df_with_vwap = calculate_monthly_vwap(df.copy())

    # Generate threshold range
    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)

    logger.info(f"Starting optimization: Testing {len(thresholds)} thresholds from {threshold_min:.1%} to {threshold_max:.1%}")

    optimization_results = []

    for i, threshold in enumerate(thresholds):
        try:
            # Run backtest for this threshold
            trades, metrics, fill_stats = backtest_strategy_with_retest(
                df=df_with_vwap,
                deviation_threshold=threshold,
                stop_loss=stop_loss,
                position_size=position_size,
                order_expiry_days=order_expiry_days
            )

            # Compile results for this threshold
            result = {
                'threshold_pct': threshold * 100,  # Convert to percentage for display
                'threshold_decimal': threshold,    # Keep decimal for internal use

                # Signal & Fill Metrics
                'signals_generated': fill_stats['signals_generated'],
                'orders_filled': fill_stats['orders_filled'],
                'fill_rate_pct': fill_stats['fill_rate'],
                'orders_expired': fill_stats['orders_expired'],
                'avg_days_to_fill': fill_stats['avg_days_to_fill'],

                # Performance Metrics
                'total_trades': metrics['total_trades'],
                'winning_trades': metrics['winning_trades'],
                'losing_trades': metrics['losing_trades'],
                'win_rate_pct': metrics['win_rate'],
                'total_pnl_dollars': metrics['total_pnl'],
                'avg_trade_pnl': metrics['avg_trade_pnl'],
                'best_trade': metrics['best_trade'],
                'worst_trade': metrics['worst_trade'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown': metrics['max_drawdown'],
                'avg_hold_days': metrics['avg_hold_days'],
                'target_exits': metrics['target_exits'],
                'stop_losses': metrics['stop_losses'],

                # Optimization timestamp
                'optimization_date': datetime.now()
            }

            optimization_results.append(result)

            logger.debug(f"Threshold {threshold:.1%}: {metrics['total_trades']} trades, "
                        f"{fill_stats['fill_rate']:.1f}% fill rate, Sharpe={metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            logger.error(f"Error testing threshold {threshold:.1%}: {e}")
            # Add error result
            result = {
                'threshold_pct': threshold * 100,
                'threshold_decimal': threshold,
                'error': str(e),
                'total_trades': 0,
                'sharpe_ratio': -999,  # Sort errors to bottom
                'optimization_date': datetime.now()
            }
            optimization_results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(optimization_results)

    # Sort by Sharpe ratio (descending) to find optimal threshold
    if not results_df.empty and 'sharpe_ratio' in results_df.columns:
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    logger.info(f"Optimization completed: Tested {len(thresholds)} thresholds, "
               f"best Sharpe ratio: {results_df.iloc[0]['sharpe_ratio']:.2f} at "
               f"{results_df.iloc[0]['threshold_pct']:.1f}% threshold")

    return results_df

def get_optimal_parameters(results_df: pd.DataFrame) -> Dict:
    """
    Extract optimal parameters from optimization results

    Args:
        results_df: DataFrame from run_parameter_optimization

    Returns:
        dict: Optimal parameter recommendations
    """
    if results_df.empty:
        return {'error': 'No optimization results available'}

    # Get best result (highest Sharpe ratio)
    best_result = results_df.iloc[0]

    # Check for realistic fill rates (not over-optimized)
    fill_rate = best_result.get('fill_rate_pct', 0)

    recommendations = {
        'optimal_threshold_pct': best_result['threshold_pct'],
        'optimal_threshold_decimal': best_result['threshold_decimal'],
        'expected_sharpe_ratio': best_result['sharpe_ratio'],
        'expected_win_rate_pct': best_result['win_rate_pct'],
        'expected_fill_rate_pct': fill_rate,
        'expected_total_trades': best_result['total_trades'],
        'expected_total_pnl': best_result['total_pnl_dollars'],
        'confidence_level': _calculate_confidence_level(best_result),
        'risk_warnings': _generate_risk_warnings(best_result)
    }

    return recommendations

def _calculate_confidence_level(result: pd.Series) -> str:
    """
    Calculate confidence level in the optimization result

    Args:
        result: Single row from optimization results

    Returns:
        str: Confidence level assessment
    """
    confidence_score = 0
    reasons = []

    # Sample size (more trades = higher confidence)
    total_trades = result.get('total_trades', 0)
    if total_trades >= 50:
        confidence_score += 3
        reasons.append("Large sample size")
    elif total_trades >= 20:
        confidence_score += 2
        reasons.append("Adequate sample size")
    elif total_trades >= 10:
        confidence_score += 1
        reasons.append("Small sample size")
    else:
        reasons.append("Very small sample size")

    # Fill rate (realistic fill rates = higher confidence)
    fill_rate = result.get('fill_rate_pct', 0)
    if 30 <= fill_rate <= 80:
        confidence_score += 2
        reasons.append("Realistic fill rate")
    elif fill_rate > 80:
        confidence_score += 1
        reasons.append("High fill rate (may be optimistic)")
    else:
        reasons.append("Low fill rate (may be too restrictive)")

    # Sharpe ratio stability
    sharpe = result.get('sharpe_ratio', 0)
    if sharpe > 1.0:
        confidence_score += 2
        reasons.append("Strong risk-adjusted returns")
    elif sharpe > 0.5:
        confidence_score += 1
        reasons.append("Moderate risk-adjusted returns")

    # Convert score to confidence level
    if confidence_score >= 5:
        return "HIGH"
    elif confidence_score >= 3:
        return "MODERATE"
    else:
        return "LOW"

def _generate_risk_warnings(result: pd.Series) -> List[str]:
    """
    Generate risk warnings based on optimization results

    Args:
        result: Single row from optimization results

    Returns:
        list: Risk warning messages
    """
    warnings = []

    # Low fill rate warning
    fill_rate = result.get('fill_rate_pct', 0)
    if fill_rate < 20:
        warnings.append(f"⚠️ Low fill rate ({fill_rate:.1f}%) - may be difficult to execute in live trading")

    # High fill rate warning (potential over-optimization)
    if fill_rate > 90:
        warnings.append(f"⚠️ Very high fill rate ({fill_rate:.1f}%) - results may be over-optimized")

    # Low sample size warning
    total_trades = result.get('total_trades', 0)
    if total_trades < 20:
        warnings.append(f"⚠️ Small sample size ({total_trades} trades) - results may not be statistically significant")

    # Negative Sharpe warning
    sharpe = result.get('sharpe_ratio', 0)
    if sharpe < 0:
        warnings.append(f"⚠️ Negative Sharpe ratio ({sharpe:.2f}) - strategy shows negative risk-adjusted returns")

    # High drawdown warning
    max_dd = result.get('max_drawdown', 0)
    if max_dd > 5000:  # $5,000 drawdown
        warnings.append(f"⚠️ High maximum drawdown (${max_dd:,.0f}) - consider position sizing")

    return warnings

def compare_thresholds(results_df: pd.DataFrame) -> Dict:
    """
    Compare different threshold ranges for robustness analysis

    Args:
        results_df: DataFrame from run_parameter_optimization

    Returns:
        dict: Comparison analysis
    """
    if results_df.empty:
        return {'error': 'No results to compare'}

    # Group by threshold ranges
    tight_thresholds = results_df[results_df['threshold_pct'] <= 1.0]  # 0.5% - 1.0%
    medium_thresholds = results_df[(results_df['threshold_pct'] > 1.0) & (results_df['threshold_pct'] <= 1.5)]  # 1.1% - 1.5%
    loose_thresholds = results_df[results_df['threshold_pct'] > 1.5]   # 1.6% - 2.0%

    comparison = {
        'tight_thresholds': {
            'count': len(tight_thresholds),
            'avg_sharpe': tight_thresholds['sharpe_ratio'].mean(),
            'avg_fill_rate': tight_thresholds['fill_rate_pct'].mean(),
            'best_sharpe': tight_thresholds['sharpe_ratio'].max()
        },
        'medium_thresholds': {
            'count': len(medium_thresholds),
            'avg_sharpe': medium_thresholds['sharpe_ratio'].mean(),
            'avg_fill_rate': medium_thresholds['fill_rate_pct'].mean(),
            'best_sharpe': medium_thresholds['sharpe_ratio'].max()
        },
        'loose_thresholds': {
            'count': len(loose_thresholds),
            'avg_sharpe': loose_thresholds['sharpe_ratio'].mean(),
            'avg_fill_rate': loose_thresholds['fill_rate_pct'].mean(),
            'best_sharpe': loose_thresholds['sharpe_ratio'].max()
        }
    }

    return comparison

def validate_optimization_results(results_df: pd.DataFrame) -> Dict:
    """
    Validate optimization results for consistency

    Args:
        results_df: DataFrame from run_parameter_optimization

    Returns:
        dict: Validation results
    """
    validation = {
        'total_thresholds_tested': len(results_df),
        'successful_tests': len(results_df[results_df.get('error').isna()]),
        'failed_tests': len(results_df[results_df.get('error').notna()]),
        'issues': []
    }

    if validation['failed_tests'] > 0:
        validation['issues'].append(f"❌ {validation['failed_tests']} threshold tests failed")

    # Check for monotonic relationships (shouldn't be perfectly monotonic)
    if len(results_df) > 5:
        sharpe_values = results_df['sharpe_ratio'].head(10)  # Check top 10
        if sharpe_values.is_monotonic_decreasing or sharpe_values.is_monotonic_increasing:
            validation['issues'].append("⚠️ Sharpe ratios show monotonic pattern - may indicate data issues")

    # Check fill rate distribution
    fill_rates = results_df['fill_rate_pct'].dropna()
    if len(fill_rates) > 0:
        if fill_rates.max() - fill_rates.min() > 80:
            validation['issues'].append("⚠️ Large variation in fill rates - thresholds may have very different tradeability")

    validation['is_valid'] = len(validation['issues']) == 0

    return validation
