"""
Statistical Tests for Feature Analysis

Provides statistical testing utilities for evaluating feature significance
in distinguishing winners from non-winners in historical stock selections.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def mann_whitney_test(winners: List[float], non_winners: List[float]) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test to check if winners and non-winners come from different distributions.

    Args:
        winners: List of feature values for winner stocks
        non_winners: List of feature values for non-winner stocks

    Returns:
        Dict with p_value, significant (bool), and statistic
    """
    try:
        # Filter out NaN values
        winners_clean = [x for x in winners if not np.isnan(x)]
        non_winners_clean = [x for x in non_winners if not np.isnan(x)]

        if len(winners_clean) < 3 or len(non_winners_clean) < 3:
            return {
                "p_value": 1.0,
                "significant": False,
                "statistic": 0.0,
                "error": "Insufficient data (need at least 3 values per group)"
            }

        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            winners_clean,
            non_winners_clean,
            alternative='two-sided'
        )

        # Check significance at 5% level
        significant = bool(p_value < 0.05)

        return {
            "p_value": float(p_value),
            "significant": significant,
            "statistic": float(statistic)
        }

    except Exception as e:
        logger.error(f"Error in Mann-Whitney test: {e}")
        return {
            "p_value": 1.0,
            "significant": False,
            "statistic": 0.0,
            "error": str(e)
        }


def cohens_d(winners: List[float], non_winners: List[float]) -> Dict[str, Any]:
    """
    Calculate Cohen's d effect size to measure the magnitude of difference between groups.

    Args:
        winners: List of feature values for winner stocks
        non_winners: List of feature values for non-winner stocks

    Returns:
        Dict with cohens_d, magnitude, and meaningful (bool)
    """
    try:
        # Filter out NaN values
        winners_clean = [x for x in winners if not np.isnan(x)]
        non_winners_clean = [x for x in non_winners if not np.isnan(x)]

        if len(winners_clean) < 2 or len(non_winners_clean) < 2:
            return {
                "cohens_d": 0.0,
                "magnitude": "negligible",
                "meaningful": False,
                "error": "Insufficient data (need at least 2 values per group)"
            }

        # Calculate means and pooled standard deviation
        mean_winners = np.mean(winners_clean)
        mean_non_winners = np.mean(non_winners_clean)

        std_winners = np.std(winners_clean, ddof=1)
        std_non_winners = np.std(non_winners_clean, ddof=1)

        # Pooled standard deviation
        n_winners = len(winners_clean)
        n_non_winners = len(non_winners_clean)

        pooled_std = np.sqrt(
            ((n_winners - 1) * std_winners**2 + (n_non_winners - 1) * std_non_winners**2) /
            (n_winners + n_non_winners - 2)
        )

        # Cohen's d
        if pooled_std == 0:
            cohens_d_value = 0.0
        else:
            cohens_d_value = (mean_winners - mean_non_winners) / pooled_std

        # Determine magnitude
        abs_d = abs(cohens_d_value)
        if abs_d < 0.2:
            magnitude = "negligible"
            meaningful = False
        elif abs_d < 0.5:
            magnitude = "small"
            meaningful = True
        elif abs_d < 0.8:
            magnitude = "medium"
            meaningful = True
        else:
            magnitude = "large"
            meaningful = True

        return {
            "cohens_d": float(cohens_d_value),
            "magnitude": magnitude,
            "meaningful": bool(meaningful)
        }

    except Exception as e:
        logger.error(f"Error calculating Cohen's d: {e}")
        return {
            "cohens_d": 0.0,
            "magnitude": "negligible",
            "meaningful": False,
            "error": str(e)
        }


def correlation_analysis(feature_values: pd.Series, existing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze correlations between the new feature and existing features.

    Args:
        feature_values: Series of the new feature values
        existing_df: DataFrame with existing features

    Returns:
        Dict with correlations, max_correlation, and redundant (bool)
    """
    try:
        # Combine feature with existing data
        analysis_df = existing_df.copy()
        analysis_df['new_feature'] = feature_values

        # Calculate correlations
        correlations = {}
        for col in existing_df.columns:
            if col != 'is_winner':  # Skip the target variable
                try:
                    corr = analysis_df['new_feature'].corr(analysis_df[col])
                    if not np.isnan(corr):
                        correlations[col] = float(corr)
                except:
                    continue

        # Find maximum absolute correlation
        if correlations:
            max_corr_feature = max(correlations.keys(), key=lambda k: abs(correlations[k]))
            max_correlation = correlations[max_corr_feature]
        else:
            max_corr_feature = None
            max_correlation = 0.0

        # Check if redundant (correlation > 0.8)
        redundant = bool(abs(max_correlation) > 0.8)

        return {
            "correlations": correlations,
            "max_correlation": max_correlation,
            "max_correlation_feature": max_corr_feature,
            "redundant": redundant
        }

    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        return {
            "correlations": {},
            "max_correlation": 0.0,
            "max_correlation_feature": None,
            "redundant": False,
            "error": str(e)
        }


def directional_analysis(
    feature_name: str,
    bullish_winners: List[float],
    bullish_non_winners: List[float],
    bearish_winners: List[float],
    bearish_non_winners: List[float]
) -> Dict[str, Any]:
    """
    Separate statistical analysis for bullish vs bearish signals.

    Args:
        feature_name: Name of the feature being analyzed
        bullish_winners: Feature values for bullish winners
        bullish_non_winners: Feature values for bullish non-winners
        bearish_winners: Feature values for bearish winners
        bearish_non_winners: Feature values for bearish non-winners

    Returns:
        Dict with separate analysis for bullish and bearish directions
    """
    try:
        # Analyze bullish direction
        bullish_mann_whitney = mann_whitney_test(bullish_winners, bullish_non_winners)
        bullish_cohens_d = cohens_d(bullish_winners, bullish_non_winners)

        # Analyze bearish direction
        bearish_mann_whitney = mann_whitney_test(bearish_winners, bearish_non_winners)
        bearish_cohens_d = cohens_d(bearish_winners, bearish_non_winners)

        # Check if directions are significantly different
        # (This is a simplified test - could be enhanced with more sophisticated methods)
        bullish_effect = abs(bullish_cohens_d.get('cohens_d', 0))
        bearish_effect = abs(bearish_cohens_d.get('cohens_d', 0))
        directional_difference = abs(bullish_effect - bearish_effect) > 0.3  # Arbitrary threshold

        return {
            "bullish": {
                "mann_whitney_p": bullish_mann_whitney.get('p_value'),
                "mann_whitney_significant": bullish_mann_whitney.get('significant'),
                "cohens_d": bullish_cohens_d.get('cohens_d'),
                "cohens_d_magnitude": bullish_cohens_d.get('magnitude'),
                "meaningful": bullish_cohens_d.get('meaningful'),
                "winner_count": len(bullish_winners),
                "non_winner_count": len(bullish_non_winners)
            },
            "bearish": {
                "mann_whitney_p": bearish_mann_whitney.get('p_value'),
                "mann_whitney_significant": bearish_mann_whitney.get('significant'),
                "cohens_d": bearish_cohens_d.get('cohens_d'),
                "cohens_d_magnitude": bearish_cohens_d.get('magnitude'),
                "meaningful": bearish_cohens_d.get('meaningful'),
                "winner_count": len(bearish_winners),
                "non_winner_count": len(bearish_non_winners)
            },
            "directional_difference": directional_difference,
            "directional_insight": _interpret_directional_difference(
                bullish_effect, bearish_effect, directional_difference
            )
        }

    except Exception as e:
        logger.error(f"Error in directional analysis for {feature_name}: {e}")
        return {
            "bullish": {"error": str(e)},
            "bearish": {"error": str(e)},
            "directional_difference": False,
            "directional_insight": "Analysis failed"
        }


def _interpret_directional_difference(bullish_effect: float, bearish_effect: float, significant_diff: bool) -> str:
    """Interpret the difference between bullish and bearish effects."""
    if not significant_diff:
        return "Similar performance in both directions"

    if bullish_effect > bearish_effect:
        return "Stronger for bullish signals"
    else:
        return "Stronger for bearish signals"


def quintile_analysis(
    feature_values: List[float],
    is_winner_flags: List[bool],
    n_quintiles: int = 5
) -> Dict[str, Any]:
    """
    Divide feature into quintiles and calculate win rate per quintile.

    Args:
        feature_values: List of feature values
        is_winner_flags: Corresponding list of winner flags
        n_quintiles: Number of quintiles to divide into (default 5)

    Returns:
        Dict with quintile analysis results
    """
    try:
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'feature_value': feature_values,
            'is_winner': is_winner_flags
        }).dropna()

        if len(df) < n_quintiles * 2:  # Need at least 2 samples per quintile
            return {
                "error": f"Insufficient data for {n_quintiles} quintiles",
                "sample_size": len(df)
            }

        # Sort by feature value and assign quintiles
        df = df.sort_values('feature_value')
        df['quintile'] = pd.qcut(df['feature_value'], n_quintiles, labels=False, duplicates='drop')

        # Calculate win rate per quintile
        quintile_results = {}
        for q in range(n_quintiles):
            quintile_data = df[df['quintile'] == q]
            win_rate = quintile_data['is_winner'].mean()
            count = len(quintile_data)

            quintile_results[q + 1] = {
                'win_rate': float(win_rate),
                'count': int(count),
                'feature_range': {
                    'min': float(quintile_data['feature_value'].min()),
                    'max': float(quintile_data['feature_value'].max()),
                    'mean': float(quintile_data['feature_value'].mean())
                }
            }

        # Check for monotonic relationship
        win_rates = [quintile_results[q]['win_rate'] for q in range(1, n_quintiles + 1)]
        monotonic_increasing = all(win_rates[i] <= win_rates[i+1] for i in range(len(win_rates)-1))
        monotonic_decreasing = all(win_rates[i] >= win_rates[i+1] for i in range(len(win_rates)-1))
        monotonic = monotonic_increasing or monotonic_decreasing

        # Calculate spread (difference between best and worst quintile)
        spread = max(win_rates) - min(win_rates)

        # Find optimal threshold (best quintile boundary)
        if monotonic_increasing:
            optimal_threshold = quintile_results[1]['feature_range']['max']  # Below this = bad
        elif monotonic_decreasing:
            optimal_threshold = quintile_results[1]['feature_range']['max']  # Above this = bad
        else:
            # Find the quintile with highest win rate
            best_quintile = max(quintile_results.keys(), key=lambda k: quintile_results[k]['win_rate'])
            if best_quintile == 1:
                optimal_threshold = quintile_results[1]['feature_range']['max']
            elif best_quintile == n_quintiles:
                optimal_threshold = quintile_results[n_quintiles]['feature_range']['min']
            else:
                optimal_threshold = None  # No clear threshold

        return {
            "quintiles": quintile_results,
            "monotonic": bool(monotonic),
            "monotonic_increasing": bool(monotonic_increasing),
            "monotonic_decreasing": bool(monotonic_decreasing),
            "spread": float(spread),
            "optimal_threshold": optimal_threshold,
            "interpretation": _interpret_quintile_analysis(quintile_results, monotonic, spread)
        }

    except Exception as e:
        logger.error(f"Error in quintile analysis: {e}")
        return {"error": str(e)}


def _interpret_quintile_analysis(quintile_results: Dict, monotonic: bool, spread: float) -> str:
    """Interpret quintile analysis results."""
    if spread < 0.05:  # Less than 5% spread
        return "No meaningful relationship with win rate"

    if monotonic:
        direction = "increasing" if quintile_results[1]['win_rate'] < quintile_results[5]['win_rate'] else "decreasing"
        return f"Strong monotonic {direction} relationship (spread: {spread:.1%})"
    else:
        return f"Non-monotonic relationship (spread: {spread:.1%}) - check individual quintiles"


def calculate_win_rate_metrics(
    feature_values: List[float],
    is_winner_flags: List[bool]
) -> Dict[str, Any]:
    """
    Calculate precision metrics at various thresholds.

    Args:
        feature_values: List of feature values
        is_winner_flags: Corresponding list of winner flags

    Returns:
        Dict with win rate metrics at different thresholds
    """
    try:
        df = pd.DataFrame({
            'feature_value': feature_values,
            'is_winner': is_winner_flags
        }).dropna().sort_values('feature_value', ascending=False)  # Sort descending for top-N

        total_samples = len(df)
        if total_samples == 0:
            return {"error": "No valid data"}

        # Overall win rate
        overall_win_rate = df['is_winner'].mean()

        # Top 10% precision
        top_10_count = max(1, int(total_samples * 0.1))
        top_10_win_rate = df.head(top_10_count)['is_winner'].mean()

        # Top 25% precision
        top_25_count = max(1, int(total_samples * 0.25))
        top_25_win_rate = df.head(top_25_count)['is_winner'].mean()

        # Top 50% precision
        top_50_count = max(1, int(total_samples * 0.5))
        top_50_win_rate = df.head(top_50_count)['is_winner'].mean()

        return {
            "overall_win_rate": float(overall_win_rate),
            "top_10_pct_precision": float(top_10_win_rate),
            "top_25_pct_precision": float(top_25_win_rate),
            "top_50_pct_precision": float(top_50_win_rate),
            "total_samples": total_samples,
            "top_10_count": top_10_count,
            "top_25_count": top_25_count,
            "top_50_count": top_50_count
        }

    except Exception as e:
        logger.error(f"Error calculating win rate metrics: {e}")
        return {"error": str(e)}


def analyze_feature_complete_enhanced(feature_name: str, selection_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced complete feature analysis with directional and quintile analysis.

    Args:
        feature_name: Name of the feature to analyze
        selection_history: The selection history data

    Returns:
        Complete enhanced analysis results
    """
    try:
        # Extract feature values by direction
        bullish_winners = []
        bullish_non_winners = []
        bearish_winners = []
        bearish_non_winners = []
        all_feature_values = []
        all_winner_flags = []
        existing_features = []

        # Process each date in history
        for date_str, date_data in selection_history.get('dates', {}).items():
            scan_results = date_data.get('scan_results', {})

            for ticker, features in scan_results.items():
                if feature_name in features:
                    value = features[feature_name]
                    signal_bias = features.get('Signal_Bias', 'NEUTRAL')
                    is_winner = features.get('is_winner', False)

                    # Skip NaN values
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        all_feature_values.append(value)
                        all_winner_flags.append(is_winner)

                        # Categorize by direction (handle emoji-prefixed signal bias)
                        signal_bias_str = str(signal_bias).upper()
                        if 'BULLISH' in signal_bias_str:
                            if is_winner:
                                bullish_winners.append(value)
                            else:
                                bullish_non_winners.append(value)
                        elif 'BEARISH' in signal_bias_str:
                            if is_winner:
                                bearish_winners.append(value)
                            else:
                                bearish_non_winners.append(value)

                        # Collect existing features for correlation analysis
                        existing_row = {}
                        for key, val in features.items():
                            if key not in ['is_winner', feature_name, 'Signal_Bias'] and isinstance(val, (int, float)):
                                existing_row[key] = val
                        if existing_row:
                            existing_features.append(existing_row)

        # Check minimum data requirements
        total_winners = len(bullish_winners) + len(bearish_winners)
        total_non_winners = len(bullish_non_winners) + len(bearish_non_winners)

        if total_winners < 3 or total_non_winners < 3:
            return {
                "feature_name": feature_name,
                "error": "Insufficient data for analysis",
                "total_winners": total_winners,
                "total_non_winners": total_non_winners,
                "bullish_winners": len(bullish_winners),
                "bearish_winners": len(bearish_winners)
            }

        # Run enhanced statistical tests
        # 1. Overall analysis (combined directions)
        all_winners = bullish_winners + bearish_winners
        all_non_winners = bullish_non_winners + bearish_non_winners

        overall_mann_whitney = mann_whitney_test(all_winners, all_non_winners)
        overall_cohens_d = cohens_d(all_winners, all_non_winners)

        # 2. Directional analysis
        directional_results = directional_analysis(
            feature_name, bullish_winners, bullish_non_winners,
            bearish_winners, bearish_non_winners
        )

        # 3. Quintile analysis
        quintile_results = quintile_analysis(all_feature_values, all_winner_flags)

        # 4. Win rate metrics
        win_rate_results = calculate_win_rate_metrics(all_feature_values, all_winner_flags)

        # 5. Correlation analysis
        correlation_result = {}
        if existing_features:
            existing_df = pd.DataFrame(existing_features)
            feature_series = pd.Series(all_feature_values)
            correlation_result = correlation_analysis(feature_series, existing_df)

        # Calculate basic statistics
        winner_mean = float(np.mean(all_winners))
        non_winner_mean = float(np.mean(all_non_winners))

        # Enhanced recommendation logic
        recommendation_data = _generate_enhanced_recommendation(
            overall_mann_whitney, overall_cohens_d, directional_results,
            quintile_results, correlation_result
        )

        return {
            "feature_name": feature_name,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),

            # Sample counts
            "total_winners": total_winners,
            "total_non_winners": total_non_winners,
            "bullish_winners": len(bullish_winners),
            "bullish_non_winners": len(bullish_non_winners),
            "bearish_winners": len(bearish_winners),
            "bearish_non_winners": len(bearish_non_winners),

            # Overall statistics
            "winner_mean": winner_mean,
            "non_winner_mean": non_winner_mean,

            # Overall statistical tests
            "overall_mann_whitney_p": overall_mann_whitney.get('p_value'),
            "overall_mann_whitney_significant": overall_mann_whitney.get('significant'),
            "overall_cohens_d": overall_cohens_d.get('cohens_d'),
            "overall_cohens_d_magnitude": overall_cohens_d.get('magnitude'),

            # Directional analysis
            "directional_analysis": directional_results,

            # Quintile analysis
            "quintile_analysis": quintile_results,

            # Win rate metrics
            "win_rate_metrics": win_rate_results,

            # Correlation analysis
            "correlation_max": correlation_result.get('max_correlation'),
            "correlation_redundant": correlation_result.get('redundant'),

            # Recommendation
            "recommendation": recommendation_data["recommendation"],
            "recommendation_text": recommendation_data["recommendation_text"],
            "recommendation_reasoning": recommendation_data["reasoning"],

            # Raw data for plotting
            "all_feature_values": all_feature_values,
            "all_winner_flags": all_winner_flags,
            "bullish_winners_values": bullish_winners,
            "bullish_non_winners_values": bullish_non_winners,
            "bearish_winners_values": bearish_winners,
            "bearish_non_winners_values": bearish_non_winners
        }

    except Exception as e:
        logger.error(f"Error in enhanced feature analysis: {e}")
        return {
            "feature_name": feature_name,
            "error": str(e)
        }


def _generate_enhanced_recommendation(
    overall_mw, overall_cd, directional, quintile, correlation
) -> Dict[str, Any]:
    """Generate enhanced recommendation based on all analysis results."""

    # Extract key metrics
    overall_significant = overall_mw.get('significant', False)
    overall_meaningful = overall_cd.get('meaningful', False)
    redundant = correlation.get('redundant', False)

    # Directional insights
    bullish_significant = directional.get('bullish', {}).get('mann_whitney_significant', False)
    bearish_significant = directional.get('bearish', {}).get('mann_whitney_significant', False)
    directional_diff = directional.get('directional_difference', False)

    # Quintile insights
    monotonic = quintile.get('monotonic', False)
    spread = quintile.get('spread', 0)

    reasoning_parts = []

    # Primary criteria
    if not overall_significant:
        recommendation = "REMOVE"
        reasoning_parts.append("Not statistically significant overall")
    elif redundant:
        recommendation = "REMOVE"
        reasoning_parts.append("Highly correlated with existing features")
    elif monotonic and spread > 0.05:  # Strong monotonic relationship
        if directional_diff:
            if bullish_significant and not bearish_significant:
                recommendation = "KEEP"
                reasoning_parts.append("Strong for bullish signals, monotonic quintiles")
            elif bearish_significant and not bullish_significant:
                recommendation = "KEEP"
                reasoning_parts.append("Strong for bearish signals, monotonic quintiles")
            else:
                recommendation = "REVIEW"
                reasoning_parts.append("Directional differences need investigation")
        else:
            recommendation = "KEEP"
            reasoning_parts.append("Strong overall performance with monotonic quintiles")
    elif spread > 0.03:  # Moderate relationship
        recommendation = "REVIEW"
        reasoning_parts.append("Moderate predictive power, consider weighting")
    else:
        recommendation = "REMOVE"
        reasoning_parts.append("Weak or no relationship with outcomes")

    # Add directional insights
    if directional_diff:
        if bullish_significant and bearish_significant:
            reasoning_parts.append("Works in both directions")
        elif bullish_significant:
            reasoning_parts.append("Primarily bullish signal")
        elif bearish_significant:
            reasoning_parts.append("Primarily bearish signal")

    reasoning = "; ".join(reasoning_parts)

    # Convert to display format
    recommendation_text_map = {
        "KEEP": "✅ KEEP - Strong predictive power",
        "REVIEW": "🤔 REVIEW - Moderate performance, investigate further",
        "REMOVE": "❌ REMOVE - Weak or redundant"
    }

    return {
        "recommendation": recommendation,
        "recommendation_text": recommendation_text_map.get(recommendation, "❓ UNKNOWN"),
        "reasoning": reasoning
    }


def analyze_feature_complete(feature_name: str, selection_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete feature analysis combining all statistical tests.
    LEGACY FUNCTION - Use analyze_feature_complete_enhanced instead.

    Args:
        feature_name: Name of the feature to analyze
        selection_history: The selection history data

    Returns:
        Complete analysis results
    """
    logger.warning("Using legacy analyze_feature_complete. Consider using analyze_feature_complete_enhanced for better analysis.")
    return analyze_feature_complete_enhanced(feature_name, selection_history)
