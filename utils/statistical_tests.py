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


def analyze_feature_complete(feature_name: str, selection_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete feature analysis combining all statistical tests.

    Args:
        feature_name: Name of the feature to analyze
        selection_history: The selection history data

    Returns:
        Complete analysis results
    """
    try:
        # Extract feature values for winners and non-winners
        winners_values = []
        non_winners_values = []
        all_feature_values = []
        existing_features = []

        # Process each date in history
        for date_str, date_data in selection_history.get('dates', {}).items():
            scan_results = date_data.get('scan_results', {})

            for ticker, features in scan_results.items():
                if feature_name in features:
                    value = features[feature_name]

                    # Skip NaN values
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        all_feature_values.append(value)

                        if features.get('is_winner', False):
                            winners_values.append(value)
                        else:
                            non_winners_values.append(value)

                        # Collect existing features for correlation analysis
                        existing_row = {}
                        for key, val in features.items():
                            if key not in ['is_winner', feature_name] and isinstance(val, (int, float)):
                                existing_row[key] = val
                        if existing_row:
                            existing_features.append(existing_row)

        if len(winners_values) < 3 or len(non_winners_values) < 3:
            return {
                "feature_name": feature_name,
                "error": "Insufficient data for analysis",
                "winner_count": len(winners_values),
                "non_winner_count": len(non_winners_values)
            }

        # Run statistical tests
        mann_whitney = mann_whitney_test(winners_values, non_winners_values)
        cohens_d_result = cohens_d(winners_values, non_winners_values)

        # Correlation analysis
        correlation_result = {}
        if existing_features:
            existing_df = pd.DataFrame(existing_features)
            feature_series = pd.Series(all_feature_values)
            correlation_result = correlation_analysis(feature_series, existing_df)

        # Calculate basic statistics
        winner_mean = float(np.mean(winners_values))
        non_winner_mean = float(np.mean(non_winners_values))

        # Determine overall recommendation
        significant = bool(mann_whitney.get('significant', False))
        meaningful = bool(cohens_d_result.get('meaningful', False))
        redundant = bool(correlation_result.get('redundant', False))

        if significant and meaningful and not redundant:
            recommendation = "STRONG_CANDIDATE"
            recommendation_text = "✅ STRONG CANDIDATE - Add to Scoring"
        elif not significant:
            recommendation = "NOT_SIGNIFICANT"
            recommendation_text = "❌ NOT SIGNIFICANT - Discard"
        elif redundant:
            recommendation = "REDUNDANT"
            recommendation_text = "⚠️ REDUNDANT - Already captured"
        else:
            recommendation = "WEAK_CANDIDATE"
            recommendation_text = "🤔 WEAK CANDIDATE - Consider with caution"

        return {
            "feature_name": feature_name,
            "winner_count": len(winners_values),
            "non_winner_count": len(non_winners_values),
            "winner_mean": winner_mean,
            "non_winner_mean": non_winner_mean,
            "mann_whitney_p": mann_whitney.get('p_value'),
            "mann_whitney_significant": significant,
            "cohens_d": cohens_d_result.get('cohens_d'),
            "cohens_d_magnitude": cohens_d_result.get('magnitude'),
            "correlation_max": correlation_result.get('max_correlation'),
            "correlation_redundant": redundant,
            "recommendation": recommendation,
            "recommendation_text": recommendation_text,
            "winners_values": winners_values,
            "non_winners_values": non_winners_values
        }

    except Exception as e:
        logger.error(f"Error in complete feature analysis: {e}")
        return {
            "feature_name": feature_name,
            "error": str(e)
        }
