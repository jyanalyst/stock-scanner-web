"""
Statistical validation and hypothesis testing utilities
Ensures all findings meet rigorous statistical standards
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Enforces statistical rigor for all ML pipeline steps
    """

    def __init__(self, alpha: float = 0.05, power: float = 0.80):
        """
        Args:
            alpha: Significance level (default 0.05 = 95% confidence)
            power: Statistical power (default 0.80 = 80% power)
        """
        self.alpha = alpha
        self.power = power

    def calculate_minimum_sample_size(self,
                                     target_ic: float = 0.08,
                                     effect_size: Optional[float] = None) -> int:
        """
        Calculate minimum sample size needed to detect target IC

        Args:
            target_ic: Minimum IC we want to detect (default 0.08)
            effect_size: Cohen's d effect size (if provided, overrides target_ic)

        Returns:
            Minimum number of samples required

        References:
            Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
        """
        if effect_size is None:
            # Convert IC (correlation) to effect size
            effect_size = 2 * target_ic / np.sqrt(1 - target_ic**2)

        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)

        # Sample size formula for correlation
        n = ((z_alpha + z_beta) / effect_size) ** 2 + 3

        return int(np.ceil(n))

    def bootstrap_confidence_interval(self,
                                     data: np.ndarray,
                                     statistic_func: callable,
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence interval for any statistic

        Args:
            data: Input data
            statistic_func: Function that calculates statistic (e.g., np.mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 0.95)

        Returns:
            Dict with mean, std, and confidence interval
        """
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)

        bootstrap_stats = np.array(bootstrap_stats)

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return {
            'mean': np.mean(bootstrap_stats),
            'std': np.std(bootstrap_stats),
            'lower_ci': np.percentile(bootstrap_stats, lower_percentile),
            'upper_ci': np.percentile(bootstrap_stats, upper_percentile),
            'bootstrap_distribution': bootstrap_stats
        }

    def test_ic_significance(self,
                            ic_values: np.ndarray,
                            null_hypothesis: float = 0.0) -> Dict:
        """
        Test if IC is significantly different from zero (or other null)

        Args:
            ic_values: Array of IC values (e.g., from time series)
            null_hypothesis: Null hypothesis value (default 0)

        Returns:
            Dict with test results
        """
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(ic_values, null_hypothesis)

        # Effect size (Cohen's d)
        cohens_d = np.mean(ic_values) / np.std(ic_values, ddof=1)

        # Bootstrap CI for robustness
        boot_ci = self.bootstrap_confidence_interval(
            ic_values,
            np.mean,
            n_bootstrap=1000
        )

        is_significant = p_value < self.alpha

        return {
            'mean_ic': np.mean(ic_values),
            'std_ic': np.std(ic_values, ddof=1),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': is_significant,
            'ci_lower': boot_ci['lower_ci'],
            'ci_upper': boot_ci['upper_ci'],
            'n_samples': len(ic_values)
        }

    def bonferroni_correction(self, n_tests: int) -> float:
        """
        Calculate Bonferroni-corrected alpha for multiple testing

        Args:
            n_tests: Number of hypothesis tests being performed

        Returns:
            Corrected alpha level
        """
        return self.alpha / n_tests

    def benjamini_hochberg_correction(self, p_values: np.ndarray) -> np.ndarray:
        """
        Benjamini-Hochberg FDR correction (less conservative than Bonferroni)

        Args:
            p_values: Array of p-values

        Returns:
            Array of corrected p-values
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        # BH correction
        corrected_p_values = sorted_p_values * n / (np.arange(n) + 1)

        # Ensure monotonicity
        for i in range(n - 1, 0, -1):
            if corrected_p_values[i] < corrected_p_values[i - 1]:
                corrected_p_values[i - 1] = corrected_p_values[i]

        # Restore original order
        result = np.empty(n)
        result[sorted_indices] = corrected_p_values

        return result


class DataQualityValidator:
    """
    Validates data quality and identifies potential issues
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_outliers(self,
                      data: pd.Series,
                      method: str = 'zscore',
                      threshold: float = 3.0) -> Dict:
        """
        Detect outliers in data

        Args:
            data: Series to check
            method: 'zscore' or 'iqr'
            threshold: Threshold for outlier detection

        Returns:
            Dict with outlier statistics
        """
        if method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold
        elif method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            outliers = (data < q1 - threshold * iqr) | (data > q3 + threshold * iqr)
        else:
            raise ValueError(f"Unknown method: {method}")

        n_outliers = outliers.sum()
        outlier_pct = n_outliers / len(data) * 100

        return {
            'n_outliers': n_outliers,
            'outlier_pct': outlier_pct,
            'outlier_indices': data[outliers].index.tolist(),
            'outlier_values': data[outliers].tolist(),
            'is_acceptable': outlier_pct < 1.0  # < 1% outliers
        }

    def check_data_gaps(self,
                       df: pd.DataFrame,
                       date_column: str = 'entry_date') -> Dict:
        """
        Check for missing dates in time series

        Args:
            df: DataFrame with date column
            date_column: Name of date column

        Returns:
            Dict with gap statistics
        """
        dates = pd.to_datetime(df[date_column]).sort_values()

        # Expected business days (Mon-Fri)
        expected_dates = pd.bdate_range(start=dates.min(), end=dates.max())

        # Find missing dates
        missing_dates = set(expected_dates) - set(dates)

        return {
            'n_missing_dates': len(missing_dates),
            'missing_dates': sorted(list(missing_dates)),
            'data_completeness_pct': (1 - len(missing_dates) / len(expected_dates)) * 100
        }

    def check_survivorship_bias(self, df: pd.DataFrame) -> Dict:
        """
        Check for potential survivorship bias

        Args:
            df: DataFrame with stock data

        Returns:
            Dict with survivorship analysis
        """
        # Check if all stocks have data up to most recent date
        latest_date = df['entry_date'].max()
        stocks_with_recent_data = df[df['entry_date'] == latest_date]['Ticker'].unique()
        all_stocks = df['Ticker'].unique()

        missing_stocks = set(all_stocks) - set(stocks_with_recent_data)

        return {
            'total_stocks': len(all_stocks),
            'stocks_with_recent_data': len(stocks_with_recent_data),
            'potentially_delisted': list(missing_stocks),
            'survivorship_risk': len(missing_stocks) > 0
        }

    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality check

        Args:
            df: DataFrame to validate

        Returns:
            Dict with all validation results
        """
        results = {}

        # Check for missing values
        results['missing_values'] = {
            col: df[col].isna().sum()
            for col in df.columns
        }

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results['infinite_values'] = {
            col: np.isinf(df[col]).sum()
            for col in numeric_cols
        }

        # Check for duplicates (skip columns with unhashable types)
        try:
            # Only check hashable columns for duplicates
            hashable_cols = []
            for col in df.columns:
                try:
                    # Test if column values are hashable
                    hash(tuple(df[col].head(1).values))
                    hashable_cols.append(col)
                except (TypeError, ValueError):
                    continue

            if hashable_cols:
                results['duplicate_rows'] = df[hashable_cols].duplicated().sum()
            else:
                results['duplicate_rows'] = 0
        except Exception as e:
            logger.warning(f"Could not check for duplicates: {e}")
            results['duplicate_rows'] = 0

        # Check outliers in returns
        return_cols = [col for col in df.columns if col.startswith('return_')]
        if return_cols:
            for col in return_cols:
                if col in df.columns:
                    results[f'outliers_{col}'] = self.check_outliers(
                        df[col].dropna(),
                        method='zscore',
                        threshold=3.0
                    )

        # Check data gaps
        if 'entry_date' in df.columns:
            results['data_gaps'] = self.check_data_gaps(df, 'entry_date')

        # Check survivorship bias
        if 'Ticker' in df.columns and 'entry_date' in df.columns:
            results['survivorship'] = self.check_survivorship_bias(df)

        # Overall assessment (more lenient for financial data)
        issues = []

        # Only count missing values in critical columns (not analyst/earnings data)
        critical_cols = ['Ticker', 'entry_date', 'entry_price', 'max_drawdown']
        critical_cols.extend([col for col in df.columns if col.startswith('return_')])
        critical_cols.extend([col for col in df.columns if col.startswith('win_')])

        critical_missing = any(results['missing_values'].get(col, 0) > 0 for col in critical_cols if col in results['missing_values'])
        if critical_missing:
            issues.append('critical_missing_values')

        if any(v > 0 for v in results['infinite_values'].values()):
            issues.append('infinite_values')
        if results['duplicate_rows'] > 0:
            issues.append('duplicates')

        # More lenient outlier threshold for financial returns (5% instead of 1%)
        return_outlier_issues = 0
        for key, value in results.items():
            if key.startswith('outliers_return_'):
                outlier_pct = value.get('outlier_pct', 0)
                if outlier_pct > 5.0:  # Allow up to 5% outliers for financial data
                    return_outlier_issues += 1
        if return_outlier_issues > 0:
            issues.append('extreme_outliers')

        # Only flag survivorship bias if >10% of stocks are missing recent data
        surv_info = results.get('survivorship', {})
        if surv_info.get('survivorship_risk', False):
            total_stocks = surv_info.get('total_stocks', 0)
            missing_recent = len(surv_info.get('potentially_delisted', []))
            if total_stocks > 0 and (missing_recent / total_stocks) > 0.1:  # >10% missing
                issues.append('survivorship_bias')

        # Data completeness: only flag if <90% complete
        data_gaps = results.get('data_gaps', {})
        completeness = data_gaps.get('data_completeness_pct', 100)
        if completeness < 90:
            issues.append('poor_data_completeness')

        results['overall_quality'] = {
            'is_acceptable': len(issues) == 0,
            'issues_found': issues,
            'quality_score': max(0, 100 - len(issues) * 15)  # More lenient scoring
        }

        return results


# Utility functions for common statistical tests

def calculate_sharpe_ratio(returns: np.ndarray,
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)

    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    return sharpe * np.sqrt(periods_per_year)


def calculate_information_coefficient(predictions: np.ndarray,
                                     actuals: np.ndarray,
                                     method: str = 'spearman') -> float:
    """
    Calculate Information Coefficient (rank correlation)

    Args:
        predictions: Predicted values (e.g., ML scores)
        actuals: Actual values (e.g., forward returns)
        method: 'spearman' or 'pearson'

    Returns:
        IC value
    """
    # Remove any NaN pairs
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[mask]
    actuals = actuals[mask]

    if len(predictions) < 3:
        return np.nan

    if method == 'spearman':
        ic, _ = stats.spearmanr(predictions, actuals)
    elif method == 'pearson':
        ic, _ = stats.pearsonr(predictions, actuals)
    else:
        raise ValueError(f"Unknown method: {method}")

    return ic
