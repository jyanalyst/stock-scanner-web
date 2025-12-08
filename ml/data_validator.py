"""
Data Quality Validator for ML Training
Validates historical data before ML collection to ensure quality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MLDataValidator:
    """
    Comprehensive data quality validator for ML training preparation

    Validates:
    - Date coverage and gaps
    - Data completeness (NULL values)
    - Data integrity (price/volume logic)
    - Forward price availability
    - Minimum requirements for ML training
    """

    def __init__(self, start_date: str, end_date: str, min_days: int = 100):
        """
        Initialize validator

        Args:
            start_date: Training period start (YYYY-MM-DD)
            end_date: Training period end (YYYY-MM-DD)
            min_days: Minimum trading days required per stock
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.min_days = min_days
        self.expected_trading_days = self._calculate_expected_trading_days()

        logger.info(f"Validator initialized: {start_date} to {end_date} ({self.expected_trading_days} expected trading days)")

    def validate_all_stocks(self) -> Dict[str, Any]:
        """
        Validate all stocks in watchlist for ML readiness

        Returns:
            Comprehensive validation report
        """
        from utils.watchlist import get_active_watchlist

        stocks = get_active_watchlist()
        logger.info(f"Validating {len(stocks)} stocks in watchlist")

        results = {
            'summary': {
                'total_stocks': len(stocks),
                'ready_count': 0,
                'partial_count': 0,
                'failed_count': 0,
                'validation_date': datetime.now().isoformat(),
                'date_range': f"{self.start_date.date()} to {self.end_date.date()}",
                'expected_trading_days': self.expected_trading_days
            },
            'ready': [],      # Stocks ready for ML (95%+ coverage, no issues)
            'partial': [],    # Stocks usable but with issues (80-95% coverage)
            'failed': [],     # Stocks unusable (<80% coverage or critical issues)
            'details': {},    # Detailed results per stock
            'issues': {
                'missing_files': [],
                'insufficient_data': [],
                'data_gaps': [],
                'null_values': [],
                'integrity_issues': [],
                'forward_unavailable': []
            }
        }

        for i, ticker in enumerate(stocks):
            logger.info(f"Validating {ticker} ({i+1}/{len(stocks)})")

            try:
                validation = self.validate_single_stock(ticker)
                results['details'][ticker] = validation

                # Categorize stock
                if validation['is_ready']:
                    results['ready'].append(ticker)
                    results['summary']['ready_count'] += 1
                elif validation['is_partial']:
                    results['partial'].append(ticker)
                    results['summary']['partial_count'] += 1
                else:
                    results['failed'].append(ticker)
                    results['summary']['failed_count'] += 1

                # Track issues
                self._categorize_issues(validation, results['issues'])

            except Exception as e:
                logger.error(f"Validation failed for {ticker}: {e}")
                validation = {
                    'ticker': ticker,
                    'is_ready': False,
                    'is_partial': False,
                    'error': str(e),
                    'total_days': 0,
                    'coverage': 0.0
                }
                results['details'][ticker] = validation
                results['failed'].append(ticker)
                results['summary']['failed_count'] += 1
                results['issues']['missing_files'].append(ticker)

        # Calculate estimated ML samples
        results['summary']['estimated_samples'] = self._estimate_ml_samples(results)

        logger.info(f"Validation complete: {results['summary']['ready_count']} ready, "
                   f"{results['summary']['partial_count']} partial, "
                   f"{results['summary']['failed_count']} failed")

        return results

    def validate_single_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Validate single stock comprehensively

        Returns:
            Detailed validation results
        """
        from core.local_file_loader import get_local_loader

        loader = get_local_loader()

        try:
            # Load data
            df = loader.load_historical_data(ticker)

            if df is None or df.empty:
                return {
                    'ticker': ticker,
                    'is_ready': False,
                    'is_partial': False,
                    'error': 'No data file found',
                    'total_days': 0,
                    'coverage': 0.0
                }

            # Ensure Date column is datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Filter to validation date range
            mask = (df['Date'] >= self.start_date) & (df['Date'] <= self.end_date)
            df_range = df[mask].copy()

            # Basic metrics
            total_days = len(df_range)
            coverage = total_days / self.expected_trading_days if self.expected_trading_days > 0 else 0

            # Detailed checks
            gaps = self._find_date_gaps(df_range)
            null_counts = self._check_null_values(df_range)
            integrity_issues = self._check_data_integrity(df_range)
            forward_available = self._check_forward_price_availability(df_range)

            # Determine status
            has_nulls = null_counts.sum() > 0
            has_integrity_issues = len(integrity_issues) > 0
            has_major_gaps = len(gaps) > 5  # More than 5 gaps is concerning

            is_ready = (
                total_days >= self.min_days and
                coverage >= 0.95 and  # 95%+ coverage
                not has_nulls and
                not has_integrity_issues and
                not has_major_gaps and
                forward_available
            )

            is_partial = (
                total_days >= self.min_days and
                coverage >= 0.80 and  # 80%+ coverage acceptable
                not is_ready  # Not already ready
            )

            return {
                'ticker': ticker,
                'is_ready': is_ready,
                'is_partial': is_partial,
                'total_days': total_days,
                'expected_days': self.expected_trading_days,
                'coverage': coverage,
                'coverage_pct': f"{coverage:.1%}",
                'gaps': gaps,
                'gap_count': len(gaps),
                'null_counts': null_counts.to_dict(),
                'has_nulls': has_nulls,
                'integrity_issues': integrity_issues,
                'has_integrity_issues': has_integrity_issues,
                'forward_available': forward_available,
                'has_major_gaps': has_major_gaps,
                'data_quality_score': self._calculate_quality_score(
                    coverage, has_nulls, has_integrity_issues, has_major_gaps, forward_available
                )
            }

        except Exception as e:
            logger.error(f"Error validating {ticker}: {e}")
            return {
                'ticker': ticker,
                'is_ready': False,
                'is_partial': False,
                'error': str(e),
                'total_days': 0,
                'coverage': 0.0
            }

    def _calculate_expected_trading_days(self) -> int:
        """Calculate expected number of trading days in date range"""
        # Create date range excluding weekends
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        trading_days = date_range[date_range.weekday < 5]  # Monday=0, Friday=4

        # Rough estimate - could be refined with actual holiday calendar
        return len(trading_days)

    def _find_date_gaps(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find gaps in date sequence"""
        if df.empty:
            return []

        df = df.sort_values('Date')
        dates = df['Date'].dt.date

        gaps = []
        expected_dates = pd.date_range(dates.min(), dates.max(), freq='D')
        expected_dates = expected_dates[expected_dates.weekday < 5]  # Trading days only

        actual_dates = set(dates)
        missing_dates = [d.date() for d in expected_dates if d.date() not in actual_dates]

        # Group consecutive missing dates
        if missing_dates:
            from itertools import groupby
            from operator import itemgetter

            ranges = []
            for k, g in groupby(enumerate(missing_dates), lambda x: x[0] - missing_dates.index(x[1])):
                group = list(map(itemgetter(1), g))
                ranges.append((group[0], group[-1]))

            for start_date, end_date in ranges:
                if start_date == end_date:
                    gaps.append({
                        'date': start_date,
                        'gap_days': 1,
                        'description': f"Missing {start_date}"
                    })
                else:
                    gap_days = (end_date - start_date).days + 1
                    gaps.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'gap_days': gap_days,
                        'description': f"Gap {start_date} to {end_date} ({gap_days} days)"
                    })

        return gaps

    def _check_null_values(self, df: pd.DataFrame) -> pd.Series:
        """Check for NULL values in critical columns"""
        critical_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        existing_columns = [col for col in critical_columns if col in df.columns]

        if not existing_columns:
            return pd.Series(dtype=int)

        return df[existing_columns].isnull().sum()

    def _check_data_integrity(self, df: pd.DataFrame) -> List[str]:
        """Check data integrity rules"""
        issues = []

        if df.empty:
            return issues

        # Price logic checks
        if 'High' in df.columns and 'Low' in df.columns:
            invalid_hl = (df['High'] < df['Low']).sum()
            if invalid_hl > 0:
                issues.append(f"{invalid_hl} rows where High < Low")

        if 'Open' in df.columns and 'Close' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            invalid_ohlc = ((df['Open'] < df['Low']) | (df['Open'] > df['High']) |
                           (df['Close'] < df['Low']) | (df['Close'] > df['High'])).sum()
            if invalid_ohlc > 0:
                issues.append(f"{invalid_ohlc} rows with invalid OHLC relationships")

        # Volume checks
        if 'Volume' in df.columns:
            negative_volume = (df['Volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"{negative_volume} rows with negative volume")

        # Price checks
        price_columns = ['Open', 'High', 'Low', 'Close']
        existing_price_cols = [col for col in price_columns if col in df.columns]

        if existing_price_cols:
            negative_prices = (df[existing_price_cols] <= 0).any(axis=1).sum()
            if negative_prices > 0:
                issues.append(f"{negative_prices} rows with zero/negative prices")

        return issues

    def _check_forward_price_availability(self, df: pd.DataFrame) -> bool:
        """Check if we can calculate 4-day forward returns"""
        if df.empty:
            return False

        latest_date = df['Date'].max()
        required_end_date = latest_date + timedelta(days=4)  # For 4-day returns

        # Check if we have data extending to required_end_date
        # This is a rough check - actual forward calculation will be done during collection
        return latest_date >= (self.end_date - timedelta(days=10))  # Buffer for safety

    def _calculate_quality_score(self, coverage: float, has_nulls: bool,
                               has_integrity_issues: bool, has_major_gaps: bool,
                               forward_available: bool) -> float:
        """Calculate overall data quality score (0-100)"""
        score = coverage * 100  # Base score from coverage

        # Penalties
        if has_nulls:
            score -= 20
        if has_integrity_issues:
            score -= 15
        if has_major_gaps:
            score -= 10
        if not forward_available:
            score -= 25

        return max(0, min(100, score))

    def _categorize_issues(self, validation: Dict[str, Any], issues_dict: Dict[str, List[str]]):
        """Categorize validation issues for reporting"""
        ticker = validation['ticker']

        if 'error' in validation:
            if 'No data file found' in validation['error']:
                issues_dict['missing_files'].append(ticker)
            else:
                issues_dict['insufficient_data'].append(ticker)
            return

        if validation.get('gap_count', 0) > 0:
            issues_dict['data_gaps'].append(ticker)

        if validation.get('has_nulls', False):
            issues_dict['null_values'].append(ticker)

        if validation.get('has_integrity_issues', False):
            issues_dict['integrity_issues'].append(ticker)

        if not validation.get('forward_available', True):
            issues_dict['forward_unavailable'].append(ticker)

    def _estimate_ml_samples(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Estimate total ML training samples"""
        avg_days_per_stock = self.expected_trading_days * 0.9  # Conservative estimate

        ready_samples = len(results['ready']) * int(avg_days_per_stock)
        total_samples = (len(results['ready']) + len(results['partial'])) * int(avg_days_per_stock)

        return {
            'ready_only': ready_samples,
            'including_partial': total_samples,
            'avg_days_per_stock': int(avg_days_per_stock)
        }

    def generate_report(self, results: Dict[str, Any], output_path: str = "data/ml_training/validation_report.html"):
        """Generate detailed HTML report"""
        html_content = self._create_html_report(results)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Validation report saved to {output_path}")
        return output_path

    def _create_html_report(self, results: Dict[str, Any]) -> str:
        """Create HTML report content"""
        summary = results['summary']

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .summary {{ display: flex; gap: 20px; margin-bottom: 30px; }}
                .metric {{ background: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .ready {{ background: #d4edda; }}
                .partial {{ background: #fff3cd; }}
                .failed {{ background: #f8d7da; }}
                .section {{ margin-bottom: 30px; }}
                .stock-list {{ background: #f8f9fa; padding: 10px; border-radius: 3px; margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .danger {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 ML Data Quality Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Date Range:</strong> {summary['date_range']} ({summary['expected_trading_days']} expected trading days)</p>
            </div>

            <div class="summary">
                <div class="metric ready">
                    <h3>✅ Ready</h3>
                    <div style="font-size: 24px;">{summary['ready_count']}</div>
                    <div>({summary['ready_count']/summary['total_stocks']:.1%})</div>
                </div>
                <div class="metric partial">
                    <h3>⚠️ Partial</h3>
                    <div style="font-size: 24px;">{summary['partial_count']}</div>
                    <div>({summary['partial_count']/summary['total_stocks']:.1%})</div>
                </div>
                <div class="metric failed">
                    <h3>❌ Failed</h3>
                    <div style="font-size: 24px;">{summary['failed_count']}</div>
                    <div>({summary['failed_count']/summary['total_stocks']:.1%})</div>
                </div>
            </div>

            <div class="section">
                <h2>📊 Estimated ML Samples</h2>
                <p><strong>Using ready stocks only:</strong> {summary['estimated_samples']['ready_only']:,} samples</p>
                <p><strong>Including partial stocks:</strong> {summary['estimated_samples']['including_partial']:,} samples</p>
                <p><em>Average {summary['estimated_samples']['avg_days_per_stock']} days per stock</em></p>
            </div>
        """

        # Ready stocks
        if results['ready']:
            html += f"""
            <div class="section">
                <h2 class="good">✅ Ready for ML ({len(results['ready'])} stocks)</h2>
                <div class="stock-list">
                    {', '.join(sorted(results['ready']))}
                </div>
            </div>
            """

        # Partial stocks
        if results['partial']:
            html += f"""
            <div class="section">
                <h2 class="warning">⚠️ Partial Data ({len(results['partial'])} stocks)</h2>
                <p>These stocks have sufficient data but some quality issues. Can be used with caution.</p>
                <div class="stock-list">
                    {', '.join(sorted(results['partial']))}
                </div>
            </div>
            """

        # Failed stocks
        if results['failed']:
            html += f"""
            <div class="section">
                <h2 class="danger">❌ Failed Validation ({len(results['failed'])} stocks)</h2>
                <p>These stocks cannot be used for ML training due to insufficient or poor quality data.</p>
                <div class="stock-list">
                    {', '.join(sorted(results['failed']))}
                </div>
            </div>
            """

        # Issues summary
        issues = results['issues']
        if any(issues.values()):
            html += """
            <div class="section">
                <h2>🔍 Issues Detected</h2>
                <ul>
            """
            if issues['missing_files']:
                html += f"<li><strong>Missing files:</strong> {', '.join(issues['missing_files'])}</li>"
            if issues['insufficient_data']:
                html += f"<li><strong>Insufficient data:</strong> {', '.join(issues['insufficient_data'])}</li>"
            if issues['data_gaps']:
                html += f"<li><strong>Data gaps:</strong> {', '.join(issues['data_gaps'])}</li>"
            if issues['null_values']:
                html += f"<li><strong>NULL values:</strong> {', '.join(issues['null_values'])}</li>"
            if issues['integrity_issues']:
                html += f"<li><strong>Data integrity issues:</strong> {', '.join(issues['integrity_issues'])}</li>"
            if issues['forward_unavailable']:
                html += f"<li><strong>Forward prices unavailable:</strong> {', '.join(issues['forward_unavailable'])}</li>"

            html += "</ul></div>"

        # Detailed table
        html += """
            <div class="section">
                <h2>📋 Detailed Results</h2>
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Status</th>
                        <th>Days</th>
                        <th>Coverage</th>
                        <th>Gaps</th>
                        <th>NULLs</th>
                        <th>Issues</th>
                        <th>Quality Score</th>
                    </tr>
        """

        for ticker in sorted(results['details'].keys()):
            detail = results['details'][ticker]

            status = "✅ Ready" if detail.get('is_ready') else "⚠️ Partial" if detail.get('is_partial') else "❌ Failed"
            status_class = "good" if detail.get('is_ready') else "warning" if detail.get('is_partial') else "danger"

            days = detail.get('total_days', 0)
            coverage = detail.get('coverage_pct', '0%')
            gaps = detail.get('gap_count', 0)
            has_nulls = "Yes" if detail.get('has_nulls') else "No"
            issues = len(detail.get('integrity_issues', []))
            quality_score = detail.get('data_quality_score', 0)

            html += f"""
                <tr>
                    <td>{ticker}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{days}</td>
                    <td>{coverage}</td>
                    <td>{gaps}</td>
                    <td>{has_nulls}</td>
                    <td>{issues}</td>
                    <td>{quality_score:.1f}</td>
                </tr>
            """

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        return html
