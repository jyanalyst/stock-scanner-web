"""
Data Validation and Quality Assurance Module
Comprehensive data validation, quality checks, and data integrity verification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ValidationResult:
    """Result of a validation check"""

    def __init__(self, check_name: str, passed: bool, severity: ValidationSeverity,
                 message: str, details: Optional[Dict[str, Any]] = None):
        self.check_name = check_name
        self.passed = passed
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'check_name': self.check_name,
            'passed': self.passed,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class DataQualityMetrics:
    """Data quality metrics for a dataset"""
    total_rows: int = 0
    total_columns: int = 0
    missing_values: int = 0
    duplicate_rows: int = 0
    outlier_count: int = 0
    data_completeness: float = 0.0
    validation_score: float = 0.0
    issues: List[ValidationResult] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

class DataValidator:
    """Comprehensive data validation and quality assurance"""

    def __init__(self):
        self.validation_rules = {
            'stock_data': self._validate_stock_data,
            'analyst_reports': self._validate_analyst_reports,
            'earnings_reports': self._validate_earnings_reports,
            'technical_indicators': self._validate_technical_indicators
        }

    def validate_dataset(self, df: pd.DataFrame, dataset_type: str = 'generic',
                        strict: bool = False) -> DataQualityMetrics:
        """
        Comprehensive validation of a dataset

        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset ('stock_data', 'analyst_reports', etc.)
            strict: If True, fail on warnings; if False, only fail on errors

        Returns:
            DataQualityMetrics with validation results
        """
        if df is None or df.empty:
            return DataQualityMetrics(
                issues=[ValidationResult(
                    "empty_dataset", False, ValidationSeverity.CRITICAL,
                    "Dataset is empty or None", {"dataset_type": dataset_type}
                )]
            )

        metrics = DataQualityMetrics(
            total_rows=len(df),
            total_columns=len(df.columns)
        )

        # Basic structural validation
        metrics.issues.extend(self._validate_basic_structure(df, dataset_type))

        # Data type validation
        metrics.issues.extend(self._validate_data_types(df, dataset_type))

        # Missing data analysis
        missing_analysis = self._analyze_missing_data(df)
        metrics.missing_values = missing_analysis['total_missing']
        metrics.data_completeness = missing_analysis['completeness_score']
        metrics.issues.extend(missing_analysis['issues'])

        # Duplicate detection
        duplicate_analysis = self._detect_duplicates(df)
        metrics.duplicate_rows = duplicate_analysis['duplicate_count']
        metrics.issues.extend(duplicate_analysis['issues'])

        # Outlier detection
        outlier_analysis = self._detect_outliers(df)
        metrics.outlier_count = outlier_analysis['outlier_count']
        metrics.issues.extend(outlier_analysis['issues'])

        # Dataset-specific validation
        if dataset_type in self.validation_rules:
            specific_issues = self.validation_rules[dataset_type](df)
            metrics.issues.extend(specific_issues)

        # Calculate overall validation score
        metrics.validation_score = self._calculate_validation_score(metrics, strict)

        return metrics

    def _validate_basic_structure(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationResult]:
        """Validate basic DataFrame structure"""
        issues = []

        # Check for required columns based on dataset type
        required_columns = self._get_required_columns(dataset_type)
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            issues.append(ValidationResult(
                "missing_required_columns", False, ValidationSeverity.ERROR,
                f"Missing required columns: {missing_columns}",
                {"missing_columns": missing_columns, "dataset_type": dataset_type}
            ))

        # Check for minimum row count
        min_rows = self._get_minimum_rows(dataset_type)
        if len(df) < min_rows:
            issues.append(ValidationResult(
                "insufficient_data", False, ValidationSeverity.WARNING,
                f"Dataset has only {len(df)} rows, minimum required: {min_rows}",
                {"actual_rows": len(df), "minimum_rows": min_rows}
            ))

        return issues

    def _validate_data_types(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationResult]:
        """Validate data types for each column"""
        issues = []
        expected_types = self._get_expected_data_types(dataset_type)

        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue

            actual_type = df[col].dtype

            # Check if data type is compatible
            if not self._is_compatible_type(actual_type, expected_type):
                issues.append(ValidationResult(
                    "incorrect_data_type", False, ValidationSeverity.WARNING,
                    f"Column '{col}' has type {actual_type}, expected {expected_type}",
                    {"column": col, "actual_type": str(actual_type), "expected_type": expected_type}
                ))

            # Check for invalid values in numeric columns
            if expected_type in ['numeric', 'float', 'int']:
                invalid_count = df[col].isna().sum() + (~df[col].apply(self._is_valid_number)).sum()
                if invalid_count > 0:
                    issues.append(ValidationResult(
                        "invalid_numeric_values", False, ValidationSeverity.ERROR,
                        f"Column '{col}' has {invalid_count} invalid numeric values",
                        {"column": col, "invalid_count": invalid_count}
                    ))

        return issues

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        issues = []
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0

        # Check columns with high missing rates
        missing_by_column = df.isna().sum()
        high_missing_cols = missing_by_column[missing_by_column > len(df) * 0.5]

        for col, missing_count in high_missing_cols.items():
            missing_pct = (missing_count / len(df)) * 100
            severity = ValidationSeverity.ERROR if missing_pct > 80 else ValidationSeverity.WARNING

            issues.append(ValidationResult(
                "high_missing_rate", False, severity,
                f"Column '{col}' has {missing_pct:.1f}% missing values ({missing_count}/{len(df)})",
                {"column": col, "missing_count": missing_count, "missing_percentage": missing_pct}
            ))

        return {
            'total_missing': missing_cells,
            'completeness_score': completeness_score,
            'issues': issues
        }

    def _detect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate rows"""
        issues = []

        # Convert unhashable columns (lists) to strings for duplicate detection
        df_for_duplicates = df.copy()
        for col in df_for_duplicates.columns:
            if df_for_duplicates[col].dtype == 'object':
                # Convert lists and other unhashable types to strings
                df_for_duplicates[col] = df_for_duplicates[col].apply(
                    lambda x: str(x) if isinstance(x, (list, dict)) else x
                )

        duplicate_count = df_for_duplicates.duplicated().sum()

        if duplicate_count > 0:
            severity = ValidationSeverity.ERROR if duplicate_count > len(df) * 0.1 else ValidationSeverity.WARNING

            issues.append(ValidationResult(
                "duplicate_rows", False, severity,
                f"Found {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.1f}%)",
                {"duplicate_count": duplicate_count, "duplicate_percentage": duplicate_count/len(df)*100}
            ))

        return {
            'duplicate_count': duplicate_count,
            'issues': issues
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical outliers in numeric columns"""
        issues = []
        outlier_count = 0

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Skip columns that shouldn't have outliers checked
            if col in ['Date', 'Analysis_Date'] or 'Date' in col:
                continue

            # Use IQR method for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            col_outliers = len(outliers)

            if col_outliers > 0:
                outlier_count += col_outliers
                outlier_pct = (col_outliers / len(df)) * 100

                # Only report if outliers are significant
                if outlier_pct > 5:  # More than 5% outliers
                    issues.append(ValidationResult(
                        "statistical_outliers", False, ValidationSeverity.INFO,
                        f"Column '{col}' has {col_outliers} outliers ({outlier_pct:.1f}%)",
                        {"column": col, "outlier_count": col_outliers, "outlier_percentage": outlier_pct}
                    ))

        return {
            'outlier_count': outlier_count,
            'issues': issues
        }

    def _validate_stock_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate stock market data"""
        issues = []

        # Check for required OHLCV columns
        required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_ohlcv = [col for col in required_ohlcv if col not in df.columns]

        if missing_ohlcv:
            issues.append(ValidationResult(
                "missing_ohlcv_columns", False, ValidationSeverity.CRITICAL,
                f"Missing OHLCV columns: {missing_ohlcv}",
                {"missing_columns": missing_ohlcv}
            ))

        # Validate OHLC logic (High >= Low, Close between High and Low, etc.)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_logic = (
                (df['High'] < df['Low']) |
                (df['Close'] > df['High']) |
                (df['Close'] < df['Low']) |
                (df['Open'] > df['High']) |
                (df['Open'] < df['Low'])
            ).sum()

            if invalid_logic > 0:
                issues.append(ValidationResult(
                    "invalid_ohlc_logic", False, ValidationSeverity.ERROR,
                    f"{invalid_logic} rows have invalid OHLC logic",
                    {"invalid_rows": invalid_logic}
                ))

        # Check volume is positive
        if 'Volume' in df.columns:
            negative_volume = (df['Volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(ValidationResult(
                    "negative_volume", False, ValidationSeverity.ERROR,
                    f"{negative_volume} rows have negative volume",
                    {"negative_volume_rows": negative_volume}
                ))

        return issues

    def _validate_analyst_reports(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate analyst report data"""
        issues = []

        # Check sentiment score range
        if 'sentiment_score' in df.columns:
            invalid_sentiment = ((df['sentiment_score'] < -1) | (df['sentiment_score'] > 1)).sum()
            if invalid_sentiment > 0:
                issues.append(ValidationResult(
                    "invalid_sentiment_range", False, ValidationSeverity.ERROR,
                    f"{invalid_sentiment} reports have sentiment scores outside [-1, 1] range",
                    {"invalid_sentiment_count": invalid_sentiment}
                ))

        # Check for future report dates
        if 'report_date' in df.columns:
            future_reports = (pd.to_datetime(df['report_date']) > datetime.now()).sum()
            if future_reports > 0:
                issues.append(ValidationResult(
                    "future_report_dates", False, ValidationSeverity.WARNING,
                    f"{future_reports} reports have future dates",
                    {"future_reports": future_reports}
                ))

        return issues

    def _validate_earnings_reports(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate earnings report data"""
        issues = []

        # Check for negative revenue (generally shouldn't happen)
        if 'revenue' in df.columns:
            negative_revenue = (df['revenue'] < 0).sum()
            if negative_revenue > 0:
                issues.append(ValidationResult(
                    "negative_revenue", False, ValidationSeverity.WARNING,
                    f"{negative_revenue} reports have negative revenue",
                    {"negative_revenue_count": negative_revenue}
                ))

        return issues

    def _validate_technical_indicators(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate technical indicator calculations"""
        issues = []

        # Check MPI values are in reasonable range
        if 'MPI' in df.columns:
            invalid_mpi = ((df['MPI'] < 0) | (df['MPI'] > 1)).sum()
            if invalid_mpi > 0:
                issues.append(ValidationResult(
                    "invalid_mpi_range", False, ValidationSeverity.ERROR,
                    f"{invalid_mpi} rows have MPI values outside [0, 1] range",
                    {"invalid_mpi_count": invalid_mpi}
                ))

        # Check IBS values are in [0, 1] range
        if 'IBS' in df.columns:
            invalid_ibs = ((df['IBS'] < 0) | (df['IBS'] > 1)).sum()
            if invalid_ibs > 0:
                issues.append(ValidationResult(
                    "invalid_ibs_range", False, ValidationSeverity.ERROR,
                    f"{invalid_ibs} rows have IBS values outside [0, 1] range",
                    {"invalid_ibs_count": invalid_ibs}
                ))

        return issues

    def _get_required_columns(self, dataset_type: str) -> List[str]:
        """Get required columns for each dataset type"""
        requirements = {
            'stock_data': ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
            'analyst_reports': ['ticker_sgx', 'report_date', 'sentiment_score'],
            'earnings_reports': ['ticker_sgx', 'report_date', 'revenue'],
            'technical_indicators': ['Date', 'Close', 'MPI', 'IBS']
        }
        return requirements.get(dataset_type, [])

    def _get_minimum_rows(self, dataset_type: str) -> int:
        """Get minimum required rows for each dataset type"""
        minimums = {
            'stock_data': 30,  # At least a month of data
            'analyst_reports': 1,
            'earnings_reports': 1,
            'technical_indicators': 10
        }
        return minimums.get(dataset_type, 1)

    def _get_expected_data_types(self, dataset_type: str) -> Dict[str, str]:
        """Get expected data types for columns"""
        type_requirements = {
            'stock_data': {
                'Open': 'numeric', 'High': 'numeric', 'Low': 'numeric',
                'Close': 'numeric', 'Volume': 'numeric', 'Date': 'datetime'
            },
            'analyst_reports': {
                'sentiment_score': 'numeric', 'report_date': 'datetime'
            },
            'earnings_reports': {
                'revenue': 'numeric', 'report_date': 'datetime'
            }
        }
        return type_requirements.get(dataset_type, {})

    def _is_compatible_type(self, actual_type, expected_type: str) -> bool:
        """Check if actual data type is compatible with expected type"""
        if expected_type == 'numeric':
            return actual_type in ['int64', 'float64', 'int32', 'float32']
        elif expected_type == 'datetime':
            return 'datetime' in str(actual_type).lower()
        elif expected_type == 'string':
            return actual_type == 'object'
        return True

    def _is_valid_number(self, value) -> bool:
        """Check if a value is a valid number"""
        try:
            float(value)
            return not np.isnan(float(value)) and not np.isinf(float(value))
        except (ValueError, TypeError):
            return False

    def _calculate_validation_score(self, metrics: DataQualityMetrics, strict: bool = False) -> float:
        """Calculate overall validation score"""
        if not metrics.issues:
            return 1.0  # Perfect score

        # Count issues by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.ERROR: 0.8,
            ValidationSeverity.WARNING: 0.4 if strict else 0.2,
            ValidationSeverity.INFO: 0.1
        }

        total_penalty = 0
        for issue in metrics.issues:
            if not issue.passed:  # Only penalize failed checks
                total_penalty += severity_weights.get(issue.severity, 0.1)

        # Normalize score (1.0 = perfect, 0.0 = many issues)
        max_penalty = len(metrics.issues) * 1.0  # Worst case
        score = max(0, 1.0 - (total_penalty / max_penalty if max_penalty > 0 else 0))

        return score

    def clean_dataset(self, df: pd.DataFrame, dataset_type: str = 'generic',
                     imputation_strategy: str = 'mean') -> Tuple[pd.DataFrame, List[ValidationResult]]:
        """
        Clean and impute missing data in a dataset

        Args:
            df: DataFrame to clean
            dataset_type: Type of dataset
            imputation_strategy: Strategy for missing data ('mean', 'median', 'forward_fill', 'drop')

        Returns:
            Tuple of (cleaned_df, cleaning_actions)
        """
        cleaned_df = df.copy()
        cleaning_actions = []

        # Handle missing values based on strategy
        if imputation_strategy == 'drop':
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.dropna()
            dropped_rows = initial_rows - len(cleaned_df)

            if dropped_rows > 0:
                cleaning_actions.append(ValidationResult(
                    "dropped_missing_rows", True, ValidationSeverity.INFO,
                    f"Dropped {dropped_rows} rows with missing values",
                    {"dropped_rows": dropped_rows}
                ))

        elif imputation_strategy in ['mean', 'median']:
            for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                missing_count = cleaned_df[col].isna().sum()
                if missing_count > 0:
                    if imputation_strategy == 'mean':
                        fill_value = cleaned_df[col].mean()
                    else:
                        fill_value = cleaned_df[col].median()

                    cleaned_df[col] = cleaned_df[col].fillna(fill_value)

                    cleaning_actions.append(ValidationResult(
                        "imputed_missing_values", True, ValidationSeverity.INFO,
                        f"Imputed {missing_count} missing values in '{col}' with {imputation_strategy}: {fill_value:.4f}",
                        {"column": col, "imputed_count": missing_count, "fill_value": fill_value}
                    ))

        elif imputation_strategy == 'forward_fill':
            initial_missing = cleaned_df.isna().sum().sum()
            cleaned_df = cleaned_df.fillna(method='ffill')
            final_missing = cleaned_df.isna().sum().sum()

            imputed_count = initial_missing - final_missing
            if imputed_count > 0:
                cleaning_actions.append(ValidationResult(
                    "forward_fill_missing", True, ValidationSeverity.INFO,
                    f"Forward-filled {imputed_count} missing values",
                    {"imputed_count": imputed_count}
                ))

        # Remove duplicate rows
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)

        if removed_duplicates > 0:
            cleaning_actions.append(ValidationResult(
                "removed_duplicates", True, ValidationSeverity.INFO,
                f"Removed {removed_duplicates} duplicate rows",
                {"removed_duplicates": removed_duplicates}
            ))

        return cleaned_df, cleaning_actions

# Global validator instance
_data_validator = DataValidator()

def validate_data_quality(df: pd.DataFrame, dataset_type: str = 'generic',
                         strict: bool = False) -> DataQualityMetrics:
    """Convenience function for data validation"""
    return _data_validator.validate_dataset(df, dataset_type, strict)

def clean_data(df: pd.DataFrame, dataset_type: str = 'generic',
               imputation_strategy: str = 'mean') -> Tuple[pd.DataFrame, List[ValidationResult]]:
    """Convenience function for data cleaning"""
    return _data_validator.clean_dataset(df, dataset_type, imputation_strategy)

def get_validation_summary(metrics: DataQualityMetrics) -> Dict[str, Any]:
    """Get a summary of validation results"""
    issues_by_severity = {}
    for severity in ValidationSeverity:
        issues_by_severity[severity.value] = len([
            issue for issue in metrics.issues if issue.severity == severity
        ])

    return {
        'total_rows': metrics.total_rows,
        'total_columns': metrics.total_columns,
        'data_completeness': f"{metrics.data_completeness:.1%}",
        'validation_score': f"{metrics.validation_score:.1%}",
        'missing_values': metrics.missing_values,
        'duplicate_rows': metrics.duplicate_rows,
        'outlier_count': metrics.outlier_count,
        'issues_by_severity': issues_by_severity,
        'total_issues': len(metrics.issues),
        'critical_issues': issues_by_severity.get('critical', 0),
        'error_issues': issues_by_severity.get('error', 0),
        'warning_issues': issues_by_severity.get('warning', 0),
        'info_issues': issues_by_severity.get('info', 0)
    }

logger.info("Data validation and quality assurance module initialized")