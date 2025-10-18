"""
Comprehensive Error Handling & Logging System
Enhanced with structured logging, error recovery, and user-friendly messages
"""

import logging
import traceback
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Union, List
from functools import wraps
import time

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create dummy st object for testing
    class DummyST:
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def info(self, msg): print(f"INFO: {msg}")
        def success(self, msg): print(f"SUCCESS: {msg}")
        def session_state(self): return {}
        def expander(self, *args, **kwargs): return self
        def code(self, text, **kwargs): print(text)
        def json(self, data): print(data)
        def write(self, text): print(text)
        def markdown(self, text): print(text)
    st = DummyST()

logger = logging.getLogger(__name__)


# Enhanced Error Hierarchy
class AppError(Exception):
    """Base exception class for application errors"""

    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None,
                 user_message: str = None, recovery_action: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now()
        self.correlation_id = str(uuid.uuid4())[:8]
        self.user_message = user_message or self._generate_user_message()
        self.recovery_action = recovery_action

    def _generate_user_message(self) -> str:
        """Generate user-friendly message from error code"""
        user_messages = {
            "DATA_LOAD_ERROR": "Unable to load stock data. Please check your data files.",
            "DATA_VALIDATION_ERROR": "Data validation failed. Please verify your input.",
            "NETWORK_TIMEOUT": "Connection timed out. Please check your internet connection.",
            "FILE_NOT_FOUND": "Required file not found. Please check file paths.",
            "PROCESSING_ERROR": "Data processing failed. Please try again.",
            "UNKNOWN_ERROR": "An unexpected error occurred. Please contact support."
        }
        return user_messages.get(self.error_code, "An error occurred. Please try again.")


class DataError(AppError):
    """Data-related errors"""
    pass

class DataLoadError(DataError):
    """Error loading data"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DATA_LOAD_ERROR", **kwargs)

class DataValidationError(DataError):
    """Data validation error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)

class DataProcessingError(DataError):
    """Error during data processing"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PROCESSING_ERROR", **kwargs)


class NetworkError(AppError):
    """Network/API errors"""
    pass

class NetworkTimeoutError(NetworkError):
    """Network timeout error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="NETWORK_TIMEOUT", **kwargs)

class NetworkConnectionError(NetworkError):
    """Network connection error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="NETWORK_CONNECTION", **kwargs)


class FileSystemError(AppError):
    """File system errors"""
    pass

class FileNotFoundError(FileSystemError):
    """File not found error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="FILE_NOT_FOUND", **kwargs)

class FilePermissionError(FileSystemError):
    """File permission error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="FILE_PERMISSION", **kwargs)


class UIError(AppError):
    """UI-related error"""
    pass


class ConfigurationError(AppError):
    """Configuration error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)


# Structured Logging System
class StructuredLogger:
    """Enhanced logging with JSON formatting and context"""

    def __init__(self, log_level: str = "INFO", log_file: str = None):
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file
        self.setup_logger()

    def setup_logger(self):
        """Setup structured logging"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(component)s | %(correlation_id)s | %(message)s',
            defaults={'component': 'Unknown', 'correlation_id': 'N/A'}
        )

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Setup file handler if specified
        handlers = [console_handler]
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            handlers=handlers,
            force=True  # Override any existing configuration
        )

    def log(self, level: str, component: str, message: str,
            correlation_id: str = None, **context):
        """Log structured message with context"""
        extra = {
            'component': component,
            'correlation_id': correlation_id or str(uuid.uuid4())[:8]
        }

        # Add context as structured data
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            message = f"{message} | {context_str}"

        logger = logging.getLogger(component)
        getattr(logger, level.lower())(message, extra=extra)


# Global structured logger instance
structured_logger = StructuredLogger()


class ErrorLogger:
    """Enhanced error logging and display with structured logging"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.debug_info = []
        self.performance_metrics = []

    def log_error(self, component: str, error: Exception, context: Dict[str, Any] = None,
                  correlation_id: str = None) -> Dict[str, Any]:
        """Log an error with full context and structured logging"""
        correlation_id = correlation_id or str(uuid.uuid4())[:8]

        error_info = {
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'user_message': getattr(error, 'user_message', None),
            'recovery_action': getattr(error, 'recovery_action', None)
        }

        self.errors.append(error_info)

        # Structured logging
        structured_logger.log('ERROR', component,
                            f"{error_info['error_type']}: {error_info['error_message']}",
                            correlation_id, **(context or {}))

        return error_info

    def log_warning(self, component: str, message: str, context: Dict[str, Any] = None,
                   correlation_id: str = None):
        """Log a warning with structured logging"""
        correlation_id = correlation_id or str(uuid.uuid4())[:8]

        warning_info = {
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'component': component,
            'message': message,
            'context': context or {}
        }

        self.warnings.append(warning_info)

        # Structured logging
        structured_logger.log('WARNING', component, message, correlation_id, **(context or {}))

        return warning_info

    def log_debug(self, component: str, message: str, data: Any = None,
                 correlation_id: str = None):
        """Log debug information with structured logging"""
        correlation_id = correlation_id or str(uuid.uuid4())[:8]

        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'component': component,
            'message': message,
            'data': str(data) if data is not None else None
        }

        self.debug_info.append(debug_info)

        # Structured logging
        structured_logger.log('DEBUG', component, message, correlation_id, data=str(data))

        return debug_info

    def log_performance(self, component: str, operation: str, duration: float,
                       success: bool = True, context: Dict[str, Any] = None):
        """Log performance metrics"""
        correlation_id = str(uuid.uuid4())[:8]

        perf_info = {
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'component': component,
            'operation': operation,
            'duration': duration,
            'success': success,
            'context': context or {}
        }

        self.performance_metrics.append(perf_info)

        level = 'INFO' if success else 'WARNING'
        structured_logger.log(level, component,
                            f"Performance: {operation} took {duration:.3f}s",
                            correlation_id, success=success, **(context or {}))

        return perf_info

    def display_errors_in_streamlit(self):
        """Display all logged errors in Streamlit interface with progressive disclosure"""
        if self.errors:
            # Show summary first
            error_summary = self._categorize_errors()
            st.error(f"❌ {len(self.errors)} error(s) occurred during execution")

            # Progressive disclosure: Simple → Detailed
            with st.expander("🔍 View Error Details", expanded=False):
                # Show error categories first
                if error_summary:
                    st.markdown("**Error Summary by Category:**")
                    for category, count in error_summary.items():
                        st.write(f"• {category}: {count} error(s)")
                    st.markdown("---")

                # Show individual errors
                for i, error in enumerate(self.errors, 1):
                    # Simple view first
                    user_msg = error.get('user_message', error['error_message'])
                    st.markdown(f"**Error {i}: {error['component']}** - {user_msg}")

                    # Detailed view in sub-expander
                    with st.expander(f"Technical Details - Error {i}", expanded=False):
                        st.code(f"Type: {error['error_type']}\nMessage: {error['error_message']}")

                        if error.get('recovery_action'):
                            st.info(f"💡 **Suggested Action:** {error['recovery_action']}")

                        if error['context']:
                            st.json(error['context'])

                        with st.expander(f"Full Traceback - Error {i}"):
                            st.code(error['traceback'], language='python')

                    st.markdown("---")

        if self.warnings:
            st.warning(f"⚠️ {len(self.warnings)} warning(s) occurred")

            with st.expander("📋 View Warnings", expanded=False):
                for warning in self.warnings:
                    st.write(f"**{warning['component']}**: {warning['message']}")
                    if warning['context']:
                        st.json(warning['context'])

    def _categorize_errors(self) -> Dict[str, int]:
        """Categorize errors by type for summary"""
        categories = {}
        for error in self.errors:
            error_type = error.get('error_type', 'Unknown')
            categories[error_type] = categories.get(error_type, 0) + 1
        return categories

    def get_summary(self) -> Dict[str, int]:
        """Get a summary of logged issues"""
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'debug_entries': len(self.debug_info),
            'performance_metrics': len(self.performance_metrics)
        }

    def clear(self):
        """Clear all logged issues"""
        self.errors = []
        self.warnings = []
        self.debug_info = []
        self.performance_metrics = []

    def export_logs(self, format: str = "json") -> str:
        """Export logs in specified format"""
        all_logs = {
            'errors': self.errors,
            'warnings': self.warnings,
            'debug_info': self.debug_info,
            'performance': self.performance_metrics,
            'summary': self.get_summary(),
            'exported_at': datetime.now().isoformat()
        }

        if format == "json":
            return json.dumps(all_logs, indent=2, default=str)
        else:
            # Plain text format
            lines = [f"Log Export - {datetime.now().isoformat()}"]
            lines.append("=" * 50)

            for error in self.errors:
                lines.append(f"ERROR [{error['component']}]: {error['error_message']}")

            for warning in self.warnings:
                lines.append(f"WARNING [{warning['component']}]: {warning['message']}")

            return "\n".join(lines)


# Error Recovery Mechanisms
class RetryMechanism:
    """Retry mechanism with exponential backoff"""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def execute(self, func: Callable, *args, component: str = "Unknown",
               context: Dict[str, Any] = None, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_error = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)

                if attempt < self.max_attempts - 1:  # Not the last attempt
                    structured_logger.log('WARNING', component,
                                        f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {str(e)}",
                                        context=context)
                    time.sleep(delay)
                else:
                    structured_logger.log('ERROR', component,
                                        f"All {self.max_attempts} attempts failed: {str(e)}",
                                        context=context)

        # All attempts failed
        raise last_error


class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise NetworkConnectionError("Circuit breaker is OPEN - service temporarily unavailable",
                                           recovery_action="Wait for automatic recovery or contact support")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


# Global instances
default_retry = RetryMechanism()
default_circuit_breaker = CircuitBreaker()


def handle_error(error: Exception, component: str, context: Dict[str, Any] = None,
                show_user_message: bool = True, correlation_id: str = None) -> Dict[str, Any]:
    """
    Enhanced centralized error handling function

    Args:
        error: The exception that occurred
        component: Component where error occurred
        context: Additional context information
        show_user_message: Whether to show user-friendly message
        correlation_id: Correlation ID for tracking

    Returns:
        Error information dictionary
    """
    # Get or create error logger from session state
    if STREAMLIT_AVAILABLE and 'error_logger' not in st.session_state:
        st.session_state.error_logger = ErrorLogger()

    error_logger = st.session_state.error_logger if STREAMLIT_AVAILABLE else ErrorLogger()

    # Log the error
    error_info = error_logger.log_error(component, error, context, correlation_id)

    # Show user-friendly message if requested
    if show_user_message and STREAMLIT_AVAILABLE:
        user_message = getattr(error, 'user_message', None)
        recovery_action = getattr(error, 'recovery_action', None)

        if user_message:
            st.error(f"❌ {user_message}")
            if recovery_action:
                st.info(f"💡 **Suggested Action:** {recovery_action}")
        else:
            # Fallback to error type based messages
            if isinstance(error, DataLoadError):
                st.error("❌ Failed to load data. Please check your data sources.")
            elif isinstance(error, DataValidationError):
                st.error("❌ Data validation failed. Please check your input data.")
            elif isinstance(error, NetworkTimeoutError):
                st.error("❌ Connection timed out. Please check your internet connection and try again.")
            elif isinstance(error, NetworkConnectionError):
                st.error("❌ Network connection failed. Please check your internet connection.")
            elif isinstance(error, FileNotFoundError):
                st.error("❌ Required file not found. Please check file paths and permissions.")
            elif isinstance(error, DataProcessingError):
                st.error("❌ Data processing failed. Please try again or check error details.")
            else:
                st.error(f"❌ An error occurred in {component}. Check error details below.")

    return error_info


def safe_execute(func: Callable, *args, component: str = "Unknown",
                context: Dict[str, Any] = None, fallback_value=None,
                enable_retry: bool = False, enable_circuit_breaker: bool = False,
                **kwargs) -> Any:
    """
    Enhanced safe execution with retry and circuit breaker support

    Args:
        func: Function to execute
        *args: Positional arguments for function
        component: Component name for error logging
        context: Additional context for error logging
        fallback_value: Value to return on error
        enable_retry: Whether to enable retry mechanism
        enable_circuit_breaker: Whether to enable circuit breaker
        **kwargs: Keyword arguments for function

    Returns:
        Function result or fallback_value on error
    """
    correlation_id = str(uuid.uuid4())[:8]

    # Performance monitoring
    start_time = time.time()

    try:
        # Apply circuit breaker if enabled
        if enable_circuit_breaker:
            result = default_circuit_breaker.call(func, *args, **kwargs)
        elif enable_retry:
            result = default_retry.execute(func, *args, component=component,
                                         context=context, **kwargs)
        else:
            result = func(*args, **kwargs)

        # Log performance success
        duration = time.time() - start_time
        if STREAMLIT_AVAILABLE and 'error_logger' in st.session_state:
            st.session_state.error_logger.log_performance(component, func.__name__, duration, True, context)

        return result

    except Exception as e:
        # Log performance failure
        duration = time.time() - start_time
        if STREAMLIT_AVAILABLE and 'error_logger' in st.session_state:
            st.session_state.error_logger.log_performance(component, func.__name__, duration, False, context)

        # Handle the error
        handle_error(e, component, context, correlation_id=correlation_id)
        return fallback_value


# Decorators for error handling
def with_error_handling(component: str = "Unknown", fallback_value=None,
                       enable_retry: bool = False, enable_circuit_breaker: bool = False):
    """
    Decorator for automatic error handling

    Args:
        component: Component name for logging
        fallback_value: Value to return on error
        enable_retry: Enable retry mechanism
        enable_circuit_breaker: Enable circuit breaker
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return safe_execute(func, *args, component=component,
                              fallback_value=fallback_value,
                              enable_retry=enable_retry,
                              enable_circuit_breaker=enable_circuit_breaker,
                              **kwargs)
        return wrapper
    return decorator


# User-Friendly Error Messages and Progressive Disclosure
class UserMessageManager:
    """Manages user-friendly error messages with progressive disclosure"""

    def __init__(self):
        self.message_templates = {
            "DATA_LOAD_ERROR": {
                "simple": "Unable to load stock data",
                "detailed": "The system couldn't load the stock data from your files. This might be due to missing files, corrupted data, or permission issues.",
                "action": "Please check that your data files exist and are accessible, then try again."
            },
            "DATA_VALIDATION_ERROR": {
                "simple": "Data validation failed",
                "detailed": "The loaded data doesn't meet the required quality standards or format expectations.",
                "action": "Please verify your data files are in the correct format and contain valid information."
            },
            "NETWORK_TIMEOUT": {
                "simple": "Connection timed out",
                "detailed": "The system couldn't connect to external services within the expected time limit.",
                "action": "Please check your internet connection and try again. If the problem persists, the service might be temporarily unavailable."
            },
            "NETWORK_CONNECTION": {
                "simple": "Network connection failed",
                "detailed": "Unable to establish a connection to required external services.",
                "action": "Please check your internet connection and firewall settings."
            },
            "FILE_NOT_FOUND": {
                "simple": "Required file not found",
                "detailed": "One or more required data files are missing from the expected location.",
                "action": "Please ensure all required data files are present in the correct directories."
            },
            "FILE_PERMISSION": {
                "simple": "File access denied",
                "detailed": "The system doesn't have permission to read or write required files.",
                "action": "Please check file permissions and ensure the application has read/write access to data directories."
            },
            "PROCESSING_ERROR": {
                "simple": "Data processing failed",
                "detailed": "An error occurred while processing the stock data for analysis.",
                "action": "Please try again. If the problem persists, check the error details for more information."
            },
            "CONFIG_ERROR": {
                "simple": "Configuration error",
                "detailed": "There's an issue with the application configuration settings.",
                "action": "Please check your configuration files and environment variables."
            }
        }

    def get_message(self, error_code: str, level: str = "simple") -> Dict[str, str]:
        """Get user message for error code at specified detail level"""
        template = self.message_templates.get(error_code, {
            "simple": "An unexpected error occurred",
            "detailed": "An unexpected error occurred during operation.",
            "action": "Please try again or contact support if the problem persists."
        })

        return {
            "simple": template["simple"],
            "detailed": template.get("detailed", template["simple"]),
            "action": template.get("action", "Please try again.")
        }


# Global user message manager
user_message_manager = UserMessageManager()


def display_user_friendly_error(error: AppError, show_detailed: bool = False):
    """Display user-friendly error with progressive disclosure"""
    if not STREAMLIT_AVAILABLE:
        print(f"ERROR: {error.user_message}")
        return

    messages = user_message_manager.get_message(error.error_code)

    # Always show simple message
    st.error(f"❌ {messages['simple']}")

    # Show detailed message if requested or if it's a critical error
    if show_detailed or error.error_code in ["CONFIG_ERROR", "FILE_PERMISSION"]:
        with st.expander("🔍 More Details", expanded=False):
            st.write(messages['detailed'])
            st.info(f"💡 **Suggested Action:** {messages['action']}")

            # Show correlation ID for support
            if hasattr(error, 'correlation_id'):
                st.caption(f"Reference ID: {error.correlation_id}")


def create_error_report(error_logger: ErrorLogger) -> str:
    """Create a comprehensive error report for user support"""
    summary = error_logger.get_summary()

    report_lines = [
        "# Error Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        f"- Total Errors: {summary['errors']}",
        f"- Total Warnings: {summary['warnings']}",
        f"- Debug Entries: {summary['debug_entries']}",
        f"- Performance Metrics: {summary['performance_metrics']}",
        "",
        "## Recent Errors"
    ]

    # Add last 5 errors
    for i, error in enumerate(error_logger.errors[-5:], 1):
        report_lines.extend([
            f"### Error {i}: {error['component']}",
            f"- Time: {error['timestamp']}",
            f"- Type: {error['error_type']}",
            f"- Message: {error['error_message']}",
            f"- Correlation ID: {error.get('correlation_id', 'N/A')}",
            ""
        ])

    return "\n".join(report_lines)


# Utility functions for common error scenarios
def handle_file_operation_error(operation: str, file_path: str, error: Exception,
                               component: str = "FileSystem") -> None:
    """Handle file operation errors with appropriate error types"""
    if isinstance(error, FileNotFoundError):
        raise FileNotFoundError(f"File not found during {operation}: {file_path}",
                               context={"operation": operation, "file_path": file_path})
    elif isinstance(error, PermissionError):
        raise FilePermissionError(f"Permission denied during {operation}: {file_path}",
                                 context={"operation": operation, "file_path": file_path})
    else:
        raise DataLoadError(f"File operation failed during {operation}: {file_path}",
                           context={"operation": operation, "file_path": file_path, "original_error": str(error)})


def handle_network_error(operation: str, url: str = None, timeout: float = None,
                        error: Exception = None, component: str = "Network") -> None:
    """Handle network-related errors"""
    context = {"operation": operation}
    if url:
        context["url"] = url
    if timeout:
        context["timeout"] = timeout

    if isinstance(error, TimeoutError) or "timeout" in str(error).lower():
        raise NetworkTimeoutError(f"Network timeout during {operation}",
                                 context=context)
    else:
        raise NetworkConnectionError(f"Network connection failed during {operation}",
                                    context=context)


def validate_data_quality(df, required_columns: List[str] = None,
                         min_rows: int = 1, component: str = "DataValidation") -> None:
    """Validate DataFrame quality and raise appropriate errors"""
    if not PANDAS_AVAILABLE:
        return  # Skip validation if pandas not available

    if df is None:
        raise DataValidationError("DataFrame is None",
                                 context={"validation_type": "null_check"})

    if df.empty:
        raise DataValidationError("DataFrame is empty",
                                 context={"validation_type": "empty_check"})

    if len(df) < min_rows:
        raise DataValidationError(f"DataFrame has insufficient rows: {len(df)} < {min_rows}",
                                 context={"validation_type": "row_count", "actual_rows": len(df), "required_rows": min_rows})

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}",
                                     context={"validation_type": "column_check", "missing_columns": missing_cols})


def with_performance_monitoring(component: str = "Unknown"):
    """
    Decorator for performance monitoring

    Args:
        component: Component name for logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            correlation_id = str(uuid.uuid4())[:8]

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                if STREAMLIT_AVAILABLE and 'error_logger' in st.session_state:
                    st.session_state.error_logger.log_performance(component, func.__name__, duration, True)

                return result
            except Exception as e:
                duration = time.time() - start_time
                if STREAMLIT_AVAILABLE and 'error_logger' in st.session_state:
                    st.session_state.error_logger.log_performance(component, func.__name__, duration, False)
                raise e

        return wrapper
    return decorator