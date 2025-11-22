"""
Performance Optimization Module
Caching, memory management, and performance monitoring for stock scanner
"""

import time
import psutil
import threading
from functools import wraps, lru_cache
from typing import Dict, Any, Optional, Callable, List
import pandas as pd
import logging
from datetime import datetime, timedelta
import gc
import weakref

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)

# Performance monitoring configuration
PERFORMANCE_CONFIG = {
    'enable_monitoring': True,
    'cache_ttl_seconds': 3600,  # 1 hour
    'memory_threshold_mb': 500,
    'cpu_threshold_percent': 80,
    'max_cache_size': 1000,
    'cleanup_interval_seconds': 300,  # 5 minutes
}

# Global performance metrics
_performance_metrics = {
    'operation_times': {},
    'memory_usage': [],
    'cache_hits': {},
    'cache_misses': {},
    'error_counts': {},
    'last_cleanup': datetime.now()
}

# Thread-safe locks
_cache_lock = threading.RLock()
_metrics_lock = threading.RLock()

class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self):
        self.start_time = None
        self.operation_name = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and self.operation_name:
            duration = time.time() - self.start_time
            with _metrics_lock:
                if self.operation_name not in _performance_metrics['operation_times']:
                    _performance_metrics['operation_times'][self.operation_name] = []
                _performance_metrics['operation_times'][self.operation_name].append(duration)

                # Keep only last 100 measurements per operation
                if len(_performance_metrics['operation_times'][self.operation_name]) > 100:
                    _performance_metrics['operation_times'][self.operation_name] = \
                        _performance_metrics['operation_times'][self.operation_name][-100:]

    def set_operation(self, name: str):
        """Set the operation name for monitoring"""
        self.operation_name = name

def performance_monitor(operation_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            with PerformanceMonitor() as monitor:
                monitor.set_operation(op_name)
                return func(*args, **kwargs)
        return wrapper
    return decorator

class SmartCache:
    """Intelligent caching with TTL and memory management"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}
        self.hit_counts = {}
        self.miss_counts = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if valid"""
        with _cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                if self._is_valid(entry):
                    self.access_times[key] = datetime.now()
                    self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
                    return entry['value']
                else:
                    # Remove expired entry
                    del self.cache[key]
                    del self.access_times[key]

            self.miss_counts[key] = self.miss_counts.get(key, 0) + 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value"""
        with _cache_lock:
            # Clean up if cache is full
            if len(self.cache) >= self.max_size:
                self._cleanup_old_entries()

            self.cache[key] = {
                'value': value,
                'timestamp': datetime.now()
            }
            self.access_times[key] = datetime.now()

    def _is_valid(self, entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in entry:
            return False
        age = (datetime.now() - entry['timestamp']).total_seconds()
        return age < self.ttl_seconds

    def _cleanup_old_entries(self) -> None:
        """Remove least recently used entries when cache is full"""
        if not self.access_times:
            return

        # Sort by access time (oldest first)
        sorted_keys = sorted(self.access_times.keys(),
                           key=lambda k: self.access_times[k])

        # Remove oldest 20% of entries
        remove_count = max(1, int(len(sorted_keys) * 0.2))
        for key in sorted_keys[:remove_count]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]

    def clear(self) -> None:
        """Clear all cached data"""
        with _cache_lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_counts.clear()
            self.miss_counts.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with _cache_lock:
            total_hits = sum(self.hit_counts.values())
            total_misses = sum(self.miss_counts.values())
            hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'ttl_seconds': self.ttl_seconds
            }

# Global cache instances
_data_cache = SmartCache(max_size=500, ttl_seconds=1800)  # 30 minutes for data
_computation_cache = SmartCache(max_size=200, ttl_seconds=3600)  # 1 hour for computations
_report_cache = SmartCache(max_size=100, ttl_seconds=7200)  # 2 hours for reports

def cached_data_operation(cache_key_prefix: str = "data"):
    """Decorator for caching data operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{cache_key_prefix}:{func.__name__}:{str(args) if args else 'no_args'}:{str(kwargs) if kwargs else 'no_kwargs'}"

            # Try to get from cache first
            cached_result = _data_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache the result
            _data_cache.set(cache_key, result)
            logger.debug(f"Cached result for {cache_key}")

            return result
        return wrapper
    return decorator

def cached_computation(operation_name: str = None):
    """Decorator for caching expensive computations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            cache_key = f"computation:{op_name}:{hash(str(args) + str(kwargs))}"

            cached_result = _computation_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            _computation_cache.set(cache_key, result)

            return result
        return wrapper
    return decorator

class MemoryManager:
    """Memory usage monitoring and optimization"""

    def __init__(self):
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
        self.baseline_memory = self.get_memory_usage()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def check_memory_pressure(self) -> bool:
        """Check if memory usage is high"""
        current_memory = self.get_memory_usage()
        return current_memory > PERFORMANCE_CONFIG['memory_threshold_mb']

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        if df.empty:
            return df

        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']):
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']):
            df[col] = pd.to_numeric(df[col], downcast='float')

        # Convert object columns to category if appropriate
        for col in df.select_dtypes(include=['object']):
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        return df

    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory"""
        gc.collect()
        logger.info("Forced garbage collection")

# Global memory manager
_memory_manager = MemoryManager()

def optimize_memory_usage():
    """Optimize memory usage across the application"""
    if _memory_manager.check_memory_pressure():
        logger.warning("High memory usage detected, optimizing...")

        # Clear caches if memory is high
        _data_cache.clear()
        _computation_cache.clear()

        # Force garbage collection
        _memory_manager.force_garbage_collection()

        # Log memory usage
        current_memory = _memory_manager.get_memory_usage()
        logger.info(f"Memory optimized. Current usage: {current_memory:.1f} MB")

def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    # Ensure cleanup thread is started (lazy initialization)
    ensure_cleanup_thread()

    with _metrics_lock:
        # Calculate averages for operation times
        operation_stats = {}
        for op_name, times in _performance_metrics['operation_times'].items():
            if times:
                operation_stats[op_name] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'last_time': times[-1]
                }

        return {
            'memory_usage_mb': _memory_manager.get_memory_usage(),
            'operation_stats': operation_stats,
            'cache_stats': {
                'data_cache': _data_cache.get_stats(),
                'computation_cache': _computation_cache.get_stats(),
                'report_cache': _report_cache.get_stats()
            },
            'error_counts': _performance_metrics['error_counts'].copy(),
            'last_cleanup': _performance_metrics['last_cleanup'].isoformat()
        }

def cleanup_performance_data():
    """Clean up old performance data"""
    with _metrics_lock:
        current_time = datetime.now()

        # Clean up old operation times (keep only last 24 hours worth)
        cutoff_time = current_time - timedelta(hours=24)
        _performance_metrics['last_cleanup'] = current_time

        # Clean up old memory usage data
        if len(_performance_metrics['memory_usage']) > 1000:
            _performance_metrics['memory_usage'] = _performance_metrics['memory_usage'][-500:]

        logger.info("Performance data cleaned up")

# Background cleanup thread
def _start_cleanup_thread():
    """Start background cleanup thread"""
    def cleanup_worker():
        while True:
            time.sleep(PERFORMANCE_CONFIG['cleanup_interval_seconds'])
            try:
                cleanup_performance_data()
                optimize_memory_usage()
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")

    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()

# Lazy initialization for cleanup thread (prevents circular import deadlock)
_cleanup_thread_started = False
_thread_lock = threading.RLock()

def ensure_cleanup_thread():
    """Ensure cleanup thread is started (lazy initialization)"""
    global _cleanup_thread_started
    with _thread_lock:
        if not _cleanup_thread_started:
            _start_cleanup_thread()
            _cleanup_thread_started = True

# Remove auto-start at module level to prevent circular import deadlock
# _start_cleanup_thread()  # REMOVED: This was causing deadlock during import

logger.info("Performance optimization module initialized")
