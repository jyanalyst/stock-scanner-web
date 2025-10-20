"""
Scanner-specific Constants
Defines progress milestones, scan scopes, and other scanner constants
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanProgress:
    """Progress milestones for scanner operations"""
    INIT = 0.1
    LOAD_DATA = 0.3
    CALCULATE = 0.4
    PROCESS_STOCKS_START = 0.4
    PROCESS_STOCKS_END = 0.8
    PREPARE_RESULTS = 0.8
    ANALYST_REPORTS = 0.85
    EARNINGS_REPORTS = 0.9
    COMPLETE = 1.0


@dataclass(frozen=True)
class ScanScope:
    """Scan scope options"""
    SINGLE_STOCK = "Single Stock"
    FULL_WATCHLIST = "Full Watchlist"


@dataclass(frozen=True)
class ScanDateType:
    """Scan date type options"""
    CURRENT = "Current Date"
    HISTORICAL = "Historical Date"


@dataclass(frozen=True)
class UpdateProgress:
    """Progress milestones for update operations"""
    INIT = 0.0
    LOAD_EOD = 0.1
    PARSE_DATE = 0.15
    PROCESS_STOCKS_START = 0.2
    PROCESS_STOCKS_END = 0.9
    CLEANUP = 0.95
    COMPLETE = 1.0


@dataclass(frozen=True)
class UpdateStatus:
    """Status values for update operations"""
    UPDATED = 'updated'
    CREATED = 'created'
    SKIPPED = 'skipped'
    ERROR = 'error'
    FAILED = 'failed'
    
    # Emoji mapping
    EMOJIS = {
        'updated': '✅',
        'created': '🆕',
        'skipped': '⏭️',
        'error': '❌',
        'failed': '❌'
    }
    
    @classmethod
    def get_emoji(cls, status: str) -> str:
        """Get emoji for status"""
        return cls.EMOJIS.get(status, '❓')


# Column display widths for dataframes
@dataclass(frozen=True)
class ColumnWidth:
    """Standard column widths for dataframe display"""
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'
