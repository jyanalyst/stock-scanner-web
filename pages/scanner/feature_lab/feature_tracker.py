"""
FeatureTracker - Historical Selection Tracking for Style Learning

Manages the recording and retrieval of historical winner selections to learn
trading style patterns and optimize scoring weights.
"""

import json
import shutil
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureTracker:
    """
    Manages historical selection tracking for style learning

    Primary Responsibilities:
    - Record manual winner selections for historical dates
    - Persist data to JSON with automatic backups
    - Retrieve selection history with filtering
    - Calculate summary statistics
    """

    def __init__(self, data_dir: str = "data/feature_lab"):
        """Initialize with data directory path"""
        self.data_dir = Path(data_dir)
        self.selection_file = self.data_dir / "selection_history.json"
        self.backups_dir = self.data_dir / "backups"
        self._ensure_directories()
        self._initialize_files()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_files(self) -> None:
        """Initialize JSON files if they don't exist"""
        if not self.selection_file.exists():
            self._create_initial_selection_history()

    def _create_initial_selection_history(self) -> None:
        """Create initial selection_history.json file"""
        initial_data = {
            "version": "1.0",
            "created_date": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "dates": {},
            "summary": {
                "total_dates_labeled": 0,
                "total_winners_selected": 0,
                "bullish_winners": 0,
                "bearish_winners": 0,
                "date_range": {
                    "earliest": None,
                    "latest": None
                },
                "completion_progress": {
                    "current": 0,
                    "target": 90,
                    "percentage": 0.0
                }
            }
        }

        with open(self.selection_file, 'w') as f:
            json.dump(initial_data, f, indent=2)

    def _create_backup(self) -> str:
        """Create backup of current selection_history.json"""
        if not self.selection_file.exists():
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"selection_history_backup_{timestamp}.json"
        backup_path = self.backups_dir / backup_filename

        shutil.copy(self.selection_file, backup_path)
        return str(backup_path)

    def _load_selection_history(self) -> Dict[str, Any]:
        """Load selection history from JSON file"""
        try:
            with open(self.selection_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load selection history: {e}")
            return self._create_initial_selection_history()

    def _save_selection_history(self, data: Dict[str, Any]) -> bool:
        """Save selection history to JSON file"""
        try:
            # Update last modified timestamp
            data["last_modified"] = datetime.now().isoformat()

            with open(self.selection_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save selection history: {e}")
            return False

    def _extract_features_from_scan_results(self, scan_results: Any) -> Dict[str, Any]:
        """
        Extract relevant features from scan results for storage

        Args:
            scan_results: Can be pandas DataFrame or list of dicts

        Returns:
            Dict mapping ticker to feature values
        """
        features = {}

        try:
            # Handle pandas DataFrame
            if hasattr(scan_results, 'to_dict'):
                df_dict = scan_results.to_dict('index')
                for idx, row in df_dict.items():
                    ticker = row.get('Ticker', str(idx))
                    features[ticker] = {
                        "Flow_Velocity_Rank": row.get('Flow_Velocity_Rank', 50.0),
                        "Flow_Rank": row.get('Flow_Rank', 50.0),
                        "Flow_Percentile": row.get('Flow_Percentile', 50.0),
                        "Volume_Conviction": row.get('Volume_Conviction', 1.0),
                        "MPI_Percentile": row.get('MPI_Percentile', 50.0),
                        "IBS_Percentile": row.get('IBS_Percentile', 50.0),
                        "VPI_Percentile": row.get('VPI_Percentile', 50.0),
                        "Signal_Bias": row.get('Signal_Bias', '⚪ NEUTRAL'),
                        "Signal_Score": row.get('Signal_Score', 50.0),
                        "Trade_Rank": row.get('Trade_Rank', 1),
                        "is_winner": False  # Will be updated based on selections
                    }

            # Handle list of dicts
            elif isinstance(scan_results, list):
                for row in scan_results:
                    ticker = row.get('Ticker', 'UNKNOWN')
                    features[ticker] = {
                        "Flow_Velocity_Rank": row.get('Flow_Velocity_Rank', 50.0),
                        "Flow_Rank": row.get('Flow_Rank', 50.0),
                        "Flow_Percentile": row.get('Flow_Percentile', 50.0),
                        "Volume_Conviction": row.get('Volume_Conviction', 1.0),
                        "MPI_Percentile": row.get('MPI_Percentile', 50.0),
                        "IBS_Percentile": row.get('IBS_Percentile', 50.0),
                        "VPI_Percentile": row.get('VPI_Percentile', 50.0),
                        "Signal_Bias": row.get('Signal_Bias', '⚪ NEUTRAL'),
                        "Signal_Score": row.get('Signal_Score', 50.0),
                        "Trade_Rank": row.get('Trade_Rank', 1),
                        "is_winner": False
                    }

        except Exception as e:
            logger.error(f"Failed to extract features from scan results: {e}")

        return features

    def _update_summary_statistics(self, data: Dict[str, Any]) -> None:
        """Update summary statistics in the data dict"""
        dates = data.get("dates", {})
        total_dates = len(dates)

        bullish_winners = 0
        bearish_winners = 0
        total_winners = 0

        earliest_date = None
        latest_date = None

        for date_str, date_data in dates.items():
            # Count winners
            bullish_winners += len(date_data.get("bullish_winners", []))
            bearish_winners += len(date_data.get("bearish_winners", []))
            total_winners += bullish_winners + bearish_winners

            # Track date range
            try:
                current_date = datetime.fromisoformat(date_str).date()
                if earliest_date is None or current_date < earliest_date:
                    earliest_date = current_date
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date
            except ValueError:
                continue

        # Update summary
        data["summary"] = {
            "total_dates_labeled": total_dates,
            "total_winners_selected": total_winners,
            "bullish_winners": bullish_winners,
            "bearish_winners": bearish_winners,
            "date_range": {
                "earliest": earliest_date.isoformat() if earliest_date else None,
                "latest": latest_date.isoformat() if latest_date else None
            },
            "completion_progress": {
                "current": total_dates,
                "target": 90,
                "percentage": round((total_dates / 90) * 100, 1) if total_dates > 0 else 0.0
            }
        }

    def record_historical_winners(
        self,
        scan_date: date,
        scan_results: Any,
        bullish_winners: List[str],
        bearish_winners: List[str],
        selection_notes: str,
        scoring_system: str = "production"
    ) -> bool:
        """
        Record winner selections for a historical date

        Args:
            scan_date: The historical date scanned
            scan_results: Full scanner results (DataFrame or list)
            bullish_winners: List of bullish winner tickers
            bearish_winners: List of bearish winner tickers
            selection_notes: Your commentary on why these worked
            scoring_system: "production" or "candidate"

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create backup
            backup_path = self._create_backup()
            if backup_path:
                logger.info(f"Created backup: {backup_path}")

            # Load current data
            data = self._load_selection_history()

            # Extract features from scan results
            features = self._extract_features_from_scan_results(scan_results)

            # Mark winners
            all_winners = set(bullish_winners + bearish_winners)
            for ticker in features:
                features[ticker]["is_winner"] = ticker in all_winners

            # Create date entry
            date_str = scan_date.isoformat()
            date_entry = {
                "scan_results": features,
                "bullish_winners": bullish_winners,
                "bearish_winners": bearish_winners,
                "selection_notes": selection_notes,
                "timestamp": datetime.now().isoformat(),
                "scoring_system": scoring_system,
                "metadata": {
                    "total_signals": len(features),
                    "bullish_signals": sum(1 for f in features.values()
                                         if '🟢 BULLISH' in str(f.get('Signal_Bias', ''))),
                    "bearish_signals": sum(1 for f in features.values()
                                         if '🔴 BEARISH' in str(f.get('Signal_Bias', '')))
                }
            }

            # Store in dates
            data["dates"][date_str] = date_entry

            # Update summary statistics
            self._update_summary_statistics(data)

            # Save
            success = self._save_selection_history(data)

            if success:
                logger.info(f"Successfully recorded winners for {date_str}: "
                          f"{len(bullish_winners)} bullish, {len(bearish_winners)} bearish")
                return True
            else:
                logger.error("Failed to save selection history")
                return False

        except Exception as e:
            logger.error(f"Failed to record historical winners: {e}")
            return False

    def get_selection_history(
        self,
        days_back: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Retrieve selection history with optional filtering

        Args:
            days_back: Number of days back from today
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Filtered selection history dict
        """
        data = self._load_selection_history()

        if not days_back and not start_date and not end_date:
            return data

        # Apply date filtering
        filtered_dates = {}

        for date_str, date_data in data["dates"].items():
            try:
                entry_date = datetime.fromisoformat(date_str).date()

                # Check date filters
                include_entry = True

                if days_back:
                    cutoff_date = date.today() - timedelta(days=days_back)
                    if entry_date < cutoff_date:
                        include_entry = False

                if start_date and entry_date < start_date:
                    include_entry = False

                if end_date and entry_date > end_date:
                    include_entry = False

                if include_entry:
                    filtered_dates[date_str] = date_data

            except ValueError:
                continue

        # Return filtered data
        filtered_data = data.copy()
        filtered_data["dates"] = filtered_dates

        # Update summary for filtered data
        self._update_summary_statistics(filtered_data)

        return filtered_data

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get current summary statistics"""
        data = self._load_selection_history()
        return data.get("summary", {})

    def delete_date_selection(self, scan_date: date) -> bool:
        """
        Remove a historical selection (for corrections)

        Args:
            scan_date: Date to remove

        Returns:
            True if deleted successfully
        """
        try:
            # Create backup
            backup_path = self._create_backup()
            if backup_path:
                logger.info(f"Created backup before deletion: {backup_path}")

            # Load data
            data = self._load_selection_history()

            # Remove date
            date_str = scan_date.isoformat()
            if date_str in data["dates"]:
                del data["dates"][date_str]

                # Update summary
                self._update_summary_statistics(data)

                # Save
                success = self._save_selection_history(data)

                if success:
                    logger.info(f"Successfully deleted selection for {date_str}")
                    return True
                else:
                    logger.error("Failed to save after deletion")
                    return False
            else:
                logger.warning(f"Date {date_str} not found in selection history")
                return False

        except Exception as e:
            logger.error(f"Failed to delete date selection: {e}")
            return False

    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export selection history to CSV for external analysis

        Args:
            output_path: Optional output path, defaults to timestamped file

        Returns:
            Path to exported CSV file
        """
        import pandas as pd

        try:
            data = self._load_selection_history()

            # Flatten the nested structure
            rows = []

            for date_str, date_data in data["dates"].items():
                scan_results = date_data.get("scan_results", {})
                bullish_winners = date_data.get("bullish_winners", [])
                bearish_winners = date_data.get("bearish_winners", [])
                notes = date_data.get("selection_notes", "")
                timestamp = date_data.get("timestamp", "")

                for ticker, features in scan_results.items():
                    is_bullish_winner = ticker in bullish_winners
                    is_bearish_winner = ticker in bearish_winners

                    row = {
                        "date": date_str,
                        "ticker": ticker,
                        "is_bullish_winner": is_bullish_winner,
                        "is_bearish_winner": is_bearish_winner,
                        "selection_notes": notes,
                        "timestamp": timestamp,
                        **features  # Include all feature values
                    }
                    rows.append(row)

            if not rows:
                logger.warning("No data to export")
                return ""

            df = pd.DataFrame(rows)

            # Generate output path
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"selection_history_export_{timestamp}.csv"

            # Export to CSV
            df.to_csv(output_path, index=False)

            logger.info(f"Exported {len(rows)} rows to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return ""
