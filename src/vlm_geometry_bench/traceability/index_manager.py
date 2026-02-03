"""Index manager for the master index.json file.

Handles reading, writing, and updating the global index of all test runs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .schemas import RunIndex, RunIndexEntry


class IndexManager:
    """Manages the master index.json file listing all test runs."""

    def __init__(self, results_dir: str):
        """Initialize index manager.

        Args:
            results_dir: Base results directory containing index.json
        """
        self.results_dir = Path(results_dir)
        self.index_path = self.results_dir / "index.json"

    def load_index(self) -> RunIndex:
        """Load the index from disk.

        Creates a new empty index if file doesn't exist.

        Returns:
            RunIndex object
        """
        if not self.index_path.exists():
            return RunIndex()

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RunIndex.model_validate(data)
        except (json.JSONDecodeError, Exception):
            # If corrupt, return empty index
            return RunIndex()

    def save_index(self, index: RunIndex) -> None:
        """Save the index to disk.

        Args:
            index: RunIndex object to save
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index.model_dump(mode="json"), f, indent=2, default=str)

    def add_run(self, entry: RunIndexEntry) -> None:
        """Add a new run entry to the index.

        If a run with the same ID already exists, it will be replaced.

        Args:
            entry: RunIndexEntry to add
        """
        index = self.load_index()

        # Remove existing entry with same ID if present
        index.runs = [r for r in index.runs if r.run_id != entry.run_id]

        # Add new entry at the beginning (most recent first)
        index.runs.insert(0, entry)

        self.save_index(index)

    def update_run(
        self,
        run_id: str,
        elapsed_seconds: Optional[float] = None,
        total_tests: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
        size_mb: Optional[float] = None,
    ) -> None:
        """Update an existing run entry.

        Args:
            run_id: Run ID to update
            elapsed_seconds: Updated elapsed time
            total_tests: Updated test count
            estimated_cost_usd: Updated cost estimate
            size_mb: Updated size
        """
        index = self.load_index()

        for entry in index.runs:
            if entry.run_id == run_id:
                if elapsed_seconds is not None:
                    entry.elapsed_seconds = elapsed_seconds
                if total_tests is not None:
                    entry.total_tests = total_tests
                if estimated_cost_usd is not None:
                    entry.estimated_cost_usd = estimated_cost_usd
                if size_mb is not None:
                    entry.size_mb = size_mb
                break

        self.save_index(index)

    def remove_run(self, run_id: str) -> bool:
        """Remove a run from the index.

        Args:
            run_id: Run ID to remove

        Returns:
            True if run was found and removed, False otherwise
        """
        index = self.load_index()
        original_count = len(index.runs)

        index.runs = [r for r in index.runs if r.run_id != run_id]

        if len(index.runs) < original_count:
            self.save_index(index)
            return True
        return False

    def get_run(self, run_id: str) -> Optional[RunIndexEntry]:
        """Get a specific run entry.

        Args:
            run_id: Run ID to find

        Returns:
            RunIndexEntry if found, None otherwise
        """
        index = self.load_index()
        for entry in index.runs:
            if entry.run_id == run_id:
                return entry
        return None

    def list_runs(
        self,
        model_filter: Optional[str] = None,
        backend_filter: Optional[str] = None,
        task_filter: Optional[str] = None,
    ) -> List[RunIndexEntry]:
        """List runs with optional filters.

        Args:
            model_filter: Filter by model name (substring match)
            backend_filter: Filter by backend
            task_filter: Filter by task

        Returns:
            List of matching RunIndexEntry objects
        """
        index = self.load_index()
        runs = index.runs

        if model_filter:
            model_lower = model_filter.lower()
            runs = [r for r in runs if any(model_lower in m.lower() for m in r.models)]

        if backend_filter:
            backend_lower = backend_filter.lower()
            runs = [r for r in runs if any(backend_lower == b.lower() for b in r.backends)]

        if task_filter:
            task_upper = task_filter.upper()
            runs = [r for r in runs if task_upper in r.tasks]

        return runs

    def create_entry_from_run(
        self,
        run_id: str,
        run_name: str,
        comment: Optional[str],
        models: List[str],
        backends: List[str],
        timestamp: datetime,
        elapsed_seconds: float,
        total_tests: int,
        estimated_cost_usd: Optional[float],
        size_mb: Optional[float],
        tasks: List[str],
        image_classes: List[str],
    ) -> RunIndexEntry:
        """Create a RunIndexEntry from run parameters.

        Args:
            run_id: Unique run identifier
            run_name: User-provided run name
            comment: User comment
            models: List of models evaluated
            backends: List of backends used
            timestamp: Run start time
            elapsed_seconds: Total elapsed time
            total_tests: Total tests across all models
            estimated_cost_usd: Total estimated cost
            size_mb: Total artifact size
            tasks: Tasks evaluated
            image_classes: Image classes evaluated

        Returns:
            RunIndexEntry object
        """
        return RunIndexEntry(
            run_id=run_id,
            run_name=run_name,
            comment=comment,
            models=models,
            backends=backends,
            timestamp=timestamp,
            elapsed_seconds=elapsed_seconds,
            total_tests=total_tests,
            estimated_cost_usd=estimated_cost_usd,
            size_mb=size_mb,
            tasks=tasks,
            image_classes=image_classes,
        )
