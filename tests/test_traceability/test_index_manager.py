"""Tests for index manager."""

import json
from datetime import datetime

import pytest

from vlm_geometry_bench.traceability.index_manager import IndexManager
from vlm_geometry_bench.traceability.schemas import RunIndex, RunIndexEntry


class TestIndexManager:
    """Tests for IndexManager class."""

    @pytest.fixture
    def temp_results_dir(self, tmp_path):
        """Create temporary results directory."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        return results_dir

    @pytest.fixture
    def manager(self, temp_results_dir):
        """Create IndexManager."""
        return IndexManager(str(temp_results_dir))

    @pytest.fixture
    def sample_entry(self):
        """Create sample RunIndexEntry."""
        return RunIndexEntry(
            run_id="20260202_130859_test",
            run_name="test",
            comment="Test comment",
            models=["llava:7b"],
            backends=["ollama"],
            timestamp=datetime(2026, 2, 2, 13, 8, 59),
            elapsed_seconds=100.5,
            total_tests=10,
            estimated_cost_usd=0.0,
            size_mb=5.5,
            tasks=["COUNT"],
            image_classes=["USSS"],
        )

    def test_load_index_nonexistent(self, manager):
        """Returns empty index if file doesn't exist."""
        index = manager.load_index()
        assert isinstance(index, RunIndex)
        assert index.runs == []

    def test_save_and_load_index(self, manager, sample_entry):
        """Can save and load index."""
        index = RunIndex(runs=[sample_entry])
        manager.save_index(index)

        loaded = manager.load_index()
        assert len(loaded.runs) == 1
        assert loaded.runs[0].run_id == "20260202_130859_test"

    def test_save_creates_directory(self, tmp_path):
        """Save creates results directory if needed."""
        manager = IndexManager(str(tmp_path / "new" / "results"))
        index = RunIndex()
        manager.save_index(index)

        assert manager.index_path.exists()

    def test_load_corrupted_index(self, manager):
        """Returns empty index if file is corrupted."""
        manager.index_path.write_text("not valid json {{{")

        index = manager.load_index()
        assert index.runs == []

    def test_add_run(self, manager, sample_entry):
        """Adds run to index."""
        manager.add_run(sample_entry)

        index = manager.load_index()
        assert len(index.runs) == 1
        assert index.runs[0].run_id == sample_entry.run_id

    def test_add_run_replaces_existing(self, manager, sample_entry):
        """Adding run with same ID replaces existing."""
        manager.add_run(sample_entry)

        # Modify and re-add
        updated = RunIndexEntry(
            run_id=sample_entry.run_id,
            run_name="updated",
            models=["gpt-4o"],
            backends=["openrouter"],
            timestamp=sample_entry.timestamp,
            elapsed_seconds=200.0,
            total_tests=20,
            tasks=["LOCATE"],
            image_classes=["CTRL"],
        )
        manager.add_run(updated)

        index = manager.load_index()
        assert len(index.runs) == 1
        assert index.runs[0].run_name == "updated"
        assert index.runs[0].elapsed_seconds == 200.0

    def test_add_run_most_recent_first(self, manager):
        """New runs added at beginning of list."""
        entry1 = RunIndexEntry(
            run_id="run1",
            run_name="first",
            models=["llava:7b"],
            backends=["ollama"],
            timestamp=datetime(2026, 1, 1),
            elapsed_seconds=100.0,
            total_tests=10,
            tasks=["COUNT"],
            image_classes=["USSS"],
        )
        entry2 = RunIndexEntry(
            run_id="run2",
            run_name="second",
            models=["llava:7b"],
            backends=["ollama"],
            timestamp=datetime(2026, 1, 2),
            elapsed_seconds=100.0,
            total_tests=10,
            tasks=["COUNT"],
            image_classes=["USSS"],
        )

        manager.add_run(entry1)
        manager.add_run(entry2)

        index = manager.load_index()
        assert index.runs[0].run_id == "run2"  # Most recent first
        assert index.runs[1].run_id == "run1"

    def test_update_run(self, manager, sample_entry):
        """Updates existing run."""
        manager.add_run(sample_entry)

        manager.update_run(
            sample_entry.run_id,
            elapsed_seconds=500.0,
            total_tests=50,
            estimated_cost_usd=1.25,
            size_mb=25.0,
        )

        index = manager.load_index()
        entry = index.runs[0]
        assert entry.elapsed_seconds == 500.0
        assert entry.total_tests == 50
        assert entry.estimated_cost_usd == 1.25
        assert entry.size_mb == 25.0

    def test_update_run_partial(self, manager, sample_entry):
        """Partial update only modifies specified fields."""
        manager.add_run(sample_entry)

        manager.update_run(
            sample_entry.run_id,
            elapsed_seconds=500.0,
        )

        index = manager.load_index()
        entry = index.runs[0]
        assert entry.elapsed_seconds == 500.0
        assert entry.total_tests == sample_entry.total_tests  # Unchanged

    def test_update_nonexistent_run(self, manager, sample_entry):
        """Updating nonexistent run does nothing."""
        manager.add_run(sample_entry)

        manager.update_run("nonexistent", elapsed_seconds=500.0)

        index = manager.load_index()
        assert index.runs[0].elapsed_seconds == sample_entry.elapsed_seconds

    def test_remove_run(self, manager, sample_entry):
        """Removes run from index."""
        manager.add_run(sample_entry)

        result = manager.remove_run(sample_entry.run_id)

        assert result is True
        index = manager.load_index()
        assert len(index.runs) == 0

    def test_remove_nonexistent_run(self, manager, sample_entry):
        """Removing nonexistent run returns False."""
        manager.add_run(sample_entry)

        result = manager.remove_run("nonexistent")

        assert result is False
        index = manager.load_index()
        assert len(index.runs) == 1

    def test_get_run(self, manager, sample_entry):
        """Gets specific run by ID."""
        manager.add_run(sample_entry)

        entry = manager.get_run(sample_entry.run_id)

        assert entry is not None
        assert entry.run_id == sample_entry.run_id

    def test_get_run_nonexistent(self, manager, sample_entry):
        """Returns None for nonexistent run."""
        manager.add_run(sample_entry)

        entry = manager.get_run("nonexistent")

        assert entry is None

    def test_list_runs_no_filter(self, manager):
        """Lists all runs without filter."""
        entries = [
            RunIndexEntry(
                run_id=f"run{i}",
                run_name=f"run{i}",
                models=["llava:7b"] if i % 2 == 0 else ["gpt-4o"],
                backends=["ollama"] if i % 2 == 0 else ["openrouter"],
                timestamp=datetime(2026, 1, i + 1),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT"] if i % 2 == 0 else ["LOCATE"],
                image_classes=["USSS"],
            )
            for i in range(4)
        ]
        for entry in entries:
            manager.add_run(entry)

        runs = manager.list_runs()
        assert len(runs) == 4

    def test_list_runs_filter_model(self, manager):
        """Filters by model name."""
        entries = [
            RunIndexEntry(
                run_id="run1",
                run_name="run1",
                models=["llava:7b"],
                backends=["ollama"],
                timestamp=datetime(2026, 1, 1),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT"],
                image_classes=["USSS"],
            ),
            RunIndexEntry(
                run_id="run2",
                run_name="run2",
                models=["gpt-4o"],
                backends=["openrouter"],
                timestamp=datetime(2026, 1, 2),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT"],
                image_classes=["USSS"],
            ),
        ]
        for entry in entries:
            manager.add_run(entry)

        runs = manager.list_runs(model_filter="llava")
        assert len(runs) == 1
        assert runs[0].run_id == "run1"

    def test_list_runs_filter_backend(self, manager):
        """Filters by backend."""
        entries = [
            RunIndexEntry(
                run_id="run1",
                run_name="run1",
                models=["llava:7b"],
                backends=["ollama"],
                timestamp=datetime(2026, 1, 1),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT"],
                image_classes=["USSS"],
            ),
            RunIndexEntry(
                run_id="run2",
                run_name="run2",
                models=["gpt-4o"],
                backends=["openrouter"],
                timestamp=datetime(2026, 1, 2),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT"],
                image_classes=["USSS"],
            ),
        ]
        for entry in entries:
            manager.add_run(entry)

        runs = manager.list_runs(backend_filter="openrouter")
        assert len(runs) == 1
        assert runs[0].run_id == "run2"

    def test_list_runs_filter_task(self, manager):
        """Filters by task."""
        entries = [
            RunIndexEntry(
                run_id="run1",
                run_name="run1",
                models=["llava:7b"],
                backends=["ollama"],
                timestamp=datetime(2026, 1, 1),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT", "LOCATE"],
                image_classes=["USSS"],
            ),
            RunIndexEntry(
                run_id="run2",
                run_name="run2",
                models=["llava:7b"],
                backends=["ollama"],
                timestamp=datetime(2026, 1, 2),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["PATTERN"],
                image_classes=["USSS"],
            ),
        ]
        for entry in entries:
            manager.add_run(entry)

        runs = manager.list_runs(task_filter="LOCATE")
        assert len(runs) == 1
        assert runs[0].run_id == "run1"

    def test_list_runs_multiple_filters(self, manager):
        """Applies multiple filters."""
        entries = [
            RunIndexEntry(
                run_id="run1",
                run_name="run1",
                models=["llava:7b"],
                backends=["ollama"],
                timestamp=datetime(2026, 1, 1),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT"],
                image_classes=["USSS"],
            ),
            RunIndexEntry(
                run_id="run2",
                run_name="run2",
                models=["llava:7b"],
                backends=["openrouter"],
                timestamp=datetime(2026, 1, 2),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT"],
                image_classes=["USSS"],
            ),
            RunIndexEntry(
                run_id="run3",
                run_name="run3",
                models=["gpt-4o"],
                backends=["ollama"],
                timestamp=datetime(2026, 1, 3),
                elapsed_seconds=100.0,
                total_tests=10,
                tasks=["COUNT"],
                image_classes=["USSS"],
            ),
        ]
        for entry in entries:
            manager.add_run(entry)

        runs = manager.list_runs(model_filter="llava", backend_filter="ollama")
        assert len(runs) == 1
        assert runs[0].run_id == "run1"

    def test_create_entry_from_run(self, manager):
        """Creates RunIndexEntry from parameters."""
        entry = manager.create_entry_from_run(
            run_id="20260202_130859_test",
            run_name="test",
            comment="Test run",
            models=["llava:7b", "gpt-4o"],
            backends=["ollama", "openrouter"],
            timestamp=datetime(2026, 2, 2, 13, 8, 59),
            elapsed_seconds=500.0,
            total_tests=184,
            estimated_cost_usd=2.50,
            size_mb=100.5,
            tasks=["COUNT", "LOCATE"],
            image_classes=["USSS", "CTRL"],
        )

        assert entry.run_id == "20260202_130859_test"
        assert entry.comment == "Test run"
        assert len(entry.models) == 2
        assert entry.estimated_cost_usd == 2.50
