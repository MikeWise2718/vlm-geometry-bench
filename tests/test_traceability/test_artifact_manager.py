"""Tests for artifact manager."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from vlm_geometry_bench.traceability.artifact_manager import (
    ArtifactManager,
    safe_model_name,
    safe_run_name,
)


class TestSafeModelName:
    """Tests for safe_model_name function."""

    def test_simple_name(self):
        """Simple name unchanged."""
        assert safe_model_name("llava") == "llava"

    def test_colon_replaced(self):
        """Colons replaced with hyphens."""
        assert safe_model_name("llava:7b") == "llava-7b"

    def test_slash_replaced(self):
        """Slashes replaced with underscores."""
        assert safe_model_name("openai/gpt-4o") == "openai_gpt-4o"

    def test_space_replaced(self):
        """Spaces replaced with underscores."""
        assert safe_model_name("my model") == "my_model"

    def test_complex_name(self):
        """Complex name with multiple special chars."""
        assert safe_model_name("anthropic/claude:v3") == "anthropic_claude-v3"


class TestSafeRunName:
    """Tests for safe_run_name function."""

    def test_simple_name(self):
        """Simple name unchanged."""
        assert safe_run_name("test") == "test"

    def test_spaces_replaced(self):
        """Spaces replaced with hyphens."""
        assert safe_run_name("my test run") == "my-test-run"

    def test_special_chars_removed(self):
        """Special chars removed."""
        assert safe_run_name("test!@#$%") == "test"

    def test_empty_returns_run(self):
        """Empty string returns 'run'."""
        assert safe_run_name("") == "run"
        assert safe_run_name("!@#$") == "run"

    def test_preserves_alphanumeric(self):
        """Alphanumeric and hyphen/underscore preserved."""
        assert safe_run_name("test-run_123") == "test-run_123"


class TestArtifactManager:
    """Tests for ArtifactManager class."""

    @pytest.fixture
    def temp_results_dir(self, tmp_path):
        """Create temporary results directory."""
        return tmp_path / "results"

    @pytest.fixture
    def manager(self, temp_results_dir):
        """Create ArtifactManager with fixed timestamp."""
        timestamp = datetime(2026, 2, 2, 13, 8, 59)
        return ArtifactManager(
            results_dir=str(temp_results_dir),
            run_name="test-run",
            timestamp=timestamp,
        )

    def test_init_creates_run_id(self, manager):
        """Run ID created from timestamp and name."""
        assert manager.run_id == "20260202_130859_test-run"

    def test_init_paths(self, manager, temp_results_dir):
        """Paths initialized correctly."""
        assert manager.run_dir == temp_results_dir / "20260202_130859_test-run"
        assert manager.images_dir == manager.run_dir / "images"
        assert manager.models_dir == manager.run_dir / "models"
        assert manager.assets_dir == manager.run_dir / "assets"

    def test_setup_run_structure(self, manager):
        """Creates folder structure."""
        manager.setup_run_structure()

        assert manager.run_dir.exists()
        assert manager.images_dir.exists()
        assert manager.models_dir.exists()
        assert manager.assets_dir.exists()

    def test_setup_model_structure(self, manager):
        """Creates model folder structure."""
        manager.setup_run_structure()
        model_dir = manager.setup_model_structure("llava:7b")

        assert model_dir.exists()
        assert model_dir.name == "llava-7b"
        assert (model_dir / "tests").exists()

    def test_get_model_dir(self, manager):
        """Gets correct model directory path."""
        model_dir = manager.get_model_dir("openai/gpt-4o")
        assert model_dir.name == "openai_gpt-4o"

    def test_get_test_dir(self, manager):
        """Gets and creates test directory."""
        manager.setup_run_structure()
        manager.setup_model_structure("llava:7b")

        test_dir = manager.get_test_dir("llava:7b", "USSS_s2_d20_wb", "LOCATE")

        assert test_dir.exists()
        assert test_dir.name == "USSS_s2_d20_wb_LOCATE"
        assert test_dir.parent.name == "tests"

    def test_copy_original_image(self, manager, tmp_path):
        """Copies image to shared folder."""
        manager.setup_run_structure()

        # Create test image
        source_path = tmp_path / "source.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(source_path)

        dest_path = manager.copy_original_image(source_path, "test_sample")

        assert dest_path.exists()
        assert dest_path.name == "test_sample.png"
        assert dest_path.parent == manager.images_dir

    def test_copy_original_image_deduplication(self, manager, tmp_path):
        """Same image not copied twice."""
        manager.setup_run_structure()

        source_path = tmp_path / "source.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(source_path)

        path1 = manager.copy_original_image(source_path, "test_sample")
        path2 = manager.copy_original_image(source_path, "test_sample")

        assert path1 == path2

    def test_create_image_symlink_fallback_to_copy(self, manager, tmp_path):
        """Creates symlink or falls back to copy."""
        manager.setup_run_structure()
        manager.setup_model_structure("llava:7b")

        # Copy original image first
        source_path = tmp_path / "source.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(source_path)
        manager.copy_original_image(source_path, "test_sample")

        test_dir = manager.get_test_dir("llava:7b", "test_sample", "LOCATE")
        link_path = manager.create_image_symlink(test_dir, "test_sample")

        assert link_path.exists()
        assert link_path.name == "original.png"

    def test_save_json(self, manager):
        """Saves JSON data."""
        manager.setup_run_structure()

        data = {"key": "value", "number": 42}
        path = manager.run_dir / "test.json"
        manager.save_json(data, path)

        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_json_pydantic(self, manager):
        """Saves Pydantic model as JSON."""
        from vlm_geometry_bench.traceability.schemas import RunIndexEntry

        manager.setup_run_structure()

        entry = RunIndexEntry(
            run_id="test",
            run_name="test",
            models=["llava:7b"],
            backends=["ollama"],
            timestamp=datetime(2026, 2, 2, 13, 8, 59),
            elapsed_seconds=100.5,
            total_tests=10,
            tasks=["COUNT"],
            image_classes=["USSS"],
        )
        path = manager.run_dir / "entry.json"
        manager.save_json(entry, path)

        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["run_id"] == "test"

    def test_save_image(self, manager):
        """Saves PIL image."""
        manager.setup_run_structure()

        img = Image.new("RGB", (100, 100), color="green")
        path = manager.run_dir / "test.png"
        manager.save_image(img, path)

        assert path.exists()
        loaded = Image.open(path)
        assert loaded.size == (100, 100)

    def test_save_html(self, manager):
        """Saves HTML content."""
        manager.setup_run_structure()

        html = "<html><body>Test</body></html>"
        path = manager.run_dir / "test.html"
        manager.save_html(html, path)

        assert path.exists()
        with open(path) as f:
            assert f.read() == html

    def test_calculate_run_size_mb(self, manager):
        """Calculates total run size."""
        manager.setup_run_structure()

        # Create some files
        (manager.run_dir / "test.txt").write_text("x" * 1024)  # 1KB
        (manager.images_dir / "img.txt").write_text("y" * 2048)  # 2KB

        size = manager.calculate_run_size_mb()
        assert size > 0
        assert size < 0.01  # Less than 10KB

    def test_get_all_test_dirs(self, manager):
        """Gets all test directories for a model."""
        manager.setup_run_structure()
        manager.setup_model_structure("llava:7b")

        # Create some test dirs
        manager.get_test_dir("llava:7b", "sample1", "COUNT")
        manager.get_test_dir("llava:7b", "sample2", "LOCATE")
        manager.get_test_dir("llava:7b", "sample3", "COUNT")

        test_dirs = manager.get_all_test_dirs("llava:7b")
        assert len(test_dirs) == 3

    def test_get_all_test_dirs_empty(self, manager):
        """Returns empty list for nonexistent model."""
        dirs = manager.get_all_test_dirs("nonexistent")
        assert dirs == []

    def test_path_helpers(self, manager):
        """Path helper methods return correct paths."""
        assert manager.run_metadata_path() == manager.run_dir / "run_metadata.json"
        assert manager.summary_html_path() == manager.run_dir / "summary.html"
        assert manager.model_metadata_path("llava:7b") == (
            manager.models_dir / "llava-7b" / "model_metadata.json"
        )
        assert manager.test_metadata_path("llava:7b", "sample1", "COUNT") == (
            manager.models_dir / "llava-7b" / "tests" / "sample1_COUNT" / "test_metadata.json"
        )
        assert manager.conversation_path("llava:7b", "sample1", "COUNT") == (
            manager.models_dir / "llava-7b" / "tests" / "sample1_COUNT" / "conversation.json"
        )
        assert manager.annotated_image_path("llava:7b", "sample1", "COUNT") == (
            manager.models_dir / "llava-7b" / "tests" / "sample1_COUNT" / "annotated.png"
        )
        assert manager.test_html_path("llava:7b", "sample1", "COUNT") == (
            manager.models_dir / "llava-7b" / "tests" / "sample1_COUNT" / "test.html"
        )

    def test_global_path_helpers(self, manager, temp_results_dir):
        """Global path helpers return correct paths."""
        assert manager.global_index_json_path() == temp_results_dir / "index.json"
        assert manager.global_index_html_path() == temp_results_dir / "index.html"

    def test_global_assets_dir_creates_folder(self, manager, temp_results_dir):
        """Global assets dir is created if needed."""
        # Ensure base results dir exists (simulating a run that has saved artifacts)
        temp_results_dir.mkdir(parents=True, exist_ok=True)
        assets_dir = manager.global_assets_dir()
        assert assets_dir == temp_results_dir / "assets"
        assert assets_dir.exists()


class TestArtifactManagerTimestamp:
    """Tests for timestamp handling."""

    def test_default_timestamp(self, tmp_path):
        """Uses current time if timestamp not provided."""
        before = datetime.now()
        manager = ArtifactManager(
            results_dir=str(tmp_path),
            run_name="test",
        )
        after = datetime.now()

        assert before <= manager.timestamp <= after

    def test_custom_timestamp(self, tmp_path):
        """Uses provided timestamp."""
        ts = datetime(2025, 6, 15, 10, 30, 0)
        manager = ArtifactManager(
            results_dir=str(tmp_path),
            run_name="test",
            timestamp=ts,
        )
        assert manager.timestamp == ts
        assert "20250615_103000" in manager.run_id
