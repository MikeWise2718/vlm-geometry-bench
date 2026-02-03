"""Tests for data loader module."""

import pytest
import tempfile
import os
from pathlib import Path

import yaml
from PIL import Image

from vlm_geometry_bench.data_loader import (
    SuiteLoader,
    BenchmarkSample,
    GroundTruth,
    SpotPosition,
)


@pytest.fixture
def sample_testsuite(tmp_path):
    """Create a minimal test suite for testing."""
    # Create manifest
    manifest = {
        "version": "1.0.0",
        "total_images": 3,
        "image_width": 700,
        "image_height": 510,
        "scale_inverse": 0.14,
        "images": [
            {
                "filename": "ctrl_empty.png",
                "class_name": "CTRL",
                "pattern": "empty",
                "spot_count": 0,
            },
            {
                "filename": "usss_5spots.png",
                "class_name": "USSS",
                "pattern": "uniform_spots",
                "spot_count": 5,
                "csv_file": "usss_5spots.csv",
            },
            {
                "filename": "hsfr_hex.png",
                "class_name": "HSFR",
                "pattern": "hexagonal",
                "spot_count": 10,
            },
        ],
    }

    manifest_path = tmp_path / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f)

    # Create dummy images
    for img_info in manifest["images"]:
        img = Image.new("RGB", (700, 510), color="white")
        img.save(tmp_path / img_info["filename"])

    # Create CSV for the USSS image
    csv_path = tmp_path / "usss_5spots.csv"
    with open(csv_path, "w") as f:
        f.write("X_um,Y_um,Diameter_um\n")
        f.write("10.0,10.0,3.5\n")
        f.write("20.0,20.0,3.5\n")
        f.write("30.0,30.0,3.5\n")
        f.write("40.0,40.0,3.5\n")
        f.write("50.0,50.0,3.5\n")

    return tmp_path


class TestSuiteLoader:
    """Tests for SuiteLoader class."""

    def test_load_manifest(self, sample_testsuite):
        """Load manifest from test suite."""
        loader = SuiteLoader(str(sample_testsuite))
        manifest = loader.load_manifest()

        assert manifest["version"] == "1.0.0"
        assert manifest["total_images"] == 3
        assert len(manifest["images"]) == 3

    def test_manifest_cached(self, sample_testsuite):
        """Manifest is cached after first load."""
        loader = SuiteLoader(str(sample_testsuite))
        manifest1 = loader.manifest
        manifest2 = loader.manifest
        assert manifest1 is manifest2

    def test_missing_manifest_raises(self, tmp_path):
        """Missing manifest raises FileNotFoundError."""
        loader = SuiteLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            loader.load_manifest()

    def test_load_all_samples(self, sample_testsuite):
        """Load all samples from test suite."""
        loader = SuiteLoader(str(sample_testsuite))
        samples = loader.load_samples()

        assert len(samples) == 3
        assert all(isinstance(s, BenchmarkSample) for s in samples)

    def test_load_samples_filter_by_class(self, sample_testsuite):
        """Filter samples by image class."""
        loader = SuiteLoader(str(sample_testsuite))
        samples = loader.load_samples(image_classes=["CTRL"])

        assert len(samples) == 1
        assert samples[0].image_class == "CTRL"

    def test_load_samples_multiple_classes(self, sample_testsuite):
        """Filter samples by multiple classes."""
        loader = SuiteLoader(str(sample_testsuite))
        samples = loader.load_samples(image_classes=["CTRL", "HSFR"])

        assert len(samples) == 2
        classes = {s.image_class for s in samples}
        assert classes == {"CTRL", "HSFR"}

    def test_load_samples_with_limit(self, sample_testsuite):
        """Limit number of samples loaded."""
        loader = SuiteLoader(str(sample_testsuite))
        samples = loader.load_samples(limit=2)

        assert len(samples) == 2

    def test_sample_has_image(self, sample_testsuite):
        """Loaded sample has PIL Image."""
        loader = SuiteLoader(str(sample_testsuite))
        samples = loader.load_samples(limit=1)

        assert isinstance(samples[0].image, Image.Image)
        assert samples[0].image.size == (700, 510)

    def test_sample_has_ground_truth(self, sample_testsuite):
        """Loaded sample has ground truth."""
        loader = SuiteLoader(str(sample_testsuite))
        samples = loader.load_samples(image_classes=["USSS"])

        gt = samples[0].ground_truth
        assert isinstance(gt, GroundTruth)
        assert gt.spot_count == 5
        assert gt.pattern_type == "uniform_spots"
        assert gt.image_class == "USSS"

    def test_sample_positions_from_csv(self, sample_testsuite):
        """Spot positions loaded from CSV."""
        loader = SuiteLoader(str(sample_testsuite))
        samples = loader.load_samples(image_classes=["USSS"])

        positions = samples[0].ground_truth.positions
        assert len(positions) == 5
        assert all(isinstance(p, SpotPosition) for p in positions)
        assert positions[0].x_um == 10.0
        assert positions[0].diameter_um == 3.5

    def test_get_class_summary(self, sample_testsuite):
        """Get count of images per class."""
        loader = SuiteLoader(str(sample_testsuite))
        summary = loader.get_class_summary()

        assert summary["CTRL"] == 1
        assert summary["USSS"] == 1
        assert summary["HSFR"] == 1

    def test_get_pattern_for_class(self, sample_testsuite):
        """Get expected pattern for class."""
        loader = SuiteLoader(str(sample_testsuite))

        assert loader.get_pattern_for_class("CTRL") == "EMPTY"
        assert loader.get_pattern_for_class("USSS") == "RANDOM"
        assert loader.get_pattern_for_class("HSFR") == "HEXAGONAL"
        assert loader.get_pattern_for_class("UNKNOWN") == "UNKNOWN"


class TestGroundTruth:
    """Tests for GroundTruth dataclass."""

    def test_has_defects(self):
        """has_defects property."""
        gt_defects = GroundTruth(
            spot_count=10,
            positions=[],
            pattern_type="hexagonal_defects",
            image_class="HSDN",
        )
        gt_no_defects = GroundTruth(
            spot_count=10,
            positions=[],
            pattern_type="hexagonal",
            image_class="HSFR",
        )

        assert gt_defects.has_defects is True
        assert gt_no_defects.has_defects is False

    def test_is_hexagonal(self):
        """is_hexagonal property."""
        gt_hex = GroundTruth(spot_count=10, positions=[], pattern_type="hexagonal", image_class="HSFR")
        gt_hex_pert = GroundTruth(spot_count=10, positions=[], pattern_type="hexagonal_perturbed", image_class="HSRP")
        gt_random = GroundTruth(spot_count=10, positions=[], pattern_type="uniform_spots", image_class="USSS")

        assert gt_hex.is_hexagonal is True
        assert gt_hex_pert.is_hexagonal is True
        assert gt_random.is_hexagonal is False

    def test_is_random(self):
        """is_random property."""
        gt_uniform = GroundTruth(spot_count=10, positions=[], pattern_type="uniform_spots", image_class="USSS")
        gt_variable = GroundTruth(spot_count=10, positions=[], pattern_type="variable_spots", image_class="USDS")
        gt_hex = GroundTruth(spot_count=10, positions=[], pattern_type="hexagonal", image_class="HSFR")

        assert gt_uniform.is_random is True
        assert gt_variable.is_random is True
        assert gt_hex.is_random is False


class TestSpotPosition:
    """Tests for SpotPosition dataclass."""

    def test_create_position(self):
        """Create a spot position."""
        pos = SpotPosition(x_um=10.5, y_um=20.5, diameter_um=3.5)

        assert pos.x_um == 10.5
        assert pos.y_um == 20.5
        assert pos.diameter_um == 3.5


class TestBenchmarkSample:
    """Tests for BenchmarkSample dataclass."""

    def test_create_sample(self, sample_testsuite):
        """Create a benchmark sample."""
        loader = SuiteLoader(str(sample_testsuite))
        samples = loader.load_samples(limit=1)
        sample = samples[0]

        assert sample.sample_id is not None
        assert sample.filename is not None
        assert sample.image_path.exists()
        assert sample.width == 700
        assert sample.height == 510
        assert sample.scale_inverse == 0.14
