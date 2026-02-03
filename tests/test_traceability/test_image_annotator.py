"""Tests for image annotator."""

import pytest
from PIL import Image

from vlm_geometry_bench.traceability.image_annotator import (
    ImageAnnotator,
    annotate_image,
    get_font,
    STATUS_BAR_HEIGHT,
)


class TestGetFont:
    """Tests for font loading."""

    def test_returns_font(self):
        """Returns a font object."""
        font = get_font()
        assert font is not None

    def test_custom_size(self):
        """Can specify custom size."""
        font = get_font(20)
        assert font is not None


class TestImageAnnotator:
    """Tests for ImageAnnotator class."""

    @pytest.fixture
    def annotator(self):
        """Create ImageAnnotator."""
        return ImageAnnotator(match_distance=0.05)

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        return Image.new("RGB", (200, 200), color="black")

    @pytest.fixture
    def test_image_rgba(self):
        """Create test RGBA image."""
        return Image.new("RGBA", (200, 200), color="black")

    def test_annotate_locate_task(self, annotator, test_image):
        """Annotates LOCATE task image."""
        gt_positions = [(0.25, 0.25), (0.75, 0.75)]
        pred_positions = [(0.26, 0.26), (0.5, 0.5)]
        metrics = {"detection_rate": 50.0, "false_positive_rate": 50.0}

        result = annotator.annotate_locate_task(
            test_image,
            gt_positions,
            pred_positions,
            sample_id="test_sample",
            model="llava:7b",
            metrics=metrics,
        )

        # Image should be taller due to status bar
        assert result.height == test_image.height + STATUS_BAR_HEIGHT
        assert result.width == test_image.width
        assert result.mode == "RGB"

    def test_annotate_locate_rgba_conversion(self, annotator, test_image_rgba):
        """Converts RGBA to RGB."""
        result = annotator.annotate_locate_task(
            test_image_rgba,
            gt_positions=[(0.5, 0.5)],
            pred_positions=[(0.5, 0.5)],
            sample_id="test",
            model="llava:7b",
            metrics={},
        )

        assert result.mode == "RGB"

    def test_annotate_locate_empty_positions(self, annotator, test_image):
        """Handles empty position lists."""
        result = annotator.annotate_locate_task(
            test_image,
            gt_positions=[],
            pred_positions=[],
            sample_id="test",
            model="llava:7b",
            metrics={"detection_rate": 100.0, "false_positive_rate": 0.0},
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_count_task(self, annotator, test_image):
        """Annotates COUNT task image."""
        metrics = {}

        result = annotator.annotate_count_task(
            test_image,
            gt_count=15,
            pred_count=14,
            sample_id="test_sample",
            model="llava:7b",
            metrics=metrics,
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_count_none_prediction(self, annotator, test_image):
        """Handles None prediction for COUNT."""
        result = annotator.annotate_count_task(
            test_image,
            gt_count=15,
            pred_count=None,
            sample_id="test",
            model="llava:7b",
            metrics={},
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_pattern_task(self, annotator, test_image):
        """Annotates PATTERN task image."""
        result = annotator.annotate_pattern_task(
            test_image,
            gt_pattern="HEXAGONAL",
            pred_pattern="HEXAGONAL",
            sample_id="test",
            model="llava:7b",
            metrics={"correct": True},
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_size_task(self, annotator, test_image):
        """Annotates SIZE task image."""
        result = annotator.annotate_size_task(
            test_image,
            gt_size=3.5,
            pred_size=3.8,
            sample_id="test",
            model="llava:7b",
            metrics={"within_tolerance": True},
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_defect_task(self, annotator, test_image):
        """Annotates DEFECT task image."""
        result = annotator.annotate_defect_task(
            test_image,
            gt_has_defects=True,
            pred_has_defects=True,
            sample_id="test",
            model="llava:7b",
            metrics={"defect_detection_correct": True},
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_defect_none_prediction(self, annotator, test_image):
        """Handles None prediction for DEFECT."""
        result = annotator.annotate_defect_task(
            test_image,
            gt_has_defects=True,
            pred_has_defects=None,
            sample_id="test",
            model="llava:7b",
            metrics={},
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_dispatcher_locate(self, annotator, test_image):
        """Dispatcher routes to LOCATE annotator."""
        ground_truth = {"positions": [(0.5, 0.5)]}
        prediction = {"positions": [(0.5, 0.5)]}
        metrics = {"detection_rate": 100.0, "false_positive_rate": 0.0}

        result = annotator.annotate(
            test_image,
            task="LOCATE",
            sample_id="test",
            model="llava:7b",
            metrics=metrics,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_dispatcher_count(self, annotator, test_image):
        """Dispatcher routes to COUNT annotator."""
        ground_truth = {"spot_count": 10}
        prediction = {"count": 10}
        metrics = {"exact_match": True}

        result = annotator.annotate(
            test_image,
            task="COUNT",
            sample_id="test",
            model="llava:7b",
            metrics=metrics,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_dispatcher_pattern(self, annotator, test_image):
        """Dispatcher routes to PATTERN annotator."""
        ground_truth = {"pattern": "HEXAGONAL"}
        prediction = {"pattern": "HEXAGONAL"}
        metrics = {"correct": True}

        result = annotator.annotate(
            test_image,
            task="PATTERN",
            sample_id="test",
            model="llava:7b",
            metrics=metrics,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_dispatcher_size(self, annotator, test_image):
        """Dispatcher routes to SIZE annotator."""
        ground_truth = {"size": 3.5}
        prediction = {"diameter_um": 3.5}
        metrics = {"within_tolerance": True}

        result = annotator.annotate(
            test_image,
            task="SIZE",
            sample_id="test",
            model="llava:7b",
            metrics=metrics,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_dispatcher_defect(self, annotator, test_image):
        """Dispatcher routes to DEFECT annotator."""
        ground_truth = {"has_defects": True}
        prediction = {"has_defects": True}
        metrics = {"defect_detection_correct": True}

        result = annotator.annotate(
            test_image,
            task="DEFECT",
            sample_id="test",
            model="llava:7b",
            metrics=metrics,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT

    def test_annotate_dispatcher_unknown_task(self, annotator, test_image):
        """Unknown task gets basic status bar."""
        result = annotator.annotate(
            test_image,
            task="UNKNOWN",
            sample_id="test",
            model="llava:7b",
            metrics={},
            ground_truth={},
            prediction={},
        )

        assert result.height == test_image.height + STATUS_BAR_HEIGHT


class TestMatchComputation:
    """Tests for matching algorithm."""

    @pytest.fixture
    def annotator(self):
        """Create annotator with known match distance."""
        return ImageAnnotator(match_distance=0.1)

    def test_perfect_match(self, annotator):
        """All predictions match ground truth."""
        gt = [(0.2, 0.2), (0.8, 0.8)]
        pred = [(0.2, 0.2), (0.8, 0.8)]

        matches = annotator._compute_matches(gt, pred)

        assert len(matches) == 2

    def test_no_match(self, annotator):
        """No predictions match ground truth."""
        gt = [(0.2, 0.2)]
        pred = [(0.8, 0.8)]  # Too far

        matches = annotator._compute_matches(gt, pred)

        assert len(matches) == 0

    def test_partial_match(self, annotator):
        """Some predictions match."""
        gt = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)]
        pred = [(0.21, 0.21), (0.9, 0.9)]  # First matches, second too far

        matches = annotator._compute_matches(gt, pred)

        assert len(matches) == 1
        assert 0 in matches  # First prediction matched

    def test_one_gt_one_pred_each(self, annotator):
        """Each GT can only match one prediction."""
        gt = [(0.5, 0.5)]
        pred = [(0.51, 0.51), (0.52, 0.52)]  # Both close to same GT

        matches = annotator._compute_matches(gt, pred)

        assert len(matches) == 1  # Only one can match

    def test_empty_gt(self, annotator):
        """Empty ground truth."""
        matches = annotator._compute_matches([], [(0.5, 0.5)])
        assert len(matches) == 0

    def test_empty_pred(self, annotator):
        """Empty predictions."""
        matches = annotator._compute_matches([(0.5, 0.5)], [])
        assert len(matches) == 0


class TestModelNameAbbreviation:
    """Tests for model name abbreviation."""

    @pytest.fixture
    def annotator(self):
        return ImageAnnotator()

    def test_known_model_abbreviated(self, annotator):
        """Known model names are abbreviated."""
        assert annotator._abbreviate_model("claude-sonnet-4-20250514") == "claude-sonnet-4"
        assert annotator._abbreviate_model("openai/gpt-4o") == "gpt-4o"

    def test_short_name_unchanged(self, annotator):
        """Short names unchanged."""
        assert annotator._abbreviate_model("llava:7b") == "llava:7b"

    def test_long_name_truncated(self, annotator):
        """Long unknown names are truncated."""
        long_name = "a" * 30
        result = annotator._abbreviate_model(long_name)
        assert len(result) <= 25
        assert result.endswith("...")


class TestAnnotateImageFunction:
    """Tests for convenience function."""

    def test_annotate_image_file(self, tmp_path):
        """Annotates image from file path."""
        # Create test image
        img = Image.new("RGB", (100, 100), color="white")
        input_path = tmp_path / "input.png"
        img.save(input_path)

        output_path = tmp_path / "output.png"

        result = annotate_image(
            image_path=input_path,
            task="COUNT",
            sample_id="test",
            model="llava:7b",
            metrics={"gt_count": 5, "pred_count": 5},
            ground_truth={"spot_count": 5},
            prediction={"count": 5},
            output_path=output_path,
        )

        assert result is not None
        assert output_path.exists()

    def test_annotate_image_no_save(self, tmp_path):
        """Returns image without saving."""
        img = Image.new("RGB", (100, 100), color="white")
        input_path = tmp_path / "input.png"
        img.save(input_path)

        result = annotate_image(
            image_path=input_path,
            task="COUNT",
            sample_id="test",
            model="llava:7b",
            metrics={},
            ground_truth={},
            prediction={},
            output_path=None,
        )

        assert result is not None
        assert result.height == 100 + STATUS_BAR_HEIGHT
