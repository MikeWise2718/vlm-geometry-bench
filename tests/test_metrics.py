"""Tests for metrics calculator module."""

import pytest
from vlm_geometry_bench.metrics import (
    MetricsCalculator,
    CountMetrics,
    PatternMetrics,
    LocateMetrics,
    SizeMetrics,
    DefectMetrics,
    SampleResult,
)


class TestCountMetrics:
    """Tests for COUNT metrics computation."""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator(count_tolerance=2)

    def test_exact_match(self, calc):
        """Exact match when predicted equals ground truth."""
        metrics = calc.compute_count_metrics(predicted=42, ground_truth=42)
        assert metrics.exact_match is True
        assert metrics.within_tolerance is True
        assert metrics.absolute_error == 0
        assert metrics.percentage_error == 0.0

    def test_within_tolerance(self, calc):
        """Within tolerance when error <= tolerance."""
        metrics = calc.compute_count_metrics(predicted=44, ground_truth=42)
        assert metrics.exact_match is False
        assert metrics.within_tolerance is True
        assert metrics.absolute_error == 2

    def test_outside_tolerance(self, calc):
        """Outside tolerance when error > tolerance."""
        metrics = calc.compute_count_metrics(predicted=50, ground_truth=42)
        assert metrics.exact_match is False
        assert metrics.within_tolerance is False
        assert metrics.absolute_error == 8

    def test_none_prediction(self, calc):
        """None prediction is always wrong."""
        metrics = calc.compute_count_metrics(predicted=None, ground_truth=42)
        assert metrics.exact_match is False
        assert metrics.within_tolerance is False
        assert metrics.predicted is None
        assert metrics.absolute_error == 42
        assert metrics.percentage_error == 100.0

    def test_zero_ground_truth(self, calc):
        """Zero ground truth has no percentage error."""
        metrics = calc.compute_count_metrics(predicted=0, ground_truth=0)
        assert metrics.exact_match is True
        assert metrics.percentage_error is None

    def test_percentage_error(self, calc):
        """Percentage error is computed correctly."""
        metrics = calc.compute_count_metrics(predicted=50, ground_truth=100)
        assert metrics.percentage_error == 50.0

    def test_to_dict(self, calc):
        """Metrics can be converted to dict."""
        metrics = calc.compute_count_metrics(predicted=42, ground_truth=42)
        d = metrics.to_dict()
        assert "exact_match" in d
        assert "predicted" in d
        assert "ground_truth" in d


class TestPatternMetrics:
    """Tests for PATTERN metrics computation."""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_correct_prediction(self, calc):
        """Correct when predicted equals ground truth."""
        metrics = calc.compute_pattern_metrics(predicted="HEXAGONAL", ground_truth="HEXAGONAL")
        assert metrics.correct is True

    def test_incorrect_prediction(self, calc):
        """Incorrect when predicted differs from ground truth."""
        metrics = calc.compute_pattern_metrics(predicted="RANDOM", ground_truth="HEXAGONAL")
        assert metrics.correct is False

    def test_none_prediction(self, calc):
        """None prediction is incorrect."""
        metrics = calc.compute_pattern_metrics(predicted=None, ground_truth="HEXAGONAL")
        assert metrics.correct is False

    def test_to_dict(self, calc):
        """Metrics can be converted to dict."""
        metrics = calc.compute_pattern_metrics(predicted="HEXAGONAL", ground_truth="HEXAGONAL")
        d = metrics.to_dict()
        assert d["correct"] is True


class TestLocateMetrics:
    """Tests for LOCATE metrics computation."""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator(locate_match_distance=20.0)

    def test_perfect_detection(self, calc):
        """Perfect detection when all spots matched."""
        gt = [(100.0, 100.0), (200.0, 200.0)]
        pred = [(100.0, 100.0), (200.0, 200.0)]
        metrics = calc.compute_locate_metrics(pred, gt)
        assert metrics.detection_rate == 100.0
        assert metrics.false_positive_rate == 0.0
        assert metrics.count_difference == 0

    def test_partial_detection(self, calc):
        """Partial detection when some spots matched."""
        gt = [(100.0, 100.0), (200.0, 200.0)]
        pred = [(100.0, 100.0)]  # Only found one
        metrics = calc.compute_locate_metrics(pred, gt)
        assert metrics.detection_rate == 50.0
        assert metrics.count_difference == 1

    def test_false_positives(self, calc):
        """False positives when predictions don't match ground truth."""
        gt = [(100.0, 100.0)]
        pred = [(100.0, 100.0), (500.0, 500.0)]  # One correct, one false positive
        metrics = calc.compute_locate_metrics(pred, gt)
        assert metrics.detection_rate == 100.0
        assert metrics.false_positive_rate == 50.0

    def test_no_ground_truth(self, calc):
        """Handle case with no ground truth spots."""
        metrics = calc.compute_locate_metrics([(100.0, 100.0)], [])
        assert metrics.ground_truth_count == 0
        assert metrics.false_positive_rate == 100.0

    def test_no_predictions(self, calc):
        """Handle case with no predictions."""
        metrics = calc.compute_locate_metrics([], [(100.0, 100.0)])
        assert metrics.predicted_count == 0
        assert metrics.detection_rate == 0.0
        assert metrics.false_positive_rate == 0.0

    def test_both_empty(self, calc):
        """Handle case with no spots at all."""
        metrics = calc.compute_locate_metrics([], [])
        assert metrics.detection_rate == 100.0
        assert metrics.false_positive_rate == 0.0

    def test_mean_distance(self, calc):
        """Mean distance is computed for predictions."""
        gt = [(100.0, 100.0)]
        pred = [(105.0, 100.0), (110.0, 100.0)]  # 5 and 10 pixels away
        metrics = calc.compute_locate_metrics(pred, gt)
        assert metrics.mean_nearest_distance == 7.5

    def test_to_dict(self, calc):
        """Metrics can be converted to dict."""
        metrics = calc.compute_locate_metrics([(100.0, 100.0)], [(100.0, 100.0)])
        d = metrics.to_dict()
        assert "detection_rate" in d


class TestSizeMetrics:
    """Tests for SIZE metrics computation."""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator(size_tolerance_pct=20.0)

    def test_exact_match(self, calc):
        """Exact match."""
        metrics = calc.compute_size_metrics(predicted=3.5, ground_truth=3.5)
        assert metrics.within_tolerance is True
        assert metrics.absolute_error == 0.0
        assert metrics.percentage_error == 0.0

    def test_within_tolerance(self, calc):
        """Within 20% tolerance."""
        metrics = calc.compute_size_metrics(predicted=4.0, ground_truth=3.5)
        assert metrics.within_tolerance is True
        assert metrics.percentage_error == pytest.approx(14.29, rel=0.01)

    def test_outside_tolerance(self, calc):
        """Outside 20% tolerance."""
        metrics = calc.compute_size_metrics(predicted=5.0, ground_truth=3.5)
        assert metrics.within_tolerance is False

    def test_none_prediction(self, calc):
        """None prediction is not within tolerance."""
        metrics = calc.compute_size_metrics(predicted=None, ground_truth=3.5)
        assert metrics.within_tolerance is False
        assert metrics.absolute_error is None

    def test_zero_ground_truth(self, calc):
        """Zero ground truth handled gracefully."""
        metrics = calc.compute_size_metrics(predicted=3.5, ground_truth=0.0)
        assert metrics.within_tolerance is False

    def test_to_dict(self, calc):
        """Metrics can be converted to dict."""
        metrics = calc.compute_size_metrics(predicted=3.5, ground_truth=3.5)
        d = metrics.to_dict()
        assert "within_tolerance" in d


class TestDefectMetrics:
    """Tests for DEFECT metrics computation."""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_correct_detection_true(self, calc):
        """Correctly detecting defects present."""
        metrics = calc.compute_defect_metrics(
            predicted_has_defects=True,
            ground_truth_has_defects=True,
        )
        assert metrics.defect_detection_correct is True

    def test_correct_detection_false(self, calc):
        """Correctly detecting no defects."""
        metrics = calc.compute_defect_metrics(
            predicted_has_defects=False,
            ground_truth_has_defects=False,
        )
        assert metrics.defect_detection_correct is True

    def test_false_positive(self, calc):
        """False positive: detecting defects when none exist."""
        metrics = calc.compute_defect_metrics(
            predicted_has_defects=True,
            ground_truth_has_defects=False,
        )
        assert metrics.defect_detection_correct is False

    def test_false_negative(self, calc):
        """False negative: missing defects."""
        metrics = calc.compute_defect_metrics(
            predicted_has_defects=False,
            ground_truth_has_defects=True,
        )
        assert metrics.defect_detection_correct is False

    def test_missing_count_error(self, calc):
        """Compute missing count error."""
        metrics = calc.compute_defect_metrics(
            predicted_has_defects=True,
            ground_truth_has_defects=True,
            predicted_missing=5,
            ground_truth_missing=3,
        )
        assert metrics.missing_count_error == 2

    def test_noise_count_error(self, calc):
        """Compute noise count error."""
        metrics = calc.compute_defect_metrics(
            predicted_has_defects=True,
            ground_truth_has_defects=True,
            predicted_noise=2,
            ground_truth_noise=4,
        )
        assert metrics.noise_count_error == 2

    def test_none_prediction(self, calc):
        """None prediction is incorrect."""
        metrics = calc.compute_defect_metrics(
            predicted_has_defects=None,
            ground_truth_has_defects=True,
        )
        assert metrics.defect_detection_correct is False

    def test_to_dict(self, calc):
        """Metrics can be converted to dict."""
        metrics = calc.compute_defect_metrics(
            predicted_has_defects=True,
            ground_truth_has_defects=True,
        )
        d = metrics.to_dict()
        assert "defect_detection_correct" in d


class TestAggregateMetrics:
    """Tests for metrics aggregation."""

    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_aggregate_count_metrics(self, calc):
        """Aggregate COUNT metrics."""
        metrics = [
            CountMetrics(exact_match=True, within_tolerance=True, absolute_error=0, percentage_error=0.0, predicted=10, ground_truth=10),
            CountMetrics(exact_match=False, within_tolerance=True, absolute_error=1, percentage_error=10.0, predicted=11, ground_truth=10),
            CountMetrics(exact_match=False, within_tolerance=False, absolute_error=5, percentage_error=50.0, predicted=15, ground_truth=10),
        ]
        agg = calc.aggregate_count_metrics(metrics)
        assert agg["total_samples"] == 3
        assert agg["exact_match_rate"] == pytest.approx(33.33, rel=0.01)
        assert agg["within_tolerance_rate"] == pytest.approx(66.67, rel=0.01)
        assert agg["mean_absolute_error"] == 2.0
        assert agg["mean_percentage_error"] == 20.0

    def test_aggregate_count_empty(self, calc):
        """Aggregate empty COUNT metrics list."""
        agg = calc.aggregate_count_metrics([])
        assert agg == {}

    def test_aggregate_pattern_metrics(self, calc):
        """Aggregate PATTERN metrics."""
        metrics = [
            PatternMetrics(correct=True, predicted="HEXAGONAL", ground_truth="HEXAGONAL"),
            PatternMetrics(correct=True, predicted="RANDOM", ground_truth="RANDOM"),
            PatternMetrics(correct=False, predicted="RANDOM", ground_truth="HEXAGONAL"),
        ]
        agg = calc.aggregate_pattern_metrics(metrics)
        assert agg["accuracy"] == pytest.approx(66.67, rel=0.01)
        assert "per_pattern_accuracy" in agg

    def test_aggregate_locate_metrics(self, calc):
        """Aggregate LOCATE metrics."""
        metrics = [
            LocateMetrics(predicted_count=5, ground_truth_count=5, count_difference=0, mean_nearest_distance=5.0, detection_rate=100.0, false_positive_rate=0.0),
            LocateMetrics(predicted_count=3, ground_truth_count=5, count_difference=2, mean_nearest_distance=10.0, detection_rate=60.0, false_positive_rate=0.0),
        ]
        agg = calc.aggregate_locate_metrics(metrics)
        assert agg["mean_detection_rate"] == 80.0
        assert agg["mean_count_difference"] == 1.0

    def test_aggregate_size_metrics(self, calc):
        """Aggregate SIZE metrics."""
        metrics = [
            SizeMetrics(predicted=3.5, ground_truth=3.5, absolute_error=0.0, percentage_error=0.0, within_tolerance=True),
            SizeMetrics(predicted=4.0, ground_truth=3.5, absolute_error=0.5, percentage_error=14.29, within_tolerance=True),
            SizeMetrics(predicted=5.0, ground_truth=3.5, absolute_error=1.5, percentage_error=42.86, within_tolerance=False),
        ]
        agg = calc.aggregate_size_metrics(metrics)
        assert agg["within_tolerance_rate"] == pytest.approx(66.67, rel=0.01)

    def test_aggregate_defect_metrics(self, calc):
        """Aggregate DEFECT metrics."""
        metrics = [
            DefectMetrics(defect_detection_correct=True, predicted_has_defects=True, ground_truth_has_defects=True, missing_count_error=0, noise_count_error=0),
            DefectMetrics(defect_detection_correct=False, predicted_has_defects=False, ground_truth_has_defects=True, missing_count_error=None, noise_count_error=None),
        ]
        agg = calc.aggregate_defect_metrics(metrics)
        assert agg["detection_accuracy"] == 50.0

    def test_aggregate_results(self, calc):
        """Aggregate sample results."""
        results = [
            SampleResult(sample_id="s1", image_class="USSS", task="COUNT", success=True, parse_error=None, metrics={}, raw_response="10"),
            SampleResult(sample_id="s2", image_class="USSS", task="COUNT", success=True, parse_error=None, metrics={}, raw_response="20"),
            SampleResult(sample_id="s3", image_class="HSFR", task="COUNT", success=False, parse_error="error", metrics={}, raw_response=""),
        ]
        agg = calc.aggregate_results(results)
        assert agg["total_samples"] == 3
        assert agg["successful_samples"] == 2
        assert agg["by_task"]["COUNT"]["total_samples"] == 3
        assert agg["by_class"]["USSS"]["COUNT"]["total_samples"] == 2
