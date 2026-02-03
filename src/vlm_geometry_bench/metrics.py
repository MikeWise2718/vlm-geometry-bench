"""Metrics calculation for VLM Geometry Bench.

Provides metrics computation for each evaluation task and aggregation
across samples, image classes, and overall results.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# INDIVIDUAL SAMPLE METRICS
# =============================================================================

@dataclass
class CountMetrics:
    """Metrics for COUNT task."""

    exact_match: bool
    within_tolerance: bool
    absolute_error: int
    percentage_error: Optional[float]  # None if ground truth is 0
    predicted: Optional[int]
    ground_truth: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "within_tolerance": self.within_tolerance,
            "absolute_error": self.absolute_error,
            "percentage_error": self.percentage_error,
            "predicted": self.predicted,
            "ground_truth": self.ground_truth,
        }


@dataclass
class PatternMetrics:
    """Metrics for PATTERN task."""

    correct: bool
    predicted: Optional[str]
    ground_truth: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "predicted": self.predicted,
            "ground_truth": self.ground_truth,
        }


@dataclass
class LocateMetrics:
    """Metrics for LOCATE task."""

    predicted_count: int
    ground_truth_count: int
    count_difference: int
    mean_nearest_distance: Optional[float]  # Mean distance to nearest ground truth
    detection_rate: float  # % of ground truth spots matched
    false_positive_rate: float  # % of predictions not matching ground truth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_count": self.predicted_count,
            "ground_truth_count": self.ground_truth_count,
            "count_difference": self.count_difference,
            "mean_nearest_distance": self.mean_nearest_distance,
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
        }


@dataclass
class SizeMetrics:
    """Metrics for SIZE task."""

    predicted: Optional[float]
    ground_truth: float
    absolute_error: Optional[float]
    percentage_error: Optional[float]
    within_tolerance: bool  # Within 20% of ground truth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted": self.predicted,
            "ground_truth": self.ground_truth,
            "absolute_error": self.absolute_error,
            "percentage_error": self.percentage_error,
            "within_tolerance": self.within_tolerance,
        }


@dataclass
class DefectMetrics:
    """Metrics for DEFECT task."""

    defect_detection_correct: bool  # Correctly identified presence/absence of defects
    predicted_has_defects: Optional[bool]
    ground_truth_has_defects: bool
    missing_count_error: Optional[int]
    noise_count_error: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_detection_correct": self.defect_detection_correct,
            "predicted_has_defects": self.predicted_has_defects,
            "ground_truth_has_defects": self.ground_truth_has_defects,
            "missing_count_error": self.missing_count_error,
            "noise_count_error": self.noise_count_error,
        }


@dataclass
class SampleResult:
    """Complete result for a single sample evaluation."""

    sample_id: str
    image_class: str
    task: str
    success: bool  # Whether API call and parsing succeeded
    parse_error: Optional[str]
    metrics: Dict[str, Any]
    raw_response: str
    latency_ms: Optional[int] = None
    input_tokens: int = 0
    output_tokens: int = 0
    # Additional fields for traceability
    model: Optional[str] = None  # Model name that generated this result
    prompt: Optional[str] = None  # Prompt text sent to the model
    ground_truth: Optional[Dict[str, Any]] = None  # Ground truth data for traceability
    prediction: Optional[Dict[str, Any]] = None  # Parsed prediction data for traceability


# =============================================================================
# AGGREGATED METRICS
# =============================================================================

@dataclass
class TaskAggregates:
    """Aggregated metrics for a single task."""

    task: str
    total_samples: int
    successful_samples: int
    success_rate: float
    metrics: Dict[str, Any]


@dataclass
class ClassAggregates:
    """Aggregated metrics for a single image class."""

    image_class: str
    total_samples: int
    by_task: Dict[str, TaskAggregates]


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    model_name: str
    total_samples: int
    successful_samples: int
    success_rate: float

    # Timing and cost
    total_time_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: Optional[float]

    # Per-task aggregates
    by_task: Dict[str, TaskAggregates]

    # Per-class aggregates
    by_class: Dict[str, ClassAggregates]

    # Raw sample results
    sample_results: List[SampleResult]


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

class MetricsCalculator:
    """Calculator for benchmark metrics."""

    def __init__(
        self,
        count_tolerance: int = 2,
        size_tolerance_pct: float = 20.0,
        locate_match_distance: float = 0.05,  # normalized distance (5% of image dimension)
    ):
        """Initialize metrics calculator.

        Args:
            count_tolerance: ±N tolerance for "within_tolerance" count metric
            size_tolerance_pct: Percentage tolerance for size estimation
            locate_match_distance: Max normalized distance to consider a match (0.05 = 5% of image)
        """
        self.count_tolerance = count_tolerance
        self.size_tolerance_pct = size_tolerance_pct
        self.locate_match_distance = locate_match_distance

    # -------------------------------------------------------------------------
    # Individual Sample Metrics
    # -------------------------------------------------------------------------

    def compute_count_metrics(
        self,
        predicted: Optional[int],
        ground_truth: int
    ) -> CountMetrics:
        """Compute metrics for COUNT task."""
        if predicted is None:
            return CountMetrics(
                exact_match=False,
                within_tolerance=False,
                absolute_error=ground_truth,
                percentage_error=100.0 if ground_truth > 0 else None,
                predicted=None,
                ground_truth=ground_truth,
            )

        abs_error = abs(predicted - ground_truth)
        pct_error = None
        if ground_truth > 0:
            pct_error = (abs_error / ground_truth) * 100

        return CountMetrics(
            exact_match=(predicted == ground_truth),
            within_tolerance=(abs_error <= self.count_tolerance),
            absolute_error=abs_error,
            percentage_error=pct_error,
            predicted=predicted,
            ground_truth=ground_truth,
        )

    def compute_pattern_metrics(
        self,
        predicted: Optional[str],
        ground_truth: str
    ) -> PatternMetrics:
        """Compute metrics for PATTERN task."""
        return PatternMetrics(
            correct=(predicted == ground_truth) if predicted else False,
            predicted=predicted,
            ground_truth=ground_truth,
        )

    def compute_locate_metrics(
        self,
        predicted_positions: List[Tuple[float, float]],
        ground_truth_positions: List[Tuple[float, float]],
    ) -> LocateMetrics:
        """Compute metrics for LOCATE task.

        Uses nearest-neighbor matching to pair predicted and ground truth positions.
        """
        pred_count = len(predicted_positions)
        gt_count = len(ground_truth_positions)

        if gt_count == 0:
            # No ground truth spots
            return LocateMetrics(
                predicted_count=pred_count,
                ground_truth_count=0,
                count_difference=pred_count,
                mean_nearest_distance=None,
                detection_rate=100.0 if pred_count == 0 else 0.0,
                false_positive_rate=100.0 if pred_count > 0 else 0.0,
            )

        if pred_count == 0:
            # No predictions
            return LocateMetrics(
                predicted_count=0,
                ground_truth_count=gt_count,
                count_difference=gt_count,
                mean_nearest_distance=None,
                detection_rate=0.0,
                false_positive_rate=0.0,
            )

        # Compute distances from each prediction to nearest ground truth
        distances = []
        matched_gt = set()

        for px, py in predicted_positions:
            min_dist = float('inf')
            min_idx = -1
            for i, (gx, gy) in enumerate(ground_truth_positions):
                dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            distances.append(min_dist)
            if min_dist <= self.locate_match_distance:
                matched_gt.add(min_idx)

        mean_dist = sum(distances) / len(distances) if distances else None
        detection_rate = (len(matched_gt) / gt_count) * 100

        # False positives: predictions that don't match any ground truth
        false_positives = sum(1 for d in distances if d > self.locate_match_distance)
        false_positive_rate = (false_positives / pred_count) * 100 if pred_count > 0 else 0.0

        return LocateMetrics(
            predicted_count=pred_count,
            ground_truth_count=gt_count,
            count_difference=abs(pred_count - gt_count),
            mean_nearest_distance=mean_dist,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
        )

    def compute_size_metrics(
        self,
        predicted: Optional[float],
        ground_truth: float
    ) -> SizeMetrics:
        """Compute metrics for SIZE task."""
        if predicted is None or ground_truth <= 0:
            return SizeMetrics(
                predicted=predicted,
                ground_truth=ground_truth,
                absolute_error=None,
                percentage_error=None,
                within_tolerance=False,
            )

        abs_error = abs(predicted - ground_truth)
        pct_error = (abs_error / ground_truth) * 100
        within_tol = pct_error <= self.size_tolerance_pct

        return SizeMetrics(
            predicted=predicted,
            ground_truth=ground_truth,
            absolute_error=abs_error,
            percentage_error=pct_error,
            within_tolerance=within_tol,
        )

    def compute_defect_metrics(
        self,
        predicted_has_defects: Optional[bool],
        ground_truth_has_defects: bool,
        predicted_missing: Optional[int] = None,
        ground_truth_missing: Optional[int] = None,
        predicted_noise: Optional[int] = None,
        ground_truth_noise: Optional[int] = None,
    ) -> DefectMetrics:
        """Compute metrics for DEFECT task."""
        detection_correct = False
        if predicted_has_defects is not None:
            detection_correct = (predicted_has_defects == ground_truth_has_defects)

        missing_error = None
        if predicted_missing is not None and ground_truth_missing is not None:
            missing_error = abs(predicted_missing - ground_truth_missing)

        noise_error = None
        if predicted_noise is not None and ground_truth_noise is not None:
            noise_error = abs(predicted_noise - ground_truth_noise)

        return DefectMetrics(
            defect_detection_correct=detection_correct,
            predicted_has_defects=predicted_has_defects,
            ground_truth_has_defects=ground_truth_has_defects,
            missing_count_error=missing_error,
            noise_count_error=noise_error,
        )

    # -------------------------------------------------------------------------
    # Aggregation Methods
    # -------------------------------------------------------------------------

    def aggregate_count_metrics(
        self,
        metrics_list: List[CountMetrics]
    ) -> Dict[str, Any]:
        """Aggregate COUNT metrics across samples."""
        if not metrics_list:
            return {}

        n = len(metrics_list)
        exact_matches = sum(1 for m in metrics_list if m.exact_match)
        within_tol = sum(1 for m in metrics_list if m.within_tolerance)
        total_abs_error = sum(m.absolute_error for m in metrics_list)

        pct_errors = [m.percentage_error for m in metrics_list if m.percentage_error is not None]
        mean_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else None

        return {
            "exact_match_rate": (exact_matches / n) * 100,
            "within_tolerance_rate": (within_tol / n) * 100,
            "mean_absolute_error": total_abs_error / n,
            "mean_percentage_error": mean_pct_error,
            "total_samples": n,
        }

    def aggregate_pattern_metrics(
        self,
        metrics_list: List[PatternMetrics]
    ) -> Dict[str, Any]:
        """Aggregate PATTERN metrics across samples."""
        if not metrics_list:
            return {}

        n = len(metrics_list)
        correct = sum(1 for m in metrics_list if m.correct)

        # Per-pattern-type accuracy (confusion matrix diagonal)
        by_pattern = defaultdict(lambda: {"correct": 0, "total": 0})
        for m in metrics_list:
            gt = m.ground_truth
            by_pattern[gt]["total"] += 1
            if m.correct:
                by_pattern[gt]["correct"] += 1

        per_pattern_acc = {
            pattern: (stats["correct"] / stats["total"]) * 100
            for pattern, stats in by_pattern.items()
        }

        # Macro accuracy (average of per-pattern accuracies)
        macro_acc = sum(per_pattern_acc.values()) / len(per_pattern_acc) if per_pattern_acc else 0

        return {
            "accuracy": (correct / n) * 100,
            "macro_accuracy": macro_acc,
            "per_pattern_accuracy": dict(per_pattern_acc),
            "total_samples": n,
        }

    def aggregate_locate_metrics(
        self,
        metrics_list: List[LocateMetrics]
    ) -> Dict[str, Any]:
        """Aggregate LOCATE metrics across samples."""
        if not metrics_list:
            return {}

        n = len(metrics_list)

        mean_detection_rate = sum(m.detection_rate for m in metrics_list) / n
        mean_false_positive_rate = sum(m.false_positive_rate for m in metrics_list) / n
        mean_count_diff = sum(m.count_difference for m in metrics_list) / n

        distances = [m.mean_nearest_distance for m in metrics_list if m.mean_nearest_distance is not None]
        mean_distance = sum(distances) / len(distances) if distances else None

        return {
            "mean_detection_rate": mean_detection_rate,
            "mean_false_positive_rate": mean_false_positive_rate,
            "mean_count_difference": mean_count_diff,
            "mean_nearest_distance": mean_distance,
            "total_samples": n,
        }

    def aggregate_size_metrics(
        self,
        metrics_list: List[SizeMetrics]
    ) -> Dict[str, Any]:
        """Aggregate SIZE metrics across samples."""
        if not metrics_list:
            return {}

        n = len(metrics_list)
        within_tol = sum(1 for m in metrics_list if m.within_tolerance)

        abs_errors = [m.absolute_error for m in metrics_list if m.absolute_error is not None]
        pct_errors = [m.percentage_error for m in metrics_list if m.percentage_error is not None]

        return {
            "within_tolerance_rate": (within_tol / n) * 100,
            "mean_absolute_error": sum(abs_errors) / len(abs_errors) if abs_errors else None,
            "mean_percentage_error": sum(pct_errors) / len(pct_errors) if pct_errors else None,
            "total_samples": n,
        }

    def aggregate_defect_metrics(
        self,
        metrics_list: List[DefectMetrics]
    ) -> Dict[str, Any]:
        """Aggregate DEFECT metrics across samples."""
        if not metrics_list:
            return {}

        n = len(metrics_list)
        detection_correct = sum(1 for m in metrics_list if m.defect_detection_correct)

        missing_errors = [m.missing_count_error for m in metrics_list if m.missing_count_error is not None]
        noise_errors = [m.noise_count_error for m in metrics_list if m.noise_count_error is not None]

        return {
            "detection_accuracy": (detection_correct / n) * 100,
            "mean_missing_count_error": sum(missing_errors) / len(missing_errors) if missing_errors else None,
            "mean_noise_count_error": sum(noise_errors) / len(noise_errors) if noise_errors else None,
            "total_samples": n,
        }

    def aggregate_results(
        self,
        sample_results: List[SampleResult],
    ) -> Dict[str, Any]:
        """Aggregate all sample results into summary statistics.

        Args:
            sample_results: List of SampleResult objects

        Returns:
            Dictionary with aggregated metrics by task and by class
        """
        if not sample_results:
            return {"error": "No results to aggregate"}

        # Group by task
        by_task: Dict[str, List[SampleResult]] = defaultdict(list)
        for r in sample_results:
            by_task[r.task].append(r)

        # Group by class
        by_class: Dict[str, Dict[str, List[SampleResult]]] = defaultdict(lambda: defaultdict(list))
        for r in sample_results:
            by_class[r.image_class][r.task].append(r)

        # Compute task aggregates
        task_aggregates = {}
        for task, results in by_task.items():
            successful = [r for r in results if r.success]
            task_aggregates[task] = {
                "total_samples": len(results),
                "successful_samples": len(successful),
                "success_rate": (len(successful) / len(results)) * 100 if results else 0,
            }

        # Compute class aggregates
        class_aggregates = {}
        for cls, task_results in by_class.items():
            class_aggregates[cls] = {}
            for task, results in task_results.items():
                successful = [r for r in results if r.success]
                class_aggregates[cls][task] = {
                    "total_samples": len(results),
                    "successful_samples": len(successful),
                    "success_rate": (len(successful) / len(results)) * 100 if results else 0,
                }

        # Overall stats
        total = len(sample_results)
        successful = sum(1 for r in sample_results if r.success)

        return {
            "total_samples": total,
            "successful_samples": successful,
            "success_rate": (successful / total) * 100 if total else 0,
            "by_task": task_aggregates,
            "by_class": class_aggregates,
        }
