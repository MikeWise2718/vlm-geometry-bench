"""Main evaluator orchestration for VLM Geometry Bench."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from tqdm import tqdm

from .config import EvaluationConfig
from .data_loader import TestSuiteLoader, BenchmarkSample
from .vision_client import VisionClient, VisionResponse
from .prompts import get_prompt_for_sample, get_expected_pattern
from .response_parser import ResponseParser
from .metrics import (
    MetricsCalculator,
    SampleResult,
    CountMetrics,
    PatternMetrics,
    LocateMetrics,
    SizeMetrics,
    DefectMetrics,
)

logger = logging.getLogger(__name__)


# Model pricing (input/output per 1M tokens)
# Used for cost estimation for OpenRouter and Anthropic backends
MODEL_PRICING = {
    # OpenRouter models
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "qwen/qwen2.5-vl-72b-instruct": (0.40, 0.40),
    "google/gemini-pro-1.5": (1.25, 5.00),
    "meta-llama/llama-3.2-90b-vision-instruct": (0.90, 0.90),
    # Anthropic direct API models (Jan 2025 pricing)
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    # Aliases for common model names
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "default": (1.00, 3.00),
}

# Backwards compatibility alias
OPENROUTER_PRICING = MODEL_PRICING


@dataclass
class UsageStats:
    """Track token usage and timing statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return ((self.total_requests - self.failed_requests) / self.total_requests) * 100

    def estimate_cost(self, model_name: str) -> Optional[float]:
        """Estimate cost based on model pricing.

        Works for both OpenRouter and Anthropic backends.
        """
        pricing = MODEL_PRICING.get("default")

        # First try exact match
        if model_name in MODEL_PRICING:
            pricing = MODEL_PRICING[model_name]
        else:
            # Try pattern matching (case-insensitive substring match)
            model_lower = model_name.lower()
            for pattern, costs in MODEL_PRICING.items():
                if pattern != "default" and pattern in model_lower:
                    pricing = costs
                    break

        input_cost, output_cost = pricing
        cost = (self.input_tokens / 1_000_000 * input_cost +
                self.output_tokens / 1_000_000 * output_cost)
        return cost


@dataclass
class EvaluationResults:
    """Complete results from an evaluation run."""

    config: Dict[str, Any]
    usage: Dict[str, Any]
    results_by_task: Dict[str, Dict[str, Any]]
    results_by_class: Dict[str, Dict[str, Dict[str, Any]]]
    overall: Dict[str, Any]
    sample_results: List[SampleResult]
    timestamp: str


class GeometryBenchEvaluator:
    """Main evaluator for VLM Geometry Bench."""

    def __init__(self, config: EvaluationConfig):
        """Initialize evaluator with configuration.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.loader = TestSuiteLoader(config.testsuite_path)
        self.parser = ResponseParser()
        self.metrics = MetricsCalculator(count_tolerance=config.count_tolerance)
        self.usage_stats = UsageStats()
        self.sample_results: List[SampleResult] = []

    async def run_evaluation(self) -> EvaluationResults:
        """Run the complete evaluation.

        Returns:
            EvaluationResults with all metrics and raw data
        """
        logger.info(f"Starting evaluation with model: {self.config.model_name}")
        logger.info(f"Backend: {self.config.backend}")
        logger.info(f"Tasks: {self.config.tasks}")
        logger.info(f"Image classes: {self.config.image_classes}")

        # Reset state
        self.sample_results = []
        self.usage_stats = UsageStats()

        # Load samples
        samples = self.loader.load_samples(
            image_classes=self.config.image_classes,
            limit=self.config.num_samples,
        )

        if not samples:
            raise ValueError("No samples loaded. Check testsuite path and image classes.")

        logger.info(f"Loaded {len(samples)} samples")

        # Run evaluation
        async with VisionClient(self.config) as client:
            for task in self.config.tasks:
                logger.info(f"Evaluating task: {task}")
                await self._evaluate_task(client, samples, task)

        # Mark end time
        self.usage_stats.end_time = time.time()

        # Compile results
        results = self._compile_results()

        return results

    async def _evaluate_task(
        self,
        client: VisionClient,
        samples: List[BenchmarkSample],
        task: str,
    ) -> None:
        """Evaluate all samples for a specific task.

        Args:
            client: VisionClient for API calls
            samples: List of benchmark samples
            task: Task to evaluate (COUNT, PATTERN, etc.)
        """
        desc = f"Evaluating {task}"

        for sample in tqdm(samples, desc=desc):
            try:
                result = await self._evaluate_single(client, sample, task)
                self.sample_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {sample.sample_id}/{task}: {e}")
                self.sample_results.append(SampleResult(
                    sample_id=sample.sample_id,
                    image_class=sample.image_class,
                    task=task,
                    success=False,
                    parse_error=str(e),
                    metrics={},
                    raw_response="",
                ))
                self.usage_stats.failed_requests += 1

    async def _evaluate_single(
        self,
        client: VisionClient,
        sample: BenchmarkSample,
        task: str,
    ) -> SampleResult:
        """Evaluate a single sample for a specific task.

        Args:
            client: VisionClient for API calls
            sample: Benchmark sample with image and ground truth
            task: Task to evaluate

        Returns:
            SampleResult with metrics
        """
        # Build prompt
        prompt = get_prompt_for_sample(task, sample, self.config.num_shots)

        # Send request
        response = await client.send_request(sample.image, prompt)

        # Track usage
        self.usage_stats.total_requests += 1
        self.usage_stats.input_tokens += response.input_tokens
        self.usage_stats.output_tokens += response.output_tokens

        if not response.success:
            self.usage_stats.failed_requests += 1
            return SampleResult(
                sample_id=sample.sample_id,
                image_class=sample.image_class,
                task=task,
                success=False,
                parse_error=response.error,
                metrics={},
                raw_response=response.content,
                latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

        # Parse response and compute metrics
        parsed = self.parser.parse(task, response.content)
        metrics_obj = self._compute_metrics(task, parsed, sample)

        return SampleResult(
            sample_id=sample.sample_id,
            image_class=sample.image_class,
            task=task,
            success=parsed.success,
            parse_error=parsed.parse_error if hasattr(parsed, 'parse_error') else None,
            metrics=metrics_obj.to_dict() if hasattr(metrics_obj, 'to_dict') else {},
            raw_response=response.content,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    def _compute_metrics(
        self,
        task: str,
        parsed: Any,
        sample: BenchmarkSample,
    ) -> Any:
        """Compute metrics for a parsed response.

        Args:
            task: Task type
            parsed: Parsed response object
            sample: Original benchmark sample

        Returns:
            Task-specific metrics object
        """
        gt = sample.ground_truth

        if task == "COUNT":
            return self.metrics.compute_count_metrics(
                predicted=parsed.count,
                ground_truth=gt.spot_count,
            )

        elif task == "PATTERN":
            expected_pattern = get_expected_pattern(gt)
            return self.metrics.compute_pattern_metrics(
                predicted=parsed.pattern,
                ground_truth=expected_pattern,
            )

        elif task == "LOCATE":
            # Convert ground truth positions to normalized (0-1) coordinates
            # Ground truth is in micrometers, convert to pixels then normalize
            scale = 1.0 / sample.scale_inverse  # pixels per micrometer
            gt_positions = [
                (
                    (pos.x_um * scale) / sample.width,   # normalize x to [0, 1]
                    (pos.y_um * scale) / sample.height,  # normalize y to [0, 1]
                )
                for pos in gt.positions
            ]
            return self.metrics.compute_locate_metrics(
                predicted_positions=parsed.positions,
                ground_truth_positions=gt_positions,
            )

        elif task == "SIZE":
            gt_size = gt.metadata.get("spot_size", 0)
            return self.metrics.compute_size_metrics(
                predicted=parsed.diameter_um,
                ground_truth=float(gt_size),
            )

        elif task == "DEFECT":
            gt_has_defects = gt.has_defects
            gt_defect_rate = gt.metadata.get("defect_rate", 0)
            gt_noise_rate = gt.metadata.get("noise_rate", 0)

            return self.metrics.compute_defect_metrics(
                predicted_has_defects=parsed.has_defects,
                ground_truth_has_defects=gt_has_defects,
                predicted_missing=parsed.missing_count,
                ground_truth_missing=None,  # We don't have exact counts in ground truth
                predicted_noise=parsed.noise_count,
                ground_truth_noise=None,
            )

        else:
            raise ValueError(f"Unknown task: {task}")

    def _compile_results(self) -> EvaluationResults:
        """Compile all sample results into aggregated metrics.

        Returns:
            EvaluationResults with all aggregated data
        """
        # Group results by task
        by_task: Dict[str, List[SampleResult]] = {}
        for r in self.sample_results:
            if r.task not in by_task:
                by_task[r.task] = []
            by_task[r.task].append(r)

        # Group results by class and task
        by_class: Dict[str, Dict[str, List[SampleResult]]] = {}
        for r in self.sample_results:
            if r.image_class not in by_class:
                by_class[r.image_class] = {}
            if r.task not in by_class[r.image_class]:
                by_class[r.image_class][r.task] = []
            by_class[r.image_class][r.task].append(r)

        # Aggregate by task
        results_by_task = {}
        for task, results in by_task.items():
            successful = [r for r in results if r.success]
            metrics_list = [self._extract_metrics_obj(r, task) for r in successful]

            aggregated = self._aggregate_task_metrics(task, metrics_list)
            aggregated["total_samples"] = len(results)
            aggregated["successful_samples"] = len(successful)
            aggregated["success_rate"] = (len(successful) / len(results)) * 100 if results else 0

            results_by_task[task] = aggregated

        # Aggregate by class
        results_by_class = {}
        for cls, task_results in by_class.items():
            results_by_class[cls] = {}
            for task, results in task_results.items():
                successful = [r for r in results if r.success]
                metrics_list = [self._extract_metrics_obj(r, task) for r in successful]

                aggregated = self._aggregate_task_metrics(task, metrics_list)
                aggregated["total_samples"] = len(results)
                aggregated["successful_samples"] = len(successful)
                aggregated["success_rate"] = (len(successful) / len(results)) * 100 if results else 0

                results_by_class[cls][task] = aggregated

        # Overall stats
        total = len(self.sample_results)
        successful = sum(1 for r in self.sample_results if r.success)

        overall = {
            "total_samples": total,
            "successful_samples": successful,
            "success_rate": (successful / total) * 100 if total else 0,
        }

        # Usage stats
        usage = {
            "elapsed_seconds": round(self.usage_stats.elapsed_seconds, 1),
            "total_requests": self.usage_stats.total_requests,
            "failed_requests": self.usage_stats.failed_requests,
            "success_rate": round(self.usage_stats.success_rate, 1),
            "input_tokens": self.usage_stats.input_tokens,
            "output_tokens": self.usage_stats.output_tokens,
            "total_tokens": self.usage_stats.total_tokens,
        }

        if self.config.backend in ("openrouter", "anthropic"):
            cost = self.usage_stats.estimate_cost(self.config.model_name)
            if cost:
                usage["estimated_cost_usd"] = round(cost, 4)

        # Config for report
        config_dict = {
            "model": self.config.model_name,
            "backend": self.config.backend,
            "testsuite": self.config.testsuite_path,
            "tasks": self.config.tasks,
            "image_classes": self.config.image_classes,
            "num_shots": self.config.num_shots,
            "num_samples": self.config.num_samples,
        }

        return EvaluationResults(
            config=config_dict,
            usage=usage,
            results_by_task=results_by_task,
            results_by_class=results_by_class,
            overall=overall,
            sample_results=self.sample_results,
            timestamp=datetime.now().isoformat(),
        )

    def _extract_metrics_obj(self, result: SampleResult, task: str) -> Any:
        """Recreate metrics object from stored dict.

        Args:
            result: SampleResult with metrics dict
            task: Task type

        Returns:
            Task-specific metrics object
        """
        m = result.metrics
        if not m:
            return None

        if task == "COUNT":
            return CountMetrics(
                exact_match=m.get("exact_match", False),
                within_tolerance=m.get("within_tolerance", False),
                absolute_error=m.get("absolute_error", 0),
                percentage_error=m.get("percentage_error"),
                predicted=m.get("predicted"),
                ground_truth=m.get("ground_truth", 0),
            )
        elif task == "PATTERN":
            return PatternMetrics(
                correct=m.get("correct", False),
                predicted=m.get("predicted"),
                ground_truth=m.get("ground_truth", ""),
            )
        elif task == "LOCATE":
            return LocateMetrics(
                predicted_count=m.get("predicted_count", 0),
                ground_truth_count=m.get("ground_truth_count", 0),
                count_difference=m.get("count_difference", 0),
                mean_nearest_distance=m.get("mean_nearest_distance"),
                detection_rate=m.get("detection_rate", 0),
                false_positive_rate=m.get("false_positive_rate", 0),
            )
        elif task == "SIZE":
            return SizeMetrics(
                predicted=m.get("predicted"),
                ground_truth=m.get("ground_truth", 0),
                absolute_error=m.get("absolute_error"),
                percentage_error=m.get("percentage_error"),
                within_tolerance=m.get("within_tolerance", False),
            )
        elif task == "DEFECT":
            return DefectMetrics(
                defect_detection_correct=m.get("defect_detection_correct", False),
                predicted_has_defects=m.get("predicted_has_defects"),
                ground_truth_has_defects=m.get("ground_truth_has_defects", False),
                missing_count_error=m.get("missing_count_error"),
                noise_count_error=m.get("noise_count_error"),
            )

        return None

    def _aggregate_task_metrics(self, task: str, metrics_list: List[Any]) -> Dict[str, Any]:
        """Aggregate metrics for a specific task.

        Args:
            task: Task type
            metrics_list: List of task-specific metrics objects

        Returns:
            Aggregated metrics dictionary
        """
        # Filter out None values
        metrics_list = [m for m in metrics_list if m is not None]

        if not metrics_list:
            return {}

        if task == "COUNT":
            return self.metrics.aggregate_count_metrics(metrics_list)
        elif task == "PATTERN":
            return self.metrics.aggregate_pattern_metrics(metrics_list)
        elif task == "LOCATE":
            return self.metrics.aggregate_locate_metrics(metrics_list)
        elif task == "SIZE":
            return self.metrics.aggregate_size_metrics(metrics_list)
        elif task == "DEFECT":
            return self.metrics.aggregate_defect_metrics(metrics_list)

        return {}


def run_evaluation_sync(config: EvaluationConfig) -> EvaluationResults:
    """Synchronous wrapper for running evaluation.

    Args:
        config: Evaluation configuration

    Returns:
        EvaluationResults
    """
    evaluator = GeometryBenchEvaluator(config)
    return asyncio.run(evaluator.run_evaluation())
