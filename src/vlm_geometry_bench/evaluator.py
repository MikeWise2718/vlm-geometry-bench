"""Main evaluator orchestration for VLM Geometry Bench."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from PIL import Image
from tqdm import tqdm

from .config import EvaluationConfig, ModelSpec
from .data_loader import SuiteLoader, BenchmarkSample
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
class ModelEvaluationResults:
    """Results from evaluating a single model."""

    model: str
    backend: str
    base_url: str
    usage: UsageStats
    sample_results: List[SampleResult]
    results_by_task: Dict[str, Dict[str, Any]]
    results_by_class: Dict[str, Dict[str, Dict[str, Any]]]
    overall: Dict[str, Any]


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
    # Multi-model support
    model_results: Optional[Dict[str, ModelEvaluationResults]] = None


class GeometryBenchEvaluator:
    """Main evaluator for VLM Geometry Bench."""

    def __init__(self, config: EvaluationConfig):
        """Initialize evaluator with configuration.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.loader = SuiteLoader(config.testsuite_path)
        self.parser = ResponseParser()
        self.metrics = MetricsCalculator(count_tolerance=config.count_tolerance)

        # Per-model tracking
        self.model_results: Dict[str, ModelEvaluationResults] = {}
        self.all_sample_results: List[SampleResult] = []

        # Traceability components (lazy initialized)
        self._artifact_manager = None
        self._html_generator = None
        self._image_annotator = None
        self._index_manager = None

    def _init_traceability(self):
        """Initialize traceability components if enabled."""
        if not self.config.traceability.enabled:
            return

        from .traceability import (
            ArtifactManager,
            HTMLGenerator,
            ImageAnnotator,
            IndexManager,
        )

        self._artifact_manager = ArtifactManager(
            results_dir=self.config.traceability.results_dir,
            run_name=self.config.run_name,
        )
        self._html_generator = HTMLGenerator()
        self._image_annotator = ImageAnnotator()
        self._index_manager = IndexManager(self.config.traceability.results_dir)

        # Setup folder structure
        self._artifact_manager.setup_run_structure()

    async def run_evaluation(self) -> EvaluationResults:
        """Run the complete evaluation.

        Returns:
            EvaluationResults with all metrics and raw data
        """
        logger.info(f"Starting evaluation")

        # Initialize traceability if enabled
        self._init_traceability()

        # Load samples
        samples = self.loader.load_samples(
            image_classes=self.config.image_classes,
            limit=self.config.num_samples,
        )

        if not samples:
            raise ValueError("No samples loaded. Check testsuite path and image classes.")

        logger.info(f"Loaded {len(samples)} samples")

        # Copy original images to shared folder if traceability enabled
        if self._artifact_manager:
            for sample in samples:
                self._artifact_manager.copy_original_image(
                    Path(sample.image_path),
                    sample.sample_id,
                )

        # Get models to evaluate
        model_specs = self.config.get_model_specs()

        # Run evaluation for each model
        for model_spec in model_specs:
            logger.info(f"Evaluating model: {model_spec.model_name} ({model_spec.backend})")

            model_results = await self._evaluate_model(model_spec, samples)
            self.model_results[model_spec.model_name] = model_results
            self.all_sample_results.extend(model_results.sample_results)

        # Compile combined results
        results = self._compile_results()

        # Generate traceability artifacts
        if self._artifact_manager:
            await self._generate_traceability_artifacts(samples, results)

        return results

    async def _evaluate_model(
        self,
        model_spec: ModelSpec,
        samples: List[BenchmarkSample],
    ) -> ModelEvaluationResults:
        """Evaluate a single model on all samples.

        Args:
            model_spec: Model specification
            samples: List of benchmark samples

        Returns:
            ModelEvaluationResults for this model
        """
        # Create a temporary config for this model
        model_config = EvaluationConfig(
            backend=model_spec.backend,
            base_url=model_spec.base_url,
            api_key=model_spec.api_key,
            model_name=model_spec.model_name,
            testsuite_path=self.config.testsuite_path,
            image_classes=self.config.image_classes,
            tasks=self.config.tasks,
            num_shots=self.config.num_shots,
            num_samples=self.config.num_samples,
            timeout_seconds=self.config.timeout_seconds,
            retry_attempts=self.config.retry_attempts,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            count_tolerance=self.config.count_tolerance,
        )

        usage_stats = UsageStats()
        sample_results: List[SampleResult] = []

        # Setup model folder if traceability enabled
        if self._artifact_manager:
            self._artifact_manager.setup_model_structure(model_spec.model_name)

        async with VisionClient(model_config) as client:
            for task in self.config.tasks:
                logger.info(f"  Evaluating task: {task}")
                task_results = await self._evaluate_task(
                    client, samples, task, model_spec.model_name, usage_stats
                )
                sample_results.extend(task_results)

        # Mark end time
        usage_stats.end_time = time.time()

        # Compile per-model results
        results_by_task, results_by_class, overall = self._compile_model_results(sample_results, task)

        return ModelEvaluationResults(
            model=model_spec.model_name,
            backend=model_spec.backend,
            base_url=model_spec.base_url,
            usage=usage_stats,
            sample_results=sample_results,
            results_by_task=results_by_task,
            results_by_class=results_by_class,
            overall=overall,
        )

    async def _evaluate_task(
        self,
        client: VisionClient,
        samples: List[BenchmarkSample],
        task: str,
        model_name: str,
        usage_stats: UsageStats,
    ) -> List[SampleResult]:
        """Evaluate all samples for a specific task.

        Args:
            client: VisionClient for API calls
            samples: List of benchmark samples
            task: Task to evaluate (COUNT, PATTERN, etc.)
            model_name: Model name for tracking
            usage_stats: Usage stats to update

        Returns:
            List of SampleResult objects
        """
        results = []
        desc = f"    {task}"

        for sample in tqdm(samples, desc=desc):
            try:
                result = await self._evaluate_single(
                    client, sample, task, model_name, usage_stats
                )
                results.append(result)

                # Save test artifacts if traceability enabled
                if self._artifact_manager:
                    await self._save_test_artifacts(sample, result)

            except Exception as e:
                logger.error(f"Error evaluating {sample.sample_id}/{task}: {e}")
                result = SampleResult(
                    sample_id=sample.sample_id,
                    image_class=sample.image_class,
                    task=task,
                    success=False,
                    parse_error=str(e),
                    metrics={},
                    raw_response="",
                    model=model_name,
                )
                results.append(result)
                usage_stats.failed_requests += 1

        return results

    async def _evaluate_single(
        self,
        client: VisionClient,
        sample: BenchmarkSample,
        task: str,
        model_name: str,
        usage_stats: UsageStats,
    ) -> SampleResult:
        """Evaluate a single sample for a specific task.

        Args:
            client: VisionClient for API calls
            sample: Benchmark sample with image and ground truth
            task: Task to evaluate
            model_name: Model name for tracking
            usage_stats: Usage stats to update

        Returns:
            SampleResult with metrics
        """
        # Build prompt
        prompt = get_prompt_for_sample(task, sample, self.config.num_shots)

        # Send request
        response = await client.send_request(sample.image, prompt)

        # Track usage
        usage_stats.total_requests += 1
        usage_stats.input_tokens += response.input_tokens
        usage_stats.output_tokens += response.output_tokens

        if not response.success:
            usage_stats.failed_requests += 1
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
                model=model_name,
                prompt=prompt,
            )

        # Parse response and compute metrics
        parsed = self.parser.parse(task, response.content)
        metrics_obj = self._compute_metrics(task, parsed, sample)

        # Build ground truth and prediction data for traceability
        ground_truth_data = self._build_ground_truth_data(task, sample)
        prediction_data = self._build_prediction_data(task, parsed)

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
            model=model_name,
            prompt=prompt,
            ground_truth=ground_truth_data,
            prediction=prediction_data,
        )

    def _build_ground_truth_data(self, task: str, sample: BenchmarkSample) -> Dict[str, Any]:
        """Build ground truth data dict for traceability."""
        gt = sample.ground_truth
        data = {
            "spot_count": gt.spot_count,
        }

        if task == "LOCATE":
            # Convert positions to normalized coordinates
            scale = 1.0 / sample.scale_inverse
            positions = [
                (
                    (pos.x_um * scale) / sample.width,
                    (pos.y_um * scale) / sample.height,
                )
                for pos in gt.positions
            ]
            data["positions"] = positions

        if task == "PATTERN":
            data["pattern"] = get_expected_pattern(gt)

        if task == "SIZE":
            data["size"] = gt.metadata.get("spot_size", 0)

        if task == "DEFECT":
            data["has_defects"] = gt.has_defects

        return data

    def _build_prediction_data(self, task: str, parsed: Any) -> Dict[str, Any]:
        """Build prediction data dict for traceability."""
        data = {}

        if task == "COUNT":
            data["count"] = parsed.count

        elif task == "LOCATE":
            data["positions"] = parsed.positions if parsed.positions else []

        elif task == "PATTERN":
            data["pattern"] = parsed.pattern

        elif task == "SIZE":
            data["diameter_um"] = parsed.diameter_um

        elif task == "DEFECT":
            data["has_defects"] = parsed.has_defects
            data["missing_count"] = parsed.missing_count
            data["noise_count"] = parsed.noise_count

        return data

    async def _save_test_artifacts(
        self,
        sample: BenchmarkSample,
        result: SampleResult,
    ) -> None:
        """Save test artifacts for traceability.

        Args:
            sample: Benchmark sample
            result: Sample result
        """
        if not self._artifact_manager or not result.model:
            return

        from .traceability.schemas import (
            SampleTestResult,
            ConversationHistory,
            ConversationTurn,
        )

        model = result.model
        task = result.task

        # Create symlink to original image
        test_dir = self._artifact_manager.get_test_dir(model, result.sample_id, task)
        self._artifact_manager.create_image_symlink(test_dir, result.sample_id)

        # Build test metadata
        test_metadata = SampleTestResult(
            test_id=f"{result.sample_id}_{task}",
            sample_id=result.sample_id,
            model=model,
            task=task,
            image_class=result.image_class,
            timestamp=datetime.now(),
            total_latency_ms=result.latency_ms or 0,
            total_input_tokens=result.input_tokens,
            total_output_tokens=result.output_tokens,
            estimated_cost_usd=self._estimate_single_cost(
                result.input_tokens, result.output_tokens, model
            ),
            num_turns=1,
            success=result.success,
            parse_error=result.parse_error,
            metrics=result.metrics,
            ground_truth=result.ground_truth or {},
            prediction=result.prediction or {},
        )

        # Save test metadata
        self._artifact_manager.save_json(
            test_metadata,
            self._artifact_manager.test_metadata_path(model, result.sample_id, task),
        )

        # Build conversation history
        conversation = ConversationHistory(
            test_id=f"{result.sample_id}_{task}",
            model=model,
            turns=[
                ConversationTurn(
                    turn=1,
                    role="user",
                    content=result.prompt or "",
                    image_attached=True,
                    timestamp=datetime.now(),
                ),
                ConversationTurn(
                    turn=1,
                    role="assistant",
                    content=result.raw_response,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                ),
            ],
            final_response=result.raw_response,
        )

        # Save conversation
        self._artifact_manager.save_json(
            conversation,
            self._artifact_manager.conversation_path(model, result.sample_id, task),
        )

        # Generate and save annotated image
        if self._image_annotator:
            try:
                # Load original image
                original_path = self._artifact_manager.get_image_path_for_sample(result.sample_id)
                image = Image.open(original_path)

                annotated = self._image_annotator.annotate(
                    image=image,
                    task=task,
                    sample_id=result.sample_id,
                    model=model,
                    metrics=result.metrics,
                    ground_truth=result.ground_truth or {},
                    prediction=result.prediction or {},
                )

                annotated.save(
                    self._artifact_manager.annotated_image_path(model, result.sample_id, task),
                    "PNG",
                )
            except Exception as e:
                logger.warning(f"Failed to generate annotated image: {e}")

    def _estimate_single_cost(
        self, input_tokens: int, output_tokens: int, model_name: str
    ) -> Optional[float]:
        """Estimate cost for a single request."""
        pricing = MODEL_PRICING.get("default")
        if model_name in MODEL_PRICING:
            pricing = MODEL_PRICING[model_name]
        else:
            model_lower = model_name.lower()
            for pattern, costs in MODEL_PRICING.items():
                if pattern != "default" and pattern in model_lower:
                    pricing = costs
                    break

        input_cost, output_cost = pricing
        return (input_tokens / 1_000_000 * input_cost +
                output_tokens / 1_000_000 * output_cost)

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

    def _compile_model_results(
        self,
        sample_results: List[SampleResult],
        task: str,
    ) -> Tuple[Dict, Dict, Dict]:
        """Compile results for a single model."""
        # Group results by task
        by_task: Dict[str, List[SampleResult]] = {}
        for r in sample_results:
            if r.task not in by_task:
                by_task[r.task] = []
            by_task[r.task].append(r)

        # Group results by class and task
        by_class: Dict[str, Dict[str, List[SampleResult]]] = {}
        for r in sample_results:
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
        total = len(sample_results)
        successful = sum(1 for r in sample_results if r.success)

        overall = {
            "total_samples": total,
            "successful_samples": successful,
            "success_rate": (successful / total) * 100 if total else 0,
        }

        return results_by_task, results_by_class, overall

    def _compile_results(self) -> EvaluationResults:
        """Compile all sample results into aggregated metrics.

        Returns:
            EvaluationResults with all aggregated data
        """
        # If single model, use its results directly
        if len(self.model_results) == 1:
            model_name = list(self.model_results.keys())[0]
            model_result = self.model_results[model_name]

            # Usage stats
            usage = {
                "elapsed_seconds": round(model_result.usage.elapsed_seconds, 1),
                "total_requests": model_result.usage.total_requests,
                "failed_requests": model_result.usage.failed_requests,
                "success_rate": round(model_result.usage.success_rate, 1),
                "input_tokens": model_result.usage.input_tokens,
                "output_tokens": model_result.usage.output_tokens,
                "total_tokens": model_result.usage.total_tokens,
            }

            if model_result.backend in ("openrouter", "anthropic"):
                cost = model_result.usage.estimate_cost(model_name)
                if cost:
                    usage["estimated_cost_usd"] = round(cost, 4)

            # Config for report
            config_dict = {
                "model": model_name,
                "backend": model_result.backend,
                "testsuite": self.config.testsuite_path,
                "tasks": self.config.tasks,
                "image_classes": self.config.image_classes,
                "num_shots": self.config.num_shots,
                "num_samples": self.config.num_samples,
            }

            return EvaluationResults(
                config=config_dict,
                usage=usage,
                results_by_task=model_result.results_by_task,
                results_by_class=model_result.results_by_class,
                overall=model_result.overall,
                sample_results=model_result.sample_results,
                timestamp=datetime.now().isoformat(),
                model_results=self.model_results,
            )

        # Multi-model: combine results
        total_elapsed = sum(mr.usage.elapsed_seconds for mr in self.model_results.values())
        total_requests = sum(mr.usage.total_requests for mr in self.model_results.values())
        failed_requests = sum(mr.usage.failed_requests for mr in self.model_results.values())
        total_input = sum(mr.usage.input_tokens for mr in self.model_results.values())
        total_output = sum(mr.usage.output_tokens for mr in self.model_results.values())
        total_cost = sum(
            mr.usage.estimate_cost(mr.model) or 0
            for mr in self.model_results.values()
        )

        usage = {
            "elapsed_seconds": round(total_elapsed, 1),
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "success_rate": round(
                ((total_requests - failed_requests) / total_requests) * 100, 1
            ) if total_requests else 0,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

        if total_cost > 0:
            usage["estimated_cost_usd"] = round(total_cost, 4)

        # Combined metrics (use first model's structure for consistency)
        first_model = list(self.model_results.values())[0]

        config_dict = {
            "models": self.config.get_all_model_names(),
            "backends": self.config.get_all_backends(),
            "testsuite": self.config.testsuite_path,
            "tasks": self.config.tasks,
            "image_classes": self.config.image_classes,
            "num_shots": self.config.num_shots,
            "num_samples": self.config.num_samples,
        }

        return EvaluationResults(
            config=config_dict,
            usage=usage,
            results_by_task=first_model.results_by_task,  # Use first model's task breakdown
            results_by_class=first_model.results_by_class,
            overall=first_model.overall,
            sample_results=self.all_sample_results,
            timestamp=datetime.now().isoformat(),
            model_results=self.model_results,
        )

    async def _generate_traceability_artifacts(
        self,
        samples: List[BenchmarkSample],
        results: EvaluationResults,
    ) -> None:
        """Generate all traceability artifacts after evaluation.

        Args:
            samples: List of benchmark samples
            results: Complete evaluation results
        """
        if not self._artifact_manager:
            return

        from .traceability.schemas import (
            RunMetadata,
            ModelRunInfo,
            ModelMetadata,
            RunIndexEntry,
        )

        logger.info("Generating traceability artifacts...")

        # Build run metadata
        model_run_infos = []
        for model_name, mr in self.model_results.items():
            model_run_infos.append(ModelRunInfo(
                model=model_name,
                backend=mr.backend,
                base_url=mr.base_url,
                elapsed_seconds=mr.usage.elapsed_seconds,
                total_tests=mr.usage.total_requests,
                success_rate=mr.usage.success_rate,
                input_tokens=mr.usage.input_tokens,
                output_tokens=mr.usage.output_tokens,
                estimated_cost_usd=mr.usage.estimate_cost(model_name),
            ))

        run_metadata = RunMetadata(
            run_id=self._artifact_manager.run_id,
            run_name=self.config.run_name,
            comment=self.config.comment,
            timestamp_start=self._artifact_manager.timestamp,
            timestamp_end=datetime.now(),
            tasks=self.config.tasks,
            image_classes=self.config.image_classes,
            num_samples=len(samples),
            models=model_run_infos,
        )

        # Save run metadata
        self._artifact_manager.save_json(
            run_metadata,
            self._artifact_manager.run_metadata_path(),
        )

        # Save model metadata for each model
        for model_name, mr in self.model_results.items():
            model_metadata = ModelMetadata(
                model=model_name,
                backend=mr.backend,
                base_url=mr.base_url,
                run_id=self._artifact_manager.run_id,
                elapsed_seconds=mr.usage.elapsed_seconds,
                total_tests=mr.usage.total_requests,
                successful_tests=mr.usage.total_requests - mr.usage.failed_requests,
                success_rate=mr.usage.success_rate,
                input_tokens=mr.usage.input_tokens,
                output_tokens=mr.usage.output_tokens,
                estimated_cost_usd=mr.usage.estimate_cost(model_name),
                results_by_task=mr.results_by_task,
                results_by_class=mr.results_by_class,
            )

            self._artifact_manager.save_json(
                model_metadata,
                self._artifact_manager.model_metadata_path(model_name),
            )

        # Generate test HTML pages
        self._generate_test_html_pages()

        # Generate summary HTML
        self._generate_summary_html(run_metadata, results)

        # Save assets
        self._html_generator.save_assets(self._artifact_manager.assets_dir)
        self._html_generator.save_assets(self._artifact_manager.global_assets_dir())

        # Update global index
        self._update_global_index(run_metadata, results)

        # Generate global index HTML
        self._generate_index_html()

        logger.info(f"Traceability artifacts saved to: {self._artifact_manager.run_dir}")

    def _generate_test_html_pages(self) -> None:
        """Generate HTML pages for each test."""
        if not self._artifact_manager or not self._html_generator:
            return

        for model_name, mr in self.model_results.items():
            for result in mr.sample_results:
                test_dir = self._artifact_manager.get_test_dir(
                    model_name, result.sample_id, result.task
                )

                # Load test metadata
                metadata_path = self._artifact_manager.test_metadata_path(
                    model_name, result.sample_id, result.task
                )

                if not metadata_path.exists():
                    continue

                import json
                with open(metadata_path) as f:
                    test_data = json.load(f)

                # Load conversation
                conv_path = self._artifact_manager.conversation_path(
                    model_name, result.sample_id, result.task
                )
                conversation = None
                if conv_path.exists():
                    with open(conv_path) as f:
                        conversation = json.load(f)

                # Find other models' results for same sample
                other_models = []
                for other_model, other_mr in self.model_results.items():
                    for other_result in other_mr.sample_results:
                        if (other_result.sample_id == result.sample_id and
                            other_result.task == result.task):
                            other_models.append({
                                "model": other_model,
                                "success": other_result.success,
                                "metrics": other_result.metrics,
                                "total_latency_ms": other_result.latency_ms or 0,
                                "estimated_cost_usd": self._estimate_single_cost(
                                    other_result.input_tokens,
                                    other_result.output_tokens,
                                    other_model,
                                ),
                            })

                # Generate HTML
                html = self._html_generator.generate_test_html(
                    test=test_data,
                    conversation=conversation,
                    other_models=other_models if len(other_models) > 1 else None,
                )

                self._artifact_manager.save_html(
                    html,
                    self._artifact_manager.test_html_path(model_name, result.sample_id, result.task),
                )

    def _generate_summary_html(self, run_metadata, results: EvaluationResults) -> None:
        """Generate summary HTML page for this run."""
        if not self._artifact_manager or not self._html_generator:
            return

        # Collect all test metadata
        tests = []
        for model_name, mr in self.model_results.items():
            for result in mr.sample_results:
                tests.append({
                    "sample_id": result.sample_id,
                    "model": model_name,
                    "task": result.task,
                    "image_class": result.image_class,
                    "success": result.success,
                    "metrics": result.metrics,
                    "total_latency_ms": result.latency_ms or 0,
                })

        # Build model metrics dict
        model_metrics = {}
        for model_name, mr in self.model_results.items():
            model_metrics[model_name] = mr.results_by_task

        # Generate HTML
        html = self._html_generator.generate_summary_html(
            run=run_metadata.model_dump(mode="json"),
            tests=tests,
            model_metrics=model_metrics,
        )

        self._artifact_manager.save_html(html, self._artifact_manager.summary_html_path())

    def _update_global_index(self, run_metadata, results: EvaluationResults) -> None:
        """Update the global index.json."""
        if not self._index_manager:
            return

        size_mb = self._artifact_manager.calculate_run_size_mb()
        total_cost = sum(
            mr.usage.estimate_cost(mr.model) or 0
            for mr in self.model_results.values()
        )

        entry = self._index_manager.create_entry_from_run(
            run_id=self._artifact_manager.run_id,
            run_name=self.config.run_name,
            comment=self.config.comment,
            models=self.config.get_all_model_names(),
            backends=self.config.get_all_backends(),
            timestamp=self._artifact_manager.timestamp,
            elapsed_seconds=sum(mr.usage.elapsed_seconds for mr in self.model_results.values()),
            total_tests=sum(mr.usage.total_requests for mr in self.model_results.values()),
            estimated_cost_usd=total_cost if total_cost > 0 else None,
            size_mb=size_mb,
            tasks=self.config.tasks,
            image_classes=self.config.image_classes,
        )

        self._index_manager.add_run(entry)

    def _generate_index_html(self) -> None:
        """Generate global index HTML page."""
        if not self._index_manager or not self._html_generator:
            return

        runs = self._index_manager.list_runs()

        html = self._html_generator.generate_index_html(
            runs=[r.model_dump(mode="json") for r in runs],
        )

        index_path = self._artifact_manager.global_index_html_path()
        self._artifact_manager.save_html(html, index_path)

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
