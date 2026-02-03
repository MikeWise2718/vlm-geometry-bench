"""VLM Geometry Bench - Benchmark for VLM geometric shape identification."""

__version__ = "0.1.0"

from .config import EvaluationConfig
from .vision_client import VisionClient, VisionResponse
from .data_loader import SuiteLoader, BenchmarkSample, GroundTruth, SpotPosition
from .response_parser import ResponseParser
from .metrics import MetricsCalculator, SampleResult
from .evaluator import GeometryBenchEvaluator, EvaluationResults, run_evaluation_sync
from .reporter import ResultsReporter, report_results

__all__ = [
    "EvaluationConfig",
    "VisionClient",
    "VisionResponse",
    "SuiteLoader",
    "BenchmarkSample",
    "GroundTruth",
    "SpotPosition",
    "ResponseParser",
    "MetricsCalculator",
    "SampleResult",
    "GeometryBenchEvaluator",
    "EvaluationResults",
    "run_evaluation_sync",
    "ResultsReporter",
    "report_results",
]
