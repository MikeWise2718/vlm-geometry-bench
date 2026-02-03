"""Pydantic schemas for traceability data structures.

These schemas define the JSON structure for all artifacts stored during evaluation runs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# MASTER INDEX SCHEMAS
# =============================================================================


class RunIndexEntry(BaseModel):
    """Entry in the master index.json listing all test runs."""

    run_id: str = Field(description="Unique run identifier (YYYYMMDD_HHMMSS_runname)")
    run_name: str = Field(description="User-provided run name")
    comment: Optional[str] = Field(default=None, description="User comment describing run purpose")
    models: List[str] = Field(description="List of model names evaluated in this run")
    backends: List[str] = Field(description="List of backends used")
    timestamp: datetime = Field(description="Run start timestamp")
    elapsed_seconds: float = Field(description="Total elapsed time for the run")
    total_tests: int = Field(description="Total number of tests across all models")
    estimated_cost_usd: Optional[float] = Field(default=None, description="Estimated total cost")
    size_mb: Optional[float] = Field(default=None, description="Total artifact size in MB")
    tasks: List[str] = Field(description="List of tasks evaluated")
    image_classes: List[str] = Field(description="List of image classes evaluated")


class RunIndex(BaseModel):
    """Master index of all test runs."""

    version: str = Field(default="1.0", description="Schema version")
    runs: List[RunIndexEntry] = Field(default_factory=list, description="List of all runs")


# =============================================================================
# RUN METADATA SCHEMAS
# =============================================================================


class ModelRunInfo(BaseModel):
    """Information about a single model's evaluation within a run."""

    model: str = Field(description="Model name")
    backend: str = Field(description="Backend used (ollama, openrouter, anthropic)")
    base_url: Optional[str] = Field(default=None, description="API base URL (e.g., Ollama host)")
    elapsed_seconds: float = Field(description="Time taken for this model's evaluation")
    total_tests: int = Field(description="Number of tests run for this model")
    success_rate: float = Field(description="Percentage of successful API calls")
    input_tokens: int = Field(default=0, description="Total input tokens used")
    output_tokens: int = Field(default=0, description="Total output tokens used")
    estimated_cost_usd: Optional[float] = Field(default=None, description="Estimated cost for this model")


class RunMetadata(BaseModel):
    """Complete metadata for a test run (multi-model)."""

    run_id: str = Field(description="Unique run identifier")
    run_name: str = Field(description="User-provided run name")
    comment: Optional[str] = Field(default=None, description="User comment describing run purpose")
    timestamp_start: datetime = Field(description="Run start time")
    timestamp_end: Optional[datetime] = Field(default=None, description="Run end time")
    tasks: List[str] = Field(description="Tasks evaluated")
    image_classes: List[str] = Field(description="Image classes evaluated")
    num_samples: int = Field(description="Number of samples evaluated per model")
    models: List[ModelRunInfo] = Field(description="Per-model evaluation info")


# =============================================================================
# MODEL METADATA SCHEMA
# =============================================================================


class ModelMetadata(BaseModel):
    """Aggregated results for a single model within a run."""

    model: str = Field(description="Model name")
    backend: str = Field(description="Backend used")
    base_url: Optional[str] = Field(default=None, description="API base URL (e.g., Ollama host)")
    run_id: str = Field(description="Parent run ID")
    elapsed_seconds: float = Field(description="Time taken for evaluation")
    total_tests: int = Field(description="Total tests run")
    successful_tests: int = Field(description="Number of successful tests")
    success_rate: float = Field(description="Success rate percentage")
    input_tokens: int = Field(default=0, description="Total input tokens")
    output_tokens: int = Field(default=0, description="Total output tokens")
    estimated_cost_usd: Optional[float] = Field(default=None, description="Estimated cost")

    # Aggregated metrics by task
    results_by_task: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Aggregated metrics per task"
    )

    # Aggregated metrics by class
    results_by_class: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Aggregated metrics per image class"
    )


# =============================================================================
# TEST METADATA SCHEMAS
# =============================================================================


class TestMetadata(BaseModel):
    """Metadata for a single test (sample + task + model)."""

    test_id: str = Field(description="Unique test identifier (sample_id_task)")
    sample_id: str = Field(description="Sample identifier")
    model: str = Field(description="Model name")
    task: str = Field(description="Task type (COUNT, LOCATE, etc.)")
    image_class: str = Field(description="Image class (CTRL, USSS, etc.)")
    timestamp: datetime = Field(description="Test execution timestamp")
    total_latency_ms: int = Field(default=0, description="Total API latency in milliseconds")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    estimated_cost_usd: Optional[float] = Field(default=None, description="Estimated cost")
    num_turns: int = Field(default=1, description="Number of conversation turns")
    success: bool = Field(description="Whether the test succeeded")
    parse_error: Optional[str] = Field(default=None, description="Parse error if any")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Task-specific metrics")
    ground_truth: Dict[str, Any] = Field(default_factory=dict, description="Ground truth data")
    prediction: Dict[str, Any] = Field(default_factory=dict, description="Model prediction data")


# =============================================================================
# CONVERSATION HISTORY SCHEMAS
# =============================================================================


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    turn: int = Field(description="Turn number (1-indexed)")
    role: str = Field(description="Role: 'user' or 'assistant'")
    content: str = Field(description="Message content")
    image_attached: bool = Field(default=False, description="Whether an image was attached (user only)")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    latency_ms: Optional[int] = Field(default=None, description="Response latency (assistant only)")
    input_tokens: Optional[int] = Field(default=None, description="Input tokens (assistant only)")
    output_tokens: Optional[int] = Field(default=None, description="Output tokens (assistant only)")


class ConversationHistory(BaseModel):
    """Full conversation history for a test."""

    test_id: str = Field(description="Test identifier")
    model: str = Field(description="Model name")
    turns: List[ConversationTurn] = Field(default_factory=list, description="All conversation turns")
    final_response: Optional[str] = Field(default=None, description="Final assistant response used for parsing")
