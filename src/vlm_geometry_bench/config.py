"""Configuration for VLM Geometry Bench evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os


# Valid image classes from imagegen test suite
VALID_IMAGE_CLASSES = {"CTRL", "USSS", "USDS", "HSFR", "HSRP", "HSDN"}

# Valid evaluation tasks
VALID_TASKS = {"COUNT", "LOCATE", "PATTERN", "SIZE", "DEFECT"}

# Valid backends
VALID_BACKENDS = {"ollama", "openrouter", "anthropic"}


@dataclass
class ModelSpec:
    """Specification for a single model to evaluate."""

    model_name: str
    backend: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        """Set default base URL based on backend."""
        if self.base_url is None:
            if self.backend == "ollama":
                self.base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            elif self.backend == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"
            elif self.backend == "anthropic":
                self.base_url = "https://api.anthropic.com"

        # Get API key from environment if not provided
        if self.api_key is None:
            if self.backend == "openrouter":
                self.api_key = os.environ.get("OPENROUTER_API_KEY")
            elif self.backend == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    @property
    def api_endpoint(self) -> str:
        """Get the chat completions API endpoint."""
        if self.backend == "ollama":
            return f"{self.base_url}/v1/chat/completions"
        elif self.backend == "anthropic":
            return f"{self.base_url}/v1/messages"
        else:  # openrouter
            return f"{self.base_url}/chat/completions"


@dataclass
class TraceabilityConfig:
    """Configuration for traceability output."""

    enabled: bool = False
    results_dir: str = "./results"


@dataclass
class EvaluationConfig:
    """Configuration for running VLM Geometry Bench evaluation."""

    # Backend settings (for single-model compatibility)
    backend: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    model_name: str = "llava:7b"

    # Multi-model support
    models: Optional[List[ModelSpec]] = None  # If set, overrides single model settings

    # Run identification
    run_name: Optional[str] = None  # User-provided run name (defaults to model name or "comparison")
    comment: Optional[str] = None  # User comment describing the purpose of this run

    # Test suite settings
    testsuite_path: str = "./testsuite"
    image_classes: List[str] = field(
        default_factory=lambda: list(VALID_IMAGE_CLASSES)
    )
    num_samples: Optional[int] = None  # None = all samples

    # Evaluation settings
    tasks: List[str] = field(default_factory=lambda: ["COUNT", "PATTERN"])
    num_shots: int = 0  # 0, 3, or 5 (few-shot examples)
    count_tolerance: int = 2  # ±N for "within tolerance" counting metric

    # API settings
    timeout_seconds: int = 120
    retry_attempts: int = 3
    temperature: float = 0.0
    max_tokens: int = 512

    # Output settings
    output_dir: str = "./results"

    # Traceability settings
    traceability: TraceabilityConfig = field(default_factory=TraceabilityConfig)

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Validate/fix base_url - must have http:// or https:// prefix
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            self.base_url = f"http://{self.base_url}"

        # Set default base_url based on backend
        if self.backend == "openrouter" and self.base_url == "http://localhost:11434":
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.backend == "anthropic" and self.base_url == "http://localhost:11434":
            self.base_url = "https://api.anthropic.com"

        # Get API key from environment if not provided
        if self.api_key is None and self.backend == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
        elif self.api_key is None and self.backend == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Validate backend
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend: {self.backend}. Must be one of: {VALID_BACKENDS}"
            )

        # Validate num_shots
        if self.num_shots not in (0, 3, 5):
            raise ValueError(f"Invalid num_shots: {self.num_shots}. Must be 0, 3, or 5")

        # Validate image classes
        for cls in self.image_classes:
            if cls not in VALID_IMAGE_CLASSES:
                raise ValueError(
                    f"Invalid image class: {cls}. Must be one of: {VALID_IMAGE_CLASSES}"
                )

        # Validate tasks
        for task in self.tasks:
            if task not in VALID_TASKS:
                raise ValueError(
                    f"Invalid task: {task}. Must be one of: {VALID_TASKS}"
                )

        # If models list is provided, validate each
        if self.models:
            for model_spec in self.models:
                if model_spec.backend not in VALID_BACKENDS:
                    raise ValueError(
                        f"Invalid backend for model {model_spec.model_name}: {model_spec.backend}"
                    )

        # Set default run name if not provided
        if self.run_name is None:
            if self.models and len(self.models) > 1:
                self.run_name = "comparison"
            else:
                # Use model name as run name
                model = self.models[0].model_name if self.models else self.model_name
                self.run_name = model.replace("/", "_").replace(":", "-")

    @property
    def api_endpoint(self) -> str:
        """Get the chat completions API endpoint."""
        if self.backend == "ollama":
            return f"{self.base_url}/v1/chat/completions"
        elif self.backend == "anthropic":
            return f"{self.base_url}/v1/messages"
        else:  # openrouter
            return f"{self.base_url}/chat/completions"

    def get_model_specs(self) -> List[ModelSpec]:
        """Get list of models to evaluate.

        Returns list of ModelSpec objects. If models list is not set,
        creates a single ModelSpec from the single-model settings.
        """
        if self.models:
            return self.models

        # Create single ModelSpec from legacy settings
        return [
            ModelSpec(
                model_name=self.model_name,
                backend=self.backend,
                base_url=self.base_url,
                api_key=self.api_key,
            )
        ]

    def get_all_model_names(self) -> List[str]:
        """Get list of all model names being evaluated."""
        return [m.model_name for m in self.get_model_specs()]

    def get_all_backends(self) -> List[str]:
        """Get list of unique backends being used."""
        return list(set(m.backend for m in self.get_model_specs()))

    @property
    def is_multi_model(self) -> bool:
        """Check if this is a multi-model evaluation."""
        return self.models is not None and len(self.models) > 1
