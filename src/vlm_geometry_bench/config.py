"""Configuration for VLM Geometry Bench evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional
import os


# Valid image classes from imagegen test suite
VALID_IMAGE_CLASSES = {"CTRL", "USSS", "USDS", "HSFR", "HSRP", "HSDN"}

# Valid evaluation tasks
VALID_TASKS = {"COUNT", "LOCATE", "PATTERN", "SIZE", "DEFECT"}

# Valid backends
VALID_BACKENDS = {"ollama", "openrouter"}


@dataclass
class EvaluationConfig:
    """Configuration for running VLM Geometry Bench evaluation."""

    # Backend settings
    backend: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    model_name: str = "llava:7b"

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

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Validate/fix base_url - must have http:// or https:// prefix
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            self.base_url = f"http://{self.base_url}"

        # Set default base_url based on backend
        if self.backend == "openrouter" and self.base_url == "http://localhost:11434":
            self.base_url = "https://openrouter.ai/api/v1"

        # Get API key from environment if not provided
        if self.api_key is None and self.backend == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY")

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

    @property
    def api_endpoint(self) -> str:
        """Get the chat completions API endpoint."""
        if self.backend == "ollama":
            return f"{self.base_url}/v1/chat/completions"
        else:  # openrouter
            return f"{self.base_url}/chat/completions"
