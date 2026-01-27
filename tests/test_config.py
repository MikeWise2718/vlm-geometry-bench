"""Tests for configuration module."""

import pytest
from vlm_geometry_bench.config import EvaluationConfig, VALID_IMAGE_CLASSES, VALID_TASKS


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        assert config.backend == "ollama"
        assert config.model_name == "llava:7b"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.0

    def test_openrouter_auto_url(self):
        """Test that OpenRouter backend sets correct URL."""
        config = EvaluationConfig(backend="openrouter")
        assert "openrouter.ai" in config.base_url

    def test_invalid_backend_raises(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            EvaluationConfig(backend="invalid")

    def test_invalid_num_shots_raises(self):
        """Test that invalid num_shots raises ValueError."""
        with pytest.raises(ValueError, match="Invalid num_shots"):
            EvaluationConfig(num_shots=2)

    def test_invalid_image_class_raises(self):
        """Test that invalid image class raises ValueError."""
        with pytest.raises(ValueError, match="Invalid image class"):
            EvaluationConfig(image_classes=["INVALID"])

    def test_invalid_task_raises(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="Invalid task"):
            EvaluationConfig(tasks=["INVALID"])

    def test_api_endpoint_ollama(self):
        """Test API endpoint for Ollama backend."""
        config = EvaluationConfig(backend="ollama")
        assert "/v1/chat/completions" in config.api_endpoint

    def test_api_endpoint_openrouter(self):
        """Test API endpoint for OpenRouter backend."""
        config = EvaluationConfig(backend="openrouter")
        assert "/chat/completions" in config.api_endpoint

    def test_url_prefix_auto_added(self):
        """Test that http:// is added to URL without prefix."""
        config = EvaluationConfig(base_url="localhost:11434")
        assert config.base_url.startswith("http://")
