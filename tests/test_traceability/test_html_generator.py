"""Tests for HTML generator."""

from datetime import datetime
from pathlib import Path

import pytest

from vlm_geometry_bench.traceability.html_generator import (
    HTMLGenerator,
    format_duration,
    safe_model_name,
)
from vlm_geometry_bench.traceability.schemas import RunIndexEntry


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_seconds(self):
        """Formats seconds."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(0.5) == "0.5s"

    def test_minutes(self):
        """Formats minutes and seconds."""
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 05s"
        assert format_duration(3599) == "59m 59s"

    def test_hours(self):
        """Formats hours and minutes."""
        assert format_duration(3600) == "1h 00m"
        assert format_duration(3660) == "1h 01m"
        assert format_duration(7200) == "2h 00m"
        assert format_duration(5400) == "1h 30m"


class TestSafeModelName:
    """Tests for safe_model_name function."""

    def test_simple_name(self):
        """Simple name unchanged."""
        assert safe_model_name("llava") == "llava"

    def test_colon_replaced(self):
        """Colons replaced with hyphens."""
        assert safe_model_name("llava:7b") == "llava-7b"

    def test_slash_replaced(self):
        """Slashes replaced with underscores."""
        assert safe_model_name("openai/gpt-4o") == "openai_gpt-4o"


class TestHTMLGenerator:
    """Tests for HTMLGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create HTMLGenerator."""
        return HTMLGenerator()

    @pytest.fixture
    def sample_runs(self):
        """Create sample run entries."""
        return [
            {
                "run_id": "20260202_130859_test",
                "run_name": "test",
                "comment": "Test comment",
                "models": ["llava:7b", "gpt-4o"],
                "backends": ["ollama", "openrouter"],
                "timestamp": "2026-02-02T13:08:59",
                "elapsed_seconds": 1680.5,
                "total_tests": 184,
                "estimated_cost_usd": 2.15,
                "size_mb": 135.6,
                "tasks": ["COUNT", "LOCATE"],
                "image_classes": ["CTRL", "USSS"],
            },
            {
                "run_id": "20260201_100000_baseline",
                "run_name": "baseline",
                "comment": None,
                "models": ["llava:7b"],
                "backends": ["ollama"],
                "timestamp": "2026-02-01T10:00:00",
                "elapsed_seconds": 300.0,
                "total_tests": 92,
                "estimated_cost_usd": None,
                "size_mb": 50.0,
                "tasks": ["COUNT"],
                "image_classes": ["USSS"],
            },
        ]

    @pytest.fixture
    def sample_run_metadata(self):
        """Create sample run metadata."""
        return {
            "run_id": "20260202_130859_test",
            "run_name": "test",
            "comment": "Test run for LOCATE task",
            "timestamp_start": "2026-02-02T12:59:38",
            "timestamp_end": "2026-02-02T13:27:38",
            "tasks": ["LOCATE"],
            "image_classes": ["CTRL", "USSS"],
            "num_samples": 92,
            "models": [
                {
                    "model": "llava:7b",
                    "backend": "ollama",
                    "base_url": "http://localhost:11434",
                    "elapsed_seconds": 300.0,
                    "total_tests": 92,
                    "success_rate": 95.5,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": None,
                },
                {
                    "model": "gpt-4o",
                    "backend": "openrouter",
                    "base_url": None,
                    "elapsed_seconds": 250.0,
                    "total_tests": 92,
                    "success_rate": 98.0,
                    "input_tokens": 50000,
                    "output_tokens": 30000,
                    "estimated_cost_usd": 1.50,
                },
            ],
        }

    @pytest.fixture
    def sample_tests(self):
        """Create sample test metadata list."""
        return [
            {
                "test_id": "USSS_s2_d20_wb_LOCATE",
                "sample_id": "USSS_s2_d20_wb",
                "model": "llava:7b",
                "task": "LOCATE",
                "image_class": "USSS",
                "timestamp": "2026-02-02T13:01:15",
                "total_latency_ms": 2500,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "estimated_cost_usd": None,
                "num_turns": 1,
                "success": True,
                "metrics": {"detection_rate": 45.0, "false_positive_rate": 30.0},
                "ground_truth": {"positions": [[0.5, 0.5]]},
                "prediction": {"positions": [[0.5, 0.5]]},
            },
            {
                "test_id": "CTRL_empty_wb_LOCATE",
                "sample_id": "CTRL_empty_wb",
                "model": "llava:7b",
                "task": "LOCATE",
                "image_class": "CTRL",
                "timestamp": "2026-02-02T13:00:15",
                "total_latency_ms": 1500,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "estimated_cost_usd": None,
                "num_turns": 1,
                "success": False,
                "parse_error": "Failed to parse response",
                "metrics": {},
                "ground_truth": {},
                "prediction": {},
            },
        ]

    def test_generate_index_html(self, generator, sample_runs):
        """Generates index.html."""
        html = generator.generate_index_html(sample_runs)

        assert "<!DOCTYPE html>" in html
        assert "VLM Geometry Bench" in html
        assert "20260202_130859_test" in html
        assert "llava:7b" in html
        assert "Test comment" in html
        assert "backend-filter" in html
        assert "task-filter" in html

    def test_generate_index_html_empty(self, generator):
        """Generates index.html with no runs."""
        html = generator.generate_index_html([])

        assert "<!DOCTYPE html>" in html
        assert "0 total" in html

    def test_generate_index_html_pydantic(self, generator):
        """Handles Pydantic models."""
        entry = RunIndexEntry(
            run_id="test",
            run_name="test",
            models=["llava:7b"],
            backends=["ollama"],
            timestamp=datetime(2026, 2, 2, 13, 8, 59),
            elapsed_seconds=100.5,
            total_tests=10,
            tasks=["COUNT"],
            image_classes=["USSS"],
        )
        html = generator.generate_index_html([entry])

        assert "test" in html

    def test_generate_summary_html(self, generator, sample_run_metadata, sample_tests):
        """Generates summary.html."""
        model_metrics = {
            "llava:7b": {
                "LOCATE": {"mean_detection_rate": 45.0, "mean_false_positive_rate": 30.0}
            },
            "gpt-4o": {
                "LOCATE": {"mean_detection_rate": 60.0, "mean_false_positive_rate": 20.0}
            },
        }

        html = generator.generate_summary_html(
            run=sample_run_metadata,
            tests=sample_tests,
            model_metrics=model_metrics,
        )

        assert "<!DOCTYPE html>" in html
        assert "20260202_130859_test" in html
        assert "Test run for LOCATE task" in html
        assert "llava:7b" in html
        assert "gpt-4o" in html
        assert "USSS_s2_d20_wb" in html
        assert "model-filter" in html

    def test_generate_summary_html_no_comment(self, generator, sample_tests):
        """Handles run without comment."""
        run = {
            "run_id": "test",
            "run_name": "test",
            "comment": None,
            "timestamp_start": "2026-02-02T12:59:38",
            "tasks": ["COUNT"],
            "image_classes": ["USSS"],
            "num_samples": 10,
            "models": [],
        }

        html = generator.generate_summary_html(
            run=run,
            tests=sample_tests,
            model_metrics={},
        )

        assert "<!DOCTYPE html>" in html
        assert "comment-box" not in html  # No comment box

    def test_generate_test_html(self, generator, sample_tests):
        """Generates test.html."""
        test = sample_tests[0]
        conversation = {
            "test_id": test["test_id"],
            "model": "llava:7b",
            "turns": [
                {
                    "turn": 1,
                    "role": "user",
                    "content": "Find the spots.",
                    "image_attached": True,
                },
                {
                    "turn": 1,
                    "role": "assistant",
                    "content": "I found spots at 0.5, 0.5",
                    "latency_ms": 2500,
                    "output_tokens": 50,
                },
            ],
            "final_response": "I found spots at 0.5, 0.5",
        }

        html = generator.generate_test_html(
            test=test,
            conversation=conversation,
            other_models=None,
        )

        assert "<!DOCTYPE html>" in html
        assert "USSS_s2_d20_wb" in html
        assert "LOCATE" in html
        assert "llava:7b" in html
        assert "Find the spots" in html
        assert "2500ms" in html

    def test_generate_test_html_with_other_models(self, generator, sample_tests):
        """Generates test.html with model comparison."""
        test = sample_tests[0]
        other_models = [
            sample_tests[0],  # Current model
            {
                "test_id": "USSS_s2_d20_wb_LOCATE",
                "sample_id": "USSS_s2_d20_wb",
                "model": "gpt-4o",
                "task": "LOCATE",
                "image_class": "USSS",
                "total_latency_ms": 1500,
                "success": True,
                "estimated_cost_usd": 0.01,
                "metrics": {"detection_rate": 60.0, "false_positive_rate": 20.0},
            },
        ]

        html = generator.generate_test_html(
            test=test,
            conversation=None,
            other_models=other_models,
        )

        assert "Compare Models" in html
        assert "gpt-4o" in html

    def test_generate_test_html_failed(self, generator, sample_tests):
        """Generates test.html for failed test."""
        test = sample_tests[1]  # Failed test

        html = generator.generate_test_html(
            test=test,
            conversation=None,
            other_models=None,
        )

        assert "FAIL" in html

    def test_generate_test_html_locate_positions(self, generator, sample_tests):
        """Shows position details for LOCATE task."""
        test = sample_tests[0]
        test["ground_truth"]["positions"] = [[0.1, 0.2], [0.3, 0.4]]
        test["prediction"]["positions"] = [[0.1, 0.2]]

        html = generator.generate_test_html(
            test=test,
            conversation=None,
            other_models=None,
        )

        assert "Position Details" in html
        assert "0.100" in html  # Formatted position

    def test_get_css(self, generator):
        """Returns CSS content."""
        css = generator.get_css()

        assert ":root" in css
        assert "--bg-primary" in css
        assert ".card" in css
        assert "table" in css

    def test_get_js(self, generator):
        """Returns JavaScript content."""
        js = generator.get_js()

        assert "applyFilters" in js
        assert "selectModel" in js
        assert "addEventListener" in js

    def test_save_assets(self, generator, tmp_path):
        """Saves CSS and JS to directory."""
        assets_dir = tmp_path / "assets"
        generator.save_assets(assets_dir)

        assert assets_dir.exists()
        assert (assets_dir / "style.css").exists()
        assert (assets_dir / "script.js").exists()

        css_content = (assets_dir / "style.css").read_text()
        assert ":root" in css_content

        js_content = (assets_dir / "script.js").read_text()
        assert "applyFilters" in js_content


class TestHTMLGeneratorMetricDisplay:
    """Tests for metric display formatting."""

    @pytest.fixture
    def generator(self):
        return HTMLGenerator()

    def test_count_task_metrics(self, generator):
        """COUNT task shows correct metrics."""
        test = {
            "test_id": "test",
            "sample_id": "test",
            "model": "llava:7b",
            "task": "COUNT",
            "image_class": "USSS",
            "total_latency_ms": 1000,
            "success": True,
            "metrics": {"exact_match": True, "absolute_error": 0},
            "ground_truth": {},
            "prediction": {},
        }

        html = generator.generate_test_html(test)

        assert "COUNT" in html

    def test_pattern_task_metrics(self, generator):
        """PATTERN task shows correct metrics."""
        test = {
            "test_id": "test",
            "sample_id": "test",
            "model": "llava:7b",
            "task": "PATTERN",
            "image_class": "HSFR",
            "total_latency_ms": 1000,
            "success": True,
            "metrics": {"correct": True, "predicted": "HEXAGONAL"},
            "ground_truth": {},
            "prediction": {},
        }

        html = generator.generate_test_html(test)

        assert "PATTERN" in html
