"""Tests for CLI argument parsing and validation."""

import pytest
from click.testing import CliRunner

from vlm_geometry_bench.__main__ import main


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


class TestCLIOptions:
    """Tests for CLI option parsing."""

    def test_help_option(self, runner):
        """--help shows help text."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "VLM Geometry Bench" in result.output
        assert "--backend" in result.output

    def test_default_backend_is_ollama(self, runner):
        """Default backend is ollama."""
        # Will fail due to missing testsuite, but parses correctly
        result = runner.invoke(main, ["--testsuite", "/nonexistent"])
        # Check it didn't fail on backend parsing
        assert "Invalid backend" not in result.output

    def test_backend_choice_validation(self, runner):
        """Backend must be ollama or openrouter."""
        result = runner.invoke(main, ["--backend", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_shots_choice_validation(self, runner):
        """Shots must be 0, 3, or 5."""
        result = runner.invoke(main, ["--shots", "2"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_openrouter_requires_api_key(self, runner):
        """OpenRouter backend requires API key."""
        result = runner.invoke(main, ["--backend", "openrouter", "--testsuite", "."])
        assert result.exit_code != 0
        assert "API key required" in result.output

    def test_invalid_class_rejected(self, runner):
        """Invalid image class is rejected."""
        result = runner.invoke(main, ["--classes", "INVALID", "--testsuite", "."])
        assert result.exit_code != 0
        assert "Invalid class" in result.output

    def test_invalid_task_rejected(self, runner):
        """Invalid task is rejected."""
        result = runner.invoke(main, ["--tasks", "INVALID", "--testsuite", "."])
        assert result.exit_code != 0
        assert "Invalid task" in result.output

    def test_valid_classes_parsed(self, runner):
        """Valid classes are parsed correctly."""
        # Will fail on testsuite, but should parse classes OK
        result = runner.invoke(main, [
            "--classes", "CTRL,USSS,HSFR",
            "--testsuite", "/nonexistent"
        ])
        # Should not fail on class validation
        assert "Invalid class" not in result.output

    def test_valid_tasks_parsed(self, runner):
        """Valid tasks are parsed correctly."""
        result = runner.invoke(main, [
            "--tasks", "COUNT,PATTERN,LOCATE",
            "--testsuite", "/nonexistent"
        ])
        assert "Invalid task" not in result.output

    def test_samples_is_optional(self, runner):
        """--samples is optional."""
        result = runner.invoke(main, ["--testsuite", "/nonexistent"])
        assert "--samples" not in str(result.exception) if result.exception else True

    def test_verbose_flag(self, runner):
        """--verbose flag is accepted."""
        result = runner.invoke(main, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_short_verbose_flag(self, runner):
        """-v short flag is accepted."""
        result = runner.invoke(main, ["-v", "--help"])
        assert result.exit_code == 0


class TestCLIOutput:
    """Tests for CLI output formatting."""

    def test_help_shows_tasks_description(self, runner):
        """Help shows task descriptions."""
        result = runner.invoke(main, ["--help"])
        assert "COUNT" in result.output
        assert "PATTERN" in result.output
        assert "spots" in result.output.lower()

    def test_help_shows_classes_description(self, runner):
        """Help shows image class descriptions."""
        result = runner.invoke(main, ["--help"])
        assert "CTRL" in result.output
        assert "USSS" in result.output
        assert "HSFR" in result.output

    def test_help_shows_examples(self, runner):
        """Help shows usage examples."""
        result = runner.invoke(main, ["--help"])
        assert "Examples" in result.output
        assert "ollama" in result.output.lower()
