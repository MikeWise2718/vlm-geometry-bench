"""Tests for traceability schemas (Pydantic models)."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from vlm_geometry_bench.traceability.schemas import (
    RunIndex,
    RunIndexEntry,
    RunMetadata,
    ModelRunInfo,
    ModelMetadata,
    SampleTestResult,
    ConversationTurn,
    ConversationHistory,
)


class TestRunIndexEntry:
    """Tests for RunIndexEntry schema."""

    def test_create_minimal(self):
        """Create entry with required fields only."""
        entry = RunIndexEntry(
            run_id="20260202_130859_test",
            run_name="test",
            models=["llava:7b"],
            backends=["ollama"],
            timestamp=datetime(2026, 2, 2, 13, 8, 59),
            elapsed_seconds=100.5,
            total_tests=10,
            tasks=["COUNT"],
            image_classes=["USSS"],
        )
        assert entry.run_id == "20260202_130859_test"
        assert entry.comment is None
        assert entry.estimated_cost_usd is None
        assert entry.size_mb is None

    def test_create_full(self):
        """Create entry with all fields."""
        entry = RunIndexEntry(
            run_id="20260202_130859_comparison",
            run_name="comparison",
            comment="Testing LOCATE task",
            models=["claude-sonnet-4", "llava:7b"],
            backends=["anthropic", "ollama"],
            timestamp=datetime(2026, 2, 2, 13, 8, 59),
            elapsed_seconds=1680.5,
            total_tests=184,
            estimated_cost_usd=2.15,
            size_mb=135.6,
            tasks=["LOCATE", "COUNT"],
            image_classes=["CTRL", "USSS", "HSDN"],
        )
        assert len(entry.models) == 2
        assert entry.estimated_cost_usd == 2.15

    def test_model_dump(self):
        """Entry can be dumped to dict."""
        entry = RunIndexEntry(
            run_id="20260202_130859_test",
            run_name="test",
            models=["llava:7b"],
            backends=["ollama"],
            timestamp=datetime(2026, 2, 2, 13, 8, 59),
            elapsed_seconds=100.5,
            total_tests=10,
            tasks=["COUNT"],
            image_classes=["USSS"],
        )
        d = entry.model_dump()
        assert d["run_id"] == "20260202_130859_test"
        assert "timestamp" in d

    def test_missing_required_field(self):
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            RunIndexEntry(
                run_id="test",
                # missing run_name
                models=["llava:7b"],
                backends=["ollama"],
                timestamp=datetime.now(),
                elapsed_seconds=100,
                total_tests=10,
                tasks=["COUNT"],
                image_classes=["USSS"],
            )


class TestRunIndex:
    """Tests for RunIndex schema."""

    def test_create_empty(self):
        """Create empty index."""
        index = RunIndex()
        assert index.version == "1.0"
        assert index.runs == []

    def test_create_with_runs(self):
        """Create index with runs."""
        entry = RunIndexEntry(
            run_id="20260202_130859_test",
            run_name="test",
            models=["llava:7b"],
            backends=["ollama"],
            timestamp=datetime(2026, 2, 2, 13, 8, 59),
            elapsed_seconds=100.5,
            total_tests=10,
            tasks=["COUNT"],
            image_classes=["USSS"],
        )
        index = RunIndex(runs=[entry])
        assert len(index.runs) == 1
        assert index.runs[0].run_id == "20260202_130859_test"

    def test_model_dump_json(self):
        """Index can be dumped to JSON-serializable dict."""
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
        index = RunIndex(runs=[entry])
        d = index.model_dump(mode="json")
        assert isinstance(d["runs"][0]["timestamp"], str)


class TestModelRunInfo:
    """Tests for ModelRunInfo schema."""

    def test_create_minimal(self):
        """Create with required fields."""
        info = ModelRunInfo(
            model="llava:7b",
            backend="ollama",
            elapsed_seconds=300.0,
            total_tests=92,
            success_rate=95.5,
        )
        assert info.model == "llava:7b"
        assert info.input_tokens == 0
        assert info.output_tokens == 0
        assert info.estimated_cost_usd is None
        assert info.base_url is None

    def test_create_full(self):
        """Create with all fields."""
        info = ModelRunInfo(
            model="claude-sonnet-4-20250514",
            backend="anthropic",
            base_url="https://api.anthropic.com",
            elapsed_seconds=560.7,
            total_tests=92,
            success_rate=98.9,
            input_tokens=58052,
            output_tokens=34740,
            estimated_cost_usd=0.70,
        )
        assert info.input_tokens == 58052
        assert info.base_url == "https://api.anthropic.com"


class TestRunMetadata:
    """Tests for RunMetadata schema."""

    def test_create_minimal(self):
        """Create with required fields."""
        metadata = RunMetadata(
            run_id="20260202_130859_test",
            run_name="test",
            timestamp_start=datetime(2026, 2, 2, 12, 59, 38),
            tasks=["LOCATE"],
            image_classes=["USSS"],
            num_samples=92,
            models=[],
        )
        assert metadata.comment is None
        assert metadata.timestamp_end is None

    def test_create_with_models(self):
        """Create with model info."""
        model_info = ModelRunInfo(
            model="llava:7b",
            backend="ollama",
            elapsed_seconds=300.0,
            total_tests=92,
            success_rate=95.5,
        )
        metadata = RunMetadata(
            run_id="20260202_130859_test",
            run_name="test",
            timestamp_start=datetime(2026, 2, 2, 12, 59, 38),
            timestamp_end=datetime(2026, 2, 2, 13, 27, 38),
            tasks=["LOCATE"],
            image_classes=["USSS"],
            num_samples=92,
            models=[model_info],
        )
        assert len(metadata.models) == 1
        assert metadata.models[0].model == "llava:7b"


class TestModelMetadata:
    """Tests for ModelMetadata schema."""

    def test_create(self):
        """Create model metadata."""
        metadata = ModelMetadata(
            model="llava:7b",
            backend="ollama",
            run_id="20260202_130859_test",
            elapsed_seconds=300.0,
            total_tests=92,
            successful_tests=87,
            success_rate=94.6,
        )
        assert metadata.results_by_task == {}
        assert metadata.results_by_class == {}

    def test_with_aggregated_results(self):
        """Create with aggregated results."""
        metadata = ModelMetadata(
            model="llava:7b",
            backend="ollama",
            base_url="http://localhost:11434",
            run_id="20260202_130859_test",
            elapsed_seconds=300.0,
            total_tests=92,
            successful_tests=87,
            success_rate=94.6,
            results_by_task={
                "LOCATE": {"mean_detection_rate": 45.2, "mean_false_positive_rate": 30.1}
            },
            results_by_class={
                "USSS": {"LOCATE": {"mean_detection_rate": 42.0}}
            },
        )
        assert metadata.results_by_task["LOCATE"]["mean_detection_rate"] == 45.2


class TestSampleTestResult:
    """Tests for SampleTestResult schema."""

    def test_create_minimal(self):
        """Create with required fields."""
        metadata = SampleTestResult(
            test_id="USSS_s2_d20_wb_LOCATE",
            sample_id="USSS_s2_d20_wb",
            model="llava:7b",
            task="LOCATE",
            image_class="USSS",
            timestamp=datetime(2026, 2, 2, 13, 1, 15),
            success=True,
        )
        assert metadata.total_latency_ms == 0
        assert metadata.num_turns == 1
        assert metadata.parse_error is None
        assert metadata.metrics == {}

    def test_create_full(self):
        """Create with all fields."""
        metadata = SampleTestResult(
            test_id="USSS_s2_d20_wb_LOCATE",
            sample_id="USSS_s2_d20_wb",
            model="claude-sonnet-4-20250514",
            task="LOCATE",
            image_class="USSS",
            timestamp=datetime(2026, 2, 2, 13, 1, 15),
            total_latency_ms=5890,
            total_input_tokens=1262,
            total_output_tokens=394,
            estimated_cost_usd=0.0097,
            num_turns=2,
            success=True,
            metrics={"detection_rate": 15.0, "false_positive_rate": 83.3},
            ground_truth={"spot_count": 20, "positions": [[0.15, 0.20]]},
            prediction={"positions": [[0.15, 0.08]]},
        )
        assert metadata.total_latency_ms == 5890
        assert metadata.metrics["detection_rate"] == 15.0

    def test_failed_test(self):
        """Create failed test metadata."""
        metadata = SampleTestResult(
            test_id="USSS_s2_d20_wb_LOCATE",
            sample_id="USSS_s2_d20_wb",
            model="llava:7b",
            task="LOCATE",
            image_class="USSS",
            timestamp=datetime(2026, 2, 2, 13, 1, 15),
            success=False,
            parse_error="Failed to parse coordinates from response",
        )
        assert metadata.success is False
        assert "Failed to parse" in metadata.parse_error


class TestConversationTurn:
    """Tests for ConversationTurn schema."""

    def test_user_turn(self):
        """Create user turn."""
        turn = ConversationTurn(
            turn=1,
            role="user",
            content="Examine this image and identify the location of each spot.",
            image_attached=True,
            timestamp=datetime(2026, 2, 2, 13, 1, 15),
        )
        assert turn.role == "user"
        assert turn.image_attached is True
        assert turn.latency_ms is None

    def test_assistant_turn(self):
        """Create assistant turn."""
        turn = ConversationTurn(
            turn=1,
            role="assistant",
            content="I can see several spots in the image...",
            latency_ms=2395,
            input_tokens=631,
            output_tokens=212,
        )
        assert turn.role == "assistant"
        assert turn.latency_ms == 2395
        assert turn.image_attached is False


class TestConversationHistory:
    """Tests for ConversationHistory schema."""

    def test_create_empty(self):
        """Create empty conversation."""
        history = ConversationHistory(
            test_id="test_1",
            model="llava:7b",
        )
        assert history.turns == []
        assert history.final_response is None

    def test_create_with_turns(self):
        """Create conversation with turns."""
        user_turn = ConversationTurn(
            turn=1,
            role="user",
            content="Count the spots.",
            image_attached=True,
        )
        assistant_turn = ConversationTurn(
            turn=1,
            role="assistant",
            content="I count 15 spots.",
            latency_ms=1500,
            output_tokens=50,
        )
        history = ConversationHistory(
            test_id="test_1",
            model="llava:7b",
            turns=[user_turn, assistant_turn],
            final_response="I count 15 spots.",
        )
        assert len(history.turns) == 2
        assert history.final_response == "I count 15 spots."
