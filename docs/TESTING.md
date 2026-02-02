# VLM Geometry Bench - Test Documentation

This document describes the test suite for VLM Geometry Bench, covering test organization, coverage, and how to run tests.

## Overview

The test suite contains **164 tests** across 6 test modules, providing approximately **70% code coverage** of the core components.

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=vlm_geometry_bench --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_response_parser.py

# Run specific test class
uv run pytest tests/test_metrics.py::TestCountMetrics

# Run specific test
uv run pytest tests/test_config.py::TestEvaluationConfig::test_default_config
```

## Test Modules

### test_config.py (9 tests)

Tests for the `EvaluationConfig` dataclass that manages evaluation settings.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestEvaluationConfig` | 9 | Configuration validation and defaults |

**Key test cases:**
- `test_default_config` - Verifies default values (ollama backend, llava:7b model)
- `test_openrouter_auto_url` - OpenRouter backend auto-configures correct URL
- `test_invalid_backend_raises` - Invalid backend raises ValueError
- `test_invalid_num_shots_raises` - Only 0, 3, 5 shots allowed
- `test_invalid_image_class_raises` - Rejects invalid image classes
- `test_invalid_task_raises` - Rejects invalid task names
- `test_api_endpoint_ollama` - Correct endpoint for Ollama
- `test_api_endpoint_openrouter` - Correct endpoint for OpenRouter
- `test_url_prefix_auto_added` - Auto-adds http:// to URLs

### test_response_parser.py (48 tests)

Tests for parsing VLM text responses into structured data.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestParseCount` | 10 | COUNT response parsing |
| `TestParseLocate` | 10 | LOCATE response parsing (normalized coords) |
| `TestParsePattern` | 10 | PATTERN response parsing |
| `TestParseSize` | 7 | SIZE response parsing |
| `TestParseDefect` | 5 | DEFECT response parsing |
| `TestParseDispatch` | 7 | Generic parse() dispatch |

**COUNT parsing tests:**
- Just number: `"42"` → 42
- With unit: `"42 spots"` → 42
- Natural language: `"I can see 15 spots"` → 15
- Zero detection: `"no spots"` → 0
- Empty/invalid responses fail gracefully

**LOCATE parsing tests (normalized coordinates):**
- Parentheses format: `"(0.18, 0.67), (0.40, 0.29)"`
- Bracket format: `"[[0.18, 0.67], [0.40, 0.29]]"`
- One per line format
- Floating point coordinates
- Out-of-bounds rejection (values > 1.1)
- Clamping slightly-over-1.0 values
- Edge values (0.0, 1.0)

**PATTERN parsing tests:**
- Exact matches: RANDOM, HEXAGONAL, GRID, EMPTY, SINGLE
- Case insensitive parsing
- Synonym recognition: honeycomb→HEXAGONAL, scattered→RANDOM
- Unknown patterns fail gracefully

**SIZE parsing tests:**
- Just number: `"3.5"` → 3.5 um
- With units: `"3.5 um"`, `"3.5 micrometers"`
- Labeled format: `"diameter: 4.2"`
- Rejects unreasonably large values

**DEFECT parsing tests:**
- Structured format with DEFECTS_FOUND, MISSING_COUNT, etc.
- Natural language detection
- Binary yes/no detection

### test_metrics.py (40 tests)

Tests for metrics computation and aggregation.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestCountMetrics` | 7 | COUNT metrics computation |
| `TestPatternMetrics` | 4 | PATTERN metrics computation |
| `TestLocateMetrics` | 8 | LOCATE metrics computation |
| `TestSizeMetrics` | 6 | SIZE metrics computation |
| `TestDefectMetrics` | 8 | DEFECT metrics computation |
| `TestAggregateMetrics` | 7 | Metrics aggregation |

**COUNT metrics tests:**
- Exact match detection
- Within tolerance (configurable, default ±2)
- Absolute and percentage error calculation
- None prediction handling
- Zero ground truth edge case

**PATTERN metrics tests:**
- Correct/incorrect classification
- None prediction handling
- Dictionary serialization

**LOCATE metrics tests:**
- Perfect detection (100% rate)
- Partial detection with count difference
- False positive rate calculation
- Empty ground truth / empty predictions
- Mean nearest distance computation

**SIZE metrics tests:**
- Exact match
- Within tolerance (default 20%)
- Outside tolerance
- None prediction handling

**DEFECT metrics tests:**
- True positive / true negative detection
- False positive / false negative detection
- Missing count error
- Noise count error

**Aggregation tests:**
- Aggregate COUNT metrics (exact match rate, mean error)
- Aggregate PATTERN metrics (accuracy, per-pattern breakdown)
- Aggregate LOCATE metrics (mean detection rate)
- Aggregate SIZE metrics (within tolerance rate)
- Aggregate DEFECT metrics (detection accuracy)
- Aggregate sample results by task and class

### test_data_loader.py (17 tests)

Tests for loading test suite data from imagegen output.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestTestSuiteLoader` | 11 | Test suite loading |
| `TestGroundTruth` | 3 | Ground truth properties |
| `TestSpotPosition` | 1 | Spot position dataclass |
| `TestBenchmarkSample` | 1 | Benchmark sample dataclass |

**Fixture:** `sample_testsuite` creates a minimal test suite in a temp directory with:
- manifest.yaml with 3 images
- Dummy PNG images
- CSV file with spot positions

**TestSuiteLoader tests:**
- `test_load_manifest` - Load and parse manifest.yaml
- `test_manifest_cached` - Manifest is cached after first load
- `test_missing_manifest_raises` - FileNotFoundError for missing manifest
- `test_load_all_samples` - Load all samples from test suite
- `test_load_samples_filter_by_class` - Filter by single class
- `test_load_samples_multiple_classes` - Filter by multiple classes
- `test_load_samples_with_limit` - Limit number of samples
- `test_sample_has_image` - Samples include PIL Image
- `test_sample_has_ground_truth` - Samples include GroundTruth
- `test_sample_positions_from_csv` - Spot positions loaded from CSV
- `test_get_class_summary` - Get image count per class

**GroundTruth tests:**
- `test_has_defects` - Detect defect classes (HSDN)
- `test_is_hexagonal` - Detect hexagonal patterns
- `test_is_random` - Detect random patterns

### test_vision_client.py (29 tests)

Tests for VLM API communication.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestVisionResponse` | 5 | Response dataclass |
| `TestVisionClientImageEncoding` | 6 | Image encoding |
| `TestVisionClientMimeTypes` | 6 | MIME type detection |
| `TestVisionClientHeaders` | 3 | Request headers |
| `TestVisionClientMessages` | 2 | Message building |
| `TestVisionClientAsyncContext` | 3 | Async context manager |
| `TestVisionClientSendRequest` | 4 | API requests (mocked) |

**VisionResponse tests:**
- Success when no error
- Not success when error present
- Token extraction from usage dict
- Zero tokens when no usage

**Image encoding tests:**
- Encode PIL Image to base64
- RGBA to RGB conversion
- Encode from file path (string)
- Encode from Path object
- FileNotFoundError for missing files
- TypeError for unsupported types

**MIME type tests:**
- PNG → image/png
- JPG/JPEG → image/jpeg
- GIF → image/gif
- WebP → image/webp
- Unknown defaults to image/jpeg

**Headers tests:**
- Ollama headers (no auth)
- OpenRouter headers (auth + referer)
- API key in Authorization header

**Message building tests:**
- Simple message (image + prompt)
- Few-shot examples (multiple turns)

**Async context tests:**
- Context manager creates session
- Context manager closes session
- Safe to close without session

**API request tests (mocked):**
- Successful request returns content
- API error returns error response
- Timeout returns error response
- KeyboardInterrupt propagates

### test_cli.py (15 tests)

Tests for CLI argument parsing and validation.

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestCLIOptions` | 11 | Option parsing |
| `TestCLIOutput` | 4 | Help output |

**Option parsing tests:**
- `test_help_option` - --help works
- `test_default_backend_is_ollama` - Default backend
- `test_backend_choice_validation` - Invalid backend rejected
- `test_shots_choice_validation` - Only 0/3/5 allowed
- `test_openrouter_requires_api_key` - API key validation
- `test_invalid_class_rejected` - Invalid class rejected
- `test_invalid_task_rejected` - Invalid task rejected
- `test_valid_classes_parsed` - Valid classes accepted
- `test_valid_tasks_parsed` - Valid tasks accepted
- `test_samples_is_optional` - --samples not required
- `test_verbose_flag` - --verbose/-v accepted

**Help output tests:**
- Help shows task descriptions (COUNT, PATTERN)
- Help shows class descriptions (CTRL, USSS, HSFR)
- Help shows usage examples

## Coverage Summary

| Module | Coverage |
|--------|----------|
| config.py | 100% |
| response_parser.py | 96% |
| metrics.py | 98% |
| data_loader.py | 94% |
| vision_client.py | 94% |
| prompts.py | 58% |
| evaluator.py | 32% |
| reporter.py | 45% |
| __main__.py | 67% |

**Note:** Lower coverage on evaluator, prompts, and reporter is expected as these modules primarily perform integration/orchestration that would require end-to-end testing with actual VLM responses.

## Test Fixtures

### Shared Fixtures

**`sample_testsuite` (test_data_loader.py)**
Creates a temporary test suite directory with:
- manifest.yaml (3 images: CTRL, USSS, HSFR)
- Dummy 700x510 PNG images
- CSV file with spot positions for USSS

**`parser` (test_response_parser.py)**
Fresh ResponseParser instance for each test.

**`calc` (test_metrics.py)**
MetricsCalculator with configurable tolerances.

**`client` (test_vision_client.py)**
VisionClient with default EvaluationConfig.

**`runner` (test_cli.py)**
Click CliRunner for CLI testing.

## Adding New Tests

### For new parsing logic

Add tests to `test_response_parser.py`:

```python
class TestParseNewTask:
    @pytest.fixture
    def parser(self):
        return ResponseParser()

    def test_parse_basic_format(self, parser):
        result = parser.parse_new_task("expected input")
        assert result.field == expected_value
        assert result.success
```

### For new metrics

Add tests to `test_metrics.py`:

```python
class TestNewMetrics:
    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_compute_new_metrics(self, calc):
        metrics = calc.compute_new_metrics(predicted=X, ground_truth=Y)
        assert metrics.field == expected
```

### For API changes

Add tests to `test_vision_client.py` with mocked responses:

```python
@pytest.mark.asyncio
async def test_new_api_behavior(self):
    config = EvaluationConfig()
    client = VisionClient(config)

    with patch("aiohttp.ClientSession.post") as mock_post:
        # Set up mock response
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value.status = 200
        mock_context.__aenter__.return_value.json = AsyncMock(return_value={...})
        mock_post.return_value = mock_context

        async with client:
            response = await client.send_request(...)

    assert response.expected_field == expected_value
```

## Dependencies

Test dependencies (in pyproject.toml):
- pytest
- pytest-asyncio
- pytest-cov
- PyYAML (for fixture data)
- Pillow (for image fixtures)
