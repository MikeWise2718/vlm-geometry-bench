# VLM Geometry Bench - Project Tasks

This document tracks the implementation tasks for VLM Geometry Bench.

## Task Summary

| ID | Status | Task |
|----|--------|------|
| 1 | Completed | Design benchmark architecture and project structure |
| 2 | Completed | Implement VisionClient for VLM API communication |
| 3 | Completed | Create data loader for imagegen test suite |
| 4 | Completed | Design and implement evaluation prompts for geometry tasks |
| 5 | Completed | Implement response parser for geometry answers |
| 6 | Completed | Implement metrics calculator for geometry evaluation |
| 7 | Completed | Build main evaluator orchestration engine |
| 8 | Completed | Implement results reporting and output generation |
| 9 | Completed | Create CLI entry point with rich-click |
| 10 | Completed | Generate initial test suite using imagegen |
| 11 | Completed | Write unit tests for core components |
| 12 | Completed | Run baseline evaluation with multiple VLMs |
| 13 | Completed | Document project and create usage guide |

## Task Details

### Task 1: Design benchmark architecture and project structure

**Description:** Design the overall architecture for the VLM geometry benchmark, including module structure, data flow, and component responsibilities.

**Deliverables:**
- `docs/ARCHITECTURE.md` - Comprehensive architecture documentation with diagrams
- Project structure with src/vlm_geometry_bench/ package layout
- Module responsibilities defined

### Task 2: Implement VisionClient for VLM API communication

**Description:** Create a client class that handles communication with VLM APIs (Ollama and OpenRouter), including image encoding, request building, and response handling.

**Deliverables:**
- `src/vlm_geometry_bench/vision_client.py`
- Support for Ollama and OpenRouter backends
- Image encoding (PIL, file path, Path object)
- Async context manager for session management
- Retry logic with configurable attempts
- Few-shot example support

### Task 3: Create data loader for imagegen test suite

**Description:** Implement a loader that reads the imagegen test suite, parses the manifest, and provides benchmark samples with ground truth.

**Deliverables:**
- `src/vlm_geometry_bench/data_loader.py`
- TestSuiteLoader class with manifest parsing
- BenchmarkSample, GroundTruth, SpotPosition dataclasses
- Filtering by image class and sample limits
- CSV loading for spot positions

### Task 4: Design and implement evaluation prompts for geometry tasks

**Description:** Create task-specific prompts that instruct VLMs to perform COUNT, LOCATE, PATTERN, SIZE, and DEFECT tasks.

**Deliverables:**
- `src/vlm_geometry_bench/prompts.py`
- PromptBuilder class with task-specific methods
- Prompts optimized for structured responses
- Scale information included where relevant

### Task 5: Implement response parser for geometry answers

**Description:** Build a parser that extracts structured data from VLM text responses for each task type.

**Deliverables:**
- `src/vlm_geometry_bench/response_parser.py`
- ResponseParser class with task-specific parsing
- ParsedCountResponse, ParsedLocateResponse, etc. dataclasses
- Natural language understanding (synonyms, various formats)
- Graceful failure with error messages

### Task 6: Implement metrics calculator for geometry evaluation

**Description:** Create a metrics system that computes accuracy metrics for each task type and aggregates results.

**Deliverables:**
- `src/vlm_geometry_bench/metrics.py`
- MetricsCalculator class
- Task-specific metrics (CountMetrics, PatternMetrics, etc.)
- Aggregation by task, class, and overall
- Configurable tolerances

### Task 7: Build main evaluator orchestration engine

**Description:** Implement the main evaluation loop that coordinates data loading, VLM queries, parsing, and metrics computation.

**Deliverables:**
- `src/vlm_geometry_bench/evaluator.py`
- Evaluator class with async run() method
- Progress tracking with tqdm
- Error handling and recovery
- Usage statistics tracking

### Task 8: Implement results reporting and output generation

**Description:** Create reporters that output evaluation results in various formats (console, JSON, CSV).

**Deliverables:**
- `src/vlm_geometry_bench/reporter.py`
- Reporter class with multiple output methods
- Console output with rich tables
- JSON metrics file
- CSV leaderboard row
- Raw responses file

### Task 9: Create CLI entry point with rich-click

**Description:** Build a command-line interface with argument parsing, validation, and help text.

**Deliverables:**
- `src/vlm_geometry_bench/__main__.py`
- rich-click based CLI
- Option validation (backend, classes, tasks, shots)
- Help text with examples
- Verbose mode

### Task 10: Generate initial test suite using imagegen

**Description:** Generate the test suite of 92 images across 6 classes using imagegen.

**Deliverables:**
- Test suite at `d:/python/imagegen/testsuite/`
- 92 images: CTRL(2), USSS(36), USDS(9), HSFR(3), HSRP(24), HSDN(18)
- manifest.yaml with ground truth
- CSV files with spot positions

### Task 11: Write unit tests for core components

**Description:** Create comprehensive unit tests for the core modules.

**Deliverables:**
- `tests/test_config.py` - 9 tests for configuration
- `tests/test_response_parser.py` - 42 tests for parsing
- `tests/test_metrics.py` - 40 tests for metrics
- `tests/test_data_loader.py` - 17 tests for data loading
- `tests/test_vision_client.py` - 29 tests for VLM client
- `tests/test_cli.py` - 15 tests for CLI

**Coverage:** 70% overall, 94-100% for core modules

### Task 12: Run baseline evaluation with multiple VLMs

**Description:** Evaluate multiple vision-language models on the test suite to establish baseline performance.

**Deliverables:**
- Evaluations with 4 Ollama models:
  - llama3.2-vision:latest (7.8 GB)
  - minicpm-v:latest (5.5 GB)
  - granite3.2-vision:2b (2.4 GB)
  - qwen3-vl:8b (6.1 GB)
- Results in `results/` directory
- Baseline metrics documented in README.md

**Key Findings:**
- COUNT exact match: 4-7% across models
- PATTERN accuracy: 39-66% across models
- RANDOM patterns easier than HEXAGONAL
- llama3.2-vision best overall performer

### Task 13: Document project and create usage guide

**Description:** Create comprehensive documentation for the project.

**Deliverables:**
- `README.md` - Main project documentation
- `docs/USAGE.md` - Detailed usage guide
- `docs/TESTING.md` - Test documentation
- `docs/PROJECT_TASKS.md` - This task tracking document
- `CLAUDE.md` - Claude Code guidance

## Dependencies

Task dependency graph:

```
1 (Architecture)
├── 2 (VisionClient) ─────────────────┐
├── 3 (DataLoader) ───────────────────┤
├── 4 (Prompts) ──────────────────────┤
├── 5 (ResponseParser) ───────────────┼── 7 (Evaluator) ── 8 (Reporter) ── 9 (CLI)
└── 6 (Metrics) ──────────────────────┘                                      │
                                                                              │
10 (TestSuite) ───────────────────────────────────────────────────────────────┤
                                                                              │
11 (Tests) ← depends on 2, 3, 5, 6                                            │
                                                                              │
12 (Baseline Eval) ← depends on 9, 10 ────────────────────────────────────────┤
                                                                              │
13 (Documentation) ← depends on 9, 12 ────────────────────────────────────────┘
```

## Timeline

All 13 tasks completed in January 2026.
