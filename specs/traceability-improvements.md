# VLM Geometry Bench - Traceability Improvements Plan

## Overview

Add comprehensive traceability to VLM Geometry Bench with:
1. **Multi-model comparison runs** - Evaluate multiple models in a single test-run
2. Structured artifact storage per test run
3. Annotated result images showing model predictions vs ground truth
4. Static HTML web interface for browsing and drilling down into results

---

## 1. Artifact Storage Structure

**Key change: Test runs can include multiple models, organized per-model within the run.**

```
results/
+-- index.json                              # Master index of all test runs
+-- 20260202_130859_comparison/             # Test run folder (timestamp_runtype)
|   +-- run_metadata.json                   # Complete run metadata (all models)
|   +-- summary.html                        # Static HTML summary with model comparison
|   +-- models/                             # Per-model results
|   |   +-- claude-sonnet-4/
|   |   |   +-- model_metadata.json         # Model-specific aggregated results
|   |   |   +-- tests/
|   |   |       +-- CTRL_empty_wb_LOCATE/
|   |   |       |   +-- test_metadata.json
|   |   |       |   +-- original.png        # (symlink or copy)
|   |   |       |   +-- annotated.png
|   |   |       |   +-- conversation.json
|   |   |       |   +-- test.html
|   |   |       +-- USSS_s2_d20_wb_COUNT/
|   |   |       +-- ...
|   |   +-- llava-7b/
|   |   |   +-- model_metadata.json
|   |   |   +-- tests/
|   |   |       +-- ...
|   +-- images/                             # Shared original images (deduplicated)
|   |   +-- CTRL_empty_wb.png
|   |   +-- USSS_s2_d20_wb.png
|   |   +-- ...
|   +-- assets/
|       +-- style.css
+-- 20260201_165631_single-model/
    +-- ...
```

**Naming Conventions:**
- Run folder: `YYYYMMDD_HHMMSS_<run-name>` (no spaces/colons)
- Model folder: `<safe-model-name>` (e.g., `claude-sonnet-4-20250514`)
- Test folder: `<sample_id>_<task>`

**Run Comment:**
- User can provide a `--comment` describing the reason/purpose for this test run
- Displayed prominently on index page and run summary page
- Stored in `run_metadata.json`

---

## 2. Data Schemas

### index.json (Master Index)
```json
{
  "version": "1.0",
  "runs": [{
    "run_id": "20260202_130859_comparison",
    "run_name": "comparison",
    "comment": "Testing LOCATE task after prompt improvements",
    "models": ["claude-sonnet-4-20250514", "llava:7b", "gpt-4o"],
    "backends": ["anthropic", "ollama", "openrouter"],
    "timestamp": "2026-02-02T13:08:59",
    "elapsed_seconds": 1680.5,
    "total_tests": 276,
    "estimated_cost_usd": 2.15,
    "size_mb": 135.6,
    "tasks": ["LOCATE"],
    "image_classes": ["CTRL", "USSS", "USDS", "HSFR", "HSRP", "HSDN"]
  }]
}
```

### run_metadata.json (Per Run - Multi-Model)
```json
{
  "run_id": "20260202_130859_comparison",
  "run_name": "comparison",
  "comment": "Testing LOCATE task after prompt improvements",
  "timestamp_start": "2026-02-02T12:59:38",
  "timestamp_end": "2026-02-02T13:27:38",
  "tasks": ["LOCATE"],
  "image_classes": ["CTRL", "USSS", "USDS", "HSFR", "HSRP", "HSDN"],
  "num_samples": 92,
  "models": [
    {
      "model": "claude-sonnet-4-20250514",
      "backend": "anthropic",
      "elapsed_seconds": 560.7,
      "total_tests": 92,
      "success_rate": 98.9,
      "input_tokens": 58052,
      "output_tokens": 34740,
      "estimated_cost_usd": 0.70
    },
    {
      "model": "llava:7b",
      "backend": "ollama",
      "elapsed_seconds": 312.1,
      "total_tests": 92,
      "success_rate": 87.5,
      "input_tokens": 0,
      "output_tokens": 0,
      "estimated_cost_usd": 0.0
    }
  ]
}
```

### test_metadata.json (Per Test)
```json
{
  "test_id": "USSS_s2_d20_wb_LOCATE",
  "sample_id": "USSS_s2_d20_wb",
  "model": "claude-sonnet-4-20250514",
  "task": "LOCATE",
  "image_class": "USSS",
  "timestamp": "2026-02-02T13:01:15",
  "total_latency_ms": 5890,
  "total_input_tokens": 1262,
  "total_output_tokens": 394,
  "estimated_cost_usd": 0.0097,
  "num_turns": 2,
  "success": true,
  "metrics": { "detection_rate": 15.0, "false_positive_rate": 83.3 },
  "ground_truth": { "spot_count": 20, "positions": [[0.15, 0.20], ...] },
  "prediction": { "positions": [[0.15, 0.08], ...] }
}
```

### conversation.json (Per Test - Full History)
```json
{
  "test_id": "USSS_s2_d20_wb_LOCATE",
  "model": "claude-sonnet-4-20250514",
  "turns": [
    {
      "turn": 1,
      "role": "user",
      "content": "Examine this image and identify the location of each circular spot...",
      "image_attached": true,
      "timestamp": "2026-02-02T13:01:15"
    },
    {
      "turn": 1,
      "role": "assistant",
      "content": "Looking at the image, I can see several spots but let me examine more carefully...",
      "latency_ms": 2395,
      "input_tokens": 631,
      "output_tokens": 212
    },
    {
      "turn": 2,
      "role": "user",
      "content": "Please provide the coordinates in the exact format requested: x, y pairs.",
      "image_attached": false,
      "timestamp": "2026-02-02T13:01:18"
    },
    {
      "turn": 2,
      "role": "assistant",
      "content": "0.15, 0.08\n0.19, 0.10\n0.23, 0.12\n...",
      "latency_ms": 3495,
      "input_tokens": 631,
      "output_tokens": 182
    }
  ],
  "final_response": "0.15, 0.08\n0.19, 0.10\n0.23, 0.12\n..."
}
```

---

## 3. Artifacts Summary

### 3.1 Global Level (results/ directory)

| Artifact | Description |
|----------|-------------|
| `index.json` | Master index listing all test runs with summary metadata |
| `index.html` | Main landing page for browsing all test runs |
| `assets/style.css` | Shared CSS stylesheet |
| `assets/script.js` | Shared JavaScript for filtering/sorting |

### 3.2 Test-Run Level (per run folder)

| Artifact | Description |
|----------|-------------|
| `run_metadata.json` | Complete run configuration, all models' aggregated results, timing, costs |
| `summary.html` | Static HTML page with model comparison tables, per-class breakdowns, test list |
| `images/` | Shared original test images (deduplicated across models) |
| `images/<sample_id>.png` | Copy of each original test image used in this run |
| `models/` | Per-model result subfolders |

### 3.3 Model Level (per model within a run)

| Artifact | Description |
|----------|-------------|
| `model_metadata.json` | Model-specific aggregated results: success rate, metrics by task/class, token usage, cost |
| `tests/` | Individual test result subfolders |

### 3.4 Individual Test Level (per sample + task + model)

| Artifact | Description |
|----------|-------------|
| `test_metadata.json` | Test details: metrics, tokens, latency, turns, success/failure, ground truth vs prediction |
| `original.png` | Symlink or copy of original test image (links to `../../images/<sample_id>.png`) |
| `annotated.png` | Result image with GT circles, prediction markers, match lines, and status bar |
| `conversation.json` | Full prompt/response history for all turns |
| `test.html` | Static HTML detail page with images, metrics, conversation history, model comparison |

### 3.5 Artifact Hierarchy Diagram

```
results/                                    [GLOBAL]
+-- index.json
+-- index.html
+-- assets/
|   +-- style.css
|   +-- script.js
|
+-- 20260202_130859_comparison/             [TEST-RUN]
|   +-- run_metadata.json
|   +-- summary.html
|   +-- images/                             [SHARED IMAGES]
|   |   +-- CTRL_empty_wb.png
|   |   +-- USSS_s2_d20_wb.png
|   |   +-- ...
|   +-- models/
|       +-- claude-sonnet-4/                [MODEL]
|       |   +-- model_metadata.json
|       |   +-- tests/
|       |       +-- CTRL_empty_wb_LOCATE/   [INDIVIDUAL TEST]
|       |       |   +-- test_metadata.json
|       |       |   +-- original.png        (symlink)
|       |       |   +-- annotated.png
|       |       |   +-- conversation.json
|       |       |   +-- test.html
|       |       +-- USSS_s2_d20_wb_LOCATE/
|       |           +-- ...
|       +-- llava-7b/
|           +-- model_metadata.json
|           +-- tests/
|               +-- ...
```

---

## 4. Annotated Image Design

```
+------------------------------------------------------------------+
|                                                                  |
|                      MAIN IMAGE AREA                             |
|                                                                  |
|     O = Ground truth (green hollow circle)                       |
|     * = Prediction (red filled circle)                           |
|     --- = Match line (blue, connecting matched pairs)            |
|                                                                  |
|         O           O                                            |
|        /           /                                             |
|       *           *              O                               |
|                                   \                              |
|              O                     *                             |
|               \                                                  |
|                *         O---*                                   |
|                                                                  |
+------------------------------------------------------------------+
| USSS_s2_d20_wb | LOCATE | claude-sonnet-4 | Det:15% | FP:83%    |
+------------------------------------------------------------------+
   ^sample_id      ^task     ^model            ^metrics (color-coded)
```

**Status Bar (40-60px):**
- Sample ID, Task, Model (abbreviated)
- Primary metric with value
- Color-coded pass/fail background

**Annotation Style (confirmed):**
- **Matched pairs**: Blue lines connecting GT position to matched prediction
- **Ground truth**: Green hollow circles (unmatched GT = missed detections)
- **Predictions**: Red filled circles (unmatched = false positives)
- **Theme**: Light theme only (simpler, works well for scientific data)

**Annotation by Task:**
| Task    | Ground Truth          | Prediction              |
|---------|-----------------------|-------------------------|
| LOCATE  | Green hollow circles  | Red filled circles + blue match lines |
| COUNT   | Count in status bar   | Predicted count         |
| PATTERN | Expected pattern      | Predicted pattern       |
| SIZE    | GT diameter           | Predicted diameter      |
| DEFECT  | Missing spot markers  | Detected defect markers |

---

## 5. Web UI Design (Static HTML)

### 4.1 Main Index Page (All Runs)

```
+=========================================================================+
|                    VLM GEOMETRY BENCH - TEST RUNS                       |
+=========================================================================+
|                                                                         |
|  Filters:                                                               |
|  +-----------------+  +-----------------+  +-----------------+          |
|  | Backend: [All v]|  | Model:   [All v]|  | Task:    [All v]|          |
|  +-----------------+  +-----------------+  +-----------------+          |
|                                                                         |
+-------------------------------------------------------------------------+
|  Run ID                    | Models          | Date       | Duration    |
|  Comment                                                                |
|  Tasks      | Tests | Cost    | Size   | Actions                        |
+-------------------------------------------------------------------------+
|  20260202_130859_comparison| 3 models        | 2026-02-02 | 28m 01s     |
|  "Testing LOCATE task after prompt improvements"                        |
|  LOCATE     |   276 | $2.15   | 136 MB | [ View ]                       |
+-------------------------------------------------------------------------+
|  20260201_175357_single    | llava:7b        | 2026-02-01 | 5m 12s      |
|  "Baseline llava performance on COUNT"                                  |
|  COUNT      |    92 | free    | 42 MB  | [ View ]                       |
+-------------------------------------------------------------------------+
|  ...                                                                    |
+=========================================================================+
```

### 4.2 Run Summary Page (Multi-Model Comparison)

```
+=========================================================================+
|  [< Back to All Runs]                                                   |
|                                                                         |
|              TEST RUN: 20260202_130859_comparison                       |
+=========================================================================+
|                                                                         |
|  +---------------------------+    +---------------------------+         |
|  | RUN DETAILS               |    | TOTAL USAGE               |         |
|  +---------------------------+    +---------------------------+         |
|  | Date: 2026-02-02 13:08    |    | Elapsed: 28m 01s          |         |
|  | Tasks: LOCATE             |    | Total Requests: 276       |         |
|  | Classes: 6                |    | Total Cost: $2.15         |         |
|  | Samples/model: 92         |    | Size: 136 MB              |         |
|  +---------------------------+    +---------------------------+         |
|                                                                         |
|  +---------------------------------------------------------------------+|
|  | MODEL COMPARISON                                                    ||
|  +---------------------------------------------------------------------+|
|  | Model             | Backend   | Success | Det.Rate | FP Rate | Cost ||
|  +---------------------------------------------------------------------+|
|  | claude-sonnet-4   | anthropic |  98.9%  |   26.1%  |  57.0%  | $0.70||
|  | gpt-4o            | openrouter|  99.0%  |   31.2%  |  48.3%  | $1.25||
|  | llava:7b          | ollama    |  87.5%  |   12.4%  |  72.1%  | free ||
|  +---------------------------------------------------------------------+|
|                                                                         |
|  +---------------------------------------------------------------------+|
|  | RESULTS BY CLASS (aggregated across models)                         ||
|  +---------------------------------------------------------------------+|
|  | Class | claude-sonnet-4 | gpt-4o      | llava:7b    | Best         ||
|  +---------------------------------------------------------------------+|
|  | CTRL  | Det:  0% FP:100%| Det: 50%... | Det:  0%... | gpt-4o       ||
|  | USSS  | Det: 21% FP: 64%| Det: 28%... | Det: 10%... | gpt-4o       ||
|  | HSFR  | Det: 61% FP: 19%| Det: 72%... | Det: 45%... | gpt-4o       ||
|  +---------------------------------------------------------------------+|
|                                                                         |
|  Select model: [claude-sonnet-4 v]  Filter: [Class: All v]              |
|                                                                         |
|  +---------------------------------------------------------------------+|
|  | INDIVIDUAL TESTS (claude-sonnet-4)                                  ||
|  +---------------------------------------------------------------------+|
|  | Sample ID          | Task   | Status | Det.Rate | FP Rate | Action ||
|  +---------------------------------------------------------------------+|
|  | CTRL_empty_wb      | LOCATE | FAIL   |  100.0%  |   0.0%  | [View] ||
|  | CTRL_single_wb     | LOCATE | OK     |    0.0%  | 100.0%  | [View] ||
|  | USSS_s2_d20_wb     | LOCATE | OK     |   15.0%  |  83.3%  | [View] ||
|  +---------------------------------------------------------------------+|
+=========================================================================+
```

### 4.3 Individual Test Page (with Model Comparison Option)

```
+=========================================================================+
|  [< Back to Run Summary]                                                |
|                                                                         |
|       TEST: USSS_s2_d20_wb - LOCATE   Model: [claude-sonnet-4 v]        |
+=========================================================================+
|                                                                         |
|  +---------------------------+    +---------------------------+         |
|  | TEST DETAILS              |    | METRICS                   |         |
|  +---------------------------+    +---------------------------+         |
|  | Sample: USSS_s2_d20_wb    |    | Detection Rate:    15.0%  |         |
|  | Task: LOCATE              |    | False Positive:    83.3%  |         |
|  | Class: USSS               |    | Count Diff:            2  |         |
|  | Model: claude-sonnet-4    |    | Mean Distance:     0.123  |         |
|  | Status: SUCCESS           |    | GT Count:             20  |         |
|  | Turns: 2                  |    | Predicted:            18  |         |
|  | Total Latency: 5890ms     |    |                           |         |
|  | Tokens: 1262 in / 394 out |    |                           |         |
|  | Cost: $0.0097             |    |                           |         |
|  +---------------------------+    +---------------------------+         |
|                                                                         |
|  +---------------------------------------------------------------------+|
|  | COMPARE MODELS ON THIS SAMPLE                                       ||
|  +---------------------------------------------------------------------+|
|  | Model             | Det.Rate | FP Rate | Turns | Latency | Cost     ||
|  +---------------------------------------------------------------------+|
|  | claude-sonnet-4   |    15.0% |   83.3% |     2 | 5890ms  | $0.0097  ||
|  | gpt-4o            |    25.0% |   60.0% |     1 | 2105ms  | $0.0082  ||
|  | llava:7b          |     5.0% |   95.0% |     3 | 4842ms  | free     ||
|  +---------------------------------------------------------------------+|
|                                                                         |
|  +---------------------------------------------------------------------+|
|  | IMAGES                                                              ||
|  +---------------------------------------------------------------------+|
|  |  +---------------------------+   +---------------------------+      ||
|  |  |                           |   |                           |      ||
|  |  |    ORIGINAL IMAGE         |   |    ANNOTATED RESULT       |      ||
|  |  |                           |   |                           |      ||
|  |  |    [test image with       |   |    [same image with       |      ||
|  |  |     white spots on        |   |     green GT circles,     |      ||
|  |  |     black background]     |   |     red predictions,      |      ||
|  |  |                           |   |     blue match lines]     |      ||
|  |  |                           |   |   +-----------------------+|      ||
|  |  |                           |   |   |sample|task|det:15%   ||      ||
|  |  +---------------------------+   +---------------------------+      ||
|  |       [Open Full Size]              [Open Full Size]                ||
|  +---------------------------------------------------------------------+|
|                                                                         |
|  +---------------------------------------------------------------------+|
|  | CONVERSATION HISTORY (2 turns)                                      ||
|  +---------------------------------------------------------------------+|
|  |                                                                     ||
|  |  [Turn 1]                                                           ||
|  |  +---------------------------------------------------------------+ ||
|  |  | USER (with image)                                             | ||
|  |  +---------------------------------------------------------------+ ||
|  |  | Examine this image and identify the location of each circular | ||
|  |  | spot. Use normalized coordinates where (0.0, 0.0) is top-left.| ||
|  |  +---------------------------------------------------------------+ ||
|  |  | ASSISTANT                                    [2395ms, 212 tok] | ||
|  |  +---------------------------------------------------------------+ ||
|  |  | Looking at the image, I can see several spots but let me      | ||
|  |  | examine more carefully...                                     | ||
|  |  +---------------------------------------------------------------+ ||
|  |                                                                     ||
|  |  [Turn 2]                                                           ||
|  |  +---------------------------------------------------------------+ ||
|  |  | USER                                                          | ||
|  |  +---------------------------------------------------------------+ ||
|  |  | Please provide the coordinates in the exact format requested. | ||
|  |  +---------------------------------------------------------------+ ||
|  |  | ASSISTANT (final)                            [3495ms, 182 tok] | ||
|  |  +---------------------------------------------------------------+ ||
|  |  | 0.15, 0.08                                                    | ||
|  |  | 0.19, 0.10                                                    | ||
|  |  | ...                                                           | ||
|  |  +---------------------------------------------------------------+ ||
|  |                                                                     ||
|  +---------------------------------------------------------------------+|
|                                                                         |
|  +--------------------------------+  +--------------------------------+ |
|  | GROUND TRUTH (20 positions)   |  | PREDICTED (18 positions)       | |
|  +--------------------------------+  +--------------------------------+ |
|  | (0.157, 0.204)                |  | (0.15, 0.08)                   | |
|  | (0.324, 0.451)                |  | (0.19, 0.10)                   | |
|  | ...                           |  | ...                            | |
|  +--------------------------------+  +--------------------------------+ |
+=========================================================================+
```

---

## 6. New Module Structure

```
src/vlm_geometry_bench/
+-- traceability/
    +-- __init__.py
    +-- schemas.py           # Pydantic models for JSON data
    +-- artifact_manager.py  # Folder/file creation and management
    +-- image_annotator.py   # Annotated image generation (Pillow)
    +-- html_generator.py    # Static HTML generation (Jinja2)
    +-- index_manager.py     # Master index.json management
```

---

## 7. Implementation Tasks

### Phase 1: Multi-Model CLI Support ✅ COMPLETE
1. ✅ Modify CLI to accept `--models` (comma-separated list) in addition to `--model`
2. ✅ Add `--run-name` option for naming comparison runs
3. ✅ Add `--comment` option for describing the purpose of the run
4. ✅ Modify `EvaluationConfig` to support multiple models and comment
5. ✅ Modify `GeometryBenchEvaluator` to iterate over models sequentially

### Phase 2: Data Model and Storage ✅ COMPLETE
5. ✅ Create `traceability/schemas.py` - Pydantic models for RunIndex, RunMetadata, ModelMetadata, SampleTestResult
6. ✅ Create `traceability/artifact_manager.py` - Folder creation, file saving, size calculation
7. ✅ Create `traceability/index_manager.py` - Read/write/update index.json
8. ✅ Add shared images folder to deduplicate original images across models

### Phase 3: Image Annotation ✅ COMPLETE
9. ✅ Create `traceability/image_annotator.py` - Base annotator class with Pillow
10. ✅ Implement LOCATE task annotation (circles, lines, status bar)
11. ✅ Implement COUNT/PATTERN/SIZE/DEFECT annotations (primarily status bar)

### Phase 4: HTML Generation ✅ COMPLETE
12. ✅ Create `traceability/html_generator.py` with Jinja2 templates
13. ✅ Implement index.html template (all runs list with filters)
14. ✅ Implement summary.html template (multi-model comparison + per-model tests)
15. ✅ Implement test.html template (individual test with model comparison table)
16. ✅ Create CSS stylesheet (clean, responsive design)
17. ✅ Add JavaScript for filtering/sorting and model selection

### Phase 5: Integration ✅ COMPLETE
18. ✅ Modify `evaluator.py` to save artifacts during evaluation
19. ✅ Modify `reporter.py` to integrate with artifact manager
20. ✅ Add CLI flags: `--traceability` to enable, `--results-dir` for output location
21. ✅ Store prompt text in SampleResult for conversation.json

### Phase 6: Testing ✅ COMPLETE
22. ✅ Unit tests for schemas and artifact manager (`test_schemas.py`, `test_artifact_manager.py`)
23. ✅ Unit tests for image annotator (`test_image_annotator.py`)
24. ✅ Unit tests for index manager (`test_index_manager.py`)
25. ✅ Unit tests for HTML generator (`test_html_generator.py`)

**Total: 119 tests in `tests/test_traceability/`**

---

## 8. Critical Files to Modify

| File | Changes |
|------|---------|
| `src/vlm_geometry_bench/__main__.py` | Add `--models`, `--run-name`, `--comment`, `--traceability` CLI flags |
| `src/vlm_geometry_bench/config.py` | Add `models` list, `run_name`, `comment`, traceability config options |
| `src/vlm_geometry_bench/evaluator.py` | Iterate over multiple models, add artifact-saving hooks |
| `src/vlm_geometry_bench/reporter.py` | Integrate with artifact manager, support multi-model runs |
| `src/vlm_geometry_bench/metrics.py` | Add `prompt` and `model` fields to SampleResult |

---

## 9. Dependencies to Add

```toml
# pyproject.toml
[project.dependencies]
pillow = ">=10.0.0"  # Already present for image loading
jinja2 = ">=3.0.0"   # HTML template generation
pydantic = ">=2.0.0" # Schema validation (optional but recommended)
```

---

## 10. Verification Plan

1. **Unit tests**: Run `uv run pytest tests/test_traceability/`

2. **Single-model test**:
   ```bash
   uv run vlm-geometry-bench --backend ollama --model llava:7b \
       --testsuite d:/python/imagegen/testsuite \
       --samples 5 --classes CTRL,USSS --tasks LOCATE \
       --traceability --results-dir ./results
   ```

3. **Multi-model comparison test**:
   ```bash
   uv run vlm-geometry-bench \
       --models llava:7b,llava:13b \
       --run-name "llava-comparison" \
       --comment "Comparing llava 7b vs 13b on LOCATE task" \
       --testsuite d:/python/imagegen/testsuite \
       --samples 3 --classes CTRL,USSS --tasks LOCATE \
       --traceability --results-dir ./results
   ```

4. **Verify outputs**:
   - Check `results/index.json` exists and lists the run(s)
   - Check run folder has `run_metadata.json`, `summary.html`
   - For multi-model: check `models/` subfolder per model
   - Check `images/` folder has deduplicated originals
   - Check test folders have all expected files (test_metadata.json, annotated.png, conversation.json, test.html)
   - Open `summary.html` in browser:
     - Verify model comparison table appears
     - Verify model selector dropdown works
     - Verify individual test links work
   - Open individual `test.html`:
     - Verify annotated image shows GT circles (green), predictions (red), match lines (blue)
     - Verify status bar at bottom of annotated image
     - Verify model comparison table shows all models' results for this sample
   - Verify annotated images are self-contained (can be viewed standalone with identifying info)
