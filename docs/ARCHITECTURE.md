# VLM Geometry Bench - Architecture Design

## Overview

VLM Geometry Bench is a benchmark for evaluating Vision Language Models' ability to identify, count, and locate simple geometric shapes. It combines:

- **Test suite generation** from `imagegen` (synthetic images with known ground truth)
- **Evaluation framework** adapted from `salbench` (VLM API integration, metrics, reporting)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VLM GEOMETRY BENCH                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   CLI        │    │  Config      │    │  Test Suite  │                  │
│  │  (run_eval)  │───▶│  Manager     │◀───│  Loader      │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                           │
│         │                   ▼                   ▼                           │
│         │            ┌──────────────────────────────────┐                  │
│         └───────────▶│         EVALUATOR                │                  │
│                      │  ┌────────────────────────────┐  │                  │
│                      │  │  For each sample:          │  │                  │
│                      │  │   1. Load image + ground   │  │                  │
│                      │  │   2. Build prompt          │  │                  │
│                      │  │   3. Send to VLM           │  │                  │
│                      │  │   4. Parse response        │  │                  │
│                      │  │   5. Compute metrics       │  │                  │
│                      │  └────────────────────────────┘  │                  │
│                      └──────────────────────────────────┘                  │
│                         │              │              │                     │
│           ┌─────────────┘              │              └─────────────┐       │
│           ▼                            ▼                            ▼       │
│  ┌──────────────┐            ┌──────────────┐            ┌──────────────┐  │
│  │   Vision     │            │   Response   │            │   Metrics    │  │
│  │   Client     │            │   Parser     │            │   Calculator │  │
│  └──────────────┘            └──────────────┘            └──────────────┘  │
│           │                                                      │          │
│           ▼                                                      ▼          │
│  ┌──────────────┐                                      ┌──────────────┐    │
│  │  OpenRouter  │                                      │   Results    │    │
│  │  or Ollama   │                                      │   Reporter   │    │
│  └──────────────┘                                      └──────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                      │
└─────────────────────────────────────────────────────────────────────────────┘

    IMAGEGEN                    VLM-GEOMETRY-BENCH                    VLM API
    (External)                                                        (External)
        │                                                                 │
        │  ┌─────────────┐                                               │
        │  │ testsuite/  │                                               │
        │  │ ├─manifest  │                                               │
        │  │ ├─CTRL_*.png│                                               │
        │  │ ├─USSS_*.png│                                               │
        │  │ └─...       │                                               │
        │  └─────────────┘                                               │
        │         │                                                       │
        ▼         ▼                                                       │
   ┌─────────────────────┐                                               │
   │   Test Suite Loader │                                               │
   │   ─────────────────  │                                               │
   │   • Parse manifest   │                                               │
   │   • Load PNG images  │                                               │
   │   • Extract metadata │                                               │
   │   • Build samples    │                                               │
   └─────────────────────┘                                               │
              │                                                           │
              │  BenchmarkSample[]                                        │
              │  ┌────────────────┐                                       │
              │  │ image: PIL     │                                       │
              │  │ ground_truth:  │                                       │
              │  │   spot_count   │                                       │
              │  │   positions    │                                       │
              │  │   pattern      │                                       │
              │  │ metadata:      │                                       │
              │  │   class, etc   │                                       │
              │  └────────────────┘                                       │
              ▼                                                           │
   ┌─────────────────────┐      ┌─────────────────────┐                  │
   │   Prompt Builder    │      │   Vision Client     │                  │
   │   ───────────────   │      │   ─────────────     │                  │
   │   • Select task     │─────▶│   • Encode image    │                  │
   │   • Format prompt   │      │   • Build request   │─────────────────▶│
   │   • Add few-shot    │      │   • Send + retry    │                  │
   └─────────────────────┘      │   • Track tokens    │◀─────────────────│
                                └─────────────────────┘                  │
                                          │                               │
                                          │  VLM Response                 │
                                          ▼                               │
                                ┌─────────────────────┐                  │
                                │   Response Parser   │                  │
                                │   ───────────────   │                  │
                                │   • Extract count   │                  │
                                │   • Parse coords    │                  │
                                │   • Classify pattern│                  │
                                └─────────────────────┘                  │
                                          │                               │
                                          │  ParsedResponse               │
                                          ▼                               │
                                ┌─────────────────────┐                  │
                                │   Metrics Calculator│                  │
                                │   ─────────────────  │                  │
                                │   • Count accuracy  │                  │
                                │   • Position error  │                  │
                                │   • Classification  │                  │
                                └─────────────────────┘                  │
                                          │                               │
                                          │  SampleMetrics                │
                                          ▼                               │
                                ┌─────────────────────┐                  │
                                │   Results Reporter  │                  │
                                │   ─────────────────  │                  │
                                │   • Aggregate stats │                  │
                                │   • Generate JSON   │                  │
                                │   • Console output  │                  │
                                └─────────────────────┘                  │
                                          │                               │
                                          ▼                               │
                                ┌─────────────────────┐                  │
                                │   results/          │                  │
                                │   ├─metrics.json    │                  │
                                │   ├─leaderboard.csv │                  │
                                │   └─raw_responses   │                  │
                                └─────────────────────┘                  │
```

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT RELATIONSHIPS                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   run_eval.py   │
                              │   (CLI Entry)   │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ EvaluationConfig│
                              │   (dataclass)   │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │ TestSuiteLoader│  │ VisionClient  │  │ PromptBuilder │
           └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                   │                  │                  │
                   │                  │                  │
                   └──────────────────┼──────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │  Evaluator    │
                              │  (orchestrate)│
                              └───────┬───────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │ResponseParser │  │MetricsCalculator│ │ResultsReporter│
           └───────────────┘  └───────────────┘  └───────────────┘
```

---

## Image Classes and Evaluation Tasks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMAGE CLASSES → EVALUATION TASKS                         │
└─────────────────────────────────────────────────────────────────────────────┘

  IMAGE CLASS              DESCRIPTION                    EVALUATION TASKS
  ───────────────────────────────────────────────────────────────────────────

  ┌─────────┐
  │  CTRL   │  Control images                    ┌─────────────────────────┐
  │         │  • Empty canvas                    │ • Count (expect 0 or 1) │
  │         │  • Single centered spot            │ • False positive check  │
  └─────────┘                                    └─────────────────────────┘

  ┌─────────┐
  │  USSS   │  Uniform Spots Same Size           ┌─────────────────────────┐
  │         │  • Random placement                │ • Count accuracy        │
  │         │  • Identical spot sizes            │ • Size estimation       │
  │         │  • Various densities (20,50,100)   │ • Pattern = "random"    │
  └─────────┘                                    └─────────────────────────┘

  ┌─────────┐
  │  USDS   │  Uniform Spots Different Sizes     ┌─────────────────────────┐
  │         │  • Random placement                │ • Count accuracy        │
  │         │  • Variable spot sizes             │ • Size range detection  │
  │         │  • Various densities               │ • Pattern = "random"    │
  └─────────┘                                    └─────────────────────────┘

  ┌─────────┐
  │  HSFR   │  Hex Spots Fixed Rigid             ┌─────────────────────────┐
  │         │  • Perfect hexagonal grid          │ • Count accuracy        │
  │         │  • Fixed positions                 │ • Pattern = "hexagonal" │
  │         │  • Various spacings (8,12,16 µm)   │ • Spacing estimation    │
  └─────────┘                                    └─────────────────────────┘

  ┌─────────┐
  │  HSRP   │  Hex Spots Random Perturbation     ┌─────────────────────────┐
  │         │  • Hexagonal base pattern          │ • Count accuracy        │
  │         │  • Random displacement ±1-2 µm     │ • Pattern = "hexagonal" │
  │         │  • Tests robustness to noise       │ • Regularity assessment │
  └─────────┘                                    └─────────────────────────┘

  ┌─────────┐
  │  HSDN   │  Hex Spots Defects + Noise         ┌─────────────────────────┐
  │         │  • Hexagonal with missing spots    │ • Count accuracy        │
  │         │  • Random noise spots added        │ • Defect detection      │
  │         │  • 5-20% defect, 5-10% noise       │ • Pattern = "hexagonal" │
  └─────────┘                                    └─────────────────────────┘
```

---

## Evaluation Tasks Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION TASK TYPES                               │
└─────────────────────────────────────────────────────────────────────────────┘

  TASK ID    NAME              PROMPT EXAMPLE                    METRICS
  ───────────────────────────────────────────────────────────────────────────

  COUNT      Spot Counting     "How many circular spots         • Exact match
                               are in this image?"              • Within ±N
                                                                • % error

  LOCATE     Localization      "List the (x,y) coordinates      • Mean distance
                               of each spot in pixels"          • Detection rate
                                                                • False positives

  PATTERN    Pattern Type      "Is the arrangement of spots:    • Accuracy
                               random, hexagonal grid,          • Confusion matrix
                               or regular grid?"

  SIZE       Size Estimation   "Estimate the diameter of        • Mean abs error
                               the spots in micrometers"        • Within tolerance

  DEFECT     Defect Detection  "Are there any missing spots     • Precision
                               or irregularities in the         • Recall
                               pattern? Describe them."         • F1 score
```

---

## VLM Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VLM BACKEND INTEGRATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │  Vision Client  │
                              │   (Abstract)    │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
           ┌───────────────────┐               ┌───────────────────┐
           │   Ollama Backend  │               │ OpenRouter Backend│
           │   ──────────────  │               │ ─────────────────  │
           │                   │               │                   │
           │  localhost:11434  │               │ openrouter.ai/api │
           │  /v1/chat/complete│               │ /v1/chat/complete │
           │                   │               │                   │
           │  Models:          │               │  Models:          │
           │  • llava:7b       │               │  • gpt-4o         │
           │  • llava:13b      │               │  • claude-3.5     │
           │  • llava:34b      │               │  • qwen2.5-vl-72b │
           │  • minicpm-v      │               │  • gemini-pro     │
           └───────────────────┘               └───────────────────┘
                    │                                     │
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────┐
                         │   Unified Response    │
                         │   ─────────────────   │
                         │   {                   │
                         │     content: string   │
                         │     tokens_in: int    │
                         │     tokens_out: int   │
                         │     latency_ms: int   │
                         │   }                   │
                         └───────────────────────┘
```

---

## Directory Structure

```
vlm-geometry-bench/
│
├── pyproject.toml              # Package configuration
├── README.md                   # Project overview
│
├── docs/
│   ├── ARCHITECTURE.md         # This document
│   └── USAGE.md                # User guide
│
├── src/
│   └── vlm_geometry_bench/
│       ├── __init__.py
│       ├── __main__.py         # CLI entry point
│       │
│       ├── config.py           # EvaluationConfig dataclass
│       ├── data_loader.py      # Test suite loading
│       ├── vision_client.py    # VLM API communication
│       ├── prompts.py          # Task-specific prompts
│       ├── response_parser.py  # Parse VLM responses
│       ├── metrics.py          # Metrics calculation
│       ├── evaluator.py        # Main orchestration
│       └── reporter.py         # Results output
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_response_parser.py
│   ├── test_metrics.py
│   └── fixtures/               # Test data
│
└── results/                    # Evaluation outputs
    └── <model>_<timestamp>/
        ├── metrics.json
        ├── leaderboard.csv
        └── raw_responses.json
```

---

## Configuration Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION CONFIG                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  EvaluationConfig
  ├── Backend Settings
  │   ├── backend: str          "ollama" | "openrouter"
  │   ├── base_url: str         API endpoint base
  │   ├── api_key: str?         For OpenRouter (or env var)
  │   └── model_name: str       e.g., "llava:7b", "gpt-4o"
  │
  ├── Test Suite Settings
  │   ├── testsuite_path: str   Path to imagegen output
  │   ├── image_classes: list   ["CTRL","USSS","USDS","HSFR","HSRP","HSDN"]
  │   └── num_samples: int?     Limit samples (None = all)
  │
  ├── Evaluation Settings
  │   ├── tasks: list           ["COUNT","LOCATE","PATTERN","SIZE","DEFECT"]
  │   ├── num_shots: int        0, 3, or 5 (few-shot examples)
  │   └── count_tolerance: int  ±N for "close enough" counting
  │
  ├── API Settings
  │   ├── timeout_seconds: int  Request timeout (default: 120)
  │   ├── retry_attempts: int   Retry count (default: 3)
  │   ├── temperature: float    Sampling temp (default: 0.0)
  │   └── max_tokens: int       Response limit (default: 512)
  │
  └── Output Settings
      └── output_dir: str       Results directory
```

---

## Metrics Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          METRICS BY TASK                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  COUNT TASK                                                             │
  │  ───────────                                                            │
  │                                                                         │
  │  exact_match    = (predicted == ground_truth)                           │
  │  within_N       = |predicted - ground_truth| ≤ N                        │
  │  pct_error      = |predicted - ground_truth| / ground_truth × 100       │
  │  mean_abs_error = mean(|predicted - ground_truth|) across samples       │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  PATTERN TASK                                                           │
  │  ────────────                                                           │
  │                                                                         │
  │  accuracy       = correct_classifications / total_samples               │
  │  per_class_f1   = F1 for each pattern type (random, hex, grid)          │
  │  macro_f1       = mean(per_class_f1)                                    │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  DEFECT TASK                                                            │
  │  ───────────                                                            │
  │                                                                         │
  │  detection_rate = correctly_identified_defects / total_defects          │
  │  false_pos_rate = false_defect_claims / non_defect_images               │
  │  f1_score       = 2 × (precision × recall) / (precision + recall)       │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  AGGREGATE METRICS                                                      │
  │  ─────────────────                                                      │
  │                                                                         │
  │  Per image class (CTRL, USSS, USDS, HSFR, HSRP, HSDN):                  │
  │    • Task-specific metrics averaged within class                        │
  │                                                                         │
  │  Overall:                                                               │
  │    • Macro average across all classes                                   │
  │    • Success rate (valid API responses / total requests)                │
  │    • Total cost (for OpenRouter)                                        │
  │    • Total time                                                         │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Sequence Diagram: Single Sample Evaluation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE SAMPLE EVALUATION SEQUENCE                        │
└─────────────────────────────────────────────────────────────────────────────┘

  Evaluator      DataLoader     PromptBuilder   VisionClient    ResponseParser
      │              │               │               │               │
      │──get_sample─▶│               │               │               │
      │              │               │               │               │
      │◀─BenchmarkSample─           │               │               │
      │   (image, ground_truth)     │               │               │
      │              │               │               │               │
      │──────────build_prompt──────▶│               │               │
      │              │               │               │               │
      │◀─────────prompt_text────────│               │               │
      │              │               │               │               │
      │───────────────send_request(image, prompt)──▶│               │
      │              │               │               │               │
      │              │               │    ┌─────────────────────┐   │
      │              │               │    │ • Encode to base64  │   │
      │              │               │    │ • Build API payload │   │
      │              │               │    │ • POST to VLM       │   │
      │              │               │    │ • Retry on failure  │   │
      │              │               │    └─────────────────────┘   │
      │              │               │               │               │
      │◀──────────────VLMResponse (text, tokens)────│               │
      │              │               │               │               │
      │──────────────────────parse_response────────────────────────▶│
      │              │               │               │               │
      │              │               │               │    ┌─────────────────┐
      │              │               │               │    │ • Extract count │
      │              │               │               │    │ • Parse coords  │
      │              │               │               │    │ • Classify type │
      │              │               │               │    └─────────────────┘
      │              │               │               │               │
      │◀─────────────────────ParsedResponse─────────────────────────│
      │              │               │               │               │
      │──compute_metrics(parsed, ground_truth)      │               │
      │              │               │               │               │
      │  SampleResult                │               │               │
      │  ┌─────────────────────┐     │               │               │
      │  │ sample_id           │     │               │               │
      │  │ predicted           │     │               │               │
      │  │ ground_truth        │     │               │               │
      │  │ metrics {}          │     │               │               │
      │  │ raw_response        │     │               │               │
      │  │ tokens_used         │     │               │               │
      │  └─────────────────────┘     │               │               │
      │              │               │               │               │
```

---

## Extensibility Points

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTENSIBILITY GUIDE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  TO ADD NEW IMAGE CLASSES:
  ────────────────────────
  1. Update imagegen's testsuite-config.yaml with new class definition
  2. Regenerate test suite with gentestsuite
  3. Add class name to VALID_CLASSES in config.py
  4. (Optional) Add class-specific prompts if needed

  TO ADD NEW EVALUATION TASKS:
  ───────────────────────────
  1. Define task ID constant in config.py (e.g., "SYMMETRY")
  2. Add prompt template in prompts.py
  3. Add parsing logic in response_parser.py
  4. Add metrics calculation in metrics.py
  5. Update evaluator.py to handle new task

  TO ADD NEW VLM BACKENDS:
  ───────────────────────
  1. Add backend name to VALID_BACKENDS in config.py
  2. Implement endpoint logic in vision_client.py
  3. Add any required authentication handling
  4. (Optional) Add pricing info for cost estimation

  TO CUSTOMIZE PROMPTS:
  ────────────────────
  • Modify templates in prompts.py
  • Add few-shot examples per task
  • Adjust for specific model capabilities
```

---

## Error Handling Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ERROR HANDLING                                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ERROR TYPE              HANDLING                      RESULT
  ────────────────────────────────────────────────────────────────────────────

  API Timeout             Retry with exponential        After 3 retries:
                          backoff (2^n seconds)         Mark as failed

  API Rate Limit          Wait and retry                Continue after delay

  Invalid Response        Log warning                   Mark as parse_error
  (unparseable)           Store raw response            Exclude from metrics

  Image Load Error        Skip sample                   Log and continue
                          Log error details

  Network Error           Retry with backoff            After 3 retries:
                                                        Mark as failed

  ────────────────────────────────────────────────────────────────────────────

  RESULT TRACKING:
  ───────────────
  • success_rate = successful_samples / total_samples
  • Each sample result includes error field (null if successful)
  • Failed samples included in raw_responses.json for debugging
  • Metrics computed only over successful samples
```

---

## Sample Output Format

```json
{
  "config": {
    "model": "gpt-4o",
    "backend": "openrouter",
    "testsuite": "./testsuite",
    "tasks": ["COUNT", "PATTERN"],
    "image_classes": ["CTRL", "USSS", "HSFR"]
  },
  "usage": {
    "elapsed_seconds": 245.3,
    "total_requests": 92,
    "failed_requests": 2,
    "input_tokens": 185000,
    "output_tokens": 4200,
    "estimated_cost_usd": 4.82
  },
  "results_by_class": {
    "CTRL": {
      "COUNT": { "exact_match": 100.0, "mean_error": 0.0 }
    },
    "USSS": {
      "COUNT": { "exact_match": 45.0, "mean_error": 8.3 },
      "PATTERN": { "accuracy": 92.0, "f1": 0.91 }
    },
    "HSFR": {
      "COUNT": { "exact_match": 33.0, "mean_error": 12.1 },
      "PATTERN": { "accuracy": 78.0, "f1": 0.76 }
    }
  },
  "overall": {
    "COUNT": { "exact_match": 52.3, "mean_error": 7.8 },
    "PATTERN": { "accuracy": 85.0, "macro_f1": 0.83 }
  }
}
```
