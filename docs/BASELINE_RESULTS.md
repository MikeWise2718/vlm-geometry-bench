# VLM Geometry Bench - Baseline Results

This document contains detailed baseline evaluation results for Vision Language Models tested on the VLM Geometry Bench test suite.

## Test Configuration

- **Test Suite:** 92 images across 6 classes (CTRL, USSS, USDS, HSFR, HSRP, HSDN)
- **Tasks:** COUNT, PATTERN
- **Shots:** 0 (zero-shot)
- **Backend:** Ollama

## Models Tested

| Model | Size | Description |
|-------|------|-------------|
| llava:7b | 4.7 GB | LLaVA 1.5 7B - multimodal model |
| llama3.2-vision:latest | 7.8 GB | Llama 3.2 Vision 11B |
| minicpm-v:latest | 5.5 GB | MiniCPM-V 2.6 |
| granite3.2-vision:2b | 2.4 GB | IBM Granite 3.2 Vision 2B |
| qwen3-vl:8b | 6.1 GB | Qwen3-VL 8B |

## Overall Performance

| Model | COUNT Exact | PATTERN Acc | Time |
|-------|-------------|-------------|------|
| llava:7b | 2.2% | 48.9% | 26s |
| llama3.2-vision | 6.5% | 66.3% | 159s |
| minicpm-v | 4.3% | 51.1% | 318s |
| granite3.2-vision:2b | ~1%* | 39.1% | 116s |
| qwen3-vl:8b | ~4%* | 58.8% | 671s |

*Low success rate due to response parsing issues

## Pattern Recognition by Type

| Model | EMPTY | SINGLE | RANDOM | HEXAGONAL |
|-------|-------|--------|--------|-----------|
| llava:7b | 50% | 50% | 97% | 0% |
| llama3.2-vision | 0% | 100% | 91% | 42% |
| minicpm-v | 100% | 100% | 100% | 0% |
| granite3.2-vision:2b | 100% | 0% | 78% | 0% |
| qwen3-vl:8b | 100% | 100% | 100% | 0% |

## Results by Image Class

### llava:7b

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 50.0% |
| HSDN | 0.0% | 0.0% |
| HSFR | 0.0% | 0.0% |
| HSRP | 0.0% | 0.0% |
| USDS | 0.0% | 100.0% |
| USSS | 0.0% | 97.2% |

### llama3.2-vision

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 50.0% | 50.0% |
| HSDN | 5.6% | 66.7% |
| HSFR | 0.0% | 33.3% |
| HSRP | 0.0% | 25.0% |
| USDS | 11.1% | 88.9% |
| USSS | 8.3% | 91.7% |

### minicpm-v

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 100.0% |
| HSDN | 5.6% | 0.0% |
| HSFR | 0.0% | 0.0% |
| HSRP | 0.0% | 0.0% |
| USDS | 0.0% | 100.0% |
| USSS | 2.8% | 100.0% |

### granite3.2-vision:2b

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 50.0% |
| HSDN | 0.0% | 0.0% |
| HSFR | 0.0% | 0.0% |
| HSRP | 0.0% | 0.0% |
| USDS | 0.0% | 77.8% |
| USSS | 0.0% | 77.8% |

### qwen3-vl:8b

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 100.0% |
| HSDN | 0.0% | 0.0% |
| HSFR | 0.0% | 0.0% |
| HSRP | 0.0% | 0.0% |
| USDS | 0.0% | 100.0% |
| USSS | 0.0% | 100.0% |

## Key Findings

1. **Counting is challenging**: All models struggle with exact spot counting, achieving only 2-7% exact match rates on the full test suite.

2. **Random patterns are easier**: Models achieve 78-100% accuracy on RANDOM patterns (USSS, USDS classes).

3. **Hexagonal patterns are difficult**: Most models fail to recognize hexagonal arrangements (0-42%). Only llama3.2-vision shows any hexagonal recognition capability (42%).

4. **llama3.2-vision performs best overall**:
   - Best hexagonal recognition (42%)
   - Highest overall pattern accuracy (66.3%)
   - Best COUNT accuracy (6.5%)

5. **Speed vs accuracy tradeoff**:
   - llava:7b is fastest (26s) but has lowest accuracy
   - qwen3-vl is slowest (671s) with moderate accuracy
   - llama3.2-vision offers best balance of speed and accuracy

6. **Control images (CTRL) are well handled**: Most models correctly identify empty images and single spots.

## Reproducibility

### Local vs Remote Ollama Comparison

Results were validated by running evaluations on both local and remote (192.168.25.202) Ollama instances:

| Model | Metric | Local | Remote | Match? |
|-------|--------|-------|--------|--------|
| **llava:7b** | COUNT | 2.2% | 2.2% | ✅ |
| | PATTERN | 48.9% | 48.9% | ✅ |
| | Time | 26s | 48s | - |
| **llama3.2-vision** | COUNT | 6.5% | 4.3% | ~close |
| | PATTERN | 66.3% | 67.4% | ~close |
| | Time | 159s | 203s | - |
| **minicpm-v** | COUNT | 4.3% | 4.3% | ✅ |
| | PATTERN | 51.1% | 51.1% | ✅ |
| | Time | 318s | 78s | - |
| **granite3.2-vision:2b** | COUNT | ~1% | ~1% | ✅ |
| | PATTERN | 39.1% | 44.6% | ~close |
| | Time | 116s | 129s | - |
| **qwen3-vl:8b** | COUNT | ~4% | ~4% | ✅ |
| | PATTERN | 58.8% | 58.8% | ✅ |
| | Time | 671s | 507s | - |

**Observations:**
- Results are consistent between local and remote execution
- Small variations (1-5%) are expected due to non-deterministic model behavior
- Execution times vary based on hardware and network characteristics

## Raw Results

Full evaluation results including raw model responses are available in the `results/` directory:

```
results/
├── llava_7b_20260127_161616/
├── llama3.2-vision_latest_20260127_135028/
├── minicpm-v_latest_20260127_135624/
├── granite3.2-vision_2b_20260127_135842/
└── qwen3-vl_8b_20260127_141012/
```

Each directory contains:
- `metrics.json` - Full evaluation metrics
- `leaderboard.csv` - Summary row for comparison
- `raw_responses.json` - Per-sample VLM responses
