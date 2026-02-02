# VLM Geometry Bench - Baseline Results

This document contains detailed baseline evaluation results for Vision Language Models tested on the VLM Geometry Bench test suite.

## Test Configuration

- **Test Suite:** 92 images across 6 classes (CTRL, USSS, USDS, HSFR, HSRP, HSDN)
- **Tasks:** COUNT, PATTERN, LOCATE
- **Shots:** 0 (zero-shot)
- **Backends:** Ollama (local), Anthropic API
- **LOCATE tolerance:** 0.05 (5% of image dimension in normalized coordinates)

## Models Tested

### Anthropic Claude 4.5 Models (via API)

| Model | Description | Cost/Run |
|-------|-------------|----------|
| claude-opus-4-5-20251101 | Claude Opus 4.5 - Anthropic's most capable model | $1.64 |
| claude-sonnet-4-5-20250929 | Claude Sonnet 4.5 - Balanced performance/cost | $0.33 |
| claude-haiku-4-5-20251001 | Claude Haiku 4.5 - Fast and efficient | $0.11 |

### Local Models (via Ollama)

| Model | Size | Description |
|-------|------|-------------|
| llava:7b | 4.7 GB | LLaVA 1.5 7B - multimodal model |
| llama3.2-vision:latest | 7.8 GB | Llama 3.2 Vision 11B |
| minicpm-v:latest | 5.5 GB | MiniCPM-V 2.6 |
| granite3.2-vision:2b | 2.4 GB | IBM Granite 3.2 Vision 2B |
| qwen3-vl:8b | 6.1 GB | Qwen3-VL 8B |

## Overall Performance

### Claude 4.5 Models (Anthropic API)

| Model | COUNT Exact | PATTERN Acc | Time | Cost |
|-------|-------------|-------------|------|------|
| claude-opus-4.5 | 3.3% | **67.4%** | 760s | $1.64 |
| claude-sonnet-4.5 | 10.9% | 65.2% | 391s | $0.33 |
| **claude-haiku-4.5** | **15.2%** | 51.1% | 272s | $0.11 |

### Local Models (Ollama)

| Model | COUNT Exact | PATTERN Acc | Time |
|-------|-------------|-------------|------|
| llama3.2-vision | 6.5% | 66.3% | 159s |
| minicpm-v | 4.3% | 51.1% | 318s |
| qwen3-vl:8b | ~4%* | 58.8% | 671s |
| llava:7b | 2.2% | 48.9% | 26s |
| granite3.2-vision:2b | ~1%* | 39.1% | 116s |

*Low success rate due to response parsing issues

## Pattern Recognition by Type

### Claude 4.5 Models

| Model | EMPTY | SINGLE | RANDOM | HEXAGONAL |
|-------|-------|--------|--------|-----------|
| claude-opus-4.5 | 100% | 100% | 100% | 24% |
| claude-sonnet-4.5 | 100% | 100% | 100% | 20% |
| claude-haiku-4.5 | 100% | 100% | 100% | 0% |

### Local Models

| Model | EMPTY | SINGLE | RANDOM | HEXAGONAL |
|-------|-------|--------|--------|-----------|
| llama3.2-vision | 0% | 100% | 91% | 42% |
| minicpm-v | 100% | 100% | 100% | 0% |
| qwen3-vl:8b | 100% | 100% | 100% | 0% |
| llava:7b | 50% | 50% | 97% | 0% |
| granite3.2-vision:2b | 100% | 0% | 78% | 0% |

## Results by Image Class

### Claude 4.5 Models

#### claude-opus-4.5

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 100.0% |
| HSDN | 0.0% | 11.1% |
| HSFR | 0.0% | 33.3% |
| HSRP | 0.0% | 50.0% |
| USDS | 0.0% | 100.0% |
| USSS | 2.8% | 100.0% |

#### claude-sonnet-4.5

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 100.0% |
| HSDN | 5.6% | 5.6% |
| HSFR | 0.0% | 33.3% |
| HSRP | 20.8% | 45.8% |
| USDS | 11.1% | 100.0% |
| USSS | 2.8% | 100.0% |

#### claude-haiku-4.5

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 100.0% |
| HSDN | 0.0% | 0.0% |
| HSFR | 33.3% | 0.0% |
| HSRP | 12.5% | 0.0% |
| USDS | 22.2% | 100.0% |
| USSS | 16.7% | 100.0% |

### Local Models

#### llava:7b

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 50.0% |
| HSDN | 0.0% | 0.0% |
| HSFR | 0.0% | 0.0% |
| HSRP | 0.0% | 0.0% |
| USDS | 0.0% | 100.0% |
| USSS | 0.0% | 97.2% |

#### llama3.2-vision

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 50.0% | 50.0% |
| HSDN | 5.6% | 66.7% |
| HSFR | 0.0% | 33.3% |
| HSRP | 0.0% | 25.0% |
| USDS | 11.1% | 88.9% |
| USSS | 8.3% | 91.7% |

#### minicpm-v

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 100.0% |
| HSDN | 5.6% | 0.0% |
| HSFR | 0.0% | 0.0% |
| HSRP | 0.0% | 0.0% |
| USDS | 0.0% | 100.0% |
| USSS | 2.8% | 100.0% |

#### granite3.2-vision:2b

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 50.0% |
| HSDN | 0.0% | 0.0% |
| HSFR | 0.0% | 0.0% |
| HSRP | 0.0% | 0.0% |
| USDS | 0.0% | 77.8% |
| USSS | 0.0% | 77.8% |

#### qwen3-vl:8b

| Class | COUNT | PATTERN |
|-------|-------|---------|
| CTRL | 100.0% | 100.0% |
| HSDN | 0.0% | 0.0% |
| HSFR | 0.0% | 0.0% |
| HSRP | 0.0% | 0.0% |
| USDS | 0.0% | 100.0% |
| USSS | 0.0% | 100.0% |

## LOCATE Task Results

The LOCATE task asks models to identify spot positions using normalized (0-1) coordinates, where (0,0) is top-left and (1,1) is bottom-right. A prediction is considered correct if it falls within 5% (0.05) of the ground truth position.

### Claude Sonnet 4 LOCATE Performance

| Class | Detection Rate | Description |
|-------|---------------|-------------|
| **HSFR** | **61.3%** | Hexagonal fixed rigid - best performance |
| HSRP | 31.6% | Hexagonal random perturbation |
| HSDN | 28.0% | Hexagonal with defects + noise |
| USSS | 20.7% | Uniform spots same size |
| USDS | 20.1% | Uniform spots different sizes |
| CTRL | 0.0% | Control images (edge case) |

**Overall LOCATE Detection Rate: 26.1%**

### LOCATE Metrics Detail

| Metric | Value |
|--------|-------|
| Mean Nearest Distance | 0.08 (8% of image) |
| Mean Count Difference | 11.8 spots |
| False Positive Rate | 69.7% |
| Success Rate | 100% (all API calls succeeded) |
| Time | 561 seconds |
| Cost | $0.70 |

### LOCATE Key Observations

1. **Hexagonal patterns are easier to locate**: HSFR achieves 61.3% detection rate because the regular grid pattern is predictable.
2. **Random patterns are harder**: USSS/USDS achieve only ~20% detection rate.
3. **Position accuracy averages 8%**: Models are within 8% of the correct position on average, but our 5% threshold is strict.
4. **Count estimation is reasonable**: Models often get close to the correct count even when positions aren't precise.

## Key Findings

1. **Claude Haiku 4.5 is best for counting**: Surprisingly, the smallest Claude model achieves the highest COUNT accuracy (15.2%), outperforming both Sonnet (10.9%) and Opus (3.3%). This suggests smaller models may be better at following simple numeric output instructions.

2. **Claude Opus 4.5 is best for pattern recognition**: Opus achieves 67.4% overall PATTERN accuracy, the highest among all models tested. It also has the best hexagonal recognition among Claude models (24%).

3. **Random patterns are easy for Claude 4.5**: All three Claude 4.5 models achieve 100% accuracy on RANDOM pattern classification.

4. **Hexagonal patterns remain challenging**: llama3.2-vision still leads hexagonal recognition at 42%. Claude Opus 4.5 (24%) and Sonnet 4.5 (20%) show improvement over previous Claude versions but still struggle compared to local models.

5. **Cost efficiency favors Haiku**: Claude Haiku 4.5 costs only $0.11 per run while achieving the best counting performance. This makes it 15x cheaper than Opus ($1.64) with better counting accuracy.

6. **Trade-off between COUNT and PATTERN**: Models that excel at pattern recognition (Opus) tend to be worse at counting, while models focused on counting (Haiku) sacrifice pattern accuracy. Sonnet offers the best balance.

7. **Control images (CTRL) are well handled**: All Claude 4.5 models achieve 100% accuracy on both COUNT and PATTERN for control images.

8. **Local models competitive on pattern recognition**: llama3.2-vision (66.3%) matches Claude Opus 4.5 (67.4%) on overall pattern accuracy at zero cost.

9. **LOCATE task favors regular patterns**: Hexagonal patterns (61.3%) are much easier to locate than random patterns (20.7%) because the grid structure is predictable.

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
├── claude-opus-4-5-20251101_20260201_180902/      # COUNT, PATTERN
├── claude-sonnet-4-5-20250929_20260201_181254/    # COUNT, PATTERN
├── claude-haiku-4-5-20251001_20260201_181727/     # COUNT, PATTERN
├── claude-sonnet-4-20250514_20260202_130859/      # LOCATE (full)
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
