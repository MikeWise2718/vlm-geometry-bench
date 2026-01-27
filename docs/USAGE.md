# VLM Geometry Bench - Usage Guide

This guide covers detailed usage scenarios for VLM Geometry Bench.

## Prerequisites

1. **Test Suite**: Generate or obtain the test suite from imagegen
2. **VLM Backend**: Either Ollama (local) or OpenRouter API key

### Setting Up Ollama

```bash
# Install Ollama (see https://ollama.ai)
# Pull a vision-capable model
ollama pull llama3.2-vision
ollama pull minicpm-v
ollama pull qwen2-vl

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Setting Up OpenRouter

```bash
# Set API key as environment variable
export OPENROUTER_API_KEY="sk-or-..."

# Or pass via CLI
vlm-geometry-bench --api-key "sk-or-..." --backend openrouter
```

## Basic Usage

### Minimal Run

```bash
uv run vlm-geometry-bench \
    --testsuite d:/python/imagegen/testsuite
```

This uses defaults: ollama backend, llava:7b model, all tasks, all classes.

### Full Evaluation

```bash
uv run vlm-geometry-bench \
    --backend ollama \
    --model llama3.2-vision:latest \
    --testsuite d:/python/imagegen/testsuite \
    --tasks COUNT,PATTERN,LOCATE,SIZE,DEFECT \
    --classes CTRL,USSS,USDS,HSFR,HSRP,HSDN \
    --output ./results \
    --verbose
```

## Common Scenarios

### Quick Validation Test

Test a model on a small subset to verify setup:

```bash
uv run vlm-geometry-bench \
    --samples 3 \
    --classes CTRL,USSS \
    --tasks COUNT \
    --testsuite d:/python/imagegen/testsuite
```

### Compare Multiple Models

Run evaluations on different models and compare:

```bash
# Model 1
uv run vlm-geometry-bench \
    --model llama3.2-vision:latest \
    --testsuite ./testsuite \
    --output ./results

# Model 2
uv run vlm-geometry-bench \
    --model minicpm-v:latest \
    --testsuite ./testsuite \
    --output ./results

# Model 3 (OpenRouter)
uv run vlm-geometry-bench \
    --backend openrouter \
    --model openai/gpt-4o \
    --testsuite ./testsuite \
    --output ./results

# Compare results
cat results/*/leaderboard.csv
```

### Focus on Specific Tasks

#### Counting Only

```bash
uv run vlm-geometry-bench \
    --tasks COUNT \
    --tolerance 5 \
    --testsuite ./testsuite
```

The `--tolerance` flag sets the "within N" threshold for counting accuracy.

#### Pattern Recognition Only

```bash
uv run vlm-geometry-bench \
    --tasks PATTERN \
    --classes USSS,HSFR,HSRP \
    --testsuite ./testsuite
```

#### Defect Detection

```bash
uv run vlm-geometry-bench \
    --tasks DEFECT \
    --classes HSDN \
    --testsuite ./testsuite
```

### Few-Shot Evaluation

Include example images in prompts:

```bash
uv run vlm-geometry-bench \
    --shots 3 \
    --tasks COUNT \
    --testsuite ./testsuite
```

Supported values: 0 (zero-shot), 3, or 5.

## Working with Results

### Output Directory Structure

```
results/llama3.2-vision_latest_20260127_135028/
├── metrics.json        # Full evaluation metrics (JSON)
├── leaderboard.csv     # One-row summary for leaderboards
└── raw_responses.json  # Per-sample VLM responses
```

### Analyzing Results

```python
import json

# Load metrics
with open("results/.../metrics.json") as f:
    metrics = json.load(f)

# Overall COUNT performance
print(metrics["results_by_task"]["COUNT"]["exact_match_rate"])

# Per-class breakdown
for cls, data in metrics["results_by_class"].items():
    print(f"{cls}: COUNT={data['COUNT']['exact_match_rate']:.1f}%")
```

### Building Leaderboards

Combine multiple evaluation results:

```bash
# Combine all leaderboard.csv files
head -1 results/*/leaderboard.csv | head -1 > combined.csv
tail -q -n1 results/*/leaderboard.csv >> combined.csv
```

## Advanced Configuration

### Custom Timeout

For slow models or large images:

```bash
uv run vlm-geometry-bench \
    --timeout 300 \
    --testsuite ./testsuite
```

### Custom Base URL

For remote Ollama or custom endpoints:

```bash
uv run vlm-geometry-bench \
    --backend ollama \
    --base-url http://192.168.1.100:11434 \
    --testsuite ./testsuite
```

### Verbose Output

Enable debug logging:

```bash
uv run vlm-geometry-bench \
    --verbose \
    --testsuite ./testsuite
```

## Troubleshooting

### Ollama Connection Failed

```
Error: Cannot connect to Ollama at http://localhost:11434
```

Solutions:
1. Ensure Ollama is running: `ollama serve`
2. Check the correct port is exposed
3. Try `--base-url http://127.0.0.1:11434`

### Model Not Found

```
Error: Model 'llava:7b' not found
```

Solutions:
1. Pull the model: `ollama pull llava:7b`
2. List available models: `ollama list`
3. Check spelling of model name

### API Key Required

```
Error: API key required for OpenRouter backend
```

Solutions:
1. Set environment variable: `export OPENROUTER_API_KEY="..."`
2. Pass via CLI: `--api-key "..."`

### Parse Failures

If many responses fail to parse:
1. Check `raw_responses.json` for actual model output
2. Model may not follow expected format
3. Consider adjusting prompts in `prompts.py`

### Timeout Errors

```
Warning: Timeout (attempt 1/3)
```

Solutions:
1. Increase timeout: `--timeout 300`
2. Use a faster model
3. Reduce image size if possible

## Performance Tips

1. **Start Small**: Test with `--samples 5` first
2. **Use Local Models**: Ollama is faster and cheaper than API calls
3. **Batch by Task**: Run one task at a time for easier debugging
4. **Monitor Progress**: Use `-v` flag to see per-sample results

## Programmatic Usage

```python
import asyncio
from vlm_geometry_bench.config import EvaluationConfig
from vlm_geometry_bench.evaluator import Evaluator

async def run_evaluation():
    config = EvaluationConfig(
        backend="ollama",
        model="llama3.2-vision:latest",
        testsuite_path="d:/python/imagegen/testsuite",
        tasks=["COUNT", "PATTERN"],
        image_classes=["USSS", "HSFR"],
    )

    evaluator = Evaluator(config)
    results = await evaluator.run()

    print(f"COUNT exact match: {results['COUNT']['exact_match_rate']:.1f}%")

asyncio.run(run_evaluation())
```

## Next Steps

- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- Check the test suite in `tests/` for API examples
- Explore `prompts.py` to customize evaluation prompts
