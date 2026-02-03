#!/usr/bin/env python3
"""CLI entry point for VLM Geometry Bench."""

import asyncio
import logging
import os
import sys

import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .config import (
    EvaluationConfig,
    ModelSpec,
    TraceabilityConfig,
    VALID_IMAGE_CLASSES,
    VALID_TASKS,
    VALID_BACKENDS,
)
from .evaluator import GeometryBenchEvaluator
from .reporter import report_results

console = Console()


def setup_logging(verbose: bool):
    """Configure logging with rich colorized output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def parse_models_string(models_str: str, default_backend: str) -> list[ModelSpec]:
    """Parse comma-separated models string into ModelSpec list.

    Supports formats:
    - "llava:7b" - uses default backend
    - "llava:7b,llava:13b" - multiple models with default backend
    - "ollama:llava:7b,anthropic:claude-sonnet-4" - explicit backend per model

    Args:
        models_str: Comma-separated model specifications
        default_backend: Default backend to use if not specified

    Returns:
        List of ModelSpec objects
    """
    specs = []
    for part in models_str.split(","):
        part = part.strip()
        if not part:
            continue

        # Check if backend is specified (format: backend:model)
        if ":" in part:
            # Could be "ollama:llava:7b" or just "llava:7b"
            # Check if first part is a valid backend
            first_colon = part.index(":")
            potential_backend = part[:first_colon]

            if potential_backend in VALID_BACKENDS:
                backend = potential_backend
                model_name = part[first_colon + 1:]
            else:
                # Not a backend prefix, use default
                backend = default_backend
                model_name = part
        else:
            backend = default_backend
            model_name = part

        specs.append(ModelSpec(model_name=model_name, backend=backend))

    return specs


@click.command()
@click.option(
    "--backend",
    type=click.Choice(["ollama", "openrouter", "anthropic"]),
    default="ollama",
    help="API backend to use (default for --models if not specified per-model)",
)
@click.option(
    "--base-url",
    default=None,
    help="API base URL (default: auto-detected based on backend)",
)
@click.option(
    "--api-key",
    default=None,
    help="API key (required for OpenRouter/Anthropic, uses OPENROUTER_API_KEY or ANTHROPIC_API_KEY env vars)",
)
@click.option(
    "--model",
    default=None,
    help="Single model name (e.g., llava:7b for Ollama, openai/gpt-4o for OpenRouter). Use --models for multiple.",
)
@click.option(
    "--models",
    "models_str",
    default=None,
    help="Comma-separated list of models to evaluate. Format: 'model1,model2' or 'backend:model' for mixed backends.",
)
@click.option(
    "--run-name",
    default=None,
    help="Name for this test run (used in artifact folder name)",
)
@click.option(
    "--comment",
    default=None,
    help="Comment describing the purpose of this test run",
)
@click.option(
    "--testsuite",
    default="./testsuite",
    help="Path to imagegen test suite directory",
)
@click.option(
    "--classes",
    default=None,
    help="Comma-separated image classes to evaluate (default: all). Options: CTRL,USSS,USDS,HSFR,HSRP,HSDN",
)
@click.option(
    "--tasks",
    default="COUNT,PATTERN",
    help="Comma-separated tasks to run. Options: COUNT,LOCATE,PATTERN,SIZE,DEFECT",
)
@click.option(
    "--shots",
    type=click.Choice(["0", "3", "5"]),
    default="0",
    help="Number of few-shot examples",
)
@click.option(
    "--samples",
    type=int,
    default=None,
    help="Limit number of samples per class (default: all)",
)
@click.option(
    "--output",
    default="./results",
    help="Output directory for results",
)
@click.option(
    "--timeout",
    type=int,
    default=120,
    help="Request timeout in seconds",
)
@click.option(
    "--tolerance",
    type=int,
    default=2,
    help="Count tolerance for 'within N' metric (default: 2)",
)
@click.option(
    "--traceability",
    is_flag=True,
    default=False,
    help="Enable traceability output (annotated images, HTML reports, conversation logs)",
)
@click.option(
    "--results-dir",
    default=None,
    help="Directory for traceability artifacts (default: same as --output)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
    models_str: str,
    run_name: str,
    comment: str,
    testsuite: str,
    classes: str,
    tasks: str,
    shots: str,
    samples: int,
    output: str,
    timeout: int,
    tolerance: int,
    traceability: bool,
    results_dir: str,
    verbose: bool,
):
    """
    Run VLM Geometry Bench evaluation against vision-language models.

    \b
    This evaluates VLMs on their ability to identify, count, and locate
    simple geometric shapes (spots/circles) in synthetic test images.

    \b
    Tasks:
      COUNT   - Count the number of spots in the image
      PATTERN - Classify spot arrangement (RANDOM, HEXAGONAL, EMPTY, SINGLE)
      LOCATE  - List coordinates of all spots
      SIZE    - Estimate spot diameter in micrometers
      DEFECT  - Detect missing/extra spots in hexagonal patterns

    \b
    Image Classes:
      CTRL  - Control images (empty, single spot)
      USSS  - Uniform Spots Same Size (random placement)
      USDS  - Uniform Spots Different Sizes (random placement)
      HSFR  - Hex Spots Fixed Rigid (perfect hexagonal grid)
      HSRP  - Hex Spots Random Perturbation (perturbed hex grid)
      HSDN  - Hex Spots Defects + Noise (hex grid with defects)

    \b
    Examples:
      # Local Ollama evaluation
      vlm-geometry-bench --backend ollama --model llava:7b --testsuite ./testsuite

      # OpenRouter with GPT-4o
      vlm-geometry-bench --backend openrouter --model openai/gpt-4o --tasks COUNT,PATTERN

      # Anthropic Claude Sonnet
      vlm-geometry-bench --backend anthropic --model claude-sonnet-4-20250514 --tasks COUNT,PATTERN

      # Quick test with limited samples
      vlm-geometry-bench --samples 5 --classes CTRL,USSS --tasks COUNT

      # Multi-model comparison
      vlm-geometry-bench --models llava:7b,llava:13b --run-name "llava-comparison" --tasks LOCATE

      # With traceability output
      vlm-geometry-bench --model llava:7b --traceability --results-dir ./results

      # Mixed backends comparison
      vlm-geometry-bench --models "ollama:llava:7b,anthropic:claude-sonnet-4-20250514" \\
          --run-name "comparison" --comment "Comparing local vs cloud models" --traceability
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Parse list options
    task_list = [t.strip().upper() for t in tasks.split(",")]
    class_list = None
    if classes:
        class_list = [c.strip().upper() for c in classes.split(",")]

    num_shots = int(shots)

    # Parse models
    model_specs = None
    if models_str:
        # Multi-model mode
        model_specs = parse_models_string(models_str, backend)
        if not model_specs:
            raise click.ClickException("No valid models specified in --models")
    elif model:
        # Single model specified via --model
        # Will use legacy single-model config
        pass
    else:
        # Default model
        model = "llava:7b"

    # Validate API keys for cloud backends
    if model_specs:
        for spec in model_specs:
            if spec.backend == "openrouter" and not spec.api_key:
                if not os.environ.get("OPENROUTER_API_KEY"):
                    raise click.ClickException(
                        f"API key required for OpenRouter model {spec.model_name}. "
                        "Set OPENROUTER_API_KEY env var."
                    )
            elif spec.backend == "anthropic" and not spec.api_key:
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise click.ClickException(
                        f"API key required for Anthropic model {spec.model_name}. "
                        "Set ANTHROPIC_API_KEY env var."
                    )
    else:
        # Single model validation
        if backend == "openrouter":
            if not api_key:
                api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise click.ClickException(
                    "API key required for OpenRouter. Set --api-key or OPENROUTER_API_KEY env var."
                )
        elif backend == "anthropic":
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise click.ClickException(
                    "API key required for Anthropic. Set --api-key or ANTHROPIC_API_KEY env var."
                )

    # Validate classes
    if class_list:
        for cls in class_list:
            if cls not in VALID_IMAGE_CLASSES:
                raise click.ClickException(
                    f"Invalid class: {cls}. Must be one of: {', '.join(sorted(VALID_IMAGE_CLASSES))}"
                )

    # Validate tasks
    for task in task_list:
        if task not in VALID_TASKS:
            raise click.ClickException(
                f"Invalid task: {task}. Must be one of: {', '.join(sorted(VALID_TASKS))}"
            )

    # Set default base URL based on backend (for single model)
    if base_url is None and not model_specs:
        if backend == "ollama":
            base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        elif backend == "anthropic":
            base_url = "https://api.anthropic.com"
        else:  # openrouter
            base_url = "https://openrouter.ai/api/v1"

    # Setup traceability config
    trace_config = TraceabilityConfig(
        enabled=traceability,
        results_dir=results_dir or output,
    )

    # Create config
    config = EvaluationConfig(
        backend=backend,
        base_url=base_url or "http://localhost:11434",
        api_key=api_key,
        model_name=model or "llava:7b",
        models=model_specs,
        run_name=run_name,
        comment=comment,
        testsuite_path=testsuite,
        image_classes=class_list or list(VALID_IMAGE_CLASSES),
        tasks=task_list,
        num_shots=num_shots,
        num_samples=samples,
        output_dir=output,
        timeout_seconds=timeout,
        count_tolerance=tolerance,
        traceability=trace_config,
    )

    # Display configuration
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="white")

    if config.is_multi_model:
        model_names = config.get_all_model_names()
        config_table.add_row("Models", f"[bold]{', '.join(model_names)}[/bold]")
        config_table.add_row("Backends", ", ".join(config.get_all_backends()))
    else:
        config_table.add_row("Backend", config.backend)
        config_table.add_row("Base URL", config.base_url)
        config_table.add_row("Model", f"[bold]{config.model_name}[/bold]")

    config_table.add_row("Run Name", config.run_name or "(auto)")
    if config.comment:
        config_table.add_row("Comment", f'"{config.comment}"')
    config_table.add_row("Test Suite", config.testsuite_path)
    config_table.add_row("Classes", ", ".join(sorted(config.image_classes)))
    config_table.add_row("Tasks", ", ".join(config.tasks))
    config_table.add_row("Shots", str(config.num_shots))
    config_table.add_row("Samples", str(config.num_samples) if config.num_samples else "all")
    config_table.add_row("Traceability", "enabled" if config.traceability.enabled else "disabled")

    console.print(Panel(config_table, title="[bold cyan]Configuration", border_style="cyan"))
    console.print()

    # Run evaluation
    evaluator = GeometryBenchEvaluator(config)

    try:
        results = asyncio.run(evaluator.run_evaluation())

        # Save and display results
        output_dir = report_results(results, output)

        console.print(f"\n[bold green]Evaluation complete![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Evaluation failed:[/bold red] {e}")
        logger.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
