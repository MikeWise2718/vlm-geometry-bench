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

from .config import EvaluationConfig, VALID_IMAGE_CLASSES, VALID_TASKS
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


@click.command()
@click.option(
    "--backend",
    type=click.Choice(["ollama", "openrouter"]),
    default="ollama",
    help="API backend to use",
)
@click.option(
    "--base-url",
    default=None,
    help="API base URL (default: auto-detected based on backend)",
)
@click.option(
    "--api-key",
    default=None,
    envvar="OPENROUTER_API_KEY",
    help="API key (required for OpenRouter, uses OPENROUTER_API_KEY env var)",
)
@click.option(
    "--model",
    default="llava:7b",
    help="Model name (e.g., llava:7b for Ollama, openai/gpt-4o for OpenRouter)",
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
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
    testsuite: str,
    classes: str,
    tasks: str,
    shots: str,
    samples: int,
    output: str,
    timeout: int,
    tolerance: int,
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

      # Quick test with limited samples
      vlm-geometry-bench --samples 5 --classes CTRL,USSS --tasks COUNT
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Parse list options
    task_list = [t.strip().upper() for t in tasks.split(",")]
    class_list = None
    if classes:
        class_list = [c.strip().upper() for c in classes.split(",")]

    num_shots = int(shots)

    # Validate API key for OpenRouter
    if backend == "openrouter" and not api_key:
        raise click.ClickException(
            "API key required for OpenRouter. Set --api-key or OPENROUTER_API_KEY env var."
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

    # Set default base URL based on backend
    if base_url is None:
        if backend == "ollama":
            base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        else:
            base_url = "https://openrouter.ai/api/v1"

    # Create config
    config = EvaluationConfig(
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        model_name=model,
        testsuite_path=testsuite,
        image_classes=class_list or list(VALID_IMAGE_CLASSES),
        tasks=task_list,
        num_shots=num_shots,
        num_samples=samples,
        output_dir=output,
        timeout_seconds=timeout,
        count_tolerance=tolerance,
    )

    # Display configuration
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_row("Backend", config.backend)
    config_table.add_row("Base URL", config.base_url)
    config_table.add_row("Model", f"[bold]{config.model_name}[/bold]")
    config_table.add_row("Test Suite", config.testsuite_path)
    config_table.add_row("Classes", ", ".join(sorted(config.image_classes)))
    config_table.add_row("Tasks", ", ".join(config.tasks))
    config_table.add_row("Shots", str(config.num_shots))
    config_table.add_row("Samples", str(config.num_samples) if config.num_samples else "all")
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
