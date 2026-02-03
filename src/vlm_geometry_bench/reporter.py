"""Results reporting and output generation."""

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .evaluator import EvaluationResults
from .metrics import SampleResult


class ResultsReporter:
    """Reporter for evaluation results."""

    def __init__(self, output_dir: str, model_name: str):
        """Initialize reporter.

        Args:
            output_dir: Base directory for results
            model_name: Model name for directory naming
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.console = Console()

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        self.run_dir = self.output_dir / f"{safe_model_name}_{timestamp}"

    def save_all(self, results: EvaluationResults) -> Path:
        """Save all results files and print summary.

        Args:
            results: EvaluationResults from evaluator

        Returns:
            Path to output directory
        """
        # Create output directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save files
        self.save_metrics(results)
        self.save_leaderboard_row(results)
        self.save_raw_responses(results.sample_results)

        # Print summary
        self.print_summary(results)

        return self.run_dir

    def save_metrics(self, results: EvaluationResults) -> Path:
        """Save complete results to JSON file.

        Args:
            results: EvaluationResults from evaluator

        Returns:
            Path to saved metrics.json file
        """
        metrics_path = self.run_dir / "metrics.json"

        # Convert to dict (excluding raw sample results to keep file manageable)
        data = {
            "config": results.config,
            "usage": results.usage,
            "results_by_task": results.results_by_task,
            "results_by_class": results.results_by_class,
            "overall": results.overall,
            "timestamp": results.timestamp,
        }

        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=2)

        return metrics_path

    def save_leaderboard_row(self, results: EvaluationResults) -> Path:
        """Save leaderboard-format CSV row.

        Args:
            results: EvaluationResults from evaluator

        Returns:
            Path to saved CSV file
        """
        csv_path = self.run_dir / "leaderboard_row.csv"

        # Build row data
        row = {
            "model": results.config.get("model", ""),
            "backend": results.config.get("backend", ""),
            "timestamp": results.timestamp,
            "total_samples": results.overall.get("total_samples", 0),
            "success_rate": round(results.overall.get("success_rate", 0), 1),
        }

        # Add task-specific metrics
        for task, metrics in results.results_by_task.items():
            task_lower = task.lower()
            if task == "COUNT":
                row[f"{task_lower}_exact_match"] = round(metrics.get("exact_match_rate", 0), 1)
                row[f"{task_lower}_within_tol"] = round(metrics.get("within_tolerance_rate", 0), 1)
                row[f"{task_lower}_mae"] = round(metrics.get("mean_absolute_error", 0), 2)
            elif task == "PATTERN":
                row[f"{task_lower}_accuracy"] = round(metrics.get("accuracy", 0), 1)
            elif task == "LOCATE":
                row[f"{task_lower}_detection"] = round(metrics.get("mean_detection_rate", 0), 1)
            elif task == "SIZE":
                row[f"{task_lower}_within_tol"] = round(metrics.get("within_tolerance_rate", 0), 1)
            elif task == "DEFECT":
                row[f"{task_lower}_accuracy"] = round(metrics.get("detection_accuracy", 0), 1)

        # Add usage stats
        row["elapsed_seconds"] = results.usage.get("elapsed_seconds", 0)
        row["total_tokens"] = results.usage.get("total_tokens", 0)
        if "estimated_cost_usd" in results.usage:
            row["cost_usd"] = results.usage["estimated_cost_usd"]

        # Write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

        return csv_path

    def save_raw_responses(self, sample_results: List[SampleResult]) -> Path:
        """Save raw per-sample responses.

        Args:
            sample_results: List of SampleResult objects

        Returns:
            Path to saved JSON file
        """
        raw_path = self.run_dir / "raw_responses.json"

        # Convert to list of dicts
        data = []
        for r in sample_results:
            entry = {
                "sample_id": r.sample_id,
                "image_class": r.image_class,
                "task": r.task,
                "success": r.success,
                "parse_error": r.parse_error,
                "metrics": r.metrics,
                "raw_response": r.raw_response,
                "latency_ms": r.latency_ms,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }
            # Add new fields if present
            if r.model:
                entry["model"] = r.model
            if r.prompt:
                entry["prompt"] = r.prompt
            if r.ground_truth:
                entry["ground_truth"] = r.ground_truth
            if r.prediction:
                entry["prediction"] = r.prediction
            data.append(entry)

        with open(raw_path, "w") as f:
            json.dump(data, f, indent=2)

        return raw_path

    def print_summary(self, results: EvaluationResults) -> None:
        """Print formatted summary to console.

        Args:
            results: EvaluationResults from evaluator
        """
        self.console.print()

        # Header
        header = Text("EVALUATION COMPLETE", style="bold green")
        self.console.print(Panel(header, border_style="green"))
        self.console.print()

        # Config info
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column("Key", style="cyan")
        config_table.add_column("Value", style="white")

        # Handle both single and multi-model configs
        if "models" in results.config:
            config_table.add_row("Models", ", ".join(results.config.get("models", [])))
            config_table.add_row("Backends", ", ".join(results.config.get("backends", [])))
        else:
            config_table.add_row("Model", results.config.get("model", ""))
            config_table.add_row("Backend", results.config.get("backend", ""))

        config_table.add_row("Tasks", ", ".join(results.config.get("tasks", [])))
        config_table.add_row("Classes", ", ".join(results.config.get("image_classes", [])))
        self.console.print(config_table)
        self.console.print()

        # Usage stats panel
        self.console.print(self._format_usage_panel(results.usage))
        self.console.print()

        # Results by task
        if results.results_by_task:
            task_table = self._create_task_table(results.results_by_task)
            self.console.print(task_table)
            self.console.print()

        # Results by class
        if results.results_by_class:
            tasks = list(results.results_by_task.keys())
            class_table = self._create_class_table(results.results_by_class, tasks)
            self.console.print(class_table)
            self.console.print()

        # Output path
        self.console.print(f"[dim]Results saved to:[/dim] [bold blue]{self.run_dir}[/bold blue]")
        self.console.print()

    def _create_task_table(self, results_by_task: Dict[str, Dict[str, Any]]) -> Table:
        """Create a table showing results by task.

        Args:
            results_by_task: Dict of task -> metrics

        Returns:
            Rich Table
        """
        table = Table(title="[bold cyan]Results by Task", border_style="blue")
        table.add_column("Task", style="cyan")
        table.add_column("Primary Metric", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("Samples", justify="right", style="dim")

        for task, metrics in results_by_task.items():
            # Determine primary metric for each task
            if task == "COUNT":
                metric_name = "Exact Match"
                value = metrics.get("exact_match_rate", 0)
            elif task == "PATTERN":
                metric_name = "Accuracy"
                value = metrics.get("accuracy", 0)
            elif task == "LOCATE":
                metric_name = "Detection Rate"
                value = metrics.get("mean_detection_rate", 0)
            elif task == "SIZE":
                metric_name = "Within 20%"
                value = metrics.get("within_tolerance_rate", 0)
            elif task == "DEFECT":
                metric_name = "Detection Acc"
                value = metrics.get("detection_accuracy", 0)
            else:
                metric_name = "Unknown"
                value = 0

            value_str = self._colorize_score(value)
            samples = f"{metrics.get('successful_samples', 0)}/{metrics.get('total_samples', 0)}"

            table.add_row(task, metric_name, value_str, samples)

        return table

    def _create_class_table(
        self,
        results_by_class: Dict[str, Dict[str, Dict[str, Any]]],
        tasks: List[str]
    ) -> Table:
        """Create a table showing results by image class.

        Args:
            results_by_class: Nested dict of class -> task -> metrics
            tasks: List of tasks evaluated

        Returns:
            Rich Table
        """
        table = Table(title="[bold cyan]Results by Image Class", border_style="blue")
        table.add_column("Class", style="cyan")

        for task in tasks:
            table.add_column(task, justify="right")

        for cls in sorted(results_by_class.keys()):
            task_results = results_by_class[cls]
            row = [cls]

            for task in tasks:
                if task in task_results:
                    metrics = task_results[task]
                    # Get primary metric
                    if task == "COUNT":
                        value = metrics.get("exact_match_rate", 0)
                    elif task == "PATTERN":
                        value = metrics.get("accuracy", 0)
                    elif task == "LOCATE":
                        value = metrics.get("mean_detection_rate", 0)
                    elif task == "SIZE":
                        value = metrics.get("within_tolerance_rate", 0)
                    elif task == "DEFECT":
                        value = metrics.get("detection_accuracy", 0)
                    else:
                        value = 0

                    row.append(self._colorize_score(value))
                else:
                    row.append("-")

            table.add_row(*row)

        return table

    def _colorize_score(self, value: float) -> str:
        """Colorize a score based on thresholds.

        Args:
            value: Score as percentage (0-100)

        Returns:
            Rich-formatted string
        """
        if value >= 70:
            return f"[green]{value:.1f}%[/green]"
        elif value >= 40:
            return f"[yellow]{value:.1f}%[/yellow]"
        else:
            return f"[red]{value:.1f}%[/red]"

    def _format_usage_panel(self, usage: Dict[str, Any]) -> Panel:
        """Create a panel showing usage statistics.

        Args:
            usage: Usage statistics dictionary

        Returns:
            Rich Panel with formatted stats
        """
        lines = [
            f"Total Requests: {usage.get('total_requests', 0)}",
            f"Failed Requests: {usage.get('failed_requests', 0)}",
            f"Success Rate: {usage.get('success_rate', 0):.1f}%",
            f"Input Tokens: {usage.get('input_tokens', 0):,}",
            f"Output Tokens: {usage.get('output_tokens', 0):,}",
            f"Elapsed Time: {usage.get('elapsed_seconds', 0):.1f}s",
        ]

        if usage.get("estimated_cost_usd"):
            lines.append(f"Estimated Cost: ${usage['estimated_cost_usd']:.4f}")

        return Panel("\n".join(lines), title="[bold cyan]Usage Statistics")


def report_results(results: EvaluationResults, output_dir: str = "./results") -> Path:
    """Convenience function to report results.

    Args:
        results: EvaluationResults from evaluator
        output_dir: Base output directory

    Returns:
        Path to output directory
    """
    # Handle both single and multi-model configs
    if "models" in results.config:
        model_name = "comparison" if len(results.config["models"]) > 1 else results.config["models"][0]
    else:
        model_name = results.config.get("model", "unknown")

    reporter = ResultsReporter(
        output_dir=output_dir,
        model_name=model_name,
    )
    return reporter.save_all(results)
