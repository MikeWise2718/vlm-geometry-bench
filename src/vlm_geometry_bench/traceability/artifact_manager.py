"""Artifact manager for traceability output.

Handles folder creation, file saving, and size calculation for all artifacts.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def safe_model_name(model: str) -> str:
    """Convert model name to safe folder name.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-20250514', 'llava:7b')

    Returns:
        Safe folder name with special chars replaced
    """
    return model.replace("/", "_").replace(":", "-").replace(" ", "_")


def safe_run_name(name: str) -> str:
    """Convert run name to safe folder name.

    Args:
        name: User-provided run name

    Returns:
        Safe folder name
    """
    # Replace spaces and special chars
    safe = name.replace(" ", "-").replace("/", "_").replace(":", "_")
    # Remove any other problematic chars
    safe = "".join(c for c in safe if c.isalnum() or c in "-_")
    return safe or "run"


class ArtifactManager:
    """Manages artifact storage for a single test run."""

    def __init__(self, results_dir: str, run_name: str, timestamp: Optional[datetime] = None):
        """Initialize artifact manager.

        Args:
            results_dir: Base results directory
            run_name: User-provided run name
            timestamp: Run start timestamp (defaults to now)
        """
        self.results_dir = Path(results_dir)
        self.timestamp = timestamp or datetime.now()
        self.run_name = run_name

        # Create run ID from timestamp and name
        ts_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        safe_name = safe_run_name(run_name)
        self.run_id = f"{ts_str}_{safe_name}"

        # Run folder path
        self.run_dir = self.results_dir / self.run_id

        # Subfolders
        self.images_dir = self.run_dir / "images"
        self.models_dir = self.run_dir / "models"
        self.assets_dir = self.run_dir / "assets"

        # Track copied images to avoid duplication
        self._copied_images: Dict[str, Path] = {}

    def setup_run_structure(self) -> None:
        """Create the folder structure for this run."""
        # Create main folders
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)

    def setup_model_structure(self, model: str) -> Path:
        """Create folder structure for a model within this run.

        Args:
            model: Model name

        Returns:
            Path to model folder
        """
        model_dir = self.models_dir / safe_model_name(model)
        model_dir.mkdir(exist_ok=True)
        (model_dir / "tests").mkdir(exist_ok=True)
        return model_dir

    def get_model_dir(self, model: str) -> Path:
        """Get the folder path for a model.

        Args:
            model: Model name

        Returns:
            Path to model folder
        """
        return self.models_dir / safe_model_name(model)

    def get_test_dir(self, model: str, sample_id: str, task: str) -> Path:
        """Get the folder path for a specific test.

        Args:
            model: Model name
            sample_id: Sample identifier
            task: Task name

        Returns:
            Path to test folder
        """
        model_dir = self.get_model_dir(model)
        test_id = f"{sample_id}_{task}"
        test_dir = model_dir / "tests" / test_id
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    def copy_original_image(self, source_path: Path, sample_id: str) -> Path:
        """Copy an original image to the shared images folder.

        Only copies if not already copied. Returns path to the shared image.

        Args:
            source_path: Path to original image
            sample_id: Sample identifier

        Returns:
            Path to the image in shared images folder
        """
        if sample_id in self._copied_images:
            return self._copied_images[sample_id]

        dest_path = self.images_dir / f"{sample_id}.png"

        if not dest_path.exists():
            # Copy the image
            shutil.copy2(source_path, dest_path)

        self._copied_images[sample_id] = dest_path
        return dest_path

    def create_image_symlink(self, test_dir: Path, sample_id: str) -> Path:
        """Create a symlink to the shared original image in a test folder.

        Falls back to copying if symlinks are not supported.

        Args:
            test_dir: Test folder path
            sample_id: Sample identifier

        Returns:
            Path to the original.png (symlink or copy)
        """
        link_path = test_dir / "original.png"
        target_path = self.images_dir / f"{sample_id}.png"

        if link_path.exists():
            return link_path

        # Calculate relative path from test dir to images dir
        # test_dir is like: results/run/models/model/tests/sample_task/
        # images_dir is like: results/run/images/
        try:
            # Try to create relative symlink
            rel_target = os.path.relpath(target_path, test_dir)
            link_path.symlink_to(rel_target)
        except (OSError, NotImplementedError):
            # Symlinks not supported, fall back to copy
            if target_path.exists():
                shutil.copy2(target_path, link_path)

        return link_path

    def save_json(self, data: Any, path: Path) -> None:
        """Save data as JSON file.

        Args:
            data: Data to save (dict, list, or Pydantic model)
            path: Destination path
        """
        # Handle Pydantic models
        if hasattr(data, "model_dump"):
            data = data.model_dump(mode="json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def save_image(self, image: Image.Image, path: Path) -> None:
        """Save a PIL Image to file.

        Args:
            image: PIL Image to save
            path: Destination path
        """
        image.save(path, "PNG")

    def save_html(self, content: str, path: Path) -> None:
        """Save HTML content to file.

        Args:
            content: HTML string
            path: Destination path
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def save_css(self, content: str, path: Path) -> None:
        """Save CSS content to file.

        Args:
            content: CSS string
            path: Destination path
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def save_js(self, content: str, path: Path) -> None:
        """Save JavaScript content to file.

        Args:
            content: JavaScript string
            path: Destination path
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def calculate_run_size_mb(self) -> float:
        """Calculate total size of the run folder in MB.

        Returns:
            Size in megabytes
        """
        total_bytes = 0
        for path in self.run_dir.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size

        return total_bytes / (1024 * 1024)

    def get_all_test_dirs(self, model: str) -> List[Path]:
        """Get all test directories for a model.

        Args:
            model: Model name

        Returns:
            List of test directory paths
        """
        tests_dir = self.get_model_dir(model) / "tests"
        if not tests_dir.exists():
            return []

        return [d for d in tests_dir.iterdir() if d.is_dir()]

    def get_image_path_for_sample(self, sample_id: str) -> Path:
        """Get the shared image path for a sample.

        Args:
            sample_id: Sample identifier

        Returns:
            Path to shared image
        """
        return self.images_dir / f"{sample_id}.png"

    # File path helpers for specific artifacts

    def run_metadata_path(self) -> Path:
        """Get path to run_metadata.json."""
        return self.run_dir / "run_metadata.json"

    def summary_html_path(self) -> Path:
        """Get path to summary.html."""
        return self.run_dir / "summary.html"

    def model_metadata_path(self, model: str) -> Path:
        """Get path to model_metadata.json for a model."""
        return self.get_model_dir(model) / "model_metadata.json"

    def test_metadata_path(self, model: str, sample_id: str, task: str) -> Path:
        """Get path to test_metadata.json for a test."""
        return self.get_test_dir(model, sample_id, task) / "test_metadata.json"

    def conversation_path(self, model: str, sample_id: str, task: str) -> Path:
        """Get path to conversation.json for a test."""
        return self.get_test_dir(model, sample_id, task) / "conversation.json"

    def annotated_image_path(self, model: str, sample_id: str, task: str) -> Path:
        """Get path to annotated.png for a test."""
        return self.get_test_dir(model, sample_id, task) / "annotated.png"

    def test_html_path(self, model: str, sample_id: str, task: str) -> Path:
        """Get path to test.html for a test."""
        return self.get_test_dir(model, sample_id, task) / "test.html"

    def style_css_path(self) -> Path:
        """Get path to style.css in assets folder."""
        return self.assets_dir / "style.css"

    def script_js_path(self) -> Path:
        """Get path to script.js in assets folder."""
        return self.assets_dir / "script.js"

    # Global index paths (at results_dir level)

    def global_index_json_path(self) -> Path:
        """Get path to global index.json."""
        return self.results_dir / "index.json"

    def global_index_html_path(self) -> Path:
        """Get path to global index.html."""
        return self.results_dir / "index.html"

    def global_assets_dir(self) -> Path:
        """Get path to global assets directory."""
        assets = self.results_dir / "assets"
        assets.mkdir(exist_ok=True)
        return assets
