"""Data loader for imagegen test suite."""

import csv
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

import yaml
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SpotPosition:
    """A single spot position with coordinates and size."""

    x_um: float
    y_um: float
    diameter_um: float


@dataclass
class GroundTruth:
    """Ground truth data extracted from manifest and CSV files."""

    spot_count: int
    positions: List[SpotPosition]
    pattern_type: str  # "empty", "single_spot", "uniform_spots", "variable_spots", "hexagonal", "hexagonal_perturbed", "hexagonal_defects"
    image_class: str  # CTRL, USSS, USDS, HSFR, HSRP, HSDN
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_defects(self) -> bool:
        """Check if this is a defect pattern."""
        return self.pattern_type == "hexagonal_defects"

    @property
    def is_hexagonal(self) -> bool:
        """Check if this is any hexagonal pattern."""
        return self.pattern_type in ("hexagonal", "hexagonal_perturbed", "hexagonal_defects")

    @property
    def is_random(self) -> bool:
        """Check if this is a random pattern."""
        return self.pattern_type in ("uniform_spots", "variable_spots")


@dataclass
class BenchmarkSample:
    """A single benchmark sample with image and ground truth."""

    sample_id: str
    image: Image.Image
    image_path: Path
    ground_truth: GroundTruth
    filename: str
    image_class: str

    # Image metadata
    width: int = 700
    height: int = 510
    scale_inverse: float = 0.14  # micrometers per pixel


class SuiteLoader:
    """Loads the imagegen test suite for evaluation."""

    def __init__(self, testsuite_path: str):
        """Initialize loader with path to test suite directory.

        Args:
            testsuite_path: Path to directory containing manifest.yaml and images
        """
        self.testsuite_path = Path(testsuite_path)
        self._manifest: Optional[Dict[str, Any]] = None

    @property
    def manifest(self) -> Dict[str, Any]:
        """Lazy-load and cache the manifest."""
        if self._manifest is None:
            self._manifest = self.load_manifest()
        return self._manifest

    def load_manifest(self) -> Dict[str, Any]:
        """Load and parse the manifest.yaml file.

        Returns:
            Parsed manifest dictionary

        Raises:
            FileNotFoundError: If manifest.yaml not found
            yaml.YAMLError: If manifest is invalid YAML
        """
        manifest_path = self.testsuite_path / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)

        logger.info(
            f"Loaded manifest: {manifest.get('total_images', 0)} images, "
            f"version {manifest.get('version', 'unknown')}"
        )
        return manifest

    def load_samples(
        self,
        image_classes: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[BenchmarkSample]:
        """Load benchmark samples from the test suite.

        Args:
            image_classes: Filter to specific classes (None = all)
            limit: Maximum number of samples to load (None = all)

        Returns:
            List of BenchmarkSample objects
        """
        samples = []
        manifest = self.manifest

        # Get global metadata
        width = manifest.get("image_width", 700)
        height = manifest.get("image_height", 510)
        scale_inverse = manifest.get("scale_inverse", 0.14)

        for image_info in manifest.get("images", []):
            # Filter by class if specified
            class_name = image_info.get("class_name", "")
            if image_classes and class_name not in image_classes:
                continue

            # Load the sample
            try:
                sample = self._load_single_sample(
                    image_info, width, height, scale_inverse
                )
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load {image_info.get('filename')}: {e}")
                continue

            # Check limit
            if limit and len(samples) >= limit:
                break

        logger.info(f"Loaded {len(samples)} samples")
        return samples

    def _load_single_sample(
        self,
        image_info: Dict[str, Any],
        width: int,
        height: int,
        scale_inverse: float,
    ) -> BenchmarkSample:
        """Load a single sample from manifest entry.

        Args:
            image_info: Entry from manifest images list
            width: Image width in pixels
            height: Image height in pixels
            scale_inverse: Scale in micrometers per pixel

        Returns:
            BenchmarkSample object
        """
        filename = image_info["filename"]
        image_path = self.testsuite_path / filename

        # Load image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path)

        # Load spot positions from CSV if available
        positions = []
        csv_file = image_info.get("csv_file")
        if csv_file:
            csv_path = self.testsuite_path / csv_file
            if csv_path.exists():
                positions = self._load_positions_from_csv(csv_path)

        # Build ground truth
        ground_truth = GroundTruth(
            spot_count=image_info.get("spot_count", len(positions)),
            positions=positions,
            pattern_type=image_info.get("pattern", "unknown"),
            image_class=image_info.get("class_name", ""),
            metadata={
                k: v for k, v in image_info.items()
                if k not in ("filename", "class_name", "pattern", "spot_count", "csv_file")
            },
        )

        # Create sample
        sample_id = Path(filename).stem
        return BenchmarkSample(
            sample_id=sample_id,
            image=image,
            image_path=image_path,
            ground_truth=ground_truth,
            filename=filename,
            image_class=image_info.get("class_name", ""),
            width=width,
            height=height,
            scale_inverse=scale_inverse,
        )

    def _load_positions_from_csv(self, csv_path: Path) -> List[SpotPosition]:
        """Load spot positions from a CSV file.

        Args:
            csv_path: Path to CSV file with X_um, Y_um, Diameter_um columns

        Returns:
            List of SpotPosition objects
        """
        positions = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pos = SpotPosition(
                        x_um=float(row["X_um"]),
                        y_um=float(row["Y_um"]),
                        diameter_um=float(row["Diameter_um"]),
                    )
                    positions.append(pos)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid row in {csv_path}: {e}")
                    continue
        return positions

    def get_class_summary(self) -> Dict[str, int]:
        """Get count of images per class.

        Returns:
            Dictionary mapping class name to image count
        """
        summary = {}
        for image_info in self.manifest.get("images", []):
            class_name = image_info.get("class_name", "unknown")
            summary[class_name] = summary.get(class_name, 0) + 1
        return summary

    def get_pattern_for_class(self, image_class: str) -> str:
        """Get the expected pattern type for evaluation based on class.

        This maps the detailed pattern types to broader categories used
        for pattern classification evaluation.

        Args:
            image_class: Image class (CTRL, USSS, etc.)

        Returns:
            Pattern category: "EMPTY", "SINGLE", "RANDOM", or "HEXAGONAL"
        """
        # Map classes to evaluation pattern categories
        pattern_map = {
            "CTRL": "EMPTY",  # Will be EMPTY or SINGLE based on spot_count
            "USSS": "RANDOM",
            "USDS": "RANDOM",
            "HSFR": "HEXAGONAL",
            "HSRP": "HEXAGONAL",
            "HSDN": "HEXAGONAL",
        }
        return pattern_map.get(image_class, "UNKNOWN")
