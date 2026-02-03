"""Image annotation for evaluation results.

Creates annotated images showing ground truth vs predictions with visual overlays.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# Color constants (RGB)
COLOR_GT = (0, 180, 0)  # Green for ground truth
COLOR_PRED = (220, 50, 50)  # Red for predictions
COLOR_MATCH = (50, 100, 220)  # Blue for match lines
COLOR_STATUS_BG = (40, 40, 40)  # Dark gray background
COLOR_STATUS_TEXT = (255, 255, 255)  # White text
COLOR_SUCCESS = (50, 180, 50)  # Green for success
COLOR_FAIL = (220, 50, 50)  # Red for failure

# Annotation parameters
GT_CIRCLE_RADIUS = 12
GT_CIRCLE_WIDTH = 2
PRED_CIRCLE_RADIUS = 6
MATCH_LINE_WIDTH = 1
STATUS_BAR_HEIGHT = 50
FONT_SIZE = 14
MATCH_DISTANCE_THRESHOLD = 0.05  # Normalized distance for matching


def get_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    """Get a font for drawing text.

    Args:
        size: Font size

    Returns:
        PIL font object (falls back to default if no TTF available)
    """
    try:
        # Try common system fonts
        font_names = [
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, size)
            except OSError:
                continue
    except Exception:
        pass

    # Fall back to default bitmap font
    return ImageFont.load_default()


class ImageAnnotator:
    """Annotates images with ground truth and prediction overlays."""

    def __init__(self, match_distance: float = MATCH_DISTANCE_THRESHOLD):
        """Initialize annotator.

        Args:
            match_distance: Normalized distance threshold for matching GT to predictions
        """
        self.match_distance = match_distance
        self.font = get_font(FONT_SIZE)
        self.font_small = get_font(FONT_SIZE - 2)

    def annotate_locate_task(
        self,
        image: Image.Image,
        gt_positions: List[Tuple[float, float]],
        pred_positions: List[Tuple[float, float]],
        sample_id: str,
        model: str,
        metrics: Dict[str, Any],
    ) -> Image.Image:
        """Annotate an image for LOCATE task results.

        Args:
            image: Original PIL Image
            gt_positions: Ground truth positions as (x, y) normalized [0-1]
            pred_positions: Predicted positions as (x, y) normalized [0-1]
            sample_id: Sample identifier for status bar
            model: Model name for status bar
            metrics: Metrics dict containing detection_rate, false_positive_rate, etc.

        Returns:
            Annotated PIL Image with status bar
        """
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Create copy to draw on
        annotated = image.copy()
        width, height = annotated.size

        # Create drawing context
        draw = ImageDraw.Draw(annotated)

        # Compute matches between predictions and ground truth
        matches = self._compute_matches(gt_positions, pred_positions)

        # Draw match lines (blue) first, so they're behind circles
        for pred_idx, gt_idx in matches.items():
            px, py = pred_positions[pred_idx]
            gx, gy = gt_positions[gt_idx]

            px_pixel = int(px * width)
            py_pixel = int(py * height)
            gx_pixel = int(gx * width)
            gy_pixel = int(gy * height)

            draw.line(
                [(px_pixel, py_pixel), (gx_pixel, gy_pixel)],
                fill=COLOR_MATCH,
                width=MATCH_LINE_WIDTH,
            )

        # Draw ground truth circles (green hollow)
        matched_gt = set(matches.values())
        for i, (gx, gy) in enumerate(gt_positions):
            gx_pixel = int(gx * width)
            gy_pixel = int(gy * height)

            # Hollow circle - draw arc
            bbox = [
                gx_pixel - GT_CIRCLE_RADIUS,
                gy_pixel - GT_CIRCLE_RADIUS,
                gx_pixel + GT_CIRCLE_RADIUS,
                gy_pixel + GT_CIRCLE_RADIUS,
            ]
            draw.ellipse(bbox, outline=COLOR_GT, width=GT_CIRCLE_WIDTH)

        # Draw prediction circles (red filled)
        for i, (px, py) in enumerate(pred_positions):
            px_pixel = int(px * width)
            py_pixel = int(py * height)

            # Filled circle
            bbox = [
                px_pixel - PRED_CIRCLE_RADIUS,
                py_pixel - PRED_CIRCLE_RADIUS,
                px_pixel + PRED_CIRCLE_RADIUS,
                py_pixel + PRED_CIRCLE_RADIUS,
            ]
            draw.ellipse(bbox, fill=COLOR_PRED)

        # Add status bar at bottom
        annotated = self._add_status_bar(
            annotated,
            sample_id=sample_id,
            task="LOCATE",
            model=model,
            metrics=metrics,
        )

        return annotated

    def annotate_count_task(
        self,
        image: Image.Image,
        gt_count: int,
        pred_count: Optional[int],
        sample_id: str,
        model: str,
        metrics: Dict[str, Any],
    ) -> Image.Image:
        """Annotate an image for COUNT task results.

        For COUNT, just adds status bar showing counts - no overlays on image.

        Args:
            image: Original PIL Image
            gt_count: Ground truth count
            pred_count: Predicted count
            sample_id: Sample identifier
            model: Model name
            metrics: Metrics dict

        Returns:
            Annotated PIL Image with status bar
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        annotated = image.copy()

        # Add count info to metrics for status bar
        metrics_with_count = metrics.copy()
        metrics_with_count["gt_count"] = gt_count
        metrics_with_count["pred_count"] = pred_count

        return self._add_status_bar(
            annotated,
            sample_id=sample_id,
            task="COUNT",
            model=model,
            metrics=metrics_with_count,
        )

    def annotate_pattern_task(
        self,
        image: Image.Image,
        gt_pattern: str,
        pred_pattern: Optional[str],
        sample_id: str,
        model: str,
        metrics: Dict[str, Any],
    ) -> Image.Image:
        """Annotate an image for PATTERN task results.

        For PATTERN, just adds status bar showing patterns.

        Args:
            image: Original PIL Image
            gt_pattern: Ground truth pattern
            pred_pattern: Predicted pattern
            sample_id: Sample identifier
            model: Model name
            metrics: Metrics dict

        Returns:
            Annotated PIL Image with status bar
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        annotated = image.copy()

        metrics_with_pattern = metrics.copy()
        metrics_with_pattern["gt_pattern"] = gt_pattern
        metrics_with_pattern["pred_pattern"] = pred_pattern

        return self._add_status_bar(
            annotated,
            sample_id=sample_id,
            task="PATTERN",
            model=model,
            metrics=metrics_with_pattern,
        )

    def annotate_size_task(
        self,
        image: Image.Image,
        gt_size: float,
        pred_size: Optional[float],
        sample_id: str,
        model: str,
        metrics: Dict[str, Any],
    ) -> Image.Image:
        """Annotate an image for SIZE task results.

        For SIZE, just adds status bar showing sizes.

        Args:
            image: Original PIL Image
            gt_size: Ground truth size
            pred_size: Predicted size
            sample_id: Sample identifier
            model: Model name
            metrics: Metrics dict

        Returns:
            Annotated PIL Image with status bar
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        annotated = image.copy()

        metrics_with_size = metrics.copy()
        metrics_with_size["gt_size"] = gt_size
        metrics_with_size["pred_size"] = pred_size

        return self._add_status_bar(
            annotated,
            sample_id=sample_id,
            task="SIZE",
            model=model,
            metrics=metrics_with_size,
        )

    def annotate_defect_task(
        self,
        image: Image.Image,
        gt_has_defects: bool,
        pred_has_defects: Optional[bool],
        sample_id: str,
        model: str,
        metrics: Dict[str, Any],
    ) -> Image.Image:
        """Annotate an image for DEFECT task results.

        For DEFECT, just adds status bar.

        Args:
            image: Original PIL Image
            gt_has_defects: Ground truth defect presence
            pred_has_defects: Predicted defect presence
            sample_id: Sample identifier
            model: Model name
            metrics: Metrics dict

        Returns:
            Annotated PIL Image with status bar
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        annotated = image.copy()

        metrics_with_defect = metrics.copy()
        metrics_with_defect["gt_defects"] = gt_has_defects
        metrics_with_defect["pred_defects"] = pred_has_defects

        return self._add_status_bar(
            annotated,
            sample_id=sample_id,
            task="DEFECT",
            model=model,
            metrics=metrics_with_defect,
        )

    def annotate(
        self,
        image: Image.Image,
        task: str,
        sample_id: str,
        model: str,
        metrics: Dict[str, Any],
        ground_truth: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Image.Image:
        """Main entry point for annotating any task type.

        Args:
            image: Original PIL Image
            task: Task type (COUNT, LOCATE, PATTERN, SIZE, DEFECT)
            sample_id: Sample identifier
            model: Model name
            metrics: Metrics dict from evaluation
            ground_truth: Ground truth data
            prediction: Prediction data

        Returns:
            Annotated PIL Image
        """
        if task == "LOCATE":
            gt_positions = ground_truth.get("positions", [])
            pred_positions = prediction.get("positions", [])
            return self.annotate_locate_task(
                image, gt_positions, pred_positions, sample_id, model, metrics
            )
        elif task == "COUNT":
            gt_count = ground_truth.get("spot_count", 0)
            pred_count = prediction.get("count")
            return self.annotate_count_task(
                image, gt_count, pred_count, sample_id, model, metrics
            )
        elif task == "PATTERN":
            gt_pattern = ground_truth.get("pattern", "")
            pred_pattern = prediction.get("pattern")
            return self.annotate_pattern_task(
                image, gt_pattern, pred_pattern, sample_id, model, metrics
            )
        elif task == "SIZE":
            gt_size = ground_truth.get("size", 0)
            pred_size = prediction.get("diameter_um")
            return self.annotate_size_task(
                image, gt_size, pred_size, sample_id, model, metrics
            )
        elif task == "DEFECT":
            gt_defects = ground_truth.get("has_defects", False)
            pred_defects = prediction.get("has_defects")
            return self.annotate_defect_task(
                image, gt_defects, pred_defects, sample_id, model, metrics
            )
        else:
            # Unknown task - just add basic status bar
            return self._add_status_bar(image, sample_id, task, model, metrics)

    def _compute_matches(
        self,
        gt_positions: List[Tuple[float, float]],
        pred_positions: List[Tuple[float, float]],
    ) -> Dict[int, int]:
        """Compute matches between predictions and ground truth.

        Uses nearest-neighbor matching with distance threshold.

        Args:
            gt_positions: Ground truth positions
            pred_positions: Predicted positions

        Returns:
            Dict mapping prediction index to matched ground truth index
        """
        matches = {}
        used_gt = set()

        # For each prediction, find nearest unmatched ground truth
        for pred_idx, (px, py) in enumerate(pred_positions):
            min_dist = float("inf")
            min_gt_idx = -1

            for gt_idx, (gx, gy) in enumerate(gt_positions):
                if gt_idx in used_gt:
                    continue

                dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_gt_idx = gt_idx

            if min_gt_idx >= 0 and min_dist <= self.match_distance:
                matches[pred_idx] = min_gt_idx
                used_gt.add(min_gt_idx)

        return matches

    def _add_status_bar(
        self,
        image: Image.Image,
        sample_id: str,
        task: str,
        model: str,
        metrics: Dict[str, Any],
    ) -> Image.Image:
        """Add a status bar at the bottom of the image.

        Args:
            image: Image to add status bar to
            sample_id: Sample identifier
            task: Task type
            model: Model name (will be abbreviated)
            metrics: Metrics dict

        Returns:
            Image with status bar added
        """
        width, height = image.size

        # Create new image with space for status bar
        new_height = height + STATUS_BAR_HEIGHT
        new_image = Image.new("RGB", (width, new_height), COLOR_STATUS_BG)
        new_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_image)

        # Abbreviate model name
        model_short = self._abbreviate_model(model)

        # Build status text based on task
        status_parts = [sample_id, task, model_short]

        # Add task-specific metrics
        if task == "LOCATE":
            det_rate = metrics.get("detection_rate", 0)
            fp_rate = metrics.get("false_positive_rate", 0)
            status_parts.append(f"Det:{det_rate:.0f}%")
            status_parts.append(f"FP:{fp_rate:.0f}%")
        elif task == "COUNT":
            gt = metrics.get("gt_count", metrics.get("ground_truth", "?"))
            pred = metrics.get("pred_count", metrics.get("predicted", "?"))
            status_parts.append(f"GT:{gt}")
            status_parts.append(f"Pred:{pred}")
        elif task == "PATTERN":
            gt = metrics.get("gt_pattern", metrics.get("ground_truth", "?"))
            pred = metrics.get("pred_pattern", metrics.get("predicted", "?"))
            correct = metrics.get("correct", False)
            status_parts.append(f"GT:{gt}")
            status_parts.append(f"Pred:{pred}")
        elif task == "SIZE":
            gt = metrics.get("gt_size", metrics.get("ground_truth", 0))
            pred = metrics.get("pred_size", metrics.get("predicted", "?"))
            if isinstance(gt, (int, float)):
                status_parts.append(f"GT:{gt:.0f}um")
            if pred is not None and isinstance(pred, (int, float)):
                status_parts.append(f"Pred:{pred:.0f}um")
        elif task == "DEFECT":
            gt = "Y" if metrics.get("gt_defects", metrics.get("ground_truth_has_defects")) else "N"
            pred = metrics.get("pred_defects", metrics.get("predicted_has_defects"))
            pred_str = "Y" if pred else ("N" if pred is False else "?")
            status_parts.append(f"GT:{gt}")
            status_parts.append(f"Pred:{pred_str}")

        # Join with separator
        status_text = " | ".join(status_parts)

        # Calculate text position (centered vertically in status bar)
        text_y = height + (STATUS_BAR_HEIGHT - FONT_SIZE) // 2

        # Draw text
        draw.text((10, text_y), status_text, fill=COLOR_STATUS_TEXT, font=self.font)

        return new_image

    def _abbreviate_model(self, model: str) -> str:
        """Abbreviate model name to fit in status bar.

        Args:
            model: Full model name

        Returns:
            Abbreviated model name (max ~25 chars)
        """
        # Common abbreviations
        abbrevs = {
            "claude-sonnet-4-20250514": "claude-sonnet-4",
            "claude-opus-4-20250514": "claude-opus-4",
            "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
            "claude-3-5-haiku-20241022": "claude-3.5-haiku",
            "openai/gpt-4o": "gpt-4o",
            "openai/gpt-4o-mini": "gpt-4o-mini",
            "anthropic/claude-3.5-sonnet": "claude-3.5-sonnet",
        }

        if model in abbrevs:
            return abbrevs[model]

        # Truncate if too long
        if len(model) > 25:
            return model[:22] + "..."

        return model


def annotate_image(
    image_path: Path,
    task: str,
    sample_id: str,
    model: str,
    metrics: Dict[str, Any],
    ground_truth: Dict[str, Any],
    prediction: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Image.Image:
    """Convenience function to annotate an image file.

    Args:
        image_path: Path to original image
        task: Task type
        sample_id: Sample identifier
        model: Model name
        metrics: Metrics dict
        ground_truth: Ground truth data
        prediction: Prediction data
        output_path: Optional path to save annotated image

    Returns:
        Annotated PIL Image
    """
    image = Image.open(image_path)
    annotator = ImageAnnotator()
    annotated = annotator.annotate(
        image, task, sample_id, model, metrics, ground_truth, prediction
    )

    if output_path:
        annotated.save(output_path, "PNG")

    return annotated
