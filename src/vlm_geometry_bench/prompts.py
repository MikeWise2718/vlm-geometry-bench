"""Task-specific prompts for VLM geometry evaluation.

This module defines prompts for evaluating VLMs on geometric shape identification tasks.
Each task has a base prompt template and optional few-shot examples.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class Task(Enum):
    """Evaluation task types."""
    COUNT = "COUNT"
    LOCATE = "LOCATE"
    PATTERN = "PATTERN"
    SIZE = "SIZE"
    DEFECT = "DEFECT"


# Valid pattern classifications for PATTERN task
VALID_PATTERNS = {"RANDOM", "HEXAGONAL", "GRID", "EMPTY", "SINGLE"}


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

TASK_PROMPTS: Dict[str, str] = {
    "COUNT": """Examine this image and count the circular spots or dots.

Instructions:
- Count ALL visible circular spots/dots in the image
- Include spots of any size
- Do not count partial spots at edges

Respond with ONLY a single integer number.

How many spots are in this image?""",

    "LOCATE": """Examine this image and identify the location of each circular spot.

Image dimensions: {width} x {height} pixels
Coordinate system: (0,0) is the top-left corner, x increases rightward, y increases downward.

Instructions:
- Report the center coordinates of each spot
- List one coordinate pair per line
- Format: x, y

Example output format:
125, 340
280, 150
450, 420

List all spot coordinates:""",

    "PATTERN": """Examine this image and classify the arrangement pattern of the spots.

Choose ONE of these pattern types:
- EMPTY: No spots visible in the image
- SINGLE: Exactly one spot visible
- RANDOM: Multiple spots scattered with no regular arrangement
- HEXAGONAL: Spots arranged in a hexagonal/honeycomb grid pattern
- GRID: Spots arranged in a rectangular grid pattern

Respond with ONLY one word from the list above.

What is the pattern type?""",

    "SIZE": """Examine this image and estimate the diameter of the circular spots.

Image scale: {scale_inverse} micrometers per pixel

Instructions:
- Estimate the typical spot diameter
- If spots vary in size, report the average
- Report the value in micrometers

Respond with ONLY a number (the diameter in micrometers).

What is the spot diameter?""",

    "DEFECT": """Examine this image showing spots in a hexagonal pattern.

Look for these types of defects:
1. MISSING: Gaps where spots should be but are absent
2. NOISE: Extra spots that don't belong to the regular pattern
3. DISPLACEMENT: Spots significantly shifted from their expected positions

Respond in exactly this format:
DEFECTS_FOUND: YES or NO
MISSING_COUNT: [number]
NOISE_COUNT: [number]
CONFIDENCE: HIGH, MEDIUM, or LOW

Analyze the pattern for defects:""",
}


# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

@dataclass
class FewShotExample:
    """A few-shot example for a task."""
    description: str  # Text description (used when no image available)
    answer: str       # Expected answer
    image_class: Optional[str] = None  # Image class for selecting real examples
    sample_id: Optional[str] = None    # Specific sample ID if using real images


# Few-shot examples for each task
# These are text-based descriptions; real image examples can be loaded from the test suite
FEW_SHOT_EXAMPLES: Dict[str, List[FewShotExample]] = {
    "COUNT": [
        FewShotExample(
            description="A black background with 20 small white circular spots scattered randomly",
            answer="20",
            image_class="USSS",
        ),
        FewShotExample(
            description="A black background with 50 white spots arranged in a hexagonal pattern",
            answer="50",
            image_class="HSFR",
        ),
        FewShotExample(
            description="A completely black image with no visible spots",
            answer="0",
            image_class="CTRL",
        ),
        FewShotExample(
            description="A black background with exactly one white spot in the center",
            answer="1",
            image_class="CTRL",
        ),
        FewShotExample(
            description="A black background with 100 small white spots densely distributed",
            answer="100",
            image_class="USSS",
        ),
    ],

    "PATTERN": [
        FewShotExample(
            description="White spots scattered randomly across a black background with no discernible pattern",
            answer="RANDOM",
            image_class="USSS",
        ),
        FewShotExample(
            description="White spots arranged in neat rows forming a honeycomb-like hexagonal grid",
            answer="HEXAGONAL",
            image_class="HSFR",
        ),
        FewShotExample(
            description="A solid black image with no spots visible",
            answer="EMPTY",
            image_class="CTRL",
        ),
        FewShotExample(
            description="A single white spot centered on a black background",
            answer="SINGLE",
            image_class="CTRL",
        ),
        FewShotExample(
            description="Spots in a roughly hexagonal arrangement but with slight random displacement",
            answer="HEXAGONAL",
            image_class="HSRP",
        ),
    ],

    "SIZE": [
        FewShotExample(
            description="Small circular spots approximately 2 micrometers in diameter",
            answer="2",
            image_class="USSS",
        ),
        FewShotExample(
            description="Medium circular spots approximately 3 micrometers in diameter",
            answer="3",
            image_class="USSS",
        ),
        FewShotExample(
            description="Larger circular spots approximately 5 micrometers in diameter",
            answer="5",
            image_class="USSS",
        ),
    ],

    "LOCATE": [
        FewShotExample(
            description="Three spots located at roughly (100, 200), (350, 150), and (500, 400)",
            answer="100, 200\n350, 150\n500, 400",
            image_class="USSS",
        ),
    ],

    "DEFECT": [
        FewShotExample(
            description="A perfect hexagonal grid pattern with no missing or extra spots",
            answer="DEFECTS_FOUND: NO\nMISSING_COUNT: 0\nNOISE_COUNT: 0\nCONFIDENCE: HIGH",
            image_class="HSFR",
        ),
        FewShotExample(
            description="A hexagonal pattern with several gaps where spots are missing and a few extra noise spots",
            answer="DEFECTS_FOUND: YES\nMISSING_COUNT: 8\nNOISE_COUNT: 5\nCONFIDENCE: MEDIUM",
            image_class="HSDN",
        ),
    ],
}


# =============================================================================
# PROMPT BUILDING FUNCTIONS
# =============================================================================

def get_prompt(task: str, **kwargs) -> str:
    """Get the prompt template for a task, formatted with provided arguments.

    Args:
        task: Task identifier (COUNT, LOCATE, PATTERN, SIZE, DEFECT)
        **kwargs: Format arguments (e.g., width, height, scale_inverse)

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If task is unknown
    """
    task_upper = task.upper()
    if task_upper not in TASK_PROMPTS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {list(TASK_PROMPTS.keys())}")

    template = TASK_PROMPTS[task_upper]

    # Only format if kwargs provided and template has placeholders
    if kwargs:
        try:
            return template.format(**kwargs)
        except KeyError:
            # Template doesn't need these args, return as-is
            pass

    return template


def get_few_shot_examples(task: str, num_shots: int) -> List[FewShotExample]:
    """Get few-shot examples for a task.

    Args:
        task: Task identifier
        num_shots: Number of examples to return (0, 3, or 5)

    Returns:
        List of FewShotExample objects
    """
    if num_shots == 0:
        return []

    task_upper = task.upper()
    examples = FEW_SHOT_EXAMPLES.get(task_upper, [])
    return examples[:num_shots]


def build_few_shot_prompt(
    task: str,
    num_shots: int = 0,
    **kwargs
) -> str:
    """Build a prompt with few-shot examples prepended.

    This creates a text-only few-shot prompt. For image-based few-shot,
    use the VisionClient's few_shot_examples parameter instead.

    Args:
        task: Task identifier
        num_shots: Number of examples to include (0, 3, or 5)
        **kwargs: Format arguments for the main prompt

    Returns:
        Complete prompt with examples prepended
    """
    base_prompt = get_prompt(task, **kwargs)

    if num_shots == 0:
        return base_prompt

    examples = get_few_shot_examples(task, num_shots)
    if not examples:
        return base_prompt

    # Build example text
    lines = ["Here are some examples:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Image: {ex.description}")
        lines.append(f"Answer: {ex.answer}")
        lines.append("")

    lines.append("Now analyze this image:\n")
    lines.append(base_prompt)

    return "\n".join(lines)


def get_prompt_for_sample(
    task: str,
    sample: Any,  # BenchmarkSample, avoid circular import
    num_shots: int = 0,
) -> str:
    """Build a prompt for a specific benchmark sample.

    This is a convenience function that extracts the relevant
    parameters from a BenchmarkSample.

    Args:
        task: Task identifier
        sample: BenchmarkSample object
        num_shots: Number of few-shot examples

    Returns:
        Formatted prompt string
    """
    # Build kwargs based on task requirements
    kwargs = {}

    task_upper = task.upper()
    if task_upper == "LOCATE":
        kwargs["width"] = sample.width
        kwargs["height"] = sample.height
    elif task_upper == "SIZE":
        kwargs["scale_inverse"] = sample.scale_inverse

    return build_few_shot_prompt(task, num_shots, **kwargs)


# =============================================================================
# EXPECTED ANSWER HELPERS
# =============================================================================

def get_expected_pattern(ground_truth: Any) -> str:
    """Determine the expected pattern classification for a sample.

    Args:
        ground_truth: GroundTruth object from a BenchmarkSample

    Returns:
        Expected pattern: EMPTY, SINGLE, RANDOM, or HEXAGONAL
    """
    if ground_truth.spot_count == 0:
        return "EMPTY"
    elif ground_truth.spot_count == 1:
        return "SINGLE"
    elif ground_truth.is_hexagonal:
        return "HEXAGONAL"
    elif ground_truth.is_random:
        return "RANDOM"
    else:
        # Default based on pattern_type string
        pattern = ground_truth.pattern_type.lower()
        if "hex" in pattern:
            return "HEXAGONAL"
        elif pattern in ("uniform_spots", "variable_spots"):
            return "RANDOM"
        elif pattern == "empty":
            return "EMPTY"
        elif pattern == "single_spot":
            return "SINGLE"
        return "RANDOM"  # Default fallback
