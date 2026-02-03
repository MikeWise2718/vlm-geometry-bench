"""Traceability module for VLM Geometry Bench.

Provides comprehensive artifact storage, image annotation, and HTML report generation
for evaluation runs with support for multi-model comparisons.
"""

from .schemas import (
    RunIndexEntry,
    RunIndex,
    ModelRunInfo,
    RunMetadata,
    ModelMetadata,
    SampleTestResult,
    ConversationTurn,
    ConversationHistory,
)
from .artifact_manager import ArtifactManager
from .index_manager import IndexManager
from .image_annotator import ImageAnnotator
from .html_generator import HTMLGenerator

__all__ = [
    # Schemas
    "RunIndexEntry",
    "RunIndex",
    "ModelRunInfo",
    "RunMetadata",
    "ModelMetadata",
    "SampleTestResult",
    "ConversationTurn",
    "ConversationHistory",
    # Managers
    "ArtifactManager",
    "IndexManager",
    # Generators
    "ImageAnnotator",
    "HTMLGenerator",
]
