"""
Pipeline stages for makroInspect.

Each stage module provides a process() function with signature:
    process(paths: ProjectPaths, cfg: dict, registry: pd.DataFrame) -> pd.DataFrame

Stages run in order: segment → crop → embed → bank → heatmap → refine

Example:
    >>> from stages import segment, crop
    >>> registry = segment.process(paths, cfg, registry)
    >>> registry = crop.process(paths, cfg, registry)
"""

from stages import segment, crop, embed, bank, heatmap, refine
from stages.segment import SegmentationError

__all__ = [
    "segment",
    "crop",
    "embed",
    "bank",
    "heatmap",
    "refine",
    "SegmentationError",
]
