# makroInspect

Industrial anomaly detection using SAM3 and DINOv2 for pixel-level defect localization.

makroInspect combines instance segmentation (SAM3) with vision transformer embeddings (DINOv2) to detect and localize manufacturing defects at the pixel level. The system uses a memory bank approach to identify anomalies by comparing test samples against learned representations of normal instances.

## Architecture

```
                              makroInspect Pipeline
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ SEGMENT  │───>│   CROP   │───>│  EMBED   │───>│   BANK   │             │
│   │  (SAM3)  │    │  (PCA)   │    │ (DINOv2) │    │(K-center)│             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│        │               │               │               │                    │
│        v               v               v               v                    │
│   Instance        Aligned         Feature         Memory                   │
│    Masks          Crops         Embeddings         Bank                    │
│                                                      │                      │
│                                      ┌───────────────┘                      │
│                                      v                                      │
│                              ┌──────────────┐    ┌──────────┐              │
│                              │   HEATMAP    │───>│  REFINE  │              │
│                              │  (Distance)  │    │(Gaussian)│              │
│                              └──────────────┘    └──────────┘              │
│                                      │                │                     │
│                                      v                v                     │
│                                 Distance        Refined                    │
│                                 Heatmaps        Heatmaps                   │
│                                                      │                      │
│                                      ┌───────────────┘                      │
│                                      v                                      │
│                              ┌──────────────┐                               │
│                              │    SCORE     │                               │
│                              │   (AUROC)    │                               │
│                              └──────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Dataset Support**: VisA, MVTec AD, MVTec LOCO, MVTec AD2
- **Instance-Aware Detection**: SAM3 segments individual objects for per-instance analysis
- **PCA Alignment**: Consistent object orientation across samples via principal component analysis
- **Smart Caching**: Config-based invalidation only reruns affected pipeline stages
- **Memory Efficient**: K-center greedy selection for memory bank with configurable size limits
- **Pixel-Level Metrics**: AUROC for both image-level classification and pixel-level localization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/makroInspect.git
cd makroInspect

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

**Requirements:**
- Python 3.12+
- CUDA 12.4+ (for GPU acceleration)
- 8GB+ VRAM recommended

## Quick Start

### CLI Usage

```bash
# 1. Prepare a dataset (downloads and initializes project)
uv run prepare.py visa capsules

# 2. Run the processing pipeline
uv run process.py capsules

# 3. Score the results
uv run score.py capsules
```

### Python API

```python
from prepare import prepare_project
from process import process_project
from score import score_project

# Initialize project from VisA dataset
paths = prepare_project("visa", "capsules")

# Run the full pipeline
result = process_project("capsules")
print(f"Processed {result['instances']} instances in {result['total_time']:.1f}s")

# Evaluate performance
metrics = score_project("capsules")
print(f"Image AUROC: {metrics['auroc_image']:.4f}")
print(f"Pixel AUROC: {metrics['auroc_pixel']:.4f}")
```

## Project Structure

```
makroInspect/
├── prepare.py           # Dataset initialization
├── process.py           # Pipeline orchestration
├── score.py             # Evaluation and metrics
│
├── core/                # Core infrastructure
│   ├── paths.py         # Centralized path management
│   ├── registry.py      # Instance registry schemas
│   └── invalidation.py  # Config change tracking
│
├── stages/              # Pipeline stages
│   ├── segment.py       # SAM3 instance segmentation
│   ├── crop.py          # PCA-aligned cropping
│   ├── embed.py         # DINOv2 feature extraction
│   ├── bank.py          # K-center memory bank selection
│   ├── heatmap.py       # Spatial distance heatmaps
│   └── refine.py        # Heatmap post-processing
│
├── config/
│   └── defaults.yaml    # Default configuration
│
├── projects/            # Project configs and results (permanent)
│   └── {project}/
│       ├── config.yaml
│       ├── runs.csv
│       └── runs/{run_id}/
│
├── artifacts/           # Generated outputs (regenerable)
│   └── {project}/
│       ├── data/
│       ├── masks/
│       ├── crops/
│       ├── embed/
│       ├── bank/
│       └── heatmaps/
│
└── datasets/            # Downloaded source datasets
```

## Configuration

Edit `projects/{project}/config.yaml` to customize the pipeline:

```yaml
# Instance Segmentation
segmentation:
  prompts: ["foreground object"]  # SAM3 text prompts
  enabled: true
  multi_instance: true            # Separate instances vs. union

# PCA Crop Alignment
crop:
  align_PCA: true
  angle_candidates: [0, 180]      # Rotation options for alignment
  scale: 1.0

# Feature Extraction
transformer:
  patch_size: 14
  map_resolution: [48, 48]        # Output feature map size

# Memory Bank
memory_bank:
  max_gb: 6                       # Maximum bank size
  selection_method: "k_center"    # k_center or random

# Post-Processing
refine:
  method: "gaussian"
  gaussian_sigma: 5

# Scoring
score:
  method: "mean"                  # mean, max, sum
  min_percentile: 99.9
  max_percentile: 99.999
```

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| **segment** | SAM3 instance segmentation using text prompts | Binary masks (PNG) |
| **crop** | PCA-aligned bounding box extraction | Cropped images + masks |
| **embed** | DINOv2 dense feature extraction | Embeddings (H×W×C) |
| **bank** | K-center greedy selection of reference embeddings | Memory bank file |
| **heatmap** | Spatial distance to nearest bank embeddings | Distance heatmaps |
| **refine** | Gaussian/bilateral filtering | Refined heatmaps |
| **score** | Percentile-based aggregation and AUROC computation | Metrics + run logs |

## CLI Reference

### prepare.py
```bash
uv run prepare.py <source> <category> [--all]

# Sources: visa, mvtec_ad, mvtec_loco, mvtec_ad2
# Examples:
uv run prepare.py visa capsules
uv run prepare.py mvtec_ad --all
```

### process.py
```bash
uv run process.py <project> [--force <stage>]

# Force rerun from a specific stage:
uv run process.py capsules --force crop
```

### score.py
```bash
uv run score.py <project> [--method <method>] [--min-pct <float>] [--max-pct <float>]

# Custom scoring parameters:
uv run score.py capsules --method mean --min-pct 99.9 --max-pct 99.999
```

## Metrics

The system computes:
- **Image AUROC**: Binary classification accuracy (normal vs. anomalous)
- **Pixel AUROC**: Localization accuracy against ground truth masks
- **Average Precision**: Ranking quality metric
- **Separation**: Gap between normal and anomaly score distributions
- **Per-Defect AUROC**: Breakdown by defect type

Results are logged to `projects/runs.csv` (global) and `projects/{project}/runs.csv` (per-project).

## License

MIT
