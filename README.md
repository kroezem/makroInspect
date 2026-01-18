# makroInspect

Visual anomaly detection system using SAM3 segmentation and DINOv2 embeddings.

## Architecture

### Process Model
- **Ingest Process** (CPU): Scans inboxes, hashes files, writes to DB
- **Main Process** (GPU): Segment -> Crop -> Evaluate -> Bank management

### Priority Lanes
| Lane | Priority | Behavior |
|------|----------|----------|
| Sprint | 10-100 | Greedy (SEG/CROP/EVAL interleaved) |
| Bulk | 5-9 | Tidal (complete each phase before next) |
| Maintenance | 0-4 | Background rescore, only when idle |

### Data Flow
1. **Ingest**: inbox/ -> artifacts/raw/ (assigns priority 10 for <=8 images, 5 for more)
2. **Segment**: SAM3 segmentation -> artifacts/masks/
3. **Crop**: PCA-aligned cropping -> artifacts/crop_images/
4. **Evaluate**: DINOv2 features + scoring -> artifacts/embeddings/, heatmaps/
5. **Bank**: K-Center Greedy selection -> bank.npy (memory bank for scoring)

### Label System
- `unlabeled`: Default, not considered for memory bank
- `normal`: Good samples, used for bank composition
- `anomaly`: User-marked defects
- `exclude`: Ignored in all analysis

### Special Inbox Folders
Place images in these folders to auto-assign labels:
- `inbox/_normal/` -> label="normal" (included in bank)
- `inbox/_anomaly/` -> label="anomaly"
- `inbox/_exclude/` -> label="exclude"

### Configuration
See `projects/defaults.yaml` for all options including:
- Segmentation prompts and multi-instance settings
- Crop scale and PCA alignment
- Anomaly scoring method and percentiles
- Memory bank size and replacement policy

## Folder Structure
```
projects/
  <project_name>/
    inbox/           # Drop images here
      _normal/       # Auto-label as normal
      _anomaly/      # Auto-label as anomaly
    artifacts/
      raw/           # Original images
      masks/         # Segmentation masks
      crop_images/   # Aligned crops
      embeddings/    # DINOv2 features
      heatmaps/      # Anomaly visualizations
    bank.npy         # Memory bank
    config.yaml      # Project settings
    project.db       # SQLite database
```
