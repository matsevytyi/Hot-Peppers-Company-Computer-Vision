# Data Directory

## MMFW-UAV Dataset

**Source**: https://doi.org/10.57760/sciencedb.07839
**Paper**: https://doi.org/10.1038/s41597-025-04482-2

### Structure
- `sample/`: Small subset for local testing (500–1000 images)
- `raw/`: Full dataset (download on Lightning AI only)
- `processed/`: Preprocessed sequences
- `splits/`: Train/val/test splits

### Sample Dataset
- 2–3 UAV types
- Top-down view only
- Zoom sensor only
- ~500 images total

### Full Dataset (Lightning AI only)
- 12 UAV types
- 3 views: Top_Down, Horizontal, Bottom_Up
- 3 sensors: Zoom (3840×2160), Wide (3840×2160), Infrared (1280×1024)
- 147,417 total images
- 30 FPS video sequences
