# Repository Guidelines

## Project Structure & Module Organization
- `mamba/`: Mamba-based detector implementation and the **single** training notebook (`mamba.ipynb`). Core modules include `model.py`, `dataset.py`, `trainer.py`, and `config.py`.
- `shared/`: Reusable utilities (VOC parser, transforms, visualization, logging helpers).
- `scripts/`: Data preparation and export utilities (`prepare_data.py`, `download_sample.py`, `export_model.py`).
- `data/`: Unified data root. Use `data/MMFW-UAV/sample/` for local testing and `data/MMFW-UAV/raw/` on Lightning AI.
- `outputs/`: Training artifacts (checkpoints, logs). This is gitignored.

## Build, Test, and Development Commands
- Create venv and install deps:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Create train/val/test splits:
  ```bash
  python scripts/prepare_data.py --data_root data/MMFW-UAV/raw
  ```
- Create a local sample subset (from a local raw copy):
  ```bash
  python scripts/download_sample.py --raw_dir data/MMFW-UAV/raw --output_dir data/MMFW-UAV/sample
  ```
- Run training notebook:
  ```bash
  cd mamba
  jupyter notebook mamba.ipynb
  ```

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` functions/variables, `PascalCase` classes.
- Keep notebook cells short and deterministic. Prefer logic in `.py` modules.
- Use repo-relative paths (e.g., `data/MMFW-UAV/sample`) to keep Mac/Lightning parity.

## Testing Guidelines
- No formal tests yet. If you add tests, use `pytest` and place them in `tests/`.
- Use `test_*.py` naming and keep fixtures near the code they validate.

## Commit & Pull Request Guidelines
- Current history is minimal, so follow conventional, imperative commit messages (e.g., “Add sequence dataset”).
- PRs should include: summary, motivation, and how changes were verified. For notebooks, clear excessive outputs before committing.

## Data & Environment Notes
- `mamba-ssm` is skipped on macOS; use LSTM fallback locally and run real Mamba on Lightning AI GPU.
- Keep full datasets and generated artifacts out of git; only commit the small `sample/` subset if needed.
