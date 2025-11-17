# User Guide — Aftershoot White Balance Prediction

This user guide shows how to set up, run, and evaluate the Aftershoot White Balance Prediction project locally on Windows (also works on Linux/macOS with minor adjustments).

**Quick links**
- Project root: `E:\aftershoot\aftershoot_wb_prediction`
- Notebook: `aftershoot_poetry_colab.ipynb`
- Main training script: `main.py`
- Model testing script: `test_model.py`

---

## 1. Prerequisites

- Python 3.8 or newer
- Git
- Optional but recommended: NVIDIA GPU and CUDA (tested with CUDA 12.x)
- A virtual environment for isolation (recommended: `.venv`)

Recommended packages (install later via pip or Poetry):
- `torch`, `torchvision`, `torchaudio` (CUDA build matching your drivers)
- `timm`, `albumentations`, `opencv-python`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `Pillow`

---

## 2. Prepare repository (one-time)

1. Clone the repo:

```powershell
git clone https://github.com/shaktiaryan/Image_Whitebalance_ML.git
cd Image_Whitebalance_ML
```

2. Create and activate a virtual environment (Windows example):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. (Optional) Install Poetry if you prefer Poetry-managed environments:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
poetry install
```

4. Install dependencies with pip (example with CUDA wheel index):

```powershell
# Adjust index-url to match your CUDA version (example: cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 3. Data layout and what to provide

Place your data in the `data/` directory:
- `data/Train/images/` — training images (not tracked in git)
- `data/Train/sliders_filtered.csv` — training CSV with ground-truth `Temperature` and `Tint` columns
- `data/Validation/images/` — validation images (no targets)
- `data/Validation/sliders_inputs.csv` — validation metadata CSV

Small `.gitkeep` files are present so folder structure is maintained in the repo.

---

## 4. Running EDA (exploratory data analysis)

EDA produces visualizations under `outputs/eda` and prints dataset statistics.

```powershell
# Run EDA via the main script
python main.py --eda --config efficientnet
```

Or run the notebook `aftershoot_poetry_colab.ipynb` and execute the EDA cells interactively.

---

## 5. Quick training (smoke test)

Run a short training session to confirm environment and GPU availability.

```powershell
# Quick 1-5 epoch test
python main.py --config lightweight --epochs 1 --device cuda --output_dir outputs/test_training
```

Watch console logs for training/validation sample counts and model initialization. Successful run will create `outputs/test_training/logs/` with `training_results.json`, `training.log` and visualization PNGs.

---

## 6. Full training (recommended)

After a successful smoke test, run the full training (example 15 epochs):

```powershell
python main.py --config lightweight --epochs 15 --device cuda --output_dir outputs/optimized_90_split
```

Notes:
- The project uses an optimized 90/10 train/val split (configured in `configs/base_config.py`).
- Check `outputs/<run>/logs/` for `training_results.json` and `training_curves.png`.

---

## 7. Model testing & evaluation

After training completes, run the testing script to generate predictions and metrics:

```powershell
python test_model.py
```

Or use the helper script created by the notebook: `test_colab_model.py`.

Testing outputs are placed in `outputs/model_testing/` and include:
- `predictions.csv` — per-image predictions and errors
- `testing_results.json` — aggregated metrics (MAE, RMSE)
- visualization PNGs: prediction scatter, error distribution

---

## 8. Monitor and inspect results via notebook

Open `aftershoot_poetry_colab.ipynb` and run the "Training Monitoring" and "Model Testing" sections. The updated notebook cells automatically look for training results across `outputs/*/logs/` and will display the best run and visualizations.

---

## 9. Troubleshooting (common issues)

- "No training log files found": Ensure you executed a training run and check `outputs/<run>/logs/` exists.
- CUDA not available: Verify GPU drivers and PyTorch CUDA build match; run `python -c "import torch; print(torch.cuda.is_available())"`.
- Poetry not found: Either install Poetry system-wide or use pip + `.venv` instead. The notebook includes a Windows-friendly installer cell.
- Long HTTP download to fetch pretrained weights: Training may download backbone weights from remote (timm/Hugging Face); ensure network access or pre-download weights.

---

## 10. Backing up results

The notebook contains steps to save important artifacts locally or to Google Drive. Locally, outputs are copied into a timestamped `backup_results_<timestamp>/` directory.

Example (from notebook):
```python
# The notebook copies these directories:
# outputs/checkpoints, outputs/logs, outputs/eda, outputs/model_testing, configs, pyproject.toml, poetry.lock
```

---

## 11. Developer notes

- Main training code is in `src/training/trainer.py`.
- Model architectures are in `src/models/` (multimodal + backbone code).
- Configs are in `configs/base_config.py` and `configs/model_configs.py` — modify carefully.

---

## 12. Useful commands summary

```powershell
# Activate venv
.venv\Scripts\Activate.ps1

# Run EDA
python main.py --eda --config efficientnet

# Quick train smoke test
python main.py --config lightweight --epochs 1 --device cuda --output_dir outputs/test_training

# Full train
python main.py --config lightweight --epochs 15 --device cuda --output_dir outputs/optimized_90_split

# Run tests
python test_model.py

# Inspect notebook
jupyter notebook aftershoot_poetry_colab.ipynb
```

---

## 13. Support & contact

If you run into problems, provide:
- OS and Python version
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- Command you ran and full console output

Contact: `https://github.com/shaktiaryan` (repo owner)

---

*Created by the project tooling — edit this file to add any project-specific deviations or operational notes.*
