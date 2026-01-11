# Image Classification — Smile Detection (Logistic Regression)

## Overview

This repository demonstrates a simple image classification pipeline for detecting smiles using logistic regression. It contains a small dataset of face images organized into smile / non-smile folders, a training notebook, and a minimal app for inference.

## Contents

- `app.py` — small script for running inference or demo (see usage).
- `smile_stalker.ipynb` — Jupyter notebook with data loading, preprocessing, training, and evaluation code.
- `requirement.txt` — Python dependencies to install.
- `Data/` — dataset directory with subfolders:
  - `Data/smile/` — images labeled as smiling
  - `Data/non_smile/` — images labeled as not smiling
  - `Data/test/` — holdout images for quick testing

## Setup

Recommended: create a virtual environment and install dependencies.

Windows PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirement.txt
```

If you use Command Prompt replace the activation command with `\.venv\Scripts\activate.bat`.

## Usage

- To explore and train the model interactively, open and run `smile_stalker.ipynb`.
- To run the demo/inference (if `app.py` exists):

```powershell
python app.py
```

Adjust any file paths inside the notebook or `app.py` to match your `Data/` location if needed.

## Dataset Notes

- Images are expected as common formats (JPEG/PNG). Preprocessing (resizing, normalization) is handled in the notebook.
- For larger datasets consider switching from logistic regression to a CNN for better accuracy.

## Reproducibility

- Seed any random number generators in the notebook before training for reproducible runs.
- Save trained model artifacts (if produced) to a `models/` folder and document their names in the notebook.


