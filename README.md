# Classifying Pianist Expertise from Motor Timing Features Alone - Deep Learning Project

## Overview

This project classifies piano performers as **amateur** or **virtuoso** based solely on expressive MIDI features (timing, duration, velocity), deliberately 
masking pitch information. It uses a bidirectional LSTM model trained on segments extracted from two datasets: **PianoVAM** (amateur) and **MAESTRO** (virtuoso).

***

## Project Structure

```
.
├── preprocessing.py       # MIDI feature extraction and dataset preparation
├── train.py               # Model training, evaluation, and run saving
├── dataset.py             # DataLoader utilities (get_dataloaders)
├── model.py               # Bidirectional LSTM model definition
├── requirements.txt       # Python dependencies
├── Runs/
├── DataSets/
│   ├── PianoVAM_v1.1/MIDI/    # Amateur MIDI files (.mid)
│   └── Maestro/maestro-v3.0.0/ # Virtuoso MIDI files (.midi)
└── Data/
    └── processed/             # Generated .npy segments + manifest.csv

```

***

## Datasets

| Dataset | Label | Format | Description |
|---------|-------|--------|-------------|
| [PianoVAM v1.1](https://github.com/cpjku/PianoVAM) | `0` (amateur) | `.mid` | Amateur piano recordings |
| [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) | `1` (virtuoso) | `.midi` | Professional concert performances |

Place the datasets under `DataSets/` following the structure above before running preprocessing.

***

## Installation

1. **Clone the repository** and navigate to the project folder.

2. **Create and activate a conda environment:**
   ```bash
   conda create -n DeepLearning_Project python=3.10
   conda activate DeepLearning_Project
   ```

3. **Install dependencies (CPU-only):**
   ```bash
   pip install -r requirements.txt
   ```
***

## Usage

### Step 1 — Preprocess the MIDI files

```bash
python3 preprocessing.py
```

This will:
- Parse all MIDI files from both datasets
- Extract per-note features (onset, duration, velocity, IOI) with pitch masked to 0
- Slice each performance into overlapping 45-second segments (22s hop, ≥20 notes)
- Normalise each segment into a `(512, 4)` float32 NumPy array
- Save `.npy` files to `Data/processed/`
- Generate a balanced `manifest.csv` with train/val/test splits (70/15/15)

### Step 2 — Train the model

```bash
python3 train.py
```

This will:
- Load data via `manifest.csv`
- Train a bidirectional LSTM with early stopping and cosine annealing LR scheduler
- Save the best model checkpoint and a `results.json` to a timestamped folder under `Runs/`

***

## Feature Representation

Each note is represented as a 4-dimensional vector, all normalised to `[0.0, 1.0]`:

| Feature | Description | Normalisation |
|---------|-------------|---------------|
| `onset` | Relative position within the segment | Divided by segment duration |
| `duration` | Key hold time | Capped at 4s, divided by 4 |
| `velocity` | Strike force | Divided by 127 |
| `ioi` | Inter-onset interval (time since previous note) | Capped at 2s, divided by 2 |

> **Note:** Pitch is intentionally masked to `0` to test whether expressive timing and dynamics alone are sufficient for skill classification.

***

## Model & Training Configuration

| Hyperparameter | Default Value |
|----------------|---------------|
| Hidden size | 128 |
| Layers | 2 |
| Bidirectional | True |
| Dropout | 0.0 |
| Batch size | 16 |
| Max epochs | 300 |
| Learning rate | 3e-4 |
| Early stopping patience | 20 epochs |
| LR scheduler | Cosine Annealing Warm Restarts (T₀=20, factor=2) |
| Features removed | `ioi` |

***

## Output

Each training run is saved in a timestamped directory under `Runs/`, e.g.:

```
Runs/20240528_172301_LR0.0003_BS16_H128_L2_DO0.0_remove-ioi_valLoss0.3210_valAcc0.8700_testAcc0.8650/
├── model.pt        # Best model weights (lowest val loss)
└── results.json    # Full config + metrics
```

***

## Dependencies

| Package | Version |
|---------|---------|
| torch | 2.12.0+cu130|
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| pretty_midi | 0.2.10 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 | # only used for plotting
| scikit-learn | 1.7.2 | # only used for plotting

***
