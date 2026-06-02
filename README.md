# FNN

A PyTorch-based Feedforward Neural Network toolkit for molecular property prediction. Supports multi-target prediction, hyperparameter tuning (Optuna), and fine-tuning.

## Quick Start

### 1. Clone

```bash
cp -r /home2/hdj/Toolkits/Machine_Learning/fnn .
```

### 2. Install

```bash
conda create -n fnn python=3.12
conda activate fnn
cd fnn
pip install -r requirements.txt
```

### 3. Configure

Copy `configs/model_parameters.example.yaml` to `configs/model_parameters.yaml` and edit. Key settings:

| Section | Key | Description |
| --- | --- | --- |
| General | `mode` | `training` / `hpo` / `prediction` / `fine-tuning` |
| General | `seed` | Random seed for reproducibility |
| Dataset | `data_file` | CSV with features and targets |
| Dataset | `smiles_column` | CSV column with SMILES (required for ECFP) |
| Dataset | `feature_list` | Column names for input features |
| Dataset | `target_list` | Column names for prediction targets |
| Model | `hidden_layer` | List of hidden layer sizes, e.g. `[64, 32]` |
| Model | `loss_fn` | `MAE` or `MSE` |
| Training | `epoch_num` | Number of training epochs |
| Training | `criteria_list` | Metrics: MAE / MSE / RMSD / R2 / AARD |

### 4. Run

All modes use the same entry point:

```bash
python main.py               # training (default)
python main.py               # hpo (set mode=hpo + copy configs/hpo.example.yaml → configs/hpo.yaml)
python main.py               # prediction (set mode=prediction)
python main.py               # fine-tuning (set mode=fine-tuning)
```

Background execution:

```bash
nohup python main.py > recording.log 2>&1 &
```

### 5. Results

All outputs are organized under `outputs/`:

```plain text
outputs/
├── training/<jobtype>/<TIME>/
│   ├── model/
│   │   ├── checkpoint/           # Periodic checkpoints
│   │   └── best_model_*.pth      # Best model by val metric
│   ├── plot/                     # Scatter plots & loss-epoch curves
│   ├── training_*.log            # Training log
│   └── model_parameters.yml      # Config snapshot
├── hpo/<jobtype>/<TIME>/
│   ├── Trial_000/ ... Trial_N/   # Per-trial results
│   ├── hpo_*.db                  # Optuna study database
│   └── hpo_*.log
└── prediction/<jobtype>/<TIME>/
    ├── data/                     # Predictions & targets (.pt)
    ├── plot/                     # Scatter plots
    └── prediction_*.log
```

## Project Structure

```plain text
fnn/
├── main.py                       # Entry point
├── configs/
│   ├── schema.py                         # Typed configuration dataclasses
│   ├── model_parameters.example.yaml     # Example model config
│   └── hpo.example.yaml                  # Example HPO config
├── data/
│   ├── dataset.py                        # FeatureDataset + FeatureSubset
│   └── data_processing.py                # Data loading, splitting, normalization
├── models/
│   ├── fnn.py                    # FNN module
│   └── factory.py                # Model creation factory
├── utils/
│   ├── gen_model.py              # Model/optimizer/scheduler wrappers
│   ├── metrics.py                # Regression metrics (MAE, MSE, R2, …)
│   ├── evaluation.py             # Inference + denormalization
│   ├── train.py                  # Training/validation loops
│   ├── save_model.py             # Checkpointing + early stopping
│   ├── file_processing.py        # Output dirs, logging, training resume
│   ├── timer.py                  # Elapsed time tracking
│   ├── setup_seed.py             # Reproducibility
│   ├── gpu_monitor.py            # GPU stats via nvidia-smi
│   ├── optuna_setup.py           # Optuna study factory
│   ├── visualization.py          # Scatter, histogram, bar, loss curves
│   ├── post_processing.py        # Log parsing
│   └── utils.py                  # Misc helpers
└── README.md
```

## Documentation

See `configs/model_parameters.example.yaml` for all configuration options with comments.
