# UAFN Urban Air Forecast Network

**Air Quality Forecasting in Urban Areas Using Graph Neural Networks**

> IU International University of Applied Sciences  
> Course: Project Computer Science Project (CSEMCSPCSP01)

---

## Overview

UAFN is a GNN-based system for short-term forecasting of urban air pollutants (PM₂.₅, NO₂, O₃). It models monitoring stations as a spatial graph and uses Graph Convolutional Networks (GCN) and GraphSAGE with GRU temporal encoding to learn spatio-temporal pollutant dynamics. An RLS-based data assimilation module handles missing and uncertain observations.

### Architecture

```
EEA Observations ─┐
                   ├──▶ Preprocessing + RLS ──▶ Graph Construction ──▶ GNN (GCN/SAGE + GRU) ──▶ Forecast
SILAM Model ───────┤                              (k-NN)                                        (1–6 hr)
ECMWF Meteo ───────┘
```

## Repository Structure

```
UAFN/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── data/
│   ├── synthetic_generator.py  # Synthetic AQ data generation
│   └── eea_downloader.py       # EEA API data downloader (in progress)
├── models/
│   ├── gcn_model.py            # GCN-based forecasting model
│   ├── sage_model.py           # GraphSAGE-based forecasting model
│   ├── mlp_baseline.py         # MLP baseline (no graph structure)
│   └── rls_filter.py           # RLS data assimilation filter
├── training/
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation and metrics
│   └── graph_builder.py        # Spatial graph construction
├── notebooks/
│   └── UAFN_Phase2.ipynb       # Complete runnable notebook
├── figures/                    # Generated figures
└── docs/
    └── UAFN_Phase2_Report.pdf  # Draft research report
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/<your-username>/UAFN.git
cd UAFN
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook notebooks/UAFN_Phase2.ipynb
```

Run all cells sequentially. The notebook generates synthetic data, trains GCN/GraphSAGE/MLP models, evaluates them, and produces all figures.

### 3. Run from Command Line

```bash
# Generate data, train all models, and evaluate
python training/train.py

# Evaluate a saved model
python training/evaluate.py --model gcn
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- PyTorch Geometric 2.x
- See `requirements.txt` for full list

## Models

| Model      | RMSE (µg/m³) | MAE (µg/m³) | R²    |
|------------|-------------|-------------|-------|
| GCN        | 2.988       | 2.388       | 0.555 |
| GraphSAGE  | 2.913       | 2.328       | 0.577 |
| MLP (base) | 2.852       | 2.316       | 0.595 |

*Results on synthetic data. Real EEA data evaluation planned for Phase 3.*

## Data Sources

- **EEA Air Quality e-Reporting**: [https://www.eea.europa.eu/en/datahub/datahubitem-view/a]
- **SILAM**: [https://silam.fmi.fi/](https://silam.fmi.fi/)
- **ECMWF ERA5**: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

## References

1. Kipf & Welling. *Semi-Supervised Classification with GCNs.* ICLR, 2017.
2. Hamilton et al. *Inductive Representation Learning on Large Graphs.* NeurIPS, 2017.
3. Guo et al. *Deep spatio-temporal learning for citywide AQ forecast.* J. Cleaner Prod., 2023.
4. Miasayedava et al. *Lightweight open data assimilation of pan-European urban AQ.* IEEE Access, 2023.

## License

This project is developed for academic purposes.
