# 🌫️ India in the Haze
### Country-Level PM2.5 Concentration Forecasting · ANRF AISE Hackathon Phase 2

---

## What is this?

A deep learning system that predicts **PM2.5 air pollution levels across all of India** for the next 16 hours — using only the past 10 hours of atmospheric data.

Built for a Kaggle competition hosted by IIT Delhi, this model goes beyond average-case forecasting. It is specifically engineered to detect and accurately predict **extreme pollution episodes** — the dangerous spikes that cause the most harm to public health.

---

## Why it matters

> India accounts for ~1.67 million deaths annually attributed to air pollution (Global Burden of Disease, 2019).

A reliable, fast PM2.5 forecasting system can:
- Trigger early health advisories before dangerous events
- Help policymakers plan interventions
- Replace slow, compute-heavy numerical models (WRF-Chem) with instant ML inference

---

## The Challenge

| | Details |
|---|---|
| **Input** | 10 hours of past data across a 140 × 124 grid (25 km/cell) |
| **Output** | PM2.5 forecasts for the next 16 hours across all of India |
| **Domain** | Entire Indian subcontinent |
| **Training data** | WRF-Chem simulations — 4 months of 2016 (April, July, October, December) |
| **Test data** | 2 unseen months from 2017 |
| **Key difficulty** | No future feature data at inference time + must excel during rare pollution episodes |

---

## How it works

### 1. Feature Engineering (21 channels)

The model ingests three types of inputs:

- **PM2.5 history** — past concentrations + frame-to-frame delta (rate of change)
- **Meteorology** — wind (u10, v10), temperature, humidity, rainfall, boundary layer height, surface pressure, solar radiation, wind speed, ventilation index
- **Emissions** — SO₂, NH₃, NOₓ, PM2.5 direct, biogenic isoprene, anthropogenic VOCs, biomass burning VOCs
- **Spatial context** — normalized latitude and longitude grids as static channels

> Heavily right-skewed features (all emissions, PM2.5, rain) are log₁p-transformed before normalization. QQ-plot analysis confirmed R² < 0.07 for raw emission distributions — confirming the necessity.

---

### 2. Model Architecture — CNN-ConvLSTM-UNet

```
Input: (Batch, 21 features, 10 timesteps, 140, 124)
         │
  Temporal Positional Embedding
         │
  ┌──────────────────────────────────┐
  │  Batched CNN Encoder (B×T)       │
  │  Conv Block 1 → 64ch  ──skip₁   │
  │  MaxPool                         │
  │  Conv Block 2 → 128ch ──skip₂   │
  │  MaxPool                         │
  └──────────────────────────────────┘
         │
  3-Layer ConvLSTM (hidden=192)
         │
  Temporal-Spatial Attention
         +
  Last LSTM Frame
         │
  Fusion Conv (→256ch)
         │
  SE Block + Spatial Attention + Temporal Channel Attention
         │
  ┌──────────────────────────────────┐
  │  UNet Decoder                    │
  │  ConvTranspose + skip₂ → 128ch  │
  │  ConvTranspose + skip₁ → 64ch   │
  └──────────────────────────────────┘
         │
  1×1 Conv head → 16 output timesteps
```

**Key design choices:**
- **ConvLSTM** captures how pollution patterns evolve spatially over time
- **Skip connections** from the *last encoder frame* (not mean) to preserve sharp spatial detail
- **Temporal Channel Attention (TCA)** re-weights feature maps by importance across time
- **SE + Spatial Attention** focuses the model on the most polluted regions

---

### 3. Episode-Aware Training

Pollution "episodes" are identified via **temporal decomposition** of PM2.5:

```
PM2.5 = Trend (24h moving avg) + Seasonal (hourly pattern) + Residual

Episode at grid point (i, j) at time t:
    Residual(i,j,t) > mean_residual(i,j) + 1.5 × std_residual(i,j)
```

This flags ~5.6% of all grid-timestep combinations as episodic.

The **custom loss function** combines:

| Component | Purpose |
|---|---|
| Weighted RMSE (log space) | Progressive weights — higher loss for high PM2.5 values |
| Episode MSE Boost (×7) | Extra penalty on episodic grid points |
| Episodic MAE | Directly penalizes peak magnitude errors |
| SMAPE | Relative error metric matching competition evaluation |
| 1 − Pearson Correlation | Ensures spatial pattern alignment |

---

### 4. Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=1e-3) |
| Scheduler | CosineAnnealingLR |
| Epochs | 55 (patience=22) |
| Batch size | 12 |
| Hidden dim | 192 |
| Mixed precision | ✅ (AMP) |
| Ensemble | 2 seeds (42, 137) |

**Proxy score for model selection:**
```
score = −(epSMAPE + 0.5 × gSMAPE − 0.3 × epCorr)
```
Episode SMAPE is the primary axis — it's the competition's hardest metric.

---

### 5. Inference & Ensemble

- Both models run inference on all 218 test samples
- Predictions are averaged across seeds
- Post-inference Gaussian smoothing was deliberately **removed** — it blurred pollution episode peaks, hurting episode metrics
- Final output: `preds.npy` of shape `(218, 140, 124, 16)`

---

## Evaluation Metrics

Three metrics, combined into a weighted final score:

| Metric | Measures |
|---|---|
| **Global SMAPE** | Overall prediction accuracy across the full India domain |
| **Episode SMAPE** | Accuracy of peak concentration magnitude during extreme events |
| **Episode Correlation** | Spatial pattern alignment during extreme events |

---

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `SciPy` · `Kaggle GPU (T4/P100)`

---

## Project Structure

```
├── The_Alchemists_v2.ipynb   # Full end-to-end pipeline
├── raw/
│   ├── APRIL_16/             # WRF-Chem training data (per-feature .npy)
│   ├── JULY_16/
│   ├── OCT_16/
│   ├── DEC_16/
│   └── lat_long.npy
├── test_in/                  # Test features (10-hour window only)
└── working/
    ├── preds.npy             # Final submission (218, 140, 124, 16)
    └── run_summary_p2.json   # Run metadata and per-seed metrics
```

---

## Key Lessons

- Raw emission values span ~10 orders of magnitude — always scale (×10⁹) and log-transform
- Episode-aware loss must balance two axes: overall accuracy vs. rare-event precision
- Gaussian smoothing improves global SMAPE but kills episode metrics — know your trade-off
- Ensemble averaging with different random seeds gives a free ~0.01–0.02 score boost
- Last-frame skip connections outperform mean-pooled skips for sharp spatial detail

---

*Competition: ANRF AISE Hackathon Phase 2, Theme 2 — hosted on Kaggle by IIT Delhi*
