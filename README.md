# 🌫️ India in the Haze — STForecaster

> **Bridging atmospheric physics and deep learning to predict severe PM2.5 pollution episodes 16 hours ahead across India.**

<p align="center">
  <img src="https://img.shields.io/badge/Competition-AISEHack%20Phase%202-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Organizer-E--Cell%20IIIT%20Hyderabad-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Theme-Pollution%20Forecasting-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Phase%201-AIR%20%2312-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Phase%202%20Finale-AIR%20%232%20Public-gold?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=flat-square&logo=kaggle"/>
  <img src="https://img.shields.io/badge/Parameters-10.01M-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Speedup-3000x%20vs%20WRF--Chem-brightgreen?style=flat-square"/>
</p>

---

## 🏆 Competition Results

| Stage | Leaderboard | Rank | Score |
|---|---|---|---|
| Phase 1 | Public | **AIR #12** | **12.753** |
| Phase 2 Grand Finale | Public | **AIR #2** | **0.8950** |
| Phase 2 Grand Finale | Private (Final) | **AIR #4** | **0.8783** |

> **Competition:** ANRF AISEHack Phase 2 — Theme 2: Pollution Forecasting
> **Hosted by:** E-Cell IIIT Hyderabad | Grand Finale held at IIIT Hyderabad (April 2026)

---

## 📌 About the Project

Air pollution is one of India's most critical public health challenges. PM2.5 — fine particulate matter — causes millions of premature deaths annually. Accurate short-term forecasting is essential.

**The challenge:** Given 10 hours of past atmospheric data across India's spatial grid, predict PM2.5 concentrations for the next **16 hours** — without access to future meteorological data (autoregressive setup).

Traditional numerical solvers like **WRF-Chem** take 45–90 minutes per forecast case. Our model does it in **~0.8 seconds** — a **>3000× speedup** — with less than 20% SMAPE degradation relative to the solver.

---

## 🌐 Physics Background

The governing equation for atmospheric pollutant transport:

```
∂C/∂t  +  u·∇C  =  ∇·(K∇C)  +  E  −  λC
```

Where `C` = PM2.5 concentration, `u` = wind velocity, `K` = turbulent diffusion, `E` = emissions, `λ` = chemical decay.

**Why standard deep learning fails here:**

| Physics Problem | Standard DL Failure | Our Fix |
|---|---|---|
| Spatiotemporal Transport | CNNs are spatially memoryless, fail to capture advection | 3-Layer ConvLSTM tracking 10-hour advection history |
| Extreme Events | Plain MSE ignores PM2.5 spikes (episodes = only 5.6% of data) | Episode-Aware Loss via STL Decomposition |
| Emission Scaling | Log-normal distributions span >10 orders of magnitude | Hard physics constraints: log1p + 1e9 scaling |

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| Source | WRF-Chem numerical solver output |
| Months | APRIL_16, JULY_16, OCT_16, DEC_16 |
| Spatial Grid | 140 × 124 (India domain) |
| Temporal Resolution | Hourly |
| Train / Val Split | 85% / 15% (chronological, no leakage) |
| Temporal Buffer | 26 steps between train and val |
| Test Samples | 218 |

---

## 📦 Features

### Base Features (16)
| Category | Features |
|---|---|
| Pollution | `cpm25` |
| Meteorology | `u10`, `v10`, `rain`, `t2`, `pblh`, `psfc`, `swdown`, `q2` |
| Emissions | `SO2`, `NH3`, `NOx`, `PM25`, `bio`, `NMVOC_finn`, `NMVOC_e` |

### Derived Features (5)
| Feature | Formula | Why |
|---|---|---|
| `wind_speed` | `√(u10² + v10²)` | Magnitude more useful than directional components |
| `vent_index` | `wind_speed × pblh` | Captures atmospheric dispersion capacity |
| `delta_cpm25` | `cpm25[t] - cpm25[t-1]` | Rate of change — is pollution rising or falling? |
| `lat` | min-max normalized | Anchors predictions to geographic terrain |
| `lon` | min-max normalized | Anchors predictions to geographic terrain |

### Log1p Transform — Applied to 9 Features
`cpm25, SO2, NH3, NOx, PM25, bio, NMVOC_finn, NMVOC_e, rain`

QQ-plot R² values (all far from normal — log1p is non-negotiable):
```
SO2=0.023 | NOx=0.042 | PM25=0.052 | NMVOC_finn=0.009
bio=0.274  | NH3=0.319 | NMVOC_e=0.184 | rain=0.013
```

**Final input tensor shape:** `(Batch, 21, 10, 140, 124)`

---

## 🧠 Model Architecture — STForecaster

```
Input  (B × 10 × 21 × 140 × 124)
           │
           ▼
  ┌─────────────────────┐
  │  Batched CNN Encoder │  B*T frames in one GPU call (no loop)
  │  1/2 & 1/4 skips    │  Outputs spatial feature maps
  └─────────────────────┘
           │
           ▼
  ┌──────────────────────────┐
  │  Positional & Temporal   │  3-Layer ConvLSTM (192 hidden dims)
  │  Core                    │  5D Temporal Positional Embeddings
  └──────────────────────────┘
           │
           ▼
  ┌──────────────────────────┐
  │  Dual Attention           │
  │  ├─ Path A: Temporal-    │  1×1 conv scorer over 10 hidden states
  │  │   Spatial Attention   │
  │  └─ Path B: Channel      │  SE-style squeeze-excite
  │      Attention           │
  └──────────────────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  U-Net Decoder       │  Upsamples back to 140×124
  │  Last-frame skip     │  Uses last input frame (NOT mean)
  └─────────────────────┘
           │
           ▼
Output (B × 16 × 140 × 124) — 16-hour PM2.5 forecast
```

**Total parameters: 10.01M**

### Key Architecture Decisions

| Decision | Why |
|---|---|
| **Batched Encoder (B\*T)** | All timesteps in one GPU call — ~10× faster than loop |
| **Last-frame skip** | Preserves most recent spatial state (mean dilutes recent info) |
| **3-Layer ConvLSTM** | Recurrent memory is essential — removing it degrades error by +68% |
| **Dual Attention** | Captures both "when" (temporal) and "which feature" (channel) matters |
| **5D Positional Embeddings** | Provides boundary-layer evolution context across the 10-hour window |

---

## 🎭 Episode Detection — STL Decomposition

Pollution episodes represent only **5.6% of all pixels** but are the most critical for public health. Standard loss functions ignore them.

```python
# Step 1: Remove trend (24hr moving average)
trend = uniform_filter1d(data, size=24)

# Step 2: Remove daily seasonal pattern
seasonal = hourly_average[t % 24]

# Step 3: Residual anomaly
residual = data - trend - seasonal

# Step 4: Mark as episode if residual > μ + 1.5σ
episode_mask = (residual > res_mean + 1.5 * res_std)
```

These episode pixels are upweighted **7×** in the loss function.

---

## 📉 Loss Function

```
L = RMSE_logz  +  0.30 × SMAPE  +  0.15 × (1 − ρ)  +  7.0 × MSE_episode
```

| Term | Weight | Purpose |
|---|---|---|
| `RMSE_logz` | 0.55 | Primary reconstruction in log space |
| `SMAPE` | 0.30 | Scale-invariant % error — direct competition metric surrogate |
| `1 − ρ` | 0.15 | Pearson correlation — preserves spatial pollution patterns |
| `MSE_episode` | **7.0×** | Heavy focus on extreme pollution events |

> **Note:** SMAPE ε = 0.1. The competition formula uses `0.5×(|y|+|ŷ|)` denominator. Using ε=1.0 (default) heavily dampens low-concentration errors.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 55 |
| Learning Rate | 3×10⁻⁴ |
| Batch Size | 12 |
| Hidden Dim | 192 |
| Warmup Epochs | 4 |
| Early Stopping Patience | 22 |
| Weight Decay | 1×10⁻³ |
| Val Fraction | 15% |
| Temporal Buffer | 26 steps |
| Episode MSE Boost | 7.0× |
| LR Scheduler | CosineAnnealingLR |
| Ensemble Seeds | [42, 137] |

---

## 📊 Results

### Model Comparison

| Model | Global RMSE (μg/m³) | Global SMAPE | Episode SMAPE | Episode Corr (ρ) |
|---|---|---|---|---|
| Persistence Baseline | ~38–45 | 0.42 | 0.51 | 0.72 |
| Plain ConvLSTM | 22.1 | 0.261 | 0.318 | 0.891 |
| **Our 2-Seed Ensemble** | **~16.5** | **0.200** | **0.224** | **0.938** |

> The model tracks high-pollution spikes with **<25% symmetric error**, completely eliminating the systemic lag found in persistence models. Inference takes **<3 minutes** for all 218 test cases.

### Computational Speedup

| System | Time per Case |
|---|---|
| WRF-Chem Numerical Solver | ~45–90 minutes |
| **STForecaster (Ours)** | **~0.8 seconds** |
| **Speedup** | **>3,000×** |

---

## 🔬 Ablation Study

| Removed Component | Episode SMAPE Degradation | Insight |
|---|---|---|
| Full Model (baseline) | 0% | — |
| Remove Temporal-Channel Attention | +12% | Moderate contribution |
| Remove Episode Boost Loss | +34% | Episode-blind networks fail on extremes |
| Replace log1p with raw normalisation | +52% | Invalidates log-normal physics of emissions |
| Replace ConvLSTM with mean-pooling | **+68%** | Atmospheric transport requires recurrent memory |

---

## 🔄 Version History

| Version | Phase 1 RMSE | Key Change |
|---|---|---|
| v1 | ~19.0 | Baseline |
| v3 | ~16.5 | Feature engineering |
| v5 | ~15.2 | Episode mask introduced |
| v7 | ~13.8 | Dual attention added |
| v9 | ~13.1 | CosineAnnealingLR + last-frame skip |
| **v11** | **~12.3** | **FIX-A to FIX-F — best submission** |

### Fixes in v11

| Fix | Description |
|---|---|
| FIX-A | LOG_FEATURES restored to all 8 emissions + cpm25 + rain |
| FIX-B | Combined progressive logz weighting + episode mask boost |
| FIX-C | SMAPE ε reduced 1.0 → 0.1 |
| FIX-D | Emission scaling guard removed |
| FIX-E | 2-seed ensemble [42, 137] |
| FIX-F | Batch size 12 with gradient accumulation |

---

## 🌍 What the Model Learned

The attention weights reveal the model autonomously discovered real atmospheric physics:

**1. Biogenic Dominance (3.1×)**
During July monsoons, the model assigns 3.1× higher attention to biogenic/NMVOC channels — accurately emulating SOA formation when anthropogenic transport is suppressed.

**2. Spatial Teleconnection: Punjab → Bangladesh**
Attention networks discovered non-local correlations at +14–16 hour lead times, perfectly matching westerly jet transport mechanics — without being explicitly told.

**3. Tendency Shift over Forecast Horizon**
For hours 1–4, model relies on recent PM2.5 tendencies. For longer horizons, it correctly shifts attention to fundamental meteorological drivers (wind, PBLH, temperature).

---

## ⚠️ Known Limitations

- **December winter fog:** Deep winter fog causes boundary-layer collapse — model shows residual prediction bias in Dec_16.
- **Autoregressive error accumulation:** Phase 2 inference uses model's own predictions as next-step input. Errors compound over the 16-hour horizon, especially beyond hour 12.

---

## 🚀 Future Work

- **3 months:** Scale to 2015–2022 dataset for interannual ENSO robustness
- **6 months:** Implement Graph Neural Networks on WRF mesh for coastal/orographic boundaries (Western Ghats)
- **Operational goal:** Deploy as a real-time FastAPI microservice for sub-hourly AQI nowcasting within India's National Clean Air Programme (NCAP)

---

## 🗂️ Repo Structure

```
india-in-the-haze/
├── phase-2-v2-0.ipynb      # Main training + inference notebook (v11)
├── README.md
└── outputs/
    └── submission.npy      # Final prediction array shape: (218, 16, H, W)
```

---

## ⚡ How to Run

### Requirements
```bash
pip install torch numpy scipy
```

### Kaggle Setup
1. Add competition dataset: `anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd`
2. Verify paths in CONFIG block:
```python
BASE_PATH = "/kaggle/input/.../raw"
TEST_PATH = "/kaggle/input/.../test_in"
```
3. Run all cells — training ~2–3 hours on Kaggle GPU (P100/T4)
4. Submission saved at `/kaggle/working/submission.npy`

### Config to Tune
```python
ENSEMBLE_SEEDS    = [42, 137]  # add more seeds for better ensemble
HIDDEN_DIM        = 192        # increase for capacity (watch VRAM)
EPISODE_MSE_BOOST = 7.0        # raise if episode SMAPE is high
EPOCHS            = 55         # increase with more compute
```

---

## 👥 Team & Acknowledgements

**Team: The Alchemists**
NMIMS Mukesh Patel School of Technology Management & Engineering (MPSTME), Shirpur

**Competition:** ANRF AISEHack — National AI Hackathon
**Organized by:** E-Cell IIIT Hyderabad
**Sponsored by:** Anusandhan National Research Foundation (ANRF)
**Dataset:** WRF-Chem atmospheric simulation outputs

---

<p align="center">
  <i>Built with ☕ and a lot of NaN loss debugging at IIIT Hyderabad Grand Finale.</i>
</p>
