<div align="center">

<br/>

```
 ░█████╗░██████╗░████████╗██╗███████╗██╗ ██████╗██╗ █████╗ ██╗
 ██╔══██╗██╔══██╗╚══██╔══╝██║██╔════╝██║██╔════╝██║██╔══██╗██║
 ███████║██████╔╝   ██║   ██║█████╗  ██║██║     ██║███████║██║
 ██╔══██║██╔══██╗   ██║   ██║██╔══╝  ██║██║     ██║██╔══██║██║
 ██║  ██║██║  ██║   ██║   ██║██║     ██║╚██████╗██║██║  ██║███████╗
 ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝     ╚═╝ ╚═════╝╚═╝╚═╝  ╚═╝╚══════╝

  ██╗   ██╗ ██████╗ ██╗      █████╗ ████████╗██╗██╗     ██╗████████╗██╗   ██╗
  ██║   ██║██╔═══██╗██║     ██╔══██╗╚══██╔══╝██║██║     ██║╚══██╔══╝╚██╗ ██╔╝
  ██║   ██║██║   ██║██║     ███████║   ██║   ██║██║     ██║   ██║    ╚████╔╝
  ╚██╗ ██╔╝██║   ██║██║     ██╔══██║   ██║   ██║██║     ██║   ██║     ╚██╔╝
   ╚████╔╝ ╚██████╔╝███████╗██║  ██║   ██║   ██║███████╗██║   ██║      ██║
    ╚═══╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝
```

### Detecting What the Market Tries to Hide

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-white?style=flat-square&labelColor=111&color=22c55e&logo=python&logoColor=white)
![TCN](https://img.shields.io/badge/Model-TCN-white?style=flat-square&labelColor=111&color=f59e0b)
![DBSCAN](https://img.shields.io/badge/Clustering-DBSCAN-white?style=flat-square&labelColor=111&color=f59e0b)
![NSE](https://img.shields.io/badge/Market-NSE%20India-white?style=flat-square&labelColor=111&color=3b82f6)
![Coverage](https://img.shields.io/badge/Coverage-2000--2026-white?style=flat-square&labelColor=111&color=a855f7)
![AUC](https://img.shields.io/badge/AUC-0.97-white?style=flat-square&labelColor=111&color=ef4444)

</div>

---

```
[ SYSTEM SUMMARY ]─────────────────────────────────────────────────────────────
  Dataset    : NIFTY 500  |  NSE India  |  2000–2026
  Core Model : Temporal Convolutional Network (TCN)
  Labeling   : Weakly supervised via DBSCAN clustering
  Task       : Binary — Natural Volatility vs. Artificial Volatility
  AUC        : 0.97   |   Accuracy : 82%   |   F1 (weighted) : 0.82
────────────────────────────────────────────────────────────────────────────────
```

<br/>

## 〉 What This Project Does

Stock markets move. Sometimes because of real economic forces — earnings, rate changes, geopolitical events. And sometimes because someone is *making* them move.

**Artificial Volatility Risk Detection** is a deep learning system trained to tell the difference.

Using 26 years of Indian equity market data from the NSE, this framework:

1. **Extracts meaningful signals** from raw price and volume data — log returns, rolling volatility, normalized volume, and index-level stress indicators
2. **Labels volatility regimes automatically** using DBSCAN clustering, without needing manually labeled fraud data
3. **Classifies every observation** using a Temporal Convolutional Network — either natural market behavior or a statistically anomalous volatility event
4. **Explains the economic significance** across time periods, market cycles, and around major structural events like India's 2016 Demonetization

No labeled fraud dataset. No black-box magic. A clean, interpretable pipeline from raw OHLCV data to a risk score.

---

## 〉 The Core Insight

```
Natural Volatility    →  Driven by earnings, macro events, sector trends
                          Clustered, predictable, structurally coherent

Artificial Volatility →  Isolated, high-dispersion, unexplained by context
                          No earnings trigger. No sector stress. Just spikes.
```

Traditional models — GARCH, ARIMA, even most LSTMs — don't draw this line. They model *how much* prices move, not *why*. This project models the **why**, using density-based outlier detection to separate the two regimes before training even begins.

---

## 〉 How It Works

### `STEP 1` — Data Collection

Daily OHLCV data pulled via **Yahoo Finance API** across a filtered universe of stable, continuously-listed companies from the NIFTY 500 index. Stability filtering ensures the analysis isn't skewed by companies that entered or exited mid-period.

Market-wide context: **NIFTY 50** (aggregate stress) and **NIFTY IT** (sector-level dynamics) included alongside stock data.

---

### `STEP 2` — Feature Engineering

| Feature | Description |
|---------|-------------|
| Log Returns | Daily price change, variance-stabilized |
| Rolling Volatility | 30-day standard deviation of returns |
| Market Stress Flag | NIFTY 50 volatility exceeds historical threshold |
| Sector Stress Flag | NIFTY IT volatility exceeds sector threshold |
| Volume Spike Flag | Trading volume exceeds historical distribution cutoff |

---

### `STEP 3` — Weak Supervision via DBSCAN

No labeled manipulation data exists — so labels are **generated**, not collected.

DBSCAN clusters observations by density in the 5-dimensional feature space:
- **Clustered observations** → normal volatility regime → labeled `0`
- **Noise / outlier observations** → anomalous behavior → labeled `1` (artificial)

A rule-based layer refines these labels: if a spike coincides with a known earnings cycle or market-wide stress event, it gets reclassified as natural. What remains unexplained stays flagged as artificial.

---

### `STEP 4` — TCN Classification

A **Temporal Convolutional Network** reads sequences of windowed feature vectors and predicts the probability of artificial volatility at each timestep.

Why TCN over LSTM?

```
LSTM  →  Sequential, slow, vanishing gradients on long sequences
TCN   →  Parallel dilated convolutions, stable over long horizons, faster
```

The output is a continuous **Artificial Volatility Risk Score** from 0 to 1:

```
0.0 – 0.35   →  Low risk    — consistent with natural market behavior
0.35 – 0.65  →  Moderate    — anomalous signal, warrants attention
0.65 – 1.0   →  High risk   — statistically unlikely under natural regime
```

---

## 〉 Performance

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Natural Volatility (0) | 0.89 | 0.85 | **0.87** | 586 |
| Artificial Volatility (1) | 0.65 | 0.72 | **0.68** | 214 |
| **Weighted Average** | **0.83** | **0.82** | **0.82** | 800 |

> Recall for artificial volatility (0.72) is prioritized over precision — in anomaly detection, missing a real event is costlier than a false alarm.

### Discriminative Power

```
ROC-AUC :  0.97
```

An AUC of 0.97 means the model maintains near-perfect separation between natural and artificial volatility across all classification thresholds — not just the default 0.5 cutoff.

---

## 〉 What the Data Reveals

### Indian Market Volatility: 2000–2026

| Period | Avg Volatility | Peak Volatility | Trend |
|--------|---------------|-----------------|-------|
| 2000–2005 | 0.0243 | 0.9359 | High baseline |
| 2005–2010 | 0.0264 | 0.9082 | Peak decade |
| 2010–2015 | 0.0177 | 0.0785 | Sharp stabilization |
| 2015–2020 | 0.0168 | 0.1204 | Mature efficiency |
| 2020–2026 | 0.0173 | 0.1343 | Resilient, stable |

The Indian equity market became measurably more efficient over 26 years. Volatility peaks shrank. Recovery from spikes became faster. The market learned.

---

### Artificial vs. Natural — Does It Matter Long-Term?

| Metric | Artificial | Natural | Verdict |
|--------|-----------|---------|---------|
| Avg Return | 0.202% | 0.086% | Marginally higher |
| Std Deviation | 0.385% | 0.073% | **Much riskier** |
| Statistical Significance | p ≈ 0 | — | Detectable but small |

The honest answer: **artificial volatility is detectable, but its long-term impact on fundamentally strong stocks is minimal.** Markets absorb and recover. Large-cap stable firms don't drift off their long-run trajectories due to short-term artificial noise.

Where it matters most: **short-term surveillance, regulatory monitoring, and intraday risk management.**

---

### The Demonetization Test

India's November 2016 demonetization was a massive structural shock. The system analyzed pre- and post-event behavior across NIFTY 500 companies.

```
Before Demonetization  →  Avg Return: 0.07%  |  Artificial Return: 0.11%
After  Demonetization  →  Avg Return: 0.09%  |  Artificial Return: 0.13%
Effect Size: –0.79  (large, unified structural shift across all stocks)
```

**Finding:** The demonetization had a stronger, more predictable effect than artificial volatility ever did. Macro events dominate. Artificial volatility is noise around those signals — detectable noise, but noise nonetheless.

---

## 〉 Architecture

```
Raw Data (Yahoo Finance API — OHLCV)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Feature Engineering Layer                                │
│  log_returns · rolling_vol · mkt_stress · vol_spike      │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  DBSCAN Clustering  (Weak Supervision)                   │
│  Dense regions  →  Natural  (label: 0)                   │
│  Outlier noise  →  Artificial (label: 1)                 │
│  + Rule refinement via earnings / market context          │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Sliding Window Sequence Generator                        │
│  [t-n ... t] → multivariate feature tensor               │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Temporal Convolutional Network (TCN)                    │
│  Dilated causal convolutions → temporal feature maps     │
│  Sigmoid output → Artificial Volatility Risk Score       │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
        Risk Score :  [0.0 ──────────── 1.0]
                       Low    Moderate   High
```

---

## 〉 Stack

| Component | Technology |
|-----------|-----------|
| Data Source | Yahoo Finance API (`yfinance`) |
| Data Processing | Pandas, NumPy |
| Clustering | Scikit-learn (DBSCAN) |
| Deep Learning | PyTorch / TensorFlow |
| Market Indices | NIFTY 50, NIFTY IT (^CNXIT) |
| Evaluation | Scikit-learn metrics, ROC-AUC |
| Visualization | Matplotlib, Seaborn |

---

## 〉 Getting Started

### Install

```bash
git clone https://github.com/Vishal8500/Artificial-Volatility-Risk-Detection.git
cd Artificial-Volatility-Risk-Detection

pip install -r requirements.txt
```

### Run

```bash
# Step 1: Pull and preprocess data
python src/data_pipeline.py

# Step 2: Generate weak supervision labels via DBSCAN
python src/clustering.py

# Step 3: Train the TCN model
python src/train.py

# Step 4: Evaluate and visualize results
python src/evaluate.py
```

### Key Parameters

```python
WINDOW_SIZE     = 30       # Sequence length for TCN input
DBSCAN_EPS      = 0.5      # Neighborhood radius
DBSCAN_MIN_PTS  = 5        # Minimum cluster density
THRESHOLD_SIGMA = 2.0      # Std deviations for volatility flagging
DROPOUT_RATE    = 0.3      # TCN regularization
```

---

## 〉 Limitations & Future Work

- **Weak labels only** — no ground truth manipulation data exists for Indian markets. Future work could validate against SEBI enforcement records.
- **Daily granularity** — intraday (minute-level) data would dramatically improve short-term surveillance capability.
- **News sentiment** — integrating NLP signals alongside price/volume features could reduce false positives during genuine information events.
- **Multi-market** — extending to BSE and cross-border spillover analysis is the natural next step.

---

## 〉 Team

**Vellore Institute of Technology, Chennai**  
School of Computer Science and Engineering

| Name | Contribution |
|------|-------------|
| M. Vishal | Model architecture & temporal analysis |
| Phoobesh S. | Feature engineering & clustering |
| Devadarishini S. | Evaluation & event-based analysis |

---

<div align="center">

<br/>

```
[ ARTIFICIAL VOLATILITY RISK DETECTION ]──────────────────────────────────────
  STATUS   : ACTIVE
  MARKET   : NSE INDIA  (NIFTY 500)
  COVERAGE : 2000 – 2026
  MODEL    : TCN + DBSCAN Weak Supervision
  AUC      : 0.97  |  ACCURACY : 82%  |  F1 : 0.82
─────────────────────────────────────────────────────────────────────────────
```

*Understanding the Indian market — one volatility regime at a time.*

<br/>

⭐ Star this repo if it was useful.

</div>
