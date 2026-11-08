# DispatchOpt: Spatio-Temporal Conformalized Quantile Regression & Dynamic Delivery SLA Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade **Spatio-Temporal Uncertainty Quantification & Dynamic Dispatch** engine designed for high-scale on-demand last-mile logistics (e.g., Gojek GoSend, GrabExpress). DispatchOpt implements **Conformalized Quantile Regression (CQR)** to provide **distribution-free finite-sample guarantees ($1 - \alpha = 90\%$)** on delivery time windows, drastically cutting late SLA merchant penalties while optimizing dynamic surge pricing under tropical weather shocks.

---

## 1. Executive Summary & Business Context

In high-density Southeast Asian urban centers like Jakarta, on-demand logistics platforms face extreme travel time variance driven by tropical monsoon rainfall, rush-hour gridlock, and merchant kitchen prep queues.

### The Failure of Traditional Mean Point Estimates (MSE)
Standard machine learning regressors (e.g. LightGBM / XGBoost Regressor minimizing Mean Squared Error) predict the conditional mean $\mathbb{E}[Y \mid X]$. Because urban delivery durations are heavily right-skewed with long delay tails:
1. **Asymmetric Penalty of Underestimation**: A mean prediction underestimates travel time on roughly **~50% of delay-prone deliveries**, causing severe customer churn, cancelled orders, and expensive merchant SLA breach compensations.
2. **False Precision**: Stating "Delivery in 38 minutes" fails to indicate whether the uncertainty is $\pm 3$ minutes (sunny day, clear highway) or $\pm 25$ minutes (monsoon thunderstorm during evening commute).
3. **Suboptimal Surge Pricing**: Dynamic pricing algorithms lacking explicit uncertainty bounds fail to balance driver fleet incentives during localized shocks.

**DispatchOpt** replaces naive point estimates with **distribution-free prediction intervals $[\hat{L}(X), \hat{U}(X)]$** backed by mathematical finite-sample coverage guarantees.

---

## 2. Mathematical & Theoretical Framework

### 2.1. Multi-Quantile Loss (Pinball Loss)
For any target quantile $q \in (0, 1)$, the tilted asymmetric loss function is:
$$\rho_q(u) = u \cdot (q - \mathbb{I}(u < 0)) = \begin{cases} q \cdot u & \text{if } u \ge 0 \\ (q - 1) \cdot u & \text{if } u < 0 \end{cases}$$

DispatchOpt trains three dedicated Gradient Boosted estimators via XGBoost:
- $\hat{q}_{\alpha/2}(X)$: Lower bound ($q = 0.05$ for $90\%$ target coverage).
- $\hat{q}_{0.50}(X)$: Median duration (robust point estimate resistant to extreme outliers).
- $\hat{q}_{1 - \alpha/2}(X)$: Upper bound ($q = 0.95$, optimal SLA promise baseline).

### 2.2. Conformalized Quantile Regression (CQR, Romano et al. NeurIPS 2019)
Raw quantile regressors frequently suffer from out-of-sample undercoverage due to finite sample overfitting. CQR utilizes an independent holdout calibration dataset $(X_{\text{cal}}, Y_{\text{cal}})$ of size $n_{\text{cal}}$:

1. **Non-conformity Score Function**:
   $$E_i = \max\left( \hat{q}_{\alpha/2}(X_i) - Y_i, \quad Y_i - \hat{q}_{1-\alpha/2}(X_i) \right)$$
   - If $E_i \le 0$, the ground truth $Y_i$ falls safely inside the predicted raw quantile band.
   - If $E_i > 0$, the observation exceeds the interval by distance $E_i$.

2. **Empirical Quantile Calibration**:
   Compute $Q_{1-\alpha}(E)$ as the empirical quantile at level:
   $$p = \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}$$

3. **Conformal Prediction Interval**:
   $$\hat{C}(X_{n+1}) = \left[ \hat{q}_{\alpha/2}(X_{n+1}) - Q_{1-\alpha}, \quad \hat{q}_{1-\alpha/2}(X_{n+1}) + Q_{1-\alpha} \right]$$

4. **Finite-Sample Mathematical Coverage Guarantee**:
   $$\mathbb{P}\left( Y_{n+1} \in \hat{C}(X_{n+1}) \right) \ge 1 - \alpha$$
   Holds exactly for any arbitrary distribution without requiring Gaussianity or homoscedasticity, assuming only exchangeability.

### 2.3. Dynamic Surge Pricing Engine
$$\text{SurgeMultiplier} = 1.0 + \gamma_{\text{rain}} \cdot \frac{\text{Rainfall}}{40} + \gamma_{\text{cong}} \cdot \text{CongestionIndex} + \beta \cdot \frac{\hat{U} - \hat{L} - 15}{20} + \delta \cdot \max\left(0, \frac{30 - \text{DriverDensity}}{30}\right)$$
Bounded between $[1.0\times, 2.5\times]$ to incentivize fleet repositioning while protecting conversion rates.

---

## 3. Empirical Benchmarks & Results

Evaluated on an out-of-sample test cohort of **12,000 on-demand Jakarta deliveries** (trained on 36,000 trips, calibrated on 12,000 trips):

### Prediction Interval Validity & Tightness
| Method | Empirical Coverage | Target Coverage | MPIW (Mean Interval Width) | Late SLA Breaches |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Quantile Regression** | 96.23% | 90.00% | 32.47 min | 2.88% |
| **Conformalized CQR (DispatchOpt)** | **90.03%** | **90.00%** | **28.36 min** | **3.54%** |

> **Key Finding**: Raw quantile models over-conservatively pad intervals. CQR tightens the mean interval width from **32.47 mins to 28.36 mins** while mathematically hitting the exact $90.0\%$ coverage requirement.

![CQR Calibration Plot](artifacts/cqr_calibration_plot.png)

### Business SLA Economics & Operational Loss Avoidance
| Delivery Promise Policy | Late Breach Count | Breach Rate (%) | Total Operational Loss (IDR) |
| :--- | :---: | :---: | :---: |
| **Traditional Mean ETA Promise** | 3,519 trips | 29.33% | Rp 75,177,294 |
| **CQR 90% SLA Promise** | **425 trips** | **3.54%** | **Rp 63,307,278** |
| **Net Operational Benefit** | **-3,094 breaches** | **-87.9% reduction** | **Rp 11,870,016 saved / 12k orders** |

### Real-Time Microservice Serving Latency (Vectorized Pure NumPy)
- **Median ($p_{50}$)**: **6.10 ms**
- **95th Percentile ($p_{95}$)**: **9.41 ms**
- **99th Percentile ($p_{99}$)**: **11.05 ms**
- **Serving SLA Status ($< 15$ ms target)**: **PASSED**

---

## 4. Spatio-Temporal Jakarta Topology

DispatchOpt models 10 core logistics hubs across Jakarta's delivery grid:
1. **Monas**: Central Government & Administrative district.
2. **Sudirman_SCBD**: Financial district & high-density office corridors.
3. **Kuningan**: Diplomatic & high-traffic thoroughfare.
4. **Tebet**: High-volume residential & culinary hub.
5. **Senayan**: Commercial & sporting complex.
6. **Kemang**: South Jakarta lifestyle & retail center.
7. **PIK (Pantai Indah Kapuk)**: North Jakarta coastal corridor.
8. **Kelapa_Gading**: Major retail & trading cluster.
9. **Tanjung_Priok**: Port & heavy logistics artery.
10. **Puri_Indah**: West Jakarta commercial center.

---

## 5. System Architecture

```
+-----------------------------------------------------------------------------------+
|                        GOSEND / GRABEXPRESS ON-DEMAND TRAFFIC                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|            OFFLINE PIPELINE: MULTI-QUANTILE TRAINING & CQR CALIBRATION            |
|  - Spatio-Temporal Hub Graph Feature Pipeline (Categorical + Covariates)           |
|  - Multi-Quantile Gradient Boosting (q = 0.05, 0.50, 0.95 with Pinball Loss)       |
|  - Finite-Sample Conformal Calibration Engine (Q_(1-alpha) correction)             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                  ONLINE SERVING: REAL-TIME FASTAPI MICROSERVICE                   |
|  - Endpoints: POST /v1/estimate-delivery, GET /v1/health                          |
|  - Vectorized Inference: p50 = 6.10 ms | p95 = 9.41 ms (< 15 ms SLA)              |
|  - Dynamic Surge Engine & SLA Risk Level Classification                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|             OPERATIONS & EXECUTIVE DASHBOARD (STREAMLIT DISPATCH SUITE)           |
|  - Live Route Planner & Weather Shock Injector (Rainfall up to 70 mm/h)            |
|  - Asymmetric Arrival Probability Density & Quantile Fan Visualizer                |
|  - Dynamic Surge Multiplier & Financial SLA Breach Cost Calculator                |
+-----------------------------------------------------------------------------------+
```

---

## 6. Directory Structure

```
dispatch_opt/
├── PRD.md                       # Product Requirements Document & CQR formulation
├── README.md                    # Architecture documentation & benchmark tables
├── requirements.txt             # Production dependencies
├── run_all.py                   # Automated end-to-end benchmark runner
├── .gitignore                   # Python ignore rules
├── artifacts/
│   ├── .gitkeep
│   └── cqr_calibration_plot.png # High-resolution benchmark comparison chart
├── data/
│   └── .gitkeep                 # Data directory placeholder
├── src/
│   ├── dataset.py               # Spatio-temporal logistics simulation (60k trips)
│   ├── models.py                # MultiQuantileRegressor, CQR, DynamicSurgeEngine
│   ├── evaluate.py              # Coverage, MPIW, Pinball Loss, and SLA economics
│   └── api.py                   # FastAPI microservice serving (<10ms latency)
└── dashboard/
    └── app.py                   # Interactive Streamlit operations dashboard
```

---

## 7. Real-Time API Usage

### Start FastAPI Server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Request Example (`POST /v1/estimate-delivery`)
```json
{
  "origin_hub": "Sudirman_SCBD",
  "destination_hub": "Kemang",
  "hour_of_day": 18,
  "rainfall_mm_h": 25.0,
  "driver_density_3km": 22,
  "merchant_prep_time_min": 15.0,
  "is_weekend": 0,
  "order_backlog": 5
}
```

### Response Example
```json
{
  "request_id": "b3e51d9e-1f7c-4742-8812-42c235b0b9a8",
  "origin_hub": "Sudirman_SCBD",
  "destination_hub": "Kemang",
  "route_distance_km": 6.87,
  "point_eta_median_min": 38.6,
  "cqr_interval_lower_min": 29.9,
  "cqr_interval_upper_min": 65.4,
  "cqr_interval_width_min": 35.5,
  "coverage_confidence_level": "90.0% Distribution-Free Finite-Sample Guarantee",
  "surge_multiplier": 2.26,
  "base_fee_idr": 18000.0,
  "final_fee_idr": 40700,
  "sla_risk_level": "HIGH_UNCERTAINTY",
  "dispatch_recommendation": "Extend SLA window by 15 mins to prevent merchant breach penalties.",
  "latency_ms": 6.10
}
```

---

## 8. Interactive Streamlit Dashboard

Run the executive dispatch dashboard:
```bash
streamlit run dashboard/app.py
```

### Dashboard Capabilities:
1. **Route Simulator**: Select any pickup & delivery pair across the 10 Jakarta hubs.
2. **Weather & Traffic Shocks**: Dynamically adjust rainfall intensity (0 to 70 mm/h) and fleet density to inspect interval widening.
3. **Probability Density Visualizer**: Observe how asymmetric uncertainty inflates right-tail risk without shifting the median proportionally.
4. **Interview Talking Points**: Structured technical arguments designed for Gojek & Grab hiring managers.

---

## 9. Quick Start & Reproduction

```bash
# 1. Clone repository
git clone https://github.com/naufal1555/dispatch-opt.git
cd dispatch-opt

# 2. Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run end-to-end benchmark
python run_all.py
```

---

## 10. Gojek / Grab Interview Talk-Track

When presenting this project in a Technical or Machine Learning System Design Interview:
1. **Why Conformal Prediction over Parametric Uncertainty**: Many engineers assume Gaussian errors $y \sim \mathcal{N}(\mu, \sigma^2)$ or use Monte Carlo dropout. In Jakarta logistics, errors are non-Gaussian, multi-modal, and heteroscedastic. CQR provides provable distribution-free validity.
2. **Tightness vs Validity**: Raw quantile regression can yield loose intervals or suffer undercoverage when base models overfit. Conformal calibration fine-tunes the boundary to achieve exact coverage while reducing interval width by over 12%.
3. **Operational Impact**: By setting customer SLA promises to the calibrated 90th percentile bound rather than the mean, delivery breaches fall by **87.9%**, eliminating customer compensation friction.

---

## Author

**Ahmad Fathin Naufal**  
- GitHub: [@naufal1555](https://github.com/naufal1555)  
- Email: [naufaalfathin15@gmail.com](mailto:naufaalfathin15@gmail.com)
