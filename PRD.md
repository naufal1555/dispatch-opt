# Product Requirements Document (PRD)
## DispatchOpt: Spatio-Temporal Quantile Regression & Conformalized Delivery SLA Engine for GoSend / GrabExpress

**Version**: 1.0.0  
**Author**: Ahmad Fathin Naufal (`naufal1555`)  
**Target Domain**: Gojek (GoSend) / Grab (GrabExpress) Logistics Data Science & Dispatch Optimization  
**Status**: Approved & In Execution  

---

## 1. Problem Statement & Business Context
Dalam logistik *on-demand last-mile delivery* (seperti GoSend Instant/Same-Day dan GrabExpress di Jakarta), akurasi estimasi waktu pengiriman (*Estimated Time of Arrival* / ETA) dan keandalan janji layanan (*Service Level Agreement* / SLA) adalah faktor penentu retensi pelanggan dan mitra pengemudi.

### Cacat Fundamental Model Regresi Rata-Rata (Mean MSE Regression):
Sebagian besar sistem logistik konvensional menggunakan model regresi berbasis *Mean Squared Error* (seperti LightGBM/XGBoost Regressor standar) yang hanya memprediksi **waktu rata-rata** $\mathbb{E}[Y \mid X]$.
1. **Asymmetric Risk of Delay**: Durasi pengiriman di Jakarta memiliki distribusi *heavy-tailed* (miring ke kanan) akibat banjir, kemacetan mendadak, atau antrean parkir merchant. Model rata-rata akan meremehkan (*underestimate*) waktu kirim pada 50% kasus, memicu komplain pelanggan, pembatalan pesanan, dan penalti klaim garansi SLA ke merchant.
2. **False Confidence (No Uncertainty Quantification)**: Janji "paket tiba dalam 35 menit" tidak memberikan informasi apakah ketidakpastian pengiriman berkisar $\pm 3$ menit (cuaca cerah, jalan lancar) atau $\pm 25$ menit (hujan lebat di jam pulang kantor).
3. **Mispriced Dynamic Surge**: Saat cuaca ekstrem atau lonjakan pesanan, penyesuaian tarif dinamis (*surge pricing*) gagal mengimbangi risiko pengemudi terjebak macet, memicu *driver unfulfillment*.

**DispatchOpt** memecahkan masalah ini dengan mengimplementasikan **Conformalized Quantile Regression (CQR)**:
Menghasilkan interval prediksi $[\hat{Y}_{\text{lower}}, \hat{Y}_{\text{upper}}]$ dengan jaminan matematis *finite-sample coverage* terbukti ($1 - \alpha = 90\%$) tanpa bergantung pada asumsi distribusi parametrik normal, dipadukan dengan mesin *Dynamic Surge Pricing* berbasis risiko ketidakpastian.

---

## 2. Success Metrics & Key Performance Indicators (KPIs)

| Kategori Metrik | KPI | Target Minimal | Definisi Matematis / Operasional |
| :--- | :--- | :--- | :--- |
| **Statistical Validity** | **Empirical Coverage Rate** | $\ge 90.0\%$ | Proporsi durasi aktual yang jatuh di dalam interval prediksi $[\hat{L}(X), \hat{U}(X)]$. |
| **Prediction Efficiency** | **Mean Prediction Interval Width (MPIW)** | $\le 16.0$ menit | Rata-rata lebar interval $\frac{1}{n}\sum (\hat{U}_i - \hat{L}_i)$. Semakin rapat, semakin informatif bagi user. |
| **Quantile Accuracy** | **Pinball / Quantile Loss** | $< 1.85$ | $\mathcal{L}_q(y, \hat{y}) = \max(q(y - \hat{y}), (q-1)(y - \hat{y}))$. |
| **Operational Impact** | **SLA Breach Reduction** | $\ge 40\%$ penurunan | Menekan tingkat pengiriman terlambat melebihi komitmen waktu pelanggan. |
| **Serving Performance** | **Real-Time Latency SLA** | $p_{95} < 15$ ms | Latensi inferensi multi-quantile + kalibrasi per request checkout. |

---

## 3. Mathematical & Theoretical Framework

### 3.1. Quantile Loss (Pinball Loss)
Untuk kuantil $q \in (0, 1)$, loss function didefinisikan sebagai:
$$\rho_q(u) = u \cdot (q - \mathbb{I}(u < 0)) = \begin{cases} q \cdot u & \text{jika } u \ge 0 \\ (q - 1) \cdot u & \text{jika } u < 0 \end{cases}$$
Model mengoptimalkan:
- $\hat{q}_{\alpha/2}(X)$: Kuantil bawah (misal $q = 0.05$ untuk 90% interval).
- $\hat{q}_{0.50}(X)$: Median durasi (estimasi titik paling robust terhadap outlier).
- $\hat{q}_{1 - \alpha/2}(X)$: Kuantil atas (misal $q = 0.95$ untuk SLA promise batas aman).

### 3.2. Conformalized Quantile Regression (CQR, Romano et al. NeurIPS 2019)
Model kuantil standar sering kali mengalami *undercoverage* pada data out-of-sample karena *overfitting*. CQR menggunakan set kalibrasi terpisah $(X_{\text{cal}}, Y_{\text{cal}})$ berukuran $n_{\text{cal}}$ untuk menjamin *validity*:

1. **Non-conformity Score**:
   $$E_i = \max\left( \hat{q}_{\alpha/2}(X_i) - Y_i, \quad Y_i - \hat{q}_{1-\alpha/2}(X_i) \right)$$
   Jika $E_i \le 0$, maka $Y_i$ berada di dalam interval kuantil mentah. Jika $E_i > 0$, maka $Y_i$ meleset keluar interval.

2. **Empirical Quantile Calibration**:
   Hitung $Q_{1-\alpha}(E, \text{cal})$ sebagai kuantil ke-$\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil / n_{\text{cal}}$ dari skor $\{E_i\}_{i=1}^{n_{\text{cal}}}$.

3. **Conformal Prediction Interval**:
   $$\hat{C}(X_{n+1}) = \left[ \hat{q}_{\alpha/2}(X_{n+1}) - Q_{1-\alpha}, \quad \hat{q}_{1-\alpha/2}(X_{n+1}) + Q_{1-\alpha} \right]$$

4. **Theoretical Guarantee**:
   $$\mathbb{P}\left( Y_{n+1} \in \hat{C}(X_{n+1}) \right) \ge 1 - \alpha$$
   Berlaku pasti untuk sampel berhingga (*finite sample*) asalkan data bersifat *exchangeable*.

### 3.3. Dynamic Surge Pricing Under Uncertainty
Faktor penyesuaian tarif dinamis:
$$\text{SurgeMultiplier} = 1.0 + \gamma_{\text{rain}} \cdot \frac{\text{Rainfall}}{50} + \gamma_{\text{cong}} \cdot \text{CongestionFactor} + \beta \cdot \frac{\hat{U} - \hat{L}}{\text{BaselineWidth}} - \delta \cdot \text{DriverDensity}$$
Dibatasi pada rentang aman $[1.0\times, 2.5\times]$ untuk menjaga ketersediaan armada tanpa memicu churn pelanggan.

---

## 4. Spatio-Temporal Topology (Jakarta Logistics Grid)

Memodelkan 10 simpul utama logistik Jakarta:
1. **Monas (Pusat / Pemerintahan)**
2. **Sudirman - SCBD (Pusat Bisnis / Perkantoran)**
3. **Kuningan - Rasuna Said (Bisnis & Kedutaan)**
4. **Tebet (Area Padat Kuliner & Hunian)**
5. **Senayan (Area Komersial & Olahraga)**
6. **Kemang (Jakarta Selatan / Kuliner & Ritel)**
7. **Pantai Indah Kapuk / PIK (Jakarta Utara / Residensial Premium)**
8. **Kelapa Gading (Jakarta Utara / Sentra Bisnis & Ritel)**
9. **Tanjung Priok (Pelabuhan / Logistik Berat & Truk)**
10. **Puri Indah / Kembangan (Jakarta Barat)**

Fitur kovariat:
- Jarak rute Haversine ($km$) & Topologi simpul rute.
- Jam pengiriman ($0 - 23$) & Jam Sibuk (Rush Hour 07:00-09:00, 17:00-20:00).
- Intensitas curah hujan ($mm/\text{jam}$).
- Kepadatan armada pengemudi aktif per radius $3\text{km}$.
- Indeks antrean restoran / merchant *prep-time*.

---

## 5. System Execution Steps
1. **Data Synthesis** (`src/dataset.py`): 60.000 log perjalanan pengiriman dengan ground truth delay non-linear dan shock cuaca ekstrem.
2. **CQR Engine** (`src/models.py`): Model Gradient Boosted Multi-Quantile + CQR Calibration Engine + Dynamic Surge Engine.
3. **Benchmarking** (`src/evaluate.py`): Evaluasi Coverage Rate, Interval Width, Pinball Loss, dan SLA penalty avoidance.
4. **Microservice API** (`src/api.py`): FastAPI serving endpoint `/v1/estimate-eta` dengan response < 15ms.
5. **Interactive Dashboard** (`dashboard/app.py`): Streamlit Spatio-Temporal Map visualizer, live route planner, weather shock injector.
6. **Git CI & GitHub Publishing**: Repo publik `naufal1555/dispatch-opt` dengan identitas kontributor bersih tunggal.
