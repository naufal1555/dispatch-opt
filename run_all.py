"""
DispatchOpt End-to-End Pipeline & Evaluation Benchmark
Author: Ahmad Fathin Naufal (naufal1555)
Target: Gojek (GoSend) / Grab (GrabExpress) Logistics Operations
Purpose:
  - Generates 60,000 simulated Jakarta on-demand delivery logs.
  - Trains Baseline Mean Regressor & Multi-Quantile Regressors.
  - Calibrates Conformalized Quantile Regression (CQR) with 90% finite-sample guarantee.
  - Evaluates Coverage Rate, Interval Width (MPIW), and SLA Economic Loss.
  - Generates artifacts/cqr_calibration_plot.png.
  - Benchmarks Real-Time Serving API latency and SLA.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from dataset import DatasetConfig, generate_spatiotemporal_logistics_data, split_train_cal_test
from models import (
    FeaturePipeline,
    BaselineMeanRegressor,
    MultiQuantileRegressor,
    ConformalizedQuantileRegressor,
    DynamicSurgeEngine,
)
from evaluate import (
    compute_pinball_loss,
    evaluate_prediction_interval,
    evaluate_sla_economics,
    plot_cqr_calibration,
)
from api import DeliveryEstimateRequest, ServingEngine


def main():
    print("=" * 80)
    print(" [DISPATCHOPT] SPATIO-TEMPORAL CONFORMALIZED QUANTILE REGRESSION ENGINE")
    print(" Candidate: Ahmad Fathin Naufal (naufal1555)")
    print(" Target: Gojek (GoSend) / Grab (GrabExpress) Logistics Data Science")
    print("=" * 80)

    # 1. Dataset Generation
    print("\n[STEP 1] Generating Simulated Jakarta Delivery Logs (60,000 trips)...")
    cfg = DatasetConfig(n_trips=60000, random_seed=42)
    df = generate_spatiotemporal_logistics_data(cfg)
    train_df, cal_df, test_df = split_train_cal_test(df, cfg)
    print(f"  -> Total Trips: {len(df):,}")
    print(f"  -> Split: Train = {len(train_df):,} | Calibration = {len(cal_df):,} | Test = {len(test_df):,}")
    print(f"  -> Duration Stats (Min/Mean/Median/Max): {df['delivery_duration_min'].min():.1f} / {df['delivery_duration_min'].mean():.1f} / {df['delivery_duration_min'].median():.1f} / {df['delivery_duration_min'].max():.1f} mins")

    # 2. Feature Transformation
    print("\n[STEP 2] Preprocessing Spatio-Temporal Graph & Categorical Hubs...")
    pipe = FeaturePipeline()
    X_train = pipe.fit_transform(train_df)
    y_train = train_df["delivery_duration_min"].values

    X_cal = pipe.transform(cal_df)
    y_cal = cal_df["delivery_duration_min"].values

    X_test = pipe.transform(test_df)
    y_test = test_df["delivery_duration_min"].values
    print(f"  [OK] Encoded Feature Vector Dimension: {X_train.shape[1]} features")

    # 3. Model Training
    print("\n[STEP 3] Training Estimators...")
    
    # 3a. Baseline Mean Regressor (MSE)
    t0 = time.time()
    mean_mod = BaselineMeanRegressor(random_seed=42)
    mean_mod.fit(X_train, y_train)
    t_mean = time.time() - t0
    print(f"  [OK] Baseline Mean Regressor (XGBoost MSE) trained in {t_mean:.2f}s")

    # 3b. Multi-Quantile Regressor
    t0 = time.time()
    mqr = MultiQuantileRegressor(alpha=0.10, random_seed=42)
    mqr.fit(X_train, y_train)
    t_mqr = time.time() - t0
    print(f"  [OK] Multi-Quantile Regressors (q=0.05, 0.50, 0.95) trained in {t_mqr:.2f}s")

    # 4. Conformal Calibration (CQR)
    print("\n[STEP 4] Performing Conformal Calibration on Holdout Calibration Set...")
    t0 = time.time()
    cqr = ConformalizedQuantileRegressor(mqr, alpha=0.10)
    cqr.calibrate(X_cal, y_cal)
    t_cqr = time.time() - t0
    print(f"  [OK] Conformal Calibration completed in {t_cqr:.2f}s")
    print(f"  -> Conformal Correction Term Q_(1-alpha): {cqr.q_correction:+.3f} mins")

    # 5. Out-of-Sample Predictions & Evaluation
    print("\n[STEP 5] Evaluating on Out-of-Sample Test Set (12,000 trips)...")
    mean_preds = mean_mod.predict(X_test)
    raw_low, raw_med, raw_high, cqr_low, cqr_high = cqr.predict(X_test)

    # Pinball Losses
    pb_low = compute_pinball_loss(y_test, raw_low, q=0.05)
    pb_med = compute_pinball_loss(y_test, raw_med, q=0.50)
    pb_high = compute_pinball_loss(y_test, raw_high, q=0.95)
    print(f"  -> Pinball Losses: q_0.05 = {pb_low:.3f} | q_0.50 = {pb_med:.3f} | q_0.95 = {pb_high:.3f}")

    # Interval Coverage & Efficiency
    res_raw = evaluate_prediction_interval(y_test, raw_low, raw_high, "Raw Quantile Regression")
    res_cqr = evaluate_prediction_interval(y_test, cqr_low, cqr_high, "Conformalized CQR (DispatchOpt)")

    print("-" * 80)
    print(f"{'Method':<32} | {'Coverage (%)':<14} | {'MPIW (Width)':<14} | {'SLA Breach (%)':<14}")
    print("-" * 80)
    for res in [res_raw, res_cqr]:
        print(f"{res['method']:<32} | {res['coverage_pct']:>11.2f}% | {res['mpiw_min']:>10.2f} min | {res['late_sla_breach_pct']:>11.2f}%")
    print("-" * 80)

    # SLA Economic Policy Comparison
    print("\n[STEP 6] Simulating Business SLA Breach Penalties (12,000 deliveries)...")
    econ_mean = evaluate_sla_economics(y_test, mean_preds, "Traditional Mean ETA Promise")
    econ_cqr = evaluate_sla_economics(y_test, cqr_high, "CQR 90% SLA Promise")

    print("-" * 90)
    print(f"{'Delivery Promise Policy':<30} | {'Breach Count':<14} | {'Breach Rate':<13} | {'Total Economic Loss':<20}")
    print("-" * 90)
    for econ in [econ_mean, econ_cqr]:
        print(f"{econ['policy']:<30} | {econ['breach_count']:>11,} trips | {econ['breach_rate_pct']:>10.2f}% | Rp {econ['total_economic_loss_idr']:>15,.0f}")
    print("-" * 90)

    breach_drop = ((econ_mean["breach_count"] - econ_cqr["breach_count"]) / econ_mean["breach_count"]) * 100.0
    loss_saved = econ_mean["total_economic_loss_idr"] - econ_cqr["total_economic_loss_idr"]
    print(f"  -> SLA Breach Reduction: -{breach_drop:.1f}% reduction in late deliveries!")
    print(f"  -> Direct Operational Loss Saved: Rp {loss_saved:,.0f} per 12,000 orders")

    # 6. Save Plot to Artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    plot_path = os.path.join(artifacts_dir, "cqr_calibration_plot.png")
    plot_cqr_calibration(
        y_test=y_test,
        mean_preds=mean_preds,
        raw_low=raw_low,
        raw_high=raw_high,
        cqr_low=cqr_low,
        cqr_high=cqr_high,
        save_path=plot_path,
        n_sample_trips=60,
    )
    print(f"\n[OK] High-Resolution CQR Calibration Plot saved to {plot_path}")

    # 7. Real-Time Serving API Latency Benchmark
    print("\n[STEP 7] Benchmarking Real-Time Serving Microservice (< 15ms target)...")
    engine = ServingEngine()
    engine.load_pipeline(pipe, cqr)

    test_req = DeliveryEstimateRequest(
        origin_hub="Sudirman_SCBD",
        destination_hub="Kemang",
        hour_of_day=18,
        rainfall_mm_h=25.0,
        driver_density_3km=22,
        merchant_prep_time_min=15.0,
        is_weekend=0,
    )

    res = engine.estimate(test_req)
    print(f"  -> Sample Inference Result:")
    print(f"     - Route: {res.origin_hub} -> {res.destination_hub} ({res.route_distance_km} km)")
    print(f"     - Point Estimate (Median): {res.point_eta_median_min} mins")
    print(f"     - Guaranteed 90% SLA Window: [{res.cqr_interval_lower_min} - {res.cqr_interval_upper_min}] mins")
    print(f"     - Dynamic Surge Multiplier: {res.surge_multiplier}x (Fee: Rp {res.final_fee_idr:,})")
    print(f"     - SLA Risk Level: {res.sla_risk_level}")

    # Latency benchmark (100 runs)
    latencies = []
    for _ in range(100):
        r = engine.estimate(test_req)
        latencies.append(r.latency_ms)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    print(f"\n[SLA BENCHMARK] Real-Time Inference Latency (100 runs):")
    print(f"  - Median (p50): {p50:.2f} ms")
    print(f"  - 95th Percentile (p95): {p95:.2f} ms")
    print(f"  - 99th Percentile (p99): {p99:.2f} ms")
    print(f"  -> Serving SLA (< 15ms target for p95): {'PASSED' if p95 < 15.0 else 'CHECK'} (p95 = {p95:.2f} ms)")

    print("\n" + "=" * 80)
    print(" [SUCCESS] ALL DISPATCHOPT PIPELINE STEPS COMPLETED WITHOUT ERROR")
    print(" Ready for Production Deployment & Interview Presentation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
