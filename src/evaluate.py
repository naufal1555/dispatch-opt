"""
Spatio-Temporal CQR Evaluation & SLA Benchmarking Engine (DispatchOpt)
Author: Ahmad Fathin Naufal (naufal1555)
Purpose:
  - Computes Empirical Coverage Rate (P(Y in [L, U])).
  - Measures Mean Prediction Interval Width (MPIW) and Quantile Pinball Loss.
  - Benchmarks Mean Regressor vs Raw Quantile vs Conformalized Quantile Regression (CQR).
  - Evaluates SLA Penalty Reductions and plots calibration curves.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Computes Pinball (Quantile) Loss for a specific quantile q."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def evaluate_prediction_interval(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    method_name: str,
) -> Dict[str, Any]:
    """
    Evaluates statistical validity and efficiency of a prediction interval:
      - Coverage rate (should meet or exceed 1 - alpha)
      - Mean Prediction Interval Width (MPIW)
      - Median Prediction Interval Width
    """
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage = float(np.mean(in_interval)) * 100.0

    widths = upper_bound - lower_bound
    mpiw = float(np.mean(widths))
    median_width = float(np.median(widths))

    under_coverage = float(np.mean(y_true > upper_bound)) * 100.0
    over_coverage = float(np.mean(y_true < lower_bound)) * 100.0

    return {
        "method": method_name,
        "coverage_pct": round(coverage, 2),
        "mpiw_min": round(mpiw, 2),
        "median_width_min": round(median_width, 2),
        "late_sla_breach_pct": round(under_coverage, 2),
        "early_delivery_pct": round(over_coverage, 2),
    }


def evaluate_sla_economics(
    y_true: np.ndarray,
    promised_sla_times: np.ndarray,
    policy_name: str,
    breach_penalty_idr: float = 20000.0,
    buffer_slack_penalty_idr: float = 250.0,
) -> Dict[str, Any]:
    """
    Evaluates business logistics impact:
      - Late deliveries (y_true > promised_sla) incur direct customer compensation (IDR 20,000).
      - Overly padded promises (promised_sla - y_true) incur customer conversion friction (IDR 250 / min).
    """
    is_breached = y_true > promised_sla_times
    breach_count = int(np.sum(is_breached))
    breach_rate = float(np.mean(is_breached)) * 100.0

    direct_penalty = breach_count * breach_penalty_idr

    # Slack padding cost
    slack_min = np.maximum(0.0, promised_sla_times - y_true)
    slack_cost = float(np.sum(slack_min) * buffer_slack_penalty_idr)
    total_cost = direct_penalty + slack_cost

    return {
        "policy": policy_name,
        "breach_count": breach_count,
        "breach_rate_pct": round(breach_rate, 2),
        "direct_penalty_idr": round(direct_penalty, 0),
        "slack_padding_cost_idr": round(slack_cost, 0),
        "total_economic_loss_idr": round(total_cost, 0),
    }


def plot_cqr_calibration(
    y_test: np.ndarray,
    mean_preds: np.ndarray,
    raw_low: np.ndarray,
    raw_high: np.ndarray,
    cqr_low: np.ndarray,
    cqr_high: np.ndarray,
    save_path: str,
    n_sample_trips: int = 60,
):
    """
    Generates high-resolution visualization comparing Point Mean vs CQR Prediction Intervals.
    """
    indices = np.arange(n_sample_trips)
    y_sub = y_test[:n_sample_trips]
    mean_sub = mean_preds[:n_sample_trips]
    cqr_l_sub = cqr_low[:n_sample_trips]
    cqr_u_sub = cqr_high[:n_sample_trips]

    plt.figure(figsize=(12, 6))

    # Plot CQR shaded interval
    plt.fill_between(
        indices,
        cqr_l_sub,
        cqr_u_sub,
        color="#2E7D32",
        alpha=0.20,
        label="CQR 90% Conformal Interval [Lower, Upper]",
    )

    # Plot Lower and Upper CQR lines
    plt.plot(indices, cqr_u_sub, color="#1B5E20", linestyle="--", linewidth=1.2, alpha=0.7)
    plt.plot(indices, cqr_l_sub, color="#1B5E20", linestyle="--", linewidth=1.2, alpha=0.7)

    # Plot Mean Regressor (Traditional point estimate)
    plt.plot(
        indices,
        mean_sub,
        color="#D32F2F",
        linestyle="-.",
        linewidth=1.8,
        label="Traditional Mean Regressor (Point Estimate)",
    )

    # Plot Ground Truth Actual Delivery Times
    plt.scatter(
        indices,
        y_sub,
        color="#1565C0",
        s=36,
        zorder=5,
        label="Actual Observed Delivery Duration (Minutes)",
    )

    plt.title(
        "DispatchOpt: Conformalized Quantile Regression vs Traditional Mean ETA",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Sample On-Demand Delivery Trips (GoSend Jakarta)", fontsize=11)
    plt.ylabel("Delivery Duration (Minutes)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    from dataset import DatasetConfig, generate_spatiotemporal_logistics_data, split_train_cal_test
    from models import FeaturePipeline, BaselineMeanRegressor, MultiQuantileRegressor, ConformalizedQuantileRegressor

    cfg = DatasetConfig(n_trips=15000)
    df = generate_spatiotemporal_logistics_data(cfg)
    train_df, cal_df, test_df = split_train_cal_test(df, cfg)

    pipe = FeaturePipeline()
    X_tr = pipe.fit_transform(train_df)
    y_tr = train_df["delivery_duration_min"].values

    X_cal = pipe.transform(cal_df)
    y_cal = cal_df["delivery_duration_min"].values

    X_te = pipe.transform(test_df)
    y_te = test_df["delivery_duration_min"].values

    mean_mod = BaselineMeanRegressor()
    mean_mod.fit(X_tr, y_tr)
    mean_preds = mean_mod.predict(X_te)

    mqr = MultiQuantileRegressor(alpha=0.10)
    mqr.fit(X_tr, y_tr)

    cqr = ConformalizedQuantileRegressor(mqr, alpha=0.10)
    cqr.calibrate(X_cal, y_cal)
    r_low, r_med, r_high, c_low, c_high = cqr.predict(X_te)

    print("Evaluating Prediction Intervals on Test Set (3,000 trips):")
    res_raw = evaluate_prediction_interval(y_te, r_low, r_high, "Raw Quantile Regression")
    res_cqr = evaluate_prediction_interval(y_te, c_low, c_high, "Conformalized CQR (DispatchOpt)")
    print(pd.DataFrame([res_raw, res_cqr]))

    print("\nEvaluating SLA Economic Policy Impact:")
    econ_mean = evaluate_sla_economics(y_te, mean_preds, "Traditional Mean ETA Promise")
    econ_cqr = evaluate_sla_economics(y_te, c_high, "CQR 90% SLA Promise")
    print(pd.DataFrame([econ_mean, econ_cqr]))
