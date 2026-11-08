"""
Conformalized Quantile Regression & Dynamic Surge Models (DispatchOpt)
Author: Ahmad Fathin Naufal (naufal1555)
Reference: Romano et al. (NeurIPS 2019) "Conformalized Quantile Regression"
Purpose:
  - Feature pipeline with OneHot encoding for Jakarta logistics hubs.
  - Baseline Mean MSE Regressor for standard point-estimate ETA.
  - Gradient Boosted Multi-Quantile Regressors (q=0.05, 0.50, 0.95).
  - Conformal Calibration Engine with finite-sample coverage guarantees (1 - alpha >= 90%).
  - Dynamic Surge Multiplier Engine balancing uncertainty and supply deficit.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

from dataset import HUB_NAMES


NUMERIC_FEATURES = [
    "route_distance_km",
    "hour_of_day",
    "is_rush_hour",
    "is_weekend",
    "rainfall_mm_h",
    "driver_density_3km",
    "order_backlog",
    "merchant_prep_time_min",
    "congestion_index",
]

CATEGORICAL_FEATURES = ["origin_hub", "destination_hub"]


class FeaturePipeline:
    """Preprocesses categorical hubs and numerical covariates into a dense array."""

    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        categories=[HUB_NAMES, HUB_NAMES],
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                    CATEGORICAL_FEATURES,
                ),
                ("num", "passthrough", NUMERIC_FEATURES),
            ]
        )
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        feats = self.preprocessor.fit_transform(df)
        self.is_fitted = True
        return np.asarray(feats, dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transforming.")
        return np.asarray(self.preprocessor.transform(df), dtype=np.float32)


class BaselineMeanRegressor:
    """Standard supervised ETA regressor minimizing Mean Squared Error (MSE)."""

    def __init__(self, random_seed: int = 42):
        self.model = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=random_seed,
            objective="reg:squarederror",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MultiQuantileRegressor:
    """
    Gradient Boosted Quantile Regressors targeting:
      - Lower Quantile (alpha / 2, default 0.05)
      - Median (0.50, robust point estimate)
      - Upper Quantile (1 - alpha / 2, default 0.95)
    """

    def __init__(self, alpha: float = 0.10, random_seed: int = 42):
        self.alpha = alpha
        self.q_low = alpha / 2.0        # 0.05 for 90% target interval
        self.q_med = 0.50
        self.q_high = 1.0 - (alpha / 2.0)  # 0.95

        self.model_low = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            objective="reg:quantileerror",
            quantile_alpha=self.q_low,
            random_state=random_seed,
            n_jobs=-1,
        )
        self.model_med = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            objective="reg:quantileerror",
            quantile_alpha=self.q_med,
            random_state=random_seed + 1,
            n_jobs=-1,
        )
        self.model_high = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            objective="reg:quantileerror",
            quantile_alpha=self.q_high,
            random_state=random_seed + 2,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model_low.fit(X, y)
        self.model_med.fit(X, y)
        self.model_high.fit(X, y)

    def predict_quantiles(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (pred_low, pred_med, pred_high)."""
        pred_low = self.model_low.predict(X)
        pred_med = self.model_med.predict(X)
        pred_high = self.model_high.predict(X)

        # Enforce monotonic quantile order: low <= med <= high
        pred_low = np.minimum(pred_low, pred_med)
        pred_high = np.maximum(pred_high, pred_med)

        return pred_low, pred_med, pred_high


class ConformalizedQuantileRegressor:
    """
    Implements Conformalized Quantile Regression (CQR, Romano et al. 2019).
    Guarantees finite-sample coverage P(Y in C(X)) >= 1 - alpha on exchangeable data.
    """

    def __init__(self, multi_quantile_model: MultiQuantileRegressor, alpha: float = 0.10):
        self.mqr = multi_quantile_model
        self.alpha = alpha
        self.q_correction = 0.0
        self.is_calibrated = False

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Computes non-conformity scores on calibration set:
          E_i = max(q_low(X_i) - y_i, y_i - q_high(X_i))
        and calculates empirical (1 - alpha) quantile.
        """
        q_low, _, q_high = self.mqr.predict_quantiles(X_cal)

        # Non-conformity score: negative if inside interval, positive if outside
        scores = np.maximum(q_low - y_cal, y_cal - q_high)

        n = len(y_cal)
        # Finite sample conformal quantile with inflation factor (n + 1) / n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(1.0, max(0.0, level))

        # Quantile method
        self.q_correction = float(np.quantile(scores, level, method="higher"))
        self.is_calibrated = True

    def predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          raw_low, raw_med, raw_high, cqr_low, cqr_high
        """
        if not self.is_calibrated:
            raise RuntimeError("CQR model must be calibrated on holdout set before prediction.")

        raw_low, raw_med, raw_high = self.mqr.predict_quantiles(X)

        # Conformalized bounds
        cqr_low = np.maximum(5.0, raw_low - self.q_correction)
        cqr_high = raw_high + self.q_correction

        return raw_low, raw_med, raw_high, cqr_low, cqr_high


class DynamicSurgeEngine:
    """
    Computes dynamic surge price multiplier & SLA delivery commitment:
      - Balances rain, congestion, driver deficit, and delivery uncertainty.
    """

    @staticmethod
    def calculate_surge(
        rainfall_mm_h: float,
        congestion_index: float,
        driver_density: int,
        interval_width_min: float,
        base_fee_idr: float = 18000.0,
    ) -> Dict[str, Any]:
        # Uncertainty ratio compared to standard baseline width of 15 minutes
        uncertainty_factor = np.clip((interval_width_min - 15.0) / 20.0, 0.0, 0.50)
        
        # Weather & traffic component
        weather_factor = np.clip(rainfall_mm_h / 40.0, 0.0, 0.40)
        traffic_factor = np.clip(congestion_index * 0.40, 0.0, 0.40)
        
        # Supply scarcity factor (density below 30 drivers incurs surge)
        supply_deficit = max(0.0, (30.0 - driver_density) / 30.0) * 0.35

        total_surge = 1.0 + weather_factor + traffic_factor + supply_deficit + uncertainty_factor
        surge_multiplier = round(float(np.clip(total_surge, 1.0, 2.5)), 2)
        final_fee = int(np.round(base_fee_idr * surge_multiplier, -2))

        # Determine SLA Risk Level
        if interval_width_min > 30.0 or congestion_index > 0.70:
            sla_risk = "HIGH_UNCERTAINTY"
            recommendation = "Extend SLA window by 15 mins to prevent merchant breach penalties."
        elif interval_width_min > 18.0 or congestion_index > 0.40:
            sla_risk = "MODERATE_UNCERTAINTY"
            recommendation = "Standard dynamic buffer applied; dispatch nearest available driver."
        else:
            sla_risk = "LOW_UNCERTAINTY"
            recommendation = "High-confidence route; guaranteed on-time delivery candidate."

        return {
            "surge_multiplier": surge_multiplier,
            "base_fee_idr": base_fee_idr,
            "final_fee_idr": final_fee,
            "sla_risk": sla_risk,
            "recommendation": recommendation,
        }


if __name__ == "__main__":
    from dataset import DatasetConfig, generate_spatiotemporal_logistics_data, split_train_cal_test

    print("[TEST] Training and calibrating CQR pipeline...")
    cfg = DatasetConfig(n_trips=15000)
    df = generate_spatiotemporal_logistics_data(cfg)
    train_df, cal_df, test_df = split_train_cal_test(df, cfg)

    pipe = FeaturePipeline()
    X_train = pipe.fit_transform(train_df)
    y_train = train_df["delivery_duration_min"].values

    X_cal = pipe.transform(cal_df)
    y_cal = cal_df["delivery_duration_min"].values

    X_test = pipe.transform(test_df)
    y_test = test_df["delivery_duration_min"].values

    print("Fitting Multi-Quantile Regressor...")
    mqr = MultiQuantileRegressor(alpha=0.10)
    mqr.fit(X_train, y_train)

    print("Calibrating Conformal Quantile Regressor...")
    cqr = ConformalizedQuantileRegressor(mqr, alpha=0.10)
    cqr.calibrate(X_cal, y_cal)
    print(f"Calibration completed! Conformal Correction Q_(1-alpha): {cqr.q_correction:.3f} mins")

    r_low, r_med, r_high, c_low, c_high = cqr.predict(X_test[:5])
    print("\nSample Predictions (minutes):")
    for i in range(5):
        print(f"Trip {i}: Actual={y_test[i]:.1f} | Raw=[{r_low[i]:.1f}, {r_high[i]:.1f}] | CQR 90%=[{c_low[i]:.1f}, {c_high[i]:.1f}] | Median={r_med[i]:.1f}")
