"""
Spatio-Temporal Dispatch Dataset Generator (DispatchOpt)
Author: Ahmad Fathin Naufal (naufal1555)
Target: GoSend / GrabExpress Jakarta Logistics Simulation
Purpose:
  - Models Jakarta road network hubs with geographic coordinates and distance tortuosity.
  - Simulates dynamic traffic, rush-hour bottlenecks, tropical monsoon rainfall shocks, and merchant prep time.
  - Generates heavy-tailed right-skewed delivery durations suitable for Conformalized Quantile Regression.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


# 10 Major Jakarta Logistics Hubs & Coordinates
JAKARTA_HUBS: Dict[str, Tuple[float, float]] = {
    "Monas": (-6.1754, 106.8272),
    "Sudirman_SCBD": (-6.2238, 106.8080),
    "Kuningan": (-6.2297, 106.8294),
    "Tebet": (-6.2343, 106.8524),
    "Senayan": (-6.2215, 106.7990),
    "Kemang": (-6.2690, 106.8152),
    "PIK": (-6.1089, 106.7381),
    "Kelapa_Gading": (-6.1584, 106.9099),
    "Tanjung_Priok": (-6.1130, 106.8827),
    "Puri_Indah": (-6.1873, 106.7348),
}

HUB_NAMES = list(JAKARTA_HUBS.keys())


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two coordinates in kilometers."""
    r = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return float(r * c)


@dataclass(frozen=True)
class DatasetConfig:
    n_trips: int = 60000
    random_seed: int = 42
    train_ratio: float = 0.60
    cal_ratio: float = 0.20
    test_ratio: float = 0.20


def generate_spatiotemporal_logistics_data(config: DatasetConfig = DatasetConfig()) -> pd.DataFrame:
    """
    Generates realistic Jakarta on-demand delivery logs with heavy-tailed delivery times.
    """
    rng = np.random.default_rng(config.random_seed)
    n = config.n_trips

    # 1. Randomly sample origin and destination hubs (different hubs)
    orig_indices = rng.integers(0, len(HUB_NAMES), size=n)
    dest_offsets = rng.integers(1, len(HUB_NAMES), size=n)
    dest_indices = (orig_indices + dest_offsets) % len(HUB_NAMES)

    origins = [HUB_NAMES[i] for i in orig_indices]
    destinations = [HUB_NAMES[i] for i in dest_indices]

    # Calculate actual road distance with urban tortuosity (factor 1.25 - 1.55)
    base_distances = np.zeros(n, dtype=np.float32)
    for i in range(n):
        o_lat, o_lon = JAKARTA_HUBS[origins[i]]
        d_lat, d_lon = JAKARTA_HUBS[destinations[i]]
        base_distances[i] = haversine_distance_km(o_lat, o_lon, d_lat, d_lon)

    tortuosity = rng.uniform(1.25, 1.55, size=n)
    route_distance_km = np.round(base_distances * tortuosity, 2)

    # 2. Temporal & Environmental Features
    # Hourly distribution with peaks at lunch (11-13) and dinner/commute (17-20)
    hour_probs = np.array([
        0.01, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04, 0.07, 0.06, 0.05,
        0.06, 0.08, 0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.08, 0.07,
        0.05, 0.04, 0.02, 0.015
    ])
    hour_probs /= hour_probs.sum()
    hour_of_day = rng.choice(np.arange(24), size=n, p=hour_probs)
    is_rush_hour = ((hour_of_day >= 7) & (hour_of_day <= 9)) | ((hour_of_day >= 17) & (hour_of_day <= 20))
    is_weekend = rng.choice([0, 1], size=n, p=[0.72, 0.28])

    # Tropical rainfall intensity in mm/hour (zero-inflated exponential)
    is_raining = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    rainfall_mm_h = np.round(is_raining * rng.exponential(scale=18.0, size=n), 1)
    rainfall_mm_h = np.clip(rainfall_mm_h, 0.0, 75.0)

    # 3. Supply & Merchant Operational Features
    # Active driver fleet density in 3km pickup radius
    driver_density_radius_3km = rng.integers(12, 75, size=n)
    # Order backlog at pickup cluster
    order_backlog_queue = rng.poisson(lam=4.2, size=n)
    # Merchant food preparation time in minutes (LogNormal distribution)
    merchant_prep_time_min = np.round(rng.lognormal(mean=2.4, sigma=0.4, size=n), 1)
    merchant_prep_time_min = np.clip(merchant_prep_time_min, 4.0, 45.0)

    # 4. Congestion Index & Urban Speed Modeling
    # High congestion hubs: Kuningan, Sudirman_SCBD, Tebet
    high_congestion_hubs = {"Sudirman_SCBD", "Kuningan", "Tebet"}
    hub_congestion_penalty = np.array([
        0.25 if (orig in high_congestion_hubs or dest in high_congestion_hubs) else 0.0
        for orig, dest in zip(origins, destinations)
    ])

    congestion_index = (
        0.35 * is_rush_hour
        + 0.25 * (rainfall_mm_h / 50.0)
        + 0.15 * hub_congestion_penalty
        + 0.15 * (1.0 - (driver_density_radius_3km / 75.0))
        + 0.10 * rng.uniform(0.0, 0.2, size=n)
    )
    congestion_index = np.clip(congestion_index, 0.05, 1.0)

    # Effective transit speed in km/h:
    # Clear road speed: ~32 km/h
    # Heavy rush/rain speed: ~10-14 km/h
    effective_speed_kmh = 32.0 * (1.0 - 0.65 * congestion_index)
    transit_time_base_min = (route_distance_km / np.maximum(effective_speed_kmh, 8.0)) * 60.0

    # 5. Right-skewed Stochastic Uncertainty Shock (Accidents, Flash floods, Police roadblocks)
    # Probability of extreme delay shock increases with rain and rush hour
    shock_prob = np.clip(0.04 + 0.15 * (rainfall_mm_h > 20.0) + 0.08 * is_rush_hour, 0.04, 0.40)
    has_shock = rng.random(size=n) < shock_prob
    shock_delay_min = np.where(has_shock, rng.exponential(scale=14.0, size=n) + 5.0, 0.0)

    # Total observed delivery duration (minutes)
    total_delivery_duration_min = np.round(
        merchant_prep_time_min + transit_time_base_min + shock_delay_min + rng.normal(0, 1.5, size=n),
        1
    )
    total_delivery_duration_min = np.clip(total_delivery_duration_min, 10.0, 150.0)

    # Build DataFrame
    df = pd.DataFrame({
        "trip_id": [f"GOSEND_JKT_{i:06d}" for i in range(n)],
        "origin_hub": origins,
        "destination_hub": destinations,
        "route_distance_km": route_distance_km,
        "hour_of_day": hour_of_day,
        "is_rush_hour": is_rush_hour.astype(int),
        "is_weekend": is_weekend,
        "rainfall_mm_h": rainfall_mm_h,
        "driver_density_3km": driver_density_radius_3km,
        "order_backlog": order_backlog_queue,
        "merchant_prep_time_min": merchant_prep_time_min,
        "congestion_index": np.round(congestion_index, 3),
        "delivery_duration_min": total_delivery_duration_min,
    })

    return df


def split_train_cal_test(
    df: pd.DataFrame,
    config: DatasetConfig = DatasetConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into Training (for quantile fitting), Calibration (for CQR conformity scores),
    and Out-of-sample Testing.
    """
    n = len(df)
    n_train = int(n * config.train_ratio)
    n_cal = int(n * config.cal_ratio)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    cal_df = df.iloc[n_train : n_train + n_cal].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_cal :].copy().reset_index(drop=True)

    return train_df, cal_df, test_df


if __name__ == "__main__":
    print("[TEST] Generating 60,000 synthetic GoSend delivery logs...")
    cfg = DatasetConfig(n_trips=60000)
    data = generate_spatiotemporal_logistics_data(cfg)
    print(f"Generated {len(data):,} rows. Sample head:")
    print(data.head(3))
    print("\nSummary Statistics of Delivery Duration (minutes):")
    print(data["delivery_duration_min"].describe())
    tr, cal, te = split_train_cal_test(data, cfg)
    print(f"\nSplit Sizes -> Train: {len(tr):,} | Calibration: {len(cal):,} | Test: {len(te):,}")
