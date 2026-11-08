"""
Real-Time Conformalized ETA & Dynamic Surge Microservice (DispatchOpt)
Author: Ahmad Fathin Naufal (naufal1555)
Purpose:
  - Real-Time inference API (< 15ms SLA target) for GoSend/GrabExpress checkout.
  - Multi-Quantile + CQR 90% guaranteed delivery window.
  - Dynamic surge pricing and operational dispatch recommendations.
"""

from typing import Dict, Any, Optional
import time
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

from dataset import JAKARTA_HUBS, HUB_NAMES, haversine_distance_km
from models import (
    FeaturePipeline,
    MultiQuantileRegressor,
    ConformalizedQuantileRegressor,
    DynamicSurgeEngine,
)


app = FastAPI(
    title="DispatchOpt: Real-Time Conformalized ETA & Dynamic Surge API",
    description="Spatio-Temporal Quantile Regression Engine with Finite-Sample Guarantees for GoSend/GrabExpress",
    version="1.0.0",
)


class DeliveryEstimateRequest(BaseModel):
    origin_hub: str = Field(..., json_schema_extra={"example": "Sudirman_SCBD"})
    destination_hub: str = Field(..., json_schema_extra={"example": "Kemang"})
    hour_of_day: int = Field(..., ge=0, le=23, json_schema_extra={"example": 18})
    rainfall_mm_h: float = Field(0.0, ge=0.0, le=100.0, json_schema_extra={"example": 25.0})
    driver_density_3km: int = Field(35, ge=1, le=150, json_schema_extra={"example": 22})
    merchant_prep_time_min: float = Field(15.0, ge=0.0, le=90.0, json_schema_extra={"example": 12.5})
    is_weekend: int = Field(0, ge=0, le=1, json_schema_extra={"example": 0})
    order_backlog: int = Field(4, ge=0, le=50, json_schema_extra={"example": 5})


class DeliveryEstimateResponse(BaseModel):
    request_id: str
    origin_hub: str
    destination_hub: str
    route_distance_km: float
    point_eta_median_min: float
    cqr_interval_lower_min: float
    cqr_interval_upper_min: float
    cqr_interval_width_min: float
    coverage_confidence_level: str
    surge_multiplier: float
    base_fee_idr: float
    final_fee_idr: int
    sla_risk_level: str
    dispatch_recommendation: str
    latency_ms: float


class ServingEngine:
    def __init__(self):
        self.pipeline: Optional[FeaturePipeline] = None
        self.cqr_model: Optional[ConformalizedQuantileRegressor] = None
        self.is_ready = False

    def load_pipeline(
        self,
        pipeline: FeaturePipeline,
        cqr_model: ConformalizedQuantileRegressor,
    ):
        self.pipeline = pipeline
        self.cqr_model = cqr_model
        self.is_ready = True

    def estimate(self, req: DeliveryEstimateRequest) -> DeliveryEstimateResponse:
        t0 = time.perf_counter()
        if not self.is_ready:
            raise RuntimeError("Serving Engine is not initialized with trained models.")

        if req.origin_hub not in JAKARTA_HUBS:
            raise ValueError(f"Origin hub '{req.origin_hub}' is invalid. Allowed: {HUB_NAMES}")
        if req.destination_hub not in JAKARTA_HUBS:
            raise ValueError(f"Destination hub '{req.destination_hub}' is invalid. Allowed: {HUB_NAMES}")

        # Compute road distance
        o_lat, o_lon = JAKARTA_HUBS[req.origin_hub]
        d_lat, d_lon = JAKARTA_HUBS[req.destination_hub]
        dist_km = round(haversine_distance_km(o_lat, o_lon, d_lat, d_lon) * 1.35, 2)

        is_rush = 1 if ((7 <= req.hour_of_day <= 9) or (17 <= req.hour_of_day <= 20)) else 0
        high_cong = {"Sudirman_SCBD", "Kuningan", "Tebet"}
        hub_penalty = 0.25 if (req.origin_hub in high_cong or req.destination_hub in high_cong) else 0.0

        congestion_index = round(float(np.clip(
            0.35 * is_rush
            + 0.25 * (req.rainfall_mm_h / 50.0)
            + 0.15 * hub_penalty
            + 0.15 * (1.0 - (req.driver_density_3km / 75.0))
            + 0.05,
            0.05,
            1.0
        )), 3)

        # Direct vectorized feature construction (bypassing pandas overhead for <10ms SLA)
        o_idx = HUB_NAMES.index(req.origin_hub)
        d_idx = HUB_NAMES.index(req.destination_hub)
        X_feat = np.zeros((1, 29), dtype=np.float32)
        X_feat[0, o_idx] = 1.0
        X_feat[0, 10 + d_idx] = 1.0
        X_feat[0, 20:] = [
            dist_km,
            req.hour_of_day,
            is_rush,
            req.is_weekend,
            req.rainfall_mm_h,
            req.driver_density_3km,
            req.order_backlog,
            req.merchant_prep_time_min,
            congestion_index,
        ]

        _, raw_med, _, cqr_low, cqr_high = self.cqr_model.predict(X_feat)

        pt_eta = round(float(raw_med[0]), 1)
        low_bound = round(float(cqr_low[0]), 1)
        high_bound = round(float(cqr_high[0]), 1)
        width = round(high_bound - low_bound, 1)

        surge_info = DynamicSurgeEngine.calculate_surge(
            rainfall_mm_h=req.rainfall_mm_h,
            congestion_index=congestion_index,
            driver_density=req.driver_density_3km,
            interval_width_min=width,
        )

        latency = round((time.perf_counter() - t0) * 1000.0, 3)

        return DeliveryEstimateResponse(
            request_id=str(uuid.uuid4()),
            origin_hub=req.origin_hub,
            destination_hub=req.destination_hub,
            route_distance_km=dist_km,
            point_eta_median_min=pt_eta,
            cqr_interval_lower_min=low_bound,
            cqr_interval_upper_min=high_bound,
            cqr_interval_width_min=width,
            coverage_confidence_level="90.0% Distribution-Free Finite-Sample Guarantee",
            surge_multiplier=surge_info["surge_multiplier"],
            base_fee_idr=surge_info["base_fee_idr"],
            final_fee_idr=surge_info["final_fee_idr"],
            sla_risk_level=surge_info["sla_risk"],
            dispatch_recommendation=surge_info["recommendation"],
            latency_ms=latency,
        )


GLOBAL_ENGINE = ServingEngine()


@app.get("/v1/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "DispatchOpt Real-Time Serving API",
        "models_ready": GLOBAL_ENGINE.is_ready,
    }


@app.post("/v1/estimate-delivery", response_model=DeliveryEstimateResponse)
def estimate_delivery(req: DeliveryEstimateRequest):
    if not GLOBAL_ENGINE.is_ready:
        raise HTTPException(status_code=503, detail="Models are initializing. Please retry.")
    try:
        return GLOBAL_ENGINE.estimate(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
