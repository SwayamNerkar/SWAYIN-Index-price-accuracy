from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

def setup_security_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

class PredictRequest(BaseModel):
    symbol: str = Field(default="^NSEI", description="Ticker symbol e.g. ^NSEI, RELIANCE.NS, AAPL")
    interval: str = Field(default="1d", description="Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1w)")
    period: str = Field(default="5y", description="Lookback period (e.g. 1d, 60d, 1y, 5y)")
    prediction_type: str = Field(default="next_candle", description="next_candle, intraday, daily, swing, trend")
    retrain: bool = Field(default=False, description="Force retrain models")

class TrainRequest(BaseModel):
    symbol: str = Field(default="^NSEI")
    interval: str = Field(default="1d")
    period: str = Field(default="5y")
    epochs: int = Field(default=25)

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    redis: str
