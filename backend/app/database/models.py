from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from datetime import datetime
from backend.app.database.connection import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True)
    interval = Column(String(10))
    current_price = Column(Float)
    predicted_price = Column(Float)
    signal = Column(String(20))
    confidence = Column(Float)
    direction = Column(String(10))
    model_used = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON, nullable=True)

class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), index=True)
    version = Column(String(20))
    rmse = Column(Float)
    mae = Column(Float)
    directional_accuracy = Column(Float)
    weight = Column(Float)
    saved_path = Column(String(255))
    updated_at = Column(DateTime, default=datetime.utcnow)

class BacktestRecord(Base):
    __tablename__ = "backtest_records"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True)
    cagr = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
