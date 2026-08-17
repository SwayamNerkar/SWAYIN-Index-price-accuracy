from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from backend.app.core.config import settings
from backend.app.core.security import PredictRequest, TrainRequest, HealthResponse
from backend.app.services.market_data import fetch_market_data, get_market_status
from backend.app.prediction.engine import prediction_engine
from backend.app.ml.trainer import ModelTrainerPipeline

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "database": "connected (SQLite/PostgreSQL)",
        "redis": "active (or in-memory fallback)"
    }

@router.post("/predict")
async def predict(req: PredictRequest):
    try:
        df_raw = fetch_market_data(symbol=req.symbol, interval=req.interval, period=req.period)
        res = prediction_engine.predict_full_pipeline(df_raw, symbol=req.symbol, retrain=req.retrain)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    def run_training_job():
        df_raw = fetch_market_data(symbol=req.symbol, interval=req.interval, period=req.period)
        trainer = ModelTrainerPipeline()
        trainer.train_pipeline(df_raw, epochs=req.epochs)

    background_tasks.add_task(run_training_job)
    return {"message": f"Training pipeline initiated asynchronously for {req.symbol}."}

@router.get("/history")
async def get_history(
    symbol: str = Query(default="^NSEI"),
    interval: str = Query(default="1d"),
    period: str = Query(default="1y")
):
    try:
        df = fetch_market_data(symbol=symbol, interval=interval, period=period)
        records = df.reset_index().to_dict(orient="records")
        return {"symbol": symbol, "count": len(records), "data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals")
async def get_signals(symbol: str = Query(default="^NSEI")):
    try:
        df_raw = fetch_market_data(symbol=symbol, interval="1d", period="60d")
        res = prediction_engine.predict_full_pipeline(df_raw, symbol=symbol)
        return {
            "symbol": symbol,
            "signal": res["signal"],
            "direction": res["direction"],
            "confidence": res["confidence"],
            "stop_loss": res["stop_loss"],
            "target_price": res["target_price"],
            "risk_reward_ratio": res["risk_reward_ratio"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics(symbol: str = Query(default="^NSEI")):
    try:
        df_raw = fetch_market_data(symbol=symbol, interval="1d", period="1y")
        res = prediction_engine.predict_full_pipeline(df_raw, symbol=symbol)
        return {
            "symbol": symbol,
            "model_metrics": res["model_metrics"],
            "ensemble_weights": res["ensemble_details"]["model_weights"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_models():
    return {
        "models": [
            "Random Forest", "XGBoost", "LightGBM", "CatBoost",
            "Linear Regression", "Gradient Boosting",
            "LSTM", "Bidirectional LSTM", "GRU", "CNN-LSTM", "Transformer"
        ],
        "ensemble": "Dynamic Performance-Weighted Ensemble"
    }

@router.get("/market-status")
async def market_status(symbol: str = Query(default="^NSEI")):
    return get_market_status(symbol=symbol)

@router.get("/feature-importance")
async def feature_importance(symbol: str = Query(default="^NSEI")):
    try:
        df_raw = fetch_market_data(symbol=symbol, interval="1d", period="60d")
        res = prediction_engine.predict_full_pipeline(df_raw, symbol=symbol)
        return {
            "symbol": symbol,
            "top_features": res["feature_importance"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
