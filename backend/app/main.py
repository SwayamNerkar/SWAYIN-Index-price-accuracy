from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.app.core.config import settings
from backend.app.core.security import setup_security_middleware
from backend.app.database.connection import init_db
from backend.app.api.router import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    try:
        init_db()
    except Exception as e:
        print(f"Database init warning: {e}")
    yield
    # Shutdown actions

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

setup_security_middleware(app)
app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Welcome to SWAYIN Predictor AI Engine API",
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
