import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.app.core.config import settings
from backend.app.core.logging import logger

Base = declarative_base()

# Safe synchronous and asynchronous engine configuration
db_file = os.path.join(settings.BASE_DIR, "swayin.db")
db_url = f"sqlite:///{db_file}"

engine = create_engine(db_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.warning(f"Database init exception: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
