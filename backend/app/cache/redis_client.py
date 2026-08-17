import json
import time
from typing import Any, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger

class MemoryCache:
    def __init__(self):
        self._store = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            data, exp = self._store[key]
            if exp is None or exp > time.time():
                return data
            else:
                del self._store[key]
        return None

    def set(self, key: str, value: Any, expire_seconds: int = 300):
        exp = time.time() + expire_seconds if expire_seconds else None
        self._store[key] = (value, exp)

    def delete(self, key: str):
        self._store.pop(key, None)

class CacheManager:
    def __init__(self):
        self.memory_cache = MemoryCache()
        self.redis_client = None
        self.use_redis = False
        self._init_redis()

    def _init_redis(self):
        try:
            import redis
            r = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1)
            if r.ping():
                self.redis_client = r
                self.use_redis = True
                logger.info("Connected to Redis Cache.")
            else:
                logger.info("Redis ping failed. Using in-memory fallback cache.")
        except Exception as e:
            logger.info(f"Redis not available ({e}). Using in-memory fallback cache.")
            self.use_redis = False

    def get(self, key: str) -> Optional[Any]:
        if self.use_redis and self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception:
                pass
        return self.memory_cache.get(key)

    def set(self, key: str, value: Any, expire_seconds: int = settings.CACHE_EXPIRE_SECONDS):
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(key, expire_seconds, json.dumps(value))
                return
            except Exception:
                pass
        self.memory_cache.set(key, value, expire_seconds)

cache_manager = CacheManager()
