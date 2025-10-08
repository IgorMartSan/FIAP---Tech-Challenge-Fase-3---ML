# redis_df_client.py
import os, io
import pandas as pd
import redis
from typing import Optional

class RedisConnection:
    def __init__(self, host: Optional[str]=None, port: Optional[int]=None, db: Optional[int]=None,
                 password: Optional[str]=None, socket_timeout: float=5.0):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", 6379))
        self.db = int(db or os.getenv("REDIS_DB", 0))
        self.password = password if password is not None else os.getenv("REDIS_PASSWORD")
        self.client = redis.Redis(host=self.host, port=self.port, db=self.db,
                                  password=self.password, decode_responses=False,
                                  socket_timeout=socket_timeout)
    def ping(self) -> bool:
        return bool(self.client.ping())

def update_df(conn: RedisConnection, key: str, df: pd.DataFrame, ttl_seconds: Optional[int]=None) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame")
    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    buf.seek(0)
    conn.client.set(name=key, value=buf.read(), ex=ttl_seconds)

def get_df(conn: RedisConnection, key: str) -> pd.DataFrame:
    data = conn.client.get(key)
    if data is None:
        raise KeyError(f"Chave não encontrada no Redis: {key}")
    return pd.read_parquet(io.BytesIO(data))
