#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class PostgresClient:
    """
    Conexão e operações em PostgreSQL para candles H1 (estilo do seu Mongo Manager).

    - Cria tabela 'candles' (PK: symbol,timeframe,time)
    - Upsert de DataFrame (ON CONFLICT DO UPDATE)
    - Ingestão incremental com preenchimento de gaps (placeholders)
    - Leitura para DataFrame (limit=-1 => todos, ordenado antigo->recente)
    """

    def __init__(
        self,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        db: Optional[str] = None,
        table: str = "candles",
        schema: Optional[str] = None,     # ex: "public" (default do PG)
    ):
        self.user = user or os.getenv("PGUSER") or os.getenv("POSTGRES_USER") or "postgres"
        self.password = password or os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD") or "postgres"
        self.host = host or os.getenv("PGHOST", "localhost")
        self.port = port or os.getenv("PGPORT", "5432")
        self.db = db or os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB") or "marketdata"
        self.schema = schema or os.getenv("PGSCHEMA") or "public"
        self.table = table

        self._engine: Optional[Engine] = None

    # -------------------- conexão --------------------
    def connect(self):
        url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
        self._engine = create_engine(url, future=True)
        self._create_schema_and_table()
        with self._engine.connect() as con:
            con.execute(text("SELECT 1"))
        print(f"✅ Conectado em postgresql://{self.host}:{self.port}/{self.db} (schema={self.schema})")

    def close(self):
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            print("🔌 Conexão Postgres encerrada.")

    # -------------------- DDL --------------------
    def _create_schema_and_table(self):
        assert self._engine is not None
        ddl_schema = f'CREATE SCHEMA IF NOT EXISTS "{self.schema}";'
        ddl_table = f"""
        CREATE TABLE IF NOT EXISTS "{self.schema}"."{self.table}" (
            symbol       TEXT        NOT NULL,
            timeframe    TEXT        NOT NULL,
            "time"       TIMESTAMPTZ NOT NULL,  -- hora cheia em UTC
            open         DOUBLE PRECISION NOT NULL,
            high         DOUBLE PRECISION NOT NULL,
            low          DOUBLE PRECISION NOT NULL,
            close        DOUBLE PRECISION NOT NULL,
            volume       BIGINT     DEFAULT 0,
            placeholder  BOOLEAN    DEFAULT FALSE,
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (symbol, timeframe, "time")
        );
        """
        with self._engine.begin() as con:
            con.execute(text(ddl_schema))
            con.execute(text(ddl_table))

    # -------------------- helpers --------------------
    @staticmethod
    def _ensure_df_columns(df: pd.DataFrame, cols: Sequence[str]):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"faltam colunas no DataFrame: {missing}")

    @staticmethod
    def _utc_floor_hour_series(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, utc=True, errors="coerce").dt.floor("H")

    # -------------------- leitura --------------------
    def get_dataframe(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        assert self._engine is not None
        base = f'SELECT "time", open, high, low, close, volume, placeholder FROM "{self.schema}"."{self.table}" WHERE symbol=:symbol AND timeframe=:tf'
        if limit == -1:
            sql = base + ' ORDER BY "time" ASC'
            params = {"symbol": symbol, "tf": timeframe}
        else:
            # pega mais recente, depois reordena no Pandas
            sql = base + ' ORDER BY "time" DESC LIMIT :lim'
            params = {"symbol": symbol, "tf": timeframe, "lim": int(limit)}

        with self._engine.begin() as con:
            df = pd.read_sql(text(sql), con, params=params)

        if df.empty:
            return df
        return df.sort_values("time").reset_index(drop=True)

    def get_last_time(self, symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
        df = self.get_dataframe(symbol, timeframe, limit=1)
        if df.empty:
            return None
        return pd.to_datetime(df["time"].iloc[-1], utc=True)

    # -------------------- upsert --------------------
    def upsert(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Insere/atualiza candles (espera colunas: time, open, high, low, close, tick_volume opcional)."""
        assert self._engine is not None
        if df is None or df.empty:
            return

        # remove duplicadas e valida
        dff = df.loc[:, ~df.columns.duplicated()].copy()
        self._ensure_df_columns(dff, ["time", "open", "high", "low", "close"])

        # normaliza tempo para UTC hora cheia
        times = self._utc_floor_hour_series(dff["time"])
        if times.isna().any():
            bad = dff.loc[times.isna(), ["time"]].head()
            raise ValueError(f"Valores inválidos na coluna 'time' (exemplos):\n{bad}")

        dff["time"] = times
        if "tick_volume" not in dff.columns:
            dff["tick_volume"] = 0
        if "placeholder" not in dff.columns:
            dff["placeholder"] = False

        # monta lote
        rows = []
        for _, r in dff.iterrows():
            rows.append({
                "symbol": symbol,
                "tf": timeframe,
                "time": r["time"].to_pydatetime(),  # psycopg2 aceita datetime tz-aware
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low":  float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r.get("tick_volume", 0)),
                "placeholder": bool(r.get("placeholder", False)),
                "updated_at": datetime.now(timezone.utc),
            })

        # upsert em lote
        sql = f"""
        INSERT INTO "{self.schema}"."{self.table}"
            (symbol, timeframe, "time", open, high, low, close, volume, placeholder, updated_at)
        VALUES
            (:symbol, :tf, :time, :open, :high, :low, :close, :volume, :placeholder, :updated_at)
        ON CONFLICT (symbol, timeframe, "time") DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low  = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            placeholder = EXCLUDED.placeholder,
            updated_at = EXCLUDED.updated_at;
        """
        try:
            with self._engine.begin() as con:
                con.execute(text(sql), rows)
            print(f"📤 Upsert OK: {len(rows)} linhas")
        except SQLAlchemyError as e:
            raise RuntimeError(f"Falha no upsert: {e}")

    # -------------------- incremental + gaps --------------------
    def incremental_complete_from_df(
        self,
        df_mt: pd.DataFrame,
        symbol: str,
        timeframe: str,
        drop_last: bool = True,
        create_placeholders: bool = True,
    ):
        """
        Insere candles históricos e incrementais, preenchendo gaps por hora.
        - Se banco vazio: insere tudo até a última hora fechada.
        - Se houver dados: grade (last_db+1h..end) + placeholders quando faltar.
        """
        if df_mt is None or df_mt.empty:
            print("⚠️ DF do provedor está vazio.")
            return

        df = df_mt.sort_values("time").reset_index(drop=True).copy()
        df = df.loc[:, ~df.columns.duplicated()].copy()
        if drop_last and len(df) > 0:
            df = df.iloc[:-1].copy()
        if df.empty:
            print("⚠️ Nada a inserir após remover o último candle.")
            return

        # normaliza tempo
        t_utc = self._utc_floor_hour_series(df["time"])
        if t_utc.isna().any():
            bad = df.loc[t_utc.isna(), ["time"]].head()
            raise ValueError(f"Valores inválidos na coluna 'time' (exemplos):\n{bad}")
        df["t_floor"] = t_utc

        last_db = self.get_last_time(symbol, timeframe)  # pd.Timestamp | None
        last_closed = pd.Timestamp.utcnow().floor("H") - pd.Timedelta(hours=1)
        end = min(last_closed, df["t_floor"].max())

        if last_db is None:
            # banco vazio: insere tudo até end
            df_ins = df[df["t_floor"] <= end].copy()
            if df_ins.empty:
                print("ℹ️ Nada a inserir (corte por última hora fechada).")
                return
            print(f"📦 Banco vazio → inserindo {len(df_ins)} candles.")
            self.upsert(df_ins.rename(columns={"t_floor": "time"}), symbol, timeframe)
            return

        start = (last_db + pd.Timedelta(hours=1)).tz_convert("UTC")
        if start > end:
            print("ℹ️ Banco já atualizado até a última hora fechada.")
            return

        # grade completa
        grid = pd.DataFrame({"time": pd.date_range(start=start, end=end, freq="H", tz="UTC")})
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0

        df_real = df[["t_floor", "open", "high", "low", "close", "tick_volume"]].rename(columns={"t_floor": "time"})
        merged = grid.merge(df_real, on="time", how="left")

        if create_placeholders:
            merged["placeholder"] = merged["open"].isna()
            merged["tick_volume"] = merged["tick_volume"].fillna(0)
        else:
            merged = merged.dropna(subset=["open", "high", "low", "close"])

        print(f"🧱 Inserindo/updatando {len(merged)} horas (placeholders={create_placeholders}).")
        self.upsert(merged, symbol, timeframe)
