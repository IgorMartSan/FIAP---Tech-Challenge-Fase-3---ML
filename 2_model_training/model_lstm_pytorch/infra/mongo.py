#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from pymongo import MongoClient, ASCENDING, UpdateOne
try:
    from dotenv import load_dotenv
    load_dotenv()  # opcional
except Exception:
    pass


class MongoClientManager:
    """
    Conexão Mongo no estilo MetaTrader:
      - índice único (symbol, timeframe, time)
      - upsert incremental
      - completar horas faltantes (placeholders)
      - leitura (limit=-1 => todos)
    """

    def __init__(
        self,
        user: str = None,
        password: str = None,
        host: str = None,
        port: str = None,
        db: str = None,
        coll: str = "candles",
    ):
        self.user = user or os.getenv("MONGO_USER") or os.getenv("MONGO_INITDB_ROOT_USERNAME")
        self.password = password or os.getenv("MONGO_PASS") or os.getenv("MONGO_INITDB_ROOT_PASSWORD")
        self.host = host or os.getenv("MONGO_HOST", "localhost")
        self.port = port or os.getenv("MONGO_PORT", "27017")
        self.db_name = db or os.getenv("MONGO_DB") or os.getenv("MONGO_INITDB_DATABASE", "marketdata")
        self.coll_name = coll or os.getenv("MONGO_COLL", "candles")

        if self.user and self.password:
            self.uri = f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}"
        else:
            self.uri = f"mongodb://{self.host}:{self.port}"

        self.client: Optional[MongoClient] = None
        self.db = None
        self.coll = None

    # ------------- CONEXÃO -------------
    def connect(self):
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.coll = self.db[self.coll_name]
        self.coll.create_index(
            [("symbol", ASCENDING), ("timeframe", ASCENDING), ("time", ASCENDING)],
            unique=True,
            name="uniq_symbol_tf_time",
        )
        self.client.admin.command("ping")
        print(f"✅ Conectado em {self.uri}/{self.db_name}.{self.coll_name}")

    def disconnect(self):
        if self.client:
            self.client.close()
            print("🔌 Conexão encerrada.")

    # ------------- HELPERS -------------
    @staticmethod
    def _to_utc_floor_single(ts) -> datetime:
        """
        Converte um único timestamp para UTC e trunca para a hora.
        NÃO usar com Series/DataFrame.
        """
        ts = pd.to_datetime(ts, utc=True)
        return ts.floor("H").to_pydatetime()

    # ------------- CONSULTA -------------
    def get_last_time(self, symbol: str, timeframe: str) -> Optional[datetime]:
        doc = self.coll.find_one({"symbol": symbol, "timeframe": timeframe}, sort=[("time", -1)])
        return doc["time"] if doc else None

    # ------------- UPSERT -------------
    def upsert(self, df: pd.DataFrame, symbol: str, timeframe: str):
        if df is None or len(df) == 0:
            return

        # remove colunas duplicadas
        dff = df.copy()
        dff = dff.loc[:, ~dff.columns.duplicated()].copy()

        need = ["time", "open", "high", "low", "close"]
        missing = [c for c in need if c not in dff.columns]
        if missing:
            raise ValueError(f"faltam colunas: {missing}")

        # normaliza coluna time (garante Series única)
        time_col = dff["time"]
        if isinstance(time_col, pd.DataFrame):
            time_col = time_col.iloc[:, 0]

        # converte para UTC, trunca para hora e transforma em datetime Python
        times = pd.to_datetime(time_col, utc=True, errors="coerce").dt.floor("H")
        if times.isna().any():
            bad = dff.loc[times.isna(), ["time"]].head()
            raise ValueError(f"Valores inválidos na coluna 'time' (exemplos):\n{bad}")

        dff["time"] = times.dt.to_pydatetime()

        if "tick_volume" not in dff.columns:
            dff["tick_volume"] = 0

        ops = []
        for _, r in dff.iterrows():
            doc = {
                "symbol": symbol,
                "timeframe": timeframe,
                "time": r["time"],
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low":  float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r.get("tick_volume", 0)),
                "updated_at": datetime.now(timezone.utc),
                "placeholder": bool(r.get("placeholder", False)),
            }
            ops.append(UpdateOne(
                {"symbol": symbol, "timeframe": timeframe, "time": r["time"]},
                {"$set": doc},
                upsert=True,
            ))
        if ops:
            result = self.coll.bulk_write(ops, ordered=False)
            inserted = (result.upserted_count or 0)
            modified = (result.modified_count or 0)
            print(f"📤 Inseridos/Atualizados: {inserted + modified}")

    # ------------- INCREMENTAL + COMPLETAÇÃO -------------
    def incremental_complete_from_df(
        self,
        df_mt: pd.DataFrame,
        symbol: str,
        timeframe: str,
        drop_last: bool = True,
        create_placeholders: bool = True,
    ):
        """
        Insere candles do MetaTrader completando horas faltantes:
        - Se banco vazio: insere tudo (até a última hora fechada).
        - Se já houver dados: cria grade (last_db+1h..end) e preenche gaps.
        """
        if df_mt is None or len(df_mt) == 0:
            print("⚠️ DF do MetaTrader vazio.")
            return

        df = df_mt.sort_values("time").reset_index(drop=True).copy()
        df = df.loc[:, ~df.columns.duplicated()].copy()

        if drop_last and len(df) > 0:
            df = df.iloc[:-1].copy()
        if len(df) == 0:
            print("⚠️ Nenhum dado restante após remover último candle.")
            return

        # normaliza tempo
        time_col = df["time"]
        if isinstance(time_col, pd.DataFrame):
            time_col = time_col.iloc[:, 0]

        df["time_utc"] = pd.to_datetime(time_col, utc=True, errors="coerce")
        if df["time_utc"].isna().any():
            bad = df.loc[df["time_utc"].isna(), ["time"]].head()
            raise ValueError(f"Valores inválidos na coluna 'time' (exemplos):\n{bad}")

        df["t_floor"] = df["time_utc"].dt.floor("h")

        # normaliza referências
        last_db = self.get_last_time(symbol, timeframe)
        if last_db is not None:
            last_db = pd.to_datetime(last_db, utc=True).to_pydatetime()

        last_closed = pd.Timestamp.utcnow().floor("h").to_pydatetime() - timedelta(hours=1)
        max_df_time = df["t_floor"].max().to_pydatetime()
        end = min(last_closed, max_df_time)

        if last_db is None:
            # inserir tudo até end
            end_ts = pd.Timestamp(end).tz_convert("UTC") if getattr(end, "tzinfo", None) else pd.Timestamp(end, tz="UTC")

            df_ins = df[df["t_floor"] <= end_ts].copy()
            if len(df_ins) == 0:
                print("ℹ️ Nada a inserir (após corte por última hora fechada).")
                return
            print(f"📦 Inserindo {len(df_ins)} candles (banco vazio).")
            self.upsert(df_ins.rename(columns={"t_floor": "time"}), symbol, timeframe)
            return

        start = last_db + timedelta(hours=1)
        if start > end:
            print("ℹ️ Banco já atualizado até a última hora fechada.")
            return

        # grade horária contínua
        full_range = pd.date_range(start=start, end=end, freq="H", tz="UTC")
        grid = pd.DataFrame({"time": full_range})

        # dados reais alinhados à grade
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

    # ------------- LEITURA -------------
    def get_dataframe(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Retorna candles:
        - limit > 0 → últimos `limit`
        - limit == -1 → todos
        Ordenado: mais antigo → mais recente.
        """
        query = {"symbol": symbol, "timeframe": timeframe}
        if limit == -1:
            cursor = self.coll.find(query, sort=[("time", 1)])  # todos em ordem crescente direto do Mongo
        else:
            cursor = self.coll.find(query, sort=[("time", -1)], limit=limit)

        docs = list(cursor)
        if not docs:
            print("⚠️ Nenhum dado encontrado.")
            return pd.DataFrame()

        df = pd.DataFrame(docs).sort_values("time").reset_index(drop=True)
        cols = ["time", "open", "high", "low", "close", "volume", "placeholder"]
        df = df[[c for c in cols if c in df.columns]]
        return df
