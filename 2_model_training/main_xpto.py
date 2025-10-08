#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBMTrainer — Treinador simples com Darts.LightGBMModel (sem exigir sequência temporal contínua).

Features principais:
- recebe um DataFrame (sem precisar reindexar/ffill),
- treina LightGBMModel com lags (e covariáveis passadas/futuras opcionais),
- faz forecast multi-step e backtest 1-step (h=1, stride configurável),
- gera e salva gráficos (overlay de forecast e backtest),
- exporta relatório JSON com métricas e metadados,
- persiste e recarrega o modelo.

Requisitos:
    pip install "u8darts[torch]" lightgbm pandas matplotlib

Uso rápido:
    from lightgbm_trainer import LightGBMTrainer
    
    trainer = LightGBMTrainer(lags=[1,2,3,6,12,24], output_chunk_length=1)
    trainer.fit(
        df, time_col="time", value_col="close",
        past_cov_cols=["volume"],
        future_cov_cols=["hour","dow","month"],  # opcional
        train_split=0.8,
    )
    trainer.backtest(horizon=1, stride=1)
    forecast_df = trainer.forecast(steps=24, return_df=True)
    trainer.save_run()

Observações:
- Se usar future_cov_cols, o df deve conter essas colunas também para o HORIZONTE futuro desejado (ao prever).
- Este trainer NÃO cria/gera covariáveis futuras automaticamente.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import LightGBMModel
from darts.metrics import mape, smape, rmse, mae


# =========================
# Utilidades
# =========================

def _ensure_sorted_df(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    dff = df.copy()
    dff[time_col] = pd.to_datetime(dff[time_col])
    dff = dff.sort_values(time_col).dropna().reset_index(drop=True)
    return dff


def _timeseries_from_df(
    df: pd.DataFrame,
    time_col: str,
    value_cols: Union[str, List[str]],
) -> TimeSeries:
    return TimeSeries.from_dataframe(df, time_col=time_col, value_cols=value_cols)


@dataclass
class LGBMRunMeta:
    dataset_rows: int
    time_start: str
    time_end: str
    freq_detected: Optional[str]
    lags: Union[int, List[int]]
    lags_past_covariates: Optional[Union[int, List[int]]]
    lags_future_covariates: Optional[Union[int, List[int]]]
    output_chunk_length: int
    multi_models: bool
    random_state: int
    train_split: float
    model_path: str
    created_at: str


class LightGBMTrainer:
    def __init__(
        self,
        lags: Union[int, List[int]] = 24,
        lags_past_covariates: Optional[Union[int, List[int]]] = None,
        lags_future_covariates: Optional[Union[int, List[int]]] = None,
        output_chunk_length: int = 1,
        multi_models: bool = False,
        random_state: int = 42,
        work_dir: Union[str, Path] = "runs_lgbm",
    ) -> None:
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.output_chunk_length = output_chunk_length
        self.multi_models = multi_models
        self.random_state = random_state

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.work_dir / f"lgbm_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Dados / artefatos do run
        self.model: Optional[LightGBMModel] = None
        self.series: Optional[TimeSeries] = None
        self.past_covs: Optional[TimeSeries] = None
        self.future_covs: Optional[TimeSeries] = None
        self.train_series: Optional[TimeSeries] = None
        self.val_series: Optional[TimeSeries] = None

        self.meta: Optional[LGBMRunMeta] = None
        self.report: Dict[str, Any] = {}

    # ----------------- FIT -----------------
    def fit(
        self,
        df: pd.DataFrame,
        time_col: str = "time",
        value_col: str = "close",
        past_cov_cols: Optional[List[str]] = None,
        future_cov_cols: Optional[List[str]] = None,
        train_split: float = 0.8,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Treina o LightGBMModel a partir de um DataFrame.
        Não reindexa e não preenche lacunas — usa apenas os pontos existentes.
        """
        assert 0.0 < train_split < 1.0, "train_split deve estar entre 0 e 1"
        dff = _ensure_sorted_df(df, time_col)

        # guarda apenas as colunas necessárias
        cols_needed = [time_col, value_col]
        if past_cov_cols:
            cols_needed += past_cov_cols
        if future_cov_cols:
            cols_needed += future_cov_cols
        missing = [c for c in cols_needed if c not in dff.columns]
        if missing:
            raise ValueError(f"Faltam colunas no DataFrame: {missing}")
        dff = dff[cols_needed]

        # Série principal e covariáveis (se existirem)
        series = _timeseries_from_df(dff, time_col, value_col)
        past_covs = _timeseries_from_df(dff, time_col, past_cov_cols) if past_cov_cols else None
        future_covs = _timeseries_from_df(dff, time_col, future_cov_cols) if future_cov_cols else None

        # split temporal
        train, val = series.split_after(train_split)
        self.series = series
        self.past_covs = past_covs
        self.future_covs = future_covs
        self.train_series = train
        self.val_series = val

        # Modelo
        self.model = LightGBMModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            output_chunk_length=self.output_chunk_length,
            multi_models=self.multi_models,
            random_state=self.random_state,
        )

        # Fit
        self.model.fit(
            series=train,
            past_covariates=past_covs,
            future_covariates=future_covs,
            verbose=verbose,
        )

        # Forecast sobre o conjunto de validação (multi-step direto)
        steps = len(val)
        pred_val = self.model.predict(
            n=steps,
            past_covariates=past_covs,
            future_covariates=future_covs,
        )

        # Métricas de validação
        real_v = series.slice_intersect(pred_val)
        pred_v = pred_val.slice_intersect(series)
        metrics_val = {
            "MAPE": float(mape(real_v, pred_v)),
            "sMAPE": float(smape(real_v, pred_v)),
            "RMSE": float(rmse(real_v, pred_v)),
            "MAE": float(mae(real_v, pred_v)),
        }

        # Metadados do run
        self.meta = LGBMRunMeta(
            dataset_rows=int(len(dff)),
            time_start=str(dff[time_col].min()),
            time_end=str(dff[time_col].max()),
            freq_detected=str(series.freq_str),
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            output_chunk_length=int(self.output_chunk_length),
            multi_models=bool(self.multi_models),
            random_state=int(self.random_state),
            train_split=float(train_split),
            model_path=str(self.run_dir / "lightgbm_model.pth.tar"),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Relatório
        self.report = {
            "model": "Darts.LightGBMModel",
            "validation": metrics_val,
            "params": {
                "lags": self.lags,
                "lags_past_covariates": self.lags_past_covariates,
                "lags_future_covariates": self.lags_future_covariates,
                "output_chunk_length": self.output_chunk_length,
                "multi_models": self.multi_models,
                "random_state": self.random_state,
            },
        }

        # Salva relatório parcial
        (self.run_dir / "report_val.json").write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if verbose:
            print("==> Métricas (val):", self.report["validation"])

        return self.report

    # ----------------- FORECAST -----------------
    def forecast(
        self,
        steps: int = 24,
        return_df: bool = True,
    ) -> Union[TimeSeries, pd.DataFrame]:
        if self.model is None or self.series is None:
            raise RuntimeError("Treine o modelo antes de forecast().")

        pred = self.model.predict(
            n=steps,
            past_covariates=self.past_covs,
            future_covariates=self.future_covs,
        )

        # plot e salvar
        plt.figure(figsize=(10, 5))
        self.series.plot(label="Real")
        pred.plot(label=f"Forecast {steps}")
        plt.title("LightGBM Forecast")
        plt.legend(); plt.tight_layout()
        out_png = self.run_dir / "forecast_overlay.png"
        plt.savefig(out_png, dpi=140)
        plt.close()

        if not return_df:
            return pred

        pdf = pred.pd_dataframe().reset_index()
        pdf.columns = ["time", "value_pred"]
        out_csv = self.run_dir / "forecast.csv"
        pdf.to_csv(out_csv, index=False)
        return pdf

    # ----------------- BACKTEST -----------------
    def backtest(
        self,
        horizon: int = 1,
        stride: int = 1,
        verbose: bool = True,
    ) -> Dict[str, float]:
        if self.model is None or self.series is None or self.train_series is None:
            raise RuntimeError("Treine o modelo antes de backtest().")

        pred_bt = self.model.historical_forecasts(
            series=self.series,
            start=self.train_series.end_time(),
            forecast_horizon=horizon,
            stride=stride,
            retrain=False,
            last_points_only=True,
            verbose=verbose,
            past_covariates=self.past_covs,
            future_covariates=self.future_covs,
        )

        real_b = self.series.slice_intersect(pred_bt)
        pred_b = pred_bt.slice_intersect(self.series)
        metrics_b = {
            "MAPE": float(mape(real_b, pred_b)),
            "sMAPE": float(smape(real_b, pred_b)),
            "RMSE": float(rmse(real_b, pred_b)),
            "MAE": float(mae(real_b, pred_b)),
        }

        # plot e salvar
        plt.figure(figsize=(10, 5))
        self.series.plot(label="Real")
        pred_bt.plot(label=f"Backtest h={horizon}, stride={stride}")
        plt.title("LightGBM Backtest")
        plt.legend(); plt.tight_layout()
        out_png = self.run_dir / "backtest_overlay.png"
        plt.savefig(out_png, dpi=140)
        plt.close()

        # salvar CSV
        pdf = pred_bt.pd_dataframe().reset_index()
        pdf.columns = ["time", "value_pred"]
        (self.run_dir / "backtest_predictions.csv").write_text("")
        pdf.to_csv(self.run_dir / "backtest_predictions.csv", index=False)

        # anexar ao report e salvar
        self.report.setdefault("backtest", {}).update(metrics_b)
        (self.run_dir / "report_val.json").write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return metrics_b

    # ----------------- SAVE/LOAD -----------------
    def save_run(self) -> None:
        if self.model is None or self.meta is None:
            raise RuntimeError("Nada para salvar: treine o modelo antes.")
        # modelo
        model_path = Path(self.meta.model_path)
        self.model.save(str(model_path))
        # metadados
        (self.run_dir / "metadata.json").write_text(
            json.dumps(asdict(self.meta), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # report final
        (self.run_dir / "report.json").write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"✔️ Artefatos salvos em: {self.run_dir.resolve()}")

    @classmethod
    def load(cls, run_dir: Union[str, Path]) -> "LightGBMTrainer":
        run_dir = Path(run_dir)
        meta_path = run_dir / "metadata.json"
        model_path = run_dir / "lightgbm_model.pth.tar"
        if not (meta_path.exists() and model_path.exists()):
            raise FileNotFoundError("metadata.json ou lightgbm_model.pth.tar não encontrados no run_dir.")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        trainer = cls(
            lags=meta.get("lags", 24),
            lags_past_covariates=meta.get("lags_past_covariates", None),
            lags_future_covariates=meta.get("lags_future_covariates", None),
            output_chunk_length=meta.get("output_chunk_length", 1),
            multi_models=meta.get("multi_models", False),
            random_state=meta.get("random_state", 42),
            work_dir=run_dir.parent,
        )
        trainer.run_dir = run_dir
        trainer.meta = LGBMRunMeta(**meta)
        trainer.model = LightGBMModel.load(str(model_path))
        # Observação: séries e covariáveis não são persistidas; reatribuí-las antes do forecast/backtest se necessário
        return trainer


# =========================
# Execução direta (exemplo)
# =========================
if __name__ == "__main__":
    # Exemplo mínimo de uso com CSV simples time/close
    CSV = "dados.csv"  # ajuste para o seu arquivo
    TIME_COL = "time"
    VALUE_COL = "close"

    df = pd.read_csv(CSV)
    # Exemplo de future covariates de calendário (se desejar):
    # df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    # df["hour"]  = df[TIME_COL].dt.hour / 23.0
    # df["dow"]   = df[TIME_COL].dt.dayofweek / 6.0
    # df["month"] = (df[TIME_COL].dt.month - 1) / 11.0

    trainer = LightGBMTrainer(lags=[1,2,3,6,12,24], output_chunk_length=1)
    trainer.fit(df, time_col=TIME_COL, value_col=VALUE_COL, train_split=0.8)
    trainer.backtest(horizon=1, stride=1)
    _ = trainer.forecast(steps=24, return_df=True)
    trainer.save_run()
