# utils/darts_forecaster.py
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import RNNModel, NaiveDrift
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse, smape
from darts.utils.statistics import plot_residuals_analysis

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

# --------------------------------- Callback de progresso ---------------------------------
class EpochProgressCSV(Callback):
    """
    Imprime métricas por época e salva um CSV incremental (live_metrics.csv).
    """
    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir
        self.live_csv = run_dir / "live_metrics.csv"
        if not self.live_csv.exists():
            self.live_csv.write_text("epoch,train_loss,val_loss,lr\n", encoding="utf-8")

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        train_loss = float(m.get("train_loss", float("nan")))
        val_loss   = float(m.get("val_loss", float("nan")))
        try:
            lr = trainer.optimizers[0].param_groups[0].get("lr", None)
        except Exception:
            lr = None
        print(f"[epoch {epoch:04d}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr}")
        with self.live_csv.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss},{val_loss},{lr}\n")

# --------------------------------- Metadados ---------------------------------
@dataclass
class DartsRunMetadata:
    framework: str
    model_type: str
    version: str
    created_at: str
    date_col: str
    target_col: str
    past_covariates_cols: List[str]
    future_covariates_cols: List[str]
    input_window: int
    output_window: int
    hidden_dim: int
    n_layers: int
    dropout: float
    batch_size: int
    epochs: int
    random_state: int
    train_range: Optional[str] = None
    val_range: Optional[str] = None
    freq_inferred: Optional[str] = None
    backtest_metric: Optional[float] = None
    backtest_metric_name: Optional[str] = None


class DartsLSTMForecaster:
    """
    Compatível com Darts 0.38.0.
    - Prepara séries (target/covariates), split/scale
    - Treina LSTM (early stopping/ckpt/logs), gera relatórios
    - Backtest completo e forecast
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        past_covariates_cols: Optional[List[str]] = None,
        future_covariates_cols: Optional[List[str]] = None,
        input_window: int = 48,
        output_window: int = 12,
        hidden_dim: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
        results_root: Union[str, Path] = "results_darts",
        salvar_grafico_fn: Optional[Callable[[pd.DataFrame, str, str, str, str], None]] = None,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.past_cols = past_covariates_cols or []
        self.future_cols = future_covariates_cols or []

        self.input_window = input_window
        self.output_window = output_window
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.run_dir: Optional[Path] = None

        self.salvar_grafico_fn = salvar_grafico_fn

        # Darts objects
        self.series: Optional[TimeSeries] = None
        self.past_covs: Optional[TimeSeries] = None
        self.future_covs: Optional[TimeSeries] = None
        self.model: Optional[RNNModel] = None

        # Scalers
        self.scaler_y = Scaler()
        self.scaler_cov = Scaler()

        # Splits escalados
        self.train_s = self.val_s = None
        self.past_train_s = self.past_val_s = None
        self.future_train_s = self.future_val_s = None

        # Logger Lightning
        self.pl_logger: Optional[CSVLogger] = None

        # Metadata
        self.metadata = DartsRunMetadata(
            framework="Darts",
            model_type="RNN(LSTM)",
            version="0.38.0",
            created_at=datetime.utcnow().isoformat() + "Z",
            date_col=self.date_col,
            target_col=self.target_col,
            past_covariates_cols=self.past_cols,
            future_covariates_cols=self.future_cols,
            input_window=self.input_window,
            output_window=self.output_window,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            batch_size=self.batch_size,
            epochs=self.epochs,
            random_state=self.random_state,
        )

    # ----------------- utils -----------------
    @staticmethod
    def _ensure_sorted_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.sort_values(date_col).reset_index(drop=True)
        return out

    @staticmethod
    def add_calendar_covariates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        out = df.copy()
        dt = pd.to_datetime(out[date_col])
        out["hour"] = dt.dt.hour.astype(int)
        out["dow"] = dt.dt.dayofweek.astype(int)
        out["month"] = dt.dt.month.astype(int)
        return out

    # ----------------- pipeline -----------------
    def prepare_from_df(self, df: pd.DataFrame) -> None:
        df = self._ensure_sorted_datetime(df, self.date_col)
        self.series = TimeSeries.from_dataframe(df, time_col=self.date_col, value_cols=self.target_col)

        self.past_covs = (
            TimeSeries.from_dataframe(df, time_col=self.date_col, value_cols=self.past_cols).astype(float)
            if self.past_cols else None
        )
        self.future_covs = (
            TimeSeries.from_dataframe(df, time_col=self.date_col, value_cols=self.future_cols).astype(float)
            if self.future_cols else None
        )

        try:
            self.metadata.freq_inferred = pd.infer_freq(df[self.date_col])
        except Exception:
            self.metadata.freq_inferred = None

    def split_and_scale(self, train_ratio: float = 0.8) -> None:
        if self.series is None:
            raise RuntimeError("Chame prepare_from_df(df) antes.")

        train, val = self.series.split_before(train_ratio)
        self.metadata.train_range = f"{train.start_time()} → {train.end_time()}"
        self.metadata.val_range = f"{val.start_time()} → {val.end_time()}"

        self.train_s = self.scaler_y.fit_transform(train)
        self.val_s = self.scaler_y.transform(val)

        if self.past_covs is not None:
            past_train, past_val = self.past_covs.split_before(train_ratio)
            self.past_train_s = self.scaler_cov.fit_transform(past_train)
            self.past_val_s = self.scaler_cov.transform(past_val)
        else:
            self.past_train_s = self.past_val_s = None

        if self.future_covs is not None:
            future_train, future_val = self.future_covs.split_before(train_ratio)
            self.future_train_s = self.scaler_cov.fit_transform(future_train)
            self.future_val_s = self.scaler_cov.transform(future_val)
        else:
            self.future_train_s = self.future_val_s = None

    def fit(self) -> None:
        if self.train_s is None:
            raise RuntimeError("Chame split_and_scale() antes de fit().")

        # cria pasta do run
        self.run_dir = self.results_root / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Logger e callbacks
        self.pl_logger = CSVLogger(save_dir=str(self.run_dir), name="pl_logs")
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10, min_delta=1e-4)
        ckpt = ModelCheckpoint(dirpath=str(self.run_dir), filename="best", monitor="val_loss", mode="min", save_top_k=1)
        progress_cb = EpochProgressCSV(self.run_dir)

        self.model = RNNModel(
            model="LSTM",
            input_chunk_length=self.input_window,
            output_chunk_length=self.output_window,
            training_length=self.input_window + self.output_window,
            hidden_dim=self.hidden_dim,
            n_rnn_layers=self.n_layers,
            dropout=self.dropout,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
            pl_trainer_kwargs={
                "accelerator": "auto",   # ou "gpu"
                "devices": 1,
                "enable_checkpointing": True,   # <<<<<< ADICIONE ISTO
                "logger": self.pl_logger,
                "callbacks": [early, ckpt, progress_cb],
                "enable_progress_bar": True,
                # "precision": "16-mixed",
            },
        )

        

        # Validação explícita para habilitar val_loss/early stopping
        self.model.fit(
            series=self.train_s,
            past_covariates=self.past_train_s,
            future_covariates=self.future_train_s,
            val_series=self.val_s,
            val_past_covariates=self.past_val_s,
            val_future_covariates=self.future_val_s,
            verbose=True,
        )

        # salva modelo e metadados
        self._save_model_and_metadata()
        # Relatórios pós-treino
        self._plot_loss_curve()
        self.epoch_table()

    # ----------------- BACKTEST -----------------
    def _ts_to_df(self, ts: TimeSeries, value_name: str) -> pd.DataFrame:
        df = ts.to_dataframe().copy().reset_index()
        time_col = df.columns[0]
        df = df.rename(columns={time_col: "date"})
        val_cols = [c for c in df.columns if c != "date"]
        if not val_cols:
            raise ValueError("TimeSeries sem coluna de valor ao converter para DataFrame.")
        return df.rename(columns={val_cols[0]: value_name})

    def _directional_accuracy(self, real_ts: TimeSeries, pred_ts: TimeSeries) -> float:
        r = self._ts_to_df(real_ts, "y").set_index("date")
        p = self._ts_to_df(pred_ts, "y_pred").set_index("date")
        df = r.join(p, how="inner").dropna()
        if len(df) < 3:
            return float("nan")
        dy_real = df["y"].diff().dropna()
        dy_pred = df["y_pred"].diff().dropna()
        n = min(len(dy_real), len(dy_pred))
        acc = (np.sign(dy_real.iloc[:n]) == np.sign(dy_pred.iloc[:n])).mean()
        return float(acc)

    def _plot_overlay_real_vs_pred(self, real_ts: TimeSeries, pred_ts: TimeSeries, path: Path) -> None:
        real_df = self._ts_to_df(real_ts, "y")
        pred_df = self._ts_to_df(pred_ts, "y_pred")
        plt.figure(figsize=(10, 5))
        plt.plot(real_df["date"], real_df["y"], label="Real")
        plt.plot(pred_df["date"], pred_df["y_pred"], label="Previsto")
        plt.legend(); plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _save_backtest_csv(self, real_ts: TimeSeries, pred_ts: TimeSeries, path: Path) -> pd.DataFrame:
        rdf = self._ts_to_df(real_ts, "y")
        pdf = self._ts_to_df(pred_ts, "y_pred")
        merged = pd.merge(rdf, pdf, on="date", how="inner")
        merged["abs_error"] = (merged["y"] - merged["y_pred"]).abs()
        merged["ape_pct"] = (merged["abs_error"] / merged["y"].abs()) * 100.0
        merged.to_csv(path, index=False)
        return merged

    def _write_backtest_report(self, metrics_model: dict, metrics_baseline: dict, path: Path) -> None:
        lines = []
        lines.append("# Backtest Report\n")
        lines.append(f"- Modelo: {self.metadata.model_type}")
        lines.append(f"- Janelas: input={self.input_window}, output={self.output_window}")
        lines.append(f"- Período treino: {self.metadata.train_range}")
        lines.append(f"- Período validação/backtest: {self.metadata.val_range}")
        lines.append("")
        lines.append("## Métricas (Modelo)")
        for k, v in metrics_model.items():
            lines.append(f"- **{k}**: {v:.6f}" if isinstance(v, (float, int)) else f"- **{k}**: {v}")
        lines.append("")
        lines.append("## Baseline — NaiveDrift (mesmas janelas)")
        for k, v in metrics_baseline.items():
            lines.append(f"- **{k}**: {v:.6f}" if isinstance(v, (float, int)) else f"- **{k}**: {v}")
        lines.append("")
        lines.append("## Artefatos")
        for fname in ["backtest_overlay.png", "backtest_predictions.csv", "backtest_metrics.json"]:
            lines.append(f"- `{fname}`")
        path.write_text("\n".join(lines), encoding="utf-8")

    def backtest(self, horizon: int = 12, stride: int = 12) -> float:
        """
        Backtest (historical_forecasts com last_points_only=True).
        Salva overlay, CSV, métricas e relatório comparando com NaiveDrift.
        """
        if self.model is None or self.series is None:
            raise RuntimeError("Treine o modelo antes de backtest().")

        assert self.run_dir is not None

        # séries escaladas para previsão
        full_s = self.scaler_y.transform(self.series)
        past_full = self.scaler_cov.transform(self.past_covs) if self.past_covs is not None else None
        future_full = self.scaler_cov.transform(self.future_covs) if self.future_covs is not None else None

        start = self.train_s.end_time()

        # previsões do modelo (um único TimeSeries)
        pred_s = self.model.historical_forecasts(
            series=full_s,
            start=start,
            forecast_horizon=horizon,
            stride=stride,
            retrain=False,
            verbose=True,
            last_points_only=True,
            past_covariates=past_full,
            future_covariates=future_full,
        )
        pred = self.scaler_y.inverse_transform(pred_s)

        # alinhar ao real
        real_common = self.series.slice_intersect(pred)
        pred_common = pred.slice_intersect(self.series)

        # métricas do modelo
        metrics_model = {
            "MAPE": float(mape(real_common, pred_common)),
            "sMAPE": float(smape(real_common, pred_common)),
            "MAE": float(mae(real_common, pred_common)),
            "RMSE": float(rmse(real_common, pred_common)),
            "Directional_Accuracy": float(self._directional_accuracy(real_common, pred_common)),
        }

        # baseline: NaiveDrift
        naive = NaiveDrift()
        naive_pred = naive.historical_forecasts(
            series=self.series,
            start=start,
            forecast_horizon=horizon,
            stride=stride,
            retrain=True,
            last_points_only=True,
            verbose=False,
        )
        real_b = self.series.slice_intersect(naive_pred)
        naive_b = naive_pred.slice_intersect(self.series)
        metrics_baseline = {
            "MAPE": float(mape(real_b, naive_b)),
            "sMAPE": float(smape(real_b, naive_b)),
            "MAE": float(mae(real_b, naive_b)),
            "RMSE": float(rmse(real_b, naive_b)),
            "Directional_Accuracy": float(self._directional_accuracy(real_b, naive_b)),
        }

        # salvar artefatos
        overlay_png = self.run_dir / "backtest_overlay.png"
        self._plot_overlay_real_vs_pred(real_common, pred_common, overlay_png)
        csv_path = self.run_dir / "backtest_predictions.csv"
        self._save_backtest_csv(real_common, pred_common, csv_path)

        # json com métricas
        (self.run_dir / "backtest_metrics.json").write_text(
            json.dumps({"model": metrics_model, "baseline_NaiveDrift": metrics_baseline}, indent=2),
            encoding="utf-8"
        )

        # atualizar metadata principal com MAPE
        self.metadata.backtest_metric = metrics_model["MAPE"]
        self.metadata.backtest_metric_name = "MAPE"
        self._save_metadata()

        # relatório markdown
        self._write_backtest_report(metrics_model, metrics_baseline, self.run_dir / "BACKTEST_REPORT.md")

        return metrics_model["MAPE"]

    # ----------------- FORECAST -----------------
    def forecast(self, steps: int = 24) -> TimeSeries:
        """Previsão futura (passos à frente) já desescalada."""
        if self.model is None:
            raise RuntimeError("Treine o modelo antes de forecast().")

        past_full = self.scaler_cov.transform(self.past_covs) if self.past_covs is not None else None
        future_full = self.scaler_cov.transform(self.future_covs) if self.future_covs is not None else None

        pred_s = self.model.predict(n=steps, past_covariates=past_full, future_covariates=future_full)
        return self.scaler_y.inverse_transform(pred_s)

    # ----------------- relatórios/plots auxiliares -----------------
    def epoch_table(self) -> pd.DataFrame:
        """
        Consolida 1 linha por época do metrics.csv e salva em epoch_metrics.csv.
        """
        if self.run_dir is None:
            raise RuntimeError("Treine o modelo antes de chamar epoch_table().")
        metrics_csv = self.run_dir / "pl_logs" / "version_0" / "metrics.csv"
        if not metrics_csv.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {metrics_csv}")

        df = pd.read_csv(metrics_csv)
        candidates = ["train_loss", "train_loss_epoch", "val_loss", "lr", "lr-Adam", "lr-AdamW"]
        keep = ["epoch"] + [c for c in candidates if c in df.columns]
        dff = df[keep].copy()

        table = (
            dff.groupby("epoch")
               .apply(lambda g: g.ffill().iloc[-1])
               .reset_index(drop=True)
               .sort_values("epoch")
        )

        if "train_loss_epoch" in table.columns and "train_loss" not in table.columns:
            table = table.rename(columns={"train_loss_epoch": "train_loss"})
        if "lr-Adam" in table.columns and "lr" not in table.columns:
            table = table.rename(columns={"lr-Adam": "lr"})
        if "lr-AdamW" in table.columns and "lr" not in table.columns:
            table = table.rename(columns={"lr-AdamW": "lr"})

        out_csv = self.run_dir / "epoch_metrics.csv"
        table.to_csv(out_csv, index=False)
        try:
            print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        except Exception:
            print(table.head())
        return table

    def _plot_loss_curve(self) -> None:
        """Plota e salva loss_curve.png a partir do metrics.csv."""
        if self.run_dir is None:
            return
        metrics_csv = self.run_dir / "pl_logs" / "version_0" / "metrics.csv"
        if not metrics_csv.exists():
            return

        df = pd.read_csv(metrics_csv)
        has_train = any(c in df.columns for c in ["train_loss", "train_loss_epoch"])
        has_val = "val_loss" in df.columns
        if not (has_train or has_val):
            return

        plt.figure(figsize=(8, 4))
        if "train_loss" in df.columns:
            plt.plot(df["epoch"], df["train_loss"], label="train_loss", alpha=0.8)
        elif "train_loss_epoch" in df.columns:
            plt.plot(df["epoch"], df["train_loss_epoch"], label="train_loss", alpha=0.8)
        if has_val:
            plt.plot(df["epoch"], df["val_loss"], label="val_loss", alpha=0.8)

        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss Curve")
        plt.legend(); plt.grid(True, alpha=0.25); plt.tight_layout()
        plt.savefig(self.run_dir / "loss_curve.png")
        plt.close()

    # ----------------- persistência -----------------
    def _save_model_and_metadata(self) -> None:
        assert self.run_dir is not None
        self.model.save(str(self.run_dir / "model.pth.tar"))  # type: ignore
        self._save_metadata()

    def _save_metadata(self) -> None:
        assert self.run_dir is not None
        with open(self.run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, run_dir: Union[str, Path]) -> Dict[str, Union[RNNModel, dict, Path]]:
        run_dir = Path(run_dir)
        model_path = run_dir / "model.pth.tar"
        meta_path = run_dir / "metadata.json"
        if not (model_path.exists() and meta_path.exists()):
            raise FileNotFoundError("Arquivos do modelo/metadata não encontrados.")
        model = RNNModel.load(str(model_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {"model": model, "metadata": metadata, "run_dir": run_dir}
