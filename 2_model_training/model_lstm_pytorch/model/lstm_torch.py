import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt


# ============ Dataset com janela deslizante ============
class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx: idx + self.seq_len]              # [T, F]
        y_lab = self.y[idx + self.seq_len - 1]               # alvo no último passo
        return torch.tensor(x_seq), torch.tensor([y_lab])


# ============ Modelo LSTM ============
class _LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)  # 1 logit (binário)

    def forward(self, x):
        out, _ = self.lstm(x)            # [B, T, H]
        last = out[:, -1, :]             # [B, H]
        logit = self.fc(last)            # [B, 1]
        return logit


# ============ Config ============
@dataclass
class LSTMConfig:
    seq_len: int = 48
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    class_weight: bool = False        # pos_weight automático se True
    early_stopping_patience: int = 5  # 0 para desativar
    device: Optional[str] = None      # "cuda", "cpu" ou None (auto)


# ============ Classe principal ============
class LSTMTimeClassifier:
    """
    Treinador LSTM para CLASSIFICAÇÃO binária em séries temporais.

    Use:
        clf = LSTMTimeClassifier(
            features=["open","high","low","close","volume","hour","dow","month"],
            config=LSTMConfig(hidden_size=64, num_layers=2, dropout=0.2, epochs=25)
        )
        report = clf.fit(df, timecol="time", target_col="target")  # ou auto_target_next_close=1
        clf.plot_curves(save_path="curvas.png")         # loss por época
        clf.plot_val_probs(save_path="val_probs.png")   # prob prevista vs alvo
    """

    def __init__(self, features: List[str], config: Optional[LSTMConfig] = None):
        self.features = features
        self.cfg = config or LSTMConfig()
        self.device = torch.device(
            self.cfg.device if self.cfg.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[_LSTMClassifier] = None
        self.scaler: Optional[Dict[str, Dict[str, float]]] = None  # min-max simples por feature
        self.history: Dict[str, List[float]] = {}
        # buffers para plots pós-treino:
        self.last_val_probs: Optional[np.ndarray] = None
        self.last_val_targets: Optional[np.ndarray] = None

    # ---------- util: min-max scaler por coluna (opcional e simples) ----------
    def _fit_minmax(self, X_df: pd.DataFrame) -> None:
        self.scaler = {}
        for c in X_df.columns:
            v = X_df[c].astype(float).values
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                self.scaler[c] = {"min": float(vmin), "max": float(vmax)}
            else:
                self.scaler[c] = {"min": 0.0, "max": 1.0}

    def _apply_minmax(self, X_df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            return X_df.values.astype(np.float32)
        Xn = np.empty_like(X_df.values, dtype=np.float32)
        for j, c in enumerate(X_df.columns):
            v = X_df[c].astype(float).values
            mn = self.scaler[c]["min"]; mx = self.scaler[c]["max"]
            if mx > mn:
                Xn[:, j] = ((v - mn) / (mx - mn)).astype(np.float32)
            else:
                Xn[:, j] = 0.0
        return Xn

    # ---------- split temporal ----------
    @staticmethod
    def _time_split(X: np.ndarray, y: np.ndarray, split: float = 0.8):
        n = len(X)
        s = int(n * split)
        return (X[:s], y[:s]), (X[s:], y[s:])

    # ---------- loop de época ----------
    def _run_epoch(self, loader, model, criterion, optimizer=None):
        train_mode = optimizer is not None
        model.train() if train_mode else model.eval()

        total_loss = 0.0
        n_batches = 0
        logits_all, targets_all = [], []

        with torch.set_grad_enabled(train_mode):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = model(xb)
                loss = criterion(logits, yb)

                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1
                logits_all.append(logits.detach().cpu().numpy())
                targets_all.append(yb.detach().cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        logits_np = np.vstack(logits_all).ravel() if logits_all else np.array([])
        targets_np = np.vstack(targets_all).ravel() if targets_all else np.array([])

        if len(logits_np) == 0:
            return avg_loss, np.nan, np.nan, np.nan, logits_np, targets_np

        probs = 1 / (1 + np.exp(-logits_np))
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(targets_np, preds)
        f1  = f1_score(targets_np, preds)
        try:
            auc = roc_auc_score(targets_np, probs)
        except Exception:
            auc = float("nan")

        return avg_loss, acc, f1, auc, probs, targets_np

    # ---------- API principal ----------
    def fit(
        self,
        df: pd.DataFrame,
        timecol: str = "time",
        target_col: Optional[str] = None,
        auto_target_next_close: Optional[int] = None,  # ex.: 1 => target=(close(t+1) > close(t))
        class_weight: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        df         : DataFrame contendo time + features (e opcionalmente target)
        timecol    : nome da coluna de tempo (ordenaremos por ela)
        target_col : nome do alvo binário (0/1). Se None e auto_target_next_close for int,
                     criaremos automaticamente a partir de 'close'.
        auto_target_next_close : int deslocamento (ex.: 1 = próxima hora), cria target=(close(t+shift) > close(t))
        class_weight           : se None, usa o da config; senão, override (True/False)
        """
        assert timecol in df.columns, f"Coluna de tempo '{timecol}' não encontrada em df"
        df = df.copy().sort_values(timecol).reset_index(drop=True)

        # cria target automaticamente?
        if target_col is None:
            if auto_target_next_close is None:
                raise ValueError("Defina 'target_col' ou 'auto_target_next_close'.")
            if "close" not in df.columns:
                raise ValueError("Para auto_target_next_close, é necessário df['close'].")
            shift = int(auto_target_next_close)
            df["__target__"] = (df["close"].shift(-shift) > df["close"]).astype(int)
            target_col = "__target__"

        # remove NaN
        base_cols = list(set(self.features + [timecol, target_col]))
        df = df[base_cols].dropna().reset_index(drop=True)

        # prepara X, y
        X_df = df[self.features]
        y = df[target_col].astype(float).values

        # escala simples min-max
        self._fit_minmax(X_df)
        X = self._apply_minmax(X_df)

        # split temporal
        (X_tr, y_tr), (X_va, y_va) = self._time_split(X, y, split=0.8)

        # datasets e loaders
        train_ds = _SeqDataset(X_tr, y_tr, self.cfg.seq_len)
        val_ds   = _SeqDataset(X_va, y_va, self.cfg.seq_len)

        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError("Dados insuficientes após janela seq_len. Reduza seq_len ou verifique o dataset.")

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg.batch_size, shuffle=False, drop_last=False)

        # modelo
        self.model = _LSTMClassifier(
            input_size=len(self.features),
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            bidirectional=self.cfg.bidirectional
        ).to(self.device)

        # loss
        use_cw = self.cfg.class_weight if class_weight is None else bool(class_weight)
        if use_cw:
            pos = float((y_tr == 1).sum())
            neg = float((y_tr == 0).sum())
            pos_weight = torch.tensor([max(1.0, neg / (pos + 1e-6))], device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # otimizador
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # histórico
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}

        # early stopping
        best_val_loss = float("inf")
        wait = 0
        patience = int(self.cfg.early_stopping_patience)
        use_es = patience > 0

        self.last_val_probs, self.last_val_targets = None, None

        # ===== treinamento =====
        for ep in range(1, self.cfg.epochs + 1):
            tr_loss, _, _, _, _, _ = self._run_epoch(train_loader, self.model, criterion, optimizer)
            va_loss, va_acc, va_f1, va_auc, va_probs, va_targets = self._run_epoch(val_loader, self.model, criterion, None)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            self.history["val_f1"].append(va_f1)
            self.history["val_auc"].append(va_auc)
            self.last_val_probs, self.last_val_targets = va_probs, va_targets

            # ---- log por época (relatório durante o treino) ----
            print(f"[{ep:02d}/{self.cfg.epochs}] "
                  f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} "
                  f"val_acc={va_acc:.4f} val_f1={va_f1:.4f} val_auc={va_auc:.4f}")

            # early stopping
            if use_es:
                if va_loss < best_val_loss:
                    best_val_loss = va_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping (patience={patience}). Melhor val_loss: {best_val_loss:.4f}")
                        break

        # ===== relatório final =====
        if self.last_val_probs is None or self.last_val_targets is None or len(self.last_val_probs) == 0:
            raise RuntimeError("Validação não gerou probabilidades; verifique dados/carregadores.")

        final_preds = (self.last_val_probs >= 0.5).astype(int)
        cm = confusion_matrix(self.last_val_targets, final_preds)
        try:
            final_auc = roc_auc_score(self.last_val_targets, self.last_val_probs)
        except Exception:
            final_auc = float("nan")

        report = {
            "epochs_run": len(self.history["val_loss"]),
            "best_val_loss": float(np.nanmin(self.history["val_loss"])) if self.history["val_loss"] else None,
            "last_val_loss": float(self.history["val_loss"][-1]),
            "last_val_acc": float(self.history["val_acc"][-1]),
            "last_val_f1": float(self.history["val_f1"][-1]),
            "last_val_auc": float(final_auc),
            "confusion_matrix": cm.tolist(),
            "features": self.features,
            "seq_len": self.cfg.seq_len,
            "hidden_size": self.cfg.hidden_size,
            "num_layers": self.cfg.num_layers,
            "dropout": self.cfg.dropout,
        }
        return report

    # ---------- GRÁFICOS ----------
    def plot_curves(self, show: bool = True, save_path: Optional[str] = None):
        """Plota loss de treino/validação por época."""
        if not self.history or "train_loss" not in self.history:
            raise RuntimeError("Histórico vazio. Treine o modelo antes (fit).")
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_loss"], marker="o", label="train_loss")
        plt.plot(epochs, self.history["val_loss"], marker="o", label="val_loss")
        plt.title("Training vs Validation Loss (BCEWithLogitsLoss)")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=140, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_val_probs(self, show: bool = True, save_path: Optional[str] = None):
        """Plota probabilidade prevista vs alvo na validação."""
        if self.last_val_probs is None or self.last_val_targets is None:
            raise RuntimeError("Sem dados de validação armazenados. Treine o modelo antes (fit).")
        plt.figure(figsize=(10, 4))
        plt.plot(self.last_val_probs, label="predicted_prob")
        plt.plot(self.last_val_targets, label="target")
        plt.title("Validation: predicted up-probability vs target")
        plt.xlabel("Sample index"); plt.ylabel("Probability / Target")
        plt.legend(); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=140, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
