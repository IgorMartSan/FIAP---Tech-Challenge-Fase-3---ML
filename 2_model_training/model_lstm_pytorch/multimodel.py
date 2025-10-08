# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from infra.mongo import UpdateOne
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from infra.mongo import MongoClientManager
from math import sqrt
import joblib
from datetime import datetime
import os
import time

# =========================
# 1) Helpers
# =========================
def make_supervised(series: np.ndarray, lags: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói X com janelas de 'lags' e y como o próximo valor (t+1).
    Retorna X, y e last_inputs (o último valor da janela) para avaliar direção.
    """
    s = np.asarray(series).astype(float).ravel()
    X, y, last_inputs = [], [], []
    for i in range(lags, len(s)):
        X.append(s[i-lags:i])
        y.append(s[i])
        last_inputs.append(s[i-1])  # último valor conhecido, para direção
    return np.array(X), np.array(y), np.array(last_inputs)

@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float
    directional_acc: float
    up_hits: int
    down_hits: int
    up_misses: int
    down_misses: int

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, last_inputs: np.ndarray) -> Metrics:
    rmse = sqrt(((y_true - y_pred) ** 2).mean())
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # Direção: sinal do retorno verdadeiro vs sinal do retorno previsto
    true_ret = y_true - last_inputs
    pred_ret = y_pred - last_inputs

    true_dir = np.sign(true_ret)
    pred_dir = np.sign(pred_ret)

    mask_up   = true_dir > 0
    mask_down = true_dir < 0

    up_hits     = int(((pred_dir > 0) & mask_up).sum())
    up_misses   = int(((pred_dir <= 0) & mask_up).sum())
    down_hits   = int(((pred_dir < 0) & mask_down).sum())
    down_misses = int(((pred_dir >= 0) & mask_down).sum())

    valid = (true_dir != 0)  # ignora empates (retorno zero)
    directional_acc = float((pred_dir[valid] == true_dir[valid]).mean()) if valid.any() else np.nan

    return Metrics(rmse, mae, r2, directional_acc, up_hits, down_hits, up_misses, down_misses)

# =========================
# 2) Model zoo (sklearn only)
# =========================
def build_models() -> Dict[str, Pipeline]:
    """
    Alguns modelos precisam de escala (SVR, KNN, regressões L1/L2/Elastic).
    Árvores/boosting não precisam, mas não atrapalha.
    """
    models = {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge":            Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))]),
        "Lasso":            Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=1e-3, random_state=42, max_iter=10000))]),
        "ElasticNet":       Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=42, max_iter=10000))]),
        "SVR":              Pipeline([("scaler", StandardScaler()), ("model", SVR(C=10.0, epsilon=0.01, kernel="rbf"))]),
        "KNN":              Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
        "RandomForest":     Pipeline([("model", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1))]),
        "HistGBDT":         Pipeline([("model", HistGradientBoostingRegressor(max_depth=None, learning_rate=0.05, max_iter=500, random_state=42))]),
    }
    return models

# =========================
# 3) Treino, avaliação e relatório
# =========================
if __name__ == "__main__":

    # ---- Parâmetros
    CSV_PATH = Path(r"2_model_training\dados_bruto.csv")  # ajuste se precisar
    COL_TARGET = "close"
    LAGS = 8
    OUT_DIR = Path("results_reports")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_SPLIT = 0.8   # Split temporal (sem embaralhar): 80% treino, 20% teste

    mongo = MongoClientManager(
        user=os.getenv("MONGO_INITDB_ROOT_USERNAME"),
        password=os.getenv("MONGO_INITDB_ROOT_PASSWORD"),
        host=os.getenv("MONGO_HOST", "localhost"),
        port=os.getenv("MONGO_PORT", "27017"),
        db=os.getenv("MONGO_INITDB_DATABASE", "test"),
        coll="candles",
    )
    mongo.connect()
    n_linhas = 0

    while True:
        time.sleep(4)
    
       
        df = mongo.get_dataframe("WDO$N", "H1", limit=-1)
        atual = len(df)
        df = df.dropna()

        if atual != n_linhas:
            print(f"Número de linhas mudou: {n_linhas} → {atual}")
            n_linhas = atual
        

            # # ---- Carregue sua série (ex.: coluna 'close')
            # df = pd.read_csv(CSV_PATH)
            # if COL_TARGET not in df.columns:
            #     raise ValueError(f"Coluna '{COL_TARGET}' não encontrada em {CSV_PATH}")

            y = df[COL_TARGET].values

            X, y_target, last_inputs = make_supervised(y, lags=LAGS)

            # Split temporal (sem embaralhar): 80% treino, 20% teste
            split = int(DATA_SPLIT * len(X))


            if split == 0 or split >= len(X):
                raise ValueError("Dados insuficientes após criar janelas. Aumente o dataset ou reduza LAGS.")

            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y_target[:split], y_target[split:]
            last_te    = last_inputs[split:]  # para direção na amostra de teste

            models = build_models()

            # Armazenamentos
            results: Dict[str, Metrics] = {}
            preds_store: Dict[str, np.ndarray] = {}
            trained_models: Dict[str, Pipeline] = {}


            # Treinamento
            for name, pipe in models.items():
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                preds_store[name] = y_pred
                trained_models[name] = pipe
                results[name] = evaluate(y_te, y_pred, last_te)

            # ----- Ranking por RMSE
            ranking = sorted(results.items(), key=lambda kv: kv[1].rmse)
            

            print("\n=== Resultados (teste) — lags=8 → horizonte +1 ===")
            for name, m in ranking:
                print(
                    f"{name:16s} | RMSE={m.rmse:8.4f} | MAE={m.mae:8.4f} | R²={m.r2:6.3f} | "
                    f"Direção={m.directional_acc:6.3f} | UP hit/miss={m.up_hits}/{m.up_misses} | DOWN hit/miss={m.down_hits}/{m.down_misses}"
                )

            # ----- Melhor modelo
            best_name, best_metrics = ranking[0]
            print(f"\n>>> Melhor por RMSE: {best_name}")
            print(best_metrics)

            # ----- Salvar melhor modelo
            save_model_path = OUT_DIR / f"best_model_{best_name}.joblib"
            joblib.dump(trained_models[best_name], save_model_path)
            print(f"Modelo salvo em: {save_model_path}")

            # ----- Previsão do próximo ponto usando a ÚLTIMA janela real
            last_window = y[-LAGS:]
            next_pred = trained_models[best_name].predict(last_window.reshape(1, -1))[0]
            print(f"\nPrevisão do próximo valor (usando {LAGS} últimos pontos): {next_pred:.6f}")




            # =========================
            # 4) Artefatos do RELATÓRIO
            # =========================
            # 4.1 Tabela de métricas (CSV)
            metrics_rows = []
            for name, m in results.items():
                row = {"model": name}
                row.update(asdict(m))
                metrics_rows.append(row)
            # ===== Tabela de métricas de TESTE (para ranking por acurácia direcional)
            TOP_K_BEST_TEST = 5  # ajuste se quiser 5, etc.
            test_df = pd.DataFrame([
                {"model": name,
                "rmse": m.rmse,
                "mae": m.mae,
                "r2": m.r2,
                "directional_acc": m.directional_acc}
                for name, m in results.items()
            ])
            metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
            metrics_csv_path = OUT_DIR / "1_metrics.csv"
            metrics_df.to_csv(metrics_csv_path, index=False)

            # 4.2 Predições do melhor modelo (CSV)
            best_pred = preds_store[best_name]
            true_ret = y_te - last_te
            pred_ret = best_pred - last_te
            best_df = pd.DataFrame({
                "y_true": y_te,
                "y_pred": best_pred,
                "last_input": last_te,
                "true_dir": np.sign(true_ret),
                "pred_dir": np.sign(pred_ret),
            })
            best_df["hit"] = (best_df["true_dir"] == best_df["pred_dir"]).astype(int)
            preds_csv_path = OUT_DIR / f"2_predicoes_{best_name}.csv"
            best_df.to_csv(preds_csv_path, index=False)

            # 4.3 Relatório em Markdown
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            relatorio_md = OUT_DIR / "relatorio.md"

            # Matriz de confusão direcional do melhor
            up_hits     = int(((best_df["pred_dir"] > 0) & (best_df["true_dir"] > 0)).sum())
            up_misses   = int(((best_df["pred_dir"] <= 0) & (best_df["true_dir"] > 0)).sum())
            down_hits   = int(((best_df["pred_dir"] < 0) & (best_df["true_dir"] < 0)).sum())
            down_misses = int(((best_df["pred_dir"] >= 0) & (best_df["true_dir"] < 0)).sum())
            dir_acc     = float((best_df["hit"][best_df["true_dir"] != 0].mean())) if (best_df["true_dir"] != 0).any() else float("nan")

            # Top-3 modelos formatados
            top_k = 3 if len(ranking) >= 3 else len(ranking)
            lines_top = []
            for i in range(top_k):
                n, m = ranking[i]
                lines_top.append(f"{i+1}. **{n}** — RMSE: `{m.rmse:.4f}` | MAE: `{m.mae:.4f}` | R²: `{m.r2:.3f}` | Direção: `{m.directional_acc:.3f}`")

            content = f"""# Relatório de Modelos (lags={LAGS}, horizonte=+1)

        **Data:** {ts}  
        **Arquivo de origem:** `{CSV_PATH}`  
        **Amostras (treino/teste):** {len(X_tr)}/{len(X_te)}

        ## Ranking (por RMSE)
        {chr(10).join(lines_top)}

        ## Melhor modelo
        **Nome:** `{best_name}`  
        **RMSE:** `{best_metrics.rmse:.6f}`  
        **MAE:** `{best_metrics.mae:.6f}`  
        **R²:** `{best_metrics.r2:.6f}`  
        **Acurácia Direcional:** `{best_metrics.directional_acc:.6f}`  

        **Matriz direcional (verdadeiro vs previsto):**

        |                | Prevista ↑ | Prevista ↓ |
        |----------------|------------|------------|
        | **Verdade ↑**  | {up_hits}  | {up_misses} |
        | **Verdade ↓**  | {down_misses} | {down_hits}  |

        > **Próximo valor previsto** (usando os últimos {LAGS} pontos reais): `{next_pred:.6f}`

        ## Artefatos gerados
        - `./{OUT_DIR.name}/metrics.csv` — métricas por modelo  
        - `./{OUT_DIR.name}/predicoes_{best_name}.csv` — predições e acertos do melhor  
        - `./{OUT_DIR.name}/{save_model_path.name}` — melhor modelo (joblib)  
        """

            relatorio_md.write_text(content, encoding="utf-8")
            print(f"\nRelatório salvo em: {relatorio_md}")
            print(f"Métricas salvas em: {metrics_csv_path}")
            print(f"Predições salvas em: {preds_csv_path}")

            # =========================
            # 5) Penúltima x Última — validação e previsão atual (por modelo)
            # =========================
            # Aliases
            X_all, y_all, last_inputs_all = X, y_target, last_inputs

            # --- (A) PENÚLTIMA JANELA → prever o ÚLTIMO valor real (validação)
            i_pen = len(y_all) - 1  # última amostra supervisionada: prevê y[-1] usando y[-LAGS: -1]
            if i_pen < 0:
                raise ValueError("Dados insuficientes para validação do penúltimo/último.")

            Xi_pen     = X_all[i_pen].reshape(1, -1)
            y_true_last = float(y_all[i_pen])
            last_in_pen = float(last_inputs_all[i_pen])

            def _sinal(x: float) -> int:
                return 1 if x > 0 else (-1 if x < 0 else 0)

            linhas_pen = []
            linhas_ult = []

            for mname, model in trained_models.items():
                # Validação: prever o último valor real com a penúltima janela
                y_pred_last = float(model.predict(Xi_pen)[0])
                true_ret = y_true_last - last_in_pen
                pred_ret = y_pred_last - last_in_pen
                dir_true = _sinal(true_ret)
                dir_pred = _sinal(pred_ret)
                hit = int(dir_true != 0 and dir_true == dir_pred)

                linhas_pen.append({
                    "model": mname,
                    "valor_anterior": last_in_pen,
                    "valor_previsto_ultimo": y_pred_last,
                    "valor_real_ultimo": y_true_last,
                    "retorno_real": true_ret,
                    "retorno_previsto": pred_ret,
                    "direcao_real": dir_true,        # -1, 0, +1
                    "direcao_prevista": dir_pred,
                    "hit_direcional": hit
                })

                # Previsão ATUAL: última janela (os LAGS últimos valores reais) → próximo valor
                last_window_current = y[-LAGS:].reshape(1, -1)
                next_pred = float(model.predict(last_window_current)[0])

                linhas_ult.append({
                    "model": mname,
                    "ultima_janela_usada": list(map(float, y[-LAGS:])),
                    "proximo_previsto": next_pred
                })

            # --- salvar CSVs
            csv_pen = OUT_DIR / "3_penultima_→_ultimo_validacao.csv"
            csv_ult = OUT_DIR / "4_ultima_→_proximo_previsao.csv"
            pd.DataFrame(linhas_pen).to_csv(csv_pen, index=False)
            pd.DataFrame(linhas_ult).to_csv(csv_ult, index=False)

            # --- imprimir resumo
            print("\n=== Validação (penúltima janela → último valor real) ===")
            for r in linhas_pen:
                status = "✔️" if r["hit_direcional"] == 1 else "❌"
                print(
                    f"{r['model']:16s} | {status} | "
                    f"anterior={r['valor_anterior']:.6f} → previsto={r['valor_previsto_ultimo']:.6f} "
                    f"vs real={r['valor_real_ultimo']:.6f} | "
                    f"dir real={r['direcao_real']}, dir prev={r['direcao_prevista']}"
                )

            # ===== Comparar com o penúltimo valor (último conhecido) e ranquear =====
            penultimo_valor = float(y[-1])  # último valor conhecido (referência)
            rank_rows = []

            print("\n=== Previsão ATUAL (última janela → próximo valor) ===")
            print(f"(Referência = penúltimo valor conhecido: {penultimo_valor:.6f})")

            for r in linhas_ult:
                model_name = r["model"]
                prox_prev  = float(r["proximo_previsto"])
                delta      = prox_prev - penultimo_valor

                if delta > 0:
                    dir_label = "↑ sobe"
                    arrow = "↑"
                elif delta < 0:
                    dir_label = "↓ desce"
                    arrow = "↓"
                else:
                    dir_label = "= neutro"
                    arrow = "="

                print(f"{model_name:16s} | proximo_previsto={prox_prev:.6f} | Δ={delta:+.6f} | {arrow} {dir_label}")

                rank_rows.append({
                    "model": model_name,
                    "penultimo_valor": penultimo_valor,
                    "proximo_previsto": prox_prev,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "direcao": dir_label   # ↑ sobe / ↓ desce / = neutro
                })

            # ===== Ranking por força do sinal (|Δ| maior primeiro) =====
            import pandas as pd
            rank_df = pd.DataFrame(rank_rows).sort_values("abs_delta", ascending=False)
            csv_rank = OUT_DIR / "5_ranking_previsao_atual.csv"
            rank_df.to_csv(csv_rank, index=False)

            print("\n=== Ranking (força do sinal |Δ|) — maior primeiro ===")
            for _, row in rank_df.iterrows():
                print(f"{row['model']:16s} | Δ={row['delta']:+.6f} | {row['direcao']} | prev={row['proximo_previsto']:.6f}")

            # ===== Anexar ao relatório =====
            top_k = min(5, len(rank_df))
            top_lines = []
            for i in range(top_k):
                rr = rank_df.iloc[i]
                top_lines.append(
                    f"- **{i+1}. {rr['model']}** — prev: `{rr['proximo_previsto']:.6f}` | "
                    f"Δ=`{rr['delta']:+.6f}` ({rr['direcao']})"
                )

                        # ===== Selecionar TOP-K por acurácia direcional no TESTE =====
            best_test_df = test_df.sort_values("directional_acc", ascending=False).head(TOP_K_BEST_TEST)
            best_models = set(best_test_df["model"].tolist())

            # ===== Filtrar as próximas previsões (rank_rows) para apenas esses melhores do TESTE
            rank_df = pd.DataFrame(rank_rows)
            rank_df = rank_df[rank_df["model"].isin(best_models)].copy()

            # ===== Ordenar pela força do sinal |Δ| (maior primeiro) só para exibição
            rank_df["abs_delta"] = rank_df["abs_delta"].astype(float)
            rank_df = rank_df.sort_values("abs_delta", ascending=False).reset_index(drop=True)

            # ===== (1) Mesclar ACURÁCIA DE TESTE no dataframe das próximas previsões
            # rank_df veio do filtro dos TOP_K_BEST_TEST; agora vamos adicionar a acurácia de teste
            rank_df = rank_df.merge(
                test_df[["model", "directional_acc", "rmse", "mae", "r2"]],
                on="model", how="left"
            )

            # ===== (2) Ranquear pelos MELHORES por ACURÁCIA DIRECIONAL (maior → melhor)
            # usa abs_delta apenas como critério de desempate (opcional)
            rank_df = rank_df.sort_values(
                ["directional_acc", "abs_delta"], ascending=[False, False]
            ).reset_index(drop=True)

            # ===== (3) Print bonitinho (mostrando também a acurácia de TESTE do modelo)
            print("\n=== 🧭 Próximas previsões — RANK por ACURÁCIA DIRECIONAL (TESTE) ===")
            print(f"Referência (penúltimo valor): {penultimo_valor:.6f}")
            for i, row in rank_df.iterrows():
                arrow = "↑" if row["delta"] > 0 else ("↓" if row["delta"] < 0 else "=")
                estrela = "⭐" if i == 0 else " "
                print(
                    f"{estrela} Rank {i+1:<2d} | {row['model']:<16s} | "
                    f"dir_acc_teste={row['directional_acc']:.3f} | "
                    f"Previsto: {row['proximo_previsto']:.6f} | "
                    f"Δ={row['delta']:+.6f} {arrow} ({row['direcao']})"
                )

            # ===== (4) Salvar CSV ranqueado por acurácia
            csv_rank_by_acc = OUT_DIR / "6_print_ranking_previsao_atual_RANK_BY_DIRECTIONAL.csv"
            rank_df.to_csv(csv_rank_by_acc, index=False)
            print(f"\nPróximas previsões (rank por acurácia direcional de TESTE) salvas em: {csv_rank_by_acc}")









