import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# =============================
# Hiperparâmetros
# =============================
CSV_PATH = "/mnt/HDD2TB/projetos_igor/novo/FIAP---Tech-Challenge-Fase-3---ML-Pipeline/2_model_training/dados_bruto.csv"
FEATURES = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']
TARGET_IDX = 3  # Índice da coluna 'close' no vetor de FEATURES
WINDOW_SIZE = 10           # Quantos passos anteriores usar como entrada
PREDICT_STEPS_AHEAD = 24    # Quantos passos à frente prever
TEST_SIZE = 0.2            # Proporção do conjunto de teste
N_ESTIMATORS = 100         # Número de árvores do RandomForest
RANDOM_SEED = 42           # Reprodutibilidade
# =============================

# 1. Carregar os dados
df = pd.read_csv(CSV_PATH, parse_dates=['time'])

# 2. Ordenar e selecionar as colunas numéricas
df = df.sort_values('time')
data = df[FEATURES].copy()

# 2.1 Verifica e converte para numérico
for col in FEATURES:
    non_numeric = df[~df[col].astype(str).str.replace('.', '', 1).str.replace('-', '', 1).str.isnumeric()][col]
    if not non_numeric.empty:
        print(f"[!] Valores inválidos na coluna '{col}':")
        print(non_numeric.head())

data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=FEATURES)

# 3. Normalizar os dados
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 4. Dividir o dataset em treino/teste antes de gerar janelas
split_index = int(len(data_scaled) * (1 - TEST_SIZE))
data_scaled_train = data_scaled[:split_index]
data_scaled_test = data_scaled[split_index - WINDOW_SIZE - PREDICT_STEPS_AHEAD + 1:]  # inclui janelas que "olham para trás"

# 5. Função para criar janelas deslizantes
def create_window_data(data, window_size, target_idx, predict_steps_ahead=1):
    X, y = [], []
    for i in range(len(data) - window_size - predict_steps_ahead + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size + predict_steps_ahead - 1][target_idx])
    return np.array(X), np.array(y)

# 6. Criar janelas para treino e teste
X_train, y_train = create_window_data(data_scaled_train, WINDOW_SIZE, TARGET_IDX, PREDICT_STEPS_AHEAD)
X_test, y_test = create_window_data(data_scaled_test, WINDOW_SIZE, TARGET_IDX, PREDICT_STEPS_AHEAD)

# 7. Remodelar para sklearn
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# 8. Treinar o modelo
model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
model.fit(X_train, y_train)

# 9. Prever
y_pred = model.predict(X_test)

# 10. Métricas de erro
print(f"\nPrevendo {PREDICT_STEPS_AHEAD} passo(s) à frente no tempo")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# 11. Desnormalizar os valores previstos e reais da coluna 'close'
def reconstruct_close(normalized_close_values):
    dummy = np.zeros((len(normalized_close_values), len(FEATURES)))
    dummy[:, TARGET_IDX] = normalized_close_values
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]

real_close = reconstruct_close(y_test)
pred_close = reconstruct_close(y_pred)

# 12. Avaliar acerto de direção
def calc_direction_accuracy(real, pred, window=1):
    # Diferença entre preços consecutivos
    real_diff = np.diff(real, n=window)
    pred_diff = np.diff(pred, n=window)

    real_sign = np.sign(real_diff)
    pred_sign = np.sign(pred_diff)

    correct = (real_sign == pred_sign).sum()
    total = len(real_sign)
    accuracy = correct / total if total > 0 else 0

    return correct, total - correct, accuracy

correct, incorrect, accuracy = calc_direction_accuracy(real_close, pred_close)

print(f"\nAvaliação de direção do preço (subiu ou caiu):")
print(f"Acertos: {correct}")
print(f"Erros:   {incorrect}")
print(f"Acurácia de direção: {accuracy * 100:.2f}%")

# 13. Plotar
plt.figure(figsize=(12, 6))
plt.plot(real_close, label='Real')
plt.plot(pred_close, label='Previsto')
plt.title(f"Preço de Fechamento: Real vs Previsto ({PREDICT_STEPS_AHEAD} passo à frente)")
plt.xlabel("Tempo (amostras)")
plt.ylabel("Preço")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()