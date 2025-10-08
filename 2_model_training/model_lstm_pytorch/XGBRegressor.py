import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import joblib

# =============================
# Hiperparâmetros
# =============================
CSV_PATH = "/mnt/HDD2TB/projetos_igor/novo/FIAP---Tech-Challenge-Fase-3---ML-Pipeline/2_model_training/dados_bruto.csv"
FEATURES = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']
TARGET_IDX = 3  # Índice da coluna 'close' no vetor de FEATURES
WINDOW_SIZE = 10            # Quantos passos anteriores usar como entrada
PREDICT_STEPS_AHEAD = 24    # Quantos passos à frente prever
TEST_SIZE = 0.2             # Proporção do conjunto de teste
RANDOM_SEED = 42
XGB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED
}
# =============================

# 1. Carregar os dados
df = pd.read_csv(CSV_PATH, parse_dates=['time'])
df = df.sort_values('time')
data = df[FEATURES].copy()

# 2. Limpar dados
for col in FEATURES:
    non_numeric = df[~df[col].astype(str).str.replace('.', '', 1).str.replace('-', '', 1).str.isnumeric()][col]
    if not non_numeric.empty:
        print(f"[!] Valores inválidos na coluna '{col}':")
        print(non_numeric.head())

data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=FEATURES)

# 3. Normalizar
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 4. Dividir treino/teste antes de criar janelas
split_index = int(len(data_scaled) * (1 - TEST_SIZE))
data_scaled_train = data_scaled[:split_index]
data_scaled_test = data_scaled[split_index - WINDOW_SIZE - PREDICT_STEPS_AHEAD + 1:]

# 5. Função de janelas
def create_window_data(data, window_size, target_idx, predict_steps_ahead=1):
    X, y = [], []
    for i in range(len(data) - window_size - predict_steps_ahead + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size + predict_steps_ahead - 1][target_idx])
    return np.array(X), np.array(y)

X_train, y_train = create_window_data(data_scaled_train, WINDOW_SIZE, TARGET_IDX, PREDICT_STEPS_AHEAD)
X_test, y_test = create_window_data(data_scaled_test, WINDOW_SIZE, TARGET_IDX, PREDICT_STEPS_AHEAD)

# 6. Achatar as janelas
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# 7. Treinar com XGBoost
model = XGBRegressor(**XGB_PARAMS)
model.fit(X_train, y_train)

# 8. Previsão
y_pred = model.predict(X_test)

# 9. Métricas
print(f"\nPrevendo {PREDICT_STEPS_AHEAD} passo(s) à frente no tempo")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# 10. Desnormalizar os valores previstos e reais da coluna 'close'
def reconstruct_close(normalized_close_values):
    dummy = np.zeros((len(normalized_close_values), len(FEATURES)))
    dummy[:, TARGET_IDX] = normalized_close_values
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]

real_close = reconstruct_close(y_test)
pred_close = reconstruct_close(y_pred)

# 11. Avaliação de direção
def calc_direction_accuracy(real, pred):
    real_diff = np.diff(real)
    pred_diff = np.diff(pred)
    real_sign = np.sign(real_diff)
    pred_sign = np.sign(pred_diff)
    correct = (real_sign == pred_sign).sum()
    total = len(real_sign)
    return correct, total - correct, correct / total if total > 0 else 0

correct, incorrect, acc = calc_direction_accuracy(real_close, pred_close)
print("\nAvaliação de direção do preço (subiu ou caiu):")
print(f"Acertos: {correct}")
print(f"Erros:   {incorrect}")
print(f"Acurácia de direção: {acc * 100:.2f}%")

# 12. Plotar
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

# 13. Prever com os últimos dados
def predict_last_window(data_scaled, window_size, model, scaler, target_idx):
    last_window = data_scaled[-window_size:]
    input_data = last_window.reshape(1, -1)
    prediction = model.predict(input_data)
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0][target_idx] = prediction[0]
    return scaler.inverse_transform(dummy)[0][target_idx]

future_close = predict_last_window(data_scaled, WINDOW_SIZE, model, scaler, TARGET_IDX)
print(f"\nPrevisão com base nos dados mais recentes: {future_close:.2f}")

# 14. Salvar modelo e scaler
joblib.dump(model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModelo e scaler salvos com sucesso.")
