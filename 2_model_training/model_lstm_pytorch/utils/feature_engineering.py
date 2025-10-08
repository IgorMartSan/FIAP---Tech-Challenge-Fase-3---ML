import pandas as pd
import pandas as pd
from collections import Counter

def preencher_lacunas(
    df: pd.DataFrame,
    col_time: str = "time",
    round_to: str = "H",  # "H" para hora cheia; pode usar "T" (minuto), etc.
) -> pd.DataFrame:
    """
    Identifica a frequência temporal automaticamente (ex.: 1H) e preenche lacunas.
    Regras de preenchimento:
      - close: ffill (com bfill no início se necessário)
      - open/high/low nas linhas sintéticas: recebem o close preenchido
      - volumes presentes: viram 0 nas linhas sintéticas
    Adiciona a coluna 'is_synthetic' marcando True só nas linhas criadas.
    Requer: col_time e, no mínimo, a coluna 'close'.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    # 1) Normaliza tempo e ordena
    df[col_time] = pd.to_datetime(df[col_time])
    # Se seus registros são por hora, garante hora cheia
    if round_to:
        df[col_time] = df[col_time].dt.floor(round_to.lower())

    df = df.sort_values(col_time).drop_duplicates(subset=[col_time])
    df = df.set_index(col_time)

    if "close" not in df.columns:
        raise ValueError("A coluna 'close' é obrigatória para o preenchimento.")

    # 2) Tenta inferir a frequência
    inferred = pd.infer_freq(df.index)
    if inferred is None:
        # Fallback: pega o delta mais comum
        diffs = (df.index[1:] - df.index[:-1])
        if len(diffs) == 0:
            # Série com um único ponto: nada a fazer
            return df.reset_index().rename(columns={"index": col_time})
        most_common_delta = Counter(diffs).most_common(1)[0][0]
        # Constrói um alias de frequência aproximado
        # Exemplos de most_common_delta: Timedelta('0 days 01:00:00') -> "1H"
        total_seconds = int(most_common_delta.total_seconds())
        if total_seconds % 3600 == 0:
            inferred = f"{total_seconds // 3600}H"
        elif total_seconds % 60 == 0:
            inferred = f"{total_seconds // 60}T"  # minutos
        else:
            # Fallback absoluto: usa o delta bruto como freq no date_range
            inferred = most_common_delta

    # 3) Índice completo pela frequência inferida
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=inferred.lower())

    # 4) Reindexa e marca lacunas
    original_index = df.index.copy()
    df = df.reindex(full_index)
    synth_mask = ~df.index.isin(original_index)
    df["is_synthetic"] = synth_mask

    # 5) Preenche close (ffill + bfill no início se precisar)
    df["close"] = df["close"].ffill()
    if df["close"].isna().any():
        df["close"] = df["close"].bfill()

    # 6) OHLC nas linhas sintéticas recebem 'close' preenchido
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = pd.NA
        df.loc[synth_mask, col] = df.loc[synth_mask, col].fillna(df.loc[synth_mask, "close"])

    # 7) Volumes nas linhas sintéticas → 0 (se existirem)
    for vol in ["tick_volume", "real_volume", "volume"]:
        if vol in df.columns:
            df.loc[synth_mask, vol] = df.loc[synth_mask, vol].fillna(0)

    # 8) Retorna com a coluna de tempo
    return df.reset_index().rename(columns={"index": col_time})

