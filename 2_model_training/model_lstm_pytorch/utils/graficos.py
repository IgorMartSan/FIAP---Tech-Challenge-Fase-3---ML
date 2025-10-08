import matplotlib.pyplot as plt
import pandas as pd

import os
import matplotlib.pyplot as plt
import pandas as pd

def salvar_grafico(df: pd.DataFrame, coluna_y: str, coluna_x: str, titulo: str, caminho_saida: str):
    """
    Gera e salva um gráfico de linha a partir de um DataFrame.

    :param df: DataFrame com os dados
    :param coluna_y: Nome da coluna que será o eixo Y
    :param coluna_x: Nome da coluna que será o eixo X
    :param titulo: Título do gráfico
    :param caminho_saida: Caminho do arquivo de saída (ex: '/mnt/data/grafico.png')
    """
    if coluna_x not in df.columns or coluna_y not in df.columns:
        raise ValueError(f"Colunas '{coluna_x}' ou '{coluna_y}' não encontradas no DataFrame.")

    # Criar diretório se não existir
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    plt.figure(figsize=(12,6))
    plt.plot(df[coluna_x], df[coluna_y], label=coluna_y, linewidth=1.5)

    plt.title(titulo if titulo else f"Evolução de {coluna_y}")
    plt.xlabel(coluna_x)
    plt.ylabel(coluna_y)
    plt.legend()
    plt.grid(True)

    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    return caminho_saida



def plotar_fechamento_ultimos_dias(
    df: pd.DataFrame,
    col_time: str = "time",
    col_close: str = "close",
    dias: int = 3,
    titulo: str = None,
    cor: str = "tab:blue"
):
    """
    Plota o gráfico de fechamento dos últimos N dias.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados (com colunas de tempo e fechamento).
    col_time : str
        Nome da coluna com o tempo (default: 'time').
    col_close : str
        Nome da coluna com o preço de fechamento (default: 'close').
    dias : int
        Quantos dias anteriores devem ser exibidos (default: 3).
    titulo : str
        Título personalizado do gráfico (default: "Evolução do preço de fechamento (últimos N dias)").
    cor : str
        Cor da linha (default: 'tab:blue').

    Retorna:
    --------
    df_filtrado : pd.DataFrame
        DataFrame contendo apenas os dados usados no gráfico.
    """

    if col_time not in df.columns or col_close not in df.columns:
        raise ValueError(f"O DataFrame deve conter as colunas '{col_time}' e '{col_close}'.")

    # Garantir tipo datetime e ordenar
    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time])
    df = df.sort_values(col_time)

    # Filtrar últimos N dias
    ultimo_dia = df[col_time].max()
    inicio_periodo = ultimo_dia - pd.Timedelta(days=dias)
    df_filtrado = df[df[col_time] >= inicio_periodo]

    # Título automático, se não definido
    if titulo is None:
        titulo = f"Evolução do preço de fechamento (últimos {dias} dias)"

    # Plotar
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtrado[col_time], df_filtrado[col_close], color=cor, linewidth=1.5)
    plt.title(titulo)
    plt.xlabel("Data/Hora")
    plt.ylabel("Preço de fechamento")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return df_filtrado