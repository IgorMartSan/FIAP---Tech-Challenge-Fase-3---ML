import MetaTrader5 as mt5
import pandas as pd


class MetaTrader:
    TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "H1": mt5.TIMEFRAME_H1,
        "D1": mt5.TIMEFRAME_D1,
    }

    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False

    def connect(self):
        """Conecta ao MetaTrader 5 com as credenciais fornecidas."""
        if not mt5.initialize(login=self.login, password=self.password, server=self.server):
            raise Exception(f"Erro ao conectar: {mt5.last_error()}")
        self.connected = True
        print("Conectado ao MetaTrader 5 com sucesso!")

    def disconnect(self):
        """Desconecta do MetaTrader 5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Desconectado do MetaTrader 5.")

    def fetch_mt5_data(self, ativo: str, timeframe: str = "H1", n_barras: int = 999999):
        """
        Obtém dados brutos do MetaTrader5.

        :param ativo: Símbolo (ex: "PETR4", "WIN$N")
        :param timeframe: String com o timeframe ("M1", "M5", "H1", "D1")
        :param n_barras: Quantidade de barras a serem buscadas
        :return: DataFrame com candles (open, high, low, close, tick_volume, spread, real_volume)
        """
        if not self.connected:
            raise RuntimeError("⚠️ Não está conectado. Use .connect() primeiro.")

        rates = mt5.copy_rates_from_pos(ativo, self.TIMEFRAMES[timeframe], 0, n_barras)
        if rates is None:
            raise Exception(f"Erro ao obter dados de {ativo}: {mt5.last_error()}")

        data = pd.DataFrame(rates)
        data["time"] = pd.to_datetime(data["time"], unit="s")
        # mantém "time" como coluna normal
        return data

