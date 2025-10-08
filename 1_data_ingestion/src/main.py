from infra.mongo import MongoClientManager
from infra.metatrader import MetaTrader
import os
import time

from dotenv import load_dotenv

load_dotenv()


def main():
    # --- inicializa Mongo no mesmo estilo do MetaTrader ---
    mongo = MongoClientManager(
        user=os.getenv("MONGO_INITDB_ROOT_USERNAME"),
        password=os.getenv("MONGO_INITDB_ROOT_PASSWORD"),
        host=os.getenv("MONGO_HOST", "localhost"),
        port=os.getenv("MONGO_PORT", "27017"),
        db=os.getenv("MONGO_INITDB_DATABASE", "test"),
        coll="candles",
    )
    mongo.connect()

    # --- inicializa MetaTrader ---
    mt = MetaTrader(
        login=int(os.getenv("LOGIN_METATRADER")),
        password=os.getenv("PASSWORD_METATRADER"),
        server=os.getenv("SERVER_METATRADER"),
    )
    mt.connect()

    try:
        while True:
            print("\n⏳ Atualizando dados do MetaTrader...")

            # --- baixa dados do MetaTrader ---
            df_mt = mt.fetch_mt5_data(ativo="WDO$N", timeframe="H1", n_barras=99999)
            print(df_mt.info())

            # --- grava incrementalmente no Mongo ---
            mongo.incremental_complete_from_df(
                df_mt,
                symbol="WDO$N",
                timeframe="H1",
                drop_last=True,             # ignora candle em formação
                create_placeholders=True,   # preenche gaps
            )

            # --- exibe últimos 100 candles ---
            df = mongo.get_dataframe("WDO$N", "H1", limit=100)
            print("\n📈 Últimos candles:")
            print(df.tail())

            # --- total de registros ---
            df_all = mongo.get_dataframe("WDO$N", "H1", limit=-1)
            print(f"📊 Total no banco: {len(df_all)} registros")

            print("⏲️ Aguardando 2 minutos para próxima atualização...\n")
            time.sleep(10)
            

    except KeyboardInterrupt:
        print("\n🛑 Encerrando loop...")

    finally:
        mt.disconnect()
        mongo.disconnect()


if __name__ == "__main__":
    main()
