import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# --- Paramètres ---
crypto = "BTC-USD"
start_date = "2014-09-17"
end_date   = "2025-08-31"

train_file_name = "1-btc_processed_train.csv"
test_file_name  = "1-btc_processed_test.csv"

save_dir = "data/data_processed_hmm"
os.makedirs(save_dir, exist_ok=True)


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Télécharge les données OHLCV via yfinance avec format cohérent."""
    df = yf.download(ticker, start=start, end=end, interval="1d")
    df = df.reset_index()

    # --- Corriger si MultiIndex ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardiser les noms de colonnes
    df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={
        "date": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })

    print("Colonnes après nettoyage:", df.columns.tolist())
    return df


def preprocess_and_save(df: pd.DataFrame, output_file: str):
    """Ajoute features + colonnes vides market_regime/signal et sauvegarde en CSV."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    if "close" not in df.columns:
        raise ValueError(f"Colonne 'close' introuvable. Colonnes actuelles: {df.columns.tolist()}")

    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Features
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_5'] = df['log_returns'].rolling(window=5).std()
    df['momentum_5'] = df['log_returns'].rolling(window=5).mean()

    df = df.dropna(subset=['log_returns', 'volatility_5', 'momentum_5']).copy()

    # Colonnes vides pour HMM
    df['market_regime'] = np.nan
    df['signal'] = np.nan

    df.to_csv(output_file, index=False)
    print("saved:", output_file, "| rows:", len(df))


if __name__ == "__main__":
    # Train
    df_train = download_data(crypto, "2014-09-17", "2020-12-31")
    preprocess_and_save(df_train, os.path.join(save_dir, train_file_name))

    # Test
    df_test = download_data(crypto, "2021-01-01", "2025-08-31")
    preprocess_and_save(df_test, os.path.join(save_dir, test_file_name))
