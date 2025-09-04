import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = "data/data_processed_hmm"
TEST_FILE_SIGNALS = "2-btc_with_signals_test.csv"
MODEL_PATH = "2-models/hmm_model.pkl"

USE_SHORT_CURVE = False  # False = long/flat
FEES_BPS = 0.0           # frais aller-retour (ex: 0.001 = 10bps)
TITLE = "BTC vs HMM (no-repaint, 2021–2025)"

# Charger modèle (signal_map, hyperparamètres)
with open(f"{BASE_DIR}/{MODEL_PATH}", "rb") as f:
    saved = pickle.load(f)

params = saved.get('params', {})

# Charger CSV enrichi
df = pd.read_csv(f"{BASE_DIR}/{TEST_FILE_SIGNALS}")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df['signal'] = df['signal'].fillna(0)

# Position
if USE_SHORT_CURVE:
    df['position'] = df['signal'].clip(-1, 1).astype(float)
else:
    df['position'] = (df['signal'] == 1).astype(float)

# Rendements quotidiens
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['log_returns'].fillna(0, inplace=True)

# Variation de position pour coûts
pos_change = df['position'].diff().abs().fillna(0.0)
fees_series = FEES_BPS * pos_change

# PnL stratégie
df['strategy_log_ret'] = df['position'] * df['log_returns'] - fees_series
df['strategy_value'] = df['strategy_log_ret'].cumsum().apply(np.exp) * df['close'].iloc[0]

# PnL Buy & Hold
df['bh_value'] = df['log_returns'].cumsum().apply(np.exp) * df['close'].iloc[0]

# Stats rapides
def stats(series):
    mu = series.mean()
    sd = series.std(ddof=0)
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(252) if sd>0 else 0.0
    cum = float(series.cumsum().apply(np.exp).iloc[-1])
    return cum, sharpe

cum_s, shp_s = stats(df['strategy_log_ret'])
cum_b, shp_b = stats(df['log_returns'])

print("Hyperparamètres retenus:", params)
print(f"Performance stratégie: x{cum_s:.2f} | Sharpe {shp_s:.2f}")
print(f"Buy & Hold BTC      : x{cum_b:.2f} | Sharpe {shp_b:.2f}")

# Plot prix BTC vs stratégie
plt.figure(figsize=(12,6))
plt.plot(df['timestamp'], df['bh_value'], label="Buy & Hold BTC")
plt.plot(df['timestamp'], df['strategy_value'], label="HMM Strategy")
plt.title(f"{TITLE}\n"
          f"n_components={params.get('n_components')} | "
          f"delay={params.get('momentum_delay')} | "
          f"n_moms={params.get('n_momentums')} | "
          f"fees={FEES_BPS:.3f}")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
