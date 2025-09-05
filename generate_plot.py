import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

ticker = "BTC-USD"

BASE_DIR = f"data/data_processed_hmm/{ticker}"
TEST_FILE_SIGNALS = f"{ticker}_with_signals_test.csv"
MODEL_PATH = "models/hmm_model.pkl"

USE_SHORT_CURVE = False  # False = long/flat
FEES_PCT = 0.0           # frais aller-retour en %
TITLE = f"HMM Strategy vs {ticker} Buy & Hold"

# --- Charger modèle ---
with open(f"{BASE_DIR}/{MODEL_PATH}", "rb") as f:
    saved = pickle.load(f)

params = saved.get('params', {})

# --- Charger CSV enrichi ---
df = pd.read_csv(f"{BASE_DIR}/{TEST_FILE_SIGNALS}")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df['signal'] = df['signal'].fillna(0)

# --- Position selon stratégie ---
if USE_SHORT_CURVE:
    df['position'] = df['signal'].clip(-1, 1).astype(float)
else:
    df['position'] = (df['signal'] == 1).astype(float)

# --- Rendements quotidiens ---
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['log_returns'].fillna(0, inplace=True)

# --- Coûts de transaction ---
pos_change = df['position'].diff().abs().fillna(0.0)
fees_series = (FEES_PCT/100) * pos_change
df['strategy_log_ret'] = df['position'] * df['log_returns'] - fees_series

# --- Cumulatif returns ---
df['strategy_cumret'] = np.exp(df['strategy_log_ret'].cumsum())
df['bh_cumret'] = np.exp(df['log_returns'].cumsum())

# --- Max drawdown ---
def compute_drawdown(cumret_series):
    rolling_max = cumret_series.cummax()
    drawdown = (cumret_series - rolling_max) / rolling_max
    return drawdown

df['strategy_dd'] = compute_drawdown(df['strategy_cumret'])
df['bh_dd'] = compute_drawdown(df['bh_cumret'])

# --- Statistiques de performance ---
def compute_stats(strategy_ret, benchmark_ret):
    mu_s, sd_s = strategy_ret.mean(), strategy_ret.std(ddof=0)
    sharpe = (mu_s / (sd_s + 1e-12)) * np.sqrt(252) if sd_s>0 else 0.0
    cum_return = np.exp(strategy_ret.cumsum().iloc[-1])

    max_dd = compute_drawdown(np.exp(strategy_ret.cumsum())).min() * 100  # en %

    # Alpha & Beta
    X = benchmark_ret.values.reshape(-1,1)
    y = strategy_ret.values
    reg = LinearRegression().fit(X, y)
    beta = reg.coef_[0]
    alpha = reg.intercept_ * 252  # annualisé approximatif

    return {
        'cumulative': cum_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'alpha': alpha,
        'beta': beta
    }

stats_strategy = compute_stats(df['strategy_log_ret'], df['log_returns'])
stats_bh = compute_stats(df['log_returns'], df['log_returns'])

# --- Affichage des stats ---
print("Hyperparamètres retenus:", params)
print("\n--- Stratégie HMM ---")
for k,v in stats_strategy.items():
    print(f"{k}: {v:.4f}")
print(f"\n--- Buy & Hold {ticker} ---")
for k,v in stats_bh.items():
    print(f"{k}: {v:.4f}")

# --- Plot cumulatif return + drawdown + positions ---
fig, axes = plt.subplots(3,1, figsize=(14,12), sharex=True)

# Cumulatif return
axes[0].plot(df['timestamp'], df['bh_cumret'], label=f"Buy & Hold {ticker}", color='blue')
axes[0].plot(df['timestamp'], df['strategy_cumret'], label="HMM Strategy", color='orange')
axes[0].set_ylabel("Cumulative Return")
axes[0].set_title(f"{TITLE}\n"
                  f"n_components={params.get('n_components')} | "
                  f"delay={params.get('momentum_delay')} | "
                  f"n_moms={params.get('n_momentums')} | "
                  f"fees={FEES_PCT:.3f}%")
axes[0].legend()
axes[0].grid(True)

# Drawdown
axes[1].plot(df['timestamp'], df['strategy_dd']*100, label="HMM Strategy DD", color='orange')
axes[1].plot(df['timestamp'], df['bh_dd']*100, label=f"{ticker} Buy & Hold DD", color='blue')
axes[1].set_ylabel("Drawdown (%)")
axes[1].legend()
axes[1].grid(True)

# Positions
axes[2].step(df['timestamp'], df['position'], label="Position (1=Long,0=Flat)", color='green', where='post')
axes[2].set_ylabel("Position")
axes[2].set_xlabel("Date")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
