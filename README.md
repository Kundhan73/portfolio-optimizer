# Portfolio Optimizer

A Python-based portfolio optimization system built on **Modern Portfolio Theory (MPT)** with **ML-enhanced return forecasting**. All market data is fetched live — no hardcoded financial values.

---

## Features

- **Live market data** via `yfinance` — real returns, volatility, and correlations computed from historical prices
- **Live risk-free rate** fetched from FRED (3-month US T-bill)
- **Efficient frontier** traced using SciPy constrained optimization (SLSQP)
- **Monte Carlo simulation** — 10,000 random portfolios to visualize the feasible set
- **ML return forecasting** — Ridge regression on momentum, volatility, and mean-reversion features with walk-forward validation
- **Stress testing** against 5 historical crash scenarios (2008 GFC, COVID-19, Dot-com bust, 2022 rate hikes, Flash Crash)
- **Rebalancing simulation** — threshold, calendar, and hybrid strategies with transaction cost modeling
- **7-panel visualization** saved as `portfolio_optimization.png`

---

## Default Asset Universe

| Ticker | Asset | Class |
|--------|-------|-------|
| SPY | US Stocks | Equity |
| EFA | International Stocks | Equity |
| EEM | Emerging Market Stocks | Equity |
| AGG | US Bonds | Bond |
| LQD | Corporate Bonds | Bond |
| VNQ | Real Estate | REIT |
| GLD | Gold | Commodity |

---

## Installation

```bash
pip install numpy pandas scipy matplotlib scikit-learn yfinance pandas-datareader
```

---

## Usage

### Google Colab
Open a new notebook, paste the script, and run. Settings are controlled by the `main()` call at the bottom of the script:

```python
main(
    tickers        = None,           # None = use all 7 default ETFs
    risk_profile   = "moderate",     # conservative / moderate / aggressive / speculative
    horizon_years  = 5,
    data_period    = "5y",           # 1y / 3y / 5y / 10y
    rebal_strategy = "hybrid",       # threshold / calendar / hybrid
    rebal_threshold= 0.05,
    tx_cost        = 0.001,
)
```

### Terminal
```bash
# Default settings
python portfolio_optimizer.py

# Custom tickers and risk profile
python portfolio_optimizer.py --tickers SPY AGG GLD TLT --risk aggressive --horizon 7 --period 10y
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tickers` | All 7 ETFs | Space-separated list of tickers |
| `--risk` | `moderate` | Risk profile: conservative / moderate / aggressive / speculative |
| `--horizon` | `5` | Investment horizon in years |
| `--period` | `5y` | Historical data window: 1y / 3y / 5y / 10y |
| `--rebal` | `hybrid` | Rebalancing strategy: threshold / calendar / hybrid |
| `--threshold` | `0.05` | Drift threshold that triggers rebalancing (5%) |
| `--txcost` | `0.001` | One-way transaction cost (0.1%) |

---

## Output

Running the script prints a full summary to the console and saves a chart:

```
======================================================
  Portfolio Optimizer  |  Moderate  |  5yr horizon
  Tickers: ['SPY', 'EFA', 'EEM', 'AGG', 'LQD', 'VNQ', 'GLD']
======================================================

[1/6] Fetching live market data ...
[2/6] Computing ML return forecasts ...
[3/6] Running 10,000 Monte Carlo portfolios ...
[4/6] Tracing efficient frontier ...
[5/6] Finding optimal portfolio ...
[6/6] Stress tests and rebalancing simulation ...

── Optimal Weights ──
  SPY    28.4%
  EFA    12.1%
  ...

── Portfolio Stats ──
  Expected Return :  9.84%
  Volatility (σ)  : 11.20%
  Sharpe Ratio    :  0.521
  ...

✓ Chart saved → portfolio_optimization.png
```

The saved chart includes: efficient frontier, optimal allocation pie, covariance heatmap, ML vs historical return comparison, stress test scenarios, rebalancing drift over 36 months, and cumulative transaction costs.

---

## How It Works

### Modern Portfolio Theory
The optimizer finds weights that maximize the Sharpe ratio — return per unit of risk — subject to constraints (no shorting, max weight per asset based on risk profile). The efficient frontier is computed by solving a series of minimum-variance problems at different target return levels.

### ML Return Forecasting
For each asset, a Ridge regression model is trained on price-derived features: short, medium, and long-term momentum, realized volatility, and a mean-reversion signal. A walk-forward train/test split prevents lookahead bias. The ML forecast is blended with the historical mean return, with more weight given to the ML signal for longer investment horizons.

### Stress Testing
Portfolio losses are estimated by applying historical crisis drawdowns (by asset class) to the optimal weights, then compared against a 60/40 stock-bond benchmark.

### Rebalancing
Monthly simulation over 36 months using correlated random returns drawn from the live covariance matrix. Rebalancing is triggered by drift, calendar schedule, or both, with transaction costs deducted from portfolio value.

---

## Disclaimer

This project is for educational purposes only. It is not financial advice. Past performance and historical correlations do not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

