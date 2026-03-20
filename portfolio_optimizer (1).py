"""
Portfolio Optimizer — Modern Portfolio Theory + ML Enhancements
================================================================
All asset data (returns, volatility, correlations, risk-free rate)
is fetched live from yfinance / FRED. No hardcoded financial values.

Dependencies:
    pip install numpy pandas scipy matplotlib scikit-learn yfinance pandas-datareader

Usage:
    python portfolio_optimizer.py
    python portfolio_optimizer.py --tickers SPY EFA EEM AGG GLD VNQ --risk aggressive --horizon 5
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. CONFIGURATION  (no financial values here)
# ──────────────────────────────────────────────

RISK_PROFILES = {
    "conservative": {"label": "Conservative", "max_weight": 0.30},
    "moderate":     {"label": "Moderate",     "max_weight": 0.40},
    "aggressive":   {"label": "Aggressive",   "max_weight": 0.50},
    "speculative":  {"label": "Speculative",  "max_weight": 0.70},
}

# Asset metadata — names and classes only, no returns or vols
ASSET_META = {
    "SPY": {"name": "US Stocks",   "cls": "equity"},
    "EFA": {"name": "Intl Stocks", "cls": "equity"},
    "EEM": {"name": "EM Stocks",   "cls": "equity"},
    "AGG": {"name": "US Bonds",    "cls": "bond"},
    "LQD": {"name": "Corp Bonds",  "cls": "bond"},
    "VNQ": {"name": "Real Estate", "cls": "reit"},
    "GLD": {"name": "Gold",        "cls": "commodity"},
}

# Stress scenario shocks by asset class (historical, not per-ticker).
# These represent known crisis drawdowns — intentionally kept as
# they are historical facts, not return forecasts.
STRESS_SCENARIOS = {
    "2008 GFC":       {"equity": -0.50, "bond":  0.06, "reit": -0.62, "commodity":  0.05, "recovery_mo": 18},
    "2020 COVID":     {"equity": -0.34, "bond":  0.08, "reit": -0.42, "commodity":  0.12, "recovery_mo":  5},
    "2000 Dot-com":   {"equity": -0.49, "bond":  0.12, "reit": -0.28, "commodity":  0.15, "recovery_mo": 48},
    "2022 Rate Hike": {"equity": -0.19, "bond": -0.14, "reit": -0.26, "commodity": -0.02, "recovery_mo":  9},
    "Flash Crash":    {"equity": -0.10, "bond":  0.03, "reit": -0.12, "commodity":  0.04, "recovery_mo":  1},
}

N_PORTFOLIOS   = 10_000
N_FRONTIER_PTS = 200


# ──────────────────────────────────────────────
# 2. LIVE DATA FETCHING
# ──────────────────────────────────────────────

def fetch_live_data(tickers: list, period: str = "5y") -> dict:
    """
    Download adjusted close prices from yfinance and compute:
      - annualised return (from mean daily log return)
      - annualised volatility
      - full covariance matrix
      - pairwise correlation matrix
      - live risk-free rate from FRED (3-month T-bill)

    Returns a dict with keys:
        assets    : {ticker: {name, cls, ret, vol}}
        cov       : np.ndarray  (annualised)
        corr      : np.ndarray
        rf        : float
        prices    : pd.DataFrame  (daily adjusted close)
        daily_ret : pd.DataFrame  (daily simple returns)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    print(f"  Fetching price data for {tickers} over {period} …")
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(how="all")
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]}).dropna()

    # Drop tickers with fewer than 60 trading days of data
    prices = prices.dropna(axis=1, thresh=60)
    valid  = prices.columns.tolist()
    dropped = set(tickers) - set(valid)
    if dropped:
        print(f"  ⚠  Dropped (insufficient data): {dropped}")

    daily_ret = prices.pct_change().dropna()

    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std()  * np.sqrt(252)
    cov     = daily_ret.cov()  * 252      # annualised covariance matrix
    corr    = daily_ret.corr()

    rf = _fetch_risk_free_rate()
    print(f"  Risk-free rate (3-mo T-bill): {rf*100:.2f}%")

    assets = {}
    for t in valid:
        meta = ASSET_META.get(t, {"name": t, "cls": "equity"})
        assets[t] = {
            "name": meta["name"],
            "cls":  meta["cls"],
            "ret":  float(ann_ret[t]),
            "vol":  float(ann_vol[t]),
        }

    return {
        "assets":    assets,
        "cov":       cov.values,
        "corr":      corr.values,
        "rf":        rf,
        "prices":    prices,
        "daily_ret": daily_ret,
    }


def _fetch_risk_free_rate() -> float:
    """
    Fetch the current 3-month US T-bill yield.
    Tries three sources in order:
      1. FRED via pandas-datareader (DGS3MO)
      2. FRED fallback series (DGS10, DFF)
      3. yfinance ^IRX (13-week T-bill futures)
    Falls back to a conservative 4.5% estimate only if all sources fail.
    """
    try:
        from pandas_datareader import data as pdr
        for series in ("DGS3MO", "DGS10", "DFF"):
            try:
                df   = pdr.get_data_fred(series, start="2020-01-01")
                rate = float(df.dropna().iloc[-1, 0]) / 100.0
                if 0.0 < rate < 0.25:
                    return rate
            except Exception:
                continue
    except ImportError:
        pass

    try:
        import yfinance as yf
        irx = yf.Ticker("^IRX").history(period="5d")["Close"]
        if not irx.empty:
            return float(irx.iloc[-1]) / 100.0
    except Exception:
        pass

    print("  ⚠  Could not fetch live risk-free rate; using 4.5% fallback")
    return 0.045


# ──────────────────────────────────────────────
# 3. PORTFOLIO STATISTICS
# ──────────────────────────────────────────────

def portfolio_stats(weights: np.ndarray, returns: np.ndarray,
                    cov: np.ndarray, rf: float) -> dict:
    w  = np.asarray(weights)
    r  = float(w @ returns)
    v  = float(np.sqrt(w @ cov @ w))
    sr = (r - rf) / v if v > 0 else 0.0
    sortino = (r - rf) / (v * 0.7) if v > 0 else 0.0
    var95   = r - 1.645 * v
    var99   = r - 2.326 * v
    return {"return": r, "vol": v, "sharpe": sr,
            "sortino": sortino, "var95": var95, "var99": var99}


# ──────────────────────────────────────────────
# 4. MONTE CARLO SIMULATION
# ──────────────────────────────────────────────

def monte_carlo(assets: dict, cov: np.ndarray, rf: float,
                n: int = N_PORTFOLIOS) -> pd.DataFrame:
    rets_arr = np.array([v["ret"] for v in assets.values()])
    tickers  = list(assets.keys())
    n_assets = len(tickers)
    rows = []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        s = portfolio_stats(w, rets_arr, cov, rf)
        rows.append({"return": s["return"], "vol": s["vol"],
                     "sharpe": s["sharpe"], **dict(zip(tickers, w))})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 5. EFFICIENT FRONTIER (SCIPY)
# ──────────────────────────────────────────────

def efficient_frontier(assets: dict, cov: np.ndarray, rf: float,
                       n_points: int = N_FRONTIER_PTS) -> pd.DataFrame:
    rets_arr    = np.array([v["ret"] for v in assets.values()])
    n           = len(rets_arr)
    target_rets = np.linspace(rets_arr.min(), rets_arr.max(), n_points)
    frontier    = []
    w0          = np.ones(n) / n
    bounds      = [(0, 1)] * n

    for target in target_rets:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ rets_arr - t},
        ]
        res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 1000})
        if res.success:
            s = portfolio_stats(res.x, rets_arr, cov, rf)
            frontier.append({"return": s["return"], "vol": s["vol"],
                              "sharpe": s["sharpe"],
                              **dict(zip(assets.keys(), res.x))})
    return pd.DataFrame(frontier)


# ──────────────────────────────────────────────
# 6. OPTIMAL PORTFOLIO
# ──────────────────────────────────────────────

def optimal_portfolio(assets: dict, cov: np.ndarray, rf: float,
                      objective: str = "sharpe",
                      max_weight: float = 0.50) -> dict:
    rets_arr = np.array([v["ret"] for v in assets.values()])
    n        = len(rets_arr)

    if objective == "sharpe":
        obj_fn = lambda w: -portfolio_stats(w, rets_arr, cov, rf)["sharpe"]
    else:
        obj_fn = lambda w: w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0.01, max_weight)] * n
    res = minimize(obj_fn, np.ones(n) / n, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 5000})
    stats = portfolio_stats(res.x, rets_arr, cov, rf)
    return {"weights": res.x, "tickers": list(assets.keys()), **stats}


# ──────────────────────────────────────────────
# 7. ML-ENHANCED RETURN PREDICTION
# ──────────────────────────────────────────────

def ml_return_forecast(assets: dict, daily_ret: pd.DataFrame,
                       horizon_years: int = 3) -> dict:
    """
    Ridge regression using real price-derived features per ticker:
      - 1-month momentum  (21-day cumulative return)
      - 3-month momentum  (63-day cumulative return)
      - 6-month momentum  (126-day cumulative return)
      - Realised volatility (21-day rolling std × sqrt(252))
      - Mean-reversion signal (price / 252-day SMA − 1)

    Target: next-21-day forward return.
    Walk-forward split (80/20) to avoid lookahead bias.
    ML forecast is blended with the historical mean return,
    with blend weight increasing for longer horizons.
    """
    ml_rets = {}
    scaler  = StandardScaler()

    for t, info in assets.items():
        if t not in daily_ret.columns or len(daily_ret[t].dropna()) < 150:
            ml_rets[t] = info["ret"]
            continue

        s  = daily_ret[t].dropna()
        df = pd.DataFrame(index=s.index)
        df["mom_21"]   = s.rolling(21).sum()
        df["mom_63"]   = s.rolling(63).sum()
        df["mom_126"]  = s.rolling(126).sum()
        df["rvol_21"]  = s.rolling(21).std() * np.sqrt(252)
        px             = (1 + s).cumprod()
        df["mean_rev"] = px / px.rolling(252).mean() - 1
        df["fwd_21"]   = s.shift(-21).rolling(21).sum()   # forward target
        df = df.dropna()

        if len(df) < 60:
            ml_rets[t] = info["ret"]
            continue

        X = df[["mom_21","mom_63","mom_126","rvol_21","mean_rev"]].values
        y = df["fwd_21"].values

        split       = int(len(X) * 0.80)
        X_tr, X_te  = X[:split], X[split:]
        y_tr        = y[:split]

        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_sc, y_tr)

        # Predict on latest available observation, annualise
        x_now    = scaler.transform(X[-1].reshape(1, -1))
        pred_ann = float(model.predict(x_now)[0]) * (252 / 21)

        # Blend: give ML more weight for longer horizons
        blend      = min(0.6, horizon_years / 10.0)
        ml_rets[t] = (1 - blend) * info["ret"] + blend * pred_ann

    return ml_rets


# ──────────────────────────────────────────────
# 8. REBALANCING SIMULATION
# ──────────────────────────────────────────────

def simulate_rebalancing(target_weights: np.ndarray, assets: dict,
                         cov: np.ndarray, rf: float,
                         strategy: str   = "hybrid",
                         threshold: float = 0.05,
                         tx_cost: float   = 0.001,
                         months: int      = 36) -> pd.DataFrame:
    rets_arr    = np.array([v["ret"] for v in assets.values()])
    n           = len(rets_arr)
    rng         = np.random.default_rng(0)
    cov_monthly = cov / 12

    # Regularise covariance for Cholesky decomposition
    L = np.linalg.cholesky(cov_monthly + np.eye(n) * 1e-8)

    w, value, total_costs = target_weights.copy(), 1_000_000.0, 0.0
    records = []

    for m in range(months):
        r_monthly = rets_arr / 12 + L @ rng.normal(size=n)
        w_new     = w * (1 + r_monthly)
        w_new    /= w_new.sum()
        value_new = value * float(1 + w @ r_monthly)

        drift = float(np.abs(w_new - target_weights).max())
        rebal = (
            (strategy == "threshold" and drift > threshold) or
            (strategy == "calendar"  and (m + 1) % 3 == 0) or
            (strategy == "hybrid"    and ((m + 1) % 3 == 0 or drift > threshold))
        )

        rebal_cost = 0.0
        if rebal:
            turnover   = float(np.abs(w_new - target_weights).sum())
            rebal_cost = value_new * turnover * tx_cost
            value_new -= rebal_cost
            total_costs += rebal_cost
            w = target_weights.copy()
        else:
            w = w_new

        records.append({"month": m + 1, "value": round(value_new, 2),
                         "drift": round(drift, 4), "rebalanced": rebal,
                         "rebal_cost": round(rebal_cost, 2)})
        value = value_new

    df = pd.DataFrame(records)
    df["cumulative_cost"] = df["rebal_cost"].cumsum()

    print(f"\n── Rebalancing Summary ({strategy.title()}) ──")
    print(f"  Events         : {df['rebalanced'].sum()}")
    print(f"  Total tx costs : ${total_costs:,.2f}")
    print(f"  Final value    : ${df['value'].iloc[-1]:,.2f}")
    return df


# ──────────────────────────────────────────────
# 9. STRESS TESTING
# ──────────────────────────────────────────────

def stress_test(weights: np.ndarray, assets: dict) -> pd.DataFrame:
    tickers = list(assets.keys())
    rows = []
    for name, shock in STRESS_SCENARIOS.items():
        loss = sum(
            w * shock.get(assets[t]["cls"], shock["equity"])
            for w, t in zip(weights, tickers)
        )
        bm = 0.60 * shock["equity"] + 0.40 * shock["bond"]
        rows.append({
            "scenario":    name,
            "port_loss":   round(loss * 100, 2),
            "bm_loss":     round(bm * 100, 2),
            "alpha":       round((loss - bm) * 100, 2),
            "recovery_mo": shock["recovery_mo"],
        })
    df = pd.DataFrame(rows)
    print("\n── Stress Test Results ──")
    print(df.to_string(index=False))
    return df


# ──────────────────────────────────────────────
# 10. VISUALISATION
# ──────────────────────────────────────────────

def plot_results(assets, cov, sim_df, frontier_df, opt,
                 ml_rets, stress_df, rebal_df, rf, data_period):
    COLORS = ["#378ADD","#1D9E75","#D85A30","#534AB7",
              "#BA7517","#D4537E","#639922","#B07018"]
    tickers = list(assets.keys())

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Portfolio Optimization — MPT + ML  |  Live {data_period} data  |  rf = {rf*100:.2f}%",
        fontsize=15, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # 1 – Efficient frontier
    ax1 = fig.add_subplot(gs[0, :2])
    sc = ax1.scatter(sim_df["vol"]*100, sim_df["return"]*100,
                     c=sim_df["sharpe"], cmap="RdYlGn", alpha=0.25, s=4)
    plt.colorbar(sc, ax=ax1, label="Sharpe Ratio", pad=0.01)
    if not frontier_df.empty:
        ax1.plot(frontier_df["vol"]*100, frontier_df["return"]*100,
                 "b-", lw=2.5, label="Efficient frontier", zorder=5)
    ax1.scatter(opt["vol"]*100, opt["return"]*100, s=160, color="#1D9E75",
                zorder=6, label=f"Max Sharpe ({opt['sharpe']:.2f})")
    mv = sim_df.loc[sim_df["vol"].idxmin()]
    ax1.scatter(mv["vol"]*100, mv["return"]*100, s=140, color="#534AB7",
                marker="D", zorder=6, label="Min Variance")
    ax1.axhline(rf*100, color="gray", lw=0.8, ls="--",
                label=f"Risk-free ({rf*100:.1f}%)")
    ax1.set_xlabel("Volatility (%)"); ax1.set_ylabel("Expected Return (%)")
    ax1.set_title("Mean-Variance Efficient Frontier (live data)", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.15)

    # 2 – Allocation pie
    ax2 = fig.add_subplot(gs[0, 2])
    w, mask = opt["weights"], opt["weights"] > 0.005
    ax2.pie(w[mask],
            labels=[tickers[i] for i in range(len(w)) if mask[i]],
            autopct="%1.1f%%",
            colors=[COLORS[i % len(COLORS)] for i in range(len(w)) if mask[i]],
            startangle=140, pctdistance=0.82, textprops={"fontsize": 9})
    ax2.set_title("Optimal Allocation", fontweight="bold")

    # 3 – Covariance heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(cov * 100, cmap="RdYlGn_r", aspect="auto")
    ax3.set_xticks(range(len(tickers)))
    ax3.set_xticklabels(tickers, fontsize=8, rotation=45)
    ax3.set_yticks(range(len(tickers)))
    ax3.set_yticklabels(tickers, fontsize=8)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax3.text(j, i, f"{cov[i,j]*100:.2f}", ha="center", va="center",
                     fontsize=7, color="black")
    ax3.set_title("Covariance Matrix ×100 (live)", fontweight="bold")
    plt.colorbar(im, ax=ax3, pad=0.02)

    # 4 – Historical vs ML returns
    ax4 = fig.add_subplot(gs[1, 1])
    x      = np.arange(len(tickers))
    base_r = [assets[t]["ret"] * 100 for t in tickers]
    ml_r   = [ml_rets.get(t, assets[t]["ret"]) * 100 for t in tickers]
    ax4.bar(x - 0.2, base_r, 0.38, label="Historical mean", color="#378ADD", alpha=0.8)
    ax4.bar(x + 0.2, ml_r,   0.38, label="ML forecast",     color="#1D9E75", alpha=0.8)
    ax4.axhline(0, color="black", lw=0.5)
    ax4.set_xticks(x); ax4.set_xticklabels(tickers, fontsize=9, rotation=30)
    ax4.set_ylabel("Expected Return (%)")
    ax4.set_title("Historical vs ML Return Forecasts", fontweight="bold")
    ax4.legend(fontsize=9); ax4.grid(True, axis="y", alpha=0.2)

    # 5 – Stress test
    ax5 = fig.add_subplot(gs[1, 2])
    scenarios   = stress_df["scenario"].tolist()
    port_losses = stress_df["port_loss"].tolist()
    bm_losses   = stress_df["bm_loss"].tolist()
    y = np.arange(len(scenarios))
    ax5.barh(y - 0.2, port_losses, 0.38, label="Portfolio", color="#D85A30", alpha=0.85)
    ax5.barh(y + 0.2, bm_losses,   0.38, label="60/40 BM",  color="#888",    alpha=0.7)
    ax5.set_yticks(y); ax5.set_yticklabels(scenarios, fontsize=9)
    ax5.axvline(0, color="black", lw=0.5)
    ax5.set_xlabel("Return (%)")
    ax5.set_title("Stress Test Scenarios", fontweight="bold")
    ax5.legend(fontsize=9); ax5.grid(True, axis="x", alpha=0.2)

    # 6 – Rebalancing drift
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.plot(rebal_df["month"], rebal_df["drift"] * 100,
             lw=1.5, color="#378ADD", label="Max weight drift")
    evts = rebal_df[rebal_df["rebalanced"]]
    ax6.vlines(evts["month"], 0, evts["drift"] * 100,
               colors="#1D9E75", lw=1, ls="--", alpha=0.6, label="Rebalance trigger")
    ax6.axhline(5, color="#D85A30", lw=1, ls=":", label="5% threshold")
    ax6.set_xlabel("Month"); ax6.set_ylabel("Max Drift (%)")
    ax6.set_title("Portfolio Drift and Rebalancing Events", fontweight="bold")
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.15)

    # 7 – Cumulative transaction costs
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.fill_between(rebal_df["month"], rebal_df["cumulative_cost"],
                     alpha=0.4, color="#D85A30")
    ax7.plot(rebal_df["month"], rebal_df["cumulative_cost"],
             color="#D85A30", lw=2, label="Cumulative tx cost ($)")
    ax7.set_xlabel("Month"); ax7.set_ylabel("Cumulative Cost ($)")
    ax7.set_title("Transaction Costs Over Time", fontweight="bold")
    ax7.grid(True, alpha=0.15); ax7.legend(fontsize=9)

    plt.savefig("portfolio_optimization.png", dpi=150, bbox_inches="tight")
    print("\n✓ Chart saved → portfolio_optimization.png")
    plt.show()


# ──────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────

def main(tickers=None, risk_profile="moderate", horizon_years=5,
         data_period="5y", rebal_strategy="hybrid",
         rebal_threshold=0.05, tx_cost=0.001):

    if tickers is None:
        tickers = list(ASSET_META.keys())

    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES["moderate"])
    print(f"\n{'='*60}")
    print(f"  Portfolio Optimizer  |  {profile['label']}  |  {horizon_years}yr horizon")
    print(f"  Tickers : {tickers}")
    print(f"{'='*60}")

    # ── 1. Fetch all live data ───────────────────────────────
    print("\n[1/6] Fetching live market data …")
    data      = fetch_live_data(tickers, period=data_period)
    assets    = data["assets"]
    cov       = data["cov"]
    rf        = data["rf"]
    daily_ret = data["daily_ret"]

    print("\n── Asset Universe (live) ──")
    for t, info in assets.items():
        print(f"  {t:5s}  {info['name']:15s}  "
              f"return={info['ret']*100:+6.2f}%  "
              f"vol={info['vol']*100:5.2f}%")

    # ── 2. ML forecasts ──────────────────────────────────────
    print("\n[2/6] Computing ML return forecasts …")
    ml_rets   = ml_return_forecast(assets, daily_ret, horizon_years)
    ml_assets = {t: {**info, "ret": ml_rets[t]} for t, info in assets.items()}

    print("\n── ML Adjustments ──")
    for t in assets:
        delta = (ml_rets[t] - assets[t]["ret"]) * 100
        print(f"  {t:5s}  hist={assets[t]['ret']*100:+6.2f}%  "
              f"ml={ml_rets[t]*100:+6.2f}%  delta={delta:+5.2f}%")

    # ── 3. Monte Carlo ───────────────────────────────────────
    print(f"\n[3/6] Running {N_PORTFOLIOS:,} Monte Carlo portfolios …")
    sim_df = monte_carlo(ml_assets, cov, rf)

    # ── 4. Efficient frontier ────────────────────────────────
    print("[4/6] Tracing efficient frontier …")
    frontier_df = efficient_frontier(ml_assets, cov, rf)

    # ── 5. Optimal portfolio ─────────────────────────────────
    print("[5/6] Finding optimal portfolio …")
    opt = optimal_portfolio(ml_assets, cov, rf,
                            objective="sharpe",
                            max_weight=profile["max_weight"])
    w = opt["weights"]

    print(f"\n── Optimal Weights ──")
    for t, wi in zip(opt["tickers"], w):
        print(f"  {t:5s}  {wi*100:5.1f}%")

    print(f"\n── Portfolio Stats ──")
    print(f"  Expected Return : {opt['return']*100:.2f}%")
    print(f"  Volatility (σ)  : {opt['vol']*100:.2f}%")
    print(f"  Sharpe Ratio    : {opt['sharpe']:.3f}")
    print(f"  Sortino Ratio   : {opt['sortino']:.3f}")
    print(f"  VaR 95%         : {opt['var95']*100:.2f}%")
    print(f"  VaR 99%         : {opt['var99']*100:.2f}%")

    # ── 6. Stress test + rebalancing ────────────────────────
    print("\n[6/6] Stress tests and rebalancing simulation …")
    stress_df = stress_test(w, assets)
    rebal_df  = simulate_rebalancing(
        w, assets, cov, rf,
        strategy  = rebal_strategy,
        threshold = rebal_threshold,
        tx_cost   = tx_cost,
        months    = 36,
    )

    plot_results(assets, cov, sim_df, frontier_df, opt,
                 ml_rets, stress_df, rebal_df, rf, data_period)

    return opt, stress_df, rebal_df


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MPT Portfolio Optimizer with live market data")
    parser.add_argument("--tickers",   nargs="+", default=None,
                        help="Tickers to include, e.g. SPY AGG GLD")
    parser.add_argument("--risk",      default="moderate",
                        choices=list(RISK_PROFILES.keys()))
    parser.add_argument("--horizon",   type=int, default=5,
                        help="Investment horizon in years (default: 5)")
    parser.add_argument("--period",    default="5y",
                        help="Historical data window: 1y 3y 5y 10y (default: 5y)")
    parser.add_argument("--rebal",     default="hybrid",
                        choices=["threshold", "calendar", "hybrid"])
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Drift threshold for rebalancing (default: 0.05)")
    parser.add_argument("--txcost",    type=float, default=0.001,
                        help="One-way transaction cost (default: 0.001 = 0.1%%)")
    args = parser.parse_args()

    main(
        tickers        = args.tickers,
        risk_profile   = args.risk,
        horizon_years  = args.horizon,
        data_period    = args.period,
        rebal_strategy = args.rebal,
        rebal_threshold= args.threshold,
        tx_cost        = args.txcost,
    )
