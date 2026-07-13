"""Portafolio IPSA: frontera eficiente (máximo Sharpe) y ticker semanal."""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------
# Parámetros
# ----------------------------------------
LOOKBACK_WEEKS = 52
PORTFOLIO_SIZE = 5
MONTE_CARLO_SAMPLES = 8_000
RISK_FREE_RATE = 0.04  # tasa anual aprox. (CLP)
TRADING_WEEKS = 52

IPSA_STOCKS = [
    "SQM-B.SN", "CHILE.SN", "BSANTANDER.SN", "COPEC.SN", "ENELAM.SN", "CENCOSUD.SN",
    "CMPC.SN", "BCI.SN", "FALABELLA.SN", "ENELCHILE.SN", "PARAUCO.SN", "COLBUN.SN",
    "CCU.SN", "ANDINA-B.SN", "VAPORES.SN", "AGUAS-A.SN", "QUINENCO.SN", "CENCOMALLS.SN",
    "LTM.SN", "CONCHATORO.SN", "ENTEL.SN", "CAP.SN", "MALLPLAZA.SN",
    "ECL.SN", "IAM.SN", "SMU.SN", "ITAUCL.SN", "SONDA.SN", "RIPLEY.SN",
]

TICKER_LABELS = {
    "SQM-B.SN": "SQM",
    "CHILE.SN": "Banco de Chile",
    "BSANTANDER.SN": "Santander",
    "COPEC.SN": "Copec",
    "ENELAM.SN": "Enel Américas",
    "CENCOSUD.SN": "Cencosud",
    "CMPC.SN": "CMPC",
    "BCI.SN": "BCI",
    "FALABELLA.SN": "Falabella",
    "ENELCHILE.SN": "Enel Chile",
    "PARAUCO.SN": "Parque Arauco",
    "COLBUN.SN": "Colbún",
    "CCU.SN": "CCU",
    "ANDINA-B.SN": "Andina",
    "VAPORES.SN": "Vapores",
    "AGUAS-A.SN": "Aguas Andinas",
    "QUINENCO.SN": "Quiñenco",
    "CENCOMALLS.SN": "Cencosud Shopping",
    "LTM.SN": "Latam",
    "CONCHATORO.SN": "Concha y Toro",
    "ENTEL.SN": "Entel",
    "CAP.SN": "CAP",
    "MALLPLAZA.SN": "Mall Plaza",
    "ECL.SN": "E-CL",
    "IAM.SN": "Inversiones Aguas",
    "SMU.SN": "SMU",
    "ITAUCL.SN": "Itaú",
    "SONDA.SN": "Sonda",
    "RIPLEY.SN": "Ripley",
}


def download_prices(tickers: list[str], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1wk",
        group_by="ticker",
        auto_adjust=True,
        threads=False,
        progress=False,
    )
    closes = {}
    for ticker in tickers:
        if ticker not in raw:
            continue
        series = raw[ticker]["Close"] if len(tickers) > 1 else raw["Close"]
        if series is not None and not series.dropna().empty:
            closes[ticker] = series.dropna()
    if not closes:
        raise RuntimeError("No se descargaron precios válidos.")
    return pd.DataFrame(closes).dropna(how="all").ffill().dropna()


def annualize(mean_returns: pd.Series, cov: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    return mean_returns * TRADING_WEEKS, cov * TRADING_WEEKS


def portfolio_stats(weights: np.ndarray, mean_returns: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights @ cov.values @ weights))
    return ret, vol


def sharpe_ratio(ret: float, vol: float, rf: float = RISK_FREE_RATE) -> float:
    if vol <= 0:
        return float("-inf")
    return (ret - rf) / vol


def _random_weights(rng: np.random.Generator, k: int, sparse: bool = False) -> np.ndarray:
    """Dirichlet diversificado, o concentrado en 1–5 activos (cubre el extremo derecho)."""
    if sparse:
        n_active = int(rng.integers(1, 6))
        idx = rng.choice(k, size=n_active, replace=False)
        w = np.zeros(k)
        w[idx] = rng.dirichlet(np.ones(n_active))
        return w
    return rng.dirichlet(np.ones(k))


def simulate_frontier(mean_returns: pd.Series, cov: pd.DataFrame, n: int) -> pd.DataFrame:
    tickers = mean_returns.index.tolist()
    k = len(tickers)
    rng = np.random.default_rng(42)
    rows = []

    # Vértices: 100% en cada acción → extremos de la frontera
    for i in range(k):
        w = np.zeros(k)
        w[i] = 1.0
        ret, vol = portfolio_stats(w, mean_returns, cov)
        rows.append({"volatility": vol, "return": ret, "sharpe": sharpe_ratio(ret, vol)})

    # Mitad diversificados, mitad concentrados (si no, Dirichlet se queda en σ ~15%)
    n_div = n // 2
    for _ in range(n_div):
        w = _random_weights(rng, k, sparse=False)
        ret, vol = portfolio_stats(w, mean_returns, cov)
        rows.append({"volatility": vol, "return": ret, "sharpe": sharpe_ratio(ret, vol)})
    for _ in range(n - n_div):
        w = _random_weights(rng, k, sparse=True)
        ret, vol = portfolio_stats(w, mean_returns, cov)
        rows.append({"volatility": vol, "return": ret, "sharpe": sharpe_ratio(ret, vol)})

    return pd.DataFrame(rows)


def max_sharpe_portfolio(mean_returns: pd.Series, cov: pd.DataFrame) -> tuple[pd.Series, float, float, float]:
    # ponytail: Monte Carlo (diversificado + concentrado) en lugar de optimizador cuadrático
    tickers = mean_returns.index.tolist()
    k = len(tickers)
    rng = np.random.default_rng(7)
    best_w = None
    best_sharpe = float("-inf")
    for i in range(25_000):
        w = _random_weights(rng, k, sparse=(i % 2 == 1))
        ret, vol = portfolio_stats(w, mean_returns, cov)
        s = sharpe_ratio(ret, vol)
        if s > best_sharpe:
            best_sharpe = s
            best_w = w
    weights = pd.Series(best_w, index=tickers)
    ret, vol = portfolio_stats(best_w, mean_returns, cov)
    return weights, ret, vol, best_sharpe


def pick_top_stocks(weights: pd.Series, n: int = PORTFOLIO_SIZE) -> pd.Series:
    top = weights.nlargest(n)
    return top / top.sum()


def stock_snapshot(
    ticker: str,
    weight: float,
    prices: pd.Series,
    mean_returns: pd.Series,
    cov: pd.DataFrame,
) -> dict:
    label = TICKER_LABELS.get(ticker, ticker.replace(".SN", ""))
    last = float(prices.iloc[-1])
    prev = float(prices.iloc[-2]) if len(prices) > 1 else last
    change_pct = ((last - prev) / prev * 100) if prev else 0.0
    week_ago = float(prices.iloc[-5]) if len(prices) >= 5 else prev
    week_change_pct = ((last - week_ago) / week_ago * 100) if week_ago else 0.0
    ret = float(mean_returns[ticker] * 100)
    vol = float(np.sqrt(cov.loc[ticker, ticker]) * 100)
    return {
        "ticker": ticker,
        "label": label,
        "weight": round(float(weight), 4),
        "price": round(last, 2),
        "change_pct": round(change_pct, 2),
        "week_change_pct": round(week_change_pct, 2),
        "annual_return_pct": round(ret, 2),
        "annual_volatility_pct": round(vol, 2),
    }


def _to_pct_points(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "volatility": round(float(row["volatility"]) * 100, 3),
            "return": round(float(row["return"]) * 100, 3),
            "sharpe": round(float(row["sharpe"]), 3),
        }
        for _, row in df.iterrows()
    ]


def downsample_frontier(df: pd.DataFrame, max_points: int = 400) -> list[dict]:
    if len(df) <= max_points:
        sample = df
    else:
        sample = df.sample(max_points, random_state=1)
    return _to_pct_points(sample)


def extract_efficient_curve(df: pd.DataFrame, n_bins: int = 50) -> list[dict]:
    """Envolvente superior sobre todo el rango de volatilidad (bins equiespaciados)."""
    work = df.copy()
    vmin, vmax = work["volatility"].min(), work["volatility"].max()
    edges = np.linspace(vmin, vmax, n_bins + 1)
    work["bin"] = pd.cut(work["volatility"], bins=edges, include_lowest=True)
    rows = []
    for _, g in work.groupby("bin", observed=True):
        if g.empty:
            continue
        rows.append(g.loc[g["return"].idxmax()])
    curve = pd.DataFrame(rows).sort_values("volatility")
    # Monótona no-decreciente en retorno (frontera eficiente clásica)
    curve = curve[curve["return"] >= curve["return"].cummax().shift(1, fill_value=-np.inf)]
    return _to_pct_points(curve)


def run() -> dict:
    end = dt.datetime.now()
    start = end - dt.timedelta(weeks=LOOKBACK_WEEKS)
    print(f"Descargando {len(IPSA_STOCKS)} acciones IPSA ({LOOKBACK_WEEKS} semanas)...")
    prices = download_prices(IPSA_STOCKS, start, end)
    returns = prices.pct_change().dropna()
    # Filtrar columnas con datos suficientes
    valid = [c for c in returns.columns if returns[c].count() >= LOOKBACK_WEEKS // 2]
    returns = returns[valid]
    if len(valid) < PORTFOLIO_SIZE:
        raise RuntimeError(f"Solo {len(valid)} acciones con datos suficientes.")

    mean_w = returns.mean()
    cov_w = returns.cov()
    mean_a, cov_a = annualize(mean_w, cov_w)

    weights, port_ret, port_vol, port_sharpe = max_sharpe_portfolio(mean_a, cov_a)
    frontier = simulate_frontier(mean_a, cov_a, MONTE_CARLO_SAMPLES)
    top_weights = pick_top_stocks(weights)

    stocks = [
        stock_snapshot(ticker, w, prices[ticker], mean_a, cov_a)
        for ticker, w in top_weights.items()
    ]

    result = {
        "updated_at": end.strftime("%Y-%m-%d"),
        "lookback_weeks": LOOKBACK_WEEKS,
        "schedule": "semanal",
        "universe_size": len(valid),
        "risk_free_rate": RISK_FREE_RATE,
        "portfolio": {
            "sharpe_ratio": round(port_sharpe, 3),
            "annual_return_pct": round(port_ret * 100, 2),
            "annual_volatility_pct": round(port_vol * 100, 2),
            "weights": {t: round(float(w), 4) for t, w in top_weights.items()},
        },
        "max_sharpe_point": {
            "volatility_pct": round(port_vol * 100, 2),
            "return_pct": round(port_ret * 100, 2),
            "sharpe": round(port_sharpe, 3),
        },
        "frontier": downsample_frontier(frontier, max_points=300),
        "frontier_curve": extract_efficient_curve(frontier),
        "stocks": stocks,
    }
    return result


def main() -> None:
    output = run()
    out_path = Path("results.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Portafolio óptimo (Sharpe {output['portfolio']['sharpe_ratio']}):")
    for s in output["stocks"]:
        print(f"  {s['label']:16} {s['weight']*100:5.1f}%  ${s['price']:>8.2f}  ({s['change_pct']:+.2f}%)")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
