"""Portafolio IPSA: frontera eficiente (máximo Sharpe) y ticker semanal."""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------
# Parámetros (literatura mean-variance)
# ----------------------------------------
# Ventana: ~3 años semanales. Con N≈30, T≫N reduce error de Σ
# (Jobson–Korkie; práctica semanal ≈ 60–120 obs. mensuales clásicas).
LOOKBACK_WEEKS = 156
PORTFOLIO_SIZE = 5
# Nube: densidad visual ∝ muestras MC, no ∝ semanas.
MONTE_CARLO_SAMPLES = 60_000
CLOUD_DISPLAY_POINTS = 2_500
CURVE_BINS = 60
RISK_FREE_RATE = 0.04  # tasa anual aprox. (CLP)
TRADING_WEEKS = 52
SHARPE_SEARCH_SAMPLES = 80_000

IPSA_STOCKS = [
    "SQM-B.SN", "CHILE.SN", "BSANTANDER.SN", "COPEC.SN", "ENELAM.SN", "CENCOSUD.SN",
    "CMPC.SN", "BCI.SN", "FALABELLA.SN", "ENELCHILE.SN", "PARAUCO.SN", "COLBUN.SN",
    "CCU.SN", "ANDINA-B.SN", "VAPORES.SN", "AGUAS-A.SN", "QUINENCO.SN", "CENCOMALLS.SN",
    "LTM.SN", "CONCHATORO.SN", "ENTEL.SN", "CAP.SN", "MALLPLAZA.SN",
    "ECL.SN", "IAM.SN", "SMU.SN", "ITAUCL.SN", "SONDA.SN", "RIPLEY.SN",
    # Ampliación IPSA / líquidas BCS
    "ILC.SN", "SALFACORP.SN", "SMSAAM.SN", "FORUS.SN",
    "SK.SN", "ENAEX.SN", "BESALCO.SN", "HABITAT.SN",
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
    "ECL.SN": "Engie Chile",
    "IAM.SN": "Inversiones Aguas",
    "SMU.SN": "SMU",
    "ITAUCL.SN": "Itaú",
    "SONDA.SN": "Sonda",
    "RIPLEY.SN": "Ripley",
    "ILC.SN": "ILC",
    "SALFACORP.SN": "SalfaCorp",
    "SMSAAM.SN": "SAAM",
    "FORUS.SN": "Forus",
    "SK.SN": "Sigdo Koppers",
    "ENAEX.SN": "Enaex",
    "BESALCO.SN": "Besalco",
    "HABITAT.SN": "Habitat",
}

WEEKDAYS_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MONTHS_ES = (
    "", "ene", "feb", "mar", "abr", "may", "jun",
    "jul", "ago", "sep", "oct", "nov", "dic",
)


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


def _batch_stats(weights: np.ndarray, mean_returns: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rets = weights @ mean_returns
    # vol_i = sqrt(w_i' Σ w_i)
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    return rets, vols


def _sample_weights(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    """Mitad Dirichlet pleno + mitad sparse 1–5 activos + vértices single-stock."""
    n_div = n // 2
    n_sparse = n - n_div
    W = np.zeros((n + k, k))
    W[:n_div] = rng.dirichlet(np.ones(k), size=n_div)
    for i in range(n_sparse):
        n_active = int(rng.integers(1, 6))
        idx = rng.choice(k, size=n_active, replace=False)
        W[n_div + i, idx] = rng.dirichlet(np.ones(n_active))
    W[n:] = np.eye(k)  # 100% cada acción
    return W


def simulate_frontier(mean_returns: pd.Series, cov: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    mu = mean_returns.values
    sigma = cov.values
    k = len(mu)
    W = _sample_weights(rng, k, n)
    rets, vols = _batch_stats(W, mu, sigma)
    sharpes = np.where(vols > 0, (rets - RISK_FREE_RATE) / vols, -np.inf)
    return pd.DataFrame({"volatility": vols, "return": rets, "sharpe": sharpes})


def max_sharpe_portfolio(mean_returns: pd.Series, cov: pd.DataFrame) -> tuple[pd.Series, float, float, float]:
    # ponytail: Monte Carlo denso en lugar de optimizador cuadrático
    rng = np.random.default_rng(7)
    tickers = mean_returns.index.tolist()
    mu = mean_returns.values
    sigma = cov.values
    k = len(mu)
    W = _sample_weights(rng, k, SHARPE_SEARCH_SAMPLES)
    rets, vols = _batch_stats(W, mu, sigma)
    sharpes = np.where(vols > 0, (rets - RISK_FREE_RATE) / vols, -np.inf)
    best = int(np.argmax(sharpes))
    w = W[best]
    return pd.Series(w, index=tickers), float(rets[best]), float(vols[best]), float(sharpes[best])


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


def business_days_within(as_of: dt.datetime, calendar_days: int = 7) -> list[dt.date]:
    """Días hábiles (lun–vie) estrictamente dentro de los próximos `calendar_days`."""
    start = as_of.date() + dt.timedelta(days=1)
    end = as_of.date() + dt.timedelta(days=calendar_days)
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def format_day_es(d: dt.date) -> str:
    return f"{WEEKDAYS_ES[d.weekday()]} {d.day} {MONTHS_ES[d.month]}"


def attach_order_validity(stocks: list[dict], as_of: dt.datetime) -> list[dict]:
    """
    Vigencia de orden en los próximos 7 días: más volátil → vence antes
    (evita órdenes stale cuando el precio se mueve rápido).
    """
    days = business_days_within(as_of, 7)
    if not days:
        for s in stocks:
            s["order_valid_until"] = None
            s["order_valid_label"] = "sin día hábil"
            s["order_valid_reason"] = "sin ventana"
        return stocks

    ranked = sorted(range(len(stocks)), key=lambda i: stocks[i]["annual_volatility_pct"], reverse=True)
    n = len(ranked)
    last = len(days) - 1
    for rank, idx in enumerate(ranked):
        day_i = int(round(rank * last / max(n - 1, 1)))
        day = days[day_i]
        # terciles de volatilidad relativa al portafolio
        if rank < max(n // 3, 1):
            reason = "alta volatilidad: vigencia corta"
        elif rank >= n - max(n // 3, 1):
            reason = "baja volatilidad: puede esperar"
        else:
            reason = "volatilidad media"
        stocks[idx]["order_valid_until"] = day.isoformat()
        stocks[idx]["order_valid_label"] = format_day_es(day)
        stocks[idx]["order_valid_reason"] = reason
    return stocks


def _to_pct_points(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "volatility": round(float(row["volatility"]) * 100, 3),
            "return": round(float(row["return"]) * 100, 3),
            "sharpe": round(float(row["sharpe"]), 3),
        }
        for _, row in df.iterrows()
    ]


def downsample_frontier(df: pd.DataFrame, max_points: int = CLOUD_DISPLAY_POINTS) -> list[dict]:
    if len(df) <= max_points:
        sample = df
    else:
        sample = df.sample(max_points, random_state=1)
    return _to_pct_points(sample)


def extract_efficient_curve(df: pd.DataFrame, n_bins: int = CURVE_BINS) -> list[dict]:
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


def optimize_window(prices: pd.DataFrame) -> tuple[pd.Series, float, float, float, pd.Series, pd.DataFrame]:
    """Retorna (top_weights, ret, vol, sharpe, mean_a, cov_a) sobre un slice de precios."""
    returns = prices.pct_change().dropna()
    valid = [c for c in returns.columns if returns[c].count() >= LOOKBACK_WEEKS // 2]
    returns = returns[valid]
    if len(valid) < PORTFOLIO_SIZE:
        raise RuntimeError(f"Solo {len(valid)} acciones con datos suficientes.")
    mean_a, cov_a = annualize(returns.mean(), returns.cov())
    weights, port_ret, port_vol, port_sharpe = max_sharpe_portfolio(mean_a, cov_a)
    top = pick_top_stocks(weights)
    return top, port_ret, port_vol, port_sharpe, mean_a, cov_a


def week_portfolio_summary(top_weights: pd.Series, as_of: str) -> dict:
    return {
        "as_of": as_of,
        "tickers": list(top_weights.index),
        "weights": {t: round(float(w), 4) for t, w in top_weights.items()},
        "labels": [TICKER_LABELS.get(t, t.replace(".SN", "")) for t in top_weights.index],
    }


def compare_top5(current: pd.Series, previous: pd.Series) -> dict:
    curr = set(current.index)
    prev = set(previous.index)
    kept = sorted(curr & prev)
    entered = sorted(curr - prev)
    exited = sorted(prev - curr)
    return {
        "unchanged": curr == prev,
        "kept": [
            {"ticker": t, "label": TICKER_LABELS.get(t, t.replace(".SN", "")),
             "weight_now": round(float(current[t]), 4),
             "weight_prev": round(float(previous[t]), 4)}
            for t in kept
        ],
        "entered": [
            {"ticker": t, "label": TICKER_LABELS.get(t, t.replace(".SN", "")),
             "weight": round(float(current[t]), 4)}
            for t in entered
        ],
        "exited": [
            {"ticker": t, "label": TICKER_LABELS.get(t, t.replace(".SN", "")),
             "weight": round(float(previous[t]), 4)}
            for t in exited
        ],
    }


def run() -> dict:
    end = dt.datetime.now()
    end_prev = end - dt.timedelta(weeks=1)
    # +1 semana para poder cortar la ventana "semana pasada" con el mismo lookback
    start = end - dt.timedelta(weeks=LOOKBACK_WEEKS + 1)
    print(f"Descargando {len(IPSA_STOCKS)} acciones IPSA ({LOOKBACK_WEEKS} semanas + 1)...")
    prices = download_prices(IPSA_STOCKS, start, end)

    # Semana actual
    prices_now = prices.loc[: end.strftime("%Y-%m-%d")].tail(LOOKBACK_WEEKS + 1)
    top_now, port_ret, port_vol, port_sharpe, mean_a, cov_a = optimize_window(prices_now)

    # Semana pasada: mismos datos truncados 1 semana atrás
    prices_prev = prices.loc[: end_prev.strftime("%Y-%m-%d")].tail(LOOKBACK_WEEKS + 1)
    top_prev, *_rest = optimize_window(prices_prev)

    stocks = [
        stock_snapshot(ticker, w, prices_now[ticker], mean_a, cov_a)
        for ticker, w in top_now.items()
    ]
    stocks = attach_order_validity(stocks, end)
    comparison = compare_top5(top_now, top_prev)
    status_by_ticker = {e["ticker"]: "new" for e in comparison["entered"]}
    for k in comparison["kept"]:
        status_by_ticker[k["ticker"]] = "kept"
    for s in stocks:
        s["vs_last_week"] = status_by_ticker.get(s["ticker"], "kept")

    frontier = simulate_frontier(mean_a, cov_a, MONTE_CARLO_SAMPLES)

    result = {
        "updated_at": end.strftime("%Y-%m-%d"),
        "lookback_weeks": LOOKBACK_WEEKS,
        "schedule": "semanal",
        "universe_size": len(mean_a),
        "risk_free_rate": RISK_FREE_RATE,
        "order_validity_window_days": 7,
        "order_validity_note": (
            "Vigencia sugerida de cada orden dentro de los próximos 7 días: "
            "mayor volatilidad = vence antes."
        ),
        "portfolio": {
            "sharpe_ratio": round(port_sharpe, 3),
            "annual_return_pct": round(port_ret * 100, 2),
            "annual_volatility_pct": round(port_vol * 100, 2),
            "weights": {t: round(float(w), 4) for t, w in top_now.items()},
        },
        "max_sharpe_point": {
            "volatility_pct": round(port_vol * 100, 2),
            "return_pct": round(port_ret * 100, 2),
            "sharpe": round(port_sharpe, 3),
        },
        "frontier": downsample_frontier(frontier, max_points=CLOUD_DISPLAY_POINTS),
        "frontier_curve": extract_efficient_curve(frontier, n_bins=CURVE_BINS),
        "stocks": stocks,
        "previous_week": week_portfolio_summary(top_prev, end_prev.strftime("%Y-%m-%d")),
        "comparison": comparison,
    }
    return result


def main() -> None:
    output = run()
    out_path = Path("results.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Portafolio óptimo (Sharpe {output['portfolio']['sharpe_ratio']}):")
    for s in output["stocks"]:
        tag = "NUEVA" if s.get("vs_last_week") == "new" else "igual"
        print(
            f"  [{tag:5}] {s['label']:16} {s['weight']*100:5.1f}%  ${s['price']:>8.2f}  "
            f"vigencia hasta {s.get('order_valid_label', '-')}"
        )
    cmp_ = output["comparison"]
    if cmp_["unchanged"]:
        print("\nSin cambios vs semana pasada.")
    else:
        if cmp_["entered"]:
            print("Entraron:", ", ".join(e["label"] for e in cmp_["entered"]))
        if cmp_["exited"]:
            print("Salieron:", ", ".join(e["label"] for e in cmp_["exited"]))
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
