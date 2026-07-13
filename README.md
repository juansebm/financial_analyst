# IPSA — Frontera Eficiente

Visualizador semanal del portafolio de **máximo ratio de Sharpe** sobre acciones del IPSA (Chile).

## Qué hace

1. Descarga precios semanales de ~29 acciones IPSA (Yahoo Finance).
2. Simula portafolios aleatorios y construye la **frontera eficiente** (riesgo vs retorno).
3. Selecciona el portafolio de **máximo Sharpe** y extrae las **5 acciones** con mayor peso.
4. Publica resultados en GitHub Pages con gráfico interactivo y cinta tipo **prompter de bolsa**.

## Instalación

```sh
git clone https://github.com/juansebm/financial_analyst.git
cd financial_analyst
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Uso local

```sh
python financial_analyst.py
cp results.json docs/results.json   # opcional, para previsualizar el sitio
```

Abre `docs/index.html` en el navegador (o sirve `docs/` con un servidor estático).

## GitHub Pages

El workflow `.github/workflows/update_results.yml` corre **cada lunes** y actualiza `results.json` + `docs/results.json`.

Configura Pages en el repo: **Settings → Pages → Source: Deploy from branch → `/docs`**.

## Salida (`results.json`)

```json
{
  "updated_at": "2026-07-12",
  "portfolio": {
    "sharpe_ratio": 1.12,
    "annual_return_pct": 14.5,
    "annual_volatility_pct": 9.3,
    "weights": { "SQM-B.SN": 0.22, "...": 0.18 }
  },
  "frontier": [{ "volatility": 8.1, "return": 6.2, "sharpe": 0.27 }],
  "stocks": [{ "label": "SQM", "price": 42000, "change_pct": 1.2, "weight": 0.22 }]
}
```

## Dependencias

- `yfinance` — datos de mercado
- `pandas` / `numpy` — optimización por simulación Monte Carlo

## Autor

**juansma** — [GitHub](https://github.com/juansebm)

## Licencia

MIT
