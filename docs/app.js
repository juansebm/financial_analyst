/******************************************
 * app.js — Frontera eficiente + ticker IPSA
 ******************************************/

const fmtPct = (v, digits = 2) =>
  v == null || Number.isNaN(v) ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(digits)}%`;

const fmtNum = (v) =>
  v == null || Number.isNaN(v) ? '—' : Number(v).toLocaleString('es-CL', { maximumFractionDigits: 2 });

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';

function buildTickerItems(stocks) {
  return stocks.flatMap((s) => [
    {
      label: s.label,
      text: `${s.label}  $${fmtNum(s.price)}  ${fmtPct(s.change_pct)}  sem ${fmtPct(s.week_change_pct)}  peso ${(s.weight * 100).toFixed(1)}%`,
      up: s.change_pct >= 0,
    },
  ]);
}

function renderTicker(stocks) {
  const track = document.getElementById('tickerTrack');
  const items = buildTickerItems(stocks);
  const html = items
    .map(
      (item) =>
        `<span class="ticker-item ${item.up ? 'up' : 'down'}">${item.text}</span>`
    )
    .join('<span class="ticker-sep">◆</span>');
  track.innerHTML = html + '<span class="ticker-sep">◆</span>' + html;
}

function renderPortfolioInfo(data) {
  const el = document.getElementById('portfolioInfo');
  const p = data.portfolio;
  const stocksHtml = (data.stocks || [])
    .map(
      (s) => `
      <div class="stock-card">
        <div class="stock-name">${s.label}</div>
        <div class="stock-ticker">${s.ticker}</div>
        <div class="stock-price">$${fmtNum(s.price)}</div>
        <div class="stock-change ${s.change_pct >= 0 ? 'up' : 'down'}">${fmtPct(s.change_pct)}</div>
        <div class="stock-weight">${(s.weight * 100).toFixed(1)}% del portafolio</div>
      </div>`
    )
    .join('');

  el.innerHTML = `
    <h2>Portafolio óptimo (5 acciones)</h2>
    <div class="metrics">
      <div class="metric"><span class="metric-label">Sharpe</span><span class="metric-value">${p.sharpe_ratio}</span></div>
      <div class="metric"><span class="metric-label">Retorno anual</span><span class="metric-value">${p.annual_return_pct}%</span></div>
      <div class="metric"><span class="metric-label">Volatilidad</span><span class="metric-value">${p.annual_volatility_pct}%</span></div>
      <div class="metric"><span class="metric-label">Universo</span><span class="metric-value">${data.universe_size} acciones</span></div>
    </div>
    <p class="note">Selección por máximo ratio de Sharpe sobre ${data.universe_size} acciones del IPSA. Datos semanales, ${data.lookback_weeks || 156} semanas (~3 años). Nube: simulación Monte Carlo.</p>
    <div class="stock-grid">${stocksHtml}</div>
  `;
}

function renderFrontierChart(data) {
  const ctx = document.getElementById('chartFrontier').getContext('2d');
  const cloud = data.frontier || [];
  const curve = (data.frontier_curve || []).slice().sort((a, b) => a.volatility - b.volatility);
  const optimal = data.max_sharpe_point;

  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Portafolios simulados',
          data: cloud.map((p) => ({ x: p.volatility, y: p.return })),
          backgroundColor: 'rgba(88, 166, 255, 0.22)',
          pointRadius: 1.5,
          pointHoverRadius: 3,
          order: 3,
        },
        {
          type: 'line',
          label: 'Frontera eficiente',
          data: curve.map((p) => ({ x: p.volatility, y: p.return, sharpe: p.sharpe })),
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.12)',
          borderWidth: 3,
          pointRadius: 0,
          pointHoverRadius: 5,
          tension: 0.35,
          fill: false,
          showLine: true,
          order: 2,
        },
        {
          label: 'Máximo Sharpe',
          data: [{ x: optimal.volatility_pct, y: optimal.return_pct }],
          backgroundColor: '#ffc107',
          borderColor: '#ffdd57',
          pointRadius: 11,
          pointHoverRadius: 13,
          pointStyle: 'star',
          order: 1,
        },
        {
          label: 'Acciones seleccionadas',
          data: (data.stocks || []).map((s) => ({
            x: s.annual_volatility_pct,
            y: s.annual_return_pct,
            label: s.label,
            weight: s.weight,
          })),
          backgroundColor: '#3fb950',
          borderColor: '#56d364',
          pointRadius: 7,
          pointHoverRadius: 9,
          order: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Frontera eficiente — Riesgo vs Retorno (anualizado)',
          color: '#e6edf3',
          font: { size: 16, weight: '600' },
        },
        legend: {
          position: 'bottom',
          labels: { color: '#8b949e', usePointStyle: true },
        },
        tooltip: {
          callbacks: {
            label(ctx) {
              const p = ctx.raw;
              if (p.label) {
                return `${p.label}: σ ${p.x.toFixed(1)}% · μ ${p.y.toFixed(1)}% · peso ${(p.weight * 100).toFixed(1)}%`;
              }
              if (p.sharpe != null) {
                return `Frontera: σ ${p.x.toFixed(2)}% · μ ${p.y.toFixed(2)}% · Sharpe ${p.sharpe.toFixed(2)}`;
              }
              return `σ ${p.x.toFixed(2)}% · μ ${p.y.toFixed(2)}%`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Volatilidad anual (%)', color: '#8b949e' },
          ticks: { color: '#8b949e' },
          grid: { color: 'rgba(48, 54, 61, 0.8)' },
        },
        y: {
          title: { display: true, text: 'Retorno anual (%)', color: '#8b949e' },
          ticks: { color: '#8b949e' },
          grid: { color: 'rgba(48, 54, 61, 0.8)' },
        },
      },
    },
  });
}

fetch('./results.json')
  .then((r) => r.json())
  .then((data) => {
    const dateEl = document.getElementById('dateContainer');
    if (data.updated_at) {
      const d = new Date(data.updated_at + 'T12:00:00');
      dateEl.textContent = `Actualizado: ${d.toLocaleDateString('es-CL', { dateStyle: 'long' })}`;
    }
    renderTicker(data.stocks || []);
    renderPortfolioInfo(data);
    renderFrontierChart(data);
  })
  .catch((err) => {
    console.error(err);
    document.getElementById('portfolioInfo').innerHTML =
      '<p>No se pudo cargar results.json. Ejecuta <code>python financial_analyst.py</code>.</p>';
  });
