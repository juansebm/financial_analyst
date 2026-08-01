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
  const wrap = document.querySelector('.ticker-wrap');
  const track = document.getElementById('tickerTrack');
  const items = buildTickerItems(stocks);
  const html = items
    .map(
      (item) =>
        `<span class="ticker-item ${item.up ? 'up' : 'down'}">${item.text}</span>`
    )
    .join('<span class="ticker-sep">◆</span>');
  track.innerHTML = html + '<span class="ticker-sep">◆</span>' + html;

  // Auto-scroll + arrastre; el contenido está duplicado → loop en halfWidth
  let offset = 0;
  let half = 0;
  let paused = false;
  let dragging = false;
  let lastX = 0;
  let lastTs = 0;
  const pxPerSec = 48;

  const apply = () => {
    if (half > 0) {
      offset = ((offset % half) + half) % half;
      track.style.transform = `translateX(${-offset}px)`;
    }
  };

  const measure = () => {
    half = track.scrollWidth / 2;
    apply();
  };

  const tick = (ts) => {
    if (!lastTs) lastTs = ts;
    const dt = (ts - lastTs) / 1000;
    lastTs = ts;
    if (!paused && !dragging && half > 0) {
      offset += pxPerSec * dt;
      apply();
    }
    requestAnimationFrame(tick);
  };

  wrap.addEventListener('pointerdown', (e) => {
    dragging = true;
    paused = true;
    lastX = e.clientX;
    wrap.classList.add('is-dragging');
    wrap.setPointerCapture(e.pointerId);
  });
  wrap.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    offset -= e.clientX - lastX;
    lastX = e.clientX;
    apply();
  });
  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    wrap.classList.remove('is-dragging');
    if (e && wrap.hasPointerCapture?.(e.pointerId)) {
      wrap.releasePointerCapture(e.pointerId);
    }
    // reanuda si el mouse no sigue encima
    paused = wrap.matches(':hover');
  };
  wrap.addEventListener('pointerup', endDrag);
  wrap.addEventListener('pointercancel', endDrag);
  wrap.addEventListener('pointerleave', () => {
    if (!dragging) paused = false;
  });
  wrap.addEventListener('pointerenter', () => {
    if (!dragging) paused = true;
  });

  requestAnimationFrame(() => {
    measure();
    requestAnimationFrame(tick);
  });
  window.addEventListener('resize', measure);
}

function fmtMoney(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toLocaleString('es-CL', { maximumFractionDigits: 0 });
}

function parseAmount(raw) {
  const n = Number(String(raw).replace(/\./g, '').replace(',', '.'));
  return Number.isFinite(n) && n > 0 ? n : 0;
}

function updateAllocations(stocks, total) {
  stocks.forEach((s, i) => {
    const amountEl = document.getElementById(`alloc-amount-${i}`);
    const sharesEl = document.getElementById(`alloc-shares-${i}`);
    if (!amountEl) return;
    if (!total) {
      amountEl.textContent = '—';
      if (sharesEl) sharesEl.textContent = '';
      return;
    }
    const amount = total * s.weight;
    amountEl.textContent = `$${fmtMoney(amount)}`;
    if (sharesEl && s.price > 0) {
      sharesEl.textContent = `≈ ${(amount / s.price).toFixed(1)} acciones`;
    }
  });
}

function renderWeekComparison(data) {
  const cmp = data.comparison;
  const prev = data.previous_week;
  if (!cmp || !prev) return '';

  const prevList = (prev.labels || [])
    .map((label, i) => {
      const ticker = prev.tickers[i];
      const w = ((prev.weights[ticker] || 0) * 100).toFixed(1);
      const exited = (cmp.exited || []).some((e) => e.ticker === ticker);
      return `<li class="${exited ? 'exited' : 'kept'}">${label} <span>${w}%</span>${exited ? ' · salió' : ''}</li>`;
    })
    .join('');

  const summary = cmp.unchanged
    ? '<p class="compare-summary same">Sin cambios: el top 5 es el mismo que la semana pasada.</p>'
    : `<p class="compare-summary changed">
        ${cmp.entered.length ? `Entraron: <strong>${cmp.entered.map((e) => e.label).join(', ')}</strong>. ` : ''}
        ${cmp.exited.length ? `Salieron: <strong>${cmp.exited.map((e) => e.label).join(', ')}</strong>.` : ''}
      </p>`;

  return `
    <div class="week-compare">
      <h3>vs semana pasada <span class="week-asof">(${prev.as_of})</span></h3>
      ${summary}
      <ol class="prev-list">${prevList}</ol>
    </div>
  `;
}

function renderPortfolioInfo(data) {
  const el = document.getElementById('portfolioInfo');
  const p = data.portfolio;
  const stocks = data.stocks || [];
  const stocksHtml = stocks
    .map(
      (s, i) => `
      <div class="stock-card ${s.vs_last_week === 'new' ? 'is-new' : 'is-kept'}">
        <div class="stock-badge">${s.vs_last_week === 'new' ? 'NUEVA' : 'IGUAL'}</div>
        <div class="stock-name">${s.label}</div>
        <div class="stock-ticker">${s.ticker}</div>
        <div class="stock-price">$${fmtNum(s.price)}</div>
        <div class="stock-change ${s.change_pct >= 0 ? 'up' : 'down'}">${fmtPct(s.change_pct)}</div>
        <div class="stock-weight">${(s.weight * 100).toFixed(1)}% del portafolio</div>
        <div class="stock-alloc" id="alloc-amount-${i}">—</div>
        <div class="stock-shares" id="alloc-shares-${i}"></div>
        <div class="stock-validity">
          <span class="validity-label">Vigencia orden</span>
          <span class="validity-until">hasta ${s.order_valid_label || '—'}</span>
          <span class="validity-reason">${s.order_valid_reason || ''}</span>
        </div>
      </div>`
    )
    .join('');

  const rfPct = ((data.risk_free_rate ?? 0.04) * 100).toFixed(0);
  el.innerHTML = `
    <h2>Portafolio óptimo (5 acciones)</h2>
    ${renderWeekComparison(data)}
    <div class="metrics">
      <div class="metric metric-sharpe">
        <span class="metric-label">Sharpe</span>
        <span class="metric-value">${p.sharpe_ratio}</span>
        <span class="metric-formula">(μ − r<sub>f</sub>) / σ = (${p.annual_return_pct}% − ${rfPct}%) / ${p.annual_volatility_pct}%</span>
      </div>
      <div class="metric"><span class="metric-label">Retorno anual</span><span class="metric-value">${p.annual_return_pct}%</span></div>
      <div class="metric"><span class="metric-label">Volatilidad</span><span class="metric-value">${p.annual_volatility_pct}%</span></div>
    </div>
    <div class="allocator">
      <label for="totalInput">Monto a invertir (CLP)</label>
      <div class="allocator-row">
        <span class="allocator-prefix">$</span>
        <input id="totalInput" type="text" inputmode="numeric" placeholder="1.000.000" autocomplete="off" />
      </div>
      <p class="allocator-hint">Se reparte según los pesos del portafolio de máximo Sharpe.</p>
    </div>
    <div class="stock-grid">${stocksHtml}</div>
  `;

  const input = document.getElementById('totalInput');
  const refresh = () => updateAllocations(stocks, parseAmount(input.value));
  input.addEventListener('input', refresh);
  refresh();
}

function renderFrontierChart(data) {
  const ctx = document.getElementById('chartFrontier').getContext('2d');
  const cloud = data.frontier || [];
  const curve = (data.frontier_curve || []).slice().sort((a, b) => a.volatility - b.volatility);
  const optimal = data.max_sharpe_point;

  const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Portafolios simulados',
          data: cloud.map((p) => ({ x: p.volatility, y: p.return })),
          backgroundColor: 'rgba(139, 148, 158, 0.28)',
          pointRadius: 1.4,
          pointHoverRadius: 3,
          pointHitRadius: 1,
          order: 3,
        },
        {
          type: 'line',
          label: 'Frontera eficiente',
          data: curve.map((p) => ({ x: p.volatility, y: p.return, sharpe: p.sharpe })),
          borderColor: '#c9d1d9',
          backgroundColor: 'rgba(201, 209, 217, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: false,
          showLine: true,
          order: 2,
        },
        {
          label: 'Máximo Sharpe',
          data: [{ x: optimal.volatility_pct, y: optimal.return_pct }],
          backgroundColor: '#e6edf3',
          borderColor: '#f0f3f6',
          pointRadius: 8,
          pointHoverRadius: 10,
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
          backgroundColor: 'transparent',
          borderColor: '#e6edf3',
          borderWidth: 1.25,
          pointRadius: 5,
          pointHoverRadius: 7,
          pointHitRadius: 8,
          pointStyle: 'crossRot',
          order: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      animations: { colors: false, x: false, y: false },
      interaction: { mode: 'nearest', intersect: true },
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
        zoom: {
          limits: {
            x: { min: 'original', max: 'original' },
            y: { min: 'original', max: 'original' },
          },
          pan: {
            enabled: true,
            mode: 'xy',
            onPan({ chart: c }) {
              c.draw();
            },
          },
          zoom: {
            wheel: { enabled: true },
            pinch: { enabled: true },
            drag: { enabled: false },
            mode: 'xy',
            onZoom({ chart: c }) {
              c.draw();
            },
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Volatilidad anual (%)', color: '#8b949e' },
          ticks: { color: '#8b949e' },
          grid: { color: 'rgba(48, 54, 61, 0.8)' },
        },
        y: {
          type: 'linear',
          title: { display: true, text: 'Retorno anual (%)', color: '#8b949e' },
          ticks: { color: '#8b949e' },
          grid: { color: 'rgba(48, 54, 61, 0.8)' },
        },
      },
    },
  });

  const resetBtn = document.getElementById('chartResetZoom');
  if (resetBtn) {
    resetBtn.onclick = () => chart.resetZoom();
  }
}

fetch('./results.json')
  .then((r) => r.json())
  .then((data) => {
    const dateEl = document.getElementById('dateContainer');
    const dateLine = data.updated_at
      ? `Actualizado: ${new Date(data.updated_at + 'T12:00:00').toLocaleDateString('es-CL', { dateStyle: 'long' })}`
      : '';
    const universeLine = `Universo: ${data.universe_size} acciones IPSA · ${data.lookback_weeks || 156} semanas.`;
    dateEl.innerHTML = `${dateLine ? `<span>${dateLine}</span>` : ''}<span class="header-universe">${universeLine}</span>`;
    renderTicker(data.stocks || []);
    renderPortfolioInfo(data);
    renderFrontierChart(data);
  })
  .catch((err) => {
    console.error(err);
    document.getElementById('portfolioInfo').innerHTML =
      '<p>No se pudo cargar results.json. Ejecuta <code>python financial_analyst.py</code>.</p>';
  });
