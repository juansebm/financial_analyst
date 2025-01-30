/******************************************
 * app.js
 * Código completo sin duplicaciones
 ******************************************/

// 1. Función para formatear texto de "analysis" (opcional)
function formatAnalysis(text) {
  // Div contenedor
  const container = document.createElement('div');
  // Separar líneas
  const lines = text.split('\n');
  let currentList = null;

  lines.forEach(line => {
    const trimmed = line.trim();
    // Ignorar líneas vacías
    if (!trimmed) {
      if (currentList) {
        container.appendChild(currentList);
        currentList = null;
      }
      return;
    }
    // Títulos con "### "
    if (trimmed.startsWith('###')) {
      if (currentList) {
        container.appendChild(currentList);
        currentList = null;
      }
      const h3 = document.createElement('h3');
      h3.textContent = trimmed.replace(/^###\s*/, '');
      container.appendChild(h3);
    }
    // Listas con "- "
    else if (trimmed.startsWith('- ')) {
      if (!currentList) {
        currentList = document.createElement('ul');
      }
      const li = document.createElement('li');
      li.textContent = trimmed.replace(/^- /, '');
      currentList.appendChild(li);
    }
    else {
      if (currentList) {
        container.appendChild(currentList);
        currentList = null;
      }
      const p = document.createElement('p');
      p.textContent = trimmed;
      container.appendChild(p);
    }
  });

  if (currentList) {
    container.appendChild(currentList);
  }
  return container.innerHTML;
}

// 2. Hacemos fetch al JSON
fetch('./results.json')
  .then(response => response.json())
  .then(data => {
    // --- A) Mostrar "analysis" como Markdown con Marked ---
    const analysisContainer = document.getElementById('analysisContainer');
    let rawHTML = marked.parse(data.analysis || '');
    // Elimina numeraciones tipo "1) " en títulos, si no quieres verlas
    rawHTML = rawHTML.replace(/<h[1-6]>(\d+\)\s+)/g, match => match.replace(/\d+\)\s+/, ''));
    analysisContainer.innerHTML = rawHTML;

    // --- B) Mostrar fecha actual en el header (opcional) ---
    const dateEl = document.getElementById('dateContainer');
    const now = new Date();
    dateEl.textContent = now.toLocaleDateString('es-CL', { dateStyle: 'long' });

    // --- C) Preparar datos para los gráficos ---
    const rows = data.data || [];
    const labels = rows.map(item => item.Stock);

    // === GRÁFICO 1: RSI (barras verticales) ===
    const ctxRSI = document.getElementById('chartRSI').getContext('2d');
    // Colores según RSI
    const barColors = rows.map(item => {
      if (item.RSI >= 70) {
        return 'rgba(255, 0, 0, 0.7)';   // sobrecompra
      } else if (item.RSI <= 30) {
        return 'rgba(0, 128, 0, 0.7)';  // sobreventa
      }
      return 'rgba(0, 44, 84, 0.7)';     // neutro
    });

    new Chart(ctxRSI, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'RSI',
          data: rows.map(item => item.RSI),
          backgroundColor: barColors,
          borderColor: '#002C54',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'RSI por Acción'
          },
          tooltip: {
            enabled: true
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100
          }
        }
      }
    });

    // === GRÁFICO 2: Índices Técnicos (barras horizontales) ===
    // Necesitamos el plugin "chartjs-plugin-annotation" incluido en el HTML
    const ctxIndices = document.getElementById('chartIndices').getContext('2d');

    // Extraemos datos (VI+ / VI- / CCI)
    const viPlusData = rows.map(item => item.VI_plus || 0);
    const viMinusData = rows.map(item => item.VI_minus || 0);
    const cciData = rows.map(item => item.CCI || 0);

    new Chart(ctxIndices, {
      type: 'bar',
      data: {
        labels: labels, // Eje Y: acciones
        datasets: [
          {
            label: 'VI+',
            data: viPlusData,
            backgroundColor: 'rgba(255, 99, 132, 0.6)',
            borderColor: 'rgb(255, 99, 132)',
            borderWidth: 1,
            xAxisID: 'xVI'   // ¡Irán al eje X "xVI"!
          },
          {
            label: 'VI-',
            data: viMinusData,
            backgroundColor: 'rgba(54, 162, 235, 0.6)',
            borderColor: 'rgb(54, 162, 235)',
            borderWidth: 1,
            xAxisID: 'xVI'   // También en el eje "xVI"
          },
          {
            label: 'CCI',
            data: cciData,
            backgroundColor: 'rgba(0, 200, 0, 0.6)',
            borderColor: 'rgb(0, 200, 0)',
            borderWidth: 1,
            xAxisID: 'xCCI'  // ¡CCI va al segundo eje "xCCI"!
          }
        ]
      },
      options: {
        indexAxis: 'y',  // barras horizontales
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Índices Técnicos (VI+/VI-/CCI)'
          },
          tooltip: {
            enabled: true
          },
          annotation: {
            annotations: {
              cciUpper: {
                type: 'line',
                // x=+100 en el eje "xCCI"
                scaleID: 'xCCI',
                value: 100,
                borderColor: 'red',
                borderWidth: 1,
                borderDash: [6, 6],
                label: {
                  enabled: true,
                  content: 'CCI = +100',
                  position: 'start',
                  backgroundColor: 'rgba(255, 0, 0, 0.2)'
                }
              },
              cciLower: {
                type: 'line',
                // x=-100 en el eje "xCCI"
                scaleID: 'xCCI',
                value: -100,
                borderColor: 'blue',
                borderWidth: 1,
                borderDash: [6, 6],
                label: {
                  enabled: true,
                  content: 'CCI = -100',
                  position: 'start',
                  backgroundColor: 'rgba(0, 0, 255, 0.2)'
                }
              }
            }
          }
        },
        scales: {
          xVI: {
            type: 'linear',
            position: 'bottom',  // O 'top'
            min: 0,             // Ajusta según valores típicos de VI
            max: 2,
            title: {
              display: true,
              text: 'VI Scale'
            }
          },
          // Eje X para CCI
          xCCI: {
            type: 'linear',
            position: 'top',    // O 'bottom'
            min: -200,          // Rango para CCI
            max: 300,
            title: {
              display: true,
              text: 'CCI Scale'
            }
          },
          y: {
            // Muestra "labels" en eje Y
          }
        }
      }
    });
  })
  .catch(error => {
    console.error('Error fetching JSON:', error);
  });
