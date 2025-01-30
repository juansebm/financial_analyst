/**
 * Formatea el texto de "analysis" para que no se vea como un bloque gigantesco.
 * - Convierte "###" al inicio de línea en <h3>
 * - Convierte líneas con "- " en <li> dentro de <ul> (opcional, ejemplo simple)
 * - O crea <p> por defecto.
 */
function formatAnalysis(text) {
    // Div contenedor para ir armando trozos de HTML
    const container = document.createElement('div');
  
    // Separar el contenido por saltos de línea
    const lines = text.split('\n');
  
    let currentList = null; // para manejar <ul>
  
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
  
      // Si detectamos un título "### "
      if (trimmed.startsWith('###')) {
        // Si veníamos construyendo una lista, la cerramos
        if (currentList) {
          container.appendChild(currentList);
          currentList = null;
        }
        // Creamos un <h3>
        const h3 = document.createElement('h3');
        // Reemplazamos "### " por nada, dejando solo el texto
        h3.textContent = trimmed.replace(/^###\s*/, '');
        container.appendChild(h3);
      }
      // Si la línea empieza con "- " la interpretamos como un ítem de lista
      else if (trimmed.startsWith('- ')) {
        if (!currentList) {
          // Si no hay una lista abierta, creamos una
          currentList = document.createElement('ul');
        }
        const li = document.createElement('li');
        li.textContent = trimmed.replace(/^- /, '');
        currentList.appendChild(li);
      }
      else {
        // Si estábamos construyendo una lista y el texto ya no es un item, la cerramos
        if (currentList) {
          container.appendChild(currentList);
          currentList = null;
        }
        // Creamos un párrafo
        const p = document.createElement('p');
        p.textContent = trimmed;
        container.appendChild(p);
      }
    });
  
    // Si al final queda una lista abierta, la cerramos
    if (currentList) {
      container.appendChild(currentList);
    }
  
    return container.innerHTML;
  }
  
  // Hacemos fetch al JSON
  fetch('./results.json')
  .then(response => response.json())
  .then(data => {
    // --------------------------------------
    // 1) Procesar y formatear el "analysis"
    // --------------------------------------
    // Conviertes la parte de "analysis" (en Markdown) a HTML usando marked:
    let rawHTML = marked.parse(data.analysis || '');

    // Opcional: Remover títulos con numeración "1)", "2)" al comienzo.
    // Este regex busca algo como <h2>1) Título</h2> y quita "1) "
    rawHTML = rawHTML.replace(/<h[1-6]>(\d+\)\s+)/g, match => {
      // match = "<hX>1) "
      // Eliminas solo la parte "1) "
      return match.replace(/\d+\)\s+/, '');
    });

    // Insertas el HTML resultante en la sección de análisis
    const analysisContainer = document.getElementById('analysisContainer');
    analysisContainer.innerHTML = rawHTML;

    // ----------------------------------------------------
    // 2) Generar el gráfico (Chart.js) con datos de RSI
    // ----------------------------------------------------
    const rows = data.data || [];

    // Eje X: nombres de acciones
    const labels = rows.map(item => item.Stock);
    // Eje Y: RSI
    const rsiData = rows.map(item => item.RSI);

    const ctx = document.getElementById('ipsaChart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'RSI',
          data: rsiData,
          backgroundColor: '#002C54',  // color base
          borderColor: '#001D3A',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false, // El canvas se adaptará al contenedor
        plugins: {
          title: {
            display: true,
            text: 'RSI de Acciones IPSA'
          },
          legend: {
            display: true
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100 // RSI va típicamente de 0 a 100
          }
        }
      }
    });
  })
  .catch(error => {
    console.error('Error fetching JSON:', error);
  });
