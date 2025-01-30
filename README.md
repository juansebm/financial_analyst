# Financial Analyst - Análisis de Acciones del IPSA

## Descripción
Este proyecto permite descargar, analizar y evaluar el desempeño de las principales acciones del índice IPSA (Chile) utilizando datos de **Yahoo Finance**. Además, incorpora inteligencia artificial basada en **LangChain** y **OpenAI GPT-4o-mini** para generar recomendaciones financieras fundamentadas.

## Características Principales
- Descarga datos históricos de las acciones del IPSA en un solo request.
- Cálculo del **RSI** (*Relative Strength Index*) para cada acción.
- Obtención de métricas financieras clave como **P/E Ratio** y **Debt/Equity**.
- Generación de análisis automatizados y recomendaciones con IA.
- Exportación de los resultados en formato JSON.

## Instalación
### 1. Clonar el repositorio
```sh
    git clone https://github.com/tu_usuario/financial_analyst.git
    cd financial_analyst
```
### 2. Crear y activar entorno virtual
```sh
    python3 -m venv venv
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate     # Windows
```
### 3. Instalar dependencias
```sh
    pip install -r requirements.txt
```

## Configuración
### Variables de entorno
Es necesario configurar la clave de API de **OpenAI**. Crear un archivo `.env` en el directorio principal y añadir:
```env
OPENAI_API_KEY = "..."
```

## Uso
Para ejecutar el *IPSA financial analyst*, simplemente ejecutar:
```sh
python3 financial_analyst.py
```
El script descargará datos, calculará métricas y generará un informe detallado con recomendaciones.

## Salida esperada
El script generará un archivo `results.json` con los datos procesados y el análisis generado por IA. Ejemplo de salida:
```json
{
    "data": [
        {"Stock": "SQM-B.SN", "Último Precio": 50.32, "RSI": 68.5, "P/E Ratio": 10.5, "Debt/Equity": 1.2},
        {"Stock": "CHILE.SN", "Último Precio": 15.48, "RSI": 55.3, "P/E Ratio": 8.7, "Debt/Equity": 0.8}
    ],
    "analysis": "Basado en los datos, se recomienda invertir en..."
}
```

## Dependencias
- `yfinance` - Para obtener datos financieros de Yahoo Finance.
- `pandas` - Para manipulación y análisis de datos.
- `dotenv` - Para manejar variables de entorno.
- `langchain_openai` - Para conectar con GPT-4o-mini o el modelo deseado.
- `langgraph` - Para crear el flujo de análisis con IA.

## Autor
**juansma** - [GitHub](https://github.com/juansebm)

## Licencia
Este proyecto se distribuye bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

