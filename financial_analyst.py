from typing import Union, Dict, List, TypedDict, Annotated
import yfinance as yf
print(yf.__version__)
import datetime as dt
import pandas as pd
import dotenv
import os
import json
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------------
# 1. Parámetros y lista de acciones IPSA
# ----------------------------------------
start_date = dt.datetime.now() - dt.timedelta(weeks=12)  # 3 meses atrás
end_date = dt.datetime.now()

ipsa_stocks = [
    "SQM-B.SN", "CHILE.SN", "BSANTANDER.SN", "COPEC.SN", "ENELAM.SN", "CENCOSUD.SN",
    "CMPC.SN", "BCI.SN", "FALABELLA.SN", "ENELCHILE.SN", "PARAUCO.SN", "COLBUN.SN",
    "CCU.SN", "ANDINA-B.SN", "VAPORES.SN", "AGUAS-A.SN", "QUINENCO.SN", "CENCOMALLS.SN",
    "LTM.SN", "CONCHATORO.SN", "ENTEL.SN", "CAP.SN", "MALLPLAZA.SN",
    "ECL.SN", "IAM.SN", "SMU.SN", "ITAUCL.SN", "SONDA.SN", "RIPLEY.SN"
]

# ----------------------------------------
# 2. Descarga de datos
# ----------------------------------------
print("Descargando datos de Yahoo Finance para todas las acciones del IPSA...")
all_data = yf.download(
    tickers=ipsa_stocks,
    start=start_date,
    end=end_date,
    interval='1d',
    group_by='ticker',  # Devuelve un dict { TICKER: DF }
    auto_adjust=False,
    threads=False  # evita multi-threading para mayor estabilidad
)
print("\nDescarga completa. Comenzando análisis...\n")

# ----------------------------------------
# 3. Funciones auxiliares
# ----------------------------------------
def calculate_rsi(series, period=14):
    """Calcula el RSI de una serie de precios de cierre."""
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Obtiene métricas financieras clave para un ticker dado."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe_ratio': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margins': info.get('profitMargins')
        }
    except Exception as e:
        return f"Error fetching ratios: {str(e)}"

def safe_get_financial_metrics(ticker):
    """Obtiene métricas financieras de forma segura (manejo de string-errores)."""
    data = get_financial_metrics(ticker)
    if isinstance(data, str):
        # Hubo error
        return {
            'pe_ratio': None,
            'debt_to_equity': None
        }
    return data

# ----------------------------------------
# 4. Bucle de análisis
# ----------------------------------------
results = []
for ticker in ipsa_stocks:
    print(f"--- Análisis para {ticker} ---")

    # 4.1) Verificamos si se descargó algo para este ticker
    #     all_data[ticker] será un DF con columnas: [Open, High, Low, Close, Adj Close, Volume]
    #     o puede no existir si no se devolvieron datos
    if ticker not in all_data:
        print(f"   ❌ No hay datos descargados para {ticker} (posible error con el sufijo .SN)")
        results.append({
            "Stock": ticker,
            "Último Precio": None,
            "RSI": None,
            "P/E Ratio": None,
            "Debt/Equity": None
        })
        continue

    df = all_data[ticker].copy()

    # 4.2) Validamos que tenga filas
    if df.empty:
        print(f"   ⚠️ DataFrame vacío para {ticker}.")
        results.append({
            "Stock": ticker,
            "Último Precio": None,
            "RSI": None,
            "P/E Ratio": None,
            "Debt/Equity": None
        })
        continue

    # 4.3) Asegurar que las columnas tengan el formato [Open, High, Low, Close, Volume, etc.]
    if "Close" not in df.columns:
        print(f"   ⚠️ La columna 'Close' no está presente en {ticker}.")
        results.append({
            "Stock": ticker,
            "Último Precio": None,
            "RSI": None,
            "P/E Ratio": None,
            "Debt/Equity": None
        })
        continue

    # 4.4) Calculamos RSI
    #     - Si no hay suficientes filas, RSI puede salir mayoritariamente NaN
    df["RSI"] = calculate_rsi(df["Close"])

    # 4.5) Tomamos el último precio y el último RSI
    last_close = df["Close"].iloc[-1]
    last_rsi = df["RSI"].iloc[-1] if not df["RSI"].isna().all() else None

    # 4.6) Buscamos métricas financieras
    fm = safe_get_financial_metrics(ticker)  # P/E, Debt/Equity, etc.
    pe_ratio = fm.get('pe_ratio')
    debt_to_equity = fm.get('debt_to_equity')

    # 4.7) Armamos registro
    stock_data = {
        "Stock": ticker,
        "Último Precio": last_close,
        "RSI": round(last_rsi, 2) if last_rsi is not None else None,
        "P/E Ratio": pe_ratio,
        "Debt/Equity": debt_to_equity
    }

    # 4.8) Calculamos porcentaje de NaN
    nan_count = sum(v is None for v in stock_data.values())
    nan_percentage = (nan_count / len(stock_data)) * 100

    # 4.9) Mostramos en consola un resumen inmediato
    print(f"   Precio: {stock_data['Último Precio']}, RSI: {stock_data['RSI']}, "
          f"P/E: {stock_data['P/E Ratio']}, D/E: {stock_data['Debt/Equity']}")
    print(f"   {nan_percentage:.2f}% de valores faltantes en {ticker}.\n")

    # 4.10) Agregamos a la lista de resultados
    results.append(stock_data)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="RSI", ascending=True, na_position='last')
df_results.fillna("No disponible", inplace=True)
print(df_results)

# ----------------------------------------
# 5. Análisis con LLM
# ----------------------------------------
class IPSAState(TypedDict):
    messages: Annotated[list, add_messages]
    df: pd.DataFrame

graph_builder = StateGraph(IPSAState)
llm = ChatOpenAI(model='gpt-4o-mini', api_key=openai_api_key, temperature=0)

def generate_ipsa_analysis(state: IPSAState):
    df = state['df']
    data_csv = df.to_csv(index=False)
    system_prompt = """
    Eres un analista financiero experto en la bolsa chilena.
    Tu tarea es examinar la información de múltiples acciones
    y dar un análisis fundamentado en español sin inventar datos.
    """
    human_prompt = f"""
    Aquí tienes datos de acciones en formato CSV:
    ```
    {data_csv}
    ```
    Proporciona:
    1) Un resumen de puntos clave.
    2) Fortalezas o debilidades observadas.
    3) Una recomendación objetiva basada en los datos.
    4) Una lista de cinco acciones recomendadas.
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    response = llm.invoke(messages)
    return {'messages': state['messages'] + [response]}

graph_builder.add_node('analyze_ipsa_df', generate_ipsa_analysis)
graph_builder.add_edge(START, 'analyze_ipsa_df')
graph_builder.add_edge('analyze_ipsa_df', END)
ipsa_graph = graph_builder.compile()

if __name__ == "__main__":
    state_input = {"messages": [("user", "Recomienda un portafolio para este conjunto de acciones.")], "df": df_results}
    events = ipsa_graph.stream(state_input, stream_mode='values')
    for event in events:
        if 'messages' in event:
            final_messages = event['messages']
            print(final_messages[-1].content)
    output_dict = {"data": df_results.to_dict(orient='records'), "analysis": final_messages[-1].content}
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)
