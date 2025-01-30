from typing import Union, Dict, List, TypedDict, Annotated
import yfinance as yf
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

def calculate_vortex(df, period=14):
    """
    Calcula el Vortex Indicator (VI+ y VI-) para un DataFrame que contenga
    columnas 'High', 'Low' y 'Close'. Devuelve dos columnas: 'VI_plus' y 'VI_minus'.
    """
    # Definimos True Range (TR) y el movimiento positivo/negativo
    high = df['High']
    low = df['Low']
    close = df['Close']
    previous_close = close.shift(1)
    
    # Typical True Range para Vortex
    tr = (high - low).abs()
    tr1 = (high - previous_close).abs()
    tr2 = (low - previous_close).abs()
    true_range = pd.concat([tr, tr1, tr2], axis=1).max(axis=1)
    
    # Movimientos positivos y negativos
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    # Suma móvil de TR, vm_plus y vm_minus
    tr_sum = true_range.rolling(window=period).sum()
    vip = vm_plus.rolling(window=period).sum() / tr_sum
    vim = vm_minus.rolling(window=period).sum() / tr_sum

    return vip, vim

def calculate_bollinger_bands(series, period=20, num_std=2):
    """
    Calcula Bollinger Bands para una serie de precios. 
    Retorna (middle_band, upper_band, lower_band).
    """
    middle_band = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)
    return middle_band, upper_band, lower_band

def calculate_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(period).mean()
    
    # MAD: mean absolute deviation sobre la diferencia tp - sma_tp
    mad = (tp - sma_tp).abs().rolling(period).mean()
    
    # Factor 0.015 es estándar en CCI
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

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
    if ticker not in all_data:
        print(f"   ❌ No hay datos descargados para {ticker} (posible error con el sufijo .SN)")
        results.append({
            "Stock": ticker,
            "Último Precio": None,
            "RSI": None,
            "P/E Ratio": None,
            "Debt/Equity": None,
            "VI_plus": None,
            "VI_minus": None,
            "Bollinger_Up": None,
            "Bollinger_Down": None,
            "CCI": None
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
            "Debt/Equity": None,
            "VI_plus": None,
            "VI_minus": None,
            "Bollinger_Up": None,
            "Bollinger_Down": None,
            "CCI": None
        })
        continue

    if "Close" not in df.columns:
        print(f"   ⚠️ La columna 'Close' no está presente en {ticker}.")
        results.append({
            "Stock": ticker,
            "Último Precio": None,
            "RSI": None,
            "P/E Ratio": None,
            "Debt/Equity": None,
            "VI_plus": None,
            "VI_minus": None,
            "Bollinger_Up": None,
            "Bollinger_Down": None,
            "CCI": None
        })
        continue

    # 4.3) Calcular RSI
    df["RSI"] = calculate_rsi(df["Close"])

    # 4.4) Calcular Vortex (VI+ y VI-)
    df["VI_plus"], df["VI_minus"] = calculate_vortex(df, period=14)

    # 4.5) Calcular Bollinger Bands
    (df["BB_Middle"], df["BB_Upper"], df["BB_Lower"]) = calculate_bollinger_bands(df["Close"], period=20, num_std=2)

    # 4.6) Calcular CCI
    df["CCI"] = calculate_cci(df, period=20)

    # 4.7) Tomamos últimos valores
    last_close = df["Close"].iloc[-1]
    last_rsi = df["RSI"].iloc[-1] if not df["RSI"].isna().all() else None
    last_vi_plus = df["VI_plus"].iloc[-1] if not df["VI_plus"].isna().all() else None
    last_vi_minus = df["VI_minus"].iloc[-1] if not df["VI_minus"].isna().all() else None
    last_bb_upper = df["BB_Upper"].iloc[-1] if not df["BB_Upper"].isna().all() else None
    last_bb_lower = df["BB_Lower"].iloc[-1] if not df["BB_Lower"].isna().all() else None
    last_cci = df["CCI"].iloc[-1] if not df["CCI"].isna().all() else None

    # 4.8) Buscamos métricas financieras
    fm = safe_get_financial_metrics(ticker)  # P/E, Debt/Equity, etc.
    pe_ratio = fm.get('pe_ratio')
    debt_to_equity = fm.get('debt_to_equity')

    # 4.9) Armamos registro
    stock_data = {
        "Stock": ticker,
        "Último Precio": last_close,
        "RSI": round(last_rsi, 2) if last_rsi is not None else None,
        "P/E Ratio": pe_ratio,
        "Debt/Equity": debt_to_equity,
        "VI_plus": round(last_vi_plus, 3) if last_vi_plus is not None else None,
        "VI_minus": round(last_vi_minus, 3) if last_vi_minus is not None else None,
        "Bollinger_Up": round(last_bb_upper, 3) if last_bb_upper is not None else None,
        "Bollinger_Down": round(last_bb_lower, 3) if last_bb_lower is not None else None,
        "CCI": round(last_cci, 3) if last_cci is not None else None
    }

    # Porcentaje de NaN (opcional, para depurar)
    nan_count = sum(v is None for v in stock_data.values())
    nan_percentage = (nan_count / len(stock_data)) * 100

    # 4.10) Mostramos en consola un resumen inmediato
    print(f"   Precio: {stock_data['Último Precio']}, RSI: {stock_data['RSI']}, "
          f"P/E: {stock_data['P/E Ratio']}, D/E: {stock_data['Debt/Equity']}, "
          f"VI+: {stock_data['VI_plus']}, VI-: {stock_data['VI_minus']}, "
          f"BollingerUp: {stock_data['Bollinger_Up']}, BollingerDown: {stock_data['Bollinger_Down']}, "
          f"CCI: {stock_data['CCI']}")
    print(f"   {nan_percentage:.2f}% de valores faltantes en {ticker}.\n")

    # 4.11) Agregamos a la lista de resultados
    results.append(stock_data)

df_results = pd.DataFrame(results)
# Ordenar solo por RSI, ascendente
df_results = df_results.sort_values(by="RSI", ascending=True, na_position='last')
df_results.fillna("No disponible", inplace=True)
print(df_results)

# ----------------------------------------
# 5. Análisis con LLM (opcional)
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
    4) Un listado de las mejores acciones según el Commodity Channel Index (CCI), EL Vortex Indicator (VI) y las Bollinger Bands, además de una breve explicación.
    4) Una lista de cinco acciones recomendadas, en base a todo lo anterior. Mientras más diferentes sean los mercados a los que pertenecen estas acciones, mejor.
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
