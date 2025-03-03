import os
import time
import streamlit as st
import requests
import pandas as pd
import random
import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------
# Session State Initialization
# -----------------
if "game_state" not in st.session_state:
    st.session_state["game_state"] = None
if "trade_history" not in st.session_state:
    st.session_state["trade_history"] = []  # initialize as a list
if "page" not in st.session_state:
    st.session_state["page"] = "Game"  # default page

# -----------------
# Page Configuration
# -----------------
st.set_page_config(page_title="Stock Trading Simulator", layout="wide")
st.title("📉 Historic Trading Simulator")

# -----------------
# Sidebar Navigation (Always Visible)
# -----------------
st.sidebar.header("Navigation")
if st.sidebar.button("Game"):
    st.session_state["page"] = "Game"
if st.sidebar.button("Leaderboard"):
    st.session_state["page"] = "Leaderboard"

# -----------------
# Sidebar Controls (Always Defined)
# -----------------
st.sidebar.header("Controls")
window_input = st.sidebar.number_input("Simulation Window (days)", min_value=10, max_value=100, value=30, step=1)
start_button = st.sidebar.button("Start New Game (Random Stock)")
next_stock_button = st.sidebar.button("Next Stock")

# Long/Short operations
buy_long = st.sidebar.number_input("Buy Long Shares", min_value=1, value=10, step=1)
sell_long = st.sidebar.number_input("Sell Long Shares", min_value=1, value=5, step=1)
short_shares = st.sidebar.number_input("Short Shares", min_value=1, value=10, step=1)
cover_shares = st.sidebar.number_input("Cover Shares", min_value=1, value=5, step=1)
buy_long_button = st.sidebar.button("Buy Long")
sell_long_button = st.sidebar.button("Sell Long")
short_button = st.sidebar.button("Short")
cover_button = st.sidebar.button("Cover")

# Disable Next Candle if game is over (if on Game page)
if (st.session_state.get("game_state") is not None and 
    st.session_state["game_state"].get("step_count", 0) >= st.session_state["game_state"].get("max_steps", 0)):
    next_candle_button = st.sidebar.button("Next Candle", disabled=True)
else:
    next_candle_button = st.sidebar.button("Next Candle")

reset_button = st.sidebar.button("Reset Game")

with st.sidebar.expander("How to Use This Simulator"):
    st.markdown("""
    **Instructions:**
    
    - **Start New Game:** A random stock is chosen from a list of 100 volatile tickers.
    - **Next Stock:** Switch to a new random stock without resetting your balance or trade history.
    - **Buy Long/Sell Long:** For entering/exiting long positions.
    - **Short/Cover:** For initiating/increasing short positions and covering them.
    - **Next Candle:** Advance the simulation one day at a time (up to 50 moves).
    - **Trade History:** Review past trades and profit/loss.
    """)

# -----------------
# Configuration Defaults
# -----------------
DEFAULT_DAYS = 220      # Days to fetch
DEFAULT_INTERVAL = "1d"
DEFAULT_WINDOW = window_input if st.session_state["page"] == "Game" else 30

# -----------------
# Leaderboard Functions
# -----------------
LEADERBOARD_FILE = "leaderboard.csv"

def load_leaderboard(filename=LEADERBOARD_FILE):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=["name", "final_balance", "profit", "timestamp"])

def update_leaderboard(name, final_balance, profit, filename=LEADERBOARD_FILE):
    leaderboard = load_leaderboard(filename)
    # Remove any existing row for this name to avoid duplicates
    leaderboard = leaderboard[leaderboard["name"] != name]
    new_entry = pd.DataFrame({
        "name": [name],
        "final_balance": [final_balance],
        "profit": [profit],
        "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
    leaderboard.to_csv(filename, index=False)
    return leaderboard

# -----------------
# Helper Functions for Data and Indicators
# -----------------
def choose_random_stock():
    random_tickers = [
        "AAPL", "TSLA", "NVDA", "AMD", "GOOGL", "MSFT", "AMZN", "META", "NFLX", "INTC",
        "SNAP", "CRM", "BABA", "BIDU", "SHOP", "SQ", "PYPL", "UBER", "LYFT", "PTON",
        "OKTA", "ZM", "DOCU", "SPOT", "ROKU", "PLTR", "FVRR", "EXAS", "ILMN", "QCOM",
        "MU", "CSCO", "AMAT", "ASML", "WBD", "DIS", "EA", "ATVI", "NTES", "TSM",
        "JD", "PDD", "BILI", "DIDI", "COIN", "MARA", "RIOT", "HIVE", "F", "GM",
        "BBY", "SIRI", "NIO", "XPEV", "LI", "ZNGA", "VRTX", "REGN", "CRSP", "EDIT",
        "FCX", "NEM", "AMGN", "GE", "BA", "CAT", "DE", "LMT", "RTX", "GS",
        "JPM", "BAC", "C", "CRWD", "ZS", "DDOG", "NET", "SNOW", "NOW", "TEAM",
        "TWLO", "ADBE", "V", "MA", "BKNG", "T", "VZ", "WMT", "KO", "PEP",
        "SBUX", "CMG", "DAL", "UAL", "NKE", "FISV", "AAL", "HUM", "SGEN", "BIIB"
    ]
    return random.choice(random_tickers)

@st.cache_data(show_spinner=False)
def fetch_polygon_data(ticker: str, days: int, interval: str) -> pd.DataFrame:
    POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
    if not POLYGON_API_KEY:
        st.error("Please set your POLYGON_API_KEY environment variable!")
        return pd.DataFrame()
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit=500&apiKey={POLYGON_API_KEY}"
    )
    attempts = 0
    while attempts < 3:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                st.warning("Rate limit exceeded, retrying in 5 seconds...")
                time.sleep(5)
                attempts += 1
            else:
                st.error(f"Error fetching data from Polygon: {e}")
                return pd.DataFrame()
    else:
        st.error("Exceeded maximum retry attempts.")
        return pd.DataFrame()
    if "results" not in data or not data["results"]:
        st.error("Polygon API did not return any results.")
        return pd.DataFrame()
    results = data["results"]
    df = pd.DataFrame(results)
    df = df.rename(columns={
        "t": "Date",
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume"
    })
    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
    try:
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except KeyError:
        st.error("Polygon data is missing required columns.")
        return pd.DataFrame()
    df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)
    df.drop(columns=["TP"], inplace=True)
    df.sort_values("Date", inplace=True)
    return df

def compute_MFI(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * df["Volume"]
    direction = tp.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    pos_mf = rmf.where(direction > 0, 0)
    neg_mf = rmf.where(direction < 0, 0)
    return 100 - (100 / (1 + pos_mf.rolling(window=period).sum() / (neg_mf.rolling(window=period).sum() + 1e-9)))

def init_game() -> dict:
    ticker = choose_random_stock()
    df = fetch_polygon_data(ticker, DEFAULT_DAYS, "1d")
    if df.empty or len(df) < 2:
        st.error(f"No data fetched for {ticker}.")
        return None
    total_candles = len(df)
    sim_window = DEFAULT_WINDOW
    desired_steps = 50
    if total_candles < sim_window + desired_steps:
        desired_steps = max(0, total_candles - sim_window)
    max_start = total_candles - (sim_window + desired_steps)
    if max_start < 0:
        max_start = 0
    start_base = random.randint(0, max_start)
    current_idx = start_base + sim_window
    return {
        "ticker": ticker,
        "df": df,
        "current_index": current_idx,
        "balance": 100000.0,
        "shares": 0,  # Positive for long, negative for short
        "avg_cost": 0.0,
        "window": sim_window,
        "step_count": 0,
        "max_steps": desired_steps
    }

def get_current_price(game_state: dict):
    idx = max(0, game_state["current_index"] - 1)
    df = game_state["df"]
    if idx < len(df):
        return df.iloc[idx]["Close"]
    return None

def advance_candle(game_state: dict, steps=1):
    if game_state["step_count"] < game_state["max_steps"]:
        df = game_state["df"]
        max_index = len(df)
        game_state["current_index"] = min(game_state["current_index"] + steps, max_index)
        game_state["step_count"] += steps
    else:
        st.warning("Game Over: Maximum candle clicks reached.")

def plot_chart_interactive_dark(game_state: dict) -> go.Figure:
    df = game_state["df"]
    window = game_state["window"]
    idx = game_state["current_index"]
    start_idx = max(0, idx - window)
    df_part = df.iloc[start_idx:idx].copy()
    if len(df_part) < 2:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, text="No data available", showarrow=False)
        return fig
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df_part[c] = pd.to_numeric(df_part[c])
    df_part["EMA_9"] = df_part["Close"].ewm(span=9, adjust=False).mean()
    df_part["EMA_20"] = df_part["Close"].ewm(span=20, adjust=False).mean()
    df_part["TP"] = (df_part["High"] + df_part["Low"] + df_part["Close"]) / 3
    df_part["VWAP"] = (df_part["TP"] * df_part["Volume"]).cumsum() / (df_part["Volume"].cumsum() + 1e-9)
    df_part["Volume_MA_8"] = df_part["Volume"].rolling(window=8).mean()
    df_part["MFI"] = compute_MFI(df_part)
    
    df_part["Volume_Up"] = df_part.apply(lambda row: row["Volume"] if row["Close"] >= row["Open"] else 0, axis=1)
    df_part["Volume_Down"] = df_part.apply(lambda row: row["Volume"] if row["Close"] < row["Open"] else 0, axis=1)
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.2],
        vertical_spacing=0.02,
        subplot_titles=("Price", "Volume", "MFI")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_part["Date"].tolist(),
            open=df_part["Open"].tolist(),
            high=df_part["High"].tolist(),
            low=df_part["Low"].tolist(),
            close=df_part["Close"].tolist(),
            name="Candlestick",
            increasing_line_color="green",
            decreasing_line_color="red"
        ),
        row=1, col=1
    )
    
    # Overlays on Price
    fig.add_trace(
        go.Scatter(
            x=df_part["Date"].tolist(), y=df_part["EMA_9"].tolist(),
            mode="lines", name="EMA 9", line=dict(color="magenta", width=1)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_part["Date"].tolist(), y=df_part["EMA_20"].tolist(),
            mode="lines", name="EMA 20", line=dict(color="orange", width=1)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_part["Date"].tolist(), y=df_part["VWAP"].tolist(),
            mode="lines", name="VWAP", line=dict(color="blue", width=1)
        ),
        row=1, col=1
    )
    
    # Volume subplot
    fig.add_trace(
        go.Bar(
            x=df_part["Date"].tolist(), y=df_part["Volume_Up"].tolist(),
            marker_color="green", name="Volume Up"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=df_part["Date"].tolist(), y=df_part["Volume_Down"].tolist(),
            marker_color="red", name="Volume Down"
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_part["Date"].tolist(), y=df_part["Volume_MA_8"].tolist(),
            mode="lines", name="Vol MA(8)", line=dict(color="white", width=1),
            fill="tozeroy", fillcolor="rgba(255,255,255,0.1)"
        ),
        row=2, col=1
    )
    
    # MFI subplot
    fig.add_trace(
        go.Scatter(
            x=df_part["Date"].tolist(), y=df_part["MFI"].tolist(),
            mode="lines", name="MFI", line=dict(color="white", width=1),
            fill="tozeroy", fillcolor="rgba(255,255,255,0.1)"
        ),
        row=3, col=1
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    # Enhanced interactivity: enable scroll zoom and pan mode (touch-friendly)
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
        'responsive': True  # Makes the chart responsive to window size changes
    }
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
        barmode="overlay"
    )
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=["2025-01-01", "2025-12-25", "2025-12-26", "2026-01-01"])
        ],
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)"
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig

# -----------------
# Trading Functions
# -----------------
def process_buy_long(game_state: dict, shares: int):
    price = get_current_price(game_state)
    if price is None:
        return "Error: No current price."
    if game_state["shares"] < 0:
        return "You have a short position. Use Cover to reduce it."
    cost = round(shares * price, 2)
    if cost > game_state["balance"]:
        return f"Insufficient balance to buy {shares} shares."
    pos = game_state["shares"]
    new_total = pos + shares
    new_avg = ((pos * game_state["avg_cost"]) + cost) / new_total if new_total > 0 else price
    game_state["shares"] = new_total
    game_state["avg_cost"] = new_avg
    game_state["balance"] -= cost
    msg = f"Bought {shares} long shares at ${price:.2f}."
    st.session_state["trade_history"].append({
        "action": "Buy Long",
        "shares": shares,
        "price": price,
        "timestamp": datetime.datetime.now(),
        "details": msg
    })
    return msg

def process_sell_long(game_state: dict, shares: int):
    price = get_current_price(game_state)
    if price is None:
        return "Error: No current price."
    if game_state["shares"] <= 0:
        return "No long position to sell."
    if shares > game_state["shares"]:
        return "Cannot sell more than you hold."
    proceeds = round(shares * price, 2)
    game_state["balance"] += proceeds
    game_state["shares"] -= shares
    if game_state["shares"] == 0:
        game_state["avg_cost"] = 0.0
    msg = f"Sold {shares} long shares at ${price:.2f}."
    st.session_state["trade_history"].append({
        "action": "Sell Long",
        "shares": shares,
        "price": price,
        "timestamp": datetime.datetime.now(),
        "details": msg
    })
    return msg

def process_short(game_state: dict, shares: int):
    price = get_current_price(game_state)
    if price is None:
        return "Error: No current price."
    if game_state["shares"] > 0:
        return "You hold a long position. Sell long shares first before shorting."
    # When shorting, do not update balance so that covering with no price change yields zero profit.
    if game_state["shares"] == 0:
        game_state["shares"] = -shares
        game_state["avg_cost"] = price
        msg = f"Shorted {shares} shares at ${price:.2f}."
    else:
        old = abs(game_state["shares"])
        new = old + shares
        new_avg = (old * game_state["avg_cost"] + shares * price) / new
        game_state["shares"] -= shares
        game_state["avg_cost"] = new_avg
        msg = f"Increased short by {shares} shares at ${price:.2f}."
    st.session_state["trade_history"].append({
        "action": "Short",
        "shares": shares,
        "price": price,
        "timestamp": datetime.datetime.now(),
        "details": msg
    })
    return msg

def process_cover(game_state: dict, shares: int):
    price = get_current_price(game_state)
    if price is None:
        return "Error: No current price."
    if game_state["shares"] >= 0:
        return "No short position to cover."
    if shares > abs(game_state["shares"]):
        return "Cannot cover more than your short position."
    # Calculate profit: (short entry price - cover price) * shares.
    diff = game_state["avg_cost"] - price
    profit = 0 if abs(diff) < 1e-6 else round(diff * shares, 2)
    # Update balance by profit
    game_state["balance"] += profit
    game_state["shares"] += shares  # moving towards 0
    if game_state["shares"] == 0:
        game_state["avg_cost"] = 0.0
    msg = f"Covered {shares} shares at ${price:.2f}, profit: ${profit:.2f}."
    st.session_state["trade_history"].append({
        "action": "Cover",
        "shares": shares,
        "price": price,
        "timestamp": datetime.datetime.now(),
        "details": msg
    })
    return msg

# -----------------
# Main App Interaction
# -----------------
def init_game_wrapper():
    return init_game()

if reset_button:
    st.session_state["game_state"] = None
    st.session_state["trade_history"] = []

if next_stock_button:
    new_ticker = choose_random_stock()
    new_df = fetch_polygon_data(new_ticker, DEFAULT_DAYS, "1d")
    if new_df.empty or len(new_df) < 2:
        st.warning(f"New stock {new_ticker} returned no data. Try again.")
    else:
        st.session_state["game_state"]["ticker"] = new_ticker
        st.session_state["game_state"]["df"] = new_df
        max_index_new = len(new_df)
        if st.session_state["game_state"]["current_index"] > max_index_new:
            st.session_state["game_state"]["current_index"] = max_index_new
        st.success("Switched to a new stock.")

if start_button:
    st.session_state["game_state"] = init_game_wrapper()
    st.session_state["trade_history"] = []

if next_candle_button:
    if st.session_state.get("game_state") is None:
        st.error("Game state is not initialized. Please start a new game.")
    elif st.session_state["game_state"].get("df") is None:
        st.error("Game state is missing data. Please restart the game.")
    else:
        df = st.session_state["game_state"]["df"]
        if df.empty:
            st.error("The data is empty. Please restart the game.")
        else:
            max_index = len(df)
            st.session_state["game_state"]["current_index"] = min(
                st.session_state["game_state"]["current_index"] + 1, max_index
            )
            st.session_state["game_state"]["step_count"] += 1
            try:
                st.experimental_rerun()
            except Exception:
                st.write("Page will refresh on next interaction.")
                
if buy_long_button:
    if st.session_state.get("game_state") is None:
        st.error("Game state is not initialized. Please start a new game.")
    else:
        msg = process_buy_long(st.session_state["game_state"], int(buy_long))
        st.sidebar.success(msg)

if sell_long_button:
    if st.session_state.get("game_state") is None:
        st.error("Game state is not initialized. Please start a new game.")
    else:
        msg = process_sell_long(st.session_state["game_state"], int(sell_long))
        st.sidebar.success(msg)

if short_button:
    if st.session_state.get("game_state") is None:
        st.error("Game state is not initialized. Please start a new game.")
    else:
        msg = process_short(st.session_state["game_state"], int(short_shares))
        st.sidebar.success(msg)

if cover_button:
    if st.session_state.get("game_state") is None:
        st.error("Game state is not initialized. Please start a new game.")
    else:
        msg = process_cover(st.session_state["game_state"], int(cover_shares))
        st.sidebar.success(msg)

# -----------------
# Page Display
# -----------------
if st.session_state["page"] == "Game":
    if st.session_state.get("game_state"):
        gs = st.session_state["game_state"]
        st.write("**Stock:** Hidden")
        st.write(f"**Current Candle Index:** {gs['current_index']} / {len(gs['df'])}")
        st.write(f"**Balance:** ${gs['balance']:.2f}")
        if gs["shares"] > 0:
            st.write(f"**Position:** Long {gs['shares']} shares (Avg Cost: ${gs['avg_cost']:.2f})")
        elif gs["shares"] < 0:
            st.write(f"**Position:** Short {abs(gs['shares'])} shares (Avg Price: ${gs['avg_cost']:.2f})")
        else:
            st.write("**Position:** Flat")
        
        cp = get_current_price(gs)
        if cp is not None:
            st.write(f"**Current Price:** ${cp:.2f}")
        
        total_invested = gs["shares"] * gs["avg_cost"]
        current_value = gs["shares"] * cp if cp else 0
        pl_total = current_value - total_invested
        st.write(f"**Unrealized P/L:** ${pl_total:.2f}")
        
        if gs["step_count"] >= gs["max_steps"]:
            st.error("Game Over: Maximum candle clicks reached.")
        
        # Configure interactivity options for Plotly chart (touch friendly)
        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'responsive': True
        }
        fig = plot_chart_interactive_dark(gs)
        fig.update_layout(dragmode="pan")
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        if st.session_state["trade_history"]:
            st.subheader("Trade History")
            trade_df = pd.DataFrame(st.session_state["trade_history"])
            trade_df["timestamp"] = trade_df["timestamp"].astype(str)
            st.dataframe(trade_df)
        
        if gs["step_count"] >= gs["max_steps"]:
            st.subheader("Submit Your Score to the Leaderboard")
            name = st.text_input("Enter your name for the leaderboard:")
            if st.button("Submit Score"):
                final_balance = gs["balance"]
                profit = final_balance - 100000.0  # starting balance of 100k
                leaderboard = update_leaderboard(name, final_balance, profit)
                st.success("Score submitted!")
                st.subheader("Leaderboard")
                leaderboard = leaderboard.sort_values(by="profit", ascending=False).reset_index(drop=True)
                leaderboard["RankInt"] = leaderboard.index + 1
                def ordinal_suffix(num: int) -> str:
                    teen_exceptions = {11, 12, 13}
                    last_two = num % 100
                    if last_two in teen_exceptions:
                        return f"{num}th"
                    last_digit = num % 10
                    if last_digit == 1:
                        return f"{num}st"
                    elif last_digit == 2:
                        return f"{num}nd"
                    elif last_digit == 3:
                        return f"{num}rd"
                    else:
                        return f"{num}th"
                leaderboard["Rank"] = leaderboard["RankInt"].apply(ordinal_suffix)
                leaderboard = leaderboard.drop(columns=["timestamp", "RankInt"])
                st.dataframe(leaderboard)
        
        with st.expander("Learn About VPA & Candlestick Patterns"):
            st.markdown("""
            ### Volume Price Analysis (VPA)
            - **Volume:** The number of shares traded in a candle.
            - **Price:** The open, high, low, and close of the candle.
            
            **VPA Insight:**  
            When price moves occur with **high volume**, it signals strong market participation. Conversely, moves on **low volume** may lack conviction.
            
            ### Common Candlestick Patterns
            - **Doji:** Open and close are nearly equal, indicating market indecision.
            - **Hammer:** A small body with a long lower wick, suggesting potential bullish reversal.
            - **Shooting Star:** A small body with a long upper wick, suggesting potential bearish reversal.
            - **Engulfing:** A candle that completely engulfs the previous candle's body, signaling a strong reversal.
            
            **How to Use This Information:**  
            Observe the candlestick shapes and volume bars on the chart. Patterns like a **hammer on high volume** or a **shooting star on high volume** may indicate upcoming reversals.
            """)
    else:
        st.info("Click 'Start New Game' in the sidebar to begin.")

elif st.session_state["page"] == "Leaderboard":
    st.header("Global Leaderboard")
    leaderboard = load_leaderboard()
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(by="profit", ascending=False).reset_index(drop=True)
        leaderboard["RankInt"] = leaderboard.index + 1
        def ordinal_suffix(num: int) -> str:
            teen_exceptions = {11, 12, 13}
            last_two = num % 100
            if last_two in teen_exceptions:
                return f"{num}th"
            last_digit = num % 10
            if last_digit == 1:
                return f"{num}st"
            elif last_digit == 2:
                return f"{num}nd"
            elif last_digit == 3:
                return f"{num}rd"
            else:
                return f"{num}th"
        leaderboard["Rank"] = leaderboard["RankInt"].apply(ordinal_suffix)
        leaderboard.drop(columns=["timestamp", "RankInt"], inplace=True)
        st.dataframe(leaderboard)
    else:
        st.info("No leaderboard data available yet.")
else:
    st.info("Click 'Start New Game' in the sidebar to begin.")
