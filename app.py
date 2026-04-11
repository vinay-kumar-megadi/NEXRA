import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================================================
# LAUNCH SCREEN
# =========================================================

if "launched" not in st.session_state:
    st.session_state.launched = False

if not st.session_state.launched:
    st.markdown("### Welcome to ⚡ Nexra")
    st.markdown("AI-powered smart energy trading platform.")

    if st.button("Launch🚀"):
        st.session_state.launched = True
        st.rerun()

    st.stop()

st.set_page_config(page_title="⚡ Nexra", layout="wide")


# =========================================================
# CACHE
# =========================================================

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp')

@st.cache_resource
def train_model(df, features, target):
    model = SARIMAX(
        df[target],
        exog=df[features],
        order=(1,1,1),
        seasonal_order=(1,1,1,24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

#=========================================================
# HELPERS
# =========================================================
def preprocess_data(df, price_col, demand_col):

    df = df.tail(4000)

    df['hour'] = df['timestamp'].dt.hour

    df['season'] = df['season'].map({
        'Winter':0,'Summer':1,'Monsoon':2,
        0:0,1:1,2:2
    })

    df['day_type'] = df['day_type'].map({
        'Weekday':0,'Weekend':1,
        0:0,1:1
    })

    features = ['season','hour','day_type', demand_col]

    df[features + [price_col]] = df[features + [price_col]].apply(
        pd.to_numeric, errors='coerce'
    )

    df = df.dropna(subset=features + [price_col])

    return df, features

def format_hour(h):
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12} {suffix}"


# =========================================================
# SIDEBAR
# =========================================================

page = st.sidebar.radio("Module", ["Home", "Help", "Sellback", "Buyback"])

step = None
if page not in ["Home", "Help"]:
    step = st.sidebar.selectbox("Workflow", [
        "1. Select Data",
        "2. Analyze History",
        "3. Run Forecast",
        "4. Strategy & Profit"
    ])

battery_capacity = None
battery_level = None

if step == "4. Strategy & Profit":
    battery_capacity = st.sidebar.slider("Battery Capacity", 0, 100, 50)
    battery_level = st.sidebar.slider("Battery Level (%)", 0, 100, 40)

#  =========================================================
# HOME
# =========================================================

if page == "Home":
    st.title("⚡ Nexra")

    st.markdown("""
    ### Intelligent Energy Trading System

    Nexra is an AI-powered energy trading platform that helps optimize electricity usage,
    storage, and trading decisions using predictive analytics.

    🔹 Forecast electricity prices using machine learning  
    🔹 Optimize buyback and sellback strategies  
    🔹 Smart battery utilization  
    🔹 Maximize profit & minimize cost  

    
    """)
    st.stop()

# =========================================================
# GUIDE PAGE
# =========================================================

if page == "Help":
    st.title("⚡ Nexra Smart Usage Guide")

    st.markdown("""

    This system helps you make **intelligent energy trading decisions** using AI.

    You will be able to:
    - Predict future electricity prices
    - Identify best time to BUY or SELL
    - Optimize battery usage
    - Maximize profit / minimize cost

    ---
    ## How to use it ?
                            
    ### Steps :           
    ### 1️⃣ Select Data
    - Choose historical year
    - Select future prediction range

    📌 Output:
    → Defines the time window for forecasting

    ---
    ### 2️⃣ Analyze History
    - View past trends
    - Identify:
        - Average price
        - Peak hour
        - Price fluctuations

    📌 Output:
    → Understand market behavior before prediction

    ---
    ### 3️⃣ Run Forecast
    - AI predicts future prices using:
        - Time patterns
        - Season
        - Demand

    📌 Output:
    → Future price curve  
    → Expected highs and lows  

    ---
    ### 4️⃣ Strategy & Profit
    - System generates decisions:
        - SELL / BUY / STORE / USE BATTERY

    📌 Output:
    → Action plan  
    → Profit or cost estimation  
    → Smart recommendations  

    ---
    ## 🔋 Battery Importance

    Battery helps you:
    - Store energy at low price
    - Use/sell at high price

    ✔ Ideal range: **20% – 80%**

    ---
    ## 🎯 Final Outcome

    Using this app, you will:
    - Avoid buying at high prices
    - Sell at peak profit times
    - Use battery efficiently
    - Make data-driven decisions
    """)
    st.stop()



# =========================================================
# LOAD DATA
# =========================================================

if page == "Sellback":
    df = load_data("SELL BACK DATASET.xlsx")
    price_col = 'electricity_price'
    demand_col = 'grid_demand_kWh'
else:
    df = load_data("BUYBACK DATASET.xlsx")
    price_col = 'buyback_price'
    demand_col = 'buyback_demand_kWh'

# =========================================================
# STEP 1
# =========================================================

if step == "1. Select Data":

    st.title("📅 Select Time Range")

    year = st.selectbox("Historical Year", [2024, 2025])

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Future Start", pd.to_datetime("2026-03-01"))
    with col2:
        end_date = st.date_input("Future End", pd.to_datetime("2026-03-07"))

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(hours=23)

    if start_dt >= end_dt:
        st.error("End must be after start")
        st.stop()

    st.session_state['inputs'] = (year, start_dt, end_dt)
    st.success("Move to next step ➡️")

# =========================================================
# STEP 2
# =========================================================

elif step == "2. Analyze History":

    st.title("📊 Historical Analysis")

    year, start_dt, end_dt = st.session_state.get('inputs', (None, None, None))

    if start_dt is None:
        st.warning("Complete Step 1 first")
        st.stop()

    start_hist = pd.to_datetime(f"{year}-{start_dt.month}-{start_dt.day}")
    end_hist = pd.to_datetime(f"{year}-{end_dt.month}-{end_dt.day}") + pd.Timedelta(hours=23)

    hist_df = df[(df['timestamp'] >= start_hist) & (df['timestamp'] <= end_hist)].copy()

    if hist_df.empty:
        st.error("No historical data")
        st.stop()

    hist_df.set_index('timestamp', inplace=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Median", f"{hist_df[price_col].median():.2f}")
    col2.metric("High (75%)", f"{hist_df[price_col].quantile(0.75):.2f}")
    col3.metric("Max", f"{hist_df[price_col].max():.2f}")

    peak_hour = hist_df.groupby(hist_df.index.hour)[price_col].mean().idxmax()
    col4.metric("Peak Hour", peak_hour)

    fig = px.line(hist_df, x=hist_df.index, y=price_col, title="Historical Trend")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# STEP 3
# =========================================================

elif step == "3. Run Forecast":

    st.title("🔮 AI Forecast")

    year, start_dt, end_dt = st.session_state.get('inputs', (None, None, None))

    if start_dt is None:
        st.warning("Complete Step 1 first")
        st.stop()

    df = df.tail(4000)

    # FEATURES
    df['hour'] = df['timestamp'].dt.hour
    df['season'] = df['season'].map({'Winter':0,'Summer':1,'Monsoon':2,0:0,1:1,2:2})
    df['day_type'] = df['day_type'].map({'Weekday':0,'Weekend':1,0:0,1:1})

    features = ['season','hour','day_type', demand_col]

    df[features + [price_col]] = df[features + [price_col]].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # TRAIN MODEL (cached)
    model_fit = train_model(df, features, price_col)

    # FUTURE DATA
    future_dates = pd.date_range(start=start_dt, end=end_dt, freq='h')

    future_df = pd.DataFrame({'timestamp': future_dates})
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['day_type'] = future_df['timestamp'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

    def get_season(m):
        return 0 if m in [12,1,2] else 1 if m in [3,4,5] else 2

    future_df['season'] = future_df['timestamp'].dt.month.apply(get_season)

    # DEMAND PATTERN (critical)
    pattern_len = min(48, len(df))  
    pattern = df[demand_col].values[-pattern_len:]

    future_df[demand_col] = (list(pattern) * (len(future_df)//pattern_len + 1))[:len(future_df)]

    # FORECAST
    forecast = model_fit.forecast(steps=len(future_df), exog=future_df[features])
    future_df['Forecast Price'] = forecast.values

    st.session_state['forecast'] = future_df

    fig = px.line(future_df, x='timestamp', y='Forecast Price', title="Forecast")
    st.plotly_chart(fig, use_container_width=True)

## =========================================================
# STEP 4
# =========================================================

elif step == "4. Strategy & Profit":

    st.title("Strategy & Profit Simulation")

    future_df = st.session_state.get('forecast')

    if future_df is None:
        st.warning("Run forecast first")
        st.stop()

    high = future_df['Forecast Price'].quantile(0.75)
    mid = future_df['Forecast Price'].quantile(0.50)

    # =========================
    # DECISION LOGIC
    # =========================

    if page == "Sellback":

        def decision(row):
            if row['Forecast Price'] >= high and battery_level > 20:
                return "SELL_FULL"
            elif row['Forecast Price'] >= mid:
                return "SELL_PARTIAL"
            elif battery_level < 80:
                return "STORE"
            else:
                return "GRID"

    else:

        def decision(row):
            if row['Forecast Price'] <= mid and battery_level < 80:
                return "BUY_FULL"
            elif row['Forecast Price'] <= high:
                return "BUY_PARTIAL"
            elif battery_level > 20:
                return "USE_BATTERY"
            else:
                return "GRID"

    future_df['Decision'] = future_df.apply(decision, axis=1)

    def explain(row):
        if row['Decision'] == "SELL_FULL":
            return "High price → sell maximum energy"
        elif row['Decision'] == "SELL_PARTIAL":
            return "Moderate price → sell some energy"
        elif row['Decision'] == "STORE":
            return "Low price → store energy for later"
        elif row['Decision'] == "BUY_FULL":
            return "Low price → buy maximum energy"
        elif row['Decision'] == "BUY_PARTIAL":
            return "Moderate price → buy limited energy"
        elif row['Decision'] == "USE_BATTERY":
            return "High price → use stored battery"
        else:
            return "No strong signal → use grid"

    future_df['Reason'] = future_df.apply(explain, axis=1)



    # =========================
    # VALUE CALCULATION
    # =========================

    future_df['Value'] = future_df.apply(
        lambda r: r['Forecast Price'] if ("SELL" in r['Decision'] or "BUY" in r['Decision']) else 0,
        axis=1
    )

    if page == "Sellback":
        st.metric("💰 Total Profit", f"{future_df['Value'].sum():.2f}")
    else:
        st.metric("💸 Total Cost", f"{future_df['Value'].sum():.2f}")

    colA, colB = st.columns(2)

    colA.metric("Most Frequent Action", future_df['Decision'].mode()[0])

    if page == "Sellback":

        sell_df = future_df[future_df['Decision'].str.contains("SELL")]

        if not sell_df.empty:
            best_row = sell_df.loc[sell_df['Forecast Price'].idxmax()]
            best_hour = best_row['timestamp'].hour
            colB.metric("Best Hour to Sell", format_hour(best_hour))
        else:
            colB.metric("Best Hour to Sell", "N/A")

    else:

        buy_df = future_df[future_df['Decision'].str.contains("BUY")]

        if not buy_df.empty:
            best_row = buy_df.loc[buy_df['Forecast Price'].idxmin()]
            best_hour = best_row['timestamp'].hour
            colB.metric("Best Hour to Buy", format_hour(best_hour))
        else:
            colB.metric("Best Hour to Buy", "N/A")

    st.dataframe(future_df[['timestamp','Forecast Price','Decision','Reason','Value']])

    # =====================================================
    # RECOMMENDATIONS SECTION
    # =====================================================

    st.markdown("## 💡 Recommendations")

    decision_options = future_df['Decision'].unique().tolist()

    selected_decision = st.selectbox(
        "Select Decision Type",
        decision_options
    )

    # =========================
    # SELLBACK SUGGESTIONS
    # =========================

    if page == "Sellback":

        if selected_decision == "SELL_FULL":
            st.success("""
            Sell maximum energy when prices are at peak.

            - Focus on high-demand hours (evening peaks)
            - Ensure battery has enough charge before selling
            - This gives highest revenue per unit
            - Avoid holding energy when price is already high
            """)

        elif selected_decision == "SELL_PARTIAL":
            st.warning("""
            Sell only a portion of stored energy.

            - Keeps backup energy for future higher prices
            - Reduces risk of price fluctuations
            - Good when price is moderately high
            """)

        elif selected_decision == "STORE":
            st.info("""
            Store energy instead of selling.

            - Best when prices are low
            - Use battery to save energy for future peaks
            - Helps maximize later profit
            """)

        elif selected_decision == "GRID":
            st.error("""
            No strong opportunity detected.

            - Prices are not favorable for selling
            - Maintain battery level and wait
            """)

    # =========================
    # BUYBACK SUGGESTIONS
    # =========================

    else:

        if selected_decision == "BUY_FULL":
            st.success("""
            Buy maximum energy at lowest price.

            - Ideal during off-peak hours
            - Charge battery fully if possible
            - Minimizes total cost
            """)

        elif selected_decision == "BUY_PARTIAL":
            st.warning("""
            Buy limited energy.

            - Avoid full commitment due to uncertain trends
            - Keep flexibility for better prices
            """)

        elif selected_decision == "USE_BATTERY":
            st.info("""
            Use stored battery energy.

            - Avoid buying at high prices
            - Reduces dependency on grid
            - Efficient cost-saving approach
            """)

        elif selected_decision == "GRID":
            st.error("""
            Depend on grid supply.

            - Battery is low
            - Market conditions are not favorable
            """)

    # =====================================================
    # GENERAL TIPS
    # =====================================================

    st.markdown("""
    ---
    ### ⚡ General Tips

    - 🔋 Keep battery between **20% – 80%**
    - ⏰ Monitor peak hours daily
    - 📉 Buy/store during lowest price periods
    - 📈 Sell only when price is significantly high
    - ⚖️ Avoid aggressive full buy/sell repeatedly
    - 🔄 Maintain balance between usage and storage
    - 📊 Observe patterns instead of reacting instantly
    - 🧠 Combine forecast + battery logic for best decisions
    """)