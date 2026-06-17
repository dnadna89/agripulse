import streamlit as st
import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AgriPulse", page_icon="🌱", layout="wide")

@st.cache_resource
def load_and_train():
    def clean_one(path):
        raw = pd.read_excel(path, header=None).rename(columns={0: 'date', 5: 'modal'})
        raw['date'] = pd.to_datetime(raw['date'], dayfirst=True, errors='coerce').ffill()
        raw['modal'] = pd.to_numeric(
            raw['modal'].astype(str).str.replace(',', '', regex=False), errors='coerce')
        return raw[(raw['date'].notna()) & (raw['modal'] > 0)][['date', 'modal']]

    files = sorted(glob.glob('Onion_*.xlsx'))
    prices = pd.concat([clean_one(f) for f in files], ignore_index=True)
    prices = (prices.groupby('date', as_index=False)['modal'].mean()
                    .sort_values('date').reset_index(drop=True)
                    .rename(columns={'modal': 'price'}))

    weather = pd.read_csv('surat_weather.csv', skiprows=3)
    rain_col = [c for c in weather.columns if 'precip' in c.lower()][0]
    temp_col = [c for c in weather.columns if 'temp' in c.lower()][0]
    weather = weather[[weather.columns[0], rain_col, temp_col]]
    weather.columns = ['date', 'rainfall', 'temperature']
    weather['date'] = pd.to_datetime(weather['date'], dayfirst=True, errors='coerce')

    data = prices.merge(weather, on='date', how='inner').sort_values('date').reset_index(drop=True)

    df = data.copy()
    df['price_lag1'] = df['price'].shift(1)
    df['price_lag7'] = df['price'].shift(7)
    df['price_roll7'] = df['price'].rolling(7).mean()
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df = df.dropna().reset_index(drop=True)

    feats = ['price_lag1','price_lag7','price_roll7','dayofyear','month','rainfall','temperature']
    X, y = df[feats], df['price']
    split = int(len(df) * 0.8)

    tm = RandomForestRegressor(n_estimators=300, random_state=42).fit(X.iloc[:split], y.iloc[:split])
    pred = tm.predict(X.iloc[split:])
    mape = (abs((y.iloc[split:] - pred) / y.iloc[split:]).mean()) * 100
    mae = mean_absolute_error(y.iloc[split:], pred)

    final = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
    return df, final, feats, round(100 - mape, 1), round(mae)

def make_forecast(df, model, feats, days):
    rain = df['rainfall'].tail(30).mean()
    temp = df['temperature'].tail(30).mean()
    hist = df[['date', 'price']].copy()
    out = []
    for _ in range(days):
        nd = hist['date'].iloc[-1] + pd.Timedelta(days=1)
        row = {'price_lag1': hist['price'].iloc[-1], 'price_lag7': hist['price'].iloc[-7],
               'price_roll7': hist['price'].tail(7).mean(), 'dayofyear': nd.dayofyear,
               'month': nd.month, 'rainfall': rain, 'temperature': temp}
        p = model.predict(pd.DataFrame([row])[feats])[0]
        hist = pd.concat([hist, pd.DataFrame([{'date': nd, 'price': p}])], ignore_index=True)
        out.append({'date': nd, 'price': p})
    return pd.DataFrame(out)

# ---------------- UI ----------------
st.title("🌱 AgriPulse")
st.caption("Climate-Aware Crop Price Intelligence — turning forecasts into waste-prevention decisions")

df, model, feats, accuracy, mae = load_and_train()

c1, c2 = st.columns(2)
with c1:
    st.selectbox("Select Crop", ["Onion"])
    st.caption("Crop-agnostic — Potato, Tomato & Pulses ready to add (just drop in a data file)")
with c2:
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14)

forecast = make_forecast(df, model, feats, horizon)
last_price = df['price'].iloc[-1]
future_price = forecast['price'].iloc[-1]
pct = (future_price - last_price) / last_price * 100

m1, m2, m3 = st.columns(3)
m1.metric(f"Predicted price in {horizon} days", f"₹{future_price:,.0f}/qtl", f"{pct:+.1f}%")
m2.metric("Model accuracy", f"{accuracy}%", help="Tested on recent weeks the model never saw")
m3.metric("Avg error", f"±₹{mae}/qtl")

if pct <= -8:
    st.error(f"⚠️ HIGH GLUT RISK: Onion predicted to fall {pct:.1f}% in {horizon} days. "
             "Likely oversupply → dumping/waste risk. Recommend staggered selling or cold storage.")
elif pct >= 8:
    st.warning(f"📈 PRICE SURGE: Onion predicted to rise {pct:.1f}% in {horizon} days. "
               "Favourable selling window — consider releasing stored stock.")
else:
    st.success(f"✅ STABLE MARKET: Onion expected within ±8% over {horizon} days. Low waste risk.")

hist = df[['date', 'price']].tail(90).rename(columns={'price': 'Historical'})
bridge = pd.DataFrame([{'date': hist['date'].iloc[-1], 'Forecast': hist['Historical'].iloc[-1]}])
fc = pd.concat([bridge, forecast.rename(columns={'price': 'Forecast'})], ignore_index=True)
chart_df = pd.merge(hist, fc, on='date', how='outer').sort_values('date').set_index('date')
st.line_chart(chart_df)

st.caption("Source: Govt. of India Agmarknet (Onion, Gujarat) + Open-Meteo climate data (Surat). "
           "Model: Random Forest fusing market history with rainfall & temperature signals.")
