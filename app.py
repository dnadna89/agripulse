import streamlit as st
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AgriPulse", page_icon="🌱", layout="wide")
CROPS = ['Onion', 'Potato', 'Tomato']
FEATS = ['price_lag1','price_lag7','price_roll7','dayofyear','month','rainfall','temp_mean']
MIN_DAYS = 150

@st.cache_resource
def load_everything():
    def load_weather(path):
        w = pd.read_csv(path, skiprows=3)
        def find(*keys):
            for c in w.columns:
                if all(k in str(c).lower() for k in keys): return c
            return None
        rain, tmean = find('precip'), find('temperature','mean')
        out = pd.DataFrame()
        out['date']      = pd.to_datetime(w[w.columns[0]], format='%Y-%m-%d', errors='coerce')
        out['rainfall']  = pd.to_numeric(w[rain],  errors='coerce') if rain  else 0.0
        out['temp_mean'] = pd.to_numeric(w[tmean], errors='coerce') if tmean else None
        return out

    weather = (pd.concat([load_weather(f) for f in glob.glob('*_weather.csv')], ignore_index=True)
                 .groupby('date', as_index=False).mean(numeric_only=True))
    wdates = set(weather['date'])

    crops_raw, varieties = {}, {}
    for crop in CROPS:
        parts = []
        for path in sorted(glob.glob(f'{crop}_*.xlsx')):
            raw = pd.read_excel(path, header=None).rename(columns={0:'date', 2:'variety', 5:'modal'})
            raw['date']    = pd.to_datetime(raw['date'], format='%d/%m/%Y', errors='coerce').ffill()
            raw['modal']   = pd.to_numeric(raw['modal'].astype(str).str.replace(',', '', regex=False),
                                           errors='coerce')
            raw['variety'] = raw['variety'].astype(str).str.strip()
            parts.append(raw[(raw['date'].notna()) & (raw['modal'] > 0)
                             & (~raw['variety'].isin(['nan','None','']))][['date','variety','modal']])
        df = pd.concat(parts, ignore_index=True)
        crops_raw[crop] = df
        good = [v for v in df['variety'].unique()
                if sum(d in wdates for d in df.loc[df['variety']==v,'date'].unique()) >= MIN_DAYS]
        varieties[crop] = sorted(good, key=lambda v: -(df['variety']==v).sum())
    return crops_raw, weather, varieties

CROPS_RAW, WEATHER, VARIETIES = load_everything()

def daily_series(crop, variety):
    df = CROPS_RAW[crop]
    sub = df if variety == "All varieties" else df[df['variety'] == variety]
    return (sub.groupby('date', as_index=False)['modal'].median()
               .rename(columns={'modal':'price'}).sort_values('date').reset_index(drop=True))

@st.cache_resource
def get_model(crop, variety):
    df = daily_series(crop, variety).merge(WEATHER, on='date', how='inner').sort_values('date').reset_index(drop=True)
    df['price_lag1']  = df['price'].shift(1)
    df['price_lag7']  = df['price'].shift(7)
    df['price_roll7'] = df['price'].rolling(7).mean()
    df['dayofyear'], df['month'] = df['date'].dt.dayofyear, df['date'].dt.month
    df = df.dropna().reset_index(drop=True)
    X, y = df[FEATS], df['price']
    split = int(len(df) * 0.8)
    tm = RandomForestRegressor(n_estimators=300, random_state=42).fit(X.iloc[:split], y.iloc[:split])
    pred = tm.predict(X.iloc[split:])
    mape = (abs((y.iloc[split:] - pred) / y.iloc[split:]).mean()) * 100
    mae = mean_absolute_error(y.iloc[split:], pred)
    final = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
    return df, final, round(100 - mape, 1), round(mae)

def forecast_ahead(df, model, days):
    rain  = df['rainfall'].tail(30).mean()
    tmean = df['temp_mean'].tail(30).mean()
    hist, out = df[['date','price']].copy(), []
    for _ in range(days):
        nd = hist['date'].iloc[-1] + pd.Timedelta(days=1)
        row = {'price_lag1': hist['price'].iloc[-1], 'price_lag7': hist['price'].iloc[-7],
               'price_roll7': hist['price'].tail(7).mean(), 'dayofyear': nd.dayofyear,
               'month': nd.month, 'rainfall': rain, 'temp_mean': tmean}
        p = model.predict(pd.DataFrame([row])[FEATS])[0]
        hist = pd.concat([hist, pd.DataFrame([{'date': nd, 'price': p}])], ignore_index=True)
        out.append({'date': nd, 'price': p})
    return pd.DataFrame(out)

# ---------------- UI ----------------
st.title("🌱 AgriPulse")
st.caption("Climate-Aware Crop Price Intelligence — multi-crop, multi-variety, statewide climate signal")

c1, c2, c3 = st.columns(3)
with c1:
    crop = st.selectbox("Crop", CROPS)
with c2:
    variety = st.selectbox("Variety", ["All varieties"] + VARIETIES[crop])
with c3:
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14)
st.caption("Generic labels (e.g. 'Onion', 'Other') = unspecified-grade reports; named ones (Nasik, Chips…) are graded varieties.")

df, model, accuracy, mae = get_model(crop, variety)
if len(df) < 40:
    st.warning("Not enough data for this variety to forecast reliably. Try 'All varieties'.")
    st.stop()

forecast = forecast_ahead(df, model, horizon)
last_price, future_price = df['price'].iloc[-1], forecast['price'].iloc[-1]
pct = (future_price - last_price) / last_price * 100
label = f"{crop} ({variety})"

m1, m2, m3 = st.columns(3)
m1.metric(f"Predicted in {horizon} days", f"₹{future_price:,.0f}/qtl", f"{pct:+.1f}%")
m2.metric("Model accuracy", f"{accuracy}%", help="Tested on recent weeks the model never saw")
m3.metric("Avg error", f"±₹{mae}/qtl")

if pct <= -8:
    st.error(f"⚠️ HIGH GLUT RISK: {label} predicted to fall {pct:.1f}% in {horizon} days. "
             "Oversupply → dumping/waste risk. Recommend staggered selling or cold storage.")
elif pct >= 8:
    st.warning(f"📈 PRICE SURGE: {label} predicted to rise {pct:.1f}% in {horizon} days. "
               "Favourable selling window — consider releasing stored stock.")
else:
    st.success(f"✅ STABLE MARKET: {label} expected within ±8% over {horizon} days. Low waste risk.")

hist = df[['date','price']].tail(120).rename(columns={'price':'Historical'})
bridge = pd.DataFrame([{'date': hist['date'].iloc[-1], 'Forecast': hist['Historical'].iloc[-1]}])
fc = pd.concat([bridge, forecast.rename(columns={'price':'Forecast'})], ignore_index=True)
st.line_chart(pd.merge(hist, fc, on='date', how='outer').sort_values('date').set_index('date'))

st.subheader("Climate context — price vs rainfall")
recent = df.tail(365)
fig, ax1 = plt.subplots(figsize=(10, 3.4))
ax1.plot(recent['date'], recent['price'], color='#2e8b57')
ax1.set_ylabel('Price (₹/qtl)', color='#2e8b57')
ax2 = ax1.twinx()
ax2.bar(recent['date'], recent['rainfall'], color='#3498db', alpha=0.35, width=2)
ax2.set_ylabel('Rainfall (mm)', color='#3498db')
fig.tight_layout()
st.pyplot(fig)

st.caption("Source: Govt. of India Agmarknet (Gujarat) + Open-Meteo composite climate (5 zones). "
           "Model: Random Forest fusing market history with rainfall & temperature signals.")
