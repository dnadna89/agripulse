import streamlit as st
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AgriPulse", page_icon="🌱", layout="wide")
CROPS  = ['Onion', 'Potato', 'Tomato']
FEATS  = ['price_lag1','price_lag7','price_roll7','dayofyear','month','rainfall','temp_mean']
BEST_H = {'Onion': 14, 'Potato': 21, 'Tomato': 30}   # each crop's most reliable horizon
MIN_DAYS = 150

@st.cache_resource
def load_everything():
    def load_weather(path):
        w = pd.read_csv(path, skiprows=3)
        def find(*k):
            for c in w.columns:
                if all(s in str(c).lower() for s in k): return c
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
            raw['modal']   = pd.to_numeric(raw['modal'].astype(str).str.replace(',', '', regex=False), errors='coerce')
            raw['variety'] = raw['variety'].astype(str).str.strip()
            parts.append(raw[(raw['date'].notna()) & (raw['modal']>0)
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
def get_model(crop, variety, h):
    base = daily_series(crop, variety).merge(WEATHER, on='date', how='inner').sort_values('date').reset_index(drop=True)
    base['price_lag1']  = base['price'].shift(1)
    base['price_lag7']  = base['price'].shift(7)
    base['price_roll7'] = base['price'].rolling(7).mean()
    base['dayofyear'], base['month'] = base['date'].dt.dayofyear, base['date'].dt.month
    base = base.dropna().reset_index(drop=True)
    d = base.copy(); d['target'] = d['price'].shift(-h); d = d.dropna().reset_index(drop=True)
    if len(d) < 60:
        return None
    X, y = d[FEATS], d['target']; split = int(len(d)*0.8)
    tm = RandomForestRegressor(n_estimators=300, random_state=42).fit(X.iloc[:max(1, split-h)], y.iloc[:max(1, split-h)])
    pred = tm.predict(X.iloc[split:]); actual = y.iloc[split:].values; today = d['price'].iloc[split:].values
    direction = (np.sign(pred-today) == np.sign(actual-today)).mean()*100
    sigma = float((actual-pred).std())
    pchg, achg = (pred-today)/today*100, (actual-today)/today*100; mask = pchg <= -8
    glut_n = int(mask.sum()); glut_prec = float((achg[mask] < 0).mean()*100) if glut_n else None
    final = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
    future = float(final.predict(base[FEATS].iloc[[-1]])[0])
    return dict(base=base, today=float(base['price'].iloc[-1]), last_date=base['date'].iloc[-1],
                future=future, direction=direction, sigma=sigma, glut_n=glut_n, glut_prec=glut_prec)

# ---------------- UI ----------------
st.title("🌱 AgriPulse")
st.caption("Climate-Aware Crop Price Intelligence — each crop forecast at its most reliable horizon")

c1, c2, c3 = st.columns(3)
with c1:
    crop = st.selectbox("Crop", CROPS)
with c2:
    variety = st.selectbox("Variety", ["All varieties"] + VARIETIES[crop])
with c3:
    h = st.slider("Forecast horizon (days)", 7, 30, BEST_H[crop], key=f"h_{crop}")

m = get_model(crop, variety, h)
if m is None:
    st.warning("Not enough data for this variety at this horizon. Try 'All varieties' or a shorter horizon.")
    st.stop()

today, future, direction, sigma = m['today'], m['future'], m['direction'], m['sigma']
pct = (future - today) / today * 100
conf = ("🟢 High confidence" if direction >= 65 else
        "🟡 Moderate confidence" if direction >= 55 else "🔴 Low confidence (volatile crop)")
st.caption(f"{conf} · direction accuracy {direction:.0f}% on unseen data (50% = chance)")

m1, m2, m3 = st.columns(3)
m1.metric(f"Predicted price in {h} days", f"₹{future:,.0f}/qtl", f"{pct:+.1f}%")
m2.metric("Direction accuracy", f"{direction:.0f}%", help="% of up/down calls correct on weeks the model never saw")
m3.metric("Glut-warning precision",
          f"{m['glut_prec']:.0f}%" if (m['glut_prec'] is not None and m['glut_n'] >= 5) else "—",
          help=f"{m['glut_n']} past glut warnings tested" if m['glut_n'] else "Too few to report")

label = f"{crop} ({variety})"
if pct <= -8:
    st.error(f"⚠️ GLUT RISK: {label} predicted to fall {pct:.1f}% in {h} days. "
             "Oversupply → dumping/waste risk. Recommend staggered selling or cold storage.")
elif pct >= 8:
    st.warning(f"📈 PRICE SURGE: {label} predicted to rise {pct:.1f}% in {h} days. "
               "Favourable selling window — consider releasing stored stock.")
else:
    st.success(f"✅ STABLE: {label} expected within ±8% over {h} days. Low waste risk.")

# Hero chart: history + forecast projection with widening confidence band
base = m['base']; last = m['last_date']
fdates = [last + pd.Timedelta(days=k) for k in range(0, h+1)]
fprice = [today + (future-today)*k/h for k in range(0, h+1)]
bw     = [1.28*sigma*(k/h) for k in range(0, h+1)]
hist = base.tail(90)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist['date'], hist['price'], color='#2e8b57', label='Actual price')
ax.plot(fdates, fprice, color='#e67e22', ls='--', label=f'{h}-day forecast')
ax.fill_between(fdates, [p-b for p,b in zip(fprice,bw)], [p+b for p,b in zip(fprice,bw)],
                color='#e67e22', alpha=0.2, label='80% confidence range')
ax.set_ylabel('Price (₹/qtl)'); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
st.pyplot(fig)

# Climate context
st.subheader("Climate context — price vs rainfall")
recent = base.tail(365)
fig2, axA = plt.subplots(figsize=(10, 3.2))
axA.plot(recent['date'], recent['price'], color='#2e8b57'); axA.set_ylabel('Price (₹/qtl)', color='#2e8b57')
axB = axA.twinx(); axB.bar(recent['date'], recent['rainfall'], color='#3498db', alpha=0.35, width=2)
axB.set_ylabel('Rainfall (mm)', color='#3498db'); fig2.tight_layout()
st.pyplot(fig2)

st.caption("Source: Govt. of India Agmarknet (Gujarat) + Open-Meteo composite climate (5 zones). "
           "Model: Random Forest, validated out-of-sample. Each crop defaults to its most reliable forecast horizon.")
