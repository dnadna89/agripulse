import streamlit as st
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="AgriPulse", page_icon="🌱", layout="wide")
CROPS    = ['Onion', 'Potato', 'Tomato']
FEATS    = ['price_lag1','price_lag7','price_roll7','dayofyear','month','rainfall','temp_mean']
BEST_H   = {'Onion': 14, 'Potato': 21, 'Tomato': 30}
DIRMODEL = {'Onion': 'reg', 'Potato': 'clf', 'Tomato': 'clf'}   # verified best direction engine per crop
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
    if len(d) < 60: return None
    X, y = d[FEATS], d['target']; today_arr = d['price'].values
    split = int(len(d)*0.8); cut = max(1, split-h)
    reg = RandomForestRegressor(n_estimators=300, random_state=42).fit(X.iloc[:cut], y.iloc[:cut])
    reg_pred = reg.predict(X.iloc[split:]); actual = y.iloc[split:].values; today_t = today_arr[split:]
    sigma = float((actual - reg_pred).std())
    yb = (y.values > today_arr).astype(int)
    if DIRMODEL[crop] == 'clf':
        clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X.iloc[:cut], yb[:cut])
        dir_test = clf.predict(X.iloc[split:])
    else:
        dir_test = (reg_pred > today_t).astype(int)
    yb_test = yb[split:]
    dir_acc = (dir_test == yb_test).mean()*100
    down = dir_test == 0; fell = actual < today_t
    glut_prec = float(fell[down].mean()*100) if down.sum() >= 5 else None
    glut_n = int(down.sum())
    reg_f = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
    future = float(reg_f.predict(base[FEATS].iloc[[-1]])[0])
    if DIRMODEL[crop] == 'clf':
        clf_f = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, yb)
        dir_now = int(clf_f.predict(base[FEATS].iloc[[-1]])[0])
    else:
        dir_now = int(future > base['price'].iloc[-1])
    return dict(base=base, today=float(base['price'].iloc[-1]), last_date=base['date'].iloc[-1],
                future=future, sigma=sigma, dir_acc=dir_acc, dir_now=dir_now,
                glut_prec=glut_prec, glut_n=glut_n, engine=DIRMODEL[crop])

# ---------------- UI ----------------
st.title("🌱 AgriPulse")
st.caption("Climate-Aware Crop Price Intelligence — direction forecasting, each crop at its best model & horizon")

c1, c2, c3 = st.columns(3)
with c1: crop = st.selectbox("Crop", CROPS)
with c2: variety = st.selectbox("Variety", ["All varieties"] + VARIETIES[crop])
with c3: h = st.slider("Forecast horizon (days)", 7, 30, BEST_H[crop], key=f"h_{crop}")

m = get_model(crop, variety, h)
if m is None:
    st.warning("Not enough data for this variety at this horizon. Try 'All varieties'."); st.stop()

today, future, dir_acc = m['today'], m['future'], m['dir_acc']
pct = (future - today) / today * 100
rising = m['dir_now'] == 1
conf = ("🟢 High confidence" if dir_acc >= 65 else "🟡 Moderate confidence" if dir_acc >= 55 else "🔴 Low confidence")
engine_name = "directional classifier" if m['engine'] == 'clf' else "regression model"
st.caption(f"{conf} · direction engine: {engine_name} · {dir_acc:.0f}% accurate on unseen data (50% = chance)")

m1, m2, m3 = st.columns(3)
m1.metric(f"Direction (next {h} days)", "📈 Rising" if rising else "📉 Falling")
m2.metric("Direction accuracy", f"{dir_acc:.0f}%", help="% of up/down calls correct on weeks the model never saw")
m3.metric(f"Est. price in {h} days", f"₹{future:,.0f}/qtl", f"{pct:+.1f}%", help="Illustrative level estimate (±band on chart)")

label = f"{crop} ({variety})"
if not rising:
    st.error(f"⚠️ GLUT RISK: {label} predicted to FALL over {h} days "
             f"(direction engine {dir_acc:.0f}% accurate). Oversupply → dumping/waste risk. "
             "Recommend staggered selling or cold storage.")
else:
    st.success(f"📈 FAVOURABLE: {label} predicted to RISE over {h} days. "
               "Good selling window — consider releasing stored stock.")

base = m['base']; last = m['last_date']; sigma = m['sigma']
fdates = [last + pd.Timedelta(days=k) for k in range(0, h+1)]
fprice = [today + (future-today)*k/h for k in range(0, h+1)]
bw     = [1.28*sigma*(k/h) for k in range(0, h+1)]
hist = base.tail(90)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist['date'], hist['price'], color='#2e8b57', label='Actual price')
ax.plot(fdates, fprice, color='#e67e22', ls='--', label='Estimated path')
ax.fill_between(fdates, [p-b for p,b in zip(fprice,bw)], [p+b for p,b in zip(fprice,bw)],
                color='#e67e22', alpha=0.2, label='Confidence range')
ax.set_ylabel('Price (₹/qtl)'); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
st.pyplot(fig)

st.subheader("Climate context — price vs rainfall")
recent = base.tail(365)
fig2, axA = plt.subplots(figsize=(10, 3.2))
axA.plot(recent['date'], recent['price'], color='#2e8b57'); axA.set_ylabel('Price (₹/qtl)', color='#2e8b57')
axB = axA.twinx(); axB.bar(recent['date'], recent['rainfall'], color='#3498db', alpha=0.35, width=2)
axB.set_ylabel('Rainfall (mm)', color='#3498db'); fig2.tight_layout()
st.pyplot(fig2)

gp = f"{m['glut_prec']:.0f}% of {m['glut_n']} 'fall' calls were correct" if m['glut_prec'] is not None else "—"
st.caption(f"Validated: {dir_acc:.0f}% direction accuracy · {gp} · engine: {engine_name}. "
           "Source: Agmarknet (Gujarat) + Open-Meteo composite climate. Each crop uses its best-performing model & horizon.")
