import streamlit as st
import pandas as pd
import numpy as np
import glob
import altair as alt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="AgriPulse", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family:'Inter', sans-serif; }
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:2rem; padding-bottom:3rem; max-width:1060px;}
h1 {font-weight:600; letter-spacing:-0.02em; color:#1c1c1c; margin-bottom:0;}
section[data-testid="stSidebar"] {background:#fafafa; border-right:1px solid #eee;}
</style>
""", unsafe_allow_html=True)

CROPS    = ['Onion', 'Potato', 'Tomato']
FEATS    = ['price_lag1','price_lag7','price_roll7','dayofyear','month','rainfall','temp_mean']
BEST_H   = {'Onion': 14, 'Potato': 21, 'Tomato': 30}
DIRMODEL = {'Onion': 'reg', 'Potato': 'clf', 'Tomato': 'clf'}
MIN_DAYS = 150
GREEN, ORANGE, BLUE = '#2f6b4f', '#c0722e', '#a9c4d6'

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

def stat_card(label, value, sub=""):
    return (f'<div style="flex:1;background:#f7f7f5;border:1px solid #ececec;border-radius:12px;padding:18px 20px;">'
            f'<div style="color:#8a8a8a;font-size:0.72rem;font-weight:500;letter-spacing:0.04em;text-transform:uppercase;">{label}</div>'
            f'<div style="color:#1c1c1c;font-size:1.5rem;font-weight:600;margin-top:8px;">{value}</div>'
            f'<div style="color:#a5a5a5;font-size:0.78rem;margin-top:3px;">{sub}</div></div>')

def banner(text, kind):
    bg, bar, fg = (('#fbf4e8','#c0722e','#8a5418') if kind=='risk' else ('#eef4f0','#2f6b4f','#2a5742'))
    st.markdown(f'<div style="background:{bg};border-left:3px solid {bar};padding:13px 18px;'
                f'border-radius:6px;color:{fg};font-size:0.92rem;margin:4px 0 8px;">{text}</div>',
                unsafe_allow_html=True)

# ---- Sidebar controls ----
with st.sidebar:
    st.markdown("#### AgriPulse")
    st.caption("Crop price intelligence")
    st.write("")
    crop = st.selectbox("Crop", CROPS)
    variety = st.selectbox("Variety", ["All varieties"] + VARIETIES[crop])
    h = st.slider("Forecast horizon (days)", 7, 30, BEST_H[crop], key=f"h_{crop}")
    st.write("")
    st.caption("Source: Agmarknet (Gujarat) and Open-Meteo climate. Each crop uses its best-performing model and horizon.")

# ---- Main ----
st.title("AgriPulse")
st.markdown('<p style="color:#888;font-size:1rem;margin-top:2px;">Climate-aware crop price direction forecasting</p>',
            unsafe_allow_html=True)

m = get_model(crop, variety, h)
if m is None:
    st.warning("Not enough data for this variety at this horizon. Try 'All varieties'.")
    st.stop()

today, future, dir_acc = m['today'], m['future'], m['dir_acc']
pct = (future - today) / today * 100
rising = m['dir_now'] == 1
conf = "High" if dir_acc >= 65 else "Moderate" if dir_acc >= 55 else "Low"
engine_name = "directional classifier" if m['engine'] == 'clf' else "regression model"
st.markdown(f'<p style="color:#999;font-size:0.85rem;margin-top:6px;">{conf} confidence · '
            f'{engine_name}, {dir_acc:.0f}% accurate on unseen data</p>', unsafe_allow_html=True)

dir_color = GREEN if rising else ORANGE
arrow = "&#9650;" if rising else "&#9660;"
dir_val = f'<span style="color:{dir_color}">{arrow} {"Rising" if rising else "Falling"}</span>'
cards = (stat_card(f"Direction · next {h}d", dir_val)
         + stat_card("Direction accuracy", f"{dir_acc:.0f}%", "on unseen data")
         + stat_card(f"Est. price in {h}d", f"&#8377;{future:,.0f}", f"{pct:+.1f}% from today"))
st.markdown(f'<div style="display:flex;gap:14px;margin:8px 0 18px;">{cards}</div>', unsafe_allow_html=True)

if not rising:
    banner(f"Glut risk. {crop} ({variety}) is predicted to fall over the next {h} days. "
           f"Likely oversupply - consider staggered selling or cold storage.", 'risk')
else:
    banner(f"Favourable window. {crop} ({variety}) is predicted to rise over the next {h} days. "
           f"Consider releasing stored stock.", 'good')

# ---- Forecast chart ----
base = m['base']; last = m['last_date']; sigma = m['sigma']
fdates = [last + pd.Timedelta(days=k) for k in range(0, h+1)]
fprice = [today + (future-today)*k/h for k in range(0, h+1)]
bw     = [1.28*sigma*(k/h) for k in range(0, h+1)]
hist90 = base.tail(90)[['date','price']].copy()
fcdf = pd.DataFrame({'date': fdates, 'price': fprice,
                     'lo': [p-b for p,b in zip(fprice,bw)], 'hi': [p+b for p,b in zip(fprice,bw)]})
ax_x = alt.Axis(title=None, format='%b %Y', labelColor='#999', tickColor='#eee', domainColor='#e5e5e5', grid=False)
ax_y = alt.Axis(title='\u20b9 / quintal', labelColor='#999', titleColor='#999', gridColor='#f2f2f2', domainColor='#e5e5e5', tickColor='#eee')
band = alt.Chart(fcdf).mark_area(color=ORANGE, opacity=0.13).encode(
    x=alt.X('date:T', axis=ax_x), y=alt.Y('lo:Q', axis=ax_y), y2='hi:Q')
l_act = alt.Chart(hist90).mark_line(color=GREEN, strokeWidth=2).encode(
    x=alt.X('date:T', axis=ax_x), y=alt.Y('price:Q', axis=ax_y),
    tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('price:Q', title='Rs/qtl', format=',.0f')])
l_fc = alt.Chart(fcdf).mark_line(color=ORANGE, strokeWidth=2, strokeDash=[5,4]).encode(
    x='date:T', y='price:Q',
    tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('price:Q', title='Est Rs/qtl', format=',.0f')])
st.altair_chart((band + l_act + l_fc).properties(height=330).configure_view(strokeWidth=0),
                use_container_width=True)

# ---- Climate context ----
st.markdown('<h3 style="font-weight:500;color:#444;margin-top:22px;margin-bottom:0;">Climate context</h3>'
            '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Price against rainfall, last 12 months</p>',
            unsafe_allow_html=True)
clim = base.tail(365)[['date','price','rainfall']].copy()
rain = alt.Chart(clim).mark_bar(color=BLUE, opacity=0.6).encode(
    x=alt.X('date:T', axis=ax_x),
    y=alt.Y('rainfall:Q', axis=alt.Axis(title='rain (mm)', labelColor='#bbb', titleColor='#bbb', grid=False)))
pr = alt.Chart(clim).mark_line(color=GREEN, strokeWidth=1.5).encode(
    x='date:T', y=alt.Y('price:Q', axis=alt.Axis(title='\u20b9 / quintal', labelColor='#999', titleColor='#999', gridColor='#f2f2f2')))
st.altair_chart(alt.layer(rain, pr).resolve_scale(y='independent').properties(height=230).configure_view(strokeWidth=0),
                use_container_width=True)

gp = f"{m['glut_prec']:.0f}% of {m['glut_n']} fall-calls correct" if m['glut_prec'] is not None else "-"
st.markdown(f'<p style="color:#b0b0b0;font-size:0.78rem;margin-top:14px;">'
            f'Validated: {dir_acc:.0f}% direction accuracy · {gp} · {engine_name}. '
            f'Data: Agmarknet (Gujarat) and Open-Meteo composite climate.</p>', unsafe_allow_html=True)
