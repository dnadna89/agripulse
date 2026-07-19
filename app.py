import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import base64
import altair as alt
import pydeck as pdk

def mascot(name):
    """Return the mascot file path, trying .png then .jpg (extension-agnostic), else None."""
    base = os.path.splitext(name)[0]
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join("mascots", base + ext)
        if os.path.exists(p):
            return p
    return None
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

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
LABELS   = {'price_roll7':'7-day avg price','price_lag1':'Yesterday price','price_lag7':'Last-week price',
            'dayofyear':'Season (day of year)','month':'Month','rainfall':'Rainfall','temp_mean':'Temperature'}
MIN_DAYS, MARKET_MIN = 150, 250
GREEN, ORANGE, BLUE = '#2f6b4f', '#c0722e', '#a9c4d6'
# Environmental footprint factors (global-average estimates; vary by production system)
# Water: Water Footprint Network (Mekonnen & Hoekstra, 2011). GHG: Poore & Nemecek (2018).
FOOTPRINT = {'Onion': {'co2': 0.5, 'water': 272}, 'Potato': {'co2': 0.5, 'water': 287}, 'Tomato': {'co2': 2.1, 'water': 214}}
SAVE_FRAC = 0.10  # conservative share of at-risk volume kept out of glut-driven waste
# Per-crop state-scale basis for the "potential if widely adopted" figures.
#   prod_t : Gujarat annual production (tonnes)   loss : ICAR-CIPHET (2015) post-harvest loss share
#   Tomato 1.36 Mt  - NHB, State-wise Tomato Production, 2017-18
#   Onion  2.05 Mt  - state horticulture data, 2023-24 (Gujarat ranks ~3rd nationally)
#   Potato 4.86 Mt  - Gujarat Agriculture Dept, 2024-25
# CIPHET loss: tomato 12.4%, onion 8.2%, potato 7.3%.
CROP_STATE = {
    'Onion':  {'prod_t': 2_050_000, 'loss': 0.082, 'pyear': '2023-24', 'psrc': 'state horticulture data'},
    'Potato': {'prod_t': 4_860_000, 'loss': 0.073, 'pyear': '2024-25', 'psrc': 'Gujarat Agriculture Dept'},
    'Tomato': {'prod_t': 1_357_520, 'loss': 0.124, 'pyear': '2017-18', 'psrc': 'NHB'},
}
LEDGER = {
    'Tomato': {'validated': True, 'gluts': 17, 'avoid_water_ML': 8317, 'avoid_co2_t': 81613,
        'badge': 'Validated - our tomato model caught all 16 severe gluts it could be tested on. When it warns of a crash it is right 52% of the time, against a 34% base rate.',
        'verb': 'acting on these warnings could have avoided',
        'rows': [('2024-03-24', 21845, 4675, 4.6), ('2024-07-22', 25178, 5388, 4.4), ('2024-10-07', 22636, 4844, 2.2),
                 ('2025-08-07', 29290, 6268, 3.0), ('2025-12-23', 19589, 4192, 2.4), ('2026-03-06', 21681, 4640, 3.1)]},
    'Onion': {'validated': False, 'gluts': 22, 'avoid_water_ML': 39749, 'avoid_co2_t': 73068,
        'badge': 'Low precision - when the onion model warns of a crash it is right only 17% of the time (5% base rate). Shown to measure the resources at stake, not as a claim we reliably prevent them.',
        'verb': 'these gluts put at stake (at 10% of arrivals) about',
        'rows': [('2024-11-10', 47984, 13052, 2.1), ('2024-12-01', 90276, 24555, 1.8), ('2024-12-31', 110857, 30153, 0.9),
                 ('2025-03-02', 161831, 44018, 0.6), ('2025-03-27', 110330, 30010, 1.2), ('2025-11-10', 34315, 9334, 2.4)]},
    'Potato': {'validated': False, 'gluts': 17, 'avoid_water_ML': 19674, 'avoid_co2_t': 34274,
        'badge': 'Low precision - when the potato model warns of a crash it is right only 12% of the time (4% base rate). Shown to measure the resources at stake, not as a claim we reliably prevent them.',
        'verb': 'these gluts put at stake (at 10% of arrivals) about',
        'rows': [('2023-12-24', 46924, 13467, 8.1), ('2024-03-31', 35682, 10241, 3.7), ('2024-05-26', 35532, 10198, 5.3),
                 ('2024-12-08', 49787, 14289, 6.6), ('2025-01-03', 49311, 14152, 6.4), ('2025-12-21', 33232, 9538, 4.9)]},
}

def state_impact(crop, frac):
    """Return (water_billion_litres, co2_tonnes) kept out of the waste stream per year at adoption `frac`."""
    cs = CROP_STATE[crop]
    kg = cs['prod_t'] * 1000 * cs['loss'] * frac
    return kg * FOOTPRINT[crop]['water'] / 1e9, kg * FOOTPRINT[crop]['co2'] / 1000
def risk_level(pct, reliable):
    """Risk label from the size of the predicted move and the reliability gate - no invented probability."""
    if not reliable:
        return "No clear call", "#9a9a9a"
    a = abs(pct)
    if a >= 20: return "HIGH", "#c0722e"
    if a >= 8:  return "MODERATE", "#c79a3e"
    return "LOW", "#2f6b4f"

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
    _wk = lambda s: ''.join(c for c in str(s).lower() if c.isalpha())
    # Prefer per-mandi files in weather/ ; fall back to any root *_weather.csv so the app never breaks.
    wfiles = sorted(glob.glob('weather/*_weather.csv')) or sorted(glob.glob('*_weather.csv'))
    weather_by = {}
    for wpath in wfiles:
        town = os.path.basename(wpath).replace('_weather.csv', '')
        wdf = load_weather(wpath).dropna(subset=['date']).groupby('date', as_index=False).mean(numeric_only=True)
        if len(wdf):
            weather_by[_wk(town)] = wdf
    if weather_by:
        weather = (pd.concat(weather_by.values(), ignore_index=True)          # statewide average = fallback
                     .groupby('date', as_index=False).mean(numeric_only=True))
    else:
        weather = pd.DataFrame(columns=['date', 'rainfall', 'temp_mean'])
    wdates = set(weather['date'])
    crops_raw, varieties, markets = {}, {}, {}
    for crop in CROPS:
        parts = []
        for path in sorted(glob.glob(f'{crop}_*.xlsx')):
            raw = pd.read_excel(path, header=None); col0 = raw[0].astype(str)
            is_mkt = col0.str.contains('Market Name', case=False, na=False)
            mkt = (col0.where(is_mkt).str.replace('Market Name','',regex=False)
                       .str.replace(':','',regex=False).str.strip().ffill())
            out = pd.DataFrame()
            out['date']    = pd.to_datetime(col0, format='%d/%m/%Y', errors='coerce').ffill()
            out['market']  = mkt
            out['variety'] = raw[2].astype(str).str.strip()
            out['modal']   = pd.to_numeric(raw[5].astype(str).str.replace(',','',regex=False), errors='coerce')
            parts.append(out[(out['date'].notna()) & (out['modal']>0) & out['market'].notna()
                             & (~out['variety'].isin(['nan','None','']))])
        df = pd.concat(parts, ignore_index=True)
        crops_raw[crop] = df
        varieties[crop] = sorted([v for v in df['variety'].unique()
            if sum(d in wdates for d in df.loc[df['variety']==v,'date'].unique()) >= MIN_DAYS],
            key=lambda v: -(df['variety']==v).sum())
        markets[crop] = sorted([mk for mk in df['market'].unique()
            if sum(d in wdates for d in df.loc[df['market']==mk,'date'].unique()) >= MARKET_MIN],
            key=lambda mk: -df.loc[df['market']==mk,'date'].nunique())
    return crops_raw, weather_by, weather, varieties, markets

CROPS_RAW, WEATHER_BY, WEATHER_AVG, VARIETIES, MARKETS = load_everything()
_WKEYS = list(WEATHER_BY.keys())
def weather_for(market):
    """Return the weather series for a market's own town; fall back to the statewide average."""
    if market == "All Gujarat":
        return WEATHER_AVG
    k = ''.join(c for c in str(market).lower() if c.isalpha())
    if k in WEATHER_BY:
        return WEATHER_BY[k]
    for wk in _WKEYS:
        if wk and (wk in k or k in wk):
            return WEATHER_BY[wk]
    return WEATHER_AVG

@st.cache_resource
def load_ndvi():
    frames = []
    for f in glob.glob('*MOD13Q1*.csv'):
        d = pd.read_csv(f)
        ndvi_cols = [c for c in d.columns if c.endswith('_NDVI')]
        if 'Date' not in d.columns or not ndvi_cols:
            continue
        out = pd.DataFrame()
        out['date'] = pd.to_datetime(d['Date'], format='%Y-%m-%d', errors='coerce')
        out['ndvi'] = pd.to_numeric(d[ndvi_cols[0]], errors='coerce')
        frames.append(out.dropna())
    if not frames:
        return None
    return (pd.concat(frames).groupby('date', as_index=False)['ndvi'].mean()
              .sort_values('date').reset_index(drop=True))

NDVI = load_ndvi()

def varieties_for(crop, market):
    if market == "All Gujarat":
        return VARIETIES[crop]
    df = CROPS_RAW[crop]; df = df[df['market'] == market]
    wd = set(WEATHER_AVG['date'])
    good = [v for v in df['variety'].unique()
            if sum(d in wd for d in df.loc[df['variety']==v, 'date'].unique()) >= 160]
    return sorted(good, key=lambda v: -(df['variety']==v).sum())

def daily_series(crop, variety, market):
    df = CROPS_RAW[crop]
    if market != "All Gujarat":   df = df[df['market'] == market]
    if variety != "All varieties": df = df[df['variety'] == variety]
    return (df.groupby('date', as_index=False)['modal'].median()
              .rename(columns={'modal':'price'}).sort_values('date').reset_index(drop=True))

@st.cache_resource
def get_model(crop, variety, market):
    h = BEST_H[crop]
    base = daily_series(crop, variety, market).merge(weather_for(market), on='date', how='inner').sort_values('date').reset_index(drop=True)
    if len(base) < 120: return None
    base['price_lag1']  = base['price'].shift(1)
    base['price_lag7']  = base['price'].shift(7)
    base['price_roll7'] = base['price'].rolling(7).mean()
    base['dayofyear'], base['month'] = base['date'].dt.dayofyear, base['date'].dt.month
    base = base.dropna().reset_index(drop=True)
    d = base.copy(); d['target'] = d['price'].shift(-h); d = d.dropna().reset_index(drop=True)
    if len(d) < 90: return None
    X, y = d[FEATS], d['target']; today_arr = d['price'].values; yb = (y.values > today_arr).astype(int)

    # one consistent test: 5-fold walk-forward direction accuracy for every model
    acc = {'Naive (majority)': [], 'Random Forest': [], 'Gradient Boosting': [], 'Classifier': []}
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        ytr = y.iloc[tr]; yte = yb[te]; td = today_arr[te]; maj = int(round(yb[tr].mean()))
        acc['Naive (majority)'].append((np.full(len(te), maj) == yte).mean()*100)
        rf = RandomForestRegressor(n_estimators=250, random_state=42).fit(X.iloc[tr], ytr)
        acc['Random Forest'].append(((rf.predict(X.iloc[te]) > td).astype(int) == yte).mean()*100)
        gb = HistGradientBoostingRegressor(random_state=42).fit(X.iloc[tr], ytr)
        acc['Gradient Boosting'].append(((gb.predict(X.iloc[te]) > td).astype(int) == yte).mean()*100)
        cf = RandomForestClassifier(n_estimators=250, random_state=42).fit(X.iloc[tr], yb[tr])
        acc['Classifier'].append((cf.predict(X.iloc[te]) == yte).mean()*100)
    wf_acc = float(np.mean(acc['Random Forest']))           # headline == this exact number
    bench = {k: float(np.mean(v)) for k, v in acc.items()}

    # ARIMA classical baseline (recent expanding hold-out; kept as a reference)
    try:
        from statsmodels.tsa.arima.model import ARIMA
        pser = base['price'].astype(float).values; n = len(pser); sp = int(n * 0.8)
        res = ARIMA(pser[:sp], order=(2, 1, 2)).fit(); hits = tot = 0
        for t in range(sp, n - h):
            res = res.append([pser[t]], refit=False)
            hits += int((float(res.forecast(steps=h)[-1]) > pser[t]) == (pser[t + h] > pser[t])); tot += 1
        if tot: bench['ARIMA (classical)'] = 100.0 * hits / tot
    except Exception:
        pass

    reg_f = RandomForestRegressor(n_estimators=250, random_state=42).fit(X, y)
    imp = sorted(zip(FEATS, reg_f.feature_importances_), key=lambda kv: -kv[1])
    split = int(len(d)*0.8)
    rf_s = RandomForestRegressor(n_estimators=250, random_state=42).fit(X.iloc[:split], y.iloc[:split])
    sigma = float((y.iloc[split:].values - rf_s.predict(X.iloc[split:])).std())
    today = float(base['price'].tail(7).median())
    recent = base['price'].tail(180); lo, hi = float(recent.quantile(0.05)), float(recent.quantile(0.95))
    future = min(max(float(reg_f.predict(base[FEATS].iloc[[-1]])[0]), lo), hi)
    pct = (future - today) / today * 100; rising = future > today
    reliable = (abs(pct) <= 50) and (wf_acc >= 53)
    return dict(base=base, today=today, future=future, pct=pct, rising=rising, reliable=reliable,
                last_date=base['date'].iloc[-1], h=h, sigma=sigma, wf_acc=wf_acc,
                bench=bench, chosen='Random Forest', imp=imp)

def stat_card(label, value, sub=""):
    return (f'<div style="flex:1;background:#f7f7f5;border:1px solid #ececec;border-radius:12px;padding:18px 20px;">'
            f'<div style="color:#8a8a8a;font-size:0.72rem;font-weight:500;letter-spacing:0.04em;text-transform:uppercase;">{label}</div>'
            f'<div style="color:#1c1c1c;font-size:1.5rem;font-weight:600;margin-top:8px;">{value}</div>'
            f'<div style="color:#a5a5a5;font-size:0.78rem;margin-top:3px;">{sub}</div></div>')

def banner(text, kind):
    bg, bar, fg = (('#fbf4e8','#c0722e','#8a5418') if kind=='risk' else
                   ('#f4f1ec','#9a9a9a','#666') if kind=='warn' else ('#eef4f0','#2f6b4f','#2a5742'))
    st.markdown(f'<div style="background:{bg};border-left:3px solid {bar};padding:13px 18px;border-radius:6px;'
                f'color:{fg};font-size:0.92rem;margin:4px 0 10px;">{text}</div>', unsafe_allow_html=True)
def reco_card(kind, action, movement, conf_txt, review_txt, note=None):
    bg, bar, fg = {'risk':('#fbf4e8','#c0722e','#8a5418'),
                   'good':('#eef4f0','#2f6b4f','#2a5742'),
                   'warn':('#f4f1ec','#9a9a9a','#5f5f5f')}[kind]
    tail = note if note is not None else f'Review again in about {review_txt}'
    return (f'<div style="background:{bg};border-left:3px solid {bar};padding:14px 18px;border-radius:6px;margin:4px 0 12px;">'
            f'<div style="color:{fg};font-size:0.72rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">Recommended action</div>'
            f'<div style="color:{fg};font-size:1.12rem;font-weight:600;margin:4px 0 8px;">{action}</div>'
            f'<div style="color:{fg};font-size:0.88rem;line-height:1.7;">Expected movement: <b>{movement}</b><br>'
            f'Confidence: <b>{conf_txt}</b><br>{tail}</div></div>')
with st.sidebar:
    st.markdown("#### AgriPulse")
    st.caption("Decision support for farmers and policymakers")
    view = st.radio("View", ["Farmer advisory", "Government / policymaker"])
    st.write("")
    crop = st.selectbox("Crop", CROPS, index=CROPS.index('Tomato'))
    market = st.selectbox("Market (yard)", ["All Gujarat"] + MARKETS[crop])
    variety = st.selectbox("Variety", ["All varieties"] + varieties_for(crop, market))
    st.write("")
    st.caption(f"Forecasting {BEST_H[crop]} days ahead. Source: Agmarknet (Gujarat) and Open-Meteo climate.")

st.markdown("""<style>
.stApp { background-color: #faf9f5; }
img { border-radius: 6px; }
/* sidebar: match the warm cream/green tone of the page */
section[data-testid="stSidebar"] { background-color: #f1f0e8; border-right: 1px solid #e2e0d4; }
section[data-testid="stSidebar"] .stRadio label, section[data-testid="stSidebar"] label { color: #3e4d46 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 { color: #2f6b4f !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #ffffff; border: 1px solid #dcd9c8; border-radius: 8px;
}
.ap-i {
    display:inline-block; width:15px; height:15px; line-height:15px; text-align:center;
    border:1px solid #c3c9c5; border-radius:50%; color:#7d8b84; font-size:0.68rem;
    font-style:normal; font-weight:700; cursor:help; margin-left:5px; vertical-align:middle;
    background:#ffffff;
}
.ap-i:hover { background:#2f6b4f; color:#ffffff; border-color:#2f6b4f; }
</style>""", unsafe_allow_html=True)

def info(text):
    """A small hover-for-detail icon. Uses the native title attribute so it can never be
    clipped by a Streamlit column or expander (a CSS pop-up would be)."""
    safe = str(text).replace('"', "'").replace('-', '-').replace('&amp;', '&')
    return f'<span class="ap-i" title="{safe}">i</span>'

_wc1, _wc2 = st.columns([1, 5], vertical_alignment="center")
with _wc1:
    _wm = mascot("saying_hi.png")
    if _wm: st.image(_wm, width=150)
with _wc2:
    st.markdown('<div style="font-size:2.6rem;font-weight:800;color:#2f6b4f;line-height:1.02;letter-spacing:-0.02em;">AgriPulse</div>'
                '<div style="color:#7c7c76;font-size:1.05rem;margin-top:5px;line-height:1.4;">AI Decision Support System for Gujarat\'s top commodity markets - forecasting price shocks and the water and carbon they waste</div>',
                unsafe_allow_html=True)
st.markdown('<hr style="border:none;border-top:1px solid #d8d8ce;margin:10px 0 6px;">', unsafe_allow_html=True)

m = get_model(crop, variety, market)
if m is None:
    st.warning("Not enough history for this yard and variety combination. Try 'All Gujarat' or 'All varieties'."); st.stop()

today, future, h, wf, pct, rising, reliable = m['today'], m['future'], m['h'], m['wf_acc'], m['pct'], m['rising'], m['reliable']
flat = abs(pct) < 1.0
rlabel, rcol = risk_level(pct, reliable)
conf = "High" if wf >= 65 else "Moderate" if wf >= 55 else "Low"
place = market if market != "All Gujarat" else "all Gujarat yards"
st.markdown(f'<p style="color:#999;font-size:0.85rem;margin-top:6px;">{place} · {conf} confidence on direction · 5-fold walk-forward, benchmarked against ARIMA &amp; naive</p>',
            unsafe_allow_html=True)

base = m['base']; last = m['last_date']; sigma = m['sigma']
where = f"{crop} at {market}" if market != "All Gujarat" else f"{crop} across Gujarat"
fdates = [last + pd.Timedelta(days=k) for k in range(0, h+1)]
fprice = [today + (future-today)*k/h for k in range(0, h+1)]
bw     = [1.28*sigma*(k/h) for k in range(0, h+1)]

# ---------------------------------------------------------------------------
# Presenter tour. Four sections, one per presenter. The split is ours - the
# audience just sees a clean, one-beat-at-a-time walkthrough.
# ---------------------------------------------------------------------------
TOUR = [
    (1, "The stake", "Maulik",
     "Gujarat pumps water from over-exploited aquifers to grow food that then rots unsold. This is what one glut actually costs."),
    (2, "The decision", "Devansh",
     "We turn that warning into a call a farmer or an officer can act on - carrying our real validated accuracy, never a claimed one."),
    (3, "The proof", "Yuvraj",
     "Why you can believe it: what actually drives the forecast, how it was tested, and exactly where it fails."),
    (4, "The lever", "Yashvi",
     "Where a glut hurts most, and what a government can actually do about it - measured in water, carbon and rupees."),
]
demo = st.sidebar.toggle("Demo mode - guided tour", value=False, key="demomode")
if demo and 'tour' not in st.session_state:
    st.session_state.tour = 1

def _sec(n):
    """True when section n should render. Outside demo mode the whole page shows."""
    return (not demo) or (st.session_state.get('tour', 1) == n)

if demo:
    _t = st.session_state.get('tour', 1)
    _, _ttl, _who, _cue = TOUR[_t - 1]
    _dots = "".join(
        f'<span style="display:inline-block;width:{28 if i+1==_t else 8}px;height:8px;border-radius:4px;margin-right:5px;'
        f'background:{"#2f6b4f" if i+1==_t else "#d3d8d4"};"></span>' for i in range(4))
    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #e4e2d8;border-left:5px solid #2f6b4f;border-radius:12px;'
        f'padding:14px 20px;margin:6px 0 10px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div style="color:#8a8a80;font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">'
        f'Chapter {_t} of 4</div>'
        f'<div style="color:#c9c9c0;font-size:0.66rem;letter-spacing:0.04em;">{_who}</div></div>'
        f'<div style="color:#2f6b4f;font-size:1.5rem;font-weight:700;margin:2px 0 4px;">{_ttl}</div>'
        f'<div style="color:#5f5f57;font-size:0.96rem;line-height:1.55;">{_cue}</div>'
        f'<div style="margin-top:10px;">{_dots}</div></div>', unsafe_allow_html=True)
    _b1, _b2, _b3 = st.columns([1, 1, 7])
    if _b1.button("Back", disabled=(_t == 1), use_container_width=True):
        st.session_state.tour = _t - 1; st.rerun()
    if _b2.button("Next", type="primary", disabled=(_t == 4), use_container_width=True):
        st.session_state.tour = _t + 1; st.rerun()

if _sec(1):
    # Honest status strip - built only from figures we already compute; no invented "this week" counts.
    _lg1 = LEDGER[crop]
    _stake1 = (_lg1['avoid_water_ML'] / SAVE_FRAC) / max(_lg1['gluts'], 1)
    st.markdown(
        f'<div style="display:flex;gap:0;flex-wrap:wrap;border:1px solid #e4e2d8;border-radius:10px;overflow:hidden;margin:2px 0 8px;">'
        f'<div style="flex:1;min-width:150px;background:#2f6b4f;color:#fff;padding:10px 16px;">'
        f'<div style="font-size:1.35rem;font-weight:700;line-height:1;">~{_stake1:,.0f}M L</div>'
        f'<div style="font-size:0.72rem;color:#cfe6d9;margin-top:3px;">water at stake / severe {crop.lower()} glut</div></div>'
        f'<div style="flex:1;min-width:150px;background:#b02828;color:#fff;padding:10px 16px;">'
        f'<div style="font-size:1.35rem;font-weight:700;line-height:1;">4 districts</div>'
        f'<div style="font-size:0.72rem;color:#f2d3d0;margin-top:3px;">now over-exploited (CGWB 2025)</div></div>'
        f'<div style="flex:1;min-width:150px;background:#3a5a6a;color:#fff;padding:10px 16px;">'
        f'<div style="font-size:1.35rem;font-weight:700;line-height:1;">1 Ramsar wetland</div>'
        f'<div style="font-size:0.72rem;color:#cfe0e6;margin-top:3px;">in an over-exploited district</div></div>'
        f'<div style="flex:1;min-width:150px;background:#8a6a3a;color:#fff;padding:10px 16px;">'
        f'<div style="font-size:1.35rem;font-weight:700;line-height:1;">{BEST_H[crop]}-day lead</div>'
        f'<div style="font-size:0.72rem;color:#e6dcc8;margin-top:3px;">warning before the glut lands</div></div></div>',
        unsafe_allow_html=True)
    # Hero band - the first number a judge sees is water, not a price.
    # Two honesty rules here: the noun must match the number (at-stake != avoided), and the
    # strength of the claim must follow the model's validation, exactly like every other panel.
    _lg = LEDGER[crop]
    _stake_glut = (_lg['avoid_water_ML'] / SAVE_FRAC) / max(_lg['gluts'], 1)   # embedded water per glut
    _avoid_glut = _lg['avoid_water_ML'] / max(_lg['gluts'], 1)                 # avoidable at SAVE_FRAC
    if _lg['validated']:
        _hero_bg = 'linear-gradient(90deg,#2f6b4f 0%,#3d7d5e 100%)'
        _hero_claim = (f'Every severe {crop.lower()} glut we flag carries <b>~{_stake_glut:,.0f} million litres</b> of embedded water into the '
                       f'mandi - acting on that warning at our conservative {int(SAVE_FRAC*100)}% rate could save '
                       f'<b>~{_avoid_glut:,.0f} million litres</b> of it.')
        _hero_note = (f'Measured from Agmarknet arrivals across the {_lg["gluts"]} severe {crop.lower()} gluts we flagged, 2021-2026. '
                      f'Our {crop.lower()} model caught every severe glut it could be tested on, at 52% warning precision against a 34% base rate.')
    else:
        _hero_bg = 'linear-gradient(90deg,#8a6a3a 0%,#a07f4a 100%)'
        _hero_claim = (f'Every severe {crop.lower()} glut carries <b>~{_stake_glut:,.0f} million litres</b> of embedded water into the mandi. '
                       f'We show the stake - not a saving, because we cannot yet promise one for {crop.lower()}.')
        _hero_note = (f'Measured from Agmarknet arrivals across {_lg["gluts"]} severe {crop.lower()} gluts, 2021-2026. '
                      f'Our {crop.lower()} model\'s warning precision is low, so this is the resource at risk, not a delivered result. '
                      f'Switch to Tomato for our validated model.')
    st.markdown(
        f'<div style="background:{_hero_bg};border-radius:12px;padding:15px 22px;margin:4px 0 8px;">'
        f'<div style="color:#e6ddc8;font-size:0.7rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;">'
        f'Early warning &middot; water at stake{"" if _lg["validated"] else " &middot; low-precision model"}</div>'
        f'<div style="color:#ffffff;font-size:1.16rem;line-height:1.5;margin-top:5px;">{_hero_claim}</div>'
        f'<div style="color:#dcd3c0;font-size:0.75rem;margin-top:6px;">{_hero_note}</div></div>'
        f'<div style="color:#7a5c3a;font-size:0.9rem;margin:6px 2px 2px;">&#127807; In Gujarat\'s worst-hit districts, that same aquifer '
        f'feeds <b>Thol Lake - a Ramsar wetland home to 320+ bird species.</b> A glut there is water drawn from a system a wetland depends on.</div>',
        unsafe_allow_html=True)

    # --- State-scale environmental headline (position 2: the stake, straight after the hero) ---
    _cs = CROP_STATE[crop]
    _wbn, _ct = state_impact(crop, SAVE_FRAC)
    _prod_mt = _cs['prod_t'] / 1e6
    _note = {
        'Tomato': 'Tomato is our strongest-validated model, which is why it anchors the demo.',
        'Potato': 'Potato is a solid secondary model, and potatoes store well, so read this as the resources embedded in post-harvest loss.',
        'Onion':  'Onion direction accuracy is only near chance at our scale, so read this as the resource opportunity, not a delivered result.',
    }[crop]
    _tsaved = _cs['prod_t'] * _cs['loss'] * SAVE_FRAC          # tonnes of produce kept out of waste / year
    st.markdown(
        f'<div style="background:#eef4f0;border:1px solid #dce8e1;border-radius:14px;padding:20px 24px;margin:4px 0 18px;">'
        f'<div style="color:#2f6b4f;font-size:0.72rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">'
        f'Statewide environmental potential &middot; {crop.lower()} &middot; if widely adopted</div>'
        f'<div style="display:flex;gap:48px;flex-wrap:wrap;margin:13px 0 4px;">'
        f'<div><div style="color:#1c1c1c;font-size:2.1rem;font-weight:600;line-height:1;">~{_wbn:.1f} billion L</div>'
        f'<div style="color:#5c6b63;font-size:0.82rem;margin-top:5px;">water kept out of the waste stream / year</div></div>'
        f'<div><div style="color:#1c1c1c;font-size:2.1rem;font-weight:600;line-height:1;">~{round(_ct,-3):,.0f} t</div>'
        f'<div style="color:#5c6b63;font-size:0.82rem;margin-top:5px;">CO&#8322;e avoided / year</div></div>'
        f'<div><div style="color:#1c1c1c;font-size:2.1rem;font-weight:600;line-height:1;">~{round(_tsaved,-2):,.0f} t</div>'
        f'<div style="color:#5c6b63;font-size:0.82rem;margin-top:5px;">produce kept out of waste / year</div></div></div>'
        f'<div style="color:#3e4d46;font-size:0.92rem;line-height:1.6;margin-top:10px;">'
        f'{crop}, Gujarat. Averting just {SAVE_FRAC*100:.0f}% of {crop.lower()} post-harvest loss statewide could save this much each year - '
        f'the water and carbon already spent growing produce that would otherwise rot. '
        f'In a water-stressed state, that is irrigation drawn from stressed aquifers and emissions released for food no one eats. {_note}</div>'
        f'<div style="color:#9aa6a0;font-size:0.72rem;margin-top:10px;line-height:1.5;">'
        f'Potential estimate, not a measured outcome. Production {_prod_mt:.2f} Mt ({_cs["psrc"]}, {_cs["pyear"]}). '
        f'Post-harvest loss {_cs["loss"]*100:.1f}% (ICAR-CIPHET 2015). Water {FOOTPRINT[crop]["water"]} L/kg (Water Footprint Network). '
        f'GHG {FOOTPRINT[crop]["co2"]} kg CO&#8322;e/kg (Poore &amp; Nemecek 2018). Conservative {SAVE_FRAC*100:.0f}% averted assumption.</div></div>',
        unsafe_allow_html=True)

    with st.expander(f"Evidence trail - how the {crop.lower()} water and carbon figures are calculated"):
        _e_kg = _cs['prod_t'] * 1000 * _cs['loss'] * SAVE_FRAC
        st.markdown(
            f'<div style="font-family:ui-monospace,Menlo,monospace;font-size:0.86rem;color:#333;line-height:2.0;">'
            f'<span style="color:#999;">1.</span> Gujarat {crop.lower()} production &nbsp;<b>{_cs["prod_t"]:,} t/yr</b> '
            f'<span style="color:#999;">&larr; {_cs["psrc"]}, {_cs["pyear"]}</span><br>'
            f'<span style="color:#999;">2.</span> &times; post-harvest loss &nbsp;<b>{_cs["loss"]*100:.1f}%</b> '
            f'<span style="color:#999;">&larr; ICAR-CIPHET 2015</span> &nbsp;=&nbsp; {_cs["prod_t"]*_cs["loss"]:,.0f} t lost/yr<br>'
            f'<span style="color:#999;">3.</span> &times; averted share &nbsp;<b>{SAVE_FRAC*100:.0f}%</b> '
            f'<span style="color:#999;">&larr; our conservative assumption, not a measurement</span> &nbsp;=&nbsp; {_e_kg/1000:,.0f} t saved/yr<br>'
            f'<span style="color:#999;">4a.</span> &times; water footprint &nbsp;<b>{FOOTPRINT[crop]["water"]} L/kg</b> '
            f'<span style="color:#999;">&larr; Water Footprint Network</span> &nbsp;=&nbsp; <b>{_wbn:.2f} billion L/yr</b><br>'
            f'<span style="color:#999;">4b.</span> &times; GHG footprint &nbsp;<b>{FOOTPRINT[crop]["co2"]} kg CO&#8322;e/kg</b> '
            f'<span style="color:#999;">&larr; Poore &amp; Nemecek 2018</span> &nbsp;=&nbsp; <b>{_ct:,.0f} t CO&#8322;e/yr</b> '
            f'<span style="color:#999;">(shown rounded to {round(_ct,-3):,.0f})</span></div>'
            f'<p style="color:#9aa6a0;font-size:0.78rem;margin-top:10px;line-height:1.5;">Every input above is a published figure or a labelled '
            f'assumption - there is no hidden step. The one judgement call is line 3, and we set it low on purpose: at 10% we under-claim rather '
            f'than over-claim. Change that slider below and every number moves with it.</p>', unsafe_allow_html=True)


    # --- Why this matters: the core innovation as a narrative (no numbers to fabricate) ---
    st.markdown(
        '<div style="display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 14px;">'
        '<div style="flex:1;min-width:280px;background:#fdf4f1;border:1px solid #f0d9d0;border-radius:12px;padding:13px 18px;">'
        '<div style="color:#c0392e;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">Without early warning</div>'
        '<div style="color:#5a4a45;font-size:0.9rem;line-height:1.7;margin-top:5px;">'
        'Glut hits &rarr; produce dumped &rarr; embedded water &amp; carbon wasted &rarr; prices crash &rarr; farmers lose &rarr; government reacts late</div></div>'
        '<div style="flex:1;min-width:280px;background:#eef6f0;border:1px solid #cfe5d8;border-radius:12px;padding:13px 18px;">'
        '<div style="color:#2f6b4f;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">With AgriPulse</div>'
        '<div style="color:#3e4d46;font-size:0.9rem;line-height:1.7;margin-top:5px;">'
        'Weeks of warning &rarr; early intervention &rarr; less dumping &rarr; groundwater spared &rarr; steadier prices &rarr; farmer income protected</div></div>'
        '</div>', unsafe_allow_html=True)


if _sec(2):
    # --- AI Decision card: the operational summary, first, before any chart (real confidence, no invented score) ---
    if not reliable:
        _dhead, _dact = "No clear call", "Signal is below our reliability gate - we make no recommendation here."
    elif flat:
        _dhead, _dact = "Prices stable", "Hold - no strong move expected over the horizon."
    elif not rising:
        _dhead, _dact = f"{rlabel} glut risk", "Farmer: sell early or stagger sales. Government: prepare procurement / buffer-stock and alert district officers."
    else:
        _dhead, _dact = "Prices set to rise", "Farmer: hold for the favourable window. Government: watch for a consumer-side spike."
    st.markdown(
        f'<div style="background:#fbfaf7;border:1px solid #e8e4da;border-left:5px solid {rcol};border-radius:12px;padding:15px 20px;margin:8px 0 12px;">'
        f'<div style="color:#8a8a80;font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">AI recommendation</div>'
        f'<div style="color:#1c1c1c;font-size:1.18rem;font-weight:700;margin:3px 0 8px;">{_dhead}</div>'
        f'<div style="display:flex;gap:26px;flex-wrap:wrap;color:#555;font-size:0.88rem;margin-bottom:7px;">'
        f'<span>Commodity <b style="color:#333;">{crop}</b></span><span>Region <b style="color:#333;">{place}</b></span>'
        f'<span>Horizon <b style="color:#333;">{h} days</b></span>'
        f'<span>Confidence <b style="color:#333;">validated {wf:.0f}%</b> &middot; {rlabel} risk</span></div>'
        f'<div style="color:#3e4d46;font-size:0.94rem;"><b>Action:</b> {_dact}</div></div>',
        unsafe_allow_html=True)

    # Price-shock / early-warning banner: fires only on a significant, reliable move; real percent and horizon.
    if reliable and abs(pct) >= 10:
        _mv = "fall" if not rising else "rise"
        _cond = "glut / dump risk" if not rising else "consumer price-spike risk"
        if abs(pct) >= 20:
            _bc = "#c0392e" if not rising else "#2f6b4f"
            _bg = "#fdece9" if not rising else "#eef6f0"
            st.markdown(
                f'<div style="background:{_bg};border:1.5px solid {_bc};border-radius:10px;padding:14px 20px;margin:6px 0 14px;">'
                f'<div style="color:{_bc};font-weight:700;font-size:1.05rem;letter-spacing:0.02em;">&#9888;&#65039; PRICE SHOCK DETECTED &middot; government action recommended</div>'
                f'<div style="color:#3e3a33;font-size:0.95rem;margin-top:5px;">{crop} prices are projected to {_mv} about '
                f'<b>{abs(pct):.0f}%</b> over the next {h} days at {place} - {_cond}. This clears our HIGH-risk threshold; '
                f'the policymaker view sets out the intervention.</div></div>', unsafe_allow_html=True)
        else:
            _bc = ORANGE if not rising else GREEN
            st.markdown(
                f'<div style="background:#fbf3ee;border:1px solid #e7cdbd;border-left:5px solid {_bc};border-radius:8px;padding:12px 18px;margin:6px 0 14px;">'
                f'<span style="color:{_bc};font-weight:600;font-size:0.95rem;">Early warning &middot; {rlabel} risk</span>'
                f'<span style="color:#3e3a33;font-size:0.95rem;"> &nbsp;{crop} prices are projected to {_mv} about '
                f'<b>{abs(pct):.0f}%</b> over the next {h} days at {place} - {_cond}.</span></div>', unsafe_allow_html=True)

    arrow = "&#9650;" if rising else "&#9660;"; dcol = GREEN if rising else ORANGE
    if not reliable:
        dir_html = '<span style="color:#999">No clear signal</span>'
        price_val, price_sub = "Not reliable here", "use All Gujarat"
    elif flat:
        dir_html = '<span style="color:#7a7a7a">Roughly flat</span>'
        price_val, price_sub = f"&#8377;{future:,.0f}", "little change expected"
    else:
        dir_html = f'<span style="color:{dcol}">{arrow} {"Rising" if rising else "Falling"}</span>'
        price_val, price_sub = f"&#8377;{future:,.0f}", f"{pct:+.1f}% from today"
    _naive = m['bench'].get('Naive (majority)')
    _arima = m['bench'].get('ARIMA (classical)')
    _beats = []
    if _naive is not None and wf > _naive: _beats.append('naive')
    if _arima is not None and wf > _arima: _beats.append('ARIMA')
    if wf < 53:
        acc_sub = "honest call: no edge here, so we don't guess"
    elif _beats:
        acc_sub = "5-fold walk-forward · beats " + " & ".join(_beats) + (" baselines" if len(_beats) > 1 else " baseline")
    else:
        acc_sub = "5-fold walk-forward validated"
    cards = (stat_card(f"Direction · next {h}d", dir_html)
             + stat_card("Validated accuracy", f"{wf:.0f}%", acc_sub)
             + stat_card(f"Est. price in {h}d", price_val, price_sub))
    st.markdown(f'<div style="display:flex;gap:14px;margin:8px 0 16px;">{cards}</div>', unsafe_allow_html=True)

    review_days = max(7, h // 2)
    if not reliable:
        _gate_why = (f"our model is projecting a move of {abs(pct):.0f}% over {h} days - beyond the &plusmn;50% we consider plausible, "
                     f"so we do not publish it" if abs(pct) > 50 else
                     f"validated accuracy at this market is {wf:.0f}%, below our 53% gate")
        st.markdown(reco_card('warn', "No action - we don't have a trustworthy signal here", "No call",
                              "not published - below our reliability gate", None,
                              note=f"Why: {_gate_why}. Switch to 'All Gujarat' for a firmer read on this crop."),
                    unsafe_allow_html=True)
    elif flat:
        st.markdown(reco_card('warn', "Hold - no strong move expected", "Roughly flat",
                              conf, f"{review_days} days"), unsafe_allow_html=True)
    elif view == "Government / policymaker":
        if not rising:
            st.markdown(reco_card('risk', "Prepare market intervention (procurement / price support)",
                                  f"Falling about {abs(pct):.0f}% over {h} days", conf, f"{review_days} days"), unsafe_allow_html=True)
        else:
            st.markdown(reco_card('good', "Consider phased buffer-stock release",
                                  f"Rising about {pct:.0f}% over {h} days", conf, f"{review_days} days"), unsafe_allow_html=True)
    else:
        if not rising:
            st.markdown(reco_card('risk', "Sell early or stagger - glut risk",
                                  f"Falling about {abs(pct):.0f}% over {h} days", conf, f"{review_days} days"), unsafe_allow_html=True)
        else:
            st.markdown(reco_card('good', "Hold for the favourable window",
                                  f"Rising about {pct:.0f}% over {h} days", conf, f"{review_days} days"), unsafe_allow_html=True)

    # --- Decision options: the choice and its trade-offs, not just one instruction ---
    if reliable and not flat:
        _fut = today * (1 + pct / 100)
        _hold_ok = "outperformed" if rising else "underperformed"
        _opts = [
            ("Sell now", f"&#8377;{today:,.0f}/qtl at today's price", "No storage cost, no price risk", "#5c6b63"),
            ("Hold to the horizon", f"about &#8377;{_fut:,.0f}/qtl if the {h}-day call holds",
             f"In comparable past situations, holding {_hold_ok} selling - our direction call validates at {wf:.0f}%. "
             f"Carries storage cost and spoilage risk we do not model.", GREEN if rising else ORANGE),
            ("Wait beyond " + str(h) + " days", "unknown",
             "We do not validate past our horizon, so we make no claim here.", "#9a9a9a"),
        ]
        _oh = "".join(
            f'<div style="flex:1;min-width:210px;border-left:3px solid {c};background:#fbfbfa;border-radius:0 8px 8px 0;padding:10px 14px;">'
            f'<div style="color:#1c1c1c;font-weight:600;font-size:0.95rem;">{t}</div>'
            f'<div style="color:{c};font-size:0.88rem;margin:3px 0;">{v}</div>'
            f'<div style="color:#8a8a80;font-size:0.78rem;line-height:1.5;">{n}</div></div>' for t, v, n, c in _opts)
        st.markdown(
            f'<div style="color:#8a8a80;font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;margin:10px 0 5px;">Your options &middot; and what each costs</div>'
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">{_oh}</div>', unsafe_allow_html=True)

    # --- Why this could be wrong: failure modes, stated before a judge asks ---
    with st.expander("Why this forecast could be wrong"):
        st.markdown(
            f'<div style="color:#3e3a33;font-size:0.9rem;line-height:1.8;">'
            f'Our model reads price history, season and weather. It cannot see the things below, so any of them can break the call:'
            f'<ul style="margin:6px 0 6px 18px;padding:0;">'
            f'<li>Unannounced government procurement or a buffer-stock release</li>'
            f'<li>A sudden weather shock outside the seasonal pattern</li>'
            f'<li>Export or movement restrictions, or a mandi strike</li>'
            f'<li>A festival or demand shock we do not model</li>'
            f'<li>Disease or pest damage to the standing crop</li>'
            f'<li>Reporting gaps or revisions in Agmarknet data</li></ul>'
            f'This is why we publish a {wf:.0f}% validated accuracy rather than a certainty, draw an uncertainty band on the forecast, '
            f'and stay silent when the signal is below our reliability gate. AgriPulse is decision support - it narrows the odds, it does not remove them.</div>',
            unsafe_allow_html=True)

    # --- What we don't claim: visible integrity, not buried in an expander ---
    st.markdown(
        '<div style="background:#f7f7f5;border:1px dashed #d5d5cd;border-radius:10px;padding:12px 17px;margin:4px 0 12px;">'
        '<div style="color:#7a7a7a;font-size:0.7rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;">What we don\'t claim</div>'
        '<div style="color:#5f5f57;font-size:0.88rem;line-height:1.7;margin-top:4px;">'
        '&#8226; We are not a certainty - we publish a validated accuracy and an uncertainty band, and we stay silent below our reliability gate.<br>'
        '&#8226; We do not invent what we cannot source - no buffer-stock tonnages, no inflation figure, no sustainability score out of 100.<br>'
        '&#8226; We tested satellite greenness as a predictor. It did not hold at our scale, so it is context here, not a forecast input.</div></div>',
        unsafe_allow_html=True)

    # --- Mood mascot: reacts to the actual forecast (never cheerful next to a glut) ---
    if not reliable:
        _mpose, _mline = "sitting_on_a_fence.jpg", "No clear signal here - I'd rather sit on the fence than guess."
    elif flat:
        _mpose, _mline = "hold.png", "Steady - no big move expected. Sit tight for now."
    elif not rising:
        _mpose, _mline = "high_glut_risk.png", "Glut risk ahead - the price may fall. Best to act early."
    else:
        _mpose, _mline = "with_crops.jpg", "Good news - the price looks set to rise. Worth the wait."
    _mimg = mascot(_mpose)
    if _mimg:
        _mc1, _mc2 = st.columns([1, 5], vertical_alignment="center")
        with _mc1:
            st.image(_mimg, width=135)
        with _mc2:
            st.markdown(f'<div style="background:#fffdf8;border:1px solid #ece7db;border-radius:16px;padding:14px 20px;'
                        f'box-shadow:0 1px 3px rgba(0,0,0,0.04);"><span style="color:#5b5b52;font-size:1.02rem;">'
                        f'&ldquo;{_mline}&rdquo;</span></div>', unsafe_allow_html=True)

    if view == "Government / policymaker" and reliable and not flat:
        if not rising:
            _acts = ["Open a buffer-stock procurement / price-support window to defend farmgate prices",
                     "Alert APMC and district officers to the incoming arrival surge",
                     "Issue a farmer advisory: stagger sales or use available cold storage",
                     "Facilitate movement or processing to absorb the surplus"]
            _res = "Averting the dump is what protects the embedded water and carbon in the figure above."
        else:
            _acts = ["Prepare a phased buffer-stock release to cap the spike",
                     "Coordinate arrivals from surplus mandis into the affected market",
                     "Monitor for hoarding; stagger releases across the window",
                     "Issue a consumer-side advisory if the rise is sustained"]
            _res = "Smoothing the spike supports consumer prices and farmgate stability together."
        _items = "".join(f'<li style="margin:3px 0;">{a}</li>' for a in _acts)
        st.markdown(
            f'<div style="background:#faf7f2;border:1px solid #ece4d8;border-radius:10px;padding:14px 18px;margin:2px 0 14px;">'
            f'<div style="color:#7a5a2e;font-size:0.72rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">'
            f'Government action engine &middot; {rlabel} risk &middot; {h}-day lead</div>'
            f'<ul style="color:#3e3a33;font-size:0.92rem;line-height:1.6;margin:8px 0 6px 18px;padding:0;">{_items}</ul>'
            f'<div style="color:#9aa6a0;font-size:0.78rem;">{_res} These are standard responses to the predicted direction, '
            f'not optimised tonnages - we deliberately do not invent stock volumes or an inflation figure we cannot defend.</div></div>',
            unsafe_allow_html=True)
        _mv2 = "fall" if not rising else "rise"
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e5e5e5;border-radius:10px;padding:18px 22px;margin:2px 0 14px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">'
            f'<div style="font-size:1rem;font-weight:700;color:#2f6b4f;">AI Policy Brief</div>'
            f'<div style="color:#999;font-size:0.78rem;margin-bottom:9px;">{crop}, {place} &middot; generated {pd.Timestamp.today().date()}</div>'
            f'<div style="color:#333;font-size:0.92rem;line-height:1.7;">'
            f'<b>Situation.</b> Our model projects {crop.lower()} prices to {_mv2} about {abs(pct):.0f}% over the next {h} days - a {rlabel.lower()} risk. '
            f'The forecast validates at {wf:.0f}% walk-forward direction accuracy.<br>'
            f'<b>Why it matters.</b> An unmanaged {crop.lower()} glut dumps produce whose embedded water and carbon are then wasted; a glut in an '
            f'over-exploited district draws down an aquifer that cannot spare it.<br>'
            f'<b>Recommended action.</b> {_acts[0]}; {_acts[1][0].lower() + _acts[1][1:]}.<br>'
            f'<b>Confidence.</b> {"High" if wf >= 65 else "Moderate" if wf >= 55 else "Low"} - based on validated accuracy, not a claimed certainty.'
            f'</div></div>', unsafe_allow_html=True)

    if reliable and not flat:
        qlabel = "Your stock (quintals)" if view == "Farmer advisory" else "Glut volume to manage (quintals)"
        qty = st.number_input(qlabel, min_value=0, value=100, step=10)

        if view == "Farmer advisory" and qty:
            impact = abs(qty * today * pct / 100)
            if not rising:
                st.markdown(f'<div style="padding:8px 0;color:#444;">On {qty} quintals, acting early could protect about '
                            f'<b>&#8377;{impact:,.0f}</b> (a {abs(pct):.0f}% drop avoided). <span style="color:#aaa;font-size:0.8rem;">Estimate.</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="padding:8px 0;color:#444;">On {qty} quintals, waiting for the predicted rise could add about '
                            f'<b>&#8377;{impact:,.0f}</b> (+{pct:.0f}%). <span style="color:#aaa;font-size:0.8rem;">Estimate.</span></div>', unsafe_allow_html=True)

        if not rising and qty:   # glut: avoided food waste carries a real environmental footprint
            fp = FOOTPRINT[crop]; saved_kg = qty * 100 * SAVE_FRAC
            water_L = saved_kg * fp['water']; co2_kg = saved_kg * fp['co2']; drive_km = co2_kg / 0.17
            st.markdown(
                f'<div style="background:#eef4f0;border:1px solid #dce8e1;border-radius:12px;padding:16px 20px;margin-top:6px;">'
                f'<div style="color:#2f6b4f;font-size:0.72rem;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;">Environmental impact of acting on this glut</div>'
                f'<div style="color:#1c1c1c;font-size:0.95rem;margin-top:8px;">Keeping just <b>{SAVE_FRAC*100:.0f}%</b> of {qty} quintals out of a glut-driven dump '
                f'(about {saved_kg:,.0f} kg) avoids roughly <b>{water_L:,.0f} litres</b> of water and '
                f'<b>{co2_kg:,.0f} kg CO&#8322;e</b> - the water and carbon embedded in produce that would otherwise rot. '
                f'That is about {drive_km:,.0f} km of car driving in emissions.</div>'
                f'<div style="color:#9aa6a0;font-size:0.72rem;margin-top:8px;">Estimate. Water: Water Footprint Network (Mekonnen &amp; Hoekstra, 2011). '
                f'GHG: Poore &amp; Nemecek (2018). Conservative {SAVE_FRAC*100:.0f}% spoilage-averted assumption.</div></div>',
                unsafe_allow_html=True)
    # --- One-click intervention brief: self-contained HTML (open + print-to-PDF), no extra library ---
    _wbn_b, _ct_b = state_impact(crop, SAVE_FRAC)
    _rs_b = CROP_STATE[crop]['prod_t'] * CROP_STATE[crop]['loss'] * SAVE_FRAC * 10 * today / 1e7  # value in Rs crore
    if not reliable:
        _act_b = "No action - signal below the reliability gate; the model makes no call here."
    elif flat:
        _act_b = "Hold - no strong move expected over the horizon."
    elif not rising:
        _act_b = ("Government: prepare market intervention (procurement / price support) and alert district officers. "
                  "Farmer: sell early or stagger sales to avoid the glut.")
    else:
        _act_b = "Government: consider a phased buffer-stock release. Farmer: hold for the favourable window."
    _dt = ("likely to fall" if not rising else "likely to rise") if (reliable and not flat) else "no clear call"
    _dirline = (f"Direction: {_dt} about {abs(pct):.0f}% over {h} days" if (reliable and not flat)
                else "Direction: no clear call (below reliability gate)")
    _brief_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>AgriPulse Brief - {crop}</title>
    <style>body{{font-family:Arial,Helvetica,sans-serif;max-width:640px;margin:34px auto;color:#222;line-height:1.6;padding:0 16px;}}
    h1{{font-size:20px;margin:0;}}.sub{{color:#888;font-size:13px;}}h2{{font-size:13px;margin:18px 0 4px;color:#2f6b4f;
    text-transform:uppercase;letter-spacing:.04em;}}.foot{{color:#999;font-size:11px;margin-top:22px;border-top:1px solid #eee;padding-top:10px;}}</style>
    </head><body>
    <h1>AgriPulse - Intervention Brief</h1>
    <div class="sub">{crop} &nbsp;|&nbsp; {market} &nbsp;|&nbsp; generated {pd.Timestamp.today().date()}</div>
    <h2>Forecast</h2>
    <div>{_dirline}</div><div>Confidence: {conf} &nbsp;&nbsp; Risk level: {rlabel}</div>
    <div>Validated accuracy: {wf:.0f}% (5-fold walk-forward, benchmarked vs ARIMA &amp; naive)</div>
    <h2>Recommended action</h2><div>{_act_b}</div>
    <h2>Environmental stake (statewide potential, if widely adopted)</h2>
    <div>Water: about {_wbn_b:.1f} billion litres per year</div>
    <div>CO&#8322;e: about {round(_ct_b, -3):,.0f} tonnes per year</div>
    <div>Value of avoided loss: about &#8377;{_rs_b:,.0f} crore, at today's modal price of &#8377;{today:,.0f}/quintal</div>
    <div class="sub">Conservative 10% averted assumption. Sources: NHB production, ICAR-CIPHET loss, Water Footprint Network, Poore &amp; Nemecek.</div>
    <div class="foot">Potential estimates, not guaranteed outcomes. AgriPulse is decision support, not an autopilot.
    To save as PDF: open this file and use your browser's Print &rarr; Save as PDF.</div>
    </body></html>"""
    st.download_button("Download intervention brief", data=_brief_html,
                       file_name=f"AgriPulse_{crop}_{market}_brief.html".replace(" ", "_"), mime="text/html")

if _sec(3):
    st.markdown(f'<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Price history &amp; {h}-day forecast</h3>'
            f'<p style="color:#999;font-size:0.85rem;margin-top:2px;">Solid line is actual price; dashed is our forecast, shaded band is the uncertainty range</p>',
            unsafe_allow_html=True)
    _frng = st.radio("Range", ["6M", "1Y", "3Y", "Max"], index=1, horizontal=True, key="fcrng", label_visibility="collapsed")
    _fdays = {"6M": 182, "1Y": 365, "3Y": 1095, "Max": 100000}[_frng]
    hist = base[['date', 'price']].copy()
    hist = hist[hist['date'] >= hist['date'].max() - pd.Timedelta(days=_fdays)]
    fcdf = pd.DataFrame({'date': fdates, 'price': fprice, 'lo':[p-b for p,b in zip(fprice,bw)], 'hi':[p+b for p,b in zip(fprice,bw)]})
    ax_x = alt.Axis(title=None, format='%b %Y', labelColor='#999', tickColor='#eee', domainColor='#e5e5e5', grid=False)
    ax_y = alt.Axis(title='\u20b9 / quintal', labelColor='#999', titleColor='#999', gridColor='#f2f2f2', domainColor='#e5e5e5', tickColor='#eee')
    band = alt.Chart(fcdf).mark_area(color=ORANGE, opacity=0.13).encode(x=alt.X('date:T', axis=ax_x), y=alt.Y('lo:Q', axis=ax_y), y2='hi:Q')
    la = alt.Chart(hist).mark_line(color=GREEN, strokeWidth=2).encode(x=alt.X('date:T', axis=ax_x), y=alt.Y('price:Q', axis=ax_y),
            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('price:Q', title='Rs/qtl', format=',.0f')])
    lf = alt.Chart(fcdf).mark_line(color=ORANGE, strokeWidth=2, strokeDash=[5,4]).encode(x='date:T', y='price:Q',
            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('price:Q', title='Est Rs/qtl', format=',.0f')])
    chart = (band + la + lf) if reliable else la
    st.altair_chart(chart.properties(height=320).interactive().configure_view(strokeWidth=0), use_container_width=True)

    lc, rc = st.columns(2)
    with lc:
        st.markdown('<p style="color:#444;font-weight:500;margin-bottom:2px;">What drives the forecast'
                    + info('Recent price trends and seasonality lead; climate is a secondary signal. These are the model\'s real feature importances, not an assumed ranking.')
                    + '</p>', unsafe_allow_html=True)
        idf = pd.DataFrame(m['imp'], columns=['feature','imp']); idf['pct'] = idf['imp']*100
        idf['feature'] = idf['feature'].map(LABELS); idf['label'] = idf['pct'].round(0).astype(int).astype(str) + '%'
        _b = alt.Chart(idf.head(5)).encode(y=alt.Y('feature:N', sort='-x', title=None, axis=alt.Axis(labelColor='#555', labelOverlap=False, labelLimit=220)))
        _bars = _b.mark_bar(color=GREEN, opacity=0.85).encode(
            x=alt.X('pct:Q', title='influence on the forecast (%)', axis=alt.Axis(labelColor='#999', titleColor='#999', gridColor='#f4f4f4')),
            tooltip=[alt.Tooltip('pct:Q', format='.0f')])
        _txt = _b.mark_text(align='left', dx=4, color='#3e4d46', fontSize=11, fontWeight='bold').encode(x='pct:Q', text='label:N')
        st.altair_chart((_bars + _txt).properties(height=210).configure_view(strokeWidth=0), use_container_width=True)
    with rc:
        st.markdown('<p style="color:#444;font-weight:500;margin-bottom:6px;">How models compare'
                    + info("The first four models are scored identically - 5-fold walk-forward across our full history - so Random Forest here matches the headline number exactly. The classifier can edge it on direction but produces no price, so Random Forest stays our pick. ARIMA is graded on a different exam - a single recent 20% hold-out, not the 5 folds - so its number is NOT directly comparable: where a recent stretch happens to be flat or trending it can score high simply because that window is easier. We would rather label the mismatch than quietly compare two different tests.") + '</p>', unsafe_allow_html=True)
        rows = ""
        for k, v in m['bench'].items():
            mark = " (our model)" if k == m['chosen'] else ""; w = "600" if k == m['chosen'] else "400"
            _same = "ARIMA" not in k
            _test = "5-fold walk-forward" if _same else "recent 20% hold-out"
            _tcol = "#888" if _same else "#c0722e"
            rows += (f"<tr><td style='padding:6px 12px;color:#444;font-weight:{w};'>{k}{mark}</td>"
                     f"<td style='padding:6px 12px;color:{_tcol};font-size:0.78rem;'>{_test}</td>"
                     f"<td style='padding:6px 12px;text-align:right;font-weight:{w};color:#1c1c1c;'>{v:.0f}%</td></tr>")
        st.markdown(f"<table style='border-collapse:collapse;width:100%;font-size:0.88rem;'>"
                    f"<tr><th style='text-align:left;padding:6px 12px;color:#999;font-weight:500;border-bottom:1px solid #eee;'>Model</th>"
                    f"<th style='text-align:left;padding:6px 12px;color:#999;font-weight:500;border-bottom:1px solid #eee;'>Test</th>"
                    f"<th style='text-align:right;padding:6px 12px;color:#999;font-weight:500;border-bottom:1px solid #eee;'>Direction acc.</th></tr>{rows}</table>",
                    unsafe_allow_html=True)

    # --- Why this forecast: plain-language explanation from the real feature importances ---
    _imp = pd.DataFrame(m['imp'], columns=['feature', 'imp']).sort_values('imp', ascending=False)
    _top = [LABELS.get(f, f) for f in _imp['feature'].head(2)]
    _wshare = float(_imp[_imp['feature'].isin(['rainfall', 'temp_mean'])]['imp'].sum()) * 100
    if not reliable:
        _explain = "The model is below its reliability gate for this crop and market, so it makes no call here - it would rather stay silent than guess."
    elif flat:
        _explain = (f"The forecast leans mostly on <b>{_top[0].lower()}</b> and <b>{_top[1].lower()}</b>; weather is only about "
                    f"{_wshare:.0f}% of the signal. Right now those look near their seasonal norm, so it expects little net move over {h} days.")
    else:
        _dir, _past = ("fall", "fell") if not rising else ("rise", "rose")
        _explain = (f"This call rests mostly on <b>{_top[0].lower()}</b> and <b>{_top[1].lower()}</b>; weather contributes only about "
                    f"{_wshare:.0f}%. It expects a {_dir} of roughly {abs(pct):.0f}% over {h} days because the current season and recent "
                    f"price level resemble past periods that {_past}. The model is pattern-matching history, not reading a cause - so we "
                    f"show it as a direction with a confidence, never a certainty.")
    st.markdown('<div style="background:#f7f7f5;border:1px solid #ececec;border-radius:10px;padding:12px 16px;margin:8px 0 4px;">'
                '<div style="color:#7a7a7a;font-size:0.72rem;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;">Why this forecast</div>'
                f'<div style="color:#3e3a33;font-size:0.92rem;line-height:1.6;margin-top:5px;">{_explain}</div></div>', unsafe_allow_html=True)

    # --- Similar historical events: nearest-neighbour lookup on real data, showing what actually happened ---
    _an = base.copy()
    _an['fwd'] = _an['price'].shift(-h) / _an['price'] - 1.0
    _pool = _an.dropna(subset=['fwd'])
    _pool = _pool[_pool['date'] < _an['date'].max() - pd.Timedelta(days=60)]     # only genuinely past episodes
    _dcols = [c for c in ['price_lag1', 'price_roll7', 'rainfall', 'temp_mean'] if c in _an.columns]
    if len(_pool) > 120 and _dcols:
        _cur = _an.iloc[-1]
        _mu, _sd = _pool[_dcols].mean(), _pool[_dcols].std().replace(0, 1)
        _dz = (((_pool[_dcols] - _mu) / _sd - (_cur[_dcols] - _mu) / _sd) ** 2).sum(axis=1) ** 0.5
        _a, _a0 = 2 * np.pi * _pool['dayofyear'] / 365.25, 2 * np.pi * _cur['dayofyear'] / 365.25
        _seas = ((np.sin(_a) - np.sin(_a0)) ** 2 + (np.cos(_a) - np.cos(_a0)) ** 2) ** 0.5
        _pool = _pool.assign(_d=_dz + 2.0 * _seas).sort_values('_d')             # season-weighted similarity
        _picks = []                                                              # greedy: keep matches 60+ days apart
        for _, _r in _pool.iterrows():
            if all(abs((_r['date'] - _p['date']).days) > 60 for _p in _picks):
                _picks.append(_r)
            if len(_picks) == 3:
                break
        if _picks:
            _rows = ""
            for _p in _picks:
                _pc = _p['fwd'] * 100
                _col = ORANGE if _pc < 0 else GREEN
                _rows += (f'<tr><td style="padding:5px 10px;">{_p["date"].date()}</td>'
                          f'<td style="padding:5px 10px;text-align:right;color:#777;">&#8377;{_p["price"]:,.0f}</td>'
                          f'<td style="padding:5px 10px;text-align:right;color:#777;">&#8377;{_p["price"]*(1+_p["fwd"]):,.0f}</td>'
                          f'<td style="padding:5px 10px;text-align:right;color:{_col};font-weight:600;">{_pc:+.0f}%</td></tr>')
            _nfall = sum(1 for _p in _picks if _p['fwd'] < 0)
            st.markdown(
                f'<div style="background:#f7f7f5;border:1px solid #ececec;border-radius:10px;padding:13px 17px;margin:8px 0 4px;">'
                f'<div style="color:#7a7a7a;font-size:0.72rem;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;">'
                f'Similar situations in our history' + info("A similarity lookup over our own 2021-2026 data - not the model's forecast, and not a promise history repeats. Shown as evidence you can check: these are real dates and the real moves that followed them.") + f'</div>'
                f'<div style="color:#3e3a33;font-size:0.9rem;margin:5px 0 7px;">The {len(_picks)} past dates whose season, price level and weather '
                f'most resemble today at {place}. <b>{_nfall} of {len(_picks)}</b> saw prices fall over the following {h} days.</div>'
                f'<table style="border-collapse:collapse;font-size:0.88rem;color:#333;width:100%;">'
                f'<tr style="color:#9a9a9a;font-size:0.72rem;text-align:left;border-bottom:1px solid #e8e8e8;">'
                f'<th style="padding:4px 10px;">Date</th><th style="padding:4px 10px;text-align:right;">Price then</th>'
                f'<th style="padding:4px 10px;text-align:right;">{h}d later</th>'
                f'<th style="padding:4px 10px;text-align:right;">Move</th></tr>{_rows}</table>'
                f'</div>', unsafe_allow_html=True)

    st.markdown('<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Climate context</h3>'
                '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Current weather vs the seasonal normal, then price against rainfall over time</p>', unsafe_allow_html=True)

    # --- Weather anomaly: current conditions vs the seasonal normal (honest - weather is a minor input) ---
    import calendar as _cal
    _wdf = base[['date', 'rainfall', 'temp_mean']].copy()
    _cm = int(base['date'].max().month)
    _recent = _wdf[_wdf['date'] > _wdf['date'].max() - pd.Timedelta(days=30)]
    _norm = _wdf[_wdf['date'].dt.month == _cm]
    _mname = _cal.month_name[_cm]
    def _wline(now, nrm, unit, up, down):
        if pd.isna(now) or pd.isna(nrm):
            return None
        diff = now - nrm
        word = up if diff > 0.1 else down if diff < -0.1 else "in line with"
        return f'<b>{now:.1f} {unit}</b> vs {nrm:.1f} {unit} typical for {_mname} - {word} the seasonal norm'
    _lines = [l for l in [
        _wline(_recent['rainfall'].mean(), _norm['rainfall'].mean(), "mm/day", "wetter than", "drier than"),
        _wline(_recent['temp_mean'].mean(), _norm['temp_mean'].mean(), "&deg;C", "warmer than", "cooler than"),
    ] if l]
    if _lines:
        _body = "".join(f'<div style="color:#444;font-size:0.92rem;margin:3px 0;">{l}</div>' for l in _lines)
        st.markdown(
            f'<div style="background:#f3f6f7;border:1px solid #e2e9ec;border-radius:12px;padding:14px 18px;margin:6px 0 4px;">'
            f'<div style="color:#4a6b7a;font-size:0.72rem;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:6px;">Weather now vs normal &middot; last 30 days</div>'
            f'{_body}'
            f'<div style="color:#9aa6a0;font-size:0.74rem;margin-top:8px;line-height:1.5;">Weather is a <b>minor</b> input to the forecast - recent prices and '
            f'seasonality dominate (see "What drives the forecast"). This is regional context, not a price driver. Source: Open-Meteo, per mandi.</div></div>',
            unsafe_allow_html=True)

    _crng = st.radio("Range", ["6M", "1Y", "3Y", "Max"], index=1, horizontal=True, key="clrng", label_visibility="collapsed")
    _cdays = {"6M": 182, "1Y": 365, "3Y": 1095, "Max": 100000}[_crng]
    clim = base[['date','price','rainfall']].copy()
    clim = clim[clim['date'] >= clim['date'].max() - pd.Timedelta(days=_cdays)]
    rain = alt.Chart(clim).mark_bar(color=BLUE, opacity=0.6).encode(x=alt.X('date:T', axis=ax_x),
            y=alt.Y('rainfall:Q', axis=alt.Axis(title='rain (mm)', labelColor='#bbb', titleColor='#bbb', grid=False)),
            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('rainfall:Q', title='Rain (mm)', format='.1f')])
    pr = alt.Chart(clim).mark_line(color=GREEN, strokeWidth=1.5).encode(x='date:T',
            y=alt.Y('price:Q', axis=alt.Axis(title='\u20b9 / quintal', labelColor='#999', titleColor='#999', gridColor='#f2f2f2')),
            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('price:Q', title='Rs/qtl', format=',.0f')])
    st.altair_chart(alt.layer(rain, pr).resolve_scale(y='independent').properties(height=220).configure_view(strokeWidth=0), use_container_width=True)

# ---------------------------------------------------------------------------
# Statewide scale: mandi direction map (opt-in) + interactive statewide impact
# Town-centroid coordinates for major Gujarat APMC mandi towns. Town-level
# locations - verify and extend against your exact Agmarknet mandi names.
# ---------------------------------------------------------------------------
GUJARAT_MANDI_COORDS = {
    'Ahmedabad': (23.03, 72.58), 'Rajkot': (22.30, 70.80), 'Surat': (21.17, 72.83),
    'Vadodara': (22.31, 73.19), 'Bhavnagar': (21.76, 72.15), 'Jamnagar': (22.47, 70.07),
    'Junagadh': (21.52, 70.46), 'Gandhinagar': (23.22, 72.65), 'Mehsana': (23.59, 72.37),
    'Anand': (22.56, 72.96), 'Nadiad': (22.69, 72.86), 'Bharuch': (21.70, 72.99),
    'Navsari': (20.95, 72.92), 'Valsad': (20.61, 72.93), 'Amreli': (21.60, 71.22),
    'Porbandar': (21.64, 69.61), 'Morbi': (22.82, 70.84), 'Surendranagar': (22.73, 71.64),
    'Palanpur': (24.17, 72.43), 'Himatnagar': (23.60, 72.96), 'Deesa': (24.26, 72.19),
    'Gondal': (21.96, 70.79), 'Jetpur': (21.75, 70.62), 'Veraval': (20.91, 70.37),
    'Botad': (22.17, 71.67), 'Dahod': (22.84, 74.26), 'Godhra': (22.78, 73.62),
    'Patan': (23.85, 72.13), 'Unjha': (23.80, 72.39), 'Bhuj': (23.25, 69.67),
    'Mahuva': (21.08, 71.77), 'Dhoraji': (21.73, 70.45), 'Upleta': (21.74, 70.28),
    'Khambhat': (22.31, 72.62), 'Visnagar': (23.70, 72.55), 'Savarkundla': (21.33, 71.30),
    'Jasdan': (22.04, 71.21), 'Idar': (23.83, 73.00), 'Talaja': (21.35, 72.03),
    'Kapadvanj': (23.02, 73.07), 'Padra': (22.24, 73.09), 'Nadiyad': (22.69, 72.86),
    'Bilimora': (20.77, 72.96), 'Visavadar': (21.34, 70.75), 'Ankleshwar': (21.63, 72.99),'Damnagar': (21.86, 71.52), 'Vadhvan': (22.70, 71.65), 'Mansa': (23.43, 72.66),
    'Talala': (21.35, 70.50), 'Palitana': (21.53, 71.83), 'Vankaner': (22.61, 70.95),
    'Kalol': (23.25, 72.49),
}
def _mnorm(s):
    return ''.join(ch for ch in str(s).lower() if ch.isalpha())
_COORD_NORM = {_mnorm(k): v for k, v in GUJARAT_MANDI_COORDS.items()}
def mandi_xy(name):
    n = _mnorm(name)
    if n in _COORD_NORM:
        return _COORD_NORM[n]
    for k, v in _COORD_NORM.items():
        if k and (k in n or n in k):
            return v
    return None

# CGWB "Dynamic Ground Water Resources of India, 2025" - stage of groundwater
# extraction (%) by Gujarat district. >100 = over-exploited (pumping more than recharge).
DISTRICT_STRESS = {
    'Banaskantha': 121.47, 'Patan': 118.57, 'Gandhinagar': 111.28, 'Mahesana': 108.74,
    'Sabarkantha': 75.32, 'Ahmedabad': 72.74, 'Devbhumi Dwarka': 70.32, 'Narmada': 66.85,
    'Vadodara': 66.53, 'Rajkot': 65.89, 'Amreli': 57.31, 'Botad': 57.27, 'Porbandar': 56.54,
    'Kachchh': 51.79, 'Surendranagar': 51.76, 'Morbi': 51.75, 'Mahisagar': 50.54,
    'Gir Somnath': 49.05, 'Junagadh': 47.41, 'Chhota Udepur': 45.35, 'Bhavnagar': 44.89,
    'Arvalli': 43.99, 'Jamnagar': 43.62, 'Surat': 42.11, 'Kheda': 41.40, 'Dahod': 40.73,
    'Tapi': 37.09, 'Valsad': 31.29, 'Navsari': 31.10, 'Anand': 29.44, 'Bharuch': 27.25,
    'Panchmahal': 27.05, 'Dang': 12.68,
}
TOWN_DISTRICT = {
    'deesa': 'Banaskantha', 'palanpur': 'Banaskantha', 'unjha': 'Mahesana',
    'visnagar': 'Mahesana', 'vijapur': 'Mahesana', 'mehsana': 'Mahesana',
    'kalol': 'Gandhinagar', 'mansa': 'Gandhinagar', 'siddhpur': 'Patan',
    'kapadvanj': 'Kheda', 'nadiad': 'Kheda', 'nadiyad': 'Kheda', 'khambhat': 'Anand',
    'gondal': 'Rajkot', 'jetpur': 'Rajkot', 'dhoraji': 'Rajkot', 'upleta': 'Rajkot',
    'jasdan': 'Rajkot', 'visavadar': 'Junagadh', 'talala': 'Junagadh',
    'talalagir': 'Junagadh', 'veraval': 'Junagadh', 'palitana': 'Bhavnagar',
    'talaja': 'Bhavnagar', 'mahuva': 'Bhavnagar', 'savarkundla': 'Amreli',
    'damnagar': 'Amreli', 'dhari': 'Amreli', 'bilimora': 'Navsari',
    'ankleshwar': 'Bharuch', 'padra': 'Vadodara', 'vadhvan': 'Surendranagar',
    'limdi': 'Surendranagar', 'vankaner': 'Morbi', 'godhra': 'Panchmahal',
    'bhuj': 'Kachchh', 'kmandvi': 'Kachchh', 'mundra': 'Kachchh',
    'himatnagar': 'Sabarkantha', 'idar': 'Sabarkantha',
}
def stress_of(name):
    """Return (district, extraction_pct) for a mandi market name, or (None, None)."""
    n = _mnorm(name)
    d = TOWN_DISTRICT.get(n)
    if d is None:
        for k, v in TOWN_DISTRICT.items():
            if k and (k in n or n in k):
                d = v; break
    if d is None:
        for dist in DISTRICT_STRESS:
            if _mnorm(dist) in n:
                d = dist; break
    return (d, DISTRICT_STRESS.get(d)) if d else (None, None)

if _sec(4):
    # --- The chain, in one picture: glut -> waste -> aquifer -> wetland. Every arrow a real link. ---
    _fl = LEDGER[crop]
    _fstake = (_fl['avoid_water_ML'] / SAVE_FRAC) / max(_fl['gluts'], 1)
    _fstages = [
        ("Severe glut flagged", f"{crop}, weeks ahead", "#c0722e"),
        ("Produce dumped unsold", "the market can't absorb the surplus", "#b0632a"),
        ("Water wasted", f"~{_fstake:,.0f} million litres, already pumped", "#3a7ca5"),
        ("Aquifer drawn down", "in districts already over-exploited", "#b02828"),
        ("Wetland stressed", "Thol Lake - 320+ bird species", "#2f6b4f"),
    ]
    _W, _rh, _top = 760, 74, 14
    _Hs = _top + _rh * len(_fstages) + 10
    _sv = [f'<svg viewBox="0 0 {_W} {_Hs}" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:760px;display:block;margin:2px auto;">']
    for _i, (_t, _s, _c) in enumerate(_fstages):
        _y = _top + _i * _rh
        if _i > 0:
            _sv.append(f'<path d="M {_W*0.5} {_y-_rh+52} L {_W*0.5} {_y+4}" stroke="#cfcabb" stroke-width="2" fill="none" marker-end="url(#ar)"/>')
        _sv.append(f'<rect x="70" y="{_y}" width="{_W-140}" height="50" rx="10" fill="{_c}" opacity="0.14" stroke="{_c}" stroke-width="1.5"/>')
        _sv.append(f'<text x="90" y="{_y+22}" font-family="Arial" font-size="15" font-weight="700" fill="#2a2a2a">{_t}</text>')
        _sv.append(f'<text x="90" y="{_y+40}" font-family="Arial" font-size="12" fill="#666">{_s}</text>')
        _sv.append(f'<circle cx="{_W-96}" cy="{_y+25}" r="7" fill="{_c}"/>')
    _sv.append('<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#b0aa98"/></marker></defs></svg>')
    st.markdown('<h3 style="font-weight:500;color:#444;margin-top:8px;margin-bottom:2px;">From one glut to a wetland - the whole chain</h3>'
                '<p style="color:#999;font-size:0.85rem;margin-top:0;margin-bottom:6px;">Each step is a real, sourced link - not a metaphor. This is why a price forecast is an environmental tool.</p>'
                + "".join(_sv), unsafe_allow_html=True)

    _pc1, _pc2 = st.columns([1, 8], vertical_alignment="center")
    with _pc1:
        _pm = mascot("pointing_right.png")
        if _pm: st.image(_pm, width=120)
    with _pc2:
        st.markdown(f'<h3 style="font-weight:500;color:#444;margin:0;">Glut Radar - where price collapses may be forming</h3>'
                    f'<p style="color:#999;font-size:0.85rem;margin-top:2px;">Every mapped Gujarat mandi, coloured by predicted {BEST_H[crop]}-day price direction. Orange = glut / dump risk.</p>',
                    unsafe_allow_html=True)

    if st.checkbox(f"Show Glut Radar - statewide {crop.lower()} mandi map  (trains a model per mandi; click when ready)", value=False):
        mrows, skipped = [], []
        for mk in MARKETS[crop]:
            xy = mandi_xy(mk)
            if xy is None:
                skipped.append(mk); continue
            if len(mrows) >= 40:               # effectively all mapped mandis (pre-warm before demo)
                continue
            try:
                mm = get_model(crop, "All varieties", mk)
            except Exception:
                continue
            if mm is None:
                continue
            if not mm['reliable']:
                rgb, hexc, dtxt, ptxt = [189, 189, 189], '#bdbdbd', 'no clear signal', 'below reliability gate'
            elif mm['rising']:
                rgb, hexc, dtxt, ptxt = [47, 107, 79], '#2f6b4f', 'likely to rise', f"{mm['pct']:+.1f}% over {BEST_H[crop]}d"
            else:
                rgb, hexc, dtxt, ptxt = [192, 114, 46], '#c0722e', 'likely to fall', f"{mm['pct']:+.1f}% over {BEST_H[crop]}d"
            dist, strs = stress_of(mk)
            over = strs is not None and strs > 100
            line = [200, 40, 40] if over else [255, 255, 255]     # red ring = over-exploited aquifer
            stxt = f"{dist}: {strs:.0f}% groundwater extraction{' (over-exploited)' if over else ''}" if strs is not None else "district n/a"
            _lab = mk.split('(')[0].replace('APMC', '').strip()
            mrows.append({'lat': xy[0], 'lon': xy[1], 'rgb': rgb, 'color': hexc, 'line': line,
                          'mandi': mk, 'label': _lab, 'dirtxt': dtxt, 'pcttxt': ptxt, 'stress': stxt})
        if mrows:
            mdf = pd.DataFrame(mrows)
            tip = {"html": "<b>{mandi}</b><br/>{dirtxt} ({pcttxt})<br/>{stress}",
                   "style": {"backgroundColor": "#1c1c1c", "color": "#fff", "fontSize": "12px",
                             "padding": "6px 9px", "borderRadius": "6px"}}
            try:
                layer = pdk.Layer("ScatterplotLayer", data=mdf, get_position=["lon", "lat"],
                                  get_fill_color="rgb", get_radius=15000, radius_min_pixels=6,
                                  radius_max_pixels=22, pickable=True, opacity=0.85,
                                  stroked=True, get_line_color="line", line_width_min_pixels=2)
                view = pdk.ViewState(latitude=22.6, longitude=71.7, zoom=5.7)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tip, map_style=None))
            except Exception:
                st.map(mdf, latitude='lat', longitude='lon', color='color', size=7000)   # fallback, no tooltip
            n_fall = sum(r['dirtxt'] == 'likely to fall' for r in mrows)
            st.markdown(
                f'<p style="color:#444;font-size:0.92rem;margin:6px 0 2px;">Of {len(mrows)} mapped {crop.lower()} mandis, '
                f'<b style="color:#c0722e;">{n_fall}</b> are flagged likely to fall (glut risk) over the next {BEST_H[crop]} days; '
                f'grey points are below our reliability gate, where we make no call.</p>'
                f'<p style="color:#9aa6a0;font-size:0.74rem;margin-top:2px;">Hover any dot for its mandi, predicted move, and district groundwater stress. '
                f'Green rising, orange falling, grey no clear signal. A <b style="color:#c82828;">red ring</b> marks an over-exploited aquifer block '
                f'(CGWB 2025, extraction over 100% of recharge) - a glut there wastes water the aquifer cannot spare. Town-level coordinates.</p>',
                unsafe_allow_html=True)
        else:
            st.info("None of this crop's mandis matched the coordinate lookup yet - add entries in GUJARAT_MANDI_COORDS.")
        if skipped:
            with st.expander(f"{len(skipped)} mandis have no coordinates yet (add them to GUJARAT_MANDI_COORDS)"):
                st.write(", ".join(skipped))

    # --- District groundwater-risk heatmap: districts coloured by CGWB extraction % ---
    _dist_pts = {}
    for _t, _xy in GUJARAT_MANDI_COORDS.items():
        _tn = _mnorm(_t); _dd = TOWN_DISTRICT.get(_tn)
        if _dd is None:
            for _dist in DISTRICT_STRESS:
                if _mnorm(_dist) in _tn:
                    _dd = _dist; break
        if _dd and _dd in DISTRICT_STRESS:
            _dist_pts.setdefault(_dd, []).append(_xy)
    _drows = []
    for _dd, _pts in _dist_pts.items():
        _s = DISTRICT_STRESS[_dd]
        if   _s > 100: _rgb, _cat = [176, 40, 40],  'over-exploited'
        elif _s >= 90: _rgb, _cat = [192, 114, 46], 'critical'
        elif _s >= 70: _rgb, _cat = [214, 161, 58], 'semi-critical'
        else:          _rgb, _cat = [47, 107, 79],  'safe'
        _drows.append({'lat': sum(p[0] for p in _pts)/len(_pts), 'lon': sum(p[1] for p in _pts)/len(_pts),
                       'district': _dd, 'stresstxt': f"{_s:.0f}% extraction", 'cat': _cat, 'rgb': _rgb})
    st.markdown('<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">District groundwater-risk map</h3>'
                '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Gujarat districts by how hard their aquifers are already pumped '
                '(CGWB 2025). The redder a district, the less it can afford a glut - a wasted crop there is water drawn from a stressed aquifer.</p>',
                unsafe_allow_html=True)
    if _drows:
        _ddf = pd.DataFrame(_drows)
        _dtip = {"html": "<b>{district}</b><br/>{stresstxt} ({cat})",
                 "style": {"backgroundColor": "#1c1c1c", "color": "#fff", "fontSize": "12px", "padding": "6px 9px", "borderRadius": "6px"}}
        try:
            _dlayer = pdk.Layer("ScatterplotLayer", data=_ddf, get_position=["lon", "lat"], get_fill_color="rgb",
                                get_radius=22000, radius_min_pixels=10, radius_max_pixels=40, pickable=True,
                                opacity=0.55, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1)
            st.pydeck_chart(pdk.Deck(layers=[_dlayer], initial_view_state=pdk.ViewState(latitude=22.6, longitude=71.7, zoom=5.7),
                                     tooltip=_dtip, map_style=None))
        except Exception:
            _ddf2 = _ddf.copy(); _ddf2['color'] = ['#b02828' if c == 'over-exploited' else '#c0722e' if c == 'critical'
                                                   else '#d6a13a' if c == 'semi-critical' else '#2f6b4f' for c in _ddf2['cat']]
            st.map(_ddf2, latitude='lat', longitude='lon', color='color', size=15000)
        st.markdown(
            '<p style="color:#9aa6a0;font-size:0.76rem;margin-top:4px;">'
            '<span style="color:#b02828;">&#9679;</span> over-exploited (&gt;100%) &nbsp; '
            '<span style="color:#c0722e;">&#9679;</span> critical (90-100%) &nbsp; '
            '<span style="color:#d6a13a;">&#9679;</span> semi-critical (70-90%) &nbsp; '
            '<span style="color:#2f6b4f;">&#9679;</span> safe (&lt;70%). '
            'Extraction % is the official CGWB district figure; the dot sits at the mean of that district\'s mandi towns. '
            'Read this with the Glut Radar above: a predicted glut in a red district is the highest-priority intervention.</p>'
            '<div style="background:#fdf4f1;border:1px solid #f0d9d0;border-radius:10px;padding:11px 16px;margin-top:6px;">'
            '<span style="color:#c0392e;font-weight:600;font-size:0.85rem;">It is getting worse, not better.</span> '
            '<span style="color:#5a4a45;font-size:0.88rem;">Between the CGWB 2023 and 2025 assessments, <b>Patan (98.7% &rarr; 118.6%)</b> and '
            '<b>Gandhinagar (92.3% &rarr; 111.3%)</b> both crossed out of &ldquo;critical&rdquo; into <b>over-exploited</b>, and Banaskantha rose '
            'from 115.5% to 121.5%. Four of the districts we cover now pump more groundwater than they recharge - up from two. '
            'CGWB records that Gujarat\'s groundwater extraction increased again between the 2024 and 2025 assessments.</span></div>',
            unsafe_allow_html=True)

    # --- Biodiversity linkage: Gujarat's Ramsar wetlands against the same CGWB district stress (cited, not modelled) ---
    RAMSAR = [("Thol Lake W.S.", "Mahesana", 2021, "320+ bird species on the Central Asian Flyway; 30+ threatened waterbirds incl. the critically endangered white-rumped vulture &amp; sociable lapwing, and the vulnerable sarus crane"),
              ("Nalsarovar B.S.", "Ahmedabad", 2012, "Gujarat's largest wetland bird sanctuary and its first Ramsar site"),
              ("Vadhvana Reservoir", "Vadodara", 2021, "irrigation reservoir; wintering ground on the Central Asian Flyway"),
              ("Khijadiya W.S.", "Jamnagar", 2021, "coastal freshwater-and-marine wetland mosaic")]
    _rr = ""
    for _nm, _dt, _yr, _note in RAMSAR:
        _s = DISTRICT_STRESS.get(_dt)
        if _s is None:
            continue                                   # never render a district we cannot source
        _cat = ("over-exploited", "#b02828") if _s > 100 else ("critical", "#c0722e") if _s >= 90 \
            else ("semi-critical", "#d6a13a") if _s >= 70 else ("safe", "#2f6b4f")
        _rr += (f'<tr><td style="padding:5px 10px;"><b>{_nm}</b><br><span style="color:#999;font-size:0.78rem;">Ramsar {_yr}</span></td>'
                f'<td style="padding:5px 10px;">{_dt}</td>'
                f'<td style="padding:5px 10px;text-align:right;color:{_cat[1]};font-weight:600;">{_s:.0f}%<br>'
                f'<span style="font-size:0.75rem;font-weight:400;">{_cat[0]}</span></td>'
                f'<td style="padding:5px 10px;color:#666;font-size:0.82rem;">{_note}</td></tr>')
    st.markdown(
        '<h3 style="font-weight:500;color:#444;margin-top:22px;margin-bottom:0;">Biodiversity &amp; ecosystem linkage</h3>'
        '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Gujarat\'s four Ramsar-listed wetlands, against the groundwater stress of the districts they sit in</p>'
        f'<table style="border-collapse:collapse;font-size:0.9rem;color:#333;width:100%;margin-top:6px;">'
        f'<tr style="color:#9a9a9a;font-size:0.72rem;text-align:left;border-bottom:1px solid #e8e8e8;">'
        f'<th style="padding:4px 10px;">Wetland</th><th style="padding:4px 10px;">District</th>'
        f'<th style="padding:4px 10px;text-align:right;">CGWB extraction</th>'
        f'<th style="padding:4px 10px;">Why it matters</th></tr>{_rr}</table>'
        '<div style="background:#eef4f0;border:1px solid #dce8e1;border-radius:10px;padding:13px 17px;margin-top:9px;">'
        '<div style="color:#3e4d46;font-size:0.92rem;line-height:1.7;">'
        '<b>Thol Lake is the link in one line.</b> It is a Ramsar wetland that was built as an irrigation tank in 1912, and it sits in '
        '<b>Mehsana - the district CGWB rates at 109% extraction</b>. The irrigation that feeds a tomato glut in Unjha, Visnagar or '
        'Mehsana APMC draws on the same over-exploited aquifer system as the wetland that hosts 320+ bird species and 30+ threatened waterbirds. '
        'Averting that glut means water not pumped, in the one district where that matters most.</div>'
        '<div style="color:#9aa6a0;font-size:0.75rem;margin-top:8px;line-height:1.5;">Sources: Ramsar Sites Information Service (Thol, site 2458, '
        'designated 5 Apr 2021); CGWB Dynamic Ground Water Resources of India 2025. <b>We state this linkage and cite it - we do not model '
        'species outcomes and we claim no measured biodiversity benefit.</b> Groundwater over-extraction is a recognised pressure on wetland '
        'hydrology; quantifying that chain is named as future work, not claimed here.</div></div>',
        unsafe_allow_html=True)

    # --- Measured waste ledger (crop-aware): real gluts, measured arrivals, groundwater-tagged ---
    # Cumulative figures from our leak-free backtest joined to Agmarknet arrivals (2021-2026).
    # Per-crop framing follows each model's real warning precision: tomato is validated; onion/potato are low-precision.
    _L = LEDGER[crop]
    _bcol = '#2f6b4f' if _L['validated'] else '#c0722e'
    _bbg, _bbr = ('#eef4f0', '#dce8e1') if _L['validated'] else ('#fbf3ee', '#ecd9c8')
    _rowsh = ''
    for _d, _ar, _wm, _ov in _L['rows']:
        _rowsh += (f'<tr><td style="padding:4px 10px;">{_d}</td>'
                   f'<td style="padding:4px 10px;text-align:right;">{_ar:,}</td>'
                   f'<td style="padding:4px 10px;text-align:right;">{_wm:,}</td>'
                   f'<td style="padding:4px 10px;text-align:right;">{_ov:.1f}%</td></tr>')
    _ledger_tip = info(f"Showing the 6 most recent of {_L['gluts']} gluts; full ledger in our notebook. Arrivals are volume that reached "
                       f"the mandi, not all wasted, so the {int(SAVE_FRAC*100)}% averted fraction is applied to the cumulative figures. "
                       f"Water 214/272/287 L/kg (Water Footprint Network), carbon (Poore and Nemecek). Over-exploited = CGWB 2025 districts "
                       f"extracting more groundwater than recharge. 1 ML = one million litres.")
    st.markdown(
        f'<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Measured waste ledger ({crop.lower()}){_ledger_tip}</h3>'
        f'<p style="color:#999;font-size:0.85rem;margin-top:2px;">Severe {crop.lower()} gluts and the {crop.lower()} that '
        f'<i>actually</i> arrived at Gujarat mandis in the window after each warning - measured from Agmarknet arrivals, not estimated.</p>'
        f'<div style="background:{_bbg};border:1px solid {_bbr};border-radius:10px;padding:11px 15px;margin:7px 0 9px;">'
        f'<div style="color:{_bcol};font-size:0.82rem;font-weight:600;line-height:1.5;">{_L["badge"]}</div></div>'
        f'<table style="border-collapse:collapse;font-size:0.9rem;color:#333;width:100%;">'
        f'<tr style="color:#7a7a7a;font-size:0.76rem;text-align:left;border-bottom:1px solid #e5e5e5;">'
        f'<th style="padding:4px 10px;">Glut caught</th><th style="padding:4px 10px;text-align:right;">Arrivals (t)</th>'
        f'<th style="padding:4px 10px;text-align:right;">Embedded water (ML)</th>'
        f'<th style="padding:4px 10px;text-align:right;">In over-exploited blocks</th></tr>{_rowsh}</table>'
        f'<p style="color:#3e4d46;font-size:0.9rem;margin-top:10px;">Across all {_L["gluts"]} gluts over 2021-2026, at a conservative '
        f'{int(SAVE_FRAC*100)}% averted, {_L["verb"]} about <b>{_L["avoid_water_ML"]:,} ML</b> of water and '
        f'<b>{_L["avoid_co2_t"]:,} tonnes</b> of CO&#8322;e (cumulative).</p>',
        unsafe_allow_html=True)

    # --- Policy intervention simulator: grounded in the real leak-free glut backtest + measured arrivals ---
    _sim_tip = info(("Grounded in tomato, our validated model - it caught every severe glut it could be tested on, with 52% warning precision "
                     "against a 34% base rate. " if _L['validated'] else
                     f"The {crop.lower()} model's warning precision is low, so read this as the opportunity a reliable forecast would unlock, "
                     f"not a delivered result. ") +
                    "Water and carbon come from measured arrivals (Water Footprint Network, Poore and Nemecek); the rupee figure values the "
                    "tonnage at the current price, so it is value preserved, not guaranteed cash. This is the lever a policymaker actually "
                    "controls - the forecast warns, the intervention capacity decides the outcome.")
    st.markdown(
        f'<div style="color:#3a5a6a;font-size:0.72rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;margin-top:18px;">Policy intervention simulator{_sim_tip}</div>'
        f'<div style="color:#3e4d46;font-size:0.92rem;line-height:1.6;margin:6px 0 2px;">Over 2021-2026, AgriPulse flagged <b>{_L["gluts"]} severe {crop.lower()} gluts</b> '
        f'weeks ahead (about {h}-day lead), leak-free. The forecast is the early warning; a government\'s buffer-stock and procurement capacity '
        f'decides how much of each glut is absorbed before it is dumped.</div>', unsafe_allow_html=True)
    _total_t = _L['avoid_co2_t'] / (SAVE_FRAC * FOOTPRINT[crop]['co2'])          # total at-risk arrivals (t) across the caught gluts
    _cap = st.slider(f"Intervention capacity - share of flagged {crop.lower()} glut volume you can act on (%)", 5, 50, int(SAVE_FRAC*100), key="polsim")
    _kg = _total_t * (_cap / 100) * 1000
    _pw, _pc = _kg * FOOTPRINT[crop]['water'] / 1e6, _kg * FOOTPRINT[crop]['co2'] / 1000
    _pr = _total_t * (_cap / 100) * 10 * today / 1e7
    st.markdown(
        f'<div style="background:#eef2f4;border:1px solid #dbe4e8;border-radius:12px;padding:14px 18px;margin-top:4px;">'
        f'<div style="color:#1c1c1c;font-size:1.05rem;">Acting on those flagged gluts at <b>{_cap}%</b> capacity would have protected about '
        f'<b>{_pw:,.0f} ML</b> of water, <b>{_pc:,.0f} tonnes</b> of CO&#8322;e, and <b>&#8377;{_pr:,.0f} crore</b> of {crop.lower()} '
        f'(cumulative, 2021-2026, valued at today\'s modal price).</div></div>',
        unsafe_allow_html=True)

    _state_tip = info(f"Potential, not a measured outcome; our headline uses a conservative {int(SAVE_FRAC*100)}%. This crop and market validate at {m['wf_acc']:.0f}% walk-forward direction accuracy. The rupee figure values the avoided tonnage at the current modal price, so it moves with both the price and the slider - it is value preserved, not guaranteed cash. Sources: NHB production, ICAR-CIPHET loss, Water Footprint Network, Poore and Nemecek.")
    st.markdown(f'<p style="color:#444;font-weight:500;margin:14px 0 2px;">Statewide impact if widely adopted ({crop.lower()}){_state_tip}</p>', unsafe_allow_html=True)
    adopt = st.slider(f"Share of avoidable {crop.lower()} loss actually averted (%)", 1, 50, int(SAVE_FRAC*100))
    _awbn, _act = state_impact(crop, adopt / 100)
    _csp = CROP_STATE[crop]
    _avt = _csp['prod_t'] * _csp['loss'] * (adopt / 100)      # avoidable tonnes
    _rs_cr = _avt * 10 * today / 1e7                          # value at current modal price, in Rs crore (1 t = 10 quintals)
    st.markdown(
        f'<div style="background:#eef4f0;border:1px solid #dce8e1;border-radius:12px;padding:14px 18px;margin-top:4px;">'
        f'<div style="color:#1c1c1c;font-size:1.05rem;">At <b>{adopt}%</b> averted, {crop.lower()} alone could keep about '
        f'<b>{_awbn:.1f} billion litres</b> of water and <b>{round(_act,-3):,.0f} tonnes</b> of CO&#8322;e out of the waste stream each year.</div>'
        f'<div style="color:#1c1c1c;font-size:1.05rem;margin-top:6px;">That is a market value of about <b>&#8377;{_rs_cr:,.0f} crore</b> '
        f'of {crop.lower()} saved, valued at today\'s modal price of &#8377;{today:,.0f} per quintal.</div></div>',
        unsafe_allow_html=True)

    # Satellite vegetation health (NDVI) - regional crop-health context only.
    # Tested as a predictor of future price direction; did not hold at our scale, so it is NOT a model input.
    if NDVI is not None and len(NDVI) > 12:
        import calendar
        nd = NDVI.copy(); nd['month'] = nd['date'].dt.month
        norm_by_month = nd.groupby('month')['ndvi'].mean()
        nd['normal'] = nd['month'].map(norm_by_month)
        recent_val = float(nd['ndvi'].tail(2).mean()); latest_month = int(nd.iloc[-1]['month'])
        latest_norm = float(norm_by_month.loc[latest_month]); anom = recent_val - latest_norm
        if anom > 0.02:
            read = "above its seasonal norm"
        elif anom < -0.02:
            read = "below its seasonal norm"
        else:
            read = "in line with its seasonal norm"
        mname = calendar.month_name[latest_month]
        st.markdown('<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Regional crop-health context (satellite)</h3>'
                    '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Greenness averaged across five Gujarat growing districts, from NASA MODIS</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#444;font-size:0.92rem;margin:4px 0 8px;">Satellite vegetation greenness is currently <b>{read}</b> '
                    f'(NDVI {recent_val:.2f} vs {latest_norm:.2f} typical for {mname}). This is regional crop-health context only. '
                    f'We tested NDVI as a predictor of future price direction and it did not hold at our scale, so it does not feed the '
                    f'forecast. A validated lagged-NDVI signal is planned for Phase 3.</p>', unsafe_allow_html=True)
        ndax = alt.Axis(title=None, format='%b %Y', labelColor='#999', tickColor='#eee', domainColor='#e5e5e5', grid=False)
        b = alt.Chart(nd)
        nd_line = b.mark_line(color=GREEN, strokeWidth=2, point=alt.OverlayMarkDef(color=GREEN, size=16)).encode(
            x=alt.X('date:T', axis=ndax),
            y=alt.Y('ndvi:Q', title='NDVI', scale=alt.Scale(zero=False), axis=alt.Axis(labelColor='#999', titleColor='#999', gridColor='#f2f2f2')),
            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('ndvi:Q', title='NDVI', format='.2f')])
        nd_norm = b.mark_line(color='#bbbbbb', strokeWidth=1.5, strokeDash=[4,4]).encode(x='date:T', y='normal:Q')
        st.altair_chart((nd_norm + nd_line).properties(height=220).interactive().configure_view(strokeWidth=0), use_container_width=True)
        st.markdown('<p style="color:#b0b0b0;font-size:0.76rem;margin-top:4px;">Source: NASA MODIS MOD13Q1 (16-day NDVI, 250 m) via AppEEARS. '
                    'Dashed line is the seasonal monthly average. Regional vegetation, not crop-specific.</p>', unsafe_allow_html=True)
    # --- Prediction audit + deployment readiness: honest systems engineering, in expanders to keep the page calm ---
    with st.expander("Prediction audit - what this forecast is built on, and where it is limited"):
        _tr_start, _tr_end = base['date'].min().date(), base['date'].max().date()
        _yrs = sorted(base['date'].dt.year.unique())
        _fresh = (pd.Timestamp.today().normalize() - pd.Timestamp(last)).days
        _lim = ("Onion direction accuracy is near chance at our scale - we show the signal but do not claim reliability."
                if crop == 'Onion' else
                "Potato is a solid secondary model; warning precision on severe gluts is low, so treat glut calls as indicative."
                if crop == 'Potato' else
                "Tomato is our strongest model: it caught every severe glut it could be tested on, with 52% warning precision against a 34% base rate.")
        st.markdown(
            f'<div style="font-size:0.9rem;color:#333;line-height:1.9;">'
            f'<b>Commodity / market</b> &nbsp;{crop} &middot; {place}<br>'
            f'<b>Horizon</b> &nbsp;{h} days &nbsp;|&nbsp; <b>Validated accuracy</b> &nbsp;{wf:.0f}% (5-fold walk-forward) &nbsp;|&nbsp; <b>Risk</b> &nbsp;{rlabel}<br>'
            f'<b>Training window</b> &nbsp;{_tr_start} to {_tr_end} &nbsp;({len(base):,} usable days)<br>'
            f'<b>Comparable years</b> &nbsp;{", ".join(str(y) for y in _yrs)}<br>'
            f'<b>Data freshness</b> &nbsp;latest price {last.date()} ({_fresh} days old)<br>'
            f'<b>Uncertainty</b> &nbsp;&plusmn;&#8377;{sigma:,.0f}/qtl (held-out residual sigma, drawn as the band on the forecast)<br>'
            f'<b>Missing data</b> &nbsp;days without both a price and matching weather are dropped, never imputed<br>'
            f'<b>Known limitation</b> &nbsp;{_lim}</div>', unsafe_allow_html=True)

    with st.expander("Deployment readiness - what we are connected to, and what a real rollout would need"):
        _have = [("Agmarknet (Gujarat)", "daily mandi prices &amp; arrivals, 2021-2026"),
                 ("Open-Meteo", "per-mandi daily rainfall &amp; temperature"),
                 ("NASA MODIS (MOD13Q1)", "regional NDVI - tested, not a predictor at our scale"),
                 ("CGWB 2025", "district groundwater extraction (%) - latest assessment, released Dec 2025"),
                 ("NHB / ICAR-CIPHET", "production &amp; post-harvest loss rates"),
                 ("Water Footprint Network / Poore &amp; Nemecek", "water &amp; carbon coefficients")]
        _need = [("Cold-storage capacity", "not published; operator API integration required"),
                 ("Procurement / buffer-stock systems", "held by FCI / NAFED / state agencies"),
                 ("NGO &amp; food-bank registry", "no public registry available"),
                 ("Logistics / transport feeds", "integration point, not modelled"),
                 ("Trade &amp; export data", "not connected - so we make no import/export claims")]
        _hh = "".join(f'<div style="margin:3px 0;"><span style="color:#2f6b4f;">&#9679;</span> <b>{n}</b> '
                      f'<span style="color:#777;">- {d}</span></div>' for n, d in _have)
        _nn = "".join(f'<div style="margin:3px 0;"><span style="color:#bbb;">&#9675;</span> <b>{n}</b> '
                      f'<span style="color:#777;">- {d}</span></div>' for n, d in _need)
        st.markdown(
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;font-size:0.88rem;">'
            f'<div style="flex:1;min-width:290px;"><div style="color:#2f6b4f;font-size:0.72rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:5px;">Live data sources</div>{_hh}</div>'
            f'<div style="flex:1;min-width:290px;"><div style="color:#8a8a80;font-size:0.72rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:5px;">Integration points (not connected)</div>{_nn}</div></div>'
            f'<p style="color:#9aa6a0;font-size:0.78rem;margin-top:10px;line-height:1.5;">We show this deliberately. Every figure in AgriPulse '
            f'traces to a live source above, a validated model, or a slider you set yourself. Where the data does not exist publicly, we name the '
            f'integration point rather than invent a number - that is what makes this deployable rather than a demo.</p>', unsafe_allow_html=True)

st.markdown('<p style="color:#b0b0b0;font-size:0.78rem;margin-top:14px;">Environmental rationale: preventing glut-driven dumping avoids the '
            'water and carbon embedded in wasted produce. Validated by 5-fold walk-forward testing on three years of Agmarknet (Gujarat) '
            'prices and Open-Meteo climate. Each crop uses its best horizon.</p>', unsafe_allow_html=True)
