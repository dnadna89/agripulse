import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import base64
import altair as alt
import pydeck as pdk

@st.cache_data(show_spinner=False)
def mascot(fname, width=110):
    """Return an <img> tag for a mascot in ./mascots, or '' if the file is absent (never breaks the app)."""
    try:
        with open(os.path.join("mascots", fname), "rb") as _f:
            _b = base64.b64encode(_f.read()).decode()
        _mime = "png" if fname.lower().endswith(".png") else "jpeg"
        return f'<img src="data:image/{_mime};base64,{_b}" width="{width}" style="display:block;height:auto;">'
    except Exception:
        return ""
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
def reco_card(kind, action, movement, conf_txt, review_txt):
    bg, bar, fg = {'risk':('#fbf4e8','#c0722e','#8a5418'),
                   'good':('#eef4f0','#2f6b4f','#2a5742'),
                   'warn':('#f4f1ec','#9a9a9a','#5f5f5f')}[kind]
    return (f'<div style="background:{bg};border-left:3px solid {bar};padding:14px 18px;border-radius:6px;margin:4px 0 12px;">'
            f'<div style="color:{fg};font-size:0.72rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">Recommended action</div>'
            f'<div style="color:{fg};font-size:1.12rem;font-weight:600;margin:4px 0 8px;">{action}</div>'
            f'<div style="color:{fg};font-size:0.88rem;line-height:1.7;">Expected movement: <b>{movement}</b><br>'
            f'Confidence: <b>{conf_txt}</b><br>Review again in about {review_txt}</div></div>')
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

st.markdown(
    f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;">'
    f'{mascot("welcoming.jpg", 92)}'
    f'<div><div style="font-size:2.1rem;font-weight:700;color:#1c1c1c;line-height:1.05;">AgriPulse</div>'
    f'<div style="color:#888;font-size:1rem;margin-top:3px;">A glut early-warning system built to cut the water and carbon wasted when crops rot unsold</div>'
    f'</div></div>', unsafe_allow_html=True)

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

# Early-warning banner: fires only on a significant, reliable move; uses the real percent and horizon.
if reliable and abs(pct) >= 10:
    _bc = ORANGE if not rising else GREEN
    _mv = "fall" if not rising else "rise"
    _cond = "glut / dump risk" if not rising else "price-spike risk"
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

where = f"{crop} at {market}" if market != "All Gujarat" else f"{crop} across Gujarat"
review_days = max(7, h // 2)
if not reliable:
    st.markdown(reco_card('warn', "No action - signal not reliable here", "Unclear",
                          conf, "switch to 'All Gujarat' for a firmer read"), unsafe_allow_html=True)
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

# --- Mood mascot: reacts to the actual forecast (never cheerful next to a glut) ---
if not reliable:
    _mpose, _mline = "sitting_on_a_fence.jpg", "No clear signal here - I'd rather sit on the fence than guess."
elif flat:
    _mpose, _mline = "hold.png", "Steady - no big move expected. Sit tight for now."
elif not rising:
    _mpose, _mline = "high_glut_risk.png", "Glut risk ahead - the price may fall. Best to act early."
else:
    _mpose, _mline = "with_crops.jpg", "Good news - the price looks set to rise. Worth the wait."
_mimg = mascot(_mpose, 84)
if _mimg:
    st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin:2px 0 16px;">{_mimg}'
                f'<div style="color:#6b6b6b;font-size:0.92rem;font-style:italic;">{_mline}</div></div>', unsafe_allow_html=True)

# --- State-scale environmental headline (the stake, right after the action) ---
_cs = CROP_STATE[crop]
_wbn, _ct = state_impact(crop, SAVE_FRAC)
_prod_mt = _cs['prod_t'] / 1e6
_note = {
    'Tomato': 'Tomato is our strongest-validated model, which is why it anchors the demo.',
    'Potato': 'Potato is a solid secondary model, and potatoes store well, so read this as the resources embedded in post-harvest loss.',
    'Onion':  'Onion direction accuracy is only near chance at our scale, so read this as the resource opportunity, not a delivered result.',
}[crop]
st.markdown(
    f'<div style="background:#eef4f0;border:1px solid #dce8e1;border-radius:14px;padding:20px 24px;margin:4px 0 18px;">'
    f'<div style="color:#2f6b4f;font-size:0.72rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">'
    f'Statewide environmental potential &middot; {crop.lower()} &middot; if widely adopted</div>'
    f'<div style="display:flex;gap:48px;flex-wrap:wrap;margin:13px 0 4px;">'
    f'<div><div style="color:#1c1c1c;font-size:2.1rem;font-weight:600;line-height:1;">~{_wbn:.1f} billion L</div>'
    f'<div style="color:#5c6b63;font-size:0.82rem;margin-top:5px;">water kept out of the waste stream / year</div></div>'
    f'<div><div style="color:#1c1c1c;font-size:2.1rem;font-weight:600;line-height:1;">~{round(_ct,-3):,.0f} t</div>'
    f'<div style="color:#5c6b63;font-size:0.82rem;margin-top:5px;">CO&#8322;e avoided / year</div></div></div>'
    f'<div style="color:#3e4d46;font-size:0.92rem;line-height:1.6;margin-top:10px;">'
    f'{crop}, Gujarat. Averting just {SAVE_FRAC*100:.0f}% of {crop.lower()} post-harvest loss statewide could save this much each year - '
    f'the water and carbon already spent growing produce that would otherwise rot. '
    f'In a water-stressed state, that is irrigation drawn from stressed aquifers and emissions released for food no one eats. {_note}</div>'
    f'<div style="color:#9aa6a0;font-size:0.72rem;margin-top:10px;line-height:1.5;">'
    f'Potential estimate, not a measured outcome. Production {_prod_mt:.2f} Mt ({_cs["psrc"]}, {_cs["pyear"]}). '
    f'Post-harvest loss {_cs["loss"]*100:.1f}% (ICAR-CIPHET 2015). Water {FOOTPRINT[crop]["water"]} L/kg (Water Footprint Network). '
    f'GHG {FOOTPRINT[crop]["co2"]} kg CO&#8322;e/kg (Poore &amp; Nemecek 2018). Conservative {SAVE_FRAC*100:.0f}% averted assumption.</div></div>',
    unsafe_allow_html=True)

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
base = m['base']; last = m['last_date']; sigma = m['sigma']
fdates = [last + pd.Timedelta(days=k) for k in range(0, h+1)]
fprice = [today + (future-today)*k/h for k in range(0, h+1)]
bw     = [1.28*sigma*(k/h) for k in range(0, h+1)]
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

hist = base[['date','price']].copy()          # full history so 2021 onward shows (chart is zoomable)
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
    st.markdown('<p style="color:#444;font-weight:500;margin-bottom:2px;">What drives the forecast</p>', unsafe_allow_html=True)
    idf = pd.DataFrame(m['imp'], columns=['feature','imp']); idf['pct'] = idf['imp']*100; idf['feature'] = idf['feature'].map(LABELS)
    st.altair_chart(alt.Chart(idf.head(5)).mark_bar(color=GREEN, opacity=0.85).encode(
        x=alt.X('pct:Q', title='importance (%)', axis=alt.Axis(labelColor='#999', titleColor='#999', gridColor='#f4f4f4')),
        y=alt.Y('feature:N', sort='-x', title=None, axis=alt.Axis(labelColor='#555', labelOverlap=False, labelLimit=220)),
        tooltip=[alt.Tooltip('pct:Q', format='.0f')]).properties(height=210).configure_view(strokeWidth=0), use_container_width=True)
    st.markdown('<p style="color:#aaa;font-size:0.78rem;margin-top:-6px;">Recent price trends and seasonality lead; climate is a secondary signal.</p>', unsafe_allow_html=True)
with rc:
    st.markdown('<p style="color:#444;font-weight:500;margin-bottom:6px;">How models compare (5-fold walk-forward)</p>', unsafe_allow_html=True)
    rows = ""
    for k, v in m['bench'].items():
        mark = " (our model)" if k == m['chosen'] else ""; w = "600" if k == m['chosen'] else "400"
        rows += (f"<tr><td style='padding:6px 12px;color:#444;font-weight:{w};'>{k}{mark}</td>"
                 f"<td style='padding:6px 12px;text-align:right;font-weight:{w};color:#1c1c1c;'>{v:.0f}%</td></tr>")
    st.markdown(f"<table style='border-collapse:collapse;width:100%;font-size:0.88rem;'>"
                f"<tr><th style='text-align:left;padding:6px 12px;color:#999;font-weight:500;border-bottom:1px solid #eee;'>Model</th>"
                f"<th style='text-align:right;padding:6px 12px;color:#999;font-weight:500;border-bottom:1px solid #eee;'>Direction acc.</th></tr>{rows}</table>",
                unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;font-size:0.78rem;margin-top:8px;">Every model here is scored the same way - 5-fold walk-forward '
                'direction accuracy - so Random Forest in this table matches the headline number exactly. The classifier can edge it on '
                'direction but produces no price, so Random Forest stays our pick. ARIMA, the classical baseline, uses a recent expanding hold-out.</p>',
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

st.markdown('<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Climate context</h3>'
            '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Price against rainfall, 2021 to present</p>', unsafe_allow_html=True)
clim = base[['date','price','rainfall']].copy()
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

# CGWB "Dynamic Ground Water Resources of India, 2023" - stage of groundwater
# extraction (%) by Gujarat district. >100 = over-exploited (pumping more than recharge).
DISTRICT_STRESS = {
    'Banaskantha': 115.5, 'Mahesana': 108.2, 'Patan': 98.7, 'Gandhinagar': 92.3,
    'Ahmedabad': 87.3, 'Sabarkantha': 71.3, 'Vadodara': 64.4, 'Rajkot': 62.8,
    'Kachchh': 54.1, 'Amreli': 50.4, 'Botad': 50.3, 'Porbandar': 50.1,
    'Surendranagar': 47.7, 'Morbi': 45.9, 'Bhavnagar': 42.8, 'Junagadh': 42.3,
    'Kheda': 40.3, 'Surat': 39.9, 'Dahod': 39.5, 'Jamnagar': 36.4,
    'Bharuch': 30.9, 'Navsari': 27.1, 'Valsad': 26.6, 'Anand': 22.8, 'Panchmahal': 21.8,
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

st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin-top:24px;">{mascot("pointing_right.jpg", 70)}'
            f'<div><h3 style="font-weight:500;color:#444;margin:0;">Glut Radar — where price collapses may be forming</h3>'
            f'<p style="color:#999;font-size:0.85rem;margin-top:2px;">Up to 12 Gujarat mandis, coloured by predicted {BEST_H[crop]}-day price direction. Orange = glut / dump risk.</p></div></div>',
            unsafe_allow_html=True)

if st.checkbox(f"Glut Radar — statewide {crop.lower()} mandi price-direction map", value=True):
    mrows, skipped = [], []
    for mk in MARKETS[crop]:
        xy = mandi_xy(mk)
        if xy is None:
            skipped.append(mk); continue
        if len(mrows) >= 12:               # cap for free-tier performance
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
        mrows.append({'lat': xy[0], 'lon': xy[1], 'rgb': rgb, 'color': hexc, 'line': line,
                      'mandi': mk, 'dirtxt': dtxt, 'pcttxt': ptxt, 'stress': stxt})
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
            f'(CGWB 2023, extraction over 100% of recharge) - a glut there wastes water the aquifer cannot spare. Town-level coordinates.</p>',
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
            '(CGWB 2023). The redder a district, the less it can afford a glut - a wasted crop there is water drawn from a stressed aquifer.</p>',
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
        'Read this with the Glut Radar above: a predicted glut in a red district is the highest-priority intervention.</p>',
        unsafe_allow_html=True)

# --- Measured waste ledger (crop-aware): real gluts, measured arrivals, groundwater-tagged ---
# Cumulative figures from our leak-free backtest joined to Agmarknet arrivals (2021-2026).
# Per-crop framing follows each model's real warning precision: tomato is validated; onion/potato are low-precision.
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
_L = LEDGER[crop]
_bcol = '#2f6b4f' if _L['validated'] else '#c0722e'
_bbg, _bbr = ('#eef4f0', '#dce8e1') if _L['validated'] else ('#fbf3ee', '#ecd9c8')
_rowsh = ''
for _d, _ar, _wm, _ov in _L['rows']:
    _rowsh += (f'<tr><td style="padding:4px 10px;">{_d}</td>'
               f'<td style="padding:4px 10px;text-align:right;">{_ar:,}</td>'
               f'<td style="padding:4px 10px;text-align:right;">{_wm:,}</td>'
               f'<td style="padding:4px 10px;text-align:right;">{_ov:.1f}%</td></tr>')
st.markdown(
    f'<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Measured waste ledger ({crop.lower()})</h3>'
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
    f'<b>{_L["avoid_co2_t"]:,} tonnes</b> of CO&#8322;e (cumulative).</p>'
    f'<p style="color:#9aa6a0;font-size:0.74rem;margin-top:4px;line-height:1.5;">Showing the 6 most recent of {_L["gluts"]} gluts; '
    f'full ledger in our notebook. Arrivals are volume that reached the mandi, not all wasted, so the {int(SAVE_FRAC*100)}% averted '
    f'fraction is applied to the cumulative figures. Water 214/272/287 L/kg (Water Footprint Network), carbon (Poore &amp; Nemecek). '
    f'"Over-exploited" = CGWB 2023 districts extracting more groundwater than recharge. 1 ML = one million litres.</p>',
    unsafe_allow_html=True)

st.markdown(f'<p style="color:#444;font-weight:500;margin:14px 0 2px;">Statewide impact if widely adopted ({crop.lower()})</p>', unsafe_allow_html=True)
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
    f'of {crop.lower()} saved, valued at today\'s modal price of &#8377;{today:,.0f} per quintal.</div>'
    f'<div style="color:#9aa6a0;font-size:0.74rem;margin-top:6px;">Potential, not a measured outcome; our headline uses a conservative '
    f'{int(SAVE_FRAC*100)}%. This crop and market validate at {m["wf_acc"]:.0f}% walk-forward direction accuracy. The rupee figure values the '
    f'avoided tonnage at the current modal price, so it moves with both the price and the slider - it is value preserved, not guaranteed cash. '
    f'Sources: {_csp["psrc"]} production ({_csp["pyear"]}), ICAR-CIPHET loss, Water Footprint Network, Poore &amp; Nemecek.</div></div>',
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
st.markdown('<p style="color:#b0b0b0;font-size:0.78rem;margin-top:14px;">Environmental rationale: preventing glut-driven dumping avoids the '
            'water and carbon embedded in wasted produce. Validated by 5-fold walk-forward testing on three years of Agmarknet (Gujarat) '
            'prices and Open-Meteo climate. Each crop uses its best horizon.</p>', unsafe_allow_html=True)
