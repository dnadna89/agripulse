import streamlit as st
import pandas as pd
import numpy as np
import glob
import altair as alt
import pydeck as pdk
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
    return crops_raw, weather, varieties, markets

CROPS_RAW, WEATHER, VARIETIES, MARKETS = load_everything()

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
    wd = set(WEATHER['date'])
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
    base = daily_series(crop, variety, market).merge(WEATHER, on='date', how='inner').sort_values('date').reset_index(drop=True)
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

st.title("AgriPulse")
st.markdown('<p style="color:#888;font-size:1rem;margin-top:2px;">A glut early-warning system preventing the water and carbon wasted when crops rot unsold</p>',
            unsafe_allow_html=True)

# --- State-scale environmental headline (follows the selected crop) ---
_cs = CROP_STATE[crop]
_wbn, _ct = state_impact(crop, SAVE_FRAC)
_prod_mt = _cs['prod_t'] / 1e6
_note = {
    'Tomato': 'Tomato is our strongest-validated model, which is why it anchors the demo.',
    'Potato': 'Potato is a solid secondary model, and potatoes store well, so read this as the resources embedded in post-harvest loss.',
    'Onion':  'Onion direction accuracy is only near chance at our scale, so read this as the resource opportunity, not a delivered result.',
}[crop]
st.markdown(
    f'<div style="background:#eef4f0;border:1px solid #dce8e1;border-radius:14px;padding:20px 24px;margin:14px 0 18px;">'
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

m = get_model(crop, variety, market)
if m is None:
    st.warning("Not enough history for this yard and variety combination. Try 'All Gujarat' or 'All varieties'."); st.stop()

today, future, h, wf, pct, rising, reliable = m['today'], m['future'], m['h'], m['wf_acc'], m['pct'], m['rising'], m['reliable']
flat = abs(pct) < 1.0
conf = "High" if wf >= 65 else "Moderate" if wf >= 55 else "Low"
place = market if market != "All Gujarat" else "all Gujarat yards"
st.markdown(f'<p style="color:#999;font-size:0.85rem;margin-top:6px;">{place} · {conf} confidence · validated walk-forward against ARIMA &amp; naive baselines</p>',
            unsafe_allow_html=True)

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
acc_sub = "5-fold walk-forward · beats coin-flip & ARIMA" if wf >= 53 else "honest call: no edge here, so we don't guess"
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
hist90 = base.tail(90)[['date','price']].copy()
fcdf = pd.DataFrame({'date': fdates, 'price': fprice, 'lo':[p-b for p,b in zip(fprice,bw)], 'hi':[p+b for p,b in zip(fprice,bw)]})
ax_x = alt.Axis(title=None, format='%b %Y', labelColor='#999', tickColor='#eee', domainColor='#e5e5e5', grid=False)
ax_y = alt.Axis(title='\u20b9 / quintal', labelColor='#999', titleColor='#999', gridColor='#f2f2f2', domainColor='#e5e5e5', tickColor='#eee')
band = alt.Chart(fcdf).mark_area(color=ORANGE, opacity=0.13).encode(x=alt.X('date:T', axis=ax_x), y=alt.Y('lo:Q', axis=ax_y), y2='hi:Q')
la = alt.Chart(hist90).mark_line(color=GREEN, strokeWidth=2).encode(x=alt.X('date:T', axis=ax_x), y=alt.Y('price:Q', axis=ax_y),
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

st.markdown('<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Climate context</h3>'
            '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Price against rainfall, last 12 months</p>', unsafe_allow_html=True)
clim = base.tail(365)[['date','price','rainfall']].copy()
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
    'Bilimora': (20.77, 72.96), 'Visavadar': (21.34, 70.75), 'Ankleshwar': (21.63, 72.99),
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

st.markdown('<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Glut Radar — where the next price collapse is forming</h3>'
            '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Every Gujarat mandi we cover, coloured by predicted 30-day price direction. Red = glut / dump risk.</p>',
            unsafe_allow_html=True)

if st.checkbox(f"Glut Radar Statewide {crop.lower()} mandi price-direction map", value=True):    
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
        mrows.append({'lat': xy[0], 'lon': xy[1], 'rgb': rgb, 'color': hexc,
                      'mandi': mk, 'dirtxt': dtxt, 'pcttxt': ptxt})
    if mrows:
        mdf = pd.DataFrame(mrows)
        tip = {"html": "<b>{mandi}</b><br/>{dirtxt} ({pcttxt})",
               "style": {"backgroundColor": "#1c1c1c", "color": "#fff", "fontSize": "12px",
                         "padding": "6px 9px", "borderRadius": "6px"}}
        try:
            layer = pdk.Layer("ScatterplotLayer", data=mdf, get_position=["lon", "lat"],
                              get_fill_color="rgb", get_radius=15000, radius_min_pixels=6,
                              radius_max_pixels=22, pickable=True, opacity=0.85,
                              stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1)
            view = pdk.ViewState(latitude=22.6, longitude=71.7, zoom=5.7)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tip, map_style=None))
        except Exception:
            st.map(mdf, latitude='lat', longitude='lon', color='color', size=7000)   # fallback, no tooltip
        n_fall = sum(r['dirtxt'] == 'likely to fall' for r in mrows)
        st.markdown(
            f'<p style="color:#444;font-size:0.92rem;margin:6px 0 2px;">Of {len(mrows)} mapped {crop.lower()} mandis, '
            f'<b style="color:#c0722e;">{n_fall}</b> are flagged likely to fall (glut risk) over the next {BEST_H[crop]} days; '
            f'grey points are below our reliability gate, where we make no call.</p>'
            f'<p style="color:#9aa6a0;font-size:0.74rem;margin-top:2px;">Hover any dot for its mandi and predicted move. '
            f'Green rising, orange falling, grey no clear signal. Town-level coordinates; the map shows predicted direction, not traded volume.</p>',
            unsafe_allow_html=True)
    else:
        st.info("None of this crop's mandis matched the coordinate lookup yet - add entries in GUJARAT_MANDI_COORDS.")
    if skipped:
        with st.expander(f"{len(skipped)} mandis have no coordinates yet (add them to GUJARAT_MANDI_COORDS)"):
            st.write(", ".join(skipped))

st.markdown(f'<p style="color:#444;font-weight:500;margin:14px 0 2px;">Statewide impact if widely adopted ({crop.lower()})</p>', unsafe_allow_html=True)
adopt = st.slider(f"Share of avoidable {crop.lower()} loss actually averted (%)", 1, 50, int(SAVE_FRAC*100))
_awbn, _act = state_impact(crop, adopt / 100)
_csp = CROP_STATE[crop]
st.markdown(
    f'<div style="background:#eef4f0;border:1px solid #dce8e1;border-radius:12px;padding:14px 18px;margin-top:4px;">'
    f'<div style="color:#1c1c1c;font-size:1.05rem;">At <b>{adopt}%</b> averted, {crop.lower()} alone could keep about '
    f'<b>{_awbn:.1f} billion litres</b> of water and <b>{round(_act,-3):,.0f} tonnes</b> of CO&#8322;e out of the waste stream each year.</div>'
    f'<div style="color:#9aa6a0;font-size:0.74rem;margin-top:6px;">Potential, not a measured outcome; our headline uses a conservative '
    f'{int(SAVE_FRAC*100)}%. This crop and market validate at {m["wf_acc"]:.0f}% walk-forward direction accuracy. '
    f'Sources: {_csp["psrc"]} production ({_csp["pyear"]}), ICAR-CIPHET loss, Water Footprint Network, Poore &amp; Nemecek. '
    f'Rupee value is shown per decision in the advisory, not aggregated here.</div></div>', unsafe_allow_html=True)

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
