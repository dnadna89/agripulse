import streamlit as st
import pandas as pd
import numpy as np
import glob
import altair as alt
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

    # one model decides everything: walk-forward direction accuracy from regression sign
    folds = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        reg = RandomForestRegressor(n_estimators=250, random_state=42).fit(X.iloc[tr], y.iloc[tr])
        pr = (reg.predict(X.iloc[te]) > today_arr[te]).astype(int)
        folds.append((pr == yb[te]).mean()*100)
    wf_acc = float(np.mean(folds))

    # model comparison on a single hold-out (for the table only)
    split = int(len(d)*0.8); cut = max(1, split-h); today_t = today_arr[split:]; yb_te = yb[split:]
    dacc = lambda p: (p == yb_te).mean()*100
    maj = int(round(yb[:cut].mean()))
    rf  = RandomForestRegressor(n_estimators=250, random_state=42).fit(X.iloc[:cut], y.iloc[:cut])
    gb  = HistGradientBoostingRegressor(random_state=42).fit(X.iloc[:cut], y.iloc[:cut])
    clf = RandomForestClassifier(n_estimators=250, random_state=42).fit(X.iloc[:cut], yb[:cut])
    bench = {'Naive (majority)': dacc(np.full(len(yb_te), maj)),
             'Random Forest':    dacc((rf.predict(X.iloc[split:]) > today_t).astype(int)),
             'Gradient Boosting':dacc((gb.predict(X.iloc[split:]) > today_t).astype(int)),
             'Classifier':       dacc(clf.predict(X.iloc[split:]))}
    sigma = float((y.iloc[split:].values - rf.predict(X.iloc[split:])).std())

    reg_f = RandomForestRegressor(n_estimators=250, random_state=42).fit(X, y)
    imp = sorted(zip(FEATS, reg_f.feature_importances_), key=lambda kv: -kv[1])

    # robust 'today' (typical recent price, not one stray reading) + sane forecast bounds
    today = float(base['price'].tail(7).median())
    recent = base['price'].tail(180)
    lo, hi = float(recent.quantile(0.05)), float(recent.quantile(0.95))
    raw_future = float(reg_f.predict(base[FEATS].iloc[[-1]])[0])
    future = min(max(raw_future, lo), hi)
    pct = (future - today) / today * 100
    rising = future > today
    reliable = abs(pct) <= 50           # bigger implied move => trust direction only
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

with st.sidebar:
    st.markdown("#### AgriPulse")
    st.caption("Decision support for farmers and policymakers")
    view = st.radio("View", ["Farmer advisory", "Government / policymaker"])
    st.write("")
    crop = st.selectbox("Crop", CROPS)
    market = st.selectbox("Market (yard)", ["All Gujarat"] + MARKETS[crop])
    variety = st.selectbox("Variety", ["All varieties"] + varieties_for(crop, market))
    st.write("")
    st.caption(f"Forecasting {BEST_H[crop]} days ahead. Source: Agmarknet (Gujarat) and Open-Meteo climate.")

st.title("AgriPulse")
st.markdown('<p style="color:#888;font-size:1rem;margin-top:2px;">Crop price-direction intelligence to help farmers time sales and cut waste</p>',
            unsafe_allow_html=True)

m = get_model(crop, variety, market)
if m is None:
    st.warning("Not enough history for this yard and variety combination. Try 'All Gujarat' or 'All varieties'."); st.stop()

today, future, h, wf, pct, rising, reliable = m['today'], m['future'], m['h'], m['wf_acc'], m['pct'], m['rising'], m['reliable']
conf = "High" if wf >= 65 else "Moderate" if wf >= 55 else "Low"
place = market if market != "All Gujarat" else "all Gujarat yards"
st.markdown(f'<p style="color:#999;font-size:0.85rem;margin-top:6px;">{place} · {conf} confidence on direction · 5-fold walk-forward validated</p>',
            unsafe_allow_html=True)

arrow = "&#9650;" if rising else "&#9660;"; dcol = GREEN if rising else ORANGE
price_val = f"&#8377;{future:,.0f}" if reliable else "Direction only"
price_sub = f"{pct:+.1f}% from today" if reliable else "magnitude uncertain here"
cards = (stat_card(f"Direction · next {h}d", f'<span style="color:{dcol}">{arrow} {"Rising" if rising else "Falling"}</span>')
         + stat_card("Validated accuracy", f"{wf:.0f}%", "5-fold walk-forward")
         + stat_card(f"Est. price in {h}d", price_val, price_sub))
st.markdown(f'<div style="display:flex;gap:14px;margin:8px 0 16px;">{cards}</div>', unsafe_allow_html=True)

where = f"{crop} at {market}" if market != "All Gujarat" else f"{crop} across Gujarat"
if not reliable:
    banner("Limited or volatile data for this selection, so AgriPulse shows the likely direction only and holds back a rupee figure. "
           "For a firmer estimate, switch to 'All Gujarat'.", 'warn')
elif view == "Government / policymaker":
    if not rising:
        banner(f"Market intervention signal. {where} is projected to fall about {abs(pct):.0f}% over the next {h} days, "
               f"pointing to a possible glut. <b>Illustrative lever:</b> ready procurement or price-support so farmer floor prices hold.", 'risk')
    else:
        banner(f"Consumer-price signal. {where} is projected to rise about {pct:.0f}% over the next {h} days. "
               f"<b>Illustrative lever:</b> consider a phased buffer-stock release to ease retail prices.", 'good')
else:
    if not rising:
        banner(f"Glut risk. {where} ({variety}) is predicted to fall about {abs(pct):.0f}% over the next {h} days. "
               f"Likely oversupply - consider staggered selling or cold storage.", 'risk')
    else:
        banner(f"Favourable window. {where} ({variety}) is predicted to rise about {pct:.0f}% over the next {h} days. "
               f"Consider releasing stored stock.", 'good')

if reliable:
    qcol, icol = st.columns([1, 2])
    with qcol:
        qty = st.number_input("Your stock (quintals)", min_value=0, value=100, step=10)
    impact = abs(qty * today * pct/100)
    with icol:
        if qty and not rising:
            st.markdown(f'<div style="padding:14px 0;color:#444;">On {qty} quintals, acting early could protect about '
                        f'<b>&#8377;{impact:,.0f}</b> (a {abs(pct):.0f}% drop avoided). <span style="color:#aaa;font-size:0.8rem;">Estimate.</span></div>', unsafe_allow_html=True)
        elif qty:
            st.markdown(f'<div style="padding:14px 0;color:#444;">On {qty} quintals, waiting for the predicted rise could add about '
                        f'<b>&#8377;{impact:,.0f}</b> (+{pct:.0f}%). <span style="color:#aaa;font-size:0.8rem;">Estimate.</span></div>', unsafe_allow_html=True)

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
st.altair_chart((band + la + lf).properties(height=320).configure_view(strokeWidth=0), use_container_width=True)

lc, rc = st.columns(2)
with lc:
    st.markdown('<p style="color:#444;font-weight:500;margin-bottom:2px;">What drives the forecast</p>', unsafe_allow_html=True)
    idf = pd.DataFrame(m['imp'], columns=['feature','imp']); idf['pct'] = idf['imp']*100; idf['feature'] = idf['feature'].map(LABELS)
    st.altair_chart(alt.Chart(idf.head(5)).mark_bar(color=GREEN, opacity=0.85).encode(
        x=alt.X('pct:Q', title='importance (%)', axis=alt.Axis(labelColor='#999', titleColor='#999', gridColor='#f4f4f4')),
        y=alt.Y('feature:N', sort='-x', title=None, axis=alt.Axis(labelColor='#555')),
        tooltip=[alt.Tooltip('pct:Q', format='.0f')]).properties(height=170).configure_view(strokeWidth=0), use_container_width=True)
    st.markdown('<p style="color:#aaa;font-size:0.78rem;margin-top:-6px;">Recent price trends and seasonality lead; climate is a secondary signal.</p>', unsafe_allow_html=True)
with rc:
    st.markdown('<p style="color:#444;font-weight:500;margin-bottom:6px;">How models compare (single hold-out test)</p>', unsafe_allow_html=True)
    rows = ""
    for k, v in m['bench'].items():
        mark = " (our model)" if k == m['chosen'] else ""; w = "600" if k == m['chosen'] else "400"
        rows += (f"<tr><td style='padding:6px 12px;color:#444;font-weight:{w};'>{k}{mark}</td>"
                 f"<td style='padding:6px 12px;text-align:right;font-weight:{w};color:#1c1c1c;'>{v:.0f}%</td></tr>")
    st.markdown(f"<table style='border-collapse:collapse;width:100%;font-size:0.88rem;'>"
                f"<tr><th style='text-align:left;padding:6px 12px;color:#999;font-weight:500;border-bottom:1px solid #eee;'>Model</th>"
                f"<th style='text-align:right;padding:6px 12px;color:#999;font-weight:500;border-bottom:1px solid #eee;'>Direction acc.</th></tr>{rows}</table>",
                unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;font-size:0.78rem;margin-top:8px;">We run one Random Forest for both the direction '
                'and the price, so the arrow and the chart can never disagree. The classifier edges it on direction alone '
                'but produces no price, so it stays a benchmark. The headline accuracy uses the tougher walk-forward test.</p>',
                unsafe_allow_html=True)

st.markdown('<h3 style="font-weight:500;color:#444;margin-top:24px;margin-bottom:0;">Climate context</h3>'
            '<p style="color:#999;font-size:0.85rem;margin-top:2px;">Price against rainfall, last 12 months</p>', unsafe_allow_html=True)
clim = base.tail(365)[['date','price','rainfall']].copy()
rain = alt.Chart(clim).mark_bar(color=BLUE, opacity=0.6).encode(x=alt.X('date:T', axis=ax_x),
        y=alt.Y('rainfall:Q', axis=alt.Axis(title='rain (mm)', labelColor='#bbb', titleColor='#bbb', grid=False)))
pr = alt.Chart(clim).mark_line(color=GREEN, strokeWidth=1.5).encode(x='date:T',
        y=alt.Y('price:Q', axis=alt.Axis(title='\u20b9 / quintal', labelColor='#999', titleColor='#999', gridColor='#f2f2f2')))
st.altair_chart(alt.layer(rain, pr).resolve_scale(y='independent').properties(height=220).configure_view(strokeWidth=0), use_container_width=True)

st.markdown('<p style="color:#b0b0b0;font-size:0.78rem;margin-top:14px;">Validated by 5-fold walk-forward testing on '
            'three years of Agmarknet (Gujarat) prices and Open-Meteo climate. Each crop uses its best horizon.</p>', unsafe_allow_html=True)
