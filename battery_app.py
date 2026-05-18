"""
Battery EMS — Application Streamlit interactive
Optimisation arbitrage spot price France (EPEX)

Lancement :  streamlit run battery_app.py
"""

import base64
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linprog
import streamlit as st

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery EMS — Arbitrage Spot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = Path(__file__).parent
DATA_CSV   = SCRIPT_DIR / "data_France_historical-spot_price_hourly.csv"
LOGO_PATH  = SCRIPT_DIR / "power_capture_logo.png"

# Fixed simulation start date (data coverage: 1.1.2025 – 15.04.2026)
START_DATE = date(2025, 1, 1)


def _render_header():
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        logo_tag = f'<img src="data:image/png;base64,{logo_b64}" style="height:64px;margin-right:18px;vertical-align:middle;">'
    else:
        logo_tag = ""
    title_tag = '<span style="font-size:2rem;font-weight:700;line-height:1.2;">BESS Intra day Market Modelisation</span>'
    subtitle   = '<p style="font-style:italic;font-size:0.82em;color:#555;margin-top:2px;">Data EPEX Spot &nbsp;(Période&nbsp;: 1.1.2025 – 15.04.2026)</p>'
    st.markdown(
        f'<div style="display:flex;align-items:center;margin-bottom:0;">{logo_tag}{title_tag}</div>{subtitle}',
        unsafe_allow_html=True,
    )


# ── Fonctions de simulation ───────────────────────────────────────────────────

def optimize_schedule(prices, soc_init_kwh, capacity_kwh, p_max,
                      eta_c, eta_d, soc_min_pct, soc_max_pct, agg_spread):
    T = len(prices)
    if T == 0:
        return np.zeros(0), np.zeros(0)
    resale      = prices + agg_spread
    soc_min_kwh = soc_min_pct * capacity_kwh
    soc_max_kwh = soc_max_pct * capacity_kwh
    c_obj       = np.concatenate([prices, -resale])
    A_rows, b_rows = [], []
    for t in range(T):
        row = np.zeros(2 * T)
        row[:t+1]    =  eta_c
        row[T:T+t+1] = -1.0 / eta_d
        A_rows.append(row);  b_rows.append(soc_max_kwh - soc_init_kwh)
        A_rows.append(-row); b_rows.append(soc_init_kwh - soc_min_kwh)
    for t in range(T):
        row = np.zeros(2 * T)
        row[t]     = 1.0
        row[T + t] = 1.0
        A_rows.append(row)
        b_rows.append(p_max)
    bounds = [(0.0, p_max)] * T + [(0.0, p_max)] * T
    res = linprog(c_obj, A_ub=np.array(A_rows), b_ub=np.array(b_rows),
                  bounds=bounds, method='highs')
    if res.status != 0:
        return np.zeros(T), np.zeros(T)
    ch = np.clip(res.x[:T], 0.0, p_max)
    di = np.clip(res.x[T:],  0.0, p_max)
    ch[ch < 0.1] = 0.0
    di[di < 0.1] = 0.0
    for t in range(T):
        if ch[t] > 0.0 and di[t] > 0.0:
            delta = ch[t] * eta_c - di[t] / eta_d
            if delta >= 0.0:
                ch[t] = min(delta / eta_c, p_max); di[t] = 0.0
            else:
                ch[t] = 0.0; di[t] = min(-delta * eta_d, p_max)
    return ch, di


def run_simulation(df_sim, params, progress_cb=None):
    records        = []
    soc_kwh        = params['soc_init_pct'] * params['capacity_kwh']
    cap_kwh        = params['capacity_kwh']
    total_fec      = 0.0
    eta_c          = params['eta_c']
    eta_d          = params['eta_d']
    min_spread_kwh = params.get('min_discharge_spread_kwh', 0.0)
    last_charge_px = None

    dates  = sorted(set(df_sim.index.date))
    day_px = {d: df_sim[df_sim.index.date == d]['price_eur_kwh'].values for d in dates}
    n_days = len(dates)

    for idx_day, day in enumerate(dates):
        p_today = day_px[day]
        if len(p_today) != 24:
            continue

        p_max   = min(params['c_rate'] * cap_kwh, params['connection_kw'])
        soc_min = params['soc_min_pct'] * cap_kwh
        soc_max = params['soc_max_pct'] * cap_kwh
        kw = dict(capacity_kwh=cap_kwh, p_max=p_max, eta_c=eta_c, eta_d=eta_d,
                  soc_min_pct=params['soc_min_pct'], soc_max_pct=params['soc_max_pct'],
                  agg_spread=params['agg_spread'])

        ch_am, di_am = optimize_schedule(p_today, soc_kwh, **kw)
        ch_pm = di_pm = None
        ts_idx = df_sim[df_sim.index.date == day].index

        for h in range(24):
            if h == 13:
                tomorrow = day + timedelta(days=1)
                p_pm = (np.concatenate([p_today[13:], day_px[tomorrow]])
                        if tomorrow in day_px and len(day_px[tomorrow]) == 24
                        else p_today[13:])
                ch_pm, di_pm = optimize_schedule(p_pm, soc_kwh, **kw)

            price = float(p_today[h])
            ch_h  = float(ch_am[h]) if h < 13 else float(ch_pm[h - 13])
            di_h  = float(di_am[h]) if h < 13 else float(di_pm[h - 13])

            # Filtre spread minimum
            if di_h > 0 and min_spread_kwh > 0 and last_charge_px is not None:
                if price - last_charge_px < min_spread_kwh:
                    di_h = 0.0

            # Plafonnement physique SOC
            ch_h = min(ch_h, max(0.0, (soc_max - soc_kwh) / eta_c))
            di_h = min(di_h, max(0.0, (soc_kwh - soc_min) * eta_d))

            # Anti-simultaneite
            if ch_h > 0.0 and di_h > 0.0:
                delta = ch_h * eta_c - di_h / eta_d
                if delta >= 0.0:
                    ch_h = min(delta / eta_c, p_max); di_h = 0.0
                else:
                    ch_h = 0.0; di_h = min(-delta * eta_d, p_max)

            if ch_h > 0:
                last_charge_px = price
            soc_kwh += ch_h * eta_c - di_h / eta_d
            soc_kwh  = float(np.clip(soc_kwh, soc_min, soc_max))

            resale   = price + params['agg_spread']
            fec_inc  = (ch_h + di_h) / (2.0 * params['capacity_kwh'])
            total_fec += fec_inc
            records.append({
                'datetime':               ts_idx[h],
                'spot_price_eur_mwh':    round(price * 1000, 4),
                'charge_from_grid_kwh':  round(ch_h, 3),
                'discharge_to_grid_kwh': round(di_h, 3),
                'soc_pct':               round(soc_kwh / cap_kwh * 100, 2),
                'capacity_kwh':          round(cap_kwh, 3),
                'purchase_cost_eur':     round(ch_h * price, 4),
                'resale_revenue_eur':    round(di_h * resale, 4),
                'net_revenue_eur':       round(di_h * resale - ch_h * price, 4),
                'cumulative_fec':        round(total_fec, 4),
            })

        cap_kwh = params['capacity_kwh'] * max(
            params['capacity_eol'],
            1.0 - params['aging_per_fec'] * total_fec)

        if progress_cb and idx_day % 10 == 0:
            progress_cb((idx_day + 1) / n_days)

    return pd.DataFrame(records).set_index('datetime')


@st.cache_data(show_spinner="Chargement des donnees spot price...")
def load_spot_data():
    df_raw = pd.read_csv(DATA_CSV, sep=';')
    df_raw['datetime'] = pd.to_datetime(
        df_raw['Datetime (Local)'], format='%d.%m.%Y %H:%M')
    df_raw['price_eur_kwh'] = (
        pd.to_numeric(df_raw['Spot Price (EUR/MWhe)'], errors='coerce') / 1000.0)
    df = (df_raw[['datetime', 'price_eur_kwh']]
          .dropna().sort_values('datetime')
          .drop_duplicates('datetime').set_index('datetime'))
    return df


# ── Sidebar — parametres ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parametres")

    with st.expander("Batterie", expanded=True):
        capacity_kwh  = st.number_input("Capacite nominale (kWh)", 10.0, 100_000.0, 1000.0, 100.0)
        c_rate        = st.slider("C-rate (h⁻¹)", 0.10, 2.00, 0.50, 0.05, format="%.2f")
        connection_kw = st.number_input("Raccordement Enedis (kW)", 10.0, 100_000.0, 1000.0, 100.0)
        eff_rt        = st.slider("Rendement aller-retour", 0.70, 0.999, 0.970, 0.005, format="%.3f")

    with st.expander("Etat de charge (SOC)", expanded=True):
        soc_min_pct  = st.slider("SOC minimum (%)",  0,  30, 10, 1) / 100
        soc_max_pct  = st.slider("SOC maximum (%)", 70, 100, 90, 1) / 100
        soc_init_pct = st.slider("SOC initial (%)",  10, 90, 50, 5) / 100

    with st.expander("Vieillissement & marche", expanded=False):
        aging_per_fec  = st.number_input("Vieillissement / FEC", value=2e-4, format="%.2e", step=1e-5)
        capacity_eol   = st.slider("Capacite fin de vie (%)", 50, 95, 80, 5) / 100
        agg_spread     = st.number_input("Spread agregateur (EUR/kWh)", value=0.0, format="%.4f")
        min_spread_mwh = st.slider("Spread min decharge (EUR/MWh)", 0, 50, 15, 1)

    start_date = START_DATE

    eta_c = eff_rt ** 0.5
    eta_d = eff_rt ** 0.5
    p_max = min(c_rate * capacity_kwh, connection_kw)

    st.markdown(f"""
    <div style='background:#f0f4f8;padding:8px 12px;border-radius:6px;font-size:0.85em'>
    <b>P_max</b> = {p_max:,.0f} kW &nbsp;|&nbsp;
    <b>eta_c=eta_d</b> = {eta_c:.4f}<br>
    SOC utile = {(soc_max_pct-soc_min_pct)*capacity_kwh:,.0f} kWh
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    run_btn = st.button("▶  LANCER LA SIMULATION", type="primary", use_container_width=True)

# ── Chargement des donnees ────────────────────────────────────────────────────
df_spot = load_spot_data()

# ── Lancement simulation ──────────────────────────────────────────────────────
if run_btn:
    df_sim = df_spot[df_spot.index.date >= start_date].copy()
    if len(df_sim) == 0:
        st.error("Aucune donnee disponible pour la periode choisie.")
    else:
        params = dict(
            capacity_kwh=capacity_kwh, c_rate=c_rate, connection_kw=connection_kw,
            eta_c=eta_c, eta_d=eta_d, soc_min_pct=soc_min_pct, soc_max_pct=soc_max_pct,
            soc_init_pct=soc_init_pct, agg_spread=agg_spread,
            aging_per_fec=aging_per_fec, capacity_eol=capacity_eol,
            min_discharge_spread_kwh=min_spread_mwh / 1000.0,
        )
        prog_bar = st.progress(0, text="Simulation en cours...")
        t0      = time.time()
        results = run_simulation(df_sim, params, progress_cb=lambda p: prog_bar.progress(p))
        prog_bar.progress(1.0, text=f"Termine en {time.time()-t0:.1f}s — {len(results):,} heures")

        st.session_state['results'] = results
        st.session_state['params']  = {**params, 'p_max': p_max}
        st.session_state['spot']    = df_spot

# ── Affichage des resultats ───────────────────────────────────────────────────
if 'results' not in st.session_state:
    _render_header()
    st.info("Definissez les parametres dans la barre laterale, puis cliquez **Lancer la simulation**.")
    st.stop()

results = st.session_state['results']
params  = st.session_state['params']
df_spot_full = st.session_state['spot']

# ── KPI globaux ───────────────────────────────────────────────────────────────
total_rev  = results['resale_revenue_eur'].sum()
total_cost = results['purchase_cost_eur'].sum()
net_profit = results['net_revenue_eur'].sum()
n_days     = results.index.normalize().nunique()
avg_daily  = net_profit / n_days if n_days else 0.0
cap_final  = results['capacity_kwh'].iloc[-1]
fec_total  = results['cumulative_fec'].iloc[-1]
cap_loss   = (1 - cap_final / capacity_kwh) * 100

_render_header()
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Contribution Brut",     f"{net_profit:,.0f} EUR")
c2.metric("Contribution / jour",   f"{avg_daily:,.1f} EUR/j")
c3.metric("Contribution annuelle", f"{avg_daily*365:,.0f} EUR/an")
c4.metric("FEC cumules",      f"{fec_total:.0f}")
c5.metric("Capacite finale",  f"{cap_final:,.0f} kWh")
c6.metric("Perte capacite",   f"{cap_loss:.2f} %")

st.markdown("---")

# ── Onglets principaux ────────────────────────────────────────────────────────
tab_monthly, tab_daily, tab_spot, tab_params = st.tabs(
    ["📊 Revenus mensuels", "📅 Detail journalier", "📈 Profil spot price", "🔧 Parametres"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 : Revenus mensuels
# ════════════════════════════════════════════════════════════════════════════════
with tab_monthly:
    monthly = results.resample('ME').agg(
        revenue    = ('resale_revenue_eur',    'sum'),
        cost       = ('purchase_cost_eur',     'sum'),
        profit     = ('net_revenue_eur',       'sum'),
        charge_kwh = ('charge_from_grid_kwh',  'sum'),
        vente_kwh  = ('discharge_to_grid_kwh', 'sum'),
        fec        = ('cumulative_fec',        'last'),
    ).round(2)
    monthly.index = monthly.index.strftime('%Y-%m')
    monthly['cumprofit'] = monthly['profit'].cumsum().round(0)
    monthly['fec_mois']  = monthly['fec'].diff().fillna(monthly['fec'].iloc[0]).round(1)

    fig_m = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ['steelblue' if v >= 0 else 'tomato' for v in monthly['profit']]
    fig_m.add_trace(go.Bar(
        x=monthly.index, y=monthly['profit'], name='Contribution mensuelle',
        marker_color=colors, text=monthly['profit'].apply(lambda v: f"{v:,.0f}"),
        textposition='outside'), secondary_y=False)
    fig_m.add_trace(go.Scatter(
        x=monthly.index, y=monthly['cumprofit'], name='Cumul',
        line=dict(color='darkorange', width=2.5)), secondary_y=True)
    fig_m.update_layout(height=380, margin=dict(t=20, b=20),
                         legend=dict(orientation='h', y=1.08))
    fig_m.update_yaxes(title_text="Contribution mensuelle (EUR)", secondary_y=False)
    fig_m.update_yaxes(title_text="Contribution cumulee (EUR)", secondary_y=True)
    st.plotly_chart(fig_m, use_container_width=True)

    # Tableau mensuel
    tbl = monthly[['revenue', 'cost', 'profit', 'charge_kwh', 'vente_kwh', 'fec_mois']].copy()
    tbl.columns = ['CA vente (EUR)', 'Cout achat (EUR)', 'Contribution (EUR)',
                   'Achat (kWh)', 'Vente (kWh)', 'FEC mois']
    st.dataframe(
        tbl.style.format({
            'CA vente (EUR)': '{:,.2f}', 'Cout achat (EUR)': '{:,.2f}',
            'Contribution (EUR)': '{:,.2f}', 'Achat (kWh)': '{:,.0f}',
            'Vente (kWh)': '{:,.0f}', 'FEC mois': '{:.1f}',
        }).background_gradient(subset=['Contribution (EUR)'], cmap='RdYlGn'),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 : Detail journalier
# ════════════════════════════════════════════════════════════════════════════════
with tab_daily:
    avail_dates = sorted(set(results.index.date))
    sel_date = st.date_input(
        "Selectionner une date",
        value=avail_dates[0],
        min_value=avail_dates[0],
        max_value=avail_dates[-1],
        key='date_picker',
    )

    day_str  = str(sel_date)
    day_data = results.loc[day_str]
    hours    = list(range(24))

    # KPIs du jour
    ch_d  = day_data['charge_from_grid_kwh'].sum()
    di_d  = day_data['discharge_to_grid_kwh'].sum()
    pr_d  = day_data['net_revenue_eur'].sum()
    soc_min_d = day_data['soc_pct'].min()
    soc_max_d = day_data['soc_pct'].max()
    spot_avg  = day_data['spot_price_eur_mwh'].mean()
    spot_max  = day_data['spot_price_eur_mwh'].max()
    spot_min  = day_data['spot_price_eur_mwh'].min()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Contribution jour", f"{pr_d:,.1f} EUR")
    k2.metric("Achat reseau",  f"{ch_d:,.0f} kWh")
    k3.metric("Vente reseau",  f"{di_d:,.0f} kWh")
    k4.metric("SOC min / max", f"{soc_min_d:.0f}% / {soc_max_d:.0f}%")
    k5.metric("Spot moyen",    f"{spot_avg:.1f} EUR/MWh")
    k6.metric("Spread jour",   f"{spot_max-spot_min:.1f} EUR/MWh")

    fig_d = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            f"Flux energie — {sel_date} (kWh)",
            "Prix spot (EUR/MWh)",
            "Etat de charge — SOC (%)",
        ],
        row_heights=[0.40, 0.30, 0.30],
    )

    # Flux
    fig_d.add_trace(go.Bar(
        x=hours, y=day_data['charge_from_grid_kwh'].values,
        name='Charge (achat)', marker_color='royalblue', opacity=0.85), row=1, col=1)
    fig_d.add_trace(go.Bar(
        x=hours, y=-day_data['discharge_to_grid_kwh'].values,
        name='Decharge (vente)', marker_color='tomato', opacity=0.85), row=1, col=1)
    fig_d.add_hline(y=p_max,  line_dash='dot', line_color='royalblue', opacity=0.4, row=1, col=1)
    fig_d.add_hline(y=-p_max, line_dash='dot', line_color='tomato',    opacity=0.4, row=1, col=1)
    fig_d.add_hline(y=0,      line_color='black', line_width=0.5, row=1, col=1)

    # Spot
    spot_vals = day_data['spot_price_eur_mwh'].values
    fig_d.add_trace(go.Scatter(
        x=hours, y=spot_vals, name='Spot', mode='lines+markers',
        line=dict(color='darkorange', width=2),
        marker=dict(size=5, color=spot_vals, colorscale='RdYlGn',
                    cmin=spot_vals.min(), cmax=spot_vals.max())), row=2, col=1)
    fig_d.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.4, row=2, col=1)

    # SOC
    soc_vals = day_data['soc_pct'].values
    fig_d.add_trace(go.Scatter(
        x=hours, y=soc_vals, name='SOC', fill='tozeroy',
        line=dict(color='steelblue', width=2),
        fillcolor='rgba(70,130,180,0.15)'), row=3, col=1)
    fig_d.add_hline(y=soc_min_pct * 100, line_dash='dash', line_color='red',
                    opacity=0.7, annotation_text=f"SOC min {soc_min_pct*100:.0f}%",
                    annotation_position='right', row=3, col=1)
    fig_d.add_hline(y=soc_max_pct * 100, line_dash='dash', line_color='green',
                    opacity=0.7, annotation_text=f"SOC max {soc_max_pct*100:.0f}%",
                    annotation_position='right', row=3, col=1)
    fig_d.add_vline(x=13, line_dash='dot', line_color='gray', opacity=0.6,
                    annotation_text="Re-plan 13h", annotation_position='top right')

    fig_d.update_xaxes(
        tickvals=hours, ticktext=[f"{h}h" for h in hours],
        tickangle=-45, row=3, col=1)
    fig_d.update_yaxes(title_text="kWh", row=1, col=1)
    fig_d.update_yaxes(title_text="EUR/MWh", row=2, col=1)
    fig_d.update_yaxes(title_text="%", row=3, col=1, range=[0, 100])
    fig_d.update_layout(height=640, showlegend=True,
                         legend=dict(orientation='h', y=1.04),
                         margin=dict(t=60, b=20, r=80))
    st.plotly_chart(fig_d, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 : Profil spot price
# ════════════════════════════════════════════════════════════════════════════════
with tab_spot:
    df_res_spot = results.copy()
    df_res_spot['_hour']  = df_res_spot.index.hour
    df_res_spot['_month'] = df_res_spot.index.strftime('%Y-%m')

    months = sorted(df_res_spot['_month'].unique())
    spot_by_mh = df_res_spot.groupby(['_month', '_hour'])['spot_price_eur_mwh'].mean()
    avg_profile = df_res_spot.groupby('_hour')['spot_price_eur_mwh'].mean()

    palette = [
        '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
        '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
        '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5',
        '#c49c94',
    ]

    fig_spot = go.Figure()
    all_vals = []
    for i, month in enumerate(months):
        vals = [spot_by_mh.get((month, h), 0) for h in range(24)]
        all_vals.append(vals)
        fig_spot.add_trace(go.Scatter(
            x=list(range(24)), y=vals, name=month, mode='lines',
            line=dict(color=palette[i % len(palette)], width=1.2), opacity=0.75))

    all_arr = np.array(all_vals)
    fig_spot.add_trace(go.Scatter(
        x=list(range(24)), y=avg_profile.values, name='Moyenne globale',
        line=dict(color='black', width=2.5, dash='dash')))
    fig_spot.add_trace(go.Scatter(
        x=list(range(24)), y=all_arr.max(axis=0),
        name='Max mensuel', line=dict(color='red', width=1, dash='dot'), opacity=0.5))
    fig_spot.add_trace(go.Scatter(
        x=list(range(24)), y=all_arr.min(axis=0),
        name='Min mensuel', line=dict(color='blue', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(100,149,237,0.08)', opacity=0.5))
    fig_spot.add_vline(x=13, line_dash='dot', line_color='gray', opacity=0.5,
                        annotation_text="13h — publication J+1")
    fig_spot.update_xaxes(tickvals=list(range(24)),
                           ticktext=[f"{h}h" for h in range(24)], tickangle=-45)
    fig_spot.update_yaxes(title_text="EUR/MWh")
    fig_spot.update_layout(height=450, title="Profil horaire moyen par mois",
                            legend=dict(orientation='h', y=-0.25),
                            margin=dict(t=40, b=80))
    st.plotly_chart(fig_spot, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 : Parametres de simulation
# ════════════════════════════════════════════════════════════════════════════════
with tab_params:
    eol_reached = cap_final / capacity_kwh <= params['capacity_eol'] + 0.001
    ratio = results['charge_from_grid_kwh'].sum() / max(results['discharge_to_grid_kwh'].sum(), 1)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Batterie")
        st.table(pd.DataFrame({
            "Parametre": ["Capacite nominale", "Capacite finale", "Perte capacite",
                          "Fin de vie atteinte", "P_max", "C-rate", "Raccordement Enedis",
                          "Rendement RT", "Vieillissement/FEC"],
            "Valeur": [
                f"{capacity_kwh:,.0f} kWh",
                f"{cap_final:,.0f} kWh",
                f"{cap_loss:.2f} %",
                "OUI ⚠️" if eol_reached else "Non",
                f"{p_max:,.0f} kW",
                f"{c_rate:.2f} h⁻¹",
                f"{connection_kw:,.0f} kW",
                f"{eff_rt*100:.1f} %",
                f"{aging_per_fec:.2e} /FEC",
            ],
        }).set_index("Parametre"))

    with col_b:
        st.subheader("Exploitation")
        st.table(pd.DataFrame({
            "Parametre": ["Periode", "Duree", "SOC plage", "FEC cumules",
                          "Spread min decharge", "Spread agregateur",
                          "Ratio achat/vente", "Contribution/FEC"],
            "Valeur": [
                f"{results.index[0].date()} → {results.index[-1].date()}",
                f"{n_days} jours",
                f"{soc_min_pct*100:.0f}% – {soc_max_pct*100:.0f}%",
                f"{fec_total:.1f}",
                f"{min_spread_mwh} EUR/MWh",
                f"{agg_spread*1000:+.2f} mEUR/kWh",
                f"{ratio:.4f}  (attendu {1/eff_rt:.4f})",
                f"{net_profit/max(fec_total,1):,.2f} EUR/FEC",
            ],
        }).set_index("Parametre"))
