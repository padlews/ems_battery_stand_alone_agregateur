"""
Battery EMS + PV — Application Streamlit interactive
Optimisation arbitrage spot price France (EPEX) avec generateur PV

Lancement :  streamlit run battery_PV_app.py
"""

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linprog
import streamlit as st

try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BESS + PV — Arbitrage Spot",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = Path(__file__).parent
DATA_CSV   = SCRIPT_DIR / "data_France_historical-spot_price_hourly.csv"
LOGO_PATH  = SCRIPT_DIR / "power_capture_logo.png"

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.05rem !important; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)

START_DATE = date(2025, 1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MODELE SOLAIRE
# ══════════════════════════════════════════════════════════════════════════════

def _solar_params(doy: int, lat_rad: float):
    """Retourne (H0_normalise, duree_jour_h, declinaison_rad)."""
    decl  = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    cos_ha = float(np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0))
    ha_rad = np.arccos(cos_ha)
    ha_deg = np.degrees(ha_rad)
    day_length = 2.0 * ha_deg / 15.0
    H0 = max(
        np.cos(lat_rad) * np.cos(decl) * np.sin(ha_rad) +
        ha_rad * np.sin(lat_rad) * np.sin(decl),
        0.0,
    )
    return float(H0), day_length, float(decl)


def _solar_noon_local(doy: int, longitude: float, month: int) -> float:
    """Midi solaire en heure legale France (UTC+1 hiver / UTC+2 ete)."""
    B       = np.radians(360 / 365 * (doy - 81))
    eot_min = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    noon_utc = 12.0 - longitude / 15.0 - eot_min / 60.0
    tz = 2 if 4 <= month <= 10 else 1
    return noon_utc + tz


@st.cache_data(show_spinner=False)
def compute_pv_profile(
    capacity_kwp: float,
    specific_yield: float,
    latitude: float,
    longitude: float,
    dt_index_tuple: tuple,
) -> np.ndarray:
    """
    Profil horaire de production PV (kWh/h).

    Distribution saisonniere : rayonnement extraterrestre H0(latitude, DOY).
    Distribution intraday    : Gaussienne centree sur le midi solaire local.
    Energie annuelle totale  : capacity_kwp x specific_yield kWh.
    """
    dt_index     = pd.DatetimeIndex(dt_index_tuple)
    annual_kwh   = capacity_kwp * specific_yield
    lat_rad      = np.radians(latitude)

    # H0 par jour de l'annee (DOY 1-365)
    H0_doy = np.zeros(366)
    for doy in range(1, 366):
        H0_doy[doy], _, _ = _solar_params(doy, lat_rad)
    H0_mean = H0_doy[1:366].mean()
    if H0_mean < 1e-9:
        return np.zeros(len(dt_index))

    pv     = np.zeros(len(dt_index))
    dates  = np.array([ts.date() for ts in dt_index])

    for d in sorted(set(dates)):
        mask    = dates == d
        indices = np.where(mask)[0]
        hours   = np.array([dt_index[i].hour for i in indices])
        doy     = dt_index[indices[0]].timetuple().tm_yday
        month   = dt_index[indices[0]].month

        H0, day_length, _ = _solar_params(doy, lat_rad)
        if day_length < 0.5:
            continue

        solar_noon = _solar_noon_local(doy, longitude, month)
        sigma      = max(day_length / 4.0, 0.5)
        sunrise    = solar_noon - day_length / 2.0
        sunset     = solar_noon + day_length / 2.0
        daily_kwh  = annual_kwh / 365.0 * (H0 / H0_mean)

        weights = np.array([
            np.exp(-0.5 * ((h + 0.5 - solar_noon) / sigma) ** 2)
            if sunrise < h + 0.5 < sunset else 0.0
            for h in hours
        ])
        w_sum = weights.sum()
        if w_sum > 0:
            pv[indices] = weights / w_sum * daily_kwh

    return pv


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISEUR LP AVEC PV
# ══════════════════════════════════════════════════════════════════════════════

def optimize_schedule_pv(prices, pv_kwh, soc_init_kwh, capacity_kwh,
                          p_max, eta_c, eta_d, soc_min_pct, soc_max_pct,
                          agg_spread, fit_price):
    """
    LP horizon T avec generateur PV.

    Variables x = [ch_grid(T), di(T), pv_sell(T)]
      ch_grid  : charge depuis reseau        [kWh]
      di       : decharge vers reseau        [kWh]
      pv_sell  : vente directe surplus PV    [kWh]
    Implicite :
      pv_to_bat = pv_kwh - pv_sell           [PV vers batterie]

    Contrainte puissance totale + anti-simultaneite :
      ch_grid[t] + (pv[t]-pv_sell[t]) + di[t] <= P_max
      <=>  ch_grid[t] - pv_sell[t] + di[t]   <= P_max - pv[t]
    """
    T  = len(prices)
    pv = np.asarray(pv_kwh, dtype=float)
    if T == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    resale      = prices + agg_spread
    soc_min_kwh = soc_min_pct * capacity_kwh
    soc_max_kwh = soc_max_pct * capacity_kwh

    # Objectif : min sum(ch_grid*spot - di*resale - pv_sell*fit)
    c_obj = np.concatenate([prices, -resale, -fit_price * np.ones(T)])

    # Contribution cumulee du PV disponible au SOC (constante, RHS)
    cumsum_pv = np.cumsum(eta_c * pv)

    A_rows, b_rows = [], []

    # Contraintes SOC (2T lignes)
    for t in range(T):
        row = np.zeros(3 * T)
        row[:t + 1]          =  eta_c
        row[T:T + t + 1]     = -1.0 / eta_d
        row[2*T:2*T + t + 1] = -eta_c        # pv_sell reduit pv_to_bat
        A_rows.append(row.copy());  b_rows.append(soc_max_kwh - soc_init_kwh - cumsum_pv[t])
        A_rows.append(-row.copy()); b_rows.append(soc_init_kwh - soc_min_kwh + cumsum_pv[t])

    # Anti-simultaneite + limite puissance totale
    for t in range(T):
        row = np.zeros(3 * T)
        row[t]       =  1.0
        row[T + t]   =  1.0
        row[2*T + t] = -1.0
        A_rows.append(row)
        b_rows.append(p_max - pv[t])

    bounds = (
        [(0.0, p_max)] * T +
        [(0.0, p_max)] * T +
        [(0.0, float(pv[t])) for t in range(T)]
    )

    res = linprog(c_obj, A_ub=np.array(A_rows), b_ub=np.array(b_rows),
                  bounds=bounds, method='highs')

    if res.status != 0:
        return np.zeros(T), np.zeros(T), pv.copy(), np.zeros(T)

    ch_grid = np.clip(res.x[:T],    0.0, p_max)
    di      = np.clip(res.x[T:2*T], 0.0, p_max)
    pv_sell = np.array([np.clip(res.x[2*T + t], 0.0, float(pv[t])) for t in range(T)])

    ch_grid[ch_grid < 0.1] = 0.0
    di[di < 0.1]           = 0.0
    pv_sell[pv_sell < 0.01] = 0.0
    pv_to_bat = np.maximum(pv - pv_sell, 0.0)

    # Filet anti-simultaneite post-LP
    for t in range(T):
        ch_tot = ch_grid[t] + pv_to_bat[t]
        if ch_tot > 0.0 and di[t] > 0.0:
            delta = ch_tot * eta_c - di[t] / eta_d
            if delta >= 0.0:
                scale        = min(delta / eta_c, p_max) / max(ch_tot, 1e-9)
                ch_grid[t]  *= scale
                pv_to_bat[t] = min(pv_to_bat[t] * scale, pv[t])
                pv_sell[t]   = pv[t] - pv_to_bat[t]
                di[t]        = 0.0
            else:
                ch_grid[t] = pv_to_bat[t] = 0.0
                pv_sell[t] = pv[t]
                di[t]      = min(-delta * eta_d, p_max)

    return ch_grid, di, pv_sell, pv_to_bat


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION AVEC PV
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation_pv(df_sim, params, pv_array, progress_cb=None):
    """Simulation horizon roulant avec generateur PV."""
    records        = []
    soc_kwh        = params['soc_init_pct'] * params['capacity_kwh']
    cap_kwh        = params['capacity_kwh']
    total_fec      = 0.0
    eta_c          = params['eta_c']
    eta_d          = params['eta_d']
    min_spread_kwh = params.get('min_discharge_spread_kwh', 0.0)
    fit_price      = params.get('fit_price', 0.10)
    last_charge_px = None

    dates  = sorted(set(df_sim.index.date))
    day_px = {d: df_sim[df_sim.index.date == d]['price_eur_kwh'].values for d in dates}
    pv_ser = pd.Series(pv_array, index=df_sim.index)
    day_pv = {d: pv_ser[pv_ser.index.date == d].values for d in dates}
    n_days = len(dates)

    for idx_day, day in enumerate(dates):
        p_today  = day_px[day]
        pv_today = day_pv.get(day, np.zeros(24))
        if len(p_today) != 24:
            continue
        pv_today = np.pad(pv_today, (0, max(0, 24 - len(pv_today))))[:24]

        p_max   = min(params['c_rate'] * cap_kwh, params['connection_kw'])
        soc_min = params['soc_min_pct'] * cap_kwh
        soc_max = params['soc_max_pct'] * cap_kwh

        kw = dict(
            capacity_kwh=cap_kwh, p_max=p_max, eta_c=eta_c, eta_d=eta_d,
            soc_min_pct=params['soc_min_pct'], soc_max_pct=params['soc_max_pct'],
            agg_spread=params['agg_spread'], fit_price=fit_price,
        )

        ch_g_am, di_am, pvs_am, pvb_am = optimize_schedule_pv(
            p_today, pv_today, soc_kwh, **kw)

        ch_g_pm = di_pm = pvs_pm = pvb_pm = None
        ts_idx = df_sim[df_sim.index.date == day].index

        for h in range(24):
            if h == 13:
                tomorrow = day + timedelta(days=1)
                if tomorrow in day_px and len(day_px[tomorrow]) == 24:
                    p_pm  = np.concatenate([p_today[13:],  day_px[tomorrow]])
                    pv_pm = np.concatenate([pv_today[13:], day_pv.get(tomorrow, np.zeros(24))])
                else:
                    p_pm  = p_today[13:]
                    pv_pm = pv_today[13:]
                ch_g_pm, di_pm, pvs_pm, pvb_pm = optimize_schedule_pv(
                    p_pm, pv_pm, soc_kwh, **kw)

            price = float(p_today[h])
            pv_h  = float(pv_today[h])

            if h < 13:
                ch_grid_h = float(ch_g_am[h]);   di_h = float(di_am[h])
                pv_sell_h = float(pvs_am[h]);     pv_bat_h = float(pvb_am[h])
            else:
                ch_grid_h = float(ch_g_pm[h-13]); di_h = float(di_pm[h-13])
                pv_sell_h = float(pvs_pm[h-13]);   pv_bat_h = float(pvb_pm[h-13])

            # Filtre spread minimum (seulement pour charges reseau)
            if di_h > 0 and min_spread_kwh > 0 and last_charge_px is not None:
                if price - last_charge_px < min_spread_kwh:
                    di_h = 0.0

            # Plafonnement physique SOC
            ch_tot   = ch_grid_h + pv_bat_h
            headroom = max(0.0, (soc_max - soc_kwh) / eta_c)
            if ch_tot > headroom:
                scale      = headroom / max(ch_tot, 1e-9)
                ch_grid_h *= scale
                pv_bat_h  *= scale
                pv_sell_h  = pv_h - pv_bat_h
            di_h = min(di_h, max(0.0, (soc_kwh - soc_min) * eta_d))

            # Filet anti-simultaneite
            ch_tot = ch_grid_h + pv_bat_h
            if ch_tot > 0.0 and di_h > 0.0:
                delta = ch_tot * eta_c - di_h / eta_d
                if delta >= 0.0:
                    scale      = min(delta / eta_c, p_max) / max(ch_tot, 1e-9)
                    ch_grid_h *= scale
                    pv_bat_h  *= scale
                    pv_sell_h  = pv_h - pv_bat_h
                    di_h = 0.0
                else:
                    ch_grid_h = pv_bat_h = 0.0
                    pv_sell_h = pv_h
                    di_h      = min(-delta * eta_d, p_max)

            if ch_grid_h > 0.01:
                last_charge_px = price

            soc_kwh += (ch_grid_h + pv_bat_h) * eta_c - di_h / eta_d
            soc_kwh  = float(np.clip(soc_kwh, soc_min, soc_max))

            resale   = price + params['agg_spread']
            fec_inc  = (ch_grid_h + pv_bat_h + di_h) / (2.0 * params['capacity_kwh'])
            total_fec += fec_inc

            records.append({
                'datetime':               ts_idx[h],
                'spot_price_eur_mwh':     round(price * 1000,            4),
                'pv_production_kwh':      round(pv_h,                    3),
                'pv_to_battery_kwh':      round(pv_bat_h,                3),
                'pv_surplus_sold_kwh':    round(pv_sell_h,               3),
                'charge_from_grid_kwh':   round(ch_grid_h,               3),
                'discharge_to_grid_kwh':  round(di_h,                    3),
                'soc_pct':                round(soc_kwh / cap_kwh * 100, 2),
                'capacity_kwh':           round(cap_kwh,                 3),
                'purchase_cost_eur':      round(ch_grid_h * price,       4),
                'pv_surplus_revenue_eur': round(pv_sell_h * fit_price,   4),
                'resale_revenue_eur':     round(di_h * resale,           4),
                'net_revenue_eur':        round(
                    di_h * resale - ch_grid_h * price + pv_sell_h * fit_price, 4),
                'cumulative_fec':         round(total_fec,               4),
            })

        cap_kwh = params['capacity_kwh'] * max(
            params['capacity_eol'],
            1.0 - params['aging_per_fec'] * total_fec,
        )
        if progress_cb and idx_day % 10 == 0:
            progress_cb((idx_day + 1) / n_days)

    return pd.DataFrame(records).set_index('datetime')


# ── Chargement donnees spot ───────────────────────────────────────────────────
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


def _render_header():
    st.markdown(
        '<span style="font-size:2rem;font-weight:700;line-height:1.2;">'
        'BESS + PV — Arbitrage Intraday</span>'
        '<p style="font-style:italic;font-size:0.82em;color:#555;margin-top:2px;">'
        'EPEX Spot France &nbsp;+&nbsp; Generateur PV &nbsp;(1.1.2025 – 30.04.2026)</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Parametres
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
        st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)
    st.title("Parametres")

    # ── Batterie ─────────────────────────────────────────────────────────────
    with st.expander("Batterie", expanded=True):
        capacity_kwh  = st.number_input("Capacite nominale (kWh)", 10.0, 100_000.0, 1000.0, 100.0)
        c_rate        = st.slider("C-rate (h-1)", 0.10, 2.00, 0.50, 0.05, format="%.2f")
        connection_kw = st.number_input("Raccordement Enedis (kW)", 10.0, 100_000.0, 1000.0, 100.0)
        eff_rt        = st.slider("Rendement aller-retour", 0.70, 0.999, 0.970, 0.005, format="%.3f")

    with st.expander("Etat de charge (SOC)", expanded=True):
        soc_min_pct  = st.slider("SOC minimum (%)",  0,  30,  5, 1) / 100
        soc_max_pct  = st.slider("SOC maximum (%)", 70, 100, 95, 1) / 100
        soc_init_pct = st.slider("SOC initial (%)",  10, 90, 50, 5) / 100

    with st.expander("Vieillissement & marche", expanded=False):
        aging_per_fec  = st.number_input("Vieillissement / FEC", value=1e-5, format="%.2e", step=1e-6)
        capacity_eol   = st.slider("Capacite fin de vie (%)", 50, 95, 80, 5) / 100
        agg_spread     = st.number_input("Spread agregateur (EUR/kWh)", value=0.0, format="%.4f")
        min_spread_mwh = st.slider("Spread min decharge (EUR/MWh)", 0, 50, 10, 1)

    # ── Generateur PV ─────────────────────────────────────────────────────────
    with st.expander("Generateur PV", expanded=True):
        pv_capacity_kwp   = st.number_input("Capacite PV (kWc)", 1.0, 10_000.0, 100.0, 10.0)
        pv_specific_yield = st.number_input(
            "Productible specifique (kWh/kWc/an)", 500.0, 2000.0, 1200.0, 50.0,
            help="Production annuelle par kWc installe (inclut toutes pertes).")
        pv_fit_price = st.number_input(
            "Tarif rachat surplus PV (EUR/kWh)", 0.0, 0.50, 0.10, 0.01, format="%.3f",
            help="Prix de rachat du surplus PV non stocke dans la batterie.")

        st.markdown("**Localisation du generateur (France)**")

        if HAS_FOLIUM:
            st.caption("Cliquez sur la carte pour positionner le generateur PV.")
            if 'pv_lat' not in st.session_state:
                st.session_state['pv_lat'] = 46.5
                st.session_state['pv_lon'] = 2.3

            m = folium.Map(
                location=[st.session_state['pv_lat'], st.session_state['pv_lon']],
                zoom_start=6, tiles='OpenStreetMap',
            )
            folium.Rectangle(
                bounds=[[41.0, -5.5], [51.5, 9.5]],
                color='#3388ff', weight=1, fill=False,
            ).add_to(m)
            folium.Marker(
                location=[st.session_state['pv_lat'], st.session_state['pv_lon']],
                tooltip=(f"PV : {st.session_state['pv_lat']:.3f}N, "
                         f"{st.session_state['pv_lon']:.3f}E"),
                icon=folium.Icon(color='orange', icon='sun', prefix='fa'),
            ).add_to(m)

            map_out = st_folium(m, width=300, height=280, key="pv_map",
                                returned_objects=["last_clicked"])

            if map_out and map_out.get("last_clicked"):
                st.session_state['pv_lat'] = round(map_out["last_clicked"]["lat"], 4)
                st.session_state['pv_lon'] = round(map_out["last_clicked"]["lng"], 4)

            pv_lat = st.session_state['pv_lat']
            pv_lon = st.session_state['pv_lon']
            st.caption(f"Lat : {pv_lat:.4f} N  |  Lon : {pv_lon:.4f} E")
        else:
            st.warning(
                "Carte interactive indisponible.\n"
                "Installer : `pip install streamlit-folium folium`"
            )
            pv_lat = st.number_input("Latitude (N)", 41.0, 51.5, 46.5, 0.1, format="%.2f")
            pv_lon = st.number_input("Longitude (E)", -5.5, 9.5, 2.3, 0.1, format="%.2f")

    eta_c = eff_rt ** 0.5
    eta_d = eff_rt ** 0.5
    p_max = min(c_rate * capacity_kwh, connection_kw)
    pv_annual_kwh = pv_capacity_kwp * pv_specific_yield

    st.markdown(f"""
    <div style='background:#f0f4f8;padding:8px 12px;border-radius:6px;font-size:0.85em'>
    <b>P_max batt.</b> = {p_max:,.0f} kW &nbsp;|&nbsp; <b>eta</b> = {eta_c:.4f}<br>
    <b>PV annuel</b> = {pv_annual_kwh:,.0f} kWh
    &nbsp;({pv_annual_kwh / max(capacity_kwh, 1) * 100:.0f}% cap. batt.)
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    run_btn = st.button("LANCER LA SIMULATION", type="primary", use_container_width=True)


# ── Chargement donnees ────────────────────────────────────────────────────────
df_spot = load_spot_data()

# ── Lancement simulation ──────────────────────────────────────────────────────
if run_btn:
    df_sim = df_spot[df_spot.index.date >= START_DATE].copy()
    if len(df_sim) == 0:
        st.error("Aucune donnee disponible.")
    else:
        params = dict(
            capacity_kwh=capacity_kwh, c_rate=c_rate, connection_kw=connection_kw,
            eta_c=eta_c, eta_d=eta_d, soc_min_pct=soc_min_pct, soc_max_pct=soc_max_pct,
            soc_init_pct=soc_init_pct, agg_spread=agg_spread,
            aging_per_fec=aging_per_fec, capacity_eol=capacity_eol,
            min_discharge_spread_kwh=min_spread_mwh / 1000.0,
            fit_price=pv_fit_price,
        )
        with st.spinner("Calcul du profil PV..."):
            pv_array = compute_pv_profile(
                pv_capacity_kwp, pv_specific_yield,
                pv_lat, pv_lon,
                tuple(df_sim.index),
            )

        prog = st.progress(0, text="Simulation en cours...")
        t0   = time.time()
        results = run_simulation_pv(
            df_sim, params, pv_array,
            progress_cb=lambda p: prog.progress(p),
        )
        elapsed = time.time() - t0
        prog.progress(1.0, text=f"Termine en {elapsed:.1f}s — {len(results):,} heures")

        st.session_state['results']   = results
        st.session_state['params']    = {**params, 'p_max': p_max}
        st.session_state['pv_params'] = {
            'capacity_kwp': pv_capacity_kwp, 'specific_yield': pv_specific_yield,
            'lat': pv_lat, 'lon': pv_lon,
            'fit_price': pv_fit_price, 'annual_kwh': pv_annual_kwh,
        }
        st.session_state['spot'] = df_spot

# ── Affichage resultats ───────────────────────────────────────────────────────
if 'results' not in st.session_state:
    _render_header()
    st.info("Definissez les parametres dans la barre laterale, puis cliquez **Lancer la simulation**.")
    st.stop()

results   = st.session_state['results']
params    = st.session_state['params']
pv_p      = st.session_state.get('pv_params', {})

total_rev     = results['resale_revenue_eur'].sum()
pv_rev        = results['pv_surplus_revenue_eur'].sum()
total_cost    = results['purchase_cost_eur'].sum()
net_profit    = results['net_revenue_eur'].sum()
n_days        = results.index.normalize().nunique()
avg_daily     = net_profit / n_days if n_days else 0.0
cap_final     = results['capacity_kwh'].iloc[-1]
fec_total     = results['cumulative_fec'].iloc[-1]
cap_loss      = (1 - cap_final / params['capacity_kwh']) * 100
pv_total      = results['pv_production_kwh'].sum()
pv_to_bat_tot = results['pv_to_battery_kwh'].sum()
pv_sold_tot   = results['pv_surplus_sold_kwh'].sum()
self_cons_pct = pv_to_bat_tot / max(pv_total, 1) * 100

_render_header()

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("Contribution Brut",      f"{net_profit:,.0f} EUR")
c2.metric("Contribution / jour",    f"{avg_daily:,.1f} EUR/j")
c3.metric("Estimation annuelle",    f"{avg_daily * 365:,.0f} EUR/an")
c4.metric("Rev. decharge BESS",     f"{total_rev:,.0f} EUR")
c5.metric("Rev. surplus PV (FIT)",  f"{pv_rev:,.0f} EUR")
c6.metric("Production PV totale",   f"{pv_total:,.0f} kWh")
c7.metric("Autoconsommation PV",    f"{self_cons_pct:.1f} %")
c8.metric("FEC cumules",            f"{fec_total:.0f}")

st.markdown("---")

tab_monthly, tab_daily, tab_pv, tab_spot, tab_params_tab = st.tabs([
    "Revenus mensuels", "Detail journalier",
    "Analyse PV", "Profil spot price", "Parametres",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Revenus mensuels
# ════════════════════════════════════════════════════════════════════════════════
with tab_monthly:
    monthly = results.resample('ME').agg(
        revenue_bess = ('resale_revenue_eur',     'sum'),
        revenue_pv   = ('pv_surplus_revenue_eur', 'sum'),
        cost         = ('purchase_cost_eur',      'sum'),
        profit       = ('net_revenue_eur',        'sum'),
        charge_kwh   = ('charge_from_grid_kwh',   'sum'),
        vente_kwh    = ('discharge_to_grid_kwh',  'sum'),
        pv_prod      = ('pv_production_kwh',      'sum'),
        pv_bat       = ('pv_to_battery_kwh',      'sum'),
        pv_sell      = ('pv_surplus_sold_kwh',    'sum'),
        fec          = ('cumulative_fec',         'last'),
    ).round(2)
    monthly.index      = monthly.index.strftime('%Y-%m')
    monthly['cumprofit'] = monthly['profit'].cumsum().round(0)
    monthly['fec_mois']  = monthly['fec'].diff().fillna(monthly['fec'].iloc[0]).round(1)

    fig_m = make_subplots(specs=[[{"secondary_y": True}]])
    fig_m.add_trace(go.Bar(
        x=monthly.index, y=monthly['revenue_bess'], name='Rev. BESS (decharge)',
        marker_color='steelblue', opacity=0.85,
        text=monthly['revenue_bess'].apply(lambda v: f"{v:,.0f}"),
        textposition='outside'), secondary_y=False)
    fig_m.add_trace(go.Bar(
        x=monthly.index, y=monthly['revenue_pv'], name='Rev. surplus PV',
        marker_color='gold', opacity=0.85,
        text=monthly['revenue_pv'].apply(lambda v: f"{v:,.0f}"),
        textposition='outside'), secondary_y=False)
    fig_m.add_trace(go.Bar(
        x=monthly.index, y=-monthly['cost'], name='Cout achat reseau',
        marker_color='tomato', opacity=0.7), secondary_y=False)
    fig_m.add_trace(go.Scatter(
        x=monthly.index, y=monthly['cumprofit'], name='Cumul contribution',
        line=dict(color='darkorange', width=2.5)), secondary_y=True)
    fig_m.update_layout(
        barmode='relative', height=400, margin=dict(t=30, b=20),
        legend=dict(orientation='h', y=1.12),
    )
    fig_m.update_yaxes(title_text="EUR / mois", secondary_y=False)
    fig_m.update_yaxes(title_text="Contribution cumulee (EUR)", secondary_y=True)
    st.plotly_chart(fig_m, use_container_width=True)

    tbl = monthly[['revenue_bess', 'revenue_pv', 'cost', 'profit',
                   'pv_prod', 'pv_bat', 'pv_sell', 'fec_mois']].copy()
    tbl.columns = ['Rev. BESS (EUR)', 'Rev. PV (EUR)', 'Cout achat (EUR)',
                   'Contribution (EUR)', 'Prod. PV (kWh)',
                   'PV batterie (kWh)', 'PV reseau (kWh)', 'FEC mois']
    st.dataframe(
        tbl.style.format({
            'Rev. BESS (EUR)': '{:,.2f}', 'Rev. PV (EUR)': '{:,.2f}',
            'Cout achat (EUR)': '{:,.2f}', 'Contribution (EUR)': '{:,.2f}',
            'Prod. PV (kWh)': '{:,.0f}', 'PV batterie (kWh)': '{:,.0f}',
            'PV reseau (kWh)': '{:,.0f}', 'FEC mois': '{:.1f}',
        }).background_gradient(subset=['Contribution (EUR)'], cmap='RdYlGn'),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Detail journalier
# ════════════════════════════════════════════════════════════════════════════════
with tab_daily:
    avail_dates = sorted(set(results.index.date))
    sel_date    = st.date_input(
        "Selectionner une date", value=avail_dates[0],
        min_value=avail_dates[0], max_value=avail_dates[-1], key='date_pv')

    day_data = results.loc[str(sel_date)]
    hours    = list(range(24))

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Contribution",       f"{day_data['net_revenue_eur'].sum():,.1f} EUR")
    kpi2.metric("Achat reseau",       f"{day_data['charge_from_grid_kwh'].sum():,.0f} kWh")
    kpi3.metric("Vente BESS",         f"{day_data['discharge_to_grid_kwh'].sum():,.0f} kWh")
    kpi4.metric("Production PV",      f"{day_data['pv_production_kwh'].sum():,.0f} kWh")
    kpi5.metric("PV batterie",        f"{day_data['pv_to_battery_kwh'].sum():,.0f} kWh")
    kpi6.metric("PV reseau (FIT)",    f"{day_data['pv_surplus_sold_kwh'].sum():,.0f} kWh")

    fig_d = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[
            f"PV : repartition batterie / reseau — {sel_date} (kWh)",
            "Flux reseau : charge grid / decharge BESS (kWh)",
            "Prix spot (EUR/MWh)",
            "Etat de charge SOC (%)",
        ],
        row_heights=[0.27, 0.27, 0.22, 0.24],
    )
    fig_d.add_trace(go.Bar(
        x=hours, y=day_data['pv_to_battery_kwh'].values,
        name='PV batterie', marker_color='gold', opacity=0.9), row=1, col=1)
    fig_d.add_trace(go.Bar(
        x=hours, y=day_data['pv_surplus_sold_kwh'].values,
        name='PV reseau (FIT)', marker_color='orange', opacity=0.8), row=1, col=1)
    fig_d.add_trace(go.Scatter(
        x=hours, y=day_data['pv_production_kwh'].values,
        name='Production PV', mode='lines',
        line=dict(color='saddlebrown', width=2, dash='dot')), row=1, col=1)

    fig_d.add_trace(go.Bar(
        x=hours, y=day_data['charge_from_grid_kwh'].values,
        name='Charge grid', marker_color='royalblue', opacity=0.85), row=2, col=1)
    fig_d.add_trace(go.Bar(
        x=hours, y=-day_data['discharge_to_grid_kwh'].values,
        name='Decharge BESS', marker_color='tomato', opacity=0.85), row=2, col=1)

    spot_v = day_data['spot_price_eur_mwh'].values
    fig_d.add_trace(go.Scatter(
        x=hours, y=spot_v, name='Spot', mode='lines+markers',
        line=dict(color='darkorange', width=2)), row=3, col=1)
    fig_d.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.3, row=3, col=1)

    soc_v = day_data['soc_pct'].values
    fig_d.add_trace(go.Scatter(
        x=hours, y=soc_v, name='SOC', fill='tozeroy',
        line=dict(color='steelblue', width=2),
        fillcolor='rgba(70,130,180,0.15)'), row=4, col=1)
    fig_d.add_hline(y=soc_min_pct * 100, line_dash='dash', line_color='red',   opacity=0.7, row=4, col=1)
    fig_d.add_hline(y=soc_max_pct * 100, line_dash='dash', line_color='green', opacity=0.7, row=4, col=1)
    fig_d.add_vline(x=13, line_dash='dot', line_color='gray', opacity=0.6,
                    annotation_text="Re-plan 13h", annotation_position='top right')

    fig_d.update_xaxes(tickvals=hours, ticktext=[f"{h}h" for h in hours],
                        tickangle=-45, row=4, col=1)
    fig_d.update_layout(barmode='stack', height=740, showlegend=True,
                         legend=dict(orientation='h', y=1.04),
                         margin=dict(t=80, b=20, r=80))
    st.plotly_chart(fig_d, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Analyse PV
# ════════════════════════════════════════════════════════════════════════════════
with tab_pv:
    col_charts, col_info = st.columns([3, 1])

    with col_info:
        st.subheader("Parametres PV")
        if pv_p:
            st.markdown(f"""
            - **Capacite** : {pv_p.get('capacity_kwp', 0):,.0f} kWc
            - **Productible** : {pv_p.get('specific_yield', 0):,.0f} kWh/kWc
            - **Prod. annuelle** : {pv_p.get('annual_kwh', 0):,.0f} kWh
            - **Tarif FIT** : {pv_p.get('fit_price', 0)*100:.1f} c€/kWh
            - **Latitude** : {pv_p.get('lat', 0):.4f} N
            - **Longitude** : {pv_p.get('lon', 0):.4f} E
            """)
        st.subheader("Bilan PV")
        st.metric("Production simulee", f"{pv_total:,.0f} kWh")
        st.metric("PV vers batterie",   f"{pv_to_bat_tot:,.0f} kWh")
        st.metric("PV vers reseau",     f"{pv_sold_tot:,.0f} kWh")
        st.metric("Autoconsommation",   f"{self_cons_pct:.1f} %")
        st.metric("Revenu FIT total",   f"{pv_rev:,.0f} EUR")

    with col_charts:
        # Production mensuelle PV
        pv_m = results.resample('ME').agg(
            pv_bat  = ('pv_to_battery_kwh',   'sum'),
            pv_sell = ('pv_surplus_sold_kwh',  'sum'),
        )
        pv_m.index = pv_m.index.strftime('%Y-%m')
        fig_pv_m = go.Figure([
            go.Bar(x=pv_m.index, y=pv_m['pv_bat'],  name='PV batterie', marker_color='gold'),
            go.Bar(x=pv_m.index, y=pv_m['pv_sell'], name='PV reseau',   marker_color='orange'),
        ])
        fig_pv_m.update_layout(
            barmode='stack', height=280, margin=dict(t=10, b=10),
            yaxis_title='kWh / mois', title='Production PV mensuelle (batterie + reseau)',
            legend=dict(orientation='h', y=1.08),
        )
        st.plotly_chart(fig_pv_m, use_container_width=True)

        # Profil horaire moyen par mois
        pv_h = results.copy()
        pv_h['_hour']  = pv_h.index.hour
        pv_h['_month'] = pv_h.index.strftime('%Y-%m')
        pv_by_mh  = pv_h.groupby(['_month', '_hour'])['pv_production_kwh'].mean()
        avg_pv_h  = pv_h.groupby('_hour')['pv_production_kwh'].mean()
        months_pv = sorted(pv_h['_month'].unique())

        pal = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
               '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
               '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5','#c49c94']
        fig_pv_h2 = go.Figure()
        for i, mo in enumerate(months_pv):
            vals = [pv_by_mh.get((mo, h), 0) for h in range(24)]
            fig_pv_h2.add_trace(go.Scatter(
                x=list(range(24)), y=vals, name=mo, mode='lines',
                line=dict(color=pal[i % len(pal)], width=1.3), opacity=0.8))
        fig_pv_h2.add_trace(go.Scatter(
            x=list(range(24)), y=avg_pv_h.values, name='Moyenne',
            line=dict(color='black', width=2.5, dash='dash')))
        fig_pv_h2.update_xaxes(
            tickvals=list(range(24)), ticktext=[f"{h}h" for h in range(24)], tickangle=-45)
        fig_pv_h2.update_layout(
            height=320, margin=dict(t=10, b=60),
            yaxis_title='kWh/h (moyenne journaliere)',
            title='Profil horaire moyen de production PV par mois',
            legend=dict(orientation='h', y=-0.30),
        )
        st.plotly_chart(fig_pv_h2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Profil spot price
# ════════════════════════════════════════════════════════════════════════════════
with tab_spot:
    r2 = results.copy()
    r2['_hour']  = r2.index.hour
    r2['_month'] = r2.index.strftime('%Y-%m')
    months_s    = sorted(r2['_month'].unique())
    spot_by_mh  = r2.groupby(['_month', '_hour'])['spot_price_eur_mwh'].mean()
    avg_spot    = r2.groupby('_hour')['spot_price_eur_mwh'].mean()

    pal2 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
            '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
            '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5','#c49c94']
    fig_sp = go.Figure()
    all_vals = []
    for i, mo in enumerate(months_s):
        vals = [spot_by_mh.get((mo, h), 0) for h in range(24)]
        all_vals.append(vals)
        fig_sp.add_trace(go.Scatter(
            x=list(range(24)), y=vals, name=mo, mode='lines',
            line=dict(color=pal2[i % len(pal2)], width=1.2), opacity=0.75))
    arr = np.array(all_vals)
    fig_sp.add_trace(go.Scatter(
        x=list(range(24)), y=avg_spot.values, name='Moyenne globale',
        line=dict(color='black', width=2.5, dash='dash')))
    fig_sp.add_trace(go.Scatter(
        x=list(range(24)), y=arr.max(axis=0), name='Max mensuel',
        line=dict(color='red', width=1, dash='dot'), opacity=0.5))
    fig_sp.add_trace(go.Scatter(
        x=list(range(24)), y=arr.min(axis=0), name='Min mensuel',
        line=dict(color='blue', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(100,149,237,0.08)', opacity=0.5))
    fig_sp.add_vline(x=13, line_dash='dot', line_color='gray', opacity=0.5,
                     annotation_text="13h — publication J+1")
    fig_sp.update_xaxes(tickvals=list(range(24)),
                         ticktext=[f"{h}h" for h in range(24)], tickangle=-45)
    fig_sp.update_yaxes(title_text='EUR/MWh')
    fig_sp.update_layout(height=450, title='Profil horaire moyen spot price par mois',
                          legend=dict(orientation='h', y=-0.25), margin=dict(t=40, b=80))
    st.plotly_chart(fig_sp, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Parametres
# ════════════════════════════════════════════════════════════════════════════════
with tab_params_tab:
    eol_reached = cap_final / params['capacity_kwh'] <= params['capacity_eol'] + 0.001
    ratio       = (results['charge_from_grid_kwh'].sum() /
                   max(results['discharge_to_grid_kwh'].sum(), 1))

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Batterie")
        st.table(pd.DataFrame({
            "Parametre": ["Capacite nominale", "Capacite finale", "Perte capacite",
                          "Fin de vie", "P_max", "C-rate", "Raccordement"],
            "Valeur": [
                f"{params['capacity_kwh']:,.0f} kWh", f"{cap_final:,.0f} kWh",
                f"{cap_loss:.2f} %", "OUI" if eol_reached else "Non",
                f"{p_max:,.0f} kW", f"{params['c_rate']:.2f} h-1",
                f"{params['connection_kw']:,.0f} kW",
            ],
        }).set_index("Parametre"))

    with col_b:
        st.subheader("Generateur PV")
        if pv_p:
            st.table(pd.DataFrame({
                "Parametre": ["Capacite PV", "Productible specifique",
                              "Prod. annuelle cible", "Prod. simulee",
                              "Tarif FIT", "Latitude", "Longitude",
                              "Autoconsommation"],
                "Valeur": [
                    f"{pv_p.get('capacity_kwp', 0):,.0f} kWc",
                    f"{pv_p.get('specific_yield', 0):,.0f} kWh/kWc",
                    f"{pv_p.get('annual_kwh', 0):,.0f} kWh",
                    f"{pv_total:,.0f} kWh",
                    f"{pv_p.get('fit_price', 0)*100:.1f} c/kWh",
                    f"{pv_p.get('lat', 0):.4f} N",
                    f"{pv_p.get('lon', 0):.4f} E",
                    f"{self_cons_pct:.1f} %",
                ],
            }).set_index("Parametre"))

    with col_c:
        st.subheader("Exploitation")
        st.table(pd.DataFrame({
            "Parametre": ["Periode", "Duree", "FEC cumules",
                          "Rev. decharge BESS", "Rev. surplus PV",
                          "Cout achat reseau", "Contribution nette",
                          "Contrib. / FEC"],
            "Valeur": [
                f"{results.index[0].date()} -> {results.index[-1].date()}",
                f"{n_days} jours",
                f"{fec_total:.1f}",
                f"{total_rev:,.0f} EUR",
                f"{pv_rev:,.0f} EUR",
                f"{total_cost:,.0f} EUR",
                f"{net_profit:,.0f} EUR",
                f"{net_profit / max(fec_total, 1):,.2f} EUR/FEC",
            ],
        }).set_index("Parametre"))
