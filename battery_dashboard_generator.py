#!/usr/bin/env python3
"""
battery_dashboard_generator.py

Lit battery_optimization_results.csv (genere par battery_arbitrage_optimizer.ipynb)
et produit un tableau de bord HTML interactif autonome : battery_dashboard.html

Utilisation : python battery_dashboard_generator.py
"""
import sys
import json
import datetime as _dt
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERREUR : pandas non installe.  pip install pandas")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
INPUT_CSV   = SCRIPT_DIR / 'battery_optimization_results.csv'
OUTPUT_HTML = SCRIPT_DIR / 'battery_dashboard.html'

# Parametres de simulation — lus depuis battery_params.json (genere par le notebook)
PARAMS_JSON = SCRIPT_DIR / 'battery_params.json'
_DEFAULTS = {
    'eff_roundtrip'            : 0.97,
    'connection_kw'            : 1_000.0,
    'aging_per_fec'            : 2e-4,
    'capacity_eol'             : 0.80,
    'agg_spread'               : 0.000,
    'c_rate'                   : 0.5,
    'capacity_kwh'             : 1_000.0,
    'p_max'                    : 500.0,
    'soc_min_pct'              : 0.10,
    'soc_max_pct'              : 0.90,
    'min_discharge_spread_mwh' : 15.0,
}
if PARAMS_JSON.exists():
    with open(PARAMS_JSON, encoding='utf-8') as _pf:
        SIMULATION_PARAMS = {**_DEFAULTS, **json.load(_pf)}
    print(f"Parametres lus depuis {PARAMS_JSON.name}")
else:
    SIMULATION_PARAMS = _DEFAULTS
    print(f"ATTENTION : {PARAMS_JSON.name} introuvable — valeurs par defaut utilisees.")
    print("  Executer la cellule 11-export-csv du notebook pour generer ce fichier.")

# ── Chargement des donnees ────────────────────────────────────────────────────
print(f"Chargement de {INPUT_CSV.name} ...")
if not INPUT_CSV.exists():
    print("ERREUR : fichier introuvable.")
    print("Executer d'abord toutes les cellules de battery_arbitrage_optimizer.ipynb.")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV, sep=';', parse_dates=['datetime'])
df = df.set_index('datetime').sort_index()
print(f"  {len(df):,} lignes  ({df.index.min().date()} => {df.index.max().date()})")

# ── Agregats mensuels ─────────────────────────────────────────────────────────
monthly_agg = df.resample('ME').agg(
    revenue       = ('resale_revenue_eur',    'sum'),
    cost          = ('purchase_cost_eur',     'sum'),
    profit        = ('net_revenue_eur',       'sum'),
    charge_kwh    = ('charge_from_grid_kwh',  'sum'),
    discharge_kwh = ('discharge_to_grid_kwh', 'sum'),
    avg_spot      = ('spot_price_eur_mwh',    'mean'),
).round(2)
monthly_agg.index.name = 'month'
monthly_agg.index = monthly_agg.index.strftime('%Y-%m')
monthly_agg['cumprofit'] = monthly_agg['profit'].cumsum().round(2)
monthly_records = monthly_agg.reset_index().to_dict(orient='records')

# ── Detail quotidien (structure allegee pour JSON compact) ────────────────────
print("Construction du detail quotidien ...")
daily_dict = {}
for day_date, grp in df.groupby(df.index.date):
    ds = day_date.strftime('%Y-%m-%d')
    daily_dict[ds] = {
        'h'     : [ts.strftime('%H:%M') for ts in grp.index],
        'spot'  : grp['spot_price_eur_mwh'].round(3).tolist(),
        'ch'    : grp['charge_from_grid_kwh'].round(3).tolist(),
        'di'    : grp['discharge_to_grid_kwh'].round(3).tolist(),
        'soc'   : grp['soc_pct'].round(2).tolist(),
        'rev'   : round(float(grp['resale_revenue_eur'].sum()), 2),
        'cost'  : round(float(grp['purchase_cost_eur'].sum()), 2),
        'prof'  : round(float(grp['net_revenue_eur'].sum()), 2),
        'chT'   : round(float(grp['charge_from_grid_kwh'].sum()), 1),
        'diT'   : round(float(grp['discharge_to_grid_kwh'].sum()), 1),
        'sp_avg': round(float(grp['spot_price_eur_mwh'].mean()), 2),
        'sp_min': round(float(grp['spot_price_eur_mwh'].min()), 2),
        'sp_max': round(float(grp['spot_price_eur_mwh'].max()), 2),
        'sm_min': round(float(grp['soc_pct'].min()), 1),
        'sm_max': round(float(grp['soc_pct'].max()), 1),
    }

# ── Stats globales ────────────────────────────────────────────────────────────
g_profit     = round(float(df['net_revenue_eur'].sum()), 2)
g_revenue    = round(float(df['resale_revenue_eur'].sum()), 2)
g_cost       = round(float(df['purchase_cost_eur'].sum()), 2)
g_charged    = int(round(float(df['charge_from_grid_kwh'].sum()), 0))
g_discharged = int(round(float(df['discharge_to_grid_kwh'].sum()), 0))
n_days       = len(daily_dict)
date_min     = df.index.min().strftime('%Y-%m-%d')
date_max     = df.index.max().strftime('%Y-%m-%d')
date_first   = df.index.min().strftime('%Y-%m-%d')
gen_date     = _dt.datetime.now().strftime('%d/%m/%Y %H:%M')

# ── Parametres derives du CSV ─────────────────────────────────────────────────
cap_nominal      = round(float(df['capacity_kwh'].max()), 1)
cap_final_v      = round(float(df['capacity_kwh'].iloc[-1]), 1)
cap_loss_pct_v   = round((cap_nominal - cap_final_v) / cap_nominal * 100, 1)
p_max_obs        = round(max(float(df['charge_from_grid_kwh'].max()),
                             float(df['discharge_to_grid_kwh'].max())), 0)
c_rate_est       = round(p_max_obs / cap_nominal, 2) if cap_nominal > 0 else 0
fec_total_v      = round(float(df['cumulative_fec'].iloc[-1]), 1)
soc_min_obs_v    = round(float(df['soc_pct'].min()), 1)
soc_max_obs_v    = round(float(df['soc_pct'].max()), 1)
agg_spread_mwh   = round(float((df['resale_price_eur_kwh'] -
                                 df['spot_price_eur_kwh']).mean()) * 1000, 3)
profit_per_fec   = round(g_profit / fec_total_v, 2) if fec_total_v > 0 else 0
avg_daily_profit = round(g_profit / n_days, 2) if n_days > 0 else 0
eol_reached      = bool(cap_final_v / cap_nominal <=
                        SIMULATION_PARAMS['capacity_eol'] + 0.005)

params_data = {
    'cap_nominal'     : cap_nominal,
    'cap_final'       : cap_final_v,
    'cap_loss_pct'    : cap_loss_pct_v,
    'p_max_obs'       : int(SIMULATION_PARAMS.get('p_max', p_max_obs)),
    'c_rate_est'      : SIMULATION_PARAMS.get('c_rate', c_rate_est),
    'connection_kw'   : int(SIMULATION_PARAMS['connection_kw']),
    'fec_total'       : fec_total_v,
    'soc_min'         : soc_min_obs_v,
    'soc_max'         : soc_max_obs_v,
    'agg_spread_mwh'        : agg_spread_mwh,
    'min_discharge_spread'  : SIMULATION_PARAMS.get('min_discharge_spread_mwh', 15.0),
    'eff_roundtrip'         : SIMULATION_PARAMS['eff_roundtrip'],
    'aging_per_fec'   : SIMULATION_PARAMS['aging_per_fec'],
    'capacity_eol'    : SIMULATION_PARAMS['capacity_eol'] * 100,
    'profit_per_fec'  : profit_per_fec,
    'avg_daily_profit': avg_daily_profit,
    'eol_reached'     : eol_reached,
    'n_days'          : n_days,
}

# ── Profils horaires spot price (moyenne par mois x heure) ───────────────────
print("Calcul des profils horaires spot price ...")
df['_hour']  = df.index.hour
df['_month'] = df.index.strftime('%Y-%m')

spot_by_mh = df.groupby(['_month', '_hour'])['spot_price_eur_mwh'].mean().round(2)
spot_profiles_dict = {}
for month, grp in spot_by_mh.groupby(level=0):
    vals = grp.droplevel(0)
    spot_profiles_dict[month] = [round(float(vals.get(h, 0.0)), 2) for h in range(24)]

avg_profile = [
    round(float(df.groupby('_hour')['spot_price_eur_mwh'].mean()[h]), 2)
    for h in range(24)
]
df.drop(columns=['_hour', '_month'], inplace=True)

spot_profiles_data = {
    'months'  : list(spot_profiles_dict.keys()),
    'profiles': list(spot_profiles_dict.values()),
    'avg'     : avg_profile,
}

# ── Serialisation JSON ────────────────────────────────────────────────────────
monthly_json       = json.dumps(monthly_records, ensure_ascii=False)
daily_json         = json.dumps(daily_dict, ensure_ascii=False, separators=(',', ':'))
params_json        = json.dumps(params_data, ensure_ascii=False)
spot_profiles_json = json.dumps(spot_profiles_data, ensure_ascii=False)

# ── Template HTML (raw string => {} JavaScript sans conflit) ──────────────────
HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Battery BESS &#8212; Dashboard Arbitrage Spot</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js" charset="utf-8"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#eef2f7;color:#2d3748;font-size:14px}

header{background:linear-gradient(135deg,#1a3a5c 0%,#2c6faa 100%);color:#fff;
       padding:16px 28px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
header h1{font-size:1.25rem;font-weight:700;letter-spacing:.02em}
header .gen{font-size:.72rem;opacity:.7;margin-top:3px}

.kpi-bar{display:flex;gap:8px;padding:12px 24px;background:#fff;
         border-bottom:1px solid #d9e4ef;flex-wrap:wrap;align-items:stretch}
.kpi-item{text-align:center;flex:1;min-width:110px;padding:6px 8px;
          border-radius:7px;background:#f8faff;border:1px solid #e2eaf6}
.kpi-item .val{font-size:1.1rem;font-weight:700;white-space:nowrap}
.kpi-item .lbl{font-size:.68rem;color:#718096;text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.pos{color:#27ae60}.neg{color:#e74c3c}.neu{color:#2c6faa}.warn{color:#e67e22}

main{max-width:1320px;margin:0 auto;padding:18px 14px}

.card{background:#fff;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.07);
      margin-bottom:18px;overflow:hidden}
.card-header{background:linear-gradient(90deg,#f0f7ff,#e8f4fd);padding:11px 20px;
             font-size:.93rem;font-weight:600;color:#1a3a5c;border-bottom:1px solid #d4e6f5;
             display:flex;align-items:center;gap:8px}
.card-body{padding:16px}

.params-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:9px}
.p-item{border-radius:8px;padding:9px 13px;background:#f8faff;border:1px solid #e2eaf6}
.p-item .p-val{font-size:1.0rem;font-weight:700;color:#1a3a5c;white-space:nowrap}
.p-item .p-lbl{font-size:.67rem;color:#718096;text-transform:uppercase;letter-spacing:.04em;margin-top:3px}
.p-item.p-warn{background:#fef9e7;border-color:#f9e79f}
.p-item.p-ok  {background:#eafaf1;border-color:#a9dfbf}
.p-item.p-info{background:#ebf5fb;border-color:#a9cce3}

.date-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.date-bar label{font-weight:600;font-size:.88rem;color:#1a3a5c}
input[type=date]{border:1.5px solid #b0c4de;border-radius:7px;padding:6px 11px;
                 font-size:.88rem;color:#2d3748;outline:none;cursor:pointer;background:#fff}
input[type=date]:focus{border-color:#2c6faa;box-shadow:0 0 0 2px rgba(44,111,170,.15)}
.btn{background:#2c6faa;color:#fff;border:none;border-radius:7px;padding:7px 14px;
     cursor:pointer;font-size:.88rem;font-weight:600;transition:background .18s;white-space:nowrap}
.btn:hover{background:#1a3a5c}
.date-label{font-size:.9rem;color:#4a5568;margin-left:4px;font-style:italic}

.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(138px,1fr));gap:9px;margin-bottom:14px}
.sc{border-radius:8px;padding:9px 12px;text-align:center;transition:transform .15s}
.sc:hover{transform:translateY(-1px)}
.sc .sv{font-size:1.05rem;font-weight:700;white-space:nowrap}
.sc .sl{font-size:.67rem;color:#718096;text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.sc-profit {background:#eafaf1;border:1px solid #a9dfbf}
.sc-revenue{background:#ebf5fb;border:1px solid #a9cce3}
.sc-cost   {background:#fef9e7;border:1px solid #f9e79f}
.sc-charge {background:#eaf4fb;border:1px solid #aed6f1}
.sc-dis    {background:#fdedec;border:1px solid #f5b7b1}
.sc-spot   {background:#fdf2e9;border:1px solid #fad7a0}
.sc-soc    {background:#f4ecf7;border:1px solid #d2b4de}
.sc-spread {background:#e8f8f5;border:1px solid #a2d9ce}

.tbl-wrap{overflow-x:auto;margin-top:14px}
table{width:100%;border-collapse:collapse;font-size:.80rem;white-space:nowrap}
th{background:#f0f7ff;padding:8px 11px;text-align:right;font-weight:600;
   color:#1a3a5c;border-bottom:2px solid #b0c4de}
th:first-child{text-align:left}
td{padding:7px 11px;text-align:right;border-bottom:1px solid #e8edf5}
td:first-child{text-align:left;font-weight:600}
tr:hover td{background:#f5f9ff}
.tbl-pos{color:#27ae60;font-weight:700}
.tbl-neg{color:#e74c3c;font-weight:700}
.tbl-foot td{font-weight:700;background:#f0f7ff;border-top:2px solid #b0c4de}

#daily-chart{min-height:680px}
</style>
</head>
<body>

<header>
  <div>
    <h1>&#9889; Battery BESS &#8212; Dashboard Arbitrage Spot Price France</h1>
    <div class="gen">
      Genere le __GEN_DATE__ &nbsp;&bull;&nbsp;
      Donnees : __DATE_MIN__ &#8594; __DATE_MAX__ &nbsp;&bull;&nbsp; __N_DAYS__ jours
    </div>
  </div>
</header>

<div class="kpi-bar">
  <div class="kpi-item">
    <div class="val pos">__G_PROFIT__ EUR</div>
    <div class="lbl">Profit net total</div>
  </div>
  <div class="kpi-item">
    <div class="val neu">__G_REVENUE__ EUR</div>
    <div class="lbl">Recettes vente</div>
  </div>
  <div class="kpi-item">
    <div class="val warn">__G_COST__ EUR</div>
    <div class="lbl">Cout achat</div>
  </div>
  <div class="kpi-item">
    <div class="val neu">__G_CHARGED__ kWh</div>
    <div class="lbl">Energie achetee</div>
  </div>
  <div class="kpi-item">
    <div class="val neu">__G_DISCHARGED__ kWh</div>
    <div class="lbl">Energie vendue</div>
  </div>
</div>

<main>

<!-- Parametres de simulation -->
<div class="card">
  <div class="card-header">&#9881;&#65039; Parametres de simulation &amp; Bilan vieillissement</div>
  <div class="card-body">
    <div class="params-grid" id="params-grid"></div>
  </div>
</div>

<!-- Revenu mensuel -->
<div class="card">
  <div class="card-header">&#128197; Revenu mensuel (EUR) &#8212; Prix vente &#8722; Prix achat</div>
  <div class="card-body">
    <div id="monthly-chart" style="height:360px"></div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Mois</th><th>Recettes (EUR)</th><th>Cout achat (EUR)</th>
          <th>Profit net (EUR)</th><th>Achete (kWh)</th><th>Vendu (kWh)</th>
          <th>Spot moy (EUR/MWh)</th><th>Profit cumule (EUR)</th>
        </tr></thead>
        <tbody id="monthly-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- Profils horaires spot price -->
<div class="card">
  <div class="card-header">&#128200; Profils horaires spot price (EUR/MWh) &#8212; Courbes moyennes par mois</div>
  <div class="card-body">
    <div id="spot-profiles-chart" style="height:400px"></div>
  </div>
</div>

<!-- Detail journalier -->
<div class="card">
  <div class="card-header">&#128269; Detail journalier &#8212; Selectionner une date</div>
  <div class="card-body">
    <div class="date-bar">
      <label>Date :</label>
      <button class="btn" onclick="navigateDay(-1)">&#9664; Precedent</button>
      <input type="date" id="date-picker"
             min="__DATE_MIN__" max="__DATE_MAX__" value="__DATE_FIRST__"
             onchange="onDateChange()">
      <button class="btn" onclick="navigateDay(+1)">Suivant &#9654;</button>
      <span class="date-label" id="date-label"></span>
    </div>

    <div class="summary-grid">
      <div class="sc sc-profit"><div class="sv pos" id="s-prof">&#8212;</div><div class="sl">Profit net</div></div>
      <div class="sc sc-revenue"><div class="sv neu" id="s-rev">&#8212;</div><div class="sl">Recettes vente</div></div>
      <div class="sc sc-cost"><div class="sv warn" id="s-cost">&#8212;</div><div class="sl">Cout achat</div></div>
      <div class="sc sc-charge"><div class="sv" id="s-chT" style="color:#2980b9">&#8212;</div><div class="sl">Achete (kWh)</div></div>
      <div class="sc sc-dis"><div class="sv" id="s-diT" style="color:#c0392b">&#8212;</div><div class="sl">Vendu (kWh)</div></div>
      <div class="sc sc-spot"><div class="sv" id="s-spot" style="color:#d35400">&#8212;</div><div class="sl">Spot moy (EUR/MWh)</div></div>
      <div class="sc sc-soc"><div class="sv" id="s-soc" style="color:#8e44ad">&#8212;</div><div class="sl">SOC min &#8211; max</div></div>
      <div class="sc sc-spread"><div class="sv" id="s-spread" style="color:#16a085">&#8212;</div><div class="sl">Spread (EUR/MWh)</div></div>
    </div>

    <div id="daily-chart"></div>
  </div>
</div>

</main>

<script>
const MONTHLY = __MONTHLY__;
const DAILY   = __DAILY__;
const PARAMS  = __PARAMS_JSON__;
const SP      = __SPOT_PROFILES_JSON__;

const f2  = v => Number(v).toLocaleString('fr-FR',{minimumFractionDigits:2,maximumFractionDigits:2});
const f1  = v => Number(v).toLocaleString('fr-FR',{minimumFractionDigits:1,maximumFractionDigits:1});
const f0  = v => Number(v).toLocaleString('fr-FR',{minimumFractionDigits:0,maximumFractionDigits:0});
const sgn = v => (v >= 0 ? '+' : '') + f2(v);

const MOIS_FR  = ['Jan','Fev','Mar','Avr','Mai','Jun','Jul','Aou','Sep','Oct','Nov','Dec'];
const JOURS_FR = ['Dim','Lun','Mar','Mer','Jeu','Ven','Sam'];
const HOURS_24 = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00',
                  '08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00',
                  '16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'];

// Palette saisonniere : bleus (hiver) → verts (printemps) → oranges (ete) → violets (automne)
const MONTH_PALETTE = [
  '#1f77b4','#4393c3','#74add1',
  '#31a354','#41ab5d','#78c679',
  '#fd8d3c','#f16913','#d94801',
  '#756bb1','#9e9ac8','#bcbddc'
];

// ── Carte parametres ──────────────────────────────────────────────────────────
function initParamsCard() {
  const P = PARAMS;
  const eolCls = P.eol_reached ? 'p-warn' : 'p-ok';
  const eolTxt = P.eol_reached
    ? '<span style="color:#e67e22">&#9888; Atteinte</span>'
    : '<span style="color:#27ae60">&#10003; Non atteinte</span>';

  const items = [
    {v: f0(P.cap_nominal) + ' kWh',                                    l: 'Capacite nominale',         cls: 'p-info'},
    {v: f0(P.cap_final) + ' kWh (' + f1(100 - P.cap_loss_pct) + '% nom.)', l: 'Capacite finale',      cls: eolCls},
    {v: f1(P.cap_loss_pct) + ' %',                                     l: 'Perte capacite',             cls: eolCls},
    {v: eolTxt,                                                         l: 'Fin de vie (' + f0(P.capacity_eol) + '% nom.)', cls: eolCls},
    {v: f0(P.p_max_obs) + ' kW',                                       l: 'P_max observee',             cls: ''},
    {v: f1(P.c_rate_est) + ' h⁻¹',                          l: 'C-rate (estime)',             cls: ''},
    {v: f0(P.connection_kw) + ' kW',                                   l: 'Raccordement Enedis',        cls: 'p-info'},
    {v: (P.eff_roundtrip * 100).toFixed(1) + ' %',                    l: 'Rendement aller-retour',      cls: ''},
    {v: P.aging_per_fec.toExponential(1),                              l: 'Vieillissement / FEC',        cls: ''},
    {v: f1(P.fec_total),                                               l: 'FEC cumules (total)',          cls: eolCls},
    {v: f1(P.soc_min) + ' % – ' + f1(P.soc_max) + ' %',        l: 'Plage SOC observee',           cls: ''},
    {v: (P.agg_spread_mwh >= 0 ? '+' : '') + f2(P.agg_spread_mwh) + ' EUR/MWh', l: 'Spread agregateur', cls: ''},
    {v: f1(P.min_discharge_spread) + ' EUR/MWh',                               l: 'Spread min decharge', cls: ''},
    {v: f2(P.avg_daily_profit) + ' EUR/j',                            l: 'Profit moyen journalier',     cls: 'p-ok'},
    {v: f2(P.profit_per_fec) + ' EUR/FEC',                            l: 'Profit par FEC equivalent',   cls: 'p-ok'},
  ];

  document.getElementById('params-grid').innerHTML = items.map(it =>
    '<div class="p-item ' + it.cls + '">' +
      '<div class="p-val">' + it.v + '</div>' +
      '<div class="p-lbl">' + it.l + '</div>' +
    '</div>'
  ).join('');
}

// ── Profils horaires spot price ───────────────────────────────────────────────
function initSpotProfilesChart() {
  const traces = [];

  // Enveloppe min-max (fond)
  const yMin = [], yMax = [];
  for (let h = 0; h < 24; h++) {
    const vals = SP.profiles.map(p => p[h]);
    yMin.push(Math.min(...vals));
    yMax.push(Math.max(...vals));
  }
  traces.push({
    name: 'Enveloppe min-max',
    x: [...HOURS_24, ...HOURS_24.slice().reverse()],
    y: [...yMax,     ...yMin.slice().reverse()],
    type: 'scatter', mode: 'none',
    fill: 'toself', fillcolor: 'rgba(44,111,170,.09)',
    line: {color: 'transparent'},
    hoverinfo: 'skip', showlegend: true,
  });

  // Courbe de chaque mois
  SP.months.forEach((month, i) => {
    traces.push({
      name: month,
      x: HOURS_24, y: SP.profiles[i],
      type: 'scatter', mode: 'lines',
      line: {color: MONTH_PALETTE[i % 12], width: 1.6},
      opacity: 0.82,
      hovertemplate: '<b>' + month + '</b><br>%{x} : <b>%{y:.1f} EUR/MWh</b><extra></extra>',
    });
  });

  // Moyenne globale (en gras)
  traces.push({
    name: 'Moyenne generale',
    x: HOURS_24, y: SP.avg,
    type: 'scatter', mode: 'lines+markers',
    line: {color: '#1a3a5c', width: 3.5},
    marker: {size: 5, color: '#1a3a5c'},
    hovertemplate: '<b>Moyenne toute periode</b><br>%{x} : <b>%{y:.1f} EUR/MWh</b><extra></extra>',
  });

  Plotly.newPlot('spot-profiles-chart', traces, {
    height: 400,
    margin: {t: 20, b: 80, l: 72, r: 20},
    plot_bgcolor: '#f8faff',
    paper_bgcolor: '#fff',
    xaxis: {
      type: 'category',
      title: {text: 'Heure de la journee', font: {size: 11}},
      tickangle: -45, tickfont: {size: 9.5},
      tickvals: HOURS_24.filter((_, i) => i % 2 === 0),
    },
    yaxis: {
      title: {text: 'Prix spot (EUR/MWh)', font: {size: 11}},
      zeroline: true, zerolinewidth: 1.5, zerolinecolor: 'rgba(200,50,50,.45)',
      tickfont: {size: 10},
    },
    legend: {orientation: 'h', x: 0, y: -0.32, font: {size: 9}, traceorder: 'reversed'},
    shapes: [{
      type: 'line', layer: 'below',
      x0: '13:00', x1: '13:00', xref: 'x',
      y0: 0, y1: 1, yref: 'paper',
      line: {color: 'rgba(44,111,170,.5)', width: 1.8, dash: 'dot'},
    }],
    annotations: [{
      x: '13:00', xref: 'x', y: 0.97, yref: 'paper',
      text: '13h00 Re-plan D+1',
      showarrow: false, font: {size: 8.5, color: '#2c6faa'},
      bgcolor: 'rgba(255,255,255,.8)', bordercolor: '#2c6faa', borderwidth: 0.8,
      xanchor: 'left',
    }],
  }, {responsive: true, displayModeBar: false});
}

// ── Graphique mensuel ─────────────────────────────────────────────────────────
function initMonthlyChart() {
  const months     = MONTHLY.map(r => r.month);
  const profits    = MONTHLY.map(r => r.profit);
  const cumprofits = MONTHLY.map(r => r.cumprofit);
  const barCol     = profits.map(v => v>=0 ? 'rgba(39,174,96,.82)' : 'rgba(231,76,60,.82)');
  const barEdge    = profits.map(v => v>=0 ? 'rgb(30,140,72)'      : 'rgb(185,50,40)');

  Plotly.newPlot('monthly-chart', [
    {
      name: 'Profit net mensuel', x: months, y: profits, type: 'bar',
      marker: {color: barCol, line: {color: barEdge, width: 1.2}},
      text: profits.map(v => sgn(v) + ' EUR'),
      textposition: months.length > 20 ? 'none' : 'outside', textfont: {size: 9.5},
      customdata: MONTHLY.map(r => [r.revenue, r.cost, r.charge_kwh, r.discharge_kwh, r.avg_spot]),
      hovertemplate:
        '<b>%{x}</b><br>Profit net : <b>%{y:+,.0f} EUR</b><br>' +
        'Recettes   : %{customdata[0]:,.0f} EUR<br>Cout achat : %{customdata[1]:,.0f} EUR<br>' +
        'Achete     : %{customdata[2]:,.0f} kWh<br>Vendu      : %{customdata[3]:,.0f} kWh<br>' +
        'Spot moy   : %{customdata[4]:.1f} EUR/MWh<extra></extra>',
      yaxis: 'y',
    },
    {
      name: 'Profit cumule', x: months, y: cumprofits,
      type: 'scatter', mode: 'lines+markers',
      line: {color: '#1a3a5c', width: 2.5}, marker: {size: 6, color: '#1a3a5c'},
      hovertemplate: '<b>%{x}</b><br>Cumule : <b>%{y:+,.0f} EUR</b><extra></extra>',
      yaxis: 'y2',
    },
  ], {
    height: 360, margin: {t: 28, b: 100, l: 85, r: 85},
    plot_bgcolor: '#f8faff', paper_bgcolor: '#fff',
    xaxis: {type: 'category', tickangle: -45, tickfont: {size: 10}},
    yaxis:  {title: 'Profit mensuel (EUR)', zeroline: true, zerolinewidth: 1.5, zerolinecolor: '#aaa'},
    yaxis2: {overlaying: 'y', side: 'right', title: 'Profit cumule (EUR)', showgrid: false},
    legend: {orientation: 'h', x: 0, y: 1.07},
    bargap: 0.28,
  }, {responsive: true, displayModeBar: false});
}

// ── Table mensuelle ───────────────────────────────────────────────────────────
function initMonthlyTable() {
  let html = '', tRev=0, tCost=0, tProf=0, tCh=0, tDi=0;
  MONTHLY.forEach(r => {
    tRev+=r.revenue; tCost+=r.cost; tProf+=r.profit; tCh+=r.charge_kwh; tDi+=r.discharge_kwh;
    const pc = r.profit >= 0 ? 'tbl-pos' : 'tbl-neg';
    const cc = r.cumprofit >= 0 ? 'tbl-pos' : 'tbl-neg';
    html += '<tr><td>' + r.month + '</td><td>' + f2(r.revenue) + '</td><td>' + f2(r.cost) +
      '</td><td class="' + pc + '">' + f2(r.profit) + '</td><td>' + f0(r.charge_kwh) +
      '</td><td>' + f0(r.discharge_kwh) + '</td><td>' + f2(r.avg_spot) +
      '</td><td class="' + cc + '">' + f2(r.cumprofit) + '</td></tr>';
  });
  const totpc = tProf >= 0 ? 'tbl-pos' : 'tbl-neg';
  html += '<tr class="tbl-foot"><td>TOTAL</td><td>' + f2(tRev) + '</td><td>' + f2(tCost) +
    '</td><td class="' + totpc + '">' + f2(tProf) + '</td><td>' + f0(tCh) +
    '</td><td>' + f0(tDi) + '</td><td>&#8212;</td><td>&#8212;</td></tr>';
  document.getElementById('monthly-tbody').innerHTML = html;
}

// ── Graphique journalier (3 sous-graphiques separes) ─────────────────────────
function updateDailyChart(dateStr) {
  const d = DAILY[dateStr];
  const labelEl = document.getElementById('date-label');

  if (!d) {
    labelEl.textContent = '— aucune donnee pour cette date';
    document.getElementById('daily-chart').innerHTML =
      '<p style="color:#999;padding:30px;text-align:center">Aucune donnee pour cette date.</p>';
    clearCards();
    return;
  }

  const jd = new Date(dateStr + 'T12:00:00');
  labelEl.textContent = JOURS_FR[jd.getDay()] + ' ' + jd.getDate() +
    ' ' + MOIS_FR[jd.getMonth()] + ' ' + jd.getFullYear();

  const profEl = document.getElementById('s-prof');
  profEl.textContent = sgn(d.prof) + ' EUR';
  profEl.className = 'sv ' + (d.prof >= 0 ? 'pos' : 'neg');
  document.getElementById('s-rev').textContent    = f2(d.rev)    + ' EUR';
  document.getElementById('s-cost').textContent   = f2(d.cost)   + ' EUR';
  document.getElementById('s-chT').textContent    = f1(d.chT)    + ' kWh';
  document.getElementById('s-diT').textContent    = f1(d.diT)    + ' kWh';
  document.getElementById('s-spot').textContent   = f2(d.sp_avg) + ' EUR/MWh';
  document.getElementById('s-soc').textContent    = f1(d.sm_min) + '–' + f1(d.sm_max) + ' %';
  document.getElementById('s-spread').textContent = f2(d.sp_max - d.sp_min) + ' EUR/MWh';

  const neg_di = d.di.map(v => -v);

  // Profil mensuel du mois courant pour comparaison
  const curMonth = dateStr.slice(0, 7);
  const mIdx = SP.months.indexOf(curMonth);
  const spotRef = mIdx >= 0 ? SP.profiles[mIdx] : null;

  const traces = [
    // Sous-graphique 1 : Flux charge / decharge
    {
      name: 'Charge (achat reseau)',
      x: d.h, y: d.ch, type: 'bar',
      marker: {color: 'rgba(41,128,185,.82)', line: {color: 'rgba(21,98,155,1)', width: .8}},
      hovertemplate: '<b>%{x}</b><br>Charge : <b>%{y:.1f} kWh</b><extra></extra>',
      xaxis: 'x', yaxis: 'y',
    },
    {
      name: 'Decharge (vente agregateur)',
      x: d.h, y: neg_di, type: 'bar',
      marker: {color: 'rgba(192,57,43,.82)', line: {color: 'rgba(140,30,20,1)', width: .8}},
      customdata: d.di,
      hovertemplate: '<b>%{x}</b><br>Decharge : <b>%{customdata:.1f} kWh</b><extra></extra>',
      xaxis: 'x', yaxis: 'y',
    },

    // Sous-graphique 2 : Prix spot EUR/MWh (separe)
    {
      name: 'Spot price (EUR/MWh)',
      x: d.h, y: d.spot, type: 'scatter', mode: 'lines+markers',
      line: {color: '#d35400', width: 2.4},
      marker: {size: 5, color: '#d35400'},
      hovertemplate: '<b>%{x}</b><br>Spot : <b>%{y:.2f} EUR/MWh</b><extra></extra>',
      xaxis: 'x2', yaxis: 'y2',
    },

    // Sous-graphique 3 : SOC %
    {
      name: 'SOC (%)',
      x: d.h, y: d.soc, type: 'scatter', mode: 'lines',
      fill: 'tozeroy',
      line: {color: 'rgba(52,73,94,.9)', width: 2},
      fillcolor: 'rgba(100,149,237,.18)',
      hovertemplate: '<b>%{x}</b><br>SOC : <b>%{y:.1f} %</b><extra></extra>',
      xaxis: 'x3', yaxis: 'y3',
    },
  ];

  // Moyenne mensuelle sur le graphique spot (courbe de reference)
  if (spotRef) {
    traces.splice(3, 0, {
      name: 'Moy. ' + curMonth,
      x: d.h, y: spotRef, type: 'scatter', mode: 'lines',
      line: {color: 'rgba(44,111,170,.55)', width: 1.5, dash: 'dot'},
      hovertemplate: '<b>%{x}</b><br>Moy. ' + curMonth + ' : <b>%{y:.1f} EUR/MWh</b><extra></extra>',
      xaxis: 'x2', yaxis: 'y2',
    });
  }

  Plotly.react('daily-chart', traces, {
    height: 680,
    margin: {t: 55, b: 55, l: 72, r: 84},
    plot_bgcolor: '#fafbff', paper_bgcolor: '#fff',
    showlegend: true,
    legend: {orientation: 'h', x: 0, y: 1.09, xanchor: 'left', font: {size: 11}},
    barmode: 'relative', bargap: 0.10,

    // X axes (synchronises via matches)
    xaxis:  {domain:[0,1], type:'category', showticklabels:false, matches:'x3'},
    xaxis2: {domain:[0,1], type:'category', anchor:'y2', showticklabels:false, matches:'x3'},
    xaxis3: {domain:[0,1], type:'category', anchor:'y3',
             tickangle:-60, tickfont:{size:9},
             title:{text:'Heure (locale)', font:{size:11}}},

    // Y axes
    yaxis:  {domain:[0.67,1.00], title:{text:'Energie (kWh)',font:{size:11}},
             zeroline:true, zerolinewidth:1.5, zerolinecolor:'#aaa', tickfont:{size:10}},
    yaxis2: {domain:[0.35,0.61], title:{text:'Spot (EUR/MWh)',font:{size:11}},
             zeroline:true, zerolinewidth:1, zerolinecolor:'rgba(200,50,50,.4)', tickfont:{size:10}},
    yaxis3: {domain:[0.00,0.28], title:{text:'SOC (%)',font:{size:11}},
             range:[0,103], zeroline:false, tickfont:{size:10}},

    shapes: [
      {type:'line',layer:'below',x0:0,x1:1,xref:'paper',
       y0:PARAMS.soc_min,y1:PARAMS.soc_min,yref:'y3',
       line:{color:'rgba(231,76,60,.55)',width:1.3,dash:'dot'}},
      {type:'line',layer:'below',x0:0,x1:1,xref:'paper',
       y0:PARAMS.soc_max,y1:PARAMS.soc_max,yref:'y3',
       line:{color:'rgba(39,174,96,.55)',width:1.3,dash:'dot'}},
    ],
    annotations: [
      {text:'<b>Flux charge / decharge (kWh)</b>',
       x:.5,xref:'paper',y:1.00,yref:'paper',
       showarrow:false,font:{size:11.5,color:'#1a3a5c'},xanchor:'center',yanchor:'bottom'},
      {text:'<b>Prix spot (EUR/MWh)</b>',
       x:.5,xref:'paper',y:0.61,yref:'paper',
       showarrow:false,font:{size:11.5,color:'#1a3a5c'},xanchor:'center',yanchor:'bottom'},
      {text:'<b>Etat de Charge — SOC (%)</b>',
       x:.5,xref:'paper',y:0.28,yref:'paper',
       showarrow:false,font:{size:11.5,color:'#1a3a5c'},xanchor:'center',yanchor:'bottom'},
    ],
  }, {responsive:true, displayModeBar:true,
      modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d']});
}

function clearCards() {
  ['s-prof','s-rev','s-cost','s-chT','s-diT','s-spot','s-soc','s-spread']
    .forEach(id => { document.getElementById(id).textContent = '—'; });
}

function navigateDay(delta) {
  const picker = document.getElementById('date-picker');
  const d = new Date(picker.value + 'T12:00:00');
  d.setDate(d.getDate() + delta);
  const nv = d.toISOString().slice(0, 10);
  if (nv >= picker.min && nv <= picker.max) {
    picker.value = nv;
    updateDailyChart(nv);
  }
}

function onDateChange() {
  const v = document.getElementById('date-picker').value;
  if (v) updateDailyChart(v);
}

document.addEventListener('DOMContentLoaded', () => {
  initParamsCard();
  initMonthlyChart();
  initMonthlyTable();
  initSpotProfilesChart();
  updateDailyChart('__DATE_FIRST__');
});
</script>
</body>
</html>"""

# ── Injection des donnees dans le template ────────────────────────────────────
def fmt_eur(v, decimals=2):
    s = f"{v:,.{decimals}f}".replace(',', ' ').replace('.', ',')
    return ('+' if v > 0 else '') + s

html = (HTML
    .replace('__MONTHLY__',            monthly_json)
    .replace('__DAILY__',              daily_json)
    .replace('__PARAMS_JSON__',        params_json)
    .replace('__SPOT_PROFILES_JSON__', spot_profiles_json)
    .replace('__DATE_MIN__',           date_min)
    .replace('__DATE_MAX__',           date_max)
    .replace('__DATE_FIRST__',         date_first)
    .replace('__G_PROFIT__',           fmt_eur(g_profit))
    .replace('__G_REVENUE__',          fmt_eur(g_revenue))
    .replace('__G_COST__',             fmt_eur(g_cost))
    .replace('__G_CHARGED__',          f"{g_charged:,}".replace(',', ' '))
    .replace('__G_DISCHARGED__',       f"{g_discharged:,}".replace(',', ' '))
    .replace('__N_DAYS__',             str(n_days))
    .replace('__GEN_DATE__',           gen_date)
)

# ── Ecriture du fichier HTML ──────────────────────────────────────────────────
OUTPUT_HTML.write_text(html, encoding='utf-8')
print(f"\nDashboard genere : {OUTPUT_HTML.name}")
print(f"  Taille : {OUTPUT_HTML.stat().st_size / 1024:.0f} KB")
print(f"  Ouvrir dans un navigateur : {OUTPUT_HTML}")
