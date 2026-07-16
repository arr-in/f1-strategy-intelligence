import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import fastf1
import os
import base64
import warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

os.makedirs('f1_cache', exist_ok=True)
# Streamlit Cloud: prefer /tmp so FastF1 can always write cache during session downloads
if os.path.isdir('/mount/src') or os.path.isdir('/home/adminuser'):
    _cache_dir = '/tmp/f1_cache'
else:
    _cache_dir = 'f1_cache'
os.makedirs(_cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(_cache_dir)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="STRAT — AI-powered Formula 1 Analytics",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# LOGO
# ─────────────────────────────────────────
def get_logo():
    for path in ['f1_logo.png', 'f1_logo.jpg', 'f1_logo.avif']:
        if os.path.exists(path):
            ext  = path.split('.')[-1]
            mime = 'image/avif' if ext == 'avif' else f'image/{ext}'
            with open(path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{b64}"
    return None

LOGO_SRC = get_logo()

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
AVAILABLE_RACES = {
    2022: ['Bahrain','Saudi Arabia','Australia','Spain','Monaco',
           'Britain','Hungary','Belgium','Italy'],
    2023: ['Bahrain','Saudi Arabia','Australia','Spain','Monaco',
           'Britain','Hungary','Italy','Singapore'],
    2024: ['Bahrain','Saudi Arabia','Australia','Spain','Monaco',
           'Britain','Hungary','Italy','Singapore'],
    2026: ['Australia','China','Japan'],
}

DRIVER_NAMES = {
    'VER':'Max Verstappen',  'LEC':'Charles Leclerc',
    'RUS':'George Russell',  'SAI':'Carlos Sainz',
    'PER':'Sergio Perez',    'ALO':'Fernando Alonso',
    'NOR':'Lando Norris',    'PIA':'Oscar Piastri',
    'HAM':'Lewis Hamilton',  'HUL':'Nico Hulkenberg',
    'TSU':'Yuki Tsunoda',    'STR':'Lance Stroll',
    'ALB':'Alexander Albon', 'RIC':'Daniel Ricciardo',
    'MAG':'Kevin Magnussen', 'BOT':'Valtteri Bottas',
    'ZHO':'Guanyu Zhou',     'SAR':'Logan Sargeant',
    'OCO':'Esteban Ocon',    'GAS':'Pierre Gasly',
    'ANT':'Kimi Antonelli',  'LAW':'Liam Lawson',
    'HAD':'Isack Hadjar',    'BEA':'Oliver Bearman',
    'BOR':'Gabriel Bortoleto','COL':'Franco Colapinto',
    'DOO':'Jack Doohan',     'LIN':'Jack Doohan',
}

TEAM_COLORS = {
    'Red Bull Racing':'#3671C6', 'Ferrari':'#E8002D',
    'Mercedes':'#27F4D2',        'McLaren':'#FF8000',
    'Aston Martin':'#229971',    'Alpine':'#FF87BC',
    'Williams':'#64C4FF',        'AlphaTauri':'#6692FF',
    'RB':'#6692FF',              'Racing Bulls':'#6692FF',
    'Alfa Romeo':'#C92D4B',      'Kick Sauber':'#52E252',
    'Haas F1 Team':'#B6BABD',    'Cadillac':'#CC0000',
    'Audi':'#F50000',
}

# ─────────────────────────────────────────
# RACE SCHEDULE
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_race_schedule():
    try:
        schedule = fastf1.get_event_schedule(2026, include_testing=False)
        now = datetime.now(timezone.utc)
        races = []
        for _, row in schedule.iterrows():
            race_date = pd.to_datetime(row['Session5Date']).to_pydatetime()
            if race_date.tzinfo is None:
                race_date = race_date.replace(tzinfo=timezone.utc)
            races.append({
                'name':  row['EventName'],
                'date':  race_date,
                'round': int(row['RoundNumber'])
            })
        future = [r for r in races if r['date'] > now]
        return future[0] if future else None
    except Exception:
        return None

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}

.stApp {
    background-color: #0B0B0B !important;
    background-image: none !important;
}

.main .block-container {
    padding: 2rem 2.75rem 5.5rem !important;
    max-width: 1480px !important;
}

/* smoother overall */
* { scroll-behavior: smooth; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0B0B0B !important;
    border-right: 1px solid #161616 !important;
    min-width: 280px !important;
    max-width: 280px !important;
}
[data-testid="stSidebar"] > div {
    padding: 0 !important;
}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"],
[aria-label="Collapse sidebar"],
[aria-label="keyboard_double_arrow_left"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-top: 2px solid #1e1e1e !important;
    border-radius: 3px !important;
    padding: 1rem 1.25rem !important;
    transition: border-top-color 0.25s ease !important;
}
[data-testid="metric-container"]:hover {
    border-top-color: #E8002D !important;
}
[data-testid="metric-container"] label {
    color: #777 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-weight: 400 !important;
}

/* ── INPUTS ── */
.stSelectbox > div > div {
    background: #0d0d0d !important;
    border: 1px solid #222 !important;
    border-radius: 3px !important;
    color: #cccccc !important;
}
.stSelectbox label, .stSlider label {
    color: #777 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
.stSlider > div > div > div { background: #E8002D !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #E8002D !important;
    color: #E8002D !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.12em !important;
    font-size: 0.72rem !important;
    padding: 0.65rem 1.5rem !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #E8002D !important;
    color: #000000 !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 30px rgba(232,0,45,0.12);
}
.stButton > button:active {
    transform: translateY(0px);
    box-shadow: none;
}

/* Cards: add breathing room + subtle lift */
.driver-card, .decision-box, .analysis-box {
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.driver-card:hover, .decision-box:hover, .analysis-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 50px rgba(0,0,0,0.45);
    border-color: rgba(232,0,45,0.18);
}

/* ── RADIO NAV ── */
[data-testid="stSidebar"] .stRadio > div {
    gap: 0 !important;
    flex-direction: column !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 12px 24px !important;
    border-left: 2px solid transparent !important;
    color: #8a8d98 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    margin: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #ffffff !important;
    border-left-color: #444 !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    color: #ffffff !important;
    border-left-color: #E10600 !important;
    background: rgba(225,6,0,0.07) !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── FLIP CLOCK IFRAME (components.html) ── */
[data-testid="stSidebar"] iframe {
    background: transparent !important;
    border: none !important;
}

/* ── FIXED FOOTER (always at bottom) ── */
.app-footer {
    position: fixed;
    left: 280px;  /* sidebar width */
    right: 0;
    bottom: 0;
    z-index: 999;
    background: rgba(11,11,11,0.95) !important;
    backdrop-filter: blur(10px);
    border-top: 1px solid #141414 !important;
    padding: 0.6rem 2.75rem 0.5rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex-wrap: wrap !important;
}
.app-footer .title {
    font-family:'Bebas Neue', monospace;
    font-size:0.9rem;
    color:#2f2f2f;
    letter-spacing:0.22em;
    text-align:center;
    margin:0;
}
.app-footer .sub {
    font-size:0.62rem;
    color:#2a2a2a;
    letter-spacing:0.16em;
    text-align:center;
    margin:0.25rem 0 0.35rem;
}
.app-footer .love {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:6px;
    font-size:0.64rem;
    color:#2a2a2a;
    letter-spacing:0.18em;
    text-transform:uppercase;
}
@media (max-width: 900px) {
  .app-footer { left: 0; }
}

/* ── SIDEBAR TEXT ── */
.sidebar-eyebrow {
    font-size: 0.52rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #141414;
}
.coverage-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 0.7rem;
    color: #666;
    letter-spacing: 0.04em;
}
.coverage-dot {
    width: 3px; height: 3px;
    border-radius: 50%;
    background: #E8002D;
    flex-shrink: 0;
}

/* ── PAGE ELEMENTS ── */
.red-rule {
    height: 1px;
    background: linear-gradient(90deg, #E8002D 0%, rgba(232,0,45,0.3) 50%, transparent 100%);
    border: none;
    margin: 1.5rem 0;
}
.dim-rule {
    height: 1px;
    background: #141414;
    border: none;
    margin: 1.5rem 0;
}
.page-title {
    font-family: 'Bebas Neue', monospace;
    font-size: 3.2rem;
    color: #fff;
    letter-spacing: 0.06em;
    line-height: 0.9;
    margin-bottom: 8px;
}
.page-subtitle {
    font-size: 0.62rem;
    color: #555;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.section-eyebrow {
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.85rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #141414;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #0f0f0f;
    font-size: 0.72rem;
}
.stat-label { color: #555; letter-spacing: 0.1em; text-transform: uppercase; }
.stat-value { color: #aaaaaa; }
.driver-card {
    background: #0a0a0a;
    border: 1px solid #161616;
    border-radius: 3px;
    padding: 1.25rem;
    margin-bottom: 12px;
}
.driver-card-name {
    font-family: 'Bebas Neue', monospace;
    font-size: 1.5rem;
    letter-spacing: 0.08em;
    line-height: 1;
    margin-bottom: 3px;
}
.driver-card-team {
    font-size: 0.58rem;
    color: #444;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.decision-box {
    background: #0a0a0a;
    border: 1px solid #161616;
    border-radius: 3px;
    padding: 2rem 1.5rem;
    text-align: center;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.decision-label {
    font-size: 0.56rem;
    color: #555;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.decision-text {
    font-family: 'Bebas Neue', monospace;
    font-size: 4rem;
    letter-spacing: 0.06em;
    line-height: 1;
}
.decision-conf {
    font-size: 0.66rem;
    color: #666;
    letter-spacing: 0.15em;
    margin-top: 14px;
}
.analysis-box {
    background: #0a0a0a;
    border: 1px solid #161616;
    border-radius: 3px;
    padding: 1.25rem;
    height: 100%;
}

/* ── STRAT broadcast cards ── */
.strat-card {
    background: #111111;
    border: 1px solid #1c1c1c;
    border-radius: 2px;
    padding: 1.35rem 1.4rem 1.5rem;
    height: 100%;
    animation: stratFade 0.45s ease both;
}
.strat-hero {
    display:flex; align-items:flex-end; justify-content:space-between; gap:16px;
    margin-bottom: 1.25rem;
}
.strat-hero-title {
    font-family:'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: 0.08em;
    color: #fff;
    line-height: 0.92;
    margin: 0;
}
.strat-hero-sub {
    font-size: 0.62rem;
    letter-spacing: 0.28em;
    color: #666;
    text-transform: uppercase;
    margin-top: 8px;
}
.zone-legend {
    display:flex; gap:14px; align-items:center; flex-wrap:wrap;
    font-size:0.58rem; letter-spacing:0.16em; color:#777; text-transform:uppercase;
}
.zone-pill {
    display:inline-flex; align-items:center; gap:6px;
}
.zone-dot { width:10px; height:3px; display:inline-block; }
@keyframes stratFade {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: none; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA + MODELS
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    clean_laps  = pd.read_csv('data/clean_laps.csv')
    optimal_pit = pd.read_csv('data/optimal_pit_df.csv')
    driver_dna  = pd.read_csv('data/driver_dna.csv')
    return clean_laps, optimal_pit, driver_dna

STRATEGY_FEATURES = [
    'tire_age_at_pit', 'compound_num', 'race_progress',
    'position_before', 'laps_remaining'
]

def _train_strategy_model(optimal_pit_df):
    """Rebuild classifier from CSV when pickles can't be loaded (e.g. sklearn/Python mismatch)."""
    from sklearn.ensemble import GradientBoostingClassifier
    df = optimal_pit_df.dropna(subset=STRATEGY_FEATURES + ['optimal_pit'])
    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
    model.fit(df[STRATEGY_FEATURES], df['optimal_pit'])
    return model

@st.cache_resource
def load_models(_optimal_pit):
    try:
        return joblib.load('models/strategy_model.pkl')
    except Exception as e:
        # Common on Streamlit Cloud when Python/sklearn ≠ training env (e.g. Python 3.14)
        st.warning(
            f"Could not load saved strategy model ({type(e).__name__}). "
            "Retraining from data — for a permanent fix, set the app Python version to **3.12** "
            "in Streamlit Cloud → Advanced settings (delete & redeploy if needed)."
        )
        return _train_strategy_model(_optimal_pit)

clean_laps, optimal_pit, driver_dna = load_data()
strategy_model = load_models(optimal_pit)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def hex_to_rgba(h, a=0.2):
    h = h.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{a})'

PLOT_BASE = dict(
    plot_bgcolor='#0B0B0B',
    paper_bgcolor='#0B0B0B',
    font=dict(color='#ffffff', family='DM Mono', size=11),
    hoverlabel=dict(bgcolor='#111', bordercolor='#222',
                    font=dict(color='#fff', size=11, family='DM Mono')),
    legend=dict(bgcolor='rgba(11,11,11,0.95)', bordercolor='#1a1a1a',
                borderwidth=1, font=dict(color='#888', size=10, family='DM Mono')),
    margin=dict(t=50, b=45, l=55, r=20),
)

def ax(title=''):
    return dict(
        title=title,
        gridcolor='#111111',
        zerolinecolor='#1a1a1a',
        tickfont=dict(color='#555', size=9),
        title_font=dict(color='#666', size=10, family='monospace')
    )

def _telemetry_csv_path(year: int, race: str) -> str:
    return os.path.join('data', 'telemetry', f'{year}_{race.replace(" ", "_")}_Q.csv')


def _fastest_car_trace(session, driver_code: str):
    """Live FastF1 fallback: return (distance, speed, lap_time_sec)."""
    try:
        laps = session.laps.pick_drivers(driver_code)
    except Exception:
        laps = session.laps.pick_driver(driver_code)
    if laps is None or getattr(laps, 'empty', False):
        raise ValueError(f"No qualifying laps found for {driver_code}.")
    lap = laps.pick_fastest()
    if lap is None or pd.isna(lap['LapTime']):
        raise ValueError(f"No valid fastest lap for {driver_code}.")
    try:
        tel = lap.get_car_data().add_distance()
    except Exception:
        tel = lap.get_telemetry().add_distance()
    if tel is None or len(tel) < 20 or 'Speed' not in tel.columns:
        raise ValueError(f"Telemetry incomplete for {driver_code}.")
    return (
        tel['Distance'].to_numpy(dtype=float),
        tel['Speed'].to_numpy(dtype=float),
        float(lap['LapTime'].total_seconds()),
    )


def _trace_from_csv(path: str, driver_code: str):
    """Load one driver's fastest-lap distance/speed from a pre-exported CSV."""
    df = pd.read_csv(path)
    sub = df[df['Driver'] == driver_code]
    if sub.empty:
        available = sorted(df['Driver'].unique().tolist())
        raise ValueError(
            f"No qualifying telemetry for {driver_code} in this session. "
            f"Available: {', '.join(available)}"
        )
    dist = sub['Distance'].to_numpy(dtype=float)
    speed = sub['Speed'].to_numpy(dtype=float)
    t_sec = float(sub['LapTimeSec'].iloc[0])
    order = np.argsort(dist)
    return dist[order], speed[order], t_sec


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_qualy_speed_trace(year: int, race: str, d1: str, d2: str):
    """
    Load qualifying speed traces for two drivers.

    Prefer shipped CSVs in data/telemetry/ (Streamlit Cloud cannot reliably
    download FastF1 car telemetry from F1 datacenter-blocked IPs). Live FastF1
    is only used as a local/dev fallback when the CSV is missing.
    """
    from scipy.interpolate import interp1d

    path = _telemetry_csv_path(year, race)
    if os.path.exists(path):
        d1_dist, d1_spd, t1s = _trace_from_csv(path, d1)
        d2_dist, d2_spd, t2s = _trace_from_csv(path, d2)
    else:
        session = fastf1.get_session(year, race, 'Q')
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        if session.laps is None or getattr(session.laps, 'empty', True):
            raise RuntimeError(
                f"No telemetry file at {path} and live FastF1 load failed for {race} {year}."
            )
        d1_dist, d1_spd, t1s = _fastest_car_trace(session, d1)
        d2_dist, d2_spd, t2s = _fastest_car_trace(session, d2)

    dmin = max(float(np.nanmin(d1_dist)), float(np.nanmin(d2_dist)))
    dmax = min(float(np.nanmax(d1_dist)), float(np.nanmax(d2_dist)))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        raise RuntimeError("Could not align distance channels for the two drivers.")

    cd = np.linspace(dmin, dmax, 1500)
    s1 = interp1d(d1_dist, d1_spd, bounds_error=False, fill_value='extrapolate')(cd)
    s2 = interp1d(d2_dist, d2_spd, bounds_error=False, fill_value='extrapolate')(cd)
    return cd, s1, s2, (s1 - s2), t1s, t2s

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    # Premium typography logo for the sidebar header
    st.markdown("""
    <div style="padding: 1.8rem 24px 1.2rem 24px; font-family: 'Bebas Neue', sans-serif; border-bottom: 1px solid #141414; margin-bottom: 15px; background: linear-gradient(180deg, rgba(225,6,0,0.03) 0%, transparent 100%);">
        <div style="display: flex; align-items: baseline; gap: 8px;">
            <span style="font-size: 2.8rem; font-weight: 900; letter-spacing: 0.05em; color: #ffffff; font-style: italic; line-height: 0.9;">STRAT</span>
            <span style="font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 900; background: #E10600; color: #ffffff; padding: 2px 6px; border-radius: 3px; letter-spacing: 0.05em; font-style: normal; transform: translateY(-4px);">AI</span>
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.6rem; font-weight: 800; color: #515462; letter-spacing: 0.35em; text-transform: uppercase; margin-top: 2px; line-height: 1;">
            F1 ANALYTICS
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Real-time flip clock — must use components.html so JS runs
    # (st.markdown strips <script> tags)
    next_race = get_race_schedule()
    if next_race:
        race_ts = int(next_race['date'].timestamp() * 1000)
        race_name = next_race['name'].upper().replace("'", "\\'")
        round_num = next_race['round']
        components.html(f"""
        <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap" rel="stylesheet">
        <style>
            html, body {{
                margin: 0; padding: 0;
                background: transparent;
                font-family: 'DM Mono', monospace;
                overflow: hidden;
            }}
            .wrap {{ padding: 0 24px 8px; box-sizing: border-box; }}
            .sidebar-eyebrow {{
                font-size: 0.52rem;
                letter-spacing: 0.28em;
                text-transform: uppercase;
                color: #444;
                margin-bottom: 10px;
                padding-bottom: 6px;
                border-bottom: 1px solid #141414;
            }}
            .next-race-name {{
                font-family: 'Bebas Neue', monospace;
                font-size: 1rem;
                color: #ffffff;
                letter-spacing: 0.12em;
                text-align: center;
                margin-bottom: 6px;
                line-height: 1.2;
            }}
            .flip-clock-wrap {{
                display: grid;
                grid-template-columns: 1fr 12px 1fr 12px 1fr 12px 1fr;
                gap: 4px;
                align-items: start;
                margin: 10px 0;
            }}
            .flip-unit {{ display: flex; flex-direction: column; align-items: center; }}
            .flip-card {{
                perspective: 1000px;
                width: 100%;
                height: 48px;
                position: relative;
            }}
            .flip-card-inner {{
                position: relative;
                width: 100%;
                height: 100%;
                transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                transform-style: preserve-3d;
            }}
            .flip-card-inner.flipped {{
                transform: rotateX(180deg);
            }}
            .flip-card-front, .flip-card-back {{
                position: absolute;
                width: 100%;
                height: 100%;
                backface-visibility: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(180deg, #161616 0%, #111 49%, #0d0d0d 51%, #161616 100%);
                border: 1px solid #222;
                border-top: 2px solid #E8002D;
                border-radius: 4px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.03);
                box-sizing: border-box;
            }}
            .flip-card-front::after, .flip-card-back::after {{
                content: '';
                position: absolute;
                left: 0; right: 0; top: 50%;
                height: 1px;
                background: rgba(0,0,0,0.9);
                z-index: 2;
            }}
            .flip-card-back {{
                transform: rotateX(180deg);
            }}
            .flip-num {{
                font-family: 'Bebas Neue', monospace;
                font-size: 1.8rem;
                color: #E8002D;
                line-height: 1;
                display: block;
                text-shadow: 0 0 20px rgba(232,0,45,0.5);
            }}
            .flip-lbl {{
                font-size: 0.46rem;
                color: #555;
                letter-spacing: 0.2em;
                text-transform: uppercase;
                display: block;
                margin-top: 5px;
            }}
            .flip-sep {{
                font-family: 'Bebas Neue', monospace;
                font-size: 1.4rem;
                color: #E8002D;
                text-align: center;
                opacity: 0.35;
                padding-top: 8px;
                align-self: flex-start;
                line-height: 1;
            }}
            .rule {{ height: 1px; background: #141414; margin: 14px 0 0; }}
        </style>
        <div class="wrap">
            <div class="sidebar-eyebrow">NEXT RACE · RD {round_num}</div>
            <div class="next-race-name">{race_name}</div>
            <div class="flip-clock-wrap">
                <div class="flip-unit">
                    <div class="flip-card" id="fc-days">
                        <div class="flip-card-inner">
                            <div class="flip-card-front"><span class="flip-num">--</span></div>
                            <div class="flip-card-back"><span class="flip-num">--</span></div>
                        </div>
                    </div>
                    <span class="flip-lbl">DAYS</span>
                </div>
                <div class="flip-sep">:</div>
                <div class="flip-unit">
                    <div class="flip-card" id="fc-hours">
                        <div class="flip-card-inner">
                            <div class="flip-card-front"><span class="flip-num">--</span></div>
                            <div class="flip-card-back"><span class="flip-num">--</span></div>
                        </div>
                    </div>
                    <span class="flip-lbl">HRS</span>
                </div>
                <div class="flip-sep">:</div>
                <div class="flip-unit">
                    <div class="flip-card" id="fc-mins">
                        <div class="flip-card-inner">
                            <div class="flip-card-front"><span class="flip-num">--</span></div>
                            <div class="flip-card-back"><span class="flip-num">--</span></div>
                        </div>
                    </div>
                    <span class="flip-lbl">MIN</span>
                </div>
                <div class="flip-sep">:</div>
                <div class="flip-unit">
                    <div class="flip-card" id="fc-secs">
                        <div class="flip-card-inner">
                            <div class="flip-card-front"><span class="flip-num">--</span></div>
                            <div class="flip-card-back"><span class="flip-num">--</span></div>
                        </div>
                    </div>
                    <span class="flip-lbl">SEC</span>
                </div>
            </div>
            <div class="rule"></div>
        </div>
        <script>
        (function() {{
            var target = {race_ts};
            function pad(n) {{ return n < 10 ? '0' + n : '' + n; }}
            function updateUnit(cardId, value) {{
                var cardEl = document.getElementById(cardId);
                if (!cardEl) return;
                var inner = cardEl.querySelector('.flip-card-inner');
                var frontNum = cardEl.querySelector('.flip-card-front .flip-num');
                var backNum = cardEl.querySelector('.flip-card-back .flip-num');
                
                var currentValue = cardEl.getAttribute('data-value');
                if (currentValue === value) return;
                cardEl.setAttribute('data-value', value);
                
                if (!currentValue) {{
                    frontNum.textContent = value;
                    backNum.textContent = value;
                    return;
                }}
                
                var isFrontActive = !inner.classList.contains('flipped');
                if (isFrontActive) {{
                    backNum.textContent = value;
                    inner.classList.add('flipped');
                }} else {{
                    frontNum.textContent = value;
                    inner.classList.remove('flipped');
                }}
            }}
            function tick() {{
                var diff = Math.max(0, Math.floor((target - Date.now()) / 1000));
                var d = Math.floor(diff / 86400);
                var h = Math.floor((diff % 86400) / 3600);
                var m = Math.floor((diff % 3600) / 60);
                var s = diff % 60;
                
                updateUnit('fc-days', pad(d));
                updateUnit('fc-hours', pad(h));
                updateUnit('fc-mins', pad(m));
                updateUnit('fc-secs', pad(s));
            }}
            tick();
            setInterval(tick, 1000);
        }})();
        </script>
        """, height=168, scrolling=False)

    # Navigation
    st.markdown('<div style="padding:0 24px 6px"><div class="sidebar-eyebrow">NAVIGATION</div></div>',
                unsafe_allow_html=True)
    page = st.radio(
        "",
        ["📡  Quali Lab", "🏁  Race Strategy", "🧬  Driver DNA", "🔮  Strategy Simulator"],
        label_visibility="collapsed",
    )

    # Data coverage
    st.markdown("""
    <div style="padding:0 24px">
        <div style="height:1px;background:#141414;margin:18px 0"></div>
        <div class="sidebar-eyebrow">DATA COVERAGE</div>
    </div>
    """, unsafe_allow_html=True)

    for item in [
        "Seasons: 2022 · 2023 · 2024 · 2026",
        "Circuits: 27 Grand Prix",
        "Drivers: 34 profiles",
        "Laps: 32,572 analyzed",
        "ML models: 3 trained",
    ]:
        st.markdown(f"""
        <div style="padding:0 24px">
            <div class="coverage-row">
                <div class="coverage-dot"></div>
                <span>{item}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar footer
    st.markdown("""
    <div style="padding:28px 24px 24px;margin-top:32px;border-top:1px solid #141414">
        <div style="text-align:center;font-size:10px;color:#515462;letter-spacing:0.05em;line-height:1.6;font-family:'Inter',sans-serif;">
            Built for the <svg viewBox="0 0 24 24" style="width: 10px; height: 10px; fill: #E10600; display: inline-block; vertical-align: middle; margin: -2px 2px 0 2px;"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg> of Formula 1.<br>
            <span style="font-weight: 700; color: #8a8d98; font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; display: block;">— Arin</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 0 — QUALI LAB (broadcast telemetry)
# ─────────────────────────────────────────
if "Quali Lab" in page:
    import quali_lab as ql

    qc1, qc2, qc3, qc4 = st.columns([1, 1.2, 1.2, 0.8])
    with qc1:
        q_year = st.selectbox("YEAR", [2024, 2026, 2023, 2022], index=0, key="ql_year")
    with qc2:
        q_race = st.selectbox("CIRCUIT", AVAILABLE_RACES[q_year], key="ql_race")
    tel_path = _telemetry_csv_path(q_year, q_race)
    drivers_in_session = []
    if os.path.exists(tel_path):
        drivers_in_session = sorted(pd.read_csv(tel_path)["Driver"].unique().tolist())
    if not drivers_in_session:
        drivers_in_session = sorted(driver_dna["driver"].unique().tolist())

    def _drv_label(code):
        return f"{DRIVER_NAMES.get(code, code)} ({code})"

    default_a = next((d for d in ["LEC", "VER", "NOR", "HAM"] if d in drivers_in_session), drivers_in_session[0])
    default_b = next((d for d in ["SAI", "HAM", "PIA", "RUS", "VER"] if d in drivers_in_session and d != default_a), drivers_in_session[-1])
    labels = [_drv_label(d) for d in drivers_in_session]
    label_to_code = {_drv_label(d): d for d in drivers_in_session}

    with qc3:
        d1_lab = st.selectbox("DRIVER A", labels, index=labels.index(_drv_label(default_a)), key="ql_d1")
    with qc4:
        d2_lab = st.selectbox(
            "DRIVER B", labels,
            index=labels.index(_drv_label(default_b)) if _drv_label(default_b) in labels else 0,
            key="ql_d2",
        )
    q_d1 = label_to_code[d1_lab]
    q_d2 = label_to_code[d2_lab]

    try:
        cd, s1, s2, _, t1s, t2s = load_qualy_speed_trace(q_year, q_race, q_d1, q_d2)
        ft1, hb1, cnr1 = ql.drive_style_from_speed(s1)
        ft2, hb2, cnr2 = ql.drive_style_from_speed(s2)
        gap = t1s - t2s
        if abs(gap) < 1e-4:
            gap1, gap2 = "LEADER", "LEADER"
        elif gap < 0:
            gap1, gap2 = "LEADER", f"+{abs(gap):.3f}s"
        else:
            gap1, gap2 = f"+{abs(gap):.3f}s", "LEADER"

        # Unique differentiable theme colors per driver (NOT team colors)
        c1, c2 = ql.pair_theme_colors(q_d1, q_d2)

        team1 = driver_dna[driver_dna["driver"] == q_d1]["team"].iloc[0] if not driver_dna[driver_dna["driver"] == q_d1].empty else "—"
        team2 = driver_dna[driver_dna["driver"] == q_d2]["team"].iloc[0] if not driver_dna[driver_dna["driver"] == q_d2].empty else "—"

        pos1, pos2 = (1, 2) if t1s <= t2s else (2, 1)

        n1 = DRIVER_NAMES.get(q_d1, q_d1)
        n2 = DRIVER_NAMES.get(q_d2, q_d2)

        # F1 TV Style Header (Customized to remove F1 & AWS marks per user request)
        circuit_upper = q_race.upper().replace("_", " ")
        header_logo_html = ""
        for p in ["header_logo.png", "logos/header_logo.png", "data/logos/header_logo.png"]:
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        h_data = base64.b64encode(f.read()).decode()
                    header_logo_html = f'<img src="data:image/png;base64,{h_data}" style="height: 25px; object-fit: contain; margin-right: 15px; display: block;" /><div style="width: 1px; height: 20px; background: #222530; margin-right: 15px;"></div>'
                    break
                except Exception:
                    pass
        st.markdown(
            f'<div style="background: #15161d; border-radius: 4px; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); border-left: 6px solid #E10600;">'
            f'<div style="display: flex; align-items: center;">'
            f'{header_logo_html}'
            f'<h1 style="margin: 0; font-family: \'Bebas Neue\', sans-serif; font-size: 1.8rem; font-weight: 900; color: #ffffff; letter-spacing: 0.05em; text-transform: uppercase; line-height: 1;">'
            f'{circuit_upper} - QUALIFYING ANALYSIS'
            f'</h1>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="zone-legend" style="margin:0 0 14px">'
            f'<span class="zone-pill"><span class="zone-dot" style="background:{c1}"></span>{n1}</span>'
            f'<span class="zone-pill"><span class="zone-dot" style="background:{c2}"></span>{n2}</span>'
            f'<span style="color:#555;letter-spacing:0.14em">MAP = FASTER DRIVER</span></div>',
            unsafe_allow_html=True,
        )

        left, mid, right = st.columns([1.15, 1.0, 1.15])
        with left:
            components.html(
                ql.build_driver_card_html(
                    pos1, q_d1, n1, team1, t1s, gap1, ft1, hb1, cnr1, c1, align="left",
                ),
                height=360, scrolling=False,
            )
        with mid:
            track = ql.load_track_outline(q_year, q_race)
            map_fig = ql.build_circuit_map(
                track, cd, s1, s2, c1, c2,
                f"{q_race.upper()}  ·  TRACK MAP",
            )
            st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})
        with right:
            components.html(
                ql.build_driver_card_html(
                    pos2, q_d2, n2, team2, t2s, gap2, ft2, hb2, cnr2, c2, align="right",
                ),
                height=360, scrolling=False,
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        fig = ql.build_speed_delta_figure(
            cd, s1, s2, n1, n2, c1, c2, f"{q_race} {q_year}",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except Exception as e:
        st.error(f"Quali Lab error: {e}")

# ─────────────────────────────────────────
# PAGE 1 — RACE STRATEGY
# ─────────────────────────────────────────
elif "Race Strategy" in page:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="page-title">RACE STRATEGY</div>
        <div class="page-subtitle">Command Center · ML-powered pit window optimization · 32,572 laps analyzed</div>
    </div>
    <div class="red-rule"></div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_race = st.selectbox("GRAND PRIX", sorted(clean_laps['race_name'].unique()))
    with c2:
        race_drivers = sorted(clean_laps[clean_laps['race_name']==selected_race]['Driver'].unique())
        driver_opts  = {f"{DRIVER_NAMES.get(d,d)} ({d})": d for d in race_drivers}
        sel_disp     = st.selectbox("DRIVER", list(driver_opts.keys()))
        selected_driver = driver_opts[sel_disp]
    with c3:
        selected_compound = st.selectbox("COMPOUND", ['SOFT','MEDIUM','HARD'], index=1)

    st.markdown('<div class="red-rule"></div>', unsafe_allow_html=True)

    driver_laps = clean_laps[
        (clean_laps['race_name']==selected_race) &
        (clean_laps['Driver']==selected_driver)
    ].copy()

    if not driver_laps.empty:
        team    = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else 'Unknown'
        tc      = TEAM_COLORS.get(team, '#E8002D')
        best    = driver_laps['LapTimeSec'].min()
        m, s    = int(best//60), best%60
        avg_deg = driver_laps['DeltaFromFastest'].mean()

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("TOTAL LAPS",      int(driver_laps['LapNumber'].max()))
        with m2: st.metric("FASTEST LAP",     f"{m}:{s:06.3f}")
        with m3: st.metric("AVG DEGRADATION", f"+{avg_deg:.2f}s")
        with m4: st.metric("TEAM",            team)

        st.markdown('<div class="dim-rule"></div>', unsafe_allow_html=True)

        y_min = driver_laps['LapTimeSec'].min() * 0.985
        y_max = driver_laps['LapTimeSec'].max() * 1.015
        yax   = ax('Lap Time (s)')
        yax['range'] = [y_min, y_max]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=driver_laps['LapNumber'], y=driver_laps['LapTimeSec'],
            mode='lines+markers',
            line=dict(color=tc, width=2),
            marker=dict(size=3),
            fill='tozeroy', fillcolor=hex_to_rgba(tc, 0.04),
            name=DRIVER_NAMES.get(selected_driver, selected_driver),
            hovertemplate='Lap %{x}: %{y:.3f}s<extra></extra>'
        ))
        fig.update_layout(
            **PLOT_BASE,
            title=dict(
                text=f'{DRIVER_NAMES.get(selected_driver,selected_driver).upper()}  ·  {selected_race}  ·  LAP TIME PROGRESSION',
                font=dict(size=11, color='#555', family='monospace'), x=0
            ),
            height=360, hovermode='x unified',
            xaxis=ax('Lap Number'), yaxis=yax
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="dim-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">TIRE DEGRADATION MODEL</div>', unsafe_allow_html=True)

        tire_model = None
        tire_model_path = f"models/tire_deg_{selected_race.replace(' ','_')}.pkl"
        if os.path.exists(tire_model_path):
            tire_model = joblib.load(tire_model_path)
        else:
            all_models = [f for f in os.listdir('models/') if f.startswith('tire_deg_')]
            race_clean = selected_race.lower().replace(' ','').replace('_','')
            for mf in all_models:
                mf_clean = mf.lower().replace('tire_deg_','').replace('.pkl','').replace('_','')
                if race_clean[:5] in mf_clean or mf_clean[:5] in race_clean:
                    tire_model = joblib.load(f'models/{mf}')
                    break

        if tire_model is None:
            race_laps = clean_laps[clean_laps['race_name']==selected_race].copy()
            if not race_laps.empty and 'DeltaFromFastest' in race_laps.columns:
                from sklearn.ensemble import GradientBoostingRegressor
                race_laps = race_laps.dropna(subset=['TyreLife','DeltaFromFastest','CompoundNum'])
                race_laps['TireLifeSq'] = race_laps['TyreLife']**2
                feats = ['TyreLife','TireLifeSq','CompoundNum','LapNum']
                if all(f in race_laps.columns for f in feats) and len(race_laps) > 20:
                    m_tmp = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    m_tmp.fit(race_laps[feats], race_laps['DeltaFromFastest'])
                    tire_model = m_tmp

        if tire_model:
            tire_ages   = np.arange(1, 46)
            comp_map    = {'SOFT':0,'MEDIUM':1,'HARD':2}
            comp_colors = {'SOFT':'#E8002D','MEDIUM':'#EF9F27','HARD':'#888780'}
            fig2 = go.Figure()
            for comp, cnum in comp_map.items():
                X_pred = pd.DataFrame({'TyreLife':tire_ages,'TireLifeSq':tire_ages**2,
                                       'CompoundNum':cnum,'LapNum':30})
                deltas = tire_model.predict(X_pred)
                fig2.add_trace(go.Scatter(
                    x=tire_ages, y=deltas, mode='lines', name=comp,
                    line=dict(color=comp_colors[comp], width=2.5),
                    hovertemplate=f'{comp} · Lap %{{x}}: +%{{y:.2f}}s<extra></extra>'
                ))
            fig2.update_layout(**PLOT_BASE, height=300,
                               xaxis=ax('Tire Age (laps)'),
                               yaxis=ax('Delta from fastest (s)'))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown('<div style="color:#555;font-size:12px;padding:1rem 0">Insufficient data for tire model on this circuit.</div>',
                        unsafe_allow_html=True)
    else:
        st.warning("No data available for this combination.")

# ─────────────────────────────────────────
# PAGE 2 — DRIVER DNA
# ─────────────────────────────────────────
elif "Driver DNA" in page:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="page-title">DRIVER DNA LAB</div>
        <div class="page-subtitle">Telemetry fingerprinting · K-Means clustering · PCA · 76,721 telemetry points</div>
    </div>
    <div class="red-rule"></div>
    """, unsafe_allow_html=True)

    drivers_avail = sorted(driver_dna['driver'].unique())
    driver_disp   = {f"{DRIVER_NAMES.get(d,d)} ({d})": d for d in drivers_avail}

    c1, c2 = st.columns(2)
    with c1:
        d1k     = st.selectbox("DRIVER 1", list(driver_disp.keys()),
                               index=next((i for i,k in enumerate(driver_disp) if driver_disp[k]=='VER'),0))
        driver1 = driver_disp[d1k]
    with c2:
        d2k     = st.selectbox("DRIVER 2", list(driver_disp.keys()),
                               index=next((i for i,k in enumerate(driver_disp) if driver_disp[k]=='HAM'),1))
        driver2 = driver_disp[d2k]

    st.markdown('<div class="red-rule"></div>', unsafe_allow_html=True)

    raw_feats = ['full_throttle_pct','heavy_brake_pct','avg_corner_speed',
                 'throttle_smoothness','coast_pct','high_speed_pct']
    cats      = ['Full Throttle','Heavy Braking','Corner Speed',
                 'Smoothness','Coasting','High Speed']

    dna_n = driver_dna.copy()
    for f in raw_feats:
        mn, mx = driver_dna[f].min(), driver_dna[f].max()
        dna_n[f+'_n'] = 30 + ((driver_dna[f]-mn)/(mx-mn))*70

    d1r  = dna_n[dna_n['driver']==driver1].iloc[0]
    d2r  = dna_n[dna_n['driver']==driver2].iloc[0]
    nc   = [f+'_n' for f in raw_feats]
    v1   = [round(d1r[c],1) for c in nc]+[round(d1r[nc[0]],1)]
    v2   = [round(d2r[c],1) for c in nc]+[round(d2r[nc[0]],1)]
    catc = cats+[cats[0]]

    t1  = d1r['team'];  t2  = d2r['team']
    c1c = TEAM_COLORS.get(t1,'#E8002D')
    c2c = TEAM_COLORS.get(t2,'#378ADD')

    col_r, col_s = st.columns([1.4, 0.6])

    with col_r:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=v1, theta=catc, fill='toself',
            fillcolor=hex_to_rgba(c1c, 0.18),
            line=dict(color=c1c, width=2.5),
            name=DRIVER_NAMES.get(driver1, driver1),
            marker=dict(size=6, color=c1c)
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=v2, theta=catc, fill='toself',
            fillcolor=hex_to_rgba(c2c, 0.18),
            line=dict(color=c2c, width=2.5),
            name=DRIVER_NAMES.get(driver2, driver2),
            marker=dict(size=6, color=c2c)
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor='#080808',
                radialaxis=dict(visible=True, range=[0,100],
                    tickvals=[25,50,75,100],
                    tickfont=dict(color='#222',size=8),
                    gridcolor='#111', linecolor='#111'),
                angularaxis=dict(
                    tickfont=dict(color='#999',size=11,family='monospace'),
                    gridcolor='#111', linecolor='#161616')
            ),
            paper_bgcolor='#080808',
            title=dict(text='DRIVING STYLE FINGERPRINT',
                       font=dict(size=11,color='#444',family='monospace'),x=0.5,xanchor='center'),
            legend=dict(bgcolor='rgba(8,8,8,0.95)',bordercolor='#141414',borderwidth=1,
                        font=dict(color='#888',size=11,family='monospace'),
                        x=0.5,y=-0.07,xanchor='center',orientation='h'),
            height=500,
            margin=dict(t=50,b=80,l=50,r=50),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_s:
        for drv, chex in [(driver1,c1c),(driver2,c2c)]:
            d_raw     = driver_dna[driver_dna['driver']==drv].iloc[0]
            fullname  = DRIVER_NAMES.get(drv, drv)
            archetype = d_raw.get('archetype','Unknown')
            if not isinstance(archetype,str): archetype = 'Unknown'
            st.markdown(f"""
            <div class="driver-card" style="border-top:2px solid {chex}">
                <div class="driver-card-name" style="color:{chex}">{fullname.upper()}</div>
                <div class="driver-card-team">{d_raw['team']}</div>
                <div class="stat-row">
                    <span class="stat-label">Archetype</span>
                    <span class="stat-value">{archetype}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Full throttle</span>
                    <span class="stat-value">{d_raw['full_throttle_pct']:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Heavy braking</span>
                    <span class="stat-value">{d_raw['heavy_brake_pct']:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Corner speed</span>
                    <span class="stat-value">{d_raw['avg_corner_speed']:.0f} km/h</span>
                </div>
                <div class="stat-row" style="border-bottom:none">
                    <span class="stat-label">Max speed</span>
                    <span class="stat-value">{d_raw['max_speed']:.0f} km/h</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Telemetry
    st.markdown('<div class="red-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">QUALIFYING SPEED TRACE · HEAD TO HEAD</div>',
                unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns([1,1,1])
    with tc1:
        tel_year = st.selectbox("YEAR", [2026,2024,2023,2022], index=1)
    with tc2:
        tel_race = st.selectbox("CIRCUIT", AVAILABLE_RACES[tel_year])
    with tc3:
        st.markdown("<div style='padding-top:1.6rem'>", unsafe_allow_html=True)
        load_tel = st.button("LOAD TELEMETRY ▶", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    # Persist button intent across reruns (Streamlit buttons are one-shot)
    if 'tel_load_requested' not in st.session_state:
        st.session_state.tel_load_requested = False
        st.session_state.tel_params = None

    if load_tel:
        st.session_state.tel_load_requested = True
        st.session_state.tel_params = (tel_year, tel_race, driver1, driver2)

    if st.session_state.tel_load_requested and st.session_state.tel_params:
        year, race, d1, d2 = st.session_state.tel_params
        with st.spinner(f"Loading {race} {year} qualifying telemetry..."):
            try:
                cd, s1, s2, delt, t1s, t2s = load_qualy_speed_trace(year, race, d1, d2)

                n1 = DRIVER_NAMES.get(d1, d1)
                n2 = DRIVER_NAMES.get(d2, d2)

                fig_t = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32],
                                      vertical_spacing=0.05, shared_xaxes=True)
                fig_t.add_trace(go.Scatter(
                    x=cd, y=s1, mode='lines',
                    name=f'{n1}  ·  {int(t1s//60)}:{t1s%60:06.3f}',
                    line=dict(color=c1c, width=2.5),
                    hovertemplate=f'{d1}: %{{y:.0f}} km/h<extra></extra>'
                ), row=1, col=1)
                fig_t.add_trace(go.Scatter(
                    x=cd, y=s2, mode='lines',
                    name=f'{n2}  ·  {int(t2s//60)}:{t2s%60:06.3f}',
                    line=dict(color=c2c, width=2.5),
                    hovertemplate=f'{d2}: %{{y:.0f}} km/h<extra></extra>'
                ), row=1, col=1)
                fig_t.add_trace(go.Scatter(
                    x=cd, y=delt, mode='lines',
                    fill='tozeroy', fillcolor=hex_to_rgba(c1c, 0.1),
                    line=dict(color=c1c, width=1.5),
                    name=f'Δ Speed ({d1} − {d2})',
                    hovertemplate='Δ: %{y:.1f} km/h<extra></extra>'
                ), row=2, col=1)
                fig_t.add_hline(y=0, line=dict(color='#1a1a1a', width=1), row=2, col=1)

                for row in [1, 2]:
                    fig_t.update_xaxes(
                        gridcolor='#0f0f0f', zerolinecolor='#111',
                        tickfont=dict(color='#555', size=9), row=row, col=1
                    )
                    fig_t.update_yaxes(
                        gridcolor='#0f0f0f', zerolinecolor='#111',
                        tickfont=dict(color='#555', size=9), row=row, col=1
                    )
                fig_t.update_yaxes(
                    title_text='Speed (km/h)',
                    title_font=dict(color='#666', size=10, family='monospace'), row=1, col=1
                )
                fig_t.update_yaxes(
                    title_text='Δ Speed',
                    title_font=dict(color='#666', size=10, family='monospace'), row=2, col=1
                )
                fig_t.update_xaxes(
                    title_text='Distance (m)',
                    title_font=dict(color='#666', size=10, family='monospace'), row=2, col=1
                )

                fig_t.update_layout(
                    **PLOT_BASE,
                    title=dict(
                        text=f'{race.upper()} {year}  ·  QUALIFYING  ·  {d1} vs {d2}',
                        font=dict(size=12, color='#555', family='monospace'), x=0.5, xanchor='center'
                    ),
                    height=600,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_t, use_container_width=True)

            except Exception as e:
                st.error(f"Telemetry error: {e}")

# ─────────────────────────────────────────
# PAGE 3 — STRATEGY SIMULATOR
# ─────────────────────────────────────────
elif "Strategy Simulator" in page:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="page-title">STRATEGY SIMULATOR</div>
        <div class="page-subtitle">ML pit decision engine · GradientBoosting · ROC-AUC 0.745 · 941 real pit stops</div>
    </div>
    <div class="red-rule"></div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown('<div class="section-eyebrow">RACE SITUATION</div>', unsafe_allow_html=True)
        sim_tire = st.slider("TIRE AGE (laps)", 1, 50, 18)
        sim_comp = st.selectbox("COMPOUND", ['SOFT','MEDIUM','HARD'], index=1)
        sim_pos  = st.slider("CURRENT POSITION", 1, 20, 5)
    with sc2:
        st.markdown('<div class="section-eyebrow">RACE STATE</div>', unsafe_allow_html=True)
        sim_prog = st.slider("RACE PROGRESS (%)", 0, 100, 45)
        sim_laps = st.slider("LAPS REMAINING", 1, 60, 25)
        sim_team = st.selectbox("TEAM", list(TEAM_COLORS.keys()), index=0)

    st.markdown('<div class="red-rule"></div>', unsafe_allow_html=True)

    cmap = {'SOFT':0,'MEDIUM':1,'HARD':2}
    feat = pd.DataFrame([{
        'tire_age_at_pit': sim_tire,
        'compound_num':    cmap[sim_comp],
        'race_progress':   sim_prog/100,
        'position_before': sim_pos/20,
        'laps_remaining':  sim_laps,
    }])
    prob     = strategy_model.predict_proba(feat)[0][1]
    decision = "PIT NOW" if prob >= 0.5 else "STAY OUT"
    dc       = "#E8002D" if prob >= 0.5 else "#1D9E75"
    tc_team  = TEAM_COLORS.get(sim_team,'#E8002D')

    rc1, rc2, rc3 = st.columns([1,1,1])

    with rc1:
        st.markdown(f"""
        <div class="decision-box" style="border-top:3px solid {dc}">
            <div class="decision-label">ML RECOMMENDATION</div>
            <div class="decision-text" style="color:{dc}">{decision}</div>
            <div style="margin:16px 0 12px;height:3px;border-radius:2px;
                        background:linear-gradient(90deg,{dc} {prob*100:.0f}%,#111 {prob*100:.0f}%)"></div>
            <div class="decision-conf">Confidence: {prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title=dict(text="PIT PROBABILITY",font=dict(color='#555',size=10,family='monospace')),
            number=dict(suffix="%",font=dict(color='#fff',size=26,family='monospace')),
            gauge=dict(
                axis=dict(range=[0,100],tickfont=dict(color='#333',size=8),tickcolor='#111'),
                bar=dict(color=dc,thickness=0.55),
                bgcolor='#0a0a0a',bordercolor='#111',borderwidth=1,
                steps=[
                    dict(range=[0,35],  color='#0a120a'),
                    dict(range=[35,65], color='#12120a'),
                    dict(range=[65,100],color='#120a0a'),
                ],
                threshold=dict(line=dict(color='#444',width=1.5),thickness=0.7,value=50)
            )
        ))
        fig_g.update_layout(
            paper_bgcolor='#080808',
            font=dict(color='#fff',family='monospace'),
            height=240,margin=dict(t=40,b=10,l=25,r=25)
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with rc3:
        if prob >= 0.7:   msg,mc = "Strong pit signal. Tire cliff approaching. Box this lap.", "#E8002D"
        elif prob >= 0.5: msg,mc = "Marginal window. Weigh track position vs fresh rubber.",  "#EF9F27"
        elif prob >= 0.3: msg,mc = "Tires still viable. Hold position if it matters.",         "#1D9E75"
        else:             msg,mc = "No pit required. Performance still strong.",               "#378ADD"

        st.markdown(f"""
        <div class="analysis-box" style="border-top:2px solid {mc}">
            <div class="section-eyebrow">STRATEGY ANALYSIS</div>
            <div style="font-size:12px;color:#777;line-height:1.8;margin-bottom:1.25rem">{msg}</div>
            <div class="stat-row">
                <span class="stat-label">Tire age</span>
                <span class="stat-value">{sim_tire} laps</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Compound</span>
                <span class="stat-value">{sim_comp}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Position</span>
                <span class="stat-value">P{sim_pos}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Race progress</span>
                <span class="stat-value">{sim_prog}%</span>
            </div>
            <div class="stat-row" style="border-bottom:none">
                <span class="stat-label">Laps remaining</span>
                <span style="color:{tc_team};font-size:0.72rem">{sim_laps}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="dim-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">PIT WINDOW SENSITIVITY · TIRE AGE vs PROBABILITY</div>',
                unsafe_allow_html=True)

    tire_ages = np.arange(1, 51)
    probs = [strategy_model.predict_proba(pd.DataFrame([{
        'tire_age_at_pit': ta,
        'compound_num':    cmap[sim_comp],
        'race_progress':   sim_prog/100,
        'position_before': sim_pos/20,
        'laps_remaining':  sim_laps
    }]))[0][1]*100 for ta in tire_ages]

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=tire_ages, y=probs, mode='lines',
        fill='tozeroy', fillcolor='rgba(232,0,45,0.06)',
        line=dict(color='#E8002D', width=2),
        hovertemplate='Age %{x}: %{y:.1f}%<extra></extra>'
    ))
    fig_s.add_hline(y=50, line=dict(color='#1a1a1a',width=1,dash='dash'),
                    annotation_text="THRESHOLD",
                    annotation_font=dict(color='#555',size=9,family='monospace'))
    fig_s.add_vline(x=sim_tire, line=dict(color='#EF9F27',width=1.5,dash='dot'),
                    annotation_text="NOW",
                    annotation_font=dict(color='#EF9F27',size=9,family='monospace'))

    sens_yax = ax('Pit Probability (%)')
    sens_yax['range'] = [0, 100]
    # Avoid passing 'margin' twice (PLOT_BASE already contains margin)
    _layout = dict(PLOT_BASE)
    _layout.update(
        height=280,
        xaxis=ax('Tire Age (laps)'),
        yaxis=sens_yax,
        margin=dict(t=30, b=45, l=55, r=20)
    )
    fig_s.update_layout(**_layout)
    st.plotly_chart(fig_s, use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; max-width: 1480px; margin: 0 auto; flex-wrap: wrap; gap: 10px;">
    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
      <span style="font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; font-weight: 900; letter-spacing: 0.05em; color: #515462;">STRAT</span>
      <span style="font-family: 'Inter', sans-serif; font-size: 0.62rem; color: #3a3d46; letter-spacing: 0.05em; text-transform: uppercase;">AI-powered Formula 1 Analytics · FastF1 · scikit-learn · Streamlit · Plotly</span>
    </div>
    <div style="font-family: 'Inter', sans-serif; font-size: 0.65rem; color: #8a8d98; letter-spacing: 0.05em; display: flex; align-items: center; gap: 4px;">
      <span>Built for the</span>
      <svg viewBox="0 0 24 24" style="width: 10px; height: 10px; fill: #E10600; display: inline-block; vertical-align: middle; margin: 0 1px 0 2px;"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>
      <span>of Formula 1.</span>
      <span style="font-weight: 700; color: #bbbbbb; margin-left: 6px; text-transform: uppercase;">— Arin</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)