import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import fastf1
import os
import warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="F1 Strategy Intelligence",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Force sidebar open always
st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { 
    transform: none !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
AVAILABLE_RACES = {
    2022: ['Bahrain', 'Saudi Arabia', 'Australia', 'Spain', 'Monaco',
           'Britain', 'Hungary', 'Belgium', 'Italy'],
    2023: ['Bahrain', 'Saudi Arabia', 'Australia', 'Spain', 'Monaco',
           'Britain', 'Hungary', 'Italy', 'Singapore'],
    2024: ['Bahrain', 'Saudi Arabia', 'Australia', 'Spain', 'Monaco',
           'Britain', 'Hungary', 'Italy', 'Singapore'],
    2026: ['Australia', 'China', 'Japan'],
}

DRIVER_NAMES = {
    'VER': 'Max Verstappen', 'LEC': 'Charles Leclerc',
    'RUS': 'George Russell', 'SAI': 'Carlos Sainz',
    'PER': 'Sergio Perez', 'ALO': 'Fernando Alonso',
    'NOR': 'Lando Norris', 'PIA': 'Oscar Piastri',
    'HAM': 'Lewis Hamilton', 'HUL': 'Nico Hulkenberg',
    'TSU': 'Yuki Tsunoda', 'STR': 'Lance Stroll',
    'ALB': 'Alexander Albon', 'RIC': 'Daniel Ricciardo',
    'MAG': 'Kevin Magnussen', 'BOT': 'Valtteri Bottas',
    'ZHO': 'Guanyu Zhou', 'SAR': 'Logan Sargeant',
    'OCO': 'Esteban Ocon', 'GAS': 'Pierre Gasly',
    'ANT': 'Kimi Antonelli', 'LAW': 'Liam Lawson',
    'HAD': 'Isack Hadjar', 'BEA': 'Oliver Bearman',
    'BOR': 'Gabriel Bortoleto', 'COL': 'Franco Colapinto',
    'DOO': 'Jack Doohan', 'LIN': 'Jack Doohan',
}

TEAM_COLORS = {
    'Red Bull Racing': '#3671C6', 'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2', 'McLaren': '#FF8000',
    'Aston Martin': '#229971', 'Alpine': '#FF87BC',
    'Williams': '#64C4FF', 'AlphaTauri': '#6692FF',
    'RB': '#6692FF', 'Racing Bulls': '#6692FF',
    'Alfa Romeo': '#C92D4B', 'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD', 'Cadillac': '#CC0000',
    'Audi': '#F50000',
}

# ─────────────────────────────────────────
# RACE SCHEDULE — auto from FastF1
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
                'name': row['EventName'],
                'date': race_date,
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
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
}

.stApp {
    background-color: #060606 !important;
    background-image:
        linear-gradient(rgba(226,75,74,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(226,75,74,0.03) 1px, transparent 1px),
        radial-gradient(ellipse at 20% 50%, rgba(226,75,74,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(54,113,198,0.04) 0%, transparent 60%);
    background-size: 40px 40px, 40px 40px, 100% 100%, 100% 100%;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a0a 0%, #080808 100%) !important;
    border-right: 1px solid #151515 !important;
    min-width: 260px !important;
}
[data-testid="stSidebar"] > div {
    padding: 0 !important;
}

/* Typography */
h1 {
    font-family: 'Bebas Neue', monospace !important;
    font-size: 3.2rem !important;
    letter-spacing: 0.06em !important;
    color: #ffffff !important;
    line-height: 0.95 !important;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-family: 'Bebas Neue', monospace !important;
    font-size: 1.6rem !important;
    letter-spacing: 0.08em !important;
    color: #ffffff !important;
}
h3 {
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #2a2a2a !important;
    margin-bottom: 0.75rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #111 !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #0a0a0a !important;
    border: 1px solid #141414 !important;
    border-top: 2px solid #1a1a1a !important;
    border-radius: 4px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"]:hover {
    border-top-color: #E24B4A !important;
    transition: border-color 0.3s !important;
}
[data-testid="metric-container"] label {
    color: #333 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.4rem !important;
    font-weight: 400 !important;
}

/* Inputs */
.stSelectbox > div > div {
    background: #0a0a0a !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 4px !important;
    color: #ccc !important;
    font-size: 0.85rem !important;
}
.stSelectbox label {
    color: #333 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
.stSlider > div > div > div { background: #E24B4A !important; }
.stSlider label {
    color: #333 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #E24B4A !important;
    color: #E24B4A !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.12em !important;
    font-size: 0.72rem !important;
    padding: 0.6rem 1.4rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #E24B4A !important;
    color: #000000 !important;
    font-weight: 500 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #111 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #2a2a2a !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.12em !important;
    font-size: 0.7rem !important;
    padding: 0.8rem 1.5rem !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #E24B4A !important;
}

/* Radio */
.stRadio label { color: #666 !important; font-size: 0.8rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #888 !important; }

/* Alerts */
.stAlert { border-radius: 3px !important; font-size: 0.8rem !important; }

#MainMenu, footer, header { visibility: hidden; }

/* Custom components */
.f1-badge {
    display: inline-flex;
    align-items: center;
    background: #E24B4A;
    color: #000;
    font-family: 'Bebas Neue', monospace;
    font-size: 1.1rem;
    padding: 3px 10px 2px;
    border-radius: 2px;
    letter-spacing: 0.05em;
    margin-right: 10px;
}
.sidebar-title {
    font-family: 'Bebas Neue', monospace;
    font-size: 1.2rem;
    color: #fff;
    letter-spacing: 0.12em;
}
.sidebar-sub {
    font-size: 0.6rem;
    color: #2a2a2a;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 2px;
}
.red-rule {
    height: 1px;
    background: linear-gradient(90deg, #E24B4A 0%, rgba(226,75,74,0.3) 60%, transparent 100%);
    border: none;
    margin: 1rem 0;
}
.dim-rule {
    height: 1px;
    background: #111;
    border: none;
    margin: 1rem 0;
}
.section-eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #2a2a2a;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #111;
}
.countdown-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
    margin-top: 8px;
}
.countdown-cell {
    background: #0d0d0d;
    border: 1px solid #141414;
    border-top: 2px solid #E24B4A;
    border-radius: 3px;
    padding: 10px 6px 8px;
    text-align: center;
}
.countdown-num {
    font-family: 'Bebas Neue', monospace;
    font-size: 2.2rem;
    color: #E24B4A;
    line-height: 1;
    display: block;
}
.countdown-lbl {
    font-size: 0.55rem;
    color: #2a2a2a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    display: block;
    margin-top: 3px;
}
.race-label {
    font-family: 'Bebas Neue', monospace;
    font-size: 0.95rem;
    color: #888;
    letter-spacing: 0.12em;
    text-align: center;
    margin-bottom: 4px;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    cursor: pointer;
    border-left: 2px solid transparent;
    transition: all 0.15s;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #444;
}
.nav-item:hover { color: #888; border-left-color: #333; }
.nav-item.active { color: #fff; border-left-color: #E24B4A; }
.coverage-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 0.7rem;
    color: #2a2a2a;
    letter-spacing: 0.05em;
}
.coverage-dot {
    width: 4px;
    height: 4px;
    background: #E24B4A;
    border-radius: 50%;
    flex-shrink: 0;
}
.page-header {
    margin-bottom: 1.5rem;
}
.page-title {
    font-family: 'Bebas Neue', monospace;
    font-size: 3rem;
    color: #fff;
    letter-spacing: 0.06em;
    line-height: 0.9;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.7rem;
    color: #2a2a2a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.driver-card {
    background: #0a0a0a;
    border: 1px solid #141414;
    border-radius: 4px;
    padding: 1.25rem;
    height: 100%;
}
.driver-card-name {
    font-family: 'Bebas Neue', monospace;
    font-size: 1.4rem;
    letter-spacing: 0.08em;
    line-height: 1;
    margin-bottom: 2px;
}
.driver-card-team {
    font-size: 0.65rem;
    color: #333;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #0f0f0f;
    font-size: 0.72rem;
}
.stat-label { color: #2a2a2a; letter-spacing: 0.1em; text-transform: uppercase; }
.stat-value { color: #888; }
.decision-box {
    background: #0a0a0a;
    border: 1px solid #141414;
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
}
.decision-label {
    font-size: 0.6rem;
    color: #2a2a2a;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.decision-text {
    font-family: 'Bebas Neue', monospace;
    font-size: 3.5rem;
    letter-spacing: 0.1em;
    line-height: 1;
}
.decision-conf {
    font-size: 0.7rem;
    color: #333;
    letter-spacing: 0.15em;
    margin-top: 12px;
}
.analysis-box {
    background: #0a0a0a;
    border: 1px solid #141414;
    border-radius: 4px;
    padding: 1.25rem;
    height: 100%;
}
.footer-wrap {
    margin-top: 4rem;
    padding: 1.5rem 0;
    border-top: 1px solid #0f0f0f;
    text-align: center;
}
.footer-main {
    font-family: 'Bebas Neue', monospace;
    font-size: 0.9rem;
    color: #1a1a1a;
    letter-spacing: 0.2em;
    margin-bottom: 4px;
}
.footer-sub {
    font-size: 0.65rem;
    color: #151515;
    letter-spacing: 0.15em;
    margin-bottom: 4px;
}
.footer-heart {
    color: #E24B4A;
    font-size: 14px;
}
.footer-by {
    font-size: 0.65rem;
    color: #1f1f1f;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA + MODELS
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    clean_laps = pd.read_csv('data/clean_laps.csv')
    optimal_pit = pd.read_csv('data/optimal_pit_df.csv')
    driver_dna = pd.read_csv('data/driver_dna.csv')
    return clean_laps, optimal_pit, driver_dna

@st.cache_resource
def load_models():
    strategy_model = joblib.load('models/strategy_model.pkl')
    dna_kmeans = joblib.load('models/dna_kmeans.pkl')
    dna_scaler = joblib.load('models/dna_scaler.pkl')
    dna_pca = joblib.load('models/dna_pca.pkl')
    return strategy_model, dna_kmeans, dna_scaler, dna_pca

clean_laps, optimal_pit, driver_dna = load_data()
strategy_model, dna_kmeans, dna_scaler, dna_pca = load_models()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def hex_to_rgba(h, a=0.2):
    h = h.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{a})'

PLOT_BASE = dict(
    plot_bgcolor='#080808',
    paper_bgcolor='#080808',
    font=dict(color='#ffffff', family='monospace', size=11),
    hoverlabel=dict(bgcolor='#0f0f0f', bordercolor='#222',
                    font=dict(color='#fff', size=11, family='monospace')),
    legend=dict(bgcolor='rgba(8,8,8,0.95)', bordercolor='#1a1a1a',
                borderwidth=1, font=dict(color='#888', size=10, family='monospace')),
    margin=dict(t=50, b=45, l=55, r=20),
)

def ax(title=''):
    return dict(title=title, gridcolor='#0f0f0f', zerolinecolor='#151515',
                tickfont=dict(color='#2a2a2a', size=9),
                title_font=dict(color='#333', size=10, family='monospace'))

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding: 1.5rem 1.25rem 0">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px">
            <div class="f1-badge">F1</div>
            <div>
                <div class="sidebar-title">STRATEGY</div>
                <div class="sidebar-sub">Intelligence System</div>
            </div>
        </div>
        <div class="red-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    # Countdown
    next_race = get_race_schedule()
    if next_race:
        now = datetime.now(timezone.utc)
        delta = next_race['date'] - now
        days = max(0, delta.days)
        hours = max(0, delta.seconds // 3600)
        minutes = max(0, (delta.seconds % 3600) // 60)
        st.markdown(f"""
        <div style="padding: 0 1.25rem">
            <div class="section-eyebrow">NEXT RACE · RD {next_race['round']}</div>
            <div class="race-label">{next_race['name'].upper()}</div>
            <div class="countdown-grid">
                <div class="countdown-cell">
                    <span class="countdown-num">{days:02d}</span>
                    <span class="countdown-lbl">DAYS</span>
                </div>
                <div class="countdown-cell">
                    <span class="countdown-num">{hours:02d}</span>
                    <span class="countdown-lbl">HRS</span>
                </div>
                <div class="countdown-cell">
                    <span class="countdown-num">{minutes:02d}</span>
                    <span class="countdown-lbl">MIN</span>
                </div>
            </div>
            <div class="dim-rule" style="margin: 1rem 0"></div>
        </div>
        """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div style="padding: 0 1.25rem"><div class="section-eyebrow">NAVIGATION</div></div>',
                unsafe_allow_html=True)
    page = st.radio("", ["🏁  Race Strategy", "🧬  Driver DNA", "🔮  Strategy Simulator"],
                    label_visibility="collapsed")

    st.markdown("""
    <div style="padding: 0 1.25rem">
        <div class="dim-rule"></div>
        <div class="section-eyebrow">DATA COVERAGE</div>
    </div>
    """, unsafe_allow_html=True)

    for item in [
        "Seasons: 2022 · 2023 · 2024 · 2026",
        "Circuits: 27 Grand Prix",
        "Drivers: 34 profiles",
        "Laps: 32,572 analyzed",
        "Models: 3 ML trained",
    ]:
        st.markdown(f"""
        <div style="padding: 0 1.25rem">
            <div class="coverage-item">
                <div class="coverage-dot"></div>
                <span>{item}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="padding: 1.5rem 1.25rem; margin-top: 2rem; border-top: 1px solid #0f0f0f">
        <div style="text-align:center">
            <div style="font-size:10px; color:#1a1a1a; letter-spacing:0.15em; line-height:2">
                FastF1 · scikit-learn · Streamlit<br>
                <span style="color:#E24B4A">♥</span>
                <span style="color:#1f1f1f; letter-spacing:0.2em"> MADE BY ARIN</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 1 — RACE STRATEGY
# ─────────────────────────────────────────
if "Race Strategy" in page:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">RACE STRATEGY</div>
        <div class="page-subtitle">Command Center · ML-powered pit window optimization · 32,572 laps analyzed</div>
    </div>
    <div class="red-rule"></div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_race = st.selectbox("GRAND PRIX", sorted(clean_laps['race_name'].unique()))
    with c2:
        race_drivers = sorted(clean_laps[clean_laps['race_name'] == selected_race]['Driver'].unique())
        driver_opts = {f"{DRIVER_NAMES.get(d, d)} ({d})": d for d in race_drivers}
        sel_disp = st.selectbox("DRIVER", list(driver_opts.keys()))
        selected_driver = driver_opts[sel_disp]
    with c3:
        selected_compound = st.selectbox("COMPOUND", ['SOFT', 'MEDIUM', 'HARD'], index=1)

    st.markdown('<div class="red-rule"></div>', unsafe_allow_html=True)

    driver_laps = clean_laps[
        (clean_laps['race_name'] == selected_race) &
        (clean_laps['Driver'] == selected_driver)
    ].copy()

    if not driver_laps.empty:
        team = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else 'Unknown'
        tc = TEAM_COLORS.get(team, '#E24B4A')
        best = driver_laps['LapTimeSec'].min()
        m, s = int(best // 60), best % 60
        avg_deg = driver_laps['DeltaFromFastest'].mean()

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("TOTAL LAPS", int(driver_laps['LapNumber'].max()))
        with m2: st.metric("FASTEST LAP", f"{m}:{s:06.3f}")
        with m3: st.metric("AVG DEGRADATION", f"+{avg_deg:.2f}s")
        with m4: st.metric("TEAM", team)

        st.markdown('<div class="dim-rule"></div>', unsafe_allow_html=True)

        # Lap time chart
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
                font=dict(size=12, color='#333', family='monospace'), x=0
            ),
            height=360, hovermode='x unified',
            xaxis=ax('Lap Number'), 
            yaxis={**ax('Lap Time (s)'), 'range': [driver_laps['LapTimeSec'].min()*0.98, driver_laps['LapTimeSec'].max()*1.02]}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tire degradation
        st.markdown('<div class="dim-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">TIRE DEGRADATION MODEL</div>', unsafe_allow_html=True)

        # Try exact match first, then fuzzy match
        tire_model = None
        tire_model_path = f"models/tire_deg_{selected_race.replace(' ', '_')}.pkl"
        if os.path.exists(tire_model_path):
            tire_model = joblib.load(tire_model_path)
        else:
            # Try to find a matching model by race name fragment
            all_models = [f for f in os.listdir('models/') if f.startswith('tire_deg_')]
            race_clean = selected_race.lower().replace(' ', '').replace('_', '')
            for mf in all_models:
                mf_clean = mf.lower().replace('tire_deg_', '').replace('.pkl','').replace('_','').replace('0','').replace('1','').replace('2','').replace('3','').replace('4','')
                if race_clean[:6] in mf_clean or mf_clean[:6] in race_clean:
                    tire_model = joblib.load(f'models/{mf}')
                    break

        if tire_model:
            tire_ages = np.arange(1, 46)
            comp_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
            comp_colors = {'SOFT': '#E24B4A', 'MEDIUM': '#EF9F27', 'HARD': '#888780'}
            fig2 = go.Figure()
            for comp, cnum in comp_map.items():
                X_pred = pd.DataFrame({
                    'TyreLife': tire_ages, 'TireLifeSq': tire_ages**2,
                    'CompoundNum': cnum, 'LapNum': 30
                })
                deltas = tire_model.predict(X_pred)
                fig2.add_trace(go.Scatter(
                    x=tire_ages, y=deltas, mode='lines',
                    name=comp, line=dict(color=comp_colors[comp], width=2.5),
                    hovertemplate=f'{comp} · Lap %{{x}}: +%{{y:.2f}}s<extra></extra>'
                ))
            fig2.update_layout(
                **PLOT_BASE, height=300,
                xaxis=ax('Tire Age (laps)'), yaxis=ax('Delta from fastest (s)')
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Build a generic model from clean_laps for this race
            race_laps = clean_laps[clean_laps['race_name'] == selected_race].copy()
            if not race_laps.empty and 'DeltaFromFastest' in race_laps.columns:
                from sklearn.ensemble import GradientBoostingRegressor
                race_laps = race_laps.dropna(subset=['TyreLife','DeltaFromFastest','CompoundNum'])
                race_laps['TireLifeSq'] = race_laps['TyreLife']**2
                features = ['TyreLife','TireLifeSq','CompoundNum','LapNum']
                if all(f in race_laps.columns for f in features):
                    X = race_laps[features]
                    y = race_laps['DeltaFromFastest']
                    if len(X) > 20:
                        m_temp = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        m_temp.fit(X, y)
                        tire_ages = np.arange(1, 46)
                        comp_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
                        comp_colors = {'SOFT': '#E24B4A', 'MEDIUM': '#EF9F27', 'HARD': '#888780'}
                        fig2 = go.Figure()
                        for comp, cnum in comp_map.items():
                            X_pred = pd.DataFrame({
                                'TyreLife': tire_ages, 'TireLifeSq': tire_ages**2,
                                'CompoundNum': cnum, 'LapNum': 30
                            })
                            deltas = m_temp.predict(X_pred)
                            fig2.add_trace(go.Scatter(
                                x=tire_ages, y=deltas, mode='lines',
                                name=comp, line=dict(color=comp_colors[comp], width=2.5),
                                hovertemplate=f'{comp} · Lap %{{x}}: +%{{y:.2f}}s<extra></extra>'
                            ))
                        fig2.update_layout(
                            **PLOT_BASE, height=300,
                            xaxis=ax('Tire Age (laps)'), yaxis=ax('Delta from fastest (s)')
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.markdown('<div style="color:#2a2a2a;font-size:12px;padding:1rem 0">Insufficient data for tire model on this circuit.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#2a2a2a;font-size:12px;padding:1rem 0">Tire model features not available for this race.</div>', unsafe_allow_html=True)
    else:
        st.warning("No data available for this combination.")

# ─────────────────────────────────────────
# PAGE 2 — DRIVER DNA
# ─────────────────────────────────────────
elif "Driver DNA" in page:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">DRIVER DNA LAB</div>
        <div class="page-subtitle">Telemetry fingerprinting · K-Means clustering · PCA · 76,721 telemetry points</div>
    </div>
    <div class="red-rule"></div>
    """, unsafe_allow_html=True)

    drivers_avail = sorted(driver_dna['driver'].unique())
    driver_disp = {f"{DRIVER_NAMES.get(d, d)} ({d})": d for d in drivers_avail}

    c1, c2 = st.columns(2)
    with c1:
        d1k = st.selectbox("DRIVER 1", list(driver_disp.keys()),
                           index=next((i for i, k in enumerate(driver_disp) if driver_disp[k]=='VER'), 0))
        driver1 = driver_disp[d1k]
    with c2:
        d2k = st.selectbox("DRIVER 2", list(driver_disp.keys()),
                           index=next((i for i, k in enumerate(driver_disp) if driver_disp[k]=='HAM'), 1))
        driver2 = driver_disp[d2k]

    st.markdown('<div class="red-rule"></div>', unsafe_allow_html=True)

    raw_feats = ['full_throttle_pct','heavy_brake_pct','avg_corner_speed',
                 'throttle_smoothness','coast_pct','high_speed_pct']
    cats = ['Full Throttle','Heavy Braking','Corner Speed','Smoothness','Coasting','High Speed']

    dna_n = driver_dna.copy()
    for f in raw_feats:
        mn, mx = driver_dna[f].min(), driver_dna[f].max()
        dna_n[f+'_n'] = 30 + ((driver_dna[f]-mn)/(mx-mn))*70

    d1r = dna_n[dna_n['driver']==driver1].iloc[0]
    d2r = dna_n[dna_n['driver']==driver2].iloc[0]
    nc = [f+'_n' for f in raw_feats]
    v1 = [round(d1r[c],1) for c in nc]+[round(d1r[nc[0]],1)]
    v2 = [round(d2r[c],1) for c in nc]+[round(d2r[nc[0]],1)]
    cats_c = cats+[cats[0]]

    t1 = d1r['team']; t2 = d2r['team']
    c1c = TEAM_COLORS.get(t1,'#E24B4A')
    c2c = TEAM_COLORS.get(t2,'#378ADD')

    col_r, col_s = st.columns([1.4, 0.6])

    with col_r:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=v1, theta=cats_c, fill='toself',
            fillcolor=hex_to_rgba(c1c, 0.18),
            line=dict(color=c1c, width=2.5),
            name=DRIVER_NAMES.get(driver1, driver1),
            marker=dict(size=6, color=c1c)
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=v2, theta=cats_c, fill='toself',
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
                    tickfont=dict(color='#1a1a1a',size=8),
                    gridcolor='#101010', linecolor='#101010'),
                angularaxis=dict(
                    tickfont=dict(color='#888',size=11,family='monospace'),
                    gridcolor='#111', linecolor='#151515')
            ),
            paper_bgcolor='#080808',
            title=dict(text='DRIVING STYLE FINGERPRINT',
                       font=dict(size=11,color='#2a2a2a',family='monospace'), x=0.5, xanchor='center'),
            legend=dict(bgcolor='rgba(8,8,8,0.95)', bordercolor='#141414', borderwidth=1,
                        font=dict(color='#888',size=11,family='monospace'),
                        x=0.5, y=-0.07, xanchor='center', orientation='h'),
            height=500,
            margin=dict(t=50,b=80,l=50,r=50),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_s:
        for drv, c_hex in [(driver1, c1c), (driver2, c2c)]:
            d_raw = driver_dna[driver_dna['driver']==drv].iloc[0]
            full_name = DRIVER_NAMES.get(drv, drv)
            archetype = d_raw.get('archetype','Unknown')
            if not isinstance(archetype, str): archetype = 'Unknown'
            st.markdown(f"""
            <div class="driver-card" style="border-top: 2px solid {c_hex}; margin-bottom: 12px">
                <div class="driver-card-name" style="color:{c_hex}">{full_name.upper()}</div>
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
    st.markdown('<div class="section-eyebrow">QUALIFYING SPEED TRACE · HEAD TO HEAD</div>', unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns([1,1,1])
    with tc1:
        tel_year = st.selectbox("YEAR", [2026,2024,2023,2022], index=1)
    with tc2:
        tel_race = st.selectbox("CIRCUIT", AVAILABLE_RACES[tel_year])
    with tc3:
        st.markdown("<div style='padding-top:1.6rem'>", unsafe_allow_html=True)
        load_tel = st.button("LOAD TELEMETRY ▶")
        st.markdown("</div>", unsafe_allow_html=True)

    if load_tel:
        with st.spinner(f"Loading {tel_race} {tel_year} Q telemetry..."):
            try:
                from scipy.interpolate import interp1d
                sess = fastf1.get_session(tel_year, tel_race, 'Q')
                sess.load(telemetry=True, weather=False, messages=False)

                lap1 = sess.laps.pick_driver(driver1).pick_fastest()
                lap2 = sess.laps.pick_driver(driver2).pick_fastest()

                if lap1.empty or lap2.empty:
                    st.error(f"No lap found for one driver in {tel_race} {tel_year}. Try a different circuit or year.")
                else:
                    tel1 = lap1.get_telemetry().add_distance()
                    tel2 = lap2.get_telemetry().add_distance()
                    dmin = max(tel1['Distance'].min(), tel2['Distance'].min())
                    dmax = min(tel1['Distance'].max(), tel2['Distance'].max())
                    cd = np.linspace(dmin, dmax, 1500)
                    s1 = interp1d(tel1['Distance'], tel1['Speed'], fill_value='extrapolate')(cd)
                    s2 = interp1d(tel2['Distance'], tel2['Speed'], fill_value='extrapolate')(cd)
                    delta = s1 - s2

                    t1s = lap1['LapTime'].total_seconds()
                    t2s = lap2['LapTime'].total_seconds()
                    n1 = DRIVER_NAMES.get(driver1, driver1)
                    n2 = DRIVER_NAMES.get(driver2, driver2)

                    fig_t = make_subplots(rows=2, cols=1, row_heights=[0.68,0.32],
                                          vertical_spacing=0.05, shared_xaxes=True)
                    fig_t.add_trace(go.Scatter(
                        x=cd, y=s1, mode='lines',
                        name=f'{n1}  ·  {int(t1s//60)}:{t1s%60:06.3f}',
                        line=dict(color=c1c, width=2.5),
                        hovertemplate=f'{driver1}: %{{y:.0f}} km/h<extra></extra>'
                    ), row=1, col=1)
                    fig_t.add_trace(go.Scatter(
                        x=cd, y=s2, mode='lines',
                        name=f'{n2}  ·  {int(t2s//60)}:{t2s%60:06.3f}',
                        line=dict(color=c2c, width=2.5),
                        hovertemplate=f'{driver2}: %{{y:.0f}} km/h<extra></extra>'
                    ), row=1, col=1)
                    fig_t.add_trace(go.Scatter(
                        x=cd, y=delta, mode='lines',
                        fill='tozeroy', fillcolor=hex_to_rgba(c1c, 0.1),
                        line=dict(color=c1c, width=1.5),
                        name=f'Δ ({driver1} − {driver2})',
                        hovertemplate='Δ: %{y:.1f} km/h<extra></extra>'
                    ), row=2, col=1)
                    fig_t.add_hline(y=0, line=dict(color='#1a1a1a', width=1), row=2, col=1)

                    for row in [1,2]:
                        fig_t.update_xaxes(gridcolor='#0f0f0f', zerolinecolor='#111',
                                           tickfont=dict(color='#2a2a2a',size=9), row=row, col=1)
                        fig_t.update_yaxes(gridcolor='#0f0f0f', zerolinecolor='#111',
                                           tickfont=dict(color='#2a2a2a',size=9), row=row, col=1)

                    fig_t.update_yaxes(title_text='Speed (km/h)',
                                       title_font=dict(color='#333',size=10,family='monospace'), row=1, col=1)
                    fig_t.update_yaxes(title_text='Δ Speed',
                                       title_font=dict(color='#333',size=10,family='monospace'), row=2, col=1)
                    fig_t.update_xaxes(title_text='Distance (m)',
                                       title_font=dict(color='#333',size=10,family='monospace'), row=2, col=1)
                    fig_t.update_layout(
                        **PLOT_BASE,
                        title=dict(
                            text=f'{tel_race.upper()} {tel_year}  ·  QUALIFYING  ·  {driver1} vs {driver2}',
                            font=dict(size=12,color='#333',family='monospace'), x=0.5, xanchor='center'
                        ),
                        height=600, hovermode='x unified'
                    )
                    st.plotly_chart(fig_t, use_container_width=True)

            except Exception as e:
                st.error(f"Telemetry error: {str(e)}")

# ─────────────────────────────────────────
# PAGE 3 — STRATEGY SIMULATOR
# ─────────────────────────────────────────
elif "Strategy Simulator" in page:
    st.markdown("""
    <div class="page-header">
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
        sim_pos = st.slider("CURRENT POSITION", 1, 20, 5)
    with sc2:
        st.markdown('<div class="section-eyebrow">RACE STATE</div>', unsafe_allow_html=True)
        sim_prog = st.slider("RACE PROGRESS (%)", 0, 100, 45)
        sim_laps = st.slider("LAPS REMAINING", 1, 60, 25)
        sim_team = st.selectbox("TEAM", list(TEAM_COLORS.keys()), index=0)

    st.markdown('<div class="red-rule"></div>', unsafe_allow_html=True)

    cmap = {'SOFT':0,'MEDIUM':1,'HARD':2}
    feat = pd.DataFrame([{
        'tire_age_at_pit': sim_tire,
        'compound_num': cmap[sim_comp],
        'race_progress': sim_prog/100,
        'position_before': sim_pos/20,
        'laps_remaining': sim_laps,
    }])
    prob = strategy_model.predict_proba(feat)[0][1]
    decision = "PIT NOW" if prob >= 0.5 else "STAY OUT"
    dc = "#E24B4A" if prob >= 0.5 else "#1D9E75"
    tc_team = TEAM_COLORS.get(sim_team,'#E24B4A')

    rc1, rc2, rc3 = st.columns([1,1,1])

    with rc1:
        st.markdown(f"""
        <div class="decision-box" style="border-top: 3px solid {dc}">
            <div class="decision-label">ML RECOMMENDATION</div>
            <div class="decision-text" style="color:{dc}">{decision}</div>
            <div style="margin: 14px 0 10px; height: 3px; border-radius: 2px;
                        background: linear-gradient(90deg, {dc} {prob*100:.0f}%, #111 {prob*100:.0f}%)"></div>
            <div class="decision-conf">Confidence: {prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with rc2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title=dict(text="PIT PROBABILITY", font=dict(color='#2a2a2a',size=10,family='monospace')),
            number=dict(suffix="%", font=dict(color='#fff',size=26,family='monospace')),
            gauge=dict(
                axis=dict(range=[0,100], tickfont=dict(color='#1a1a1a',size=8), tickcolor='#111'),
                bar=dict(color=dc, thickness=0.55),
                bgcolor='#0a0a0a', bordercolor='#111', borderwidth=1,
                steps=[
                    dict(range=[0,35], color='#0a120a'),
                    dict(range=[35,65], color='#12120a'),
                    dict(range=[65,100], color='#120a0a'),
                ],
                threshold=dict(line=dict(color='#333',width=1.5), thickness=0.7, value=50)
            )
        ))
        fig_g.update_layout(
            paper_bgcolor='#080808',
            font=dict(color='#fff',family='monospace'),
            height=240, margin=dict(t=40,b=10,l=25,r=25)
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with rc3:
        if prob >= 0.7: msg, mc = "Strong pit signal. Tire cliff approaching. Box this lap.", "#E24B4A"
        elif prob >= 0.5: msg, mc = "Marginal window. Weigh track position vs fresh rubber.", "#EF9F27"
        elif prob >= 0.3: msg, mc = "Tires still viable. Hold position if it matters.", "#1D9E75"
        else: msg, mc = "No pit required. Performance still strong.", "#378ADD"

        st.markdown(f"""
        <div class="analysis-box" style="border-top: 2px solid {mc}">
            <div class="section-eyebrow">STRATEGY ANALYSIS</div>
            <div style="font-size:12px; color:#555; line-height:1.8; margin-bottom:1.25rem">{msg}</div>
            <div class="stat-row"><span class="stat-label">Tire age</span><span class="stat-value">{sim_tire} laps</span></div>
            <div class="stat-row"><span class="stat-label">Compound</span><span class="stat-value">{sim_comp}</span></div>
            <div class="stat-row"><span class="stat-label">Position</span><span class="stat-value">P{sim_pos}</span></div>
            <div class="stat-row"><span class="stat-label">Race progress</span><span class="stat-value">{sim_prog}%</span></div>
            <div class="stat-row" style="border-bottom:none">
                <span class="stat-label">Laps remaining</span>
                <span style="color:{tc_team};font-size:0.72rem">{sim_laps}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="dim-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">PIT WINDOW SENSITIVITY · TIRE AGE vs PROBABILITY</div>', unsafe_allow_html=True)

    tire_ages = np.arange(1, 51)
    probs = [strategy_model.predict_proba(pd.DataFrame([{
        'tire_age_at_pit': ta, 'compound_num': cmap[sim_comp],
        'race_progress': sim_prog/100, 'position_before': sim_pos/20,
        'laps_remaining': sim_laps
    }]))[0][1]*100 for ta in tire_ages]

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=tire_ages, y=probs, mode='lines',
        fill='tozeroy', fillcolor='rgba(226,75,74,0.06)',
        line=dict(color='#E24B4A', width=2),
        hovertemplate='Age %{x}: %{y:.1f}%<extra></extra>'
    ))
    fig_s.add_hline(y=50, line=dict(color='#1a1a1a',width=1,dash='dash'),
                    annotation_text="THRESHOLD", annotation_font=dict(color='#2a2a2a',size=9,family='monospace'))
    fig_s.add_vline(x=sim_tire, line=dict(color='#EF9F27',width=1.5,dash='dot'),
                    annotation_text="NOW", annotation_font=dict(color='#EF9F27',size=9,family='monospace'))
    fig_s.update_layout(**PLOT_BASE, height=280,
                        xaxis=ax('Tire Age (laps)'),
                        yaxis=dict(**ax('Pit Probability (%)'), range=[0,100]),
                        margin=dict(t=30,b=45,l=55,r=20))
    st.plotly_chart(fig_s, use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer-wrap">
    <div class="footer-main">F1 STRATEGY INTELLIGENCE SYSTEM</div>
    <div class="footer-sub">FastF1 · scikit-learn · Streamlit · Plotly · 2022–2026</div>
    <div style="margin: 8px 0">
        <span class="footer-heart">♥</span>
        <span class="footer-by"> &nbsp;Made with love by Arin</span>
    </div>
</div>
""", unsafe_allow_html=True)