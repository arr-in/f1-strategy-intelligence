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

# ─────────────────────────────────────────
# DRIVER NAMES
# ─────────────────────────────────────────
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
    'LIN': 'Jack Doohan', 'DOO': 'Jack Doohan',
    'SAR': 'Logan Sargeant', 'HUL': 'Nico Hulkenberg',
}

# ─────────────────────────────────────────
# TEAM COLORS
# ─────────────────────────────────────────
TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'AlphaTauri': '#6692FF',
    'RB': '#6692FF',
    'Racing Bulls': '#6692FF',
    'Alfa Romeo': '#C92D4B',
    'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD',
    'Cadillac': '#CC0000',
    'Audi': '#F50000',
}

# ─────────────────────────────────────────
# UPCOMING RACES 2026
# ─────────────────────────────────────────
UPCOMING_RACES = [
    {'name': 'Miami Grand Prix', 'circuit': 'Miami', 'date': datetime(2026, 5, 3, 14, 0, tzinfo=timezone.utc)},
    {'name': 'Emilia Romagna Grand Prix', 'circuit': 'Imola', 'date': datetime(2026, 5, 17, 13, 0, tzinfo=timezone.utc)},
    {'name': 'Monaco Grand Prix', 'circuit': 'Monaco', 'date': datetime(2026, 5, 24, 13, 0, tzinfo=timezone.utc)},
    {'name': 'Spanish Grand Prix', 'circuit': 'Barcelona', 'date': datetime(2026, 6, 1, 13, 0, tzinfo=timezone.utc)},
    {'name': 'Canadian Grand Prix', 'circuit': 'Montreal', 'date': datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)},
    {'name': 'Austrian Grand Prix', 'circuit': 'Spielberg', 'date': datetime(2026, 6, 29, 13, 0, tzinfo=timezone.utc)},
    {'name': 'British Grand Prix', 'circuit': 'Silverstone', 'date': datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc)},
]

def get_next_race():
    now = datetime.now(timezone.utc)
    future = [r for r in UPCOMING_RACES if r['date'] > now]
    return future[0] if future else None

def format_countdown(race):
    now = datetime.now(timezone.utc)
    delta = race['date'] - now
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes = rem // 60
    return days, hours, minutes

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bebas+Neue&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    background-color: #080808 !important;
    color: #dddddd !important;
}

.stApp {
    background-color: #080808 !important;
    background-image:
        repeating-linear-gradient(
            45deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.012) 2px,
            rgba(255,255,255,0.012) 4px
        ),
        repeating-linear-gradient(
            -45deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.008) 2px,
            rgba(255,255,255,0.008) 4px
        );
}

.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #1a1a1a !important;
    background-image:
        repeating-linear-gradient(
            45deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.015) 2px,
            rgba(255,255,255,0.015) 4px
        );
}
[data-testid="stSidebar"] * { color: #cccccc !important; }

h1 {
    font-family: 'Bebas Neue', monospace !important;
    font-size: 3rem !important;
    letter-spacing: 0.08em !important;
    color: #ffffff !important;
    line-height: 1 !important;
}
h2 {
    font-family: 'Bebas Neue', monospace !important;
    font-size: 1.8rem !important;
    letter-spacing: 0.06em !important;
    color: #ffffff !important;
}
h3 {
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #444444 !important;
    font-weight: 500 !important;
}

[data-testid="metric-container"] {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: #444 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
}

.stSelectbox > div > div {
    background: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 6px !important;
    color: #ddd !important;
}
.stSlider > div > div > div { background: #E24B4A !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid #E24B4A !important;
    color: #E24B4A !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.08em !important;
    font-size: 0.8rem !important;
    transition: all 0.2s !important;
    padding: 0.5rem 1.2rem !important;
}
.stButton > button:hover {
    background: #E24B4A !important;
    color: #ffffff !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a1a1a !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #444 !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.1em !important;
    font-size: 0.75rem !important;
    padding: 0.75rem 1.5rem !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #ffffff !important;
    border-bottom: 2px solid #E24B4A !important;
}

.stRadio > div { gap: 0.5rem !important; }
.stRadio label {
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    color: #888 !important;
}

hr { border-color: #1a1a1a !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.countdown-box {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-top: 2px solid #E24B4A;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
    margin: 0.5rem 0;
}
.countdown-num {
    font-family: 'Bebas Neue', monospace;
    font-size: 2rem;
    color: #E24B4A;
    line-height: 1;
}
.countdown-label {
    font-size: 0.6rem;
    color: #444;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2px;
}
.race-name {
    font-family: 'Bebas Neue', monospace;
    font-size: 1rem;
    color: #ffffff;
    letter-spacing: 0.1em;
    text-align: center;
    margin-bottom: 0.5rem;
}
.red-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #E24B4A, transparent);
    margin: 1rem 0;
    border: none;
}
.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #333;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1a1a;
}
.footer-text {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #2a2a2a;
    text-align: center;
    padding: 1rem 0;
    letter-spacing: 0.1em;
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
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    # F1 Logo + Title
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem 0'>
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:4px'>
            <div style='
                background: #E24B4A;
                color: white;
                font-family: Bebas Neue, monospace;
                font-size: 1.4rem;
                padding: 4px 10px;
                letter-spacing: 0.05em;
                border-radius: 3px;
                line-height: 1;
            '>F1</div>
            <div>
                <div style='font-family: Bebas Neue, monospace; font-size: 1.3rem;
                            color: #fff; letter-spacing: 0.1em; line-height:1'>STRATEGY</div>
                <div style='font-size: 0.6rem; color: #444; letter-spacing: 0.2em;
                            text-transform: uppercase'>INTELLIGENCE SYSTEM</div>
            </div>
        </div>
        <div style='height:1px; background:linear-gradient(90deg,#E24B4A,transparent);
                    margin: 0.75rem 0'></div>
    </div>
    """, unsafe_allow_html=True)

    # Upcoming race countdown
    next_race = get_next_race()
    if next_race:
        days, hours, minutes = format_countdown(next_race)
        st.markdown(f"""
        <div style='margin-bottom: 0.5rem'>
            <div class='section-label'>NEXT RACE</div>
            <div class='race-name'>{next_race['name'].upper()}</div>
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px'>
                <div class='countdown-box'>
                    <div class='countdown-num'>{days:02d}</div>
                    <div class='countdown-label'>DAYS</div>
                </div>
                <div class='countdown-box'>
                    <div class='countdown-num'>{hours:02d}</div>
                    <div class='countdown-label'>HRS</div>
                </div>
                <div class='countdown-box'>
                    <div class='countdown-num'>{minutes:02d}</div>
                    <div class='countdown-label'>MIN</div>
                </div>
            </div>
        </div>
        <div style='height:1px; background:linear-gradient(90deg,#E24B4A,transparent);
                    margin: 0.75rem 0'></div>
        """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="section-label">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🏁  Race Strategy", "🧬  Driver DNA", "🔮  Strategy Simulator"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='height:1px; background:linear-gradient(90deg,#E24B4A,transparent);
                margin: 0.75rem 0'></div>
    """, unsafe_allow_html=True)

    # Data coverage
    st.markdown('<div class="section-label">DATA COVERAGE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; color:#3a3a3a; line-height:2.2; font-family:monospace'>
    <span style='color:#E24B4A'>▸</span> Seasons: 2022 · 2023 · 2024 · 2026<br>
    <span style='color:#E24B4A'>▸</span> Circuits: 27 Grand Prix<br>
    <span style='color:#E24B4A'>▸</span> Drivers: 34 profiles<br>
    <span style='color:#E24B4A'>▸</span> Laps analyzed: 32,572<br>
    <span style='color:#E24B4A'>▸</span> ML models: 3 trained
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style='position:absolute; bottom:1rem; left:0; right:0; padding: 0 1rem'>
        <div style='height:1px; background:linear-gradient(90deg,transparent,#1a1a1a,transparent);
                    margin-bottom:0.75rem'></div>
        <div style='font-size:10px; color:#2a2a2a; text-align:center;
                    font-family:monospace; letter-spacing:0.1em; line-height:1.8'>
            Made with ♥ by Arin<br>
            <span style='color:#1a1a1a'>FastF1 · scikit-learn · Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────
PLOT_LAYOUT = dict(
    plot_bgcolor='#0a0a0a',
    paper_bgcolor='#0a0a0a',
    font=dict(color='#ffffff', family='monospace'),
    margin=dict(t=60, b=50, l=60, r=20),
    hoverlabel=dict(
        bgcolor='#111111',
        bordercolor='#333333',
        font=dict(color='#ffffff', size=12, family='monospace')
    ),
    legend=dict(
        bgcolor='rgba(10,10,10,0.95)',
        bordercolor='#1e1e1e',
        borderwidth=1,
        font=dict(color='#aaaaaa', size=11, family='monospace')
    )
)

def styled_axis(title=''):
    return dict(
        title=title,
        gridcolor='#111111',
        zerolinecolor='#1e1e1e',
        tickfont=dict(color='#333', size=10),
        title_font=dict(color='#444', size=11, family='monospace')
    )

def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

# ─────────────────────────────────────────
# PAGE 1 — RACE STRATEGY
# ─────────────────────────────────────────
if "Race Strategy" in page:
    st.markdown("# RACE STRATEGY")
    st.markdown("### Command Center · ML-powered pit window optimization")
    st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_race = st.selectbox(
            "GRAND PRIX",
            sorted(clean_laps['race_name'].unique()),
        )
    with col2:
        race_drivers = sorted(clean_laps[clean_laps['race_name'] == selected_race]['Driver'].unique())
        driver_options = {f"{DRIVER_NAMES.get(d, d)} ({d})": d for d in race_drivers}
        selected_display = st.selectbox("DRIVER", list(driver_options.keys()))
        selected_driver = driver_options[selected_display]
    with col3:
        selected_compound = st.selectbox("COMPOUND", ['SOFT', 'MEDIUM', 'HARD'], index=1)

    st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)

    driver_laps = clean_laps[
        (clean_laps['race_name'] == selected_race) &
        (clean_laps['Driver'] == selected_driver)
    ].copy()

    if not driver_laps.empty:
        team = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else 'Unknown'
        team_color = TEAM_COLORS.get(team, '#E24B4A')
        best_lap = driver_laps['LapTimeSec'].min()
        mins = int(best_lap // 60)
        secs = best_lap % 60
        avg_deg = driver_laps['DeltaFromFastest'].mean()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("TOTAL LAPS", int(driver_laps['LapNumber'].max()))
        with m2:
            st.metric("FASTEST LAP", f"{mins}:{secs:06.3f}")
        with m3:
            st.metric("AVG DEGRADATION", f"+{avg_deg:.2f}s")
        with m4:
            st.metric("TEAM", team)

        st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)

        # Lap time chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=driver_laps['LapNumber'],
            y=driver_laps['LapTimeSec'],
            mode='lines+markers',
            line=dict(color=team_color, width=2),
            marker=dict(size=3, color=team_color),
            fill='tozeroy',
            fillcolor=hex_to_rgba(team_color, 0.05),
            name=f'{DRIVER_NAMES.get(selected_driver, selected_driver)}',
            hovertemplate='Lap %{x}: %{y:.3f}s<extra></extra>'
        ))
        layout = {**PLOT_LAYOUT}
        layout['title'] = dict(
            text=f'{DRIVER_NAMES.get(selected_driver, selected_driver).upper()}  ·  {selected_race}  ·  LAP TIME PROGRESSION',
            font=dict(size=14, color='#ffffff', family='monospace'),
        )
        layout['height'] = 380
        layout['xaxis'] = styled_axis('Lap Number')
        layout['yaxis'] = styled_axis('Lap Time (s)')
        layout['hovermode'] = 'x unified'
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        # Tire degradation
        st.markdown("### TIRE DEGRADATION MODEL")
        tire_model_path = f"models/tire_deg_{selected_race.replace(' ', '_')}.pkl"
        if os.path.exists(tire_model_path):
            tire_model = joblib.load(tire_model_path)
            tire_ages = np.arange(1, 46)
            compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
            colors = {'SOFT': '#E24B4A', 'MEDIUM': '#EF9F27', 'HARD': '#888780'}

            fig2 = go.Figure()
            for compound, c_num in compound_map.items():
                X_pred = pd.DataFrame({
                    'TyreLife': tire_ages,
                    'TireLifeSq': tire_ages ** 2,
                    'CompoundNum': c_num,
                    'LapNum': 30
                })
                deltas = tire_model.predict(X_pred)
                fig2.add_trace(go.Scatter(
                    x=tire_ages, y=deltas,
                    mode='lines',
                    name=compound,
                    line=dict(color=colors[compound], width=2.5),
                    hovertemplate=f'{compound} · Lap %{{x}}: +%{{y:.2f}}s<extra></extra>'
                ))

            layout2 = {**PLOT_LAYOUT}
            layout2['height'] = 320
            layout2['xaxis'] = styled_axis('Tire Age (laps)')
            layout2['yaxis'] = styled_axis('Delta from fastest (s)')
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(f"""
            <div style='background:#0d0d0d; border:1px solid #1a1a1a; border-left:3px solid #333;
                        border-radius:4px; padding:1rem; font-size:12px; color:#444'>
                No tire model for {selected_race} — available circuits: Bahrain, Britain, Saudi Arabia
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No data for this combination.")

# ─────────────────────────────────────────
# PAGE 2 — DRIVER DNA
# ─────────────────────────────────────────
elif "Driver DNA" in page:
    st.markdown("# DRIVER DNA LAB")
    st.markdown("### Telemetry fingerprinting · K-Means clustering · PCA analysis")
    st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)

    drivers_available = sorted(driver_dna['driver'].unique())
    driver_display = {f"{DRIVER_NAMES.get(d, d)} ({d})": d for d in drivers_available}

    col1, col2 = st.columns(2)
    with col1:
        d1_display = st.selectbox(
            "DRIVER 1",
            list(driver_display.keys()),
            index=list(driver_display.keys()).index(
                next(k for k in driver_display if driver_display[k] == 'VER'), 0
            ) if 'VER' in drivers_available else 0
        )
        driver1 = driver_display[d1_display]
    with col2:
        d2_display = st.selectbox(
            "DRIVER 2",
            list(driver_display.keys()),
            index=list(driver_display.keys()).index(
                next(k for k in driver_display if driver_display[k] == 'HAM'), 0
            ) if 'HAM' in drivers_available else 1
        )
        driver2 = driver_display[d2_display]

    st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)

    raw_features = ['full_throttle_pct', 'heavy_brake_pct', 'avg_corner_speed',
                    'throttle_smoothness', 'coast_pct', 'high_speed_pct']
    categories = ['Full Throttle', 'Heavy Braking', 'Corner Speed',
                  'Smoothness', 'Coasting', 'High Speed']

    def normalize_30_100(series):
        return 30 + ((series - series.min()) / (series.max() - series.min())) * 70

    dna_norm = driver_dna.copy()
    for f in raw_features:
        dna_norm[f + '_norm'] = normalize_30_100(driver_dna[f])

    d1 = dna_norm[dna_norm['driver'] == driver1].iloc[0]
    d2 = dna_norm[dna_norm['driver'] == driver2].iloc[0]
    norm_cols = [f + '_norm' for f in raw_features]
    v1 = [round(d1[c], 1) for c in norm_cols] + [round(d1[norm_cols[0]], 1)]
    v2 = [round(d2[c], 1) for c in norm_cols] + [round(d2[norm_cols[0]], 1)]
    cats = categories + [categories[0]]

    t1 = d1['team']
    t2 = d2['team']
    c1 = TEAM_COLORS.get(t1, '#E24B4A')
    c2 = TEAM_COLORS.get(t2, '#378ADD')

    col_radar, col_stats = st.columns([1.3, 0.7])

    with col_radar:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=v1, theta=cats,
            fill='toself',
            fillcolor=hex_to_rgba(c1, 0.2),
            line=dict(color=c1, width=3),
            name=f'{DRIVER_NAMES.get(driver1, driver1)}',
            marker=dict(size=7, color=c1)
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=v2, theta=cats,
            fill='toself',
            fillcolor=hex_to_rgba(c2, 0.2),
            line=dict(color=c2, width=3),
            name=f'{DRIVER_NAMES.get(driver2, driver2)}',
            marker=dict(size=7, color=c2)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#0a0a0a',
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickvals=[25, 50, 75, 100],
                    tickfont=dict(color='#222', size=8),
                    gridcolor='#151515',
                    linecolor='#151515',
                ),
                angularaxis=dict(
                    tickfont=dict(color='#aaaaaa', size=11, family='monospace'),
                    gridcolor='#151515',
                    linecolor='#222',
                )
            ),
            paper_bgcolor='#080808',
            title=dict(
                text=f'DRIVING STYLE COMPARISON',
                font=dict(size=13, color='#444', family='monospace'),
                x=0.5, xanchor='center'
            ),
            legend=dict(
                bgcolor='rgba(10,10,10,0.95)',
                bordercolor='#1e1e1e', borderwidth=1,
                font=dict(color='#aaa', size=11, family='monospace'),
                x=0.5, y=-0.08,
                xanchor='center', orientation='h'
            ),
            height=480,
            margin=dict(t=60, b=80, l=60, r=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_stats:
        for drv, d_raw, color in [(driver1, driver_dna[driver_dna['driver'] == driver1].iloc[0], c1),
                                   (driver2, driver_dna[driver_dna['driver'] == driver2].iloc[0], c2)]:
            full_name = DRIVER_NAMES.get(drv, drv)
            archetype = d_raw.get('archetype', 'Unknown') if isinstance(d_raw.get('archetype'), str) else 'Unknown'
            st.markdown(f"""
            <div style='background:#0d0d0d; border:1px solid #1a1a1a;
                        border-left:3px solid {color}; border-radius:4px;
                        padding:1rem; margin-bottom:1rem'>
                <div style='font-family:Bebas Neue,monospace; font-size:1.3rem;
                            color:{color}; letter-spacing:0.08em'>{full_name.upper()}</div>
                <div style='font-size:10px; color:#333; letter-spacing:0.15em;
                            margin-bottom:0.75rem'>{d_raw["team"]}</div>
                <div style='font-size:11px; color:#555; line-height:2.2; font-family:monospace'>
                    <span style='color:#2a2a2a'>ARCHETYPE</span><br>
                    <span style='color:#aaa'>{archetype}</span><br>
                    <span style='color:#2a2a2a'>FULL THROTTLE</span><br>
                    <span style='color:#aaa'>{d_raw["full_throttle_pct"]:.1f}%</span><br>
                    <span style='color:#2a2a2a'>HEAVY BRAKING</span><br>
                    <span style='color:#aaa'>{d_raw["heavy_brake_pct"]:.1f}%</span><br>
                    <span style='color:#2a2a2a'>CORNER SPEED</span><br>
                    <span style='color:#aaa'>{d_raw["avg_corner_speed"]:.0f} km/h</span><br>
                    <span style='color:#2a2a2a'>MAX SPEED</span><br>
                    <span style='color:#aaa'>{d_raw["max_speed"]:.0f} km/h</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Telemetry speed trace
    st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)
    st.markdown("### QUALIFYING SPEED TRACE")

    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    with col_t1:
        tel_year = st.selectbox("YEAR", [2026, 2024, 2023, 2022], index=1)
    with col_t2:
        tel_race = st.selectbox(
            "CIRCUIT",
            ['Bahrain', 'Monaco', 'Britain', 'Italy', 'Australia', 'Japan'],
            index=0
        )
    with col_t3:
        st.markdown("<div style='padding-top:1.8rem'>", unsafe_allow_html=True)
        load_tel = st.button("LOAD TELEMETRY ▶")
        st.markdown("</div>", unsafe_allow_html=True)

    if load_tel:
        with st.spinner(f"Loading {tel_race} {tel_year} telemetry..."):
            try:
                from scipy.interpolate import interp1d
                session = fastf1.get_session(tel_year, tel_race, 'Q')
                session.load(telemetry=True, weather=False, messages=False)

                lap1 = session.laps.pick_driver(driver1).pick_fastest()
                lap2 = session.laps.pick_driver(driver2).pick_fastest()

                if lap1.empty or lap2.empty:
                    st.error(f"No qualifying lap found for one of the drivers in {tel_race} {tel_year}")
                else:
                    tel1 = lap1.get_telemetry().add_distance()
                    tel2 = lap2.get_telemetry().add_distance()

                    dist_min = max(tel1['Distance'].min(), tel2['Distance'].min())
                    dist_max = min(tel1['Distance'].max(), tel2['Distance'].max())
                    common_dist = np.linspace(dist_min, dist_max, 1500)

                    s1 = interp1d(tel1['Distance'], tel1['Speed'],
                                  kind='linear', fill_value='extrapolate')(common_dist)
                    s2 = interp1d(tel2['Distance'], tel2['Speed'],
                                  kind='linear', fill_value='extrapolate')(common_dist)
                    delta = s1 - s2

                    t1_sec = lap1['LapTime'].total_seconds()
                    t2_sec = lap2['LapTime'].total_seconds()
                    m1t, s1t = int(t1_sec // 60), t1_sec % 60
                    m2t, s2t = int(t2_sec // 60), t2_sec % 60

                    n1 = DRIVER_NAMES.get(driver1, driver1)
                    n2 = DRIVER_NAMES.get(driver2, driver2)

                    fig_tel = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.68, 0.32],
                        vertical_spacing=0.06,
                        shared_xaxes=True
                    )

                    fig_tel.add_trace(go.Scatter(
                        x=common_dist, y=s1,
                        mode='lines',
                        name=f'{n1}  ·  {m1t}:{s1t:06.3f}',
                        line=dict(color=c1, width=2.5),
                        hovertemplate=f'{driver1}: %{{y:.0f}} km/h<extra></extra>'
                    ), row=1, col=1)

                    fig_tel.add_trace(go.Scatter(
                        x=common_dist, y=s2,
                        mode='lines',
                        name=f'{n2}  ·  {m2t}:{s2t:06.3f}',
                        line=dict(color=c2, width=2.5),
                        hovertemplate=f'{driver2}: %{{y:.0f}} km/h<extra></extra>'
                    ), row=1, col=1)

                    fig_tel.add_trace(go.Scatter(
                        x=common_dist, y=delta,
                        mode='lines',
                        fill='tozeroy',
                        fillcolor=hex_to_rgba(c1, 0.12),
                        line=dict(color=c1, width=1.5),
                        name=f'Δ Speed ({driver1} − {driver2})',
                        hovertemplate='Δ: %{y:.1f} km/h<extra></extra>'
                    ), row=2, col=1)

                    fig_tel.add_hline(
                        y=0,
                        line=dict(color='#2a2a2a', width=1),
                        row=2, col=1
                    )

                    for row in [1, 2]:
                        fig_tel.update_xaxes(
                            gridcolor='#111', zerolinecolor='#1e1e1e',
                            tickfont=dict(color='#333', size=10),
                            row=row, col=1
                        )
                        fig_tel.update_yaxes(
                            gridcolor='#111', zerolinecolor='#1e1e1e',
                            tickfont=dict(color='#444', size=10),
                            row=row, col=1
                        )

                    fig_tel.update_yaxes(
                        title_text='Speed (km/h)',
                        title_font=dict(color='#444', size=11, family='monospace'),
                        row=1, col=1
                    )
                    fig_tel.update_yaxes(
                        title_text='Δ Speed (km/h)',
                        title_font=dict(color='#444', size=11, family='monospace'),
                        row=2, col=1
                    )
                    fig_tel.update_xaxes(
                        title_text='Distance (m)',
                        title_font=dict(color='#444', size=11, family='monospace'),
                        row=2, col=1
                    )

                    fig_tel.update_layout(
                        title=dict(
                            text=f'{tel_race.upper()} {tel_year}  ·  QUALIFYING  ·  {driver1} vs {driver2}',
                            font=dict(size=14, color='#ffffff', family='monospace'),
                            x=0.5, xanchor='center'
                        ),
                        plot_bgcolor='#0a0a0a',
                        paper_bgcolor='#0a0a0a',
                        font=dict(color='#ffffff', family='monospace'),
                        legend=dict(
                            bgcolor='rgba(10,10,10,0.95)',
                            bordercolor='#1e1e1e', borderwidth=1,
                            font=dict(color='#aaa', size=11, family='monospace'),
                            x=0.01, y=0.99
                        ),
                        height=620,
                        margin=dict(t=60, b=50, l=70, r=20),
                        hovermode='x unified',
                        hoverlabel=dict(
                            bgcolor='#111', bordercolor='#333',
                            font=dict(color='#fff', family='monospace')
                        )
                    )
                    st.plotly_chart(fig_tel, use_container_width=True)

            except Exception as e:
                st.error(f"Telemetry error: {e}")

# ─────────────────────────────────────────
# PAGE 3 — STRATEGY SIMULATOR
# ─────────────────────────────────────────
elif "Strategy Simulator" in page:
    st.markdown("# STRATEGY SIMULATOR")
    st.markdown("### ML pit stop decision engine · GradientBoosting · ROC-AUC 0.745")
    st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### RACE SITUATION")
        sim_tire_age = st.slider("TIRE AGE (laps)", 1, 50, 18)
        sim_compound = st.selectbox("CURRENT COMPOUND", ['SOFT', 'MEDIUM', 'HARD'], index=1)
        sim_position = st.slider("CURRENT POSITION", 1, 20, 5)
    with col2:
        st.markdown("### RACE STATE")
        sim_race_progress = st.slider("RACE PROGRESS (%)", 0, 100, 45)
        sim_laps_remaining = st.slider("LAPS REMAINING", 1, 60, 25)
        sim_team = st.selectbox("TEAM", list(TEAM_COLORS.keys()), index=0)

    st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)

    compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
    features = pd.DataFrame([{
        'tire_age_at_pit': sim_tire_age,
        'compound_num': compound_map[sim_compound],
        'race_progress': sim_race_progress / 100,
        'position_before': sim_position / 20,
        'laps_remaining': sim_laps_remaining,
    }])

    prob = strategy_model.predict_proba(features)[0][1]
    decision = "PIT NOW" if prob >= 0.5 else "STAY OUT"
    decision_color = "#E24B4A" if prob >= 0.5 else "#1D9E75"
    team_color = TEAM_COLORS.get(sim_team, '#E24B4A')

    col_dec, col_gauge, col_info = st.columns([1, 1, 1])

    with col_dec:
        st.markdown(f"""
        <div style='
            background: #0d0d0d;
            border: 1px solid #1a1a1a;
            border-top: 3px solid {decision_color};
            border-radius: 6px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            height: 100%;
        '>
            <div style='font-size:10px; color:#333; letter-spacing:0.2em;
                        text-transform:uppercase; margin-bottom:12px; font-family:monospace'>
                ML RECOMMENDATION
            </div>
            <div style='font-family:Bebas Neue,monospace; font-size:3.5rem;
                        color:{decision_color}; letter-spacing:0.1em; line-height:1'>
                {decision}
            </div>
            <div style='font-size:11px; color:#555; margin-top:12px; font-family:monospace'>
                Confidence: {prob*100:.1f}%
            </div>
            <div style='margin-top:1rem; height:4px; border-radius:2px;
                        background:linear-gradient(90deg, {decision_color} {prob*100:.0f}%, #1a1a1a {prob*100:.0f}%)'>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title=dict(
                text="PIT PROBABILITY",
                font=dict(color='#444', size=11, family='monospace')
            ),
            number=dict(
                suffix="%",
                font=dict(color='#ffffff', size=28, family='monospace')
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickfont=dict(color='#333', size=9),
                    tickcolor='#222'
                ),
                bar=dict(color=decision_color, thickness=0.6),
                bgcolor='#0d0d0d',
                bordercolor='#1a1a1a',
                borderwidth=1,
                steps=[
                    dict(range=[0, 35], color='#0d150d'),
                    dict(range=[35, 65], color='#15150d'),
                    dict(range=[65, 100], color='#150d0d'),
                ],
                threshold=dict(
                    line=dict(color='#ffffff', width=1.5),
                    thickness=0.7,
                    value=50
                )
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor='#080808',
            font=dict(color='#ffffff', family='monospace'),
            height=250,
            margin=dict(t=40, b=10, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        if prob >= 0.7:
            msg = "⚠️ Strong pit signal. Tire cliff approaching. Box this lap."
            msg_color = "#E24B4A"
        elif prob >= 0.5:
            msg = "🟡 Marginal window. Weigh track position vs fresh rubber."
            msg_color = "#EF9F27"
        elif prob >= 0.3:
            msg = "🟢 Tires viable. Stay out if position is valuable."
            msg_color = "#1D9E75"
        else:
            msg = "✅ No pit required. Performance still strong."
            msg_color = "#378ADD"

        st.markdown(f"""
        <div style='background:#0d0d0d; border:1px solid #1a1a1a;
                    border-left:3px solid {msg_color}; border-radius:4px;
                    padding:1.5rem; height:100%; font-family:monospace'>
            <div style='font-size:10px; color:#333; letter-spacing:0.2em;
                        text-transform:uppercase; margin-bottom:1rem'>ANALYSIS</div>
            <div style='font-size:12px; color:#888; line-height:1.8; margin-bottom:1.5rem'>
                {msg}
            </div>
            <div style='font-size:11px; color:#2a2a2a; line-height:2.5'>
                <span style='color:#333'>TIRE AGE</span> · 
                <span style='color:#666'>{sim_tire_age} laps</span><br>
                <span style='color:#333'>COMPOUND</span> · 
                <span style='color:#666'>{sim_compound}</span><br>
                <span style='color:#333'>POSITION</span> · 
                <span style='color:#666'>P{sim_position}</span><br>
                <span style='color:#333'>PROGRESS</span> · 
                <span style='color:#666'>{sim_race_progress}%</span><br>
                <span style='color:#333'>LAPS LEFT</span> · 
                <span style='color:{team_color}'>{sim_laps_remaining}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sensitivity analysis
    st.markdown('<div class="red-line"></div>', unsafe_allow_html=True)
    st.markdown("### PIT WINDOW SENSITIVITY")

    tire_ages = np.arange(1, 51)
    probs = []
    for ta in tire_ages:
        f = pd.DataFrame([{
            'tire_age_at_pit': ta,
            'compound_num': compound_map[sim_compound],
            'race_progress': sim_race_progress / 100,
            'position_before': sim_position / 20,
            'laps_remaining': sim_laps_remaining,
        }])
        probs.append(strategy_model.predict_proba(f)[0][1] * 100)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=tire_ages, y=probs,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(226,75,74,0.08)',
        line=dict(color='#E24B4A', width=2.5),
        hovertemplate='Tire age %{x}: %{y:.1f}%<extra></extra>'
    ))
    fig_sens.add_hline(
        y=50,
        line=dict(color='#333', width=1, dash='dash'),
        annotation_text="PIT THRESHOLD",
        annotation_font=dict(color='#444', size=9, family='monospace')
    )
    fig_sens.add_vline(
        x=sim_tire_age,
        line=dict(color='#EF9F27', width=1.5, dash='dot'),
        annotation_text=f"NOW",
        annotation_font=dict(color='#EF9F27', size=9, family='monospace')
    )

    sens_layout = {**PLOT_LAYOUT}
    sens_layout['height'] = 300
    sens_layout['xaxis'] = styled_axis('Tire Age (laps)')
    sens_layout['yaxis'] = {**styled_axis('Pit Probability (%)'), 'range': [0, 100]}
    sens_layout['margin'] = dict(t=30, b=50, l=60, r=20)
    fig_sens.update_layout(**sens_layout)
    st.plotly_chart(fig_sens, use_container_width=True)

# ─────────────────────────────────────────
# GLOBAL FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style='margin-top:4rem; padding-top:1rem;
            border-top:1px solid #111; text-align:center'>
    <div style='font-family:monospace; font-size:11px; color:#222;
                letter-spacing:0.15em; line-height:2'>
        F1 STRATEGY INTELLIGENCE SYSTEM<br>
        <span style='color:#1a1a1a'>
            Built with FastF1 · scikit-learn · Streamlit · Plotly
        </span><br>
        <span style='color:#E24B4A; font-size:13px'>♥</span>
        <span style='color:#1a1a1a'> Made with love by Arin</span>
    </div>
</div>
""", unsafe_allow_html=True)