import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import fastf1
import warnings
warnings.filterwarnings('ignore')

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
# CUSTOM CSS — F1 PIT WALL AESTHETIC
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bebas+Neue&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
    background-color: #080808 !important;
    color: #dddddd !important;
}

/* Main background */
.stApp { background-color: #080808 !important; }
.main .block-container { 
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 1px solid #1a1a1a !important;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }

/* Headers */
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
    font-size: 0.9rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #666666 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f0f0f !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    color: #555 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem !important;
}

/* Selectbox + sliders */
.stSelectbox > div > div {
    background: #0f0f0f !important;
    border: 1px solid #222 !important;
    border-radius: 6px !important;
    color: #ddd !important;
}
.stSlider > div > div > div {
    background: #E24B4A !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #E24B4A !important;
    color: #E24B4A !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #E24B4A !important;
    color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0d0d !important;
    border-bottom: 1px solid #1a1a1a !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #555 !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.08em !important;
    padding: 0.75rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #ffffff !important;
    border-bottom: 2px solid #E24B4A !important;
}

/* Divider */
hr { border-color: #1a1a1a !important; }

/* Info boxes */
.stInfo {
    background: #0f0f0f !important;
    border: 1px solid #1e1e1e !important;
    border-left: 3px solid #378ADD !important;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

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
    st.markdown("# 🏎 F1 STRATEGY\n### INTELLIGENCE SYSTEM")
    st.markdown("---")
    st.markdown("### NAVIGATION")
    page = st.radio(
        "",
        ["🏁  Race Strategy", "🧬  Driver DNA", "🔮  Strategy Simulator"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### DATA COVERAGE")
    st.markdown("""
    <div style='font-size:12px; color:#555; line-height:2'>
    📅 Seasons: 2022 · 2023 · 2024 · 2026<br>
    🏟 Circuits: 27 Grand Prix<br>
    🚗 Drivers: 34 profiles<br>
    📊 Laps: 32,572 analyzed<br>
    🔬 Models: 3 ML models
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#333; text-align:center'>
    Built with FastF1 · scikit-learn<br>
    Data: 2022–2026 seasons
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 1 — RACE STRATEGY
# ─────────────────────────────────────────
if "Race Strategy" in page:
    st.markdown("# RACE STRATEGY COMMAND CENTER")
    st.markdown("### Real-time pit window optimization using ML")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_race = st.selectbox(
            "SELECT RACE",
            sorted(clean_laps['race_name'].unique()),
            index=0
        )
    with col2:
        race_drivers = clean_laps[clean_laps['race_name'] == selected_race]['Driver'].unique()
        selected_driver = st.selectbox("SELECT DRIVER", sorted(race_drivers))
    with col3:
        selected_compound = st.selectbox(
            "CURRENT COMPOUND",
            ['SOFT', 'MEDIUM', 'HARD'],
            index=1
        )

    st.markdown("---")

    # Driver lap data
    driver_laps = clean_laps[
        (clean_laps['race_name'] == selected_race) &
        (clean_laps['Driver'] == selected_driver)
    ].copy()

    if not driver_laps.empty:
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("TOTAL LAPS", f"{int(driver_laps['LapNumber'].max())}")
        with m2:
            best_lap = driver_laps['LapTimeSec'].min()
            mins = int(best_lap // 60)
            secs = best_lap % 60
            st.metric("FASTEST LAP", f"{mins}:{secs:06.3f}")
        with m3:
            avg_deg = driver_laps['DeltaFromFastest'].mean()
            st.metric("AVG DEGRADATION", f"+{avg_deg:.2f}s")
        with m4:
            team = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else 'Unknown'
            st.metric("TEAM", team)

        st.markdown("---")

        # Lap time chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=driver_laps['LapNumber'],
            y=driver_laps['LapTimeSec'],
            mode='lines+markers',
            line=dict(
                color=TEAM_COLORS.get(team, '#E24B4A'),
                width=2
            ),
            marker=dict(size=4),
            name=f'{selected_driver} lap times',
            hovertemplate='Lap %{x}: %{y:.3f}s<extra></extra>'
        ))

        # Highlight pit laps
        pit_laps = driver_laps[driver_laps['TyreLife'] == driver_laps['TyreLife'].min()]

        fig.update_layout(
            title=dict(
                text=f'{selected_driver}  ·  {selected_race}  ·  Lap Time Progression',
                font=dict(size=16, color='#ffffff', family='monospace'),
            ),
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#ffffff', family='monospace'),
            xaxis=dict(
                title='Lap Number',
                gridcolor='#141414',
                title_font=dict(color='#555'),
                tickfont=dict(color='#555')
            ),
            yaxis=dict(
                title='Lap Time (s)',
                gridcolor='#141414',
                title_font=dict(color='#555'),
                tickfont=dict(color='#555')
            ),
            height=400,
            margin=dict(t=50, b=50, l=60, r=20),
            hoverlabel=dict(
                bgcolor='#111', bordercolor='#333',
                font=dict(color='#fff', family='monospace')
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tire degradation section
        st.markdown("### TIRE DEGRADATION ANALYSIS")
        
        # Load per-circuit tire model if exists
        import os
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

            fig2.update_layout(
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#0a0a0a',
                font=dict(color='#ffffff', family='monospace'),
                xaxis=dict(
                    title='Tire Age (laps)',
                    gridcolor='#141414',
                    title_font=dict(color='#555'),
                    tickfont=dict(color='#555')
                ),
                yaxis=dict(
                    title='Delta from fastest (s)',
                    gridcolor='#141414',
                    title_font=dict(color='#555'),
                    tickfont=dict(color='#555')
                ),
                height=350,
                margin=dict(t=30, b=50, l=60, r=20),
                legend=dict(
                    bgcolor='#111', bordercolor='#222',
                    font=dict(color='#aaa', family='monospace')
                ),
                hoverlabel=dict(
                    bgcolor='#111', bordercolor='#333',
                    font=dict(color='#fff', family='monospace')
                )
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"No tire model available for {selected_race} — run Module 01 with this circuit to generate one.")
    else:
        st.warning("No data available for this driver/race combination.")

# ─────────────────────────────────────────
# PAGE 2 — DRIVER DNA
# ─────────────────────────────────────────
elif "Driver DNA" in page:
    st.markdown("# DRIVER DNA LAB")
    st.markdown("### Telemetry-based driving style fingerprinting")
    st.markdown("---")

    drivers_available = sorted(driver_dna['driver'].unique())

    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.selectbox("DRIVER 1", drivers_available,
                               index=drivers_available.index('VER') if 'VER' in drivers_available else 0)
    with col2:
        driver2 = st.selectbox("DRIVER 2", drivers_available,
                               index=drivers_available.index('HAM') if 'HAM' in drivers_available else 1)

    st.markdown("---")

    # DNA features
    raw_features = ['full_throttle_pct', 'heavy_brake_pct', 'avg_corner_speed',
                    'throttle_smoothness', 'coast_pct', 'high_speed_pct']
    categories = ['Full Throttle %', 'Heavy Braking %', 'Corner Speed',
                  'Smoothness', 'Coast %', 'High Speed %']

    def normalize_0_100(series):
        return 30 + ((series - series.min()) / (series.max() - series.min())) * 70

    dna_norm = driver_dna.copy()
    for f in raw_features:
        dna_norm[f + '_norm'] = normalize_0_100(driver_dna[f])

    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r},{g},{b},{alpha})'

    # Radar chart
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

    col_radar, col_stats = st.columns([1.2, 0.8])

    with col_radar:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=v1, theta=cats,
            fill='toself',
            fillcolor=hex_to_rgba(c1, 0.25),
            line=dict(color=c1, width=3),
            name=f'{driver1}  ·  {t1}',
            marker=dict(size=8, color=c1)
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=v2, theta=cats,
            fill='toself',
            fillcolor=hex_to_rgba(c2, 0.25),
            line=dict(color=c2, width=3),
            name=f'{driver2}  ·  {t2}',
            marker=dict(size=8, color=c2)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#0a0a0a',
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickvals=[25, 50, 75, 100],
                    tickfont=dict(color='#333', size=9),
                    gridcolor='#1a1a1a',
                    linecolor='#1a1a1a',
                ),
                angularaxis=dict(
                    tickfont=dict(color='#cccccc', size=12, family='monospace'),
                    gridcolor='#1e1e1e',
                    linecolor='#333',
                )
            ),
            paper_bgcolor='#080808',
            title=dict(
                text=f'DNA COMPARISON  ·  {driver1} vs {driver2}',
                font=dict(size=16, color='#ffffff', family='monospace'),
                x=0.5, xanchor='center'
            ),
            legend=dict(
                bgcolor='rgba(12,12,12,0.95)',
                bordercolor='#222', borderwidth=1,
                font=dict(color='#cccccc', size=11, family='monospace'),
                x=0.5, y=-0.1,
                xanchor='center', orientation='h'
            ),
            height=500,
            margin=dict(t=80, b=100, l=60, r=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_stats:
        st.markdown(f"### {driver1} PROFILE")
        d1_raw = driver_dna[driver_dna['driver'] == driver1].iloc[0]
        st.metric("Archetype", d1_raw.get('archetype', 'N/A'))
        st.metric("Full Throttle", f"{d1_raw['full_throttle_pct']:.1f}%")
        st.metric("Heavy Braking", f"{d1_raw['heavy_brake_pct']:.1f}%")
        st.metric("Avg Corner Speed", f"{d1_raw['avg_corner_speed']:.0f} km/h")
        st.metric("Max Speed", f"{d1_raw['max_speed']:.0f} km/h")

        st.markdown("---")
        st.markdown(f"### {driver2} PROFILE")
        d2_raw = driver_dna[driver_dna['driver'] == driver2].iloc[0]
        st.metric("Archetype", d2_raw.get('archetype', 'N/A'))
        st.metric("Full Throttle", f"{d2_raw['full_throttle_pct']:.1f}%")
        st.metric("Heavy Braking", f"{d2_raw['heavy_brake_pct']:.1f}%")
        st.metric("Avg Corner Speed", f"{d2_raw['avg_corner_speed']:.0f} km/h")
        st.metric("Max Speed", f"{d2_raw['max_speed']:.0f} km/h")

    # Telemetry speed trace
    st.markdown("---")
    st.markdown("### QUALIFYING SPEED TRACE")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        tel_year = st.selectbox("YEAR", [2026, 2024, 2023, 2022], index=1)
    with col_t2:
        tel_race = st.selectbox("CIRCUIT", ['Bahrain', 'Monaco', 'Britain', 'Italy',
                                             'Australia', 'Japan'], index=0)
    with col_t3:
        load_tel = st.button("LOAD TELEMETRY ▶")

    if load_tel:
        with st.spinner(f"Loading {tel_race} {tel_year} qualifying telemetry..."):
            try:
                from scipy.interpolate import interp1d
                session = fastf1.get_session(tel_year, tel_race, 'Q')
                session.load(telemetry=True, weather=False, messages=False)

                lap1 = session.laps.pick_driver(driver1).pick_fastest()
                lap2 = session.laps.pick_driver(driver2).pick_fastest()
                tel1 = lap1.get_telemetry().add_distance()
                tel2 = lap2.get_telemetry().add_distance()

                dist_min = max(tel1['Distance'].min(), tel2['Distance'].min())
                dist_max = min(tel1['Distance'].max(), tel2['Distance'].max())
                common_dist = np.linspace(dist_min, dist_max, 1000)

                s1 = interp1d(tel1['Distance'], tel1['Speed'],
                              kind='linear', fill_value='extrapolate')(common_dist)
                s2 = interp1d(tel2['Distance'], tel2['Speed'],
                              kind='linear', fill_value='extrapolate')(common_dist)
                delta = s1 - s2

                time1 = lap1['LapTime'].total_seconds()
                time2 = lap2['LapTime'].total_seconds()
                m1, s1t = int(time1 // 60), time1 % 60
                m2, s2t = int(time2 // 60), time2 % 60

                fig_tel = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.06
                )

                fig_tel.add_trace(go.Scatter(
                    x=common_dist, y=s1,
                    mode='lines',
                    name=f'{driver1}  ·  {m1}:{s1t:06.3f}',
                    line=dict(color=c1, width=2.5),
                    hovertemplate=f'{driver1}: %{{y:.0f}} km/h<extra></extra>'
                ), row=1, col=1)

                fig_tel.add_trace(go.Scatter(
                    x=common_dist, y=s2,
                    mode='lines',
                    name=f'{driver2}  ·  {m2}:{s2t:06.3f}',
                    line=dict(color=c2, width=2.5),
                    hovertemplate=f'{driver2}: %{{y:.0f}} km/h<extra></extra>'
                ), row=1, col=1)

                fig_tel.add_trace(go.Scatter(
                    x=common_dist, y=delta,
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=f'rgba({int(c1[1:3],16)},{int(c1[3:5],16)},{int(c1[5:7],16)},0.15)',
                    line=dict(color=c1, width=1.5),
                    name=f'Δ Speed ({driver1} - {driver2})',
                    hovertemplate='Δ: %{y:.1f} km/h<extra></extra>'
                ), row=2, col=1)

                fig_tel.add_hline(y=0, line=dict(color='#333', width=1), row=2, col=1)

                for row in [1, 2]:
                    fig_tel.update_xaxes(
                        gridcolor='#141414', zerolinecolor='#222',
                        tickfont=dict(color='#444', size=10),
                        title_font=dict(color='#555', size=11),
                        row=row, col=1
                    )
                    fig_tel.update_yaxes(
                        gridcolor='#141414', zerolinecolor='#222',
                        tickfont=dict(color='#555', size=10),
                        title_font=dict(color='#555', size=11),
                        row=row, col=1
                    )

                fig_tel.update_layout(
                    title=dict(
                        text=f'{tel_race.upper()} {tel_year}  ·  QUALIFYING  ·  {driver1} vs {driver2}',
                        font=dict(size=16, color='#ffffff', family='monospace'),
                        x=0.5, xanchor='center'
                    ),
                    plot_bgcolor='#0a0a0a',
                    paper_bgcolor='#0a0a0a',
                    font=dict(color='#ffffff', family='monospace'),
                    legend=dict(
                        bgcolor='rgba(12,12,12,0.95)',
                        bordercolor='#222', borderwidth=1,
                        font=dict(color='#ccc', size=11, family='monospace'),
                        x=0.01, y=0.99
                    ),
                    height=600,
                    margin=dict(t=70, b=50, l=70, r=30),
                    hovermode='x unified',
                    hoverlabel=dict(
                        bgcolor='#111', bordercolor='#333',
                        font=dict(color='#fff', family='monospace')
                    )
                )
                st.plotly_chart(fig_tel, use_container_width=True)

            except Exception as e:
                st.error(f"Could not load telemetry: {e}")

# ─────────────────────────────────────────
# PAGE 3 — STRATEGY SIMULATOR
# ─────────────────────────────────────────
elif "Strategy Simulator" in page:
    st.markdown("# STRATEGY SIMULATOR")
    st.markdown("### ML-powered pit stop decision engine")
    st.markdown("---")

    st.markdown("### SET RACE SITUATION")

    col1, col2 = st.columns(2)
    with col1:
        sim_tire_age = st.slider("TIRE AGE (laps)", 1, 50, 18)
        sim_compound = st.selectbox("CURRENT COMPOUND", ['SOFT', 'MEDIUM', 'HARD'], index=1)
        sim_position = st.slider("CURRENT POSITION", 1, 20, 5)
    with col2:
        sim_race_progress = st.slider("RACE PROGRESS (%)", 0, 100, 45)
        sim_laps_remaining = st.slider("LAPS REMAINING", 1, 60, 25)
        sim_team = st.selectbox("TEAM", list(TEAM_COLORS.keys()), index=0)

    st.markdown("---")

    # ML prediction
    compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
    team_map = {
        'Red Bull Racing': 0, 'Ferrari': 1, 'Mercedes': 2,
        'McLaren': 3, 'Aston Martin': 4, 'Alpine': 5,
        'Williams': 6, 'AlphaTauri': 7, 'Alfa Romeo': 8,
        'Haas F1 Team': 9, 'RB': 7, 'Kick Sauber': 8,
        'Racing Bulls': 7, 'Cadillac': 10, 'Audi': 11
    }

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

    # Big decision display
    st.markdown(f"""
    <div style='
        background: #0f0f0f;
        border: 1px solid #1e1e1e;
        border-left: 4px solid {decision_color};
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    '>
        <div style='font-size:12px; color:#555; letter-spacing:0.15em; 
                    text-transform:uppercase; margin-bottom:8px; font-family:monospace'>
            ML STRATEGY RECOMMENDATION
        </div>
        <div style='font-family: Bebas Neue, monospace; font-size: 4rem; 
                    color: {decision_color}; letter-spacing: 0.1em; line-height:1'>
            {decision}
        </div>
        <div style='font-size:14px; color:#888; margin-top:8px; font-family:monospace'>
            Confidence: {prob*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title=dict(
            text="PIT PROBABILITY",
            font=dict(color='#888', size=14, family='monospace')
        ),
        number=dict(
            suffix="%",
            font=dict(color='#ffffff', size=32, family='monospace')
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickfont=dict(color='#444', size=10),
                tickcolor='#333'
            ),
            bar=dict(color=decision_color),
            bgcolor='#0f0f0f',
            bordercolor='#1a1a1a',
            steps=[
                dict(range=[0, 35], color='#0d1a0d'),
                dict(range=[35, 65], color='#1a1a0d'),
                dict(range=[65, 100], color='#1a0d0d'),
            ],
            threshold=dict(
                line=dict(color='#ffffff', width=2),
                thickness=0.75,
                value=50
            )
        )
    ))
    fig_gauge.update_layout(
        paper_bgcolor='#080808',
        font=dict(color='#ffffff', family='monospace'),
        height=300,
        margin=dict(t=40, b=20, l=40, r=40)
    )

    col_g, col_info = st.columns([1, 1])
    with col_g:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_info:
        st.markdown("### SITUATION ANALYSIS")
        st.markdown(f"""
        <div style='font-family:monospace; font-size:13px; 
                    line-height:2.2; color:#888; margin-top:1rem'>
            <span style='color:#555'>TIRE AGE</span>      
            <span style='color:#fff'>{sim_tire_age} laps</span><br>
            <span style='color:#555'>COMPOUND</span>      
            <span style='color:#fff'>{sim_compound}</span><br>
            <span style='color:#555'>POSITION</span>      
            <span style='color:#fff'>P{sim_position}</span><br>
            <span style='color:#555'>RACE PROG</span>     
            <span style='color:#fff'>{sim_race_progress}%</span><br>
            <span style='color:#555'>LAPS LEFT</span>     
            <span style='color:#fff'>{sim_laps_remaining}</span><br>
            <span style='color:#555'>TEAM</span>          
            <span style='color:{TEAM_COLORS.get(sim_team, "#fff")}'>{sim_team}</span>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation text
        st.markdown("---")
        if prob >= 0.7:
            msg = "⚠️ Strong pit signal. Tire performance likely falling off. Box this lap."
        elif prob >= 0.5:
            msg = "🟡 Marginal pit window. Consider track position vs fresh tires."
        elif prob >= 0.3:
            msg = "🟢 Tires still viable. Stay out if track position is valuable."
        else:
            msg = "✅ No pit required. Tires performing well for this race stage."
        st.info(msg)

    # Sensitivity analysis
    st.markdown("---")
    st.markdown("### SENSITIVITY ANALYSIS")
    st.markdown("How does pit probability change as tire age increases?")

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
        fillcolor='rgba(226,75,74,0.1)',
        line=dict(color='#E24B4A', width=2.5),
        hovertemplate='Tire age %{x}: %{y:.1f}% pit probability<extra></extra>'
    ))
    fig_sens.add_hline(
        y=50,
        line=dict(color='#ffffff', width=1, dash='dash'),
        annotation_text="PIT THRESHOLD",
        annotation_font=dict(color='#555', size=10, family='monospace')
    )
    fig_sens.add_vline(
        x=sim_tire_age,
        line=dict(color='#EF9F27', width=2, dash='dot'),
        annotation_text=f"NOW (lap {sim_tire_age})",
        annotation_font=dict(color='#EF9F27', size=10, family='monospace')
    )
    fig_sens.update_layout(
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font=dict(color='#ffffff', family='monospace'),
        xaxis=dict(
            title='Tire Age (laps)',
            gridcolor='#141414',
            title_font=dict(color='#555'),
            tickfont=dict(color='#555')
        ),
        yaxis=dict(
            title='Pit Probability (%)',
            gridcolor='#141414',
            title_font=dict(color='#555'),
            tickfont=dict(color='#555'),
            range=[0, 100]
        ),
        height=350,
        margin=dict(t=30, b=50, l=60, r=20),
        hoverlabel=dict(
            bgcolor='#111', bordercolor='#333',
            font=dict(color='#fff', family='monospace')
        )
    )
    st.plotly_chart(fig_sens, use_container_width=True)