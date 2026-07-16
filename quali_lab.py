"""Broadcast-style qualifying analysis visuals for STRAT."""
from __future__ import annotations

import hashlib
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

RED = "#E10600"
YELLOW = "#FFEB00"
CYAN = "#00E5FF"
ORANGE = "#FF6A00"
LIME = "#B4FF00"
MAGENTA = "#FF2BD6"
BLUE = "#3D7EFF"
WHITE = "#F2F2F2"
BG = "#101116"      # The dark blue/black background from the F1 graphics
MUTED = "#515462"   # Medium/dark grey for labels
GRID = "#1b1c23"    # Very dark blue-grey grid lines

# High-contrast palette — never rely on team colors
DRIVER_PALETTE = [
    RED, YELLOW, CYAN, ORANGE, LIME, MAGENTA, BLUE, WHITE,
    "#7CFFB2", "#FF8FAB", "#A78BFA", "#38BDF8",
]

def format_laptime(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:06.3f}"

def driver_theme_color(code: str) -> str:
    """Stable unique-ish color per driver code."""
    h = int(hashlib.md5(code.encode()).hexdigest(), 16)
    return DRIVER_PALETTE[h % len(DRIVER_PALETTE)]

def pair_theme_colors(code1: str, code2: str) -> Tuple[str, str]:
    """Always return two visually different theme colors for a head-to-head."""
    c1 = driver_theme_color(code1)
    c2 = driver_theme_color(code2)
    if c1 == c2:
        # pick next palette slot for driver 2
        i = DRIVER_PALETTE.index(c1)
        c2 = DRIVER_PALETTE[(i + 1) % len(DRIVER_PALETTE)]
    return c1, c2

def drive_style_from_speed(speed: np.ndarray) -> Tuple[float, float, float]:
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 5:
        return 45.0, 20.0, 35.0
    vmax = float(np.nanmax(speed)) or 1.0
    ds = np.diff(speed, prepend=speed[0])
    full_throttle = float(np.clip(np.mean(speed > 0.78 * vmax) * 100, 8, 92))
    heavy_brake = float(np.clip(np.mean(ds < -6.5) * 100 * 2.2, 5, 55))
    cornering = float(np.clip(np.mean(speed < 0.58 * vmax) * 100, 10, 70))
    return full_throttle, heavy_brake, cornering

def cumulative_time_delta(distance: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    v1 = np.maximum(np.asarray(s1, dtype=float), 1.0) / 3.6
    v2 = np.maximum(np.asarray(s2, dtype=float), 1.0) / 3.6
    dd = np.diff(distance, prepend=distance[0])
    dd[0] = 0.0
    return np.cumsum(dd / v1) - np.cumsum(dd / v2)

def detect_turn_distances(distance: np.ndarray, speed: np.ndarray, max_turns: int = 14) -> np.ndarray:
    speed = np.asarray(speed, dtype=float)
    distance = np.asarray(distance, dtype=float)
    if len(speed) < 30:
        return np.array([])
    peaks, _ = find_peaks(-speed, distance=max(8, len(speed) // 28), prominence=12)
    if len(peaks) == 0:
        return np.array([])
    order = np.argsort(speed[peaks])[:max_turns]
    return distance[np.sort(peaks[order])]

def load_track_outline(year: int, race: str) -> Optional[pd.DataFrame]:
    path = os.path.join("data", "telemetry", f"track_{year}_{race.replace(' ', '_')}.csv")
    if not os.path.exists(path):
        folder = os.path.join("data", "telemetry")
        for fname in sorted(os.listdir(folder), reverse=True):
            if fname.startswith("track_") and race.replace(" ", "_") in fname and fname.endswith(".csv"):
                path = os.path.join(folder, fname)
                break
        else:
            return None
    df = pd.read_csv(path)
    if not {"X", "Y"}.issubset(df.columns):
        return None
    return df

def get_team_logo_svg(team: str, color: str) -> str:
    team_lower = team.lower()
    if 'ferrari' in team_lower:
        return """
        <svg viewBox="0 0 40 50" width="32" height="40" style="display: block;">
            <path d="M20 2 C32 2, 36 8, 36 28 C36 42, 20 48, 20 48 C20 48, 4 42, 4 28 C4 8, 8 2, 20 2 Z" fill="#FFEB00"/>
            <path d="M20,12 c-0.4,1 -1,2.3 -0.5,3.3 c0.4,0.8 1.8,0.7 2.1,1.5 c0.3,0.8 -0.8,1.4 -0.6,2.2 c0.2,0.8 1.5,0.7 1.3,1.9 c-0.2,1.2 -1.4,1 -1.4,2 c0,1 0.7,0.9 0.7,1.8 c0,0.9 -1.1,1 -1,1.8 c0.1,0.8 1.4,0.7 1,1.8 c-0.4,1.1 -2.3,1.6 -3.2,1.3 c-0.9,-0.3 -0.4,-2 -0.8,-2.6 c-0.4,-0.6 -1.8,0 -2.3,-0.6 c-0.5,-0.6 0.1,-1.5 0.2,-2.3 c0.1,-0.8 -0.7,-1 -0.9,-1.8 c-0.2,-0.8 0.3,-1.2 0.3,-2 c0,-0.8 -0.9,-1 -0.7,-1.8 c0.2,-0.8 0.8,-0.9 0.5,-1.8 c-0.3,-0.9 -1.2,-0.5 -1.2,-1.3 c0,-0.8 1,-1.4 1.7,-2 c0.7,-0.6 0.8,-1.7 1.8,-1.8 c1,-0.1 1.7,0.7 2.3,0.3 c0.6,-0.4 0.6,-1.7 1.2,-1.8 C19.9,9.9 20.4,11 20,12 Z" fill="#000000"/>
        </svg>
        """
    elif 'mercedes' in team_lower:
        return f"""
        <svg viewBox="0 0 40 40" width="32" height="32" style="display: block;">
            <circle cx="20" cy="20" r="18" fill="none" stroke="{color}" stroke-width="2"/>
            <path d="M20 20 L20 4 M20 20 L6 28 M20 20 L34 28" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
        </svg>
        """
    elif 'red bull' in team_lower:
        return f"""
        <svg viewBox="0 0 42 30" width="38" height="28" style="display: block;">
            <circle cx="21" cy="15" r="9" fill="#FFCC00"/>
            <path d="M 4 20 C 8 14, 12 14, 18 18 C 24 22, 28 16, 32 18 L 30 12 C 26 14, 22 10, 16 14 C 10 18, 6 16, 4 20 Z" fill="{color}"/>
            <path d="M 8 24 C 12 20, 16 18, 22 20 C 28 22, 30 18, 34 20" fill="none" stroke="#E8002D" stroke-width="2" stroke-linecap="round"/>
        </svg>
        """
    elif 'mclaren' in team_lower:
        return f"""
        <svg viewBox="0 0 40 30" width="34" height="26" style="display: block;">
            <path d="M10 25 C18 21, 28 17, 34 9 C30 13, 20 15, 10 25 Z" fill="{color}"/>
        </svg>
        """
    elif 'aston martin' in team_lower:
        return f"""
        <svg viewBox="0 0 50 30" width="40" height="24" style="display: block;">
            <path d="M5 15 C15 10, 20 12, 25 15 C30 12, 35 10, 45 15 C35 22, 15 22, 5 15 Z" fill="none" stroke="{color}" stroke-width="2"/>
            <path d="M10 15 L40 15 M15 15 L20 18 M35 15 L30 18" fill="none" stroke="{color}" stroke-width="1.5"/>
            <rect x="22" y="11" width="6" height="6" fill="{color}"/>
        </svg>
        """
    elif 'alpine' in team_lower:
        return f"""
        <svg viewBox="0 0 40 40" width="32" height="32" style="display: block;">
            <path d="M10 34 L20 6 L30 34 L25 34 L20 20 L15 34 Z M16 28 L24 28" fill="{color}"/>
        </svg>
        """
    elif 'williams' in team_lower:
        return f"""
        <svg viewBox="0 0 40 40" width="32" height="32" style="display: block;">
            <path d="M6 10 L14 30 L20 18 L26 30 L34 10 L28 10 L23 22 L17 10 Z" fill="{color}"/>
        </svg>
        """
    elif 'haas' in team_lower:
        return f"""
        <svg viewBox="0 0 40 40" width="32" height="32" style="display: block;">
            <circle cx="20" cy="20" r="16" fill="none" stroke="{color}" stroke-width="2"/>
            <path d="M14 12 L14 28 M26 12 L26 28 M14 20 L26 20" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
        </svg>
        """
    elif 'sauber' in team_lower or 'alfa romeo' in team_lower:
        return f"""
        <svg viewBox="0 0 40 40" width="32" height="32" style="display: block;">
            <circle cx="20" cy="20" r="16" fill="none" stroke="{color}" stroke-width="2"/>
            <path d="M14 14 C18 10, 26 10, 26 18 C26 22, 14 22, 14 26 C14 30, 22 30, 26 26" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
        </svg>
        """
    else:
        initial = team[0].upper() if team else 'T'
        return f"""
        <svg viewBox="0 0 40 40" width="32" height="32" style="display: block;">
            <circle cx="20" cy="20" r="16" fill="none" stroke="{color}" stroke-width="2"/>
            <text x="20" y="26" font-family="'Inter', sans-serif" font-weight="900" font-size="18" fill="{color}" text-anchor="middle">{initial}</text>
        </svg>
        """

def _bar_html(label: str, pct: float, color: str) -> str:
    w = max(4.0, min(100.0, pct))
    return f"""
    <div style="margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-family: 'Inter', sans-serif; font-weight: 700; font-size: 0.65rem; color: #8a8d98; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 3px;">
            <span>{label}</span>
            <span style="color: #ffffff; font-family: 'Bebas Neue', sans-serif; font-size: 0.9rem; font-style: italic;">{pct:.0f}%</span>
        </div>
        <div style="height: 5px; background: #222530; border-radius: 2px; overflow: hidden; width: 100%;">
            <div style="width: {w:.1f}%; height: 100%; background: {color}; border-radius: 2px;"></div>
        </div>
    </div>
    """

def get_team_logo_html(team: str, color: str) -> str:
    team_lower = team.lower()
    
    # Map team name to filename key
    if 'ferrari' in team_lower:
        logo_key = 'ferrari'
    elif 'mercedes' in team_lower:
        logo_key = 'mercedes'
    elif 'red bull' in team_lower:
        logo_key = 'red_bull_racing'
    elif 'mclaren' in team_lower:
        logo_key = 'mclaren'
    elif 'aston martin' in team_lower:
        logo_key = 'aston_martin'
    elif 'alpine' in team_lower:
        logo_key = 'alpine'
    elif 'williams' in team_lower:
        logo_key = 'williams'
    elif 'haas' in team_lower:
        logo_key = 'haas'
    elif 'sauber' in team_lower or 'alfa romeo' in team_lower:
        logo_key = 'sauber'
    elif 'alphatauri' in team_lower or 'rb' in team_lower or 'bulls' in team_lower:
        logo_key = 'rb'
    else:
        logo_key = team_lower.replace(" ", "_")
        
    # Check paths in root or logos/ directory
    paths = [
        f"logos/{logo_key}.png",
        f"logos/{logo_key}.jpg",
        f"data/logos/{logo_key}.png",
        f"{logo_key}.png"
    ]
    
    import base64
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                ext = p.split(".")[-1]
                return f'<img src="data:image/{ext};base64,{data}" style="height: 40px; max-width: 60px; object-fit: contain; display: block;" />'
            except Exception:
                pass
                
    # Fallback to premium CSS/SVG monograms
    return get_team_logo_svg(team, color)

def build_driver_card_html(
    pos: int,
    code: str,
    full_name: str,
    team: str,
    lap_sec: float,
    gap_text: str,
    ft: float,
    hb: float,
    corner: float,
    accent: str,
    align: str = "left",
) -> str:
    parts = full_name.split(" ", 1)
    first = parts[0]
    last = parts[1] if len(parts) > 1 else code
    
    logo_html = get_team_logo_html(team, accent)
    
    # Text alignments
    text_align = "left" if align == "left" else "right"
    flex_dir = "row" if align == "left" else "row-reverse"
    border_style = f"border-left: 6px solid {accent};" if align == "left" else f"border-right: 6px solid {accent};"
    
    # Progress bars layout
    bar_margin = "margin-left: 10px;" if align == "left" else "margin-right: 10px;"
    bars_flex_dir = "row" if align == "left" else "row-reverse"
    
    # Pre-render the vertical labels to avoid backslashes inside f-string expressions
    lbl_left = "<div class='vertical-lbl-container' style='width: 15px; display: flex; align-items: center; justify-content: center; height: 75px;'><div style='transform: rotate(-90deg); white-space: nowrap; font-family: \"Inter\", sans-serif; font-size: 0.58rem; font-weight: 800; color: #4e515d; letter-spacing: 0.1em;'>% LAP TIME</div></div>" if align == "left" else ""
    lbl_right = "<div class='vertical-lbl-container' style='width: 15px; display: flex; align-items: center; justify-content: center; height: 75px;'><div style='transform: rotate(90deg); white-space: nowrap; font-family: \"Inter\", sans-serif; font-size: 0.58rem; font-weight: 800; color: #4e515d; letter-spacing: 0.1em;'>% LAP TIME</div></div>" if align == "right" else ""
    
    # Render the bars HTML
    bars_html = f"""
    <div style="display: flex; flex-direction: {bars_flex_dir}; align-items: center; margin-top: 15px; background: rgba(0,0,0,0.15); padding: 8px 12px; border-radius: 4px;">
        {lbl_left}
        <div style="flex: 1; {bar_margin}">
            {_bar_html("FULL THROTTLE", ft, accent)}
            {_bar_html("HEAVY BRAKING", hb, accent)}
            {_bar_html("CORNERING", corner, accent)}
        </div>
        {lbl_right}
    </div>
    """
    
    body = f"""
    <div class="card-container" style="background: #15161d; {border_style} padding: 16px 20px; box-sizing: border-box; height: 100%; display: flex; flex-direction: column; justify-content: space-between; border-radius: 4px; box-shadow: 0 8px 24px rgba(0,0,0,0.4);">
        <!-- Top Row: Position, Name, Logo -->
        <div style="display: flex; flex-direction: {flex_dir}; align-items: center; justify-content: space-between;">
            <div style="display: flex; flex-direction: {flex_dir}; align-items: center; gap: 16px;">
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 4.8rem; line-height: 0.8; font-weight: bold; color: #ffffff;">{pos}</div>
                <div style="text-align: {text_align};">
                    <div style="font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.85rem; color: #8a8d98; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 2px;">{first}</div>
                    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem; line-height: 0.85; font-weight: 900; color: {accent}; letter-spacing: 0.05em; text-transform: uppercase;">{last}</div>
                    <div style="font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.65rem; color: #515462; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 4px;">{team}</div>
                </div>
            </div>
            <div style="flex-shrink: 0;">
                {logo_html}
            </div>
        </div>
        
        <!-- Middle Row: Lap Time and Gap -->
        <div style="display: flex; flex-direction: {flex_dir}; justify-content: space-between; align-items: flex-end; margin-top: 15px; border-top: 1px solid #222530; padding-top: 12px;">
            <div style="text-align: {text_align};">
                <div style="font-family: 'Inter', sans-serif; font-weight: 800; font-size: 0.65rem; color: #8a8d98; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 2px;">LAP TIME</div>
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem; font-weight: bold; font-style: italic; color: #ffffff; letter-spacing: 0.02em; line-height: 1;">{format_laptime(lap_sec)}</div>
            </div>
            <div style="text-align: {"right" if align == "left" else "left"};">
                <div style="font-family: 'Inter', sans-serif; font-weight: 800; font-size: 0.65rem; color: #8a8d98; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 2px;">GAP</div>
                <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; font-weight: bold; font-style: italic; color: {accent}; letter-spacing: 0.02em; line-height: 1;">{gap_text}</div>
            </div>
        </div>
        
        <!-- Bottom Row: Progress Bars -->
        {bars_html}
    </div>
    """
    
    return f"""
    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800;900&display=swap" rel="stylesheet">
    <style>
        html, body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; height: 100%; }}
        .card-container {{ box-sizing: border-box; }}
    </style>
    {body}
    """

def build_circuit_map(
    track: Optional[pd.DataFrame],
    distance: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    c1: str,
    c2: str,
    title: str,
) -> go.Figure:
    """Color track by which driver is faster at each point (broadcast style)."""
    fig = go.Figure()
    if track is not None and len(track) > 10:
        x = track["X"].to_numpy(dtype=float)
        y = track["Y"].to_numpy(dtype=float)
        seg = np.sqrt(np.diff(x, prepend=x[0]) ** 2 + np.diff(y, prepend=y[0]) ** 2)
        arc = np.cumsum(seg)
        arc = arc / max(arc[-1], 1e-9) * float(distance[-1] - distance[0]) + float(distance[0])
        sp1 = interp1d(distance, s1, bounds_error=False, fill_value=(s1[0], s1[-1]))(arc)
        sp2 = interp1d(distance, s2, bounds_error=False, fill_value=(s2[0], s2[-1]))(arc)
        colors = [c1 if a >= b else c2 for a, b in zip(sp1, sp2)]

        # Batch into runs of same color for fewer traces / cleaner look
        start = 0
        for i in range(1, len(x)):
            if colors[i] != colors[start] or i == len(x) - 1:
                end = i if colors[i] != colors[start] else i + 1
                fig.add_trace(go.Scatter(
                    x=x[start:end], y=y[start:end], mode="lines",
                    line=dict(color=colors[start], width=6),
                    hoverinfo="skip", showlegend=False,
                ))
                start = i

        turns = detect_turn_distances(distance, (s1 + s2) / 2.0)
        for n, td in enumerate(turns, start=1):
            j = int(np.argmin(np.abs(arc - td)))
            fig.add_annotation(
                x=x[j], y=y[j], text=str(n), showarrow=False,
                font=dict(size=11, color="#ffffff", family="Bebas Neue, sans-serif", weight="bold"),
                bgcolor="#15161d", bordercolor="#222530", borderwidth=1, borderpad=3,
                yshift=10, xshift=10
            )

        # Place speed zone labels based on speed data
        avg_speed = (sp1 + sp2) / 2.0
        low_peaks, _ = find_peaks(-avg_speed, distance=len(avg_speed)//5, prominence=15)
        high_peaks, _ = find_peaks(avg_speed, distance=len(avg_speed)//5, prominence=15)
        
        labeled_points = []
        for idx in low_peaks[:2]:
            if avg_speed[idx] < 140:
                fig.add_annotation(
                    x=x[idx], y=y[idx], text="LOW SPEED", showarrow=False,
                    font=dict(size=8, color="#8a8d98", family="Inter, sans-serif", weight="bold"),
                    bgcolor="rgba(16,17,22,0.85)", bordercolor="#222530", borderwidth=1, borderpad=3,
                    yshift=-15, xshift=-15
                )
                labeled_points.append(idx)
                
        for idx in high_peaks[:2]:
            if avg_speed[idx] > 240:
                fig.add_annotation(
                    x=x[idx], y=y[idx], text="HIGH SPEED", showarrow=False,
                    font=dict(size=8, color="#ffffff", family="Inter, sans-serif", weight="bold"),
                    bgcolor="rgba(225,6,0,0.85)" if c1 == RED or c2 == RED else "rgba(16,17,22,0.85)", 
                    bordercolor="#222530", borderwidth=1, borderpad=3,
                    yshift=15, xshift=15
                )
                labeled_points.append(idx)
                
        if len(avg_speed) > 100:
            for idx in range(len(avg_speed)//3, 2*len(avg_speed)//3, len(avg_speed)//10):
                if idx not in labeled_points and 160 < avg_speed[idx] < 220:
                    fig.add_annotation(
                        x=x[idx], y=y[idx], text="MEDIUM SPEED", showarrow=False,
                        font=dict(size=8, color="#8a8d98", family="Inter, sans-serif", weight="bold"),
                        bgcolor="rgba(16,17,22,0.85)", bordercolor="#222530", borderwidth=1, borderpad=3,
                        yshift=15, xshift=-15
                    )
                    break
    else:
        fig.add_trace(go.Scatter(
            x=distance, y=np.zeros_like(distance), mode="lines",
            line=dict(color=c1, width=8), hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=360, margin=dict(t=40, b=16, l=16, r=16),
        title=dict(text=title, font=dict(size=11, color=MUTED, family="Inter, sans-serif", weight="bold"),
                   x=0.5, xanchor="center"),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return fig

def build_speed_delta_figure(
    distance: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    name1: str,
    name2: str,
    c1: str,
    c2: str,
    race_label: str,
) -> go.Figure:
    delta = cumulative_time_delta(distance, s1, s2)
    turns = detect_turn_distances(distance, (s1 + s2) / 2.0)
    vmax = float(np.nanmax(np.concatenate([s1, s2])))
    vmin = float(np.nanmin(np.concatenate([s1, s2])))
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
    )

    # Shaded columns and turn/speed labels for grouped turns
    grouped_turns = []
    i = 0
    while i < len(turns):
        t_start = turns[i]
        t_group = [i + 1]
        j = i + 1
        while j < len(turns) and (turns[j] - turns[j-1]) < 180:
            t_group.append(j + 1)
            j += 1
        t_end = turns[j-1] if len(t_group) > 1 else t_start
        grouped_turns.append((t_start, t_end, t_group))
        i = j

    for t_start, t_end, t_nums in grouped_turns:
        pad_start = max(0.0, t_start - 60)
        pad_end = min(float(distance[-1]), t_end + 60)
        
        fig.add_vrect(
            x0=pad_start, x1=pad_end, 
            fillcolor="rgba(255,255,255,0.015)", 
            line_width=0, row=1, col=1
        )
        
        idx_start = np.searchsorted(distance, pad_start)
        idx_end = np.searchsorted(distance, pad_end)
        avg_speed_zone = np.mean((s1[idx_start:idx_end] + s2[idx_start:idx_end]) / 2.0)
        
        if avg_speed_zone < 130:
            zone_lbl = "LOW SPEED"
            zone_color = "#8a8d98"
        elif avg_speed_zone < 220:
            zone_lbl = "MEDIUM SPEED"
            zone_color = "#a8abb6"
        else:
            zone_lbl = "HIGH SPEED"
            zone_color = "#E10600"
            
        turn_str = "TURN " + " ".join(map(str, t_nums))
        mid_x = (t_start + t_end) / 2.0
        
        fig.add_annotation(
            x=mid_x, y=vmax * 1.08, text=zone_lbl, showarrow=False,
            font=dict(size=7, color=zone_color, family="Inter, sans-serif", weight="bold"),
            yref="y", row=1, col=1, yshift=15
        )
        
        fig.add_annotation(
            x=mid_x, y=vmax * 1.08, text=turn_str, showarrow=False,
            font=dict(size=12, color="#ffffff", family="Bebas Neue, sans-serif", weight="bold"),
            yref="y", row=1, col=1
        )

    # Plot speed lines
    fig.add_trace(go.Scatter(
        x=distance, y=s1, mode="lines", name=name1,
        line=dict(color=c1, width=3.0),
        hovertemplate="%{y:.0f} km/h<extra>" + name1 + "</extra>",
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=distance, y=s2, mode="lines", name=name2,
        line=dict(color=c2, width=3.0),
        hovertemplate="%{y:.0f} km/h<extra>" + name2 + "</extra>",
    ), row=1, col=1)

    # Add vertical divider lines for turns
    for t_val in turns:
        fig.add_vline(x=t_val, line=dict(color="#222530", width=1.5, dash="dash"), row=1, col=1)
        fig.add_vline(x=t_val, line=dict(color="#222530", width=1.5, dash="dash"), row=2, col=1)

    # Horizontal zero line for delta plot
    fig.add_hline(y=0, line=dict(color="#515462", width=2.0), row=2, col=1)

    # Plot delta line (plot negative delta to show lead as positive values)
    y_delta = -delta
    fig.add_trace(go.Scatter(
        x=distance, y=y_delta, mode="lines", name="DELTA",
        line=dict(color="#ffffff", width=2.2),
        fill='tozeroy', fillcolor='rgba(255,255,255,0.02)',
        hovertemplate="%{y:.3f}s<extra>Δ</extra>",
        showlegend=False,
    ), row=2, col=1)

    # Add delta boxes at the bottom of the Speed plot
    for t_val in turns:
        idx = int(np.searchsorted(distance, t_val))
        if idx >= len(delta):
            idx = len(delta) - 1
        t_delta = delta[idx]
        d_color = c1 if t_delta <= 0 else c2
        d_text = f"{t_delta:+.3f}"
        
        fig.add_annotation(
            x=t_val, y=vmax * 0.05 + vmin * 0.95, text=d_text, showarrow=False,
            font=dict(size=10, color=d_color, family="Inter, sans-serif", weight="bold"),
            bgcolor="#101116", bordercolor="#222530", borderwidth=1, borderpad=3,
            yref="y", row=1, col=1
        )

    # Add large watermarks "FASTER" and "SLOWER" to background of Delta plot
    fig.add_annotation(
        text="FASTER", xref="paper", yref="y2", x=0.03, y=max(0.1, float(np.max(y_delta)) * 0.6) if len(y_delta) > 0 else 0.1,
        showarrow=False, font=dict(size=24, color="rgba(255,255,255,0.035)", family="Bebas Neue, sans-serif", weight="bold"),
        xanchor="left", yanchor="bottom"
    )
    fig.add_annotation(
        text="SLOWER", xref="paper", yref="y2", x=0.03, y=min(-0.1, float(np.min(y_delta)) * 0.6) if len(y_delta) > 0 else -0.1,
        showarrow=False, font=dict(size=24, color="rgba(255,255,255,0.035)", family="Bebas Neue, sans-serif", weight="bold"),
        xanchor="left", yanchor="top"
    )

    # Custom Y-axis tick formatting
    fig.update_yaxes(
        tickvals=[int(vmin), int(vmax)],
        ticktext=[f"{int(vmin)} KM/H", f"{int(vmax)} KM/H"],
        tickfont=dict(color="#ffffff", size=10, family="Bebas Neue, sans-serif", weight="bold"),
        gridcolor=GRID, zeroline=False, row=1, col=1
    )
    
    fig.update_yaxes(
        tickfont=dict(color="#8a8d98", size=9, family="Inter, sans-serif"),
        gridcolor=GRID, title_text="DELTA (s)", title_font=dict(size=9, color=MUTED, family="Inter, sans-serif"),
        zeroline=False, row=2, col=1
    )
    
    fig.update_xaxes(
        gridcolor=GRID, title_text="DISTANCE (m)", title_font=dict(size=9, color=MUTED, family="Inter, sans-serif"),
        tickfont=dict(color="#8a8d98", size=9, family="Inter, sans-serif"), row=2, col=1
    )
    fig.update_xaxes(gridcolor=GRID, showticklabels=False, row=1, col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=580, margin=dict(t=70, b=40, l=55, r=24),
        title=dict(
            text=f"{race_label.upper()}  ·  QUALIFYING HEAD-TO-HEAD SPEED & TIME DELTA",
            font=dict(size=12, color="#8a8d98", family="Inter, sans-serif", weight="bold"), x=0, xanchor="left",
        ),
        legend=dict(
            orientation="h", y=1.04, x=1, xanchor="right",
            font=dict(size=11, color="#ffffff", family="Bebas Neue, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    # Legend indicators next to delta y-axis
    fig.add_annotation(
        text="▲ FASTER", xref="paper", yref="paper", x=-0.045, y=0.18,
        showarrow=False, font=dict(size=8, color="#8a8d98", family="Inter, sans-serif", weight="bold"),
        textangle=-90
    )
    fig.add_annotation(
        text="▼ SLOWER", xref="paper", yref="paper", x=-0.045, y=0.04,
        showarrow=False, font=dict(size=8, color="#8a8d98", family="Inter, sans-serif", weight="bold"),
        textangle=-90
    )

    return fig
