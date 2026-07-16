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
BG = "#0B0B0B"
MUTED = "#6B6B6B"
GRID = "#1A1A1A"

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


def _bar(label: str, pct: float, color: str) -> str:
    w = max(4.0, min(100.0, pct))
    # NO blank lines — Streamlit markdown treats blank lines as end-of-HTML
    return (
        f'<div style="margin-top:10px">'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;'
        f'letter-spacing:0.14em;color:#777;margin-bottom:4px;text-transform:uppercase;'
        f'font-family:DM Mono,monospace">'
        f'<span>{label}</span><span style="color:#bbb">{pct:.0f}%</span></div>'
        f'<div style="height:8px;background:#1a1a1a;overflow:hidden">'
        f'<div style="width:{w:.1f}%;height:100%;background:{color}"></div></div></div>'
    )


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
    text_align = "left" if align == "left" else "right"
    flex_dir = "row" if align == "left" else "row-reverse"
    body = (
        f'<div style="background:#111;border:1px solid #1c1c1c;padding:1.25rem 1.3rem 1.35rem;'
        f'text-align:{text_align};height:100%;box-sizing:border-box;font-family:DM Mono,monospace">'
        f'<div style="display:flex;flex-direction:{flex_dir};align-items:flex-start;gap:14px">'
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:4.2rem;line-height:0.85;'
        f'color:{accent};letter-spacing:0.02em">{pos}</div>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-size:0.7rem;letter-spacing:0.22em;color:#666;text-transform:uppercase">{first}</div>'
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:2.1rem;letter-spacing:0.06em;'
        f'color:#fff;line-height:1;margin:2px 0 6px">{last.upper()}</div>'
        f'<div style="font-size:0.62rem;letter-spacing:0.2em;color:#555;text-transform:uppercase">{team}</div></div></div>'
        f'<div style="margin-top:16px;display:flex;justify-content:space-between;gap:12px;'
        f'flex-direction:{flex_dir};align-items:baseline">'
        f'<div><div style="font-size:0.55rem;letter-spacing:0.2em;color:#555">LAP TIME</div>'
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:1.85rem;color:#fff;'
        f'letter-spacing:0.04em">{format_laptime(lap_sec)}</div></div>'
        f'<div><div style="font-size:0.55rem;letter-spacing:0.2em;color:#555">GAP</div>'
        f'<div style="font-family:DM Mono,monospace;font-size:0.95rem;color:{accent};'
        f'font-weight:500">{gap_text}</div></div></div>'
        f'{_bar("Full Throttle", ft, accent)}'
        f'{_bar("Heavy Braking", hb, accent)}'
        f'{_bar("Cornering", corner, "#888")}'
        f'</div>'
    )
    return (
        '<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">'
        '<style>html,body{margin:0;padding:0;background:transparent;overflow:hidden}</style>'
        + body
    )


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
                font=dict(size=11, color="#ddd", family="DM Mono, monospace"),
                bgcolor="rgba(0,0,0,0.65)", borderpad=3,
            )
    else:
        fig.add_trace(go.Scatter(
            x=distance, y=np.zeros_like(distance), mode="lines",
            line=dict(color=c1, width=8), hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=340, margin=dict(t=40, b=16, l=16, r=16),
        title=dict(text=title, font=dict(size=11, color=MUTED, family="DM Mono, monospace"),
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
    mono = "DM Mono, Courier New, monospace"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30],
        vertical_spacing=0.05,
    )

    # subtle zone washes
    fig.add_hrect(y0=0, y1=0.45 * vmax, fillcolor="rgba(80,80,80,0.10)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=0.45 * vmax, y1=0.75 * vmax, fillcolor="rgba(255,235,0,0.05)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=0.75 * vmax, y1=vmax * 1.05, fillcolor="rgba(225,6,0,0.06)", line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=distance, y=s1, mode="lines", name=name1,
        line=dict(color=c1, width=2.6),
        hovertemplate="%{y:.0f} km/h<extra>" + name1 + "</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=distance, y=s2, mode="lines", name=name2,
        line=dict(color=c2, width=2.6),
        hovertemplate="%{y:.0f} km/h<extra>" + name2 + "</extra>",
    ), row=1, col=1)

    for i, td in enumerate(turns, start=1):
        fig.add_vline(x=td, line=dict(color="#252525", width=1), row=1, col=1)
        fig.add_vline(x=td, line=dict(color="#252525", width=1), row=2, col=1)
        fig.add_annotation(
            x=td, y=vmax * 0.97, text=f"T{i}", showarrow=False,
            font=dict(size=9, color="#777", family=mono), yref="y",
        )

    # Delta line colored by who is ahead: use two fills
    fig.add_trace(go.Scatter(
        x=distance, y=delta, mode="lines", name="DELTA",
        line=dict(color="#ddd", width=1.6),
        hovertemplate="%{y:.3f}s<extra>Δ</extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#444", width=1), row=2, col=1)

    # Floating delta numbers in mono — color by which driver is ahead
    # (negative delta => driver1 faster => c1)
    n_labels = min(11, max(6, len(turns) + 2))
    for frac in np.linspace(0.08, 0.95, n_labels):
        i = int(frac * (len(delta) - 1))
        val = float(delta[i])
        col = c1 if val <= 0 else c2
        fig.add_annotation(
            x=float(distance[i]),
            y=float(val),
            text=f"{val:+.3f}",
            showarrow=False,
            font=dict(size=12, color=col, family=mono),
            yref="y2",
            yshift=14 if val <= 0 else -14,
        )

    fig.update_yaxes(
        title_text="SPEED", title_font=dict(size=10, color=MUTED, family=mono),
        gridcolor=GRID, tickfont=dict(color="#777", size=9, family=mono), row=1, col=1,
    )
    fig.update_yaxes(
        title_text="DELTA", title_font=dict(size=10, color=MUTED, family=mono),
        gridcolor=GRID, tickfont=dict(color="#777", size=9, family=mono), row=2, col=1,
    )
    fig.update_xaxes(
        title_text="DISTANCE (m)", title_font=dict(size=10, color=MUTED, family=mono),
        gridcolor=GRID, tickfont=dict(color="#777", size=9, family=mono), row=2, col=1,
    )
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color="#777", size=9, family=mono), row=1, col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=580, margin=dict(t=52, b=42, l=55, r=24),
        title=dict(
            text=f"{race_label.upper()}  ·  QUALIFYING ANALYSIS",
            font=dict(size=13, color="#aaa", family=mono), x=0, xanchor="left",
        ),
        legend=dict(
            orientation="h", y=1.10, x=1, xanchor="right",
            font=dict(size=11, color="#ccc", family=mono),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        font=dict(family=mono),
    )
    fig.add_annotation(
        text="FASTER", xref="paper", yref="paper", x=0.012, y=0.22,
        showarrow=False, font=dict(size=9, color="#666", family=mono),
    )
    fig.add_annotation(
        text="SLOWER", xref="paper", yref="paper", x=0.012, y=0.03,
        showarrow=False, font=dict(size=9, color="#666", family=mono),
    )
    return fig
