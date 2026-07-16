"""Broadcast-style qualifying analysis visuals for STRAT."""
from __future__ import annotations

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
WHITE = "#FFFFFF"
BG = "#0B0B0B"
PANEL = "#111111"
MUTED = "#6B6B6B"
GRID = "#1A1A1A"


def format_laptime(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:06.3f}"


def drive_style_from_speed(speed: np.ndarray) -> Tuple[float, float, float]:
    """Approximate % full throttle / heavy brake / cornering from speed trace."""
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 5:
        return 45.0, 20.0, 35.0
    vmax = float(np.nanmax(speed)) or 1.0
    ds = np.diff(speed, prepend=speed[0])
    # share of samples in each regime (not forced to sum 100 — closer to broadcast bars)
    full_throttle = float(np.clip(np.mean(speed > 0.78 * vmax) * 100, 8, 92))
    heavy_brake = float(np.clip(np.mean(ds < -6.5) * 100 * 2.2, 5, 55))
    cornering = float(np.clip(np.mean(speed < 0.58 * vmax) * 100, 10, 70))
    return full_throttle, heavy_brake, cornering


def cumulative_time_delta(distance: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Time delta (driver1 - driver2) along the lap. Negative => driver1 ahead."""
    v1 = np.maximum(np.asarray(s1, dtype=float), 1.0) / 3.6
    v2 = np.maximum(np.asarray(s2, dtype=float), 1.0) / 3.6
    dd = np.diff(distance, prepend=distance[0])
    dd[0] = 0.0
    t1 = np.cumsum(dd / v1)
    t2 = np.cumsum(dd / v2)
    return t1 - t2


def detect_turn_distances(distance: np.ndarray, speed: np.ndarray, max_turns: int = 14) -> np.ndarray:
    speed = np.asarray(speed, dtype=float)
    distance = np.asarray(distance, dtype=float)
    if len(speed) < 30:
        return np.array([])
    inv = -speed
    peaks, _ = find_peaks(inv, distance=max(8, len(speed) // 28), prominence=12)
    if len(peaks) == 0:
        return np.array([])
    # keep deepest minima
    order = np.argsort(speed[peaks])[:max_turns]
    peaks = np.sort(peaks[order])
    return distance[peaks]


def load_track_outline(year: int, race: str) -> Optional[pd.DataFrame]:
    path = os.path.join("data", "telemetry", f"track_{year}_{race.replace(' ', '_')}.csv")
    if not os.path.exists(path):
        # try any year for same circuit
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


def speed_zone(speed: float, vmax: float) -> str:
    if speed >= 0.75 * vmax:
        return "HIGH"
    if speed >= 0.45 * vmax:
        return "MEDIUM"
    return "LOW"


def zone_color(zone: str, c_high: str, c_med: str) -> str:
    if zone == "HIGH":
        return c_high
    if zone == "MEDIUM":
        return c_med
    return "#3A3A3A"


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
    first, *rest = full_name.split(" ", 1)
    last = rest[0] if rest else ""
    text_align = "left" if align == "left" else "right"
    flex_dir = "row" if align == "left" else "row-reverse"

    def bar(label: str, pct: float, color: str) -> str:
        w = max(4.0, min(100.0, pct))
        return f"""
        <div style="margin:10px 0 0">
          <div style="display:flex;justify-content:space-between;font-size:10px;
                      letter-spacing:0.14em;color:#777;margin-bottom:4px;text-transform:uppercase">
            <span>{label}</span><span style="color:#aaa">{pct:.0f}%</span>
          </div>
          <div style="height:7px;background:#1a1a1a;border-radius:1px;overflow:hidden">
            <div style="width:{w:.1f}%;height:100%;background:{color}"></div>
          </div>
        </div>"""

    return f"""
    <div class="strat-card" style="text-align:{text_align}">
      <div style="display:flex;flex-direction:{flex_dir};align-items:flex-start;gap:14px">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:4.2rem;line-height:0.85;
                    color:{accent};letter-spacing:0.02em">{pos}</div>
        <div style="flex:1;min-width:0">
          <div style="font-size:0.7rem;letter-spacing:0.22em;color:#666;text-transform:uppercase">{first}</div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:2.1rem;letter-spacing:0.06em;
                      color:#fff;line-height:1;margin:2px 0 6px">{last.upper() or code}</div>
          <div style="font-size:0.62rem;letter-spacing:0.2em;color:#555;text-transform:uppercase">{team}</div>
        </div>
      </div>
      <div style="margin-top:18px;display:flex;justify-content:space-between;gap:12px;
                  flex-direction:{flex_dir};align-items:baseline">
        <div>
          <div style="font-size:0.55rem;letter-spacing:0.2em;color:#555">LAP TIME</div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:1.85rem;color:#fff;
                      letter-spacing:0.04em">{format_laptime(lap_sec)}</div>
        </div>
        <div>
          <div style="font-size:0.55rem;letter-spacing:0.2em;color:#555">GAP</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.95rem;color:{accent}">{gap_text}</div>
        </div>
      </div>
      {bar("Full Throttle", ft, accent)}
      {bar("Heavy Braking", hb, YELLOW if accent == RED else accent)}
      {bar("Cornering", corner, "#888")}
    </div>
    """


def build_circuit_map(
    track: Optional[pd.DataFrame],
    distance: np.ndarray,
    speed_ref: np.ndarray,
    c1: str,
    c2: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    vmax = float(np.nanmax(speed_ref)) if len(speed_ref) else 1.0

    if track is not None and len(track) > 10:
        x = track["X"].to_numpy(dtype=float)
        y = track["Y"].to_numpy(dtype=float)
        # color segments by mapped speed along arc length
        seg = np.sqrt(np.diff(x, prepend=x[0]) ** 2 + np.diff(y, prepend=y[0]) ** 2)
        arc = np.cumsum(seg)
        arc = arc / max(arc[-1], 1e-9) * float(distance[-1] - distance[0]) + float(distance[0])
        spd = interp1d(distance, speed_ref, bounds_error=False, fill_value=(speed_ref[0], speed_ref[-1]))(arc)
        colors = [zone_color(speed_zone(s, vmax), c1, c2) for s in spd]

        for i in range(1, len(x)):
            fig.add_trace(go.Scatter(
                x=x[i - 1:i + 1], y=y[i - 1:i + 1], mode="lines",
                line=dict(color=colors[i], width=5),
                hoverinfo="skip", showlegend=False,
            ))
        # turn numbers at low-speed vertices
        turns = detect_turn_distances(distance, speed_ref)
        if len(turns):
            t_xy = []
            for td in turns:
                j = int(np.argmin(np.abs(arc - td)))
                t_xy.append((x[j], y[j]))
            for n, (tx, ty) in enumerate(t_xy, start=1):
                fig.add_annotation(
                    x=tx, y=ty, text=str(n), showarrow=False,
                    font=dict(size=10, color="#ccc", family="DM Mono"),
                    bgcolor="rgba(0,0,0,0.55)", borderpad=2,
                )
    else:
        # fallback ribbon
        fig.add_trace(go.Scatter(
            x=distance, y=np.zeros_like(distance), mode="lines",
            line=dict(color=c1, width=8), hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=320, margin=dict(t=36, b=20, l=20, r=20),
        title=dict(text=title, font=dict(size=11, color=MUTED, family="DM Mono"), x=0.5, xanchor="center"),
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

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    # zone backgrounds on speed chart
    zones = [
        (0.0, 0.45 * vmax, "rgba(80,80,80,0.12)", "LOW SPEED"),
        (0.45 * vmax, 0.75 * vmax, "rgba(255,235,0,0.06)", "MEDIUM"),
        (0.75 * vmax, vmax * 1.05, "rgba(225,6,0,0.07)", "HIGH SPEED"),
    ]
    for y0, y1, fill, _ in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=distance, y=s1, mode="lines", name=name1,
        line=dict(color=c1, width=2.4),
        hovertemplate="%{y:.0f} km/h<extra>" + name1 + "</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=distance, y=s2, mode="lines", name=name2,
        line=dict(color=c2, width=2.4),
        hovertemplate="%{y:.0f} km/h<extra>" + name2 + "</extra>",
    ), row=1, col=1)

    for i, td in enumerate(turns, start=1):
        fig.add_vline(x=td, line=dict(color="#2a2a2a", width=1), row=1, col=1)
        fig.add_vline(x=td, line=dict(color="#2a2a2a", width=1), row=2, col=1)
        fig.add_annotation(
            x=td, y=vmax * 0.98, text=f"T{i}", showarrow=False,
            font=dict(size=9, color="#666", family="DM Mono"), yref="y",
        )

    # delta fill
    fig.add_trace(go.Scatter(
        x=distance, y=delta, mode="lines", name="DELTA",
        line=dict(color=c1, width=1.8),
        fill="tozeroy", fillcolor="rgba(225,6,0,0.15)",
        hovertemplate="%{y:.3f}s<extra>Δ</extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#333", width=1), row=2, col=1)

    # annotate a few delta values
    for frac in (0.2, 0.45, 0.7, 0.9):
        i = int(frac * (len(delta) - 1))
        fig.add_annotation(
            x=distance[i], y=delta[i],
            text=f"{delta[i]:+.3f}", showarrow=False,
            font=dict(size=10, color=c1, family="DM Mono"),
            yref="y2", yshift=12 if delta[i] < 0 else -12,
        )

    fig.update_yaxes(
        title_text="SPEED (km/h)", title_font=dict(size=10, color=MUTED, family="DM Mono"),
        gridcolor=GRID, tickfont=dict(color="#777", size=9), row=1, col=1,
    )
    fig.update_yaxes(
        title_text="DELTA (s)", title_font=dict(size=10, color=MUTED, family="DM Mono"),
        gridcolor=GRID, tickfont=dict(color="#777", size=9), row=2, col=1,
        zeroline=False,
    )
    fig.update_xaxes(
        title_text="DISTANCE (m)", title_font=dict(size=10, color=MUTED, family="DM Mono"),
        gridcolor=GRID, tickfont=dict(color="#777", size=9), row=2, col=1,
    )
    fig.update_xaxes(gridcolor=GRID, tickfont=dict(color="#777", size=9), row=1, col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=560, margin=dict(t=48, b=40, l=55, r=24),
        title=dict(
            text=f"{race_label.upper()}  ·  QUALIFYING ANALYSIS",
            font=dict(size=13, color="#aaa", family="DM Mono"), x=0, xanchor="left",
        ),
        legend=dict(
            orientation="h", y=1.08, x=1, xanchor="right",
            font=dict(size=11, color="#ccc", family="DM Mono"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    # FASTER / SLOWER labels on delta pane
    fig.add_annotation(
        text="FASTER", xref="paper", yref="paper", x=0.01, y=0.20,
        showarrow=False, font=dict(size=9, color="#666", family="DM Mono"),
    )
    fig.add_annotation(
        text="SLOWER", xref="paper", yref="paper", x=0.01, y=0.03,
        showarrow=False, font=dict(size=9, color="#666", family="DM Mono"),
    )
    return fig
