"""
F1 Race Strategy Analyzer — Streamlit Web App
------------------------------------------------
Interactive version of the original analysis script. Pick a year/race/session/
drivers from dropdowns in a browser and get the analysis instantly — no code
editing, no re-running scripts.

HOW TO RUN LOCALLY
1. pip install streamlit fastf1 pandas matplotlib requests
2. streamlit run app.py   (or: python -m streamlit run app.py)

HOW TO DEPLOY (FREE, public link)
1. Push this file + requirements.txt to your GitHub repo
2. Go to share.streamlit.io, sign in with GitHub
3. Click "New app", select this repo, set main file to app.py, click Deploy
4. You get a public URL like: https://your-app-name.streamlit.app

NOTES ON IMAGES
- Circuit photos are fetched live from Wikipedia's public API at runtime
  (not hardcoded), with a graceful fallback if a page/image isn't found.
- Driver cards use real team colors pulled from FastF1 itself, not team
  logos — this avoids using trademarked F1 team/sponsor imagery while
  still looking sharp and being instantly recognizable by color.
"""

import itertools
import os
import struct
import time

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="F1 Race Strategy Analyzer", layout="wide", page_icon="🏎️")


def get_mp4_duration_seconds(path: str) -> float | None:
    """Reads an MP4's moov/mvhd atom directly to get its duration, with no
    external dependencies (no ffmpeg needed). Returns None if it can't be read.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()

        def find_atom(buf, atom_type, start=0, end=None):
            end = end if end is not None else len(buf)
            pos = start
            while pos < end - 8:
                size = struct.unpack(">I", buf[pos:pos + 4])[0]
                atype = buf[pos + 4:pos + 8]
                if size == 0:
                    size = end - pos
                if atype == atom_type:
                    return pos, size
                pos += size if size >= 8 else 8
            return None, None

        moov_pos, moov_size = find_atom(data, b"moov")
        if moov_pos is None:
            return None
        mvhd_pos, mvhd_size = find_atom(data, b"mvhd", moov_pos + 8, moov_pos + moov_size)
        if mvhd_pos is None:
            return None

        version = data[mvhd_pos + 8]
        if version == 1:
            timescale = struct.unpack(">I", data[mvhd_pos + 28:mvhd_pos + 32])[0]
            duration = struct.unpack(">Q", data[mvhd_pos + 32:mvhd_pos + 40])[0]
        else:
            timescale = struct.unpack(">I", data[mvhd_pos + 20:mvhd_pos + 24])[0]
            duration = struct.unpack(">I", data[mvhd_pos + 24:mvhd_pos + 28])[0]

        return duration / timescale if timescale else None
    except Exception:
        return None


# ── Intro video gate ─────────────────────────────────────────────────────
# Shows a true full-screen video the moment the site loads — no title text,
# no controls, no button. The site advances automatically once the video's
# actual runtime has elapsed, timed from the Python side.
#
# Why not just detect "video ended" in JavaScript? Browsers block a video's
# "ended" event from navigating the top-level page unless a real user click
# triggered it — an automatic event doesn't count, for security reasons.
# So instead of fighting that restriction, the app itself waits exactly as
# long as the video runs, then moves on — no browser permission needed.
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

# ── Intro video gate ─────────────────────────────────────────────────────
# Shows a true full-screen video the moment the site loads — no title text,
# no controls, no button. The site advances automatically once the video's
# actual runtime has elapsed, timed from the Python side.
#
# The video is served via Streamlit's built-in static file serving (a real
# HTTP request the browser streams normally) rather than being embedded as
# base64 text in the page. Base64-embedding a large video bloats the page
# payload substantially and delays playback start — which is what caused
# the video to get cut off before finishing. A normal streamed file starts
# playing almost immediately, so the timing below stays accurate.
#
# Why not just detect "video ended" in JavaScript? Browsers block a video's
# "ended" event from navigating the top-level page unless a real user click
# triggered it — an automatic event doesn't count, for security reasons.
# So instead of fighting that restriction, the app itself waits exactly as
# long as the video runs, then moves on — no browser permission needed.
#
# SETUP REQUIRED: place intro.mp4 inside a folder named "static" next to
# this app.py file (i.e. static/intro.mp4), and make sure the accompanying
# .streamlit/config.toml has "enableStaticServing = true" under [server]
# — both are provided alongside this script.
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    intro_path = os.path.join("static", "intro.mp4")
    if os.path.exists(intro_path):
        duration = get_mp4_duration_seconds(intro_path) or 6.0  # fallback if unreadable

        components.html(
            f"""
            <style>
                html, body {{ margin: 0; padding: 0; overflow: hidden; background: #0a0a0f; }}
                #introVideo {{
                    width: 100vw; height: 100vh; object-fit: cover; display: block;
                    animation: fadeToDark 0.7s ease-in forwards;
                    animation-delay: {max(duration - 0.7, 0):.2f}s;
                }}
                @keyframes fadeToDark {{
                    from {{ opacity: 1; }}
                    to {{ opacity: 0; }}
                }}
            </style>
            <video id="introVideo" autoplay muted playsinline preload="auto">
                <source src="app/static/intro.mp4" type="video/mp4">
            </video>
            <script>
                const frame = window.frameElement;
                if (frame) {{
                    frame.style.position = "fixed";
                    frame.style.top = "0";
                    frame.style.left = "0";
                    frame.style.width = "100vw";
                    frame.style.height = "100vh";
                    frame.style.zIndex = "999999";
                    frame.style.border = "none";
                    frame.style.backgroundColor = "#0a0a0f";
                }}
            </script>
            """,
            height=0,
        )

        # Wait exactly as long as the video plays, then move to the dashboard.
        # A small buffer accounts for the brief moment the browser takes to
        # start playback after the component first renders.
        time.sleep(duration + 0.7)
        st.session_state.intro_done = True
        st.rerun()

    else:
        st.warning(
            "static/intro.mp4 not found — create a folder named 'static' next to "
            "app.py and place your intro video inside it, named exactly 'intro.mp4'."
        )
        if st.button("🏁 Enter Site", type="primary"):
            st.session_state.intro_done = True
            st.rerun()

    st.stop()  # halts execution here — nothing below runs until intro finishes


os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

COMPOUND_COLORS = {
    "SOFT": "#DA291C",
    "MEDIUM": "#FFD12E",
    "HARD": "#F0F0F0",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
}

# ── F1-themed styling ───────────────────────────────────────────────────────
# Background: an original abstract circuit-outline graphic (not based on any
# real track's actual layout, not any team/brand artwork) — drawn from
# scratch as a low-opacity SVG watermark so it adds atmosphere without
# competing with the dashboard content on top of it.
CIRCUIT_BG_B64 = (
    "PHN2ZyB2aWV3Qm94PSIwIDAgMTYwMCA5MDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2"
    "ZyI+CiAgPHBhdGggZD0iTSAyNjAsNTYwIAogICAgICAgICAgIEMgMjYwLDQyMCAzODAsMzQwIDUyMCwz"
    "NTAgCiAgICAgICAgICAgQyA2NDAsMzU4IDY2MCwyODAgNzgwLDI3MCAKICAgICAgICAgICBDIDkyMCwy"
    "NTggOTYwLDM2MCA5MDAsNDIwIAogICAgICAgICAgIEMgODUwLDQ3MCA5NzAsNTAwIDEwODAsNDcwIAog"
    "ICAgICAgICAgIEMgMTIzMCw0MzAgMTM0MCw1NDAgMTI5MCw2MzAgCiAgICAgICAgICAgQyAxMjUwLDcw"
    "MCAxMTQwLDY2MCAxMDgwLDcwMCAKICAgICAgICAgICBDIDEwMTAsNzQ1IDkyMCw3MjAgODgwLDY1MCAK"
    "ICAgICAgICAgICBDIDg0MCw1ODAgNzAwLDYxMCA2NTAsNjkwIAogICAgICAgICAgIEMgNjAwLDc3MCA0"
    "NjAsNzUwIDQxMCw2ODAgCiAgICAgICAgICAgQyAzNzAsNjI1IDI2MCw2NTAgMjYwLDU2MCBaIgogICAg"
    "ICAgIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2UxMDYwMCIgc3Ryb2tlLW9wYWNpdHk9IjAuMDciCiAgICAg"
    "ICAgc3Ryb2tlLXdpZHRoPSIxMCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49"
    "InJvdW5kIi8+CiAgPHBhdGggZD0iTSA1MjAsMzUwIEwgNTIwLDMyMCIgc3Ryb2tlPSIjZmZmZmZmIiBz"
    "dHJva2Utb3BhY2l0eT0iMC4xMCIgc3Ryb2tlLXdpZHRoPSI2Ii8+CiAgPHBhdGggZD0iTSA1MDAsMzM1"
    "IEwgNTQwLDMzNSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utb3BhY2l0eT0iMC4xMCIgc3Ryb2tlLXdp"
    "ZHRoPSI0Ii8+Cjwvc3ZnPg=="
)

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@600;700;900&display=swap" rel="stylesheet">
<style>
.stApp {{
    background-color: #0a0a0f;
    background-image: url("data:image/svg+xml;base64,{CIRCUIT_BG_B64}");
    background-repeat: no-repeat;
    background-position: center 60px;
    background-size: min(1400px, 95vw) auto;
    background-attachment: fixed;
}}
h1, h2, h3 {{ font-family: 'Titillium Web', 'Trebuchet MS', sans-serif; letter-spacing: 0.5px; }}
h1 {{
    color: #ffffff !important; font-weight: 900 !important; font-style: italic;
    text-transform: uppercase; letter-spacing: 1px;
}}
h1 .accent {{ color: #e10600; }}
.circuit-banner {{
    width: 100%; border-radius: 12px; overflow: hidden; margin-bottom: 1rem;
}}
.driver-card {{
    border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
    color: white; font-family: 'Trebuchet MS', sans-serif;
}}
.driver-card .code {{ font-size: 1.4rem; font-weight: 800; letter-spacing: 1px; }}
.driver-card .team {{ font-size: 0.8rem; opacity: 0.9; }}
[data-testid="stMetricValue"] {{ color: #e10600; }}
</style>
""", unsafe_allow_html=True)


# ── Data functions (cached so repeat visits/re-plots don't re-download) ────

@st.cache_resource(show_spinner=False)
def load_session(year: int, grand_prix: str, session_type: str):
    """Cached as a resource (not serialized data) since Session objects
    are needed later to look up official team colors."""
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load(telemetry=False, weather=False)
    return session


def get_driver_team_map(session) -> dict:
    """driver code -> team name, used for coloring and grouping."""
    laps = session.laps
    return laps.drop_duplicates("Driver").set_index("Driver")["Team"].to_dict()


def get_team_color(team_name: str, session) -> str:
    try:
        return "#" + fastf1.plotting.get_team_color(team_name, session).lstrip("#")
    except Exception:
        return "#e10600"  # neutral F1-red fallback if a team isn't recognized


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_circuit_image(grand_prix: str) -> str | None:
    """Look up a circuit photo from Wikipedia's public API at runtime.
    Returns an image URL, or None if nothing suitable was found — caller
    should handle the None case gracefully rather than assuming an image exists.
    """
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "format": "json", "prop": "pageimages",
            "piprop": "original", "generator": "search",
            "gsrsearch": f"{grand_prix} Grand Prix circuit", "gsrlimit": 1,
        }
        resp = requests.get(search_url, params=params, timeout=5)
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            original = page.get("original", {})
            if "source" in original:
                return original["source"]
    except Exception:
        pass
    return None


def clean_laps(laps: pd.DataFrame, drivers: list) -> pd.DataFrame:
    clean = laps[laps["Driver"].isin(drivers)].copy()
    clean = clean[clean["PitOutTime"].isna() & clean["PitInTime"].isna()]
    clean = clean[clean["TrackStatus"] == "1"]
    clean = clean.dropna(subset=["LapTime", "TyreLife", "Compound"])
    clean["LapTimeSeconds"] = clean["LapTime"].dt.total_seconds()
    return clean


def build_degradation_table(clean_laps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = clean_laps.groupby(["Driver", "Compound", "Stint"])
    for (driver, compound, stint), stint_laps in grouped:
        if len(stint_laps) < 4:
            continue
        slope = stint_laps["LapTimeSeconds"].corr(stint_laps["TyreLife"]) * (
            stint_laps["LapTimeSeconds"].std() / stint_laps["TyreLife"].std()
        )
        rows.append({
            "Driver": driver, "Compound": compound, "Stint": stint,
            "LapsOnStint": len(stint_laps),
            "AvgLapTime": round(stint_laps["LapTimeSeconds"].mean(), 3),
            "DegradationSecPerLap": round(slope, 4),
        })
    if not rows:
        return pd.DataFrame(columns=["Driver", "Compound", "Stint", "LapsOnStint",
                                      "AvgLapTime", "DegradationSecPerLap"])
    return pd.DataFrame(rows).sort_values(["Driver", "Stint"])


def fit_compound_models(driver_laps: pd.DataFrame) -> dict:
    """Fits lap_time ≈ base + slope × tyre_age per compound via linear
    regression on the driver's actual laps. Returns {compound: (base, slope)}.
    Skips compounds with too few laps to fit reliably (needs >= 4), and skips
    the first lap of each stint (age 1) since out-lap effects add noise.
    """
    models = {}
    for compound, grp in driver_laps.groupby("Compound"):
        grp = grp[grp["TyreLife"] >= 2]
        if len(grp) < 4:
            continue
        slope, intercept = np.polyfit(grp["TyreLife"], grp["LapTimeSeconds"], 1)
        models[compound] = (intercept, slope)
    return models


def _stint_time(compound: str, length: int, models: dict) -> float | None:
    """Closed-form predicted total time for a stint: sum_{age=1}^{L} (base + slope*age)."""
    if compound not in models or length <= 0:
        return None
    base, slope = models[compound]
    return length * base + slope * length * (length + 1) / 2


def search_best_strategy(total_laps: int, models: dict, max_observed: dict,
                          n_stops: int, pit_loss: float, min_stint: int = 5,
                          extrapolation_factor: float = 1.2):
    """Brute-force search over pit-stop lap(s) and compound assignment for a
    fixed number of stops, minimizing total predicted race time.

    Deliberately refuses to recommend running a compound longer than
    (observed max stint length x extrapolation_factor) — a linear fit
    extrapolated far past the laps it was actually trained on becomes
    unreliable, and real tyres don't degrade linearly forever anyway.
    Returns (total_time, [(compound, length), ...]) or None if nothing
    within those bounds can cover the full race distance.
    """
    n_stints = n_stops + 1
    compounds = list(models.keys())
    if not compounds:
        return None
    best = None
    for cuts in itertools.combinations(range(min_stint, total_laps - min_stint + 1), n_stops):
        boundaries = [0] + list(cuts) + [total_laps]
        lengths = [boundaries[i + 1] - boundaries[i] for i in range(n_stints)]
        if any(l < min_stint for l in lengths):
            continue
        for combo in itertools.product(compounds, repeat=n_stints):
            if any(l > max_observed.get(c, 0) * extrapolation_factor for c, l in zip(combo, lengths)):
                continue
            total = pit_loss * n_stops
            ok = True
            for c, l in zip(combo, lengths):
                t = _stint_time(c, l, models)
                if t is None:
                    ok = False
                    break
                total += t
            if not ok:
                continue
            if best is None or total < best[0]:
                best = (total, list(zip(combo, lengths)))
    return best


def analyze_driver_strategy(driver_laps: pd.DataFrame, total_laps: int, pit_loss: float) -> dict | None:
    """Full analysis for one driver: fits degradation models, replays their
    actual strategy through the same model for a fair comparison, then
    searches 1/2/3-stop alternatives within realistic tyre-life bounds.
    Returns a dict of results, or None if there isn't enough clean data
    to fit reliable models (e.g., a driver with only a handful of laps).
    """
    models = fit_compound_models(driver_laps)
    if not models:
        return None

    max_observed = driver_laps.groupby("Compound")["TyreLife"].max().to_dict()

    actual_stints = (
        driver_laps.groupby(["Stint", "Compound"])["TyreLife"].max()
        .reset_index().sort_values("Stint")
    )
    actual_stops = max(len(actual_stints) - 1, 0)
    actual_time = pit_loss * actual_stops
    for _, row in actual_stints.iterrows():
        t = _stint_time(row["Compound"], int(row["TyreLife"]), models)
        if t is None:
            return None  # can't fairly compare if a compound they ran has no fitted model
        actual_time += t

    alternatives = {}
    for k in [1, 2, 3]:
        result = search_best_strategy(total_laps, models, max_observed, k, pit_loss)
        if result:
            alternatives[k] = result

    return {
        "models": models,
        "actual_stops": actual_stops,
        "actual_plan": list(actual_stints[["Compound", "TyreLife"]].itertuples(index=False, name=None)),
        "actual_time": actual_time,
        "alternatives": alternatives,
    }


# ── Plotting functions (dark-theme matplotlib to match the page) ───────────

plt.style.use("dark_background")


COMPOUND_LINESTYLES = {
    "SOFT": "-",
    "MEDIUM": "--",
    "HARD": ":",
    "INTERMEDIATE": "-.",
    "WET": (0, (1, 1)),
}
TEAMMATE_MARKERS = ["o", "s", "^", "D", "v", "P"]


def fig_degradation(clean_laps: pd.DataFrame, year, grand_prix, team_colors, x_mode="tyre_age"):
    """Interactive Plotly chart instead of a static image. With many drivers
    selected, a static chart just piles up overlapping lines with no way to
    declutter it. Here, each driver gets one legend entry (click to hide/show
    their line, double-click to isolate it), compounds are distinguished by
    line style, and hovering shows exact lap time / tyre age values —
    genuinely usable even with 15+ drivers selected at once.

    x_mode="tyre_age": x-axis resets to 1 at the start of every stint — the
        standard way to compare degradation, since stints start at different
        points in the race. This naturally tops out around stint length
        (often 20-30 laps), not the full race distance — that's expected,
        not missing data.
    x_mode="lap_number": x-axis is the actual race lap (1 through the full
        race distance), so you can see the whole race timeline. Each stint
        is plotted as its own line segment (grouped by Stint) so the plot
        doesn't draw a false connecting line across a pit stop gap.
    """
    dash_map = {"SOFT": "solid", "MEDIUM": "solid", "HARD": "solid",
                "INTERMEDIATE": "dot", "WET": "dash"}
    # Distinguish compounds within a driver's color using marker symbols too,
    # since teammates already share a team color.
    symbol_map = {"SOFT": "circle", "MEDIUM": "square", "HARD": "diamond",
                  "INTERMEDIATE": "triangle-up", "WET": "x"}

    x_col = "TyreLife" if x_mode == "tyre_age" else "LapNumber"
    group_cols = ["Driver", "Compound"] if x_mode == "tyre_age" else ["Driver", "Stint", "Compound"]

    fig = go.Figure()
    for driver in sorted(clean_laps["Driver"].unique()):
        driver_data = clean_laps[clean_laps["Driver"] == driver]
        color = team_colors.get(driver, "#999999")
        for i, (key, grp) in enumerate(driver_data.groupby(group_cols)):
            compound = key[-1]  # last element of the groupby key is always Compound
            grp = grp.sort_values(x_col)
            fig.add_trace(go.Scatter(
                x=grp[x_col], y=grp["LapTimeSeconds"],
                mode="lines+markers",
                name=driver,
                legendgroup=driver,
                showlegend=(i == 0),  # one legend entry per driver, not per driver+compound+stint
                line=dict(color=color, dash=dash_map.get(compound, "solid"), width=2),
                marker=dict(symbol=symbol_map.get(compound, "circle"), size=6),
                hovertemplate=(
                    f"<b>{driver}</b> ({compound})<br>"
                    + ("Tyre age: %{x} laps<br>" if x_mode == "tyre_age" else "Lap: %{x}<br>")
                    + "Lap time: %{y:.2f}s<extra></extra>"
                ),
            ))

    # Mark pit stops with a dashed vertical line + a hoverable tick explaining
    # the gap, so it reads as "pit stop happened here" instead of looking like
    # missing data. Only meaningful in lap_number mode — in tyre_age mode every
    # stint already starts at x=1, so there's no gap to annotate.
    if x_mode == "lap_number" and not clean_laps.empty:
        y_min = clean_laps["LapTimeSeconds"].min() - 0.4
        y_max = clean_laps["LapTimeSeconds"].max() + 0.4
        for driver, driver_data in clean_laps.groupby("Driver"):
            color = team_colors.get(driver, "#999999")
            stints_sorted = sorted(driver_data["Stint"].unique())
            for s_prev, s_next in zip(stints_sorted[:-1], stints_sorted[1:]):
                prev_data = driver_data[driver_data["Stint"] == s_prev]
                next_data = driver_data[driver_data["Stint"] == s_next]
                if prev_data.empty or next_data.empty:
                    continue
                lap_before = prev_data["LapNumber"].max()
                lap_after = next_data["LapNumber"].min()
                mid_x = (lap_before + lap_after) / 2

                # Full-height dashed line marking the stop
                fig.add_trace(go.Scatter(
                    x=[mid_x, mid_x], y=[y_min, y_max],
                    mode="lines", line=dict(color=color, dash="dot", width=1),
                    opacity=0.35, legendgroup=driver, showlegend=False, hoverinfo="skip",
                ))
                # Hoverable tick explaining why the gap exists
                fig.add_trace(go.Scatter(
                    x=[mid_x], y=[(y_min + y_max) / 2],
                    mode="markers",
                    marker=dict(color=color, size=9, symbol="line-ns", line=dict(width=2, color=color)),
                    opacity=0.7, legendgroup=driver, showlegend=False,
                    hovertemplate=(
                        f"<b>{driver}</b> pit stop — lap {int(lap_before)} → {int(lap_after)}<br>"
                        "In-lap &amp; out-lap excluded (pit-lane time skews pace)<extra></extra>"
                    ),
                ))

    x_title = "Tyre Age (laps)" if x_mode == "tyre_age" else "Race Lap Number"
    subtitle = "Lap Time vs Tyre Age" if x_mode == "tyre_age" else "Lap Time Across the Full Race"

    fig.update_layout(
        title=f"{year} {grand_prix} GP — {subtitle}",
        xaxis_title=x_title,
        yaxis_title="Lap Time (seconds)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,28,0.6)",
        legend=dict(title="Driver (click to toggle)", font=dict(size=10)),
        height=560,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def fig_strategy_timeline(clean_laps: pd.DataFrame, year, grand_prix):
    stints = (
        clean_laps.groupby(["Driver", "Stint", "Compound"])["LapNumber"]
        .agg(["min", "max"]).reset_index().sort_values(["Driver", "Stint"])
    )
    drivers = stints["Driver"].unique()
    fig, ax = plt.subplots(figsize=(10, max(3, len(drivers) * 0.6)))
    for i, driver in enumerate(drivers):
        for _, row in stints[stints["Driver"] == driver].iterrows():
            color = COMPOUND_COLORS.get(row["Compound"], "#999999")
            ax.barh(i, row["max"] - row["min"] + 1, left=row["min"], color=color,
                    edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers)
    ax.set_xlabel("Lap Number")
    ax.set_title(f"{year} {grand_prix} GP — Tyre Strategy Timeline")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black") for c in COMPOUND_COLORS.values()]
    ax.legend(handles, COMPOUND_COLORS.keys(), fontsize=8, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


def fig_average_pace(clean_laps: pd.DataFrame, year, grand_prix, team_colors):
    avg_pace = clean_laps.groupby("Driver")["LapTimeSeconds"].mean().sort_values()
    colors = [team_colors.get(d, "#e10600") for d in avg_pace.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(avg_pace.index, avg_pace.values, color=colors)
    ax.set_ylabel("Average Lap Time (seconds)")
    ax.set_title(f"{year} {grand_prix} GP — Average Race Pace by Driver")
    ax.set_ylim(avg_pace.min() - 0.5, avg_pace.max() + 0.5)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, avg_pace.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                ha="center", fontsize=8, color="white")
    fig.tight_layout()
    return fig


# ── UI ───────────────────────────────────────────────────────────────────

# One unified overlay covering the whole page right as the dashboard begins
# rendering. It sits above every element while Streamlit streams them in
# underneath (so nothing pops in individually), then fades away as a single
# clean motion — picking up visually right where the video's fade-to-dark
# left off, since both use the same background color.
st.markdown("""
<div id="transitionOverlay" style="
    position: fixed; inset: 0; background: #0a0a0f; z-index: 999998;
    pointer-events: none; animation: overlayFadeOut 0.8s ease-out 0.15s forwards;
"></div>
<style>
@keyframes overlayFadeOut {
    from { opacity: 1; }
    to { opacity: 0; visibility: hidden; }
}
</style>
""", unsafe_allow_html=True)

# ── Title, using the Titillium Web racing font imported above ──────────────
st.markdown(
    '<h1>🏁 <span class="accent">F1</span> Race Strategy &amp; Tyre Degradation Analyzer</h1>',
    unsafe_allow_html=True,
)
st.caption("Real F1 telemetry data via FastF1. Pick a race below — no code editing needed.")

with st.sidebar:
    st.header("Race Selection")
    year = st.selectbox("Year", options=list(range(2026, 2017, -1)), index=0)
    grand_prix = st.text_input("Grand Prix", value="Bahrain",
                                help='e.g. "Monaco", "Singapore", "Silverstone"')
    session_type = st.selectbox(
        "Session", options=["R", "Q", "FP1", "FP2", "FP3"],
        format_func=lambda s: {"R": "Race", "Q": "Qualifying", "FP1": "Practice 1",
                                "FP2": "Practice 2", "FP3": "Practice 3"}[s],
    )
    load_clicked = st.button("🏁 Load Race", type="primary", use_container_width=True)
    st.caption(
        "Note: for 2026, only races already held this season will have data available."
    )

if "laps_loaded" not in st.session_state:
    st.session_state.laps_loaded = False

if load_clicked:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            f"<p style='text-align:center; color:#ccc; margin-bottom:0.5rem;'>"
            f"Loading {year} {grand_prix} {session_type} data... "
            f"(first load takes a minute)</p>",
            unsafe_allow_html=True,
        )
        # Loops for as long as the real data fetch takes — unlike the intro video,
        # this has no fixed duration, so it just keeps playing until the try/except
        # block below finishes and the placeholder is cleared.
        components.html(
            """
            <div style="display:flex; justify-content:center;">
                <video autoplay muted loop playsinline
                       style="width:100%; max-width:420px; border-radius:12px;
                              box-shadow:0 6px 24px rgba(225,6,0,0.25);">
                    <source src="app/static/loading.mp4" type="video/mp4">
                </video>
            </div>
            """,
            height=260,
        )

    try:
        session_obj = load_session(year, grand_prix, session_type)
        st.session_state.session_obj = session_obj
        st.session_state.raw_laps = session_obj.laps
        st.session_state.team_map = get_driver_team_map(session_obj)
        st.session_state.available_drivers = sorted(st.session_state.team_map.keys())
        st.session_state.circuit_image = fetch_circuit_image(grand_prix)
        st.session_state.laps_loaded = True
        st.session_state.year, st.session_state.grand_prix = year, grand_prix
    except Exception as e:
        st.session_state.laps_loaded = False
        st.error(f"Couldn't load that session. Check the Grand Prix name and try again. ({e})")

    loading_placeholder.empty()  # remove the loading video now that fetching is done

if st.session_state.laps_loaded:
    # Circuit banner image (fetched live — falls back to nothing if unavailable)
    if st.session_state.get("circuit_image"):
        st.image(st.session_state.circuit_image, use_container_width=True,
                  caption=f"{st.session_state.grand_prix} Grand Prix — image via Wikipedia")

    st.subheader(f"{st.session_state.year} {st.session_state.grand_prix} GP")

    drivers = st.multiselect(
        "Compare drivers",
        options=st.session_state.available_drivers,
        default=st.session_state.available_drivers[:3],
    )

    if not drivers:
        st.info("Pick at least one driver to see the analysis.")
    else:
        # Build team colors for the selected drivers
        team_colors = {}
        for d in drivers:
            team = st.session_state.team_map.get(d, "")
            team_colors[d] = get_team_color(team, st.session_state.session_obj)

        # Driver cards row
        cols = st.columns(len(drivers))
        for col, d in zip(cols, drivers):
            team = st.session_state.team_map.get(d, "Unknown Team")
            color = team_colors[d]
            col.markdown(
                f"""<div class="driver-card" style="background:linear-gradient(135deg,{color}dd,{color}55);">
                <div class="code">{d}</div><div class="team">{team}</div></div>""",
                unsafe_allow_html=True,
            )

        clean = clean_laps(st.session_state.raw_laps, drivers)

        if clean.empty:
            st.warning("No clean racing laps found for this selection — try different drivers or a race session.")
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📉 Tyre Degradation", "🔧 Strategy Timeline", "⏱️ Average Pace",
                 "📋 Data Table", "🏆 Strategy Verdict"]
            )
            with tab1:
                x_mode_choice = st.radio(
                    "View by",
                    options=["Tyre Age (per stint)", "Full Race (lap number)"],
                    horizontal=True,
                    help=(
                        "Tyre Age resets to 1 every pit stop — best for comparing "
                        "degradation. Full Race shows every lap across the whole "
                        "race distance."
                    ),
                )
                x_mode = "tyre_age" if x_mode_choice.startswith("Tyre Age") else "lap_number"
                st.plotly_chart(
                    fig_degradation(clean, st.session_state.year, st.session_state.grand_prix,
                                     team_colors, x_mode=x_mode),
                    use_container_width=True,
                )
            with tab2:
                st.pyplot(fig_strategy_timeline(clean, st.session_state.year, st.session_state.grand_prix))
            with tab3:
                st.pyplot(fig_average_pace(clean, st.session_state.year, st.session_state.grand_prix, team_colors))
            with tab4:
                deg_table = build_degradation_table(clean)
                st.dataframe(deg_table, use_container_width=True)
                st.download_button(
                    "Download degradation summary (CSV)",
                    deg_table.to_csv(index=False),
                    file_name=f"tyre_degradation_summary_{grand_prix.lower()}_{year}.csv",
                )
            with tab5:
                st.caption(
                    "Fits each driver's actual lap times to a degradation curve per "
                    "compound, then searches alternate pit-stop counts and timings "
                    "to see what the data suggests could have been faster."
                )
                with st.expander("⚙️ Assumptions used in this simulation"):
                    pit_loss = st.slider(
                        "Assumed pit-stop time loss (seconds)", 15.0, 30.0, 22.0, 0.5,
                        help="Rough time lost driving through the pit lane vs a flying lap. "
                             "Varies by circuit — adjust if you know this track's actual value.",
                    )
                    st.caption(
                        "This model assumes lap time degrades linearly with tyre age, based only "
                        "on this session's own data. It doesn't account for traffic, safety cars, "
                        "fuel-corrected pace, grip evolution, or overtaking difficulty — treat this "
                        "as a data-driven estimate, not a guarantee."
                    )

                total_laps = int(clean["LapNumber"].max())

                for driver in drivers:
                    driver_laps = clean[clean["Driver"] == driver]
                    analysis = analyze_driver_strategy(driver_laps, total_laps, pit_loss)

                    st.markdown(f"#### {driver}")
                    if analysis is None:
                        st.info(f"Not enough clean laps to model {driver}'s degradation reliably.")
                        continue

                    actual_plan_str = " → ".join(
                        f"{c} ({int(l)} laps)" for c, l in analysis["actual_plan"]
                    )
                    st.markdown(
                        f"**Actual strategy:** {analysis['actual_stops']}-stop — {actual_plan_str}  \n"
                        f"**Modeled time for actual strategy:** {analysis['actual_time']:.1f}s"
                    )

                    if not analysis["alternatives"]:
                        st.markdown(
                            "No alternative strategy within realistic tyre-life bounds "
                            "was found to compare against."
                        )
                        st.divider()
                        continue

                    best_k, (best_time, best_plan) = min(
                        analysis["alternatives"].items(), key=lambda kv: kv[1][0]
                    )
                    time_diff = analysis["actual_time"] - best_time
                    best_plan_str = " → ".join(f"{c} ({int(l)} laps)" for c, l in best_plan)

                    if time_diff > 1.0:
                        st.markdown(
                            f"**Model's suggested alternative:** {best_k}-stop — {best_plan_str}  \n"
                            f"**Modeled time:** {best_time:.1f}s "
                            f"(**{time_diff:.1f}s faster** than the actual strategy)"
                        )
                        st.markdown(
                            f"📝 **Conclusion:** Based on how {driver}'s pace actually fell off with "
                            f"tyre age this session, the data suggests a {best_k}-stop strategy "
                            f"using {best_plan_str} could have gained roughly **{time_diff:.1f} seconds** "
                            f"over the race distance. This is a modeled estimate from this session's "
                            f"pace trends alone — it can't account for where safety cars fell, traffic "
                            f"in the pack, or tyre availability."
                        )
                    else:
                        st.markdown(
                            f"**Model's suggested alternative:** {best_k}-stop — {best_plan_str}  \n"
                            f"**Modeled time:** {best_time:.1f}s "
                            f"({abs(time_diff):.1f}s slower than actual)"
                        )
                        st.markdown(
                            f"📝 **Conclusion:** {driver}'s actual strategy already looks close to "
                            f"optimal based on this session's pace data — the model couldn't find "
                            f"a meaningfully faster alternative within realistic tyre-life limits."
                        )
                    st.divider()

else:
    st.info("Pick a year, race, and session in the sidebar, then click **🏁 Load Race** to begin.")
