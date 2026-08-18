"""
F1 Race Strategy Analyzer — Tyre Degradation & Pit Stop Comparison
--------------------------------------------------------------------
A starter portfolio project using real F1 telemetry via the FastF1 library.

WHAT THIS SCRIPT DOES
1. Pulls official lap-by-lap data for a chosen race (compound, tyre age, lap time, pit stops)
2. Cleans it (drops in/out laps, safety car laps — these distort degradation curves)
3. Fits a degradation trend (lap time vs tyre age) per compound per driver
4. Produces three visualizations:
   - Tyre degradation curves (lap time vs tyre age)
   - Strategy timeline (which compound each driver ran, and when they pitted)
   - Average race pace comparison (simplest chart — good for a quick recruiter skim)
5. Exports a clean table to CSV — this is what you'd load into Power BI or a SQL database

HOW TO RUN
1. pip install fastf1 pandas matplotlib
2. Just run: python f1_tyre_degradation_analysis.py
   (First run downloads and caches data — takes a minute or two. After that it's instant.)
3. Edit YEAR / GP / SESSION / DRIVERS below to analyze any race you like.

WHY THIS PROJECT MATTERS FOR YOUR PORTFOLIO
This is the exact type of analysis race strategists do before and during a Grand Prix:
"how much time do we lose per lap as this tyre ages, and does that justify pitting earlier?"
"""

import os
import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG — change these to analyze a different race ──────────────────────
YEAR = 2026
GRAND_PRIX = "Barcelona"     # e.g. "Monaco", "Singapore", "Silverstone"
SESSION = "R"               # R = Race, Q = Qualifying, FP1/FP2/FP3 = Practice
DRIVERS = ["VER", "HAM", "LEC" , "NOR" , "PIA" , "ALO" , "ANT" , "RUS"]   # driver 3-letter codes to compare, or None for all
# ─────────────────────────────────────────────────────────────────────────

os.makedirs("f1_cache", exist_ok=True)  # fastf1 requires this folder to exist first
fastf1.Cache.enable_cache("f1_cache")  # local folder, avoids re-downloading every run

# Used to build unique output filenames per race, e.g. "singapore_2024"
# so re-running for a new race doesn't overwrite a previous race's results.
RACE_TAG = f"{GRAND_PRIX.lower().replace(' ', '_')}_{YEAR}"


def load_race_data():
    session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION)
    session.load(telemetry=False, weather=False)  # lap/tyre data only — faster
    laps = session.laps
    if DRIVERS:
        laps = laps[laps["Driver"].isin(DRIVERS)]
    return laps


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Remove laps that don't reflect true pace: in/out laps, laps under yellow/SC/VSC."""
    clean = laps.copy()
    clean = clean[clean["PitOutTime"].isna() & clean["PitInTime"].isna()]
    clean = clean[clean["TrackStatus"] == "1"]  # '1' = green flag racing
    clean = clean.dropna(subset=["LapTime", "TyreLife", "Compound"])
    clean["LapTimeSeconds"] = clean["LapTime"].dt.total_seconds()
    return clean


def build_degradation_table(clean_laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver/compound/stint with a simple linear degradation slope (sec/lap)."""
    rows = []
    grouped = clean_laps.groupby(["Driver", "Compound", "Stint"])
    for (driver, compound, stint), stint_laps in grouped:
        if len(stint_laps) < 4:  # need a few laps to fit a meaningful trend
            continue
        slope = stint_laps["LapTimeSeconds"].corr(stint_laps["TyreLife"]) * (
            stint_laps["LapTimeSeconds"].std() / stint_laps["TyreLife"].std()
        )
        rows.append({
            "Driver": driver,
            "Compound": compound,
            "Stint": stint,
            "LapsOnStint": len(stint_laps),
            "AvgLapTime": round(stint_laps["LapTimeSeconds"].mean(), 3),
            "DegradationSecPerLap": round(slope, 4),
        })
    return pd.DataFrame(rows).sort_values(["Driver", "Stint"])


def plot_degradation(clean_laps: pd.DataFrame, save_path=None):
    if save_path is None:
        save_path = f"tyre_degradation_{RACE_TAG}.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    for (driver, compound), grp in clean_laps.groupby(["Driver", "Compound"]):
        ax.plot(grp["TyreLife"], grp["LapTimeSeconds"], marker="o", markersize=3,
                label=f"{driver} - {compound}", alpha=0.7)
    ax.set_xlabel("Tyre Age (laps)")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title(f"{YEAR} {GRAND_PRIX} GP — Lap Time vs Tyre Age")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


# Standard F1 broadcast colours for tyre compounds — makes the strategy chart instantly readable
COMPOUND_COLORS = {
    "SOFT": "#DA291C",
    "MEDIUM": "#FFD12E",
    "HARD": "#F0F0F0",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
}


def plot_strategy_timeline(clean_laps: pd.DataFrame, save_path=None):
    """Horizontal stacked bar chart: each driver's stints by compound across the race.
    This is the classic 'who pitted when, on what tyre' strategy chart used in F1 broadcasts —
    the easiest of the three charts for a non-technical viewer (e.g. a recruiter) to read.
    """
    if save_path is None:
        save_path = f"strategy_timeline_{RACE_TAG}.png"

    # Build one row per driver/stint: which laps it covered and which compound
    stints = (
        clean_laps.groupby(["Driver", "Stint", "Compound"])["LapNumber"]
        .agg(["min", "max"])
        .reset_index()
        .sort_values(["Driver", "Stint"])
    )

    drivers = stints["Driver"].unique()
    fig, ax = plt.subplots(figsize=(10, max(3, len(drivers) * 0.6)))

    for i, driver in enumerate(drivers):
        driver_stints = stints[stints["Driver"] == driver]
        for _, row in driver_stints.iterrows():
            lap_start, lap_end = row["min"], row["max"]
            color = COMPOUND_COLORS.get(row["Compound"], "#999999")
            ax.barh(i, lap_end - lap_start + 1, left=lap_start, color=color,
                    edgecolor="black", linewidth=0.5)

    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers)
    ax.set_xlabel("Lap Number")
    ax.set_title(f"{YEAR} {GRAND_PRIX} GP — Tyre Strategy Timeline")
    ax.invert_yaxis()  # first driver on top

    # Legend built manually since bars don't carry per-compound labels
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black") for c in COMPOUND_COLORS.values()]
    ax.legend(handles, COMPOUND_COLORS.keys(), fontsize=8, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


def plot_average_pace(clean_laps: pd.DataFrame, save_path=None):
    """Simple bar chart of average lap time per driver — the quickest chart to read,
    good for a portfolio README or a non-technical audience skimming your project.
    """
    if save_path is None:
        save_path = f"average_pace_{RACE_TAG}.png"

    avg_pace = clean_laps.groupby("Driver")["LapTimeSeconds"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(avg_pace.index, avg_pace.values, color="#1f77b4")
    ax.set_ylabel("Average Lap Time (seconds)")
    ax.set_title(f"{YEAR} {GRAND_PRIX} GP — Average Race Pace by Driver")
    ax.set_ylim(avg_pace.min() - 0.5, avg_pace.max() + 0.5)  # zoom in — differences are small but real
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, avg_pace.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}",
                ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


def main():
    print(f"Loading {YEAR} {GRAND_PRIX} {SESSION} data...")
    laps = load_race_data()
    clean = clean_laps(laps)

    deg_table = build_degradation_table(clean)
    print("\n=== Tyre Degradation Summary (sec lost per additional lap of tyre age) ===")
    print(deg_table.to_string(index=False))

    # Export — this CSV is what you'd load into Power BI or import into a SQL table
    summary_path = f"tyre_degradation_summary_{RACE_TAG}.csv"
    laps_path = f"clean_laps_export_{RACE_TAG}.csv"
    deg_table.to_csv(summary_path, index=False)
    clean[["Driver", "LapNumber", "Compound", "TyreLife", "Stint", "LapTimeSeconds"]].to_csv(
        laps_path, index=False
    )
    print(f"\nExported: {summary_path}, {laps_path}")

    plot_degradation(clean)
    plot_strategy_timeline(clean)
    plot_average_pace(clean)


if __name__ == "__main__":
    main()
