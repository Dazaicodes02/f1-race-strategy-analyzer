import os
import fastf1
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG — change these to analyze a different race ──────────────────────
YEAR = 2024
GRAND_PRIX = "Bahrain"     # e.g. "Monaco", "Singapore", "Silverstone"
SESSION = "R"               # R = Race, Q = Qualifying, FP1/FP2/FP3 = Practice
DRIVERS = ["VER", "HAM", "LEC"]   # driver 3-letter codes to compare, or None for all
# ─────────────────────────────────────────────────────────────────────────

os.makedirs("f1_cache", exist_ok=True)  # fastf1 requires this folder to exist first
fastf1.Cache.enable_cache("f1_cache")  # local folder, avoids re-downloading every run


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


def plot_degradation(clean_laps: pd.DataFrame, save_path="tyre_degradation.png"):
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


def main():
    print(f"Loading {YEAR} {GRAND_PRIX} {SESSION} data...")
    laps = load_race_data()
    clean = clean_laps(laps)

    deg_table = build_degradation_table(clean)
    print("\n=== Tyre Degradation Summary (sec lost per additional lap of tyre age) ===")
    print(deg_table.to_string(index=False))

    # Export — this CSV is what you'd load into Power BI or import into a SQL table
    deg_table.to_csv("tyre_degradation_summary.csv", index=False)
    clean[["Driver", "LapNumber", "Compound", "TyreLife", "Stint", "LapTimeSeconds"]].to_csv(
        "clean_laps_export.csv", index=False
    )
    print("\nExported: tyre_degradation_summary.csv, clean_laps_export.csv")

    plot_degradation(clean)


if __name__ == "__main__":
    main()
