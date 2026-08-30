"""
Batch pre-fetch script — downloads every session (FP1, FP2, FP3, Qualifying,
Race) for every 2026 Grand Prix held so far, building up your local
f1_cache folder in one run instead of clicking through the app dozens of
times.

HOW TO RUN
1. Put this file in the same folder as your app.py (so it shares the same
   f1_cache folder).
2. pip install fastf1
3. python prefetch_all_2026.py

This can take a while (10-30+ minutes depending on how many races have run
and your internet speed) since it's genuinely downloading a season's worth
of timing data. It's safe to stop and re-run — sessions already cached are
skipped automatically, so you can resume later if interrupted.

WHAT TO DO AFTER
Once it finishes, zip your f1_cache folder and update your GitHub Release
asset with it, following the same process as before.
"""

import datetime
import os
import time

import fastf1

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

YEAR = 2026
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]


def get_completed_races(year: int) -> list[str]:
    """Returns the names of every event in this year's schedule that has
    already happened (so we don't waste time trying to fetch races that
    haven't run yet).

    Uses Session5DateUtc rather than Session5Date — each race's local session
    time is in a different timezone (Australia is +11, Miami is -04, etc.),
    so the plain Session5Date column can't be treated as one uniform
    datetime type. The UTC-normalized column sidesteps that entirely.
    """
    schedule = fastf1.get_event_schedule(year)
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    completed = schedule[schedule["Session5DateUtc"] < now_utc]
    return completed["EventName"].tolist()


def main():
    print(f"Fetching {YEAR} season schedule...")
    races = get_completed_races(YEAR)
    print(f"Found {len(races)} completed race weekends: {races}\n")

    total = len(races) * len(SESSIONS)
    done = 0
    failed = []

    for race in races:
        for session_type in SESSIONS:
            done += 1
            label = f"[{done}/{total}] {YEAR} {race} - {session_type}"
            try:
                session = fastf1.get_session(YEAR, race, session_type)
                session.load(telemetry=False, weather=False)
                print(f"{label}: OK ({len(session.laps)} laps)")
            except Exception as e:
                print(f"{label}: FAILED ({e})")
                failed.append((race, session_type))
            time.sleep(1)  # small pause to be polite to the API

    print(f"\nDone. {total - len(failed)}/{total} sessions fetched successfully.")
    if failed:
        print("Failed sessions (this is normal for e.g. Sprint weekends with no FP2/FP3):")
        for race, session_type in failed:
            print(f"  - {race} {session_type}")

    # Report final cache size so you know what you're about to zip
    cache_size_mb = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk("f1_cache")
        for f in filenames
    ) / (1024 * 1024)
    print(f"\nTotal f1_cache size: {cache_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
