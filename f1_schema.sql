-- ============================================================
-- F1 Race Strategy Analyzer — Relational Schema
-- Load clean_laps_export.csv and tyre_degradation_summary.csv
-- (produced by f1_tyre_degradation_analysis.py) into these tables.
-- ============================================================

CREATE TABLE sessions (
    session_id      INTEGER PRIMARY KEY,
    year            INTEGER NOT NULL,
    grand_prix      VARCHAR(50) NOT NULL,
    session_type    VARCHAR(5) NOT NULL   -- 'R', 'Q', 'FP1' etc.
);

CREATE TABLE drivers (
    driver_code     VARCHAR(3) PRIMARY KEY,   -- e.g. 'VER', 'HAM'
    driver_name     VARCHAR(50),
    team            VARCHAR(50)
);

CREATE TABLE laps (
    lap_id          INTEGER PRIMARY KEY,
    session_id      INTEGER REFERENCES sessions(session_id),
    driver_code     VARCHAR(3) REFERENCES drivers(driver_code),
    lap_number      INTEGER NOT NULL,
    compound        VARCHAR(10) NOT NULL,     -- SOFT / MEDIUM / HARD / INTERMEDIATE / WET
    tyre_life       INTEGER NOT NULL,          -- laps on this tyre at time of this lap
    stint           INTEGER NOT NULL,
    lap_time_sec    DECIMAL(6,3) NOT NULL
);

CREATE TABLE stint_degradation (
    stint_id        INTEGER PRIMARY KEY,
    session_id      INTEGER REFERENCES sessions(session_id),
    driver_code     VARCHAR(3) REFERENCES drivers(driver_code),
    compound        VARCHAR(10) NOT NULL,
    stint           INTEGER NOT NULL,
    laps_on_stint   INTEGER NOT NULL,
    avg_lap_time    DECIMAL(6,3),
    degradation_sec_per_lap DECIMAL(6,4)   -- positive = getting slower as tyre ages
);

-- ============================================================
-- Example analysis queries once data is loaded
-- ============================================================

-- 1. Which compound degrades fastest across all drivers in a race?
SELECT compound, ROUND(AVG(degradation_sec_per_lap), 4) AS avg_degradation
FROM stint_degradation
GROUP BY compound
ORDER BY avg_degradation DESC;

-- 2. Compare two drivers' pace drop-off on the same compound
SELECT driver_code, compound, stint, avg_lap_time, degradation_sec_per_lap
FROM stint_degradation
WHERE driver_code IN ('VER', 'HAM')
ORDER BY compound, driver_code;

-- 3. Rolling lap time trend for one driver's stint (for a line chart in Power BI)
SELECT lap_number, tyre_life, lap_time_sec
FROM laps
WHERE driver_code = 'VER' AND stint = 1
ORDER BY lap_number;
