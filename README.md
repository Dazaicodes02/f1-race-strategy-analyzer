# 🏁 F1 Race Strategy & Tyre Degradation Analyzer

An interactive Formula 1 strategy dashboard — pulls real F1 telemetry data, models tyre degradation per driver, and simulates alternate pit-stop strategies to evaluate whether a faster race was actually on the table. Built as a portfolio project combining a data analytics/AI background with a long-standing interest in F1 strategy.

**🌐 Live demo:** [https://f1-race-strategy-analyzer.streamlit.app/]

## What it does

Pick any Grand Prix from 2026 onward, and the dashboard:

- Pulls official lap-by-lap timing, tyre compound, tyre age, and stint data via [FastF1](https://github.com/theOehrly/Fast-F1)
- Cleans it — removes in/out laps and laps run under Safety Car/VSC that would distort degradation numbers
- Visualizes it four ways, then goes a step further and **models what the optimal strategy would have been**

### Tabs

| Tab | What it shows |
|---|---|
| 📉 **Tyre Degradation** | Interactive lap-time-vs-tyre-age chart. Toggle between per-stint tyre age and full-race lap number; pit stops are marked with dashed lines and hoverable explanations. Handles 15+ drivers at once — click any driver's legend entry to isolate or hide their line. |
| 🔧 **Strategy Timeline** | The classic broadcast-style stint chart — which compound each driver ran and when they pitted. |
| ⏱️ **Average Pace** | Quick bar-chart ranking of average lap time per driver, colored by team. |
| 📋 **Data Table** | Raw degradation summary (seconds lost per lap of tyre age, per driver/compound/stint), downloadable as CSV. |
| 🏆 **Strategy Verdict** | The core analysis — see below. |

### Strategy Verdict — the interesting part

For each driver, the app:

1. **Fits a real degradation curve** per compound (linear regression on their actual lap times vs. tyre age)
2. **Replays their actual strategy** through that model, for a fair baseline
3. **Brute-force searches** every realistic 1-stop, 2-stop, and 3-stop alternative — every possible pit lap and compound combination — to find what the data says would have been fastest
4. **Refuses to recommend unrealistic strategies** — won't suggest running a compound far beyond how long it was ever actually observed lasting that session
5. Optionally factors in **real tyre-set availability**: loads FP1, FP2, FP3, and Qualifying for that weekend and counts how many fresh sets of each compound were actually used before the race (via FastF1's `FreshTyre` flag), then applies the current FIA allocation (13 sets — 2 Hard / 3 Medium / 8 Soft on a normal weekend, 12 sets on a Sprint weekend) to cap what's realistically left for the race
6. Optionally factors in **real safety car / VSC windows** from that session — a pit stop landing under a neutralized track costs far less real time than a green-flag stop, and the model prices that in rather than assuming every stop costs the same

Every conclusion is written in plain English with the actual numbers, and only presented as a "faster alternative" when the model found something genuinely quicker — not just the least-bad option in a limited search.

**Caveats, stated plainly in the app itself:** this assumes linear degradation from the session's own data, and can't account for traffic, tyre-cliff effects, or race-day risk (punctures, damage, etc.). It's a data-driven estimate, not a guarantee.

## Presentation

- Full-screen intro video on load, auto-advancing once it finishes (no click needed)
- A looping video plays during data loading instead of a plain spinner
- Dark, F1-themed styling (Titillium Web font, team colors pulled directly from FastF1 rather than official logos/branding, to avoid trademarked assets)
- Personal background photo (Spa-Francorchamps), embedded directly in the app

## Tech stack

| Layer | Tool |
|---|---|
| Data source | [FastF1](https://github.com/theOehrly/Fast-F1) (official F1 timing API wrapper) |
| App framework | [Streamlit](https://streamlit.io) |
| Data processing | Python, pandas, NumPy |
| Interactive charts | Plotly |
| Static charts | Matplotlib |

## Project structure

```
f1-race-strategy-analyzer/
├── app.py                  # the full Streamlit app
├── requirements.txt
├── static/
│   ├── intro.mp4            # full-screen intro video
│   └── loading.mp4          # loading-state video
└── .streamlit/
    └── config.toml          # dark theme + static file serving
```

> Note: the background photo is embedded directly as base64 inside `app.py`, so it needs no separate file or static-serving configuration — one less thing that can be misconfigured on deployment.

## Running it locally

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `intro.mp4` and `loading.mp4` inside a `static/` folder next to `app.py`, and make sure `.streamlit/config.toml` includes:

```toml
[server]
enableStaticServing = true
```

On first loading a race, FastF1 downloads and caches session data (takes 1–2 minutes, needs internet access). Subsequent loads of the same race are near-instant thanks to local caching.

## Deploying (free, public link)

1. Push this repo to GitHub, including `static/` and `.streamlit/config.toml`
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. Click **New app** → select this repo → set the main file to `app.py` → **Deploy**
4. You'll get a public URL like `https://your-app-name.streamlit.app`

## Roadmap

- [ ] Sector-by-sector telemetry overlay for qualifying lap comparisons
- [ ] Season-level pace comparison across multiple races
- [ ] Export the Strategy Verdict conclusions as a shareable summary/report

## Author

Thangavel K — B.Tech, Artificial Intelligence & Data Science, Saranathan College of Engineering
[LinkedIn](https://www.linkedin.com/in/thangavel-kbtech) · [GitHub](https://github.com/Dazaicodes02)
