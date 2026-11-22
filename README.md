# tradepulse-data

Nightly market data for the TradePulse app. A GitHub Action scans every NSE stock after market close and writes JSON to `data/`:

| File | What |
|---|---|
| `data/scan.json` | All stocks (core fields), horizon scores (1W…>1Y), top-20 picks per horizon, Multibagger Radar, full detail for picks |
| `data/d/<A-Z>.json` | Per-stock detail (checklist, reasons, plans), loaded on demand |
| `data/themes.json` | Big Themes board with live gauges (Yahoo + FRED) |
| `data/events.json` | Event Radar: verified macro calendar (Fed, BLS, RBI, NSE) + company result dates |
| `data/buzz.json` | Creator Buzz: latest finance YouTube titles and stock mentions |

Scores are rule-based screens from public data, not investment advice.
