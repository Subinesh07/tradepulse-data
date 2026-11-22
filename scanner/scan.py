#!/usr/bin/env python3
"""TradePulse nightly scanner.

Scans every NSE equity, scores it for six horizons, runs the long-term research
checklist, builds the Big Themes board, the Event Radar and Creator Buzz, and
writes compact JSON into data/ for the TradePulse app to download.

Run:  python scanner/scan.py            (live, used by GitHub Actions)
      python scanner/scan.py --demo     (synthetic data, offline test)
"""
import io, json, math, os, re, sys, time, random, datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DATA, exist_ok=True)
DEMO = '--demo' in sys.argv
FUND_BATCH = int(os.environ.get('FUND_BATCH', '450'))   # fundamentals refreshed per run (rotating)
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime.now(IST)
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
      'Accept-Language': 'en-US,en;q=0.9'}
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def rd(x, n=2):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return round(float(x), n)
    except Exception:
        return None


def load(name, default):
    p = os.path.join(DATA, name)
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def save(name, obj):
    with open(os.path.join(DATA, name), 'w') as f:
        json.dump(obj, f, separators=(',', ':'), ensure_ascii=False)


def http(url, headers=None, timeout=25, session=None):
    import requests
    s = session or requests
    r = s.get(url, headers=headers or UA, timeout=timeout)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------- holidays / trading days
NSE_HOLIDAYS = {  # weekday NSE equity holidays from the NSE 2026 circular (verified). NSE publishes the 2027 list in December.
    '2026-01-15', '2026-01-26', '2026-03-03', '2026-03-26', '2026-03-31', '2026-10-02', '2026-10-20', '2026-11-10', '2026-11-24', '2026-12-25'}


def is_trading_day(d):
    return d.weekday() < 5 and d.isoformat() not in NSE_HOLIDAYS


def next_trading_day(d):
    d = d + dt.timedelta(days=1)
    while not is_trading_day(d):
        d += dt.timedelta(days=1)
    return d


# ---------------------------------------------------------------- universe
SECTOR_FALLBACK = {}


def get_universe():
    """All NSE EQ-series stocks. Falls back to Nifty Total Market, then to the last saved universe."""
    import requests
    uni, sectors = [], {}
    sess = requests.Session()
    try:  # industry map for ~750 stocks
        r = http('https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv', session=sess)
        df = pd.read_csv(io.StringIO(r.text))
        for _, row in df.iterrows():
            sectors[str(row['Symbol']).strip()] = (str(row.get('Industry', '')).strip(), str(row.get('Company Name', '')).strip())
        log('nifty total market list:', len(sectors))
    except Exception as e:
        log('WARN total market list failed:', e)
    try:
        try:
            sess.get('https://www.nseindia.com', headers=UA, timeout=15)
        except Exception:
            pass
        r = http('https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv', session=sess)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        df = df[df['SERIES'].astype(str).str.strip().isin(['EQ', 'BE', 'BZ'])]
        for _, row in df.iterrows():
            s = str(row['SYMBOL']).strip()
            uni.append({'s': s, 'n': str(row['NAME OF COMPANY']).strip().title(), 'sec': sectors.get(s, ('', ''))[0]})
        log('NSE equity list:', len(uni))
    except Exception as e:
        log('WARN NSE equity list failed:', e)
    if len(uni) < 500 and sectors:
        uni = [{'s': s, 'n': v[1], 'sec': v[0]} for s, v in sectors.items()]
        log('using Nifty Total Market universe:', len(uni))
    if len(uni) < 200:
        uni = load('universe.json', [])
        log('using cached universe:', len(uni))
    else:
        save('universe.json', uni)
    return uni


# ---------------------------------------------------------------- prices
def download_prices(symbols, period='3y'):
    import yfinance as yf
    out = {}
    tick = [s + '.NS' for s in symbols]
    for i in range(0, len(tick), 150):
        chunk = tick[i:i + 150]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=period, interval='1d', group_by='ticker', auto_adjust=True,
                                 threads=True, progress=False)
                break
            except Exception as e:
                log('WARN price chunk', i, attempt, e)
                time.sleep(10 * (attempt + 1))
                df = None
        if df is None or df.empty:
            continue
        for t in chunk:
            try:
                d = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                d = d.dropna(subset=['Close'])
                if len(d) >= 60:
                    out[t[:-3]] = d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            except Exception:
                pass
        log(f'prices {min(i + 150, len(tick))}/{len(tick)} ok={len(out)}')
        time.sleep(1.5)
    return out


def yseries(sym, period='2y'):
    import yfinance as yf
    try:
        d = yf.download(sym, period=period, interval='1d', auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return d.dropna(subset=['Close'])
    except Exception as e:
        log('WARN series', sym, e)
        return None


# ---------------------------------------------------------------- technical features
def rsi(c, p=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def tech(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume'].fillna(0)
    n = len(c)
    px = float(c.iloc[-1])
    ret = lambda k: (px / float(c.iloc[-1 - k]) - 1) * 100 if n > k else None
    d50 = float(c.rolling(50).mean().iloc[-1]) if n >= 50 else None
    d200 = float(c.rolling(200).mean().iloc[-1]) if n >= 200 else None
    d200_prev = float(c.rolling(200).mean().iloc[-21]) if n >= 221 else None
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    y = df.iloc[-250:]
    hi52, lo52 = float(y['High'].max()), float(y['Low'].min())
    y3 = df.iloc[-750:]
    hi3, lo3 = float(y3['High'].max()), float(y3['Low'].min())
    r = c.pct_change().iloc[-120:]
    vol = float(r.std() * math.sqrt(252) * 100) if len(r) > 20 else None
    peak = c.iloc[-250:].cummax()
    dd = float(((c.iloc[-250:] / peak) - 1).min() * 100)
    v20, v120 = float(v.iloc[-20:].mean()), float(v.iloc[-120:].mean() or 1)
    traded = float((c.iloc[-20:] * v.iloc[-20:]).mean())  # avg daily traded value (₹)
    # base: how tight was the range in the 9 months before the last month
    base = df.iloc[-210:-21] if n > 230 else None
    base_tight = None
    if base is not None and len(base) > 50:
        base_tight = (float(base['High'].max()) / float(base['Low'].min()) - 1) * 100
    return {
        'px': px, 'chg1d': (px / float(c.iloc[-2]) - 1) * 100 if n > 1 else 0,
        'r': {'w1': ret(5), 'm1': ret(21), 'm3': ret(63), 'm6': ret(126), 'y1': ret(250), 'y3': ret(745)},
        'd50': d50, 'd200': d200, 'd200up': (d200 > d200_prev) if (d200 and d200_prev) else None,
        'atr': atr, 'atrp': atr / px * 100, 'hi52': hi52, 'lo52': lo52, 'hi3': hi3, 'lo3': lo3,
        'fromhi': (px / hi52 - 1) * 100, 'vol': vol, 'dd': dd, 'rsi': float(rsi(c).iloc[-1]),
        'volx': v20 / v120 if v120 else 1, 'traded': traded, 'base_tight': base_tight,
        'brk20': px >= float(h.iloc[-21:-1].max()) if n > 21 else False,
        'spark': [rd(x, 2) for x in c.iloc[-60::3].tolist()],
    }


# ---------------------------------------------------------------- fundamentals (yfinance statements) + checklist
def row(df, names):
    if df is None or getattr(df, 'empty', True):
        return None
    for nm in names:
        if nm in df.index:
            s = df.loc[nm].dropna()
            if len(s):
                return s.sort_index()  # oldest -> newest
    return None


def cagr(s, years=3):
    try:
        if s is None or len(s) < 2:
            return None
        k = min(years, len(s) - 1)
        a, b = float(s.iloc[-1 - k]), float(s.iloc[-1])
        if a <= 0 or b <= 0:
            return None
        return ((b / a) ** (1 / k) - 1) * 100
    except Exception:
        return None


def fetch_fund(sym, px_df):
    import yfinance as yf
    t = yf.Ticker(sym + '.NS')
    info = {}
    try:
        info = t.info or {}
    except Exception as e:
        log('WARN info', sym, e)
    inc = bal = cf = None
    try:
        inc = t.income_stmt
    except Exception:
        pass
    try:
        bal = t.balance_sheet
    except Exception:
        pass
    try:
        cf = t.cashflow
    except Exception:
        pass
    res_date = None
    try:
        cal = t.calendar
        if isinstance(cal, dict) and cal.get('Earnings Date'):
            ed = cal['Earnings Date']
            ed = ed[0] if isinstance(ed, (list, tuple)) else ed
            res_date = str(ed)[:10]
    except Exception:
        pass

    rev = row(inc, ['Total Revenue', 'Operating Revenue'])
    ebit = row(inc, ['EBIT', 'Operating Income'])
    opi = row(inc, ['Operating Income', 'EBIT'])
    ni = row(inc, ['Net Income', 'Net Income Common Stockholders'])
    pbt = row(inc, ['Pretax Income'])
    cogs = row(inc, ['Cost Of Revenue', 'Reconciled Cost Of Revenue'])
    intexp = row(inc, ['Interest Expense'])
    ta = row(bal, ['Total Assets'])
    cl = row(bal, ['Current Liabilities'])
    ca = row(bal, ['Current Assets'])
    inv = row(bal, ['Inventory'])
    recv = row(bal, ['Accounts Receivable', 'Receivables'])
    eq = row(bal, ['Stockholders Equity', 'Common Stock Equity'])
    debt = row(bal, ['Total Debt'])
    shares = row(bal, ['Ordinary Shares Number', 'Share Issued'])
    ocf = row(cf, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'])
    capex = row(cf, ['Capital Expenditure'])

    last = lambda s: float(s.iloc[-1]) if s is not None and len(s) else None
    f = {
        'mcap': info.get('marketCap') or ((info.get('sharesOutstanding') or 0) * float(px_df['Close'].iloc[-1]) if (info.get('sharesOutstanding') and px_df is not None) else None), 'pe': info.get('trailingPE'), 'pb': info.get('priceToBook'),
        'roe': (info.get('returnOnEquity') or 0) * 100 if info.get('returnOnEquity') is not None else None,
        'de': (info.get('debtToEquity') / 100) if info.get('debtToEquity') is not None else None,
        'cr': info.get('currentRatio'), 'div': info.get('dividendYield'), 'beta': info.get('beta'),
        'prom': (info.get('heldPercentInsiders') or 0) * 100 if info.get('heldPercentInsiders') is not None else None,
        'sector': info.get('sector'), 'industry': info.get('industry'), 'eps': info.get('trailingEps'),
        'target': info.get('targetMeanPrice'), 'rec': info.get('recommendationKey'), 'res': res_date,
        'name': info.get('longName') or info.get('shortName'),
    }
    if f['div'] is not None and f['div'] > 1.5:  # yfinance sometimes returns percent already
        pass
    elif f['div'] is not None:
        f['div'] = f['div'] * 100
    # derived checklist metrics
    try:
        cap_emp = (ta - cl) if (ta is not None and cl is not None) else None
        if ebit is not None and cap_emp is not None:
            ce = cap_emp.reindex(ebit.index).dropna()
            eb = ebit.reindex(ce.index)
            if len(ce):
                f['roce'] = float(eb.iloc[-1] / ce.iloc[-1] * 100) if ce.iloc[-1] else None
                if len(ce) >= 3 and (ce.iloc[-1] - ce.iloc[0]) != 0:
                    f['iroce'] = float((eb.iloc[-1] - eb.iloc[0]) / (ce.iloc[-1] - ce.iloc[0]) * 100)
        f['sales3'] = cagr(rev)
        f['profit3'] = cagr(ni)
        f['profit_last'] = (float(ni.iloc[-1] / ni.iloc[-2] - 1) * 100) if ni is not None and len(ni) > 1 and ni.iloc[-2] > 0 else None
        if opi is not None and rev is not None:
            o = (opi / rev.reindex(opi.index)).dropna() * 100
            if len(o):
                f['opm'] = float(o.iloc[-1])
                f['opm_old'] = float(o.iloc[0])
        if ocf is not None:
            f['ocf'] = last(ocf)
            if capex is not None:
                fcf = (ocf + capex.reindex(ocf.index).fillna(0))
                f['fcf'] = last(fcf)
                f['fcf_pos_years'] = int((fcf > 0).sum())
                f['fcf_years'] = int(len(fcf))
            if pbt is not None and last(pbt):
                f['cfo_pbt'] = last(ocf) / last(pbt) if last(pbt) > 0 else None
        if inv is not None and cogs is not None and last(cogs):
            f['inv_days'] = last(inv) / last(cogs) * 365
        if recv is not None and rev is not None and last(rev):
            f['debt_days'] = last(recv) / last(rev) * 365
            r2 = (recv / rev.reindex(recv.index)).dropna() * 365
            if len(r2) >= 2:
                f['debt_days_old'] = float(r2.iloc[0])
        if ca is not None and cl is not None and rev is not None:
            wc = ((ca - cl) / rev.reindex(ca.index)).dropna() * 100
            if len(wc) >= 2:
                f['wc_trend'] = float(wc.iloc[-1] - wc.iloc[0])
                f['wc_pct'] = float(wc.iloc[-1])
        if ebit is not None and intexp is not None and last(intexp):
            f['icov'] = last(ebit) / abs(last(intexp)) if last(intexp) else None
        if shares is not None and len(shares) >= 2 and shares.iloc[0] > 0:
            f['dilution'] = (float(shares.iloc[-1] / shares.iloc[0]) ** (1 / (len(shares) - 1)) - 1) * 100
        if f.get('roe') is None and ni is not None and eq is not None and last(eq):
            f['roe'] = last(ni) / last(eq) * 100
        if f.get('de') is None and debt is not None and eq is not None and last(eq):
            f['de'] = last(debt) / last(eq)
        # 3-year median PE: price at each fiscal year end / EPS of that year
        if ni is not None and shares is not None and px_df is not None:
            pes = []
            for d in ni.index[-3:]:
                sh = shares.reindex([d]).iloc[0] if d in shares.index else float(shares.iloc[-1])
                eps = float(ni[d]) / float(sh) if sh else None
                p = px_df['Close'][px_df.index <= pd.Timestamp(d).tz_localize(px_df.index.tz) if px_df.index.tz else px_df.index <= pd.Timestamp(d)]
                if eps and eps > 0 and len(p):
                    pes.append(float(p.iloc[-1]) / eps)
            if pes:
                f['pe_med3'] = float(np.median(pes))
        # loss history (turnaround detection)
        if ni is not None and len(ni) >= 3:
            f['had_loss'] = bool((ni.iloc[:-1] < 0).any()) and float(ni.iloc[-1]) > 0
        if f.get('pe') and f.get('profit3') and f['profit3'] > 0:
            f['peg'] = f['pe'] / f['profit3']
    except Exception as e:
        log('WARN derive', sym, e)
    return {k: (rd(v, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in f.items()}


CYCLICAL = re.compile(r'metal|steel|alumin|mining|cement|auto|chemical|oil|gas|energy|realty|construction|capital goods|commodit|sugar|paper|shipping', re.I)


def lynch(f, sec):
    mc = (f.get('mcap') or 0) / 1e7  # crores
    g = f.get('profit3')
    if f.get('had_loss'):
        return 'Turnaround'
    if CYCLICAL.search(sec or '') or CYCLICAL.search(f.get('industry') or ''):
        return 'Cyclical'
    if f.get('pb') is not None and f['pb'] < 1:
        return 'Asset play'
    if g is None:
        return None
    if g >= 20:
        return 'Fast grower'
    if g >= 10 and mc >= 20000:
        return 'Stalwart'
    if g >= 10:
        return 'Fast grower' if mc < 5000 and g >= 15 else 'Stalwart'
    return 'Slow grower'


def checklist(f, t, sec):
    """Friend's long-term research checklist. Each item: [label, value text, status g/a/r/n, note]."""
    items, pts, maxp = [], 0, 0

    def add(label, val, st, note='', w=1):
        nonlocal pts, maxp
        items.append([label, val, st])
        if st in 'gar':
            maxp += 2 * w
            pts += {'g': 2, 'a': 1, 'r': 0}[st] * w

    def fv(x, suf='', n=1):
        return '—' if x is None else f'{x:.{n}f}{suf}'

    g = lambda c: 'g' if c else None
    roce = f.get('roce')
    add('ROCE', fv(roce, '%'), 'n' if roce is None else 'g' if roce >= 18 else 'a' if roce >= 10 else 'r', 'Return on capital employed: above 15–18% is high quality', 2)
    peg = f.get('peg')
    add('PEG', fv(peg, '', 2), 'n' if peg is None else 'g' if peg <= 1 else 'a' if peg <= 2 else 'r', 'P/E divided by profit growth: below 1 means growth is cheap', 1.5)
    p3 = f.get('profit3')
    add('3Y profit growth', fv(p3, '%/yr'), 'n' if p3 is None else 'g' if p3 >= 15 else 'a' if p3 >= 5 else 'r', '', 1.5)
    pb = f.get('pb')
    add('Price to book', fv(pb, '', 2), 'n' if pb is None else 'g' if pb <= 3 else 'a' if pb <= 8 else 'r', 'Higher is OK only if ROE is also high')
    s3 = f.get('sales3')
    add('3Y sales growth', fv(s3, '%/yr'), 'n' if s3 is None else 'g' if s3 >= 12 else 'a' if s3 >= 5 else 'r', '', 1.5)
    pe = f.get('pe')
    pbpe = pb * pe if (pb and pe and pe > 0) else None
    add('P/B × P/E (Graham)', fv(pbpe, '', 1), 'n' if pbpe is None else 'g' if pbpe <= 22.5 else 'a' if pbpe <= 75 else 'r', 'Graham’s rule: below 22.5 is classic value')
    opm, opm0 = f.get('opm'), f.get('opm_old')
    add('Operating margin', fv(opm, '%') + (f' ({"↑" if opm0 is not None and opm > opm0 else "↓"} vs 3Y ago)' if (opm is not None and opm0 is not None) else ''),
        'n' if opm is None else 'g' if opm >= 15 and (opm0 is None or opm >= opm0 - 1) else 'a' if opm >= 8 else 'r', '')
    add('P/E', fv(pe, '', 1), 'n' if pe is None else 'r' if pe <= 0 else 'g' if pe <= 25 else 'a' if pe <= 50 else 'r', 'Negative = loss-making')
    add('Current price', f'₹{t["px"]:,.2f}', 'n', '')
    fcf, fy, fpos = f.get('fcf'), f.get('fcf_years'), f.get('fcf_pos_years')
    add('Free cash flow', '—' if fcf is None else f'₹{fcf / 1e7:,.0f} Cr' + (f' ({fpos}/{fy} yrs positive)' if fy else ''),
        'n' if fcf is None else 'g' if fcf > 0 and (not fy or fpos >= fy - 1) else 'a' if fcf > 0 else 'r', 'Cash left after capex', 1.5)
    med = f.get('pe_med3')
    add('3Y median P/E', fv(med, '', 1) + (f' (now {pe:.1f})' if (med and pe) else ''),
        'n' if (med is None or not pe) else 'g' if pe <= med * 1.05 else 'a' if pe <= med * 1.4 else 'r', 'Now cheaper than its own history = green')
    idays = f.get('inv_days')
    add('Inventory days', fv(idays, ' d', 0), 'n' if idays is None else 'g' if idays <= 90 else 'a' if idays <= 180 else 'r', '')
    dd, dd0 = f.get('debt_days'), f.get('debt_days_old')
    add('Debtor days', fv(dd, ' d', 0) + (f' ({"↑" if dd0 is not None and dd > dd0 + 10 else "steady"})' if dd is not None and dd0 is not None else ''),
        'n' if dd is None else 'g' if dd <= 60 and not (dd0 is not None and dd > dd0 + 20) else 'a' if dd <= 120 else 'r', 'Rising debtor days can hide weak sales')
    dv = f.get('div')
    add('Dividend yield', fv(dv, '%', 2), 'n' if dv is None else 'g' if dv >= 1.5 else 'a', '', 0.5)
    add('Promoter holding change (5Y)', 'Not in free data', 'n', 'Check the NSE shareholding pattern or the AI report')
    lb = lynch(f, sec)
    add('Peter Lynch bucket', lb or '—', 'n', {'Fast grower': 'Small/mid company growing 20%+ — biggest upside, biggest risk', 'Stalwart': 'Large steady grower — aim for 30–50% gains, then review',
                                              'Slow grower': 'Mature company — mainly for dividends', 'Cyclical': 'Buy when P/E looks high (profits at bottom), sell when P/E looks low',
                                              'Turnaround': 'Recovering from losses — check debt and cash carefully', 'Asset play': 'Trades below book value — hidden asset value'}.get(lb, ''))
    pos3 = (t['px'] - t['lo3']) / (t['hi3'] - t['lo3']) * 100 if t['hi3'] > t['lo3'] else 50
    add('3Y high / low', f'₹{t["lo3"]:,.0f} – ₹{t["hi3"]:,.0f} (at {pos3:.0f}% of range)', 'n', '')
    add('Pledged %', 'Not in free data', 'n', 'Check NSE/BSE shareholding filings — any pledge above 10% is a red flag')
    pr = f.get('prom')
    add('Promoter / insider holding', fv(pr, '%'), 'n' if pr is None else 'g' if pr >= 45 else 'a' if pr >= 25 else 'r', 'Skin in the game')
    roe = f.get('roe')
    add('Return on equity', fv(roe, '%'), 'n' if roe is None else 'g' if roe >= 15 else 'a' if roe >= 10 else 'r', '', 1.5)
    de = f.get('de')
    add('Debt to equity', fv(de, '', 2), 'n' if de is None else 'g' if de <= 0.5 else 'a' if de <= 1.2 else 'r', 'Banks/NBFCs naturally carry high debt', 1.5)
    cr = f.get('cr')
    add('Current ratio', fv(cr, '', 2), 'n' if cr is None else 'g' if cr >= 1.5 else 'a' if cr >= 1 else 'r', '')
    cp = f.get('cfo_pbt')
    add('Cash flow / profit before tax', fv(cp, '', 2), 'n' if cp is None else 'g' if cp >= 0.8 else 'a' if cp >= 0.5 else 'r', 'Below 0.5 = profits not turning into cash (red flag)', 1.5)
    ir = f.get('iroce')
    add('Incremental ROCE (3Y)', fv(ir, '%'), 'n' if ir is None else 'g' if ir >= 18 else 'a' if ir >= 8 else 'r', 'Return on the NEW capital invested')
    wct = f.get('wc_trend')
    add('Working capital trend', '—' if wct is None else ('rising (cash getting stuck)' if wct > 5 else 'falling (efficient)' if wct < -5 else 'stable'),
        'n' if wct is None else 'r' if wct > 10 else 'a' if wct > 5 else 'g', '')
    for q in ['Prospects for the industry', 'Capacity utilisation', 'Market share trend', 'Related party transactions', 'Auditor qualifications', 'Valuation vs mid-cycle multiples']:
        add(q, 'AI report', 'n', 'Needs annual reports — use the AI Deep Report button')
    # red flags
    flags = []
    if cp is not None and cp < 0.5:
        flags.append('Profits are not converting into operating cash')
    if dd is not None and dd0 is not None and dd > dd0 * 1.4 and dd > 60:
        flags.append('Debtor days rising sharply')
    if f.get('icov') is not None and f['icov'] < 2.5:
        flags.append(f'Weak interest cover ({f["icov"]:.1f}x)')
    if f.get('dilution') is not None and f['dilution'] > 5:
        flags.append(f'Share count growing {f["dilution"]:.0f}%/yr (dilution)')
    if opm is not None and opm0 is not None and opm < opm0 - 5:
        flags.append('Operating margin shrinking')
    if fcf is not None and fcf < 0 and (p3 or 0) > 15:
        flags.append('Growing profits but negative free cash flow')
    if pe is not None and pe > 80:
        flags.append('Extremely high valuation (P/E above 80)')
    add('Other red flags', ', '.join(flags) if flags else 'None detected in numbers', 'r' if len(flags) >= 2 else 'a' if flags else 'g', '', 1.5)
    score = round(pts / maxp * 100) if maxp >= 10 else None
    return {'score': score, 'items': items, 'flags': flags, 'lynch': lb}


# ---------------------------------------------------------------- horizon scoring
THEME_SECTOR = {  # sector keyword -> theme id (used for tailwind/headwind)
    'jewel': 'gold', 'gems': 'gold', 'information technology': 'ai', 'software': 'ai', 'it - ': 'ai', 'semicon': 'ai', 'power': 'capex', 'capital goods': 'capex',
    'construction': 'capex', 'defence': 'defence', 'aerospace': 'defence', 'oil': 'crude', 'refiner': 'crude', 'paint': 'crude', 'airline': 'crude',
    'bank': 'rates', 'financial': 'rates', 'pharma': 'usdinr', 'healthcare': 'usdinr'}


def theme_of(sec, industry, s):
    txt = f'{sec} {industry}'.lower()
    if s in ('HAL', 'BEL', 'BDL', 'MAZDOCK', 'COCHINSHIP', 'GRSE', 'DATAPATTNS', 'BEML', 'SOLARINDS', 'ASTRAMICRO', 'PARAS'):
        return 'defence'
    for k, v in THEME_SECTOR.items():
        if k in txt:
            return v
    return None


KEEP_F = {'pe', 'pb', 'roe', 'de', 'cr', 'div', 'beta', 'prom', 'sector', 'industry', 'eps', 'target', 'rec', 'roce', 'iroce', 'sales3', 'profit3', 'profit_last',
          'opm', 'opm_old', 'fcf', 'cfo_pbt', 'inv_days', 'debt_days', 'wc_trend', 'pe_med3', 'peg', 'icov', 'dilution', 'mcap', 'name'}


def shard_of(s):
    c = s[0].upper()
    return c if c.isalpha() else '0'


def clamp(x, a=0, b=100):
    return max(a, min(b, x))


def pct_rank(series):
    return series.rank(pct=True) * 100


def comp_scores(t, f, rs_pct, theme_heat):
    """Sub-scores 0-100 with reason text."""
    S, W = {}, {}
    px, d50, d200 = t['px'], t['d50'], t['d200']
    # trend
    tr = 50
    if d200:
        tr += 18 if px > d200 else -20
        if d50:
            tr += 12 if d50 > d200 else -10
        if t['d200up'] is True:
            tr += 8
        elif t['d200up'] is False:
            tr -= 8
    S['trend'] = clamp(tr)
    W['trend'] = ('Long-term uptrend (above rising 200-day average)' if tr >= 70 else 'Below the 200-day average — trend is down' if tr <= 35 else 'Trend is mixed')
    # momentum (relative strength percentile + 6/12m)
    S['mom'] = clamp(rs_pct)
    W['mom'] = f'Stronger than {rs_pct:.0f}% of NSE stocks over 6–12 months' if rs_pct >= 60 else f'Weaker than {100 - rs_pct:.0f}% of NSE stocks'
    # short-term setup
    st = 50
    if 52 <= t['rsi'] <= 68:
        st += 15
    elif t['rsi'] > 75:
        st -= 10
    elif t['rsi'] < 35:
        st -= 8
    if t['brk20']:
        st += 15
    if t['volx'] > 1.5:
        st += 10
    if (t['r']['w1'] or 0) > 0:
        st += 5
    if d50 and px > d50:
        st += 5
    S['short'] = clamp(st)
    W['short'] = ('Fresh breakout above the 20-day high' + (' on heavy volume' if t['volx'] > 1.5 else '')) if t['brk20'] else f'RSI {t["rsi"]:.0f} — ' + ('healthy strength' if 52 <= t['rsi'] <= 68 else 'overheated' if t['rsi'] > 75 else 'weak')
    # breakout from base (multibagger pattern)
    bk = 40
    if t['base_tight'] is not None and t['base_tight'] < 45 and t['fromhi'] > -6:
        bk = 85 if t['volx'] > 1.4 else 72
    elif t['fromhi'] > -5:
        bk = 65
    S['breakout'] = bk
    W['breakout'] = 'Breaking out of a long sideways base near its 52-week high' if bk >= 72 else 'Trading near its 52-week high' if bk >= 65 else 'Not near a breakout'
    # risk (low vol & shallow drawdown = higher score)
    vol = t['vol'] or 40
    S['risk'] = clamp(100 - (vol - 15) * 1.6 + max(t['dd'], -60) * 0.4)
    W['risk'] = f'Calm stock (volatility {vol:.0f}%/yr)' if vol < 25 else f'Volatile (swings ~{vol:.0f}%/yr)'
    has_f = f is not None and f.get('roce') is not None or (f is not None and f.get('roe') is not None)
    if has_f:
        q = 50
        roce, roe = f.get('roce'), f.get('roe')
        base_r = roce if roce is not None else roe
        if base_r is not None:
            q += clamp((base_r - 12) * 2, -25, 25)
        if f.get('de') is not None:
            q += 8 if f['de'] < 0.5 else -10 if f['de'] > 1.5 else 0
        if f.get('cfo_pbt') is not None:
            q += 8 if f['cfo_pbt'] >= 0.8 else -12 if f['cfo_pbt'] < 0.5 else 0
        if f.get('fcf') is not None:
            q += 5 if f['fcf'] > 0 else -6
        if f.get('opm') is not None:
            q += 5 if f['opm'] > 15 else -4 if f['opm'] < 6 else 0
        S['quality'] = clamp(q)
        W['quality'] = f'High-quality business (ROCE {base_r:.0f}%, ' + ('low debt' if (f.get('de') or 0) < 0.5 else 'manageable debt') + ')' if q >= 65 else f'Weak returns on capital ({(base_r or 0):.0f}%)' if q <= 40 else 'Average business quality'
        g = 50
        if f.get('profit3') is not None:
            g += clamp((f['profit3'] - 10) * 1.5, -25, 30)
        if f.get('sales3') is not None:
            g += clamp((f['sales3'] - 8) * 1.2, -15, 20)
        if f.get('profit_last') is not None and f.get('profit3') is not None and f['profit_last'] > f['profit3'] + 5 and f['profit_last'] > 20:
            g += 8
        S['growth'] = clamp(g)
        W['growth'] = f'Profits growing {f["profit3"]:.0f}%/yr over 3 years' if (f.get('profit3') or 0) >= 12 else 'Slow or falling profit growth' if f.get('profit3') is not None else 'Growth data limited'
        v = 50
        pe, med, peg = f.get('pe'), f.get('pe_med3'), f.get('peg')
        if pe is not None and pe <= 0:
            v = 25
        else:
            if peg is not None:
                v += 18 if peg < 1 else 6 if peg < 1.5 else -12 if peg > 2.5 else 0
            if pe and med:
                v += 12 if pe < med * 0.9 else -12 if pe > med * 1.4 else 0
            if pe and pe > 70:
                v -= 12
        S['value'] = clamp(v)
        W['value'] = ('Cheaper than its own 3-year average valuation' if (pe and med and pe < med * 0.9) else f'Growth looks cheap (PEG {peg:.1f})' if (peg and peg < 1) else 'Expensive vs its history' if (pe and med and pe > med * 1.4) else 'Fairly valued')
        gv = 50
        if f.get('prom') is not None:
            gv += 12 if f['prom'] >= 50 else -10 if f['prom'] < 25 else 0
        if f.get('dilution') is not None and f['dilution'] > 5:
            gv -= 12
        S['gov'] = clamp(gv)
        W['gov'] = f'Promoters/insiders own {f["prom"]:.0f}%' if f.get('prom') is not None else 'Ownership data limited'
    S['theme'] = theme_heat
    W['theme'] = None
    return S, W


HZ = {  # horizon weights
    '1W': {'short': .45, 'trend': .2, 'mom': .2, 'breakout': .1, 'risk': .05},
    '1M': {'mom': .3, 'trend': .25, 'short': .15, 'breakout': .15, 'theme': .1, 'risk': .05},
    '3M': {'mom': .25, 'trend': .2, 'breakout': .1, 'quality': .15, 'growth': .1, 'theme': .12, 'risk': .08},
    '6M': {'mom': .15, 'trend': .15, 'quality': .22, 'growth': .2, 'value': .08, 'theme': .1, 'risk': .1},
    '1Y': {'quality': .28, 'growth': .24, 'value': .16, 'trend': .12, 'mom': .05, 'theme': .1, 'risk': .05},
    '>1Y': {'quality': .32, 'growth': .24, 'value': .2, 'gov': .1, 'theme': .08, 'risk': .06},
}
NEEDS_FUND = {'6M', '1Y', '>1Y'}


def horizon_plan(hz, t, f):
    px, atr = t['px'], t['atr']
    if hz == '1W':
        sl = px - 1.5 * atr; tg = px + 3 * atr
    elif hz == '1M':
        sl = px - 2.5 * atr; tg = px + 5 * atr
    elif hz == '3M':
        sl = min(px - 3.5 * atr, (t['d50'] or px) * 0.97); tg = px + (px - sl) * 2
    else:
        sl = max(min((t['d200'] or px * 0.85) * 0.95, px * 0.88), px * 0.75)
        g = (f or {}).get('profit3') or 12
        g = clamp(g, 8, 30) / 100
        yrs = {'6M': 0.5, '1Y': 1, '>1Y': 3}[hz]
        rer = 1.0
        if f and f.get('pe') and f.get('pe_med3') and f['pe'] > 0:
            rer = clamp(f['pe_med3'] / f['pe'], 0.75, 1.3) ** (min(yrs, 1))
        tg = px * ((1 + g) ** yrs) * rer
    return rd(sl, 2), rd(tg, 2)


# ---------------------------------------------------------------- themes
THEMES = [
    {'id': 'ai', 'e': '🤖', 'name': 'AI boom or bubble?',
     'what': 'Huge spending on AI chips and data centres has driven US tech to records. Bubbles usually burst when that spending slows or interest rates bite, not when people first start worrying.',
     'bull': 'AI spending keeps rising: chip makers, data-centre power and cooling, and Indian firms building AI services gain.',
     'bear': 'Spending pauses or returns disappoint: US tech can fall 20–40%, global risk-off hits Indian IT and high-valuation stocks first.',
     'win': ['NETWEB', 'TATAELXSI', 'KAYNES', 'CGPOWER', 'POWERGRID', 'TATAPOWER'], 'lose': ['INFY', 'TCS', 'WIPRO'],
     'series': ['^IXIC', 'NVDA', 'SOXX']},
    {'id': 'rates', 'e': '🏦', 'name': 'US rates & dollar liquidity (repo)',
     'what': 'When US yields rise or dollar liquidity tightens (repo rates spiking above the Fed’s rate, reverse-repo cushion drained), foreign money leaves emerging markets like India.',
     'bull': 'Yields fall and liquidity is plentiful: foreign investors buy Indian banks and large caps, and the rupee steadies.',
     'bear': 'Yields jump or repo stress appears: foreign selling, weaker rupee, rate-sensitive sectors fall.',
     'win': ['HDFCBANK', 'ICICIBANK', 'BAJFINANCE', 'DLF'], 'lose': [], 'series': ['^TNX']},
    {'id': 'gold', 'e': '🥇', 'name': 'Gold cycle',
     'what': 'Gold rises when real interest rates fall, the dollar weakens, or fear rises (wars, central-bank buying). Jewellers and gold-loan lenders ride it.',
     'bull': 'Gold keeps climbing: jewellers’ inventory gains boost profits (part of the TBZ story), and gold-loan NBFCs grow faster.',
     'bear': 'Gold corrects after a big run: jeweller stocks often fall 2–3× harder than gold itself.',
     'win': ['TBZ', 'TITAN', 'KALYANKJIL', 'SENCO', 'MUTHOOTFIN', 'MANAPPURAM'], 'lose': [], 'series': ['GC=F']},
    {'id': 'crude', 'e': '🛢️', 'name': 'Crude oil',
     'what': 'India imports most of its oil. Expensive crude raises inflation, widens the trade deficit and hurts companies that use oil as raw material.',
     'bull': 'Crude falls: oil marketers, paints, tyres, airlines and chemicals gain; the rupee and inflation improve.',
     'bear': 'Crude spikes (e.g. Middle East conflict): OMCs, paints and airlines fall; ONGC and Oil India gain.',
     'win': ['BPCL', 'HINDPETRO', 'ASIANPAINT', 'INDIGO'], 'lose': ['ONGC', 'OIL'], 'series': ['BZ=F']},
    {'id': 'usdinr', 'e': '💵', 'name': 'Dollar vs rupee',
     'what': 'A weaker rupee helps exporters (IT, pharma, textiles) earn more rupees, but raises import costs and scares foreign investors.',
     'bull': 'Rupee weakens gradually: exporters’ margins improve.',
     'bear': 'Rupee falls sharply: foreign selling, imported inflation, RBI may tighten.',
     'win': ['TCS', 'INFY', 'SUNPHARMA', 'DRREDDY'], 'lose': [], 'series': ['USDINR=X', 'DX-Y.NYB']},
    {'id': 'capex', 'e': '🏗️', 'name': 'India capex & power',
     'what': 'Government and private spending on roads, railways, power grids and factories. Power demand is also rising with data centres and EVs.',
     'bull': 'Order books keep growing: capital goods, power equipment and cables re-rate.',
     'bear': 'Spending slows (elections, fiscal squeeze) or valuations are already stretched: sharp corrections.',
     'win': ['LT', 'SIEMENS', 'ABB', 'BHEL', 'POLYCAB', 'KEI', 'NTPC'], 'lose': [], 'basket': ['LT', 'SIEMENS', 'ABB', 'BHEL', 'POLYCAB', 'NTPC', 'POWERGRID']},
    {'id': 'defence', 'e': '🛡️', 'name': 'Defence & make-in-India',
     'what': 'Record defence budgets and export orders. These stocks run hard on order news, then cool for long stretches.',
     'bull': 'New big orders and exports: steady long-term growth.',
     'bear': 'Valuations far ahead of earnings: 30–50% drawdowns have happened even in strong years.',
     'win': ['HAL', 'BEL', 'BDL', 'MAZDOCK', 'COCHINSHIP', 'DATAPATTNS'], 'lose': [], 'basket': ['HAL', 'BEL', 'BDL', 'MAZDOCK', 'COCHINSHIP']},
    {'id': 'fear', 'e': '😨', 'name': 'Market fear gauge',
     'what': 'India VIX and US VIX measure expected volatility. Low readings mean calm, sometimes complacent, markets; spikes mean panic, which is often when long-term bargains appear.',
     'bull': 'VIX low and steady: trends tend to continue.',
     'bear': 'VIX spiking: expect wide swings; reduce leverage and buy only in parts.',
     'win': [], 'lose': [], 'series': ['^INDIAVIX', '^VIX']},
]


def fred(series_id):
    try:
        r = None
        for attempt in range(2):
            try:
                r = http(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={(NOW.date() - dt.timedelta(days=400)).isoformat()}', timeout=75)
                break
            except Exception as e:
                if attempt:
                    raise
                time.sleep(5)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ['d', 'v']
        df['v'] = pd.to_numeric(df['v'], errors='coerce')
        return df.dropna()
    except Exception as e:
        log('WARN fred', series_id, e)
        return None


def series_stats(d):
    c = d['Close']
    px = float(c.iloc[-1])
    d200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float(c.mean())
    y1 = (px / float(c.iloc[-250]) - 1) * 100 if len(c) > 250 else None
    m1 = (px / float(c.iloc[-21]) - 1) * 100 if len(c) > 21 else None
    return {'px': px, 'vs200': (px / d200 - 1) * 100, 'y1': y1, 'm1': m1, 'spark': [rd(x, 2) for x in c.iloc[-120::4].tolist()]}


def build_themes(prices, demo=False):
    out = []
    fr = {}
    if not demo:
        for sid in ['T10Y2Y', 'RRPONTSYD', 'SOFR', 'IORB', 'WALCL']:
            fr[sid] = fred(sid)
    for th in THEMES:
        g, heat = [], 50
        try:
            if 'series' in th:
                for sym in th['series']:
                    d = demo_series(sym) if demo else yseries(sym)
                    if d is None or len(d) < 60:
                        continue
                    s = series_stats(d)
                    g.append({'k': sym, 'v': rd(s['px'], 2), 'vs200': rd(s['vs200'], 1), 'y1': rd(s['y1'], 1), 'm1': rd(s['m1'], 1), 'spark': s['spark']})
            if 'basket' in th:
                vals = [tech(prices[s]) for s in th['basket'] if s in prices and len(prices[s]) > 210]
                if vals:
                    g.append({'k': 'Basket of ' + ', '.join(th['basket'][:4]) + '…', 'vs200': rd(np.mean([(v['px'] / v['d200'] - 1) * 100 for v in vals if v['d200']]), 1),
                              'y1': rd(np.mean([v['r']['y1'] for v in vals if v['r']['y1'] is not None]), 1), 'm1': rd(np.mean([v['r']['m1'] for v in vals if v['r']['m1'] is not None]), 1)})
            # heat: how stretched/strong the theme is
            vs = [x['vs200'] for x in g if x.get('vs200') is not None]
            y1 = [x['y1'] for x in g if x.get('y1') is not None]
            if vs:
                heat = clamp(50 + np.mean(vs) * 2 + (np.mean(y1) * 0.25 if y1 else 0))
            if th['id'] == 'rates':
                notes = []
                if fr.get('SOFR') is None:  # NY Fed fallback for SOFR
                    try:
                        j = http('https://markets.newyorkfed.org/api/rates/secured/sofr/last/1.json').json()
                        fr['SOFR'] = pd.DataFrame({'d': [j['refRates'][0]['effectiveDate']], 'v': [float(j['refRates'][0]['percentRate'])]})
                        g.append({'k': 'SOFR (overnight repo rate, %)', 'v': rd(fr['SOFR']['v'].iloc[-1], 2), 'note': 'Source: New York Fed'})
                    except Exception as e2:
                        log('WARN nyfed', e2)
                if fr.get('SOFR') is not None and fr.get('IORB') is not None and len(fr['SOFR']) and len(fr['IORB']):
                    spread = (fr['SOFR']['v'].iloc[-1] - fr['IORB']['v'].iloc[-1]) * 100
                    g.append({'k': 'SOFR − IORB (repo stress, bp)', 'v': rd(spread, 1), 'note': 'Above 0 = repo funding stress'})
                    heat += clamp(spread * 2, -10, 25)
                if fr.get('RRPONTSYD') is not None and len(fr['RRPONTSYD']):
                    v = fr['RRPONTSYD']['v'].iloc[-1]
                    g.append({'k': 'Fed reverse-repo cushion ($bn)', 'v': rd(v, 0), 'note': 'Near zero = liquidity buffer used up'})
                    if v < 100:
                        heat += 10
                if fr.get('T10Y2Y') is not None and len(fr['T10Y2Y']):
                    g.append({'k': 'US yield curve 10Y−2Y (%)', 'v': rd(fr['T10Y2Y']['v'].iloc[-1], 2), 'note': 'Negative = recession warning'})
                if fr.get('WALCL') is not None and len(fr['WALCL']) > 13:
                    w = fr['WALCL']['v']
                    g.append({'k': 'Fed balance sheet, 3-month change (%)', 'v': rd((w.iloc[-1] / w.iloc[-13] - 1) * 100, 2), 'note': 'Shrinking = draining liquidity'})
                heat = clamp(heat)
            if th['id'] == 'fear':
                vix = [x for x in g if x['k'] == '^INDIAVIX']
                if vix:
                    heat = clamp((vix[0]['v'] - 10) * 5)
        except Exception as e:
            log('WARN theme', th['id'], e)
        status = 'Overheated' if heat >= 80 else 'Heating up' if heat >= 62 else 'Cooling' if heat <= 38 else 'Neutral'
        if th['id'] == 'fear':
            status = 'Panic' if heat >= 70 else 'Nervous' if heat >= 45 else 'Calm'
        if th['id'] == 'rates':
            status = 'Liquidity stress' if heat >= 70 else 'Tightening' if heat >= 55 else 'Easy money' if heat <= 40 else 'Neutral'
        out.append({k: v for k, v in th.items() if k not in ('series', 'basket')} | {'heat': rd(heat, 0), 'status': status, 'g': g})
    return out


def theme_heat_for(tid, themes):
    for t in themes:
        if t['id'] == tid:
            h = t['heat'] or 50
            if tid in ('rates', 'fear', 'crude'):  # high heat here is a headwind
                return clamp(100 - h)
            return clamp(40 + (h - 50) * 0.8) if h < 85 else 55  # overheated themes get no extra credit
    return 50


# ---------------------------------------------------------------- events
EFFECTS = {
    'us_cpi': {'up': 'Hotter inflation → Fed keeps rates high → dollar up, gold & Nifty pressured, IT and metals dip', 'down': 'Cooler inflation → rate-cut hopes → gold, Bitcoin and Indian stocks usually rise', 'assets': ['XAUUSD', 'DXY', 'NIFTY', 'BTC']},
    'us_jobs': {'up': 'Strong jobs → fewer rate cuts → dollar up, gold down, rupee weaker', 'down': 'Weak jobs → rate-cut bets rise → gold up; very weak data can trigger recession fears', 'assets': ['XAUUSD', 'DXY', 'USDINR']},
    'fomc': {'up': 'Hawkish (rates held high / hike) → dollar up, gold & emerging markets down', 'down': 'Dovish (cut / signals cuts) → gold, crypto and Indian stocks usually rally', 'assets': ['XAUUSD', 'DXY', 'NIFTY', 'BTC']},
    'rbi': {'up': 'Rate hike → banks’ loan growth slows, rate-sensitive sectors (realty, autos) dip, rupee firms', 'down': 'Rate cut / softer stance → banks, NBFCs, realty and autos usually rally', 'assets': ['BANKNIFTY', 'USDINR', 'NIFTY']},
    'budget': {'up': 'Big capex & tax relief → infra, defence, consumption rally', 'down': 'Higher taxes on capital gains / fiscal slippage → market sell-off', 'assets': ['NIFTY', 'BANKNIFTY']},
    'india_cpi': {'up': 'Higher inflation → RBI stays tight → banks & realty pressured', 'down': 'Lower inflation → room for rate cuts → rate-sensitive stocks gain', 'assets': ['BANKNIFTY', 'NIFTY']},
    'india_gdp': {'up': 'Strong growth → cyclicals, banks and capital goods gain', 'down': 'Weak growth → defensives (FMCG, pharma) outperform', 'assets': ['NIFTY']},
    'holiday': {'up': 'Market closed', 'down': 'Market closed', 'assets': []},
    'results': {'up': 'Beat → stock often gaps up; watch whether the gap holds', 'down': 'Miss → gap down; don’t rush to buy the dip', 'assets': []},
}


def build_events(fund_cache, uni_names, demo=False):
    with open(os.path.join(HERE, 'events_static.json')) as f:
        ev = json.load(f)
    today = NOW.date().isoformat()
    ev = [e for e in ev if e['date'] >= today]
    # results dates from fundamentals cache
    res = []
    for s, f in fund_cache.items():
        d = (f.get('f') or {}).get('res')
        if d and d >= today and d <= (NOW.date() + dt.timedelta(days=100)).isoformat():
            res.append({'date': d, 'time': '', 'type': 'results', 's': s, 'title': f'{uni_names.get(s, s)} quarterly results', 'imp': 'M', 'src': 'Company/Yahoo (expected)', 'exp': True})
    res.sort(key=lambda x: x['date'])
    # ForexFactory this week (high impact)
    ff = []
    if not demo:
        for u in ['https://nfs.faireconomy.media/ff_calendar_thisweek.json', 'https://nfs.faireconomy.media/ff_calendar_nextweek.json']:
            try:
                for e in http(u).json():
                    if e.get('impact') == 'High' and e.get('country') in ('USD', 'INR', 'EUR', 'CNY', 'GBP', 'JPY'):
                        ff.append({'date': e['date'][:10], 'iso': e['date'], 'type': 'ff', 'title': f'{e["country"]} {e["title"]}', 'imp': 'H', 'fc': e.get('forecast'), 'prev': e.get('previous'), 'src': 'ForexFactory'})
            except Exception as e2:
                log('WARN ff', u, e2)
    allev = ev + ff + res[:300]
    allev.sort(key=lambda x: (x['date'], x.get('time', '')))
    return {'events': allev, 'effects': EFFECTS}


# ---------------------------------------------------------------- creator buzz
CHANNELS = [  # name, handle or channel id, language
    ('Money Pechu (Anand Srinivasan)', '@MoneyPechu', 'ta'),
    ('Thathuva Pechu (Anand Srinivasan)', 'UCrfwMUFeZZ4RBO7xTkmD3QA', 'ta'),
    ('Madurai Veeran Money Coach', 'UCIsk_cJWVlZcaMFpcdduvZA', 'ta'),
    ('CA Rachana Ranade', '@CArachanaRanade', 'en'),
    ('Pranjal Kamra (Finology)', '@PranjalKamra', 'en'),
    ('Akshat Shrivastava', '@AkshatZayn', 'en'),
    ('SOIC', '@SOIC', 'en'),
    ('Asset Yogi', '@AssetYogi', 'en'),
    ('Markets by Zerodha', '@marketsbyzerodha', 'en'),
]
BUZZ_THEMES = {'ai': r'\bai\b|artificial intel|nvidia|data cent', 'gold': r'gold|silver|jewel|தங்கம்', 'defence': r'defen[cs]e|hal\b|bel\b|மாசகான்',
               'bank': r'bank|nbfc|rbi|repo', 'it': r'\bit\b stocks|infosys|tcs\b|wipro', 'crash': r'crash|bubble|fall|சரிவு|வீழ்ச்சி', 'ipo': r'\bipo\b', 'smallcap': r'small ?cap|multibagger|penny', 'crypto': r'bitcoin|crypto'}


def resolve_channel(h):
    if h.startswith('UC'):
        return h
    try:
        txt = http(f'https://www.youtube.com/{h}', timeout=20).text
        m = re.search(r'"channelId":"(UC[\w-]{22})"', txt) or re.search(r'channel/(UC[\w-]{22})', txt)
        return m.group(1) if m else None
    except Exception as e:
        log('WARN yt resolve', h, e)
        return None


def build_buzz(uni, tech_map, lt_map, demo=False):
    cache = load('yt_channels.json', {})
    chans, vids = [], []
    for name, h, lang in CHANNELS:
        cid = cache.get(h) or (None if demo else resolve_channel(h))
        if cid:
            cache[h] = cid
        entry = {'name': name, 'lang': lang, 'ok': False, 'url': f'https://www.youtube.com/{h}' if h.startswith('@') else f'https://www.youtube.com/channel/{h}'}
        if cid and not demo:
            try:
                root = ET.fromstring(http(f'https://www.youtube.com/feeds/videos.xml?channel_id={cid}', timeout=20).content)
                ns = {'a': 'http://www.w3.org/2005/Atom', 'yt': 'http://www.youtube.com/xml/schemas/2015'}
                for e in root.findall('a:entry', ns)[:12]:
                    title = e.find('a:title', ns).text or ''
                    vid = e.find('yt:videoId', ns).text
                    pub = e.find('a:published', ns).text
                    vids.append({'ch': name, 't': title, 'id': vid, 'd': pub[:10]})
                entry['ok'] = True
            except Exception as e2:
                log('WARN yt feed', name, e2)
        chans.append(entry)
    save('yt_channels.json', cache)
    if demo:
        vids = [{'ch': 'Money Pechu (Anand Srinivasan)', 't': 'தங்கம் விலை இன்னும் உயருமா? Gold, TBZ, Titan analysis', 'id': 'demo', 'd': NOW.date().isoformat()},
                {'ch': 'CA Rachana Ranade', 't': 'Is the AI bubble about to burst? What it means for Indian IT stocks', 'id': 'demo', 'd': NOW.date().isoformat()}]
    # mentions: match NSE symbols and distinctive company names
    STOP_SYM = {'GOLD', 'IT', 'AI', 'IPO', 'NSE', 'SIP', 'ETF', 'USA', 'RBI', 'GDP', 'TV', 'EMI', 'NRI', 'PSU', 'FII', 'DII', 'BANK', 'TAX', 'GST', 'CEO', 'LIVE', 'NEWS', 'WAR', 'BEST', 'TOP', 'NEW', 'OIL', 'SBI'}
    STOP_W = {'anand', 'srinivasan', 'gandhi', 'silver', 'golden', 'money', 'market', 'markets', 'stock', 'stocks', 'share', 'shares', 'bharat', 'tamil', 'nifty', 'india', 'indian',
              'legend', 'legends', 'success', 'future', 'global', 'world', 'great', 'super', 'smart', 'power', 'energy', 'capital', 'finance', 'royal', 'prime', 'sunday', 'national', 'metal', 'universal'}
    phrase, single, sym_ok = {}, {}, {}
    for u in uni:
        s = u['s']
        if s not in STOP_SYM and len(s) >= 3:
            sym_ok[s] = s
        w = [x for x in re.sub(r'\b(limited|ltd|india|industries|the|company|corporation|of|and)\b|[^a-z ]', ' ', u['n'].lower()).split() if len(x) > 1]
        if len(w) >= 2:
            phrase[w[0] + ' ' + w[1]] = s
        tv = (tech_map.get(s) or {}).get('traded', 0)
        if w and len(w[0]) >= 5 and w[0] not in STOP_W and tv >= 5e8:  # single-word names only for big, liquid companies
            single[w[0]] = s
    names = sym_ok
    cutoff = (NOW.date() - dt.timedelta(days=21)).isoformat()
    cnt, themes = {}, {k: 0 for k in BUZZ_THEMES}
    for v in vids:
        if v['d'] < cutoff:
            continue
        low = v['t'].lower()
        for k, rx in BUZZ_THEMES.items():
            if re.search(rx, low):
                themes[k] += 1
        found = set()
        for tok in re.findall(r'[A-Za-z&]{3,}', v['t']):
            if tok.isupper() and tok in sym_ok:
                found.add(sym_ok[tok])
            elif tok.lower() in single:
                found.add(single[tok.lower()])
        for ph, s2 in phrase.items():
            if ph in low:
                found.add(s2)
        for s in found:
            c = cnt.setdefault(s, {'s': s, 'n': 0, 'ch': set(), 'titles': []})
            c['n'] += 1
            c['ch'].add(v['ch'])
            if len(c['titles']) < 3:
                c['titles'].append(v['t'][:110])
    ment = []
    for s, c in sorted(cnt.items(), key=lambda kv: -kv[1]['n'])[:25]:
        lt = lt_map.get(s)
        verdict = None if lt is None else ('Data agrees' if lt >= 60 else 'Hype ahead of data' if lt < 45 else 'Mixed')
        ment.append({'s': s, 'n': c['n'], 'ch': sorted(c['ch']), 'titles': c['titles'], 'lt': lt, 'verdict': verdict})
    vids.sort(key=lambda x: x['d'], reverse=True)
    return {'channels': chans, 'videos': vids[:80], 'mentions': ment, 'themes': themes}


# ---------------------------------------------------------------- demo data
def demo_series(sym, n=760, base=None):
    rnd = random.Random(sym)
    base = base or rnd.uniform(50, 3000)
    p, rows, drift = base, [], 0
    dates = pd.bdate_range(end=pd.Timestamp(NOW.date()), periods=n)
    for i in range(n):
        if i % 60 == 0:
            drift = rnd.uniform(-0.0012, 0.0018)
        o = p
        c = o * (1 + drift + rnd.gauss(0, 0.017))
        h, l = max(o, c) * (1 + abs(rnd.gauss(0, 0.006))), min(o, c) * (1 - abs(rnd.gauss(0, 0.006)))
        rows.append((o, h, l, c, rnd.uniform(1e5, 3e6)))
        p = c
    return pd.DataFrame(rows, index=dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])


def demo_fund(s):
    r = random.Random(s + 'f')
    pe = r.uniform(8, 70)
    return {'mcap': r.uniform(500, 300000) * 1e7, 'pe': pe, 'pb': r.uniform(0.8, 12), 'roe': r.uniform(4, 32), 'de': r.uniform(0, 1.8), 'cr': r.uniform(0.8, 3),
            'div': r.uniform(0, 3), 'prom': r.uniform(20, 75), 'roce': r.uniform(5, 38), 'iroce': r.uniform(-5, 40), 'sales3': r.uniform(-3, 30), 'profit3': r.uniform(-10, 45),
            'profit_last': r.uniform(-20, 60), 'opm': r.uniform(4, 30), 'opm_old': r.uniform(4, 30), 'fcf': r.uniform(-200, 2000) * 1e7, 'fcf_years': 4, 'fcf_pos_years': r.randint(1, 4),
            'cfo_pbt': r.uniform(0.3, 1.4), 'inv_days': r.uniform(10, 200), 'debt_days': r.uniform(15, 140), 'debt_days_old': r.uniform(15, 140), 'wc_trend': r.uniform(-10, 12),
            'pe_med3': pe * r.uniform(0.7, 1.4), 'peg': pe / max(1, r.uniform(5, 40)), 'icov': r.uniform(1.5, 30), 'dilution': r.uniform(0, 6), 'sector': None, 'industry': None,
            'res': (NOW.date() + dt.timedelta(days=r.randint(5, 80))).isoformat(), 'name': None}


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    if DEMO:
        uni = [{'s': s, 'n': n, 'sec': sec} for s, n, sec in [
            ('RELIANCE', 'Reliance Industries Limited', 'Oil Gas & Consumable Fuels'), ('TCS', 'Tata Consultancy Services Limited', 'Information Technology'),
            ('HDFCBANK', 'HDFC Bank Limited', 'Financial Services'), ('TBZ', 'Tribhovandas Bhimji Zaveri Limited', 'Consumer Durables'), ('TITAN', 'Titan Company Limited', 'Consumer Durables'),
            ('HAL', 'Hindustan Aeronautics Limited', 'Capital Goods'), ('BEL', 'Bharat Electronics Limited', 'Capital Goods'), ('INFY', 'Infosys Limited', 'Information Technology'),
            ('ITC', 'ITC Limited', 'Fast Moving Consumer Goods'), ('LT', 'Larsen & Toubro Limited', 'Construction'), ('POLYCAB', 'Polycab India Limited', 'Capital Goods'),
            ('KALYANKJIL', 'Kalyan Jewellers India Limited', 'Consumer Durables'), ('SUNPHARMA', 'Sun Pharmaceutical Industries Limited', 'Healthcare'),
            ('TATASTEEL', 'Tata Steel Limited', 'Metals & Mining'), ('NTPC', 'NTPC Limited', 'Power')] + [(f'DEMO{i}', f'Demo Company {i} Limited', 'Capital Goods') for i in range(40)]]
        prices = {u['s']: demo_series(u['s']) for u in uni}
    else:
        uni = get_universe()
        prices = download_prices([u['s'] for u in uni])
    log(f'universe {len(uni)}, prices {len(prices)} ({time.time() - t0:.0f}s)')
    nifty = demo_series('^NSEI', base=26000) if DEMO else yseries('^NSEI', '3y')

    # technicals for all
    T = {}
    for s, d in prices.items():
        try:
            T[s] = tech(d)
        except Exception as e:
            log('WARN tech', s, e)
    # relative strength percentile (blend of 6m and 12m returns)
    rs_raw = pd.Series({s: 0.4 * (t['r']['m6'] or 0) + 0.6 * (t['r']['y1'] if t['r']['y1'] is not None else (t['r']['m6'] or 0)) for s, t in T.items()})
    rs_pct = pct_rank(rs_raw).to_dict() if len(rs_raw) else {}

    # fundamentals: rotating cache
    fc = load('fund_cache.json', {})
    order = sorted(T.keys(), key=lambda s: (fc.get(s, {}).get('at', ''), -T[s]['traded']))
    todo = [s for s in order if T[s]['traded'] >= 5e6][:FUND_BATCH]  # liquid stocks first (₹50 lakh+/day)
    log(f'fundamentals: refreshing {len(todo)} (cached {len(fc)})')
    for i, s in enumerate(todo):
        try:
            f = demo_fund(s) if DEMO else fetch_fund(s, prices.get(s))
            fc[s] = {'at': NOW.date().isoformat(), 'f': f}
        except Exception as e:
            log('WARN fund', s, e)
        if not DEMO:
            time.sleep(0.35)
        if i % 50 == 0:
            log(f'  fund {i}/{len(todo)} ({time.time() - t0:.0f}s)')
            save('fund_cache.json', fc)
        if time.time() - t0 > 60 * 95:  # keep the job well under the 6h limit and polite to the data source
            log('time budget reached, continuing next run')
            break
    save('fund_cache.json', fc)

    themes = build_themes(prices, DEMO)
    names = {u['s']: u['n'] for u in uni}
    secs = {u['s']: u['sec'] for u in uni}

    stocks, lt_map = [], {}
    for s, t in T.items():
        try:
            f = (fc.get(s) or {}).get('f')
            sec = secs.get(s) or (f or {}).get('sector') or ''
            tid = theme_of(sec, (f or {}).get('industry') or '', s)
            th = theme_heat_for(tid, themes) if tid else 50
            S, W = comp_scores(t, f, rs_pct.get(s, 50), th)
            if tid:
                W['theme'] = f'Theme: {next(x["name"] for x in THEMES if x["id"] == tid)} ({next(x["status"] for x in themes if x["id"] == tid)})'
            ck = checklist(f, t, sec) if f else None
            hz, why, risk, plan = {}, {}, {}, {}
            for h, w in HZ.items():
                if h in NEEDS_FUND and 'quality' not in S:
                    continue
                tot = sum(w[k] for k in w if k in S)
                sc = sum(S[k] * w[k] for k in w if k in S) / tot
                if h in NEEDS_FUND and ck and ck['score'] is not None:
                    sc = sc * 0.75 + ck['score'] * 0.25
                if len(ck['flags'] if ck else []) >= 2 and h in NEEDS_FUND:
                    sc -= 8
                hz[h] = round(clamp(sc))
                contrib = sorted([(S[k] * w[k], k) for k in w if k in S], reverse=True)
                why[h] = [W[k] for _, k in contrib if S[k] >= 60 and W.get(k)][:3]
                risk[h] = [W[k] for k in sorted(w, key=lambda k: S.get(k, 100)) if k in S and S[k] <= 42 and W.get(k)][:2]
                plan[h] = horizon_plan(h, t, f)
            lt_map[s] = hz.get('1Y') or hz.get('6M') or hz.get('3M')
            mb, mbw = 0, []
            if f:
                if (f.get('profit_last') or 0) > 25 and (f.get('profit_last') or 0) > (f.get('profit3') or 0) + 5:
                    mb += 22; mbw.append(f'Profit growth accelerating ({f["profit_last"]:.0f}% last year vs {(f.get("profit3") or 0):.0f}%/yr over 3Y)')
                if (f.get('opm') or 0) > (f.get('opm_old') or 99) + 2:
                    mb += 12; mbw.append('Margins expanding')
                if (f.get('de') or 9) < 0.6:
                    mb += 8; mbw.append('Low debt')
                mc = (f.get('mcap') or 0) / 1e7
                if 300 < mc < 25000:
                    mb += 12; mbw.append(f'Small/mid size (₹{mc:,.0f} Cr) — room to grow')
                if (f.get('roce') or 0) > 18:
                    mb += 10; mbw.append('High ROCE')
            if S['breakout'] >= 72:
                mb += 20; mbw.append('Breaking out of a long base near the 52-week high')
            if rs_pct.get(s, 0) >= 85:
                mb += 10; mbw.append('Top-15% price strength on NSE')
            if t['volx'] > 1.6:
                mb += 6; mbw.append('Volume surging (big buyers stepping in?)')
            if tid and th >= 58:
                mb += 8; mbw.append(W['theme'])
            fl = {k: v for k, v in (f or {}).items() if k in KEEP_F} if f else None
            stocks.append({
                's': s, 'n': names.get(s) or (f or {}).get('name') or s, 'sec': sec, 'th': tid,
                'px': rd(t['px']), 'c1': rd(t['chg1d'], 1), 'r': {k: rd(v, 1) for k, v in t['r'].items()},
                'd50': rd(t['d50']), 'd200': rd(t['d200']), 'hi52': rd(t['hi52']), 'lo52': rd(t['lo52']), 'hi3': rd(t['hi3']), 'lo3': rd(t['lo3']),
                'rsi': rd(t['rsi'], 0), 'vol': rd(t['vol'], 0), 'dd': rd(t['dd'], 0), 'atrp': rd(t['atrp'], 2), 'tv': rd(t['traded'] / 1e7, 2), 'rs': rd(rs_pct.get(s, 50), 0),
                'hz': hz, 'ckS': ck['score'] if ck else None, 'lynch': ck['lynch'] if ck else None, 'nfl': len(ck['flags']) if ck else None,
                'mb': mb if f else None, 'res': (f or {}).get('res'), 'mcap': rd(f['mcap'] / 1e7, 0) if (f and f.get('mcap')) else None,
                'pe': (f or {}).get('pe'), 'beta': (f or {}).get('beta'),
                '_d': {'sp': t['spark'], 'why': why, 'risk': risk, 'plan': plan, 'S': {k: rd(v, 0) for k, v in S.items()}, 'f': fl, 'ck': ck, 'mbw': mbw},
            })
        except Exception as e:
            log('WARN stock', s, e)

    # picks per horizon: liquid (₹1 Cr+/day), price > ₹20
    picks = {}
    elig = [x for x in stocks if (x['tv'] or 0) >= 1 and (x['px'] or 0) > 20]
    for h in HZ:
        ranked = sorted([x for x in elig if h in x['hz']], key=lambda x: -x['hz'][h])
        picks[h] = [x['s'] for x in ranked[:20]]
    mbr = sorted([x for x in elig if x['mb']], key=lambda x: -x['mb'])[:15]

    # market regime
    regime = {}
    try:
        n = series_stats(nifty)
        above = np.mean([1 if (t['d200'] and t['px'] > t['d200']) else 0 for t in T.values()]) * 100
        regime = {'nifty': rd(n['px']), 'nifty_vs200': rd(n['vs200'], 1), 'nifty_y1': rd(n['y1'], 1), 'breadth': rd(above, 0),
                  'mood': 'bull' if above >= 60 and n['vs200'] > 0 else 'narrow' if (above < 45 and n['vs200'] > 0) else 'bear' if above < 40 else 'mixed'}
    except Exception as e:
        log('WARN regime', e)

    last_bar = max((d.index[-1] for d in prices.values()), default=pd.Timestamp(NOW.date())).date()
    meta = {'generated': NOW.isoformat(timespec='minutes'), 'asof': last_bar.isoformat(), 'buy_on': next_trading_day(last_bar).isoformat(),
            'universe': len(uni), 'priced': len(T), 'with_fund': sum(1 for x in stocks if x['ckS'] is not None or x['mcap']), 'regime': regime, 'demo': DEMO,
            'holidays_known_until': '2026-12-31', 'secs': round(time.time() - t0)}
    # core file (all stocks, small) + detail shards loaded on demand + full detail for picks
    pick_set = set(sum(picks.values(), [])) | {x['s'] for x in mbr}
    shards, detail_picks = {}, {}
    for x in stocks:
        d = x.pop('_d')
        shards.setdefault(shard_of(x['s']), {})[x['s']] = d
        if x['s'] in pick_set:
            detail_picks[x['s']] = d
    os.makedirs(os.path.join(DATA, 'd'), exist_ok=True)
    for k, v in shards.items():
        save(f'd/{k}.json', v)
    save('scan.json', {'meta': meta, 'stocks': stocks, 'picks': picks, 'mb': [x['s'] for x in mbr], 'detail': detail_picks})
    save('themes.json', {'meta': meta, 'themes': themes})
    save('events.json', {'meta': meta} | build_events(fc, names, DEMO))
    save('buzz.json', {'meta': meta} | build_buzz(uni, T, lt_map, DEMO))
    save('meta.json', meta | {'log': LOG[-60:]})
    log(f'done in {time.time() - t0:.0f}s: {len(stocks)} stocks, picks {", ".join(f"{h}:{len(v)}" for h, v in picks.items())}')


if __name__ == '__main__':
    main()
