"""
Stable Market Board - daily data pull
--------------------------------------
Pulls recent daily bars from Yahoo Finance (via yfinance), computes breadth,
theme scores, momentum, RVOL and extension stats, and writes data.json.

This is meant to run server-side (e.g. GitHub Actions), not in a browser --
Yahoo's endpoints don't send CORS headers, so a browser-based fetch will fail.

Run: python fetch_data.py
Requires: pip install yfinance pandas numpy
"""
import json
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Universe: ticker -> theme. Trim/expand this list freely -- it drives both
# the breadth stats and the theme scores. Keep it a few hundred tickers max
# or the daily pull gets slow / rate-limited.
# ---------------------------------------------------------------------------
UNIVERSE = {
    # Robotics / Automation
    "ISRG": "Robotics", "ROK": "Robotics", "ABB": "Robotics", "FANUY": "Robotics",
    "IRBT": "Robotics", "TER": "Robotics", "ZBRA": "Robotics", "CGNX": "Robotics",
    # Software Infrastructure
    "MSFT": "Software Infrastructure", "CRM": "Software Infrastructure", "NOW": "Software Infrastructure",
    "SNOW": "Software Infrastructure", "DDOG": "Software Infrastructure", "MDB": "Software Infrastructure",
    "NET": "Software Infrastructure", "PANW": "Software Infrastructure", "CRWD": "Software Infrastructure",
    "FTNT": "Software Infrastructure",
    # Defensives
    "PG": "Defensives", "KO": "Defensives", "PEP": "Defensives", "WMT": "Defensives",
    "COST": "Defensives", "JNJ": "Defensives", "CL": "Defensives", "KMB": "Defensives",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "OXY": "Energy", "PXD": "Energy", "MPC": "Energy",
    # Biotech
    "LLY": "Biotech", "VRTX": "Biotech", "REGN": "Biotech", "AMGN": "Biotech",
    "BIIB": "Biotech", "GILD": "Biotech", "MRNA": "Biotech",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "WFC": "Financials", "SCHW": "Financials", "BLK": "Financials",
    # EV / Autonomy
    "TSLA": "EV/Autonomy", "RIVN": "EV/Autonomy", "LI": "EV/Autonomy", "NIO": "EV/Autonomy",
    "APTV": "EV/Autonomy", "MBLY": "EV/Autonomy",
    # China Tech
    "BABA": "China Tech", "PDD": "China Tech", "JD": "China Tech", "BIDU": "China Tech",
    "NTES": "China Tech", "TCEHY": "China Tech",
    # Consumer Momentum
    "AMZN": "Consumer Momentum", "MELI": "Consumer Momentum", "SHOP": "Consumer Momentum",
    "CMG": "Consumer Momentum", "LULU": "Consumer Momentum", "ULTA": "Consumer Momentum",
    # Datacenter REITs
    "DLR": "Datacenter REITs", "EQIX": "Datacenter REITs", "AMT": "Datacenter REITs",
    # Semiconductors
    "NVDA": "Semiconductors", "AVGO": "Semiconductors", "AMD": "Semiconductors",
    "TSM": "Semiconductors", "MU": "Semiconductors", "ASML": "Semiconductors",
    "QCOM": "Semiconductors", "ARM": "Semiconductors",
    # Optics
    "COHR": "Optics", "LITE": "Optics", "IIVI": "Optics",
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "HON": "Industrials", "GE": "Industrials",
    # Media & Streaming
    "NFLX": "Media & Streaming", "DIS": "Media & Streaming", "ROKU": "Media & Streaming",
    "WBD": "Media & Streaming",
    # Housing
    "DHI": "Housing", "LEN": "Housing", "PHM": "Housing", "HD": "Housing",
    # Regional Banks
    "KRE": "Regional Banks", "ZION": "Regional Banks", "CFG": "Regional Banks", "RF": "Regional Banks",
    # Cybersecurity
    "ZS": "Cybersecurity", "OKTA": "Cybersecurity", "S": "Cybersecurity",
    # Gold Miners
    "GDX": "Gold Miners", "NEM": "Gold Miners", "GOLD": "Gold Miners",
    # Shipping
    "ZIM": "Shipping", "MATX": "Shipping",
    # Agriculture
    "ADM": "Agriculture", "BG": "Agriculture", "DE": "Agriculture",
}

INDEXES = {"SPY": "SPX", "QQQ": "NDX", "IWM": "R2K", "RSP": "EQUAL WEIGHT SPX", "DIA": "DJIA"}
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "KRE", "IBB", "ARKK", "TAN"]

ALL_TICKERS = sorted(set(UNIVERSE) | set(INDEXES) | set(SECTOR_ETFS))


def pct(a, b):
    """percent change from b to a"""
    if b == 0 or pd.isna(b) or pd.isna(a):
        return 0.0
    return (a / b - 1.0) * 100.0


def download_history(tickers, period="4mo"):
    df = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    return df


def stats_for_ticker(hist):
    """hist: DataFrame with Close/Volume columns, most recent row last."""
    hist = hist.dropna(subset=["Close"])
    if len(hist) < 25:
        return None
    close = hist["Close"]
    vol = hist["Volume"]

    def snapshot(upto):
        c = close.iloc[:upto]
        v = vol.iloc[:upto]
        if len(c) < 21:
            return None
        last = c.iloc[-1]
        sma20 = c.tail(20).mean()
        sma50 = c.tail(50).mean() if len(c) >= 50 else c.mean()
        # true-range based ATR(14) approximation using close-to-close moves
        tr = c.diff().abs()
        atr = tr.tail(14).mean() or 1e-9
        ret1 = pct(last, c.iloc[-2]) if len(c) >= 2 else 0.0
        ret5 = pct(last, c.iloc[-6]) if len(c) >= 6 else 0.0
        ret20 = pct(last, c.iloc[-21]) if len(c) >= 21 else 0.0
        avgvol20 = v.tail(20).mean() or 1e-9
        rvol = (v.iloc[-1] / avgvol20) if avgvol20 else 1.0
        dist50 = pct(last, sma50)
        atr_ext = (last - sma50) / atr if atr else 0.0
        above20 = last > sma20
        above50 = last > sma50
        hist15 = [round(float(x), 2) for x in c.tail(22).tolist()]
        # daily series (last 22 rows) used for theme score history
        sma20_s = close.rolling(20, min_periods=10).mean()
        sma50_s = close.rolling(50, min_periods=20).mean()
        above20_s = (close > sma20_s).tail(22)
        above50_s = (close > sma50_s).tail(22)
        ret5_s = (close.pct_change(5) * 100).tail(22)
        return dict(last=last, ret1=ret1, ret5=ret5, ret20=ret20, rvol=rvol,
                    dist50=dist50, atr_ext=atr_ext, above20=above20, above50=above50,
                    hist15=hist15,
                    above20_series=above20_s, above50_series=above50_s, ret5_series=ret5_s)

    today = snapshot(len(close))
    yesterday = snapshot(len(close) - 1)
    if today is None:
        return None
    today["prev"] = yesterday
    return today


def build():
    print(f"Downloading {len(ALL_TICKERS)} tickers...")
    raw = download_history(ALL_TICKERS)

    per_ticker = {}
    for t in ALL_TICKERS:
        try:
            h = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            s = stats_for_ticker(h)
            if s:
                per_ticker[t] = s
        except Exception as e:
            print(f"  skip {t}: {e}")

    total = len(per_ticker)
    above20 = sum(1 for s in per_ticker.values() if s["above20"])
    above50 = sum(1 for s in per_ticker.values() if s["above50"])
    # 200dma not computed (needs longer history window); approximate w/ 50dma pool for now
    above200 = above50
    up3 = sum(1 for s in per_ticker.values() if s["ret1"] >= 3)
    down3 = sum(1 for s in per_ticker.values() if s["ret1"] <= -3)
    new_hi20 = sum(1 for t, s in per_ticker.items() if s["ret20"] > 0 and s["dist50"] > 8)
    new_hi52 = sum(1 for s in per_ticker.values() if s["ret20"] > 15)
    new_lows = sum(1 for s in per_ticker.values() if s["ret5"] < -5)

    pct20 = 100 * above20 / total if total else 0
    pct50 = 100 * above50 / total if total else 0
    pct200 = 100 * above200 / total if total else 0

    regime_label = "RISK-OFF" if pct50 < 45 else ("RISK-ON" if pct50 > 65 else "NEUTRAL")

    # theme scores
    theme_groups = {}
    for t, theme in UNIVERSE.items():
        if t in per_ticker:
            theme_groups.setdefault(theme, []).append(per_ticker[t])

    def theme_score(members):
        if not members:
            return 0, 0
        pct_above50 = 100 * sum(1 for m in members if m["above50"]) / len(members)
        pct_above20 = 100 * sum(1 for m in members if m["above20"]) / len(members)
        avg_ret5 = np.mean([m["ret5"] for m in members])
        score = 0.5 * pct_above50 + 0.3 * pct_above20 + 0.2 * min(max(avg_ret5 * 10 + 50, 0), 100)
        # yesterday version using prev snapshot
        prev_members = [m["prev"] for m in members if m.get("prev")]
        if prev_members:
            p_above50 = 100 * sum(1 for m in prev_members if m["above50"]) / len(prev_members)
            p_above20 = 100 * sum(1 for m in prev_members if m["above20"]) / len(prev_members)
            p_avg_ret5 = np.mean([m["ret5"] for m in prev_members])
            prev_score = 0.5 * p_above50 + 0.3 * p_above20 + 0.2 * min(max(p_avg_ret5 * 10 + 50, 0), 100)
        else:
            prev_score = score
        return round(score), round(score - prev_score, 1)

    # keep ticker symbol alongside stats so we can build per-theme constituent lists
    theme_members_with_ticker = {}
    for t, theme in UNIVERSE.items():
        if t in per_ticker:
            theme_members_with_ticker.setdefault(theme, []).append((t, per_ticker[t]))

    themes = []
    for theme, members in theme_groups.items():
        score, delta = theme_score(members)
        if score > 85:
            status = "DOMINANT"
        elif score < 45:
            status = "DETERIORATING" if delta < 0 else "WEAK"
        elif delta < -6:
            status = "FADING"
        else:
            status = "STRONG"
        constituents = [
            dict(t=t, ret1=round(s["ret1"], 2), ret5=round(s["ret5"], 2), dist50=round(s["dist50"], 1))
            for t, s in theme_members_with_ticker.get(theme, [])
        ]
        constituents.sort(key=lambda x: -x["ret1"])

        # score history over the last ~22 trading days
        score_hist = []
        pairs = theme_members_with_ticker.get(theme, [])
        if pairs:
            a20 = pd.concat([s["above20_series"] for _, s in pairs], axis=1)
            a50 = pd.concat([s["above50_series"] for _, s in pairs], axis=1)
            r5 = pd.concat([s["ret5_series"] for _, s in pairs], axis=1)
            for i in range(len(a50)):
                p50 = 100 * a50.iloc[i].mean()
                p20 = 100 * a20.iloc[i].mean()
                ar5 = r5.iloc[i].mean()
                if pd.isna(ar5):
                    ar5 = 0.0
                sc = 0.5 * p50 + 0.3 * p20 + 0.2 * min(max(ar5 * 10 + 50, 0), 100)
                score_hist.append(round(float(sc)))
        themes.append(dict(name=theme, count=len(members), score=score, delta=delta, status=status,
                            constituents=constituents, scoreHist=score_hist))
    themes.sort(key=lambda x: -x["score"])

    # date labels for the score-history chart (use SPY's index as reference)
    chart_dates = []
    try:
        spy_hist = raw["SPY"] if isinstance(raw.columns, pd.MultiIndex) else raw
        chart_dates = [d.strftime("%m/%d") for d in spy_hist.dropna(subset=["Close"]).tail(22).index]
    except Exception:
        pass

    idx_out = []
    for t, label in INDEXES.items():
        s = per_ticker.get(t)
        if s:
            idx_out.append(dict(t=t, s=label, d1=round(s["ret1"], 2), d5=round(s["ret5"], 2),
                                 dist50=round(s["dist50"], 1), atrExt=round(s["atr_ext"], 1)))

    etf_out = []
    for t in SECTOR_ETFS:
        s = per_ticker.get(t)
        if s:
            etf_out.append(dict(t=t, d1=round(s["ret1"], 2), d5=round(s["ret5"], 2), rvol=round(s["rvol"], 2),
                                 hist=s.get("hist15", [])))

    momentum = sorted(
        [dict(t=t, d1=round(s["ret1"], 2), d20=round(s["ret20"], 2), rvol=round(s["rvol"], 2))
         for t, s in per_ticker.items() if t in UNIVERSE],
        key=lambda x: -x["d20"])[:15]

    rvol_tbl = sorted(
        [dict(t=t, rvol=round(s["rvol"], 2), d1=round(s["ret1"], 2))
         for t, s in per_ticker.items() if t in UNIVERSE],
        key=lambda x: -x["rvol"])[:15]

    ext_tbl = sorted(
        [dict(t=t, extAtr=round(s["atr_ext"], 1), pctFrom50=round(s["dist50"], 1))
         for t, s in per_ticker.items() if t in UNIVERSE],
        key=lambda x: -abs(x["extAtr"]))[:15]

    data = dict(
        asOf=dt.date.today().isoformat(),
        totalNames=total,
        regime=dict(label=regime_label, cls="risk-off" if regime_label == "RISK-OFF" else ("risk-on" if regime_label == "RISK-ON" else "")),
        pct20=round(pct20, 1), pct50=round(pct50, 1), pct200=round(pct200, 1),
        newHi20=new_hi20, newHi52=new_hi52, newHighs=new_hi20, newLows=new_lows,
        up3=up3, down3=down3,
        chartDates=chart_dates,
        themes=themes, idx=idx_out, etfs=etf_out,
        momentum=momentum, rvolTbl=rvol_tbl, ext=ext_tbl,
    )

    def sanitize(obj):
        if isinstance(obj, float):
            return None if (np.isnan(obj) or np.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    data = sanitize(data)

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    print(f"Wrote data.json -- {total} tickers, regime={regime_label}")


if __name__ == "__main__":
    build()
