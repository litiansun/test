"""
USD/JPY, US 30Y Treasury Yield, Gold Futures, WTI Crude Oil — 5-year daily chart.
Data loaded from investing.com CSV exports.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

USDJPY_CSV  = ("/root/.claude/uploads/60eb40a6-d9ec-4243-9e7b-8e049c70b9c9/"
               "7f9e7fb8-USD_JPY_Historical_Data.csv")
TY30_CSV    = ("/root/.claude/uploads/60eb40a6-d9ec-4243-9e7b-8e049c70b9c9/"
               "d0d5b067-United_States_30Year_Bond_Yield_Historical_Data.csv")
GOLD_CSV    = ("/root/.claude/uploads/151409b9-499a-4db8-80ec-25b67d3daddd/"
               "14dae4a6-Gold_Futures_Historical_Data.csv")
OIL_CSV     = ("/root/.claude/uploads/151409b9-499a-4db8-80ec-25b67d3daddd/"
               "d3488827-Crude_Oil_WTI_Futures_Historical_Data.csv")

def load_investing_csv(path, date_col, price_col, date_fmt=None):
    """Load an investing.com CSV; returns a date-indexed price Series."""
    df = pd.read_csv(path, thousands=',')
    df[date_col] = pd.to_datetime(df[date_col], format=date_fmt)
    s = df.set_index(date_col)[price_col].astype(float).sort_index()
    return s

# USD/JPY: Japanese headers already renamed in earlier session, re-load cleanly
usdjpy_raw = pd.read_csv(USDJPY_CSV)
usdjpy_raw.columns = ['date', 'close', 'open', 'high', 'low', 'volume', 'change']
usdjpy_raw['date'] = pd.to_datetime(usdjpy_raw['date'])
usdjpy = usdjpy_raw.set_index('date')['close'].astype(float).sort_index()
usdjpy.name = "USD/JPY"

ty30  = load_investing_csv(TY30_CSV,  "Date", "Price", "%m/%d/%Y")
ty30.name = "US 30Y Yield (%)"

gold  = load_investing_csv(GOLD_CSV,  "Date", "Price", "%m/%d/%Y")
gold.name = "Gold (USD/oz)"

oil   = load_investing_csv(OIL_CSV,   "Date", "Price", "%m/%d/%Y")
oil.name = "WTI Crude (USD/bbl)"

# ── align on common dates ─────────────────────────────────────────────────────
df = pd.concat([usdjpy, ty30, gold, oil], axis=1)
df.columns = ["USD/JPY", "US 30Y Yield (%)", "Gold (USD/oz)", "WTI Crude (USD/bbl)"]
df = df.dropna()

# ── range summary ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  Period       : {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Trading days : {len(df)}")
print("=" * 60)
for col in df.columns:
    lo, hi  = df[col].min(), df[col].max()
    latest  = df[col].iloc[-1]
    unit    = "%" if "Yield" in col else ""
    print(f"\n  {col}")
    print(f"    Min    : {lo:.3f}{unit}")
    print(f"    Max    : {hi:.3f}{unit}")
    print(f"    Range  : {hi - lo:.3f}{unit}")
    print(f"    Latest : {latest:.3f}{unit}")
print("=" * 60)

# ── normalise 0-1 ─────────────────────────────────────────────────────────────
df_norm = (df - df.min()) / (df.max() - df.min())

# ── detect notable peaks and troughs ─────────────────────────────────────────
def get_extrema(series, prominence_frac=0.12):
    arr  = series.values
    prom = (arr.max() - arr.min()) * prominence_frac
    peaks,   _ = find_peaks( arr, prominence=prom, distance=30)
    troughs, _ = find_peaks(-arr, prominence=prom, distance=30)
    peaks   = np.union1d(peaks,   [arr.argmax()])
    troughs = np.union1d(troughs, [arr.argmin()])
    return peaks, troughs

extrema = {col: get_extrema(df[col]) for col in df.columns}

# ── colours ───────────────────────────────────────────────────────────────────
COLORS = {
    "USD/JPY":            "#1565C0",   # blue
    "US 30Y Yield (%)":   "#C62828",   # red
    "Gold (USD/oz)":      "#E65100",   # deep orange / gold
    "WTI Crude (USD/bbl)":"#2E7D32",   # dark green
}

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))

for col, color in COLORS.items():
    ax.fill_between(df_norm.index, df_norm[col], alpha=0.05, color=color)
    ax.plot(df_norm.index, df_norm[col], color=color, linewidth=1.3, label=col)

# ── annotate extrema ─────────────────────────────────────────────────────────
def fmt_value(col, val):
    if "Yield" in col:
        return f"{val:.2f}%"
    elif "JPY" in col:
        return f"{val:.2f}"
    else:
        return f"{val:,.0f}"

def annotate_extrema(ax, df_orig, df_norm, idxs, col, color, is_peak):
    for i in idxs:
        date   = df_orig.index[i]
        orig   = df_orig[col].iloc[i]
        norm_y = df_norm[col].iloc[i]
        label  = f"{fmt_value(col, orig)}\n{date.strftime('%b %y')}"
        va     = "bottom" if is_peak else "top"
        ax.plot(date, norm_y, marker="^" if is_peak else "v",
                ms=5, color=color, zorder=5)
        ax.annotate(label, xy=(date, norm_y),
                    xytext=(0, 18 if is_peak else -18),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=7, color=color,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color,
                                   lw=0.7, alpha=0.6))

for col, color in COLORS.items():
    peaks, troughs = extrema[col]
    annotate_extrema(ax, df, df_norm, peaks,   col, color, True)
    annotate_extrema(ax, df, df_norm, troughs, col, color, False)

# reference lines
ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
ax.axhline(1, color="grey", linewidth=0.4, linestyle=":")

# year separators
for yr in range(2022, 2027):
    ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="grey",
               linewidth=0.6, linestyle="--", alpha=0.35)

ax.set_ylim(-0.30, 1.38)
ax.set_ylabel("Min-max normalised  (0 = period low, 1 = period high)", fontsize=10)

# x-axis: tick every 3 months
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.grid(axis="x", which="major", alpha=0.15, linestyle=":")

# range label boxes (stacked top-left)
box_y = 0.998
for col, color in COLORS.items():
    lo, hi = df[col].min(), df[col].max()
    tag = fmt_value(col, lo) + " – " + fmt_value(col, hi) + "  (Δ" + fmt_value(col, hi - lo) + ")"
    ax.text(0.005, box_y,
            f"{col}  range: {tag}",
            transform=ax.transAxes, fontsize=8.5, va="top",
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.10))
    box_y -= 0.068

ax.set_title(
    "USD/JPY  ·  US 30Y Yield  ·  Gold  ·  WTI Crude  —  5-Year Daily  (min-max normalised)\n"
    f"(investing.com data, {df.index[0].strftime('%b %Y')} – {df.index[-1].strftime('%b %Y')})",
    fontsize=12, fontweight="bold", pad=10
)
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.88)

plt.tight_layout()
out = "/home/user/test/usdjpy_vs_30y.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved → {out}")
