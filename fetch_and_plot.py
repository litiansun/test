"""
USD/JPY and US 30-Year Treasury Yield — 5-year daily chart.
Data loaded from investing.com CSV exports.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

# ── load USD/JPY (Japanese headers, YYYY-MM-DD dates) ────────────────────────
usdjpy_raw = pd.read_csv(
    "/root/.claude/uploads/60eb40a6-d9ec-4243-9e7b-8e049c70b9c9/"
    "7f9e7fb8-USD_JPY_Historical_Data.csv"
)
usdjpy_raw.columns = ['date', 'close', 'open', 'high', 'low', 'volume', 'change']
usdjpy_raw['date'] = pd.to_datetime(usdjpy_raw['date'])
usdjpy = usdjpy_raw.set_index('date')['close'].astype(float).sort_index()
usdjpy.name = "USD/JPY"

# ── load US 30Y Yield (English headers, MM/DD/YYYY dates) ────────────────────
ty30_raw = pd.read_csv(
    "/root/.claude/uploads/60eb40a6-d9ec-4243-9e7b-8e049c70b9c9/"
    "d0d5b067-United_States_30Year_Bond_Yield_Historical_Data.csv"
)
ty30_raw.columns = ['date', 'price', 'open', 'high', 'low', 'change']
ty30_raw['date'] = pd.to_datetime(ty30_raw['date'], format='%m/%d/%Y')
ty30 = ty30_raw.set_index('date')['price'].astype(float).sort_index()
ty30.name = "US 30Y Yield (%)"

# ── align on common dates ─────────────────────────────────────────────────────
df = pd.concat([usdjpy, ty30], axis=1).dropna()
df.columns = ["USD/JPY", "US 30Y Yield (%)"]

# ── range summary ─────────────────────────────────────────────────────────────
print("=" * 56)
print(f"  Period : {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Trading days : {len(df)}")
print("=" * 56)
for col in df.columns:
    lo, hi = df[col].min(), df[col].max()
    rng_val = hi - lo
    latest  = df[col].iloc[-1]
    unit = "%" if "Yield" in col else ""
    print(f"\n  {col}")
    print(f"    Min    : {lo:.3f}{unit}")
    print(f"    Max    : {hi:.3f}{unit}")
    print(f"    Range  : {rng_val:.3f}{unit}")
    print(f"    Latest : {latest:.3f}{unit}")
print("=" * 56)

# ── normalise 0-1 ─────────────────────────────────────────────────────────────
df_norm = (df - df.min()) / (df.max() - df.min())

# ── detect notable peaks and troughs ─────────────────────────────────────────
def get_extrema(series, prominence_frac=0.12):
    arr = series.values
    prom = (arr.max() - arr.min()) * prominence_frac
    peaks,   _ = find_peaks( arr, prominence=prom, distance=30)
    troughs, _ = find_peaks(-arr, prominence=prom, distance=30)
    peaks   = np.union1d(peaks,   [arr.argmax()])
    troughs = np.union1d(troughs, [arr.argmin()])
    return peaks, troughs

fx_peaks,   fx_troughs   = get_extrema(df["USD/JPY"])
bond_peaks, bond_troughs = get_extrema(df["US 30Y Yield (%)"])

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7))

COLOR_FX   = "#1565C0"
COLOR_BOND = "#C62828"

ax.fill_between(df_norm.index, df_norm["USD/JPY"],
                alpha=0.07, color=COLOR_FX)
ax.fill_between(df_norm.index, df_norm["US 30Y Yield (%)"],
                alpha=0.07, color=COLOR_BOND)

ax.plot(df_norm.index, df_norm["USD/JPY"],
        color=COLOR_FX,   linewidth=1.4, label="USD/JPY")
ax.plot(df_norm.index, df_norm["US 30Y Yield (%)"],
        color=COLOR_BOND, linewidth=1.4, label="US 30Y Treasury Yield")

# ── annotate extrema ─────────────────────────────────────────────────────────
def annotate_extrema(ax, df_orig, df_norm, idxs, col, color, is_peak):
    for i in idxs:
        date   = df_orig.index[i]
        orig   = df_orig[col].iloc[i]
        norm_y = df_norm[col].iloc[i]
        unit   = "%" if "Yield" in col else ""
        label  = f"{orig:.2f}{unit}\n{date.strftime('%b %y')}"
        va     = "bottom" if is_peak else "top"
        ax.plot(date, norm_y, marker="^" if is_peak else "v",
                ms=5, color=color, zorder=5)
        ax.annotate(label, xy=(date, norm_y),
                    xytext=(0, 18 if is_peak else -18),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=7.5, color=color,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color,
                                   lw=0.7, alpha=0.6))

annotate_extrema(ax, df, df_norm, fx_peaks,    "USD/JPY",          COLOR_FX,   True)
annotate_extrema(ax, df, df_norm, fx_troughs,  "USD/JPY",          COLOR_FX,   False)
annotate_extrema(ax, df, df_norm, bond_peaks,  "US 30Y Yield (%)", COLOR_BOND, True)
annotate_extrema(ax, df, df_norm, bond_troughs,"US 30Y Yield (%)", COLOR_BOND, False)

# reference lines
ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
ax.axhline(1, color="grey", linewidth=0.4, linestyle=":")

# year separators
for yr in range(2022, 2027):
    ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="grey",
               linewidth=0.6, linestyle="--", alpha=0.35)

ax.set_ylim(-0.22, 1.28)
ax.set_ylabel("Min-max normalised  (0 = period low, 1 = period high)", fontsize=10)

# x-axis: tick every 3 months
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.grid(axis="x", which="major", alpha=0.15, linestyle=":")

# range labels
usdjpy_lo, usdjpy_hi = df["USD/JPY"].min(), df["USD/JPY"].max()
ty_lo,     ty_hi     = df["US 30Y Yield (%)"].min(), df["US 30Y Yield (%)"].max()

ax.text(0.005, 0.995,
        f"USD/JPY  range: {usdjpy_lo:.1f} – {usdjpy_hi:.1f}  (Δ{usdjpy_hi-usdjpy_lo:.1f})",
        transform=ax.transAxes, fontsize=9, va="top",
        color=COLOR_FX, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLOR_FX, alpha=0.12))
ax.text(0.005, 0.920,
        f"US 30Y Yield  range: {ty_lo:.2f}% – {ty_hi:.2f}%  (Δ{ty_hi-ty_lo:.2f}%)",
        transform=ax.transAxes, fontsize=9, va="top",
        color=COLOR_BOND, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLOR_BOND, alpha=0.12))

ax.set_title(
    "USD/JPY  vs  US 30-Year Treasury Yield  —  5-Year Daily  (min-max normalised)\n"
    f"(investing.com data, {df.index[0].strftime('%b %Y')} – {df.index[-1].strftime('%b %Y')})",
    fontsize=12, fontweight="bold", pad=10
)
ax.legend(loc="upper right", fontsize=10, framealpha=0.85)

plt.tight_layout()
out = "/home/user/test/usdjpy_vs_30y.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved → {out}")
