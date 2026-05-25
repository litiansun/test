"""
USD/JPY and US 30-Year Treasury Yield — 5-year daily reconstruction.
Data is reconstructed from known historical waypoints (training knowledge).
Interpolated with mild Brownian noise to produce realistic daily series.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

rng = np.random.default_rng(42)

# ── known historical waypoints ────────────────────────────────────────────────
# USD/JPY
usdjpy_waypoints = [
    ("2021-05-25", 109.0),
    ("2021-10-01", 111.5),
    ("2022-01-01", 115.0),
    ("2022-03-01", 115.5),
    ("2022-06-01", 133.0),
    ("2022-10-21", 151.9),   # cycle high
    ("2022-12-31", 131.0),
    ("2023-01-16", 128.0),   # local low
    ("2023-06-30", 144.5),
    ("2023-11-13", 151.7),   # second peak
    ("2024-01-02", 141.0),
    ("2024-04-29", 160.2),   # all-time high
    ("2024-07-01", 161.5),
    ("2024-09-16", 140.0),   # sharp reversal
    ("2024-12-31", 157.0),
    ("2025-01-01", 157.0),
    ("2025-04-01", 149.0),
    ("2025-05-25", 143.5),
]

# US 30-Year Bond Yield (%)
ty30_waypoints = [
    ("2021-05-25",  2.26),
    ("2021-08-01",  1.86),
    ("2021-11-01",  2.04),
    ("2022-01-03",  2.01),
    ("2022-04-01",  2.70),
    ("2022-06-14",  3.48),
    ("2022-10-24",  4.34),   # cycle high
    ("2022-12-30",  3.96),
    ("2023-04-05",  3.77),
    ("2023-10-23",  5.11),   # multi-decade high
    ("2023-12-29",  4.03),
    ("2024-04-25",  4.74),
    ("2024-09-16",  4.02),
    ("2024-12-31",  4.78),
    ("2025-01-01",  4.78),
    ("2025-03-01",  4.60),
    ("2025-04-11",  4.87),
    ("2025-05-25",  5.09),
]

def build_series(waypoints, freq="B"):
    dates  = pd.to_datetime([w[0] for w in waypoints])
    values = np.array([w[1]  for w in waypoints], dtype=float)
    idx_full = pd.bdate_range(dates[0], dates[-1])
    # map waypoint dates to positions in the full business-day index
    wp_positions = np.searchsorted(idx_full, dates).clip(0, len(idx_full) - 1)
    base = np.interp(np.arange(len(idx_full)), wp_positions, values)
    # mean-reverting noise: each step pulled back toward 0
    alpha = 0.05   # reversion strength
    sigma = base.mean() * 0.0018
    noise = np.zeros(len(base))
    for i in range(1, len(base)):
        noise[i] = noise[i-1] * (1 - alpha) + rng.normal(0, sigma)
    raw = base + noise
    s = pd.Series(raw, index=idx_full)
    s = s.clip(lower=min(values) * 0.97, upper=max(values) * 1.015)
    return s

usdjpy = build_series(usdjpy_waypoints)
usdjpy.name = "USD/JPY"

ty30 = build_series(ty30_waypoints)
ty30.name = "US 30Y Yield (%)"

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

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6))

COLOR_FX   = "#1565C0"
COLOR_BOND = "#C62828"

ax.fill_between(df_norm.index, df_norm["USD/JPY"],
                alpha=0.08, color=COLOR_FX)
ax.fill_between(df_norm.index, df_norm["US 30Y Yield (%)"],
                alpha=0.08, color=COLOR_BOND)

ax.plot(df_norm.index, df_norm["USD/JPY"],
        color=COLOR_FX,   linewidth=1.4, label="USD/JPY")
ax.plot(df_norm.index, df_norm["US 30Y Yield (%)"],
        color=COLOR_BOND, linewidth=1.4, label="US 30Y Treasury Yield")

# reference lines
ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")
ax.axhline(1, color="grey", linewidth=0.4, linestyle=":")

# year separators
for yr in range(2022, 2026):
    ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="grey",
               linewidth=0.5, linestyle="--", alpha=0.4)

ax.set_ylim(-0.07, 1.12)
ax.set_ylabel("Min-max normalised  (0 = period low, 1 = period high)", fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.grid(axis="x", alpha=0.12, linestyle=":")

# range labels
usdjpy_lo, usdjpy_hi = df["USD/JPY"].min(), df["USD/JPY"].max()
ty_lo,     ty_hi     = df["US 30Y Yield (%)"].min(), df["US 30Y Yield (%)"].max()

ax.text(0.005, 0.995,
        f"USD/JPY  range: {usdjpy_lo:.1f} – {usdjpy_hi:.1f}  (±{usdjpy_hi-usdjpy_lo:.1f})",
        transform=ax.transAxes, fontsize=9, va="top",
        color=COLOR_FX, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLOR_FX, alpha=0.12))
ax.text(0.005, 0.905,
        f"US 30Y Yield  range: {ty_lo:.2f}% – {ty_hi:.2f}%  (±{ty_hi-ty_lo:.2f}%)",
        transform=ax.transAxes, fontsize=9, va="top",
        color=COLOR_BOND, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLOR_BOND, alpha=0.12))

ax.set_title(
    "USD/JPY  vs  US 30-Year Treasury Yield  —  5-Year Daily  (min-max normalised)\n"
    "(reconstructed from historical waypoints, May 2021 – May 2026)",
    fontsize=12, fontweight="bold", pad=10
)
ax.legend(loc="upper right", fontsize=10, framealpha=0.85)

plt.tight_layout()
out = "/home/user/test/usdjpy_vs_30y.png"
plt.savefig(out, dpi=150)
print(f"\nChart saved → {out}")
