# ============================================================
#  CHOKEPOINT SHOCK — K&T Quant Labs
#  Quantitative Analysis of Strait of Hormuz Disruption
#  on Global Capital Markets and Commodity Trades
# ============================================================
#  Author  : K&T Quant Labs  |  Version : 2.1  |  2026
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import stats
from sklearn.mixture import GaussianMixture


# ──────────────────────────────────────────────────────────────
# 1.  GLOBAL STYLE — DARK GEOPOLITICAL THEME
# ──────────────────────────────────────────────────────────────
PALETTE = {
    "bg":         "#0A0D12",
    "panel":      "#10141C",
    "border":     "#1E2535",
    "text":       "#E8ECEF",
    "subtext":    "#7A8599",
    "accent1":    "#F5A623",   # amber  – crude oil
    "accent2":    "#E84040",   # red    – crisis / disruption
    "accent3":    "#3DD9B3",   # teal   – safe havens
    "accent4":    "#5B8CFF",   # blue   – equities
    "accent5":    "#C97BFF",   # purple – nat-gas
    "accent6":    "#FF7B54",   # orange – energy eq
    "calm":       "#3DD9B3",
    "stress":     "#F5A623",
    "disruption": "#E84040",
}

ASSET_COLORS = {
    "Brent":    PALETTE["accent1"],
    "WTI":      "#FFD166",
    "NatGas":   PALETTE["accent5"],
    "EnergyEq": PALETTE["accent6"],
    "Airlines": "#FF4D8D",
    "USD":      PALETTE["accent3"],
    "Gold":     "#F0C040",
    "SPY":      PALETTE["accent4"],
    "TLT":      "#74C0FC",
    "VIX":      PALETTE["accent2"],
}

FONT_TITLE = {"fontsize": 15, "fontweight": "bold",
              "color": PALETTE["text"], "fontfamily": "monospace"}
FONT_LABEL = {"fontsize": 10, "color": PALETTE["subtext"],
              "fontfamily": "monospace"}


def apply_dark_style():
    matplotlib.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["subtext"],
        "axes.titlecolor":   PALETTE["text"],
        "axes.grid":         True,
        "grid.color":        PALETTE["border"],
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,
        "xtick.color":       PALETTE["subtext"],
        "ytick.color":       PALETTE["subtext"],
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.facecolor":  "#151A24",
        "legend.edgecolor":  PALETTE["border"],
        "legend.labelcolor": PALETTE["text"],
        "legend.fontsize":   8,
        "text.color":        PALETTE["text"],
        "font.family":       "monospace",
        "lines.linewidth":   1.8,
        "figure.dpi":        150,
    })

apply_dark_style()


# ──────────────────────────────────────────────────────────────
# 2.  USER SETTINGS
# ──────────────────────────────────────────────────────────────
LOGO_PATH  = r"C:\Users\Niraj\Downloads\K & T Quant Labs.png"
START_DATE = "2024-01-01"
END_DATE   = None

TICKERS = {
    "Brent":    "BZ=F",
    "WTI":      "CL=F",
    "NatGas":   "NG=F",
    "EnergyEq": "XLE",
    "Airlines": "JETS",
    "USD":      "UUP",
    "Gold":     "GLD",
    "SPY":      "SPY",
    "TLT":      "TLT",
    "VIX":      "^VIX",
}

EVENT_DATE = "2026-03-15"
OUTPUT_DIR = r"C:\Users\Niraj\Downloads"   # all charts saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 3.  UTILITIES
# ──────────────────────────────────────────────────────────────
def to_series(col):
    """
    Safely coerce a DataFrame column (MultiIndex yfinance result)
    to a plain 1-D Series — fixes fill_between / plot TypeErrors.
    """
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0].squeeze()
    return col.squeeze()


def save_show(fig, filename, pad=1.8):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout(pad=pad)
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  [saved] {path}")


def add_logo(fig, logo_path=LOGO_PATH):
    """
    FIX: logo attached to the FIGURE level (not an axes),
    placed bottom-right at small size so it never overlaps data.
    [left, bottom, width, height] in figure-fraction coords.
    """
    if not os.path.exists(logo_path):
        return
    try:
        img     = plt.imread(logo_path)
        ax_logo = fig.add_axes([0.88, 0.01, 0.10, 0.055])
        ax_logo.imshow(img)
        ax_logo.axis("off")
        ax_logo.patch.set_alpha(0.0)
    except Exception as e:
        print(f"  [logo] {e}")


def watermark(ax, text="K&T QUANT LABS — RESEARCH"):
    ax.text(0.50, 0.50, text,
            transform=ax.transAxes,
            fontsize=22, color=PALETTE["border"],
            alpha=0.18, ha="center", va="center",
            rotation=30, fontweight="bold", fontfamily="monospace")


def add_event_vline(ax, event_date, label="Shock Event"):
    ev   = pd.to_datetime(event_date)
    ylim = ax.get_ylim()
    ax.axvline(ev, color=PALETTE["accent2"], linewidth=1.2,
               linestyle="--", zorder=5)
    ax.text(ev, ylim[1] - (ylim[1] - ylim[0]) * 0.04,
            f" ◀ {label}", color=PALETTE["accent2"],
            fontsize=7, va="top", fontfamily="monospace")


# ──────────────────────────────────────────────────────────────
# 4.  DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────
def download_market_data(tickers=TICKERS, start=START_DATE, end=END_DATE):
    print("  Downloading market data …")
    frames = []
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False)
            if df.empty:
                print(f"    [warn] No data: {name} ({ticker})")
                continue
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            s   = df[col].squeeze().copy()
            if isinstance(s, pd.DataFrame):   # flatten MultiIndex
                s = s.iloc[:, 0]
            s.name = name
            frames.append(s)
        except Exception as e:
            print(f"    [warn] {name}: {e}")

    if not frames:
        raise ValueError("No data downloaded.")

    data = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
    print(f"  ✓  {len(data.columns)} assets  |  "
          f"{data.index[0].date()} → {data.index[-1].date()}")
    return data


# ──────────────────────────────────────────────────────────────
# 5.  FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
def prepare_features(prices):
    returns = prices.pct_change().dropna()
    ft = pd.DataFrame(index=prices.index)

    if {"Brent", "WTI"}.issubset(prices.columns):
        ft["Brent_WTI_Spread"] = prices["Brent"] - prices["WTI"]

    if "Brent" in returns.columns:
        ft["Brent_Return"]  = returns["Brent"]
        ft["Brent_20D_Vol"] = returns["Brent"].rolling(20).std() * np.sqrt(252)
        ft["Brent_60D_Vol"] = returns["Brent"].rolling(60).std() * np.sqrt(252)
        ft["Brent_RSI_14"]  = _rsi(prices["Brent"], 14)

    if "SPY" in returns.columns and "Brent" in returns.columns:
        ft["Brent_SPY_60D_Corr"] = (
            returns["Brent"].rolling(60).corr(returns["SPY"]))

    if "VIX" in prices.columns:
        ft["VIX"]            = prices["VIX"]
        ft["VIX_30D_Change"] = prices["VIX"].pct_change(30)

    normed = prices / prices.iloc[0] * 100
    return returns, ft, normed


def _rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ──────────────────────────────────────────────────────────────
# 6.  RISK ANALYTICS
# ──────────────────────────────────────────────────────────────
def compute_risk_metrics(returns, confidence=0.95):
    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 20:
            continue
        var    = float(np.percentile(r, (1 - confidence) * 100))
        cvar   = float(r[r <= var].mean())
        sharpe = (float((r.mean() / r.std()) * np.sqrt(252))
                  if r.std() > 0 else np.nan)
        cum    = (1 + r).cumprod()
        peak   = cum.cummax()
        mdd    = float(((cum - peak) / peak).min())
        rows.append({
            "Asset":         col,
            f"VaR {int(confidence*100)}%":  round(var  * 100, 3),
            f"CVaR {int(confidence*100)}%": round(cvar * 100, 3),
            "Ann. Sharpe":   round(sharpe, 3),
            "Max Drawdown":  round(mdd * 100, 3),
            "Ann. Ret %":    round(r.mean() * 252 * 100, 2),
            "Ann. Vol %":    round(r.std()  * np.sqrt(252) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Asset")


def stress_scenario_returns(returns):
    if "Brent" not in returns.columns:
        return pd.Series(dtype=float)
    mask    = returns["Brent"] > 0.03
    shocked = returns[mask]
    if shocked.empty:
        return pd.Series(dtype=float)
    return shocked.median() * 100


# ──────────────────────────────────────────────────────────────
# 7.  REGIME DETECTION (3-STATE GMM)
# ──────────────────────────────────────────────────────────────
def classify_regimes(returns):
    if "Brent" not in returns.columns:
        return pd.Series(index=returns.index, dtype="object", name="Regime")

    tmp = pd.DataFrame({
        "ret": returns["Brent"],
        "vol": returns["Brent"].rolling(20).std(),
    }).dropna()

    if len(tmp) < 50:
        return pd.Series(index=returns.index, dtype="object", name="Regime")

    gmm    = GaussianMixture(n_components=3, covariance_type="full",
                             random_state=42)
    labels = gmm.fit_predict(tmp.values)
    tmp["label"] = labels

    order = (tmp.groupby("label")["vol"].mean()
               .sort_values().index.tolist())
    rmap  = {order[0]: "Calm", order[1]: "Stress", order[2]: "Disruption"}
    tmp["Regime"] = tmp["label"].map(rmap)

    out = pd.Series(index=returns.index, dtype="object", name="Regime")
    out.loc[tmp.index] = tmp["Regime"]
    return out


# ──────────────────────────────────────────────────────────────
# 8.  EVENT WINDOW
# ──────────────────────────────────────────────────────────────
def event_window_returns(prices, event_date=EVENT_DATE, window=10):
    ev  = pd.to_datetime(event_date)
    idx = prices.index
    if ev not in idx:
        loc_arr = idx.get_indexer([ev], method="nearest")
        ev = idx[loc_arr[0]]
        print(f"  [event] Using nearest trading day: {ev.date()}")

    loc   = idx.get_loc(ev)
    s_loc = max(loc - window, 0)
    e_loc = min(loc + window, len(idx) - 1)

    ep       = prices.iloc[s_loc:e_loc + 1].copy()
    ep       = ep / ep.loc[ev] * 100
    ep.index = np.arange(s_loc - loc, e_loc - loc + 1)
    return ep


# ──────────────────────────────────────────────────────────────
# 9.  PLOTS
# ──────────────────────────────────────────────────────────────

# 9-A  TITLE BANNER
def plot_title_banner():
    fig, ax = plt.subplots(figsize=(16, 2.2))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    ax.text(0.5, 0.72,
            "C H O K E P O I N T   S H O C K",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=32, fontweight="bold", color=PALETTE["accent1"],
            fontfamily="monospace")
    ax.text(0.5, 0.25,
            "Quantitative Analysis  •  Strait of Hormuz Disruption  "
            "•  Global Capital Markets & Commodity Trades",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color=PALETTE["subtext"], fontfamily="monospace")

    ax.plot([0.08, 0.92], [0.10, 0.10],
            color=PALETTE["accent2"], linewidth=0.7,
            transform=ax.transAxes, clip_on=False)

    add_logo(fig)
    save_show(fig, "00_title_banner.png", pad=0.3)


# 9-B  NORMALIZED CROSS-ASSET PERFORMANCE
def plot_normalized_prices(normed, event_date):
    SHOW = [c for c in
            ["Brent","WTI","NatGas","EnergyEq","Airlines","Gold","SPY","TLT"]
            if c in normed.columns]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.subplots_adjust(bottom=0.10)

    for col in SHOW:
        ax.plot(normed.index, to_series(normed[col]),
                color=ASSET_COLORS.get(col, "#FFFFFF"),
                linewidth=1.8, label=col, alpha=0.9)

    ev = pd.to_datetime(event_date)
    ax.axvspan(ev - pd.Timedelta(days=5), ev + pd.Timedelta(days=10),
               color=PALETTE["accent2"], alpha=0.06, label="Shock window")
    ax.axvline(ev, color=PALETTE["accent2"], linewidth=1.3, linestyle="--")
    ax.text(ev, ax.get_ylim()[0] * 1.01, " ◀ EVENT",
            color=PALETTE["accent2"], fontsize=7,
            va="bottom", fontfamily="monospace")

    ax.set_title("NORMALIZED CROSS-ASSET PERFORMANCE  (Base = 100)",
                 **FONT_TITLE)
    ax.set_ylabel("Index Level", **FONT_LABEL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.legend(ncol=4, loc="upper left")
    watermark(ax)
    add_logo(fig)
    save_show(fig, "01_normalized_cross_asset_performance.png")


# 9-C  BRENT-WTI SPREAD + REALIZED VOL
def plot_brent_wti_spread(features, event_date):
    if "Brent_WTI_Spread" not in features.columns:
        return

    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.10)
    gs  = gridspec.GridSpec(2, 1, hspace=0.40)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    spread = to_series(features["Brent_WTI_Spread"]).dropna()
    ax1.fill_between(spread.index, spread, 0,
                     where=(spread >= 0), alpha=0.25,
                     color=PALETTE["accent1"])
    ax1.fill_between(spread.index, spread, 0,
                     where=(spread < 0),  alpha=0.25,
                     color=PALETTE["accent2"])
    ax1.plot(spread.index, spread, color=PALETTE["accent1"], linewidth=1.5)
    ax1.axhline(0, color=PALETTE["border"], linewidth=0.8, linestyle="--")
    ax1.set_title("BRENT – WTI SPREAD", **FONT_TITLE)
    ax1.set_ylabel("USD / barrel", **FONT_LABEL)
    add_event_vline(ax1, event_date)

    if "Brent_20D_Vol" in features.columns:
        vol20 = to_series(features["Brent_20D_Vol"]).dropna()
        vol60 = (to_series(features["Brent_60D_Vol"]).dropna()
                 if "Brent_60D_Vol" in features.columns else None)
        ax2.fill_between(vol20.index, vol20 * 100,
                         alpha=0.30, color=PALETTE["accent5"])
        ax2.plot(vol20.index, vol20 * 100,
                 color=PALETTE["accent5"], linewidth=1.5, label="20-Day HV")
        if vol60 is not None:
            ax2.plot(vol60.index, vol60 * 100,
                     color=PALETTE["accent3"], linewidth=1.2,
                     linestyle="--", label="60-Day HV")
        ax2.set_title("BRENT REALIZED VOLATILITY (Annualized)", **FONT_TITLE)
        ax2.set_ylabel("Volatility %", **FONT_LABEL)
        ax2.legend()
        add_event_vline(ax2, event_date)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        watermark(ax)

    add_logo(fig)
    save_show(fig, "02_brent_wti_spread_and_vol.png")


# 9-D  ROLLING CORRELATION
def plot_rolling_correlation(features, event_date):
    if "Brent_SPY_60D_Corr" not in features.columns:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.subplots_adjust(bottom=0.10)
    corr = to_series(features["Brent_SPY_60D_Corr"]).dropna()

    ax.fill_between(corr.index, corr, 0,
                    where=(corr >= 0), alpha=0.20, color=PALETTE["accent2"],
                    label="Positive (risk-on / oil)")
    ax.fill_between(corr.index, corr, 0,
                    where=(corr < 0),  alpha=0.20, color=PALETTE["accent3"],
                    label="Negative (decoupled / haven)")
    ax.plot(corr.index, corr, color=PALETTE["accent1"], linewidth=1.8)
    ax.axhline(0,    color=PALETTE["border"],  linewidth=0.8, linestyle="--")
    ax.axhline(0.5,  color=PALETTE["accent2"], linewidth=0.6, linestyle=":")
    ax.axhline(-0.5, color=PALETTE["accent3"], linewidth=0.6, linestyle=":")

    add_event_vline(ax, event_date)
    ax.set_title("60-DAY ROLLING CORRELATION: BRENT vs. S&P 500", **FONT_TITLE)
    ax.set_ylabel("Pearson Correlation", **FONT_LABEL)
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    watermark(ax)
    add_logo(fig)
    save_show(fig, "03_brent_spy_rolling_correlation.png")


# 9-E  EVENT WINDOW
def plot_event_window(event_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(bottom=0.10)

    ax = axes[0]
    for col in event_df.columns:
        ax.plot(event_df.index, to_series(event_df[col]),
                color=ASSET_COLORS.get(col, "#AAAAAA"),
                linewidth=1.8, label=col, alpha=0.9)
    ax.axvline(0, color=PALETTE["accent2"], linewidth=1.5, linestyle="--")
    ax.axhline(100, color=PALETTE["border"], linewidth=0.7, linestyle=":")
    ax.set_title("EVENT WINDOW — ALL ASSETS", **FONT_TITLE)
    ax.set_xlabel("Days Relative to Event", **FONT_LABEL)
    ax.set_ylabel("Indexed to 100 on Event Day", **FONT_LABEL)
    ax.legend(ncol=2, loc="upper left")
    watermark(ax)

    ax2 = axes[1]
    cum    = event_df.iloc[-1] - 100
    colors = [PALETTE["accent3"] if v >= 0 else PALETTE["accent2"]
              for v in cum.values]
    bars   = ax2.barh(cum.index, cum.values, color=colors,
                      edgecolor=PALETTE["border"], height=0.6, linewidth=0.5)
    ax2.axvline(0, color=PALETTE["subtext"], linewidth=0.8)
    for bar, val in zip(bars, cum.values):
        ax2.text(val + (0.15 if val >= 0 else -0.15),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.2f}%", va="center",
                 ha="left" if val >= 0 else "right",
                 fontsize=8, color=PALETTE["text"], fontfamily="monospace")
    ax2.set_title("CUMULATIVE RETURN — DAY 0 → +10", **FONT_TITLE)
    ax2.set_xlabel("% Change from Event Date", **FONT_LABEL)
    watermark(ax2)

    add_logo(fig)
    save_show(fig, "04_event_window_performance.png")


# 9-F  REGIME DETECTION
def plot_regimes(prices, regimes):
    if "Brent" not in prices.columns:
        return

    df = pd.DataFrame({
        "Brent":  to_series(prices["Brent"]),
        "Regime": regimes
    }).dropna()

    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(bottom=0.10)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(df.index, df["Brent"],
             color=PALETTE["accent1"], linewidth=1.6, zorder=3)

    regime_colors = {
        "Calm":       PALETTE["calm"],
        "Stress":     PALETTE["stress"],
        "Disruption": PALETTE["disruption"],
    }
    bmin = df["Brent"].min() * 0.98
    bmax = df["Brent"].max() * 1.02
    for regime, color in regime_colors.items():
        mask = df["Regime"] == regime
        if mask.any():
            ax1.fill_between(df.index, bmin, bmax,
                             where=mask, color=color,
                             alpha=0.10, label=regime)

    ax1.set_title("BRENT CRUDE — MARKET REGIME DETECTION (3-STATE GMM)",
                  **FONT_TITLE)
    ax1.set_ylabel("Price (USD)", **FONT_LABEL)
    ax1.legend(ncol=3, loc="upper left")
    ax1.tick_params(labelbottom=False)
    watermark(ax1)

    numeric_regime = df["Regime"].map(
        {"Calm": 0, "Stress": 1, "Disruption": 2})
    cmap = LinearSegmentedColormap.from_list(
        "regime",
        [PALETTE["calm"], PALETTE["stress"], PALETTE["disruption"]])
    ax2.scatter(df.index, [1] * len(df), c=numeric_regime, cmap=cmap,
                s=6, marker="|", linewidths=0.4, vmin=0, vmax=2)
    ax2.set_yticks([])
    ax2.set_ylim(0.5, 1.5)
    ax2.set_ylabel("Regime", **FONT_LABEL)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    patches = [mpatches.Patch(color=v, label=k)
               for k, v in regime_colors.items()]
    ax2.legend(handles=patches, ncol=3, loc="upper left", fontsize=7)

    add_logo(fig)
    save_show(fig, "05_brent_market_regimes.png")


# 9-G  CORRELATION HEATMAP
def plot_correlation_heatmap(returns):
    COLS = [c for c in
            ["Brent","WTI","NatGas","EnergyEq","Airlines",
             "SPY","TLT","Gold","USD"]
            if c in returns.columns]
    corr = returns[COLS].corr()
    n    = len(corr)

    cmap_custom = LinearSegmentedColormap.from_list(
        "kt_corr",
        [PALETTE["accent4"], PALETTE["panel"], PALETTE["accent1"]])

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.subplots_adjust(bottom=0.12)
    im = ax.imshow(corr.values, cmap=cmap_custom, vmin=-1, vmax=1,
                   aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            txt_color = PALETTE["bg"] if abs(val) > 0.55 else PALETTE["text"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color, fontfamily="monospace",
                    fontweight="bold" if abs(val) > 0.7 else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
    cbar.set_label("Pearson r", color=PALETTE["subtext"], fontsize=9)

    ax.set_title("CROSS-ASSET RETURN CORRELATION MATRIX", **FONT_TITLE)
    watermark(ax)
    add_logo(fig)
    save_show(fig, "06_cross_asset_correlation_heatmap.png")


# 9-H  RISK METRICS TABLE
def plot_risk_metrics(risk_df):
    fig, ax = plt.subplots(figsize=(14, len(risk_df) * 0.72 + 2))
    fig.subplots_adjust(bottom=0.10)
    ax.axis("off")

    cols   = risk_df.reset_index().columns.tolist()
    rows   = risk_df.reset_index().values.tolist()
    n_cols = len(cols)

    for j, col in enumerate(cols):
        ax.text(j / n_cols + 0.5 / n_cols, 0.97, col,
                ha="center", va="top", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=PALETTE["accent1"],
                fontfamily="monospace")

    ax.plot([0, 1], [0.93, 0.93], color=PALETTE["border"],
            linewidth=0.8, transform=ax.transAxes, clip_on=False)

    y_step = 0.88 / max(len(rows), 1)
    for i, row in enumerate(rows):
        y        = 0.92 - i * y_step
        bg_color = PALETTE["panel"] if i % 2 == 0 else "#12161F"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y - y_step * 0.8), 1, y_step * 0.85,
            transform=ax.transAxes, boxstyle="square,pad=0",
            facecolor=bg_color, edgecolor="none", zorder=0))

        for j, val in enumerate(row):
            cell_color = PALETTE["text"]
            if j > 0:
                try:
                    fval = float(val)
                    if j in (1, 2, 4):
                        cell_color = (PALETTE["accent2"] if fval < -1
                                      else PALETTE["text"])
                    elif j == 3:
                        cell_color = (PALETTE["accent3"] if fval > 0.8
                                      else (PALETTE["accent2"] if fval < 0
                                            else PALETTE["text"]))
                except (TypeError, ValueError):
                    pass

            fmt_val = val if isinstance(val, str) else f"{val:,.3f}"
            ax.text(j / n_cols + 0.5 / n_cols, y - y_step * 0.3,
                    fmt_val, ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=8.5, color=cell_color, fontfamily="monospace")

    ax.set_title("RISK METRICS — VaR · CVaR · SHARPE · MAX DRAWDOWN",
                 **FONT_TITLE, pad=18)
    add_logo(fig)
    save_show(fig, "07_risk_metrics_table.png")


# 9-I  STRESS SCENARIO
def plot_stress_scenario(returns):
    response = stress_scenario_returns(returns)
    if response.empty:
        print("  [skip] Not enough Brent shock days for stress scenario.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(bottom=0.10)
    cols   = response.index.tolist()
    values = response.values
    colors = [PALETTE["accent3"] if v >= 0 else PALETTE["accent2"]
              for v in values]

    bars = ax.bar(range(len(cols)), values, color=colors,
                  edgecolor=PALETTE["border"], linewidth=0.6, width=0.65)
    ax.axhline(0, color=PALETTE["subtext"], linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.002 if val >= 0 else -0.002),
                f"{val:+.3f}%", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8, color=PALETTE["text"], fontfamily="monospace")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_title(
        "STRESS SCENARIO — CROSS-ASSET RESPONSE ON HIGH BRENT SHOCK DAYS (>+3%)",
        **FONT_TITLE)
    ax.set_ylabel("Median Daily Return (%)", **FONT_LABEL)
    watermark(ax)
    add_logo(fig)
    save_show(fig, "08_stress_scenario_response.png")


# 9-J  VIX vs BRENT DUAL AXIS  — FIXED fill_between TypeError
def plot_vix_crude(prices, event_date):
    if not {"VIX", "Brent"}.issubset(prices.columns):
        return

    # FIX: squeeze both to plain 1-D float Series before any plotting
    vix   = to_series(prices["VIX"]).dropna()
    brent = to_series(prices["Brent"]).dropna()

    # Align on shared trading dates
    common = vix.index.intersection(brent.index)
    vix    = vix.loc[common]
    brent  = brent.loc[common]

    fig, ax1 = plt.subplots(figsize=(16, 6))
    fig.subplots_adjust(bottom=0.10)
    ax2 = ax1.twinx()

    ax1.fill_between(vix.index, vix.values,
                     alpha=0.20, color=PALETTE["accent2"])
    ax1.plot(vix.index, vix.values,
             color=PALETTE["accent2"], linewidth=1.5, label="VIX")
    ax1.set_ylabel("VIX Level", **FONT_LABEL, color=PALETTE["accent2"])
    ax1.tick_params(axis="y", colors=PALETTE["accent2"])

    ax2.plot(brent.index, brent.values,
             color=PALETTE["accent1"], linewidth=1.8, label="Brent Crude")
    ax2.set_ylabel("Brent (USD/bbl)", **FONT_LABEL, color=PALETTE["accent1"])
    ax2.tick_params(axis="y", colors=PALETTE["accent1"])

    add_event_vline(ax1, event_date)
    ax1.set_title("VIX vs. BRENT CRUDE — FEAR vs. OIL PREMIUM", **FONT_TITLE)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left")

    watermark(ax1)
    add_logo(fig)
    save_show(fig, "09_vix_vs_brent.png")


# ──────────────────────────────────────────────────────────────
# 10.  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 60)
    print("  CHOKEPOINT SHOCK — K&T QUANT LABS")
    print("  Hormuz Disruption  ·  Capital Markets Analysis")
    print("═" * 60 + "\n")

    prices = download_market_data()

    print("  Engineering features …")
    returns, features, normed = prepare_features(prices)

    print("  Detecting market regimes …")
    regimes = classify_regimes(returns)

    print("  Building event window …")
    ev_cols  = [c for c in
                ["Brent","WTI","NatGas","SPY","EnergyEq","Airlines","Gold","TLT"]
                if c in prices.columns]
    event_df = event_window_returns(prices[ev_cols], event_date=EVENT_DATE)

    print("  Computing risk metrics …")
    risk_cols = [c for c in
                 ["Brent","WTI","NatGas","EnergyEq","Airlines",
                  "SPY","TLT","Gold","USD"]
                 if c in returns.columns]
    risk_df   = compute_risk_metrics(returns[risk_cols])

    print("\n  Generating & saving charts to:", OUTPUT_DIR)
    plot_title_banner()
    plot_normalized_prices(normed, EVENT_DATE)
    plot_brent_wti_spread(features, EVENT_DATE)
    plot_rolling_correlation(features, EVENT_DATE)
    plot_event_window(event_df)
    plot_regimes(prices, regimes)
    plot_correlation_heatmap(returns)
    plot_risk_metrics(risk_df)
    plot_stress_scenario(returns)
    plot_vix_crude(prices, EVENT_DATE)

    print("\n  Saving CSVs …")
    prices.to_csv(os.path.join(OUTPUT_DIR,   "market_prices.csv"))
    returns.to_csv(os.path.join(OUTPUT_DIR,  "market_returns.csv"))
    features.to_csv(os.path.join(OUTPUT_DIR, "derived_features.csv"))
    regimes.to_csv(os.path.join(OUTPUT_DIR,  "regimes.csv"))
    risk_df.to_csv(os.path.join(OUTPUT_DIR,  "risk_metrics.csv"))

    print("\n" + "═" * 60)
    print(f"  ✓  All outputs saved → {OUTPUT_DIR}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
