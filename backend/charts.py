# backend/charts.py
"""
SmartEdge Copilot – Chart Generation Layer
Produces matplotlib figures for the Streamlit dashboard.
Every function returns both the Figure object AND a base64-encoded PNG string
so callers can choose between st.pyplot(fig) or st.image(b64).
"""

import io
import base64
from typing import Optional, Any

import numpy as np
import pandas as pd

def _mpl():
    from importlib import import_module
    mpl = import_module("matplotlib")
    plt = import_module("matplotlib.pyplot")
    mdates = import_module("matplotlib.dates")
    mticker = import_module("matplotlib.ticker")
    try:
        mpl.use("Agg")
    except Exception:
        pass
    return mpl, plt, mdates, mticker

# ─────────────────────────────────────────────
# Design System
# ─────────────────────────────────────────────

PALETTE = {
    "bg":          "#0F1117",   # dark canvas
    "panel":       "#1A1D27",   # card background
    "border":      "#2E3250",   # subtle borders
    "text_primary":"#F0F2FF",
    "text_muted":  "#8B90B8",
    "accent_blue": "#4F8EF7",
    "accent_cyan": "#00D4C8",
    "accent_green":"#3DDC97",
    "accent_amber":"#F7B731",
    "accent_red":  "#FF6B6B",
    "accent_purple":"#A78BFA",
    "gradient_start": "#4F8EF7",
    "gradient_end":   "#A78BFA",
}

FEATURE_COLORS = [
    PALETTE["accent_blue"],
    PALETTE["accent_cyan"],
    PALETTE["accent_green"],
    PALETTE["accent_amber"],
    PALETTE["accent_red"],
    PALETTE["accent_purple"],
]

_FONT_FAMILY = "DejaVu Sans"


def _apply_dark_theme(fig: Any, axes) -> None:
    """Apply the SmartEdge dark theme to a figure and one or more Axes."""
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    fig.patch.set_facecolor(PALETTE["bg"])

    for ax in axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text_muted"])
        ax.yaxis.label.set_color(PALETTE["text_muted"])
        ax.title.set_color(PALETTE["text_primary"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.grid(True, color=PALETTE["border"], linewidth=0.5,
                linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)


def _fig_to_b64(fig: Any) -> str:
    """Encode a matplotlib Figure as a base64 PNG string."""
    _, plt, _, _ = _mpl()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _styled_fig(
    ncols: int = 1,
    nrows: int = 1,
    figsize: tuple = (10, 4),
) -> tuple[Any, Any]:
    """Create a pre-styled dark figure and return (fig, axes)."""
    _, plt, _, _ = _mpl()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    _apply_dark_theme(fig, axes if hasattr(axes, "__iter__") else [axes])
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    return fig, axes


def _add_watermark(ax, text: str = "SmartEdge Copilot") -> None:
    ax.text(
        0.99, 0.02, text,
        transform=ax.transAxes,
        fontsize=7, color=PALETTE["text_muted"],
        alpha=0.4, ha="right", va="bottom",
        fontfamily=_FONT_FAMILY,
    )


def _no_data_figure(title: str) -> tuple[Any, str]:
    """Return a placeholder figure when the DataFrame is empty."""
    fig, ax = _styled_fig(figsize=(8, 3))
    ax.text(
        0.5, 0.5, "No data available yet.\nMake some LLM calls to populate this chart.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=11, color=PALETTE["text_muted"], fontfamily=_FONT_FAMILY,
    )
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["text_primary"], pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    return fig, _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 1. Latency Over Time
# ─────────────────────────────────────────────

def chart_latency_over_time(
    df: pd.DataFrame,
    freq: str = "1h",
    show_p95: bool = True,
    feature_filter: Optional[str] = None,
) -> tuple[Any, str]:
    """
    Line chart: average (and optional p95) latency resampled over time.

    Args:
        df:             Raw metrics DataFrame from query_metrics().
        freq:           Pandas resample frequency string – "15min", "1h", "1D".
        show_p95:       Overlay a p95 latency band.
        feature_filter: If supplied, only plot rows for that feature.

    Returns:
        (matplotlib Figure, base64 PNG string)
    """
    title = "Latency Over Time"
    if df is None or df.empty:
        return _no_data_figure(title)

    if feature_filter:
        df = df[df["feature"] == feature_filter]
        title += f"  ·  {feature_filter}"

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    ts = df["latency_ms"].resample(freq).agg(
        avg=("mean"),
        p95=(lambda s: s.quantile(0.95) if len(s) > 0 else np.nan),
        count=("count"),
    ).dropna(subset=["avg"])

    if ts.empty:
        return _no_data_figure(title)

    fig, ax = _styled_fig(figsize=(11, 4))

    ax.plot(
        ts.index, ts["avg"],
        color=PALETTE["accent_blue"], linewidth=2,
        marker="o", markersize=3, label="Avg Latency",
    )

    if show_p95:
        ax.fill_between(
            ts.index, ts["avg"], ts["p95"],
            alpha=0.18, color=PALETTE["accent_purple"], label="p95 band",
        )
        ax.plot(
            ts.index, ts["p95"],
            color=PALETTE["accent_purple"], linewidth=1.2,
            linestyle="--", alpha=0.8, label="p95 Latency",
        )

    # x-axis date formatting
    _, _, mdates, mticker = _mpl()
    locator   = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    _, plt, _, _ = _mpl()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f} ms"
    ))
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["text_primary"], pad=12)
    ax.set_xlabel("Time", labelpad=8)
    ax.set_ylabel("Latency (ms)", labelpad=8)

    legend = ax.legend(
        framealpha=0.15, edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text_muted"], fontsize=9,
    )
    legend.get_frame().set_facecolor(PALETTE["panel"])

    _add_watermark(ax)
    plt.tight_layout()
    return fig, _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 2. Tokens Per Feature
# ─────────────────────────────────────────────

def chart_tokens_per_feature(
    df: pd.DataFrame,
    mode: str = "total",
    top_n: int = 10,
) -> tuple[Any, str]:
    """
    Horizontal bar chart: token consumption broken down by feature.

    Args:
        df:    Raw metrics DataFrame from query_metrics().
        mode:  "total" for sum of tokens, "avg" for mean per call.
        top_n: Show only the top N features by token usage.

    Returns:
        (matplotlib Figure, base64 PNG string)
    """
    title = f"{'Total' if mode == 'total' else 'Avg'} Tokens per Feature"
    if df is None or df.empty:
        return _no_data_figure(title)

    grp = (
        df.groupby("feature")["tokens_used"]
        .agg(["sum", "mean", "count"])
        .rename(columns={"sum": "total", "mean": "avg"})
        .sort_values("total", ascending=False)
        .head(top_n)
    )

    if grp.empty:
        return _no_data_figure(title)

    metric_col = "total" if mode == "total" else "avg"
    grp = grp.sort_values(metric_col, ascending=True)   # ascending for horizontal bars

    colors = [
        FEATURE_COLORS[i % len(FEATURE_COLORS)]
        for i in range(len(grp))
    ]

    fig, ax = _styled_fig(figsize=(10, max(3, len(grp) * 0.55 + 1.5)))

    bars = ax.barh(
        grp.index, grp[metric_col],
        color=colors, height=0.6,
        edgecolor=PALETTE["bg"], linewidth=0.4,
    )

    # Value labels
    for bar, val in zip(bars, grp[metric_col]):
        ax.text(
            bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center", ha="left",
            fontsize=8.5, color=PALETTE["text_muted"],
        )

    _, _, _, mticker = _mpl()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}"
    ))
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["text_primary"], pad=12)
    ax.set_xlabel("Tokens", labelpad=8)
    ax.set_ylabel("Feature", labelpad=8)

    _add_watermark(ax)
    plt.tight_layout()
    return fig, _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 3. Cost Trend
# ─────────────────────────────────────────────

def chart_cost_trend(
    df: pd.DataFrame,
    freq: str = "1D",
    show_cumulative: bool = True,
) -> tuple[Any, str]:
    """
    Dual-axis chart: daily spend (bars) + cumulative cost (line).

    Args:
        df:               Raw metrics DataFrame from query_metrics().
        freq:             Resample frequency – "1h", "1D", "1W".
        show_cumulative:  Overlay cumulative cost on the right y-axis.

    Returns:
        (matplotlib Figure, base64 PNG string)
    """
    title = "Cost Trend"
    if df is None or df.empty:
        return _no_data_figure(title)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    ts = df["cost"].resample(freq).sum().fillna(0)
    ts_cum = ts.cumsum()

    if ts.empty:
        return _no_data_figure(title)

    fig, ax1 = _styled_fig(figsize=(11, 4))

    # Bar: daily spend
    bar_width = max(0.6, (ts.index[-1] - ts.index[0]).total_seconds()
                   / max(len(ts), 1) / 86400 * 0.7) if len(ts) > 1 else 0.6

    ax1.bar(
        ts.index, ts.values,
        width=pd.tseries.frequencies.to_offset(freq).nanos / 1e9 / 86400 * 0.65
              if hasattr(pd.tseries.frequencies.to_offset(freq), "nanos") else bar_width,
        color=PALETTE["accent_blue"],
        alpha=0.75, label="Period Cost",
        align="center",
    )
    _, _, _, mticker = _mpl()
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:.4f}"
    ))
    ax1.set_ylabel("Cost per Period (USD)", labelpad=8, color=PALETTE["accent_blue"])
    ax1.tick_params(axis="y", colors=PALETTE["accent_blue"])

    # Line: cumulative cost (right axis)
    if show_cumulative:
        ax2 = ax1.twinx()
        ax2.set_facecolor(PALETTE["panel"])
        ax2.plot(
            ts_cum.index, ts_cum.values,
            color=PALETTE["accent_green"],
            linewidth=2.2, linestyle="-",
            marker="", label="Cumulative Cost",
        )
        ax2.fill_between(
            ts_cum.index, ts_cum.values,
            alpha=0.08, color=PALETTE["accent_green"],
        )
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"${v:.4f}"
        ))
        ax2.set_ylabel("Cumulative Cost (USD)", labelpad=8,
                       color=PALETTE["accent_green"])
        ax2.tick_params(axis="y", colors=PALETTE["accent_green"])
        ax2.spines["right"].set_edgecolor(PALETTE["border"])

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(
            lines1 + lines2, labels1 + labels2,
            framealpha=0.15, edgecolor=PALETTE["border"],
            labelcolor=PALETTE["text_muted"], fontsize=9,
        )
        legend.get_frame().set_facecolor(PALETTE["panel"])

    _, _, mdates, _ = _mpl()
    locator   = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    _, plt, _, _ = _mpl()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha="right")

    ax1.set_title(title, fontsize=13, fontweight="bold",
                  color=PALETTE["text_primary"], pad=12)
    ax1.set_xlabel("Time", labelpad=8)

    _add_watermark(ax1)
    plt.tight_layout()
    return fig, _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 4. Optimization Comparison %
# ─────────────────────────────────────────────

def chart_optimization_comparison(
    df: pd.DataFrame,
    baseline_version: str = "v1",
    compare_version:  str = "v2",
    metrics: Optional[list[str]] = None,
) -> tuple[Any, str]:
    """
    Grouped bar chart comparing two prompt versions across key metrics,
    expressed as percentage change relative to the baseline.

    Args:
        df:                Raw metrics DataFrame from query_metrics().
        baseline_version:  prompt_version tag treated as the baseline (100 %).
        compare_version:   prompt_version tag to compare against baseline.
        metrics:           List of numeric columns to compare.
                           Defaults to ["latency_ms", "tokens_used", "cost"].

    Returns:
        (matplotlib Figure, base64 PNG string)
    """
    title = f"Optimization: {baseline_version} → {compare_version}"
    if df is None or df.empty:
        return _no_data_figure(title)

    metrics = metrics or ["latency_ms", "tokens_used", "cost"]

    base = df[df["prompt_version"] == baseline_version][metrics].mean()
    comp = df[df["prompt_version"] == compare_version][metrics].mean()

    if base.isna().all() or comp.isna().all():
        return _no_data_figure(title + "\n(insufficient version data)")

    pct_change = ((comp - base) / base.replace(0, np.nan) * 100).fillna(0)

    labels     = [m.replace("_", " ").title() for m in metrics]
    bar_colors = [
        PALETTE["accent_green"] if v <= 0 else PALETTE["accent_red"]
        for v in pct_change.values
    ]

    fig, axes = _styled_fig(ncols=2, figsize=(12, 4.5))
    ax_bar, ax_table = axes

    # ── Left: percentage-change bars ──
    x = np.arange(len(labels))
    bars = ax_bar.bar(
        x, pct_change.values,
        color=bar_colors, width=0.5,
        edgecolor=PALETTE["bg"], linewidth=0.5,
    )

    # Zero-line
    ax_bar.axhline(0, color=PALETTE["text_muted"],
                   linewidth=0.8, linestyle="-")

    for bar, val in zip(bars, pct_change.values):
        ypos = bar.get_height() + (2 if val >= 0 else -5)
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.1f}%",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9.5, fontweight="bold",
            color=PALETTE["accent_green"] if val <= 0 else PALETTE["accent_red"],
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10)
    _, _, _, mticker = _mpl()
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:+.0f}%"
    ))
    ax_bar.set_title(title, fontsize=12, fontweight="bold",
                     color=PALETTE["text_primary"], pad=10)
    ax_bar.set_ylabel("% Change vs Baseline", labelpad=8)
    _add_watermark(ax_bar)

    # ── Right: raw values table ──
    ax_table.axis("off")

    table_data = []
    for m, lbl in zip(metrics, labels):
        b_val = base[m] if not np.isnan(base[m]) else 0
        c_val = comp[m] if not np.isnan(comp[m]) else 0
        delta = pct_change[m]
        arrow = "▼" if delta < 0 else ("▲" if delta > 0 else "–")
        fmt   = ".2f" if m == "cost" else ".1f"
        table_data.append([
            lbl,
            f"{b_val:{fmt}}",
            f"{c_val:{fmt}}",
            f"{arrow} {abs(delta):.1f}%",
        ])

    col_labels = ["Metric", baseline_version, compare_version, "Δ Change"]
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.1, 1, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(PALETTE["panel"] if row > 0 else PALETTE["border"])
        cell.set_edgecolor(PALETTE["border"])
        cell.set_text_props(
            color=PALETTE["text_primary"] if row == 0 else PALETTE["text_muted"]
        )

    ax_table.set_title("Raw Comparison", fontsize=11,
                        color=PALETTE["text_primary"], pad=8)

    plt.tight_layout()
    return fig, _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 5. Bonus: Call Volume Heatmap
# ─────────────────────────────────────────────

def chart_call_volume_heatmap(
    df: pd.DataFrame,
) -> tuple[Any, str]:
    """
    Calendar-style heatmap of LLM call volume by hour-of-day × day-of-week.
    Reveals usage patterns (e.g. peak hours, quiet weekends).

    Returns:
        (matplotlib Figure, base64 PNG string)
    """
    title = "Call Volume Heatmap  ·  Hour × Day"
    if df is None or df.empty:
        return _no_data_figure(title)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = (
        df.groupby(["weekday", "hour"]).size()
        .unstack(fill_value=0)
        .reindex(day_order, fill_value=0)
    )

    fig, ax = _styled_fig(figsize=(13, 4))

    mpl, _, _, _ = _mpl()
    cmap = mpl.colormaps.get_cmap("YlOrRd")
    im   = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(day_order)))
    ax.set_yticklabels(day_order, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    _, _, _, mticker = _mpl()
    cbar.ax.tick_params(colors=PALETTE["text_muted"], labelsize=8)
    cbar.set_label("Call Count", color=PALETTE["text_muted"], fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["text_primary"], pad=12)
    ax.grid(False)

    _add_watermark(ax)
    _, plt, _, _ = _mpl()
    plt.tight_layout()
    return fig, _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 6. Bonus: Model Comparison Radar
# ─────────────────────────────────────────────

def chart_model_radar(
    model_stats: pd.DataFrame,
) -> tuple[Figure, str]:
    """
    Radar (spider) chart comparing models across normalised performance axes:
    Speed, Efficiency, Economy, Volume.

    Args:
        model_stats: DataFrame from metrics.query_model_comparison().

    Returns:
        (matplotlib Figure, base64 PNG string)
    """
    title = "Model Performance Radar"
    if model_stats is None or model_stats.empty:
        return _no_data_figure(title)

    axes_cfg = {
        "Speed":      ("avg_latency_ms", True),   # lower is better → invert
        "Efficiency": ("avg_tokens",     True),
        "Economy":    ("avg_cost",       True),
        "Volume":     ("call_count",     False),  # higher is better
    }

    radar_labels = list(axes_cfg.keys())
    n_axes       = len(radar_labels)
    angles       = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles      += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    for i, (_, row) in enumerate(model_stats.head(5).iterrows()):
        values = []
        for ax_name, (col, invert) in axes_cfg.items():
            col_min = model_stats[col].min()
            col_max = model_stats[col].max()
            rng     = col_max - col_min if col_max != col_min else 1
            norm    = (row[col] - col_min) / rng
            values.append(1 - norm if invert else norm)
        values += values[:1]

        color = FEATURE_COLORS[i % len(FEATURE_COLORS)]
        ax.plot(angles, values, color=color, linewidth=2,
                linestyle="-", label=row["model"])
        ax.fill(angles, values, color=color, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, color=PALETTE["text_primary"],
                       fontsize=10, fontweight="bold")
    ax.set_yticklabels([])
    ax.spines["polar"].set_edgecolor(PALETTE["border"])
    ax.grid(color=PALETTE["border"], linewidth=0.5)

    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        framealpha=0.15, edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text_muted"], fontsize=9,
    )
    legend.get_frame().set_facecolor(PALETTE["panel"])
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["text_primary"], pad=18)

    plt.tight_layout()
    return fig, _fig_to_b64(fig)


# ─────────────────────────────────────────────
# Dashboard Bundle
# ─────────────────────────────────────────────

def generate_all_charts(
    df: pd.DataFrame,
    model_stats: Optional[pd.DataFrame] = None,
    freq: str = "1h",
) -> dict[str, tuple[Figure, str]]:
    """
    Generate all charts in a single call.
    Returns a dict keyed by chart name → (Figure, b64_string).

    Usage in Streamlit::

        charts = generate_all_charts(df, model_stats)
        st.pyplot(charts["latency"][0])
        # or
        st.image(base64.b64decode(charts["latency"][1]))
    """
    return {
        "latency":      chart_latency_over_time(df, freq=freq),
        "tokens":       chart_tokens_per_feature(df),
        "cost_trend":   chart_cost_trend(df, freq=freq),
        "optimization": chart_optimization_comparison(df),
        "heatmap":      chart_call_volume_heatmap(df),
        "radar": chart_model_radar(
            model_stats if model_stats is not None and not model_stats.empty
            else pd.DataFrame()
        ),
    }
