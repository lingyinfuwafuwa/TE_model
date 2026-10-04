"""Terminator strength ranked weakest -> strongest, with 95% CI, linear vs log y-axis.

Reproduces (and extends) the Excel version: all 582 constructs rather than the
synthetic sheet alone, coloured by library, with the CI drawn as a filled band.

The point of the figure is the comparison between the two panels:
  - linear y : ~150 terminators crushed against zero, CI invisible at the bottom
               and enormous at the top  -> error grows with the measurement
  - log y    : a smooth monotone ramp, CI band of roughly constant width
               -> the error is a constant PERCENTAGE, not a constant amount

That constant-width band is the argument for training on ln(Ts).

Also writes strength_ranked_linear_and_logts.png: panel A of the above beside the
ln(Ts) figure, untitled, on plain white -- the version for slides and papers.

Run:  python probing/plot_strength_ranked.py
"""

import ctypes
import os
import tempfile
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- palette (dataviz reference instance, light surface) -------------------
SURFACE = "#fcfcfb"
INK = "#10151c"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NATURAL = "#2a78d6"    # categorical slot 1
SYNTHETIC = "#eb6834"  # categorical slot 2
BAND = "#8e9299"

# chart chrome per surface. WARM is the reference look above (the two original
# figures); PLAIN is pure white with neutral grays, for the combined A/B figure.
# NATURAL / SYNTHETIC re-validated on #ffffff: all checks pass.
WARM = dict(surface=SURFACE, grid=GRID, axis=AXIS)
PLAIN = dict(surface="#ffffff", grid="#e5e5e5", axis="#c4c4c4")

N_REP = 3
T_CRIT = 4.302653     # t(df=2, 0.975) -- Chen reports n=3


def repo_root(start=None):
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "datasets_v2.csv").exists():
            return cand
    raise FileNotFoundError("repo root not found")


def read_excel_unlocked(path, **kw):
    """Survive the workbook being open in Excel (see CLAUDE.md)."""
    try:
        return pd.read_excel(path, **kw)
    except PermissionError:
        tmp = Path(tempfile.gettempdir()) / ("_locked_" + Path(path).name)
        if os.name == "nt":
            if not ctypes.windll.kernel32.CopyFileW(str(Path(path).resolve()), str(tmp), False):
                raise ctypes.WinError()
        else:
            import shutil
            shutil.copyfile(path, tmp)
        print(f"  note: {Path(path).name} is locked -- read a copy")
        return pd.read_excel(tmp, **kw)


def style(ax, theme=WARM):
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(theme["axis"])
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def load(repo):
    d = repo / "Chen_2013_terminators"
    nat = read_excel_unlocked(d / "41592_2013_BFnmeth2515_MOESM206_ESM.xlsx").assign(library="natural")
    syn = read_excel_unlocked(d / "41592_2013_BFnmeth2515_MOESM207_ESM.xlsx").assign(library="synthetic")
    df = pd.concat([nat, syn], ignore_index=True)
    df["Ts"] = pd.to_numeric(df["Average Strength"], errors="coerce")
    df["SD"] = pd.to_numeric(df["Standard Deviation of Strength"], errors="coerce")
    df = df.dropna(subset=["Ts"]).copy()

    # rank weakest -> strongest
    df = df.sort_values("Ts", ascending=True).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    half = T_CRIT * df["SD"] / np.sqrt(N_REP)      # NaN where SD is missing
    df["lo"] = df["Ts"] - half
    df["hi"] = df["Ts"] + half
    return df


def draw(ax, df, logy, theme=WARM, title=True):
    style(ax, theme)
    # CI band. NaN rows simply leave a gap, which is honest.
    ax.fill_between(df["rank"], df["lo"], df["hi"], color=BAND, alpha=0.30,
                    linewidth=0, zorder=2, label="95% CI (n=3)")
    # One scatter call, painted in rank order. Drawing the two libraries as
    # separate calls put synthetic systematically on top of natural, which made
    # the strong end look more synthetic than it is. (It IS more synthetic --
    # 78% of the top decile -- but the layering exaggerated it.)
    colors = np.where(df.library.to_numpy() == "natural", NATURAL, SYNTHETIC)
    ax.scatter(df["rank"], df["Ts"], s=9, c=colors, alpha=0.8,
               linewidths=0, zorder=4)

    ax.axhline(1.0, color=theme["axis"], linewidth=1.0, linestyle="--", zorder=3)
    ax.set_xlim(0, len(df) + 4)
    ax.set_xlabel("terminator rank  (weakest $\\rightarrow$ strongest)",
                  color=INK_2, fontsize=9.5)

    if logy:
        ax.set_yscale("log")
        ax.set_ylim(0.02, 2000)
        ax.set_ylabel("$T_S$   (log scale)", color=INK_2, fontsize=9.5)
        if title:
            ax.set_title("B.  log y-axis", color=INK, fontsize=10.5,
                         fontweight="bold", loc="left", pad=10)
        ax.annotate("CI band is roughly CONSTANT width\n"
                    "$\\Rightarrow$ error is a constant percentage",
                    xy=(0.04, 0.93), xycoords="axes fraction", va="top",
                    color=INK_2, fontsize=8.5, linespacing=1.5)
        ax.annotate("$T_S = 1$", xy=(len(df) - 4, 1.15), color=MUTED,
                    fontsize=8, ha="right")
    else:
        ax.set_ylim(-20, 700)
        ax.set_ylabel("$T_S$", color=INK_2, fontsize=9.5)
        if title:
            ax.set_title("A.  linear y-axis", color=INK, fontsize=10.5,
                         fontweight="bold", loc="left", pad=10)
        ax.annotate("~150 terminators crushed against zero;\n"
                    "CI invisible here, enormous at the top",
                    xy=(0.04, 0.93), xycoords="axes fraction", va="top",
                    color=INK_2, fontsize=8.5, linespacing=1.5)


def draw_logts(df, out):
    """Single panel: the TRAINING TARGET itself, ln(Ts), on a linear axis.

    Not the same as panel B of the other figure. That one plots raw Ts on a log
    AXIS -- same curve shape, but ticks labelled in Ts. This one plots the log
    VALUES, so the y-axis is the quantity the model actually regresses on, and
    y = 0 is exactly Ts = 1, the point where logit(TE) becomes undefined.
    """
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    fig.patch.set_facecolor(SURFACE)
    n_below, n_undef = draw_logts_panel(ax, df)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}   ({n_below} rows at or below ln(Ts)=0; "
          f"{n_undef} CI lower bounds non-positive)")


def draw_logts_panel(ax, df, theme=WARM, title=True, legend=True):
    """Draw the ln(Ts) panel onto `ax`; returns (n_below, n_undef) for captions."""
    style(ax, theme)

    y = np.log(df["Ts"].to_numpy())
    lo, hi = df["lo"].to_numpy(), df["hi"].to_numpy()
    drawable = lo > 0                       # ln() of a non-positive bound does not exist
    band_lo = np.where(drawable, np.log(np.where(drawable, lo, 1.0)), np.nan)
    band_hi = np.log(hi)

    ax.fill_between(df["rank"], band_lo, band_hi, color=BAND, alpha=0.30,
                    linewidth=0, zorder=2)
    ax.axhline(0.0, color=theme["axis"], linewidth=1.2, linestyle="--", zorder=3)
    colors = np.where(df.library.to_numpy() == "natural", NATURAL, SYNTHETIC)
    ax.scatter(df["rank"], y, s=9, c=colors, alpha=0.8, linewidths=0, zorder=4)

    n_undef = int((~drawable & df["lo"].notna().to_numpy()).sum())
    n_below = int((df["Ts"] <= 1).sum())
    ax.set_xlim(0, len(df) + 4)
    ax.set_ylim(-1.5, 6.4)
    ax.set_xlabel("terminator rank  (weakest $\\rightarrow$ strongest)",
                  color=INK_2, fontsize=9.5)
    ax.set_ylabel("$\\ln(T_S)$   — the training target", color=INK_2, fontsize=10)
    if title:
        ax.set_title("The target the model actually sees", color=INK, fontsize=12.5,
                     fontweight="bold", loc="left", pad=14)

    # both notes in the one genuinely free region (upper left); the data runs
    # lower-left to upper-right, so the legend gets the lower-right corner
    ax.annotate("Roughly a straight ramp — no single terminator\n"
                "dominates the squared error.\n\n"
                f"The dashed line is $\\ln(T_S) = 0$, i.e. $T_S = 1$.\n"
                f"{n_below} terminators fall below it, where\n"
                f"$\\mathrm{{TE}} = 1 - 1/T_S$ turns negative and\n"
                f"$\\mathrm{{logit}}(\\mathrm{{TE}})$ is undefined.",
                xy=(0.035, 0.96), xycoords="axes fraction", va="top",
                color=INK_2, fontsize=8.5, linespacing=1.6)
    ax.annotate("$T_S = 1$", xy=(len(df) - 6, 0.12), color=MUTED,
                fontsize=8, ha="right")

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=NATURAL, markersize=5,
               label=f"natural (n = {int((df.library=='natural').sum())})"),
        Line2D([], [], marker="o", linestyle="none", color=SYNTHETIC, markersize=5,
               label=f"synthetic (n = {int((df.library=='synthetic').sum())})"),
        Patch(facecolor=BAND, alpha=0.30,
              label=f"95% CI, n = {N_REP}  ({n_undef} rows omitted: lower bound $\\leq$ 0)"),
    ]
    if legend:
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
                  labelcolor=INK_2)
    return n_below, n_undef


def draw_combined(df, out):
    """A = panel A of the linear-vs-log figure, B = the ln(Ts) figure, side by side.

    For slides / papers: no titles (the caption carries them), one shared legend,
    and a plain white surface instead of the warm reference look.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    fig.patch.set_facecolor(PLAIN["surface"])
    draw(ax_a, df, logy=False, theme=PLAIN, title=False)
    _, n_undef = draw_logts_panel(ax_b, df, theme=PLAIN, title=False, legend=False)

    for ax, letter in ((ax_a, "A"), (ax_b, "B")):
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-36, 12),
                    textcoords="offset points", ha="left", va="bottom",
                    color=INK, fontsize=14, fontweight="bold")

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=NATURAL, markersize=5,
               label=f"natural (n = {int((df.library=='natural').sum())})"),
        Line2D([], [], marker="o", linestyle="none", color=SYNTHETIC, markersize=5,
               label=f"synthetic (n = {int((df.library=='synthetic').sum())})"),
        Patch(facecolor=BAND, alpha=0.30,
              label=f"95% CI, n = {N_REP}  (B omits {n_undef} rows whose lower bound is ≤ 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=200, facecolor=PLAIN["surface"], bbox_inches="tight")
    print(f"wrote {out}")


def main():
    repo = repo_root(Path(__file__).parent)
    df = load(repo)
    n_ci = int(df["lo"].notna().sum())
    print(f"n = {len(df)} constructs | {n_ci} with a reported SD | "
          f"Ts {df.Ts.min():.3f} - {df.Ts.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
    draw(axes[0], df, logy=False)
    draw(axes[1], df, logy=True)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=NATURAL, markersize=5,
               label=f"natural (n = {int((df.library=='natural').sum())})"),
        Line2D([], [], marker="o", linestyle="none", color=SYNTHETIC, markersize=5,
               label=f"synthetic (n = {int((df.library=='synthetic').sum())})"),
        Patch(facecolor=BAND, alpha=0.30, label=f"95% CI, n = {N_REP} ({n_ci} rows)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Chen 2013 terminator strength, ranked weakest to strongest",
                 color=INK, fontsize=12.5, fontweight="bold", x=0.008, ha="left", y=0.99)
    fig.text(0.008, 0.915,
             "Same data, same ranking, two y-axes. The linear panel is why "
             "$R^2$ on raw $T_S$ is dominated by a handful of terminators.",
             color=INK_2, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.89))

    out = Path(__file__).parent / "figures" / "strength_ranked_linear_vs_log.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")

    draw_logts(df, out.parent / "logts_ranked.png")
    draw_combined(df, out.parent / "strength_ranked_linear_and_logts.png")

    # the quantitative version of what the panels show
    band_w = (np.log10(df["hi"]) - np.log10(df["lo"].where(df["lo"] > 0)))
    q = pd.qcut(df["Ts"], 4, labels=False)
    print("\nCI width by Ts quartile:")
    print(f"{'quartile':<10}{'median Ts':>11}{'CI width, linear':>19}{'CI width, log10':>18}")
    for k in range(4):
        m = q == k
        lin = (df.loc[m, "hi"] - df.loc[m, "lo"]).median()
        print(f"Q{k+1:<9}{df.loc[m,'Ts'].median():>11.2f}{lin:>19.2f}{band_w[m].median():>18.2f}")
    print("\n-> linear width grows ~100x across quartiles; log10 width is near-constant.")


if __name__ == "__main__":
    main()
