"""Relationship between Chen's terminator strength T_S, termination efficiency TE,
and the transition-state free-energy difference ddG++.

    TE      = 1 - 1/T_S
    logit(TE) = ln(TE / (1 - TE)) = ln(T_S - 1) = -ddG++/RT
    ddG++/RT  = -ln(T_S - 1),      ddG++ = dG++_term - dG++_read

Writes probing/figures/te_energy_relationship.png.

Run:  C:\\venvs\\TE_model\\Scripts\\python.exe probing/plot_te_energy.py
"""

import os
import shutil
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
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
USABLE = "#2a78d6"   # categorical slot 1
BROKEN = "#d03b3b"   # status: critical -- validated vs slot 1, CVD dE 23.8

R_KCAL = 1.987204e-3      # kcal / (mol K)
T_KELVIN = 310.15         # 37 C, Chen's assay
RT = R_KCAL * T_KELVIN    # ~0.6164 kcal/mol


def repo_root(start=None):
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "datasets_v2.csv").exists():
            return cand
    raise FileNotFoundError("repo root not found (looked for datasets_v2.csv)")


def read_excel_unlocked(path, **kw):
    """pd.read_excel, but survive the workbook being open in Excel.

    Excel takes an exclusive lock that makes open() raise PermissionError, while
    a shell copy still succeeds. Fall back to reading a temp copy rather than
    asking the user to close their spreadsheet.
    """
    try:
        return pd.read_excel(path, **kw)
    except PermissionError:
        tmp = Path(tempfile.gettempdir()) / f"_locked_{Path(path).name}"
        # shutil.copyfile uses open() and hits the same lock; the Win32 CopyFileW
        # API (what Copy-Item calls) opens with wider share flags and succeeds.
        if os.name == "nt":
            import ctypes
            if not ctypes.windll.kernel32.CopyFileW(str(path), str(tmp), False):
                raise ctypes.WinError()
        else:
            shutil.copyfile(path, tmp)
        print(f"  note: {Path(path).name} is locked (open in Excel?) -- read a copy")
        return pd.read_excel(tmp, **kw)


def load_chen(repo):
    """582 terminators with Chen's 'Average Strength' (= T_S)."""
    d = repo / "Chen_2013_terminators"
    nat = read_excel_unlocked(d / "41592_2013_BFnmeth2515_MOESM206_ESM.xlsx",
                              sheet_name="Fig2a-natural")
    syn = read_excel_unlocked(d / "41592_2013_BFnmeth2515_MOESM207_ESM.xlsx",
                              sheet_name="Sheet1")
    both = pd.concat([nat.assign(src="natural"), syn.assign(src="synthetic")],
                     ignore_index=True)
    both["T_S"] = pd.to_numeric(both["Average Strength"], errors="coerce")
    return both.dropna(subset=["T_S"])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def main():
    repo = repo_root(Path(__file__).parent)
    chen = load_chen(repo)
    ts = chen["T_S"].to_numpy()
    ok = ts > 1
    te_ok = 1 - 1 / ts[ok]
    ddg_ok = -np.log(ts[ok] - 1)          # in units of RT

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    fig.patch.set_facecolor(SURFACE)

    def rug(ax, values, y, color, alpha):
        """Real data as marginal ticks. The three relations are deterministic, so
        plotting the 582 points as a scatter would just retrace the curve and hide
        it -- the only thing the data adds is *where along the curve it sits*."""
        ax.plot(values, np.full(len(values), y), "|", color=color,
                markersize=6, alpha=alpha, zorder=6, clip_on=False)

    # ---------------------------------------------------------------- panel A
    ax = axes[0]
    style(ax)
    x = np.logspace(np.log10(0.3), np.log10(400), 800)
    ax.axhspan(-2.35, 0, color=BROKEN, alpha=0.07, zorder=1)
    ax.plot(x, 1 - 1 / x, color=INK_2, linewidth=2.0, zorder=4)
    ax.axhline(1, color=AXIS, linewidth=1.0, linestyle=":", zorder=2)
    ax.axvline(1, color=BROKEN, linewidth=1.2, linestyle="--", zorder=3)
    rug(ax, ts[ok], 1.30, USABLE, 0.35)
    rug(ax, ts[~ok], 1.30, BROKEN, 0.60)

    ax.set_xscale("log")
    ax.set_xlim(0.3, 400)
    ax.set_ylim(-2.35, 1.20)
    ax.set_xlabel("$T_S$   (Chen 'Average Strength')", color=INK_2, fontsize=9.5)
    ax.set_ylabel("TE  $=\\; 1 - 1/T_S$", color=INK_2, fontsize=9.5)
    ax.set_title("A.  Strength $\\rightarrow$ efficiency", color=INK,
                 fontsize=10.5, fontweight="bold", loc="left", pad=16)
    ax.annotate("TE $\\rightarrow$ 1", xy=(320, 0.90), color=MUTED, fontsize=8,
                ha="right", va="top")
    ax.annotate("$T_S = 1$", xy=(0.92, -2.18), color=BROKEN, fontsize=8.5,
                ha="right")
    ax.annotate("TE $<$ 0 — not a probability", xy=(1.6, -0.42), color=BROKEN,
                fontsize=8.5)

    # ---------------------------------------------------------------- panel B
    ax = axes[1]
    style(ax)
    xr = np.logspace(np.log10(1.0009), np.log10(400), 1200)
    ax.axvspan(0.3, 1, color=BROKEN, alpha=0.07, zorder=1)
    ax.plot(xr, -np.log(xr - 1), color=INK_2, linewidth=2.0, zorder=4)
    ax.axhline(0, color=AXIS, linewidth=1.0, linestyle="--", zorder=2)
    ax.axvline(1, color=BROKEN, linewidth=1.2, linestyle="--", zorder=3)
    rug(ax, ts[ok], 7.15, USABLE, 0.35)
    rug(ax, ts[~ok], 7.15, BROKEN, 0.60)

    ax.set_xscale("log")
    ax.set_xlim(0.3, 400)
    ax.set_ylim(-6.8, 6.8)
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_xlabel("$T_S$", color=INK_2, fontsize=9.5)
    ax.set_ylabel("$\\Delta\\Delta G^{\\ddagger}/RT \\;=\\; -\\ln(T_S - 1)$",
                  color=INK_2, fontsize=9.5)
    ax.set_title("B.  Strength $\\rightarrow$ energy", color=INK, fontsize=10.5,
                 fontweight="bold", loc="left", pad=16)
    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: v * RT, lambda v: v / RT))
    sec.set_ylabel("$\\Delta\\Delta G^{\\ddagger}$   (kcal/mol at 37 $\\degree$C)",
                   color=MUTED, fontsize=9)
    sec.set_yticks([-4, -2, 0, 2, 4])
    sec.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    sec.spines["right"].set_color(AXIS)
    ax.annotate("readthrough favoured  $\\uparrow$", xy=(2.1, 5.4), color=MUTED,
                fontsize=8, style="italic")
    ax.annotate("termination favoured  $\\downarrow$", xy=(2.1, -5.9),
                color=MUTED, fontsize=8, style="italic")
    # keep this inside the narrow shaded band rather than letting it spill across
    # the T_S = 1 boundary into the valid region
    ax.text(0.55, 0.0, "90 rows — no $\\Delta\\Delta G^{\\ddagger}$",
            color=BROKEN, fontsize=8.5, rotation=90, ha="center", va="center")

    # ---------------------------------------------------------------- panel C
    ax = axes[2]
    style(ax)
    te = np.linspace(2.4e-4, 1 - 2.4e-4, 2000)
    ax.plot(te, -np.log(te / (1 - te)), color=INK_2, linewidth=2.0, zorder=4)
    ax.axhline(0, color=AXIS, linewidth=1.0, linestyle="--", zorder=2)
    ax.axvline(0.5, color=AXIS, linewidth=1.0, linestyle=":", zorder=2)
    rug(ax, te_ok, 7.15, USABLE, 0.35)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-6.8, 6.8)
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_xlabel("TE", color=INK_2, fontsize=9.5)
    ax.set_ylabel(
        "$\\Delta\\Delta G^{\\ddagger}/RT \\;=\\; -\\mathrm{logit}(\\mathrm{TE})$",
        color=INK_2, fontsize=9.5)
    ax.set_title("C.  Efficiency $\\rightarrow$ energy", color=INK, fontsize=10.5,
                 fontweight="bold", loc="left", pad=16)
    ax.annotate("TE = 0.5,  $\\Delta\\Delta G^{\\ddagger} = 0$", xy=(0.5, 0),
                xytext=(0.56, 2.6), color=MUTED, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1))
    ax.annotate("diverges as TE $\\rightarrow$ 0", xy=(0.055, 5.4), color=MUTED,
                fontsize=8, style="italic")
    ax.annotate("and as TE $\\rightarrow$ 1", xy=(0.60, -5.9), color=MUTED,
                fontsize=8, style="italic")

    # ---------------------------------------------------------------- chrome
    handles = [
        Line2D([], [], color=INK_2, linewidth=2.0, label="exact relation"),
        Line2D([], [], marker="|", linestyle="none", color=USABLE,
               markersize=9, alpha=0.8,
               label=f"Chen 2013, usable  ($T_S > 1$, n = {int(ok.sum())})"),
        Line2D([], [], marker="|", linestyle="none", color=BROKEN,
               markersize=9,
               label=f"Chen 2013, undefined  ($T_S \\leq 1$, n = {int((~ok).sum())})"),
        Patch(facecolor=BROKEN, alpha=0.10,
              label="TE $\\leq$ 0 — no $\\Delta\\Delta G^{\\ddagger}$ exists"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Terminator strength, termination efficiency, and the transition-state "
        "energy difference",
        color=INK, fontsize=12.5, fontweight="bold", x=0.008, ha="left", y=0.995)
    fig.text(0.008, 0.925,
             "$\\mathrm{TE} = 1 - 1/T_S$   and   "
             "$\\mathrm{logit}(\\mathrm{TE}) = -\\Delta\\Delta G^{\\ddagger}/RT$,   "
             "so   $\\Delta\\Delta G^{\\ddagger}/RT = -\\ln(T_S - 1)$."
             "   Energy axis uses $RT$ = 0.616 kcal/mol at 37 $\\degree$C.",
             color=INK_2, fontsize=9, ha="left")

    fig.tight_layout(rect=(0, 0.055, 1, 0.90))

    out = Path(__file__).parent / "figures" / "te_energy_relationship.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  usable  n={int(ok.sum())}  ddG/RT range "
          f"{ddg_ok.min():.2f} .. {ddg_ok.max():.2f}"
          f"  ({ddg_ok.min()*RT:.2f} .. {ddg_ok.max()*RT:.2f} kcal/mol)")
    print(f"  undefined n={int((~ok).sum())}  T_S range "
          f"{ts[~ok].min():.3f} .. {ts[~ok].max():.3f}")


if __name__ == "__main__":
    main()
