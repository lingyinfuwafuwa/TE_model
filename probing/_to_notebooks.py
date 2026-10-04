"""Turns this session's analysis scripts into readable notebooks. Run once, then delete."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def nb(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python (TE_model)",
                                        "language": "python", "name": "te_model"},
                         "language_info": {"name": "python"}}}


def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": s.strip("\n").splitlines(keepends=True)}


def co(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


SETUP = r'''
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")
REPO = Path.cwd().parent if Path.cwd().name == "probing" else Path.cwd()
EMB, RES = REPO / "probing/embeddings", REPO / "probing/results"
ALPHAS = np.logspace(-3, 6, 40)

FOLDS = json.loads((RES / "folds.json").read_text())
print("folds:", FOLDS["n_folds"], "| seed:", FOLDS["seed"], "| schemes:", list(FOLDS["schemes"]))
'''

HELPERS = r'''
def per_fold(make_model, X, y, folds):
    """Fit on each training fold, score the held-out fold. Returns (r2 array, rho array)."""
    r2, rho = [], []
    for tr, te in folds:
        tr, te = np.array(tr), np.array(te)
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2.append(r2_score(y[te], p))
        rho.append(spearmanr(y[te], p).statistic)
    return np.array(r2), np.array(rho)

RIDGE = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))

def one_hot(seqs, L):
    idx = {c: i for i, c in enumerate("ACGU")}
    M = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            M[i, j, idx[c]] = 1.0
    return M.reshape(len(seqs), -1)
'''

# ---------------------------------------------------------------- 02 window
w = [
md(r"""
# 02 - does the 19 nt downstream window help?

`probe_rnafm.ipynb` (notebook 01) ran the 90 nt window. This one runs the **90 + 19 nt**
window and compares them **paired on identical folds and rows**, so the difference is not
confounded by the split.

Embeddings for the 109 nt window were extracted by `_run_window109.py`; this notebook loads
the cached arrays.
"""),
co(SETUP),
co(HELPERS),
co(r'''
d90  = pd.read_csv(REPO/"probing/data/chen2013_window_up90.csv").rename(columns={"window":"sequence","strength":"Ts"})
d109 = pd.read_csv(REPO/"probing/data/chen2013_window_up90_dn19.csv").rename(columns={"window":"sequence","strength":"Ts"})
assert d90.name.equals(d109.name), "row order differs -> folds would not be comparable"
y = np.log(d90.Ts.to_numpy())                       # target: ln(Ts)
print("n =", len(y), "| 90 nt and 109 nt CSVs are row-aligned")
'''),
co(r'''
def reps(L, tag):
    st = {l: np.load(EMB / (f"rnafm_layer{l}_tokens.npy" if tag == "" else f"rnafm_L{L}_layer{l}_tokens.npy"))
          for l in [6, 9, 12]}
    R = {"A: L12 mean":  st[12][:, 1:L+1, :].mean(1),
         "B: L12 <cls>": st[12][:, 0, :],
         "C: L6 mean":   st[6][:, 1:L+1, :].mean(1),
         "D: L9 mean":   st[9][:, 1:L+1, :].mean(1)}
    return R

R90, R109 = reps(90, ""), reps(109, "L")
R90["E: one-hot"]  = one_hot(d90.sequence.tolist(), 90)
R109["E: one-hot"] = one_hot(d109.sequence.tolist(), 109)
for k in R90:
    print(f"{k:<14} 90nt {R90[k].shape}   109nt {R109[k].shape}")
'''),
md("## Paired comparison, per arm"),
co(r'''
for scheme, folds in FOLDS["schemes"].items():
    print(f"=== target ln(Ts) | {scheme} ===")
    print(f"{'arm':<14}{'90 nt':>9}{'109 nt':>9}{'diff':>9}{'wins':>7}{'p':>8}")
    for k in R90:
        a, _ = per_fold(RIDGE, R90[k], y, folds)
        b, _ = per_fold(RIDGE, R109[k], y, folds)
        d = b - a
        print(f"{k:<14}{a.mean():>+9.3f}{b.mean():>+9.3f}{d.mean():>+9.3f}"
              f"{int((d>0).sum()):>5}/10{wilcoxon(a, b).pvalue:>8.3f}")
    print()
'''),
md(r"""
**Reading:** the 19 nt downstream carries real signal - one-hot gains most, and it is a purely
linear model on raw nucleotides, so the information is unambiguously present. But *mean-pooled
layer 12* is the one arm that cannot use it. That motivated the pooling ablation in notebook 03.
"""),
]

# ---------------------------------------------------------------- 03 pooling
p = [
md(r"""
# 03 - pooling ablation

Hypothesis from notebook 02: mean-pooling over 109 positions **dilutes** a signal concentrated
near the 3' end, which is why L12-mean was the only arm that failed to gain from the downstream
window.

Window layout for the 109 nt case:

```
token   0        <cls>
tokens  1..90    upstream, ending AT the termination site
tokens 91..109   the 19 nt downstream
token  110       <eos>
```

Everything below uses frozen embeddings, identical folds, target `ln(Ts)`.
"""),
co(SETUP),
co(HELPERS),
co(r'''
L, UP = 109, 90
df = pd.read_csv(REPO/"probing/data/chen2013_window_up90_dn19.csv").rename(columns={"window":"sequence","strength":"Ts"})
y = np.log(df.Ts.to_numpy())

def poolings(E):
    nt   = E[:, 1:L+1, :]        # nucleotides only
    upst = E[:, 1:UP+1, :]       # through the TTS
    down = E[:, UP+1:L+1, :]     # the 19 nt downstream
    return {
        "mean (all)":            nt.mean(1),
        "max (all)":             nt.max(1),
        "mean+max":              np.hstack([nt.mean(1), nt.max(1)]),
        "<cls>":                 E[:, 0, :],
        "mean upstream only":    upst.mean(1),
        "mean downstream only":  down.mean(1),
        "upstream+downstream":   np.hstack([upst.mean(1), down.mean(1)]),
        "mean last 30":          nt[:, -30:, :].mean(1),
        "mean last 10":          nt[:, -10:, :].mean(1),
        "flatten all positions": nt.reshape(len(E), -1),   # free positional weighting
    }
print("target ln(Ts) | n =", len(y))
'''),
co(r'''
rows = []
for layer in [6, 12]:
    E = np.load(EMB / f"rnafm_L{L}_layer{layer}_tokens.npy")
    P = poolings(E)
    for scheme, folds in FOLDS["schemes"].items():
        base = None
        print(f"=== layer {layer} | {scheme} ===")
        print(f"{'pooling':<24}{'dim':>8}{'R2':>9}{'sd':>8}{'vs mean':>10}{'wins':>7}{'p':>8}")
        for name, X in P.items():
            r, _ = per_fold(RIDGE, X, y, folds)
            if name == "mean (all)":
                base = r
                print(f"{name:<24}{X.shape[1]:>8}{r.mean():>+9.3f}{r.std():>8.3f}{'--':>10}{'--':>7}{'--':>8}")
            else:
                d = r - base
                print(f"{name:<24}{X.shape[1]:>8}{r.mean():>+9.3f}{r.std():>8.3f}"
                      f"{d.mean():>+10.3f}{int((d>0).sum()):>5}/10{wilcoxon(base, r).pvalue:>8.3f}")
            rows.append(dict(layer=layer, cv=scheme, pooling=name, dim=X.shape[1],
                             r2_mean=r.mean(), r2_sd=r.std()))
        print()
    del P, E
abl = pd.DataFrame(rows)
'''),
co(r'''
print("top 5 by GroupKFold R2:")
print(abl[abl.cv=="GroupKFold"].sort_values("r2_mean", ascending=False).head(5)
      .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
'''),
md(r"""
**Reading:** the dilution hypothesis holds for **layer 12** (restricting to the last 30
positions gains a lot) but is **false for layer 6**, which instead wants a second summary
statistic over the same positions (`mean+max`).

Two negative results worth keeping: `flatten all positions` (69,760 features, free positional
weighting) does **not** win - ridge cannot exploit that freedom at n=582, so the implicit
averaging in pooling is acting as useful regularisation. And `mean downstream only` scores as
well as the full window at layer 6, i.e. those 19 positions already encode the upstream through
self-attention.

**Caveat:** 10 poolings x 2 layers x 2 schemes = 40 configurations, and the best was selected
post hoc. With fold sd around 0.13, re-run the winner on a fresh fold seed before relying on it.
"""),
]

# ---------------------------------------------------------------- 04 baselines
b = [
md(r"""
# 04 - baselines: representation x model class

Is RNA-FM actually better than the preprint's 16 hand-crafted features? The answer depends on
the head you put on top, so this notebook fills in the full 2x2.

All four cells use **identical GroupKFold folds** and target `ln(Ts)`.

Hand-crafted features come from the May reproduction in `handcrafted_features/features_output-all.csv`.
Note those were computed before the pheA-1 / ECK120034435 annotation swap, so 2 of 582 rows
carry the uncorrected values there.
"""),
co(SETUP),
co(HELPERS),
co(r'''
from xgboost import XGBRegressor

df = pd.read_csv(REPO/"probing/data/chen2013_window_up90.csv").rename(columns={"strength":"Ts"})
y  = np.log(df.Ts.to_numpy())

# best RNA-FM arm from notebook 02: layer 6, mean, 109 nt window
E6 = np.load(EMB/"rnafm_L109_layer6_tokens.npy")[:, 1:110, :]
RNAFM = E6.mean(1)

hc = pd.read_csv(REPO/"handcrafted_features/features_output-all.csv")
drop = ['Name','Average Strength','Orig_index','Utract','Atract','Stem','Hairpin','Loop','Structure']
HC = hc.set_index("Name").reindex(df.name)[[c for c in hc.columns if c not in drop]].to_numpy()
assert np.isfinite(HC).all(), "feature join failed"
print("RNA-FM", RNAFM.shape, "  hand-crafted", HC.shape)
'''),
co(r'''
# the preprint's tuned hyperparameters
TUNED = dict(n_estimators=4000, learning_rate=0.001, max_depth=5,
             min_child_weight=3, subsample=0.65, random_state=42)
XGB = lambda: XGBRegressor(**TUNED)
folds = FOLDS["schemes"]["GroupKFold"]

cells = {}
for rn, X in [("RNA-FM L6 mean", RNAFM), ("hand-crafted 16", HC)]:
    for mn, mk in [("Ridge", RIDGE), ("XGBoost", XGB)]:
        r, _ = per_fold(mk, X, y, folds)
        cells[(rn, mn)] = r
        print(f"  {rn:<20} + {mn:<8} R2 = {r.mean():+.3f} +/- {r.std():.3f}", flush=True)
'''),
co(r'''
print("=== 2x2: representation x model class, GroupKFold, ln(Ts) ===")
def ms(r):
    return f"{r.mean():+.3f} +/- {r.std():.3f}"   # per-fold R2 as mean +/- sd, as in the cell above
print(f"{'':<22}{'Ridge':>20}{'XGBoost':>20}")
for rn in ["RNA-FM L6 mean", "hand-crafted 16"]:
    print(f"{rn:<22}{ms(cells[(rn,'Ridge')]):>20}{ms(cells[(rn,'XGBoost')]):>20}")

print("\n=== paired tests (identical folds) ===")
def cmp(a, b, label):
    d = cells[a] - cells[b]
    print(f"{label:<50}{d.mean():>+8.3f}  wins {int((d>0).sum())}/10  p={wilcoxon(cells[a], cells[b]).pvalue:.3f}")
cmp(("RNA-FM L6 mean","Ridge"),   ("hand-crafted 16","XGBoost"), "RNA-FM+Ridge   - handcrafted+XGBoost")
cmp(("RNA-FM L6 mean","XGBoost"), ("hand-crafted 16","XGBoost"), "RNA-FM+XGBoost - handcrafted+XGBoost")
cmp(("RNA-FM L6 mean","XGBoost"), ("RNA-FM L6 mean","Ridge"), "RNA-FM+XGBoost - RNA-FM+Ridge")
cmp(("hand-crafted 16","XGBoost"),    ("hand-crafted 16","Ridge"),    "handcrafted+XGBoost - handcrafted+Ridge")
'''),
md(r"""
**Why this matters:** an earlier ridge-vs-ridge comparison made RNA-FM look far ahead, but the
hand-crafted features need a nonlinear head to work at all. Comparing them under the same head
is the fair test of the *representation*; comparing best-pipeline to best-pipeline is the fair
test of the *method*. They give different answers, so report which one you mean.
"""),
]

for name, cells_ in [("nb02_window_comparison.ipynb", w),
                     ("nb03_pooling_ablation.ipynb", p),
                     ("nb04_baselines.ipynb", b)]:
    (HERE / name).write_text(json.dumps(nb(cells_), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name, "-", len(cells_), "cells")
