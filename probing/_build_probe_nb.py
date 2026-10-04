"""Builds probe_rnafm.ipynb. Run once, then delete."""
import json
from pathlib import Path

C = []
def md(s): C.append({"cell_type": "markdown", "metadata": {},
                     "source": s.strip("\n").splitlines(keepends=True)})
def co(s): C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""
# Linear probe of RNA-FM embeddings for terminator strength

Does a pretrained RNA language model carry information about termination efficiency
beyond hand-crafted thermodynamic features?

Data: Chen et al. 2013, 582 E. coli constructs, in vivo dual-fluorescence reporter.
Backbone frozen throughout - only a ridge head is fitted.

**Targets:** `ln(Ts)` (primary) and raw `Ts` (secondary).
Ts spans 0.326-382, errors are multiplicative, so the log matches the noise structure
and stops a handful of very strong terminators dominating squared error. `ln(Ts)` is
defined for all 582 rows (min Ts > 0), so no rows are dropped.
""")

co(r"""
import json, argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import fm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED     = 42
N_FOLDS  = 10
LAYERS   = [6, 9, 12]
BATCH    = 16
ALPHAS   = np.logspace(-3, 6, 40)

REPO   = Path.cwd().parent if Path.cwd().name == "probing" else Path.cwd()
CSV    = REPO / "probing" / "data" / "chen2013_window_up90.csv"
CKPT   = REPO / "RNA-FM_stuff" / "pre-trained-models" / "RNA-FM_pretrained.pth"
EMB    = REPO / "probing" / "embeddings"; EMB.mkdir(parents=True, exist_ok=True)
RES    = REPO / "probing" / "results";    RES.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED); torch.manual_seed(SEED)
print("seed", SEED, "| folds", N_FOLDS, "| layers", LAYERS)
print("csv ", CSV.exists(), CSV)
print("ckpt", CKPT.exists(), CKPT)
""")

co(r"""
df = pd.read_csv(CSV)
# column mapping: this CSV uses name / window / strength for the spec's name / sequence / Ts
df = df.rename(columns={"window": "sequence", "strength": "Ts"})
print("columns:", list(df.columns))
print("shape  :", df.shape)

L = df.sequence.str.len().unique()
assert len(L) == 1, f"sequences are not all the same length: {L}"
L = int(L[0])
assert set("".join(df.sequence)) <= set("ACGU"), "non-ACGU character present"
assert df.name.is_unique, "duplicate names"
assert (df.Ts > 0).all(), "Ts must be positive for log()"
print(f"n = {len(df)} | window length L = {L} | alphabet OK | Ts {df.Ts.min():.3f}-{df.Ts.max():.2f}")
""")

md("## Step 1 - extract embeddings")

co(r"""
torch.serialization.add_safe_globals([argparse.Namespace])   # torch>=2.6 needs this
model, alphabet = fm.pretrained.rna_fm_t12(str(CKPT))
batch_converter = alphabet.get_batch_converter()
model.eval()
print("params:", f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M")
""")

co(r"""
# verify the token layout on ONE example before processing everything
probe = [(df.name[0], df.sequence[0])]
_, _, tok = batch_converter(probe)
print("sequence length      :", len(df.sequence[0]))
print("batch_tokens shape   :", tuple(tok.shape))
assert tok.shape[1] == L + 2, f"expected L+2={L+2} tokens, got {tok.shape[1]}"
print("index 0 (cls)        :", int(tok[0, 0]), "==", alphabet.cls_idx)
print("index L+1 (eos)      :", int(tok[0, -1]), "==", alphabet.eos_idx)
assert int(tok[0, 0]) == alphabet.cls_idx and int(tok[0, -1]) == alphabet.eos_idx
print("padding idx          :", alphabet.padding_idx, "-> should never appear (fixed length)")
assert (tok == alphabet.padding_idx).sum() == 0, "padding found in a fixed-length batch"
print("\nOK: token 0 is <cls>, tokens 1..L are nucleotides, token L+1 is <eos>")
""")

co(r"""
# full per-token embeddings, all requested layers in ONE forward pass
n, D = len(df), 640
store = {l: np.zeros((n, L + 2, D), dtype=np.float32) for l in LAYERS}

with torch.no_grad():
    for s in range(0, n, BATCH):
        chunk = list(zip(df.name[s:s+BATCH], df.sequence[s:s+BATCH]))
        _, _, tk = batch_converter(chunk)
        assert (tk == alphabet.padding_idx).sum() == 0
        out = model(tk, repr_layers=LAYERS)
        for l in LAYERS:
            store[l][s:s+len(chunk)] = out["representations"][l].numpy()
        if s % (BATCH * 8) == 0:
            print(f"  {s:4d}/{n}", flush=True)

for l in LAYERS:
    p = EMB / f"rnafm_layer{l}_tokens.npy"
    np.save(p, store[l])
    print(f"layer {l:2d}: {store[l].shape} -> {p.name}  ({p.stat().st_size/1e6:.0f} MB)")

# row order, so embeddings can always be matched back to names
(EMB / "row_order.json").write_text(json.dumps(list(df.name)), encoding="utf-8")
print("row order saved:", len(df), "names")
""")

md("## Step 2 - build feature matrices")

co(r"""
# tokens 1..L are the nucleotides; 0 is <cls>, L+1 is <eos>
def mean_pool(arr):  return arr[:, 1:L+1, :].mean(axis=1)
def cls_vec(arr):    return arr[:, 0, :]

def one_hot(seqs):
    idx = {c: i for i, c in enumerate("ACGU")}
    M = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            M[i, j, idx[ch]] = 1.0
    return M.reshape(len(seqs), L * 4)

REPS = {
    "A: L12 mean":  mean_pool(store[12]),
    "B: L12 <cls>": cls_vec(store[12]),
    "C: L6 mean":   mean_pool(store[6]),
    "D: L9 mean":   mean_pool(store[9]),
    "E: one-hot":   one_hot(df.sequence.tolist()),
}
for k, v in REPS.items():
    print(f"{k:<14} {v.shape}   finite={np.isfinite(v).all()}")
assert all(v.shape[0] == len(df) for v in REPS.values())
""")

md("## Step 3 - targets, groups, folds")

co(r"""
TARGETS = {
    "ln(Ts)": np.log(df.Ts.to_numpy()),
    "Ts":      df.Ts.to_numpy(),
}
for k, v in TARGETS.items():
    print(f"{k:<9} n={len(v)}  range {v.min():8.3f} .. {v.max():8.3f}  "
          f"mean {v.mean():7.3f}  sd {v.std():6.3f}  NaN={int(np.isnan(v).sum())}")
""")

co(r"""
# Are the synthetic constructs variants on shared parent scaffolds?
groups = df.hairpin.to_numpy()
vc = pd.Series(groups).value_counts()
print(f"distinct hairpins            : {len(vc)} for {len(df)} rows")
print(f"singleton groups             : {int((vc == 1).sum())}")
print(f"multi-member groups          : {int((vc > 1).sum())}  covering {int(vc[vc>1].sum())} rows")
print(f"largest groups               : {list(vc.head(5).values)}")

# how similar are two rows inside the same group, over the whole window?
same, diff = [], []
rng = np.random.default_rng(SEED)
S = df.sequence.to_numpy()
for hp, g in df.groupby("hairpin").groups.items():
    g = list(g)
    if len(g) > 1:
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                same.append(np.mean([a == b for a, b in zip(S[g[i]], S[g[j]])]))
for _ in range(2000):
    i, j = rng.integers(0, len(df), 2)
    if groups[i] != groups[j]:
        diff.append(np.mean([a == b for a, b in zip(S[i], S[j])]))
print(f"\nwindow identity, same hairpin      : mean {np.mean(same):.1%}  (n={len(same)} pairs)")
print(f"window identity, different hairpin : mean {np.mean(diff):.1%}  (n={len(diff)} pairs)")
print("\n=> near-duplicate groups DO exist. Reporting BOTH plain KFold and GroupKFold.")
""")

co(r"""
# folds generated ONCE, saved, reloaded for every arm
FOLD_FILE = RES / "folds.json"
if not FOLD_FILE.exists():
    schemes = {}
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    schemes["KFold"] = [[tr.tolist(), te.tolist()] for tr, te in kf.split(np.arange(len(df)))]
    gkf = GroupKFold(n_splits=N_FOLDS)
    schemes["GroupKFold"] = [[tr.tolist(), te.tolist()]
                             for tr, te in gkf.split(np.arange(len(df)), groups=groups)]
    FOLD_FILE.write_text(json.dumps({"seed": SEED, "n_folds": N_FOLDS, "schemes": schemes}),
                         encoding="utf-8")
    print("folds generated and saved")
FOLDS = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
print("loaded folds from", FOLD_FILE.name, "| seed", FOLDS["seed"])
for name, sch in FOLDS["schemes"].items():
    sizes = [len(te) for _, te in sch]
    leak = np.mean([len(set(groups[te]) & set(groups[tr])) / max(1, len(set(groups[te])))
                    for tr, te in sch])
    print(f"  {name:<11} test sizes {min(sizes)}-{max(sizes)}   "
          f"test groups also in train: {leak:.1%}")
""")

md("## Step 4 - run the arms")

co(r"""
def run_arm(X, y, folds):
    preds = np.full(len(y), np.nan)
    r2s, rhos, alphas = [], [], []
    for tr, te in folds:
        tr, te = np.array(tr), np.array(te)
        pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        pipe.fit(X[tr], y[tr])                       # alpha chosen on train only
        p = pipe.predict(X[te])
        preds[te] = p
        r2s.append(r2_score(y[te], p))
        rhos.append(spearmanr(y[te], p).statistic)
        alphas.append(pipe[-1].alpha_)
    return np.array(r2s), np.array(rhos), preds, alphas

rows, all_preds = [], {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for tname, y in TARGETS.items():
        for scheme, folds in FOLDS["schemes"].items():
            for rname, X in REPS.items():
                r2s, rhos, preds, al = run_arm(X, y, folds)
                rows.append(dict(target=tname, cv=scheme, arm=rname,
                                 r2_mean=r2s.mean(), r2_sd=r2s.std(),
                                 rho_mean=rhos.mean(), rho_sd=rhos.std()))
                all_preds[f"{tname}|{scheme}|{rname}"] = preds
                print(f"{tname:<8} {scheme:<11} {rname:<14} "
                      f"R2 {r2s.mean():+.3f}+/-{r2s.std():.3f}   "
                      f"rho {rhos.mean():+.3f}+/-{rhos.std():.3f}", flush=True)

results = pd.DataFrame(rows)
results.to_csv(RES / "probe_results.csv", index=False)
np.savez_compressed(RES / "fold_predictions.npz", **all_preds)
print("\nsaved probe_results.csv and fold_predictions.npz")
""")

md("## Step 5 - report")

co(r"""
for tname in TARGETS:
    for scheme in FOLDS["schemes"]:
        sub = results[(results.target == tname) & (results.cv == scheme)]
        print(f"\n=== target {tname} | {scheme} ===")
        print(sub[["arm", "r2_mean", "r2_sd", "rho_mean", "rho_sd"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
""")

co(r"""
print("Per-fold R2 (target ln(Ts)), to show the spread:\n")
y = TARGETS["ln(Ts)"]
for scheme, folds in FOLDS["schemes"].items():
    print(f"--- {scheme} ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rname, X in REPS.items():
            r2s, _, _, _ = run_arm(X, y, folds)
            print(f"  {rname:<14} " + " ".join(f"{v:+.2f}" for v in r2s)
                  + f"   | mean {r2s.mean():+.3f}")
""")

co(r"""
def get(t, cv, arm, col="r2_mean"):
    return results[(results.target==t)&(results.cv==cv)&(results.arm==arm)][col].iloc[0]

lines = []
for scheme in FOLDS["schemes"]:
    a  = get("ln(Ts)", scheme, "A: L12 mean")
    e  = get("ln(Ts)", scheme, "E: one-hot")
    sd = max(get("ln(Ts)", scheme, "A: L12 mean", "r2_sd"),
             get("ln(Ts)", scheme, "E: one-hot",  "r2_sd"))
    best_layer = max([("L12", get("ln(Ts)", scheme, "A: L12 mean")),
                      ("L9",  get("ln(Ts)", scheme, "D: L9 mean")),
                      ("L6",  get("ln(Ts)", scheme, "C: L6 mean"))], key=lambda t: t[1])
    cls  = get("ln(Ts)", scheme, "B: L12 <cls>")
    mean = get("ln(Ts)", scheme, "A: L12 mean")
    lines += [
        f"[{scheme}]  target ln(Ts)",
        f"  RNA-FM L12 mean vs one-hot : {a:+.3f} vs {e:+.3f}  "
        f"(difference {a-e:+.3f} = {abs(a-e)/sd:.1f}x the fold-to-fold sd of {sd:.3f})",
        f"  best layer                 : {best_layer[0]} at R2 {best_layer[1]:+.3f}",
        f"  <cls> vs mean pooling      : {cls:+.3f} vs {mean:+.3f} -> "
        f"{'<cls> better' if cls > mean else 'mean pooling better'}",
        "",
    ]
summary = "\n".join(lines)
print(summary)
(RES / "summary.txt").write_text(summary, encoding="utf-8")
print("saved summary.txt")
""")

nb = {"cells": C, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python (TE_model)",
                                  "language": "python", "name": "te_model"},
                   "language_info": {"name": "python"}}}
out = Path(__file__).parent / "probe_rnafm.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out.name, "-", len(C), "cells")
