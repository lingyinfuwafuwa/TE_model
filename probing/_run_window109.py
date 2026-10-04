"""Same probe, on the 90+19 nt window. Reuses folds.json so the comparison is paired."""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import fm
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.simplefilter("ignore")

SEED, LAYERS, BATCH = 42, [6, 9, 12], 16
ALPHAS = np.logspace(-3, 6, 40)
REPO = Path(__file__).resolve().parent.parent
EMB, RES = REPO / "probing/embeddings", REPO / "probing/results"
np.random.seed(SEED); torch.manual_seed(SEED)

df = (pd.read_csv(REPO / "probing/data/chen2013_window_up90_dn19.csv")
        .rename(columns={"window": "sequence", "strength": "Ts"}))
ref = pd.read_csv(REPO / "probing/data/chen2013_window_up90.csv")
assert df.name.equals(ref.name), "row order differs -> folds would not be comparable"
L = int(df.sequence.str.len().unique()[0])
assert set("".join(df.sequence)) <= set("ACGU")
print(f"n={len(df)}  L={L}  (90 upstream + 19 downstream)")

cache = EMB / f"rnafm_L{L}_layer12_tokens.npy"
if cache.exists():
    store = {l: np.load(EMB / f"rnafm_L{L}_layer{l}_tokens.npy") for l in LAYERS}
    print("loaded cached embeddings")
else:
    torch.serialization.add_safe_globals([argparse.Namespace])
    model, alphabet = fm.pretrained.rna_fm_t12(
        str(REPO / "RNA-FM_stuff/pre-trained-models/RNA-FM_pretrained.pth"))
    model.eval()
    bc = alphabet.get_batch_converter()
    _, _, t0 = bc([(df.name[0], df.sequence[0])])
    assert t0.shape[1] == L + 2, f"expected {L+2} tokens, got {t0.shape[1]}"
    print(f"token check OK: {tuple(t0.shape)} == (1, L+2)")
    store = {l: np.zeros((len(df), L + 2, 640), dtype=np.float32) for l in LAYERS}
    with torch.no_grad():
        for s in range(0, len(df), BATCH):
            ch = list(zip(df.name[s:s+BATCH], df.sequence[s:s+BATCH]))
            _, _, tk = bc(ch)
            assert (tk == alphabet.padding_idx).sum() == 0
            out = model(tk, repr_layers=LAYERS)
            for l in LAYERS:
                store[l][s:s+len(ch)] = out["representations"][l].numpy()
            if s % (BATCH * 8) == 0:
                print(f"  {s:4d}/{len(df)}", flush=True)
    for l in LAYERS:
        np.save(EMB / f"rnafm_L{L}_layer{l}_tokens.npy", store[l])
    print("embeddings saved")

def one_hot(seqs, L):
    idx = {c: i for i, c in enumerate("ACGU")}
    M = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            M[i, j, idx[c]] = 1.0
    return M.reshape(len(seqs), -1)

REPS = {
    "A: L12 mean":  store[12][:, 1:L+1, :].mean(1),
    "B: L12 <cls>": store[12][:, 0, :],
    "C: L6 mean":   store[6][:, 1:L+1, :].mean(1),
    "D: L9 mean":   store[9][:, 1:L+1, :].mean(1),
    "E: one-hot":   one_hot(df.sequence.tolist(), L),
}
for k, v in REPS.items():
    print(f"  {k:<14} {v.shape}")

FOLDS = json.loads((RES / "folds.json").read_text())
TARGETS = {"ln(Ts)": np.log(df.Ts.to_numpy()), "Ts": df.Ts.to_numpy()}

def per_fold(X, y, folds):
    r2, rho, preds = [], [], np.full(len(y), np.nan)
    for tr, te in folds:
        tr, te = np.array(tr), np.array(te)
        p = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[tr], y[tr]).predict(X[te])
        preds[te] = p
        r2.append(r2_score(y[te], p)); rho.append(spearmanr(y[te], p).statistic)
    return np.array(r2), np.array(rho), preds

rows, allp = [], {}
for tn, y in TARGETS.items():
    for sc, folds in FOLDS["schemes"].items():
        for rn, X in REPS.items():
            r2, rho, pr = per_fold(X, y, folds)
            rows.append(dict(window=f"90+19 ({L}nt)", target=tn, cv=sc, arm=rn,
                             r2_mean=r2.mean(), r2_sd=r2.std(),
                             rho_mean=rho.mean(), rho_sd=rho.std()))
            allp[f"{tn}|{sc}|{rn}"] = pr
            print(f"{tn:<8} {sc:<11} {rn:<14} R2 {r2.mean():+.3f}+/-{r2.std():.3f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "probe_results_window109.csv", index=False)
np.savez_compressed(RES / "fold_predictions_window109.npz", **allp)

# paired 90 vs 109, same folds, same rows
print("\n" + "=" * 74)
print("PAIRED COMPARISON  90 nt  vs  90+19 nt   (identical folds and rows)\n")
e90 = {l: np.load(EMB / f"rnafm_layer{l}_tokens.npy") for l in [12]}
A90 = e90[12][:, 1:91, :].mean(1)
A109 = REPS["A: L12 mean"]
for tn, y in TARGETS.items():
    for sc, folds in FOLDS["schemes"].items():
        a, _, _ = per_fold(A90, y, folds)
        b, _, _ = per_fold(A109, y, folds)
        d = b - a
        print(f"{tn:<8} {sc:<11}  90nt {a.mean():+.3f}   109nt {b.mean():+.3f}   "
              f"diff {d.mean():+.3f}+/-{d.std():.3f}  109 wins {int((d>0).sum())}/{len(d)}  "
              f"p={wilcoxon(a, b).pvalue:.3f}")
print("\nsaved probe_results_window109.csv")
