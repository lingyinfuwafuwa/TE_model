"""Pooling ablation.

Hypothesis from the 90-vs-109 result: mean-pooling over 109 positions dilutes a
signal concentrated in the last 19 nt, which is why L12-mean was the only arm that
failed to gain from the downstream window.

Tests pooling strategies on identical folds, identical target, frozen embeddings.
Window layout, 109 nt case:  token 0 = <cls>, tokens 1..90 = upstream through the
TTS, tokens 91..109 = the 19 nt downstream, token 110 = <eos>.
"""
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.simplefilter("ignore")

REPO = Path(__file__).resolve().parent.parent
EMB, RES = REPO / "probing/embeddings", REPO / "probing/results"
ALPHAS = np.logspace(-3, 6, 40)
L, UP = 109, 90                       # 90 upstream (ends at TTS) + 19 downstream

df = (pd.read_csv(REPO / "probing/data/chen2013_window_up90_dn19.csv")
        .rename(columns={"window": "sequence", "strength": "Ts"}))
y = np.log(df.Ts.to_numpy())
FOLDS = json.loads((RES / "folds.json").read_text())
print(f"n={len(df)}  L={L}  target=ln(Ts)  folds={FOLDS['n_folds']}  seed={FOLDS['seed']}\n")


def poolings(E):
    """E: (n, L+2, 640). Token 0 is <cls>; 1..L are nucleotides; L+1 is <eos>."""
    nt   = E[:, 1:L+1, :]                 # nucleotides only
    upst = E[:, 1:UP+1, :]                # 1..90   -> upstream, ends at the TTS
    down = E[:, UP+1:L+1, :]              # 91..109 -> the 19 nt downstream
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


def per_fold(X, y, folds):
    out = []
    for tr, te in folds:
        tr, te = np.array(tr), np.array(te)
        p = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[tr], y[tr]).predict(X[te])
        out.append(r2_score(y[te], p))
    return np.array(out)


rows = []
for layer in [6, 12]:
    E = np.load(EMB / f"rnafm_L{L}_layer{layer}_tokens.npy")
    P = poolings(E)
    for scheme, folds in FOLDS["schemes"].items():
        base = None
        print(f"=== layer {layer} | {scheme} ===")
        print(f"{'pooling':<24}{'dim':>8}{'R2':>9}{'sd':>8}{'vs mean':>10}{'wins':>7}{'p':>8}")
        for name, X in P.items():
            r = per_fold(X, y, folds)
            if name == "mean (all)":
                base = r
                print(f"{name:<24}{X.shape[1]:>8}{r.mean():>+9.3f}{r.std():>8.3f}"
                      f"{'--':>10}{'--':>7}{'--':>8}")
            else:
                d = r - base
                print(f"{name:<24}{X.shape[1]:>8}{r.mean():>+9.3f}{r.std():>8.3f}"
                      f"{d.mean():>+10.3f}{int((d>0).sum()):>5}/10"
                      f"{wilcoxon(base, r).pvalue:>8.3f}")
            rows.append(dict(layer=layer, cv=scheme, pooling=name, dim=X.shape[1],
                             r2_mean=r.mean(), r2_sd=r.std()))
        print()
    del P, E

out = pd.DataFrame(rows)
out.to_csv(RES / "probe_pooling_ablation.csv", index=False)
print("saved probe_pooling_ablation.csv")

best = out[out.cv == "GroupKFold"].sort_values("r2_mean", ascending=False).head(5)
print("\ntop 5 by GroupKFold R2:")
print(best.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
