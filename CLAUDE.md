# TE_model — working notes

Predicting bacterial intrinsic transcription termination efficiency (TE) from sequence.

The root `README.md` is the project overview and the GitHub front page. Behind it, the repo has
two halves, and they are different kinds of work:

- **Survey** — `public_dataset_survey/` (`README.md` + `datasets_v2.csv`): what public data
  exists and what it can support. Scripts find the repo root by walking up to that CSV.
- **Implementation** — `probing/`: frozen RNA-FM embeddings into a linear head on Chen 2013.
  See `probing/README.md`. The hand-crafted-feature baseline it is compared against
  (the Kundlatsch 2026 preprint's 16 features + XGBoost) lives in `handcrafted_features/`.

## Environment

Analysis stack is pinned in `requirements.txt`; the modeling stack (torch, rna-fm, sklearn) is
in `requirements-ml.txt`. **No seaborn anywhere — write plotting code against matplotlib only.**

Machine-specific setup (venv path, activation, kernel name) is in `ENVIRONMENT.local.md`, which
is gitignored. Do not put local paths in any `README.md` — they go to GitHub.

Gitignored on purpose: `*.local.md`, `RNA-FM_stuff/` (1.19 GB of weights; provenance and loading
instructions in `RNA-FM_stuff/SOURCE.md`), and the usual checkpoint extensions.

## Gotchas that have already cost time

- **`public_dataset_survey/datasets_v2.csv` is not UTF-8.** Byte `0x96` (cp1252 en-dash) at offset 7647.
  `pd.read_csv(..., encoding="cp1252")`.
- **The Chen `.xlsx` files are often open in Excel**, which makes `pd.read_excel` raise
  `PermissionError`. `shutil.copyfile` fails the same way (it uses `open()`); Win32 `CopyFileW`
  gets through. Working fallback: `probing/plot_te_energy.py:read_excel_unlocked`.
- **`Sequence` is DNA (`T`), every parts column is RNA (`U`).** Convert before matching or *zero*
  hairpins match, and the failure looks like a data problem rather than an alphabet problem.
  Work in `U` throughout — RNA-FM wants RNA anyway.
- **rna-fm 0.2.2 cannot load its own checkpoint on torch >= 2.6.** It calls `torch.load()` without
  `weights_only=False`. Fix with `torch.serialization.add_safe_globals([argparse.Namespace])` —
  never with `weights_only=False`. Details in `RNA-FM_stuff/SOURCE.md`.
- **`fm.pretrained.rna_fm_t12(path)` silently re-downloads 1.19 GB if `path` does not exist.**
  Assert the file is there first.

## Chen 2013 (D02) specifics

The main labeled dataset. 582 terminators; `Chen_2013_terminators/SOURCE.md` describes the tidy
parts table, `probing/README.md` the label, and `~/.claude/plans/agile-beaming-babbage.md` the
window-extraction algorithm.

- **Anchor on the `Hairpin` column, nothing else.** Exact-match rate inside the insert: hairpin
  570/582, A-tract 337, U-tract 235. A-tract is 8 nt and U-tract 12 nt for *all* rows, so they
  are fixed annotation windows — the "termination site" is just `hairpin_end + 12`, a convention
  and not a measurement.
- **`Sequence` is only the cloned insert, and the terminator often runs off both ends.** 342/582
  rows have their U-tract past the 3' end; only 1/582 has 90 nt of upstream context inside the
  insert. Vector flanking sequence is required, not optional.
- **`pheA-1` and `ECK120034435` have each other's annotation blocks** (adjacent rows, off-by-one
  in the supplement). Confirmed two ways: each one's listed hairpin sits in the other's insert,
  and the `Scaffold 1` column marks pheA-1 together with the `L3S1*` family whose hairpin is the
  one misfiled under ECK120034435.
- **90 of 582 rows have `T_S <= 1`**, so `TE = 1 - 1/T_S` is negative and `logit(TE)` is
  undefined — `np.log` returns `nan` silently. Distinct from `logit(TE)` merely being negative,
  which is fine and applies to 133 rows. Options are weighed in `probing/README.md`.
- **Do not split randomly.** 47% of rows share a hairpin with another row; group identity
  explains 87% of label variance (ICC 0.866), and a random 80/20 split leaks ~43% of the test
  set. Chen's synthetic libraries are combinatorial variants on three natural scaffolds. Group
  on hairpin similarity; recipe in the plan file.
- **Do not reverse-complement the `-R` rows** — they are already tabulated in transcript sense.
- `Predicted TS` is the authors' predicted *strength*, not a termination site.

## Conventions

- Figures are PNG committed under `outputs/<analysis>/figures/` or `probing/figures/`, written by
  a script that can regenerate them. Tables go in a sibling `tables/`.
- Load the `dataviz` skill before writing chart code.
- Math in Markdown uses `$...$` / `$$...$$` and `\mathrm`, not `\(...\)` or `\operatorname` —
  GitHub renders neither of the latter.
