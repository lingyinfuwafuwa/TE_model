# Hand-crafted terminator features (Kundlatsch et al. 2026 reproduction)

The 16 sequence features of the Kundlatsch et al. 2026 preprint (Terminator Factory, TerFac),
recomputed for the 582 Chen 2013 terminators (`D02`). This is the hand-crafted baseline that
`probing/nb04_baselines.ipynb` compares RNA-FM against.

Feature definitions follow the authors' code:
[gkundlatsch/TerFac-offline](https://github.com/gkundlatsch/TerFac-offline).

## Files

* `feature_extraction.ipynb` — reads the two Chen 2013 supplements from
  `../Chen_2013_terminators/` (MOESM206 natural + MOESM207 synthetic), splits each hairpin
  into stem and apical loop from the dot-bracket `Structure`, computes the features and writes
  `features_output-all.csv`.
* `features_output-all.csv` — 582 rows × 25 columns: `Orig_index`, `Name`,
  `Average Strength` (Chen's $T_S$), the part sequences kept for inspection (`Utract`, `Atract`,
  `Loop`, `Hairpin`, `Structure`, `Stem`), and the 16 features.
* `train_xgboost.ipynb` — XGBoost on the 16 features, with default settings and with the
  preprint's tuned hyperparameters, under 5-fold CV.

Both notebooks use paths relative to this folder, so run them from here.

## Caveats

* Computed from the supplement **as published**, so it does not include the pheA-1 /
  ECK120034435 annotation-swap correction that `make_csv_chen.ipynb` applies to the training
  tables. Those 2 rows carry each other's features.
* `train_xgboost.ipynb` uses a shuffled `KFold`, which puts near-duplicate constructs on both
  sides of a split. `probing/nb04_baselines.ipynb` re-runs the same model on `GroupKFold`.
* The preprint used xgboost 1.7.6; this repo pins 3.2.0.

## Provenance

These were written in May 2026 in a separate working folder as `test.ipynb` and `test2.ipynb`.
They were copied here with only the input paths and kernel metadata changed. Re-running
`feature_extraction.ipynb` against `Chen_2013_terminators/` reproduces the CSV: all 16 features
and every sequence column are identical. The only difference is in one `Average Strength`
value (L2U5H11), which differs by $3 \times 10^{-16}$ because the xlsx was re-saved.
