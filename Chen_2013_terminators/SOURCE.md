# Chen 2013 terminator length table

`terminator_lengths.csv` — one row per terminator, 582 rows.

**Origin.** Derived from the supplementary tables of

> Chen Y-J. *et al.* (2013) "Characterization of 582 natural and synthetic terminators and
> quantification of their design constraints." *Nature Methods* **10**, 659–664.
> doi:10.1038/nmeth.2515  — dataset `D02` in `datasets_v2.csv`.

* `41592_2013_BFnmeth2515_MOESM206_ESM.xlsx`, sheet `Fig2a-natural` — 317 **natural**
  *E. coli* / phage terminators (`ECK…` IDs)
* `41592_2013_BFnmeth2515_MOESM207_ESM.xlsx`, sheet `Sheet1` — 265 **synthetic**
  terminators (`L3S…` IDs)

The decomposition was computed in `terminator_length_distribution.ipynb` in the
`intri_term` working folder; this CSV is that notebook's export, copied here so
`terminator_length_distributions.ipynb` is reproducible without a cross-project path.

**Columns.** `terminator_len` = `A_tract_len` + `hairpin_len` + `U_tract_len`. In this
dataset the A-tract is always 8 nt and the U-tract always 12 nt — the authors' fixed
annotation windows — so all length variation is in the hairpin. `stem_bp` / `loop_len` /
`bulge_len` come from the dot-bracket `Structure` string. `cloned_seq_len` is the full
cloned insert (terminator **plus** vector scaffold, 23–168 nt) and is *not* a terminator
length.

**Caveat when comparing with the TERMITe atlas.** The two datasets use different tract
conventions (8 nt / 12 nt here vs 15 nt / 15 nt in TERMITe) and different anchor points,
so absolute medians are not directly comparable. What *is* comparable is the upper end:
both put every terminator inside an 80 nt window.
