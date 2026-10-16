# TE_model

Predicting bacterial intrinsic transcription termination efficiency (TE) from sequence.

Maintained by Siwei Qian.

Goal: build a model that predicts TE from the terminator sequence, see how it compares to the
current state of the art (hand-crafted features + XGBoost), and later look into the model to see
if it tells us anything new about the mechanism of intrinsic termination.

## Writeup

[`ms3_probing_writeup-v3.pdf`](ms3_probing_writeup-v3.pdf): *A frozen RNA language model matches
hand-crafted features for terminator strength prediction*. A short paper-style report
(introduction, methods, results, discussion) on parts 2–4 below, and the quickest way to see what
the repo found. On the 582 Chen 2013 terminators, with ln(Ts) as the target and 10-fold CV grouped
by hairpin:

- Every RNA-FM embedding arm (layers 6, 9 and 12, mean pooled or `<cls>`) beats one-hot sequence
  with the same ridge head. Best is layer 12 mean pooling, R² 0.595 against 0.467.
- Adding 19 nt downstream of the termination site helps layer 6 embeddings and one-hot.
- RNA-FM layer 6 embeddings (109 nt) with ridge reach R² 0.634, level with the 16 hand-crafted
  features of Kundlatsch et al 2026 with XGBoost (0.620). The hand-crafted features need XGBoost to
  get there; with ridge they reach 0.429.

Where the numbers come from: Tables 1–3 in `probing/probe_rnafm.ipynb`, Table 4 in
`probing/nb02_window_comparison.ipynb`, Tables 5–6 in `probing/nb04_baselines.ipynb`. Figure 2 is
`probing/figures/strength_ranked_linear_and_logts.png`, Figure 3 is
`outputs/terminator_length/figures/fig08_input_window_clean.png`. The rough noise-ceiling check in
the discussion is `probing/nb05_noise_ceiling.ipynb`.

The repo started as a survey of public datasets and grew from there. Four parts, roughly in the
order they were done:

## 1. Public dataset survey

[`public_dataset_survey/`](public_dataset_survey/README.md): what public TE data exists and what
it can and cannot support. `datasets_v2.csv` in there is the inventory. Picked Chen et al 2013
(`D02`) as the main labeled dataset: 582 natural + synthetic terminators in E. coli, TE measured
in vivo with a dual fluorescence reporter. The README also includes a brief brainstormed study plan (as of July 2026).

## 2. How many nt the input window needs

[`terminator_length_distributions.ipynb`](terminator_length_distributions.ipynb), figures and
tables in [`outputs/terminator_length/`](outputs/terminator_length/). Looks at terminator lengths
in the TERMITe atlas (Kosinski 2025, 11,769 terminators from 13 species, raw tables in
[`Kosinski_2025_term-seq_sources_terminators_altas/`](Kosinski_2025_term-seq_sources_terminators_altas/))
and in Chen 2013. Every terminator in both fits within 80 nt upstream of its 3' end, so the 90 nt
upstream window used for training covers all of them. 90 nt upstream + 19 nt downstream is the
second option.

## 3. Chen 2013 data

[`Chen_2013_terminators/`](Chen_2013_terminators/) has the original supplementary tables from the
paper. [`make_csv_chen.ipynb`](make_csv_chen.ipynb) turns them into the training tables in
[`probing/data/`](probing/data/): fixes the pheA-1 / ECK120034435 annotation swap in the
supplement, adds back the plasmid flanking sequence, anchors on the hairpin, and cuts the 90 nt
and 90 + 19 nt windows.

## 4. Training models

- [`probing/`](probing/README.md): frozen RNA-FM embeddings + a ridge or XGBoost head, target
  ln(Ts). `probe_rnafm.ipynb` is the probe itself, `nb02`–`nb04` go through window size, pooling
  and the baselines. Main results use CV grouped by hairpin, since many Chen constructs are
  variants of the same hairpin and a random split leaks.
- [`handcrafted_features/`](handcrafted_features/README.md): reproduction of the 16 hand-crafted
  features + XGBoost from Kundlatsch et al 2026, the baseline RNA-FM is compared against.

## Setup

`requirements.txt` for the analysis notebooks, `requirements-ml.txt` for the modeling (torch,
rna-fm, xgboost).

The RNA-FM weights (1.19 GB) are not in the repo. Download
[`RNA-FM_pretrained.pth`](https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth)
and put it at `RNA-FM_stuff/pre-trained-models/RNA-FM_pretrained.pth`.

