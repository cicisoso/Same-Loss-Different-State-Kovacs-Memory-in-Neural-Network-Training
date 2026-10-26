# Kovacs memory in neural-network training: code

Two training runs reach the same held-out loss along different learning-rate histories and are then trained identically.
Do they stay together? This directory has the code for every run, analysis and figure in the manuscript "Same Loss,
Different State: Kovacs Memory in Neural-Network Training". The analysis plan for the withheld protocols and the norm
interventions is in [`PREREG.md`](../PREREG.md), and its timestamps precede the runs it covers.

Requirements: Python 3.11, PyTorch 2.x with CUDA, NumPy, SciPy, pandas, matplotlib, tiktoken. All runs were made on one
NVIDIA RTX A6000 (48 GB). Each script states its usage in its header.

## Data
| file | role |
|---|---|
| `dl_domains.sh`, `prep_domains.py` | download and tokenise Python code (codeparrot-clean) and PubMed abstracts (MedRAG PubMed chunks) into GPT-2 token shards; FineWeb uses the public GPT-2 token shards of FineWeb-10B |
| `unigram.py` | unigram counts of a training stream, used to bin held-out target tokens by frequency rank |

## Training
| file | role |
|---|---|
| `kov_train.py` | GPT-style transformer (30M or 124M parameters) trained under piecewise-constant learning-rate schedules given as JSON. Every run from a checkpoint sees the same minibatch at each step. A phase with `"until": "match"` ends when the held-out loss crosses a reference run's evaluation log. Options cover optimizer-state transplant (`--init-opt`), optimizer reset, the exact norm reparameterisations (`--rescale-proj`, `--rescale-resid`) and angular-update logging (`--log-angle`). Writes `<tag>_eval.tsv`, `<tag>_train.tsv`, `<tag>_meta.json` and `<tag>_detail.npz` (per-sequence losses on held-out sets A and B, loss by token-frequency bin and context position, per-matrix norms) |
| `kov_small.py` | the same protocols for a matrix-factorisation recommender (MovieLens-1M) and a residual CNN (CIFAR-10), including the exact filter-norm exchange of the CNN |
| `run_k1.sh` | pilot: base run to step 4000, reference, Kovacs run, single pulses and constant steps |
| `run_k2.sh` | withheld protocol family (100/1000 heating steps, heating at 2x, cooling at 1/4, reverse protocol, optimizer reset, replicates on unseen data) |
| `run_seeds.sh` | five independently initialised networks, each with its own reference and both protocols |
| `run_swap.sh` | matched checkpoints with full optimizer state, the 2x2 optimizer-state transplant, and identical anneals |
| `run_norm.sh` | registered norm interventions at the match (exact reparameterisations; see PREREG.md) |
| `run_k3.sh`, `run_lin.sh` | quench probes; linear-regime steps and pulses, two-pulse superposition, ageing |
| `run_dom.sh`, `run_big.sh`, `run_wd0.sh` | code and PubMed domains, the 124M model, zero weight decay |
| `run_small.sh`, `run_small2.sh`, `run_small_chain.sh`, `run_cifN.sh` | MovieLens and CIFAR-10 learners, with and without momentum; CNN filter-norm exchange |
| `run_pool.sh` | the jobs of `run_seeds.sh` after the first seed's heat->cool run, and of `run_lin.sh`, `run_k3.sh`, `run_wd0.sh` and `run_big.sh`, with identical commands, run as chains that start when the GPU has memory for them |
| `run_pool2.sh` | replaces `run_pool.sh` after it had stopped `run_seeds.sh` (08:35) but before it started any chain: same chains, a 30M chain starts at 8000 instead of 9100 MiB free (nvidia-smi reserves ~600 MiB) |
| `run_bncheck.sh`, `kov_small_bn.py` | exploratory CIFAR check of the batch-norm running-statistics lag: the matched pair of the norm exchange continued again, every evaluation also scored with running statistics re-estimated over 20 fixed minibatches (tmux `kov_bn`, waits for the plain-SGD CIFAR chain) |
| `ts_noise0.py` | check of the zero-label-noise case of `ts_noise.py` (crossing, gap at release, extremes of the excursion) |
| `stop_dom.sh` | stops the domain queue (`run_dom.sh`) after `pubmed_km`: the PubMed pulses and constant steps that follow are used by no analysis or figure; their GPU slot goes to the pool |
| `k_repro_ep.py` | resolution of the registered endpoints from two identical continuations of one matched checkpoint (area, late offset, smoothed maximum) |
| `run_pool3.sh` | takes over from the dispatcher of `run_pool2.sh` (stopped with SIGSTOP at 10:02 after S2-S4 had started; those chains continue): same chain commands; km2 and the two unseen-data replicates of `run_k2.sh` run as parallel chains; the 124M chain follows S5 and WD0; the 124M and zero-weight-decay pulses (no registered use) are separate chains at the end |
| `run_pool4.sh` | takes over from the dispatcher of `run_pool3.sh` (stopped with SIGSTOP at 10:20 after S5 had started): B = 0 anneals of independent matched pairs (seeds 1-3; preparation replayed and saved at the match) and continuations of the transplant pair with fresh optimizer moments (PREREG addendum 10:20) run first; quench probes and the 124M / zero-weight-decay pulses dropped |
| `run_pool5.sh` | takes over from the dispatcher of `run_pool4.sh` (stopped with SIGSTOP at 11:25 after AN1 had started): same chains, with the 124M model, the unseen-data replicates and the ageing pair ahead of zero weight decay and the linear-regime runs |
| `run_pool6.sh` | takes over from the dispatcher of `run_pool5.sh` (stopped with SIGSTOP at 12:27 after AN2 had started) after a CUDA MPS fault killed every GPU job at 12:22; re-queues the failed chains first (AN1 BIG S4 S5 AN3 RST S2 S3, then the rest) and also retries a run after a CUDA error, keeping the failed log |
| `run_pool7.sh` | takes over from the dispatcher of `run_pool6.sh` (stopped with SIGSTOP at 12:55 after AN1, BIG and S4 had started); same chains and retry, new order of the remaining chains (S5 AN3 RST R1 R2 AGE WD0 K2b S2 S3 S1 LINa LINb) and starts spaced by 2.5 min |
| `stop_k2.sh` | stops `run_k2.sh` once `refL_reset` has finished, so that its remaining jobs run in `run_pool3.sh` |
| `run_annN.sh` | exploratory (PREREG addendum 09:44): B = 0 anneals of the matched checkpoints after the function-preserving norm changes (`an_kpN_B0`, `an_refN_B0`) |
| `run_an2.sh` | registered stage-2 analyses (contest and peak-time law on the withheld runs and seeds) |
| `run_an2b.sh` | replaces `run_an2.sh` (its triggers never come after the queues moved to the pool dispatchers): the registered contest (`k_models.py`, `k_score.py`) on the withheld pairs and seeds whose runs have finished, rerun as more finish, and the pair statistics of `km2` from its release; the peak-time law is scored by `k_peaklaw_status.py` |
| (manual stop, 10:40) | `run_small_chain.sh` was stopped during `cifar0_pulse_cold`: the plain-SGD CIFAR pulses and constant steps are used by no analysis or figure (the plain-SGD `cifar0_refL`, `cifar0_kp` and `cifar0_km` had finished). The partial output was removed and ALLDONE appended to `logs/kov/run_cifar0.log`, so that `run_bncheck.sh` starts |
| `run_anal_ann.sh` | analyses only, each started when its runs finish: B = 1000 anneal and the anneal model contest, norm-changed anneals, both reset pairs (from the match, PREREG note 10:36), anneal replications on seeds 1-3 |
| `run_an3b.sh` | replaces `run_an3.sh` (whose triggers never come after the queues moved to the pool dispatchers): the same exploratory detail analyses, each run when both runs of its pair have finished |
| `make_tables.sh` | rebuilds Supporting Information Tables S2-S7 from the analysis outputs (pending rows for missing inputs; `MISSING=label` for the final tables) |
| `run_linfinal.sh` | final linear-regime analyses after the last run: `k_lin.py` on every constant step (`k_lin_final.json`), `k_lin_report.py`, and the frozen registered models on the six linear-regime runs (`k_models_lin.json`) |
| `run_anal_rst2.sh` | decision rule of PREREG note 10:36 on both reset pairs (`k_reset_rule.py`), run as each reset analysis of `run_anal_ann.sh` lands |

## Analysis
| file | role |
|---|---|
| `k_pairs.py`, `k_table.py` | excursions of run pairs with the registered statistics; summary table of all pairs |
| `k_detail.py` | sequence bootstrap of the excursion; decomposition by target-token frequency and context position |
| `k_models.py`, `k_score.py` | blind contest of schedule-to-loss models (single-timescale law, multi-power law, functional scaling law, two-mode and spectral models) fitted on steps and pulses only, with the registered scoring |
| `k_models_ms.py` | sensitivity of the contest to the starting point of the parametric fits: 16 starts per model on the fit set only (Supporting Information) |
| `k_models_x.py` | exploratory diagnostics of the contest |
| `k_peaklaw.py` | registered peak-time law with its baselines |
| `k_lin.py`, `k_super.py`, `k_asym.py` | linear response of constant steps, superposition of step responses (exploratory, with the norm-corrected clock), asymmetry of approach |
| `k_lin_report.py` | the linear-regime runs as reported in the Supporting Information: per-unit responses of the +-20% steps and pulses on the nominal and intrinsic clocks, the two-pulse superposition test, and the whole-run RMSE of the frozen contest models on these runs |
| `k_text_numbers.py` | every number quoted in the manuscript text, read from the analysis outputs |
| `contest_grid.py`, `kmdip.py` | registered contest per case and model (pass flag, height and time ratios, NRMSE); early minima of the cool-heat excursions |
| `k_swap.py`, `k_norm.py`, `k_anneal.py` | transplant factorial, norm interventions, anneals from matched checkpoints |
| `k_anneal_pairs.py`, `k_annealN.py`, `k_anneal_table.py` | anneal endpoints against the free excursion and the shared-clock prediction (joint block bootstrap); norm-changed anneals; Supporting Information Table S4 |
| `k_peaklaw_status.py`, `k_peaklaw_table.py`, `k_interv_table.py` | peak-time law over every planned withheld run with an explicit status per run; Supporting Information Tables S5 and S3 |
| `k_anneal_rep_table.py` | anneal replications on seeds 1-3: replay provenance, final gaps with intervals, secondary contrasts and the registered outcome per pair (Supporting Information) |
| `k_anneal_fbin.py` | anneal gap split by target-token frequency at the start and end of each annealed pair (exploratory; Figure S2d) |
| `k_reset_rule.py` | ratios reset/unreset of the maximum and the area, with the declared classification (retained, reduced but retained, removed, reshaped) |
| `k_hump.py`, `k_an.py`, `k_plot.py` | two-mode fits of the excursion, pilot analysis, quick-look plots |

## Theory and exactly solvable model
| file | role |
|---|---|
| `ts_exact.py`, `ts_kovacs.py` | exact second-moment dynamics of online SGD on Gaussian linear regression |
| `ts_kappa.py`, `ts_noise.py`, `ts_peak.py`, `ts_sectors.py` | two curvature levels, noise strength, peak-time law, spectral interventions (manuscript Figure 3, drawn by `figs/fig2_theory.py`) |

## Figures
`figs/` holds one data script and one plotting script per figure (`fig1_data.py`/`fig1_phenom.py`, ...; the manuscript's Figure 2 is drawn by `fig3_mech.py` and its Figure 3 by `fig2_theory.py`, an order changed after the third review), the ToC image
(`toc_image.py`) and the shared style (`figstyle.py`). The data scripts read only the result files above. `render_all.sh`
renders every figure. The panel-alignment gate and the PDF collision audit used during rendering are helper scripts of a
figure-checking toolkit (`audit_panel_alignment.py`, `audit_figure_collisions.py`, `audit_pdf_text.py`,
`figure_safety.py`, `validate_figure.py`) that are not redistributed here; without them `figstyle.py` skips the gate
and the figures are rendered unchanged.
