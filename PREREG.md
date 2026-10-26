# E10 analysis plan, fixed 2026-10-20 05:45 CST (before any withheld protocol below had been run)

## Excursion statistics (all Kovacs runs)
- Excursion ΔL(t) = L_run(t) − L_ref(t) on held-out set A (matched set) and on the untouched set B; t counted from the
  release step (first step back at η_ref). Reference interpolated linearly between its evaluations.
- Peak: maximum of the 3-evaluation running mean over 10 ≤ t ≤ 150 steps. Area: trapezoidal integral over 0 ≤ t ≤ 300.
  Late offset: mean over 1000 ≤ t ≤ 2000.
- Uncertainty: (i) independent base-training seeds 1–5 (each with its own reference), mean ± s.e.m. across seeds;
  (ii) within a run pair, block bootstrap over validation sequences (blocks of 8 consecutive sequences, 1000 resamples).
  Temporal jitter is not used as an uncertainty.

## Blind prediction contest
- Fit set (only): base (steps ≥ 1000), reference, single pulses (3η_ref, η_ref/8 for 300 steps), constant steps
  η ∈ {η_ref/8, η_ref/4, η_ref/2, 2η_ref, 3η_ref}. All model choices are made on this set, including the (p, q) grid
  choice of the spectral model and the river form.
- Models: single-timescale schedule law (Tissue et al.), multi-power law (Luo et al.), functional scaling law (Li et al.,
  power-law kernel in intrinsic time), river + two modes, river + 26-mode log-grid spectrum (NNLS).
- Withheld test protocols: kp_ta100, kp_ta1000 (heating 100 / 1000 steps), kp_h2 (heat 2η_ref), kp_l2 (cool η_ref/4), km2
  (reverse), the replicate and seed Kovacs runs (models transferred unchanged), pulse_two (two-pulse superposition),
  linear-regime steps/pulses at 1.2η_ref and 0.8η_ref, the ageing pair (protocol started at step 8000), annealed
  continuations from matched checkpoints. kp itself is exploratory (it was seen while building the models).
- Metrics: sign of the excursion; peak height ratio pred/obs; peak time ratio; NRMSE = RMSE/range of the observed
  smoothed post-release excursion; RMSE of the absolute loss over the whole run. Success: correct sign, peak height and time
  within a factor 1.5, NRMSE < 0.3.
- No amplitude or time-axis rescaling of any prediction.

## Addendum, 2026-10-20 05:53 CST (before any withheld Kovacs run had started): peak-time law
(The heading first read "05:58" by mistake; 05:53:20 is the file's save time. Corrected at 05:58, see the end.)
- For relaxation linear in intrinsic time with a power-law density of rates, excess weight a(r) ∝ r^(ν−1), the excursion
  is ΔE(σ) = (g_h−g_l)A(s_c+σ) − (g_h−g_0)A(s_c+s_a+σ) − (g_0−g_l)A(σ), A(x) ∝ x^(−ν) (−ln x for ν = 0), so the lag of the
  maximum σ* depends only on the protocol levels, s_a and s_c (intrinsic heating and cooling durations) and ν.
- Prediction rule, fixed now: g ∝ η (p = 1), intrinsic time = steps × η/η_ref (q = 1), ν = 0 (log-uniform spectrum, the
  value implied by the exploratory kp run: observed t* = 41 ± 1 steps vs 44 predicted). s_c is the measured cooling time of
  each run × η_l/η_ref. Compared with the two-mode-fit t* (and the smoothed maximum) of kp_ta100, kp_ta1000, kp_h2, kp_l2,
  the seed runs, the replicates, code/PubMed/124M Kovacs runs. Success: |log(pred/obs)| < log 1.5.

## Correction, 2026-10-20 05:58 CST
- code_kp (finished 05:41) and the MovieLens run mfn_kp had been run, and their peak times inspected, before the addendum
  was written. They are exploratory checks of the peak-time rule, not withheld tests, and are reported as such.
- Withheld tests of the rule (none had started at 05:53): kp_ta100, kp_ta1000, kp_h2, kp_l2, the seed and replicate
  Kovacs runs, the PubMed and 124M Kovacs runs. The rule itself is unchanged.

## Addendum, 2026-10-20 06:51 CST (before any intervention branch below had been run): norm interventions at the match
- Hypothesis (weight-norm thermostat): heating inflates the weight norms; a longer weight turns less per update (smaller
  effective learning rate), so the norm acts as a slow variable of opposite sign to the hidden modes.
- Branches, all from the matched checkpoints of the transplant pair (kpS at the match, refS at the same step), 1500 steps at
  eta_ref with identical data, angular updates logged (run_norm.sh):
  nm_kp_sham / nm_ref_sham: unchanged; nm_kp_proj: in every block the qkv and fc matrices rescaled to the reference's norms
  with reciprocal LayerNorm gains (exact reparameterisation); nm_ref_proj: the reference's qkv and fc to the Kovacs norms;
  nm_kp_resid: the Kovacs residual stream rescaled (embeddings, attention and MLP outputs times c, final LayerNorm gain over
  c) to the reference's residual-writer norm; nm_ref_resid: the converse; nm_kp_both: nm_kp_resid and nm_kp_proj together.
- Expected direction under the hypothesis (relative to nm_ref_sham): deflating the Kovacs network (nm_kp_proj, nm_kp_resid,
  nm_kp_both) gives a larger peak (10-150) and area (0-300) than nm_kp_sham and a late offset (1000-1500) that is less
  negative; inflating the reference (nm_ref_proj, nm_ref_resid) gives a negative area (0-300) and a negative late offset.
  The hypothesis is contradicted if deflation lowers the Kovacs area or inflation gives a positive area.
- Checks: held-out loss before the first update within 1e-3 nats of the sham's (residual rescale: LayerNorm epsilon and bf16)
  and per-sequence losses identical up to bf16 rounding (norm/gain reallocation); per-step angular updates per matrix group.
- Not a test of norm independently of gains: the reallocation moves norm into the (undecayed) LayerNorm gains.

## Note, 2026-10-20 06:55 CST (scoring code brought in line with this plan; no withheld result had been seen)
- k_score.py had used the signed extremum where this plan registers the maximum; the registered maximum is now primary and
  the signed extremum is reported separately as a secondary statistic (it differs only for mainly negative excursions).
- k_peaklaw.py had taken the smoothed maximum over 1-600 steps; the registered window 10-150 is now used (the wide window is
  kept as a secondary column). Success of the peak-time rule is reported against both the two-mode-fit t* and the smoothed
  maximum, never against the better of the two.
- Added timing baselines calibrated on the exploratory pilot (kp): constant lag 41.3 steps; constant fraction 0.469 of the
  intrinsic cooling time.

## Note, 2026-10-20 08:34 CST (scoring code brought in line with this plan; no withheld contest, peak-law, transplant or anneal result had been computed)
- k_models.py: the registered "RMSE of the absolute loss over the whole run" is now computed on absolute losses
  (`rmse_abs`); the former `rmse_all`, which compared loss differences to the reference, is kept as `rmse_diff`.
- k_table.py, k_norm.py, k_swap.py: a statistic whose window the pair does not cover is reported as missing, never
  replaced by another window (k_table.py had used the last 500 steps) or by zero (k_norm.py had used `or 0`). k_norm.py
  issues no verdict on an incomplete window. Coverage of a window counts as complete if the last evaluation lies within one
  sparse evaluation interval (60 steps) of its end; the window actually used is recorded (only the pilot is affected:
  1000-1950).
- k_swap.py: the four transplant branches are compared only on steps all four cover. Its late window is 1000-1500 (the
  transplant continuations are 1500 steps long), as for the norm branches.
- k_anneal.py: the last-evaluation difference and the mean over the last three evaluations each get a block bootstrap of
  their own evaluations.
- Seen before this note: the first 300 steps of nm_kp_proj and nm_ref_proj (registered directions so far), the sham pair,
  and the kp_ta100 excursion statistics. The registered statistics and success criteria are unchanged.

## Note, 2026-10-20 08:47 CST (interim scoring; nothing in the plan changed)
- k_peaklaw.py (unchanged since 06:55) was run on the three withheld runs that had finished: kp_ta100, kp_ta1000, kp_sd1.
  Law: 32.7 / 22.6 / 44.4 steps against t_max 35 / 25 / 55 and t* 33.7 / 18.2 / 40.3 (all within the factor 1.5); the
  constant-lag baseline fails kp_ta1000, the constant-fraction baseline passes all three. Further runs are scored with
  the same code; the final count includes these three.
- k_norm.py: reading a branch whose evaluation file exists but is still empty no longer raises (treated as not started).
  Statistics and verdict rules are unchanged. nm_kp_proj is complete (peak, area and late offset all above the sham);
  nm_ref_proj has its area (negative) but not yet its late window.
- Seen: k_table statistics of kp_ta1000 and kp_sd1 (set A and B). Manuscript text drafted after this point quotes them.

## Note, 2026-10-20 09:02 CST (exploratory addition after the third review; not part of the registered plan)
- run_bncheck.sh / kov_small_bn.py: the matched CIFAR pair of the filter-norm exchange is continued once more as in
  cifar_cn_own / cifar_cn_ref, and every evaluation is also scored with the batch-norm running statistics re-estimated over
  20 fixed minibatches (training untouched). Reported as an exploratory sensitivity check of the running-statistics lag.
- 09:06: k_table.py lists the repeat kpS (matching from kpS, release from its sham continuations nm_kp_sham/nm_ref_sham,
  1500 steps, so its 1000-2000 late window is reported as missing). No statistic or criterion changed.

## Note, 2026-10-20, written shortly after 09:13 CST, when kp_h2 finished (interim scoring; nothing in the plan changed)
- k_norm.py on the completed branches nm_kp_resid and nm_ref_resid (nm_kp_both still running): every registered direction
  holds for all four completed branches (deflated Kovacs: larger maximum and area, less negative late offset; inflated
  reference: negative area and late offset). Endpoint resolution from two identical continuations (k_repro_ep.py, new,
  exploratory): area within 0.03 nats*steps, late offset within 5e-4 nats, smoothed maximum within 7e-4 nats.
- k_peaklaw.py on kp_h2 (withheld, finished 09:13), with the same code: law 19.1 steps vs t* 25.1 (pass) and t_max 30
  (fail, ratio 1.57); constant lag 41.3 fails t*, passes t_max; constant fraction 9.6 fails both. Interim counts over the
  four withheld runs scored so far (kp_ta100, kp_ta1000, kp_sd1, kp_h2), against t* and t_max: law 4/4 and 3/4; constant
  lag 2/4 and 3/4; constant fraction 3/4 and 3/4 (k_peaklaw_interim.json). Seen: k_table statistics of kp_h2.

## Note, 2026-10-20, written after 09:24 CST, when sw_ref_kpopt finished (transplant scored; nothing in the plan changed)
- k_swap.py on the complete factorial (sw_ref_kpopt finished 09:24), set A: Y_KK area 14.48 (max +0.065 at 55), W_R = Y_KR
  14.45 (+0.064 at 70), O_R = Y_RK 1.70 (+0.020 at 20), I -1.67; late offsets within 5e-4. Set B: 13.81, 13.70, 1.71, -1.61.
  Reported with the precommitted wording of the third review ("The response depends predominantly on the weight checkpoint
  under the tested continuations"); the reset control (kp_reset / refL_reset) is pending and will be reported as it falls.
- k_interv_table.py: the interaction row no longer shows a maximum (maxima do not decompose additively).
- Added after 09:38 CST, when nm_kp_both finished (k_norm.py): maximum +0.0796 (d +0.0146), area 18.72 (d +4.28), late +0.0140 (d +0.0147). All
  five registered norm branches point in the registered direction; the two single reductions add almost exactly
  (d area 1.87 + 2.36 = 4.23 against 4.28 together).
- Added after 09:37 CST, when pubmed_kp finished (withheld test of the rule): matched after 546 cooling steps, maximum +0.059 at 45
  steps (set B +0.057 at 45), area 11.6, late -0.004. Law 33.2 vs t* 38.2 and t_max 45 (both pass); both baselines pass.
  Interim counts over five withheld runs (t* / t_max): law 5/5, 4/5; constant lag 3/5, 4/5; constant fraction 4/5, 4/5.

## Addendum, 2026-10-20 09:44 CST (exploratory; written before the runs below were started, after the B = 0 anneal pair)
- Seen: the B = 0 anneal pair (an_kp_B0, an_ref_B0). Final gap +0.073 nats (set A; B +0.072), larger than the maximum of the
  free excursion (+0.065); during the anneal the gap stays near 0.065 from 50 to 300 steps and grows to 0.073 as the learning
  rate approaches zero. A linear relaxation model on a clock common to both networks predicts that any common schedule only
  reparameterises time, so the anneal gap would equal the free excursion at intrinsic time 250 steps (+0.037, sham pair).
- Question: does parameter scale control the anneal gap? Two further anneals (B = 0, identical schedule and data), from the
  matched checkpoints after the function-preserving norm changes of the norm test: an_kpN_B0 (Kovacs, both reductions, as
  nm_kp_both) and an_refN_B0 (reference, both increases to the Kovacs norms). run_annN.sh starts them when run_swap.sh ends.
- Contrasts (final difference, last evaluation and mean of the last three, sets A and B, block bootstrap as k_anneal.py):
  G_N = an_kpN_B0 - an_ref_B0 against G = an_kp_B0 - an_ref_B0; H_N = an_refN_B0 - an_ref_B0.
- Primary hypothesis (scale-dependent clock: a longer network turns less per update and lags further behind the falling
  learning rate): G_N < G and H_N > 0. The opposite pattern (G_N > G, H_N < 0), by analogy with the free continuations of
  the norm test, would contradict it. Exploratory; reported whichever way it falls.

## Note, added after 10:02 CST, when kp_l2 finished (withheld test of the rule)
- k_peaklaw.py, unchanged: kp_l2 (cooling at η_ref/4, matched after 362 cooling steps, s_c = 90.5) law 38.8 steps vs t* 34.5
  and t_max 45 (both pass); constant lag and constant fraction (42.4) both pass. Interim counts over six withheld runs
  (t* / t_max): law 6/6, 5/6; constant lag 4/6, 5/6; constant fraction 5/6, 5/6 (`k_peaklaw_interim2.json`).
- Queue (no result involved): from 10:02 the remaining jobs run in run_pool3.sh (same commands, new order; see README).

## Addendum, 2026-10-20 10:20 CST (before any run below had started): anneal replication on independent networks; optimizer reset from the transplant pair
- Why: the B = 0 and B = 200 anneals (Section 2.6) come from one matched pair (the transplant pair, kpS/refS). Codex round 4
  (`01_ideas/codex_reviews/kov_eval4_out.md`) asked for B = 0 anneals of independent matched pairs, and for reset
  continuations from the same saved checkpoints as the transplant. The anneal design was chosen after the transplant pair's
  anneal had been seen; for the independent pairs below it is fixed here, before any of them has run.
- Anneal replication, seeds 1, 2, 3 (each starts once that seed's reference refL_sd<s> has finished):
  - kpM_sd<s>: replay of the heat->cool preparation of kp_sd<s> from base_sd<s>_s4000 (same seeds and data, 3η_ref for 300
    steps, then η_ref/8 until the match, checked every step against refL_sd<s>_eval.tsv), saved with its AdamW state at the
    match (kpM_sd<s>_end1.pt).
  - refM_sd<s>: the reference of that seed continued from base_sd<s>_s4000 to the same step and saved (refM_sd<s>_s<TM>.pt).
  - anM_kp_sd<s> and anM_ref_sd<s>: both annealed with the schedule of run_swap.sh (linear decay from η_ref to 0 over 500
    steps, identical minibatches).
  - Statistic: final gap G_s = L(anM_kp_sd<s>) - L(anM_ref_sd<s>) at the last evaluation and as the mean of the last three, on
    sets A and B, with the block bootstrap of k_anneal.py (blocks of 8 sequences, 1000 resamples).
  - Hypothesis: G_s > 0 (95% interval above zero) for every pair. Secondary, descriptive: G_s against the shared-clock
    prediction, the free excursion of the seed's registered pair (kp_sd<s> - refL_sd<s>, the original rather than the replayed
    run) at 250 steps after the match, and against that pair's largest evaluated free excursion.
- Reset from the transplant pair (rs_kp, rs_ref): kpS_end1.pt and refS_s4987.pt continued for 1500 steps at η_ref with fresh
  AdamW moments (--reset-opt 1) on identical minibatches. Contrast rs_kp - rs_ref with the unreset continuations of the same
  checkpoints (sw_kp_own - sw_ref_own; nm_kp_sham - nm_ref_sham as a repeat), on the registered endpoints: smoothed maximum
  and its lag (10-150 steps), area over 0-300 steps, late offset over the last 500 steps. Peak and area are reported
  separately; the wording for each outcome (retained / reduced but retained / removed / reshaped) is Codex round 4,
  section 2, verbatim. The queued reset pair from the base checkpoint (kp_reset / refL_reset) is still run and reported.
- Anneal endpoints against the free excursion (both for the transplant pair and the replications): the difference between
  the final gap and the largest evaluated free excursion is bootstrapped directly, resampling the same validation blocks for
  the annealed and the free runs and recomputing the free maximum in each resample (k_anneal_pairs.py).
- Queue: the quench probes (PRkp, PRkm) and the zero-weight-decay and 124M pulses are dropped; no registered analysis uses
  them, and their GPU time goes to the runs above (run_pool4.sh).

## Note, 2026-10-20 10:36 CST (decision margins; none of an_kpN_B0, an_refN_B0, rs_kp, rs_ref, kpM_sd*, refM_sd*, anM_* had started)
- Codex round 4 asks for declared margins before the wording of its section 2 is applied. They are fixed here.
- Norm-changed anneals (exploratory, 09:44): C_K = G_N - G = an_kpN_B0 - an_kp_B0 and C_R = H_N = an_refN_B0 - an_ref_B0,
  last evaluation and mean of the last three, sets A and B, block bootstrap (k_annealN.py). A direction is resolved when the
  95% interval excludes zero. Equivalence margin +-0.005 nats (7% of G, 14% of its excess over the shared-clock
  prediction): "no practically resolved effect" requires both intervals (C_K and C_R) inside (-0.005, +0.005). An interval
  that crosses zero and leaves the margin is "unresolved". The wording follows the outcome on set A at the last evaluation;
  set B and the last-three mean are reported with it, and a disagreement between them is stated.
- Optimizer reset from the transplant pair: ratios rho = reset / unreset of the smoothed maximum (10-150 steps) and of the
  area (0-300 steps) of rs_kp - rs_ref, against sw_kp_own - sw_ref_own (nm_kp_sham - nm_ref_sham as a repeat), both from
  the match (k_pairs.py with @4987). Retained: rho >= 0.8 for both; reduced but retained: 0.2 <= rho < 0.8 for both;
  removed: rho < 0.2 for both; otherwise reshaped. The lag of the maximum and the late offset are reported, not classified.
  The same rule applies to kp_reset - refL_reset against kp - refL (both from the match of the Kovacs run).

## Note, 2026-10-20 11:25 CST (queue only; no result involved)
- The remaining chains run in run_pool5.sh (tmux kov_pool5); the dispatcher of run_pool4.sh was stopped with SIGSTOP after
  it had started AN1. New order: AN2 AN3 RST BIG R1 R2 AGE K2b S1 WD0 LINa LINb. The 124M chain sets the end of the queue,
  so it moves ahead of zero weight decay and the linear-regime runs, which now come last. Commands, analyses and success
  criteria are unchanged. Every registered run is still queued; any run that does not finish is reported as not run.

## Note, 2026-10-20 11:49 CST (counting convention for the anneal replications; no kpM_sd*, refM_sd* or anM_* run had finished and no replication had been analysed)
- From Codex round 5 (`01_ideas/codex_reviews/kov_eval5_out.md`, section 1), fixed before any replication is analysed.
  Replication counts use the final evaluation on set A. A pair meets the registered positive-gap criterion when its 95%
  block-bootstrap interval lies wholly above zero. Set B and the last-three-evaluation mean are mandatory robustness
  analyses; any disagreement is reported explicitly.
- Outcomes per pair: met (interval above zero); unresolved (interval crosses zero); opposite direction (interval below
  zero); unscored (run missing or incomplete, not a negative result). Counts are reported as n+/3 prospective replications
  and, separately, (n+ + 1)/4 including the exploratory transplant pair.
- Secondary contrasts, counted separately with the same interval rule: D_clock = G_s - F_s(250.5) (shared-clock prediction,
  k ∝ η) and D_max = G_s - the largest evaluated free excursion over 0-1500 steps of the seed's original registered pair.
  The free benchmarks come from the original preparations and the anneals from replayed ones; replay matching times and
  initial gaps are reported.
- The wording for each outcome is Codex round 5, section 1, verbatim (tables "Outcome (a)/(b)/(c)" and "Secondary contrasts").

## Note, 2026-10-20 12:29 CST (queue only; no result involved)
- At 12:22 a fault of the CUDA MPS server (Xid 31) killed every GPU job: km_sd2, km_sd3, kp_sd4, kp_sd5 and refM_sd1 (AN1)
  failed with "illegal memory access". kpM_sd1 had already finished (match at 5007, registered kp_sd1 at 5008).
- run_pool6.sh (tmux kov_pool6) re-queues the failed chains, which resume at their first missing output; a failed run is
  repeated from its initial checkpoint with the same command. Order: AN1 BIG S4 S5 AN3 RST S2 S3 R1 R2 AGE K2b S1 WD0 LINa
  LINb. The dispatcher of run_pool5.sh is stopped with SIGSTOP after it had started AN2, which keeps running. Runs, analyses
  and success criteria are unchanged; failed logs are kept as <name>_fail<try>.log.

## Note, 2026-10-20 12:56 CST (queue only; no result involved)
- run_pool7.sh (tmux kov_pool7) takes over from the dispatcher of run_pool6.sh (SIGSTOP at 12:55 after it had started AN1,
  BIG and S4). New order of the chains not yet started: S5 AN3 RST R1 R2 AGE WD0 K2b S2 S3 S1 LINa LINb. The commands,
  analyses and success criteria are unchanged.

## Note, 2026-10-20 13:06 CST (wording only; written before RST had started and while AN3's replay was running, so before
## any result of either)
- Codex round 6 (`01_ideas/codex_reviews/kov_eval6_out.md`) reviewed the interim results. Its wording replaces earlier
  wording where the two differ.
- Reset from the saved transplant pair (rs_kp, rs_ref): the primary comparator is the own-state continuation of the same
  checkpoints (sw_kp_own - sw_ref_own), with the sham continuation (nm_kp_sham - nm_ref_sham) as the repeat check. The
  rule of note 10:36 is unchanged. Wording for each outcome is round 6, section 1, table "Saved-pair outcome", verbatim.
  When the sign flag is set, the affected endpoint is named explicitly, the numerical category stays in the table, and
  the flag has interpretive priority. The Discussion opens with round 6's sentence for the base-checkpoint repeat, then
  adds the saved-pair sentence. If the two classifications differ, round 6's sentence on the two preparations is added.
- Independent anneal pairs: the outcome sentences are round 6, section 2 (Abstract, end of Introduction, §2.6, Discussion
  for outcomes (a) and (b), and the sentence for the secondary counts). They revise round 5, section 1: present tense in
  the Abstract, and in outcome (b) separate Discussion sentences for an unresolved and for an opposite third pair. A
  missing seed-3 result stays unscored; it does not produce outcome (b).
