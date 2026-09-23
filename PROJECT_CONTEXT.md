# PROJECT CONTEXT — Predictive Churn-Aware CF-ISAC

## Project
Predictive Churn-Aware Overlapping User- and Target-Centric Clustering for Mobile Cell-Free ISAC Networks.

Working directory:
~/predictive-churn-aware-cfisac

This is the ONLY main project. Do not create another project copy.
Do NOT modify ~/fpga-ai-accelerator.

## Research goal
Build a paper-quality mobile cell-free ISAC system where mobility prediction is used to make future-aware overlapping user/target clustering decisions, while explicitly accounting for reconfiguration/churn cost.

The intended research chain is:

mobility
→ prediction
→ predicted future network state
→ overlapping clustering
→ churn-aware decision
→ communication/sensing resource allocation
→ two-timescale optimization
→ robust/risk-aware evaluation.

## Current verified state

### Tests
Full suite:
63/63 tests passing.

Phase-2 tests:
18/18 passing.

### Phase 1 — COMPLETE
Implemented:
- constant-velocity mobility
- stochastic OU-inspired mobility
- mobility validation
- corrected per-object temporal train/validation/test split
- CV trajectory predictor
- LSTM trajectory predictor
- deterministic domain normalization
- LSTM early stopping
- best-validation checkpoint restore
- 2x2 predictor experiment
- prediction metrics and horizon breakdown

Current LSTM architecture is intentionally frozen:
input=2, hidden=64, 2 layers, dropout=0.1, direct multi-step output, horizon=5.

Do NOT redesign the LSTM unless a later controlled experiment explicitly requires it.

### Latest prediction findings
After normalization + convergence correction, LSTM became dramatically better.

Users:
CV mobility + CV:
MAE 1.78, ADE 3.52, FDE 7.47

CV mobility + LSTM:
MAE 1.73, ADE 2.79, FDE 3.36

Stochastic mobility + CV:
MAE 6.63, ADE 10.42, FDE 18.88

Stochastic mobility + LSTM:
MAE 6.64, ADE 10.40, FDE 17.24

Targets:
CV mobility + CV:
MAE 0.59, ADE 1.19, FDE 2.55

CV mobility + LSTM:
MAE 3.98, ADE 6.66, FDE 9.14

Stochastic mobility + CV:
MAE 8.64, ADE 13.63, FDE 23.95

Stochastic mobility + LSTM:
MAE 8.30, ADE 13.01, FDE 20.15

Important interpretation:
Do not claim that LSTM universally beats CV.
The evidence shows LSTM becomes competitive and has useful longer-horizon behavior under stochastic mobility, while CV remains strong in constant-velocity cases.

## Phase 2 — IMPLEMENTED AND TESTED

A predictor interface now exists:
src/prediction/predictor_interface.py

It supports CV and LSTM predictors.

Prediction information is connected into the simulation/optimization pipeline.

Horizon-aware objective plumbing was added.

Relevant files:
- src/prediction/predictor_interface.py
- src/prediction/predict.py
- src/prediction/normalization.py
- src/simulation/end_to_end.py
- src/optimization/baselines.py
- src/optimization/churn_aware.py
- src/clustering/churn.py
- src/clustering/overlapping.py
- tests/test_phase2.py

## Phase 3 — EXECUTED

Experiment:
experiments/run_phase3.py

Experiment matrix:
- CV mobility
- stochastic mobility
- CV predictor
- LSTM predictor
- STATIC
- REACTIVE
- PREDICTIVE
- PREDICTIVE+CHURN

Executed with:
seed=42
blocks=100

Results:
results/phase3/summary.csv
results/phase3/block_metrics.csv
results/phase3/plots/

Important Phase-3 findings:

CV mobility:
REACTIVE:
objective -48.4913
reconfigs 67

PREDICTIVE + CV:
objective -49.5888
reconfigs 72

PREDICTIVE + LSTM:
objective -49.5053
reconfigs 67

PREDICTIVE+CHURN + CV:
objective -54.0886
reconfigs 0

PREDICTIVE+CHURN + LSTM:
objective -54.0886
reconfigs 0

Stochastic mobility:
REACTIVE:
objective -46.8397
reconfigs 49

PREDICTIVE + CV:
objective -50.5824
reconfigs 89

PREDICTIVE + LSTM:
objective -48.5155
reconfigs 70

PREDICTIVE+CHURN + CV:
objective -49.8378
reconfigs 0

PREDICTIVE+CHURN + LSTM:
objective -49.8378
reconfigs 0

### CRITICAL ISSUE TO AUDIT

PREDICTIVE reacts differently to CV vs LSTM, but PREDICTIVE+CHURN produces EXACTLY identical results for CV and LSTM and always has zero reconfigurations.

This may mean:
- predictor information is not reaching the churn-aware decision;
- candidate generation is identical;
- churn penalty/normalization dominates;
- KEEP is always selected;
- or another legitimate formulation effect exists.

Do NOT assume it is a bug.
Do NOT change the churn penalty merely to make results look better.

First trace the actual decision path and establish the real cause.

## Existing results
results/
- prediction/
- mobility_validation/
- four_policy_comparison/
- objective_sensitivity/
- phase3/

## Current major missing work

1. Audit/fix/validate PREDICTIVE+CHURN behavior.
2. Complete robust Phase-3 evaluation after the churn audit.
3. Real communication beamforming.
4. Joint communication + sensing optimization.
5. Two-timescale architecture:
   slow = clustering/AP/sensing assignment
   fast = beamforming/power/sensing covariance
6. Prediction uncertainty.
7. Risk-aware/chance/CVaR optimization if justified.
8. Strong baseline comparisons.
9. Multi-seed evaluation.
10. Ablations.
11. Sensitivity.
12. Scalability.
13. Final publication-quality figures/tables.
14. Reproducibility and documentation.
15. Paper.

## Beamforming status
src/beamforming/__init__.py currently exists, but there is no substantive beamforming implementation yet.

Do not claim beamforming is implemented.

## Research principles
- Never fabricate results.
- Never report an experiment as complete unless it actually ran.
- Keep CV as a real baseline.
- Do not tune specifically to force LSTM to win.
- Do not leak future ground truth into the proposed method.
- Oracle future information may be used only as an explicit evaluation baseline.
- Preserve reproducibility and explicit seeds.
- Keep existing passing tests.
- Make minimum defensible changes.
- Prefer existing reference implementations in vendor/ when appropriate.
- Do not modify ~/fpga-ai-accelerator.

## Immediate next objective
Audit PREDICTIVE+CHURN before implementing beamforming.

Inspect:
- src/optimization/churn_aware.py
- src/optimization/baselines.py
- src/simulation/end_to_end.py
- src/clustering/churn.py
- src/clustering/overlapping.py
- src/prediction/predictor_interface.py
- experiments/run_phase3.py

Trace a real stochastic+CV decision and a real stochastic+LSTM decision.

Record:
- predicted positions
- candidate configurations
- current objective
- predicted objective
- predicted gain
- normalized churn cost
- final KEEP/RECONFIGURE decision.

## Phase 4A — IMPLEMENTED (minimum physical-layer extension)

Audit note: the archive this phase started from
(predictive-churn-aware-cfisac-phase3-validated-20260919-070625) contained
NO partial Phase 4A implementation. `src/beamforming/__init__.py` was
empty (0 bytes) and no `phase4-7_implementation_map.md` file existed.
Everything below was implemented from scratch in this pass, using
vendor/cordis/cordis/algorithms/beamforming.py as the reference
convention for the MRT/RZF signal model.

New files (additive only — nothing pre-existing was modified):
- src/channels/physical.py — per-antenna complex channel (ULA steering +
  Rician small-scale fading), reuses generate_comm_gains from
  src/channels/communication.py for the large-scale magnitude.
- src/beamforming/precoders.py — mrt_weights, rzf_weights,
  normalize_per_ap_power. src/beamforming/__init__.py now re-exports
  these (was previously an empty placeholder file).
- src/metrics/physical_layer.py — physical_sinr, physical_rate. Rate
  reuses src.metrics.system.communication_rate directly.
- tests/test_phase4a_physical_layer.py — 30 focused unit tests.
- experiments/run_phase4a_beamforming_validation.py — deterministic
  scalar-vs-MRT-vs-RZF validation experiment.
- configs/default.yaml — new additive `physical_layer:` section
  (num_antennas_per_ap, antenna_spacing, rician_k_factor,
  small_scale_fading, rzf_epsilon_rel, tx_power_per_ap_w, noise_power_w,
  seed). No existing keys were changed.

The existing scalar channel/SINR/rate path
(src/channels/communication.py, src/metrics/system.py,
src/optimization/churn_aware.py) is completely unmodified and continues
to be used as-is by the frozen prediction/clustering/churn pipeline.

Test results actually executed in the Anthropic sandbox (no network,
so torch is NOT installed here and could not be verified in this
environment):
- 64/64 pre-existing non-torch tests: PASS
- 30/30 new Phase 4A tests: PASS
- tests/test_prediction.py (21 tests, requires torch): NOT RUN here —
  torch is unavailable in this sandbox. These must be re-verified on
  Ubuntu where torch is installed, alongside the other 94.

Validation experiment: actually executed twice (default config: 20 APs /
5 users / 4 antennas/AP; and a denser overlap config: 6 APs / 8 users /
4 antennas/AP). Results saved to results/phase4a/ and
results/phase4a_stressed/. Under denser AP-user overlap, RZF measurably
outperforms MRT (higher avg SINR/rate, same per-AP power); under the
sparser default topology the two are close because most APs serve at
most 1-2 users, giving RZF little multi-user interference to cancel.
Both are strictly better than the scalar baseline metric in this
experiment, which is expected since the scalar model has no array/
beamforming gain at all — this is not evidence that the scalar path is
"wrong" for its own purpose (it is a different, coarser model kept for
backward compatibility), only that it does not model directional gain.

## Phase 4A — explicitly NOT done (deferred to Phase 4B+)
- Joint communication + sensing optimization
- Two-timescale (slow clustering / fast beamforming-power) optimization
- Prediction uncertainty
- Risk-aware / CVaR objectives
- Multi-seed / large-scale evaluation of the physical layer
- Any change to the LSTM or the prediction/clustering/churn architecture

First diagnose.
Then propose the minimum scientifically defensible correction.
Only then modify code.

## Phase 4B — IMPLEMENTED (SINR-constrained communication beamforming)

Audit note: at the start of this pass, the only beamforming code present
was the Phase 4A MRT/RZF (`src/beamforming/precoders.py`) and physical-layer
SINR/rate (`src/metrics/physical_layer.py`) modules. There was NO
partial Phase 4B implementation (no SOCP/bisection/constrained code, no
`src/beamforming/constrained.py`, no Phase-4B tests). Everything below
was implemented from scratch in this pass.

References used:
- `vendor/matlab_isac/optimization/opt_comm_SOCP_vec.m` — the target
  problem (feasibility SOCP: per-user SINR constraints + per-AP power
  constraints, no association mask, global/joint precoding).
- `vendor/matlab_isac/optimization/bisection_SINR.m` — the bisection
  pattern used for `bisection_sinr_beamforming`.
- `vendor/cordis/cordis/algorithms/beamforming.py` — reused (read-only)
  as the confirmed signal-model convention already adopted by Phase 4A's
  MRT/RZF; NOT modified and not directly called by the new Phase 4B code.
- `vendor/cordis/cordis/algorithms/centralized.py` — inspected for
  convention/architecture consistency; not directly reused (no function
  from it matched the SINR-constrained-beamforming problem).

**Important solver honesty note:** this sandbox has no CVX/cvxpy/MOSEK
or any general SOCP solver (`requirements.txt` lists only
numpy/scipy/matplotlib/pandas/scikit-learn/pyyaml/torch/networkx, and
there is no network access to install one). The MATLAB reference uses
CVX. Rather than fake an SOCP solve, `src/beamforming/constrained.py`
implements the classical **uplink-downlink duality fixed-point
algorithm** for QoS/SINR-constrained beamforming with per-AP power
constraints (Schubert & Boche 2004 for the sum-power fixed point; Yu &
Lan 2007 for the per-AP-group generalization via an outer dual-weight
update). Strong duality holds for this problem class, so when the
fixed point converges this recovers the same optimum CVX's SOCP would,
not a heuristic approximation. Full derivation, the exact iteration
equations, and the documented fallback/infeasibility-detection behavior
are in that file's module docstring — see it for the authoritative
solver documentation; do not duplicate/re-derive it here.

New files (additive only — nothing pre-existing was modified):
- `src/beamforming/constrained.py` — `constrained_sinr_beamforming`
  (single target-SINR feasibility + design) and
  `bisection_sinr_beamforming` (bisection over a uniform SINR floor,
  mirroring `bisection_SINR.m`). Global/joint design across all AP
  antennas (matches `opt_comm_SOCP_vec.m`'s convention exactly — no
  per-user association mask), as opposed to Phase 4A's MRT/RZF which
  remain per-AP-local/association-restricted and are completely
  unmodified.
- `src/beamforming/__init__.py` — additive re-export of the two new
  functions; `mrt_weights` / `rzf_weights` / `normalize_per_ap_power`
  exports are untouched.
- `tests/test_phase4b_constrained_beamforming.py` — 26 focused tests
  covering: feasible SINR cases, two distinct infeasibility mechanisms
  (interference-limited — target unachievable at ANY power — and
  power-limited — spatially achievable but budget too small),
  single-user, multi-user/multi-AP, achieved-SINR-vs-target, per-AP
  power-constraint enforcement (including heterogeneous per-AP
  budgets), convergence/iteration reporting, bisection monotonic
  bracket shrinkage, determinism (bitwise-identical repeated calls,
  feasible and infeasible), and side-by-side comparison against MRT/RZF
  including a regression guard that MRT/RZF outputs are byte-identical
  before/after exercising the new module.
- `experiments/run_phase4b_beamforming_validation.py` — deterministic
  4-way comparison (MRT / RZF / fixed-target SINR-constrained /
  bisection max-min SINR floor) on the same topology, association, and
  per-AP power budget.

The existing scalar channel/SINR/rate path
(`src/channels/communication.py`, `src/metrics/system.py`,
`src/optimization/churn_aware.py`) and the Phase 4A MRT/RZF path
(`src/beamforming/precoders.py`, `src/metrics/physical_layer.py`) are
completely unmodified. Prediction, clustering, and churn logic were not
touched.

Test results actually executed in the Anthropic sandbox (no network, so
torch is NOT installed here, same limitation as Phase 4A):
- 95/95 pre-existing non-torch tests: PASS (unchanged from before this
  pass; includes the 30 Phase 4A tests)
- 26/26 new Phase 4B tests: PASS
- `tests/test_prediction.py` (requires torch): NOT RUN here — must be
  re-verified on Ubuntu alongside the other 121, exactly as Phase 4A
  flagged. Combined with the 21 torch tests already counted in the
  115/115 Ubuntu baseline, the full suite is expected to be 141/141 on
  Ubuntu; this was NOT executed on Ubuntu in this pass and that number
  is a prediction, not a verified result.

Validation experiments: actually executed twice.
- Default topology (20 APs / 5 users / 4 antennas/AP, seed 42):
  MRT and RZF both reach ~9 dB / ~8.6 dB min-SINR at 1.0 W/AP; the
  SINR-constrained design hits a 0 dB (linear 1.0) target exactly using
  only ~0.00013 W/AP (the topology has plenty of slack at that modest
  target); bisection finds a ~68.3 dB max-min-fair SINR floor at the
  full 1.0 W/AP budget. Saved to `results/phase4b/`.
- Denser/stressed topology (6 APs / 8 users / 4 antennas/AP, target
  5.0 linear, seed 42): here MRT and RZF's worst-served user is BELOW
  0 dB (-4.7 dB and -6.7 dB respectively — interference-limited), while
  the SINR-constrained design still guarantees exactly 5.0 linear
  (~7.0 dB) minimum SINR for every user at a fraction of the power
  budget, and bisection finds a ~47.0 dB max-min floor at full power.
  This is the qualitative payoff of Phase 4B: MRT/RZF have no SINR
  guarantee for the worst user under contention; the constrained design
  does, by construction. Saved to `results/phase4b_stressed/`.

## Phase 4B — explicitly NOT done (deferred to later phases)
- Per-user (non-uniform) SINR targets were implemented in
  `constrained_sinr_beamforming` (it accepts a per-user gamma vector)
  but NOT exercised in the bisection driver, which bisects a single
  uniform floor only (matching `bisection_SINR.m`'s scalar-gamma
  pattern) — a per-user-weighted bisection is future work if needed.
- No association-mask / per-AP-local variant of the constrained solver
  (it is intentionally global/joint, matching the vendor SOCP
  reference exactly) — a locally-constrained SOCP variant is future
  work if the research question specifically calls for it.
- Joint communication + sensing optimization.
- Two-timescale (slow clustering / fast beamforming-power) optimization.
- Any change to the LSTM, prediction, clustering, or churn architecture.

## Current completion estimate
Core implementation/prototype: about 90–92%.
Full paper-grade research system: about 70–72%.

The remaining major research work is mostly joint comm+sensing
optimization, the two-timescale architecture, and rigorous multi-seed
evaluation, not the basic prediction infrastructure or the physical-layer
beamforming building blocks (MRT/RZF/SINR-constrained), which are now
all implemented and tested.

## Phase 5A — Physical Communication/Sensing Bridge

**Status: COMPLETE**

Phase 5A adds an additive physical communication-to-sensing bridge without modifying
the existing scalar Fisher-information sensing model, Phase 4A/4B beamforming, or
churn-aware optimization.

### Implementation

New module:
- `src/metrics/physical_sensing.py`

The bridge provides:
- communication power per AP from physical beamforming weights
- isotropic sensing covariance from explicit per-AP sensing power
- sensing SNR from `p_sens * G_sens / noise`
- SNR-to-measurement-variance mapping
- physical-power-aware aggregation of the existing bistatic Fisher information
- posterior covariance, information gain, and tracking error
- combined AP transmit power:
  `P_tx,m = P_comm,m + P_sens,m`
- per-AP power-constraint checking

The sensing covariance approximation is isotropic:
`S_m = (p_sens,m / N) I_N`

Phase 5A does **not** introduce:
- joint communication/sensing power optimization
- sensing interference into communication SINR
- changes to the existing scalar sensing channel/FIM implementation
- changes to Phase 4A or Phase 4B beamforming

### Validation

New tests:
- `tests/test_phase5a_physical_sensing_bridge.py`

Phase 5A tests:
- **30/30 PASS**

Full project regression:
- **171/171 PASS**

Dedicated validation experiment:
- `experiments/run_phase5a_joint_physical_sensing_validation.py`

Validation output:
- `results/phase5a/phase5a_joint_validation.json`

Deterministic validation configuration:
- 20 APs
- 5 users
- 2 sensing targets
- 4 antennas/AP
- seed 42
- 1.0 W/AP maximum transmit power
- fixed MRT communication beamforming at 0.5 W/AP on active communication APs

Representative sensing-power sweep:

| Condition | Sensing power/AP | Mean information gain | Mean tracking error | Max combined AP power | Feasible |
|---|---:|---:|---:|---:|---|
| zero_sensing | 0.00 W | 0.000000 | 2.000000 | 0.500000 W | Yes |
| low_sensing | 0.10 W | 0.236266 | 1.792923 | 0.600000 W | Yes |
| medium_sensing | 0.25 W | 0.505082 | 1.607443 | 0.750000 W | Yes |
| budget_edge | 0.50 W | 0.840359 | 1.418813 | 1.000000 W | Yes |
| over_budget | 0.60 W | 0.951360 | 1.363163 | 1.100000 W | No |

The validation confirms:
1. information gain is non-decreasing with sensing power;
2. tracking error is non-increasing with sensing power;
3. zero sensing power produces zero sensing information gain;
4. communication and sensing power are summed exactly once;
5. the per-AP power constraint is detected correctly;
6. existing scalar sensing and Phase 4A/4B behavior remain unchanged.
