"""
src/optimization/cvar.py

Phase 7 -- Empirical Value-at-Risk (VaR) and Conditional Value-at-Risk
(CVaR), via the Rockafellar-Uryasev (2000) formulation, applied to
scenario losses produced by Phase 7's empirical scenario resampling
(src.optimization.uncertainty).

This module is ADDITIVE and self-contained. It does not import, touch,
or depend on any Phase 1-6 module. It operates purely on caller-supplied
arrays of scenario losses (plain numpy), so it has no coupling to
MobilitySimulator, the predictor interface, or the physical/sensing
channel models.

===========================================================================
1. Scenario loss -- explicit mathematical definition
===========================================================================

CVaR is a coherent risk measure defined over LOSSES (larger = worse).
Phase 6's horizon objective (src.simulation.end_to_end._horizon_objective,
via churn_aware_decision's "combined_objective") is a GAIN to be
maximized. To reuse that objective under a risk measure, this module
fixes the loss of scenario i to be the negation of that scenario's
objective value, and nothing else:

    l_i := -g_i ,   i = 1, ..., N

where g_i is the (deterministic, already-computed) scalar objective
value for scenario i -- e.g. _horizon_objective evaluated on one
predicted trajectory drawn from Phase 7's residual-resampling scenario
generator. No noise, smoothing, weighting or distributional assumption
is added on top of g_i; the ONLY transformation applied is the sign
flip that turns "gain to maximize" into "loss to risk-measure".

The scenario set {g_i} (equivalently {l_i}) is empirical: each of the N
scenarios carries equal probability weight 1/N. No parametric family
(Gaussian, etc.) is fit to it anywhere in this module -- this matches
Phase 7's constraint of using only measured CV residuals / empirical
resampling, never fabricated predictor variance.

===========================================================================
2. Empirical VaR
===========================================================================

For a finite empirical sample l_1, ..., l_N (equal weight 1/N each),
the alpha-Value-at-Risk (0 < alpha < 1) is the smallest loss threshold
covering at least an alpha fraction of the empirical probability mass:

    VaR_alpha(L) = inf { v in R : (1/N) sum_i 1[l_i <= v] >= alpha }

For a finite sample this is exactly the ceil(alpha*N)-th order
statistic:

    VaR_alpha(L) = l_(k),      k = ceil(alpha * N)

where l_(1) <= l_(2) <= ... <= l_(N) is the ascending-sorted sample
(1-indexed).

===========================================================================
3. Empirical CVaR -- Rockafellar-Uryasev (2000) formulation
===========================================================================

Rockafellar & Uryasev (2000), "Optimization of Conditional
Value-at-Risk", show CVaR_alpha(L) is the value of the convex
minimization

    CVaR_alpha(L) = min_{z in R}  F_alpha(z),

    F_alpha(z) := z + 1/(1-alpha) * E[ max(L - z, 0) ]

and that the minimizing z* equals VaR_alpha(L) (their Theorem 1). For
the empirical (equal-weight, finite-sample) distribution used here,
E[max(L - z, 0)] = (1/N) sum_i max(l_i - z, 0), so:

    CVaR_alpha(L) = F_alpha(z*)
                  = z* + 1/((1-alpha)*N) * sum_i max(l_i - z*, 0),
                  where z* = VaR_alpha(L)  (Section 2).

This module evaluates F_alpha AT z* = VaR_alpha directly -- it does not
use a separate "tail average" shortcut -- so the VaR and CVaR reported
are always mutually consistent with the single RU definition above (they
agree with the tail-average shortcut exactly when alpha*N is an
integer, and generalize it correctly via partial weighting otherwise;
see tests/test_phase7_cvar.py for a worked numeric check).

===========================================================================
4. Zero-uncertainty reduction to deterministic Phase 6 (identity)
===========================================================================

When there is exactly one scenario (N = 1), for EVERY alpha in (0, 1):

    k = ceil(alpha * 1) = 1  =>  VaR_alpha(L) = l_1
    shortfall = max(l_1 - l_1, 0) = 0
    CVaR_alpha(L) = l_1 + 0 = l_1

So both VaR and CVaR collapse, exactly (not approximately), to the
single scenario's loss, independent of alpha. Composed with the
gain<->loss sign flip in cvar_risk_adjusted_objective, the risk-adjusted
objective collapses to g_1: precisely Phase 6's deterministic objective,
with no residual risk term. This is a structural property of the RU
formula above, not a special case that had to be coded separately.

===========================================================================
5. Design choices this module does NOT make
===========================================================================

  * No invented probability distribution: only the empirical (sample)
    distribution over caller-supplied scenario losses is used.
  * No randomness inside this module: scenario generation (seeded
    residual resampling) lives in src.optimization.uncertainty; this
    module is a pure, deterministic function of its inputs.
  * No silent handling of zero scenarios: an empty scenario array
    raises ValueError (Section 6) rather than returning some
    fallback/sentinel value.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "CVaRResult",
    "empirical_var",
    "empirical_cvar",
    "evaluate_cvar",
    "cvar_risk_adjusted_objective",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_alpha(alpha) -> float:
    """Requirement 3: alpha must satisfy 0 < alpha < 1, strictly."""
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(
            f"alpha must satisfy 0 < alpha < 1 (got alpha={alpha}). "
            "alpha=0 and alpha=1 are excluded because VaR/CVaR degenerate "
            "(alpha=0 -> the minimum loss is always covered trivially; "
            "alpha=1 -> the tail probability mass 1-alpha is zero, making "
            "the Rockafellar-Uryasev denominator 1/((1-alpha)*N) undefined)."
        )
    return alpha


def _validate_losses(losses) -> np.ndarray:
    """Requirement 5: zero-scenario handling -- reject empty input explicitly."""
    losses = np.asarray(losses, dtype=float).ravel()
    if losses.shape[0] == 0:
        raise ValueError(
            "Zero scenarios provided: VaR/CVaR are undefined for an empty "
            "scenario set. A single scenario (N=1) is the valid "
            "deterministic case (see module docstring, Section 4) and "
            "reduces VaR=CVaR=that scenario's loss; N=0 is not handled by "
            "extrapolation or by inventing a fallback value."
        )
    if not np.all(np.isfinite(losses)):
        raise ValueError("Scenario losses must all be finite (no NaN/inf).")
    return losses


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CVaRResult:
    """
    Output of evaluate_cvar.

    Attributes
    ----------
    var             : VaR_alpha(L), the empirical alpha-quantile of losses.
    cvar            : CVaR_alpha(L), the Rockafellar-Uryasev value at z*=var.
    alpha           : confidence level used (0 < alpha < 1).
    num_scenarios   : N, number of scenario losses supplied.
    losses          : the (validated, flattened) input loss array, unchanged.
    """

    var: float
    cvar: float
    alpha: float
    num_scenarios: int
    losses: np.ndarray


# ---------------------------------------------------------------------------
# Core empirical VaR / CVaR
# ---------------------------------------------------------------------------

def empirical_var(losses, alpha) -> float:
    """
    Empirical Value-at-Risk at confidence level alpha (see module
    docstring, Section 2, for the exact definition).

    Deterministic: for a fixed `losses` array and `alpha`, always returns
    the same value (pure function of its inputs; uses np.sort, no RNG).
    """
    alpha = _validate_alpha(alpha)
    losses = _validate_losses(losses)
    n = losses.shape[0]

    sorted_losses = np.sort(losses)
    k = int(np.ceil(alpha * n - 1e-12))  # tolerance guards float round-off
    k = min(max(k, 1), n)
    return float(sorted_losses[k - 1])


def empirical_cvar(losses, alpha, var: Optional[float] = None) -> float:
    """
    Empirical CVaR at confidence level alpha, evaluated via the
    Rockafellar-Uryasev formula at z* = VaR_alpha (module docstring,
    Section 3).

    Parameters
    ----------
    losses : array-like of scenario losses l_1..l_N.
    alpha  : confidence level, 0 < alpha < 1.
    var    : optional precomputed VaR_alpha(losses); if omitted it is
             computed internally via empirical_var (kept as a parameter
             purely so evaluate_cvar can compute VaR once and reuse it,
             not as a way to substitute an externally-fit quantile).
    """
    alpha = _validate_alpha(alpha)
    losses = _validate_losses(losses)
    n = losses.shape[0]

    if var is None:
        var = empirical_var(losses, alpha)

    shortfall = np.maximum(losses - var, 0.0)
    cvar = var + shortfall.sum() / ((1.0 - alpha) * n)
    return float(cvar)


def evaluate_cvar(losses, alpha) -> CVaRResult:
    """
    Compute VaR and CVaR together (VaR computed once, reused for CVaR so
    the two numbers are always mutually consistent under one RU
    evaluation). Returns a CVaRResult.
    """
    alpha = _validate_alpha(alpha)
    losses = _validate_losses(losses)

    var = empirical_var(losses, alpha)
    cvar = empirical_cvar(losses, alpha, var=var)

    return CVaRResult(
        var=var,
        cvar=cvar,
        alpha=alpha,
        num_scenarios=int(losses.shape[0]),
        losses=losses,
    )


# ---------------------------------------------------------------------------
# Gain -> loss bridge for the Phase 6 slow-update objective
# ---------------------------------------------------------------------------

def cvar_risk_adjusted_objective(
    scenario_objectives, alpha
) -> Tuple[float, CVaRResult]:
    """
    Turn a set of per-scenario Phase 6 horizon-objective (GAIN) values
    into a single risk-adjusted scalar objective, via CVaR of the
    corresponding losses (module docstring, Section 1).

        l_i            = -g_i                          (Section 1)
        G_CVaR_alpha    = -CVaR_alpha( {l_i} )          (Section 3)

    G_CVaR_alpha is the alpha-worst-case-averaged objective: the
    (RU-weighted) mean objective value over the worst (1-alpha) fraction
    of scenarios. It satisfies G_CVaR_alpha <= mean_i(g_i) always
    (risk-averse, never optimistic relative to the plain empirical mean
    objective), and reduces EXACTLY to g_1 when only one scenario is
    supplied, for every alpha in (0,1) -- see Section 4.

    Parameters
    ----------
    scenario_objectives : array-like of N scalar gain values g_1..g_N
                           (e.g. _horizon_objective evaluated once per
                           resampled scenario trajectory).
    alpha                : confidence level, 0 < alpha < 1.

    Returns
    -------
    (risk_adjusted_objective, CVaRResult)
        risk_adjusted_objective : float, G_CVaR_alpha as defined above.
        CVaRResult               : the underlying VaR/CVaR-of-losses
                                    result, for logging/diagnostics
                                    (result.cvar = -risk_adjusted_objective,
                                    result.losses = -scenario_objectives).
    """
    scenario_objectives = np.asarray(scenario_objectives, dtype=float).ravel()
    losses = -scenario_objectives
    result = evaluate_cvar(losses, alpha)
    return -result.cvar, result
