"""
Four-Pathway Mass Screening Simulation
Wilde (2026): Structural Causal Modeling for Mass Screening Programs

Implements structural equations from §4, produces:
1. Point estimates for Chat Control vs. Police Polygraph
2. Sensitivity table: Y* as base rate varies from 1/100 to 1/10,000
3. Uncertainty propagation (Gelman & Carpenter approach)

Parameters grounded in §11 literature estimates.
"""

import numpy as np
import pandas as pd

# ── 1. STRUCTURAL EQUATION FUNCTIONS ─────────────────────────────────────────

def classification_output(T, X, pi, sensitivity, specificity):
    """
    C := T · X · [π·sensitivity + (1-π)·(1-specificity)]
    Returns (true_positives, false_positives, false_negatives, true_negatives)
    """
    flagged      = T * X * (pi * sensitivity + (1 - pi) * (1 - specificity))
    true_pos     = T * X * pi * sensitivity
    false_pos    = T * X * (1 - pi) * (1 - specificity)
    false_neg    = T * X * pi * (1 - sensitivity)
    true_neg     = T * X * (1 - pi) * specificity
    return true_pos, false_pos, false_neg, true_neg

def strategy_effect(T, awareness, evasion_rate, sophistication):
    """
    S := θ_deterrence · T · awareness - θ_evasion · sophistication
    Positive = net deterrence; negative = net evasion.
    awareness: how credible/visible the program is [0,1]
    sophistication: proportion of targets who can evade [0,1]
    """
    deterrence = 0.15 * T * awareness      # r~0.15 from Pratt et al. (2006)
    evasion    = evasion_rate * sophistication
    return deterrence - evasion

def information_yield(T, S, X, bogus_pipeline_d, has_interrogation):
    """
    I := θ_confession · T · (1 - S/X) · bogus_pipeline_effect
    bogus_pipeline_d: Cohen's d from Roese & Jamieson (1993) = 0.41
    has_interrogation: 1 if human interrogation present, 0 if automated
    """
    if not has_interrogation:
        return 0.0
    # Convert d to probability scale (approximate)
    confession_prob = bogus_pipeline_d / (bogus_pipeline_d + 1)
    residual_pool   = max(0, 1 - S / max(X, 1))
    return T * residual_pool * confession_prob

def resource_cost(false_pos, capacity, cost_per_applicant):
    """
    R := base_allocation + θ_fp_burden · C/capacity
    Returns ratio of capacity consumed by false positive triage.
    """
    triage_hours  = false_pos * (0.5 / 60)   # 30 seconds per flag in hours
    return min(triage_hours / max(capacity, 1), 10.0)  # cap at 10x capacity

def outcome(true_pos, false_pos, S, I, R, X, pi):
    """
    Y* := θ_C·TP - θ_FP·FP + θ_S·S + θ_I·I - θ_R·R·opportunity_cost
    Normalised to population so programs of different scale are comparable.
    """
    theta_C  =  1.0   # benefit per true positive detected
    theta_FP = -0.5   # harm per false positive (investigation burden + liberty)
    theta_S  =  0.8   # benefit per unit net deterrence
    theta_I  =  0.6   # benefit per unit information yield
    theta_R  = -2.0   # cost per unit resource overload

    classification_contrib   = (theta_C * true_pos + theta_FP * false_pos) / max(X, 1)
    strategy_contrib         = theta_S * S
    information_contrib      = theta_I * I
    resource_contrib         = theta_R * R

    return classification_contrib + strategy_contrib + information_contrib + resource_contrib


# ── 2. PROGRAM PARAMETER PROFILES ────────────────────────────────────────────

PROGRAMS = {
    "Chat Control": {
        "T"               : 1.0,          # full coverage
        "X"               : 10_000_000_000,
        "sensitivity"     : 0.80,         # generous assumption per §5.1
        "specificity"     : 0.84,         # 1 - false_positive_rate(0.16)
        "awareness"       : 0.3,          # low: automated, invisible to most
        "evasion_rate"    : 0.8,          # high: encryption trivially available
        "sophistication"  : 0.9,          # serious offenders are technically capable
        "bogus_pipeline_d": 0.41,
        "has_interrogation": False,
        "capacity"        : 3000 * 2000,  # ~3000 EU investigators × 2000 hrs/yr
        "cost_per_unit"   : 0,
        "label"           : "Chat Control (10B msgs, π=1/1000)",
    },
    "Police Polygraph": {
        "T"               : 1.0,
        "X"               : 50_000,       # approximate annual US LEA applicants using polygraph
        "sensitivity"     : 0.75,         # NAS (2003) midpoint
        "specificity"     : 0.65,         # NAS (2003) midpoint
        "awareness"       : 0.8,          # high: well-known, publicised
        "evasion_rate"    : 0.2,          # low: applicant pool less sophisticated
        "sophistication"  : 0.3,
        "bogus_pipeline_d": 0.41,         # Roese & Jamieson (1993)
        "has_interrogation": True,
        "capacity"        : 50_000,       # one-to-one: each applicant gets interview
        "cost_per_unit"   : 350,          # $200-500/applicant midpoint
        "label"           : "Police Polygraph (50k applicants)",
    },
}


# ── 3. SINGLE-RUN SIMULATION ──────────────────────────────────────────────────

def run_simulation(prog, pi):
    tp, fp, fn, tn = classification_output(
        prog["T"], prog["X"], pi,
        prog["sensitivity"], prog["specificity"]
    )
    S = strategy_effect(
        prog["T"], prog["awareness"],
        prog["evasion_rate"], prog["sophistication"]
    )
    I = information_yield(
        prog["T"], S, prog["X"],
        prog["bogus_pipeline_d"], prog["has_interrogation"]
    )
    R = resource_cost(fp, prog["capacity"], prog["cost_per_unit"])
    Y = outcome(tp, fp, S, I, R, prog["X"], pi)

    return {
        "pi"           : pi,
        "true_pos"     : tp,
        "false_pos"    : fp,
        "false_neg"    : fn,
        "ppv"          : tp / max(tp + fp, 1),   # precision
        "strategy"     : S,
        "information"  : I,
        "resource_load": R,
        "Y_star"       : Y,
    }


# ── 4. POINT ESTIMATES ────────────────────────────────────────────────────────

print("=" * 70)
print("FOUR-PATHWAY SIMULATION: POINT ESTIMATES")
print("Wilde (2026) — §5 & §6 working paper")
print("=" * 70)

BASE_RATES = {
    "Chat Control" : 1/1000,
    "Police Polygraph": 0.10,   # ~10% problematic applicants (conservative)
}

results = {}
for name, prog in PROGRAMS.items():
    pi = BASE_RATES[name]
    r  = run_simulation(prog, pi)
    results[name] = r

    print(f"\n{'─'*60}")
    print(f"Program : {prog['label']}")
    print(f"Base rate (π): 1/{int(round(1/pi))}")
    print(f"\n  CLASSIFICATION PATHWAY")
    print(f"    True positives  : {r['true_pos']:>15,.0f}")
    print(f"    False positives : {r['false_pos']:>15,.0f}")
    print(f"    False negatives : {r['false_neg']:>15,.0f}")
    print(f"    Precision (PPV) : {r['ppv']:>15.4f}  ({r['ppv']*100:.2f}% of flags are real)")
    print(f"\n  STRATEGY PATHWAY")
    print(f"    Net deterrence  : {r['strategy']:>15.4f}  ({'positive' if r['strategy']>0 else 'negative/evasion dominates'})")
    print(f"\n  INFORMATION PATHWAY")
    print(f"    Information yield: {r['information']:>14.4f}")
    print(f"\n  RESOURCE PATHWAY")
    print(f"    Capacity consumed: {r['resource_load']:>13.1f}x  ({'CATASTROPHIC' if r['resource_load']>1 else 'manageable'})")
    print(f"\n  NET OUTCOME (Y*): {r['Y_star']:>+.4f}  ({'NET BENEFIT' if r['Y_star']>0 else 'NET HARM'})")


# ── 5. SENSITIVITY ANALYSIS: BASE RATE VARIATION ─────────────────────────────

print("\n\n" + "=" * 70)
print("SENSITIVITY ANALYSIS: Y* AS BASE RATE VARIES")
print("(Chat Control only — polygraph applicant pool less sensitive to π)")
print("=" * 70)

pi_values = [1/100, 1/500, 1/1000, 1/2000, 1/5000, 1/10000]
cc_prog   = PROGRAMS["Chat Control"]

rows = []
for pi in pi_values:
    r = run_simulation(cc_prog, pi)
    rows.append({
        "Base rate"       : f"1/{int(round(1/pi)):,}",
        "True positives"  : f"{r['true_pos']:>12,.0f}",
        "False positives" : f"{r['false_pos']:>15,.0f}",
        "PPV (%)"         : f"{r['ppv']*100:>8.3f}",
        "Capacity (×)"    : f"{r['resource_load']:>8.1f}",
        "Y* (net)"        : f"{r['Y_star']:>+10.4f}",
        "Verdict"         : "HARM" if r["Y_star"] < 0 else "benefit",
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

print("""
Notes:
  - Y* normalised per screened individual; negative = net harm
  - Resource load > 1.0 means false positive triage exceeds full investigator capacity
  - Parameters: sensitivity=0.80, specificity=0.84, evasion_rate=0.80
  - Strategy and information pathway contributions held constant across π
  - Theta weights (θ_C=1.0, θ_FP=-0.5, θ_S=0.8, θ_I=0.6, θ_R=-2.0) are 
    illustrative; sensitivity analysis over theta values is future work
  - See §11 for all parameter sources
""")


# ── 6. UNCERTAINTY PROPAGATION (Gelman & Carpenter approach) ─────────────────

print("=" * 70)
print("UNCERTAINTY PROPAGATION: MONTE CARLO OVER KEY PARAMETERS")
print("(Chat Control, π = 1/1000)")
print("=" * 70)

np.random.seed(42)
N_DRAWS = 10_000
pi_fixed = 1/1000

# Sample uncertain parameters
sensitivity_draws = np.random.beta(8, 2,   N_DRAWS)   # mean~0.80, uncertain
specificity_draws = np.random.beta(8, 2,   N_DRAWS)   # mean~0.80 (FPR~0.20)
evasion_draws     = np.random.beta(8, 2,   N_DRAWS)   # high evasion likely
pi_draws          = np.random.beta(1, 999, N_DRAWS)   # mean~1/1000, skewed

Y_draws = []
for i in range(N_DRAWS):
    prog_i = dict(cc_prog)
    prog_i["sensitivity"] = sensitivity_draws[i]
    prog_i["specificity"] = specificity_draws[i]
    prog_i["evasion_rate"]= evasion_draws[i]
    r = run_simulation(prog_i, pi_draws[i])
    Y_draws.append(r["Y_star"])

Y_draws = np.array(Y_draws)
print(f"\n  Y* distribution over {N_DRAWS:,} parameter draws:")
print(f"    Mean              : {Y_draws.mean():>+.4f}")
print(f"    Median            : {np.median(Y_draws):>+.4f}")
print(f"    2.5th percentile  : {np.percentile(Y_draws, 2.5):>+.4f}")
print(f"    97.5th percentile : {np.percentile(Y_draws, 97.5):>+.4f}")
print(f"    P(Y* < 0)         : {(Y_draws < 0).mean()*100:.1f}%  (probability of net harm)")
print(f"\n  Interpretation: Under uncertainty over sensitivity, specificity,")
print(f"  evasion rate, and base rate, Chat Control produces net harm in")
print(f"  {(Y_draws<0).mean()*100:.0f}% of simulated parameter combinations.")
