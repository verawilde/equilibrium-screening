"""
Disaggregated Strategy Pathway for Mass Screening Programs
===========================================================
Extension of Wilde (2026) four-pathway framework.

THREE-SUBPOPULATION MODEL
--------------------------
The prior two-subpopulation model (casual vs. dedicated) mislocated casual offenders
entirely in the strategy pathway. This version corrects that and further disaggregates
casual offenders into two mechanistically distinct groups:

  Subpopulation 1 — Truly unaware casual offenders:
    - Distribute CSAM not knowing it is illegal (memes, "Stammtisch humour")
    - Hartmann (2025): "no deterrent effect on stupid people"
    - Deterrence mechanism structurally absent: no illegal-act calculus to affect
    - These are a CLASSIFICATION pathway matter (detectable, not deterrable)
    - Near-zero evasion, near-zero deterrence

  Subpopulation 2 — Vaguely aware / careless casual offenders:
    - Know conduct is "sketchy" but haven't thought through legal status
    - Plausibly deterrable via public awareness campaigns ("this IS a crime, you WILL be scanned")
    - The only subpopulation for which Chat Control's strategy pathway could be positive
    - This fraction (p_aware_casual) is the key sensitivity parameter
    - Still low harm per case relative to dedicated offenders

  Subpopulation 3 — Dedicated offenders:
    - Systematic CSAM distribution/production; know exactly what they are doing
    - Already on Dark Net with evasion tutorials (Hartmann 2025)
    - Near-zero deterrence, high evasion
    - High harm per case; dominate child safety outcomes

SENSITIVITY ANALYSIS
---------------------
The key question for the working paper: even under generous assumptions about
the deterrable fraction of casual offenders, does dedicated-offender evasion
still dominate the harm-weighted aggregate?

Sweep over p_aware_casual ∈ [0.0, 1.0]:
  - 0.0 = Hartmann's strong view: casual offenders entirely unaware, non-deterrable
  - 0.5 = moderate assumption: half are vaguely aware and deterrable
  - 1.0 = generous upper bound: all casual offenders are deterrable

Result: harm-weighted strategy effect remains negative across the full sweep.
The dedicated-offender evasion term (~3.5× harm weight, 75% evasion rate) swamps
any plausible casual-offender deterrence contribution.

This means the working paper claim can be framed as:
  "Robust to generous assumptions" — stronger rhetorically than a logical assertion,
  harder to attack, and directly supported by the simulation.

References:
  Hartmann (expert interview, May 21, 2025) — ZAC NRW cybercrime prosecutor
  Rubenstein et al. (2017) — causal consistency of structural equation models
  Working paper §3.5–3.6, §5.3
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class StrategySubpopulation:
    name: str
    share: float            # Fraction of total offender population [0,1]
    harm_per_case: float    # Relative harm weight (1.0 = baseline)
    deterrence_rate: float  # P(deterred | aware of illegality) [0,1]
    evasion_rate: float     # P(evades detection | aware of program) [0,1]
    awareness: float        # P(aware that conduct is illegal) [0,1]
    description: str = ""


def build_chat_control_subpops(p_aware_casual: float = 0.15) -> List[StrategySubpopulation]:
    """
    Three-subpopulation Chat Control model.

    p_aware_casual: fraction of casual offenders who are vaguely aware of illegality
                    and thus plausibly deterrable. Key sensitivity parameter.
                    Hartmann's view ~ 0.05–0.15; generous upper bound ~ 1.0.
    """
    p_aware_casual = float(np.clip(p_aware_casual, 0.0, 1.0))

    # Total casual share held constant at 0.45 (Hartmann: 40–50%)
    casual_share = 0.45
    dedicated_share = 0.55

    return [
        StrategySubpopulation(
            name="casual_unaware",
            share=casual_share * (1 - p_aware_casual),
            harm_per_case=0.15,
            deterrence_rate=0.00,   # Structural zero: no illegal-act calculus
            evasion_rate=0.01,      # Near-zero: nothing to evade from
            awareness=0.02,         # Essentially unaware
            description=(
                f"Truly unaware casual offenders ({100*(1-p_aware_casual):.0f}% of casual pool). "
                "Hartmann: 'no deterrent effect on stupid people.' "
                "Classification pathway matter, not strategy."
            ),
        ),
        StrategySubpopulation(
            name="casual_aware",
            share=casual_share * p_aware_casual,
            harm_per_case=0.25,
            deterrence_rate=0.45,   # Plausibly deterrable via public awareness
            evasion_rate=0.08,      # Low: not sophisticated enough to evade
            awareness=0.70,         # Vaguely aware conduct is legally risky
            description=(
                f"Vaguely aware / careless casual offenders ({100*p_aware_casual:.0f}% of casual pool). "
                "Know it's 'sketchy'; deterrable via PR campaign around program existence. "
                "Only subpop with potentially positive strategy contribution."
            ),
        ),
        StrategySubpopulation(
            name="dedicated_offenders",
            share=dedicated_share,
            harm_per_case=3.50,
            deterrence_rate=0.03,   # Near-zero: expect to evade, not be deterred
            evasion_rate=0.75,      # High: Dark Net tutorials already in use
            awareness=0.98,
            description=(
                "Systematic CSAM distribution/production. "
                "Hartmann: 'already on the Dark Net, specialized online forums offer "
                "users a tutorial, education for avoiding detection.' "
                "Dominate harm-weighted outcome."
            ),
        ),
    ]


POLYGRAPH_SUBPOPS = [
    StrategySubpopulation(
        name="casual_problematic_applicants",
        share=0.70,
        harm_per_case=0.5,
        deterrence_rate=0.30,
        evasion_rate=0.15,
        awareness=0.85,
        description="Applicants with past misconduct; self-select out under interrogation pressure.",
    ),
    StrategySubpopulation(
        name="sophisticated_deceivers",
        share=0.30,
        harm_per_case=2.0,
        deterrence_rate=0.05,
        evasion_rate=0.40,
        awareness=0.99,
        description=(
            "Determined bad actors. Lower evasion than digital context: "
            "face-to-face interrogation limits preparation."
        ),
    ),
]


class DisaggregatedStrategyModel:

    def __init__(self, subpopulations: List[StrategySubpopulation]):
        self.subpops = subpopulations
        total = sum(s.share for s in subpopulations)
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"Subpopulation shares must sum to ~1.0 (got {total:.3f})")

    def subpop_effect(self, pop: StrategySubpopulation, T: float, X: float) -> Dict:
        n = pop.share * X
        aware = T * pop.awareness
        n_deterred = aware * pop.deterrence_rate * n
        n_evaders = aware * pop.evasion_rate * n
        net_raw = n_deterred - n_evaders
        net_hw = pop.harm_per_case * net_raw
        return dict(
            name=pop.name,
            description=pop.description,
            n=n,
            n_deterred=n_deterred,
            n_evaders=n_evaders,
            net_raw=net_raw,
            harm_per_case=pop.harm_per_case,
            net_harm_weighted=net_hw,
        )

    def aggregate(self, T: float = 1.0, X: float = 10_000.0) -> Dict:
        results = [self.subpop_effect(p, T, X) for p in self.subpops]
        net_raw = sum(r["net_raw"] for r in results)
        net_hw = sum(r["net_harm_weighted"] for r in results)

        naive_det = sum(p.share * p.deterrence_rate * p.awareness for p in self.subpops)
        naive_eva = sum(p.share * p.evasion_rate * p.awareness for p in self.subpops)
        naive_net = (naive_det - naive_eva) * T * X

        return dict(
            subpop_results=results,
            net_raw=net_raw,
            net_harm_weighted=net_hw,
            naive_net=naive_net,
            sign_agree=np.sign(naive_net) == np.sign(net_hw),
        )


def sensitivity_sweep(
    p_aware_range: Optional[np.ndarray] = None,
    T: float = 1.0,
    X: float = 10_000.0,
) -> Dict[str, np.ndarray]:
    """
    Sweep p_aware_casual from 0 → 1 and track:
      - harm-weighted strategy effect for each subpopulation
      - total harm-weighted aggregate
      - naive aggregate (for comparison)
      - sign of total (does it ever go positive?)
    """
    if p_aware_range is None:
        p_aware_range = np.linspace(0.0, 1.0, 101)

    results = {
        "p_aware": p_aware_range,
        "casual_unaware_hw": np.zeros_like(p_aware_range),
        "casual_aware_hw": np.zeros_like(p_aware_range),
        "dedicated_hw": np.zeros_like(p_aware_range),
        "total_hw": np.zeros_like(p_aware_range),
        "naive_net": np.zeros_like(p_aware_range),
    }

    for i, p in enumerate(p_aware_range):
        subpops = build_chat_control_subpops(p_aware_casual=p)
        model = DisaggregatedStrategyModel(subpops)
        agg = model.aggregate(T=T, X=X)

        for r in agg["subpop_results"]:
            if r["name"] == "casual_unaware":
                results["casual_unaware_hw"][i] = r["net_harm_weighted"]
            elif r["name"] == "casual_aware":
                results["casual_aware_hw"][i] = r["net_harm_weighted"]
            elif r["name"] == "dedicated_offenders":
                results["dedicated_hw"][i] = r["net_harm_weighted"]

        results["total_hw"][i] = agg["net_harm_weighted"]
        results["naive_net"][i] = agg["naive_net"]

    return results


def print_sensitivity_report(sweep: Dict[str, np.ndarray]):
    """Print key quantitative findings from the sensitivity sweep."""
    print("\n" + "=" * 70)
    print("SENSITIVITY SWEEP: p_aware_casual ∈ [0, 1]")
    print("Harm-weighted strategy pathway effect, Chat Control")
    print("X = 10,000 offenders | T = 1.0")
    print("=" * 70)

    p = sweep["p_aware"]
    total = sweep["total_hw"]
    dedicated = sweep["dedicated_hw"]
    ca_aware = sweep["casual_aware_hw"]

    print(f"\n{'p_aware':>8}  {'casual_aware_hw':>16}  {'dedicated_hw':>14}  "
          f"{'total_hw':>10}  {'sign':>5}")
    print("-" * 62)

    for idx in [0, 10, 25, 50, 75, 100]:
        if idx >= len(p):
            idx = len(p) - 1
        sign = "+" if total[idx] > 0 else "-"
        print(
            f"{p[idx]:>8.2f}  {ca_aware[idx]:>16.1f}  {dedicated[idx]:>14.1f}  "
            f"{total[idx]:>10.1f}  {sign:>5}"
        )

    ever_positive = np.any(total > 0)
    crossover = None if not ever_positive else p[np.argmax(total > 0)]

    print("\n" + "-" * 70)
    print(f"Total harm-weighted effect range: [{total.min():.1f}, {total.max():.1f}]")
    print(f"Effect ever positive?  {ever_positive}")
    if crossover is not None:
        print(f"Sign crossover at p_aware = {crossover:.2f}")
    else:
        print("No sign crossover across full [0,1] range.")

    # How much does casual-aware deterrence contribute at p=1.0 (max generous)?
    max_casual = ca_aware[-1]
    ded = dedicated[0]  # dedicated is constant across sweep
    ratio = abs(ded) / (abs(max_casual) + 1e-9)
    print(
        f"\nAt p_aware=1.0 (most generous): "
        f"dedicated evasion ({ded:.1f}) is {ratio:.1f}× "
        f"casual deterrence ({max_casual:.1f})"
    )
    print(
        "Interpretation: even if ALL casual offenders are vaguely aware and deterrable,\n"
        "dedicated-offender evasion dominates the harm-weighted aggregate."
    )
    print("=" * 70)


def print_full_report(p_aware_casual: float = 0.15, X: float = 10_000.0):
    """Full subpopulation breakdown at a given p_aware_casual value."""
    print("\n" + "=" * 70)
    print(f"DISAGGREGATED STRATEGY PATHWAY — CHAT CONTROL")
    print(f"p_aware_casual = {p_aware_casual:.2f}, X = {X:.0f}, T = 1.0")
    print("=" * 70)

    subpops = build_chat_control_subpops(p_aware_casual)
    model = DisaggregatedStrategyModel(subpops)
    agg = model.aggregate(T=1.0, X=X)

    for r in agg["subpop_results"]:
        print(f"\n  [{r['name']}]")
        print(f"    {r['description']}")
        print(f"    N:                   {r['n']:.0f}")
        print(f"    N deterred:          {r['n_deterred']:.1f}")
        print(f"    N evaders:           {r['n_evaders']:.1f}")
        print(f"    Net (unweighted):    {r['net_raw']:.1f}")
        print(f"    Harm weight:         {r['harm_per_case']:.2f}")
        print(f"    Net (harm-weighted): {r['net_harm_weighted']:.1f}")

    print(f"\n  AGGREGATE (naive):        {agg['naive_net']:.1f}")
    print(f"  AGGREGATE (harm-weighted): {agg['net_harm_weighted']:.1f}")
    print(f"  Signs agree:               {agg['sign_agree']}")

    # Contribution breakdown
    total_hw = agg["net_harm_weighted"]
    for r in agg["subpop_results"]:
        pct = 100 * r["net_harm_weighted"] / (total_hw + 1e-9)
        print(f"  {r['name']:25s} share of harm-weighted total: {pct:.1f}%")

    print("=" * 70)


if __name__ == "__main__":

    # 1. Full breakdown at Hartmann's conservative estimate
    print_full_report(p_aware_casual=0.10)

    # 2. Full breakdown at a moderate assumption
    print_full_report(p_aware_casual=0.50)

    # 3. Sensitivity sweep
    sweep = sensitivity_sweep()
    print_sensitivity_report(sweep)

    # 4. Polygraph comparison
    print("\n" + "=" * 70)
    print("POLYGRAPH STRATEGY PATHWAY (for contrast)")
    print("=" * 70)
    poly = DisaggregatedStrategyModel(POLYGRAPH_SUBPOPS)
    agg = poly.aggregate(T=1.0, X=1000.0)
    for r in agg["subpop_results"]:
        print(f"\n  [{r['name']}]")
        print(f"    Net (harm-weighted): {r['net_harm_weighted']:.1f}")
    print(f"\n  AGGREGATE (harm-weighted): {agg['net_harm_weighted']:.1f}")
    print(
        "\n  NOTE: Polygraph strategy pathway is also mildly negative here.\n"
        "  Polygraph's positive Y* in the full simulation is driven by the\n"
        "  Information pathway (bogus pipeline, d=0.41), not by deterrence.\n"
        "  This confirms the working paper's interpretation of the LEMAS finding."
    )
    print("=" * 70)
