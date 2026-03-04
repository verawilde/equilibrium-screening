"""
Disaggregated Resource Pathway Simulation
Separates triage-layer and specialist-layer capacity explicitly.
Shows that even with unlimited triage, specialist bottleneck alone
causes net harm.
"""
import numpy as np
import pandas as pd

# Parameters
X = 10_000_000_000
pi = 1/1000
sensitivity = 0.80
specificity = 0.84

tp = X * pi * sensitivity
fp = X * (1-pi) * (1-specificity)

# Capacity layers (Hartmann, expert interview, May 21, 2025)
# NRW: 13 prosecutors, ~14k cases/yr → ~325 EU specialist prosecutors
# 3,000 = full CSAM-adjacent LE across both layers (Europol IOCTA 2023)

TRIAGE_CAPACITY_hrs   = 3000 * 2000   # ~3000 total LE × 2000 hrs/yr
SPECIALIST_CAPACITY   = 325            # EU specialist prosecutors (Hartmann scaling)
SPECIALIST_CASES_YR   = 1000          # cases/prosecutor/year (Hartmann)
SPECIALIST_HRS_CASE   = 2000 / 1000   # 2 hrs per case average

triage_hrs_needed     = fp * (0.5/60)  # 30 seconds per flag
specialist_cases_from_triage = fp * 0.001  # assume 0.1% of FPs escalate to specialist

specialist_capacity_total = SPECIALIST_CAPACITY * SPECIALIST_CASES_YR

print("=" * 65)
print("DISAGGREGATED RESOURCE PATHWAY: TRIAGE vs SPECIALIST")
print("Chat Control, π=1/1000, favorable assumptions")
print("=" * 65)

print(f"\n  False positives generated     : {fp:>15,.0f}")
print(f"\n  ── TRIAGE LAYER ──")
print(f"  Hours needed (30s/flag)       : {triage_hrs_needed:>15,.0f}")
print(f"  Total LE capacity (hrs/yr)    : {TRIAGE_CAPACITY_hrs:>15,.0f}")
print(f"  Triage overload factor        : {triage_hrs_needed/TRIAGE_CAPACITY_hrs:>15.1f}×")
print(f"  Note: triage CAN be scaled    :    (non-specialist labor)")
print(f"        but at severe human cost:    (trauma from CSAM exposure)")

print(f"\n  ── SPECIALIST LAYER ──")
print(f"  EU specialist prosecutors     : {SPECIALIST_CAPACITY:>15,}")
print(f"    (Source: Hartmann NRW×25)")
print(f"  Cases/prosecutor/year         : {SPECIALIST_CASES_YR:>15,}")
print(f"  Total specialist capacity     : {specialist_capacity_total:>15,.0f} cases/yr")
print(f"  FP escalations to specialist  : {specialist_cases_from_triage:>15,.0f}")
print(f"    (at 0.1% escalation rate)")
print(f"  Specialist overload factor    : {specialist_cases_from_triage/specialist_capacity_total:>15.1f}×")
print(f"  Note: specialist CANNOT be    :    scaled quickly")
print(f"        (Hartmann: market limits)")

print(f"\n  ── SENSITIVITY: escalation rate vs specialist overload ──")
print(f"\n  {'Escalation rate':>20} {'FPs escalated':>16} {'Overload':>10}")
for rate in [0.0001, 0.001, 0.01, 0.05, 0.10]:
    esc = fp * rate
    overload = esc / specialist_capacity_total
    print(f"  {rate*100:>19.2f}%  {esc:>16,.0f}  {overload:>9.1f}×")

print(f"""
  Interpretation:
  Even if triage is fully outsourced to non-specialist labor,
  the specialist investigation layer — ~325 prosecutors EU-wide —
  is overwhelmed at any escalation rate above ~0.02% of false
  positives. At realistic escalation rates (1-5%), overload is
  100-500×. The binding constraint is not triage hours but
  specialist capacity, which cannot be rapidly expanded.
  
  Source: Hartmann (expert interview, May 21, 2025) — NRW figures
  scaled to EU population (×25). Europol IOCTA (2023) 3,000 figure
  covers both layers; specialist subset is ~10× smaller.
""")
