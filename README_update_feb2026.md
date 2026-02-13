# equilibrium-screening — February 2026 Update

## What's New: Ribers & Ullrich Extension + Cross-Domain Validation Hierarchy

### Background

The original repository implemented Besserve & Schölkopf's (2022) equilibrium
framework applied to the four-pathway structural equations for security screening
programs (polygraph, iBorderCtrl, Chat Control). The medical modules contained
starter code parameterized from Budzyń et al. (2025) published data.

This update integrates the Ribers & Ullrich (2024) / Kleinberg et al. (2018)
**complementarity analysis** framework — asking where AI outperforms humans and
vice versa, stratified by case difficulty — across the full validation hierarchy.

### The Key Insight

The R&U complementarity framework requires a **solved validation problem**: an
observable answer sheet that tells you whether each classification decision was
correct. This requirement is met in some screening domains and not others:

| Domain | Validation Level | Gold Standard | R&U Applicable? |
|--------|-----------------|---------------|-----------------|
| HIV testing | **Fully solved** | Confirmatory serology | ✅ Yes |
| Colonoscopy | **Largely solved** | Histopathology | ✅ Yes |
| Mammography | **Persistently open** | DCIS ambiguity contaminates | ❌ No |
| Security (polygraph, Chat Control) | **Unsolvable** | No gold standard for deception | ❌ No |

The equilibrium framework (B&S 2022) applies to ALL domains — it models
system-level feedback dynamics regardless of gold standard availability. But the
**empirical precision** achievable differs fundamentally by domain.

**In security, we model what the ROC space WOULD look like under assumed
parameters. In colonoscopy, we can OBSERVE it. This asymmetry is the point.**

### Colonoscopy as Secondary Screening

A critical feature formalized in this update: colonoscopy in European CRC
screening is a **secondary screening**, preceded by FIT (fecal immunochemical
test). The population entering ACCEPT's colonoscopy randomization has been
pre-selected by FIT with varying national thresholds (Norway: 15 μg/g; others:
20–150 μg/g). This means:

- Cross-site comparison in ACCEPT is not "same AI, different populations" but
  "same AI, **differently FIT-selected populations**"
- The FIT threshold is itself a product of the **strategy pathway** operating at
  the system-design level (Porter 1995; Greenland 2023; Welch et al. 2011)
- The FIT stage has its own **largely-solved validation problem** (colonoscopy
  reveals true status for FIT-positives; cancer registries track interval cancers
  for FIT-negatives) — creating an **unexploited R&U-style complementarity
  analysis opportunity** at the upstream screening stage

To our knowledge, no one has framed FIT threshold-setting as a signal detection
problem amenable to R&U-style complementarity analysis.

### New Module: `cross_domain/`

```
cross_domain/
├── __init__.py
├── validation_hierarchy.py       # Domain specs, gold standards, analytical tools
└── complementarity_analysis.py   # R&U logic: observed vs hypothetical
```

**`validation_hierarchy.py`**: Formal specification of where each screening domain
sits on the validation solvability spectrum. Includes multi-stage pipeline
representation (FIT → colonoscopy), characterization of gold standards and their
limitations, and mapping of which analytical tools are available at each level.

**`complementarity_analysis.py`**: Applies R&U/Kleinberg difficulty-stratified
deployment logic across domains. For colonoscopy: specifies empirically testable
predictions using ACCEPT histopathology data (Prediction 5a-c from working paper).
For security: simulates hypothetical complementarity under stated NAS assumptions
with prominent caveats.

### Updated Repository Structure

```
equilibrium-screening/
├── README.md
├── README_update_feb2026.md      # This file
├── requirements.txt
├── security/
│   └── screening_equilibrium.py  # Four-pathway model (Chat Control, Polygraph, iBorderCtrl)
├── medical/
│   ├── colonoscopy_deskilling.py # Budzyń parameterization (starter, awaiting ACCEPT data)
│   ├── mammography_observational.py   # Placeholder (awaiting Oslo registry data)
│   └── mammography_ai_experimental.py # Placeholder (awaiting MASAI data)
└── cross_domain/                 # NEW
    ├── __init__.py
    ├── validation_hierarchy.py
    └── complementarity_analysis.py
```

### Usage

```python
# Print the validation hierarchy
from cross_domain.validation_hierarchy import print_hierarchy_summary
print_hierarchy_summary()

# Run cross-domain complementarity comparison
from cross_domain.complementarity_analysis import run_cross_domain_comparison
run_cross_domain_comparison()

# Simulate hypothetical difficulty-stratified deployment (security)
from cross_domain.complementarity_analysis import (
    simulate_difficulty_stratified_deployment,
    ParameterStatus,
)
results = simulate_difficulty_stratified_deployment(
    base_rate=0.10,
    parameter_status=ParameterStatus.ASSUMED,  # security: no gold standard
)
print(results['difficulty_stratified']['sensitivity'])
print(results['CAVEAT'])  # Prominent caveat about assumed parameters
```

### What This Does NOT Do (and Why)

The R&U complementarity framework **cannot be properly applied** in security
domains because the validation problem is unsolvable there. This update does
NOT pretend otherwise. Instead, it:

1. **Formalizes the impossibility** — making the unsolvable validation problem
   an explicit, coded feature rather than just prose in a paper
2. **Simulates hypothetical ROC positions** under NAS-assumed parameters with
   prominent caveats, showing the *structure* of what difficulty-stratified
   deployment would predict
3. **Contrasts with colonoscopy** where the same analysis is empirically testable,
   making the epistemological asymmetry the methodological contribution

### Connection to MSCA Objectives

Note: Iterativity is now folded into O1–O3 as a structural feedback feature
of the model, not a standalone objective.

| Objective | Module | Status |
|-----------|--------|--------|
| O1 (Structural causal model + iterativity) | `security/screening_equilibrium.py` | Implemented (security); awaiting data (medical) |
| O2 (AI colonoscopy) | `medical/colonoscopy_deskilling.py` | Starter from published data |
| O3 (Mammography) | `medical/mammography_*.py` | Placeholder |
| O4 (Policy synthesis) | — | Requires O2+O3 completion |
| O5 (Cross-domain comparison) | `cross_domain/` | **NEW: validation hierarchy + complementarity** |

### References

- Besserve, M., & Schölkopf, B. (2022). Learning soft interventions in complex equilibrium systems. UAI 2022.
- Budzyń, K., et al. (2025). Endoscopist deskilling risk after exposure to AI in colonoscopy. Lancet Gastroenterol Hepatol.
- Kleinberg, J., et al. (2018). Human decisions and machine predictions. QJE.
- Ribers, M., & Ullrich, H. (2024). Complementarities between algorithmic and human decision-making. QME.
- Wilde, V. (2026). Structural causal modeling for mass screening programs. Working paper.
- Wilde, V. (2026). Equilibrium effects in AI-assisted screening. Working paper.
