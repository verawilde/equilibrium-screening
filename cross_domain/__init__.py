"""
Cross-domain analysis modules for the equilibrium screening framework.

These modules formalize the validation hierarchy across screening domains
and apply Ribers & Ullrich (2024) / Kleinberg et al. (2018) complementarity
logic where it is applicable (colonoscopy, HIV) and model hypothetical
complementarity where it is not (security, mammography).

The equilibrium framework (Besserve & Schölkopf 2022) applies everywhere.
The complementarity framework requires a solved validation problem.
This asymmetry is the methodological contribution of O5.

Modules:
    validation_hierarchy: Domain specifications, gold standard characterization,
        analytical tools available at each validation level. Includes FIT
        two-stage pipeline analysis for colonoscopy as secondary screening.
    complementarity_analysis: R&U/Kleinberg difficulty-stratified deployment
        logic applied across domains. Observed (colonoscopy) vs. hypothetical
        (security) ROC positions. Simulation and visualization.
"""

from cross_domain.validation_hierarchy import (
    ValidationLevel,
    ScreeningDomain,
    ScreeningStage,
    GoldStandardSpec,
    build_validation_hierarchy,
    print_hierarchy_summary,
)

from cross_domain.complementarity_analysis import (
    ParameterStatus,
    ROCPosition,
    DifficultyStratum,
    colonoscopy_complementarity,
    security_hypothetical_complementarity,
    simulate_difficulty_stratified_deployment,
    run_cross_domain_comparison,
)
