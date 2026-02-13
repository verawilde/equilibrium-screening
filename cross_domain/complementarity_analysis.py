"""
Complementarity Analysis Across Screening Domains

Applies the Ribers & Ullrich (2024) / Kleinberg et al. (2018) complementarity
logic across the validation hierarchy:

- Where the validation problem IS solved (colonoscopy, HIV): 
  observed ROC positions, empirically testable complementarity
- Where it is NOT solved (security, mammography): 
  hypothetical ROC positions under stated assumptions, with prominent caveats

The contrast itself is analytically informative: "In security, we can model
what the ROC space WOULD look like under assumed parameters. In colonoscopy,
we can OBSERVE it. This asymmetry is the point."

Extends:
    Ribers & Ullrich (2024). Complementarities between algorithmic and human
        decision-making: The case of antibiotic prescribing. QME.
    Kleinberg et al. (2018). Human decisions and machine predictions. QJE.
    Wilde (2026). Equilibrium effects in AI-assisted screening. Working paper.

Builds on:
    Besserve & Schölkopf (2022). Learning soft interventions in complex
        equilibrium systems. UAI 2022.

Author: Vera Wilde
Date: February 2026
"""

# IMPORTANT LIMITATION: Colonoscopy complementarity analysis operates on the
# CONDITIONAL confusion matrix (detected-and-removed lesions only). Unlike
# Ribers & Ullrich's UTI case where lab culture validated every patient's
# classification regardless of the physician's decision (unconditional 2x2
# table), histopathology in colonoscopy validates only removed lesions.
# False negatives (missed lesions) must be estimated from ACCEPT's
# randomization (AI-on vs. AI-off detection rate comparisons) or from
# tandem colonoscopy miss rate literature (22-27%, Zhao et al. 2019).
# The complementarity analysis is therefore more uncertain than in the
# UTI case, particularly for undetected lesions.

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from cross_domain.validation_hierarchy import ValidationLevel


# ──────────────────────────────────────────────────────────────────────
# 1. ROC SPACE REPRESENTATION
# ──────────────────────────────────────────────────────────────────────

class ParameterStatus(Enum):
    """Whether a parameter is observed from data or assumed from literature."""
    OBSERVED = "observed"         # From gold-standard-validated data
    ESTIMATED = "estimated"       # Statistically estimated with uncertainty
    ASSUMED = "assumed"           # From literature ranges, no direct observation
    HYPOTHETICAL = "hypothetical" # Illustrative, no empirical basis


@dataclass
class ROCPosition:
    """
    A decision-maker's position in the ROC space.
    
    In R&U-tractable domains (colonoscopy, HIV), these are OBSERVED
    from gold-standard-validated data.
    
    In validation-unsolved domains (security), these are ASSUMED from
    literature ranges — and the distinction matters fundamentally.
    """
    false_positive_rate: float  # x-axis in ROC space
    true_positive_rate: float   # y-axis in ROC space
    
    # Critically: is this observed or assumed?
    status: ParameterStatus
    
    # Uncertainty
    fpr_ci: Optional[Tuple[float, float]] = None  # 95% CI
    tpr_ci: Optional[Tuple[float, float]] = None
    
    # Metadata
    label: str = ""
    source: str = ""
    caveats: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        assert 0 <= self.false_positive_rate <= 1
        assert 0 <= self.true_positive_rate <= 1


@dataclass 
class DifficultyStratum:
    """
    A case-difficulty stratum for complementarity analysis.
    
    R&U (2024) key finding: AI outperformed physicians on easy and hard cases;
    physicians outperformed on intermediate-difficulty cases (>50% of decisions).
    
    The logic applies across domains, but:
    - In colonoscopy: difficulty proxied by lesion size, morphology, location
      (directly observable in ACCEPT data)
    - In security: difficulty is conceptual (naive vs sophisticated deception)
      but not empirically stratifiable without a gold standard
    """
    name: str
    description: str
    difficulty_level: str  # "easy", "intermediate", "hard"
    
    # Who performs better at this difficulty level?
    # R&U prediction: AI for easy+hard, human for intermediate
    human_position: Optional[ROCPosition] = None
    ai_position: Optional[ROCPosition] = None
    
    # Is this empirically testable in this domain?
    empirically_testable: bool = True
    
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────
# 2. DOMAIN-SPECIFIC COMPLEMENTARITY SPECIFICATIONS
# ──────────────────────────────────────────────────────────────────────

def colonoscopy_complementarity() -> Dict:
    """
    Colonoscopy: OBSERVED complementarity (with ACCEPT data).
    
    Histopathology provides the answer sheet:
    - TP = conventional adenoma confirmed by histopathology
    - FP = hyperplastic (non-neoplastic) polyp resected unnecessarily
    
    This is the first context where BOTH requirements are jointly met:
    (a) answer sheet exists (→ complementarity analysis applicable)
    (b) deskilling demonstrates feedback dynamics (→ equilibrium modeling applicable)
    
    NB: Colonoscopy is a SECONDARY screening. The population entering ACCEPT's
    randomization has been pre-selected by FIT with varying national thresholds.
    The complementarity structure at colonoscopy is CONDITIONAL on FIT selection,
    meaning cross-site variation may reflect differential pre-selection as much
    as genuine population-level heterogeneity.
    
    DIFFICULTY STRATA (from ACCEPT lesion characteristics):
    - Easy: Large pedunculated polyps (obvious neoplastic features)
    - Intermediate: Small flat adenomas, ambiguous mucosal patterns
    - Hard: Diminutive lesions in difficult anatomical locations
    
    PREDICTION (from working paper Section 7, Prediction 5):
    Difficulty-stratified AI deployment outperforms uniform deployment,
    observable in the ROC space via histopathology validation.
    """
    strata = [
        DifficultyStratum(
            name="Easy cases",
            description="Large pedunculated polyps with obvious neoplastic features",
            difficulty_level="easy",
            empirically_testable=True,
            notes=(
                "R&U prediction: AI outperforms endoscopists here. "
                "ACCEPT data: lesion size ≥10mm, pedunculated morphology. "
                "Histopathology validates classification."
            )
        ),
        DifficultyStratum(
            name="Intermediate cases",
            description="Small flat adenomas, ambiguous mucosal patterns",
            difficulty_level="intermediate",
            empirically_testable=True,
            notes=(
                "R&U prediction: endoscopists outperform AI here. "
                "ACCEPT data: lesion size 6-9mm, sessile morphology. "
                "This is where human judgment adds value and where "
                "deskilling is most costly."
            )
        ),
        DifficultyStratum(
            name="Hard cases",
            description="Diminutive lesions in difficult anatomical locations",
            difficulty_level="hard",
            empirically_testable=True,
            notes=(
                "R&U prediction: AI outperforms endoscopists here. "
                "ACCEPT data: lesion size ≤5mm, flat morphology, "
                "right colon / sigmoid colon location. "
                "Miss rate literature identifies these as consistently "
                "high-miss-rate characteristics (Zhao et al. 2019)."
            )
        ),
    ]
    
    return {
        "domain": "AI-assisted colonoscopy",
        "validation_level": ValidationLevel.LARGELY_SOLVED,
        "gold_standard": "Histopathology (adenoma vs hyperplastic)",
        "parameter_status": ParameterStatus.OBSERVED,  # With ACCEPT data
        "strata": strata,
        "complementarity_applicable": True,
        "two_stage_note": (
            "CRITICAL: Colonoscopy is a secondary screening, preceded by FIT. "
            "The complementarity structure estimated at colonoscopy is conditional "
            "on FIT pre-selection. Cross-site variation in ACCEPT may reflect "
            "differences in FIT threshold (15-150 μg/g) — itself a product of "
            "the strategy pathway at the system-design level — as much as genuine "
            "population-level variation in disease and morphology."
        ),
        "fit_complementarity_note": (
            "UNEXPLOITED EXTENSION: The FIT stage itself has a largely-solved "
            "validation problem (colonoscopy reveals true status for FIT-positives; "
            "cancer registries track interval cancers for FIT-negatives). "
            "AI-human complementarity at FIT threshold-setting or interpretation "
            "is a natural extension no one has explored."
        ),
        "testable_predictions": [
            "P5a: Endoscopists using difficulty-stratified AI show less deskilling "
            "on intermediate-difficulty cases than those using uniform AI assistance",
            "P5b: Non-neoplastic resection rates (FP) decrease under difficulty-"
            "stratified protocols vs uniform AI",
            "P5c: ROC positions show improving (northwest movement) on easy and "
            "hard cases under AI, while remaining stable on intermediate cases "
            "under human judgment",
        ],
    }


def security_hypothetical_complementarity() -> Dict:
    """
    Security: HYPOTHETICAL complementarity under stated assumptions.
    
    The validation problem is unsolvable — no gold standard for deception.
    We CANNOT observe polygraph examiners or Chat Control classifiers in
    the ROC space in the R&U sense.
    
    What we CAN do: model what the ROC space WOULD look like under the
    NAS (2003) assumed parameter ranges, and show what difficulty-stratified
    deployment WOULD predict — making the contrast with colonoscopy
    (where we can observe it) a teaching tool about what the solved
    validation problem buys you.
    
    DIFFICULTY STRATA (conceptual, NOT empirically stratifiable):
    - Easy: Naive deception, obvious indicators
    - Intermediate: Ordinary concealment, ambiguous signals
    - Hard: Sophisticated countermeasures, trained subjects
    
    For Chat Control:
    - Easy: Known-hash matching (trivially validated, but not the policy case)
    - Intermediate: Typical CSAM-adjacent content (high FP risk)
    - Hard: Novel, sophisticated CSAM (where detection is needed most)
    """
    polygraph_strata = [
        DifficultyStratum(
            name="Naive deception (easy for detector)",
            description="Untrained subjects, strong autonomic responses",
            difficulty_level="easy",
            human_position=ROCPosition(
                false_positive_rate=0.15,
                true_positive_rate=0.85,
                status=ParameterStatus.ASSUMED,
                fpr_ci=(0.10, 0.30),
                tpr_ci=(0.70, 0.95),
                label="Examiner (naive subjects)",
                source="NAS 2003, upper range",
                caveats=[
                    "ASSUMED, not observed — no gold standard for deception",
                    "NAS ranges are themselves uncertain",
                    "Laboratory vs field conditions differ substantially"
                ]
            ),
            empirically_testable=False,
            notes=(
                "Even in the 'easy' case, the examiner's ROC position is "
                "ASSUMED from NAS literature ranges. We cannot validate this "
                "because we cannot determine ground truth for deception."
            )
        ),
        DifficultyStratum(
            name="Ordinary concealment (intermediate)",
            description="Typical applicant concealing disqualifying information",
            difficulty_level="intermediate",
            human_position=ROCPosition(
                false_positive_rate=0.25,
                true_positive_rate=0.75,
                status=ParameterStatus.ASSUMED,
                fpr_ci=(0.15, 0.40),
                tpr_ci=(0.60, 0.85),
                label="Examiner (typical applicants)",
                source="NAS 2003, mid range",
                caveats=[
                    "ASSUMED — no gold standard",
                    "Most relevant case for pre-employment screening",
                    "Information pathway (bogus pipeline) may be more important "
                    "than classification accuracy here"
                ]
            ),
            empirically_testable=False,
            notes=(
                "R&U logic WOULD predict human superiority here — the examiner "
                "integrates contextual cues, behavioral observations, interview "
                "dynamics. But we cannot test this prediction because we cannot "
                "observe the answer sheet."
            )
        ),
        DifficultyStratum(
            name="Trained countermeasures (hard for detector)",
            description="Subjects trained in countermeasure techniques",
            difficulty_level="hard",
            human_position=ROCPosition(
                false_positive_rate=0.35,
                true_positive_rate=0.55,
                status=ParameterStatus.ASSUMED,
                fpr_ci=(0.20, 0.50),
                tpr_ci=(0.40, 0.70),
                label="Examiner (countermeasure-trained)",
                source="NAS 2003, lower range; countermeasure literature",
                caveats=[
                    "ASSUMED — no gold standard",
                    "Countermeasure effectiveness itself contested",
                    "Waid, Orne & Wilson 1979: deceptive subjects with lower "
                    "socialization scores more likely to pass"
                ]
            ),
            empirically_testable=False,
            notes=(
                "For sophisticated deception, the polygraph approaches random "
                "classification — the entire program's value (if any) comes "
                "from Strategy and Information pathways, not Classification."
            )
        ),
    ]
    
    chat_control_strata = [
        DifficultyStratum(
            name="Known-hash matching (trivially easy)",
            description="Previously identified CSAM, hash in database",
            difficulty_level="easy",
            empirically_testable=True,  # Hash matching IS verifiable
            notes=(
                "Hash matching (PhotoDNA etc) has high accuracy for KNOWN "
                "material. But this is not the policy-relevant case — it "
                "only finds previously-identified CSAM. The novel detection "
                "case is where the policy need lies."
            )
        ),
        DifficultyStratum(
            name="CSAM-adjacent content (high FP risk)",
            description="Ambiguous content: family photos, medical images, art",
            difficulty_level="intermediate",
            empirically_testable=False,
            notes=(
                "This is where false positives overwhelm: billions of "
                "messages containing child photos, medical content, etc. "
                "No AI classifier can reliably distinguish in this zone, "
                "and no R&U-style difficulty stratification can help because "
                "there is no gold standard for intent/context at scale."
            )
        ),
        DifficultyStratum(
            name="Novel, sophisticated CSAM (hard)",
            description="New material, possibly encrypted/steganographic",
            difficulty_level="hard",
            empirically_testable=False,
            notes=(
                "Where detection is most needed, the validation problem is "
                "most severe. Sophisticated actors migrate to encrypted "
                "channels — strategy pathway dominates, and the classification "
                "pathway cannot reach them regardless of AI accuracy."
            )
        ),
    ]
    
    return {
        "domain": "Security screening (polygraph, Chat Control, iBorderCtrl)",
        "validation_level": ValidationLevel.UNSOLVABLE,
        "gold_standard": "None (no gold standard for deception/threat status)",
        "parameter_status": ParameterStatus.ASSUMED,
        "polygraph_strata": polygraph_strata,
        "chat_control_strata": chat_control_strata,
        "complementarity_applicable": False,
        "what_we_can_do_instead": (
            "Model HYPOTHETICAL complementarity under stated assumptions from "
            "NAS (2003) parameter ranges. The contrast with colonoscopy — where "
            "we can observe the ROC space via histopathology — demonstrates what "
            "the solved validation problem buys you. In security, we show what "
            "the ROC space WOULD look like. In colonoscopy, we OBSERVE it."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# 3. HYPOTHETICAL ROC SIMULATION (SECURITY DOMAINS)
# ──────────────────────────────────────────────────────────────────────

def simulate_hypothetical_roc(
    sensitivity_range: Tuple[float, float] = (0.50, 0.90),
    specificity_range: Tuple[float, float] = (0.50, 0.80),
    n_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate hypothetical ROC positions for a decision-maker with
    ASSUMED (not observed) classification parameters.
    
    Used for security domains where the validation problem is unsolvable.
    Each sample represents one possible true operating point, drawn
    uniformly from the NAS-reported parameter ranges.
    
    Returns array of shape (n_samples, 2): [FPR, TPR] pairs.
    
    CAVEAT: These are NOT observed positions. They represent the SPACE
    of positions compatible with assumed parameter ranges. The actual
    position is unknown and unknowable without a gold standard.
    """
    rng = np.random.default_rng(seed)
    
    sensitivities = rng.uniform(sensitivity_range[0], sensitivity_range[1], n_samples)
    specificities = rng.uniform(specificity_range[0], specificity_range[1], n_samples)
    
    # FPR = 1 - specificity; TPR = sensitivity
    fpr = 1 - specificities
    tpr = sensitivities
    
    return np.column_stack([fpr, tpr])


def simulate_difficulty_stratified_deployment(
    base_rate: float = 0.10,
    n_cases: int = 10000,
    seed: int = 42,
    # Assumed parameters (security: from NAS; colonoscopy: from ACCEPT once available)
    human_sens_by_difficulty: Dict[str, Tuple[float, float]] = None,
    ai_sens_by_difficulty: Dict[str, Tuple[float, float]] = None,
    parameter_status: ParameterStatus = ParameterStatus.ASSUMED,
) -> Dict:
    """
    Simulate R&U-style difficulty-stratified deployment under stated assumptions.
    
    For security: parameters are ASSUMED from literature (NAS 2003).
    For colonoscopy: parameters would be OBSERVED from ACCEPT histopathology.
    
    The simulation structure is identical — the epistemological status differs.
    
    Returns dict with performance metrics for:
    - Uniform AI deployment
    - Uniform human deployment
    - Difficulty-stratified deployment (R&U logic)
    """
    if human_sens_by_difficulty is None:
        # Default: NAS-range polygraph parameters (ASSUMED)
        human_sens_by_difficulty = {
            "easy": (0.85, 0.85),      # (sensitivity, specificity)
            "intermediate": (0.75, 0.70),
            "hard": (0.55, 0.60),
        }
    if ai_sens_by_difficulty is None:
        # Hypothetical AI parameters for illustration
        ai_sens_by_difficulty = {
            "easy": (0.92, 0.90),
            "intermediate": (0.65, 0.75),
            "hard": (0.70, 0.65),
        }
    
    rng = np.random.default_rng(seed)
    
    # Generate cases with difficulty levels
    # Roughly: 30% easy, 50% intermediate, 20% hard (following R&U proportions)
    difficulty_proportions = {"easy": 0.30, "intermediate": 0.50, "hard": 0.20}
    
    results = {
        "parameter_status": parameter_status.value,
        "base_rate": base_rate,
        "n_cases": n_cases,
    }
    
    for strategy_name, strategy_rule in [
        ("uniform_human", lambda d: "human"),
        ("uniform_ai", lambda d: "ai"),
        ("difficulty_stratified", lambda d: "ai" if d in ["easy", "hard"] else "human"),
    ]:
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        
        for difficulty, proportion in difficulty_proportions.items():
            n_stratum = int(n_cases * proportion)
            n_positive = int(n_stratum * base_rate)
            n_negative = n_stratum - n_positive
            
            # Who decides in this stratum under this strategy?
            decider = strategy_rule(difficulty)
            if decider == "human":
                sens, spec = human_sens_by_difficulty[difficulty]
            else:
                sens, spec = ai_sens_by_difficulty[difficulty]
            
            tp = int(n_positive * sens)
            fn = n_positive - tp
            tn = int(n_negative * spec)
            fp = n_negative - tn
            
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
        
        total_screened = total_tp + total_fp + total_tn + total_fn
        results[strategy_name] = {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "true_negatives": total_tn,
            "false_negatives": total_fn,
            "sensitivity": total_tp / max(total_tp + total_fn, 1),
            "specificity": total_tn / max(total_tn + total_fp, 1),
            "ppv": total_tp / max(total_tp + total_fp, 1),
            "fpr": total_fp / max(total_fp + total_tn, 1),
        }
    
    # Add caveat
    if parameter_status == ParameterStatus.ASSUMED:
        results["CAVEAT"] = (
            "ALL parameters in this simulation are ASSUMED from literature ranges, "
            "NOT observed from gold-standard-validated data. The validation problem "
            "is unsolvable in this domain. These results illustrate the STRUCTURE "
            "of difficulty-stratified deployment, not its empirical magnitude. "
            "For observed complementarity, see colonoscopy analysis with ACCEPT "
            "histopathology data."
        )
    elif parameter_status == ParameterStatus.OBSERVED:
        results["NOTE"] = (
            "Parameters observed from gold-standard-validated data. "
            "NB: If colonoscopy data, remember this is a SECONDARY screening — "
            "the population was pre-selected by FIT, and cross-site variation "
            "may reflect differential FIT threshold selection."
        )
    
    return results


# ──────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION
# ──────────────────────────────────────────────────────────────────────

def plot_roc_comparison(
    observed_positions: Optional[List[ROCPosition]] = None,
    assumed_positions: Optional[List[ROCPosition]] = None,
    hypothetical_cloud: Optional[np.ndarray] = None,
    title: str = "ROC Space: Observed vs. Hypothetical Decision-Maker Positions",
    save_path: Optional[str] = None,
):
    """
    Plot observed (colonoscopy) vs assumed (security) ROC positions.
    
    The visual contrast makes the epistemological point:
    - Observed positions: solid markers with tight CIs
    - Assumed positions: hollow markers with wide uncertainty clouds
    - Hypothetical cloud: shaded region of possible positions
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random classifier')
    
    # Hypothetical cloud (assumed parameter ranges)
    if hypothetical_cloud is not None:
        ax.scatter(
            hypothetical_cloud[:, 0], hypothetical_cloud[:, 1],
            alpha=0.05, c='red', s=10,
            label='Hypothetical positions (ASSUMED)'
        )
    
    # Assumed positions (security: hollow markers, wide CIs)
    if assumed_positions:
        for pos in assumed_positions:
            ax.scatter(
                pos.false_positive_rate, pos.true_positive_rate,
                marker='o', facecolors='none', edgecolors='red',
                s=100, linewidths=2, zorder=5
            )
            if pos.fpr_ci and pos.tpr_ci:
                ax.errorbar(
                    pos.false_positive_rate, pos.true_positive_rate,
                    xerr=[[pos.false_positive_rate - pos.fpr_ci[0]],
                           [pos.fpr_ci[1] - pos.false_positive_rate]],
                    yerr=[[pos.true_positive_rate - pos.tpr_ci[0]],
                           [pos.tpr_ci[1] - pos.true_positive_rate]],
                    fmt='none', ecolor='red', alpha=0.5, capsize=3
                )
            if pos.label:
                ax.annotate(
                    f'{pos.label}\n(ASSUMED)',
                    (pos.false_positive_rate + 0.02, pos.true_positive_rate - 0.02),
                    fontsize=7, color='red', fontstyle='italic'
                )
    
    # Observed positions (colonoscopy: solid markers, tighter CIs)
    if observed_positions:
        for pos in observed_positions:
            ax.scatter(
                pos.false_positive_rate, pos.true_positive_rate,
                marker='s', c='blue', s=100, zorder=5
            )
            if pos.fpr_ci and pos.tpr_ci:
                ax.errorbar(
                    pos.false_positive_rate, pos.true_positive_rate,
                    xerr=[[pos.false_positive_rate - pos.fpr_ci[0]],
                           [pos.fpr_ci[1] - pos.false_positive_rate]],
                    yerr=[[pos.true_positive_rate - pos.tpr_ci[0]],
                           [pos.tpr_ci[1] - pos.true_positive_rate]],
                    fmt='none', ecolor='blue', alpha=0.7, capsize=3
                )
            if pos.label:
                ax.annotate(
                    f'{pos.label}\n(OBSERVED)',
                    (pos.false_positive_rate + 0.02, pos.true_positive_rate + 0.02),
                    fontsize=7, color='blue', fontweight='bold'
                )
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    
    # Add epistemological note
    ax.text(
        0.02, 0.02,
        'Blue (solid) = OBSERVED via gold standard\n'
        'Red (hollow) = ASSUMED from literature ranges\n'
        'Shaded = space of possible positions given assumed ranges',
        transform=ax.transAxes, fontsize=7,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────
# 5. CROSS-DOMAIN COMPARISON RUNNER
# ──────────────────────────────────────────────────────────────────────

def run_cross_domain_comparison():
    """
    Run and print the cross-domain complementarity comparison.
    
    Demonstrates:
    1. In colonoscopy: empirically testable complementarity predictions
       (awaiting ACCEPT individual-level data)
    2. In security: hypothetical complementarity under stated assumptions
       (NAS parameter ranges)
    3. The asymmetry is the methodological contribution
    """
    print("=" * 75)
    print("CROSS-DOMAIN COMPLEMENTARITY ANALYSIS")
    print("Ribers & Ullrich (2024) / Kleinberg et al. (2018) logic")
    print("applied across the validation hierarchy")
    print("=" * 75)
    
    # Colonoscopy: what we CAN do
    colo = colonoscopy_complementarity()
    print(f"\n{'─' * 75}")
    print(f"DOMAIN: {colo['domain']}")
    print(f"Validation level: {colo['validation_level'].value}")
    print(f"Parameters: {colo['parameter_status'].value}")
    print(f"R&U complementarity applicable: {colo['complementarity_applicable']}")
    print(f"\n  Two-stage note: {colo['two_stage_note']}")
    print(f"\n  FIT extension: {colo['fit_complementarity_note']}")
    print(f"\n  Testable predictions:")
    for pred in colo['testable_predictions']:
        print(f"    • {pred}")
    
    # Security: what we CANNOT do (but can model hypothetically)
    sec = security_hypothetical_complementarity()
    print(f"\n{'─' * 75}")
    print(f"DOMAIN: {sec['domain']}")
    print(f"Validation level: {sec['validation_level'].value}")
    print(f"Parameters: {sec['parameter_status'].value}")
    print(f"R&U complementarity applicable: {sec['complementarity_applicable']}")
    print(f"\n  What we can do instead: {sec['what_we_can_do_instead']}")
    
    # Hypothetical simulation for security
    print(f"\n{'─' * 75}")
    print("HYPOTHETICAL DIFFICULTY-STRATIFIED DEPLOYMENT (Security)")
    print("Parameters ASSUMED from NAS (2003)")
    sec_results = simulate_difficulty_stratified_deployment(
        base_rate=0.10,  # Assumed problematic applicant rate
        parameter_status=ParameterStatus.ASSUMED,
    )
    
    for strategy in ["uniform_human", "uniform_ai", "difficulty_stratified"]:
        r = sec_results[strategy]
        print(f"\n  {strategy.upper()}:")
        print(f"    Sensitivity: {r['sensitivity']:.3f}")
        print(f"    Specificity: {r['specificity']:.3f}")
        print(f"    PPV:         {r['ppv']:.3f}")
        print(f"    FPR:         {r['fpr']:.3f}")
    
    print(f"\n  CAVEAT: {sec_results['CAVEAT']}")
    
    # The asymmetry
    print(f"\n{'═' * 75}")
    print("THE ASYMMETRY (methodological contribution):")
    print("  In SECURITY: we model what the ROC space WOULD look like")
    print("    under assumed parameters from NAS (2003) ranges.")
    print("  In COLONOSCOPY: we can OBSERVE it via histopathology,")
    print("    including at the upstream FIT stage via registry linkage.")
    print("  This asymmetry is the point.")
    print("═" * 75)


if __name__ == "__main__":
    run_cross_domain_comparison()
