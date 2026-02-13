"""
Validation Hierarchy Across Screening Domains

Formalizes where each screening domain sits on the validation problem's
solvability spectrum, and what analytical tools are available at each level.

This is the conceptual backbone of MSCA Objective O5 (cross-domain comparison).
The validation problem — the challenge of establishing ground truth to evaluate
screening decisions — varies systematically across domains:

    FULLY SOLVED:    HIV testing (confirmatory serology)
    LARGELY SOLVED:  Colonoscopy (histopathology resolves adenoma vs. hyperplastic)
    PERSISTENTLY OPEN: Mammography (DCIS ambiguity contaminates gold standard)
    UNSOLVABLE:      Security (no gold standard for deception/threat status)

This variation determines which analytical tools from the complementarity
literature (Ribers & Ullrich 2024; Kleinberg et al. 2018) can be applied.
The equilibrium framework (Besserve & Schölkopf 2022, extended in Wilde 2026)
applies across ALL domains — it models system-level feedback dynamics that
operate regardless of gold standard availability. But the empirical precision
achievable differs fundamentally by domain.

Key insight for colonoscopy: It occupies a uniquely tractable position because
(a) it is itself a SECONDARY screening — preceded by FIT in most European
programs — creating a two-stage validation structure where BOTH stages have
(at least partially) solvable validation problems, and (b) histopathology
provides the answer sheet that complementarity analysis demands, while the
deskilling finding (Budzyń et al. 2025) demonstrates the feedback dynamics
that equilibrium modeling captures. No other screening context currently
satisfies both requirements simultaneously.

References:
    Besserve & Schölkopf (2022). Learning soft interventions in complex
        equilibrium systems. UAI 2022.
    Budzyń et al. (2025). Endoscopist deskilling risk after exposure to AI
        in colonoscopy. Lancet Gastroenterol Hepatol.
    Kleinberg et al. (2018). Human decisions and machine predictions. QJE.
    Ribers & Ullrich (2024). Complementarities between algorithmic and human
        decision-making: The case of antibiotic prescribing. QME.
    Wilde (2026). Structural causal modeling for mass screening programs.
        Working paper.

Author: Vera Wilde
Date: February 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict


# ──────────────────────────────────────────────────────────────────────
# 1. VALIDATION SOLVABILITY LEVELS
# ──────────────────────────────────────────────────────────────────────

class ValidationLevel(Enum):
    """
    Where a screening domain sits on the validation hierarchy.
    Determines which analytical tools can be applied.
    """
    FULLY_SOLVED = "fully_solved"
    LARGELY_SOLVED = "largely_solved"
    PERSISTENTLY_OPEN = "persistently_open"
    UNSOLVABLE = "unsolvable"


class AnalyticalCapability(Enum):
    """What a given validation level enables."""
    FULL_ROC_OBSERVATION = "full_roc_observation"
    PARTIAL_ROC_OBSERVATION = "partial_roc_observation"
    AGGREGATE_METRICS_ONLY = "aggregate_metrics_only"
    ASSUMED_PARAMETERS_ONLY = "assumed_parameters_only"


# ──────────────────────────────────────────────────────────────────────
# 2. SCREENING DOMAIN SPECIFICATIONS
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GoldStandardSpec:
    """Characterizes the gold standard (or lack thereof) for a domain."""
    name: str
    description: str
    resolves_classification: bool  # Can it answer: was this a TP or FP?
    resolves_clinical_importance: bool  # Can it answer: did this matter?
    contamination_source: Optional[str] = None  # What contaminates it, if anything
    delay_to_result: Optional[str] = None  # How long until gold standard available


@dataclass
class ScreeningStage:
    """
    A single stage in a possibly multi-stage screening pipeline.
    
    Key insight: Colonoscopy in European CRC screening is a SECONDARY screening,
    preceded by FIT. The population entering colonoscopy has already been
    pre-filtered by FIT — a selection mechanism that itself operates with
    variable thresholds reflecting the strategy pathway at the system-design
    level (Porter 1995; Greenland 2023; Welch et al. 2011).
    """
    name: str
    gold_standard: GoldStandardSpec
    validation_level: ValidationLevel
    
    # What the validation level enables
    roc_observable: bool  # Can we place decision-makers in ROC space?
    complementarity_applicable: bool  # Can we apply R&U/Kleinberg framework?
    equilibrium_applicable: bool  # Can we apply B&S equilibrium framework?
    # (Always True — equilibrium effects operate regardless of gold standard)
    
    notes: str = ""


@dataclass
class ScreeningDomain:
    """
    Complete specification of a screening domain's validation structure.
    Multi-stage pipelines (e.g., FIT → colonoscopy) are represented as
    ordered lists of ScreeningStages.
    """
    name: str
    stages: List[ScreeningStage]
    population_scale: str  # e.g., "thousands/year", "billions/day"
    base_rate_range: str  # e.g., "0.001-0.01"
    
    # Which four-pathway components dominate
    dominant_pathways: List[str]
    
    # What parameter sources exist
    parameter_sources: List[str]
    
    # Whether classification parameters are observed or assumed
    classification_params_observed: bool
    
    notes: str = ""
    
    @property
    def overall_validation_level(self) -> ValidationLevel:
        """
        The domain's analytical tractability is limited by its weakest stage.
        But for complementarity analysis, only the relevant stage matters.
        """
        levels_ordered = [
            ValidationLevel.FULLY_SOLVED,
            ValidationLevel.LARGELY_SOLVED,
            ValidationLevel.PERSISTENTLY_OPEN,
            ValidationLevel.UNSOLVABLE
        ]
        worst = ValidationLevel.FULLY_SOLVED
        for stage in self.stages:
            if levels_ordered.index(stage.validation_level) > levels_ordered.index(worst):
                worst = stage.validation_level
        return worst
    
    @property
    def any_stage_supports_complementarity(self) -> bool:
        """Whether ANY stage supports R&U-style complementarity analysis."""
        return any(s.complementarity_applicable for s in self.stages)


# ──────────────────────────────────────────────────────────────────────
# 3. DOMAIN DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

def hiv_testing() -> ScreeningDomain:
    """
    HIV testing: the canonical FULLY SOLVED case.
    Confirmatory serology definitively establishes infection status.
    Full 2x2 classification table directly observable.
    """
    return ScreeningDomain(
        name="HIV Testing",
        stages=[
            ScreeningStage(
                name="Antibody/antigen screening",
                gold_standard=GoldStandardSpec(
                    name="Confirmatory serology (Western blot / NAT)",
                    description="Definitive laboratory confirmation of HIV status",
                    resolves_classification=True,
                    resolves_clinical_importance=True,
                    delay_to_result="days to weeks"
                ),
                validation_level=ValidationLevel.FULLY_SOLVED,
                roc_observable=True,
                complementarity_applicable=True,
                equilibrium_applicable=True,
                notes="Full 2x2 table directly observable for every screened individual"
            )
        ],
        population_scale="millions/year (e.g., prenatal, blood supply)",
        base_rate_range="0.001-0.05 depending on population",
        dominant_pathways=["classification", "information", "resource_reallocation"],
        parameter_sources=["Confirmatory lab results", "Population registries"],
        classification_params_observed=True,
        notes=(
            "The cleanest case for complementarity analysis. "
            "Relatively rare example where the validation problem is fully "
            "solved at the classification level AND the clinical importance level."
        )
    )


def colorectal_cancer_screening() -> ScreeningDomain:
    """
    CRC screening via FIT → colonoscopy pipeline.
    
    CRITICAL: Colonoscopy is a SECONDARY screening. In most European programs,
    the population entering colonoscopy has been pre-selected by FIT (fecal
    immunochemical test) — a continuous biomarker dichotomized at a positivity
    threshold that varies dramatically across national programs (Norway: 15 μg/g;
    other European countries: 20-150 μg/g).
    
    This creates a screening-into-screening selection structure:
    - FIT selects who enters the colonoscopy population
    - Colonoscopy (with or without AI) classifies what is found
    - The populations entering ACCEPT's colonoscopy randomization have already
      been pre-filtered differently by country
    
    The FIT threshold is not a neutral technical optimization. It is already a
    product of the strategy pathway operating at the system-design level, where
    the strategic actors are not patients but expert committees and clinical
    institutions (Porter 1995; Greenland 2023; Welch et al. 2011).
    
    To our knowledge, no one has framed FIT screening accuracy as a signal
    detection problem amenable to R&U-style complementarity analysis. The full
    2x2 table CAN be constructed using Scandinavian population registries:
    FIT-positive patients receive colonoscopy (revealing true status), while
    FIT-negative patients can be followed through cancer registries for interval
    cancers (revealing false negatives).
    """
    return ScreeningDomain(
        name="Colorectal Cancer Screening (FIT → Colonoscopy)",
        stages=[
            # STAGE 1: FIT
            ScreeningStage(
                name="FIT (fecal immunochemical test)",
                gold_standard=GoldStandardSpec(
                    name="Colonoscopy findings + cancer registry follow-up",
                    description=(
                        "FIT-positive patients receive colonoscopy (revealing true "
                        "status). FIT-negative patients followed through national "
                        "cancer registries for interval cancers (revealing FN). "
                        "Norway, Sweden, Denmark have registry infrastructure for "
                        "this linkage."
                    ),
                    resolves_classification=True,
                    resolves_clinical_importance=False,  # Not all found adenomas progress
                    delay_to_result="weeks (colonoscopy) to years (registry follow-up)"
                ),
                validation_level=ValidationLevel.LARGELY_SOLVED,
                roc_observable=True,
                complementarity_applicable=True,  # UNEXPLOITED
                equilibrium_applicable=True,
                notes=(
                    "UNEXPLOITED OPPORTUNITY: No one has framed FIT threshold-setting "
                    "as a signal detection problem amenable to R&U-style complementarity "
                    "analysis. Could AI-human complementarity at the FIT interpretation "
                    "stage reduce false positives (unnecessary colonoscopies) while "
                    "holding false negatives (missed cancers) constant? "
                    "NB: FIT thresholds vary by country (15-150 μg/g), reflecting "
                    "strategy pathway dynamics at the system-design level — expert "
                    "committees subject to cognitive biases and perverse incentives "
                    "(clinicians who perform colonoscopies benefit from lower thresholds "
                    "generating more referrals). See Welch, Schwartz & Woloshin 2011."
                )
            ),
            # STAGE 2: Colonoscopy
            ScreeningStage(
                name="Colonoscopy (with or without AI assistance)",
                gold_standard=GoldStandardSpec(
                    name="Histopathology",
                    description=(
                        "Definitive classification of removed lesions: "
                        "conventional adenoma (TP), hyperplastic polyp (FP). "
                        "Sessile serrated lesions — evolving classification, "
                        "addressed in sensitivity analyses."
                    ),
                    resolves_classification=True,
                    resolves_clinical_importance=False,
                    contamination_source=(
                        "Sessile serrated lesion classification is evolving. "
                        "Not all adenomas progress to cancer. But the "
                        "CLASSIFICATION question (adenoma vs hyperplastic?) is "
                        "definitively resolved — unlike mammography."
                    ),
                    delay_to_result="days (histopathology turnaround)"
                ),
                validation_level=ValidationLevel.LARGELY_SOLVED,
                roc_observable=True,
                complementarity_applicable=True,
                equilibrium_applicable=True,
                notes=(
                    "Histopathology parallels Ribers & Ullrich's UTI lab result: "
                    "decision point (endoscopist assesses lesion) → gold standard "
                    "(histopathology) returns days later. The answer sheet exists for "
                    "every removed polyp. Spadaccini et al. (2025) found AI increases "
                    "non-neoplastic resections (RR 1.39, 95% CI 1.23-1.57), making "
                    "the FP side of the ROC space directly observable. ACCEPT's "
                    "randomization provides statistical solution to the missing cell "
                    "(lesions never detected)."
                )
            )
        ],
        population_scale="millions/year across European programs",
        base_rate_range="0.01-0.05 (adenomas in screened population)",
        dominant_pathways=[
            "classification",  # AI ADR improvement: RR 1.22 (Soleymanjahi 2024)
            "strategy",  # Deskilling: -6.0pp (Budzyń 2025)
            "resource_reallocation"  # More surveillance colonoscopies from AI-detected adenomas
        ],
        parameter_sources=[
            "ACCEPT trial (individual-level, 6 countries, 2 continents)",
            "NordICC trial (colonoscopy vs usual care)",
            "Budzyń et al. 2025 (deskilling: published aggregate data)",
            "Soleymanjahi et al. 2024 meta-analysis (44 RCTs)",
            "Spadaccini et al. 2025 (non-neoplastic resections)",
            "National cancer registries (Norway, Sweden, Denmark)",
        ],
        classification_params_observed=True,
        notes=(
            "UNIQUELY TRACTABLE: The only screening context where both "
            "requirements for integrated analysis are jointly met: "
            "(a) histopathology provides the answer sheet that complementarity "
            "analysis demands, AND (b) the deskilling finding demonstrates the "
            "feedback dynamics that equilibrium modeling captures. "
            "\n\nCRITICAL TWO-STAGE STRUCTURE: Colonoscopy is a secondary screening. "
            "Cross-site comparison in ACCEPT is not 'same AI, different populations' "
            "but 'same AI, differently FIT-selected populations' — where the FIT "
            "selection mechanism varies in ways reflecting national differences in "
            "clinical convention, resource allocation, and the political economy of "
            "screening program design."
        )
    )


def mammography_screening() -> ScreeningDomain:
    """
    Mammography screening: validation problem PERSISTENTLY OPEN.
    
    DCIS (ductal carcinoma in situ) confounds the gold standard at the
    classification level itself. A biopsy-confirmed DCIS diagnosis does
    not tell us whether the woman has clinically important cancer. Experts
    genuinely disagree about whether DCIS constitutes cancer at all.
    
    This means the Ribers & Ullrich / Kleinberg complementarity framework
    CANNOT be applied: we cannot observe radiologists in the ROC space in
    the sense required, because the answer sheet is contaminated.
    
    Lauritzen et al. (2024) found DCIS diagnoses were significantly more
    frequent with AI support (20.4% vs 15.1%, P=.04), raising the possibility
    that reported reductions in false positives are partly artifacts of
    reclassifying probable false positives (overdiagnosed DCIS) as true
    positives.
    """
    return ScreeningDomain(
        name="Mammography Screening",
        stages=[
            ScreeningStage(
                name="Mammography (with or without AI assistance)",
                gold_standard=GoldStandardSpec(
                    name="Biopsy + histopathology",
                    description=(
                        "Biopsy results do NOT definitively resolve classification "
                        "for the outcome that matters. DCIS diagnoses — which "
                        "increased with AI support (Lauritzen et al. 2024) — "
                        "may represent overdiagnosis rather than true positive "
                        "detection."
                    ),
                    resolves_classification=False,  # THIS IS THE KEY DIFFERENCE
                    resolves_clinical_importance=False,
                    contamination_source=(
                        "DCIS ambiguity: most DCIS does not progress to invasive "
                        "cancer (Welch et al. 2011; Esserman et al. 2014). "
                        "Unknown proportion of biopsy-confirmed findings are "
                        "overdiagnosed cases that should arguably be classified "
                        "as false positives."
                    ),
                    delay_to_result="days to weeks (biopsy); years (progression/mortality)"
                ),
                validation_level=ValidationLevel.PERSISTENTLY_OPEN,
                roc_observable=False,
                complementarity_applicable=False,
                equilibrium_applicable=True,
                notes=(
                    "R&U complementarity framework INAPPLICABLE. "
                    "Mammography analysis relies on different tools: "
                    "decomposition of aggregate screening program effects "
                    "(Kalager et al. 2010) and comparison of observational "
                    "registry data with experimental trial data (MASAI), "
                    "rather than individual-level ROC space observation."
                )
            )
        ],
        population_scale="millions/year",
        base_rate_range="0.003-0.008 (breast cancer in screening population)",
        dominant_pathways=[
            "resource_reallocation",  # Kalager 2010: ~2/3 of benefit from non-classification
            "classification",  # But classification harms (overdiagnosis) may dominate benefits
            "strategy"  # Provider strategic behavior: malpractice asymmetry
        ],
        parameter_sources=[
            "Norwegian Cancer Registry (decades of observational data)",
            "MASAI trial (AI mammography, N>105,000; Lång collaboration)",
            "Lauritzen et al. 2024 (Danish AI mammography indicators)",
            "Eisemann et al. 2025 (German nationwide AI implementation)",
            "Kalager et al. 2010 NEJM (screening vs non-classification effects)",
            "Zahl, Kalager et al. 2020 (net QALY negative in Norway)",
            "Bretthauer et al. 2024 (lifetime gain: -190 to +237 days)"
        ],
        classification_params_observed=False,  # Contaminated by DCIS ambiguity
        notes=(
            "Classification pathway parameters must be estimated under persistent "
            "uncertainty about the gold standard — propagating through all "
            "downstream equilibrium calculations. Sensitivity analyses in O3 "
            "must account for this.\n\n"
            "GENDER DIMENSION: Potential harms — overdiagnosis, unnecessary surgery, "
            "radiation, psychological burden — fall exclusively on women. "
            "Asymmetric legal incentives (malpractice for missed diagnoses) "
            "may drive provider strategic behavior differently here."
        )
    )


def security_polygraph() -> ScreeningDomain:
    """
    Police polygraph screening: validation problem UNSOLVABLE.
    
    No gold standard for deception or threat status exists.
    Ground truth is rarely if ever obtainable.
    
    However, the LEMAS data provide an indirect outcome measure
    (sustained complaints), and the four-pathway framework explains
    how the program may still produce effects — primarily through
    Strategy and Information pathways, not Classification accuracy.
    """
    return ScreeningDomain(
        name="Police Polygraph (Pre-employment Screening)",
        stages=[
            ScreeningStage(
                name="Polygraph examination + interrogation",
                gold_standard=GoldStandardSpec(
                    name="None",
                    description=(
                        "No gold standard for deception exists. "
                        "Polygraph measures autonomic arousal, not deception. "
                        "Ground truth about applicant integrity is not obtainable "
                        "at the time of screening."
                    ),
                    resolves_classification=False,
                    resolves_clinical_importance=False,
                    contamination_source="Fundamental: the construct being measured is undefined",
                    delay_to_result="N/A — no confirmatory test exists"
                ),
                validation_level=ValidationLevel.UNSOLVABLE,
                roc_observable=False,
                complementarity_applicable=False,
                equilibrium_applicable=True,
                notes=(
                    "R&U complementarity framework INAPPLICABLE — no answer sheet. "
                    "But equilibrium effects still operate: Strategy pathway "
                    "(applicant self-selection/deterrence) and Information pathway "
                    "(bogus pipeline confession elicitation, d=0.41) may dominate. "
                    "LEMAS data suggest strong effect on sustained complaints "
                    "(-15.57 log, 95% CI [-25.1, -6.0]) but not total complaints, "
                    "consistent with selecting non-confessors rather than reducing "
                    "actual misconduct."
                )
            )
        ],
        population_scale="thousands of applicants/year",
        base_rate_range="0.05-0.20 (problematic applicants, assumed)",
        dominant_pathways=[
            "strategy",  # Deterrence/self-selection
            "information"  # Bogus pipeline, confession elicitation
        ],
        parameter_sources=[
            "NAS 2003 polygraph report (sensitivity/specificity ranges)",
            "LEMAS 2003-2007 (diff-in-diff, Wilde 2014)",
            "Roese & Jamieson 1993 (bogus pipeline: d=0.41 [0.25, 0.57])",
            "Pratt et al. 2006 (deterrence: r~0.15 [0.05, 0.25])",
        ],
        classification_params_observed=False,  # Assumed from NAS ranges
        notes=(
            "Classification parameters are ASSUMED, not observed. "
            "Sensitivity 0.70-0.90, specificity 0.50-0.80 (NAS 2003). "
            "These wide ranges reflect genuine scientific uncertainty, "
            "not measurement imprecision — the construct validity of "
            "polygraph testing is contested.\n\n"
            "KEY INSIGHT: The program's effects likely operate through "
            "non-classification pathways. The divergence between sustained "
            "and total complaint effects in LEMAS is consistent with a "
            "selection mechanism on confession/admission behavior rather "
            "than actual misconduct reduction."
        )
    )


def chat_control() -> ScreeningDomain:
    """
    EU Chat Control: proposed AI scanning of all digital communications for CSAM.
    Validation problem unsolvable for novel material; trivially solved for
    known-hash matching (but known-hash matching is not the policy-relevant case).
    """
    return ScreeningDomain(
        name="Chat Control (EU CSAM Scanning Proposal)",
        stages=[
            ScreeningStage(
                name="AI content classifier (novel CSAM detection)",
                gold_standard=GoldStandardSpec(
                    name="Human review (but overwhelmed by scale)",
                    description=(
                        "Human review of flagged content is the only available "
                        "'gold standard', but at the proposed scale (~1.5B false "
                        "positives), human review capacity is catastrophically "
                        "exceeded. Hash matching of known CSAM is trivially "
                        "validated but only detects previously-identified material."
                    ),
                    resolves_classification=False,  # At scale, cannot be done
                    resolves_clinical_importance=False,
                    contamination_source=(
                        "Scale: ~1.5B false positives overwhelm ~thousands of "
                        "investigators. No independent accuracy verification exists "
                        "for novel-material AI classifiers."
                    ),
                    delay_to_result="N/A at proposed scale"
                ),
                validation_level=ValidationLevel.UNSOLVABLE,
                roc_observable=False,
                complementarity_applicable=False,
                equilibrium_applicable=True,
                notes=(
                    "ZERO Information pathway: automated scanning has no "
                    "confession elicitation mechanism (unlike polygraph). "
                    "Strategy pathway NEGATIVE: sophisticated offenders "
                    "migrate to unmonitored channels. "
                    "Resource reallocation CATASTROPHIC: ~200:1 false positive "
                    "to true positive ratio drowns investigative capacity."
                )
            )
        ],
        population_scale="~450M people, billions of messages/day",
        base_rate_range="~0.00001 (CSAM prevalence in messages, assumed)",
        dominant_pathways=[
            "classification",  # Backfires catastrophically: 99.5% of flags are FP
            "resource_reallocation"  # Catastrophic: investigators overwhelmed
        ],
        parameter_sources=[
            "EU Commission 2023 (Microsoft 88% accuracy claim — unverified)",
            "Thorn marketing (>99% claim — no independent verification)",
            "Wilde 2023 Hertie School blog (Bayesian analysis)",
            "Steinebach 2023 (PhotoDNA analysis)"
        ],
        classification_params_observed=False,
        notes=(
            "Even under GENEROUS assumptions (90% accuracy), the program "
            "backfires catastrophically due to base rate rarity. "
            "P(innocent | flagged) ≈ 99.5%. "
            "Prediction: Net harm to child safety."
        )
    )


def iborderctrl() -> ScreeningDomain:
    """iBorderCtrl: EU AI border security pilot (2016-2019). Rejected."""
    return ScreeningDomain(
        name="iBorderCtrl (EU AI Border Security, 2016-2019)",
        stages=[
            ScreeningStage(
                name="AI facial micro-expression analysis",
                gold_standard=GoldStandardSpec(
                    name="None",
                    description="No validated accuracy data; AI emotion detection lacks scientific basis",
                    resolves_classification=False,
                    resolves_clinical_importance=False,
                    contamination_source="Fundamental: construct validity of 'deception detection' via micro-expressions is unsupported"
                ),
                validation_level=ValidationLevel.UNSOLVABLE,
                roc_observable=False,
                complementarity_applicable=False,
                equilibrium_applicable=True,
                notes="Correctly rejected after scientific/civil liberties criticism"
            )
        ],
        population_scale="millions of border crossings/year",
        base_rate_range="unknown",
        dominant_pathways=["strategy"],
        parameter_sources=["No validated accuracy data published"],
        classification_params_observed=False,
        notes="Success case for evidence-based critique."
    )


# ──────────────────────────────────────────────────────────────────────
# 4. ANALYTICAL TOOLS BY VALIDATION LEVEL
# ──────────────────────────────────────────────────────────────────────

TOOLS_BY_LEVEL = {
    ValidationLevel.FULLY_SOLVED: {
        "available": [
            "Full ROC observation (individual decision-makers)",
            "R&U/Kleinberg complementarity analysis",
            "Difficulty-stratified deployment optimization",
            "Observed classification pathway parameterization",
            "Equilibrium modeling (B&S framework)",
            "Counterfactual policy simulation with observed parameters",
        ],
        "examples": ["HIV testing", "Danish UTI prescribing (Ribers & Ullrich 2024)"],
    },
    ValidationLevel.LARGELY_SOLVED: {
        "available": [
            "Partial ROC observation (for removed/treated cases)",
            "R&U/Kleinberg complementarity analysis (with caveats)",
            "Difficulty-stratified deployment optimization",
            "Largely-observed classification pathway parameterization",
            "Equilibrium modeling (B&S framework)",
            "Sensitivity analyses for unobserved cells",
        ],
        "examples": [
            "Colonoscopy polyp detection (histopathology gold standard)",
            "FIT-to-colonoscopy pipeline (registry follow-up for FN)"
        ],
        "caveats": [
            "Miss rate (lesions never detected) requires estimation from randomized comparison",
            "Clinical importance (would adenoma have progressed?) remains open",
            "Sessile serrated lesion classification is evolving"
        ],
    },
    ValidationLevel.PERSISTENTLY_OPEN: {
        "available": [
            "Aggregate metrics (recall rate, cancer detection rate)",
            "Observational vs experimental comparison",
            "Pathway decomposition (Kalager et al. 2010 approach)",
            "Equilibrium modeling (B&S framework)",
            "Sensitivity analyses propagating gold standard uncertainty",
        ],
        "unavailable": [
            "Individual-level ROC observation",
            "R&U/Kleinberg complementarity analysis",
            "Difficulty-stratified deployment optimization",
        ],
        "examples": ["Mammography screening (DCIS ambiguity)"],
    },
    ValidationLevel.UNSOLVABLE: {
        "available": [
            "Equilibrium modeling with ASSUMED parameters (B&S framework)",
            "Bayesian classification analysis under generous assumptions",
            "Indirect outcome measures (e.g., LEMAS sustained complaints)",
            "Sensitivity analysis across parameter ranges",
            "Comparative structural analysis across domains",
        ],
        "unavailable": [
            "ROC observation of any kind",
            "R&U/Kleinberg complementarity analysis",
            "Observed classification pathway parameterization",
        ],
        "examples": [
            "Polygraph screening (no gold standard for deception)",
            "Chat Control (scale overwhelms human review)",
            "iBorderCtrl (construct validity unsupported)"
        ],
        "what_we_can_still_do": (
            "Model hypothetical complementarity under stated assumptions. "
            "The contrast with domains where ROC observation IS possible "
            "becomes a teaching tool about what the solved validation problem "
            "buys you empirically. In security, we can model what the ROC "
            "space WOULD look like under assumed parameters; in colonoscopy, "
            "we can OBSERVE it. This asymmetry is the point."
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────
# 5. CROSS-DOMAIN COMPARISON
# ──────────────────────────────────────────────────────────────────────

def build_validation_hierarchy() -> Dict[str, ScreeningDomain]:
    """
    Construct the full validation hierarchy across screening domains.
    
    Returns dict keyed by domain name, ordered from most to least
    analytically tractable.
    """
    domains = {
        "hiv": hiv_testing(),
        "colorectal_cancer": colorectal_cancer_screening(),
        "mammography": mammography_screening(),
        "polygraph": security_polygraph(),
        "chat_control": chat_control(),
        "iborderctrl": iborderctrl(),
    }
    return domains


def print_hierarchy_summary():
    """Print a human-readable summary of the validation hierarchy."""
    domains = build_validation_hierarchy()
    
    print("=" * 75)
    print("VALIDATION HIERARCHY ACROSS SCREENING DOMAINS")
    print("Wilde (2026), extending Besserve & Schölkopf (2022)")
    print("=" * 75)
    
    for key, domain in domains.items():
        level = domain.overall_validation_level.value.upper().replace("_", " ")
        complementarity = "YES" if domain.any_stage_supports_complementarity else "NO"
        observed = "Observed" if domain.classification_params_observed else "Assumed"
        
        print(f"\n{'─' * 75}")
        print(f"  {domain.name}")
        print(f"  Validation level: {level}")
        print(f"  R&U complementarity applicable: {complementarity}")
        print(f"  Classification parameters: {observed}")
        print(f"  Dominant pathways: {', '.join(domain.dominant_pathways)}")
        
        if len(domain.stages) > 1:
            print(f"  Multi-stage pipeline ({len(domain.stages)} stages):")
            for stage in domain.stages:
                slevel = stage.validation_level.value.upper().replace("_", " ")
                print(f"    → {stage.name}: {slevel}")
    
    print(f"\n{'─' * 75}")
    print("\nKEY METHODOLOGICAL IMPLICATION:")
    print("  The equilibrium framework (B&S 2022) applies to ALL domains.")
    print("  The R&U/Kleinberg complementarity framework requires a solved")
    print("  validation problem and thus applies only to HIV testing and")
    print("  colonoscopy (including the unexploited FIT stage).")
    print("  In security, we model HYPOTHETICAL complementarity under")
    print("  ASSUMED parameters. In colonoscopy, we can OBSERVE it.")
    print("  This asymmetry is the point.")
    print("=" * 75)


if __name__ == "__main__":
    print_hierarchy_summary()
