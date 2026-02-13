"""
Quick test of the cross-domain modules.
Runs without PyTorch (numpy + stdlib only).

Usage:
    cd equilibrium-screening
    python test_cross_domain.py
"""

import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cross_domain.validation_hierarchy import (
    build_validation_hierarchy,
    print_hierarchy_summary,
    ValidationLevel,
)
from cross_domain.complementarity_analysis import (
    colonoscopy_complementarity,
    security_hypothetical_complementarity,
    simulate_difficulty_stratified_deployment,
    simulate_hypothetical_roc,
    ParameterStatus,
    run_cross_domain_comparison,
)


def test_validation_hierarchy():
    print("\n" + "=" * 60)
    print("TEST 1: Validation Hierarchy")
    print("=" * 60)
    
    domains = build_validation_hierarchy()
    
    # Check expected levels
    assert domains["hiv"].overall_validation_level == ValidationLevel.FULLY_SOLVED
    assert domains["colorectal_cancer"].overall_validation_level == ValidationLevel.LARGELY_SOLVED
    assert domains["mammography"].overall_validation_level == ValidationLevel.PERSISTENTLY_OPEN
    assert domains["polygraph"].overall_validation_level == ValidationLevel.UNSOLVABLE
    assert domains["chat_control"].overall_validation_level == ValidationLevel.UNSOLVABLE
    
    # CRC should have 2 stages (FIT → colonoscopy)
    assert len(domains["colorectal_cancer"].stages) == 2, \
        f"Expected 2 stages for CRC, got {len(domains['colorectal_cancer'].stages)}"
    
    # Both CRC stages should support complementarity
    for stage in domains["colorectal_cancer"].stages:
        assert stage.complementarity_applicable, \
            f"CRC stage '{stage.name}' should support complementarity"
    
    # Mammography should NOT support complementarity
    assert not domains["mammography"].any_stage_supports_complementarity
    
    # Security should NOT support complementarity
    assert not domains["polygraph"].any_stage_supports_complementarity
    assert not domains["chat_control"].any_stage_supports_complementarity
    
    print("  ✓ All validation levels correct")
    print("  ✓ CRC has 2-stage pipeline (FIT → colonoscopy)")
    print("  ✓ Both CRC stages support complementarity")
    print("  ✓ Mammography does NOT support complementarity (DCIS)")
    print("  ✓ Security domains do NOT support complementarity (no gold standard)")


def test_complementarity_specs():
    print("\n" + "=" * 60)
    print("TEST 2: Complementarity Specifications")
    print("=" * 60)
    
    colo = colonoscopy_complementarity()
    assert colo["complementarity_applicable"] is True
    assert colo["parameter_status"] == ParameterStatus.OBSERVED
    assert len(colo["strata"]) == 3  # easy, intermediate, hard
    assert len(colo["testable_predictions"]) >= 3
    assert "FIT" in colo["two_stage_note"]
    assert "secondary screening" in colo["two_stage_note"].lower()
    
    sec = security_hypothetical_complementarity()
    assert sec["complementarity_applicable"] is False
    assert sec["parameter_status"] == ParameterStatus.ASSUMED
    
    # All security strata should be empirically untestable
    for stratum in sec["polygraph_strata"]:
        assert not stratum.empirically_testable, \
            f"Security stratum '{stratum.name}' should NOT be empirically testable"
    
    # Except known-hash matching in Chat Control (trivially validated)
    known_hash = sec["chat_control_strata"][0]
    assert known_hash.empirically_testable, \
        "Known-hash matching should be testable (trivially validated)"
    
    print("  ✓ Colonoscopy: complementarity applicable, OBSERVED parameters")
    print("  ✓ Colonoscopy: 3 difficulty strata, ≥3 testable predictions")
    print("  ✓ Colonoscopy: FIT two-stage pipeline documented")
    print("  ✓ Security: complementarity NOT applicable, ASSUMED parameters")
    print("  ✓ Security: polygraph strata NOT empirically testable")
    print("  ✓ Chat Control: known-hash IS testable (but not the policy case)")


def test_hypothetical_simulation():
    print("\n" + "=" * 60)
    print("TEST 3: Hypothetical Difficulty-Stratified Deployment")
    print("=" * 60)
    
    # Security case (assumed parameters)
    results = simulate_difficulty_stratified_deployment(
        base_rate=0.10,
        parameter_status=ParameterStatus.ASSUMED,
    )
    
    assert "CAVEAT" in results, "Security simulation must include CAVEAT"
    
    # Difficulty-stratified should generally outperform uniform strategies
    # (this is the R&U prediction, though with assumed params the magnitude
    # is illustrative only)
    strat = results["difficulty_stratified"]
    human = results["uniform_human"]
    ai = results["uniform_ai"]
    
    print(f"  Uniform human:  sens={human['sensitivity']:.3f}, spec={human['specificity']:.3f}")
    print(f"  Uniform AI:     sens={ai['sensitivity']:.3f}, spec={ai['specificity']:.3f}")
    print(f"  Stratified:     sens={strat['sensitivity']:.3f}, spec={strat['specificity']:.3f}")
    print(f"  CAVEAT present: ✓")
    
    # Test with observed status (colonoscopy case)
    colo_results = simulate_difficulty_stratified_deployment(
        base_rate=0.05,
        parameter_status=ParameterStatus.OBSERVED,
        human_sens_by_difficulty={
            "easy": (0.90, 0.80),
            "intermediate": (0.75, 0.70),
            "hard": (0.60, 0.65),
        },
        ai_sens_by_difficulty={
            "easy": (0.95, 0.85),
            "intermediate": (0.70, 0.75),
            "hard": (0.80, 0.60),
        },
    )
    assert "NOTE" in colo_results, "Colonoscopy simulation should include NOTE (not CAVEAT)"
    assert "FIT" in colo_results["NOTE"], "Colonoscopy note should mention FIT pre-selection"
    print(f"  Colonoscopy simulation includes FIT caveat: ✓")


def test_hypothetical_roc():
    print("\n" + "=" * 60)
    print("TEST 4: Hypothetical ROC Cloud")
    print("=" * 60)
    
    cloud = simulate_hypothetical_roc(
        sensitivity_range=(0.50, 0.90),
        specificity_range=(0.50, 0.80),
        n_samples=500,
    )
    
    assert cloud.shape == (500, 2)
    assert (cloud[:, 0] >= 0.20).all() and (cloud[:, 0] <= 0.50).all(), \
        "FPR should be in [0.20, 0.50] given specificity [0.50, 0.80]"
    assert (cloud[:, 1] >= 0.50).all() and (cloud[:, 1] <= 0.90).all(), \
        "TPR should be in [0.50, 0.90] given sensitivity [0.50, 0.90]"
    
    print(f"  Generated {cloud.shape[0]} hypothetical ROC positions")
    print(f"  FPR range: [{cloud[:, 0].min():.3f}, {cloud[:, 0].max():.3f}]")
    print(f"  TPR range: [{cloud[:, 1].min():.3f}, {cloud[:, 1].max():.3f}]")
    print("  ✓ All within expected NAS parameter bounds")


if __name__ == "__main__":
    print("Testing cross-domain modules...")
    print("(No PyTorch required — numpy + stdlib only)")
    
    test_validation_hierarchy()
    test_complementarity_specs()
    test_hypothetical_simulation()
    test_hypothetical_roc()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    
    print("\n\nRunning full cross-domain comparison...\n")
    run_cross_domain_comparison()
