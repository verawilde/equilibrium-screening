"""
Equilibrium Effects in AI-Assisted Colonoscopy: Deskilling Analysis
Based on Besserve & Schölkopf (2022) and Budzyń et al. (2025)

STATUS: STARTER CODE — awaiting Oslo/CERG data for full implementation

What's here:
- Structural equations from medical applications working paper (Section 4)
- Can fit to Budzyń et al. (2025) published point estimate
- Testable predictions (Section 7)

What's needed for full equilibrium analysis (Section 6):
- Endoscopist ID (individual identifier)
- Date of each colonoscopy (timestamp)
- AI status per procedure (on/off)
- ADR per procedure
- Endoscopist experience (years, volume)
- Ideally: eye-tracking, AI detection events, long-term patient outcomes

Collaboration: Clinical Effectiveness Research Group, University of Oslo
Contacts: Mette Kalager, Michael Bretthauer

Author: Vera Wilde
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class BudzynPublishedData:
    """
    Published data from Budzyń et al. (2025) Lancet Gastroenterol Hepatol.
    This is what we CAN parameterize from.
    """
    # Pre-AI baseline
    H_0: float = 0.284                    # Baseline ADR (226/795)
    H_0_n: int = 795
    H_0_events: int = 226
    
    # Post-exposure (no AI available)
    H_E: float = 0.224                    # Post-exposure ADR (145/648)
    H_E_n: int = 648
    H_E_events: int = 145
    
    # Effect estimates
    absolute_diff: float = -0.060         # -6.0 percentage points
    absolute_diff_ci_low: float = -0.105  # -10.5 pp
    absolute_diff_ci_high: float = -0.016 # -1.6 pp
    
    adjusted_or: float = 0.69
    adjusted_or_ci_low: float = 0.53
    adjusted_or_ci_high: float = 0.89
    
    @property
    def relative_skill_loss(self) -> float:
        """Point estimate: 6.0/28.4 = 21.1%"""
        return abs(self.absolute_diff) / self.H_0
    
    @property
    def relative_skill_loss_ci(self) -> Tuple[float, float]:
        """95% CI: [5.6%, 37.0%]"""
        return (
            abs(self.absolute_diff_ci_high) / self.H_0,
            abs(self.absolute_diff_ci_low) / self.H_0
        )


@dataclass
class DataNeededFromOslo:
    """
    What we need from CERG to do full equilibrium analysis.
    See working paper Section 6.
    """
    minimum_required = [
        "Endoscopist ID (individual identifier)",
        "Date of each colonoscopy (timestamp)",
        "AI status per procedure (on/off)",
        "ADR per procedure (outcome)",
        "Endoscopist experience (years, volume)",
    ]
    
    ideal_for_mechanism = [
        "Eye-tracking data (gaze patterns)",
        "AI detection events (what AI flagged)",
        "Patient outcomes (CRC at 5-10 years)",
        "Training status (resident vs attending)",
    ]


class ColonoscopyDeskillingModel(nn.Module):
    """
    Structural equations from working paper Section 4.
    
    Variables:
        H: Human skill (baseline ADR without AI), range [0,1]
        A: AI detection contribution, range [0,1]
        E: Exposure duration (time using AI)
        D: Deskilling factor (skill loss), range [0,1]
        S: System ADR (combined), range [0,1]
    
    Structural Assignments:
        H(E) := H_0 * (1 - D(E))
        D(E) := θ_deskill * (1 - exp(-λ * E))
        S_AI := H(E) + A * (1 - H(E))
        S_noAI := H(E)
    
    Critical insight: RCTs measure S_AI vs H_0.
    Real-world involves S_AI vs S_noAI (with deskilling).
    """
    
    def __init__(
        self,
        H_0: float = 0.284,           # From Budzyń
        theta_deskill: float = 0.211, # Point estimate of skill loss
        lambda_rate: float = 1.0,     # Unknown without Oslo data
        A: float = 0.15,              # AI contribution (estimated)
    ):
        super().__init__()
        self.H_0 = torch.tensor(H_0)
        self.theta_deskill = nn.Parameter(torch.tensor(theta_deskill))
        self.lambda_rate = nn.Parameter(torch.tensor(lambda_rate))
        self.A = nn.Parameter(torch.tensor(A))
    
    def deskilling_function(self, E: torch.Tensor) -> torch.Tensor:
        """D(E) := θ_deskill * (1 - exp(-λ * E))"""
        return self.theta_deskill * (1 - torch.exp(-self.lambda_rate * E))
    
    def human_skill(self, E: torch.Tensor) -> torch.Tensor:
        """H(E) := H_0 * (1 - D(E))"""
        D = self.deskilling_function(E)
        return self.H_0 * (1 - D)
    
    def system_adr_with_ai(self, E: torch.Tensor) -> torch.Tensor:
        """S_AI := H(E) + A * (1 - H(E))"""
        H_E = self.human_skill(E)
        return H_E + self.A * (1 - H_E)
    
    def system_adr_without_ai(self, E: torch.Tensor) -> torch.Tensor:
        """S_noAI := H(E) — this is what Budzyń measured"""
        return self.human_skill(E)


def fit_to_budzyn_point_estimate(
    target_H_E: float = 0.224,
    H_0: float = 0.284,
    E_assumed: float = 1.0,
    n_iter: int = 1000
) -> Dict[str, float]:
    """
    Fit deskilling parameters to single published data point.
    
    LIMITATION: We can only identify θ_deskill * f(λ, E) 
    since E is not reported. We assume E=1 (normalized).
    
    Full identification requires Oslo data with individual exposure times.
    """
    model = ColonoscopyDeskillingModel(H_0=H_0)
    optimizer = torch.optim.Adam([model.theta_deskill, model.lambda_rate], lr=0.01)
    
    E = torch.tensor(E_assumed)
    target = torch.tensor(target_H_E)
    
    for _ in range(n_iter):
        optimizer.zero_grad()
        pred = model.system_adr_without_ai(E)
        loss = (pred - target) ** 2
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            model.theta_deskill.clamp_(0.0, 1.0)
            model.lambda_rate.clamp_(0.01, 10.0)
    
    return {
        'theta_deskill': model.theta_deskill.item(),
        'lambda_rate': model.lambda_rate.item(),
        'fitted_H_E': model.system_adr_without_ai(E).item(),
        'target_H_E': target_H_E,
        'NOTE': 'λ and θ not separately identified without exposure duration data'
    }


if __name__ == '__main__':
    print("=" * 60)
    print("AI Colonoscopy Deskilling: STARTER CODE")
    print("Awaiting Oslo/CERG data for full equilibrium analysis")
    print("=" * 60)
    
    # Show published data
    data = BudzynPublishedData()
    print(f"\nBudzyń et al. (2025) published findings:")
    print(f"  Pre-AI baseline ADR: {data.H_0:.1%}")
    print(f"  Post-exposure ADR (no AI): {data.H_E:.1%}")
    print(f"  Absolute difference: {data.absolute_diff:.1%}")
    print(f"  95% CI: [{data.absolute_diff_ci_low:.1%}, {data.absolute_diff_ci_high:.1%}]")
    print(f"  Relative skill loss: {data.relative_skill_loss:.1%}")
    
    # Fit to point estimate
    print(f"\nFitting to published point estimate...")
    fit = fit_to_budzyn_point_estimate()
    print(f"  θ_deskill: {fit['theta_deskill']:.3f}")
    print(f"  λ: {fit['lambda_rate']:.3f}")
    print(f"  Fitted H(E): {fit['fitted_H_E']:.3f} (target: {fit['target_H_E']})")
    print(f"  NOTE: {fit['NOTE']}")
    
    # What we need
    print(f"\nTo proceed with full analysis, need from Oslo:")
    for item in DataNeededFromOslo.minimum_required:
        print(f"  - {item}")
    
    print("\n" + "=" * 60)
