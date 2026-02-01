"""
Structural Causal Modeling for Mass Screening Programs: A Four-Pathway Framework
Based on Besserve & Schölkopf (2022) "Learning soft interventions in complex equilibrium systems"

This implements the structural equations from Wilde (2026) security applications working paper,
using PyTorch automatic differentiation for equilibrium finding and intervention optimization.

Four pathways from Test to Outcome:
1. CLASSIFICATION: True/False Positives/Negatives (Fienberg/NAS focus)
2. STRATEGY: Deterrence, Evasion, Gaming
3. INFORMATION: Elicitation, Bogus Pipeline, Confession
4. RESOURCE REALLOCATION: Zero-sum capacity, Framing effects

Applications (chronological):
- Police Polygraph Programs (Wilde 2014 LEMAS analysis)
- iBorderCtrl (EU border AI pilot, 2016-2019)
- Chat Control (EU CSAM scanning proposal)

Author: Vera Wilde
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ParameterEstimates:
    """
    Literature-derived parameter estimates with uncertainty bounds.
    Sources documented in security applications working paper.
    """
    # Bogus pipeline effect - Roese & Jamieson (1993) meta-analysis
    bogus_pipeline_d: float = 0.41
    bogus_pipeline_ci_low: float = 0.25
    bogus_pipeline_ci_high: float = 0.57
    
    # Deterrence (certainty) - Pratt et al. (2006), Nagin (2013)
    deterrence_certainty_r: float = 0.15
    deterrence_certainty_low: float = 0.05
    deterrence_certainty_high: float = 0.25
    
    # Deterrence (severity) - Doob & Webster (2003), Nagin (2013)
    deterrence_severity_r: float = 0.0
    deterrence_severity_low: float = -0.05
    deterrence_severity_high: float = 0.05
    
    # Polygraph operating characteristics - NAS (2003)
    polygraph_sensitivity_low: float = 0.70
    polygraph_sensitivity_high: float = 0.90
    polygraph_specificity_low: float = 0.50
    polygraph_specificity_high: float = 0.80
    
    # LEMAS polygraph effects - Wilde (2014)
    lemas_sustained_complaints_log: float = -15.57
    lemas_sustained_ci_low: float = -25.1
    lemas_sustained_ci_high: float = -6.0
    lemas_total_complaints_log: float = -62.34
    lemas_total_ci_low: float = -157.8
    lemas_total_ci_high: float = 33.1


class FourPathwayModel(nn.Module):
    """
    Structural causal model for mass screening programs with four pathways.
    
    Variables (from working paper Section 4.1):
        X: Population screened
        T: Test intensity/coverage [0,1]
        C: Classification output (flagged positives) [0,X]
        S: Strategic behavior (net deterrence, positive=deterred, negative=evasion)
        I: Information yield (confessions/tips) [0,X]
        R: Resource allocation proportion (mass vs targeted) [0,1]
        Y: Outcome (security/integrity measure, higher=better)
        π: Base rate (true prevalence) [0,1]
    """
    
    def __init__(
        self,
        # Test characteristics
        sensitivity: float = 0.80,
        specificity: float = 0.65,
        # Strategy parameters
        theta_deterrence: float = 0.15,  # From deterrence literature
        theta_evasion: float = 0.10,
        # Information parameters  
        theta_confession: float = 0.41,  # Bogus pipeline effect size
        # Resource parameters
        base_allocation: float = 0.5,
        theta_false_positive_burden: float = 0.01,
        theta_tip_efficiency: float = 0.05,
        investigative_capacity: float = 1000.0,
        # Outcome weights
        theta_C_tp: float = 1.0,   # True positive value
        theta_C_fp: float = -0.5,  # False positive cost
        theta_S: float = 0.3,      # Strategy contribution weight
        theta_I: float = 0.5,      # Information contribution weight
        theta_R: float = 0.2,      # Resource cost weight
        opportunity_cost: float = 1.0,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        
        # Test characteristics
        self.sensitivity = nn.Parameter(torch.tensor(sensitivity, device=device))
        self.specificity = nn.Parameter(torch.tensor(specificity, device=device))
        
        # Strategy parameters
        self.theta_deterrence = nn.Parameter(torch.tensor(theta_deterrence, device=device))
        self.theta_evasion = nn.Parameter(torch.tensor(theta_evasion, device=device))
        
        # Information parameters
        self.theta_confession = nn.Parameter(torch.tensor(theta_confession, device=device))
        
        # Resource parameters
        self.base_allocation = torch.tensor(base_allocation, device=device)
        self.theta_fp_burden = nn.Parameter(torch.tensor(theta_false_positive_burden, device=device))
        self.theta_tip_eff = nn.Parameter(torch.tensor(theta_tip_efficiency, device=device))
        self.capacity = torch.tensor(investigative_capacity, device=device)
        
        # Outcome weights
        self.theta_C_tp = torch.tensor(theta_C_tp, device=device)
        self.theta_C_fp = torch.tensor(theta_C_fp, device=device)
        self.theta_S = torch.tensor(theta_S, device=device)
        self.theta_I = torch.tensor(theta_I, device=device)
        self.theta_R = torch.tensor(theta_R, device=device)
        self.opportunity_cost = torch.tensor(opportunity_cost, device=device)
    
    def classification_pathway(
        self, 
        T: torch.Tensor,
        X: torch.Tensor,
        pi: torch.Tensor,
        effective_sensitivity: Optional[torch.Tensor] = None,
        effective_specificity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        C := f_C(T, X, π) = T · X · [π·sensitivity + (1-π)·(1-specificity)]
        
        Under rarity (π → 0), C is dominated by false positives.
        """
        sens = effective_sensitivity if effective_sensitivity is not None else self.sensitivity
        spec = effective_specificity if effective_specificity is not None else self.specificity
        
        true_positives = T * X * pi * sens
        false_positives = T * X * (1 - pi) * (1 - spec)
        C = true_positives + false_positives
        
        return C, true_positives, false_positives
    
    def g_awareness(self, T: torch.Tensor) -> torch.Tensor:
        """Awareness function: how visible/credible is the screening program?"""
        return torch.sigmoid(5 * (T - 0.5))
    
    def h_sophistication(self, C: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Sophistication function: learning to evade based on classification feedback."""
        flag_rate = C / (X + 1e-6)
        return torch.tanh(flag_rate * 10)
    
    def strategy_pathway(
        self,
        T: torch.Tensor,
        C: torch.Tensor,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        S := θ_deterrence · T · g_awareness(T) - θ_evasion · h_sophistication(C)
        
        Net deterrence: positive = deterred, negative = evasion dominates
        """
        deterrence = self.theta_deterrence * T * self.g_awareness(T)
        evasion = self.theta_evasion * self.h_sophistication(C, X)
        return deterrence - evasion
    
    def information_pathway(
        self,
        T: torch.Tensor,
        S: torch.Tensor,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        I := θ_confession · T · (1 - S/X) · bogus_pipeline_effect
        
        People confess more when they believe a test works, regardless of accuracy.
        """
        confession_pool = torch.clamp(1 - S / (X + 1e-6), 0, 1)
        return self.theta_confession * T * X * confession_pool
    
    def resource_pathway(
        self,
        C: torch.Tensor,
        I: torch.Tensor
    ) -> torch.Tensor:
        """
        R := base_allocation + θ_fp_burden · C/capacity - θ_tip_efficiency · I
        
        More false positives drain resources; good tips improve efficiency.
        """
        R = (
            self.base_allocation 
            + self.theta_fp_burden * C / self.capacity 
            - self.theta_tip_eff * I / self.capacity
        )
        return torch.clamp(R, 0, 1)
    
    def outcome(
        self,
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        S: torch.Tensor,
        I: torch.Tensor,
        R: torch.Tensor
    ) -> torch.Tensor:
        """
        Y* := θ_C·TP - θ_FP·FP + θ_S·S + θ_I·I - θ_R·R·opportunity_cost
        """
        classification_contrib = self.theta_C_tp * true_positives + self.theta_C_fp * false_positives
        strategy_contrib = self.theta_S * S
        information_contrib = self.theta_I * I
        resource_cost = self.theta_R * R * self.opportunity_cost
        
        return classification_contrib + strategy_contrib + information_contrib - resource_cost
    
    def forward(
        self,
        T: torch.Tensor,
        X: torch.Tensor,
        pi: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """Full forward pass through all four pathways."""
        # Classification pathway
        C, TP, FP = self.classification_pathway(T, X, pi)
        
        # Strategy pathway
        S = self.strategy_pathway(T, C, X)
        
        # Strategy ↔ Classification moderation (evasion degrades test performance)
        evasion_factor = torch.clamp(torch.abs(S) / (X + 1e-6) * 10, 0, 0.3)
        effective_sens = self.sensitivity * (1 - evasion_factor)
        effective_spec = self.specificity * (1 - evasion_factor)
        
        # Recalculate with degraded accuracy
        C, TP, FP = self.classification_pathway(T, X, pi, effective_sens, effective_spec)
        
        # Information pathway
        I = self.information_pathway(T, S, X)
        
        # Resource pathway
        R = self.resource_pathway(C, I)
        
        # Outcome
        Y = self.outcome(TP, FP, S, I, R)
        
        if return_components:
            return {
                'Y': Y, 'C': C, 'TP': TP, 'FP': FP,
                'S': S, 'I': I, 'R': R,
                'effective_sensitivity': effective_sens,
                'effective_specificity': effective_spec
            }
        return Y


class PolygraphModel(FourPathwayModel):
    """
    Police polygraph application: Pre-employment screening for law enforcement.
    
    From Wilde (2014) LEMAS analysis:
    - Sustained complaints (log): -15.57, 95% CI [-25.1, -6.0]
    - Total complaints (log): -62.34, 95% CI [-157.8, +33.1]
    
    Key insight: Effect likely operates through Strategy (self-selection) and/or
    Information (bogus pipeline), not Classification accuracy.
    """
    
    def __init__(self, **kwargs):
        defaults = {
            'sensitivity': 0.80,
            'specificity': 0.65,
            'theta_deterrence': 0.20,
            'theta_evasion': 0.05,
            'theta_confession': 0.41,  # Bogus pipeline effect
            'investigative_capacity': 1000.0,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
    
    def analyze(
        self,
        applicants: float = 1000.0,
        problematic_applicant_rate: float = 0.10
    ) -> Dict[str, float]:
        T = torch.tensor(1.0)
        X = torch.tensor(applicants)
        pi = torch.tensor(problematic_applicant_rate)
        
        results = self.forward(T, X, pi, return_components=True)
        
        return {
            'outcome': results['Y'].item(),
            'flagged_applicants': results['C'].item(),
            'true_positives': results['TP'].item(),
            'false_positives': results['FP'].item(),
            'deterrence_effect': results['S'].item(),
            'confession_yield': results['I'].item(),
            'resource_allocation': results['R'].item()
        }


class IBorderCtrlModel(FourPathwayModel):
    """
    iBorderCtrl: EU-funded pilot (2016-2019) testing AI "lie detection" at borders.
    
    The project faced substantial scientific criticism and was not adopted.
    This represents a success case for evidence-based critique.
    """
    
    def __init__(self, **kwargs):
        defaults = {
            'sensitivity': 0.60,  # No validated accuracy
            'specificity': 0.70,
            'theta_deterrence': 0.02,  # Weak
            'theta_evasion': 0.25,     # Strong - traffickers evade
            'theta_confession': 0.10,  # Weak - automated kiosk
            'investigative_capacity': 500.0,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class ChatControlModel(FourPathwayModel):
    """
    Chat Control: EU proposal for AI scanning of all digital communications for CSAM.
    
    From Wilde (2023) Fienberg-style Bayesian analysis:
    - Messages scanned: 10 billion
    - Base rate: 1/1000
    - False positives: >1.5 billion
    - P(innocent | flagged): 99.5%
    - FP per TP: ~200:1
    
    Prediction: Net harm to child safety
    """
    
    def __init__(self, **kwargs):
        defaults = {
            'sensitivity': 0.80,
            'specificity': 0.90,  # Even "high" accuracy fails under rarity
            'theta_deterrence': 0.05,  # Weak
            'theta_evasion': 0.30,     # Strong - easy to evade
            'theta_confession': 0.0,   # Zero - no interrogation
            'investigative_capacity': 10000.0,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
    
    def analyze(
        self,
        messages_scanned: float = 10e9,
        base_rate: float = 0.001
    ) -> Dict[str, float]:
        T = torch.tensor(1.0)
        X = torch.tensor(messages_scanned)
        pi = torch.tensor(base_rate)
        
        results = self.forward(T, X, pi, return_components=True)
        
        fp_per_tp = results['FP'] / (results['TP'] + 1e-6)
        p_innocent_given_flagged = results['FP'] / (results['C'] + 1e-6)
        
        return {
            'outcome': results['Y'].item(),
            'flagged_positives': results['C'].item(),
            'true_positives': results['TP'].item(),
            'false_positives': results['FP'].item(),
            'fp_per_tp': fp_per_tp.item(),
            'p_innocent_given_flagged': p_innocent_given_flagged.item(),
            'net_deterrence': results['S'].item(),
            'information_yield': results['I'].item(),
            'resource_burden': results['R'].item()
        }


def compare_programs(
    messages_scanned: float = 10e9,
    csam_base_rate: float = 0.001,
    polygraph_applicants: float = 1000.0,
    problematic_rate: float = 0.10
) -> Dict[str, Dict[str, float]]:
    """Compare all three security screening programs."""
    
    polygraph = PolygraphModel()
    iborderctrl = IBorderCtrlModel()
    chat_control = ChatControlModel()
    
    return {
        'polygraph': polygraph.analyze(polygraph_applicants, problematic_rate),
        'iborderctrl': iborderctrl.analyze(polygraph_applicants, problematic_rate),
        'chat_control': chat_control.analyze(messages_scanned, csam_base_rate)
    }


if __name__ == '__main__':
    print("=" * 70)
    print("Structural Causal Modeling for Mass Screening Programs")
    print("Four-Pathway Framework - Security Applications")
    print("=" * 70)
    
    comparison = compare_programs()
    
    print("\n--- POLICE POLYGRAPH (Wilde 2014 LEMAS) ---")
    for k, v in comparison['polygraph'].items():
        print(f"   {k}: {v:.4f}")
    
    print("\n--- iBORDERCTRL (EU 2016-2019) ---")
    for k, v in comparison['iborderctrl'].items():
        print(f"   {k}: {v:.4f}")
    
    print("\n--- CHAT CONTROL (EU proposal) ---")
    for k, v in comparison['chat_control'].items():
        if abs(v) > 1e6:
            print(f"   {k}: {v:.2e}")
        else:
            print(f"   {k}: {v:.4f}")
    
    print("\n" + "=" * 70)
    print("KEY PREDICTIONS:")
    print("  Polygraph: Uncertain - may help via Information pathway (bogus pipeline)")
    print("  iBorderCtrl: Net harm - correctly rejected")
    print("  Chat Control: Net harm - evasion + zero Information + catastrophic FP burden")
    print("=" * 70)
