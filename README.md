# Equilibrium Effects in Mass Screening Programs

A four-pathway structural causal framework for analyzing mass screening programs, building on [Besserve & Schölkopf (2022)](https://proceedings.mlr.press/v180/besserve22a.html) "Learning soft interventions in complex equilibrium systems."

## The Problem

Mass screening programs for low-prevalence problems share a common mathematical structure that makes them prone to backfiring. The National Academy of Sciences' 2003 polygraph report (Fienberg et al.) analyzed this using only the **classification pathway** — applying Bayes' Rule to estimate true/false positive outcomes.

This single-pathway analysis misses three additional causal pathways:

1. **Classification** — test accuracy producing TP/FP/TN/FN (Fienberg's focus)
2. **Strategy** — behavioral changes in response to test existence (deterrence, evasion, gaming)
3. **Information** — knowledge generated through screening (elicitation, bogus pipeline, iatrogenic effects)
4. **Resource reallocation** — both quantitative (zero-sum capacity) and qualitative (framing)

**Key insight**: Even if a test backfires via Classification (due to rarity + accuracy-error tradeoff), the Strategy and Information pathways may compensate — or amplify the harm. We cannot determine net effects from classification accuracy alone.

## Repository Structure

```
equilibrium-screening/
├── security/                    # Security/surveillance applications (implemented)
│   └── screening_equilibrium.py # Polygraph, iBorderCtrl, Chat Control
├── medical/                     # Medical screening applications  
│   ├── colonoscopy_deskilling.py    # Starter from Budzyń (awaiting Oslo data)
│   └── README.md
└── requirements.txt
```

## Applications

### Security (implemented — can run simulations now)

| Application | Classification | Strategy | Information | Resources | Prediction |
|------------|---------------|----------|-------------|-----------|------------|
| **Police Polygraph** | Weak/contested | Deterrence possible | Strong (bogus pipeline) | Moderate | **Uncertain** |
| **iBorderCtrl** | Unknown/poor | Negative | Weak | Uncertain | **Net harm** |
| **Chat Control** | Backfires (99.5% FP) | Evasion dominates | Zero (automated) | Catastrophic | **Net harm** |

### Medical (awaiting data for full implementation)

| Application | Status | Data Source | Key Question |
|------------|--------|-------------|--------------|
| **AI Colonoscopy Deskilling** | Starter code from published data | Budzyń et al. (2025); full analysis awaits Oslo/CERG | Does AI degrade human skill? |
| **Mammography Observational** | Placeholder | Awaiting Oslo collaboration | Do non-classification pathways dominate? |
| **Mammography AI (MASAI)** | Placeholder | Awaiting Lång/MASAI data | Does AI increase overdiagnosis? |

## Installation

```bash
git clone https://github.com/verawilde/equilibrium-screening
cd equilibrium-screening
pip install -r requirements.txt
```

## Usage

```python
from security.screening_equilibrium import ChatControlModel, PolygraphModel, compare_programs

# Compare programs
results = compare_programs()
print(results['chat_control'])  # Predicts net harm
print(results['polygraph'])     # Uncertain - may help via non-classification pathways
```

## Key Literature Parameters

| Parameter | Estimate | 95% CI | Source |
|-----------|----------|--------|--------|
| Bogus pipeline effect | d = 0.41 | [0.25, 0.57] | Roese & Jamieson (1993) |
| Deterrence (certainty) | r ~ 0.15 | [0.05, 0.25] | Pratt et al. (2006) |
| Deterrence (severity) | r ~ 0 | [-0.05, 0.05] | Doob & Webster (2003) |
| Polygraph sensitivity | 0.70-0.90 | Wide | NAS (2003) |
| Polygraph specificity | 0.50-0.80 | Wide | NAS (2003) |
| AI colonoscopy deskilling | -6.0 pp | [-10.5, -1.6] | Budzyń et al. (2025) |

## References

- Besserve, M., & Schölkopf, B. (2022). Learning soft interventions in complex equilibrium systems. UAI 2022.
- Budzyń, K., et al. (2025). Endoscopist deskilling risk after exposure to AI in colonoscopy. Lancet Gastroenterol Hepatol.
- National Academy of Sciences. (2003). The Polygraph and Lie Detection.
- Roese, N. J., & Jamieson, D. W. (1993). Twenty years of bogus pipeline research. Psychological Bulletin.
- Wilde, V. (2014). Neutral Competence? Polygraphy and Technology-Mediated Administrative Decisions. PhD Dissertation, UVA.

## Author

Vera Wilde — [Wilde Truth](https://wildetruth.substack.com)

## License

MIT
