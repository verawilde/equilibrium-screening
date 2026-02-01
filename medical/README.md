# Medical Screening Applications

Applying the four-pathway equilibrium framework to medical screening programs.

## Status Overview

| Application | Code Status | Data Status |
|-------------|-------------|-------------|
| AI Colonoscopy Deskilling | **Starter** — structural equations, fits to published point | Awaiting Oslo/CERG |
| Mammography Observational | Placeholder | Awaiting Oslo/CERG |
| Mammography AI (MASAI) | Placeholder | Awaiting Lång/MASAI |

---

## 1. AI Colonoscopy Deskilling (`colonoscopy_deskilling.py`)

**What's implemented:**
- Structural equations from working paper Section 4
- Fits to Budzyń et al. (2025) published point estimate
- Testable predictions framework

**What's needed from Oslo (working paper Section 6):**

*Minimum:*
- Endoscopist ID
- Date of each colonoscopy  
- AI status per procedure
- ADR per procedure
- Endoscopist experience

*Ideal:*
- Eye-tracking data
- AI detection events
- Long-term patient outcomes (CRC at 5-10 years)

**Key finding from Budzyń:**
- Pre-AI ADR: 28.4%
- Post-exposure ADR: 22.4%
- Skill loss: ~21% (95% CI: 6% to 37%)

---

## 2. Mammography Observational

**Not yet implemented** — placeholder only.

**Key findings to model (from Oslo publications):**
- Kalager 2009: Unscreened women also benefited from program
- Kalager 2010: Only ~1/3 of mortality reduction from screening itself
- Zahl 2020: Net QALY may be negative

**Four-pathway interpretation:**
Benefits came primarily through resource reallocation (multidisciplinary teams) and information pathways, not classification accuracy.

---

## 3. Mammography AI Experimental  

**Not yet implemented** — placeholder only.

**Relevant published findings:**
- Eisemann 2025: German AI increased DCIS diagnoses
- Lauritzen 2024: Danish AI DCIS 20.4% vs 15.1% pre-AI

**Key question:** Does AI mammography increase overdiagnosis without reducing mortality?

---

## Complementary Failure Modes

| Modality | Classification Effect | Primary Failure Mode |
|----------|----------------------|---------------------|
| Colonoscopy | AI improves ADR ✓ | **Strategy**: skill atrophy |
| Mammography | AI improves detection ✓ | **Classification**: overdiagnosis |

Both illustrate: **Classification accuracy ≠ system effectiveness**

---

## Collaboration Contacts

- **Oslo/CERG**: Mette Kalager, Michael Bretthauer
- **MASAI**: Kristina Lång
