# Verification & Engineering Assessment Report
### Problem Statement 10 | AI Manthan 2.0 (Aerospace & Materials Science)
**Author / Backend System:** AI Manthan Senior Scientific Computing & AI Team  
**Dataset Generation Date:** 2026-10-28 | **Mode:** FULL (6,400 rows)

---

## 1. Data Honesty & Simulation Labeling
In strict adherence to scientific data integrity:
- **Simulation Data Label:** All data in `data/lattice_dataset_v1.csv` (6,400 rows) represents **numerical simulation data from our verified beam-element periodic homogenization solver**. It is **NOT experimental test data**.
- **No Fabricated Claims:** All numbers quoted in this report and documentation are compiled dynamically from `results/headline_numbers.json`, reflecting code that actually executed end-to-end.

---

## 2. What Was Verified and How

### 2.1 Physics Solver Verification Suite (10 of 10 Passed)
- **a) Single Beam Cantilever:** Deflection matches beam theory to 1.76e-16; axial stretch to 0.00e+00.
- **b) Symmetry & Positive Definiteness:** Elasticity tensor $C = C^T$ to $< 10^{-9}$ and all eigenvalues $> 0$ across all 8 topologies.
- **c) Simple Cubic Analytical Formulation:** $C_{11} = \rho/3$ matches analytical solution to 0.00e+00; coupling $C_{12} = 0$.
- **d) Cubic Isotropy:** $E_x = E_y = E_z$ verified to $< 10^{-6}$ for cubic configurations (SC, BCC, Octet, Diamond).
- **e) Upper Bound:** $E_x, E_y, E_z \le \rho$ strictly satisfied across all topologies and densities.
- **f) Scaling Slopes:** Octet slope = 1.006 (stretch-dominated $\approx 1.0$), BCC slope = 1.998 (bending-dominated $\approx 2.0$).
- **g) Literature Comparison:** Octet at $\rho=0.01$ matches $E_z/\rho = 1/9$ within 0.30% (Deshpande et al. 2001; citation unconfirmed).
- **h) Size Invariance:** Scaling cell size from 1 to 3 leaves normalized moduli unchanged ($< 10^{-9}$).
- **i) Energy Self-Consistency:** Microscopic internal strain energy equals macroscopic $\frac{1}{2} \varepsilon^T C \varepsilon V$ to $< 10^{-9}$.
- **j) Determinism:** Solver produces bit-exact identical stiffness tensors across independent runs.

### 2.2 First-Failure Strength Formulation
- **Simple Cubic Exact Yield & Buckling:** $yield\_z = \rho/3$ (diff = 0.00e+00) and $buckle\_z = \pi \rho^2 / 36$ (diff = 0.00e+00).
- **Linear Elastic First-Failure Scope:** First-failure strain derived as $failure\_strain\_z = failure\_stress\_z / E_z$. Post-failure plasticity and plateaus are NOT modeled.

---

## 3. Surrogate Accuracy & Generalization

### 3.1 In-Distribution Test Set Accuracy (FULL Mode: 6,400 Rows)
Evaluated on a 15% stratified test split (960 configurations):
- **Primary Model:** MLP_Ensemble
- **Minimum $R^2$ across all targets:** 0.9949
- **Mean Absolute Percent Error (MAPE):** 2.89%
- **Mean Median Absolute Percent Error (MedAPE):** 1.12%

| Target | HGB $R^2$ | HGB MedAPE (%) | HGB MAPE (%) | HGB MSPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Ex** | 0.9974 | 0.86% | 1.68% | -0.06% |
| **Ez** | 0.9975 | 1.69% | 2.42% | -0.27% |
| **Gxy** | 0.9990 | 0.55% | 0.93% | -0.02% |
| **Gxz** | 0.9993 | 0.47% | 0.79% | -0.02% |
| **nu_xy** | 0.9988 | 1.31% | 4.04% | +0.46% |
| **nu_xz** | 0.9949 | 2.05% | 10.09% | +4.95% |
| **yield_z** | 0.9963 | 1.95% | 2.77% | -0.09% |
| **buckle_z** | 0.9971 | 0.08% | 0.38% | -0.07% |

### 3.2 Held-Out Topology Generalization (Leave-One-Topology-Out)
Surrogates were evaluated by holding out an entire topology during training. We compare baseline geometric descriptors against models augmented with transferrable affine Voigt physics features:

| Held-Out Topology | Base $R^2$ | Base MedAPE (%) | Phys $R^2$ (with Physics) | Phys MedAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **sc** | -1.250 | 98.2% | -1.268 | 98.3% |
| **bcc** | -26.047 | 1329.6% | -14.572 | 876.3% |
| **octet** | 0.992 | 5.1% | 0.991 | 5.0% |
| **diamond** | -8.515 | 551.6% | 0.189 | 61.7% |
| **bcc_z** | -0.702 | 93.9% | -0.647 | 93.4% |
| **sc_bcc** | 0.850 | 22.9% | 0.976 | 6.7% |
| **sc_octet** | 0.933 | 20.2% | 0.991 | 5.1% |
| **fcc_face** | 0.268 | 66.6% | 0.330 | 63.5% |

**Honest Generalization Assessment:**  
Surrogate models are **accurate inside the trained topology set, but unreliable for unseen topologies**. While transferrable affine physics features noticeably improve extrapolation for diamond and hybrid lattices, architectures with unique bending modes (such as BCC) exhibit high error when unobserved in the training set.

### 3.3 Fresh Solver Verification (200 Designs, Seed 999) & $G_{xz}$ Bias Analysis
- **Overall Mean APE:** 3.12% across all 8 targets.
- **Shear Modulus $G_{xz}$ Investigation:** The mean signed percent error for $G_{xz}$ is -0.07%. On the FULL dataset (6,400 rows), the earlier 5-7% under-prediction bias is reduced to 0.07%.

---

## 4. Uncertainty Quantification Assessment
- **In-Distribution 90th Percentile Threshold:** 0.0670
- **Pearson Correlation ($r$):** 0.301
- **Spearman Rank Correlation ($\rho$):** 0.245
- **Honest Finding:** The correlation between ensemble disagreement and point-wise true error is **weak** within the in-distribution test set. Uncertainty scores rise sharply on out-of-distribution and held-out topologies, making the metric useful for detecting out-of-envelope domain shift, but not for reliably predicting exact point-wise errors within distribution.

---

## 5. Optimizer Comparison: Genetic Algorithm vs. Baseline Random Search
Both optimization methods were executed with an identical surrogate evaluation budget of **2,000 evaluations**:
- **GA Viable Candidates:** 1669 (263 on non-dominated Pareto front) | Wall Time: 101.40 s
- **Random Search Viable Candidates:** 13
- **Best Relative Density Found:** GA $\rho = 0.0958$ vs Random Search $\rho = 0.1874$
- **Diversity:** Top returned candidates represent 2 distinct topologies via farthest-point selection in normalized design space.
- **Boundary Diagnostics:** 0 of the top designs are within 2% of the training parameter envelope boundary.
- **First-Principles Solver Re-Check on Top 10 Designs:**
  - $E_z$ Mean Absolute Error: 2.79% (Signed Mean: +0.90%)
  - $G_{xz}$ Mean Absolute Error: 2.01% (Signed Mean: -1.23%)
  - Strength Mean Absolute Error: 3.05% (Signed Mean: +1.16%)

---

## 6. Benchmarks & Scalability
- **Single Solve:** Beam Solver = 5.22 ms | Single Surrogate Call = 5.63 ms | Single-call speedup: 0.93x
- **Batch Scaling:** Throughput scales from 156,148 designs/s at batch 1,000 to 1,346,560 designs/s at batch 1,000,000.
- **Baseline Scope:** The baseline solver is a reduced-order beam homogenization model, not solid continuum 3D FEA.

---

## 7. Scope Limits and What We Do NOT Claim
1. **Linear First-Failure Only:** Progressive post-yield plastic flow, buckling load shedding, and densification plateaus are NOT modeled.
2. **Reduced-Order Beam Kinematics:** Struts are modeled with 3D Euler-Bernoulli beam theory valid for slender struts ($r/L \le 0.15$). Strut intersection volume overlaps are neglected.
3. **Materials Database:** Except for the normalized reference material, all materials are UNVERIFIED bulk values. User must open primary datasheets before flight certification.
4. **Generalization Across Topologies:** Surrogates do not reliably predict properties for unobserved topological graphs.

---

## 8. NOT DONE and UNVERIFIED Items
- **Section 7 External Dataset Check:** Marked **NOT DONE (offline / rate-limited)** due to HTTP 429 response from the external ETH Zurich DOI resolver.
- **Materials Database:** Marked **UNVERIFIED - user must open the source and confirm** for Ti-6Al-4V, AlSi10Mg, 316L, and PA12.
- **Stress-Strain Response:** Marked **PARTIAL (linear to first failure only)**.

---

## 9. Manual Action Items (TODO for User)
1. **Materials Datasheet Confirmation:**
   - Open MMPDS-01 (or ASM Handbook) and verify bulk $E, \sigma_y, \rho$ for Ti-6Al-4V.
   - Open EOS AlSi10Mg datasheet and verify bulk properties.
   - Open ASTM F3184-16 for 316L Stainless Steel and confirm yield strength.
   - Open EOS PA 2200 datasheet, confirm laser-sintered part density (fill `density_kg_m3`), and set `verified: true` only after personal confirmation.
2. **Literature Citations:**
   - Verify Caulfield et al. (2007) in *Journal of Materials Processing Technology* 182, pp. 477-488.
   - Verify Deshpande, Fleck, Ashby (2001) for octet stiffness scaling before publishing.
3. **External Dataset Download:**
   - Download the ETH Zurich HyperCAN dataset manually in a browser via DOI `10.3929/ethz-b-000699994` (scripted access received HTTP 429).
