# AI-Generated Lattice Design and Property Prediction Backend
### Problem Statement 10 | AI Manthan 2.0 (Aerospace & Materials Science)

A verified scientific computing backend and prototype REST API for generating, evaluating, and optimizing micro-architected cellular metamaterial lattices.

---

## Quick Start: How to Run in 3 Commands

From the project root directory:

```bash
# Command 1: Install verified runtime dependencies
pip install -r requirements.txt

# Command 2: Execute the complete verified notebook top to bottom
jupyter nbconvert --to notebook --execute lattice_backend.ipynb --output lattice_backend.ipynb

# Command 3: Launch the interactive Jupyter environment
jupyter lab lattice_backend.ipynb
```

*(Optional: To run the prototype FastAPI service directly from terminal, execute the notebook cells in Section 13 or launch with `uvicorn`).*

---

## Which 3 Numbers You Should Check First

When evaluating the validity and performance of this backend, check these **3 critical numbers**:

1. **Analytical Solver Verification: diff = 0.00e+00 (Section 4 & 5)**  
   The periodic homogenization solver matches the analytical beam solution for Simple Cubic ($C_{11} = \rho/3$, $yield\_z = \rho/3$, $buckle\_z = \pi \rho^2 / 36$) to within machine floating-point precision ($< 10^{-9}$).
2. **Surrogate In-Distribution Accuracy: Mean MedAPE = 1.12% (Section 8)**  
   Evaluated on a stratified 15% held-out test split of the full 6,400-sample dataset, the primary surrogate achieves a minimum $R^2$ of 0.9949 and mean median absolute percent error of 1.12% across all 8 mechanical targets.
3. **Aerospace Core Mass Fraction: \rho = 0.0958 (90.4% Core Mass Saving) (Section 12)**  
   In the realistic satellite sandwich-panel core design scenario ($G_{xz} \ge 400$ MPa, $\sigma_z \ge 30$ MPa in Ti-6Al-4V), the multi-objective genetic algorithm identified an optimal OCTET configuration with relative density $\rho = 0.0958$, reducing core mass by 90.4% compared to a solid titanium block of identical thickness (valid because flight requirements are below solid capacity).

---

## Key Capabilities & System Architecture

| Component | Scientific Method | Verification Benchmark |
| :--- | :--- | :--- |
| **Geometry Engine** | Parametric unit cells for 8 topologies with affine Voigt stiffness and geometric invariants | Strut counts, coordination numbers, and descriptor uniqueness rigorously verified |
| **Physics Solver** | 3D Euler-Bernoulli beam elements with periodic microscopic boundary conditions | 10 of 10 automated verification checks passed (cantilever, symmetry, slopes) |
| **Strength Model** | First-yield (von Mises) and Euler strut buckling under uniaxial compression | Simple Cubic exact analytical match to $< 10^{-9}$ |
| **Surrogate AI** | Dual HistGradientBoosting + 5-Seed MLP Ensemble | In-distribution MedAPE = 1.12%, 200-sample fresh solver check MAPE = 3.12% |
| **Inverse Optimizer** | Multi-Objective Genetic Algorithm with tournament selection & Pareto front extraction | Evaluated against baseline random search at 2,000 evaluations; diverse candidates with boundary flags |
| **Prototype API** | FastAPI REST service with Pydantic validation and CAD JSON exports | 13 automated endpoint tests passed including error handling and solver fallback |

---

## Benchmark Scalability (relative to reduced-order beam solver)

- **Single Solve:** Solver = 3.85 ms | Surrogate = 2.17 ms | Single-call speedup: 1.78x
- **Batch Scaling:** Throughput reaches 1,485,580 designs/sec for batch size 1,000,000.
- *Note:* The baseline solver is a reduced-order beam model; speedup is measured relative to that model.
