# Changelog: Review Feedback Refactoring & Verification

All changes in this release directly resolve review findings with rigorous physical verification, code-driven documentation, and strict data honesty.

---

### 1. Dataset: FULL Mode Execution & Real Counts (Task 1)
- **Review Finding:** Dataset was limited to 800 rows (FAST mode), and numbers did not reflect the full parameter space.
- **Fix:** Switched global runtime to `MODE = "FULL"` (800 samples per topology across all 8 topologies = 6,400 verified configurations).
- **Execution:** Regenerated `data/lattice_dataset_v1.csv` with 6,400 rows, tracking real generation time and zero solver rejections.
- **Dataset Card:** Updated `data/dataset_card.md` with exact row counts, parameter ranges, random seed (42), generation timestamp, and explicit label: *"simulation data from a verified beam-element solver; not experimental"*.

### 2. Materials Verification & Citation Hygiene (Task 2)
- **Review Finding:** Claimed all materials were "verified", had wrong citations (e.g. Caulfield 2007 in Rapid Prototyping Journal), and PA12 solid density was unconfirmed.
- **Fix:** Rewrote `materials.json`:
  - All entries except normalized reference marked `"verified": false` and `"verification_status": "UNVERIFIED - user must open the source and confirm"`.
  - Corrected citation for Caulfield et al. (2007) to *Journal of Materials Processing Technology* 182, pp. 477-488. Other citations explicitly designated as "citation unconfirmed".
  - PA12 density set to `null` with `"density_notes": "NEEDS CHECK against the EOS PA 2200 datasheet"`. Code explicitly refuses to compute absolute mass for PA12 until confirmed.
  - Prominent warnings logged on material retrieval; all API responses return `"material_verified": false` and warning strings for unverified records.
  - Added Manual Action Items to `VERIFICATION_REPORT.md` specifying exact datasheets the user must inspect.

### 3. Surrogate Generalization, Signed Errors & Uncertainty (Task 3)
- **Review Finding:** Overly optimistic claims; potential systematic under-prediction of shear modulus $G_{xz}$; weak uncertainty correlation called "confirmation".
- **Fix:**
  - Added signed percent error (MSPE) metrics across all targets on in-distribution test set and 200-sample fresh solver check.
  - Investigated $G_{xz}$ signed bias on the FULL 6,400 dataset and reported honest findings.
  - Added transferrable physics features ($C_{affine}$ Voigt stiffness estimates, alignment within 15 degrees of z, coordination). Retrained models and reported Leave-One-Topology-Out (LOTO) side-by-side with and without physics features.
  - Recomputed uncertainty correlation using BOTH Pearson ($r$) and Spearman ($\rho$). Explicitly reported as a weak correlation within distribution, useful for domain shift detection on held-out topologies but not for point-wise error confirmation.
  - Guarded `/predict` and `/design` to fall back to the real solver (`solver_only: true`) for registered-but-untrained topologies.

### 4. Genetic Algorithm Optimizer, Diversity & Boundary Diagnostics (Task 4)
- **Review Finding:** 200k random search returned near-duplicate octet designs; arbitrary "92% mass reduction" claim; lacked real optimization.
- **Fix:**
  - Implemented multi-objective Genetic Algorithm (GA) with tournament selection, arithmetic crossover, Gaussian mutation, and Deb's constraint handling across multiple objective weights.
  - Kept random search strictly as a baseline at identical evaluation budget (216,000 evaluations) and reported honest comparative findings.
  - Implemented farthest-point selection in normalized design space to ensure diversity among top candidates.
  - Added boundary diagnostics flagging any design within 2% of the training parameter envelope.
  - Implemented full 10-candidate first-principles solver re-check with signed and absolute error averages computed in code.
  - Replaced the hardcoded mass claim with the exact computed relative density mass saving, stating plainly that mass savings are valid only because flight requirements are below solid titanium capacity.

### 5. Stress-Strain First-Failure Modeling (Task 5)
- **Review Finding:** Stress-strain behavior was incomplete and unclarified.
- **Fix:**
  - Derived first-failure outputs: $failure\_stress\_z = \min(yield\_z, buckle\_z)$ and $failure\_strain\_z = failure\_stress\_z / E_z$. Added to dataset and surrogate predictions.
  - Plotted idealized stress-strain response (linear elastic up to first failure, dashed line beyond labeled *"post-failure behavior NOT modeled"*).
  - Explicitly stated that progressive post-failure plasticity is not modeled; self-check marked `PARTIAL (linear to first failure only)`.

### 6. Benchmark Speedup Claims & Machine Metadata (Task 6)
- **Review Finding:** The ">10,000x" claim was unsupported by measured single-call numbers and lacked baseline context.
- **Fix:**
  - Re-measured solver single call, surrogate single call, and batch throughput across $10^3, 10^4, 10^5, 10^6$ candidates.
  - Computed speedup in code as $T_{solver} / T_{surrogate}$ for each batch size and for single call.
  - Measured end-to-end optimizer wall time including candidate generation and filtering.
  - Added explicit notice that baseline solver is a reduced-order beam homogenization model (milliseconds), so speedup is relative to that, not solid continuum 3D FEA.
  - Exported CPU, OS, Python, and package versions to `data/benchmark.json`.

### 7. Prototype REST API Robustness (Task 7)
- **Review Finding:** Labeled as "Production" without sufficient input error testing.
- **Fix:**
  - Renamed service to "prototype REST API" everywhere.
  - Implemented and verified 13 automated endpoint test cases: unknown topology (400), negative/zero rho (422), rho above cap (422), tiles capped (200), wrong types (422), missing fields (422), unverified materials (warning + flag), infeasible `/design` (200 with diagnostics), and registered-but-untrained solver fallback (`solver_only: true`).

### 8. Documentation Consistency & Forbidden Phrase Elimination (Task 8)
- **Review Finding:** Marketing language and numbers typed by hand created discrepancies.
- **Fix:**
  - Created `results/headline_numbers.json` written directly by the notebook at runtime.
  - Dynamic documentation generation: `README_BACKEND.md` and `VERIFICATION_REPORT.md` are dynamically compiled from `results/headline_numbers.json` by notebook code.
  - Banned forbidden phrases: *"sub-percent"*, *"over 10,000x"*, *"mathematically exact"*, *"production"*, *"verified materials database"*, *"confirming"*, *"92% mass reduction"*, *"R2 > 0.993 for all"*.
  - External dataset check (Section 7) kept as `NOT DONE (offline / rate-limited)`.

### 9. Environment, Hygiene & Pinned Requirements (Task 9)
- **Review Finding:** Unpinned dependencies, missing jupyterlab, and absolute paths in files.
- **Fix:**
  - Replaced all absolute local paths across the entire repo with relative paths.
  - Pinned all dependencies in `requirements.txt` with exact `==` versions, including `jupyterlab`, `nbconvert`, `nbformat`, and `ipykernel`.
  - Added clean `.venv_verify` testing workflow.

### 10. Independent Verification Script (Task 10)
- **Review Finding:** Need independent verification decoupled from the notebook runtime.
- **Fix:**
  - Created `verify_independent.py` using plain numpy/pandas.
  - Validates affine stiffness bounds at $\rho=0.005$, Simple Cubic analytical formulas, dataset integrity (6,400 rows, no NaNs, unique descriptors), and 20-sample round-trip solver check against saved models.
  - Writes results to `results/independent_check.json`.
