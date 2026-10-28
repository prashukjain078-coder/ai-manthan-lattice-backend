# Dataset Card: lattice_dataset_v1.csv
### AI Manthan 2.0 | Problem Statement 10

**Dataset Status:** simulation data from a verified beam-element solver; not experimental  
**Label:** Verified Simulation Data (3D Euler-Bernoulli Periodic Homogenization)

- **Generation Date:** 2026-10-28
- **Total Valid Configurations:** 6,400
- **Configurations per Topology:** 800 (balanced across all 8 topologies)
- **Topologies Included:** ['sc', 'bcc', 'octet', 'diamond', 'bcc_z', 'sc_bcc', 'sc_octet', 'fcc_face']
- **Parameter Exploration:**
  - Relative density $\rho$: [0.0200, 0.2999]
  - Z-cell stretch ratio $s_z$: [0.6001, 1.7998]
  - Strut radius ratio $k$: [0.6000, 1.5998]
- **Physical Bounds Enforced:**
  - Slenderness limit: $\max(r/L) \le 0.15$
  - Positive definiteness: strictly checked on every sample
  - Hashin-Shtrikman / Voigt upper bound: $E_x, E_y, E_z \le \rho$ strictly verified
- **Derived First-Failure Outputs:**
  - `failure_stress_z`: first-yield or Euler buckling capacity
  - `failure_strain_z`: failure_stress_z / Ez (linear first-failure strain)
- **Random Seed:** 42 (reproducible Sobol quasi-random sequence)
- **Failed Solves / Rejections:** 0
- **Runtime:** 18.13 seconds (FULL mode)
