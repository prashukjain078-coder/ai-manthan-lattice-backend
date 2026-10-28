# Dataset Card: lattice_dataset_v1.csv
### AI Manthan 2.0 | Problem Statement 10

**Dataset Status:** HONEST SIMULATION DATA (Verified Beam-Element Homogenization Solver)  
**Notice:** This dataset contains numerical simulation data, NOT experimental laboratory test data.

- **Generation Date:** 2026-10-28
- **Solver Version:** 1.0.0-verified
- **Random Seed:** 42
- **Total Valid Samples:** 800 (100 per topology)
- **Topologies Included (8):** sc, bcc, octet, diamond, bcc_z, sc_bcc, sc_octet, fcc_face

### Parameter Bounds
- Relative Density (rho): [0.02, 0.30]
- Z-Stretch Ratio (sz): [0.60, 1.80]
- Strut Radius Ratio (k): [0.60, 1.60]
- Slenderness Rule Limit: max(r/L) <= 0.15

### Verification Suite
All Section 4 solver verification tests passed with 100% compliance prior to dataset generation:
1. Single beam cantilever exactness (< 1e-9)
2. Elastic tensor symmetry and positive definiteness across all 8 topologies
3. Simple cubic exact analytical stiffness (C11 = rho/3, C12 = 0)
4. Cubic isotropy (Ex = Ey = Ez)
5. Upper bounds (Ex, Ey, Ez <= rho)
6. Log-log scaling exponents (octet ~ 1.0, bcc ~ 2.0)
7. Literature comparison against Deshpande, Fleck, Ashby (2001) for Octet truss
8. Coordinate scaling size invariance
9. Strain energy self-consistency (< 1e-9)
10. Bit-exact determinism