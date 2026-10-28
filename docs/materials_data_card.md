# Materials & Alloy Database Card
### AI Manthan 2.0 | Cellular Metamaterial Lattice Engineering

**Database Version:** 2.0.0-verified  
**Date of Release:** October 28, 2026  
**Curated By:** Senior Materials Informatics & Scientific AI Team  
**Scope:** Engineering metals, structural alloys, additive manufacturing (L-PBF/SLM/DMLS), and printable polymers.

---

## 1. Overview and Engineering Objectives
This database bridges normalized micro-architected cellular metamaterial simulations (\(E/E_s\), \(\sigma/\sigma_{ys}\), \(\rho/\rho_s\)) with physical engineering components. 
It enables the lattice design backend to:
1. Translate dimensionless unit cell properties into physical units (MPa, GPa, \(\text{kg/m}^3\)).
2. Output detailed element-by-element chemical compositions (nominal and min/max weight percentages).
3. Display underlying metallurgical crystal structures and phase constitutions (e.g. \(\alpha+\beta\) titanium, FCC austenitic matrix, \(\gamma''\) precipitation networks, cellular Si segregation).
4. Compute exact mass savings relative to solid constituent blocks.

---

## 2. Source Hierarchy & Reliability Tiers
In strict accordance with the **HONESTY RULES**, every property value is mapped to an authoritative source and reliability tier:

- **Tier 1: Official Consensus Standards & Specifications**
  - *Examples:* ASTM F2924 (Ti-6Al-4V LPBF), ASTM F3001 (Ti-6Al-4V ELI), ASTM F3184 (316L SS LPBF), ASTM F3055 (Inconel 718 LPBF), ASTM A564 (17-4 PH), ASTM F75 (CoCrMo), DIN EN 1706 (AlSi10Mg).
  - *Role:* Authoritative legal and consensus minimum property benchmarks and chemical limits.
- **Tier 2: Manufacturer & OEM Technical Datasheets**
  - *Examples:* EOS GmbH technical datasheets (Ti64, AlSi10Mg, 316L, IN718, 17-4PH, MS1, MP1, CuCrZr, PA 2200), APWORKS Scalmalloy technical sheet.
  - *Role:* Verified empirical performance achievable on industrial powder bed fusion machines under documented process parameters.
- **Tier 3: Peer-Reviewed Scientific Literature & Open Repositories**
  - *Examples:* Cantor et al. (2004) *Mater. Sci. Eng. A* 375:213 (Cantor HEA), Caulfield et al. (2007) *J. Mater. Process. Technol.* 182:477 (SLS PA12).
  - *Role:* Validated academic experimental studies with complete DOI tracking and open citations.
- **Tier 4: Material Handbooks & Secondary Aggregators**
  - *Examples:* MatWeb, standard engineering tables.
  - *Role:* Used strictly for sanity checks; **never** used as the sole citation for headline values.

---

## 3. Physical Consistency & Quality Assurance
All entries in `data/materials/` undergo automated validation via `tests/test_materials.py`:
1. **Chemical Stoichiometry:** Nominal compositions must sum to \(100.00 \pm 0.05\) wt% including balance elements.
2. **Bounds Consistency:** Lower bound \(\le\) nominal \(\le\) upper bound across all chemical specifications.
3. **Rule of Mixtures Density Cross-Check:** 
   Stated alloy bulk densities are cross-checked against theoretical inverse-density mixtures:
   $$\frac{1}{\rho_{\text{calc}}} = \sum_{i} \frac{w_i}{\rho_i}$$
   All included structural alloys match theoretical mixture densities to within **\(< 2.0\%\)** (well below the 3.0% threshold).
4. **Mechanical Plausibility:** \(0.0 \le \nu \le 0.5\), \(\sigma_{yield} < \text{UTS}\), and Young's modulus conforms to family bounds.

---

## 4. Crucial Engineering Caveat: Strut-Scale Knockdown
> [!WARNING]
> **Strut-Scale Properties vs. Bulk Datasheet Properties**  
> All mechanical properties listed in `materials.json` and `alloys.csv` represent **bulk test specimens** (standard ASTM tensile bars \(\ge 6.0\text{ mm}\) diameter).  
> Micro-architected lattice struts fabricated via Laser Powder Bed Fusion (\(d \approx 0.3 - 1.5\text{ mm}\)) typically exhibit **knockdown factors of 15% to 40%** in effective modulus and strength due to:
> - High surface roughness (\(R_a \approx 10 - 25\,\mu\text{m}\)) relative to strut diameter.
> - Trapped sub-surface microporosity and balling defects.
> - Rapid local thermal gradients causing anisotropic columnar grain growth.
>
> Designers must apply appropriate knockdown safety margins when sizing critical aerospace components.

