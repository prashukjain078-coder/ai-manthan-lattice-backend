import nbformat

nb = nbformat.read('lattice_backend.ipynb', as_version=4)
cell37 = nb.cells[37].source

# 1. Update optimizer dict in headline_numbers
old_opt = """    'optimizer': {
        'ga_total_evaluations': int(opt_res['total_evaluations']),
        'ga_viable_candidates': int(opt_res['n_survivors']),
        'ga_pareto_candidates': int(opt_res['n_pareto']),
        'ga_wall_time_s': float(round(t_opt_ga, 2)),
        'random_evaluations': int(res_random['total_evaluations']),
        'random_viable_candidates': int(res_random['n_survivors']),
        'random_min_rho': float(round(res_random['min_rho_found'], 4)) if res_random['min_rho_found'] else None,"""

new_opt = """    'optimizer': {
        'ga_actual_evaluations': int(opt_res.get('ga_evaluations_actual', opt_res['total_evaluations'])),
        'ga_total_evaluations': int(opt_res.get('ga_evaluations_actual', opt_res['total_evaluations'])),
        'ga_viable_candidates': int(opt_res['n_survivors']),
        'ga_distinct_viable': int(opt_res.get('n_distinct_feasible', opt_res['n_survivors'])),
        'ga_pareto_candidates': int(opt_res['n_pareto']),
        'ga_wall_time_s': float(round(t_opt_ga, 2)),
        'random_evaluations': int(res_random['total_evaluations']),
        'random_viable_candidates': int(res_random['n_survivors']),
        'random_distinct_viable': int(res_random.get('n_distinct_feasible', res_random['n_survivors'])),
        'random_min_rho': float(round(res_random['min_rho_found'], 4)) if res_random['min_rho_found'] else None,"""

assert old_opt in cell37, 'old_opt block not found in Cell 37'
cell37 = cell37.replace(old_opt, new_opt, 1)

# 2. In report_template, reorder Section 3 so Held-Out is 3.1 and In-Distribution is 3.2
old_report_sec3_and_5 = """## 3. Surrogate Accuracy & Generalization

### 3.1 In-Distribution Test Set Accuracy (FULL Mode: 6,400 Rows)
Evaluated on a 15% stratified test split ({int(headline_numbers['dataset']['total_rows']*0.15):,} configurations):
- **Primary Model:** {headline_numbers['surrogate_in_distribution']['primary_model']}
- **Minimum $R^2$ across all targets:** {headline_numbers['surrogate_in_distribution']['hgb_min_r2']:.4f}
- **Mean Absolute Percent Error (MAPE):** {headline_numbers['surrogate_in_distribution']['hgb_mean_mape']:.2f}%
- **Mean Median Absolute Percent Error (MedAPE):** {headline_numbers['surrogate_in_distribution']['hgb_mean_medape']:.2f}%

| Target | HGB $R^2$ | HGB MedAPE (%) | HGB MAPE (%) | HGB MSPE (%) |
| :--- | :--- | :--- | :--- | :--- |
\"\"\"
for row in headline_numbers['surrogate_in_distribution']['per_target_table']:
    report_template += f"| **{row['Target']}** | {row['HGB R2']:.4f} | {row['HGB MedAPE (%)']:.2f}% | {row['HGB MAPE (%)']:.2f}% | {row['HGB MSPE (%)']:+.2f}% |\\n"

report_template += f\"\"\"
### 3.2 Held-Out Topology Generalization (Leave-One-Topology-Out)
Surrogates were evaluated by holding out an entire topology during training. We compare baseline geometric descriptors against models augmented with transferrable affine Voigt physics features:

| Held-Out Topology | Base $R^2$ | Base MedAPE (%) | Phys $R^2$ (with Physics) | Phys MedAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
\"\"\"
for row in headline_numbers['surrogate_held_out_topology']['loto_table']:
    report_template += f"| **{row['Held-Out Topology']}** | {row['Base R2']:.3f} | {row['Base MedAPE (%)']:.1f}% | {row['Phys R2']:.3f} | {row['Phys MedAPE (%)']:.1f}% |\\n"

report_template += f\"\"\"
**Honest Generalization Assessment:**  
Surrogate models are **accurate inside the trained topology set, but unreliable for unseen topologies**. While transferrable affine physics features noticeably improve extrapolation for diamond and hybrid lattices, architectures with unique bending modes (such as BCC) exhibit high error when unobserved in the training set.

### 3.3 Fresh Solver Verification (200 Designs, Seed 999) & $G_{{xz}}$ Bias Analysis
- **Overall Mean APE:** {headline_numbers['surrogate_fresh_solver_check']['mean_mape']:.2f}% across all 8 targets.
- **Shear Modulus $G_{{xz}}$ Investigation:** The mean signed percent error for $G_{{xz}}$ is {headline_numbers['surrogate_fresh_solver_check']['gxz_mean_signed_error']:+.2f}%. On the FULL dataset ({headline_numbers['dataset']['total_rows']:,} rows), the earlier 5-7% under-prediction bias is reduced to {abs(headline_numbers['surrogate_fresh_solver_check']['gxz_mean_signed_error']):.2f}%.

---

## 4. Uncertainty Quantification Assessment
- **In-Distribution 90th Percentile Threshold:** {headline_numbers['uncertainty']['threshold_90th']:.4f}
- **Pearson Correlation ($r$):** {headline_numbers['uncertainty']['pearson_correlation']:.3f}
- **Spearman Rank Correlation ($\\\\rho$):** {headline_numbers['uncertainty']['spearman_correlation']:.3f}
- **Honest Finding:** The correlation between ensemble disagreement and point-wise true error is **weak** within the in-distribution test set. Uncertainty scores rise sharply on out-of-distribution and held-out topologies, making the metric useful for detecting out-of-envelope domain shift, but not for reliably predicting exact point-wise errors within distribution.

---

## 5. Optimizer Comparison: Genetic Algorithm vs. Baseline Random Search
Both optimization methods were executed with an identical surrogate evaluation budget of **{headline_numbers['optimizer']['ga_total_evaluations']:,} evaluations**:
- **GA Viable Candidates:** {headline_numbers['optimizer']['ga_viable_candidates']} ({headline_numbers['optimizer']['ga_pareto_candidates']} on non-dominated Pareto front) | Wall Time: {headline_numbers['optimizer']['ga_wall_time_s']:.2f} s
- **Random Search Viable Candidates:** {headline_numbers['optimizer']['random_viable_candidates']}
- **Best Relative Density Found:** GA $\\\\rho = {headline_numbers['aerospace_sandwich_core']['relative_density']:.4f}$ vs Random Search $\\\\rho = {headline_numbers['optimizer']['random_min_rho'] if headline_numbers['optimizer']['random_min_rho'] else 'None (infeasible)'}$"""

new_report_sec3_and_5 = """## 3. Surrogate Accuracy & Generalization (Held-Out Limits Stated First)

### 3.1 Held-Out Topology Generalization (Leave-One-Topology-Out Extrapolation Limits)
Surrogates were evaluated by holding out an entire topology during training. We compare baseline geometric descriptors against models augmented with transferrable affine Voigt physics features:

| Held-Out Topology | Base $R^2$ | Base MedAPE (%) | Phys $R^2$ (with Physics) | Phys MedAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
\"\"\"
for row in headline_numbers['surrogate_held_out_topology']['loto_table']:
    report_template += f"| **{row['Held-Out Topology']}** | {row['Base R2']:.3f} | {row['Base MedAPE (%)']:.1f}% | {row['Phys R2']:.3f} | {row['Phys MedAPE (%)']:.1f}% |\\n"

report_template += f\"\"\"
**Honest Generalization Assessment:**  
Surrogate models are **accurate inside the trained topology set, but unreliable for unseen topologies**. While transferrable affine physics features noticeably improve extrapolation for diamond and hybrid lattices, architectures with unique bending modes (such as BCC) exhibit high error when unobserved in the training set.

### 3.2 In-Distribution Test Set Accuracy (FULL Mode: 6,400 Rows)
Evaluated on a 15% stratified test split ({int(headline_numbers['dataset']['total_rows']*0.15):,} configurations):
- **Primary Model:** {headline_numbers['surrogate_in_distribution']['primary_model']}
- **Minimum $R^2$ across all targets:** {headline_numbers['surrogate_in_distribution']['hgb_min_r2']:.4f}
- **Mean Absolute Percent Error (MAPE):** {headline_numbers['surrogate_in_distribution']['hgb_mean_mape']:.2f}%
- **Mean Median Absolute Percent Error (MedAPE):** {headline_numbers['surrogate_in_distribution']['hgb_mean_medape']:.2f}%

| Target | HGB $R^2$ | HGB MedAPE (%) | HGB MAPE (%) | HGB MSPE (%) |
| :--- | :--- | :--- | :--- | :--- |
\"\"\"
for row in headline_numbers['surrogate_in_distribution']['per_target_table']:
    report_template += f"| **{row['Target']}** | {row['HGB R2']:.4f} | {row['HGB MedAPE (%)']:.2f}% | {row['HGB MAPE (%)']:.2f}% | {row['HGB MSPE (%)']:+.2f}% |\\n"

report_template += f\"\"\"
### 3.3 Fresh Solver Verification (200 Designs, Seed 999) & $G_{{xz}}$ Bias Analysis
- **Overall Mean APE:** {headline_numbers['surrogate_fresh_solver_check']['mean_mape']:.2f}% across all 8 targets.
- **Shear Modulus $G_{{xz}}$ Investigation:** The mean signed percent error for $G_{{xz}}$ is {headline_numbers['surrogate_fresh_solver_check']['gxz_mean_signed_error']:+.2f}%. On the FULL dataset ({headline_numbers['dataset']['total_rows']:,} rows), the earlier 5-7% under-prediction bias is reduced to {abs(headline_numbers['surrogate_fresh_solver_check']['gxz_mean_signed_error']):.2f}%.

---

## 4. Uncertainty Quantification Assessment
- **In-Distribution 90th Percentile Threshold:** {headline_numbers['uncertainty']['threshold_90th']:.4f}
- **Pearson Correlation ($r$):** {headline_numbers['uncertainty']['pearson_correlation']:.3f}
- **Spearman Rank Correlation ($\\\\rho$):** {headline_numbers['uncertainty']['spearman_correlation']:.3f}
- **Honest Finding:** The correlation between ensemble disagreement and point-wise true error is **weak** within the in-distribution test set. Uncertainty scores rise sharply on out-of-distribution and held-out topologies, making the metric useful for detecting out-of-envelope domain shift, but not for reliably predicting exact point-wise errors within distribution.

---

## 5. Optimizer Comparison: Genetic Algorithm vs. Baseline Random Search
Both optimization methods were executed with an identical actual surrogate evaluation budget of **{headline_numbers['optimizer']['ga_actual_evaluations']:,} evaluations**:
- **GA Actual Evaluations:** {headline_numbers['optimizer']['ga_actual_evaluations']:,} | Viable in Final Population: {headline_numbers['optimizer']['ga_viable_candidates']} (Distinct: {headline_numbers['optimizer']['ga_distinct_viable']}, Pareto Front: {headline_numbers['optimizer']['ga_pareto_candidates']}) | Wall Time: {headline_numbers['optimizer']['ga_wall_time_s']:.2f} s
- **Random Search Evaluations:** {headline_numbers['optimizer']['random_evaluations']:,} | Viable Candidates: {headline_numbers['optimizer']['random_viable_candidates']:,} (Distinct: {headline_numbers['optimizer']['random_distinct_viable']:,})
- **Best Relative Density Found:** GA $\\\\rho = {headline_numbers['aerospace_sandwich_core']['relative_density']:.4f}$ vs Random Search $\\\\rho = {headline_numbers['optimizer']['random_min_rho'] if headline_numbers['optimizer']['random_min_rho'] else 'None (infeasible)'}$ (Difference: {abs(headline_numbers['aerospace_sandwich_core']['relative_density'] - (headline_numbers['optimizer']['random_min_rho'] or 0.0)):.4f})
- **Honest Comparative Assessment:** At an identical actual evaluation budget ({headline_numbers['optimizer']['ga_actual_evaluations']:,} evaluations), the GA converged to a slightly lighter feasible core ($\\\\rho = {headline_numbers['aerospace_sandwich_core']['relative_density']:.4f}$ vs $\\\\rho = {headline_numbers['optimizer']['random_min_rho']:.4f}$). However, Random Search identified thousands of viable designs ({headline_numbers['optimizer']['random_viable_candidates']:,}), disproving the earlier artifact where Random Search was under-budgeted at only 2,000 evaluations."""

assert old_report_sec3_and_5 in cell37, 'old_report_sec3_and_5 not found in Cell 37'
cell37 = cell37.replace(old_report_sec3_and_5, new_report_sec3_and_5, 1)

nb.cells[37].source = cell37
nbformat.write(nb, 'lattice_backend.ipynb')
print('PASS: Cell 37 updated with honest evaluation accounting and held-out-first reporting!')

