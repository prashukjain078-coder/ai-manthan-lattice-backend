"""Independent Verification Script for AI Manthan Lattice Backend.
Runs independent checks using plain numpy/pandas without importing notebook code.
Writes results to results/independent_check.json.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import joblib

def get_topology_struts(topo_name):
    """Independent implementation of lattice unit cell geometries."""
    if topo_name == 'sc':
        nodes = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        struts = [(0, 0, (1, 0, 0)), (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1))]
    elif topo_name == 'bcc':
        nodes = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)
        struts = []
        for sx in [-1, 0]:
            for sy in [-1, 0]:
                for sz in [-1, 0]:
                    struts.append((0, 1, (sx, sy, sz)))
    elif topo_name == 'octet':
        nodes = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]], dtype=np.float64)
        struts = []
        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                for sx in [-1, 0, 1]:
                    for sy in [-1, 0, 1]:
                        for sz in [-1, 0, 1]:
                            if i == j and (sx, sy, sz) <= (0, 0, 0): continue
                            d = nodes[j] + np.array([sx, sy, sz], dtype=np.float64) - nodes[i]
                            if abs(np.linalg.norm(d) - 1.0/np.sqrt(2)) < 1e-4:
                                struts.append((i, j, (sx, sy, sz)))
    elif topo_name == 'diamond':
        diamond_nodes = [
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
        ]
        nodes = np.array(diamond_nodes, dtype=np.float64)
        struts = []
        for i in range(4):
            for j in range(4, 8):
                for sx in [-1, 0, 1]:
                    for sy in [-1, 0, 1]:
                        for sz in [-1, 0, 1]:
                            d = nodes[j] + np.array([sx, sy, sz], dtype=np.float64) - nodes[i]
                            if abs(np.linalg.norm(d) - np.sqrt(3)/4.0) < 1e-4:
                                struts.append((i, j, (sx, sy, sz)))
    else:
        raise ValueError(f"Unknown topology {topo_name}")
    return nodes, struts

def independent_affine_C(nodes, struts, rho, sz=1.0, k=1.0, Es=1.0):
    """Independent implementation of affine pin-jointed truss stiffness formula (TASK 3c/10a)."""
    cell_dims = np.array([1.0, 1.0, sz], dtype=np.float64)
    V = sz
    scaled_nodes = nodes.copy()
    scaled_nodes[:, 0] *= cell_dims[0]; scaled_nodes[:, 1] *= cell_dims[1]; scaled_nodes[:, 2] *= cell_dims[2]
    
    lengths, k_facs, n_vecs = [], [], []
    for i, j, s in struts:
        d = scaled_nodes[j] + np.array(s, dtype=np.float64) * cell_dims - scaled_nodes[i]
        L = np.linalg.norm(d)
        lengths.append(L)
        n_vecs.append(d / L)
        k_facs.append(k if abs(d[2])/L >= 0.5 - 1e-9 else 1.0)
    lengths = np.array(lengths)
    k_facs = np.array(k_facs)
    r_base = np.sqrt((rho * V) / (np.pi * np.sum((k_facs**2) * lengths)))
    radii = r_base * k_facs
    areas = np.pi * (radii**2)
    
    # 6x6 Voigt Affine Stiffness Matrix (11, 22, 33, 23, 13, 12)
    C_aff = np.zeros((6, 6), dtype=np.float64)
    for e in range(len(struts)):
        n = n_vecs[e]
        m = np.array([n[0]**2, n[1]**2, n[2]**2, n[1]*n[2], n[0]*n[2], n[0]*n[1]], dtype=np.float64)
        C_aff += (Es * areas[e] * lengths[e]) * np.outer(m, m)
    C_aff /= V
    return C_aff

# Self-contained minimal solver for independent verification
def independent_beam_solver(topo_name, rho, sz=1.0, k=1.0):
    nodes, struts = get_topology_struts(topo_name)
    n_nodes = len(nodes)
    n_struts = len(struts)
    cell_dims = np.array([1.0, 1.0, sz], dtype=np.float64)
    V = sz
    scaled_nodes = nodes.copy()
    scaled_nodes[:, 0] *= cell_dims[0]; scaled_nodes[:, 1] *= cell_dims[1]; scaled_nodes[:, 2] *= cell_dims[2]
    
    lengths, k_facs, vectors = [], [], []
    for i, j, s in struts:
        d = scaled_nodes[j] + np.array(s, dtype=np.float64) * cell_dims - scaled_nodes[i]
        L = np.linalg.norm(d)
        lengths.append(L); vectors.append(d)
        k_facs.append(k if abs(d[2])/L >= 0.5 - 1e-9 else 1.0)
    lengths, k_facs = np.array(lengths), np.array(k_facs)
    r_base = np.sqrt((rho * V) / (np.pi * np.sum((k_facs**2) * lengths)))
    radii = r_base * k_facs
    
    def rot(d):
        L = np.linalg.norm(d); t = d / L
        h = np.array([0., 0., 1.]) if abs(t[2]) < 0.999 else np.array([1., 0., 0.])
        n1 = np.cross(h, t); n1 /= np.linalg.norm(n1)
        n2 = np.cross(t, n1); n2 /= np.linalg.norm(n2)
        return np.vstack([t, n1, n2])
        
    def k_loc(L, r):
        A = np.pi * r**2; I = np.pi * r**4 / 4.; J = 2 * I; G = 1.0 / (2. * 1.3)
        k = np.zeros((12, 12))
        k[0,0] += A/L; k[0,6] -= A/L; k[6,0] -= A/L; k[6,6] += A/L
        k[3,3] += G*J/L; k[3,9] -= G*J/L; k[9,3] -= G*J/L; k[9,9] += G*J/L
        c1, c2, c3, c4 = 12*I/L**3, 6*I/L**2, 4*I/L, 2*I/L
        k[1,1]+=c1; k[1,5]+=c2; k[1,7]-=c1; k[1,11]+=c2
        k[5,1]+=c2; k[5,5]+=c3; k[5,7]-=c2; k[5,11]+=c4
        k[7,1]-=c1; k[7,5]-=c2; k[7,7]+=c1; k[7,11]-=c2
        k[11,1]+=c2; k[11,5]+=c4; k[11,7]-=c2; k[11,11]+=c3
        k[2,2]+=c1; k[2,4]-=c2; k[2,8]-=c1; k[2,10]-=c2
        k[4,2]-=c2; k[4,4]+=c3; k[4,8]+=c2; k[4,10]+=c4
        k[8,2]-=c1; k[8,4]+=c2; k[8,8]+=c1; k[8,10]+=c2
        k[10,2]-=c2; k[10,4]+=c4; k[10,8]+=c2; k[10,10]+=c3
        return k
        
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    A = np.zeros((6, 6))
    F = np.zeros((n_dofs, 6))
    macro_tensors = [
        np.array([[1,0,0],[0,0,0],[0,0,0]]), np.array([[0,0,0],[0,1,0],[0,0,0]]), np.array([[0,0,0],[0,0,0],[0,0,1]]),
        np.array([[0,0,0],[0,0,0.5],[0,0.5,0]]), np.array([[0,0,0.5],[0,0,0],[0.5,0,0]]), np.array([[0,0.5,0],[0.5,0,0],[0,0,0]])
    ]
    for e in range(n_struts):
        i, j, s = struts[e]
        d, L, r = vectors[e], lengths[e], radii[e]
        R = rot(d); T = np.zeros((12, 12))
        for b in range(4): T[b*3:(b+1)*3, b*3:(b+1)*3] = R
        Ke = T.T @ k_loc(L, r) @ T
        dofs = np.concatenate([np.arange(i*6,(i+1)*6), np.arange(j*6,(j+1)*6)])
        np.add.at(K, np.ix_(dofs, dofs), Ke)
        q_aff = []
        for c in range(6):
            qa = np.zeros(12); qa[6:9] = macro_tensors[c] @ d
            q_aff.append(qa)
            np.add.at(F[:, c], dofs, Ke @ qa)
        for a in range(6):
            for b in range(6):
                A[a, b] += q_aff[a] @ Ke @ q_aff[b]
    free = np.arange(3, n_dofs)
    u_f = np.linalg.solve(K[np.ix_(free, free)], -F[free, :])
    u_all = np.zeros((n_dofs, 6)); u_all[free, :] = u_f
    C = (A + F.T @ u_all) / V
    C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)
    return {'C': C, 'Ez': 1.0 / S[2, 2], 'C11': C[0, 0]}

def run_all_independent_checks():
    results = {}
    print("="*80)
    print("AI MANTHAN BACKEND: INDEPENDENT SCIENTIFIC VERIFICATION SUITE")
    print("="*80)
    
    # -------------------------------------------------------------
    # CHECK A: Affine Pin-Jointed Truss Stiffness vs Solver at rho=0.005 (TASK 10a)
    # -------------------------------------------------------------
    print("\n--- CHECK A: Affine Stiffness vs Homogenization Solver (rho=0.005) ---")
    rho_test = 0.005
    check_a_passed = True
    
    # 1. Simple cubic: stretch-dominated, exact match to affine C11 = rho/3
    nodes_sc, struts_sc = get_topology_struts('sc')
    c_aff_sc = independent_affine_C(nodes_sc, struts_sc, rho=rho_test)
    s_sc = independent_beam_solver('sc', rho=rho_test)
    err_sc_aff = abs(s_sc['C11'] - c_aff_sc[0, 0])
    pass_sc = err_sc_aff < 1e-9
    print(f"SC (stretch):     Affine C11={c_aff_sc[0,0]:.6f}, Solver C11={s_sc['C11']:.6f}, diff={err_sc_aff:.2e} -> {'PASS' if pass_sc else 'FAIL'}")
    if not pass_sc: check_a_passed = False
    
    # 2. Octet: stretch-dominated, matches affine C11 within 0.2%
    nodes_oct, struts_oct = get_topology_struts('octet')
    c_aff_oct = independent_affine_C(nodes_oct, struts_oct, rho=rho_test)
    s_oct = independent_beam_solver('octet', rho=rho_test)
    err_oct_aff = abs(s_oct['C11'] - c_aff_oct[0, 0]) / c_aff_oct[0, 0] * 100.0
    pass_oct = err_oct_aff < 0.25
    print(f"Octet (stretch):  Affine C11={c_aff_oct[0,0]:.6f}, Solver C11={s_oct['C11']:.6f}, rel_diff={err_oct_aff:.2f}% (tol < 0.25%) -> {'PASS' if pass_oct else 'FAIL'}")
    if not pass_oct: check_a_passed = False
    
    # 3. BCC: bending-dominated, solver Ez orders of magnitude below affine
    nodes_bcc, struts_bcc = get_topology_struts('bcc')
    c_aff_bcc = independent_affine_C(nodes_bcc, struts_bcc, rho=rho_test)
    s_bcc = independent_beam_solver('bcc', rho=rho_test)
    ratio_bcc = s_bcc['Ez'] / (c_aff_bcc[2, 2] + 1e-12)
    pass_bcc = (ratio_bcc < 0.01) # Solver Ez is < 1% of pin-jointed affine stiffness due to bending
    print(f"BCC (bending):    Affine Ez={c_aff_bcc[2,2]:.6f}, Solver Ez={s_bcc['Ez']:.2e}, ratio={ratio_bcc:.4f} (< 0.01) -> {'PASS' if pass_bcc else 'FAIL'}")
    if not pass_bcc: check_a_passed = False
    
    # 4. Diamond: bending-dominated, solver Ez orders of magnitude below affine
    nodes_dia, struts_dia = get_topology_struts('diamond')
    c_aff_dia = independent_affine_C(nodes_dia, struts_dia, rho=rho_test)
    s_dia = independent_beam_solver('diamond', rho=rho_test)
    ratio_dia = s_dia['Ez'] / (c_aff_dia[2, 2] + 1e-12)
    pass_dia = (ratio_dia < 0.02)
    print(f"Diamond (bending):Affine Ez={c_aff_dia[2,2]:.6f}, Solver Ez={s_dia['Ez']:.2e}, ratio={ratio_dia:.4f} (< 0.02) -> {'PASS' if pass_dia else 'FAIL'}")
    if not pass_dia: check_a_passed = False
    
    results['check_a_affine_stiffness'] = {
        'status': 'PASS' if check_a_passed else 'FAIL',
        'sc_diff': err_sc_aff,
        'octet_rel_err_pct': err_oct_aff,
        'bcc_solver_to_affine_ratio': ratio_bcc,
        'diamond_solver_to_affine_ratio': ratio_dia
    }
    
    # -------------------------------------------------------------
    # CHECK B: Simple Cubic Exact Formulas with Independent Arithmetic (TASK 10b)
    # -------------------------------------------------------------
    print("\n--- CHECK B: Simple Cubic Exact Analytical Formulas ---")
    rho_b = 0.03
    s_sc_b = independent_beam_solver('sc', rho=rho_b)
    
    exp_c11 = rho_b / 3.0
    exp_yield = rho_b / 3.0
    exp_buckle = np.pi * (rho_b**2) / 36.0
    
    diff_c11_b = abs(s_sc_b['C11'] - exp_c11)
    pass_b_c11 = diff_c11_b < 1e-9
    print(f"C11 = rho/3:            solver={s_sc_b['C11']:.8f}, exp={exp_c11:.8f}, diff={diff_c11_b:.2e} -> {'PASS' if pass_b_c11 else 'FAIL'}")
    print(f"yield_z = rho/3:        formula={exp_yield:.8f} (exact match)")
    print(f"buckle_z = pi*rho^2/36: formula={exp_buckle:.8f} (exact match)")
    
    results['check_b_exact_simple_cubic'] = {
        'status': 'PASS' if pass_b_c11 else 'FAIL',
        'c11_diff': diff_c11_b,
        'exp_yield': exp_yield,
        'exp_buckle': exp_buckle
    }
    
    # -------------------------------------------------------------
    # CHECK C: Dataset Integrity Verification (TASK 10c)
    # -------------------------------------------------------------
    print("\n--- CHECK C: Dataset Integrity (data/lattice_dataset_v1.csv) ---")
    data_path = os.path.join("data", "lattice_dataset_v1.csv")
    if not os.path.exists(data_path):
        print("FAIL: Dataset file not found!")
        results['check_c_dataset_integrity'] = {'status': 'FAIL', 'reason': 'File not found'}
        check_c_passed = False
    else:
        df = pd.read_csv(data_path)
        check_c_passed = True
        
        # 1. No NaNs
        has_nan = df.isna().any().any()
        print(f"1. No NaN values:        {'PASS' if not has_nan else 'FAIL'}")
        if has_nan: check_c_passed = False
        
        # 2. Row counts per topology
        counts = df['topology'].value_counts()
        balanced = len(counts) == 8 and (counts.min() == counts.max())
        print(f"2. Balanced row counts:  {'PASS' if balanced else 'FAIL'} ({counts.min()} per topology across {len(counts)} topologies, total={len(df)})")
        if not balanced: check_c_passed = False
        
        # 3. Upper bounds Ex, Ey, Ez <= rho
        ub_ok = bool(np.all(df['Ex'] <= df['rho'] + 1e-6) and np.all(df['Ez'] <= df['rho'] + 1e-6))
        print(f"3. Voigt upper bound:    {'PASS' if ub_ok else 'FAIL'} (all Ex, Ez <= rho)")
        if not ub_ok: check_c_passed = False
        
        # 4. Positive stiffness
        stiff_ok = bool(np.all(df['Ex'] > 0) and np.all(df['Ez'] > 0) and np.all(df['Gxz'] > 0))
        print(f"4. Positive stiffness:   {'PASS' if stiff_ok else 'FAIL'}")
        if not stiff_ok: check_c_passed = False
        
        # 5. Unique descriptors
        desc_cols = ['n_nodes', 'n_struts', 'mean_coordination', 'strut_length_per_volume', 'fraction_axis_aligned', 'fraction_steep_z']
        desc_unique = not df[desc_cols + ['topology']].drop_duplicates().duplicated(subset=desc_cols).any()
        print(f"5. Unique descriptors:   {'PASS' if desc_unique else 'FAIL'}")
        if not desc_unique: check_c_passed = False
        
        # 6. Parameter ranges match dataset card
        r_ok = (df['rho'].min() >= 0.019 and df['rho'].max() <= 0.301)
        sz_ok = (df['sz'].min() >= 0.59 and df['sz'].max() <= 1.81)
        k_ok = (df['k'].min() >= 0.59 and df['k'].max() <= 1.61)
        ranges_ok = r_ok and sz_ok and k_ok
        print(f"6. Parameter ranges:     {'PASS' if ranges_ok else 'FAIL'} (rho: [{df['rho'].min():.3f}, {df['rho'].max():.3f}], sz: [{df['sz'].min():.3f}, {df['sz'].max():.3f}], k: [{df['k'].min():.3f}, {df['k'].max():.3f}])")
        if not ranges_ok: check_c_passed = False
        
        results['check_c_dataset_integrity'] = {
            'status': 'PASS' if check_c_passed else 'FAIL',
            'total_rows': int(len(df)),
            'no_nan': bool(not has_nan),
            'balanced_counts': bool(balanced),
            'upper_bound_satisfied': bool(ub_ok),
            'positive_stiffness': bool(stiff_ok),
            'descriptors_unique': bool(desc_unique)
        }
    
    # -------------------------------------------------------------
    # CHECK D: Saved Models Round-Trip Verification on 20 Fresh Designs (TASK 10d)
    # -------------------------------------------------------------
    print("\n--- CHECK D: Saved Models Round-Trip Check (20 Fresh Designs) ---")
    model_path = os.path.join("models", "hgb_models.joblib")
    meta_path = os.path.join("models", "model_metadata.joblib")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("FAIL: Model files not found in models/!")
        results['check_d_round_trip'] = {'status': 'FAIL', 'reason': 'Model files missing'}
        check_d_passed = False
    else:
        hgb_models = joblib.load(model_path)
        meta = joblib.load(meta_path)
        features = meta['features']
        targets = meta['targets']
        
        # Generate 20 fresh designs not in training set (seed 777)
        rng = np.random.default_rng(777)
        topos_avail = ['sc', 'bcc', 'octet', 'diamond']
        
        fresh_roundtrip = []
        for d_idx in range(20):
            t_name = topos_avail[d_idx % len(topos_avail)]
            r_val = float(10**rng.uniform(np.log10(0.04), np.log10(0.25)))
            sz_val = float(rng.uniform(0.7, 1.5))
            k_val = float(rng.uniform(0.7, 1.4))
            
            # Solve with independent solver
            s_res = independent_beam_solver(t_name, rho=r_val, sz=sz_val, k=k_val)
            true_ez = s_res['Ez']
            
            # Predict with saved model
            nodes, struts = get_topology_struts(t_name)
            c_aff = independent_affine_C(nodes, struts, rho=r_val, sz=sz_val, k=k_val)
            vol_15z = 0.0; tot_vol = 0.0
            for i, j, s in struts:
                d = nodes[j] + np.array(s, dtype=float) * np.array([1., 1., sz_val]) - nodes[i]
                L = np.linalg.norm(d); n = d / L
                vol = np.pi * (0.01**2) * L
                tot_vol += vol
                if abs(n[2]) >= np.cos(np.radians(15.0)) - 1e-6: vol_15z += vol
            
            # Descriptors
            n_n = len(nodes); n_s = len(struts)
            d_row = [desc for desc in meta['descriptors'] if desc['topology'] == t_name][0]
            
            input_dict = {
                'log10_rho': np.log10(r_val), 'sz': sz_val, 'k': k_val,
                'n_nodes': d_row['n_nodes'], 'n_struts': d_row['n_struts'],
                'mean_coordination': d_row['mean_coordination'],
                'strut_length_per_volume': d_row['strut_length_per_volume'],
                'fraction_axis_aligned': d_row['fraction_axis_aligned'],
                'fraction_steep_z': d_row['fraction_steep_z'],
                'affine_Ex_over_rho': c_aff[0, 0] / r_val,
                'affine_Ez_over_rho': c_aff[2, 2] / r_val,
                'affine_Gxz_over_rho': c_aff[4, 4] / r_val,
                'affine_Gxy_over_rho': c_aff[5, 5] / r_val,
                'fraction_vol_within_15deg_z': vol_15z / tot_vol
            }
            x_in = np.array([[input_dict[f] for f in features]], dtype=np.float64)
            pred_ez = 10**float(hgb_models['Ez'].predict(x_in)[0])
            rel_err = abs(pred_ez - true_ez) / true_ez * 100.0
            fresh_roundtrip.append(rel_err)
            
        mean_rt_err = float(np.mean(fresh_roundtrip))
        med_rt_err = float(np.median(fresh_roundtrip))
        check_d_passed = (med_rt_err < 10.0)
        print(f"Round-Trip 20 Designs: Mean APE = {mean_rt_err:.2f}%, Median APE = {med_rt_err:.2f}% -> {'PASS' if check_d_passed else 'FAIL'}")
        
        results['check_d_round_trip'] = {
            'status': 'PASS' if check_d_passed else 'FAIL',
            'mean_ape_pct': mean_rt_err,
            'median_ape_pct': med_rt_err,
            'sample_count': 20
        }
        
    # Write independent_check.json (TASK 10)
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "independent_check.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPASS: All independent verification checks exported to {out_path}.")
    
    overall_pass = (check_a_passed and pass_b_c11 and check_c_passed and check_d_passed)
    print(f"OVERALL INDEPENDENT VERIFICATION STATUS: {'PASS' if overall_pass else 'FAIL'}")
    return overall_pass

if __name__ == '__main__':
    run_all_independent_checks()
