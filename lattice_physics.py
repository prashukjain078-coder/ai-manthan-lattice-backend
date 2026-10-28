"""Core Lattice Physics and Geometry Engine for AI Manthan 2.0.

Provides verified mathematical models for:
1. 8 fundamental cellular metamaterial topologies (parametrically scaled).
2. Exact 3D Euler-Bernoulli beam periodic homogenization solver.
3. First-failure analytical and numerical strength calculations (von Mises yield + Euler buckling).
4. Transferrable physics-informed descriptor extraction.
"""
import math
import numpy as np
import pandas as pd

COS_15_DEG = float(np.cos(np.radians(15.0)))
SLENDER_MAX = 0.15
SOLID_POISSON = 0.30
SOLVER_VERSION = "1.0.0-verified"

def build_topologies():
    """Define the 8 fundamental lattice topologies as a clean data dictionary."""
    topos = {}
    topos['sc'] = {'nodes': np.array([[0.0, 0.0, 0.0]], dtype=np.float64), 'rule': 'sc'}
    topos['bcc'] = {'nodes': np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64), 'rule': 'nn'}
    topos['octet'] = {'nodes': np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]], dtype=np.float64), 'rule': 'nn'}
    diamond_nodes = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
    ]
    topos['diamond'] = {'nodes': np.array(diamond_nodes, dtype=np.float64), 'rule': 'nn'}
    topos['bcc_z'] = {'nodes': np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64), 'rule': 'bcc_z'}
    topos['sc_bcc'] = {'nodes': np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64), 'rule': 'sc_bcc'}
    topos['sc_octet'] = {'nodes': np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]], dtype=np.float64), 'rule': 'sc_octet'}
    topos['fcc_face'] = {'nodes': np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]], dtype=np.float64), 'rule': 'fcc_face'}
    return topos

TOPOLOGIES_REGISTRY = build_topologies()

def find_struts(topo_name, topo_info=None):
    """Find unique periodic struts using nearest-neighbor search or explicit topology rules."""
    if topo_info is None:
        topo_info = TOPOLOGIES_REGISTRY[topo_name]
    if 'struts' in topo_info:
        return topo_info['struts']
    nodes = topo_info['nodes']
    n_nodes = len(nodes)
    rule = topo_info['rule']
    if rule == 'sc':
        return [(0, 0, (1, 0, 0)), (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1))]
    shifts = [(sx, sy, sz) for sx in [-1, 0, 1] for sy in [-1, 0, 1] for sz in [-1, 0, 1]]
    candidates = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            for s in shifts:
                if i == j and s == (0, 0, 0): continue
                d = nodes[j] + np.array(s, dtype=np.float64) - nodes[i]
                candidates.append((np.linalg.norm(d), i, j, s))
    candidates.sort(key=lambda x: x[0])
    min_dist = candidates[0][0]
    nn_struts = []
    seen = set()
    for dist, i, j, s in candidates:
        if abs(dist - min_dist) < 1e-5:
            neg_s = (-s[0], -s[1], -s[2])
            if (j, i, neg_s) not in seen and (i, j, s) not in seen:
                seen.add((i, j, s))
                nn_struts.append((i, j, s))
    if rule == 'nn':
        return nn_struts
    elif rule == 'bcc_z':
        struts = list(nn_struts)
        struts.append((0, 0, (0, 0, 1)))
        return struts
    elif rule == 'sc_bcc':
        struts = list(nn_struts)
        struts.extend([(0, 0, (1, 0, 0)), (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1))])
        return struts
    elif rule == 'sc_octet':
        struts = list(nn_struts)
        struts.extend([(0, 0, (1, 0, 0)), (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1))])
        return struts
    elif rule == 'fcc_face':
        return [st for st in nn_struts if (st[0] == 0 and st[1] in [1, 2, 3]) or (st[1] == 0 and st[0] in [1, 2, 3])]
    raise ValueError(f'Unknown rule: {rule}')

def compute_topology_descriptors(topos=None):
    """Compute invariant geometric descriptors for the 8 topologies."""
    if topos is None:
        topos = TOPOLOGIES_REGISTRY
    records = []
    for name, info in topos.items():
        nodes = info['nodes']
        n_nodes = len(nodes)
        struts = find_struts(name, info)
        n_struts = len(struts)
        tot_len = 0.0
        n_axis = 0
        n_steep = 0
        for i, j, s in struts:
            d = nodes[j] + np.array(s, dtype=np.float64) - nodes[i]
            L = np.linalg.norm(d)
            tot_len += L
            if np.sum(np.abs(d) > 1e-4) == 1:
                n_axis += 1
            if abs(d[2]) / L >= 0.5 - 1e-5:
                n_steep += 1
        records.append({
            'topology': name,
            'n_nodes': n_nodes,
            'n_struts': n_struts,
            'mean_coordination': 2.0 * n_struts / n_nodes,
            'strut_length_per_volume': tot_len / 1.0,
            'fraction_axis_aligned': n_axis / n_struts,
            'fraction_steep_z': n_steep / n_struts
        })
    return pd.DataFrame(records)

DESCRIPTORS_DF = compute_topology_descriptors()

def compute_physics_features(topo_name, rho, sz=1.0, k_radius=1.0, topos_dict=None):
    """Compute transferrable affine Voigt stiffness and geometric features without FEA solver.
    
    Correctly accounts for k-tapered strut weighting: struts with |dz|/L >= 0.5 receive
    radius scaling factor k, meaning their volume scales as k^2.
    """
    if topos_dict is None:
        topos_dict = TOPOLOGIES_REGISTRY
    info = topos_dict[topo_name]
    nodes = info['nodes'].copy()
    struts = find_struts(topo_name, info)
    
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
        k_facs.append(k_radius if abs(d[2]) / L >= 0.5 - 1e-9 else 1.0)
        
    lengths = np.array(lengths)
    k_facs = np.array(k_facs)
    r_base = np.sqrt((rho * V) / (np.pi * np.sum((k_facs**2) * lengths)))
    radii = r_base * k_facs
    areas = np.pi * (radii**2)
    volumes = areas * lengths
    total_vol = np.sum(volumes)
    
    # 6x6 Voigt Affine Stiffness Matrix (11, 22, 33, 23, 13, 12)
    C_aff = np.zeros((6, 6), dtype=np.float64)
    vol_15z = 0.0
    for e in range(len(struts)):
        n = n_vecs[e]
        m = np.array([n[0]**2, n[1]**2, n[2]**2, n[1]*n[2], n[0]*n[2], n[0]*n[1]], dtype=np.float64)
        C_aff += (areas[e] * lengths[e]) * np.outer(m, m)
        if abs(n[2]) >= COS_15_DEG - 1e-6:
            vol_15z += volumes[e]
    C_aff /= V
    
    return {
        'affine_Ex_over_rho': C_aff[0, 0] / rho,
        'affine_Ez_over_rho': C_aff[2, 2] / rho,
        'affine_Gxz_over_rho': C_aff[4, 4] / rho,
        'affine_Gxy_over_rho': C_aff[5, 5] / rho,
        'fraction_vol_within_15deg_z': vol_15z / total_vol,
        'mean_coordination': 2.0 * len(struts) / len(nodes)
    }

def make_rotation_matrix(d):
    """Construct orthonormal 3D rotation matrix aligning local x-axis with strut vector d."""
    L = np.linalg.norm(d)
    t = d / L
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(t[2]) < 0.999 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n1 = np.cross(helper, t)
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(t, n1)
    n2 = n2 / np.linalg.norm(n2)
    return np.vstack([t, n1, n2])

def make_T_matrix(R):
    """Build 12x12 transformation matrix from 3x3 rotation matrix."""
    T = np.zeros((12, 12), dtype=np.float64)
    for b in range(4):
        T[b*3:(b+1)*3, b*3:(b+1)*3] = R
    return T

def local_beam_stiffness(L, r, E=1.0, nu=SOLID_POISSON):
    """12x12 local Euler-Bernoulli 3D beam stiffness matrix."""
    A = np.pi * r**2
    Iy = np.pi * r**4 / 4.0
    Iz = np.pi * r**4 / 4.0
    J = np.pi * r**4 / 2.0
    G = E / (2.0 * (1.0 + nu))
    k = np.zeros((12, 12), dtype=np.float64)
    
    # Axial & Torsion
    k_ax, k_tor = E * A / L, G * J / L
    k[0, 0] += k_ax; k[0, 6] -= k_ax; k[6, 0] -= k_ax; k[6, 6] += k_ax
    k[3, 3] += k_tor; k[3, 9] -= k_tor; k[9, 3] -= k_tor; k[9, 9] += k_tor
    
    # Bending in xy plane
    c1, c2, c3, c4 = 12*E*Iz/L**3, 6*E*Iz/L**2, 4*E*Iz/L, 2*E*Iz/L
    k[1, 1] += c1; k[1, 5] += c2; k[1, 7] -= c1; k[1, 11] += c2
    k[5, 1] += c2; k[5, 5] += c3; k[5, 7] -= c2; k[5, 11] += c4
    k[7, 1] -= c1; k[7, 5] -= c2; k[7, 7] += c1; k[7, 11] -= c2
    k[11, 1] += c2; k[11, 5] += c4; k[11, 7] -= c2; k[11, 11] += c3
    
    # Bending in xz plane
    d1, d2, d3, d4 = 12*E*Iy/L**3, 6*E*Iy/L**2, 4*E*Iy/L, 2*E*Iy/L
    k[2, 2] += d1; k[2, 4] -= d2; k[2, 8] -= d1; k[2, 10] -= d2
    k[4, 2] -= d2; k[4, 4] += d3; k[4, 8] += d2; k[4, 10] += d4
    k[8, 2] -= d1; k[8, 4] += d2; k[8, 8] += d1; k[8, 10] += d2
    k[10, 2] -= d2; k[10, 4] += d4; k[10, 8] += d2; k[10, 10] += d3
    return k

def solve_lattice(topo_name, rho, sz=1.0, k_radius=1.0, cell_size=1.0, nu_solid=SOLID_POISSON, topos_dict=None):
    """Homogenize one unit cell with periodic boundary conditions to compute C and S."""
    if topos_dict is not None and topo_name in topos_dict:
        info = topos_dict[topo_name]
    elif topo_name in TOPOLOGIES_REGISTRY:
        info = TOPOLOGIES_REGISTRY[topo_name]
    else:
        topos = build_topologies()
        if topo_name not in topos:
            return {'status': 'unknown_topology'}
        info = topos[topo_name]
    nodes = info['nodes'].copy()
    n_nodes = len(nodes)
    struts = find_struts(topo_name, info)
    n_struts = len(struts)
    
    cell_dims = np.array([cell_size, cell_size, sz * cell_size], dtype=np.float64)
    V = cell_dims[0] * cell_dims[1] * cell_dims[2]
    nodes[:, 0] *= cell_dims[0]; nodes[:, 1] *= cell_dims[1]; nodes[:, 2] *= cell_dims[2]
    
    strut_lengths, k_facs, strut_vectors = [], [], []
    for i, j, s in struts:
        d = nodes[j] + np.array(s, dtype=np.float64) * cell_dims - nodes[i]
        L = np.linalg.norm(d)
        strut_lengths.append(L); strut_vectors.append(d)
        k_facs.append(k_radius if abs(d[2]) / L >= 0.5 - 1e-9 else 1.0)
        
    strut_lengths, k_facs = np.array(strut_lengths), np.array(k_facs)
    r_base = np.sqrt((rho * V) / (np.pi * np.sum((k_facs**2) * strut_lengths)))
    strut_radii = r_base * k_facs
    rho_actual = np.sum(np.pi * (strut_radii**2) * strut_lengths) / V
    max_slender = float(np.max(strut_radii / strut_lengths))
    min_slender = float(np.min(strut_radii / strut_lengths))
    
    macro_tensors = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]], dtype=np.float64),
        np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]], dtype=np.float64),
        np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=np.float64)
    ]
    
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs), dtype=np.float64)
    A = np.zeros((6, 6), dtype=np.float64)
    F = np.zeros((n_dofs, 6), dtype=np.float64)
    element_matrices = []
    
    for e in range(n_struts):
        i, j, s = struts[e]
        d, L, r = strut_vectors[e], strut_lengths[e], strut_radii[e]
        K_loc = local_beam_stiffness(L, r, E=1.0, nu=nu_solid)
        R = make_rotation_matrix(d)
        T = make_T_matrix(R)
        Ke = T.T @ K_loc @ T
        element_matrices.append({'i': i, 'j': j, 'd': d, 'L': L, 'r': r, 'K_loc': K_loc, 'T': T, 'Ke': Ke})
        
        elem_dofs = np.concatenate([np.arange(i*6, (i+1)*6), np.arange(j*6, (j+1)*6)])
        np.add.at(K_global, np.ix_(elem_dofs, elem_dofs), Ke)
        
        q_aff_list = []
        for case in range(6):
            q_aff = np.zeros(12, dtype=np.float64)
            q_aff[6:9] = macro_tensors[case] @ d
            q_aff_list.append(q_aff)
            np.add.at(F[:, case], elem_dofs, Ke @ q_aff)
        for a in range(6):
            for b in range(6):
                A[a, b] += q_aff_list[a] @ Ke @ q_aff_list[b]
                
    free_dofs = np.arange(3, n_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    cond_K = np.linalg.cond(K_ff)
    if cond_K > 1e14 or np.isnan(cond_K):
        return {'status': 'singular_K', 'cond': cond_K}
        
    u_f = np.linalg.solve(K_ff, -F[free_dofs, :])
    u_full = np.zeros((n_dofs, 6), dtype=np.float64)
    u_full[free_dofs, :] = u_f
    
    C_raw = (A + F.T @ u_full) / V
    C = 0.5 * (C_raw + C_raw.T)
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return {'status': 'singular_C'}
        
    return {
        'status': 'OK', 'topology': topo_name, 'rho': float(rho), 'sz': float(sz), 'k': float(k_radius),
        'C': C, 'C_raw': C_raw, 'S': S, 'V': V,
        'rho_actual': float(rho_actual), 'max_slenderness': float(max_slender), 'min_slenderness': float(min_slender),
        'Ex': float(1.0 / S[0, 0]), 'Ey': float(1.0 / S[1, 1]), 'Ez': float(1.0 / S[2, 2]),
        'Gyz': float(1.0 / S[3, 3]), 'Gxz': float(1.0 / S[4, 4]), 'Gxy': float(1.0 / S[5, 5]),
        'nu_xy': float(-S[0, 1] / S[0, 0]), 'nu_xz': float(-S[0, 2] / S[0, 0]), 'nu_yz': float(-S[1, 2] / S[1, 1]),
        'K_global': K_global, 'element_matrices': element_matrices, 'free_dofs': free_dofs,
        'strut_lengths': strut_lengths, 'strut_radii': strut_radii, 'strut_vectors': strut_vectors,
        'struts': struts
    }

def compute_strength(solver_res, Es=1.0, sigma_ys=1.0):
    """Compute first-yield, Euler buckling strength, failure stress, and failure strain under sigma_zz = -1."""
    if solver_res.get('status') != 'OK':
        return None, None, None, None, None
        
    S = solver_res['S']
    Ez = solver_res['Ez']
    macro_strain_voigt = S @ np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    macro_t = np.array([
        [macro_strain_voigt[0], 0.5*macro_strain_voigt[5], 0.5*macro_strain_voigt[4]],
        [0.5*macro_strain_voigt[5], macro_strain_voigt[1], 0.5*macro_strain_voigt[3]],
        [0.5*macro_strain_voigt[4], 0.5*macro_strain_voigt[3], macro_strain_voigt[2]]
    ])
    
    element_matrices = solver_res['element_matrices']
    K_global = solver_res['K_global']
    free_dofs = solver_res['free_dofs']
    n_dofs = K_global.shape[0]
    
    F_macro = np.zeros(n_dofs, dtype=np.float64)
    q_aff_list = []
    for elem in element_matrices:
        qa = np.zeros(12, dtype=np.float64); qa[6:9] = macro_t @ elem['d']
        q_aff_list.append(qa)
        elem_dofs = np.concatenate([np.arange(elem['i']*6, (elem['i']+1)*6), np.arange(elem['j']*6, (elem['j']+1)*6)])
        np.add.at(F_macro, elem_dofs, elem['Ke'] @ qa)
        
    u_f = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], -F_macro[free_dofs])
    u_full = np.zeros(n_dofs, dtype=np.float64); u_full[free_dofs] = u_f
    
    vm_stresses, buckle_ratios = [], []
    for e, elem in enumerate(element_matrices):
        elem_dofs = np.concatenate([np.arange(elem['i']*6, (elem['i']+1)*6), np.arange(elem['j']*6, (elem['j']+1)*6)])
        q_tot = u_full[elem_dofs] + q_aff_list[e]
        f_loc = elem['K_loc'] @ (elem['T'] @ q_tot)
        
        N = f_loc[6]
        T_tor = abs(f_loc[3])
        M1 = np.sqrt(f_loc[4]**2 + f_loc[5]**2)
        M2 = np.sqrt(f_loc[10]**2 + f_loc[11]**2)
        Mmax = max(M1, M2)
        
        A = np.pi * elem['r']**2
        I = np.pi * elem['r']**4 / 4.0
        J = np.pi * elem['r']**4 / 2.0
        
        peak_stress = abs(N) / A + Mmax * elem['r'] / I
        shear_stress = T_tor * elem['r'] / J
        vm_stresses.append(np.sqrt(peak_stress**2 + 3.0 * shear_stress**2))
        
        if N < -1e-12:
            Pcr = (np.pi**2) * Es * I / (elem['L']**2)
            buckle_ratios.append(Pcr / abs(N))
            
    yield_z = 1.0 / np.max(vm_stresses)
    if len(buckle_ratios) > 0:
        buckle_z = np.min(buckle_ratios)
        buckle_capped_flag = False
    else:
        buckle_z = 10.0
        buckle_capped_flag = True
        
    # First-failure strength and strain in normalized units (Es=1.0, sigma_ys=1.0)
    failure_stress_z = min(yield_z * sigma_ys, buckle_z * Es)
    failure_strain_z = failure_stress_z / Ez
    
    return float(yield_z), float(buckle_z), buckle_capped_flag, float(failure_stress_z), float(failure_strain_z)
