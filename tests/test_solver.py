"""Pytest suite for physics solver, periodic homogenization, and strength models."""
import pytest
import numpy as np
import lattice_physics as lp

def test_simple_cubic_exact_closed_form():
    """Verify Simple Cubic matches exact closed-form analytical formulas to < 1e-9."""
    rho_val = 0.03
    res = lp.solve_lattice('sc', rho=rho_val, sz=1.0)
    assert res['status'] == 'OK'
    
    # 1. Axial stiffness C11 = rho / 3
    exp_c11 = rho_val / 3.0
    c11_act = res['C'][0, 0]
    assert abs(c11_act - exp_c11) < 1e-9, f"C11 {c11_act} != expected {exp_c11}"
    
    # 2. First-failure strength: yield = rho / 3, buckle = pi * rho^2 / 36
    y_z, b_z, flag, f_str, f_eps = lp.compute_strength(res)
    exp_yield = rho_val / 3.0
    exp_buckle = np.pi * (rho_val**2) / 36.0
    assert abs(y_z - exp_yield) < 1e-9, f"yield {y_z} != expected {exp_yield}"
    assert abs(b_z - exp_buckle) < 1e-9, f"buckle {b_z} != expected {exp_buckle}"
    # When struts buckle, buckle_capped_flag is False (it is only True when capped to 10.0)
    assert flag == False
    assert abs(f_str - min(exp_yield, exp_buckle)) < 1e-9

def test_cubic_symmetry():
    """Verify cubic symmetry (C11=C22=C33, C12=C13=C23, C44=C55=C66) for Octet and SC."""
    for topo in ['sc', 'octet']:
        res = lp.solve_lattice(topo, rho=0.08, sz=1.0, k_radius=1.0)
        assert res['status'] == 'OK'
        C = res['C']
        # Diagonal normal components
        assert abs(C[0, 0] - C[1, 1]) / C[0, 0] < 1e-4
        assert abs(C[0, 0] - C[2, 2]) / C[0, 0] < 1e-4
        # Off-diagonal normal components
        assert abs(C[0, 1] - C[0, 2]) / (abs(C[0, 1]) + 1e-6) < 1e-4
        assert abs(C[0, 1] - C[1, 2]) / (abs(C[0, 1]) + 1e-6) < 1e-4
        # Shear components
        assert abs(C[3, 3] - C[4, 4]) / C[3, 3] < 1e-4
        assert abs(C[3, 3] - C[5, 5]) / C[3, 3] < 1e-4

def test_voigt_upper_bound():
    """Verify that normalized Young's moduli never exceed relative density (Voigt bound: E <= rho)."""
    for topo in lp.TOPOLOGIES_REGISTRY.keys():
        res = lp.solve_lattice(topo, rho=0.06, sz=1.1, k_radius=1.0)
        if res['status'] == 'OK':
            assert res['Ex'] <= res['rho_actual'] + 1e-7, f"Voigt bound violated in Ex for {topo}"
            assert res['Ey'] <= res['rho_actual'] + 1e-7, f"Voigt bound violated in Ey for {topo}"
            assert res['Ez'] <= res['rho_actual'] + 1e-7, f"Voigt bound violated in Ez for {topo}"

def test_gibson_ashby_scaling_slopes():
    """Verify that stretch-dominated octet has slope ~1.0, and bending BCC has slope ~2.0."""
    rhos = [0.03, 0.05, 0.08, 0.12]
    # Octet: stretch dominated (E ~ rho^1)
    ez_oct = [lp.solve_lattice('octet', rho=r, sz=1.0)['Ez'] for r in rhos]
    slope_oct = np.polyfit(np.log10(rhos), np.log10(ez_oct), deg=1)[0]
    assert 0.95 <= slope_oct <= 1.08, f"Octet slope {slope_oct} should be ~1.0"
    
    # BCC: bending dominated (E ~ rho^2)
    ez_bcc = [lp.solve_lattice('bcc', rho=r, sz=1.0)['Ez'] for r in rhos]
    slope_bcc = np.polyfit(np.log10(rhos), np.log10(ez_bcc), deg=1)[0]
    assert 1.90 <= slope_bcc <= 2.10, f"BCC slope {slope_bcc} should be ~2.0"

def test_cell_size_independence():
    """Verify that normalized effective properties are strictly independent of cell size."""
    res_1 = lp.solve_lattice('octet', rho=0.08, sz=1.2, k_radius=1.1, cell_size=1.0)
    res_10 = lp.solve_lattice('octet', rho=0.08, sz=1.2, k_radius=1.1, cell_size=10.0)
    assert abs(res_1['Ez'] - res_10['Ez']) / res_1['Ez'] < 1e-7
    assert abs(res_1['Gxz'] - res_10['Gxz']) / res_1['Gxz'] < 1e-7
    assert abs(res_1['C'][0, 0] - res_10['C'][0, 0]) / res_1['C'][0, 0] < 1e-7
