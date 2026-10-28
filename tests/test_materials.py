"""Pytest suite for materials database quality, stoichiometry, physical plausibility, and integrity."""
import os
import json
import pytest
import numpy as np
import pandas as pd

SOURCES_PATH = os.path.join("data", "materials", "sources.csv")
ELEMENTS_PATH = os.path.join("data", "materials", "elements.csv")
COMPS_PATH = os.path.join("data", "materials", "compositions_long.csv")
ALLOYS_PATH = os.path.join("data", "materials", "alloys.csv")
MATERIALS_JSON_PATH = "materials.json"

@pytest.fixture(scope="module")
def materials_data():
    sources_df = pd.read_csv(SOURCES_PATH)
    elements_df = pd.read_csv(ELEMENTS_PATH).set_index('symbol')
    comps_df = pd.read_csv(COMPS_PATH)
    alloys_df = pd.read_csv(ALLOYS_PATH)
    with open(MATERIALS_JSON_PATH, 'r') as f:
        mat_json = json.load(f)
    return {
        'sources': sources_df, 'elements': elements_df,
        'comps': comps_df, 'alloys': alloys_df, 'json': mat_json
    }

def test_source_foreign_keys(materials_data):
    """Verify that every source_id in elements, compositions, and alloys exists in sources.csv."""
    valid_sources = set(materials_data['sources']['source_id'])
    
    # Elements
    for sid in materials_data['elements']['source_id'].dropna():
        assert sid in valid_sources, f"Unknown source_id {sid} in elements.csv"
        
    # Compositions
    for sid in materials_data['comps']['source_id'].dropna():
        assert sid in valid_sources, f"Unknown source_id {sid} in compositions_long.csv"
        
    # Alloys
    for sid in materials_data['alloys']['source_id'].dropna():
        assert sid in valid_sources, f"Unknown source_id {sid} in alloys.csv"

def test_composition_stoichiometry(materials_data):
    """Verify nominal compositions sum to 100 wt% (+/- 0.05%) for all alloy conditions."""
    comps = materials_data['comps']
    grouped = comps.groupby('condition_id')['nominal_wt_pct'].sum()
    for cond_id, total in grouped.items():
        assert abs(total - 100.0) < 0.05, f"Composition {cond_id} sums to {total}%, expected 100%"

def test_composition_bounds_validity(materials_data):
    """Verify min <= nominal <= max and no negative percentages."""
    comps = materials_data['comps']
    for idx, row in comps.iterrows():
        assert row['min_wt_pct'] >= 0.0, f"Negative min wt% in {row['condition_id']} for {row['element']}"
        assert row['nominal_wt_pct'] >= row['min_wt_pct'] - 1e-4, f"nominal < min in {row['condition_id']} for {row['element']}"
        assert row['nominal_wt_pct'] <= row['max_wt_pct'] + 1e-4, f"nominal > max in {row['condition_id']} for {row['element']}"

def test_no_duplicate_alloy_conditions(materials_data):
    """Verify no duplicate (grade, condition, build_direction) combinations in alloys.csv."""
    alloys = materials_data['alloys']
    dups = alloys.duplicated(subset=['grade', 'condition', 'build_direction'])
    assert not dups.any(), f"Duplicate alloy rows found: {alloys[dups]}"

def test_mechanical_property_plausibility(materials_data):
    """Verify physical bounds: 0 <= poisson <= 0.5, yield <= UTS, and positive moduli."""
    alloys = materials_data['alloys']
    for idx, row in alloys.iterrows():
        assert 0.0 <= row['poisson'] <= 0.5, f"Invalid Poisson ratio {row['poisson']} for {row['grade']}"
        assert row['E_MPa'] > 0.0, f"Non-positive E {row['E_MPa']} for {row['grade']}"
        assert row['yield_MPa'] <= row['UTS_MPa'] + 1e-3, f"Yield {row['yield_MPa']} > UTS {row['UTS_MPa']} for {row['grade']}"
        assert row['density_kg_m3'] > 0.0, f"Non-positive density for {row['grade']}"

def test_rule_of_mixtures_density_cross_check(materials_data):
    """Verify bulk alloy density against theoretical inverse-rule-of-mixtures (< 3% tolerance)."""
    elements = materials_data['elements']
    comps = materials_data['comps']
    alloys = materials_data['alloys']
    
    # Mapping between alloy condition and condition_id in comps
    cond_map = {
        ('Ti-6Al-4V', 'Annealed', 'XY'): 'TI64_LPBF_ANN',
        ('Ti-6Al-4V ELI', 'Stress-Relieved', 'XY'): 'TI64_ELI_LPBF',
        ('AlSi10Mg', 'Stress-Relieved', 'XY'): 'ALSI10MG_LPBF_SR',
        ('Scalmalloy', 'Heat-Treated', 'XY'): 'SCALMALLOY_LPBF_HT',
        ('316L Stainless Steel', 'Stress-Relieved', 'XY'): '316L_LPBF_SR',
        ('17-4 PH Stainless Steel', 'Solution-Treated + H900 Aged', 'XY'): '174PH_LPBF_H900',
        ('Maraging Steel 18Ni300', 'Age-Hardened (490C 6h)', 'XY'): 'MS1_LPBF_AGED',
        ('Inconel 718', 'Solution-Annealed + Double-Aged', 'XY'): 'IN718_LPBF_HT',
        ('CoCrMo', 'Heat-Treated', 'XY'): 'COCRMO_LPBF_HT',
        ('CuCr1Zr', 'Heat-Treated (Age-Hardened)', 'XY'): 'CUCRZR_LPBF_HT',
        ('Cantor Alloy CoCrFeMnNi', 'Homogenized (1100C)', 'Isotropic'): 'CANTOR_CAST_HOMOG'
    }
    
    for (grade, cond, bdir), cond_id in cond_map.items():
        alloy_sub = alloys[(alloys['grade'] == grade) & (alloys['condition'] == cond) & (alloys['build_direction'] == bdir)]
        if alloy_sub.empty: continue
        stated_density_g = float(alloy_sub['density_kg_m3'].iloc[0]) / 1000.0
        
        comp_sub = comps[comps['condition_id'] == cond_id]
        inv_rho = 0.0
        for _, c_row in comp_sub.iterrows():
            el = c_row['element']
            wt = c_row['nominal_wt_pct'] / 100.0
            el_rho = float(elements.loc[el, 'density_g_cm3'])
            if pd.notna(el_rho):
                inv_rho += wt / el_rho
        calc_density_g = 1.0 / inv_rho
        diff_pct = abs(calc_density_g - stated_density_g) / stated_density_g * 100.0
        assert diff_pct <= 3.0, f"Density mismatch for {cond_id}: stated {stated_density_g} vs calc {calc_density_g} ({diff_pct:.2f}% > 3.0%)"

def test_materials_json_backward_compatibility(materials_data):
    """Verify that materials.json retains all original keys and works with existing codebase."""
    mat_json = materials_data['json']
    required_keys = ['E_MPa', 'poisson', 'yield_MPa', 'density_kg_m3', 'source', 'verified', 'verification_status', 'notes']
    original_materials = ['Demo (normalized)', 'Ti-6Al-4V', 'AlSi10Mg', '316L Stainless Steel', 'Printable Polymer (PA12)']
    
    for mat_name in original_materials:
        assert mat_name in mat_json, f"Missing original material {mat_name} in materials.json"
        mat = mat_json[mat_name]
        for rk in required_keys:
            assert rk in mat, f"Key {rk} missing in {mat_name}"
            
    # Check that new structured metadata is present
    for mat_name, mat in mat_json.items():
        assert 'structure' in mat
        assert 'process' in mat
        assert 'condition'
        assert 'composition' in mat

