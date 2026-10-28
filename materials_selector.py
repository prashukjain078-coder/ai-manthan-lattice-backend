"""Material selection and physical property translation engine for AI Manthan 2.0.

Provides:
1. select_material(requirements): Filters and ranks bulk materials meeting engineering constraints.
2. format_material_composition(material_name): Returns element-by-element chemical composition.
3. convert_lattice_to_physical(relative_props, material_name): Scales dimensionless lattice ratios to physical units.
"""
import os
import json
import pandas as pd
import numpy as np

ALLOYS_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "materials", "alloys.csv")
COMPS_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "materials", "compositions_long.csv")
MATERIALS_JSON_PATH = os.path.join(os.path.dirname(__file__), "materials.json")

STRUT_KNOCKDOWN_NOTE = (
    "CRITICAL ENGINEERING NOTE: Bulk datasheet values shown. "
    "Additive strut-scale cellular lattices typically exhibit a 15% to 40% knockdown "
    "in effective modulus and strength compared to bulk tensile specimens due to "
    "as-built surface roughness (Ra ~ 10-25 um), sub-surface microporosity, and columnar grain morphology."
)

def get_materials_db():
    """Load cached materials database tables."""
    alloys_df = pd.read_csv(ALLOYS_CSV_PATH)
    comps_df = pd.read_csv(COMPS_CSV_PATH)
    with open(MATERIALS_JSON_PATH, "r") as f:
        mat_json = json.load(f)
    return alloys_df, comps_df, mat_json

def select_material(requirements):
    """Filter and rank candidate engineering materials meeting bulk constraints.
    
    Parameters:
    -----------
    requirements : dict
        - min_E_MPa : float, optional
        - min_yield_MPa : float, optional
        - max_density_kg_m3 : float, optional
        - process : str, optional (e.g. 'L-PBF', 'SLS')
        - family : str, optional (e.g. 'Titanium', 'Aluminum', 'Steel')
        - max_service_temp_C : float, optional
        - sort_by : str, default 'specific_yield' ('specific_yield', 'specific_stiffness', 'yield_MPa')
        
    Returns:
    --------
    dict with:
        - viable_count: int
        - candidates: list of dicts with composition, crystal structure, mechanical specs, and justification.
        - strut_scale_knockdown_note: str
    """
    alloys_df, comps_df, mat_json = get_materials_db()
    
    min_E = requirements.get('min_E_MPa', 0.0)
    min_yield = requirements.get('min_yield_MPa', 0.0)
    max_rho = requirements.get('max_density_kg_m3', float('inf'))
    target_process = requirements.get('process')
    target_family = requirements.get('family')
    
    # Filter
    mask = (
        (alloys_df['E_MPa'] >= min_E) &
        (alloys_df['yield_MPa'] >= min_yield) &
        (alloys_df['density_kg_m3'] <= max_rho)
    )
    if target_process:
        mask &= (alloys_df['process'].str.upper() == target_process.upper())
    if target_family:
        mask &= (alloys_df['family'].str.upper() == target_family.upper())
        
    filtered = alloys_df[mask].copy()
    if filtered.empty:
        return {
            'viable_count': 0,
            'candidates': [],
            'strut_scale_knockdown_note': STRUT_KNOCKDOWN_NOTE,
            'diagnostic': f"No materials met criteria: min_E={min_E} MPa, min_yield={min_yield} MPa, max_rho={max_rho} kg/m3"
        }
        
    # Derived performance metrics
    filtered['specific_yield_kNm_kg'] = (filtered['yield_MPa'] * 1e3) / filtered['density_kg_m3']
    filtered['specific_stiffness_MNm_kg'] = (filtered['E_MPa'] * 1e3) / filtered['density_kg_m3']
    
    sort_by = requirements.get('sort_by', 'specific_yield')
    if sort_by == 'specific_stiffness':
        filtered = filtered.sort_values(by='specific_stiffness_MNm_kg', ascending=False)
    else:
        filtered = filtered.sort_values(by='specific_yield_kNm_kg', ascending=False)
        
    candidate_records = []
    for _, row in filtered.iterrows():
        grade = row['grade']
        cond = row['condition']
        
        # Get composition
        alloy_comps = comps_df[comps_df['grade'] == grade]
        comp_summary = {}
        if not alloy_comps.empty:
            for _, c_row in alloy_comps.iterrows():
                comp_summary[c_row['element']] = {
                    'nominal_wt_pct': c_row['nominal_wt_pct'],
                    'min_wt_pct': c_row['min_wt_pct'],
                    'max_wt_pct': c_row['max_wt_pct'],
                    'balance': c_row['is_balance']
                }
        else:
            # Fallback to json composition if available
            if grade in mat_json and 'composition' in mat_json[grade]:
                comp_summary = mat_json[grade]['composition']
                
        candidate_records.append({
            'grade': grade,
            'family': row['family'],
            'process': row['process'],
            'condition': cond,
            'build_direction': row['build_direction'],
            'crystal_structure_or_phases': row['crystal_structure_or_phases'],
            'E_MPa': float(row['E_MPa']),
            'yield_MPa': float(row['yield_MPa']),
            'UTS_MPa': float(row['UTS_MPa']),
            'elongation_pct': float(row['elongation_pct']),
            'density_kg_m3': float(row['density_kg_m3']),
            'specific_yield_kNm_kg': round(float(row['specific_yield_kNm_kg']), 2),
            'specific_stiffness_MNm_kg': round(float(row['specific_stiffness_MNm_kg']), 2),
            'thermal_conductivity_W_mK': float(row['thermal_conductivity_W_mK']) if pd.notna(row['thermal_conductivity_W_mK']) else None,
            'composition': comp_summary,
            'source_id': row['source_id'],
            'source_tier': row['tier'],
            'verified': bool(row['verified']),
            'verification_status': row['verification_status'],
            'justification': (
                f"{grade} ({row['process']} {cond}) provides {row['yield_MPa']:.0f} MPa yield strength "
                f"at {row['density_kg_m3']:.0f} kg/m3 (specific yield: {row['specific_yield_kNm_kg']:.1f} kN*m/kg)."
            )
        })
        
    return {
        'viable_count': len(candidate_records),
        'candidates': candidate_records,
        'strut_scale_knockdown_note': STRUT_KNOCKDOWN_NOTE
    }

def convert_lattice_to_physical(relative_props, material_name):
    """Translate normalized lattice simulation output into physical engineering units.
    
    Parameters:
    -----------
    relative_props : dict
        Contains normalized dimensionless ratios: 'rho', 'Ex', 'Ez', 'Gxz', 'strength_z' (first-failure).
    material_name : str
        Key matching materials.json (e.g. 'Ti-6Al-4V', 'Scalmalloy', 'AlSi10Mg', 'Inconel 718').
        
    Returns:
    --------
    dict with physical moduli (MPa), strengths (MPa), apparent density (kg/m3), mass saved (%),
    and constituent metallurgical metadata.
    """
    alloys_df, comps_df, mat_json = get_materials_db()
    if material_name not in mat_json:
        raise ValueError(f"Material {material_name} not found in materials database.")
        
    mat = mat_json[material_name]
    Es = mat['E_MPa']
    ys = mat['yield_MPa']
    rho_solid = mat['density_kg_m3']
    
    rho_rel = relative_props['rho']
    apparent_density = rho_rel * rho_solid if rho_solid else None
    mass_saving_pct = (1.0 - rho_rel) * 100.0
    
    return {
        'material': material_name,
        'process': mat.get('process'),
        'condition': mat.get('condition'),
        'crystal_structure': mat.get('structure'),
        'composition': mat.get('composition', {}),
        'solid_E_MPa': Es,
        'solid_yield_MPa': ys,
        'solid_density_kg_m3': rho_solid,
        'relative_density': rho_rel,
        'apparent_density_kg_m3': round(apparent_density, 1) if apparent_density else None,
        'mass_saving_vs_solid_pct': round(mass_saving_pct, 2),
        'physical_Ex_MPa': round(relative_props.get('Ex', 0.0) * Es, 1),
        'physical_Ez_MPa': round(relative_props.get('Ez', 0.0) * Es, 1),
        'physical_Gxz_MPa': round(relative_props.get('Gxz', 0.0) * Es, 1),
        'first_failure_strength_z_MPa': round(relative_props.get('strength_z', 0.0) * ys, 1),
        'source': mat.get('source'),
        'verified': mat.get('verified', False),
        'verification_status': mat.get('verification_status'),
        'strut_knockdown_note': STRUT_KNOCKDOWN_NOTE
    }

