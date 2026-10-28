"""Pytest suite for surrogate model persistence, prediction shapes, determinism, and regression tests."""
import os
import pytest
import joblib
import numpy as np
import pandas as pd
import lattice_physics as lp

@pytest.fixture(scope="module")
def model_assets():
    meta = joblib.load("models/model_metadata.joblib")
    hgb = joblib.load("models/hgb_models.joblib")
    mlp = joblib.load("models/mlp_ensemble.joblib")
    scalers = joblib.load("models/scalers.joblib")
    return {'meta': meta, 'hgb': hgb, 'mlp': mlp, 'scalers': scalers}

def test_model_files_exist():
    """Verify that all required model assets are present on disk."""
    for fn in ['hgb_models.joblib', 'mlp_ensemble.joblib', 'scalers.joblib', 'model_metadata.joblib']:
        p = os.path.join("models", fn)
        assert os.path.exists(p), f"Missing model asset: {p}"

def test_surrogate_prediction_shapes(model_assets):
    """Verify that predictions return exactly 8 mechanical targets."""
    meta = model_assets['meta']
    hgb = model_assets['hgb']
    mlp = model_assets['mlp']
    scalers = model_assets['scalers']
    
    desc_lookup = lp.DESCRIPTORS_DF.set_index('topology')
    d_row = desc_lookup.loc['octet']
    p_feat = lp.compute_physics_features('octet', 0.1, sz=1.0, k_radius=1.0)
    x = np.array([[
        np.log10(0.1), 1.0, 1.0,
        d_row['n_nodes'], d_row['n_struts'], d_row['mean_coordination'],
        d_row['strut_length_per_volume'], d_row['fraction_axis_aligned'], d_row['fraction_steep_z'],
        p_feat['affine_Ex_over_rho'], p_feat['affine_Ez_over_rho'],
        p_feat['affine_Gxz_over_rho'], p_feat['affine_Gxy_over_rho'],
        p_feat['fraction_vol_within_15deg_z']
    ]], dtype=np.float64)
    
    # Check HGB has 8 regressors
    assert len(hgb) == 8
    for tgt in meta['targets']:
        val = hgb[tgt].predict(x)
        assert val.shape == (1,)
        
    # Check MLP ensemble has 5 models, each predicting 8 targets
    assert len(mlp) == 5
    x_scaled = scalers['scaler_X'].transform(x)
    for model in mlp:
        pred_scaled = model.predict(x_scaled)
        assert pred_scaled.shape == (1, 8)

def test_model_prediction_determinism(model_assets):
    """Verify that repeated predictions on identical input produce identical outputs."""
    mlp = model_assets['mlp']
    scalers = model_assets['scalers']
    desc_lookup = lp.DESCRIPTORS_DF.set_index('topology')
    d_row = desc_lookup.loc['bcc']
    p_feat = lp.compute_physics_features('bcc', 0.08, sz=1.2, k_radius=1.0)
    x = np.array([[
        np.log10(0.08), 1.2, 1.0,
        d_row['n_nodes'], d_row['n_struts'], d_row['mean_coordination'],
        d_row['strut_length_per_volume'], d_row['fraction_axis_aligned'], d_row['fraction_steep_z'],
        p_feat['affine_Ex_over_rho'], p_feat['affine_Ez_over_rho'],
        p_feat['affine_Gxz_over_rho'], p_feat['affine_Gxy_over_rho'],
        p_feat['fraction_vol_within_15deg_z']
    ]], dtype=np.float64)
    
    x_scaled = scalers['scaler_X'].transform(x)
    pred_1 = mlp[0].predict(x_scaled)
    pred_2 = mlp[0].predict(x_scaled)
    np.testing.assert_allclose(pred_1, pred_2, atol=1e-12)

def test_frozen_regression_5_designs(model_assets):
    """Frozen regression test on 5 canonical designs to prevent silent drift."""
    mlp = model_assets['mlp']
    scalers = model_assets['scalers']
    desc_lookup = lp.DESCRIPTORS_DF.set_index('topology')
    
    benchmarks = [
        ('sc', 0.05, 1.0, 1.0, 0.016968, 0.000133),
        ('bcc', 0.10, 1.2, 1.1, 0.001288, 0.012305),
        ('octet', 0.08, 1.4, 1.0, 0.019751, 0.006395),
        ('diamond', 0.15, 0.9, 1.2, 0.005524, 0.005446),
        ('sc_octet', 0.12, 1.1, 1.0, 0.022590, 0.008825)
    ]
    
    for topo, rho, sz, k, ref_Ez, ref_Gxz in benchmarks:
        d_row = desc_lookup.loc[topo]
        p_feat = lp.compute_physics_features(topo, rho, sz=sz, k_radius=k)
        x = [
            np.log10(rho), sz, k,
            d_row['n_nodes'], d_row['n_struts'], d_row['mean_coordination'],
            d_row['strut_length_per_volume'], d_row['fraction_axis_aligned'], d_row['fraction_steep_z'],
            p_feat['affine_Ex_over_rho'], p_feat['affine_Ez_over_rho'],
            p_feat['affine_Gxz_over_rho'], p_feat['affine_Gxy_over_rho'],
            p_feat['fraction_vol_within_15deg_z']
        ]
        x_scaled = scalers['scaler_X'].transform([x])
        raw_mlp = [scalers['scaler_Y'].inverse_transform(model.predict(x_scaled))[0] for model in mlp]
        pred_Ez = float(np.mean([10**p[1] for p in raw_mlp]))
        pred_Gxz = float(np.mean([10**p[3] for p in raw_mlp]))
        
        assert abs(pred_Ez - ref_Ez) < 1e-4, f"{topo} Ez drift: {pred_Ez} != {ref_Ez}"
        assert abs(pred_Gxz - ref_Gxz) < 1e-4, f"{topo} Gxz drift: {pred_Gxz} != {ref_Gxz}"

