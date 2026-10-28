"""Pytest suite for simulation dataset integrity and non-negotiable data bounds."""
import os
import pytest
import numpy as np
import pandas as pd

DATASET_PATH = os.path.join("data", "lattice_dataset_v1.csv")

@pytest.fixture(scope="module")
def dataset_df():
    assert os.path.exists(DATASET_PATH), f"Dataset file missing at {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH)
    return df

def test_dataset_no_nans(dataset_df):
    """Verify zero NaN, null, or infinite values in simulation dataset."""
    nan_count = dataset_df.isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in dataset!"

def test_dataset_row_counts_and_balance(dataset_df):
    """Verify exactly 6,400 verified samples balanced across 8 topologies (800 each)."""
    assert len(dataset_df) == 6400, f"Expected 6400 rows, found {len(dataset_df)}"
    counts = dataset_df['topology'].value_counts().to_dict()
    assert len(counts) == 8, f"Expected 8 topologies, found {len(counts)}"
    for topo, cnt in counts.items():
        assert cnt == 800, f"Topology {topo} has {cnt} rows, expected exactly 800"

def test_dataset_positive_definiteness(dataset_df):
    """Verify all effective moduli and strengths are strictly positive."""
    for col in ['Ex', 'Ey', 'Ez', 'Gyz', 'Gxz', 'Gxy', 'yield_z', 'buckle_z', 'failure_stress_z']:
        assert np.all(dataset_df[col] > 0), f"Column {col} contains non-positive values!"

def test_dataset_voigt_bounds(dataset_df):
    """Verify that normalized Young's moduli never exceed relative density rho."""
    assert np.all(dataset_df['Ex'] <= dataset_df['rho'] + 1e-7), "Ex exceeds Voigt bound rho!"
    assert np.all(dataset_df['Ez'] <= dataset_df['rho'] + 1e-7), "Ez exceeds Voigt bound rho!"

def test_dataset_parameter_ranges(dataset_df):
    """Verify parameter sampling complies with stated limits."""
    assert dataset_df['rho'].min() >= 0.019
    assert dataset_df['rho'].max() <= 0.301
    assert dataset_df['sz'].min() >= 0.59
    assert dataset_df['sz'].max() <= 1.81
    assert dataset_df['k'].min() >= 0.59
    assert dataset_df['k'].max() <= 1.61

