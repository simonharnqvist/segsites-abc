from abiss.demographic_model import DemographicModel
import pytest
import numpy as np
from numpy import testing

@pytest.fixture
def make_model():
    return DemographicModel(
        population_sizes=(100_000, 150_000, 300_000, 50_000, 200_000),
        tau_split=1/3,
        tau_change=1/6,
        Ms=(3, 1.5, 1.5, 1),
    ).msprime_demography

def test_population_sizes(make_model):
    msprime_pop_sizes = [pop.initial_size for pop in make_model.populations]
    reordered_pop_sizes = [200_000, 100_000, 150_000, 300_000, 50_000]
    assert msprime_pop_sizes == reordered_pop_sizes

def test_times(make_model):
    msprime_times = [event.time for event in make_model.events][-2:]
    assert msprime_times == [50_000, 100_000]

def test_mig_rates(make_model):
    expected_migration_matrix = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 5e-6, 0, 0],
        [0, 1.5e-5, 0, 0, 0],
        [0, 0, 0, 0, 1e-5],
        [0, 0, 0, 2.5e-6, 0]
    ])

    migmat = make_model.migration_matrix
    print(migmat)
    testing.assert_array_equal(migmat, expected_migration_matrix)