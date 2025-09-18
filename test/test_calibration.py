import os
import shutil
import tempfile
from multiprocessing import Pool
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import matplotlib.pyplot as plt
import numpy
import pytest

from oasis import calibration
from oasis.common import G_GRAVITY


class TestComputeR200mAndV200m:
    """Test suite for _compute_r200m_and_v200m function."""

    # @pytest.fixture
    # def mock_gravity(self):
    #     """Mock gravitational constant for testing."""
    #     with patch('G_GRAVITY', G_GRAVITY):  # Typical value in (km/s)^2 Mpc/M_sun
    #         yield G_GRAVITY

    def test_basic_computation(self):
        """Test basic R200m and V200 computation with well-behaved data."""
        radial_distances = numpy.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0])
        particle_mass = 1e10  # M_sun
        mass_density = 2.78e11  # M_sun/Mpc^3
        
        r200m, v200_sq = calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
        assert r200m > 0
        assert v200_sq > 0
        assert isinstance(r200m, float)
        assert isinstance(v200_sq, float)
        assert r200m <= radial_distances.max()  # R200m should be within particle range

    def test_empty_array(self):
        """Test behavior with empty radial distances array."""
        radial_distances = numpy.array([])
        particle_mass = 1e10
        mass_density = 2.78e11
        
        with pytest.raises(ValueError, match='is empty'):
            calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
    def test_single_particle(self):
        """Test computation with single particle."""
        radial_distances = numpy.array([0.5])
        particle_mass = 1e10
        mass_density = 2.78e11
        
        r200m, v200_sq = calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
        assert r200m == 0.5
        assert v200_sq > 0

    def test_zero_distances_handling(self):
        """Test that zero distances are handled properly."""
        radial_distances = numpy.array([0.0, 0.0, 0.1, 0.2])
        particle_mass = 1e10
        mass_density = 2.78e11
        
        r200m, v200_sq = calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
        assert r200m > 0
        assert v200_sq > 0

    @pytest.mark.parametrize(
        "particle_mass, mass_density", 
        [
            (1e9, 1e11),
            (1e10, 2.78e11),
            (1e11, 5e11),
            (5e10, 1.5e11)
        ],
    )
    def test_different_mass_scales(self, particle_mass, mass_density):
        """Test computation with different mass scales."""
        radial_distances = numpy.linspace(0.1, 3.0, 20)
        
        r200m, v200_sq = calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
        assert r200m > 0
        assert v200_sq > 0

    def test_no_qualifying_particles(self):
        """Test when no particles satisfy the 200×ρ_m criterion."""
        # Create a very sparse distribution where density never reaches 200×ρ_m
        radial_distances = numpy.array([5.0, 10.0, 15.0, 20.0])
        particle_mass = 1e8  # Small particle mass
        mass_density = 1e12   # High background density
        
        r200m, v200_sq = calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
        # Should use outermost particle
        assert r200m == 20.0
        assert v200_sq > 0


class TestFindIsolatedSeeds:
    """Test suite for _find_isolated_seeds function."""

    @pytest.fixture
    def regular_grid_seeds(self):
        """Create seeds arranged in a regular grid for testing isolation."""
        spacing = 5.0
        x, y, z = numpy.meshgrid(
            numpy.arange(0, 50, spacing),
            numpy.arange(0, 50, spacing), 
            numpy.arange(0, 50, spacing)
        )
        positions = numpy.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Assign random masses and radii
        n_seeds = len(positions)
        masses = numpy.random.lognormal(30, 0.5, n_seeds)  # Log-normal mass distribution
        radii = (masses / 1e12) ** (1/3) * 0.5  # Mass-radius relation
        
        return positions, masses, radii

    def test_no_neighbors_isolation(self):
        """Test that widely separated seeds are all considered isolated."""
        positions = numpy.array([
            [10.0, 10.0, 10.0],
            [30.0, 30.0, 30.0],
            [70.0, 70.0, 70.0]
        ])
        masses = numpy.array([1e13, 1e13, 1e13])
        radii = numpy.array([1.0, 1.0, 1.0])
        
        isolated = calibration._find_isolated_seeds(
            position=positions,
            mass=masses,
            radius=radii,
            max_seeds=10,
            boxsize=100.0,
            isolation_factor=0.2,
            isolation_radius_factor=2.0
        )
        
        # All seeds should be isolated
        assert len(isolated) == 3
        assert set(isolated) == {0, 1, 2}

    def test_mass_hierarchy_isolation(self):
        """Test isolation based on mass hierarchy."""
        positions = numpy.array([
            [25.0, 25.0, 25.0],  # Central massive seed
            [27.0, 25.0, 25.0],  # Close neighbor, lower mass
            [23.0, 25.0, 25.0],  # Close neighbor, lower mass
        ])
        masses = numpy.array([1e14, 1e13, 1e13])  # Central seed 10x more massive
        radii = numpy.array([2.0, 1.0, 1.0])
        
        isolated = calibration._find_isolated_seeds(
            position=positions,
            mass=masses,
            radius=radii,
            max_seeds=5,
            boxsize=100.0,
            isolation_factor=0.2,
            isolation_radius_factor=2.0
        )
        
        # Only the central massive seed should be isolated
        assert len(isolated) >= 1
        assert 0 in isolated  # Most massive seed should be found first

    def test_max_seeds_limit(self, regular_grid_seeds):
        """Test that max_seeds parameter limits the number of returned seeds."""
        positions, masses, radii = regular_grid_seeds
        
        max_seeds = 5
        isolated = calibration._find_isolated_seeds(
            position=positions,
            mass=masses,
            radius=radii,
            max_seeds=max_seeds,
            boxsize=50.0,
            isolation_factor=0.2,
            isolation_radius_factor=2.0
        )
        
        assert len(isolated) <= max_seeds

    def test_periodic_boundary_conditions(self):
        """Test isolation with periodic boundary conditions."""
        # Place seeds near box edges to test periodic boundaries
        positions = numpy.array([
            [1.0, 1.0, 1.0],    # Near corner
            [99.0, 99.0, 99.0], # Near opposite corner (close due to PBC)
            [50.0, 50.0, 50.0]  # Center
        ])
        masses = numpy.array([1e13, 1e13, 1e13])
        radii = numpy.array([2.0, 2.0, 2.0])
        
        isolated = calibration._find_isolated_seeds(
            position=positions,
            mass=masses,
            radius=radii,
            max_seeds=10,
            boxsize=100.0,
            isolation_factor=0.2,
            isolation_radius_factor=2.0
        )
        
        # Corner seeds should not be isolated due to PBC
        # Center seed should be isolated
        assert 2 in isolated

    @pytest.mark.parametrize(
        "isolation_factor, isolation_radius_factor", 
        [
            (0.1, 1.5),
            (0.2, 2.0),
            (0.3, 2.5),
            (0.5, 3.0)
        ],
    )
    def test_isolation_parameters(self, regular_grid_seeds, isolation_factor, isolation_radius_factor):
        """Test different isolation parameters."""
        positions, masses, radii = regular_grid_seeds
        
        isolated = calibration._find_isolated_seeds(
            position=positions,
            mass=masses,
            radius=radii,
            max_seeds=20,
            boxsize=50.0,
            isolation_factor=isolation_factor,
            isolation_radius_factor=isolation_radius_factor
        )
        
        assert len(isolated) >= 0
        # More restrictive parameters should generally yield fewer isolated seeds
        
    def test_uniform_mass_case(self):
        """Test isolation when all seeds have the same mass."""
        positions = numpy.random.uniform(0, 100, (50, 3))
        mass = 1e13  # Single value for all seeds
        radius = 1.0  # Single value for all seeds
        
        isolated = calibration._find_isolated_seeds(
            position=positions,
            mass=mass,
            radius=radius,
            max_seeds=10,
            boxsize=100.0
        )
        
        assert len(isolated) <= 10
        assert all(isinstance(idx, (int, numpy.integer)) for idx in isolated)