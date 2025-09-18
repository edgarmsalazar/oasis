import os
import shutil
import tempfile
from multiprocessing import Pool
from pathlib import Path
# from unittest.mock import MagicMock, patch

import h5py
import matplotlib.pyplot as plt
import numpy as np
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
        radial_distances = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0])
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
        radial_distances = np.array([])
        particle_mass = 1e10
        mass_density = 2.78e11
        
        with pytest.raises(ValueError, match='is empty'):
            calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
    def test_single_particle(self):
        """Test computation with single particle."""
        radial_distances = np.array([0.5])
        particle_mass = 1e10
        mass_density = 2.78e11
        
        r200m, v200_sq = calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
        assert r200m == 0.5
        assert v200_sq > 0

    def test_zero_distances_handling(self):
        """Test that zero distances are handled properly."""
        radial_distances = np.array([0.0, 0.0, 0.1, 0.2])
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
        radial_distances = np.linspace(0.1, 3.0, 20)
        
        r200m, v200_sq = calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
        assert r200m > 0
        assert v200_sq > 0

    def test_no_qualifying_particles(self):
        """Test when no particles satisfy the 200×ρ_m criterion."""
        # Create a very sparse distribution where density never reaches 200×ρ_m
        radial_distances = np.array([5.0, 10.0, 15.0, 20.0])
        particle_mass = 1e8  # Small particle mass
        mass_density = 1e12   # High background density
        
        r200m, v200_sq = calibration._compute_r200m_and_v200m(radial_distances, particle_mass, mass_density)
        
        # Should use outermost particle
        assert r200m == 20.0
        assert v200_sq > 0