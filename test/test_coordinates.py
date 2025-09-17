import numpy
import pytest

from oasis import coordinates


class TestRelativeCoordinates:
    """Tests for relative_coordinates."""

    @pytest.mark.parametrize(
        "positions, reference, boxsize, periodic, expected",
        [
            # Non-periodic simple displacement
            (numpy.array([[60.0, 40.0, 55.0]]),
             numpy.array([50.0, 50.0, 50.0]), 100.0, False,
             numpy.array([[10.0, -10.0, 5.0]])),

            # Periodic: normal case (no wrapping needed)
            (numpy.array([[60.0, 40.0, 55.0]]),
             numpy.array([50.0, 50.0, 50.0]), 100.0, True,
             numpy.array([[10.0, -10.0, 5.0]])),

            # Periodic: positive wrap-around
            (numpy.array([[4.0, 50.0, 50.0]]),
             numpy.array([99.5, 50.0, 50.0]), 100.0, True,
             numpy.array([[4.5, 0.0, 0.0]])),

            # Periodic: negative wrap-around
            (numpy.array([[99.5, 50.0, 50.0]]),
             numpy.array([4.0, 50.0, 50.0]), 100.0, True,
             numpy.array([[-4.5, 0.0, 0.0]])),

            # Periodic: exactly at half-box boundary (choose negative direction)
            (numpy.array([[50.0, 0.0, 0.0]]),
             numpy.array([0.0, 0.0, 0.0]), 100.0, True,
             numpy.array([[-50.0, 0.0, 0.0]])),
        ],
    )
    def test_displacements(self, positions, reference, boxsize, periodic, expected):
        """Test displacements with and without periodic boundary conditions."""
        result = coordinates.relative_coordinates(
            positions, reference, boxsize, periodic)
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_multiple_particles(self):
        """Multiple particles with mixed periodic displacements."""
        x0 = numpy.array([50.0, 50.0, 50.0])
        x = numpy.array([
            [60.0, 40.0, 55.0],
            [5.0, 95.0, 25.0],
            [45.0, 45.0, 45.0],
        ])
        expected = numpy.array([
            [10.0, -10.0, 5.0],
            [-45.0, 45.0, -25.0],
            [-5.0, -5.0, -5.0],
        ])
        result = coordinates.relative_coordinates(x, x0, 100.0, periodic=True)
        numpy.testing.assert_array_almost_equal(result, expected)

    def test_dtype_and_small_box(self):
        """Check behavior with different dtypes and very small box size."""
        x0 = numpy.array([50.0, 50.0, 50.0])
        x_int = numpy.array([[60, 40, 55]], dtype=int)
        x_float32 = numpy.array([[60.0, 40.0, 55.0]], dtype=numpy.float32)

        assert coordinates.relative_coordinates(
            x_int, x0, 100.0).shape == (1, 3)
        assert coordinates.relative_coordinates(
            x_float32, x0, 100.0).shape == (1, 3)

        # Very small boxsize
        small_box = 1e-10
        x = numpy.array([[small_box / 4, 0.0, 0.0]])
        x0 = numpy.zeros(3)
        result = coordinates.relative_coordinates(x, x0, small_box)
        assert result.shape == (1, 3)

    @pytest.mark.parametrize(
        "positions, reference, boxsize",
        [
            # Wrong shape (positions)
            (numpy.array([[1.0, 2.0]]), numpy.zeros(3), 1.0),

            # Wrong shape (reference)
            (numpy.array([[1.0, 2.0, 3.0]]), numpy.array([0.0, 0.0]), 1.0),

            # box size is zero
            (numpy.zeros((1, 3)), numpy.zeros(3), 0.0),

            # negative box size
            (numpy.zeros((1, 3)), numpy.zeros(3), -1.0),

        ],
    )
    def test_value_errors(self, positions, reference, boxsize):
        """Invalid inputs should raise ValueError."""
        with pytest.raises(ValueError):
            coordinates.relative_coordinates(positions, reference, boxsize)

    @pytest.mark.parametrize(
        "positions, reference, boxsize",
        [   
            ("bad-input", numpy.zeros(3), 1.0),
            # non-scalar box size
            (numpy.zeros((1, 3)), numpy.zeros(3), [1.0, 2.0]),
        ],
    )
    def test_type_error(self, positions, reference, boxsize):
        """Non-numpy-convertible inputs should raise TypeError."""
        with pytest.raises(TypeError):
            coordinates.relative_coordinates(positions, reference, boxsize)


class TestVelocityComponents:
    """Tests for velocity_components."""

    def test_single_and_multiple_particles(self):
        """Check single particle, zero velocity, and multiple particle cases."""
        # Single particle
        x = numpy.array([[1.0, 1.0, 1.0]])
        v = numpy.array([[1.0, 1.0, -1.0]])
        vr, vt, v2 = coordinates.velocity_components(x, v)

        vr_expected = numpy.sin(numpy.arccos(1./numpy.sqrt(3.))) * \
            (numpy.cos(numpy.pi/4.) + numpy.sin(numpy.pi/4.)) - 1./numpy.sqrt(3.)
        vt_expected = numpy.sin(numpy.arccos(1./numpy.sqrt(3.))) + \
            1./numpy.sqrt(3.) * (numpy.cos(numpy.pi/4.) +
                                 numpy.sin(numpy.pi/4.))

        assert vr[0] == pytest.approx(vr_expected)
        assert vt[0] == pytest.approx(vt_expected)
        assert v2[0] == pytest.approx(3.0)
        assert pytest.approx(vr[0]**2 + vt[0]**2) == v2[0]

        # Zero velocity vector
        vr, vt, v2 = coordinates.velocity_components(
            numpy.array([[2.0, 0.0, 0.0]]), numpy.array([[0.0, 0.0, 0.0]]))
        assert vr[0] == pytest.approx(0.0)
        assert vt[0] == pytest.approx(0.0)
        assert v2[0] == pytest.approx(0.0)

        # Multiple particles
        x = numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        v = numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        vr, vt, v2 = coordinates.velocity_components(x, v)
        assert vr.shape == (3,)
        assert vt.shape == (3,)
        assert v2.shape == (3,)
        assert numpy.allclose(vt, 0.0)  # purely radial
        assert numpy.allclose(vr**2, v2)

    @pytest.mark.parametrize(
        "pos, vel, expected_vr, expected_vt, expected_v2",
        [
            # Origin particle: radial=0, tangential=|v|, v2=|v|^2
            (numpy.array([0.0, 0.0, 0.0]), numpy.array(
                [1.0, 2.0, 2.0]), 0.0, 3.0, 9.0),

            # Purely radial
            (numpy.array([1.0, 0.0, 0.0]), numpy.array(
                [2.0, 0.0, 0.0]), 2.0, 0.0, 4.0),

            # Purely tangential
            (numpy.array([1.0, 0.0, 0.0]), numpy.array(
                [0.0, 2.0, 0.0]), 0.0, 2.0, 4.0),
        ],
    )
    def test_radial_and_tangential_cases(self, pos, vel, expected_vr, expected_vt, expected_v2):
        """Test special cases: origin, purely radial, purely tangential."""
        vr, vt, v2 = coordinates.velocity_components(pos, vel)
        numpy.testing.assert_allclose(vr, [expected_vr])
        numpy.testing.assert_allclose(vt, [expected_vt])
        numpy.testing.assert_allclose(v2, [expected_v2])

    def test_negative_coordinates_and_shape_errors(self):
        """Check negative coordinates and shape mismatch handling."""
        vr, vt, v2 = coordinates.velocity_components(numpy.array(
            [[-1.0, -1.0, -1.0]]), numpy.array([[-1.0, -1.0, 1.0]]))
        assert vr.shape == (1,)
        assert vt.shape == (1,)
        assert v2.shape == (1,)
        assert pytest.approx(vr[0]**2 + vt[0]**2) == v2[0]
        assert v2[0] > 0  # total velocity magnitude should be positive

        # Shape mismatch
        with pytest.raises(ValueError):
            coordinates.velocity_components(
                numpy.zeros((2, 3)), numpy.zeros((3, 3)))

    def test_type_error(self):
        """Non-array-convertible inputs should raise TypeError."""
        with pytest.raises(TypeError):
            coordinates.velocity_components("bad-input", numpy.zeros((1, 3)))
