import numpy
import pytest

from oasis import coordinates


def test_relative_coordinates():
    """Comprehensive test for relative_coordinates function."""
    
    # Test parameters
    boxsize = 100.0
    
    # Test 1: Basic non-periodic case
    x0 = numpy.array([50.0, 50.0, 50.0])
    x = numpy.array([[60.0, 40.0, 55.0]])
    expected = numpy.array([[10.0, -10.0, 5.0]])
    
    result = coordinates.relative_coordinates(x, x0, boxsize, periodic=False)
    numpy.testing.assert_array_almost_equal(result, expected)
    
    # Test 2: Periodic boundary conditions - standard case
    x0 = numpy.array([50.0, 50.0, 50.0])
    x = numpy.array([[60.0, 40.0, 55.0]])
    expected = numpy.array([[10.0, -10.0, 5.0]])  # Same as non-periodic for this case
    
    result = coordinates.relative_coordinates(x, x0, boxsize, periodic=True)
    numpy.testing.assert_array_almost_equal(result, expected)
    
    # Test 3: Periodic boundary conditions - wrap around case
    x0 = numpy.array([99.5, 50.0, 50.0])
    x = numpy.array([[4.0, 50.0, 50.0]])  # Should wrap around
    expected = numpy.array([[4.5, 0.0, 0.0]])  # 4.0 - 99.5 + 100 = 4.5 (closest distance)
    
    result = coordinates.relative_coordinates(x, x0, boxsize, periodic=True)
    numpy.testing.assert_array_almost_equal(result, expected)
    
    # Test 4: Periodic boundary conditions - negative wrap
    x0 = numpy.array([4.0, 50.0, 50.0])
    x = numpy.array([[99.5, 50.0, 50.0]])  # Should wrap in negative direction
    expected = numpy.array([[-4.5, 0.0, 0.0]])  # 99.5 - 4.0 - 100 = -4.5
    
    result = coordinates.relative_coordinates(x, x0, boxsize, periodic=True)
    numpy.testing.assert_array_almost_equal(result, expected)
    
    # Test 5: Multiple particles
    x0 = numpy.array([50.0, 50.0, 50.0])
    x = numpy.array([
        [60.0, 40.0, 55.0],
        [5.0, 95.0, 25.0],
        [45.0, 45.0, 45.0]
    ])
    
    expected_periodic = numpy.array([
        [10.0, -10.0, 5.0],
        [-45.0, 45.0, -25.0],
        [-5.0, -5.0, -5.0]
    ])
    
    result = coordinates.relative_coordinates(x, x0, boxsize, periodic=True)
    numpy.testing.assert_array_almost_equal(result, expected_periodic)
    
    # Test 6: Edge cases - exactly at box boundaries
    x0 = numpy.array([0.0, 0.0, 0.0])
    x = numpy.array([[50.0, 0.0, 0.0]])  # Exactly half box away
    
    result = coordinates.relative_coordinates(x, x0, boxsize, periodic=True)
    # Should be exactly 50.0 or -50.0, the implementation chooses -50.0
    expected = numpy.array([[-50.0, 0.0, 0.0]])
    numpy.testing.assert_array_almost_equal(result, expected)
    
    # Test 7: 1D arrays (should work)
    x0_1d = numpy.array([50.0])
    x_1d = numpy.array([[60.0], [40.0]])
    expected_1d = numpy.array([[10.0], [-10.0]])
    
    result = coordinates.relative_coordinates(x_1d, x0_1d, boxsize, periodic=False)
    numpy.testing.assert_array_almost_equal(result, expected_1d)
    
    # Test 8: Shape consistency
    x0 = numpy.array([50.0, 50.0, 50.0])
    x_single = numpy.array([[60.0, 40.0, 55.0]])
    x_multi = numpy.array([
        [60.0, 40.0, 55.0],
        [45.0, 45.0, 45.0]
    ])
    
    result_single = coordinates.relative_coordinates(x_single, x0, boxsize)
    result_multi = coordinates.relative_coordinates(x_multi, x0, boxsize)
    
    assert result_single.shape == (1, 3)
    assert result_multi.shape == (2, 3)
    
    # Test 9: TypeError for list inputs
    with pytest.raises(TypeError, match="Input 'x' must be an array"):
        coordinates.relative_coordinates([60.0, 40.0, 55.0], x0, boxsize)
    
    with pytest.raises(TypeError, match="Input 'x' must be an array"):
        coordinates.relative_coordinates((60.0, 40.0, 55.0), x0, boxsize)
    
    # Test 10: Different data types
    x0 = numpy.array([50.0, 50.0, 50.0])
    x_int = numpy.array([[60, 40, 55]], dtype=int)
    x_float32 = numpy.array([[60.0, 40.0, 55.0]], dtype=numpy.float32)
    
    result_int = coordinates.relative_coordinates(x_int, x0, boxsize)
    result_float32 = coordinates.relative_coordinates(x_float32, x0, boxsize)
    
    # Should work with different numeric types
    assert result_int.shape == (1, 3)
    assert result_float32.shape == (1, 3)
    
    # Test 11: Zero boxsize (edge case for non-periodic)
    result_zero_box = coordinates.relative_coordinates(x_single, x0, 0.0, periodic=False)
    expected_zero_box = numpy.array([[10.0, -10.0, 5.0]])
    numpy.testing.assert_array_almost_equal(result_zero_box, expected_zero_box)
    
    # Test 12: Very small boxsize with periodic (potential numerical issues)
    small_boxsize = 1e-10
    x0_small = numpy.array([0.0, 0.0, 0.0])
    x_small = numpy.array([[small_boxsize/4, 0.0, 0.0]])
    
    result_small = coordinates.relative_coordinates(x_small, x0_small, small_boxsize, periodic=True)
    # Should handle small numbers correctly
    assert result_small.shape == (1, 3)
    


def test_get_vr_vt_from_coordinates():
    # Single particle, mismatched shape (should raise)
    x = numpy.array([1., 1., 1.])
    v = numpy.array([1., 1., -1.])
    with pytest.raises(ValueError):
        coordinates.velocity_components(x, v)

    # Single particle, correct shape
    x = numpy.array([x])
    v = numpy.array([v])
    vr, vt, v2 = coordinates.velocity_components(x, v)

    vr_expected = numpy.sin(numpy.arccos(1./numpy.sqrt(3.))) * \
        (numpy.cos(numpy.pi/4.) + numpy.sin(numpy.pi/4.)) - 1./numpy.sqrt(3.)
    vt_expected = numpy.sin(numpy.arccos(1./numpy.sqrt(3.))) + 1./numpy.sqrt(3.) * \
        (numpy.cos(numpy.pi/4.) + numpy.sin(numpy.pi/4.))

    assert vr[0] == pytest.approx(vr_expected)
    assert vt[0] == pytest.approx(vt_expected)
    assert v2[0] == 3.

    # Zero velocity vector
    x = numpy.array([[2., 0., 0.]])
    v = numpy.array([[0., 0., 0.]])
    vr, vt, v2 = coordinates.velocity_components(x, v)
    assert vr[0] == pytest.approx(0.)
    assert vt[0] == pytest.approx(0.)
    assert v2[0] == pytest.approx(0.)

    # # Zero position vector (should raise or handle gracefully)
    # x = numpy.array([[0., 0., 0.]])
    # v = numpy.array([[1., 0., 0.]])
    # with pytest.raises(Exception):
    #     coordinates.velocity_components(x, v)

    # Multiple particles
    x = numpy.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    v = numpy.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    vr, vt, v2 = coordinates.velocity_components(x, v)
    assert vr.shape == (3,)
    assert vt.shape == (3,)
    assert v2.shape == (3,)
    assert all(isinstance(val, float) or isinstance(val, numpy.floating) for val in vr)
    assert all(isinstance(val, float) or isinstance(val, numpy.floating) for val in vt)
    assert all(isinstance(val, float) or isinstance(val, numpy.floating) for val in v2)

    # Negative coordinates and velocities
    x = numpy.array([[-1., -1., -1.]])
    v = numpy.array([[-1., -1., 1.]])
    vr, vt, v2 = coordinates.velocity_components(x, v)
    assert vr.shape == (1,)
    assert vt.shape == (1,)
    assert v2.shape == (1,)
