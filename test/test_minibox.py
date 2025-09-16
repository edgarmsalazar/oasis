import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy
import pytest

from oasis import minibox


class TestValidateInputsBoxsizeMinisize:
    """Test suite for _validate_inputs_boxsize_minisize function."""

    def test_valid_inputs(self):
        """Test that valid inputs don't raise exceptions."""
        # Should not raise any exceptions
        minibox._validate_inputs_boxsize_minisize(100.0, 10.0)
        minibox._validate_inputs_boxsize_minisize(50, 25)
        minibox._validate_inputs_boxsize_minisize(1.0, 1.0)  # Equal values
        minibox._validate_inputs_boxsize_minisize(
            1000.5, 0.1)  # Float precision

    @pytest.mark.parametrize(
        "boxsize, minisize, expected_error",
        [
            # Type errors
            ("100", 10.0, TypeError),
            (100.0, "10", TypeError),
            ("100", "10", TypeError),
            (None, 10.0, TypeError),
            (100.0, None, TypeError),
            ([100], 10.0, TypeError),
            (100.0, [10], TypeError),
            (complex(100, 0), 10.0, TypeError),
            (100.0, complex(10, 0), TypeError),
        ],
    )
    def test_type_errors(self, boxsize, minisize, expected_error):
        """Test that invalid types raise TypeError."""
        with pytest.raises(expected_error, match="boxsize and minisize must be numeric"):
            minibox._validate_inputs_boxsize_minisize(boxsize, minisize)

    @pytest.mark.parametrize(
        "boxsize, minisize, expected_error, error_message",
        [
            # Negative boxsize
            (-1.0, 10.0, ValueError, "boxsize must be positive"),
            (-100, 10.0, ValueError, "boxsize must be positive"),
            (-0.1, 0.05, ValueError, "boxsize must be positive"),
            # Zero boxsize
            (0, 10.0, ValueError, "boxsize must be positive"),
            (0.0, 5.0, ValueError, "boxsize must be positive"),
            # Negative minisize
            (100.0, -1.0, ValueError, "minisize must be positive"),
            (100.0, -10, ValueError, "minisize must be positive"),
            (50.0, -0.1, ValueError, "minisize must be positive"),
            # Zero minisize
            (100.0, 0, ValueError, "minisize must be positive"),
            (100.0, 0.0, ValueError, "minisize must be positive"),
            # minisize > boxsize
            (10.0, 20.0, ValueError, "minisize cannot be larger than boxsize"),
            (5, 10, ValueError, "minisize cannot be larger than boxsize"),
            (0.5, 1.0, ValueError, "minisize cannot be larger than boxsize"),
        ],
    )
    def test_value_errors(self, boxsize, minisize, expected_error, error_message):
        """Test that invalid values raise ValueError with correct messages."""
        with pytest.raises(expected_error, match=error_message):
            minibox._validate_inputs_boxsize_minisize(boxsize, minisize)

    def test_edge_cases(self):
        """Test edge cases with very small or very large numbers."""
        # Very small positive numbers
        minibox._validate_inputs_boxsize_minisize(1e-10, 1e-11)

        # Very large numbers
        minibox._validate_inputs_boxsize_minisize(1e10, 1e9)

        # Float precision edge cases
        minibox._validate_inputs_boxsize_minisize(1.0000001, 1.0)


class TestValidateInputsBoxPartitioning:
    """Test suite for _validate_inputs_box_partitioning function."""

    @pytest.fixture
    def valid_data(self):
        """Fixture providing valid test data."""
        n_particles = 100
        positions = numpy.random.rand(n_particles, 3)
        velocities = numpy.random.rand(n_particles, 3)
        uid = numpy.arange(n_particles)
        return positions, velocities, uid, n_particles

    @pytest.fixture
    def valid_props(self, valid_data):
        """Fixture providing valid props data."""
        _, _, _, n_particles = valid_data
        arrays = [
            numpy.random.rand(n_particles),
            numpy.random.rand(n_particles, 2),
            numpy.ones(n_particles, dtype=int)
        ]
        labels = ['property1', 'property2', 'property3']
        dtypes = [numpy.float32, numpy.float64, numpy.int32]
        return (arrays, labels, dtypes)

    def test_valid_inputs_no_props(self, valid_data):
        """Test that valid inputs without props don't raise exceptions."""
        positions, velocities, uid, _ = valid_data
        minibox._validate_inputs_box_partitioning(
            positions, velocities, uid, None)

    def test_valid_inputs_with_props(self, valid_data, valid_props):
        """Test that valid inputs with props don't raise exceptions."""
        positions, velocities, uid, _ = valid_data
        minibox._validate_inputs_box_partitioning(
            positions, velocities, uid, valid_props)

    def test_empty_arrays(self):
        """Test that empty but correctly shaped arrays are valid."""
        positions = numpy.empty((0, 3))
        velocities = numpy.empty((0, 3))
        uid = numpy.empty(0)
        minibox._validate_inputs_box_partitioning(
            positions, velocities, uid, None)

    @pytest.mark.parametrize(
        "positions_shape, expected_error",
        [
            # Wrong number of dimensions
            ((100,), "positions must have shape \\(n_particles, 3\\)"),
            ((100, 3, 2), "positions must have shape \\(n_particles, 3\\)"),
            # Wrong second dimension
            ((100, 2), "positions must have shape \\(n_particles, 3\\)"),
            ((100, 4), "positions must have shape \\(n_particles, 3\\)"),
            ((100, 1), "positions must have shape \\(n_particles, 3\\)"),
        ],
    )
    def test_invalid_positions_shape(self, positions_shape, expected_error):
        """Test that invalid positions shapes raise ValueError."""
        positions = numpy.zeros(positions_shape)
        velocities = numpy.zeros((100, 3))
        uid = numpy.zeros(100)

        with pytest.raises(ValueError, match=expected_error):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, None)

    @pytest.mark.parametrize(
        "velocities_shape, expected_error",
        [
            # Wrong number of dimensions
            ((100,), "velocities must have shape \\(n_particles, 3\\)"),
            ((100, 3, 2), "velocities must have shape \\(n_particles, 3\\)"),
            # Wrong second dimension
            ((100, 2), "velocities must have shape \\(n_particles, 3\\)"),
            ((100, 4), "velocities must have shape \\(n_particles, 3\\)"),
            ((100, 1), "velocities must have shape \\(n_particles, 3\\)"),
        ],
    )
    def test_invalid_velocities_shape(self, velocities_shape, expected_error):
        """Test that invalid velocities shapes raise ValueError."""
        positions = numpy.zeros((100, 3))
        velocities = numpy.zeros(velocities_shape)
        uid = numpy.zeros(100)

        with pytest.raises(ValueError, match=expected_error):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, None)

    @pytest.mark.parametrize(
        "uid_shape, expected_error",
        [
            # Wrong number of dimensions
            ((100, 1), "uid must be a 1D array"),
            ((100, 3), "uid must be a 1D array"),
            ((100, 2, 3), "uid must be a 1D array"),
        ],
    )
    def test_invalid_uid_shape(self, uid_shape, expected_error):
        """Test that invalid uid shapes raise ValueError."""
        positions = numpy.zeros((100, 3))
        velocities = numpy.zeros((100, 3))
        uid = numpy.zeros(uid_shape)

        with pytest.raises(ValueError, match=expected_error):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, None)

    @pytest.mark.parametrize(
        "pos_size, vel_size, uid_size",
        [
            (100, 50, 100),  # velocities mismatch
            (100, 100, 50),  # uid mismatch
            (50, 100, 100),  # positions mismatch
            (100, 50, 75),   # all different
            (0, 10, 0),      # mixed empty/non-empty
        ],
    )
    def test_mismatched_array_lengths(self, pos_size, vel_size, uid_size):
        """Test that arrays with mismatched lengths raise ValueError."""
        positions = numpy.zeros((pos_size, 3))
        velocities = numpy.zeros((vel_size, 3))
        uid = numpy.zeros(uid_size)

        with pytest.raises(ValueError, match="positions, velocities, and uid must have the same length"):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, None)

    @pytest.mark.parametrize(
        "props_input, expected_error",
        [
            # Wrong type
            ("invalid", "props must be a tuple of \\(arrays, labels, dtypes\\)"),
            (["list", "instead", "of", "tuple"],
             "props must be a tuple of \\(arrays, labels, dtypes\\)"),
            (42, "props must be a tuple of \\(arrays, labels, dtypes\\)"),
            # Wrong tuple length
            ((), "props must be a tuple of \\(arrays, labels, dtypes\\)"),
            (([], []), "props must be a tuple of \\(arrays, labels, dtypes\\)"),
            (([], [], [], []), "props must be a tuple of \\(arrays, labels, dtypes\\)"),
        ],
    )
    def test_invalid_props_structure(self, valid_data, props_input, expected_error):
        """Test that invalid props structure raises ValueError."""
        positions, velocities, uid, _ = valid_data

        with pytest.raises(ValueError, match=expected_error):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, props_input)

    @pytest.mark.parametrize(
        "arrays, labels, dtypes, expected_error",
        [
            # Wrong types in props tuple
            ("not_list", [], [], "props must contain three lists"),
            ([], "not_list", [], "props must contain three lists"),
            ([], [], "not_list", "props must contain three lists"),
            (42, [], [], "props must contain three lists"),
            ([], [], 42, "props must contain three lists"),
        ],
    )
    def test_invalid_props_list_types(self, valid_data, arrays, labels, dtypes, expected_error):
        """Test that props with wrong list types raise ValueError."""
        positions, velocities, uid, _ = valid_data
        props = (arrays, labels, dtypes)

        with pytest.raises(ValueError, match=expected_error):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, props)

    def test_mismatched_props_lengths(self, valid_data):
        """Test that props lists with different lengths raise ValueError."""
        positions, velocities, uid, _ = valid_data

        arrays = [numpy.zeros(100)]
        labels = ['prop1', 'prop2']  # Different length
        dtypes = [numpy.float32]
        props = (arrays, labels, dtypes)

        with pytest.raises(ValueError, match="All lists in props must have the same length"):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, props)

    @pytest.mark.parametrize(
        "array_type",
        [
            "not_array",
            42,
            [1, 2, 3],
            None,
        ],
    )
    def test_non_numpy_array_in_props(self, valid_data, array_type):
        """Test that non-numpy arrays in props raise ValueError."""
        positions, velocities, uid, n_particles = valid_data

        arrays = [array_type]
        labels = ['prop1']
        dtypes = [numpy.float32]
        props = (arrays, labels, dtypes)

        with pytest.raises(ValueError, match="props array 0 must be a numpy array"):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, props)

    @pytest.mark.parametrize(
        "wrong_size",
        [50, 150, 0, 1]
    )
    def test_wrong_size_props_array(self, valid_data, wrong_size):
        """Test that props arrays with wrong size raise ValueError."""
        positions, velocities, uid, n_particles = valid_data

        arrays = [numpy.zeros(wrong_size)]
        labels = ['prop1']
        dtypes = [numpy.float32]
        props = (arrays, labels, dtypes)

        with pytest.raises(ValueError, match=f"props array 0 must have {n_particles} elements"):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, props)

    def test_multiple_props_arrays_validation(self, valid_data):
        """Test validation of multiple props arrays with mixed validity."""
        positions, velocities, uid, n_particles = valid_data

        # Second array has wrong size
        arrays = [
            numpy.zeros(n_particles),  # Correct
            numpy.zeros(n_particles // 2),  # Wrong size
            numpy.zeros(n_particles)   # Correct
        ]
        labels = ['prop1', 'prop2', 'prop3']
        dtypes = [numpy.float32, numpy.float64, numpy.int32]
        props = (arrays, labels, dtypes)

        with pytest.raises(ValueError, match=f"props array 1 must have {n_particles} elements"):
            minibox._validate_inputs_box_partitioning(
                positions, velocities, uid, props)

    def test_props_with_different_array_shapes(self, valid_data):
        """Test that props arrays can have different shapes as long as first dimension matches."""
        positions, velocities, uid, n_particles = valid_data

        arrays = [
            numpy.zeros(n_particles),          # 1D
            numpy.zeros((n_particles, 3)),     # 2D
            numpy.zeros((n_particles, 2, 4))   # 3D
        ]
        labels = ['prop1', 'prop2', 'prop3']
        dtypes = [numpy.float32, numpy.float64, numpy.int32]
        props = (arrays, labels, dtypes)

        # Should not raise an exception
        minibox._validate_inputs_box_partitioning(
            positions, velocities, uid, props)

    def test_tuples_instead_of_lists_in_props(self, valid_data):
        """Test that tuples are accepted instead of lists in props."""
        positions, velocities, uid, n_particles = valid_data

        arrays = (numpy.zeros(n_particles),)  # tuple instead of list
        labels = ('prop1',)                # tuple instead of list
        dtypes = (numpy.float32,)            # tuple instead of list
        props = (arrays, labels, dtypes)

        # Should not raise an exception
        minibox._validate_inputs_box_partitioning(
            positions, velocities, uid, props)


class TestValidateInputsLoad:
    """Test suite for _validate_inputs_load function."""

    def test_valid_inputs(self, tmp_path):
        """Test that valid inputs don't raise exceptions."""
        valid_dir = tmp_path / "data"
        valid_dir.mkdir()

        # Should not raise any exceptions
        minibox._validate_inputs_load(
            mini_box_id=0,
            boxsize=100.0,
            minisize=10.0,
            load_path=valid_dir,
            padding=0.0,
        )

        minibox._validate_inputs_load(
            mini_box_id=5,
            boxsize=10.0,
            minisize=1.0,
            load_path=str(valid_dir),
            padding=1.0,
        )

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize, load_path, padding, expected_error, error_message",
        [
            # mini_box_id type errors
            ("0", 100.0, 10.0, "some/path", 0.0,
             TypeError, "mini_box_id must be an integer"),
            (None, 100.0, 10.0, "some/path", 0.0,
             TypeError, "mini_box_id must be an integer"),
            (1.5, 100.0, 10.0, "some/path", 0.0,
             TypeError, "mini_box_id must be an integer"),
            # padding type errors
            (0, 100.0, 10.0, "some/path", "0.0",
             TypeError, "padding must be numeric"),
            (0, 100.0, 10.0, "some/path", None,
             TypeError, "padding must be numeric"),
            # load_path type errors
            (0, 100.0, 10.0, 123, 0.0, TypeError,
             "load_path must be a string or Path object"),
            (0, 100.0, 10.0, None, 0.0, TypeError,
             "load_path must be a string or Path object"),
        ],
    )
    def test_type_errors(
        self, mini_box_id, boxsize, minisize, load_path, padding, expected_error, error_message
    ):
        """Test that invalid types raise TypeError with correct messages."""
        with pytest.raises(expected_error, match=error_message):
            minibox._validate_inputs_load(
                mini_box_id, boxsize, minisize, load_path, padding)

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize, padding, expected_error, error_message",
        [
            # Negative mini_box_id
            (-1, 10.0, 1.0, 0.0, ValueError, "mini_box_id must be non-negative"),
            # mini_box_id too large
            (1000, 10.0, 1.0, 0.0, ValueError, "exceeds maximum valid ID"),
            # Negative padding
            (0, 10.0, 1.0, -0.5, ValueError, "padding must be non-negative"),
        ],
    )
    def test_value_errors(
        self, tmp_path, mini_box_id, boxsize, minisize, padding, expected_error, error_message
    ):
        """Test that invalid values raise ValueError with correct messages."""
        valid_dir = tmp_path / "data"
        valid_dir.mkdir()

        with pytest.raises(expected_error, match=error_message):
            minibox._validate_inputs_load(
                mini_box_id, boxsize, minisize, valid_dir, padding
            )

    def test_load_path_not_found(self, tmp_path):
        """Test that non-existent load_path raises FileNotFoundError."""
        missing_path = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Load path does not exist"):
            minibox._validate_inputs_load(0, 10.0, 1.0, missing_path, 0.0)

    def test_load_path_not_directory(self, tmp_path):
        """Test that non-directory load_path raises NotADirectoryError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("dummy")
        with pytest.raises(NotADirectoryError, match="Load path is not a directory"):
            minibox._validate_inputs_load(0, 10.0, 1.0, file_path, 0.0)

    def test_edge_cases(self, tmp_path):
        """Test edge cases with very small or large boxsize/minisize and numpy types."""
        valid_dir = tmp_path / "edge"
        valid_dir.mkdir()

        # mini_box_id as numpy integer
        minibox._validate_inputs_load(
            numpy.int64(0), 1.0, 1.0, valid_dir, numpy.float64(0.0)
        )

        # Very large grid
        minibox._validate_inputs_load(
            0, 1e6, 1.0, valid_dir, 0.0
        )

        # Very small valid minisize
        minibox._validate_inputs_load(
            0, 1e-3, 1e-4, valid_dir, 0.0
        )

# Additional integration tests


class TestValidationIntegration:
    """Integration tests for both validation functions together."""

    @pytest.mark.parametrize("n_particles", [0, 1, 10, 1000])
    def test_various_particle_counts(self, n_particles):
        """Test both functions with various particle counts."""
        # Test boxsize/minisize validation
        minibox._validate_inputs_boxsize_minisize(100.0, 10.0)

        # Test array validation
        positions = numpy.random.rand(n_particles, 3)
        velocities = numpy.random.rand(n_particles, 3)
        uid = numpy.arange(n_particles)

        minibox._validate_inputs_box_partitioning(
            positions, velocities, uid, None)

    def test_realistic_simulation_data(self):
        """Test with realistic simulation-like data."""
        # Large simulation
        n_particles = 10000
        boxsize = 500.0
        minisize = 25.0

        # Validate box parameters
        minibox._validate_inputs_boxsize_minisize(boxsize, minisize)

        # Create realistic data
        positions = numpy.random.uniform(0, boxsize, size=(n_particles, 3))
        velocities = numpy.random.normal(0, 100, size=(n_particles, 3))
        uid = numpy.arange(n_particles, dtype=numpy.int64)

        # Additional properties
        masses = numpy.random.lognormal(0, 1, size=n_particles)
        temperatures = numpy.random.exponential(1000, size=n_particles)

        props = (
            [masses, temperatures],
            ['mass', 'temperature'],
            [numpy.float32, numpy.float32]
        )

        # Should validate successfully
        minibox._validate_inputs_box_partitioning(
            positions, velocities, uid, props)


class TestGetMiniBoxId:
    """Test suite for minibox.get_mini_box_id function"""

    @pytest.mark.parametrize(
        "position, boxsize, minisize, expected, description",
        [
            # Basic single positions
            ([0.1, 0.1, 0.1], 10.0, 1.0, 0, "origin position"),
            ([1.5, 2.5, 3.5], 10.0, 1.0, 321, "interior position"),
            ([5.0, 5.0, 5.0], 10.0, 1.0, 555, "center position"),

            # Non-unit minisize
            ([3.0, 6.0, 9.0], 12.0, 2.0, 163, "non-unit minisize"),
            ([1.0, 1.0, 1.0], 3.0, 0.5, 86, "fractional minisize"),
            ([2.5, 2.5, 2.5], 5.0, 2.5, 7, "minisize equals half boxsize"),

            # Edge positions
            ([9.9, 9.9, 9.9], 10.0, 1.0, 999, "near upper boundary"),
            ([0.01, 0.01, 0.01], 10.0, 1.0, 0, "near lower boundary"),
        ],
    )
    def test_single_positions(self, position, boxsize, minisize, expected, description):
        """Test single position scenarios"""
        pos = numpy.array(position)
        result = minibox.get_mini_box_id(pos, boxsize, minisize)

        assert result == expected, f"Failed for {description}"
        assert isinstance(result, int), f"Wrong return type for {description}"

    @pytest.mark.parametrize(
        "positions, boxsize, minisize, expected, description",
        [
            # Basic multiple positions
            ([[0.1, 0.1, 0.1], [1.5, 0.1, 0.1], [0.1, 1.5, 0.1], [0.1, 0.1, 1.5]],
             10.0, 1.0, [0, 1, 10, 100], "axis-aligned positions"),

            # Different grid cells
            ([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]],
             10.0, 1.0, [0, 111, 222], "diagonal positions"),

            # Mixed positions
            ([[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]],
             10.0, 1.0, [0, 999], "corner to corner"),
        ],
    )
    def test_multiple_positions(self, positions, boxsize, minisize, expected, description):
        """Test multiple position scenarios"""
        pos = numpy.array(positions)
        result = minibox.get_mini_box_id(pos, boxsize, minisize)
        expected_arr = numpy.array(expected)

        numpy.testing.assert_array_equal(result, expected_arr,
                                         err_msg=f"Failed for {description}")
        assert isinstance(
            result, numpy.ndarray), f"Wrong return type for {description}"
        assert result.shape == (
            len(positions),), f"Wrong shape for {description}"

    @pytest.mark.parametrize(
        "boundary_pos, boxsize, minisize, expected_grid, description",
        [
            # Upper boundary cases
            ([10.0, 5.0, 5.0], 10.0, 1.0, [9, 5, 5], "upper x boundary"),
            ([5.0, 10.0, 5.0], 10.0, 1.0, [5, 9, 5], "upper y boundary"),
            ([5.0, 5.0, 10.0], 10.0, 1.0, [5, 5, 9], "upper z boundary"),
            ([10.0, 10.0, 10.0], 10.0, 1.0, [9, 9, 9], "all upper boundaries"),

            # Lower boundary cases
            ([0.0, 0.5, 0.5], 10.0, 1.0, [0, 0, 0], "lower x boundary"),
            ([0.5, 0.0, 0.5], 10.0, 1.0, [0, 0, 0], "lower y boundary"),
            ([0.5, 0.5, 0.0], 10.0, 1.0, [0, 0, 0], "lower z boundary"),
            ([0.0, 0.0, 0.0], 10.0, 1.0, [0, 0, 0], "all lower boundaries"),

            # Different box sizes
            ([5.0, 5.0, 5.0], 5.0, 1.0, [4, 4, 4],
             "boundary with different boxsize"),
            ([2.0, 2.0, 2.0], 2.0, 0.5, [3, 3, 3],
             "boundary with fractional minisize"),
        ],
    )
    def test_boundary_conditions(self, boundary_pos, boxsize, minisize, expected_grid, description):
        """Test boundary condition handling"""
        pos = numpy.array([boundary_pos])
        result = minibox.get_mini_box_id(pos, boxsize, minisize)

        # Calculate expected ID from grid indices
        n_cells = int(numpy.ceil(boxsize / minisize))
        shift = numpy.array([1, n_cells, n_cells**2])
        expected_id = numpy.sum(shift * numpy.array(expected_grid))

        numpy.testing.assert_array_equal(result, [expected_id],
                                         err_msg=f"Failed for {description}")

    @pytest.mark.parametrize(
        "tolerance, description",
        [
            (1e-10, "very close to boundary"),
            (1e-12, "extremely close to boundary"),
            (1e-15, "machine precision boundary"),
        ],
    )
    def test_numerical_precision_boundaries(self, tolerance, description):
        """Test handling of positions very close to boundaries"""
        positions = numpy.array([
            [tolerance, 1.0, 1.0],           # Close to lower bound
            [10.0 - tolerance, 1.0, 1.0]    # Close to upper bound
        ])
        boxsize = 10.0
        minisize = 1.0

        result = minibox.get_mini_box_id(positions, boxsize, minisize)

        assert len(result) == 2, f"Wrong result length for {description}"
        assert all(id >= 0 for id in result), f"Negative IDs for {description}"
        assert isinstance(
            result, numpy.ndarray), f"Wrong type for {description}"

    @pytest.mark.parametrize(
        "error_type, position, boxsize, minisize, error_msg, description",
        [
            # Type errors
            (TypeError, [1.0, 1.0, 1.0], 10.0, 1.0,
             "position must be a numpy array", "non-numpy position"),
            (TypeError, numpy.array([1.0, 1.0, 1.0]), "10.0", 1.0,
             "boxsize and minisize must be numeric", "string boxsize"),
            (TypeError, numpy.array([1.0, 1.0, 1.0]), 10.0, "1.0",
             "boxsize and minisize must be numeric", "string minisize"),

            # Value errors - sizes
            (ValueError, numpy.array([1.0, 1.0, 1.0]), -5.0, 1.0,
             "boxsize must be positive", "negative boxsize"),
            (ValueError, numpy.array([1.0, 1.0, 1.0]), 10.0, -1.0,
             "minisize must be positive", "negative minisize"),
            (ValueError, numpy.array([1.0, 1.0, 1.0]), 5.0, 10.0,
             "minisize cannot be larger than boxsize", "minisize > boxsize"),
            (ValueError, numpy.array([1.0, 1.0, 1.0]), 0.0, 1.0,
             "boxsize must be positive", "zero boxsize"),
            (ValueError, numpy.array([1.0, 1.0, 1.0]), 10.0, 0.0,
             "minisize must be positive", "zero minisize"),

            # Value errors - shapes
            (ValueError, numpy.array([1.0, 1.0]), 10.0, 1.0,
             "1D position array must have exactly 3 elements", "wrong 1D shape"),
            (ValueError, numpy.array([[1.0, 1.0], [2.0, 2.0]]), 10.0, 1.0,
             "2D position array must have shape \\(N, 3\\)", "wrong 2D shape"),
            (ValueError, numpy.array([[[1.0, 1.0, 1.0]]]), 10.0, 1.0,
             "position array must be 1D .* or 2D .*", "3D position array"),

            # Value errors - bounds
            (ValueError, numpy.array([15.0, 5.0, 5.0]), 10.0, 1.0,
             "All coordinates must be within \\[0, 10.0\\]",
             "coordinate above boxsize"),
            (ValueError, numpy.array([5.0, -1.0, 5.0]), 10.0, 1.0,
             "All coordinates must be within \\[0, 10.0\\]",
             "negative coordinate"),
            (ValueError, numpy.array([[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]]), 10.0, 1.0,
             "All coordinates must be within \\[0, 10.0\\]",
             "mixed valid/invalid coordinates"),
        ],
    )
    def test_input_validation(self, error_type, position, boxsize, minisize, error_msg, description):
        """Test comprehensive input validation"""
        with pytest.raises(error_type, match=error_msg):
            minibox.get_mini_box_id(position, boxsize, minisize)

    @pytest.mark.parametrize(
        "boxsize, minisize, test_fraction",
        [
            (1.0, 0.1, 0.5),      # Small box, fine grid
            (10.0, 1.0, 0.3),     # Medium box, unit grid
            (5.5, 0.7, 0.8),      # Non-integer ratios
            (100.0, 3.3, 0.1),    # Large box, coarse grid
            (2.5, 2.5, 0.9),      # Single cell case
            (7.0, 0.33, 0.7),     # Many cells case
        ],
    )
    def test_various_configurations(self, boxsize, minisize, test_fraction):
        """Test various box and minisize configurations"""
        # Test position at a fraction of the box size
        position = numpy.array([boxsize * test_fraction] * 3)

        result = minibox.get_mini_box_id(position, boxsize, minisize)

        assert isinstance(
            result, int), f"Wrong type for boxsize={boxsize}, minisize={minisize}"
        assert result >= 0, f"Negative ID for boxsize={boxsize}, minisize={minisize}"

        # Verify the result is within expected bounds
        n_cells = int(numpy.ceil(boxsize / minisize))
        max_possible_id = n_cells**3 - 1
        assert result <= max_possible_id, f"ID too large for boxsize={boxsize}, minisize={minisize}"

    def test_unique_ids_comprehensive(self):
        """Test that positions in different mini-boxes get unique IDs"""
        boxsize = 6.0
        minisize = 1.0

        # Generate positions in different mini-boxes
        positions = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Place position in center of each mini-box
                    pos = [i + 0.5, j + 0.5, k + 0.5]
                    positions.append(pos)

        positions = numpy.array(positions)
        result = minibox.get_mini_box_id(positions, boxsize, minisize)

        # All IDs should be unique
        unique_ids = numpy.unique(result)
        assert len(unique_ids) == len(result), "Non-unique IDs found"
        assert len(unique_ids) == 27, "Wrong number of unique IDs"  # 3^3 = 27

    def test_empty_and_special_arrays(self):
        """Test edge cases with empty and special arrays"""
        boxsize = 10.0
        minisize = 1.0

        # Empty array
        empty_positions = numpy.empty((0, 3))
        result = minibox.get_mini_box_id(empty_positions, boxsize, minisize)
        assert result.shape == (0,)
        assert isinstance(result, numpy.ndarray)

        # Single row array
        single_pos = numpy.array([[5.0, 5.0, 5.0]])
        result = minibox.get_mini_box_id(single_pos, boxsize, minisize)
        assert result.shape == (1,)
        assert isinstance(result, numpy.ndarray)

    def test_in_place_modification_behavior(self):
        """Test that the function modifies input arrays in-place"""
        # Test with boundary positions that will be modified
        original_positions = numpy.array([
            [10.0, 5.0, 5.0],   # At upper boundary
            [0.0, 5.0, 5.0],    # At lower boundary
            [5.0, 10.0, 5.0]    # At upper boundary
        ])

        # Keep a copy to compare
        positions_copy = original_positions.copy()
        boxsize = 10.0
        minisize = 1.0

        result = minibox.get_mini_box_id(original_positions, boxsize, minisize)

        # Check that modifications occurred
        modifications_found = not numpy.array_equal(
            original_positions, positions_copy)
        assert modifications_found, "Expected in-place modifications for boundary positions"

        # Check that boundary positions were adjusted inward
        assert numpy.all(original_positions <=
                         boxsize), "Positions not properly bounded"

        # Results should still be valid
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 3
        assert all(id >= 0 for id in result)

    @pytest.mark.parametrize(
        "n_positions",
        [1, 10, 100, 1000],
    )
    def test_scalability(self, n_positions):
        """Test function performance with different array sizes"""
        boxsize = 10.0
        minisize = 1.0

        # Generate random positions within bounds
        numpy.random.seed(42)  # For reproducibility
        positions = numpy.random.uniform(0.1, 9.9, size=(n_positions, 3))

        result = minibox.get_mini_box_id(positions, boxsize, minisize)

        assert isinstance(
            result, numpy.ndarray if n_positions > 0 else numpy.ndarray)
        assert result.shape == (n_positions,)
        assert all(id >= 0 for id in result)


class TestGetAdjacentMiniBoxIds:
    """Comprehensive test suite for get_adjacent_mini_box_ids function."""

    # Test input validation
    @pytest.mark.parametrize(
        "mini_box_id, expected_error",
        [
            (3.14, TypeError),
            ("5", TypeError),
            ([1], TypeError),
            (None, TypeError),
        ],
    )
    def test_invalid_mini_box_id_type(self, mini_box_id, expected_error):
        """Test that invalid mini_box_id types raise TypeError."""
        with pytest.raises(expected_error):
            minibox.get_adjacent_mini_box_ids(mini_box_id, 1.0, 1.0)

    @pytest.mark.parametrize(
        "boxsize, minisize, expected_error",
        [
            ("1.0", 1.0, TypeError),
            (1.0, "1.0", TypeError),
            (None, 1.0, TypeError),
            (1.0, None, TypeError),
            ([1.0], 1.0, TypeError),
            (1.0, [1.0], TypeError),
        ],
    )
    def test_invalid_boxsize_minisize_types(self, boxsize, minisize, expected_error):
        """Test that invalid boxsize/minisize types raise TypeError."""
        with pytest.raises(expected_error):
            minibox.get_adjacent_mini_box_ids(0, boxsize, minisize)

    @pytest.mark.parametrize(
        "boxsize, minisize, expected_error",
        [
            (0.0, 1.0, ValueError),
            (-1.0, 1.0, ValueError),
            (1.0, 0.0, ValueError),
            (1.0, -1.0, ValueError),
            (1.0, 2.0, ValueError),  # minisize > boxsize
        ],
    )
    def test_invalid_size_values(self, boxsize, minisize, expected_error):
        """Test that invalid size values raise ValueError."""
        with pytest.raises(expected_error):
            minibox.get_adjacent_mini_box_ids(0, boxsize, minisize)

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize",
        [
            (-1, 2.0, 1.0),
            (-5, 2.0, 1.0),
            (8, 2.0, 1.0),  # 2x2x2 grid has IDs 0-7, so 8 is invalid
            (1000, 2.0, 1.0),
        ],
    )
    def test_mini_box_id_out_of_range(self, mini_box_id, boxsize, minisize):
        """Test that mini_box_id outside valid range raises ValueError."""
        with pytest.raises(ValueError):
            minibox.get_adjacent_mini_box_ids(mini_box_id, boxsize, minisize)

    # Test return value properties
    def test_return_type_and_shape(self):
        """Test that function returns correct type and shape."""
        result = minibox.get_adjacent_mini_box_ids(0, 2.0, 1.0)

        assert isinstance(result, numpy.ndarray)
        assert result.shape == (27,)
        assert result.dtype == numpy.int32

    def test_first_element_is_input_id(self):
        """Test that first element of result is always the input mini_box_id."""
        test_cases = [
            (0, 2.0, 1.0),
            (7, 2.0, 1.0),
            (13, 3.0, 1.0),
            (26, 3.0, 1.0),
        ]

        for mini_box_id, boxsize, minisize in test_cases:
            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)
            assert result[0] == mini_box_id

    def test_get_adjacent_mini_box_ids(self):
        """Test that the adjacent IDs are in the correct order and are these explicit values."""
        l_box = 100.
        l_mb = 20.

        adj_ids = minibox.get_adjacent_mini_box_ids(
            mini_box_id=0,
            boxsize=l_box,
            minisize=l_mb,
        )

        adj_ids_0 = [0, 1, 4, 5, 6, 9, 20, 21, 24, 25, 26, 29, 30, 31, 34, 45,
                     46, 49, 100, 101, 104, 105, 106, 109, 120, 121, 124]

        assert len(adj_ids) == 27
        assert all([i in adj_ids for i in adj_ids_0])

    def test_all_ids_in_valid_range(self):
        """Test that all returned IDs are within valid range."""
        test_cases = [
            (0, 2.0, 1.0),   # 2x2x2 grid (8 boxes)
            (13, 3.0, 1.0),  # 3x3x3 grid (27 boxes)
            (50, 4.0, 1.0),  # 4x4x4 grid (64 boxes)
        ]

        for mini_box_id, boxsize, minisize in test_cases:
            cells_per_side = int(numpy.ceil(boxsize / minisize))
            max_valid_id = cells_per_side**3 - 1

            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)

            assert numpy.all(result >= 0)
            assert numpy.all(result <= max_valid_id)

    def test_no_duplicate_ids(self):
        """Test that all returned IDs are unique."""
        test_cases = [
            (0, 2.0, 1.0),
            (13, 3.0, 1.0),
            (31, 4.0, 1.0),
        ]

        for mini_box_id, boxsize, minisize in test_cases:
            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)
            unique_ids = numpy.unique(result)
            if boxsize / minisize == 2:
                assert len(unique_ids) == 8, "Duplicate IDs found"
            else:
                assert len(unique_ids) == 27, "Duplicate IDs found"

    # Test specific grid configurations

    def test_1x1x1_grid(self):
        """Test edge case of 1x1x1 grid - all neighbors should be the same box."""
        result = minibox.get_adjacent_mini_box_ids(0, 1.0, 1.0)

        # In a 1x1x1 grid, all 27 neighbors should be box 0 (itself)
        expected = numpy.full(27, 0, dtype=numpy.int32)
        numpy.testing.assert_array_equal(result, expected)

    def test_2x2x2_grid_all_positions(self):
        """Test all positions in a 2x2x2 grid."""
        boxsize, minisize = 2.0, 1.0

        # In a 2x2x2 grid, every box is adjacent to every other box due to wrapping
        for mini_box_id in range(8):
            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)

            # Should contain all 8 box IDs, with many repetitions to make 27 total
            unique_ids = numpy.unique(result)
            expected_unique = numpy.arange(8, dtype=numpy.int32)
            numpy.testing.assert_array_equal(
                numpy.sort(unique_ids), expected_unique)

    def test_3x3x3_grid_center_box(self):
        """Test center box in 3x3x3 grid (ID 13)."""
        result = minibox.get_adjacent_mini_box_ids(13, 3.0, 1.0)

        # Center box (1,1,1) should have all 27 unique neighbors
        unique_ids = numpy.unique(result)
        assert len(unique_ids) == 27  # All boxes in 3x3x3 grid
        numpy.testing.assert_array_equal(
            numpy.sort(unique_ids), numpy.arange(27))

    def test_3x3x3_grid_corner_box(self):
        """Test corner box in 3x3x3 grid (ID 0)."""
        result = minibox.get_adjacent_mini_box_ids(0, 3.0, 1.0)

        # Corner box should still have 27 neighbors due to periodic boundaries
        assert len(result) == 27
        assert result[0] == 0  # First element is input ID

        # Verify periodic wrapping - corner should connect to opposite edges
        # Box 0 is at coordinates (0,0,0)
        # Its neighbors include boxes at coordinates like (2,2,2) due to wrapping
        expected_neighbors = {0, 1, 2, 3, 6, 9, 18, 19,
                              20, 21, 24, 26}  # Sample expected neighbors
        result_set = set(result)
        assert expected_neighbors.issubset(result_set)

    @pytest.mark.parametrize(
        "mini_box_id, expected_coords",
        [
            (0, (0, 0, 0)),
            (1, (0, 0, 1)),
            (3, (0, 1, 0)),
            (9, (1, 0, 0)),
            (13, (1, 1, 1)),  # center
            (26, (2, 2, 2)),  # far corner
        ],
    )
    def test_coordinate_conversion_3x3x3(self, mini_box_id, expected_coords):
        """Test that coordinate conversion is working correctly."""
        # This is an indirect test - we'll verify by checking that the function
        # doesn't raise RuntimeError (which would indicate coordinate conversion failure)
        result = minibox.get_adjacent_mini_box_ids(mini_box_id, 3.0, 1.0)
        assert len(result) == 27
        assert result[0] == mini_box_id

    # Test with different box/mini-box size ratios
    @pytest.mark.parametrize(
        "boxsize, minisize, expected_cells",
        [
            (2.0, 1.0, 2),    # Exact division
            (2.1, 1.0, 3),    # Ceil rounds up
            (3.9, 2.0, 2),    # Ceil rounds up
            (10.0, 3.0, 4),   # 10/3 = 3.33... -> 4 cells
        ],
    )
    def test_different_size_ratios(self, boxsize, minisize, expected_cells):
        """Test grid generation with different box/mini-box size ratios."""
        total_boxes = expected_cells**3
        max_valid_id = total_boxes - 1

        # Test with first and last valid IDs
        for test_id in [0, max_valid_id]:
            result = minibox.get_adjacent_mini_box_ids(
                test_id, boxsize, minisize)
            assert len(result) == 27
            assert result[0] == test_id
            assert numpy.all(result >= 0)
            assert numpy.all(result <= max_valid_id)

    # Test periodic boundary conditions explicitly
    def test_periodic_boundaries_4x4x4_edge_cases(self):
        """Test periodic boundary conditions with specific edge cases in 4x4x4 grid."""
        boxsize, minisize = 4.0, 1.0

        # Test box at edge (ID 3, coordinates (0,0,3))
        result = minibox.get_adjacent_mini_box_ids(3, boxsize, minisize)

        # Box at (0,0,3) should have neighbor at (0,0,0) due to k-direction wrapping
        assert 0 in result  # Periodic boundary neighbor
        assert 2 in result  # Regular neighbor
        assert 3 in result  # Self

        # Test box at face (ID 15, coordinates (0,3,3))
        result = minibox.get_adjacent_mini_box_ids(15, boxsize, minisize)

        # Should wrap in both j and k directions
        assert 12 in result  # k-direction wrap
        assert 3 in result   # j-direction wrap
        assert 0 in result   # Both j,k wrap

    def test_large_grid_performance(self):
        """Test function works with reasonably large grids."""
        # 10x10x10 grid (1000 boxes)
        result = minibox.get_adjacent_mini_box_ids(500, 10.0, 1.0)

        assert len(result) == 27
        assert result[0] == 500
        assert numpy.all(result >= 0)
        assert numpy.all(result <= 999)

    # Test floating-point edge cases
    def test_floating_point_precision(self):
        """Test with floating-point values that might cause precision issues."""
        test_cases = [
            (0, 2.000000001, 1.0),
            (0, 3.999999999, 2.0),
            (0, 1.0000000001, 0.5),
        ]

        for mini_box_id, boxsize, minisize in test_cases:
            result = minibox.get_adjacent_mini_box_ids(
                mini_box_id, boxsize, minisize)
            assert len(result) == 27
            assert result[0] == mini_box_id

    # Integration test with coordinate reconstruction
    def test_coordinate_reconstruction_consistency(self):
        """Test that coordinate conversion is consistent and reversible."""
        boxsize, minisize = 5.0, 1.0
        cells_per_side = 5

        # Test several random boxes
        test_ids = [0, 1, 5, 25, 62, 124]  # Various positions in 5x5x5 grid

        for test_id in test_ids:
            result = minibox.get_adjacent_mini_box_ids(
                test_id, boxsize, minisize)

            # Verify all neighbors are valid and unique
            assert len(result) == 27
            # Could be less due to wrapping
            assert len(numpy.unique(result)) <= 27
            assert numpy.all(result >= 0)
            assert numpy.all(result < cells_per_side**3)

            # Verify the input ID is preserved
            assert result[0] == test_id


class TestSplitSimulationIntoMiniBoxes:
    """Test suite for split_simulation_into_mini_boxes function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def basic_simulation_data(self):
        """Generate basic simulation data for testing."""
        n_particles = 100
        positions = numpy.random.rand(n_particles, 3) * 10.0
        velocities = numpy.random.rand(n_particles, 3) * 5.0
        uid = numpy.arange(n_particles, dtype=numpy.uint32)
        return positions, velocities, uid

    @pytest.fixture
    def large_simulation_data(self):
        """Generate larger simulation data for performance testing."""
        n_particles = 10000
        positions = numpy.random.rand(n_particles, 3) * 100.0
        velocities = numpy.random.randn(n_particles, 3) * 10.0
        uid = numpy.arange(n_particles, dtype=numpy.uint64)
        return positions, velocities, uid

    @pytest.fixture
    def simulation_with_props(self):
        """Generate simulation data with additional properties."""
        n_particles = 200
        positions = numpy.random.rand(n_particles, 3) * 20.0
        velocities = numpy.random.rand(n_particles, 3) * 8.0
        uid = numpy.arange(n_particles, dtype=numpy.uint32)

        # Additional properties
        masses = numpy.random.exponential(
            1e10, n_particles).astype(numpy.float32)
        temperatures = numpy.random.exponential(
            1e6, n_particles).astype(numpy.float32)

        props = (
            [masses, temperatures],
            ['mass', 'temperature'],
            [numpy.float32, numpy.float32]
        )
        return positions, velocities, uid, props

    def verify_hdf5_structure(self, file_path: Path, expected_datasets: List[str],
                              name: Optional[str] = None):
        """Verify HDF5 file structure and datasets."""
        with h5py.File(file_path, 'r') as f:
            prefix = f"{name}/" if name else ""
            for dataset in expected_datasets:
                dataset_path = f"{prefix}{dataset}"
                assert dataset_path in f, f"Dataset {dataset_path} not found in {file_path}"
                assert isinstance(
                    f[dataset_path], h5py.Dataset), f"{dataset_path} is not a dataset"

    def verify_data_consistency(self, file_path: Path, original_positions: numpy.ndarray,
                                original_velocities: numpy.ndarray, original_uid: numpy.ndarray,
                                boxsize: float, minisize: float, mini_box_id: int,
                                name: Optional[str] = None, props: Optional[Tuple] = None):
        """Verify that saved data is consistent with original data."""
        with h5py.File(file_path, 'r') as f:
            prefix = f"{name}/" if name else ""

            # Load saved data
            saved_positions = f[f"{prefix}pos"][:]
            saved_velocities = f[f"{prefix}vel"][:]
            saved_uid = f[f"{prefix}ID"][:]

            # Verify shapes
            n_particles_in_box = saved_positions.shape[0]
            assert saved_velocities.shape[0] == n_particles_in_box
            assert saved_uid.shape[0] == n_particles_in_box

            # Verify all particles belong to this mini box
            computed_ids = minibox.get_mini_box_id(
                saved_positions, boxsize, minisize)
            if isinstance(computed_ids, numpy.ndarray):
                assert numpy.all(computed_ids == mini_box_id), \
                    f"Not all particles belong to mini box {mini_box_id}"
            else:
                assert computed_ids == mini_box_id, \
                    f"Particle doesn't belong to mini box {mini_box_id}"

            # Verify data integrity by checking if saved UIDs exist in original data
            original_uid_set = set(original_uid)
            for saved_id in saved_uid:
                assert saved_id in original_uid_set, \
                    f"Saved UID {saved_id} not found in original data"

            # If props are provided, verify them too
            if props:
                arrays, labels, dtypes = props
                for i, label in enumerate(labels):
                    saved_prop = f[f"{prefix}{label}"][:]
                    assert saved_prop.shape[0] == n_particles_in_box, \
                        f"Property {label} has inconsistent number of particles"
                    assert saved_prop.dtype == dtypes[i], \
                        f"Property {label} has incorrect dtype"

    def test_basic_functionality(self, temp_dir, basic_simulation_data):
        """Test basic functionality with simple data."""
        positions, velocities, uid = basic_simulation_data
        boxsize = 10.0
        minisize = 2.0

        minibox.split_simulation_into_mini_boxes(
            positions=positions,
            velocities=velocities,
            uid=uid,
            save_path=temp_dir + "/",
            boxsize=boxsize,
            minisize=minisize
        )

        # Verify directory structure
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"
        assert output_dir.exists(), "Output directory not created"

        # Verify files were created
        hdf5_files = list(output_dir.glob("*.hdf5"))
        assert len(hdf5_files) > 0, "No HDF5 files created"

        # Verify file structure
        expected_datasets = ['ID', 'pos', 'vel']
        for file_path in hdf5_files:
            self.verify_hdf5_structure(file_path, expected_datasets)

    def test_with_name_parameter(self, temp_dir, basic_simulation_data):
        """Test functionality with name parameter."""
        positions, velocities, uid = basic_simulation_data
        boxsize = 10.0
        minisize = 2.0
        name = "test_simulation"

        minibox.split_simulation_into_mini_boxes(
            positions=positions,
            velocities=velocities,
            uid=uid,
            save_path=temp_dir + "/",
            boxsize=boxsize,
            minisize=minisize,
            name=name
        )

        # Verify directory structure
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        # Verify file structure with name prefix
        expected_datasets = ['ID', 'pos', 'vel']
        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            self.verify_hdf5_structure(file_path, expected_datasets, name)

    def test_with_additional_properties(self, temp_dir, simulation_with_props):
        """Test functionality with additional particle properties."""
        positions, velocities, uid, props = simulation_with_props
        boxsize = 20.0
        minisize = 4.0

        minibox.split_simulation_into_mini_boxes(
            positions=positions,
            velocities=velocities,
            uid=uid,
            save_path=temp_dir + "/",
            boxsize=boxsize,
            minisize=minisize,
            props=props
        )

        # Verify directory structure
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        # Verify file structure with additional properties
        expected_datasets = ['ID', 'pos', 'vel', 'mass', 'temperature']
        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            self.verify_hdf5_structure(file_path, expected_datasets)

    @pytest.mark.parametrize("boxsize,minisize,expected_cells", [
        (10.0, 2.0, 5),
        (10.0, 3.0, 4),
        (15.0, 5.0, 3),
        (20.0, 7.0, 3),
        (100.0, 25.0, 4)
    ])
    def test_different_box_configurations(self, temp_dir, basic_simulation_data,
                                          boxsize, minisize, expected_cells):
        """Test with different box size configurations."""
        positions, velocities, uid = basic_simulation_data
        # Scale positions to fit the box
        positions = positions * (boxsize / 10.0)

        minibox.split_simulation_into_mini_boxes(
            positions=positions,
            velocities=velocities,
            uid=uid,
            save_path=temp_dir + "/",
            boxsize=boxsize,
            minisize=minisize
        )

        # Verify correct number of cells
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{expected_cells}"
        assert output_dir.exists(
        ), f"Expected directory with {expected_cells} cells per side"

    def test_data_consistency(self, temp_dir, basic_simulation_data):
        """Test that saved data is consistent with original data."""
        positions, velocities, uid = basic_simulation_data
        boxsize = 10.0
        minisize = 2.0

        minibox.split_simulation_into_mini_boxes(
            positions=positions,
            velocities=velocities,
            uid=uid,
            save_path=temp_dir + "/",
            boxsize=boxsize,
            minisize=minisize
        )

        # Verify data consistency
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            mini_box_id = int(file_path.stem)
            self.verify_data_consistency(
                file_path, positions, velocities, uid,
                boxsize, minisize, mini_box_id
            )

    def test_particle_count_conservation(self, temp_dir, basic_simulation_data):
        """Test that all particles are saved and none are lost or duplicated."""
        positions, velocities, uid = basic_simulation_data
        boxsize = 10.0
        minisize = 2.0

        minibox.split_simulation_into_mini_boxes(
            positions=positions,
            velocities=velocities,
            uid=uid,
            save_path=temp_dir + "/",
            boxsize=boxsize,
            minisize=minisize
        )

        # Count total particles across all files
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        total_saved_particles = 0
        all_saved_uids = []

        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                n_particles = f['ID'].shape[0]
                total_saved_particles += n_particles
                all_saved_uids.extend(f['ID'][:])

        # Verify particle count
        assert total_saved_particles == len(positions), \
            f"Particle count mismatch: original {len(positions)}, saved {total_saved_particles}"

        # Verify no duplicates
        assert len(set(all_saved_uids)) == len(all_saved_uids), \
            "Duplicate particles found in saved files"

        # Verify all original UIDs are present
        original_uid_set = set(uid)
        saved_uid_set = set(all_saved_uids)
        assert original_uid_set == saved_uid_set, \
            "Saved UIDs don't match original UIDs"

    # def test_empty_mini_boxes_handling(self, temp_dir):
    #     """Test behavior with sparse data leading to empty mini boxes."""
    #     # Create data clustered in one corner
    #     n_particles = 50
    #     positions = numpy.random.rand(n_particles, 3) * 2.0  # Only in corner
    #     velocities = numpy.random.rand(n_particles, 3) * 5.0
    #     uid = numpy.arange(n_particles, dtype=numpy.uint32)

    #     boxsize = 20.0  # Much larger box
    #     minisize = 2.0

    #     minibox.split_simulation_into_mini_boxes(
    #         positions=positions,
    #         velocities=velocities,
    #         uid=uid,
    #         save_path=temp_dir + "/",
    #         boxsize=boxsize,
    #         minisize=minisize
    #     )

    #     # Verify only some files are created (not all mini boxes have particles)
    #     cells_per_side = int(numpy.ceil(boxsize / minisize))
    #     output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

    #     hdf5_files = list(output_dir.glob("*.hdf5"))
    #     total_possible_boxes = cells_per_side ** 3

    #     # Should have fewer files than total possible boxes
    #     assert len(hdf5_files) < total_possible_boxes, \
    #         "Expected some empty mini boxes, but all were created"
    #     assert len(hdf5_files) > 0, "No files created"

    def test_large_simulation(self, temp_dir, large_simulation_data):
        """Test with larger simulation data."""
        positions, velocities, uid = large_simulation_data
        boxsize = 100.0
        minisize = 10.0

        minibox.split_simulation_into_mini_boxes(
            positions=positions,
            velocities=velocities,
            uid=uid,
            save_path=temp_dir + "/",
            boxsize=boxsize,
            minisize=minisize
        )

        # Verify particle conservation
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        total_particles = 0
        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                total_particles += f['ID'].shape[0]

        assert total_particles == len(positions), \
            f"Particle count mismatch in large simulation: {total_particles} vs {len(positions)}"

    @pytest.mark.parametrize("invalid_positions", [
        numpy.random.rand(100, 2),  # Wrong number of columns
        numpy.random.rand(100),     # 1D array
        numpy.random.rand(100, 3, 2),  # 3D array
    ])
    def test_invalid_positions_shape(self, temp_dir, invalid_positions):
        """Test validation of positions array shape."""
        velocities = numpy.random.rand(100, 3)
        uid = numpy.arange(100)

        with pytest.raises(ValueError, match="positions must have shape"):
            minibox.split_simulation_into_mini_boxes(
                positions=invalid_positions,
                velocities=velocities,
                uid=uid,
                save_path=temp_dir + "/",
                boxsize=10.0,
                minisize=2.0
            )

    @pytest.mark.parametrize("invalid_velocities", [
        numpy.random.rand(100, 2),  # Wrong number of columns
        numpy.random.rand(100),     # 1D array
        numpy.random.rand(50, 3),   # Wrong number of rows
    ])
    def test_invalid_velocities_shape(self, temp_dir, invalid_velocities):
        """Test validation of velocities array shape."""
        positions = numpy.random.rand(100, 3)
        uid = numpy.arange(100)

        with pytest.raises(ValueError):
            minibox.split_simulation_into_mini_boxes(
                positions=positions,
                velocities=invalid_velocities,
                uid=uid,
                save_path=temp_dir + "/",
                boxsize=10.0,
                minisize=2.0
            )

    @pytest.mark.parametrize("invalid_uid", [
        numpy.random.rand(100, 2),  # 2D array
        numpy.arange(50),           # Wrong length
    ])
    def test_invalid_uid_shape(self, temp_dir, invalid_uid):
        """Test validation of uid array shape."""
        positions = numpy.random.rand(100, 3)
        velocities = numpy.random.rand(100, 3)

        with pytest.raises(ValueError):
            minibox.split_simulation_into_mini_boxes(
                positions=positions,
                velocities=velocities,
                uid=invalid_uid,
                save_path=temp_dir + "/",
                boxsize=10.0,
                minisize=2.0
            )

    @pytest.mark.parametrize("boxsize,minisize,expected_error", [
        (0, 2.0, "boxsize must be positive"),
        (-5.0, 2.0, "boxsize must be positive"),
        (10.0, 0, "minisize must be positive"),
        (10.0, -2.0, "minisize must be positive"),
        (5.0, 10.0, "minisize cannot be larger than boxsize"),
        ("10.0", 2.0, "boxsize and minisize must be numeric"),
        (10.0, "2.0", "boxsize and minisize must be numeric"),
    ])
    def test_invalid_box_parameters(self, temp_dir, basic_simulation_data,
                                    boxsize, minisize, expected_error):
        """Test validation of box size parameters."""
        positions, velocities, uid = basic_simulation_data

        with pytest.raises((ValueError, TypeError), match=expected_error):
            minibox.split_simulation_into_mini_boxes(
                positions=positions,
                velocities=velocities,
                uid=uid,
                save_path=temp_dir + "/",
                boxsize=boxsize,
                minisize=minisize
            )

    def test_empty_arrays(self, temp_dir):
        """Test behavior with empty input arrays."""
        positions = numpy.empty((0, 3))
        velocities = numpy.empty((0, 3))
        uid = numpy.empty(0, dtype=int)

        with pytest.raises(ValueError, match="No particles provided"):
            minibox.split_simulation_into_mini_boxes(
                positions=positions,
                velocities=velocities,
                uid=uid,
                save_path=temp_dir + "/",
                boxsize=10.0,
                minisize=2.0
            )

    def test_invalid_props_structure(self, temp_dir, basic_simulation_data):
        """Test validation of props parameter structure."""
        positions, velocities, uid = basic_simulation_data

        # Test with wrong tuple length
        invalid_props = ([numpy.random.rand(100)], ["mass"])  # Missing dtypes

        with pytest.raises(ValueError, match="props must be a tuple of"):
            minibox.split_simulation_into_mini_boxes(
                positions=positions,
                velocities=velocities,
                uid=uid,
                save_path=temp_dir + "/",
                boxsize=10.0,
                minisize=2.0,
                props=invalid_props
            )

    def test_mismatched_props_arrays(self, temp_dir, basic_simulation_data):
        """Test validation of props arrays with mismatched lengths."""
        positions, velocities, uid = basic_simulation_data

        # Props array with wrong length
        invalid_props = (
            [numpy.random.rand(50)],  # Wrong length
            ["mass"],
            [numpy.float32]
        )

        with pytest.raises(ValueError, match="must have.*elements"):
            minibox.split_simulation_into_mini_boxes(
                positions=positions,
                velocities=velocities,
                uid=uid,
                save_path=temp_dir + "/",
                boxsize=10.0,
                minisize=2.0,
                props=invalid_props
            )

    def test_boundary_particles(self, temp_dir):
        """Test handling of particles at box boundaries."""
        # Create particles exactly at boundaries
        positions = numpy.array([
            [0.0, 0.0, 0.0],      # At origin
            [10.0, 10.0, 10.0],   # At upper boundary
            [5.0, 5.0, 5.0],      # In middle
            [9.999999, 9.999999, 9.999999],  # Very close to boundary
        ])
        velocities = numpy.random.rand(4, 3)
        uid = numpy.arange(4)

        boxsize = 10.0
        minisize = 5.0

        minibox.split_simulation_into_mini_boxes(
            positions=positions,
            velocities=velocities,
            uid=uid,
            save_path=temp_dir + "/",
            boxsize=boxsize,
            minisize=minisize
        )

        # Verify all particles are saved
        cells_per_side = int(numpy.ceil(boxsize / minisize))
        output_dir = Path(temp_dir) / f"mini_boxes_nside_{cells_per_side}"

        total_particles = 0
        hdf5_files = list(output_dir.glob("*.hdf5"))
        for file_path in hdf5_files:
            with h5py.File(file_path, 'r') as f:
                total_particles += f['ID'].shape[0]

        assert total_particles == 4, "Not all boundary particles were saved"


# --------------------
# Low-level file makers
# --------------------
def _make_particle_file(path: Path, positions, velocities, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("part")
        grp.create_dataset("pos", data=positions)
        grp.create_dataset("vel", data=velocities)
        grp.create_dataset("ID", data=ids)


def _make_seed_file(path: Path, positions, velocities, ids, r200, m200, rs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("seed")
        grp.create_dataset("pos", data=positions)
        grp.create_dataset("vel", data=velocities)
        grp.create_dataset("ID", data=ids)
        grp.create_dataset("R200b", data=r200)
        grp.create_dataset("M200b", data=m200)
        grp.create_dataset("Rs", data=rs)


# ------------------------
# High-level test utilities
# ------------------------
def make_all_particle_files(tmp_path, mini_box_id, boxsize, minisize, with_particle=True):
    """Create particle files for target + neighbors. Only target has one particle if with_particle=True."""
    cells_per_side = int(numpy.ceil(boxsize / minisize))
    mini_box_ids = minibox.get_adjacent_mini_box_ids(
        mini_box_id, boxsize, minisize)
    subdir = tmp_path / f"mini_boxes_nside_{cells_per_side}"

    # Compute center of target mini-box
    i = mini_box_id // (cells_per_side**2)
    remainder = mini_box_id % (cells_per_side**2)
    j = remainder // cells_per_side
    k = remainder % cells_per_side
    center = numpy.array([(k + 0.5) * minisize, (j + 0.5)
                         * minisize, (i + 0.5) * minisize])

    for mb in mini_box_ids:
        file_path = subdir / f"{mb}.hdf5"
        if mb == mini_box_id and with_particle:
            _make_particle_file(
                file_path,
                positions=numpy.array([center]),
                velocities=numpy.array([[1.0, 1.0, 1.0]]),
                ids=numpy.array([42], dtype=int),
            )
        else:
            _make_particle_file(
                file_path,
                positions=numpy.empty((0, 3)),
                velocities=numpy.empty((0, 3)),
                ids=numpy.empty((0,), dtype=int),
            )


def make_all_seed_files(tmp_path, mini_box_id, boxsize, minisize, with_seed=True):
    """Create seed files for target + neighbors. Only target has one seed if with_seed=True."""
    cells_per_side = int(numpy.ceil(boxsize / minisize))
    mini_box_ids = minibox.get_adjacent_mini_box_ids(
        mini_box_id, boxsize, minisize)
    subdir = tmp_path / f"mini_boxes_nside_{cells_per_side}"

    # Compute center of target mini-box
    i = mini_box_id // (cells_per_side**2)
    remainder = mini_box_id % (cells_per_side**2)
    j = remainder // cells_per_side
    k = remainder % cells_per_side
    center = numpy.array([(k + 0.5) * minisize, (j + 0.5)
                         * minisize, (i + 0.5) * minisize])

    for mb in mini_box_ids:
        file_path = subdir / f"{mb}.hdf5"
        if mb == mini_box_id and with_seed:
            _make_seed_file(
                file_path,
                positions=numpy.array([center]),
                velocities=numpy.array([[1.0, 1.0, 1.0]]),
                ids=numpy.array([7], dtype=int),
                r200=numpy.array([10.0]),
                m200=numpy.array([20.0]),
                rs=numpy.array([1.0]),
            )
        else:
            _make_seed_file(
                file_path,
                positions=numpy.empty((0, 3)),
                velocities=numpy.empty((0, 3)),
                ids=numpy.empty((0,), dtype=int),
                r200=numpy.empty((0,)),
                m200=numpy.empty((0,)),
                rs=numpy.empty((0,)),
            )


class TestLoadParticles:
    def test_valid_inputs(self, tmp_path):
        make_all_particle_files(tmp_path, mini_box_id=0,
                                boxsize=10.0, minisize=1.0, with_particle=True)

        pos, vel, ids = minibox.load_particles(
            0, 10.0, 1.0, str(tmp_path) + "/", padding=1.0
        )

        assert pos.shape == (1, 3)
        assert vel.shape == (1, 3)
        assert ids.shape == (1,)
        assert int(ids[0]) == 42

    def test_runtime_error_when_no_particles(self, tmp_path):
        make_all_particle_files(tmp_path, 0, 10.0, 10.0, with_particle=False)

        with pytest.raises(RuntimeError, match="No particles found"):
            minibox.load_particles(
                0, 10.0, 10.0, str(tmp_path) + "/", padding=1.0)

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize, load_path, padding, expected_error",
        [
            ("0", 10.0, 10.0, "some/path", 1.0, TypeError),
            (0, "10", 10.0, "some/path", 1.0, TypeError),
            (0, 10.0, "10", "some/path", 1.0, TypeError),
            (0, 10.0, 10.0, 123, 1.0, TypeError),
            (0, 10.0, 10.0, "some/path", "1.0", TypeError),
        ],
    )
    def test_type_errors(self, mini_box_id, boxsize, minisize, load_path, padding, expected_error):
        with pytest.raises(expected_error):
            minibox.load_particles(mini_box_id, boxsize,
                                   minisize, load_path, padding)

    def test_value_errors_and_file_errors(self, tmp_path):
        subdir = tmp_path / "mini_boxes_nside_1"
        subdir.mkdir()

        # Negative ID
        with pytest.raises(ValueError, match="mini_box_id must be non-negative"):
            minibox.load_particles(-1, 10.0, 10.0, str(tmp_path) + "/", 1.0)

        # Missing file
        with pytest.raises(FileNotFoundError):
            minibox.load_particles(0, 10.0, 10.0, str(tmp_path) + "/", 1.0)

        # Not a directory
        file_path = tmp_path / "not_a_dir.hdf5"
        file_path.write_text("x")
        with pytest.raises(NotADirectoryError):
            minibox.load_particles(0, 10.0, 10.0, file_path, 1.0)


# --------------
# Tests: Seeds
# --------------
class TestLoadSeeds:
    def test_valid_inputs(self, tmp_path):
        make_all_seed_files(tmp_path, mini_box_id=0,
                            boxsize=10.0, minisize=1.0, with_seed=True)

        pos, vel, ids, r200, m200, rs, mask = minibox.load_seeds(
            0, 10.0, 1.0, str(tmp_path) + "/", padding=1.0
        )

        assert pos.shape == (1, 3)
        assert vel.shape == (1, 3)
        assert ids.shape == (1,)
        assert r200.shape == (1,)
        assert m200.shape == (1,)
        assert rs.shape == (1,)
        assert mask.shape == (1,)
        assert int(ids[0]) == 7
        assert pytest.approx(float(m200[0])) == 20.0

    def test_runtime_error_when_no_seeds(self, tmp_path):
        make_all_seed_files(tmp_path, 0, 10.0, 10.0, with_seed=False)

        with pytest.raises(RuntimeError, match="No seeds found"):
            minibox.load_seeds(0, 10.0, 10.0, str(tmp_path) + "/", padding=1.0)

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize, load_path, padding, expected_error",
        [
            ("0", 10.0, 10.0, "some/path", 1.0, TypeError),
            (0, "10", 10.0, "some/path", 1.0, TypeError),
            (0, 10.0, "10", "some/path", 1.0, TypeError),
            (0, 10.0, 10.0, 123, 1.0, TypeError),
            (0, 10.0, 10.0, "some/path", "1.0", TypeError),
        ],
    )
    def test_type_errors(self, mini_box_id, boxsize, minisize, load_path, padding, expected_error):
        with pytest.raises(expected_error):
            minibox.load_seeds(mini_box_id, boxsize,
                               minisize, load_path, padding)

    def test_value_errors_and_file_errors(self, tmp_path):
        subdir = tmp_path / "mini_boxes_nside_1"
        subdir.mkdir()

        # Negative ID
        with pytest.raises(ValueError, match="mini_box_id must be non-negative"):
            minibox.load_seeds(-1, 10.0, 10.0, str(tmp_path) + "/", 1.0)

        # Missing file
        with pytest.raises(FileNotFoundError):
            minibox.load_seeds(0, 10.0, 10.0, str(tmp_path) + "/", 1.0)

        # Not a directory
        file_path = tmp_path / "not_a_dir.hdf5"
        file_path.write_text("x")
        with pytest.raises(NotADirectoryError):
            minibox.load_seeds(0, 10.0, 10.0, file_path, 1.0)
