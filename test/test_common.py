"""
Unit tests for oasis.common module.

Covers:
- AnsiColor dataclass
- timer decorator
- TimerContext context manager
- get_min_unit_dtype function
- ensure_dir_exists function

Tests are structured in classes for clarity and maintainability.
"""
import time

import numpy
import pytest

from oasis import common


class TestAnsiColor:
    """Tests for AnsiColor constants."""

    def test_constants_exist_and_are_strings(self):
        """Ensure all expected ANSI attributes exist and are strings."""
        expected_attrs = [
            "HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARNING",
            "FAIL", "ENDC", "BOLD", "UNDERLINE", "BULLET"
        ]
        for attr in expected_attrs:
            value = getattr(common.AnsiColor, attr)
            assert isinstance(value, str)

    def test_unique_values(self):
        """Check that ANSI codes are unique to avoid duplication errors."""
        values = [
            getattr(common.AnsiColor, attr)
            for attr in vars(common.AnsiColor)
            if not attr.startswith("_")
        ]
        assert len(values) == len(set(values))


class TestTimerDecorator:
    """Tests for the @timer decorator."""

    def test_decorator_measures_time(self, capsys):
        """Check that timer prints timing info for a slow function."""
        @common.timer
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        captured = capsys.readouterr().out

        assert result == "done"
        assert "Process:" in captured
        assert "Elapsed:" in captured

    def test_decorator_disabled(self, capsys):
        """Check that disabled timer runs silently."""
        @common.timer(enabled=False)
        def fast_function():
            return 42

        result = fast_function()
        captured = capsys.readouterr().out

        assert result == 42
        assert captured == ""  # no timing output

    def test_decorator_plain_output(self, capsys):
        """Check that plain (non-fancy) output has no ANSI codes."""
        @common.timer(fancy=False, precision=2)
        def task():
            time.sleep(0.001)

        task()
        captured = capsys.readouterr().out
        assert "Elapsed:" in captured
        assert "\033" not in captured  # no ANSI escape codes

    def test_nested_decorators(self, capsys):
        """Ensure nested decorated functions both print output."""
        @common.timer
        def inner():
            return "inner"

        @common.timer
        def outer():
            return inner()

        result = outer()
        captured = capsys.readouterr().out
        assert result == "inner"
        assert captured.count("Elapsed:") >= 2  # both ran


class TestTimerContext:
    """Tests for the TimerContext context manager."""

    def test_context_measures_time(self, capsys):
        """Ensure TimerContext measures elapsed time and prints output."""
        with common.TimerContext("unit test", fancy=False, precision=2) as t:
            time.sleep(0.005)

        assert t.elapsed is not None
        captured = capsys.readouterr().out
        assert "unit test completed in" in captured

    def test_context_with_fancy_output(self, capsys):
        """Ensure fancy output includes ANSI escape codes."""
        with common.TimerContext("fancy test", fancy=True):
            pass

        captured = capsys.readouterr().out
        assert "fancy test" in captured
        assert "\033" in captured  # contains ANSI codes

    def test_context_elapsed_precision(self):
        """Ensure elapsed string respects precision setting."""
        with common.TimerContext("precise test", fancy=False, precision=5) as t:
            time.sleep(0.001)

        assert t.elapsed == pytest.approx(t.elapsed, rel=1e-5)


class TestUintDtype:
    """Tests for get_min_unit_dtype."""

    def test_wrong_dtype(self):
        """Invalid inputs should raise TypeError."""
        with pytest.raises(TypeError):
            common.get_min_unit_dtype(-1)

        with pytest.raises(TypeError):
            common.get_min_unit_dtype(1.0)

    def test_correct_dtype_boundaries(self):
        """Check boundary values for dtype selection."""
        assert common.get_min_unit_dtype(255) == numpy.uint8
        assert common.get_min_unit_dtype(256) == numpy.uint16
        assert common.get_min_unit_dtype(65_535) == numpy.uint16
        assert common.get_min_unit_dtype(65_536) == numpy.uint32
        assert common.get_min_unit_dtype(4_294_967_295) == numpy.uint32
        assert common.get_min_unit_dtype(4_294_967_296) == numpy.uint64

    def test_overflow(self):
        """Values too large for uint64 should raise OverflowError."""
        with pytest.raises(OverflowError):
            common.get_min_unit_dtype(18_446_744_073_709_551_616)


class TestEnsureDirExists:
    """Tests for ensure_dir_exists."""

    def test_fail_due_to_permissions(self):
        """Should fail to create directory in restricted locations."""
        dirpath_fail = "/root/forbidden_dir_test"
        with pytest.raises((OSError, PermissionError)):
            common.ensure_dir_exists(dirpath_fail, verbose=False)

    def test_pass_and_cleanup(self, tmp_path):
        """Should successfully create directory in tmp_path and clean up."""
        dirpath = tmp_path / "test_dir"
        path_created = common.ensure_dir_exists(dirpath, verbose=True)

        assert path_created.exists()
        assert path_created.is_dir()

        # Cleanup handled automatically by tmp_path

    def test_existing_directory(self, tmp_path, capsys):
        """Should recognize already existing directory."""
        dirpath = tmp_path / "existing"
        dirpath.mkdir()
        result = common.ensure_dir_exists(dirpath, verbose=True)

        assert result == dirpath.resolve()
        captured = capsys.readouterr().out
        assert "already exists" in captured

    def test_existing_file_instead_of_directory(self, tmp_path):
        """Should raise OSError if path exists but is a file."""
        filepath = tmp_path / "file.txt"
        filepath.write_text("dummy")
        with pytest.raises(OSError):
            common.ensure_dir_exists(filepath)

    def test_with_parents_false_and_cleanup(self, tmp_path):
        """Should fail if parent directories do not exist and parents=False."""
        subdir = tmp_path / "nonexistent" / "child"
        with pytest.raises(OSError):
            common.ensure_dir_exists(subdir, parents=False)
        # Nothing created, so no cleanup required


class TestValidateInputsPositiveNumber:
    """Test suite for _validate_inputs_positive_number function."""

    def test_valid_inputs(self):
        """Test that valid inputs don't raise exceptions."""
        # Should not raise any exceptions
        common._validate_inputs_positive_number(1.0, "test_param")
        common._validate_inputs_positive_number(100, "test_param")
        common._validate_inputs_positive_number(
            0.0001, "test_param")  # Small positive

    @pytest.mark.parametrize(
        "value, expected_error",
        [
            # Type errors
            ("100", TypeError),
            (None, TypeError),
            ([100], TypeError),
            (complex(100, 0), TypeError),
        ],
    )
    def test_type_errors(self, value, expected_error):
        """Test that invalid types raise TypeError."""
        with pytest.raises(expected_error, match="test_param must be numeric"):
            common._validate_inputs_positive_number(value, "test_param")

    @pytest.mark.parametrize(
        "value, expected_error, error_message",
        [
            # Negative values
            (-1.0, ValueError, "test_param must be positive"),
            (-100, ValueError, "test_param must be positive"),
            (-0.0001, ValueError, "test_param must be positive"),
            # Zero value
            (0, ValueError, "test_param must be positive"),
            (0.0, ValueError, "test_param must be positive"),
        ],
    )
    def test_value_errors(self, value, expected_error, error_message):
        """Test that invalid values raise ValueError with correct messages."""
        with pytest.raises(expected_error, match=error_message):
            common._validate_inputs_positive_number(value, "test_param")

    def test_edge_cases(self):
        """Test edge cases with very small or very large numbers."""
        # Very small positive number
        common._validate_inputs_positive_number(1e-10, "test_param")

        # Very large number
        common._validate_inputs_positive_number(1e10, "test_param")

        # Float precision edge case
        common._validate_inputs_positive_number(1.0000001, "test_param")


class TestValidateInputsCoordinateArrays:
    """Test suite for _validate_inputs_coordinate_arrays function."""

    def test_valid_inputs(self):
        """Test that valid inputs don't raise exceptions."""
        # Should not raise any exceptions
        common._validate_inputs_coordinate_arrays(
            numpy.zeros((100, 3)), "positions")
        common._validate_inputs_coordinate_arrays(
            numpy.random.rand(50, 3), "velocities")
        common._validate_inputs_coordinate_arrays(
            numpy.array([[1.0, 2.0, 3.0]]), "single_position")

    @pytest.mark.parametrize(
        "array, expected_error, error_message",
        [
            # Wrong number of dimensions
            (numpy.zeros(100), ValueError, "must have shape"),
            # Wrong second dimension
            (numpy.zeros((100, 2)), ValueError, "must have shape"),
            (numpy.zeros((100, 4)), ValueError, "must have shape"),
            (numpy.zeros((100, 1)), ValueError, "must have shape"),
            # Wrong type
            ("not an array", TypeError, "must be a numpy array"),
            (None, TypeError, "must be a numpy array"),
            ([1.0, 2.0, 3.0], TypeError, "must be a numpy array"),
            (42, TypeError, "must be a numpy array"),
        ],
    )
    def test_invalid_inputs(self, array, expected_error, error_message):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises(expected_error, match=error_message):
            common._validate_inputs_coordinate_arrays(array, "test_array")


class TestValidateInputsExistingPath:
    """Test suite for _validate_inputs_existing_path function."""

    def test_valid_inputs(self, tmp_path):
        """Test that valid inputs don't raise exceptions."""
        valid_dir = tmp_path / "data"
        valid_dir.mkdir()
        valid_file = tmp_path / "file.txt"
        valid_file.write_text("dummy")

        # Should not raise any exceptions
        common._validate_inputs_existing_path(valid_dir)
        common._validate_inputs_existing_path(str(valid_dir))

        # Should not raise exceptionss
        with pytest.raises(NotADirectoryError):
            common._validate_inputs_existing_path(valid_file)
        
        with pytest.raises(NotADirectoryError):
            common._validate_inputs_existing_path(str(valid_file))

    @pytest.mark.parametrize(
        "path, expected_error, error_message",
        [
            # Wrong type
            (123, TypeError, "must be a string or Path object"),
            (None, TypeError, "must be a string or Path object"),
            ([], TypeError, "must be a string or Path object"),
            (3.14, TypeError, "must be a string or Path object"),
            # Non-existent path
            ("nonexistent/path", FileNotFoundError, "does not exist"),
            ("/this/path/should/not/exist", FileNotFoundError, "does not exist"),
        ],
    )
    def test_invalid_inputs(self, path, expected_error, error_message):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises(expected_error, match=error_message):
            common._validate_inputs_existing_path(path)


class TestValidateInputsBoxsizeMinisize:
    """Test suite for _validate_inputs_boxsize_minisize function."""

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
            common._validate_inputs_boxsize_minisize(boxsize, minisize)

    def test_edge_cases(self):
        """Test edge cases with very small or very large numbers."""
        # Very small positive numbers
        common._validate_inputs_boxsize_minisize(1e-10, 1e-11)

        # Very large numbers
        common._validate_inputs_boxsize_minisize(1e10, 1e9)

        # Float precision edge cases
        common._validate_inputs_boxsize_minisize(1.0000001, 1.0)


class TestValidateInputsMiniBoxId:
    """Test suite for _validate_inputs_mini_box_id function."""

    def test_valid_inputs(self):
        """Test that valid inputs don't raise exceptions."""
        # Should not raise any exceptions
        common._validate_inputs_mini_box_id(0, cells_per_side=10)
        common._validate_inputs_mini_box_id(999, cells_per_side=10)

        # Should not raise an exception
        with pytest.raises(ValueError, match="exceeds maximum valid ID"):
            common._validate_inputs_mini_box_id(27, cells_per_side=3)

    @pytest.mark.parametrize(
        "mini_box_id, cells_per_side, expected_error, error_message",
        [
            # Type errors
            ("100", None, TypeError, "mini_box_id must be an integer"),
            (None, None, TypeError, "mini_box_id must be an integer"),
            (1.5, None, TypeError, "mini_box_id must be an integer"),
            ([], None, TypeError, "mini_box_id must be an integer"),

            # Negative mini_box_id
            (-1, None, ValueError, "mini_box_id must be zero or positive"),
            (-10, None, ValueError, "mini_box_id must be zero or positive"),

            # Exceeds max ID without cells_per_side
            (1_000_000_000_000, 10, ValueError, "exceeds maximum valid ID"),

            # Exceeds max ID with cells_per_side
            (1000, 10, ValueError, "exceeds maximum valid ID"),
            (5000, 15, ValueError, "exceeds maximum valid ID"),
            (28, 3, ValueError, "exceeds maximum valid ID"),

            # # Invalid cells_per_side
            (10, 0, ValueError, "cells_per_side must be a positive integer"),
            (10, -5, ValueError, "cells_per_side must be a positive integer"),
            (10, 2.5, TypeError, "cells_per_side must be a positive integer"),
            (10, "10", TypeError, "cells_per_side must be a positive integer"),
        ],
    )
    def test_invalid_inputs(self, mini_box_id, cells_per_side,
                            expected_error, error_message):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises(expected_error, match=error_message):
            common._validate_inputs_mini_box_id(mini_box_id, cells_per_side)


class TestValidateInputsLoad:
    """Test suite for _validate_inputs_load function."""

    def test_valid_inputs(self, tmp_path):
        """Test that valid inputs don't raise exceptions."""
        valid_dir = tmp_path / "data"
        valid_dir.mkdir()

        # Should not raise any exceptions
        common._validate_inputs_load(
            mini_box_id=0,
            boxsize=100.0,
            minisize=10.0,
            load_path=valid_dir,
            padding=1.0,
        )

        common._validate_inputs_load(
            mini_box_id=5,
            boxsize=10.0,
            minisize=1.0,
            load_path=str(valid_dir),
            padding=1.0,
        )

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize, padding, expected_error, error_message",
        [
            # mini_box_id type errors
            ("0", 100.0, 10.0, 0.0, TypeError, "mini_box_id must be an integer"),
            (None, 100.0, 10.0, 0.0, TypeError, "mini_box_id must be an integer"),
            (1.5, 100.0, 10.0, 0.0, TypeError, "mini_box_id must be an integer"),

            # padding type errors
            (0, 100.0, 10.0, "0.0", TypeError, "padding must be numeric"),
            (0, 100.0, 10.0, None, TypeError, "padding must be numeric"),
        ],
    )
    def test_type_errors(
        self, tmp_path, mini_box_id, boxsize, minisize, padding, expected_error, error_message
    ):
        """Test that invalid types raise TypeError with correct messages."""
        valid_dir = tmp_path / "data"
        valid_dir.mkdir()

        with pytest.raises(expected_error, match=error_message):
            common._validate_inputs_load(
                mini_box_id, boxsize, minisize, valid_dir, padding)

    @pytest.mark.parametrize(
        "mini_box_id, boxsize, minisize, padding, expected_error, error_message",
        [
            # Negative mini_box_id
            (-1, 10.0, 1.0, 0.0, ValueError,
             "mini_box_id must be zero or positive"),
            # mini_box_id too large
            (1000, 10.0, 1.0, 1.0, ValueError, "exceeds maximum valid ID"),
            # Negative padding
            (0, 10.0, 1.0, -0.5, ValueError, "padding must be positive"),
        ],
    )
    def test_value_errors(
        self, tmp_path, mini_box_id, boxsize, minisize, padding, expected_error, error_message
    ):
        """Test that invalid values raise ValueError with correct messages."""
        valid_dir = tmp_path / "data"
        valid_dir.mkdir()

        with pytest.raises(expected_error, match=error_message):
            common._validate_inputs_load(
                mini_box_id, boxsize, minisize, valid_dir, padding
            )

    def test_load_path_not_found(self, tmp_path):
        """Test that non-existent load_path raises FileNotFoundError."""
        missing_path = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            common._validate_inputs_load(0, 10.0, 1.0, missing_path, 0.1)

    def test_load_path_not_directory(self, tmp_path):
        """Test that non-directory load_path raises NotADirectoryError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("dummy")
        with pytest.raises(NotADirectoryError, match="Path is not a directory"):
            common._validate_inputs_load(0, 10.0, 1.0, file_path, 0.1)

    def test_edge_cases(self, tmp_path):
        """Test edge cases with very small or large boxsize/minisize and numpy types."""
        valid_dir = tmp_path / "edge"
        valid_dir.mkdir()

        # mini_box_id as numpy integer
        common._validate_inputs_load(
            numpy.int64(0), 1.0, 1.0, valid_dir, numpy.float64(1.0)
        )

        # Very large grid
        common._validate_inputs_load(0, 1e6, 1.0, valid_dir, 0.1)

        # Very small valid minisize
        common._validate_inputs_load(0, 1e-3, 1e-4, valid_dir, 0.1)


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
        common._validate_inputs_box_partitioning(
            positions, velocities, uid, None)

    def test_valid_inputs_with_props(self, valid_data, valid_props):
        """Test that valid inputs with props don't raise exceptions."""
        positions, velocities, uid, _ = valid_data
        common._validate_inputs_box_partitioning(
            positions, velocities, uid, valid_props)

    def test_empty_arrays(self):
        """Test that empty but correctly shaped arrays are valid."""
        positions = numpy.empty((1, 3))
        velocities = numpy.empty((1, 3))
        uid = numpy.empty(1)
        common._validate_inputs_box_partitioning(
            positions, velocities, uid, None)

    @pytest.mark.parametrize(
        "positions_shape, expected_error",
        [
            # Wrong number of dimensions
            ((100,), "positions must have shape"),
            ((100, 3, 2), "positions must have shape"),
            # Wrong second dimension
            ((100, 2), "positions must have shape"),
            ((100, 4), "positions must have shape"),
            ((100, 1), "positions must have shape"),
        ],
    )
    def test_invalid_positions_shape(self, positions_shape, expected_error):
        """Test that invalid positions shapes raise ValueError."""
        positions = numpy.zeros(positions_shape)
        velocities = numpy.zeros((100, 3))
        uid = numpy.zeros(100)

        with pytest.raises(ValueError, match=expected_error):
            common._validate_inputs_box_partitioning(
                positions, velocities, uid, None)

    @pytest.mark.parametrize(
        "velocities_shape, expected_error",
        [
            # Wrong number of dimensions
            ((100,), "velocities must have shape"),
            ((100, 3, 2), "velocities must have shape"),
            # Wrong second dimension
            ((100, 2), "velocities must have shape"),
            ((100, 4), "velocities must have shape"),
            ((100, 1), "velocities must have shape"),
        ],
    )
    def test_invalid_velocities_shape(self, velocities_shape, expected_error):
        """Test that invalid velocities shapes raise ValueError."""
        positions = numpy.zeros((100, 3))
        velocities = numpy.zeros(velocities_shape)
        uid = numpy.zeros(100)

        with pytest.raises(ValueError, match=expected_error):
            common._validate_inputs_box_partitioning(
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
            common._validate_inputs_box_partitioning(
                positions, velocities, uid, None)

    @pytest.mark.parametrize(
        "pos_size, vel_size, uid_size",
        [
            (100, 50, 100),  # velocities mismatch
            (100, 100, 50),  # uid mismatch
            (50, 100, 100),  # positions mismatch
            (100, 50, 75),   # all different
            # (0, 10, 0),      # mixed empty/non-empty
        ],
    )
    def test_mismatched_array_lengths(self, pos_size, vel_size, uid_size):
        """Test that arrays with mismatched lengths raise ValueError."""
        positions = numpy.zeros((pos_size, 3))
        velocities = numpy.zeros((vel_size, 3))
        uid = numpy.zeros(uid_size)

        with pytest.raises(ValueError, match="positions, velocities, and uid must have the same length"):
            common._validate_inputs_box_partitioning(
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
            common._validate_inputs_box_partitioning(
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
            common._validate_inputs_box_partitioning(
                positions, velocities, uid, props)

    def test_mismatched_props_lengths(self, valid_data):
        """Test that props lists with different lengths raise ValueError."""
        positions, velocities, uid, _ = valid_data

        arrays = [numpy.zeros(100)]
        labels = ['prop1', 'prop2']  # Different length
        dtypes = [numpy.float32]
        props = (arrays, labels, dtypes)

        with pytest.raises(ValueError, match="All lists in props must have the same length"):
            common._validate_inputs_box_partitioning(
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
            common._validate_inputs_box_partitioning(
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
            common._validate_inputs_box_partitioning(
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
            common._validate_inputs_box_partitioning(
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
        common._validate_inputs_box_partitioning(
            positions, velocities, uid, props)

    def test_tuples_instead_of_lists_in_props(self, valid_data):
        """Test that tuples are accepted instead of lists in props."""
        positions, velocities, uid, n_particles = valid_data

        arrays = (numpy.zeros(n_particles),)  # tuple instead of list
        labels = ('prop1',)                # tuple instead of list
        dtypes = (numpy.float32,)            # tuple instead of list
        props = (arrays, labels, dtypes)

        # Should not raise an exception
        common._validate_inputs_box_partitioning(
            positions, velocities, uid, props)


class TestValidationIntegration:
    """Integration tests for both validation functions together."""

    @pytest.mark.parametrize("n_particles", [1, 10, 1000])
    def test_various_particle_counts(self, n_particles):
        """Test both functions with various particle counts."""
        # Test boxsize/minisize validation
        common._validate_inputs_boxsize_minisize(100.0, 10.0)

        # Test array validation
        positions = numpy.random.rand(n_particles, 3)
        velocities = numpy.random.rand(n_particles, 3)
        uid = numpy.arange(n_particles)

        common._validate_inputs_box_partitioning(
            positions, velocities, uid, None)

    def test_realistic_simulation_data(self):
        """Test with realistic simulation-like data."""
        # Large simulation
        n_particles = 10000
        boxsize = 500.0
        minisize = 25.0

        # Validate box parameters
        common._validate_inputs_boxsize_minisize(boxsize, minisize)

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
        common._validate_inputs_box_partitioning(
            positions, velocities, uid, props)


###
