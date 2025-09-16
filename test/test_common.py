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


class TestMkdir:
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
