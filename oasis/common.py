import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial, wraps
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Union

import numpy

# Gravitational constant
G_GRAVITY: float = 4.3e-09     # Mpc (km/s)^2 / M_sun


@dataclass(frozen=True)
class AnsiColor:
    """ANSI escape codes for colored and styled terminal output."""
    HEADER: str = "\033[35m"
    OKBLUE: str = "\033[34m"
    OKCYAN: str = "\033[36m"
    OKGREEN: str = "\033[32m"
    WARNING: str = "\033[33m"
    FAIL: str = "\033[31m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    BULLET: str = "\u25CF"


def timer(
    func: Optional[Callable] = None,
    *,
    fancy: bool = True,
    enabled: bool = True,
    precision: int = 3,
) -> Callable:
    """
    Decorator that measures and prints a function's execution time.

    Parameters
    ----------
    func : Callable, optional
        Function to be timed.
    fancy : bool, optional
        Whether to use colored output, by default True.
    enabled : bool, optional
        Whether timing is enabled, by default True.
    precision : int, optional
        Decimal precision for time display, by default 3.

    Returns
    -------
    Callable
        Decorated function or decorator factory.

    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)

    >>> @timer(fancy=False, precision=2)
    ... def another_function():
    ...     return 42
    """
    if func is None:
        return partial(timer, fancy=fancy, enabled=enabled, precision=precision)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        fmt = '%Y-%m-%d %H:%M:%S'
        start_time = datetime.now()
        perf_start = perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            finish_time = datetime.now()
            elapsed = perf_counter() - perf_start
            elapsed_str = f"{elapsed:.{precision}f}s"

            if fancy:
                print(
                    f"\t{AnsiColor.BOLD}Process:{AnsiColor.ENDC} "
                    f"{AnsiColor.FAIL}{func.__name__}{AnsiColor.ENDC}\n"
                    f"Start:  {AnsiColor.OKBLUE}{start_time.strftime(fmt)}{AnsiColor.ENDC}\n"
                    f"Finish: {AnsiColor.OKBLUE}{finish_time.strftime(fmt)}{AnsiColor.ENDC}\n"
                    f"{AnsiColor.BULLET}{AnsiColor.BOLD}{AnsiColor.OKGREEN} Elapsed: "
                    f"{AnsiColor.ENDC}{AnsiColor.WARNING}{elapsed_str}{AnsiColor.ENDC}"
                )
            else:
                print(
                    f"\tProcess: {func.__name__}\n"
                    f"Start:  {start_time.strftime(fmt)}\n"
                    f"Finish: {finish_time.strftime(fmt)}\n"
                    f"{AnsiColor.BULLET} Elapsed: {elapsed_str}"
                )

        return result

    return wrapper


class TimerContext:
    """
    Context manager for timing code blocks.

    Examples
    --------
    >>> with TimerContext("data processing"):
    ...     # some time-consuming operation
    ...     time.sleep(1)
    """

    def __init__(self, name: str = "operation", fancy: bool = True, precision: int = 3):
        self.name = name
        self.fancy = fancy
        self.precision = precision
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = perf_counter() - self.start_time
        elapsed_str = f"{self.elapsed:.{self.precision}f}s"

        if self.fancy:
            print(
                f"{AnsiColor.BULLET}{AnsiColor.BOLD}{AnsiColor.OKGREEN}"
                f"{self.name} completed in {AnsiColor.ENDC}"
                f"{AnsiColor.WARNING}{elapsed_str}{AnsiColor.ENDC}"
            )
        else:
            print(f"{self.name} completed in {elapsed_str}")


def get_min_unit_dtype(num: int) -> numpy.dtype:
    """
    Determine the minimum unsigned integer dtype to represent a number.

    Parameters
    ----------
    num : int
        Non-negative integer value.

    Returns
    -------
    numpy.dtype
        Minimum unsigned integer dtype that can represent the number.

    Raises
    ------
    TypeError
        If input is not a non-negative integer.
    OverflowError
        If number exceeds uint64 maximum value.

    Examples
    --------
    >>> get_min_uint_dtype(255)
    dtype('uint8')
    >>> get_min_uint_dtype(70000)
    dtype('uint32')
    """
    if not isinstance(num, int) or num < 0:
        raise TypeError("Input must be a non-negative integer")

    # Check in order of size
    for dtype in [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
        if num <= numpy.iinfo(dtype).max:
            return dtype

    raise OverflowError(f"Number {num} exceeds maximum uint64 value")


def ensure_dir_exists(
    path: Union[str, Path],
    verbose: bool = False,
    parents: bool = True,
) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to ensure exists.
    verbose : bool, optional
        Whether to print status messages, by default False.
    parents : bool, optional
        Whether to create parent directories, by default True.

    Returns
    -------
    Path
        Absolute path to the directory.

    Raises
    ------
    PermissionError
        If lacking permissions to create directory.
    OSError
        If directory creation fails for other reasons.

    Examples
    --------
    >>> ensure_dir_exists("./data/output")
    PosixPath('/absolute/path/to/data/output')
    """
    path_obj = Path(path).resolve()

    if path_obj.exists():
        if not path_obj.is_dir():
            raise OSError(f"Path exists but is not a directory: {path_obj}")
        if verbose:
            print(f"Directory already exists: {path_obj}")
        return path_obj

    try:
        path_obj.mkdir(parents=parents, exist_ok=True)
        if verbose:
            print(f"Created directory: {path_obj}")
        return path_obj
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied creating directory {path_obj}") from e
    except OSError as e:
        raise OSError(f"Failed to create directory {path_obj}: {e}") from e


###
