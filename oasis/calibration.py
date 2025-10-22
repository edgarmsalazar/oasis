import os
from multiprocessing import Pool
from typing import Dict, Tuple, Union

import h5py
import numpy
from matplotlib import pyplot
from matplotlib.colorbar import ColorbarBase
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, minimize
from scipy.stats import iqr as interquartile_range
from scipy.spatial import cKDTree
from tqdm import tqdm

from oasis.common import (G_GRAVITY, _validate_inputs_boxsize_minisize,
                          _validate_inputs_coordinate_arrays,
                          _validate_inputs_existing_path,
                          _validate_inputs_mini_box_id,
                          _validate_inputs_positive_number,
                          _validate_inputs_positive_number_non_zero,
                          _validate_inputs_seed_data)
from oasis.coordinates import relative_coordinates, velocity_components
from oasis.minibox import get_mini_box_id, load_particles

__all__ = [
    'get_calibration_data',
    'calibrate',
]

# Label sizes
SIZE_TICKS: int = 12
SIZE_LEGEND: int = 14
SIZE_LABELS: int = 16

COLOR_BLUE = '#6591b5'
COLOR_GRAY = '#808080'
COLOR_RED = '#db715b'


def _compute_r200m_and_v200m(
    radial_distances: numpy.ndarray,
    particle_mass: float,
    mass_density: float
) -> Tuple[float, float]:
    """Compute virial radius R200m and circular velocity V200 for a halo.

    This function calculates the virial properties of a halo by analyzing the
    radial density profile. R200m is defined as the radius at which the mean
    enclosed density equals 200 times the mean matter density of the universe.
    V200 is the corresponding circular velocity at R200m.

    Parameters
    ----------
    radial_distances : numpy.ndarray
        Array of radial distances from the halo center with shape (N,).
        Values must be non-negative and in simulation units (typically Mpc).
    particle_mass : float
        Mass of each simulation particle in simulation units (typically M_sun).
        Must be positive.
    mass_density : float
        Mean matter density of the universe in simulation units
        (typically M_sun/Mpc^3). Must be positive.

    Returns
    -------
    r200m : float
        Virial radius R200m in simulation units. Returns 0.0 if computation fails.
    v200_squared : float
        Square of circular velocity V200² in (simulation_units)². Returns 0.0
        if computation fails.

    Raises
    ------
    TypeError
        If radial_distances cannot be converted to numpy array.
    ValueError
        If particle_mass or mass_density are not positive scalars, or if 
        radial_distances contains negative values.
    RuntimeError
        If density profile computation encounters numerical issues.

    Notes
    -----
    - The algorithm sorts particles by radius and computes cumulative mass profile
    - Density is computed as rho(r) = M(<r) / (4π/3 x r³)  
    - R200m corresponds to the largest radius where rho(r) ≥ 200xrho_m
    - If no radius satisfies the criterion, uses the outermost particle
    - V200² = GM200m/R200m where G is the gravitational constant

    Examples
    --------
    >>> import numpy
    >>> distances = numpy.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
    >>> r200m, v200_sq = _compute_r200m_and_v200(distances, 1e10, 2.78e11)
    >>> print(f"R200m = {r200m:.3f} Mpc, V200 = {numpy.sqrt(v200_sq):.1f} km/s")

    >>> # Handle edge case with no particles
    >>> r200m, v200_sq = _compute_r200m_and_v200(numpy.array([]), 1e10, 2.78e11)
    >>> print(f"Empty input: R200m = {r200m}, V200² = {v200_sq}")
    Empty input: R200m = 0.0, V200² = 0.0

    See Also
    --------
    G_GRAVITY : Gravitational constant used in V200 computation
    """
    # Input validation
    if radial_distances.size == 0:
        raise ValueError(f"radial_distances is empty")
    _validate_inputs_positive_number_non_zero(particle_mass, 'particle_mass')
    _validate_inputs_positive_number_non_zero(mass_density, 'mass_density')

    # Sort distances for cumulative mass profile
    sorted_distances = numpy.sort(radial_distances)
    n_particles = len(sorted_distances)

    # Avoid division by zero for particles at the center
    # Replace zero distances with a small value
    min_distance = numpy.finfo(float).eps
    sorted_distances = numpy.maximum(sorted_distances, min_distance)

    # Compute cumulative mass profile
    mass_profile = particle_mass * numpy.arange(1, n_particles + 1)

    # Compute volume-averaged densities
    volumes = (4.0 / 3.0) * numpy.pi * sorted_distances**3
    densities = mass_profile / volumes

    # Find where density drops below 200 * mass_density
    target_density = 200.0 * mass_density
    mask_above_target = densities >= target_density

    if not numpy.any(mask_above_target):
        # If no particles satisfy criterion, use outermost particle
        r200m = sorted_distances[-1]
        m200m = mass_profile[-1]
    else:
        # Use last particle that satisfies the density criterion
        last_valid_idx = numpy.where(mask_above_target)[0][-1]
        r200m = sorted_distances[last_valid_idx]
        m200m = mass_profile[last_valid_idx]

    # Compute V200² with safety check
    if r200m > 0 and m200m > 0:
        v200_squared = G_GRAVITY * m200m / r200m
        if not numpy.isfinite(v200_squared) or v200_squared <= 0:
            v200_squared = 0.0
    else:
        v200_squared = 0.0

    return float(r200m), float(v200_squared)


def _get_candidate_particle_data(
    mini_box_id: int,
    position_seeds: numpy.ndarray,
    velocity_seeds: numpy.ndarray,
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_mass: float,
    mass_density: float,
) -> numpy.ndarray:
    """Extract and process particle data for all seeds within a single minibox.

    This function loads particles from a specified minibox and its neighbors, then
    processes each seed to compute scaled radial distances, radial velocities, and
    velocity magnitudes in units of the virial properties (R200m, V200).

    The function performs the following steps for each seed:
    1. Loads particles within r_max of the seed position
    2. Computes relative positions and velocities 
    3. Calculates R200m and V200 from the density profile
    4. Scales all quantities by the virial properties

    Parameters
    ----------
    mini_box_id : int
        ID of the minibox containing the seeds. Must be a valid minibox ID
        for the given box configuration.
    position_seeds : numpy.ndarray
        Seed positions with shape (n_seeds, 3). Each row contains the 
        (x, y, z) coordinates of a seed in simulation units.
    velocity_seeds : numpy.ndarray
        Seed velocities with shape (n_seeds, 3). Each row contains the
        (vx, vy, vz) velocity components of a seed in simulation units.
    r_max : float
        Maximum distance from seed centers to consider particles, in 
        simulation units. Must be positive.
    boxsize : float
        Size of the cubic simulation box in simulation units. Must be positive.
    minisize : float
        Size of each cubic minibox in simulation units. Must be positive
        and typically smaller than boxsize.
    save_path : str
        Path to directory containing minibox HDF5 files. Must be a valid
        directory path with the expected minibox file structure.
    particle_mass : float
        Mass of each simulation particle in simulation units (typically M_sun).
        Must be positive.
    mass_density : float
        Mean matter density of the universe in simulation units 
        (typically M_sun/Mpc^3). Must be positive.

    Returns
    -------
    numpy.ndarray
        Particle data array with shape (n_particles, 3) where each row contains:
        - Column 0: r/R200m - Radial distance scaled by R200m
        - Column 1: vr/V200 - Radial velocity scaled by V200  
        - Column 2: ln(v²/V200²) - Natural log of velocity squared scaled by V200²

        If no valid particles are found, returns empty array with shape (0, 3).

    Raises
    ------
    TypeError
        If mini_box_id is not an integer, or if array inputs cannot be
        converted to numpy arrays.
    ValueError
        If mini_box_id is negative, array shapes are incompatible, or
        if any scalar parameters are non-positive.
    FileNotFoundError
        If save_path doesn't exist or required minibox files are missing.
    OSError
        If minibox HDF5 files cannot be read or are corrupted.
    RuntimeError
        If particle loading fails or density profile computation encounters
        numerical issues.

    Notes
    -----
    - Uses periodic boundary conditions when computing relative coordinates
    - R200m is defined as the radius where mean enclosed density equals 200xrho_m
    - V200 = √(GM200m/R200m) where M200m is the mass within R200m
    - Seeds with insufficient particles or invalid virial properties are skipped
    - Memory usage scales with the number of particles within r_max of all seeds

    Examples
    --------
    >>> # Process seeds in minibox 42
    >>> pos_seeds = numpy.array([[10.0, 15.0, 20.0], [25.0, 30.0, 35.0]])
    >>> vel_seeds = numpy.array([[100.0, 50.0, -75.0], [-80.0, 120.0, 90.0]])
    >>> data = _get_candidate_particle_data(
    ...     mini_box_id=42,
    ...     position_seeds=pos_seeds, 
    ...     velocity_seeds=vel_seeds,
    ...     r_max=5.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     save_path="/data/miniboxes/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11
    ... )
    >>> print(f"Processed {data.shape[1]} particles from {len(pos_seeds)} seeds")

    See Also
    --------
    _compute_r200m_and_v200m : Computes virial radius and velocity
    load_particles : Loads particles from minibox files
    velocity_components : Decomposes velocities into radial/tangential components
    """
    # Validate inputs
    _validate_inputs_boxsize_minisize(boxsize, minisize)
    _validate_inputs_mini_box_id(
        mini_box_id, int(numpy.ceil(boxsize/minisize)))
    _validate_inputs_coordinate_arrays(position_seeds, 'seed positions')
    _validate_inputs_coordinate_arrays(velocity_seeds, 'seed velocities')
    _validate_inputs_existing_path(save_path)
    _validate_inputs_positive_number_non_zero(r_max, 'r_max')
    _validate_inputs_positive_number_non_zero(particle_mass, 'particle_mass')
    _validate_inputs_positive_number_non_zero(mass_density, 'rhom')

    # Load particles in minibox.
    position_particles, velocity_particles, _ = \
        load_particles(mini_box_id, boxsize, minisize, save_path)

    # Iterate over seeds in current mini box.
    radius, radial_velocity, log_velocity_squared = ([] for _ in range(3))
    for position_seed_i, velocity_seed_i in zip(position_seeds, velocity_seeds):
        # Find particles within r_max of the seed
        relative_position = relative_coordinates(
            position_particles, position_seed_i, boxsize)
        mask_close = numpy.prod(
            numpy.abs(relative_position) <= r_max, axis=1, dtype=bool)

        # Apply mask
        relative_position = relative_position[mask_close]
        relative_velocity = velocity_particles[mask_close] - velocity_seed_i

        # Compute radial distance (L2 norm). No need to further filter by r_max
        # since we already applied a cubic mask.
        rps = numpy.linalg.norm(relative_position, axis=1)

        # Compute velocity components
        vrp, _, v2p = velocity_components(relative_position, relative_velocity)

        # Compute R200m and M200m
        r200m, v200m_sq = _compute_r200m_and_v200m(
            rps, particle_mass, mass_density)

        # Append rescaled quantities to containers
        radius.append(rps / r200m)
        radial_velocity.append(vrp / numpy.sqrt(v200m_sq))

        # Prevent log(0) or log(negative) by setting minimum value
        v_sq_ratio = v2p / v200m_sq
        v_sq_ratio = numpy.maximum(v_sq_ratio, 1e-10)

        log_velocity_squared.append(numpy.log(v_sq_ratio))

    # Concatenate arrays
    radius = numpy.concatenate(radius)
    radial_velocity = numpy.concatenate(radial_velocity)
    log_velocity_squared = numpy.concatenate(log_velocity_squared)

    return numpy.vstack([radius, radial_velocity, log_velocity_squared])


def _find_isolated_seeds(
    position: numpy.ndarray,
    mass: Union[float, numpy.ndarray],
    radius: Union[float, numpy.ndarray],
    max_seeds: int,
    boxsize: float,
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
) -> numpy.ndarray:
    """Find seeds that are isolated from other massive neighbors.

    This function identifies seeds that dominate their local environment by
    checking that all neighboring seeds within a specified isolation radius
    have masses below a threshold fraction of the candidate seed's mass.

    The isolation criterion requires that all neighbors within an isolation
    radius (isolation_radius_factor x R200) must have masses less than
    isolation_factor x M200 of the candidate seed.

    Parameters
    ----------
    position : numpy.ndarray
        Seed positions with shape (n_seeds, 3). Each row contains the 
        (x, y, z) coordinates of a seed in simulation units.
    mass : Union[float, numpy.ndarray]
        Mass of each seed in simulation units (typically M_sun). Can be
        a scalar if all seeds have the same mass, or array with shape (n_seeds,).
        Must be positive.
    radius : Union[float, numpy.ndarray]
        Virial radius R200 of each seed in simulation units. Can be
        a scalar if all seeds have the same radius, or array with shape (n_seeds,).
        Must be positive.
    max_seeds : int
        Maximum number of isolated seeds to return. Must be positive.
        Function stops searching once this many isolated seeds are found.
    boxsize : float
        Size of the cubic simulation box in simulation units. Must be positive.
        Used for periodic boundary condition calculations.
    isolation_factor : float, optional
        Maximum allowed mass ratio for neighbors. Neighbors must have
        mass < isolation_factor x seed_mass to satisfy isolation criterion.
        Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200 to define isolation radius. Neighbors within
        isolation_radius_factor x R200 are checked. Default is 2.0.

    Returns
    -------
    numpy.ndarray
        Array of indices of isolated seeds with shape (n_isolated,).
        Indices correspond to rows in the input position array.
        Returned in order of discovery, up to max_seeds entries.

    Raises
    ------
    TypeError
        If position cannot be converted to numpy array, or if max_seeds
        is not an integer.
    ValueError
        If array shapes are incompatible, if any scalar parameters are
        non-positive, or if position array has wrong dimensions.

    Notes
    -----
    - Uses periodic boundary conditions for distance calculations
    - Searches seeds in input order, stopping when max_seeds are found
    - A seed with no neighbors within isolation radius is automatically isolated
    - Isolation radius scales with each seed's individual R200 value
    - Memory usage is O(n_seeds²) in worst case for distance calculations

    Examples
    --------
    >>> # Find up to 100 isolated seeds
    >>> positions = numpy.random.uniform(0, 100, (1000, 3))
    >>> masses = numpy.random.uniform(1e12, 1e15, 1000)
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> isolated = _find_isolated_seeds(
    ...     position=positions,
    ...     mass=masses, 
    ...     radius=radii,
    ...     max_seeds=100,
    ...     boxsize=100.0,
    ...     isolation_factor=0.3,
    ...     isolation_radius_factor=2.5
    ... )
    >>> print(f"Found {len(isolated)} isolated seeds")

    >>> # Handle uniform mass case
    >>> isolated = _find_isolated_seeds(
    ...     position=positions,
    ...     mass=1e13,  # All seeds have same mass
    ...     radius=1.0, # All seeds have same radius
    ...     max_seeds=50,
    ...     boxsize=100.0
    ... )

    See Also
    --------
    relative_coordinates : Computes relative positions with periodic boundaries
    """
    # Input validation
    _validate_inputs_coordinate_arrays(position, 'position')
    _validate_inputs_positive_number_non_zero(max_seeds, 'max_seeds')
    _validate_inputs_positive_number_non_zero(boxsize, 'boxsize')
    _validate_inputs_positive_number_non_zero(isolation_factor, 'isolation_factor')
    _validate_inputs_positive_number_non_zero(
        isolation_radius_factor, 'isolation_radius_factor')

    # This should never be needed as all seeds are assumed to have different
    # masses and radii. But just in case.
    if isinstance(mass, (int, float)):
        mass = numpy.full(position.shape[0], mass)

    if isinstance(radius, (int, float)):
        radius = numpy.full(position.shape[0], radius)

    n_seeds = position.shape[0]
    isolated_indices = []

    # Pre-compute isolation radii for efficiency
    isolation_radii = isolation_radius_factor * radius
    max_neighbor_masses = isolation_factor * mass

    # Compute KDTree with seed positions. Saw a 40% improvement over previous 
    # method.
    position_tree = cKDTree(position, boxsize=boxsize)

    for i in range(n_seeds):
        # If all requested isolated seeds have been found, exit loop
        if len(isolated_indices) >= max_seeds:
            break

        current_position = position[i]
        isolation_radius = isolation_radii[i]
        max_neighbor_mass = max_neighbor_masses[i]

        # Use a KDTree to find neighbouring seeds.
        idx_neighbor = position_tree.query_ball_point(current_position, 
                                                      isolation_radius,
                                                      p=numpy.inf,
                                                      return_sorted=True)
        idx_neighbor = [item for item in idx_neighbor if item != i]

        # No neighbors found - seed is isolated
        if len(idx_neighbor) == 0:
            isolated_indices.append(i)
            continue

        neighbor_masses = mass[idx_neighbor]

        # Check isolation criterion: all neighbors must have mass < threshold
        if numpy.all(neighbor_masses < max_neighbor_mass):
            isolated_indices.append(i)

    return numpy.array(isolated_indices)


def _group_seeds_by_minibox(
    position: numpy.ndarray,
    velocity: numpy.ndarray,
    boxsize: float,
    minisize: float,
) -> Dict[int, Tuple[numpy.ndarray, numpy.ndarray]]:
    """Group seeds by their containing minibox for efficient parallel processing.

    This function assigns each seed to its containing minibox based on spatial
    coordinates and returns a dictionary mapping minibox IDs to the positions
    and velocities of seeds within that minibox.

    Parameters
    ----------
    position : numpy.ndarray
        Seed positions with shape (n_seeds, 3). Each row contains the 
        (x, y, z) coordinates of a seed in simulation units.
    velocity : numpy.ndarray
        Seed velocities with shape (n_seeds, 3). Each row contains the
        (vx, vy, vz) velocity components of a seed in simulation units.
    boxsize : float
        Size of the cubic simulation box in simulation units. Must be positive.
    minisize : float
        Size of each cubic minibox in simulation units. Must be positive
        and typically smaller than boxsize. The number of miniboxes per
        dimension is ceil(boxsize/minisize).

    Returns
    -------
    Dict[int, Tuple[numpy.ndarray, numpy.ndarray]]
        Dictionary mapping minibox IDs to tuples of (positions, velocities).
        Each minibox ID maps to:
        - positions: numpy.ndarray with shape (n_seeds_in_box, 3)
        - velocities: numpy.ndarray with shape (n_seeds_in_box, 3)
        Only miniboxes containing at least one seed are included.

    Raises
    ------
    TypeError
        If position or velocity arrays cannot be converted to numpy arrays.
    ValueError
        If array shapes are incompatible, if boxsize or minisize are
        non-positive, or if minisize > boxsize.

    Notes
    -----
    - Minibox IDs are computed using spatial hashing of seed coordinates
    - Empty miniboxes (containing no seeds) are not included in the result
    - Seeds exactly on minibox boundaries are assigned consistently
    - Memory usage scales linearly with the number of seeds

    Examples
    --------
    >>> # Group 1000 seeds into miniboxes
    >>> positions = numpy.random.uniform(0, 100, (1000, 3))
    >>> velocities = numpy.random.normal(0, 200, (1000, 3))
    >>> groups = _group_seeds_by_minibox(
    ...     position=positions,
    ...     velocity=velocities, 
    ...     boxsize=100.0,
    ...     minisize=10.0
    ... )
    >>> print(f"Seeds distributed across {len(groups)} miniboxes")
    >>> for box_id, (pos, vel) in groups.items():
    ...     print(f"Minibox {box_id}: {len(pos)} seeds")

    See Also
    --------
    get_mini_box_id : Computes minibox ID from spatial coordinates
    """
    # Validate inputs
    _validate_inputs_coordinate_arrays(position, 'position')
    _validate_inputs_coordinate_arrays(velocity, 'velocity')
    _validate_inputs_boxsize_minisize(boxsize, minisize)

    # Get mini-box IDs for all seeds
    mini_box_ids = get_mini_box_id(position, boxsize, minisize)

    # Group by minibox ID using dictionary comprehension
    unique_ids = numpy.unique(mini_box_ids)
    minibox_groups = {}
    for minibox_id in unique_ids:
        mask = mini_box_ids == minibox_id
        minibox_groups[int(minibox_id)] = (position[mask], velocity[mask])

    return minibox_groups


def _select_candidate_seeds(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_mass: float,
    mass_density: float,
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
    n_threads: int = None,
) -> tuple[numpy.ndarray]:
    """Select isolated massive seeds and extract scaled particle data around them.

    This function identifies the most massive isolated seeds from the input
    catalog and processes particles within r_max of each seed to compute
    scaled kinematic quantities. Only seeds that dominate their local environment
    are considered (isolation criterion: all neighbors within 2 x R200 must have
    mass < 0.2 x M200 of the seed).

    The function performs the following workflow:
    1. Sorts seeds by mass in descending order
    2. Finds up to n_seeds isolated seeds using the isolation criterion
    3. Groups selected seeds by minibox for efficient parallel processing
    4. Extracts and processes particles around each seed
    5. Returns scaled kinematic data for all processed particles

    Parameters
    ----------
    n_seeds : int
        Maximum number of seeds to process. Must be positive.
        Function selects up to this many of the most massive isolated seeds.
    seed_data : Tuple[numpy.ndarray]
        Tuple containing (position, velocity, mass, radius) arrays:
        - position: shape (n_total_seeds, 3) - seed coordinates
        - velocity: shape (n_total_seeds, 3) - seed velocities  
        - mass: shape (n_total_seeds,) - seed masses (M200)
        - radius: shape (n_total_seeds,) - seed virial radii (R200)
    r_max : float
        Maximum distance from seed centers to consider particles, in 
        simulation units. Must be positive.
    boxsize : float
        Size of the cubic simulation box in simulation units. Must be positive.
    minisize : float
        Size of each cubic minibox in simulation units. Must be positive
        and typically smaller than boxsize.
    save_path : str
        Path to directory containing minibox HDF5 files. Must be a valid
        directory path with the expected minibox file structure.
    particle_mass : float
        Mass of each simulation particle in simulation units (typically M_sun).
        Must be positive.
    mass_density : float
        Mean matter density of the universe in simulation units 
        (typically M_sun/Mpc^3). Must be positive.
    isolation_factor : float, optional
        Maximum allowed mass ratio for isolation criterion. Neighbors must have
        mass < isolation_factor x seed_mass. Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200 to define isolation radius for neighbor search.
        Default is 2.0.
    n_threads : int, optional
        Number of threads for parallel processing. If None, uses half of
        available CPU cores, capped by the number of miniboxes to process.

    Returns
    -------
    numpy.ndarray
        Particle data array with shape (n_particles, 3) where each row contains:
        - Column 0: r/R200m - Radial distance scaled by R200m
        - Column 1: vr/V200 - Radial velocity scaled by V200  
        - Column 2: ln(v²/V200²) - Natural log of velocity squared scaled by V200²

        All quantities are dimensionless and scaled by the virial properties
        computed individually for each seed.

    Raises
    ------
    TypeError
        If n_seeds is not an integer, or if array inputs in seed_data cannot
        be converted to numpy arrays.
    ValueError
        If n_seeds is non-positive, if array shapes in seed_data are incompatible,
        or if any scalar parameters are non-positive.
    FileNotFoundError
        If save_path doesn't exist or required minibox files are missing.
    RuntimeError
        If parallel processing fails and fallback to sequential mode is needed,
        or if particle loading encounters errors.

    Notes
    -----
    - Seeds are ranked by mass, with the most massive considered first
    - Isolation criterion ensures selected seeds dominate their local environment
    - Uses periodic boundary conditions for all distance calculations
    - Automatically falls back to sequential processing if parallelization fails
    - Memory usage scales with the number of particles within r_max of all seeds
    - Progress bars show processing status for miniboxes

    Examples
    --------
    >>> # Process 100 most massive isolated seeds
    >>> positions = numpy.random.uniform(0, 100, (10000, 3))
    >>> velocities = numpy.random.normal(0, 200, (10000, 3)) 
    >>> masses = numpy.random.lognormal(30, 1, 10000)
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> seed_data = (positions, velocities, masses, radii)
    >>> 
    >>> results = _select_candidate_seeds(
    ...     n_seeds=100,
    ...     seed_data=seed_data,
    ...     r_max=5.0,
    ...     boxsize=100.0,
    ...     minisize=10.0, 
    ...     save_path="/data/miniboxes/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     n_threads=8
    ... )
    >>> print(f"Processed {len(results)} particles from isolated seeds")

    See Also
    --------
    _find_isolated_seeds : Identifies seeds meeting isolation criterion
    _group_seeds_by_minibox : Groups seeds for efficient parallel processing
    _get_candidate_particle_data : Processes particles around seeds in one minibox
    """
    # Validate inputs
    _validate_inputs_positive_number_non_zero(n_seeds, 'n_seeds')
    _validate_inputs_seed_data(seed_data)
    _validate_inputs_positive_number_non_zero(r_max, 'r_max')
    _validate_inputs_existing_path(save_path)
    _validate_inputs_boxsize_minisize(boxsize, minisize)
    _validate_inputs_positive_number_non_zero(particle_mass, 'particle_mass')
    _validate_inputs_positive_number_non_zero(mass_density, 'mass_density')
    if n_threads is not None:
        _validate_inputs_positive_number_non_zero(n_threads, 'n_threads')

    # Unpack seed data
    position, velocity, mass, radius = seed_data

    # Trim values outside boxsize due to floating point precision
    position = numpy.mod(position, boxsize)

    # Rank order by mass.
    order = numpy.argsort(-mass)
    position = position[order]
    velocity = velocity[order]
    radius = radius[order]
    mass = mass[order]

    # Search for eligible (isolated) seeds.
    isolated_indices = _find_isolated_seeds(position, mass, radius, n_seeds,
                                            boxsize, isolation_factor,
                                            isolation_radius_factor)

    # Select only the candidate seeds.
    position = position[isolated_indices]
    velocity = velocity[isolated_indices]

    # Group seeds by minibox for efficient processing
    minibox_groups = _group_seeds_by_minibox(
        position, velocity, boxsize, minisize)

    # Cap the number of threads to the total number of miniboxes to process
    n_miniboxes = len(minibox_groups)
    if n_threads is None:
        n_threads = min(max(1, os.cpu_count()//2), n_miniboxes)
    else:
        n_threads = min(n_threads, n_miniboxes)
    print(f"Processing {n_miniboxes} miniboxes using {n_threads} threads...")

    # Set up multiprocessing
    processing_args = [
        (minibox_id, position_group, velocity_group, r_max, boxsize, minisize,
         save_path, particle_mass, mass_density)
        for minibox_id, (position_group, velocity_group) in minibox_groups.items()
    ]

    results = []
    # Safely handle multiprocessing falure with a fall back to a single thread.
    if n_threads > 1 and n_miniboxes > 1:
        try:
            with Pool(n_threads) as pool, tqdm(total=n_miniboxes, colour="blue",
                                               desc='Processing candidates',
                                               ncols=100) as pbar:
                for result in pool.starmap(_get_candidate_particle_data, processing_args):
                    results.append(result)
                    pbar.update()
                    pbar.refresh()
        except Exception as e:
            print(
                f"Warning: Parallel processing failed ({e}), falling back to sequential")
            # Fall back to sequential processing
            n_threads = 1

    if n_threads == 1:
        for args in tqdm(processing_args, desc='Processing miniboxes',
                         colour='blue', ncols=100):
            result = _get_candidate_particle_data(*args)
            results.append(result)

    results = numpy.concatenate(results, axis=1).T

    return results


def _diagnostic_calibration_data_plot(
    save_path: str,
    radius: numpy.ndarray,
    radial_velocity: numpy.ndarray,
    log_velocity_squared: numpy.ndarray,
) -> None:
    """Generate diagnostic plots for calibration data visualization.

    This function creates a two-panel diagnostic plot showing the distribution
    of particle data in the (r/R200m, ln(v²/V200²)) space, separated by the
    sign of the radial velocity. The plots help visualize the phase space
    distribution of particles around halos for calibration purposes.

    Parameters
    ----------
    save_path : str
        Directory path where the diagnostic plot will be saved. Must be a valid
        directory path with write permissions. The plot is saved as 
        'calibration_data.png'.
    radius : numpy.ndarray
        Array of radial distances scaled by R200m with shape (n_particles,).
        Values should typically be positive and in the range [0, 3].
    radial_velocity : numpy.ndarray
        Array of radial velocities scaled by V200 with shape (n_particles,).
        Values can be positive (outflow) or negative (inflow).
    log_velocity_squared : numpy.ndarray
        Array of natural logarithm of velocity squared scaled by V200² with
        shape (n_particles,). Values typically range from -3 to +3.

    Returns
    -------
    None
        Function saves the plot to disk but returns nothing.

    Raises
    ------
    TypeError
        If any array inputs cannot be converted to numpy arrays.
    ValueError
        If array shapes are incompatible or if save_path is not a valid string.
    OSError
        If save_path directory doesn't exist or lacks write permissions.

    Notes
    -----
    - Creates two side-by-side 2D histograms for vr > 0 and vr < 0
    - Uses 200x200 bins for high-resolution visualization
    - Plot ranges are fixed: r/R200m ∈ [0,2], ln(v²/V200²) ∈ [-2,2.5]
    - Uses 'terrain' colormap for particle density visualization
    - Includes LaTeX formatting for axis labels and titles
    - Colorbar shows relative particle counts but without tick labels
    - Figure is saved at 300 DPI for publication quality

    Examples
    --------
    >>> # Generate diagnostic plot for calibration data
    >>> radii = numpy.random.uniform(0.1, 2.0, 50000) 
    >>> vr = numpy.random.normal(0, 1, 50000)
    >>> log_v2 = numpy.random.normal(0, 1.5, 50000)
    >>> _diagnostic_calibration_data_plot(
    ...     save_path="/output/plots/",
    ...     radius=radii,
    ...     radial_velocity=vr, 
    ...     log_velocity_squared=log_v2
    ... )
    >>> # Plot saved to /output/plots/calibration_data.png

    See Also
    --------
    matplotlib.pyplot.hist2d : Creates 2D histogram plots
    matplotlib.colorbar.ColorbarBase : Creates standalone colorbars
    """
    pyplot.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 150,
    })

    cmap = 'terrain'
    limits = ((0, 2), (-2, 2.5))
    levels = 80

    mask_negative_vr = (radial_velocity < 0)
    mask_positive_vr = ~mask_negative_vr

    # Compute counts histograms for top two panels
    x_mesh_positive, y_mesh_positive, z_positive = \
        _hist2d_mesh(radius[mask_positive_vr],
                     log_velocity_squared[mask_positive_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=False,
                     )
    z_positive = _smooth_2d_hist(z_positive)

    x_mesh_negative, y_mesh_negative, z_negative = \
        _hist2d_mesh(radius[mask_negative_vr],
                     log_velocity_squared[mask_negative_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=False,
                     )
    z_negative = _smooth_2d_hist(z_negative)

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4.2))
    fig.suptitle('Calibration data', fontsize=SIZE_LABELS)
    axes = axes.flatten()

    for ax in axes:
        ax.set_xlabel(r'$r/R_{\rm 200m}$', fontsize=SIZE_LABELS)
        ax.set_ylabel(r'$\ln(a^2v^2/v_{\rm 200m}^2)$', fontsize=SIZE_LABELS)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)

    # Create a color bar to display mass ranges.
    cax = fig.add_axes([1.01, 0.2, 0.01, 0.7])
    cbar = ColorbarBase(cax, cmap=cmap, orientation="vertical", extend='max')
    cbar.set_label(r'Counts (a.u.)', fontsize=SIZE_LEGEND)
    cbar.set_ticklabels([], fontsize=SIZE_TICKS)
    cbar.ax.tick_params(size=0, labelleft=False, labelright=False,
                        labeltop=False, labelbottom=False)

    pyplot.sca(axes[0])
    pyplot.title(r'$v_r > 0$', fontsize=SIZE_LABELS)
    pyplot.contourf(x_mesh_positive, y_mesh_positive, z_positive.T,
                    levels=levels, cmap=cmap)

    pyplot.sca(axes[1])
    pyplot.title(r'$v_r < 0$', fontsize=SIZE_LABELS)
    pyplot.contourf(x_mesh_negative, y_mesh_negative, z_negative.T,
                    levels=levels, cmap=cmap)

    pyplot.tight_layout()
    pyplot.savefig(save_path + 'calibration_data.png', dpi=300,
                   bbox_inches='tight')

    return None


def get_calibration_data(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_mass: float,
    mass_density: float,
    redshift: float,
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
    n_threads: int = None,
    diagnostics: bool = True,
    overwrite: bool = False,
) -> Tuple[numpy.ndarray]:
    """Generate or load calibration data from isolated massive seed halos.

    This function either loads pre-computed calibration data from an HDF5 file
    or generates new calibration data by processing particles around isolated
    massive seeds. The calibration data consists of scaled kinematic quantities
    (radial distance, radial velocity, velocity magnitude) that can be used
    for statistical analysis or machine learning applications.

    The function implements a caching mechanism: if 'calibration_data.hdf5' exists
    in save_path, the data is loaded from disk. Otherwise, new data is computed
    and saved for future use.

    Parameters
    ----------
    n_seeds : int
        Maximum number of seeds to process for calibration. Must be positive.
        Function selects up to this many of the most massive isolated seeds.
    seed_data : Tuple[numpy.ndarray]
        Tuple containing (position, velocity, mass, radius) arrays:
        - position: shape (n_total_seeds, 3) - seed coordinates in simulation units
        - velocity: shape (n_total_seeds, 3) - seed velocities in simulation units
        - mass: shape (n_total_seeds,) - seed masses M200 in simulation units
        - radius: shape (n_total_seeds,) - seed virial radii R200 in simulation units
    r_max : float
        Maximum distance from seed centers to consider particles, in 
        simulation units. Must be positive, typically 2-5 x R200.
    boxsize : float
        Size of the cubic simulation box in simulation units. Must be positive.
    minisize : float
        Size of each cubic minibox in simulation units. Must be positive
        and smaller than boxsize.
    save_path : str
        Directory path for reading/writing calibration data and diagnostic plots.
        Must be a valid directory path with read/write permissions.
    particle_mass : float
        Mass of each simulation particle in simulation units (typically M_sun).
        Must be positive.
    mass_density : float
        Mean matter density of the universe in simulation units 
        (typically M_sun/Mpc^3). Must be positive.
    redshift : float
        Cosmological redshift of the universe in simulation.
    isolation_factor : float, optional
        Maximum allowed mass ratio for isolation criterion. Neighbors must have
        mass < isolation_factor x seed_mass. Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200 to define isolation radius for neighbor search.
        Default is 2.0.
    n_threads : int, optional
        Number of threads for parallel processing. If None, uses half of
        available CPU cores. Only used when generating new data.
    diagnostics : bool, optional
        Whether to generate diagnostic plots of the calibration data.
        Default is True.
    overwrite : bool, optional
        Whether to overwrite the existing calibration data. Default is False.

    Returns
    -------
    Tuple[numpy.ndarray]
        Tuple containing three arrays with calibration data:
        - radius: numpy.ndarray with shape (n_particles,)
            Radial distances scaled by R200m (dimensionless)
        - radial_velocity: numpy.ndarray with shape (n_particles,) 
            Radial velocities scaled by V200 (dimensionless)
        - log_velocity_squared: numpy.ndarray with shape (n_particles,)
            Natural log of velocity squared scaled by V200² (dimensionless)

    Raises
    ------
    TypeError
        If n_seeds is not an integer, or if array inputs in seed_data cannot
        be converted to numpy arrays.
    ValueError
        If n_seeds is non-positive, if array shapes in seed_data are incompatible,
        or if any scalar parameters are non-positive.
    FileNotFoundError
        If save_path doesn't exist, or if required minibox files are missing
        when generating new data.
    OSError
        If HDF5 file cannot be read/written, or if directory permissions
        are insufficient.

    Notes
    -----
    - Implements caching via HDF5 file to avoid recomputation
    - Automatically generates diagnostic plots when diagnostics=True
    - Uses isolation criterion to ensure seeds dominate their environment
    - All returned quantities are dimensionless and scaled by virial properties
    - HDF5 file structure: 'r', 'vr', 'lnv2' datasets
    - Diagnostic plot saved as 'calibration_data.png' in save_path

    Examples
    --------
    >>> # Generate calibration data from 500 massive isolated seeds
    >>> positions = numpy.random.uniform(0, 100, (50000, 3))
    >>> velocities = numpy.random.normal(0, 200, (50000, 3))
    >>> masses = numpy.random.lognormal(30, 1, 50000) 
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> seed_data = (positions, velocities, masses, radii)
    >>> 
    >>> r, vr, lnv2 = get_calibration_data(
    ...     n_seeds=500,
    ...     seed_data=seed_data,
    ...     r_max=4.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     save_path="/data/calibration/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=0,
    ...     n_threads=16,
    ...     diagnostics=True
    ... )
    >>> print(f"Calibration data: {len(r)} particles")
    >>> print(f"Radius range: [{r.min():.2f}, {r.max():.2f}]")

    >>> # Load existing calibration data (fast)
    >>> r, vr, lnv2 = get_calibration_data(
    ...     n_seeds=500,  # Parameters don't matter for loading
    ...     seed_data=seed_data,
    ...     r_max=4.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     save_path="/data/calibration/",  # Must contain calibration_data.hdf5
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=0,
    ...     diagnostics=False  # Skip plot generation
    ... )

    See Also
    --------
    _select_candidate_seeds : Core function for generating calibration data
    _diagnostic_calibration_data_plot : Creates diagnostic visualizations
    h5py.File : HDF5 file interface for data persistence
    """
    file_name = save_path + 'calibration_data.hdf5'
    _validate_inputs_positive_number(redshift, 'redshift')
    
    # Flag to trigger except clause given the overwrite flag.
    execute_fallback = overwrite

    if not execute_fallback:
        try:
            with h5py.File(file_name, 'r') as hdf:
                radius = hdf['r'][()]
                radial_velocity = hdf['vr'][()]
                log_velocity_squared = hdf['lnv2'][()]
        except:
            execute_fallback = True
    
    if execute_fallback:
        results = _select_candidate_seeds(
            n_seeds=n_seeds,
            seed_data=seed_data,
            r_max=r_max,
            boxsize=boxsize,
            minisize=minisize,
            save_path=save_path,
            particle_mass=particle_mass,
            mass_density=mass_density,
            isolation_factor=isolation_factor,
            isolation_radius_factor=isolation_radius_factor,
            n_threads=n_threads,
        )

        radius = results[:, 0]
        radial_velocity = results[:, 1]
        log_velocity_squared = results[:, 2]

        if redshift > 0.:
            a = 1 / (1 + redshift)
            log_velocity_squared += numpy.log(a**2)

        with h5py.File(file_name, 'w') as hdf:
            hdf.create_dataset('r', data=radius)
            hdf.create_dataset('vr', data=radial_velocity)
            hdf.create_dataset('lnv2', data=log_velocity_squared)

    if diagnostics:
        _diagnostic_calibration_data_plot(
            save_path=save_path,
            radius=radius,
            radial_velocity=radial_velocity,
            log_velocity_squared=log_velocity_squared
        )

    return radius, radial_velocity, log_velocity_squared


def _cost_percentile(abscissa: float, *data) -> float:
    """Compute cost for fitting line abscissa to achieve target percentile.

    This cost function optimizes the y-intercept (at a pivot radius) of a line
    with fixed slope such that a specified percentile of data points falls below
    the line. The cost is minimized when the fraction of points below the line
    matches the target percentile. 

    Parameters
    ----------
    abscissa : float
        Y-intercept of the line at the pivot radius (radius_pivot). In the
        context of velocity analysis, this represents ln(v²/v_200m²) at the
        pivot radius. Can be positive or negative.
    *data : tuple
        Variable-length argument tuple containing exactly 5 elements:
        - radius : numpy.ndarray with shape (n_particles,)
            Scaled radial distances (r/R200m) for all particles.
        - log_velocity_squared : numpy.ndarray with shape (n_particles,)
            Natural log of velocity squared scaled by V200² (ln(v²/v_200m²)).
        - slope : float
            Fixed slope of the line in (ln(v²), r/R200m) space. Can be
            positive, negative, or zero.
        - target : float
            Target percentile as a fraction in range [0, 1]. For example,
            0.9 means 90% of points should be below the line.
        - radius_pivot : float
            Pivot radius where the line has value abscissa. Typically chosen 
            at 0.5 x R200 for numerical stability.

    Returns
    -------
    cost : float
        Logarithmic cost function value. Returns ln((target - actual_fraction)²).
        Minimum occurs when actual fraction equals target. Always finite for
        target in (0, 1).

    Raises
    ------
    ValueError
        If data tuple doesn't contain exactly 5 elements, or if arrays have
        incompatible shapes.
    TypeError
        If inputs cannot be converted to appropriate numeric types.

    Notes
    -----
    - The line equation is: y = slope x (radius - radius_pivot) + abscissa
    - Cost function uses logarithm to handle small differences robustly
    - The function is designed for use with scipy.optimize.minimize_scalar
    - When target = 0.5, finds the median line through the data
    - Cost is symmetric around the optimal value

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import minimize_scalar
    >>> 
    >>> # Generate sample data
    >>> radius = np.random.uniform(0.1, 2.0, 1000)
    >>> log_v2 = 0.5 * radius + np.random.normal(0, 0.3, 1000)
    >>> 
    >>> # Find intercept for 90th percentile line
    >>> data = (radius, log_v2, 0.5, 0.9, 1.0)
    >>> result = minimize_scalar(_cost_percentile, args=data, method='brent')
    >>> optimal_b = result.x
    >>> print(f"Optimal intercept: {optimal_b:.3f}")

    >>> # Verify that ~90% of points are below the line
    >>> line = 0.5 * (radius - 1.0) + optimal_b
    >>> fraction_below = (log_v2 < line).sum() / len(radius)
    >>> print(f"Actual fraction below: {fraction_below:.2%}")

    See Also
    --------
    _cost_perpendicular_distance_abscissa : Alternative cost based on distances
    scipy.optimize.minimize_scalar : Optimization routine for this cost function
    """
    radius, log_velocity_squared, slope, target, radius_pivot = data
    line = slope * (radius - radius_pivot) + abscissa

    # Total number of elements below the line
    below_line = (log_velocity_squared < line).sum()

    # Fraction of elements below the line
    fraction = target - below_line / radius.shape[0]

    # Cost to minimize
    cost = numpy.log(fraction ** 2)
    return cost


def _cost_perpendicular_distance_abscissa(abscissa: float, *data) -> float:
    """Compute cost for fitting line abscissa to maximize perpendicular distances.

    This cost function optimizes the y-intercept (at a pivot radius) of a line
    with fixed slope to maximize the mean squared perpendicular distance of
    points within a specified band around the line. This helps find lines that
    pass through regions of high data density in phase space.

    Parameters
    ----------
    abscissa : float
        Y-intercept of the line at the pivot radius (radius_pivot). In the
        context of velocity analysis, this represents ln(v²/v_200m²) at the
        pivot radius. Can be positive or negative.
    *data : tuple
        Variable-length argument tuple containing exactly 5 elements:
        - radius : numpy.ndarray with shape (n_particles,)
            Scaled radial distances (r/R200m) for all particles.
        - log_velocity_squared : numpy.ndarray with shape (n_particles,)
            Natural log of velocity squared scaled by V200² (ln(v²/v_200m²)).
        - slope : float
            Fixed slope of the line in (ln(v²), r/R200m) space. Can be
            positive, negative, or zero.
        - width : float
            Half-width of band around the line in same units as
            log_velocity_squared. Only points within this perpendicular
            distance contribute to the cost. Must be positive.
        - radius_pivot : float
            Pivot radius where the line has value b. Typically chosen near
            the center of the data range for numerical stability.

    Returns
    -------
    cost : float
        Negative logarithm of mean squared perpendicular distance for points
        within the band. Returns -ln(mean(distance²)). More negative values
        indicate better fits (larger mean distances to maximize).

    Raises
    ------
    ValueError
        If data tuple doesn't contain exactly 5 elements, if arrays have
        incompatible shapes, or if no points fall within the band.
    TypeError
        If inputs cannot be converted to appropriate numeric types.

    Notes
    -----
    - Line equation: y = slope x (radius - radius_pivot) + abscissa
    - Perpendicular distance: d = |y - line| / √(1 + slope²)
    - Only points with d < width contribute to cost
    - Cost is negated because we maximize distance but minimize cost
    - Returns -∞ if no points are within the band (handled by optimizer)
    - For steep slopes (|slope| >> 1), perpendicular distance ≈ horizontal distance

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import minimize_scalar
    >>> 
    >>> # Generate data with a clear ridge
    >>> radius = np.random.uniform(0.1, 2.0, 2000)
    >>> log_v2 = 0.3 * radius + 0.5 + np.random.normal(0, 0.2, 2000)
    >>> 
    >>> # Find intercept that maximizes distances within band
    >>> data = (radius, log_v2, 0.3, 0.5, 1.0)
    >>> result = minimize_scalar(_cost_perpendicular_distance_abscissa, 
    ...                          args=data, method='brent')
    >>> optimal_b = result.x
    >>> print(f"Optimal intercept: {optimal_b:.3f}")

    >>> # Visualize the fitted line
    >>> import matplotlib.pyplot as plt
    >>> line = 0.3 * (radius - 1.0) + optimal_b
    >>> plt.scatter(radius, log_v2, alpha=0.3, s=1)
    >>> plt.plot(radius, line, 'r-', linewidth=2)
    >>> plt.xlabel('r/R200m')
    >>> plt.ylabel('ln(v²/v_200m²)')
    >>> plt.show()

    See Also
    --------
    _cost_perpendicular_distance_slope : Optimize slope with fixed intercept
    _cost_percentile : Alternative cost based on percentiles
    scipy.optimize.minimize_scalar : Optimization routine for this cost function
    """
    radius, log_velocity_squared, slope, width, radius_pivot = data
    line = slope * (radius - radius_pivot) + abscissa

    # Perpendicular distance to the line
    distance = numpy.abs(log_velocity_squared - line) / \
        numpy.sqrt(1 + slope**2)

    # Select only elements within the width of the band
    distance_within_band = distance[(distance < width)]

    # Cost to maximize (thus negative)
    cost = -numpy.log(numpy.mean(distance_within_band ** 2))
    return cost
    

def _cost_perpendicular_distance_slope(slope: float, *data) -> float:
    """Compute cost for fitting line slope to maximize perpendicular distances.

    This cost function optimizes the slope of a line with fixed y-intercept
    (at a pivot radius) to maximize the mean squared perpendicular distance
    of points within a specified band around the line. This helps find the
    optimal orientation of lines through high-density regions in phase space.

    Parameters
    ----------
    slope : float
        Slope of the line in (ln(v²), r/R200m) space. Represents the rate of
        change of ln(v²/v_200m²) with respect to r/R200m. Can be positive
        (increasing velocity with radius), negative (decreasing), or zero
        (constant velocity).
    *data : tuple
        Variable-length argument tuple containing exactly 5 elements:
        - radius : numpy.ndarray with shape (n_particles,)
            Scaled radial distances (r/R200m) for all particles.
        - log_velocity_squared : numpy.ndarray with shape (n_particles,)
            Natural log of velocity squared scaled by V200² (ln(v²/v_200m²)).
        - abscissa : float
            Fixed y-intercept of the line at the pivot radius (radius_pivot).
            In velocity analysis, represents ln(v²/v_200m²) at radius_pivot.
        - width : float
            Half-width of band around the line in same units as
            log_velocity_squared. Only points within this perpendicular
            distance contribute to the cost. Must be positive.
        - radius_pivot : float
            Pivot radius where the line has value b. Typically chosen near
            the center of the data range for numerical stability.

    Returns
    -------
    cost : float
        Negative logarithm of mean squared perpendicular distance for points
        within the band. Returns -ln(mean(distance²)). More negative values
        indicate better fits (larger mean distances to maximize).

    Raises
    ------
    ValueError
        If data tuple doesn't contain exactly 5 elements, if arrays have
        incompatible shapes, or if no points fall within the band.
    TypeError
        If inputs cannot be converted to appropriate numeric types.
    RuntimeWarning
        If the band contains very few points, results may be unstable.

    Notes
    -----
    - Line equation: y = slope x (radius - radius_pivot) + abscissa
    - Perpendicular distance: d = |y - line| / √(1 + slope²)
    - Only points with d < width contribute to cost
    - Cost is negated because we maximize distance but minimize cost
    - The denominator √(1 + slope²) ensures true perpendicular distance
    - For very steep slopes, optimization may become numerically unstable
    - Recommended to constrain slope to reasonable physical range

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import minimize_scalar
    >>> 
    >>> # Generate data along a sloped line with scatter
    >>> radius = np.random.uniform(0.1, 2.0, 2000)
    >>> true_slope = 0.4
    >>> log_v2 = true_slope * (radius - 1.0) + 0.3 + np.random.normal(0, 0.15, 2000)
    >>> 
    >>> # Find slope that maximizes distances within band
    >>> data = (radius, log_v2, 0.3, 0.4, 1.0)
    >>> result = minimize_scalar(_cost_perpendicular_distance_slope,
    ...                          args=data, method='brent',
    ...                          bounds=(-1.0, 1.0))
    >>> optimal_slope = result.x
    >>> print(f"True slope: {true_slope:.3f}")
    >>> print(f"Fitted slope: {optimal_slope:.3f}")

    >>> # Joint optimization of slope and intercept
    >>> # First optimize slope with initial intercept guess
    >>> data_slope = (radius, log_v2, 0.0, 0.4, 1.0)
    >>> result_slope = minimize_scalar(_cost_perpendicular_distance_slope,
    ...                                args=data_slope)
    >>> optimal_slope = result_slope.x
    >>> 
    >>> # Then optimize intercept with fitted slope
    >>> data_intercept = (radius, log_v2, optimal_slope, 0.4, 1.0)
    >>> result_intercept = minimize_scalar(_cost_perpendicular_distance_abscissa,
    ...                                    args=data_intercept)
    >>> optimal_b = result_intercept.x

    See Also
    --------
    _cost_perpendicular_distance_abscissa : Optimize intercept with fixed slope
    _cost_percentile : Alternative cost based on percentiles
    scipy.optimize.minimize : For joint optimization of slope and intercept
    """
    radius, log_velocity_squared, abscissa, width, radius_pivot = data
    line = slope * (radius - radius_pivot) + abscissa

    # Perpendicular distance to the line
    distance = numpy.abs(log_velocity_squared - line) / \
        numpy.sqrt(1 + slope**2)

    # Select only elements within the width of the band
    distance_within_band = distance[(distance < width)]

    # Cost to maximize (thus negative)
    cost = -numpy.log(numpy.mean(distance_within_band ** 2))
    return cost


def _diagnostic_gradient_minima_plot(
    save_path,
    log_velocity_squared_bins,
    counts,
    counts_gradient,
    counts_gradient_smooth,
    radial_bins,
    counts_gradient_minima,
    diagnostics_title,
) -> None:

    pyplot.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 150,
    })

    limits = ((-2, 2.5), (-1.1, 1.1))

    # Figure out the number of panels
    n_panels = log_velocity_squared_bins.shape[0]
    n_rows = int(numpy.floor(numpy.sqrt(n_panels)))
    n_cols = int(numpy.ceil(numpy.sqrt(n_panels)))

    fig, axes = pyplot.subplots(n_rows, n_cols, sharex=True, sharey=True,
                                figsize=(2.5*n_cols, 2.0*n_rows),
                                gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(diagnostics_title, fontsize=SIZE_LABELS)

    for i, ax in enumerate(axes.flatten()):
        ax.text(-1.8, 0.8, r'$r/R_{\rm 200m} = %.2f$' % radial_bins[i])
        if i % n_cols == 0:
            ax.set_ylabel('Counts (a.u.)', fontsize=SIZE_LABELS)
        if i >= n_panels - n_cols:
            ax.set_xlabel(r'$\ln(a^2v^2/v_{\rm 200m}^2)$', fontsize=SIZE_LABELS)
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])

        ax.plot(log_velocity_squared_bins[i, :], counts[i, :],
                color=COLOR_GRAY, lw=1.0, label='Counts')
        ax.plot(log_velocity_squared_bins[i, :], counts_gradient[i,
                :], color=COLOR_BLUE, lw=1.0, label='Gradient')
        ax.plot(log_velocity_squared_bins[i, :], counts_gradient_smooth[i,
                :], color=COLOR_RED, lw=1.0, label='Smoothed gradient')
        ax.axvline(
            counts_gradient_minima[i], color='k', lw=1.5, ls='--', label='Minimum')

        if i == 0:
            ax.legend()

    pyplot.tight_layout()
    plot_suffix = diagnostics_title.split()[0]
    pyplot.savefig(save_path + f'calibration_gradient_minima_{plot_suffix.lower()}.png',
                   dpi=300, bbox_inches='tight')

    return None


def _gradient_minima(
    radius: numpy.ndarray,
    log_velocity_squared: numpy.ndarray,
    n_points: int,
    r_min: float,
    r_max: float,
    save_path: str,
    sigma_smooth: float = 1.5,
    lnvsq_lims: tuple = (-2.0, 2.5),
    diagnostics: bool = True,
    diagnostics_title: str = None,
) -> Tuple[numpy.ndarray]:
    """Find gradient minima in velocity-radius phase space to identify boundaries.

    This function analyzes the velocity distribution as a function of radius by
    computing histograms in radial bins, then finding the minimum of the gradient
    of the velocity distribution in each bin. These minima correspond to boundaries
    or edges in the phase space distribution, useful for identifying distinct
    kinematic populations (e.g., splashback radius, infall region boundaries).

    The algorithm proceeds as follows for each radial bin:
    1. Histogram the log_velocity_squared values using automatic bin width
    2. Compute the gradient of the histogram counts
    3. Apply Gaussian smoothing to the gradient
    4. Find the minimum of the smoothed gradient for positive velocities
    5. This minimum represents a boundary in velocity space

    The number of velocity bins is automatically determined using Silverman's
    rule of thumb for optimal histogram bin width, based on the data within
    the specified radial range [r_min, r_max].

    Parameters
    ----------
    radius : numpy.ndarray
        Scaled radial distances (r/R200m) with shape (n_particles,).
        Values should typically be in range [0, 3]. Must be non-negative.
    log_velocity_squared : numpy.ndarray
        Natural log of velocity squared scaled by V200² with shape (n_particles,).
        Values ln(v²/v_200m²) typically range from -3 to +3.
    n_points : int
        Number of radial bins to compute gradient minima in. Must be positive
        and non-zero. Typical values are 10-50. More points give finer radial
        resolution but require more particles for stable statistics.
    r_min : float
        Minimum radius for analysis in units of R200m. Must be non-negative
        and less than r_max. Typical value is 0.0 or 0.2.
    r_max : float
        Maximum radius for analysis in units of R200m. Must be positive and
        greater than r_min. Typical value is 0.5 for calibration or 2.5 for
        broader analysis.
    save_path : str
        Directory path where diagnostic plots will be saved. Must be a valid
        directory path with write permissions. Plot saved as
        'gradient_minima_{diagnostics_title}.png' or 'gradient_minima.png'.
    sigma_smooth : float, optional
        Standard deviation for Gaussian smoothing kernel applied to gradient.
        Must be positive and non-zero. Default is 1.5. Larger values produce
        smoother gradients but may obscure fine structure. Typical range is
        [1.0, 3.0].
    lnvsq_lims : tuple, optional
        Lower and upper bounds (min, max) for log_velocity_squared histogram
        range, as a tuple of two floats. Default is (-2.0, 2.5). This range
        should encompass all relevant velocity data. Used with automatic bin
        width calculation to determine number of bins.
    diagnostics : bool, optional
        Whether to generate diagnostic plots showing histograms, gradients,
        and identified minima. Default is True.
    diagnostics_title : str, optional
        Additional identifier for diagnostic plot filename. If None, uses
        default filename 'gradient_minima.png'. If provided, saves as
        'gradient_minima_{diagnostics_title}.png'. Default is None.

    Returns
    -------
    radial_bins : numpy.ndarray
        Center positions of radial bins with shape (n_points,). Values are
        in units of R200m and span [r_min, r_max].
    counts_gradient_minima : numpy.ndarray
        Velocity values at gradient minima for each radial bin with shape
        (n_points,). Values are ln(v²/v_200m²) and identify boundaries in
        velocity space as a function of radius.

    Raises
    ------
    ValueError
        If array shapes are incompatible, if r_min >= r_max, if n_points is
        non-positive, if sigma_smooth is non-positive, or if
        radius/log_velocity_squared contain invalid values.
    TypeError
        If inputs cannot be converted to appropriate numeric types, or if
        lnvsq_lims is not a tuple/sequence.
    OSError
        If save_path doesn't exist or lacks write permissions when diagnostics=True.
    RuntimeWarning
        If radial bins contain very few particles, statistics may be unreliable.

    Notes
    -----
    **Automatic Bin Width Selection**:
    
    The number of velocity bins is automatically computed using Silverman's
    rule of thumb:
    
    h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
    
    where:
    - h is the optimal bin width
    - σ is the standard deviation of log_velocity_squared in [r_min, r_max]
    - IQR is the interquartile range (scaled to normal distribution)
    - n is the number of particles in [r_min, r_max]
    
    Number of bins = ceil((lnvsq_lims[1] - lnvsq_lims[0]) / h)
    
    This adaptive approach ensures appropriate resolution for different
    data distributions and sample sizes.
    
    **Algorithm Details**:
    
    - The gradient is normalized by its maximum absolute value in each bin
    - Only positive log_velocity_squared values are considered when finding minima
    - The smoothing kernel helps avoid spurious minima from statistical noise
    - Empty radial bins will have undefined gradient minima (may be NaN or zero)
    - The method is most reliable with >100 particles per radial bin
    - Diagnostic plots show: counts, gradient, smoothed gradient, and minima
    
    **Velocity Range Masking**:
    
    Before finding the minimum, the algorithm masks to only consider positive
    velocity values (log_velocity_squared > 0). This ensures the identified
    boundary corresponds to kinematically significant features rather than
    low-velocity artifacts.

    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Generate sample phase space data
    >>> n_particles = 10000
    >>> radius = np.random.uniform(0.1, 2.5, n_particles)
    >>> # Create bimodal velocity distribution
    >>> log_v2 = np.where(radius < 1.0,
    ...                   np.random.normal(0.5, 0.3, n_particles),
    ...                   np.random.normal(-0.2, 0.4, n_particles))
    >>> 
    >>> # Find gradient minima with default parameters
    >>> r_bins, v_minima = _gradient_minima(
    ...     radius=radius,
    ...     log_velocity_squared=log_v2,
    ...     n_points=20,
    ...     r_min=0.2,
    ...     r_max=0.5,
    ...     save_path="/output/",
    ...     diagnostics=True
    ... )
    >>> 
    >>> # Plot the boundary
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(radius, log_v2, alpha=0.1, s=1)
    >>> plt.plot(r_bins, v_minima, 'r-', linewidth=2, label='Boundary')
    >>> plt.xlabel('r/R200m')
    >>> plt.ylabel('ln(v²/v_200m²)')
    >>> plt.legend()
    >>> plt.show()

    >>> # Calibration for positive radial velocity
    >>> r_bins_pos, v_minima_pos = _gradient_minima(
    ...     radius=radius[radial_velocity > 0],
    ...     log_velocity_squared=log_v2[radial_velocity > 0],
    ...     n_points=20,
    ...     r_min=0.2,
    ...     r_max=0.5,
    ...     save_path="/output/calibration/",
    ...     sigma_smooth=1.5,
    ...     lnvsq_lims=(-2.0, 2.5),
    ...     diagnostics=True,
    ...     diagnostics_title='Positive vr slope calibration'
    ... )

    >>> # Calibration for negative radial velocity with custom parameters
    >>> r_bins_neg, v_minima_neg = _gradient_minima(
    ...     radius=radius[radial_velocity < 0],
    ...     log_velocity_squared=log_v2[radial_velocity < 0],
    ...     n_points=25,
    ...     r_min=0.2,
    ...     r_max=0.5,
    ...     save_path="/output/calibration/",
    ...     sigma_smooth=2.0,  # More smoothing
    ...     lnvsq_lims=(-3.0, 3.0),  # Wider velocity range
    ...     diagnostics=True,
    ...     diagnostics_title='Negative vr slope calibration'
    ... )

    >>> # Use for splashback radius identification
    >>> # Find where gradient minimum crosses zero velocity
    >>> zero_crossings = np.where(np.diff(np.sign(v_minima)))[0]
    >>> if len(zero_crossings) > 0:
    ...     r_splash = r_bins[zero_crossings[0]]
    ...     print(f"Splashback radius: {r_splash:.2f} R200m")

    >>> # Compare different smoothing scales
    >>> fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    >>> for ax, sigma in zip(axes, [1.0, 1.5, 2.5]):
    ...     r_b, v_m = _gradient_minima(
    ...         radius=radius,
    ...         log_velocity_squared=log_v2,
    ...         n_points=20,
    ...         r_min=0.2,
    ...         r_max=0.5,
    ...         save_path="/output/",
    ...         sigma_smooth=sigma,
    ...         diagnostics=False
    ...     )
    ...     ax.scatter(radius, log_v2, alpha=0.1, s=1)
    ...     ax.plot(r_b, v_m, 'r-', linewidth=2)
    ...     ax.set_title(f'σ = {sigma}')
    ...     ax.set_xlabel('r/R200m')
    ...     ax.set_ylabel('ln(v²/v_200m²)')
    >>> plt.tight_layout()
    >>> plt.show()

    See Also
    --------
    _diagnostic_gradient_minima_plot : Creates visualization of results
    scipy.ndimage.gaussian_filter : Gaussian smoothing implementation
    scipy.stats.iqr : Interquartile range computation
    numpy.gradient : Gradient computation
    self_calibration : Uses this function for phase space calibration
    """
    # Validate inputs
    _validate_inputs_positive_number_non_zero(n_points, 'n_points')
    _validate_inputs_positive_number(r_min, 'r_min')
    _validate_inputs_positive_number_non_zero(sigma_smooth, 'sigma_smooth')

    if r_min >= r_max:
        raise ValueError('r_max must be larger than r_min')

    mask = (radius >= r_min) & (radius <= r_max)
    iqr = interquartile_range(log_velocity_squared[mask], scale='normal')
    # Silvermann's rule of thumb.
    h = 0.9 * numpy.min([numpy.std(log_velocity_squared[mask]), iqr]) * \
        mask.sum()**(-0.2)
    n_bins = int(numpy.ceil((lnvsq_lims[1] - lnvsq_lims[0]) / h))

    # Compute radial bins and initialize empty container for gradient minima
    radius_edges = numpy.linspace(r_min, r_max, n_points + 1)
    counts_gradient_minima = numpy.zeros(n_points)

    # Initialize empty containers for diagnostic plot even if diagnostics = False
    log_velocity_squared_bins_out = numpy.zeros((n_points, n_bins))
    counts_out = numpy.zeros((n_points, n_bins))
    counts_gradient_out = numpy.zeros((n_points, n_bins))
    counts_gradient_smooth_out = numpy.zeros((n_points, n_bins))

    for i in range(n_points):
        # Create mask for current r bin
        radius_mask = (radius > radius_edges[i]) * \
            (radius < radius_edges[i + 1])

        # Compute histogram of lnv2 values within the r bin and the vr mask
        counts, log_velocity_squared_edges = numpy.histogram(
            log_velocity_squared[radius_mask], bins=numpy.linspace(*lnvsq_lims, n_bins+1))

        # Compute the gradient of the histogram
        counts_gradient = numpy.gradient(
            counts, numpy.mean(numpy.diff(log_velocity_squared_edges)))
        counts_gradient /= numpy.max(numpy.abs(counts_gradient))

        # Smooth the gradient
        counts_gradient_smooth = gaussian_filter(
            counts_gradient, sigma_smooth)

        # Find the lnv2 value corresponding to the minimum of the smoothed gradient
        log_velocity_squared_bins = 0.5 * (log_velocity_squared_edges[:-1] +
                                           log_velocity_squared_edges[1:])
        
        # This is should be the choice but it is not giving the same approximate 
        # result as before.
        mask_log_velocity_squared = (0. < log_velocity_squared_bins)
        counts_gradient_minima[i] = log_velocity_squared_bins[mask_log_velocity_squared][numpy.argmin(
            counts_gradient_smooth[mask_log_velocity_squared])]

        # counts_gradient_minima[i] = log_velocity_squared_bins[mask_log_velocity_squared][numpy.argmin(
        #     counts_gradient[mask_log_velocity_squared])]

        # Store diagnostics
        log_velocity_squared_bins_out[i, :] = log_velocity_squared_bins
        counts_out[i, :] = counts / numpy.max(counts)
        counts_gradient_out[i, :] = counts_gradient
        counts_gradient_smooth_out[i, :] = counts_gradient_smooth

    # Compute r bin centres
    radial_bins = 0.5 * (radius_edges[:-1] + radius_edges[1:])

    # Return diagnostics if requested
    if diagnostics:
        _diagnostic_gradient_minima_plot(
            save_path,
            log_velocity_squared_bins_out,
            counts_out,
            counts_gradient_out,
            counts_gradient_smooth_out,
            radial_bins,
            counts_gradient_minima,
            diagnostics_title,
        )
    return radial_bins, counts_gradient_minima


def _hist2d_mesh(
    x: numpy.ndarray,
    y: numpy.ndarray,
    limits: tuple,
    n_bins: int,
    gradient: bool = False,
) -> Tuple[numpy.ndarray]:
    """Compute 2D histogram with optional gradient for phase space analysis.

    This function creates a 2D histogram of (x, y) data and optionally computes
    the gradient of the histogram density with respect to y in each x bin.
    Returns meshgrid arrays suitable for contour plots and gradient analysis.
    Useful for analyzing phase space distributions and identifying boundaries.

    Parameters
    ----------
    x : numpy.ndarray
        X-coordinates of data points with shape (n_points,). Typically
        represents radial distance (r/R200m) in halo analysis.
    y : numpy.ndarray
        Y-coordinates of data points with shape (n_points,). Typically
        represents velocity quantities (ln(v²/v_200m²)) in halo analysis.
    limits : tuple
        2D histogram limits as ((x_min, x_max), (y_min, y_max)). Must be
        a tuple of two tuples, each containing two floats. Defines the
        rectangular region for histogram computation.
    n_bins : int
        Number of bins in each dimension. Must be positive. The total number
        of bins in the 2D histogram is n_bins². Typical values are 50-200.
    gradient : bool, optional
        Whether to compute gradient of density with respect to y in each x bin.
        If True, returns gradient array instead of density. Default is False.

    Returns
    -------
    x_mesh : numpy.ndarray
        X-coordinates of bin centers with shape (n_bins, n_bins). Created by
        numpy.meshgrid for use in contour/surface plots.
    y_mesh : numpy.ndarray
        Y-coordinates of bin centers with shape (n_bins, n_bins). Created by
        numpy.meshgrid for use in contour/surface plots.
    z : numpy.ndarray
        If gradient=False: 2D histogram density with shape (n_bins, n_bins).
            Values are normalized probability densities.
        If gradient=True: Gradient of density with respect to y with shape
            (n_bins, n_bins). Values are ∂ρ/∂y for each x bin.

    Raises
    ------
    ValueError
        If x and y have different shapes, if limits tuple has wrong structure,
        or if n_bins is non-positive.
    TypeError
        If inputs cannot be converted to appropriate numeric types.

    Notes
    -----
    - Histogram is computed using numpy.histogram2d with density=True
    - Bin centers are computed as midpoints between edges
    - Gradient is computed using numpy.gradient along y-axis for each x bin
    - Empty bins will have zero density/gradient
    - Meshgrids follow numpy convention: first index is x, second is y
    - The gradient computation uses mean bin width for numerical derivative

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Generate sample phase space data
    >>> n_points = 5000
    >>> x = np.random.uniform(0, 2, n_points)
    >>> y = 0.5 * x + np.random.normal(0, 0.3, n_points)
    >>> 
    >>> # Compute 2D histogram
    >>> limits = ((0, 2), (-1, 2))
    >>> x_mesh, y_mesh, density = _hist2d_mesh(x, y, limits, n_bins=100)
    >>> 
    >>> # Plot density
    >>> plt.contourf(x_mesh, y_mesh, density.T, levels=20)
    >>> plt.colorbar(label='Density')
    >>> plt.xlabel('r/R200m')
    >>> plt.ylabel('ln(v²/v_200m²)')
    >>> plt.show()

    >>> # Compute and plot gradient
    >>> x_mesh, y_mesh, grad = _hist2d_mesh(x, y, limits, n_bins=100, 
    ...                                     gradient=True)
    >>> plt.contourf(x_mesh, y_mesh, grad.T, levels=20, cmap='RdBu_r')
    >>> plt.colorbar(label='∂ρ/∂y')
    >>> plt.xlabel('r/R200m')
    >>> plt.ylabel('ln(v²/v_200m²)')
    >>> plt.show()

    >>> # Find ridges in phase space (where gradient ≈ 0)
    >>> ridge_mask = np.abs(grad) < 0.1
    >>> plt.scatter(x_mesh[ridge_mask], y_mesh[ridge_mask], c='red', s=1)

    See Also
    --------
    numpy.histogram2d : Underlying 2D histogram computation
    numpy.gradient : Gradient computation method
    numpy.meshgrid : Meshgrid creation for plotting
    _gradient_minima : Alternative method for finding boundaries
    """
    
    hist_z, hist_x, hist_y = numpy.histogram2d(x, y, bins=n_bins, range=limits,
                                               density=True)

    # Bin centres
    x = 0.5 * (hist_x[:-1] + hist_x[1:])
    y = 0.5 * (hist_y[:-1] + hist_y[1:])

    # Bin widths
    dy = numpy.mean(numpy.diff(y))

    # Meshgrids
    x_mesh, y_mesh = numpy.meshgrid(x, y)

    # Compute gradient of z as a function of y for each x bin
    if gradient:
        z_grad = numpy.zeros_like(hist_z)
        for i in range(n_bins):
            z_grad[i, :] = numpy.gradient(hist_z[i, :], dy)
        return x_mesh, y_mesh, z_grad
    else:
        return x_mesh, y_mesh, hist_z


def _smooth_2d_hist(arr: numpy.ndarray, sigma: float = 2.0) -> numpy.ndarray:
    """Apply Gaussian smoothing to 2D histogram for noise reduction.

    This function smooths a 2D array (typically a histogram or density map) using
    a Gaussian filter. Smoothing reduces statistical noise and makes underlying
    structures more visible, which is useful for feature detection and boundary
    identification in phase space distributions.

    Parameters
    ----------
    arr : numpy.ndarray
        2D array to smooth, typically with shape (n_bins, n_bins). Usually
        represents a histogram density or gradient field. Can contain any
        numeric values including zeros and NaNs (which will be smoothed).
    sigma : float, optional
        Standard deviation of the Gaussian kernel in bin units. Must be
        non-negative. Default is 2.0. Larger values produce stronger smoothing:
        - sigma < 1: Minimal smoothing, preserves fine structure
        - sigma = 1-3: Moderate smoothing, good for most applications
        - sigma > 5: Strong smoothing, may obscure real features

    Returns
    -------
    arr_smooth : numpy.ndarray
        Smoothed 2D array with same shape as input. Values are weighted averages
        of neighboring bins using Gaussian kernel. Edge effects are handled by
        scipy's boundary conditions (typically reflection).

    Raises
    ------
    ValueError
        If arr is not 2-dimensional or if sigma is negative.
    TypeError
        If arr cannot be converted to numpy array or sigma is not numeric.

    Notes
    -----
    - Uses scipy.ndimage.gaussian_filter for efficient computation
    - The Gaussian kernel has standard deviation sigma in array index units
    - Effective smoothing radius is approximately 3×sigma bins
    - Smoothing is applied equally in both dimensions
    - Edge effects are handled by reflecting values across boundaries
    - NaN values are propagated through smoothing (consider masking first)

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Create noisy 2D histogram
    >>> np.random.seed(42)
    >>> x = np.random.uniform(0, 2, 5000)
    >>> y = 0.3 * x + np.random.normal(0, 0.2, 5000)
    >>> hist, x_edges, y_edges = np.histogram2d(x, y, bins=50, 
    ...                                         range=((0, 2), (-1, 1)))
    >>> 
    >>> # Add noise to simulate low statistics
    >>> hist_noisy = hist + np.random.poisson(2, hist.shape)
    >>> 
    >>> # Apply smoothing
    >>> hist_smooth = _smooth_2d_hist(hist_noisy, sigma=2.0)
    >>> 
    >>> # Compare original and smoothed
    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> im1 = axes[0].imshow(hist_noisy.T, origin='lower', aspect='auto')
    >>> axes[0].set_title('Noisy histogram')
    >>> plt.colorbar(im1, ax=axes[0])
    >>> 
    >>> im2 = axes[1].imshow(hist_smooth.T, origin='lower', aspect='auto')
    >>> axes[1].set_title('Smoothed histogram')
    >>> plt.colorbar(im2, ax=axes[1])
    >>> plt.show()

    >>> # Effect of different sigma values
    >>> sigmas = [0.5, 1.0, 2.0, 5.0]
    >>> fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    >>> for ax, sigma in zip(axes, sigmas):
    ...     smoothed = _smooth_2d_hist(hist_noisy, sigma=sigma)
    ...     ax.imshow(smoothed.T, origin='lower', aspect='auto')
    ...     ax.set_title(f'σ = {sigma}')
    >>> plt.tight_layout()
    >>> plt.show()

    >>> # Smoothing gradient fields
    >>> # Compute gradient of density
    >>> grad_y = np.gradient(hist, axis=1)
    >>> grad_y_smooth = _smooth_2d_hist(grad_y, sigma=1.5)
    >>> 
    >>> # Smoothed gradients are better for finding features
    >>> plt.imshow(grad_y_smooth.T, origin='lower', aspect='auto', cmap='RdBu_r')
    >>> plt.colorbar(label='Smoothed ∂ρ/∂y')
    >>> plt.title('Smoothed density gradient')
    >>> plt.show()

    See Also
    --------
    scipy.ndimage.gaussian_filter : Underlying smoothing implementation
    _hist2d_mesh : Creates 2D histograms for smoothing
    scipy.ndimage.uniform_filter : Alternative box-car smoothing
    scipy.signal.wiener : Adaptive Wiener filtering for noise reduction
    """
    arr_smooth = gaussian_filter(arr, sigma=sigma)
    return arr_smooth


def _diagnostic_self_calibration_plot(
    save_path: str,
    radius: numpy.ndarray,
    radial_velocity: numpy.ndarray,
    log_velocity_squared: numpy.ndarray,
    parameters: Tuple[float],
    gradient_values: Tuple[numpy.ndarray],
) -> None:

    pyplot.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 150,
    })

    radius_pivot, slope_positive_vr, abscissa_positive_vr, slope_negative_vr, \
        abscissa_negative_vr, alpha, beta, gamma = parameters

    # Compute cut lines
    x = numpy.linspace(0, 2, 1000)
    line_positive_vr = slope_positive_vr * \
        (x - radius_pivot) + abscissa_positive_vr

    x_low = numpy.linspace(0, radius_pivot, 1000)
    x_high = numpy.linspace(radius_pivot, 2, 1000)
    line_negative_vr = slope_negative_vr * \
        (x - radius_pivot) + abscissa_negative_vr
    curve_low = alpha*x_low**2 + beta*x_low + gamma
    curve_high = slope_negative_vr * \
        (x_high - radius_pivot) + abscissa_negative_vr

    label_positive_vr = fr"${slope_positive_vr:.3f}(x-{radius_pivot:.2f}){abscissa_positive_vr:+.3f}$"
    label_negative_vr = fr"${slope_negative_vr:.3f}(x-{radius_pivot:.2f}){abscissa_negative_vr:+.3f}$"

    cmap = 'terrain'
    limits = ((0, 2), (-2, 2.5))
    levels = 80
    mask_negative_vr = (radial_velocity < 0)
    mask_positive_vr = ~mask_negative_vr

    radial_bins_positive, gradient_minumum_positive, radial_bins_negative, \
        gradient_minumum_negative = gradient_values

    fig, axes = pyplot.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Calibration data', fontsize=SIZE_LABELS)
    axes = axes.flatten()

    for ax in axes:
        ax.set_xlabel(r'$r/R_{\rm 200m}$', fontsize=SIZE_LABELS)
        ax.set_ylabel(r'$\ln(a^2v^2/v_{\rm 200m}^2)$', fontsize=SIZE_LABELS)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)

    # Create a color bar to display mass ranges.
    cax = fig.add_axes([1.01, 0.2, 0.01, 0.7])
    cbar = ColorbarBase(cax, cmap=cmap, orientation="vertical", extend='max')
    cbar.set_label(r'Counts (a.u.)', fontsize=SIZE_LEGEND)
    cbar.set_ticklabels([], fontsize=SIZE_TICKS)
    cbar.ax.tick_params(size=0, labelleft=False, labelright=False,
                        labeltop=False, labelbottom=False)

    # Compute counts histograms for top two panels
    x_mesh_positive, y_mesh_positive, z_positive = \
        _hist2d_mesh(radius[mask_positive_vr],
                     log_velocity_squared[mask_positive_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=False,
                     )
    z_positive = _smooth_2d_hist(z_positive)

    x_mesh_negative, y_mesh_negative, z_negative = \
        _hist2d_mesh(radius[mask_negative_vr],
                     log_velocity_squared[mask_negative_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=False,
                     )
    z_negative = _smooth_2d_hist(z_negative)

    # Top-left
    pyplot.sca(axes[0])
    pyplot.title(r'$v_r > 0$', fontsize=SIZE_LABELS)
    pyplot.contourf(x_mesh_positive, y_mesh_positive, z_positive.T,
                    levels=levels, cmap=cmap)
    pyplot.plot(x, line_positive_vr, lw=2.0,
                color="r", label=label_positive_vr)
    pyplot.legend(loc='lower left', fontsize=SIZE_LEGEND)

    # Top-right
    pyplot.sca(axes[1])
    pyplot.title(r'$v_r < 0$', fontsize=SIZE_LABELS)
    pyplot.contourf(x_mesh_negative, y_mesh_negative, z_negative.T,
                    levels=levels, cmap=cmap)
    pyplot.plot(x, line_negative_vr, lw=2.0,
                color="k", label=label_negative_vr)
    pyplot.plot(x_low, curve_low, lw=2.0, color='r', ls='--')
    pyplot.plot(x_high, curve_high, lw=2.0, color='r',
                ls='--', label='Low radius correction')
    pyplot.legend(loc='lower left', fontsize=SIZE_LEGEND)

    # Compute counts gradient for lower two panels
    x_mesh_positive, y_mesh_positive, z_grad_positive = \
        _hist2d_mesh(radius[mask_positive_vr],
                     log_velocity_squared[mask_positive_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=True,
                     )
    z_grad_positive = _smooth_2d_hist(z_grad_positive)

    x_mesh_negative, y_mesh_negative, z_grad_negative = \
        _hist2d_mesh(radius[mask_negative_vr],
                     log_velocity_squared[mask_negative_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=True,
                     )
    z_grad_negative = _smooth_2d_hist(z_grad_negative)

    # Bottom-left
    pyplot.sca(axes[2])
    pyplot.contourf(x_mesh_positive, y_mesh_positive, z_grad_positive.T,
                    levels=levels, cmap=cmap)
    pyplot.plot(radial_bins_positive,
                gradient_minumum_positive, lw=1.0, color="r")
    pyplot.plot(x, line_positive_vr, lw=2.0, color="r", label='Cut line')

    # Bottom-right
    pyplot.sca(axes[3])
    pyplot.contourf(x_mesh_negative, y_mesh_negative, z_grad_negative.T,
                    levels=levels, cmap=cmap)
    pyplot.plot(radial_bins_negative,
                gradient_minumum_negative, lw=1.0, color="r")
    pyplot.plot(x, line_negative_vr, lw=2.0, color="k", label='Cut line')
    pyplot.plot(x_low, curve_low, lw=2.0, color='r', ls='--')
    pyplot.plot(x_high, curve_high, lw=2.0, color='r',
                ls='--', label='Low radius correction')

    pyplot.tight_layout()
    pyplot.savefig(save_path + 'calibration_classification_lines.png', dpi=300,
                   bbox_inches='tight')

    return None


def self_calibration(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_mass: float,
    mass_density: float,
    redshift: float,
    n_points: int = 20,
    percent: float = 0.995,
    width: float = 0.05,
    gradient_radial_lims: Tuple[float] = (0.2, 0.5),
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
    n_threads: int = None,
    diagnostics: bool = True,
    overwrite: bool = False,
) -> None:
    """Runs calibration from isolated halo samples.

    This function calibrates boundary lines in velocity-radius phase space by
    analyzing particle distributions around isolated massive halos. It separately
    calibrates boundaries for particles with positive (outflowing) and negative
    (inflowing) radial velocities using different optimization strategies:
    
    - **Positive radial velocity**: Finds a percentile-based upper boundary line
      that contains a specified fraction of particles.
    - **Negative radial velocity**: Finds a boundary line that maximizes
      perpendicular distances within a specified band width, then applies
      a low-radius quadratic correction.
    
    The calibration parameters are saved to 'calibration_pars.hdf5' for use
    in halo finding algorithms.

    Parameters
    ----------
    n_seeds : int
        Number of isolated massive seeds to use for calibration. Must be positive.
        Typical values are 100-500. More seeds improve statistics but increase
        computation time. Only the most massive isolated halos are selected.
    seed_data : Tuple[numpy.ndarray]
        Tuple containing (position, velocity, mass, radius) arrays:
        - position: shape (n_total_seeds, 3) - seed coordinates in simulation units
        - velocity: shape (n_total_seeds, 3) - seed velocities in simulation units
        - mass: shape (n_total_seeds,) - seed masses M200 in simulation units
        - radius: shape (n_total_seeds,) - seed virial radii R200 in simulation units
    r_max : float
        Maximum distance from seed centers to consider particles, in 
        simulation units. Must be positive. Typical values are 3-5 * R200.
    boxsize : float
        Size of the cubic simulation box in simulation units. Must be positive.
    minisize : float
        Size of each cubic minibox in simulation units. Must be positive
        and smaller than boxsize.
    save_path : str
        Directory path for reading minibox data and writing calibration results.
        Must be a valid directory path with read/write permissions.
        Calibration parameters saved as 'calibration_pars.hdf5'.
    particle_mass : float
        Mass of each simulation particle in simulation units (typically M_sun).
        Must be positive.
    mass_density : float
        Mean matter density of the universe at the simulation redshift in 
        simulation units (typically M_sun/Mpc³). Must be positive.
    redshift : float
        Redshift of the simulation snapshot. Must be non-negative.
        Used to scale the band width for negative radial velocity calibration.
        Higher redshifts use wider bands to handle increased numerical noise.
    n_points : int, optional
        Number of radial bins for gradient minima computation. Must be positive.
        Default is 20. More points give finer resolution but require more particles.
    percent : float, optional
        Target percentile for positive radial velocity boundary as a fraction
        in range (0, 1). Default is 0.995 (99.5%). This fraction of positive-vr
        particles will fall below the calibrated boundary line.
    width : float, optional
        Base band width for negative radial velocity calibration in units of
        ln(v²/v_200m²). Must be positive. Default is 0.05. This is scaled by
        (1 + z/3) to handle redshift dependence. Smaller values fit tighter
        to the data but may be noisier.
    gradient_radial_lims : Tuple[float], optional
        Radial interval (r_min, r_max) in units of R200m for computing gradient
        minima. Default is (0.2, 0.5). This range should capture the linear
        regime of the phase space boundaries.
    isolation_factor : float, optional
        Maximum allowed mass ratio for isolation criterion. Neighbors must have
        mass < isolation_factor * seed_mass. Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200 to define isolation radius for neighbor search.
        Default is 2.0.
    n_threads : int, optional
        Number of threads for parallel processing. If None, uses half of
        available CPU cores. Only used when generating calibration data.
    diagnostics : bool, optional
        Whether to generate diagnostic plots of calibration process and results.
        Default is True. Plots saved to save_path.
    overwrite : bool, optional
        Whether to overwrite the existing calibration data. Default is False.

    Returns
    -------
    None
        Function saves calibration parameters to HDF5 file but returns nothing.

    Raises
    ------
    TypeError
        If n_seeds is not an integer, or if array inputs in seed_data cannot
        be converted to numpy arrays.
    ValueError
        If n_seeds is non-positive, if array shapes in seed_data are incompatible,
        if any scalar parameters are non-positive, if percent is not in (0, 1),
        or if gradient_radial_lims is invalid.
    FileNotFoundError
        If save_path doesn't exist or if required minibox files are missing.
    OSError
        If HDF5 files cannot be read/written, or if directory permissions
        are insufficient.
    RuntimeError
        If optimization fails to converge or if insufficient data for calibration.

    Notes
    -----
    **Algorithm Overview**:
    
    1. Load calibration data from isolated massive halos
    2. Separate particles by radial velocity sign
    3. For positive radial velocity:
       - Compute gradient minima to find initial boundary slope
       - Fit linear model to minima (with outlier rejection)
       - Optimize intercept to achieve target percentile
    4. For negative radial velocity:
       - Compute gradient minima similarly
       - Optimize intercept to maximize perpendicular distances
       - Refine slope with fixed intercept
       - Compute low-radius quadratic correction
    5. Save all parameters to HDF5 file
    
    **Calibration Parameters Saved**:
    
    - 'pos': [slope, intercept] for positive vr boundary (linear)
    - 'neg/line': [slope, intercept] for negative vr boundary (linear at large r)
    - 'neg/quad': [alpha, beta, gamma] for negative vr correction (quadratic at small r)
    
    The negative vr boundary uses a piecewise model:
    - Linear: y = slope * (r - 0.5) + intercept for r > r_transition
    - Quadratic: y = alpha * r² + beta * r + gamma for r ≤ r_transition
    
    **Outlier Rejection**:
    
    Uses median absolute deviation (MAD) with scale factor 1.4826 to identify
    and reject outliers in gradient minima before fitting. Points beyond 3*MAD
    from the median are excluded.
    
    **Redshift Scaling**:
    
    The band width for negative vr calibration is scaled as width * (1 + z/3)
    to handle increased numerical noise at higher redshifts. Calibrated up to z~3.

    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Prepare seed catalog from halo finder
    >>> positions = np.random.uniform(0, 100, (1000, 3))
    >>> velocities = np.random.normal(0, 200, (1000, 3))
    >>> masses = np.random.lognormal(30, 1, 1000)
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> seed_data = (positions, velocities, masses, radii)
    >>> 
    >>> # Run self-calibration
    >>> self_calibration(
    ...     n_seeds=200,
    ...     seed_data=seed_data,
    ...     r_max=4.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     save_path="/data/calibration/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=0.0,
    ...     percent=0.995,
    ...     diagnostics=True
    ... )
    >>> 
    >>> # Load calibrated parameters
    >>> import h5py
    >>> with h5py.File("/data/calibration/calibration_pars.hdf5", 'r') as f:
    ...     slope_pos, b_pos = f['pos'][()]
    ...     slope_neg, b_neg = f['neg/line'][()]
    ...     alpha, beta, gamma = f['neg/quad'][()]
    >>> print(f"Positive vr: slope={slope_pos:.3f}, intercept={b_pos:.3f}")
    >>> print(f"Negative vr: slope={slope_neg:.3f}, intercept={b_neg:.3f}")

    >>> # Calibration at higher redshift
    >>> self_calibration(
    ...     n_seeds=300,
    ...     seed_data=seed_data_z2,
    ...     r_max=4.0,
    ...     boxsize=200.0,
    ...     minisize=10.0,
    ...     save_path="/data/calibration_z2/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=2.0,  # Higher redshift
    ...     width=0.05,     # Will be scaled to 0.05 * (1 + 2/3) ≈ 0.083
    ...     n_threads=16,
    ...     diagnostics=True
    ... )

    See Also
    --------
    calibrate : High-level interface with cosmology-based calibration option
    get_calibration_data : Loads particle data for calibration
    _gradient_minima : Finds boundaries from gradient analysis
    _cost_percentile : Cost function for percentile-based fitting
    _cost_perpendicular_distance_abscissa : Cost function for distance-based fitting
    """
    radius, radial_velocity, log_velocity_squared = get_calibration_data(
        n_seeds=n_seeds,
        seed_data=seed_data,
        r_max=r_max,
        boxsize=boxsize,
        minisize=minisize,
        save_path=save_path,
        particle_mass=particle_mass,
        mass_density=mass_density,
        redshift=redshift,
        n_threads=n_threads,
        isolation_factor=isolation_factor,
        isolation_radius_factor=isolation_radius_factor,
        diagnostics=diagnostics,
        overwrite=overwrite,
    )

    mask_negative_vr = (radial_velocity < 0)
    mask_positive_vr = ~mask_negative_vr
    mask_radius = radius <= 2.0
    mask_low_radius = radius <= 1.0
    radius_pivot = 0.5
    def line_model(x, slope, abscissa): return slope * \
        (x - radius_pivot) + abscissa

    # ==========================================================================
    # 
    #                           Positive radial velocity
    # 
    # ==========================================================================
    radial_bins_positive, gradient_minumum_positive = _gradient_minima(
        radius=radius[mask_positive_vr],
        log_velocity_squared=log_velocity_squared[mask_positive_vr],
        n_points=n_points,
        r_min=gradient_radial_lims[0],
        r_max=gradient_radial_lims[1],
        save_path=save_path,
        diagnostics=diagnostics,
        diagnostics_title=r'Positive $v_r$ slope calibration',
    )

    # Mask outliers
    abs_dev = numpy.abs(gradient_minumum_positive - numpy.median(gradient_minumum_positive))
    median_abs_dev = 1.4286 * numpy.median(abs_dev)
    mask_outliers = abs_dev <= 3. * median_abs_dev
    
    # Find slope by fitting to the minima.
    (slope_positive_vr, abscissa_p), _ = curve_fit(line_model, 
                                                   radial_bins_positive[mask_outliers],
                                                   gradient_minumum_positive[mask_outliers],
                                                   p0=[-1, 2], bounds=((-5, 0), (0, 5)))

    # Find intercept by finding the value that contains 'perc' percent of
    # particles below the line at fixed slope 'm_pos'.
    result = minimize(
        fun=_cost_percentile,
        x0=1.1 * abscissa_p,
        bounds=((abscissa_p, 5.0),),
        args=(radius[mask_positive_vr & mask_radius],
              log_velocity_squared[mask_positive_vr & mask_radius],
              slope_positive_vr, percent, radius_pivot),
        method='Nelder-Mead',
    )
    abscissa_positive_vr = result.x[0]

    # ==========================================================================
    # 
    #                           Negative radial velocity
    # 
    # ==========================================================================
    radial_bins_negative, gradient_minumum_negative = _gradient_minima(
        radius=radius[mask_negative_vr],
        log_velocity_squared=log_velocity_squared[mask_negative_vr],
        n_points=n_points,
        r_min=gradient_radial_lims[0],
        r_max=gradient_radial_lims[1],
        save_path=save_path,
        diagnostics=diagnostics,
        diagnostics_title=r'Negative $v_r$ slope calibration',
    )

    # Mask outliers
    abs_dev = numpy.abs(gradient_minumum_negative - numpy.median(gradient_minumum_negative))
    median_abs_dev = 1.4286 * numpy.median(abs_dev)
    mask_outliers = abs_dev <= 3. * median_abs_dev

    # Find slope by fitting to the minima.
    (slope_n, abscissa_n), _ = curve_fit(line_model, 
                                         radial_bins_negative[mask_outliers],
                                         gradient_minumum_negative[mask_outliers],
                                         p0=[-1, 2], bounds=((-5, 0), (0, 3)))
    
    # The user input width is scaled with redshift such that it is double at 
    # z = 3. This was needed due to the numerical noise in the cost function 
    # when using small band widths. Has not been tested beyond this redshift.
    width *= (1. + redshift/3.)

    # Find intercept by finding the value that maximizes the perpendicular
    # distance to the line at fixed slope of all points within a perpendicular
    # 'width' distance from the line (ignoring all others).
    result = minimize(
        fun=_cost_perpendicular_distance_abscissa,
        x0=0.5 * abscissa_n,
        bounds=((0., abscissa_n),),
        args=(radius[mask_negative_vr & mask_low_radius], 
              log_velocity_squared[mask_negative_vr & mask_low_radius],
              slope_n, width, radius_pivot),
        method='Nelder-Mead',
    )
    abscissa_negative_vr = result.x[0]

    # Refine the slope by fitting with fixed abscissa.
    result = minimize(
        fun=_cost_perpendicular_distance_slope,
        x0=slope_n,
        bounds=((1.5*slope_n, 0.5*slope_n),),
        args=(radius[mask_negative_vr & mask_low_radius], 
              log_velocity_squared[mask_negative_vr & mask_low_radius],
              abscissa_negative_vr, width, radius_pivot),
        method='Nelder-Mead',
    )
    slope_negative_vr = result.x[0]

    # Compute low radius correction parameters
    b_neg = abscissa_negative_vr - slope_negative_vr * radius_pivot
    gamma = 2.
    alpha = (gamma - b_neg) / radius_pivot**2
    beta = slope_negative_vr - 2 * alpha * radius_pivot

    # Save to file
    with h5py.File(save_path + 'calibration_pars.hdf5', 'w') as hdf:
        hdf.create_dataset(
            'pos', data=[slope_positive_vr, abscissa_positive_vr])
        hdf.create_dataset(
            'neg/line', data=[slope_negative_vr, abscissa_negative_vr])
        hdf.create_dataset('neg/quad', data=[alpha, beta, gamma])

    if diagnostics:
        parameters = (
            radius_pivot,
            slope_positive_vr,
            abscissa_positive_vr,
            slope_negative_vr,
            abscissa_negative_vr,
            alpha,
            beta,
            gamma,
        )

        gradient_values = [
            radial_bins_positive,
            gradient_minumum_positive,
            radial_bins_negative,
            gradient_minumum_negative,
        ]

        _diagnostic_self_calibration_plot(
            save_path=save_path,
            radius=radius,
            radial_velocity=radial_velocity,
            log_velocity_squared=log_velocity_squared,
            parameters=parameters,
            gradient_values=gradient_values,
        )

    return


def calibrate(
    save_path: str,
    omega_m: float = None,
    **kwargs,
) -> None:
    """Calibrate halo finder parameters using cosmology-based or simulation-based 
    approach.

    This is a high-level interface function that provides two calibration modes:
    
    1. **Cosmology-based calibration** (omega_m provided): Uses pre-calibrated
       empirical relations between calibration parameters and matter density
       parameter Ωₘ. This is fast and doesn't require simulation data, but
       assumes standard ΛCDM cosmology.
       
    2. **Self-calibration** (omega_m=None): Performs full self-calibration from
       simulation data by analyzing particle distributions around isolated halos.
       This is more accurate for non-standard cosmologies or specific simulations
       but requires more computation.

    Both modes save calibration parameters to 'calibration_pars.hdf5' in the
    specified directory for use in halo finding.

    Parameters
    ----------
    save_path : str
        Directory path where calibration parameters will be saved. Must be a
        valid directory path with write permissions. For self-calibration mode,
        this directory must also contain the minibox data files.
        Calibration parameters saved as 'calibration_pars.hdf5'.
    omega_m : float, optional
        Matter density parameter Ωₘ at z=0 for cosmology-based calibration.
        If provided, uses empirical scaling relations. Must be positive and
        typically in range [0.2, 0.4]. Standard ΛCDM values:
        - Planck 2018: Ωₘ ≈ 0.315
        - WMAP: Ωₘ ≈ 0.27
        If None (default), performs full self-calibration from simulation data.
    **kwargs : dict
        Additional keyword arguments passed to self_calibration when omega_m=None.
        Ignored when using cosmology-based calibration. Required parameters for
        self-calibration include:
        - n_seeds : int - Number of halos for calibration
        - seed_data : tuple - Halo catalog data
        - r_max : float - Maximum radius for particle selection
        - boxsize : float - Simulation box size
        - minisize : float - Minibox size
        - particle_mass : float - Particle mass
        - mass_density : float - Mean matter density
        - redshift : float - Simulation redshift
        See self_calibration docstring for complete parameter descriptions.

    Returns
    -------
    None
        Function saves calibration parameters to HDF5 file but returns nothing.

    Raises
    ------
    ValueError
        If omega_m is provided but is non-positive or unreasonably large (>1).
    OSError
        If save_path doesn't exist or lacks write permissions, or if
        required minibox files are missing (self-calibration mode only).
    TypeError
        If required kwargs are missing or have wrong types (self-calibration mode).
    RuntimeError
        If self-calibration optimization fails to converge.

    Notes
    -----
    **Cosmology-based Calibration**:
    
    Uses empirical fitting formulas calibrated from a suite of simulations:
    
    - Positive radial velocity boundary:
      * slope = -1.915 (fixed)
      * intercept = 1.664 + 0.74 * (Ωₘ - 0.3)
      
    - Negative radial velocity boundary (at r_pivot = 0.5):
      * slope = -1.592 + 0.696 * (Ωₘ - 0.3)
      * intercept = 0.8 + 0.525 * (Ωₘ - 0.3)
      * Quadratic correction computed to match boundary at r=0.5
      
    These relations are valid for ΛCDM cosmologies with Ωₘ ∈ [0.2, 0.4] and
    are calibrated at z=0. For other redshifts or non-standard cosmologies,
    self-calibration is recommended.
    
    **Self-calibration**:
    
    Performs full calibration by analyzing particle phase space around isolated
    massive halos. See self_calibration docstring for algorithm details.
    Recommended when:
    - Using non-standard cosmology
    - Simulations with modified gravity or dark energy
    - Requiring maximum accuracy for specific simulation
    - Calibrating at high redshift (z > 1)
    
    **HDF5 File Structure**:
    
    The output file 'calibration_pars.hdf5' contains:
    - 'pos': [slope, intercept] for positive vr boundary
    - 'neg/line': [slope, intercept] for negative vr boundary (linear part)
    - 'neg/quad': [alpha, beta, gamma] for negative vr boundary (quadratic correction)

    Examples
    --------
    >>> # Example 1: Fast cosmology-based calibration
    >>> calibrate(
    ...     save_path="/data/calibration/",
    ...     omega_m=0.3  # Use Ωₘ = 0.3 (close to Planck 2018)
    ... )
    >>> 
    >>> # Verify saved parameters
    >>> import h5py
    >>> with h5py.File("/data/calibration/calibration_pars.hdf5", 'r') as f:
    ...     slope_pos, b_pos = f['pos'][()]
    ...     print(f"Positive vr: slope={slope_pos:.3f}, intercept={b_pos:.3f}")
    Positive vr: slope=-1.915, intercept=1.664

    >>> # Example 2: Self-calibration from simulation data
    >>> import numpy as np
    >>> 
    >>> # Prepare halo catalog
    >>> positions = np.random.uniform(0, 100, (1000, 3))
    >>> velocities = np.random.normal(0, 200, (1000, 3))
    >>> masses = np.random.lognormal(30, 1, 1000)
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> seed_data = (positions, velocities, masses, radii)
    >>> 
    >>> calibrate(
    ...     save_path="/data/calibration/",
    ...     omega_m=None,  # Trigger self-calibration mode
    ...     n_seeds=200,
    ...     seed_data=seed_data,
    ...     r_max=4.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=0.0,
    ...     diagnostics=True
    ... )

    >>> # Example 3: Different cosmologies
    >>> for omega in [0.25, 0.30, 0.35]:
    ...     output_dir = f"/data/calibration_Om{omega:.2f}/"
    ...     calibrate(save_path=output_dir, omega_m=omega)
    ...     print(f"Calibrated for Ωₘ = {omega}")

    >>> # Example 4: High-redshift self-calibration
    >>> calibrate(
    ...     save_path="/data/calibration_z2/",
    ...     omega_m=None,
    ...     n_seeds=300,
    ...     seed_data=seed_data_z2,
    ...     r_max=4.0,
    ...     boxsize=200.0,
    ...     minisize=10.0,
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=2.0,  # High redshift
    ...     n_threads=16,
    ...     diagnostics=True
    ... )

    See Also
    --------
    self_calibration : Full self-calibration from simulation data
    get_calibration_data : Loads particle data for calibration
    _cost_percentile : Cost function for boundary fitting
    _gradient_minima : Identifies phase space boundaries
    """
    if omega_m:
        slope_pos = -1.915
        b_pivot_pos = 1.664 + 0.74 * (omega_m - 0.3)

        x0 = 0.5
        slope_neg = -1.592 + 0.696 * (omega_m - 0.3)
        b_pivot_neg = 0.8 + 0.525 * (omega_m - 0.3)
        b_neg = b_pivot_neg - slope_neg * x0
        gamma = 2.
        alpha = (gamma - b_neg) / x0**2
        beta = slope_neg - 2 * alpha * x0

        with h5py.File(save_path + 'calibration_pars.hdf5', 'w') as hdf:
            hdf.create_dataset('pos', data=[slope_pos, b_pivot_pos])
            hdf.create_dataset('neg/line', data=[slope_neg, b_pivot_neg])
            hdf.create_dataset('neg/quad', data=[alpha, beta, gamma])
    else:
        self_calibration(save_path=save_path, **kwargs)

    return


###
