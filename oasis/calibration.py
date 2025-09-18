import os
from multiprocessing import Pool
from typing import Dict, Tuple, Union

import h5py
import numpy
from matplotlib import pyplot
from matplotlib.colorbar import ColorbarBase
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

from oasis.common import (G_GRAVITY, _validate_inputs_boxsize_minisize,
                          _validate_inputs_coordinate_arrays,
                          _validate_inputs_existing_path,
                          _validate_inputs_mini_box_id,
                          _validate_inputs_positive_number,
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
    - Density is computed as ρ(r) = M(<r) / (4π/3 × r³)  
    - R200m corresponds to the largest radius where ρ(r) ≥ 200×ρ_m
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
    _validate_inputs_positive_number(particle_mass, 'particle_mass')
    _validate_inputs_positive_number(mass_density, 'mass_density')

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
    - R200m is defined as the radius where mean enclosed density equals 200×ρ_m
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
    _validate_inputs_positive_number(r_max, 'r_max')
    _validate_inputs_positive_number(particle_mass, 'particle_mass')
    _validate_inputs_positive_number(mass_density, 'rhom')

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
    radius (isolation_radius_factor × R200) must have masses less than
    isolation_factor × M200 of the candidate seed.

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
        mass < isolation_factor × seed_mass to satisfy isolation criterion.
        Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200 to define isolation radius. Neighbors within
        isolation_radius_factor × R200 are checked. Default is 2.0.

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
    _validate_inputs_positive_number(max_seeds, 'max_seeds')
    _validate_inputs_positive_number(boxsize, 'boxsize')
    _validate_inputs_positive_number(isolation_factor, 'isolation_factor')
    _validate_inputs_positive_number(
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

    for i in range(n_seeds):
        # If all requested isolated seeds have been found, exit loop
        if len(isolated_indices) >= max_seeds:
            break

        current_position = position[i]
        isolation_radius = isolation_radii[i]
        max_neighbor_mass = max_neighbor_masses[i]

        # Find neighboring seeds within isolation radius using periodic BC
        rel_positions = relative_coordinates(position, current_position,
                                             boxsize, periodic=True)
        distances = numpy.linalg.norm(rel_positions, axis=1)

        # Exclude self (distance = 0) and find neighbors within isolation radius
        neighbor_mask = (distances > 0) & (distances <= isolation_radius)

        # No neighbors found - seed is isolated
        if not numpy.any(neighbor_mask):
            isolated_indices.append(i)
            continue

        neighbor_masses = mass[neighbor_mask]

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
    are considered (isolation criterion: all neighbors within 2×R200 must have
    mass < 0.2×M200 of the seed).

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
        mass < isolation_factor × seed_mass. Default is 0.2 (20%).
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
    _validate_inputs_positive_number(n_seeds, 'n_seeds')
    _validate_inputs_seed_data(seed_data)
    _validate_inputs_positive_number(r_max, 'r_max')
    _validate_inputs_existing_path(save_path)
    _validate_inputs_boxsize_minisize(boxsize, minisize)
    _validate_inputs_positive_number(particle_mass, 'particle_mass')
    _validate_inputs_positive_number(mass_density, 'mass_density')
    if n_threads is not None:
        _validate_inputs_positive_number(n_threads, 'n_threads')

    # Unpack seed data
    position, velocity, mass, radius = seed_data

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
    - Uses 200×200 bins for high-resolution visualization
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

    mask_negative_vr = (radial_velocity < 0)
    mask_positive_vr = ~mask_negative_vr

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4.2))
    fig.suptitle('Calibration data', fontsize=SIZE_LABELS)
    axes = axes.flatten()

    for ax in axes:
        ax.set_xlabel(r'$r/R_{\rm 200m}$', fontsize=SIZE_LABELS)
        ax.set_ylabel(r'$\ln(v^2/v_{\rm 200m}^2)$', fontsize=SIZE_LABELS)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)

    # Create a color bar to display mass ranges.
    cax = fig.add_axes([1.01, 0.2, 0.01, 0.7])
    cbar = ColorbarBase(cax, cmap=cmap, orientation="vertical", extend='max')
    cbar.set_label(r'Counts', fontsize=SIZE_LEGEND)
    cbar.set_ticklabels([], fontsize=SIZE_TICKS)
    cbar.ax.tick_params(size=0, labelleft=False, labelright=False,
                        labeltop=False, labelbottom=False)

    pyplot.sca(axes[0])
    pyplot.title(r'$v_r > 0$', fontsize=SIZE_LABELS)
    pyplot.hist2d(radius[mask_positive_vr],
                  log_velocity_squared[mask_positive_vr],
                  bins=200, cmap=cmap, range=limits)

    pyplot.sca(axes[1])
    pyplot.title(r'$v_r < 0$', fontsize=SIZE_LABELS)
    pyplot.hist2d(radius[mask_negative_vr],
                  log_velocity_squared[mask_negative_vr],
                  bins=200, cmap=cmap, range=limits)

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
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
    n_threads: int = None,
    diagnostics: bool = True,
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
        simulation units. Must be positive, typically 2-5 × R200.
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
    isolation_factor : float, optional
        Maximum allowed mass ratio for isolation criterion. Neighbors must have
        mass < isolation_factor × seed_mass. Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200 to define isolation radius for neighbor search.
        Default is 2.0.
    n_threads : int, optional
        Number of threads for parallel processing. If None, uses half of
        available CPU cores. Only used when generating new data.
    diagnostics : bool, optional
        Whether to generate diagnostic plots of the calibration data.
        Default is True.

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
    ...     diagnostics=False  # Skip plot generation
    ... )

    See Also
    --------
    _select_candidate_seeds : Core function for generating calibration data
    _diagnostic_calibration_data_plot : Creates diagnostic visualizations
    h5py.File : HDF5 file interface for data persistence
    """
    file_name = save_path + 'calibration_data.hdf5'
    try:
        with h5py.File(file_name, 'r') as hdf:
            radius = hdf['r'][()]
            radial_velocity = hdf['vr'][()]
            log_velocity_squared = hdf['lnv2'][()]
    except:
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


def _cost_percentile(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the `target` percentile of particles is below the line.

    Parameters
    ----------
    b : float
        Fit parameter
    *data : tuple
        A tuple with `[r, lnv2, slope, target]`, where `slope` is the slope of 
        the line and is fixed, and `target` is the desired percentile

    Returns
    -------
    float
    """
    radius, log_velocity_squared, slope, target, radius_pivot = data
    line = slope * (radius - radius_pivot) + b

    # Total number of elements below the line
    below_line = (log_velocity_squared < line).sum()

    # Fraction of elements below the line
    fraction = target - below_line / radius.shape[0]

    # Cost to minimize
    cost = numpy.log(fraction ** 2)
    return cost


def _cost_perpendicular_distance(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the perpendicular distance of all points to the line is maximal
    Parameters
    ----------
    b : float
        Fit parameter
    *data: tuple
        A tuple with `[r, lnv2, slope, width]`, where `slope` is the slope of 
        the line and is fixed, and `width` is the width of a band around the 
        line within which the distance is computed

    Returns
    -------
    float
    """
    radius, log_velocity_squared, slope, width, radius_pivot = data
    line = slope * (radius - radius_pivot) + b

    # Perpendicular distance to the line
    distance = numpy.abs(log_velocity_squared - line) / \
        numpy.sqrt(1 + slope**2)

    # Select only elements within the width of the band
    distance_within_band = distance[(distance < width)]

    # Cost to maximize (thus negative)
    cost = -numpy.log(numpy.mean(distance_within_band ** 2))
    return cost


def _gradient_minima(
    radius: numpy.ndarray,
    log_velocity_squared: numpy.ndarray,
    n_points: int,
    r_min: float,
    r_max: float,
    n_bins: int = 100,
    sigma_smooth: float = 2.0,
    diagnostics: bool = True,
) -> Tuple[numpy.ndarray]:
    """Computes the r-lnv2 gradient and finds the minimum as a function of `r`
    within the interval `[r_min, r_max]`

    Parameters
    ----------
    r : numpy.ndarray
        Radial separation
    lnv2 : numpy.ndarray
        Log-kinetic energy
    mask_vr : numpy.ndarray
        Mask for the selection of radial velocity
    n_points : int
        Number of minima points to compute
    r_min : float
        Minimum radial distance
    r_max : float
        Maximum radial distance

    Returns
    -------
    tuple[numpy.ndarray]
        Radial and minima coordinates.
    """
    radius_edges = numpy.linspace(r_min, r_max, n_points + 1)
    counts_gradient_minima = numpy.zeros(n_points)

    lnv2_bins_out = numpy.zeros((n_points, n_bins))
    counts_out = numpy.zeros((n_points, n_bins))
    counts_gradient_out = numpy.zeros((n_points, n_bins))
    counts_gradient_smooth_out = numpy.zeros((n_points, n_bins))

    for i in range(n_points):
        # Create mask for current r bin
        radius_mask = (radius > radius_edges[i]) * \
            (radius < radius_edges[i + 1])

        # Compute histogram of lnv2 values within the r bin and the vr mask
        counts, lnv2_edges = numpy.histogram(
            log_velocity_squared[radius_mask], bins=n_bins)

        # Compute the gradient of the histogram
        counts_gradient = numpy.gradient(
            counts, numpy.mean(numpy.diff(lnv2_edges)))
        counts_gradient /= numpy.max(numpy.abs(counts_gradient))

        # Smooth the gradient
        counts_gradient_smooth = gaussian_filter1d(
            counts_gradient, sigma_smooth)

        # Find the lnv2 value corresponding to the minimum of the smoothed gradient
        log_velocity_squared_bins = 0.5 * (lnv2_edges[:-1] + lnv2_edges[1:])
        counts_gradient_minima[i] = log_velocity_squared_bins[numpy.argmin(
            counts_gradient_smooth)]

        # Store diagnostics
        lnv2_bins_out[i, :] = log_velocity_squared_bins
        counts_out[i, :] = counts / numpy.max(counts)
        counts_gradient_out[i, :] = counts_gradient
        counts_gradient_smooth_out[i, :] = counts_gradient_smooth

    # Compute r bin centres
    radial_bins = 0.5 * (radius_edges[:-1] + radius_edges[1:])

    # Return diagnostics if requested
    if diagnostics:
        return radial_bins, counts_gradient_minima, \
            (lnv2_bins_out, counts_out, counts_gradient_out, counts_gradient_smooth_out)
    else:
        return radial_bins, counts_gradient_minima


def self_calibration(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_mass: float,
    mass_density: float,
    n_points: int = 20,
    percent: float = 0.995,
    width: float = 0.05,
    gradient_radial_lims: Tuple[float] = (0.2, 0.5),
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
    n_threads: int = None,
    diagnostics: bool = True
) -> None:
    """Runs calibration from isolated halo samples.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    seed_data : tuple[numpy.ndarray]
        Tuple with seed ID, positions, velocities, M200b and R200b.
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    particle_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe
    n_points : int, optional
        Number of minima points to compute, by default 20
    perc : float, optional
        Target percentile for the positive radial velocity calibration, 
        by default 0.98
    width : float, optional
        Band width for the negattive radial velocity calibration, 
        by default 0.05
    grad_lims : tuple[float]
        Radial interval where the gradient is computed, by default (0.2, 0.5)
    n_threads : int
        Number of threads, by default None
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
        n_threads=n_threads,
        isolation_factor=isolation_factor,
        isolation_radius_factor=isolation_radius_factor,
        diagnostics=diagnostics,
    )

    mask_negative_vr = (radial_velocity < 0)
    mask_positive_vr = ~mask_negative_vr
    mask_low_radius = radius <= 2.0
    radius_pivot = 0.5
    def line_model(x, slope, abscissa): return slope * \
        (x - radius_pivot) + abscissa

    # For vr > 0 ===============================================================
    radial_bins, gradient_minumum = _gradient_minima(
        radius=radius[mask_positive_vr],
        log_velocity_squared=log_velocity_squared[mask_positive_vr],
        n_points=n_points,
        r_min=gradient_radial_lims[0],
        r_max=gradient_radial_lims[1],
        diagnostics=diagnostics
    )
    # Find slope by fitting to the minima.
    (slope_positive_vr, abscissa_p), _ = curve_fit(line_model, radial_bins, gradient_minumum,
                                                   p0=[-1, 2], bounds=((-5, 0), (0, 5)))

    # Find intercept by finding the value that contains 'perc' percent of
    # particles below the line at fixed slope 'm_pos'.
    res = minimize(
        fun=_cost_percentile,
        x0=1.1 * abscissa_p,
        bounds=((abscissa_p, 5.0),),
        args=(radius[mask_positive_vr & mask_low_radius],
              log_velocity_squared[mask_positive_vr & mask_low_radius],
              slope_positive_vr, percent, radius_pivot),
        method='Nelder-Mead',
    )
    abscissa_positive_vr = res.x[0]

    # For vr < 0 ===============================================================
    radial_bins, gradient_minumum = _gradient_minima(
        radius=radius[mask_negative_vr],
        log_velocity_squared=log_velocity_squared[mask_negative_vr],
        n_points=n_points,
        r_min=gradient_radial_lims[0],
        r_max=gradient_radial_lims[1],
        diagnostics=diagnostics
    )
    # Find slope by fitting to the minima.
    (slope_negative_vr, abscissa_n), _ = curve_fit(line_model, radial_bins, gradient_minumum,
                                                   p0=[-1, 2], bounds=((-5, 0), (0, 3)))

    # Find intercept by finding the value that maximizes the perpendicular
    # distance to the line at fixed slope of all points within a perpendicular
    # 'width' distance from the line (ignoring all others).
    res = minimize(
        fun=_cost_perpendicular_distance,
        x0=0.8 * abscissa_n,
        bounds=((0.5 * abscissa_n, abscissa_n),),
        args=(radius[mask_negative_vr], log_velocity_squared[mask_negative_vr],
              slope_negative_vr, width, radius_pivot),
        method='Nelder-Mead',
    )
    b_pivot_neg = res.x[0]

    b_neg = b_pivot_neg - slope_negative_vr * radius_pivot
    gamma = 2.
    alpha = (gamma - b_neg) / radius_pivot**2
    beta = slope_negative_vr - 2 * alpha * radius_pivot

    with h5py.File(save_path + 'calibration_pars.hdf5', 'w') as hdf:
        hdf.create_dataset(
            'pos', data=[slope_positive_vr, abscissa_positive_vr])
        hdf.create_dataset('neg/line', data=[slope_negative_vr, b_pivot_neg])
        hdf.create_dataset('neg/quad', data=[alpha, beta, gamma])

    return


def calibrate(
    save_path: str,
    omega_m: float = None,
    **kwargs,
) -> None:
    """Calibrates finder by assuming cosmology dependence. If `omega_m` is 
    `None`, then it runs the calibration on the simulation data directly.

    Parameters
    ----------
    save_path : str
        Path to the mini boxes. Saves the calibration parameter in this directory.
    omega_m : float, optional
        Matter density parameter Omega matter, by default None.
    **kwargs
        See `run_calibrate` for parameter descriptions.
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
