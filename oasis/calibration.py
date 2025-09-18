import os
from functools import partial
from multiprocessing import Pool
from typing import Tuple

import h5py
import numpy
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

from oasis.common import (G_GRAVITY, _validate_inputs_boxsize_minisize,
                          _validate_inputs_coordinate_arrays,
                          _validate_inputs_existing_path,
                          _validate_inputs_mini_box_id,
                          _validate_inputs_positive_number)
from oasis.coordinates import relative_coordinates, velocity_components
from oasis.minibox import get_mini_box_id, load_particles

__all__ = [
    'get_calibration_data',
    'self_calibration',
    'calibrate',
]


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
    _validate_inputs_coordinate_arrays(radial_distances, 'radial_distances')
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
    pos_seed : np.ndarray
        Seed positions with shape (n_seeds, 3). Each row contains the 
        (x, y, z) coordinates of a seed in simulation units.
    vel_seed : np.ndarray
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
    np.ndarray
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
    >>> pos_seeds = np.array([[10.0, 15.0, 20.0], [25.0, 30.0, 35.0]])
    >>> vel_seeds = np.array([[100.0, 50.0, -75.0], [-80.0, 120.0, 90.0]])
    >>> data = _get_candidate_particle_data(
    ...     mini_box_id=42,
    ...     pos_seed=pos_seeds, 
    ...     vel_seed=vel_seeds,
    ...     r_max=5.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     save_path="/data/miniboxes/",
    ...     part_mass=1e10,
    ...     rhom=2.78e11
    ... )
    >>> print(f"Processed {len(data)} particles from {len(pos_seeds)} seeds")

    See Also
    --------
    _compute_r200m_and_v200 : Computes virial radius and velocity
    load_particles : Loads particles from minibox files
    velocity_components : Decomposes velocities into radial/tangential components
    """
    # Validate inputs
    _validate_inputs_mini_box_id(mini_box_id)
    _validate_inputs_coordinate_arrays(position_seeds, 'seed positions')
    _validate_inputs_coordinate_arrays(velocity_seeds, 'seed velocities')
    _validate_inputs_existing_path(save_path)
    _validate_inputs_boxsize_minisize(boxsize, minisize)
    _validate_inputs_positive_number(r_max, 'r_max')
    _validate_inputs_positive_number(particle_mass, 'particle_mass')
    _validate_inputs_positive_number(mass_density, 'rhom')

    # Load particles in minibox.
    position_particles, velocity_particles, _ = \
        load_particles(mini_box_id, boxsize, minisize, save_path)

    # Iterate over seeds in current mini box.
    radius, radial_velocity, log_velocity_sq = ([] for _ in range(3))
    for position_seed_i, velocity_seed_i in zip(position_seeds, velocity_seeds):
        # Find particles within r_max of the seed
        relative_position = relative_coordinates(position_particles, position_seed_i, boxsize)
        mask_close = numpy.prod(numpy.abs(relative_position) <= r_max, axis=1, dtype=bool)

        # Apply mask
        relative_position = relative_position[mask_close]
        relative_velocity = velocity_particles[mask_close] - velocity_seed_i
        
        # Compute radial distance (L2 norm). No need to further filter by r_max
        # since we already applied a cubic mask.
        rps = numpy.linalg.norm(relative_position, axis=1)

        # Compute velocity components
        vrp, _, v2p = velocity_components(relative_position, relative_velocity)

        # Compute R200m and M200m
        r200m, v200m_sq = _compute_r200m_and_v200m(rps, particle_mass, mass_density)
        
        # Append rescaled quantities to containers
        radius.append(rps / r200m)
        radial_velocity.append(vrp / numpy.sqrt(v200m_sq))

        # Prevent log(0) or log(negative) by setting minimum value
        v_sq_ratio = v2p / v200m_sq
        v_sq_ratio = numpy.maximum(v_sq_ratio, 1e-10)

        log_velocity_sq.append(numpy.log(v_sq_ratio))
    
    # Concatenate arrays
    radius = numpy.concatenate(radius)
    radial_velocity = numpy.concatenate(radial_velocity)
    log_velocity_sq = numpy.concatenate(log_velocity_sq)

    return numpy.vstack([radius, radial_velocity, log_velocity_sq])


def _select_candidate_seeds(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_mass: float,
    mass_density: float,
    n_threads: int = None,
) -> tuple[numpy.ndarray]:
    """Locates for the largest `M_200b` seeds and searches for all the particles
    around them up to a distance `r_max`.

    Only seeds that dominate their environment are eligible. This means that the 
    mass of all other seeds up to a distance of 2*R_200b must be at most 20% the
    mass of the seed.

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

    Returns
    -------
    numpy.ndarray
        Radial distance, radial velocity, and log of the square of the velocity 
        in units of R200m and M200m.
    """
    # Load seed data
    hid, pos_seed, vel_seed, m200b, r200b  = seed_data

    # Rank order by mass.
    order = numpy.argsort(-m200b)
    hid = hid[order]
    pos_seed = pos_seed[order]
    vel_seed = vel_seed[order]
    r200b = r200b[order]
    m200b = m200b[order]

    # Search for eligible seeds.
    # A seed is considered eligible if it dominates its own environment. This is
    # enforced by requiring that the next most-massive seed within 2*R200 is, at
    # least five times smaller than the seed, i.e. has a mass <= 20% of M200.
    # The loop will stop once it has found `n_seeds` eligible seeds.
    # NOTE: When using multiple threads, this is the part that takes most of the
    # execution time and scales with the number of candidate seeds requested.
    seed_i = []
    i = 0
    # print('Looking for candidate seeds...')
    while len(seed_i) < n_seeds:
        # Exit the loop if there are no more seeds in the list.
        if i >= len(hid)-1: 
            print(f'Only found {i}/{n_seeds} seeds.')
            break
        mask_close = numpy.prod(numpy.abs(pos_seed - pos_seed[i]) <=  2.*r200b[i],
                             axis=1, dtype=bool)
        mask_self = m200b != m200b[i]
        if numpy.all(m200b[mask_close & mask_self] < (0.2 * m200b[i])):
            seed_i.append(i)
        i += 1
    # print(f'Found candidate seeds.')

    hid = hid[seed_i]
    pos_seed = pos_seed[seed_i]
    vel_seed = vel_seed[seed_i]

    # Locate mini box IDs for all seeds.
    seed_mini_box_id = get_mini_box_id(pos_seed, boxsize, minisize)
    # Sort by mini box ID
    order = numpy.argsort(seed_mini_box_id)
    seed_mini_box_id = seed_mini_box_id[order]
    hid = hid[order]
    pos_seed = pos_seed[order]
    vel_seed = vel_seed[order]

    # Get unique mini box ids
    unique_mini_box_ids = numpy.unique(seed_mini_box_id)
    n_unique = len(unique_mini_box_ids)

    pos_unique = [
        pos_seed[seed_mini_box_id == mini_box_id] 
        for mini_box_id in unique_mini_box_ids
    ]
    vel_unique = [
        vel_seed[seed_mini_box_id == mini_box_id] 
        for mini_box_id in unique_mini_box_ids
    ]

    func = partial(_get_candidate_particle_data, r_max=r_max, boxsize=boxsize, 
                   minisize=minisize, save_path=save_path, particle_mass=particle_mass, 
                   rhom=mass_density)
    
    # Cap the number of threads to the total number of miniboxes to process.
    if not n_threads:
        n_threads = numpy.min([os.cpu_count()-10, n_unique])
    else:
        n_threads = numpy.min([n_threads, n_unique])

    data = zip(unique_mini_box_ids, pos_unique, vel_unique)
    out = []
    with Pool(n_threads) as pool, \
        tqdm(total=n_unique, colour="blue", ncols=100,
             desc='Processing candidates') as pbar:
        # The 
        for res in pool.starmap(func, data):
            out.append(res)
            pbar.update()
            pbar.refresh()
    out = numpy.concatenate(out, axis=1).T

    # Return an array where each column corresponds to r, vr, lnv2 respectively
    return out


def get_calibration_data(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_mass: float,
    mass_density: float,
    n_threads: int = None,
) -> Tuple[numpy.ndarray]:
    """_summary_

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
    n_threads : int
        Number of threads, by default None

    Returns
    -------
    tuple[numpy.ndarray]
        Radial distance, radial velocity, and log of the square of the velocity
    """
    file_name = save_path + 'calibration_data.hdf5'
    try:
        with h5py.File(file_name, 'r') as hdf:
            r = hdf['r'][()]
            vr = hdf['vr'][()]
            lnv2 = hdf['lnv2'][()]
        return r, vr, lnv2
    except:
        out = _select_candidate_seeds(
            n_seeds=n_seeds,
            seed_data=seed_data,
            r_max=r_max,
            boxsize=boxsize,
            minisize=minisize,
            save_path=save_path,
            particle_mass=particle_mass,
            mass_density=mass_density,
            n_threads=n_threads,
        )

        with h5py.File(file_name, 'w') as hdf:
            hdf.create_dataset('r', data=out[:, 0])
            hdf.create_dataset('vr', data=out[:, 1])
            hdf.create_dataset('lnv2', data=out[:, 2])

        return out[:, 0], out[:, 1], out[:, 2]



def cost_percentile(b: float, *data) -> float:
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
    r, lnv2, slope, target, r0 = data
    line = slope * (r - r0) + b
    below_line = (lnv2 < line).sum()
    return numpy.log((target - below_line / r.shape[0]) ** 2)


def cost_perp_distance(b: float, *data) -> float:
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
    r, lnv2, slope, width, r0 = data
    line = slope * (r - r0) + b
    d = numpy.abs(lnv2 - line) / numpy.sqrt(1 + slope**2)
    return -numpy.log(numpy.mean(d[(d < width)] ** 2))


def gradient_minima(
    r: numpy.ndarray,
    lnv2: numpy.ndarray,
    n_points: int,
    r_min: float,
    r_max: float,
    n_bins: int = 100,
    sigma_smooth: float = 2.0,
    diagnostics: bool = False,
) -> tuple[numpy.ndarray]:
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
    r_edges = numpy.linspace(r_min, r_max, n_points + 1)
    counts_gradient_minima = numpy.zeros(n_points)

    lnv2_bins_out = numpy.zeros((n_points, n_bins))
    counts_out = numpy.zeros((n_points, n_bins))
    counts_gradient_out = numpy.zeros((n_points, n_bins))
    counts_gradient_smooth_out = numpy.zeros((n_points, n_bins))
    
    for i in range(n_points):
        # Create mask for current r bin
        r_mask = (r > r_edges[i]) * (r < r_edges[i + 1])

        # Compute histogram of lnv2 values within the r bin and the vr mask
        counts, lnv2_edges = numpy.histogram(lnv2[r_mask], bins=n_bins)
        
        # Compute the gradient of the histogram
        counts_gradient = numpy.gradient(counts, numpy.mean(numpy.diff(lnv2_edges)))
        counts_gradient /= numpy.max(numpy.abs(counts_gradient))
        
        # Smooth the gradient
        counts_gradient_smooth = gaussian_filter1d(counts_gradient, sigma_smooth)
        
        # Find the lnv2 value corresponding to the minimum of the smoothed gradient
        lnv2_bins = 0.5 * (lnv2_edges[:-1] + lnv2_edges[1:])
        counts_gradient_minima[i] = lnv2_bins[numpy.argmin(counts_gradient_smooth)]

        # Store diagnostics
        lnv2_bins_out[i, :] = lnv2_bins
        counts_out[i, :] = counts / numpy.max(counts)
        counts_gradient_out[i, :] = counts_gradient
        counts_gradient_smooth_out[i, :] = counts_gradient_smooth
    
    # Compute r bin centres
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Return diagnostics if requested
    if diagnostics:
        return r_bins, counts_gradient_minima, \
            (lnv2_bins_out, counts_out, counts_gradient_out, counts_gradient_smooth_out)
    else:
        return r_bins, counts_gradient_minima


def self_calibration(
    n_seeds: int,
    seed_data: tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_mass: float,
    rhom: float,
    n_points: int = 20,
    perc: float = 0.995,
    width: float = 0.05,
    grad_lims: tuple[float] = (0.2, 0.5),
    n_threads: int = None,
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
    r, vr, lnv2 = get_calibration_data(
        n_seeds=n_seeds,
        seed_data=seed_data,
        r_max=r_max,
        boxsize=boxsize,
        minisize=minisize,
        save_path=save_path,
        particle_mass=particle_mass,
        mass_density=rhom,
        n_threads=n_threads,
    )

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r <= 2.0
    x0 = 0.5

    # For vr > 0 ===============================================================
    r_grad, min_grad = gradient_minima(r[mask_vr_pos], lnv2[mask_vr_pos], n_points, 
                                       *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * (x - x0) + b, r_grad, min_grad, 
                        p0=[-1, 2], bounds=((-5, 0), (0, 5)))
    slope_pos, pivot_0 = popt

    # Find intercept by finding the value that contains 'perc' percent of
    # particles below the line at fixed slope 'm_pos'.
    res = minimize(
        fun=cost_percentile,
        x0=1.1 * pivot_0,
        bounds=((pivot_0, 5.0),),
        args=(r[mask_vr_pos&mask_r], lnv2[mask_vr_pos&mask_r], slope_pos, perc, x0),
        method='Nelder-Mead',
    )
    b_pivot_pos = res.x[0]

    # For vr < 0 ===============================================================
    r_grad, min_grad = gradient_minima(r[mask_vr_neg], lnv2[mask_vr_neg], n_points, 
                                       *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * (x - x0) + b, r_grad, min_grad, 
                        p0=[-1, 2], bounds=((-5, 0), (0, 3)))
    slope_neg, pivot_1 = popt

    # Find intercept by finding the value that maximizes the perpendicular
    # distance to the line at fixed slope of all points within a perpendicular
    # 'width' distance from the line (ignoring all others).
    res = minimize(
        fun=cost_perp_distance,
        x0=0.8 * pivot_1,
        bounds=((0.5 * pivot_1, pivot_1),),
        args=(r[mask_vr_neg], lnv2[mask_vr_neg], slope_neg, width, x0),
        method='Nelder-Mead',
    )
    b_pivot_neg = res.x[0]
    
    b_neg = b_pivot_neg - slope_neg * x0
    gamma = 2.
    alpha = (gamma - b_neg) / x0**2
    beta = slope_neg - 2 * alpha * x0
    

    with h5py.File(save_path + 'calibration_pars.hdf5', 'w') as hdf:
        hdf.create_dataset('pos', data=[slope_pos, b_pivot_pos])
        hdf.create_dataset('neg/line', data=[slope_neg, b_pivot_neg])
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
