import os
from functools import partial
from multiprocessing import Pool

import h5py as h5
import numpy as np
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

from oasis.common import G_gravity
from oasis.coordinates import relative_coordinates, velocity_components
from oasis.minibox import get_mini_box_id, load_particles


def _get_candidate_particle_data(
    mini_box_id: int,
    pos_seed: list[np.ndarray],
    vel_seed: list[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    part_mass: float,
    rhom: float,
) -> np.ndarray:
    """Extracts all requested seeds from a single minibox.

    Parameters
    ----------
    mini_box_id : int
        Mini-box ID
    pos_seed : list[np.ndarray]
        Seed positions
    vel_seed : list[np.ndarray]
        Seed velocities
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe

    Returns
    -------
    np.ndarray
        Particle's radial distance, radial velocity and log of the square of the
        velocity in units of R200m and M200m.
    """
    # Load particles in minibox.
    pos, vel, *_ = load_particles(mini_box_id, boxsize, minisize, save_path)

    # Iterate over seeds in current mini box.
    r, vr, lnv2 = ([] for _ in range(3))
    for i in range(len(pos_seed)):
        # Compute the relative positions of all particles in the box
        rel_pos = relative_coordinates(pos, pos_seed[i], boxsize)
        # Only work with those close to the seed
        mask_close = np.prod(np.abs(rel_pos) <= r_max, axis=1, dtype=bool)

        rel_pos = rel_pos[mask_close]
        rel_vel = vel[mask_close] - vel_seed[i]
        
        # Compute radial distance, radial and tangential velocities
        rps = np.sqrt(np.sum(np.square(rel_pos), axis=1))
        vrp, _, v2p = velocity_components(rel_pos, rel_vel)

        # Compute R200m and M200m
        rps_prof = rps[np.argsort(rps)]
        mass_prof = part_mass * np.arange(1, len(rps_prof)+1)
        # Find \rho(r) = 200*\rhom
        rho200_loc = np.argmax(mass_prof / (4 / 3 * np.pi * rps_prof ** 3) \
                               <= 200 * rhom)
        r200m = rps_prof[rho200_loc]
        m200m = mass_prof[rho200_loc]

        # Compute V200
        v200sq = G_gravity * m200m / r200m
        
        # Append rescaled quantities to containers
        r.append(rps/r200m)
        vr.append(vrp/np.sqrt(v200sq))
        lnv2.append(np.log(v2p/v200sq))
    
    # Concatenate into a single array
    r = np.concatenate(r)
    vr = np.concatenate(vr)
    lnv2 = np.concatenate(lnv2)

    return np.vstack([r, vr, lnv2])

def _select_candidate_seeds(
    n_seeds: int,
    seed_data: tuple[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    part_mass: float,
    rhom: float,
    n_threads: int = None,
) -> tuple[np.ndarray]:
    """Locates for the largest `M_200b` seeds and searches for all the particles
    around them up to a distance `r_max`.

    Only seeds that dominate their environment are eligible. This means that the 
    mass of all other seeds up to a distance of 2*R_200b must be at most 20% the
    mass of the seed.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    seed_data : tuple[np.ndarray]
        Tuple with seed ID, positions, velocities, M200b and R200b.
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe

    Returns
    -------
    np.ndarray
        Radial distance, radial velocity, and log of the square of the velocity 
        in units of R200m and M200m.
    """
    # Load seed data
    hid, pos_seed, vel_seed, m200b, r200b  = seed_data

    # Rank order by mass.
    order = np.argsort(-m200b)
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
    print('Looking for candidate seeds...')
    while len(seed_i) < n_seeds:
        # Exit the loop if there are no more seeds in the list.
        if i >= len(hid)-1: 
            print(f'Only found {i}/{n_seeds} seeds.')
            break
        mask_close = np.prod(np.abs(pos_seed - pos_seed[i]) <=  2.*r200b[i],
                             axis=1, dtype=bool)
        mask_self = m200b != m200b[i]
        if np.all(m200b[mask_close & mask_self] < (0.2 * m200b[i])):
            seed_i.append(i)
        i += 1
    print(f'Found candidate seeds.')

    hid = hid[seed_i]
    pos_seed = pos_seed[seed_i]
    vel_seed = vel_seed[seed_i]

    # Locate mini box IDs for all seeds.
    seed_mini_box_id = get_mini_box_id(pos_seed, boxsize, minisize)
    # Sort by mini box ID
    order = np.argsort(seed_mini_box_id)
    seed_mini_box_id = seed_mini_box_id[order]
    hid = hid[order]
    pos_seed = pos_seed[order]
    vel_seed = vel_seed[order]

    # Get unique mini box ids
    unique_mini_box_ids = np.unique(seed_mini_box_id)
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
                   minisize=minisize, save_path=save_path, part_mass=part_mass, 
                   rhom=rhom)
    
    # Cap the number of threads to the total number of miniboxes to process.
    if not n_threads:
        n_threads = np.min([os.cpu_count()-10, n_unique])
    else:
        n_threads = np.min([n_threads, n_unique])

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
    out = np.concatenate(out, axis=1).T

    # Return an array where each column corresponds to r, vr, lnv2 respectively
    return out


def get_calibration_data(
    n_seeds: int,
    seed_data: tuple[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    part_mass: float,
    rhom: float,
    n_threads: int = None,
) -> tuple[np.ndarray]:
    """_summary_

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    seed_data : tuple[np.ndarray]
        Tuple with seed ID, positions, velocities, M200b and R200b.
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe
    n_threads : int
        Number of threads, by default None

    Returns
    -------
    tuple[np.ndarray]
        Radial distance, radial velocity, and log of the square of the velocity
    """
    file_name = save_path + 'calibration_data.hdf5'
    try:
        with h5.File(file_name, 'r') as hdf:
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
            part_mass=part_mass,
            rhom=rhom,
            n_threads=n_threads,
        )

        with h5.File(file_name, 'w') as hdf:
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
    r, lnv2, slope, target = data
    below_line = (lnv2 < (slope * r + b)).sum()
    return np.log((target - below_line / r.shape[0]) ** 2)


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
    r, lnv2, slope, width = data
    d = np.abs(lnv2 - slope * r - b) / np.sqrt(1 + slope**2)
    return -np.log(np.mean(d[(d < width)] ** 2))


def gradient_minima(
    r: np.ndarray,
    lnv2: np.ndarray,
    mask_vr: np.ndarray,
    n_points: int,
    r_min: float,
    r_max: float,
) -> tuple[np.ndarray]:
    """Computes the r-lnv2 gradient and finds the minimum as a function of `r`
    within the interval `[r_min, r_max]`

    Parameters
    ----------
    r : np.ndarray
        Radial separation
    lnv2 : np.ndarray
        Log-kinetic energy
    mask_vr : np.ndarray
        Mask for the selection of radial velocity
    n_points : int
        Number of minima points to compute
    r_min : float
        Minimum radial distance
    r_max : float
        Maximum radial distance

    Returns
    -------
    tuple[np.ndarray]
        Radial and minima coordinates.
    """
    r_edges_grad = np.linspace(r_min, r_max, n_points + 1)
    grad_r = 0.5 * (r_edges_grad[:-1] + r_edges_grad[1:])
    grad_min = np.zeros(n_points)
    for i in range(n_points):
        r_mask = (r > r_edges_grad[i]) * (r < r_edges_grad[i + 1])
        hist_yv, hist_edges = np.histogram(lnv2[mask_vr * r_mask], bins=200)
        hist_lnv2 = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        hist_lnv2_grad = np.gradient(hist_yv, np.mean(np.diff(hist_edges)))
        lnv2_mask = (1.0 < hist_lnv2) * (hist_lnv2 < 2.0)
        grad_min[i] = hist_lnv2[lnv2_mask][np.argmin(hist_lnv2_grad[lnv2_mask])]

    return grad_r, grad_min


def calibrate(
    n_seeds: int,
    seed_data: tuple[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    part_mass: float,
    rhom: float,
    n_points: int = 20,
    perc: float = 0.995,
    width: float = 0.05,
    grad_lims: tuple[float] = (0.2, 0.5),
    n_threads: int = None,
) -> None:
    """_summary_

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    seed_data : tuple[np.ndarray]
        Tuple with seed ID, positions, velocities, M200b and R200b.
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
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
        part_mass=part_mass,
        rhom=rhom,
        n_threads=n_threads,
    )

    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r < 2.0

    # For vr > 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_pos, n_points, 
                                       *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * x + b, r_grad, min_grad, p0=[-1, 2])
    m_pos, b01 = popt

    # Find intercept by finding the value that contains 'perc' percent of
    # particles below the line at fixed slope 'm_pos'.
    res = minimize(
        cost_percentile,
        1.1 * b01,
        bounds=((0.8 * b01, 3.0),),
        args=(r[mask_vr_pos * mask_r], lnv2[mask_vr_pos * mask_r], m_pos, perc),
        method='Nelder-Mead',
    )
    b_pos = res.x[0]

    # For vr < 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_neg, n_points, 
                                       *grad_lims)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * x + b, r_grad, min_grad, p0=[-1, 2])
    m_neg, b02 = popt

    # Find intercept by finding the value that maximizes the perpendicular
    # distance to the line at fixed slope of all points within a perpendicular
    # 'width' distance from the line (ignoring all others).
    res = minimize(
        cost_perp_distance,
        0.75 * b02,
        bounds=((1.2, b02),),
        args=(r[mask_vr_neg], lnv2[mask_vr_neg], m_neg, width),
        method='Nelder-Mead',
    )
    b_neg = res.x[0]

    with h5.File(save_path + 'calibration_pars.hdf5', 'w') as hdf:
        hdf.create_dataset('pos', data=[m_pos, b_pos])
        hdf.create_dataset('neg', data=[m_neg, b_neg])

    return


###
