"""Template file for running OASIS: Orbiting mAss asSIngment Scheme

All loading and processing of simulation data and catalogues is assumed to be 
done before running OASIS. This is done to avoid adding dependencies specific
any simulation and simplifying the API.

"""
import numpy as np

from oasis.calibration import calibrate
from oasis.catalogue import run_orbiting_mass_assignment
from oasis.common import mkdir, timer
from oasis.minibox import split_box_into_mini_boxes

# Simulation box parameters ====================================================
# EXAMPLE values. Change them to your simulation parameters.
boxsize: float = 1000.                     # h^-1 Mpc
minisize: float = 100.                     # h^-1 Mpc
padding: float = 5.                        # h^-1 Mpc
rhom: float = 0.3 * 277_536_627_000.0      # h^-2 M_sun / Mpc^3
part_mass: float = 77546570000.0           # h^-1 M_sun

# ==============================================================================
#
#                               OASIS configuration
#
# ==============================================================================
# OASIS will save results to this path
save_path: str = '/'

# Data preparation =============================================================
chunksize_seed: int = 100_000
chunksize_part: int = 10_000_000

# Calibration ==================================================================
n_seeds: int = 500
r_max: float = 5.0
calib_n_points: int = 20
calib_p: float = 0.995
calib_w: float = 0.050
calib_grad_lims: tuple[float] = (0.2, 0.5)
n_threads: int = 50

# Run ==========================================================================
n_orb_min: int = 200
fast_mass: bool = False
run_name = '<cool name>'        # OASIS appends this name to identify the run


# ==============================================================================
#
#                               OASIS sample run
#
# ==============================================================================
@timer
def process_seeds(data: tuple[np.ndarray]) -> None:
    # Load your data
    hid, pos, vel, r200b, m200b, rs = data

    # Additional properties to include in seed catalogue.
    props = [r200b, m200b, rs]
    labels = ('R200b', 'M200b', 'Rs')
    dtypes = (np.float32, np.float32, np.float32)
    props_zip = (props, labels, dtypes)

    # Save seeds into miniboxes according to their minibox ID.
    split_box_into_mini_boxes(
        positions=pos,
        velocities=vel,
        uid=hid,
        save_path=save_path,
        boxsize=boxsize,
        minisize=minisize,
        chunksize=chunksize_seed,
        name='seed',
        props=props_zip,
    )
    return None


@timer
def process_particles(data: tuple[np.ndarray]) -> None:
    # Load your data
    pid, pos, vel = data

    split_box_into_mini_boxes(
        positions=pos,
        velocities=vel,
        uid=pid,
        save_path=save_path,
        boxsize=boxsize,
        minisize=minisize,
        chunksize=chunksize_part,
        name='part',
    )

    return


@timer
def calibration(data: tuple[np.ndarray]) -> None:
    # Load your data. No `rs` needed here.
    hid, pos, vel, m200b, r200b, _ = data

    calibrate(
        n_seeds=n_seeds,
        seed_data=data,
        r_max=r_max,
        boxsize=boxsize,
        minisize=minisize,
        save_path=save_path,
        part_mass=part_mass,
        rhom=rhom,
        n_points=calib_n_points,
        perc=calib_p,
        width=calib_w,
        grad_lims=calib_grad_lims,
        n_threads=n_threads,
    )

    return


@timer
def run_oasis():

    run_orbiting_mass_assignment(
        load_path=save_path,
        min_num_part=n_orb_min,
        boxsize=boxsize,
        minisize=minisize,
        run_name=run_name,
        padding=padding,
        fast_mass=fast_mass,
        part_mass=part_mass,
        n_threads=n_threads,
    )

    return


@timer
def main():

    # LOAD YOUR DATA HERE
    seed_data = ()
    particle_data = ()

    process_seeds(data=seed_data)
    process_particles(data=particle_data)
    calibration(data=seed_data)
    run_oasis()

    return


if __name__ == '__main__':
    mkdir(save_path)
    main()

#####
