"""Template file for running OASIS: Orbiting mAss asSIngment Scheme

All loading and processing of simulation data and catalogues is assumed to be 
done before running OASIS. This is done to avoid adding dependencies specific
any simulation and simplifying the API.

"""
import numpy as np

from oasis.calibration import calibrate
from oasis.catalogue import run_orbiting_mass_assignment
from oasis.common import G_GRAVITY, ensure_dir_exists, timer
from oasis.minibox import process_simulation_data

# ==============================================================================
#
#                               Simulation parameters
#
#   These are EXAMPLE values. Change them to your simulation parameters BEFORE
#   running OASIS or you will get wrong results.
# ==============================================================================
boxsize: float = 1000.                              # h^-1 Mpc
minisize: float = 100.                              # h^-1 Mpc
particle_mass: float = 77546570000.0                # h^-1 M_sun
padding: float = 5.                                 # h^-1 Mpc

# Cosmological parameters.
omega_m: float = 0.3
redshift: float = 0.
h: float = 0.67

# This is an example for z=0. Modify H(z)^2 accordingly or calculate the mass 
# density externally.
Hz_sq: float = (100. * h)**2
critical_density: float = 3. * Hz_sq / 8. / np.pi / G_GRAVITY # h^-2 M_sun / Mpc^3
mass_density: float = omega_m * critical_density              # h^-2 M_sun / Mpc^3

# ==============================================================================
#
#                               OASIS configuration
#
# ==============================================================================
# OASIS will save results to this path.
save_path: str = '/'

# Calibration.
n_seeds: int = 500
r_max: float = 5.0                                          # h^-1 Mpc
calib_n_points: int = 20
calib_vrp_percent: float = 0.995
calib_vrn_width: float = 0.050
calib_gradient_radial_limits: tuple[float] = (0.2, 0.5)
n_threads: int = 50

# Catalogue.
n_orb_min: int = 200
fast_mass: bool = False
run_name = '<cool name>'        # OASIS appends this name to identify the run.
                                # This is useful to generate catalogues for the 
                                # same simulation box with different parameter
                                # combinations.

# ==============================================================================
#
#                               OASIS sample run
#
# ==============================================================================
def seed_data() -> tuple[np.ndarray]:
    # LOAD YOUR DATA HERE
    pos, vel, m200b, r200b, hid, rs = ()
    return pos, vel, m200b, r200b, hid, rs


def particle_data() -> tuple[np.ndarray]:
    # LOAD YOUR DATA HERE
    pid, pos, vel = ()
    return pid, pos, vel


@timer(fancy=False)
def process_seeds() -> None:
    # Load your data.
    pos, vel, r200b, m200b, hid, rs = seed_data()

    # Additional properties to include in seed catalogue.
    props = [r200b, rs]
    labels = ('R200b', 'Rs')
    dtypes = (np.float32, np.float32)
    props_zip = (props, labels, dtypes)

    # Save seeds into miniboxes according to their minibox ID.
    process_simulation_data(
        save_path=save_path,
        particle_type='seed',
        boxsize=boxsize,
        minisize=minisize,
        positions=pos,
        velocities=vel,
        ids=hid,
        mass=(m200b, 'M200b'),
        data=props,
        n_threads=n_threads,
    )
    return None


@timer(fancy=False)
def process_particles() -> None:
    # Load your data.
    pid, pos, vel = particle_data()

    # Save particles into miniboxes according to their minibox ID.
    process_simulation_data(
        save_path=save_path,
        particle_type='dm',
        boxsize=boxsize,
        minisize=minisize,
        positions=pos,
        velocities=vel,
        ids=pid,
        mass=(particle_mass, 'mass'),
        n_threads=n_threads,
    )

    return


@timer(fancy=False)
def calibration() -> None:

    # OPTION 1 ======================================================
    # Assume cosmological dependence on calibration parameters.
    calibrate(
        save_path=save_path,
        omega_m=omega_m,
    )

    # OPTION 2 ======================================================
    # Calibrate finder on simulation data directly.
    
    # Load your data. No `rs` needed here.
    *data, _, _ = seed_data()

    # Calibrate OASIS
    calibrate(
        n_seeds=n_seeds,
        seed_data=data,
        r_max=r_max,
        boxsize=boxsize,
        minisize=minisize,
        save_path=save_path,
        particle_type='dm',
        particle_mass=particle_mass,
        mass_density=mass_density,
        redshift=redshift,
        percent=calib_vrp_percent,
        width=calib_vrn_width,
        gradient_radial_lims=calib_gradient_radial_limits,
        n_threads=n_threads,
        overwrite=False,
    )

    return None


@timer(fancy=False)
def run_oasis():

    run_orbiting_mass_assignment(
        load_path=save_path,
        min_num_part=n_orb_min,
        boxsize=boxsize,
        minisize=minisize,
        run_name=run_name,
        padding=padding,
        fast_mass=fast_mass,
        part_mass=particle_mass,
        n_threads=n_threads,
    )

    return


@timer(fancy=False)
def main():

    process_seeds()
    process_particles()
    calibration()
    run_oasis()

    return


if __name__ == '__main__':
    ensure_dir_exists(save_path)
    main()

#####
