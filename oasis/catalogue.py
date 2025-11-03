import os
from functools import partial
from multiprocessing import Pool
from warnings import filterwarnings

import h5py as h5
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.spatial import cKDTree
from tqdm import tqdm

from oasis.common import G_GRAVITY, ensure_dir_exists
from oasis.coordinates import relative_coordinates
from oasis.minibox import load_particles, load_seeds

filterwarnings('ignore')


class MiniBoxClassifier:
    """
    Parallel-friendly wrapper for classify_single_mini_box().
    Each instance handles exactly == one mini box ==.
    """

    def __init__(
        self,
        mini_box_id: int,
        min_num_part: int,
        boxsize: float,
        minisize: float,
        load_path: str,
        run_name: str,
        particle_type: str,
        padding: float = 5.0,
        fast_mass: bool = False,
        disable_tqdm: bool = True,
    ):
        # Input parameters
        self.mini_box_id = mini_box_id
        self.min_num_part = min_num_part
        self.boxsize = boxsize
        self.minisize = minisize
        self.load_path = load_path
        self.run_name = run_name
        self.particle_type = particle_type
        self.padding = padding
        self.fast_mass = fast_mass
        self.disable_tqdm = disable_tqdm

        # Internal parameters
        self.save_path = None
        # Seed properties
        self.pos_seed = self.vel_seed = self.hid = None
        self.r200b = self.m200b = self.rs = self.mask_mb = None
        # Particle properties
        self.pos_part = self.vel_part = self.pid_part = self.mass_part = None
        # Classification parameters
        self.pars = None
        self.deltac = None
        self.n_seeds = 0

    # ==========================================================================
    # Phase 1: Setup and loading
    # ==========================================================================
    def _make_output_dir(self):
        self.save_path = self.load_path + f"run_{self.run_name}/mini_box_catalogues/"
        ensure_dir_exists(self.save_path)

    def _load_seeds_and_filter(self):
        (
            self.pos_seed,
            self.vel_seed,
            self.hid,
            self.r200b,
            self.m200b,
            self.rs,
            self.mask_mb,
        ) = load_seeds(
            self.mini_box_id,
            self.boxsize,
            self.minisize,
            self.load_path,
            self.padding,
        )

        if self.fast_mass:
            m200b_mask = self.m200b > (5.0 * np.min(self.m200b))

            self.pos_seed = self.pos_seed[m200b_mask]
            self.vel_seed = self.vel_seed[m200b_mask]
            self.hid = self.hid[m200b_mask]
            self.r200b = self.r200b[m200b_mask]
            self.m200b = self.m200b[m200b_mask]
            self.rs = self.rs[m200b_mask]
            self.mask_mb = self.mask_mb[m200b_mask]

        self.n_seeds = len(self.hid)

    def _early_exit_if_no_seeds(self):
        if not any(self.hid):
            return True
        return False

    def _compute_deltac(self):
        """Computes the characteristic density of an NFW profile.

        Parameters
        ----------
        r200 : float
            Halo radius
        rs : float
            Scale radius

        Returns
        -------
        float
            Delta characteristic

        Raises
        ------
        ZeroDivisionError
            If `r200` or `rs` are zero.
        """
        if np.any(self.rs == 0.) or np.any(self.r200b == 0.):
            raise ZeroDivisionError('Neither r200 nor rs can be zero.')

        c200 = self.r200b / self.rs
        self.deltac = (200./3.) * (c200 ** 3 / (np.log(1 + c200) - (c200 / (1 + c200))))

    @staticmethod
    def _rho_nfw_roots(
        x: float,
        delta1: float,
        rs1: float,
        delta2: float,
        rs2: float,
        r12: float,
    ):
        """Returns the value of 

        \begin{equation*}
            \frac{\rho_1(R-r)}{\rho_{c}} &= \frac{\rho_2(r)}{\rho_{c}}
        \end{equation*}

        where $\rho(r)$ is the NFW profile.

        \begin{equation*}
            \frac{\rho(r)}{\rho_{c}} = 
                \frac{\delta_c}{\frac{r}{R_s}\left(1+\frac{r}{R_s}\right)^2}
        \end{equation*}

        Parameters
        ----------
        x : float
            Radial coordinate
        delta1 : float
            Characteristic density of the central object
        rs1 : float
            Scale radius of the central object
        delta2 : float
            Characteristic density of the substructure
        rs2 : float
            Scale radius of the substructure
        r12 : float
            Radial separation between central and substructure R=|x2-x1|.

        Returns
        -------
        float

        """
        x1 = (r12 - x) / rs1
        x2 = x / rs2
        frac1 = delta1 / (x1 * (1 + x1)**2)
        frac2 = delta2 / (x2 * (1 + x2)**2)
        return frac1 - frac2

    def _load_particles(self):
        (
            self.pos_part,
            self.vel_part,
            self.pid_part,
            self.mass_part,
        ) = load_particles(
            self.mini_box_id,
            self.boxsize,
            self.minisize,
            self.load_path,
            self.particle_type,
            self.padding,
        )
        self.position_tree = cKDTree(self.pos_part, boxsize=self.boxsize)

    def _load_calibration_parameters(self):
        with h5.File(self.load_path + "calibration_pars.hdf5", "r") as hdf:
            self.pars = (
                *hdf["pos"][()],
                *hdf["neg/line"][()],
                *hdf["neg/quad"][()],
            )

    def _init_catalogue_dataframe(self):
        col_names = (
            "Halo_ID",
            "pos",
            "vel",
            "R200b",
            "M200b",
            "Morb",
            "Norb",
            "LIDX",
            "RIDX",
            "INMB",
            "NSUBS",
            "PID",
            "SLIDX",
            "SRIDX",
        )
        
        # Empty dataframe with column names
        self.haloes = pd.DataFrame(columns=col_names)
        self.haloes_perc = pd.DataFrame(columns=col_names)

        self.orb_hid = []
        self.orb_pid = []
        self.orb_mass = []
        self.n_tot_p = 0
        self.n_tot_s = 0

        # If not -1 then seed is an orbiting structure
        self.parent_id_seed = np.full(self.n_seeds, -1, dtype=int)

    # ==========================================================================
    @staticmethod
    def _classify(
        rel_pos: np.ndarray,
        rel_vel: np.ndarray,
        r200: float,
        m200: float,
        class_pars: list | tuple | np.ndarray,
        max_radius: float = 2.0,
        pivot_radius: float = 0.5
    ) -> np.ndarray:
        """Classifies particles as orbiting.

        Parameters
        ----------
        rel_pos : np.ndarray
            Relative position of particles around seed
        rel_vel : np.ndarray
            Relative velocity of particles around seed
        r200 : float
            Seed R200
        m200 : float
            Seed M200
        class_pars : Union[List, Tuple, np.ndarray]
            Classification parameters [m_pos, b_pos, m_neg, b_neg]
        max_radius : float
            Maximum radius where orbiting particles can be found. All particles 
            above this value are set to be infalling. By default 2.0.
        pivot_radius : float
            Pivot value for the cut transition from linear to quadratic. By default
            0.5.

        Returns
        -------
        np.ndarray
            A boolean array where True == orbiting
        """
        m_pos, b_pos, m_neg, b_neg, alpha, beta, gamma = class_pars
        # Compute V200
        v200 = G_GRAVITY * m200 / r200

        # Compute the radius to seed_i in r200 units, and ln(v^2) in v200 units
        part_ln_vel = np.log(np.sum(np.square(rel_vel), axis=1) / v200)
        part_radius = np.sqrt(np.sum(np.square(rel_pos), axis=1)) / r200

        # Create a mask for particles with positive radial velocity
        mask_vr_positive = np.sum(rel_vel * rel_pos, axis=1) > 0

        # Orbiting classification for vr > 0
        line = m_pos * (part_radius - pivot_radius) + b_pos
        mask_cut_pos = part_ln_vel < line

        # Orbiting classification for vr < 0
        mask_small_radius = part_radius <= pivot_radius
        curve = alpha * part_radius ** 2 + beta * part_radius + gamma
        line = m_neg * (part_radius - pivot_radius) + b_neg

        mask_cut_neg = ((part_ln_vel < curve) & mask_small_radius) ^ \
            ((part_ln_vel < line) & ~mask_small_radius)

        # Particle is infalling if it is below both lines and 2*R00
        mask_orb = (part_radius <= max_radius) & (
            (mask_cut_pos & mask_vr_positive) ^
            (mask_cut_neg & ~mask_vr_positive)
        )

        return mask_orb


    def _classify_particles(self, i: int):
        """
        Perform classification for one seed (halo candidate).
        """
        # Select all particles around the seed
        within_r200b = self.position_tree.query_ball_point(self.pos_seed[i], 
                                                           2.*self.r200b[i],
                                                           p=np.inf,
                                                           return_sorted=True)
        # Skip if not enough particles within R200b
        if len(within_r200b) < self.min_num_part:
            return None
        
        # Relative coordinates of particles w.r.t seed position
        rel_pos = relative_coordinates(self.pos_part[within_r200b], 
                                       self.pos_seed[i], self.boxsize)
        rel_vel = self.vel_part[within_r200b] - self.vel_seed[i]

        # Classify particles around the seed.
        orb_mask = self._classify(rel_pos, rel_vel, self.r200b[i], self.m200b[i],
        self.pars)
        
        # Ignore seed if it does not have the minimum mass to be considered a
        # halo. Early exit to avoid further computation for a non-halo seed.
        if orb_mask.sum() < self.min_num_part:
            return None

        return (within_r200b, orb_mask)

    def _apply_6d_ball(
        self, 
        i: int, 
        within_r200b: np.ndarray, 
        orb_mask: np.ndarray,
    ):
        """
        Apply 6D ball criterion for orbiting structures
        """
        # Only work with less massive seeds within a 2*R200b sphere.
        rel_pos = relative_coordinates(self.pos_seed, self.pos_seed[i], 
                                       self.boxsize)
        r_max = 2.0 * self.r200b[i]
        mask_seed = (np.sum(np.square(rel_pos[i+1:]), axis=1) <= r_max**2) & \
            (self.parent_id_seed[i+1:] == -1)

        # Skip if not enough seeds within 2*R200b
        n_seeds_near = mask_seed.sum()
        if n_seeds_near <= 0:
            return 0

        # Select seeds in the vicinity
        rel_pos = rel_pos[i+1:][mask_seed]
        pos_near = self.pos_seed[i+1:][mask_seed]
        vel_near = self.vel_seed[i+1:][mask_seed]
        deltac_near = self.deltac[i+1:][mask_seed]
        m200b_near = self.m200b[i+1:][mask_seed]
        r200b_near = self.r200b[i+1:][mask_seed]
        rs_near = self.rs[i+1:][mask_seed]
        hid_near = self.hid[i+1:][mask_seed]

        # Loop over other seeds
        j = 0
        is_halo = True
        orb_seeds = []
        orb_mask_new = np.copy(orb_mask)

        while is_halo and (j < n_seeds_near):
            # Select particles around jth seed.
            rel_pos_part = relative_coordinates(self.pos_part[within_r200b],
                                                pos_near[j], self.boxsize)
            rel_vel_part = self.vel_part[within_r200b] - vel_near[j]
            rp_sq = np.sum(np.square(rel_pos_part), axis=1)
            vp_sq = np.sum(np.square(rel_vel_part), axis=1)

            # Distance from the current seed to the substructure.
            r_ij = np.linalg.norm(rel_pos[j])
            
            # Defines the search radius of the 6D ball. Distance from the 
            # substructure where the NFW density of both objects is equal.
            r_ball = fsolve(
                func=self._rho_nfw_roots,
                # Start at half the distance bewteen seeds.
                x0=0.5*r_ij,
                args=(
                    self.deltac[i], 
                    self.rs[i], 
                    deltac_near[j], 
                    rs_near[j], 
                    r_ij
                ))
            r_ball = np.min([r_ball[0], r200b_near[j]])

            # Defines the search velocity  of the 6D ball.
            v_ball_sq = 2**2 * G_GRAVITY * m200b_near[j] / r200b_near[j]

            # Check the fraction of orbiting particles in the 6D ball
            ball6d = (rp_sq <= r_ball**2) & (vp_sq <= v_ball_sq)
            # Compare to the original orbiting population.
            frac_inside = (ball6d * orb_mask_new).sum() / ball6d.sum()

            # If more than half the particles in the vicinity of the seed are 
            # orbiting, the seed is tagged as orbiting. The seed is infalling 
            # otherwise and all the particles within the box are tagged as 
            # infalling too.
            upper_threshold = 1. - np.exp(-(r_ij / r200b_near[j])**2)
            f_threshold = np.max([0.5, upper_threshold])

            if frac_inside >= f_threshold:
                orb_seeds.append(hid_near[j])
                orb_mask_new[ball6d] = True
            else:
                orb_mask_new[ball6d] = False

            # Check wether seed is still a halo.
            is_halo = orb_mask_new.sum() >= self.min_num_part

            # Next item.
            j += 1

        if is_halo:
            return (orb_mask_new, orb_seeds)
        else:
            return None

    def _classify_single_seed(self, i: int):
        """
        Full classification scheme for a single seed
        """
        # Classify particles around the seed
        result_1 = self._classify_particles(i=i)
        if result_1 is None:
            return None
        within_r200b, orb_mask = result_1

        # Classify seeds in the vicinity as orbiting structures
        result_2 = self._apply_6d_ball(i, within_r200b, orb_mask)
        if result_2 == 0:
            orb_mask_final = orb_mask
            n_subs = 0
            orb_seeds = []
        elif result_2 is None:
            return None
        else:
            # Un-pack results
            orb_mask_final, orb_seeds = result_2
            
            # Set parent halo ID for seeds (these are no longer free).
            mask_subs = np.isin(self.hid[i+1:], orb_seeds)
            self.parent_id_seed[i+1:][mask_subs] = self.hid[i]
            n_subs = mask_subs.sum()

        # Compute orbiting mass, and append orbiting objects to global lists
        n_orb = orb_mask_final.sum()
        if isinstance(self.mass_part, np.ndarray):
            morb = np.sum(self.mass_part[within_r200b][orb_mask_final])
            self.orb_mass.append(self.mass_part[within_r200b][orb_mask_final])
        else:
            morb = n_orb * self.mass_part
            self.orb_mass.append(np.full(n_orb, self.mass_part))
        
        self.orb_hid.append(orb_seeds)
        self.orb_pid.append(self.pid_part[within_r200b][orb_mask_final])
        
        # Build the row
        row = dict(
            Halo_ID=int(self.hid[i]),
            pos=self.pos_seed[i],
            vel=self.vel_seed[i],
            R200b=float(self.r200b[i]),
            M200b=float(self.m200b[i]),
            Morb=float(morb),
            Norb=int(n_orb),
            LIDX=int(self.n_tot_p),
            RIDX=int(self.n_tot_p + n_orb),
            INMB=bool(self.mask_mb[i]),
            NSUBS=int(n_subs),
            PID=self.parent_id_seed[i],
            SLIDX=int(self.n_tot_s),
            SRIDX=int(self.n_tot_s + n_subs),
        )

        # Update running counters for packed outputs
        self.n_tot_p += n_orb
        self.n_tot_s += n_subs

        return row

    def _process_all_seeds(self):
        """
        Main loop over all seeds in mini-box.
        """
        results = []
        for i in tqdm(range(self.n_seeds), ncols=100, desc='Finding haloes',
                      colour='green', disable=self.disable_tqdm):
            row = self._classify_single_seed(i)
            # print(i, row is None)
            if row is not None:
                results.append(row)
                

        if results:
            df = pd.DataFrame(results)
            self.haloes = pd.concat([self.haloes, df], ignore_index=True)
        
        # Sort haloes by their orbiting mass
        self.haloes.sort_values(by='Morb', ascending=False, inplace=True, 
                                ignore_index=True)

        self.orb_hid = np.concatenate(self.orb_hid).astype(self.hid[0].dtype)
        self.orb_pid = np.concatenate(self.orb_pid).astype(self.pid_part[0].dtype)
        self.orb_mass = np.concatenate(self.orb_mass)

    # ==========================================================================
    def _percolation(self):
        """
        This pass ensures each particle/seed only counts as orbiting the most
        massive host it belongs to, and recomputes orbital masses accordingly.
        Produces:
            self.haloes_perc, self.orb_pid_perc, self.orb_hid_perc
        """

        orb_pid_perc = []
        orb_hid_perc = []
        n_tot_perc = 0
        n_tot_s_perc = 0

        # Use sets to speed up isin queries
        orb_pid_seen = set()
        orb_hid_seen = set()

        for i in tqdm(range(len(self.haloes.index)), ncols=100, 
                      desc='Percolating particles', colour='green', 
                      disable=self.disable_tqdm):

            lidx = self.haloes["LIDX"][i]
            ridx = self.haloes["RIDX"][i]
            slidx = self.haloes["SLIDX"][i]
            sridx = self.haloes["SRIDX"][i]

            # Select particles not orbiting anything more massive
            pid_range = self.orb_pid[lidx:ridx]
            new_orb = [p for p in pid_range if p not in orb_pid_seen]
            orb_pid_seen.update(new_orb)
            n_orb = len(new_orb)
            
            # Skip to next seed if no longer a halo
            if n_orb < self.min_num_part:
                continue

            # Compute final orbiting mass
            mask = np.isin(pid_range, np.array(new_orb))
            morb = np.sum(np.array(self.orb_mass[lidx:ridx])[mask])

            # Select seeds not orbiting anything more massive.
            hid_range = self.orb_hid[slidx:sridx]
            new_orb_s = [h for h in hid_range if h not in orb_hid_seen]
            orb_hid_seen.update(new_orb_s)
            n_orb_s = len(new_orb_s)

            # Ignore seed if it is not within the current mini-box
            if not self.haloes["INMB"][i]:
                continue

            orb_pid_perc.append(new_orb)
            orb_hid_perc.append(new_orb_s)

            self.haloes_perc.loc[len(self.haloes_perc.index)] = [
                self.haloes["Halo_ID"][i],
                self.haloes["pos"][i],
                self.haloes["vel"][i],
                self.haloes["R200b"][i],
                self.haloes["M200b"][i],
                morb,
                n_orb,
                n_tot_perc,
                n_tot_perc + n_orb,
                True,
                n_orb_s,
                self.haloes["PID"][i],
                n_tot_s_perc,
                n_tot_s_perc + n_orb_s,
            ]

            n_tot_perc += n_orb
            n_tot_s_perc += n_orb_s

        # Concatenate lists into arrays
        self.orb_pid_perc = np.concatenate(orb_pid_perc) if orb_pid_perc else np.array([])
        self.orb_hid_perc = np.concatenate(orb_hid_perc) if orb_hid_perc else np.array([])
        
        # If no haloes where fuond in this mini-box
        if len(self.haloes_perc.index) <= 0:
            self.haloes_perc = None

    # ==========================================================================
    def _save_catalogues(self):
        """
        Writes halo and member catalogues to an HDF5 file in `self.save_path`.
        """

        full_path = self.save_path + f"{self.mini_box_id}.hdf5"
        dtypes = (
            np.uint32, np.float32, np.float32, np.float32, np.float32, np.float32,
            np.uint32, np.uint32, np.uint32, None, np.int32, np.uint32, np.uint32,
            np.uint32
        )
        with h5.File(full_path, "w") as hdf:
            # Halo catalogue
            for i, key in enumerate(self.haloes_perc.columns):
                if key == 'INMB':
                    continue
                if key in ['pos', 'vel']:
                    data = np.stack(self.haloes_perc[key].values)
                else:
                    data = self.haloes_perc[key].values
                hdf.create_dataset(f'halo/{key}', data=data, dtype=dtypes[i])

            # Particles
            hdf.create_dataset('memb/PID', data=self.orb_pid_perc,
                                dtype=self.orb_pid_perc[0].dtype)

            # Seeds
            hdf.create_dataset('memb/Halo_ID', data=self.orb_hid_perc,
                               dtype=self.orb_hid_perc[0].dtype)

    def run(self):
        """Main loop"""
        self._make_output_dir()
        self._load_seeds_and_filter()

        if self._early_exit_if_no_seeds():
            return None
        
        self._compute_deltac()
        self._load_particles()
        self._load_calibration_parameters()
        self._init_catalogue_dataframe()
        self._process_all_seeds()
        self._percolation()
        self._save_catalogues()
        
        return


def process_minibox(i, **kwargs):
    """Calls mini-box classifier for parallel computing"""
    classifier = MiniBoxClassifier(mini_box_id=i, **kwargs)
    classifier.run()
    return


def process_all_miniboxes(
    load_path: str,
    run_name: str,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    padding: float,
    particle_type: str,
    fast_mass: bool = False,
    n_threads: int = None,
) -> None:
    # Create directory if it does not exist
    save_path = load_path + f'run_{run_name}/mini_box_catalogues/'
    ensure_dir_exists(save_path)

    # Number of miniboxes
    n_mini_boxes = np.int_(np.ceil(boxsize / minisize))**3
    
    # Cap the number of threads to the total number of mini-boxes to process
    if n_threads is None:
        n_threads = min(max(1, os.cpu_count()//2), n_mini_boxes)
    else:
        n_threads = min(n_threads, n_mini_boxes)

    # Parallel processing of miniboxes.
    func = partial(
        process_minibox, 
        min_num_part=min_num_part,
        boxsize=boxsize, 
        minisize=minisize, 
        load_path=load_path,
        run_name=run_name, 
        particle_type=particle_type, 
        padding=padding, 
        fast_mass=fast_mass, 
        disable_tqdm=True,
    )
    
    # Safely handle multiprocessing falure with a fall back to a single thread.
    if n_threads > 1:
        try:
            with Pool(n_threads) as pool, \
                tqdm(total=n_mini_boxes, colour="green", ncols=100,
                    desc='Generating halo catalogue') as pbar:
                for _ in pool.imap(func, range(n_mini_boxes)):
                    pbar.update()
        except Exception as e:
            print(f"Warning: Parallel processing failed ({e}), falling back to"
                  " sequential")
            # Fall back to sequential processing
            n_threads = 1
    
    if n_threads == 1:
        for box_i in tqdm(range(n_mini_boxes), colour="green", ncols=100,
                        desc='Generating halo catalogue'):
            func(box_i)

    return None


def merge_catalogues(
    load_path: str,
    run_name: str,
    n_mini_boxes: int,
) -> None:
    save_path = load_path + f'run_{run_name}/mini_box_catalogues/'

    # Find dataset keys from file
    first_file = None
    for i in range(n_mini_boxes):
        file_path = os.path.join(save_path, f"{i}.hdf5")
        if os.path.exists(file_path):
            first_file = file_path
            break
    if first_file is None:
        raise FileNotFoundError("No mini-box catalogue files found.")
    
    with h5.File(first_file, 'r') as hdf_load:
        halo_keys = list((hdf_load['halo'].keys()))
    
    # Load and concatenate data
    halo_data = {key: [] for key in halo_keys}

    n_part, n_seed = 0, 0
    hdf_memb = h5.File(load_path + f'run_{run_name}/members.hdf5', 'w')
    for i in tqdm(range(n_mini_boxes), ncols=100, desc='Merging catalogues',
                  colour='green'):
        
        # Check if current file exists
        file_path = os.path.join(save_path, f"{i}.hdf5")
        if not os.path.exists(file_path):
            continue

        with h5.File(file_path, 'r') as hdf_load:
            if 'halo' not in hdf_load.keys():
                continue
            # Member data ======================================================
            # Number of particles in current file
            n_part_this = hdf_load['memb/PID'].shape[0]
            n_seed_this = hdf_load['memb/Halo_ID'].shape[0]

            # This reshaping of the dataset after every new file...
            if first_file:  # Create the dataset at first pass.
                hdf_memb.create_dataset(name='PID',
                                        chunks=True, maxshape=(None,),
                                        data=hdf_load['memb/PID'][()])
                hdf_memb.create_dataset(name='Halo_ID',
                                        chunks=True, maxshape=(None,),
                                        data=hdf_load['memb/Halo_ID'][()])
                first_file = False
            else:
                # Number of particles so far plus this file's total.
                new_shape = n_part + n_part_this
                # Resize axes and save incoming data
                hdf_memb['PID'].resize((new_shape), axis=0)
                hdf_memb['PID'][n_part:] = hdf_load['memb/PID'][()]

                # Number of seeds so far plus this file's total.
                new_shape = n_seed + n_seed_this
                # Resize axes and save incoming data
                hdf_memb['Halo_ID'].resize((new_shape), axis=0)
                hdf_memb['Halo_ID'][n_seed:] = \
                    hdf_load['memb/Halo_ID'][()]

            # Halo data ========================================================
            for key in halo_keys:
                data = hdf_load[f"halo/{key}"][()]
                # Offset indices by number of objects in this file
                if key in {"LIDX", "RIDX"}:
                    data += n_part
                elif key in {"SLIDX", "SRIDX"}:
                    data += n_seed
                halo_data[key].append(data)

            # Add the total number of particles in this file to the next.
            n_part += n_part_this
            n_seed += n_seed_this

    hdf_memb.close()

    # Set the seed index to -1 for all those haloes without subhaloes.
    slidx = np.concatenate(halo_data.get("SLIDX", []), dtype=np.int32)
    sridx = np.concatenate(halo_data.get("SRIDX", []), dtype=np.int32)
    mask = (sridx-slidx == 0)
    slidx[mask] = -1
    sridx[mask] = -1

    with h5.File(load_path + f'run_{run_name}/catalogue.hdf5', 'w') as hdf:
        hdf.create_dataset('SLIDX', data=slidx)
        hdf.create_dataset('SRIDX', data=sridx)
        for key, chunks in halo_data.items():
            if key in {'SLIDX', 'SRIDX'}:
                continue
            hdf.create_dataset(key, data=np.concatenate(chunks))
    
    return None


def run_orbiting_mass_assignment(
    load_path: str,
    run_name: str,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    padding: float,
    particle_type: str,
    fast_mass: bool = False,
    n_threads: int = None,
    cleanup: bool = False,
) -> None:
    """Generates a halo catalogue using the kinetic mass criterion to classify
    particles into orbiting or infalling.

    Parameters
    ----------
    load_path : str
        Location from where to load the file
    min_num_part : int
        Minimum number of particles needed to be considered a halo
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    run_name : str
        Label for the current run. The directory created will be `run_<run_name>`
    padding : float, optional
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5
    n_threads : int
        Number of threads, by default None
    cleanup : bool
        Removes individual minibox catalogues after contatenation.

    Returns
    -------
    None
    """
    process_all_miniboxes(
        load_path=load_path,
        run_name=run_name,
        min_num_part=min_num_part,
        boxsize=boxsize,
        minisize=minisize,
        padding=padding,
        particle_type=particle_type,
        fast_mass=fast_mass,
        n_threads=n_threads,
    )

    n_mini_boxes = np.int_(np.ceil(boxsize / minisize))**3
    merge_catalogues(
        load_path=load_path,
        run_name=run_name,
        n_mini_boxes=n_mini_boxes,
    )

    save_path = load_path + f'run_{run_name}/mini_box_catalogues/'
    if cleanup:
        for item in os.listdir(save_path):
            os.remove(save_path + item)
        os.removedirs(save_path)

    return None


###
