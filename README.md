<!-- ![](res/oasis_logo_1.png "title") -->
# Orbiting Mass Assignment Scheme

When using this code please cite Salazar et. al. (2025) ([arXiv:XXXX.XXXX]()).

## Requirements
Create a conda environment. An `environment.yml` file is supplied as an example but feel free to use your own. It must include `mpi4py` and `python>=3.10`.
```sh
$ conda env create -f environment.yml
```
Alternatively
```sh
$ conda create --name <env name> python=3.10 mpi4py
```

## Installation
Build from source
```sh
$ git clone https://github.com/edgarmsalazar/oasis.git
$ cd oasis
$ python -m pip install .
```

## Usage
A [template](template.py) file is provided as an example of how to use the code. It is very simple and it consists of four basic steps: preparing data, calibrating and runnning.

### User parameters
The choice of units is not strict to the ones shown below. However, it must be consisten across all quantities with the same physical dimensions. For example, if the units for the `boxsize` are in $h^{-1}{\rm Mpc}$, then all disntances, positions and radii must also be provided in $h^{-1}{\rm Mpc}$.

- `boxsize`: side length of the simulation box in units of $h^{-1}{\rm Mpc}$.
- `minisize`: side length of the subvolume or minibox in units of $h^{-1}{\rm Mpc}$.
- `padding`: length of the padding distance form the edge of the minibox in unnits of $h^{-1}{\rm Mpc}$.
- `rhom`: mass density of the Universe in units of $h^{-2} M_{\odot} / {\rm Mpc}^{3}$.
- `part_mass`: dark matter particle mass in units of $h^{-2} M_{\odot}$.
- `save_path`: path to the directory where <span style="font-variant:small-caps;">Oasis</span> will save all data products.

### Step 1: Prepare data
<span style="font-variant:small-caps;">Oasis</span> does not assume any type or form of data storage like AREPO, Gadget, HDF, or any other, and leaves the task of loading data to memory to the user. The input data must include:

- Particles: ID, $\vec{x}$, $\vec{v}$.
- Seeds: ID, $\vec{x}$, $\vec{v}$, $M_{\rm 200b}$, $R_{\rm 200b}$, $R_{s}$.

where $R_{s}$ is the NFW scale radius, and $M_{\rm 200b}$ is [<span style="font-variant:small-caps;">Rockstar</span>](https://ui.adsabs.harvard.edu/abs/2013ApJ...762..109B/abstract) mass definition.

<span style="font-variant:small-caps;">Oasis</span> splits the simulation volume into  $\left\lceil\right.$`boxsize//minisize`$\left.\right\rceil^{3}$ miniboxes or subvolumes in order to process the full catalogue in parallel. For this reason it effectively duplicates the data in disc meaning the necessary free storage must be at least the same size of the input catalogues. Although this might not be an issue for most HPCs, please keep it in mind when running in personal computers. The current implementation also loads the full catalogue into RAM so it could be a problem on smaller systems.

There are two user parameters that the `split_box_into_mini_boxes` method takes to pay attention to which relate to how many seeds/particles are processed at a time when saving into miniboxes. The values for `chunksize` below are acceptable for a catalogue with $\sim 4\times10^{6}$ seeds and $1024^3$ particles. The actual values will depend on your own data.

- `chunksize_seed` = 100_000
- `chunksize_part` = 10_000_000

Here is an example on how to run seed preparation.
```python
import numpy as np
from oasis.minibox import split_box_into_mini_boxes

data = hid, pos, vel, r200b, m200b, rs

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
```

The same function is used for particles with the only exception that `props=None` and `name=part`.

### Step 2: Calibration
Before running <span style="font-variant:small-caps;">Oasis</span> on the full volume, it needs to be calibrated. That is find the cut line in $\ln v^2-r$ space that classifies particles into orbiting and infalling. Below are the parameters for the `calibrate` method and example.

- `n_seeds`: number of seeds to load.
- `r_max`: search radius in units of $h^{-1}{\rm Mpc}$.. All particles within this radius will be collected for calibration.
- `calib_p`: calibration parameter for $v_r>0$. Sets the target fraction of particles below the cut line. Defaults to 0.995.
- `calib_w`: calibration parameter for $v_r<0$. Sets the width of the band around the cut line. Defaults to 0.050.
- `calib_n_points`: number of gradient points to use when finding the slope of the cut line. Defaults to 20.
- `calib_grad_lims`: radial interval where the gradient points are taken from. Defaults to (0.2, 0.5) in units of $r/R_{\rm 200b}$.
- `n_threads`: number of multiprocessing threads to use. Speeds up loading all seeds from each distinct minibox. Defaults to `None`.

```python
from oasis.calibration import calibrate

data = hid, pos, vel, m200b, r200b

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
```

> Input data order matters: ID, $\vec{x}$, $\vec{v}$, $M_{\rm 200b}$, $R_{\rm 200b}$.

As a recommendation, always check that the calibration was done properly and makes sense. Here is the output of the previous function call.

<img src="res/calibration.png" alt="calibration" class="center" height="300"/>

### Step 3: Run orbiting mass assingment
Once calibrated, you can simply call `run_orbiting_mass_assignment` to generate a dynamical halo catalogue and the members catalogue. The additional parameters to set are:

- `n_orb_min`: only dynamical haloes with at least this number of orbiting particles will be considered.
- `fast_mass`: when `True`,  <span style="font-variant:small-caps;">Oasis</span> will perform a simple percolaition. It speeds things up but results are only comparable to the full percolation at the mass function level, i.e. members will differ from _true_ percolation. Defaults to `False`.
- `run_name`: <span style="font-variant:small-caps;">Oasis</span> appends this name to identify the results and distinguish them from subsequent runs in the same box.

```python
from oasis.catalogue import run_orbiting_mass_assignment

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
```
