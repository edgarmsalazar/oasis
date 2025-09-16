from typing import Tuple, Union

import h5py
import numpy
from tqdm import tqdm

from oasis.common import ensure_dir_exists, get_min_unit_dtype
from oasis.coordinates import relative_coordinates


def get_mini_box_id(
    position: numpy.ndarray,
    boxsize: float,
    minisize: float,
) -> Union[int, numpy.ndarray]:
    """
    Returns the mini-box ID(s) to which the given coordinates fall into.

    This function divides a simulation box into smaller cubic mini-boxes and 
    determines which mini-box each position belongs to. Mini-boxes are indexed
    using a 3D grid system converted to unique 1D IDs.

    Parameters
    ----------
    position : numpy.ndarray
        Position(s) in cartesian coordinates. Can be:
        - 1D array of shape (3,) for a single position
        - 2D array of shape (N, 3) for N positions
        Each position should have [x, y, z] coordinates within [0, boxsize).
    boxsize : float
        Size of the cubic simulation box. Must be positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.

    Returns
    -------
    int or numpy.ndarray
        Mini-box ID(s). Returns:
        - int: if input was a single position (1D array)
        - numpy.ndarray: if input was multiple positions (2D array)

    Raises
    ------
    TypeError
        If position is not a numpy array or boxsize/minisize are not numeric.
    ValueError
        If minisize > boxsize, or if any coordinate is outside [0, boxsize),
        or if position array has wrong dimensions, or if boxsize/minisize ≤ 0.

    Notes
    -----
    - Positions exactly at the upper boundary (boxsize) are adjusted inward
      by a small tolerance to ensure they fall within valid mini-boxes.
    - The function modifies the input array in-place for memory efficiency.
    - The function uses a 3D-to-1D mapping: ID = i + j*nx + k*nx*ny
      where (i,j,k) are grid indices and nx, ny, nz are grid dimensions.

    Examples
    --------
    >>> import numpy as numpy
    >>> pos = numpy.array([1.5, 2.5, 3.5])
    >>> get_mini_box_id(pos, boxsize=10.0, minisize=2.0)
    32

    >>> positions = numpy.array([[1.5, 2.5, 3.5], [0.1, 0.1, 0.1]])
    >>> get_mini_box_id(positions, boxsize=10.0, minisize=2.0)
    array([32,  0])
    """
    EDGE_TOL = 1e-8

    # Input validation
    if not isinstance(position, numpy.ndarray):
        raise TypeError("position must be a numpy array")

    if not isinstance(boxsize, (int, float)) or not isinstance(minisize, (int, float)):
        raise TypeError("boxsize and minisize must be numeric")

    if boxsize <= 0:
        raise ValueError("boxsize must be positive")

    if minisize <= 0:
        raise ValueError("minisize must be positive")

    if minisize > boxsize:
        raise ValueError("minisize cannot be larger than boxsize")

    # Handle input dimensions
    if position.ndim == 1:
        if position.size != 3:
            raise ValueError("1D position array must have exactly 3 elements")
        position = position.reshape(1, 3)
        return_scalar = True
    elif position.ndim == 2:
        if position.shape[1] != 3:
            raise ValueError("2D position array must have shape (N, 3)")
        return_scalar = False
    else:
        raise ValueError(
            "position array must be 1D (shape (3,)) or 2D (shape (N, 3))")

    # Validate coordinate bounds
    if numpy.any(position < 0) or numpy.any(position > boxsize):
        raise ValueError(f"All coordinates must be within [0, {boxsize}]")

    # Pre-compute grid parameters
    n_cells_per_side = int(numpy.ceil(boxsize / minisize))
    shift = numpy.array([1, n_cells_per_side, n_cells_per_side**2])

    # Handle boundary conditions
    # Points at upper boundary (within numerical precision)
    upper_mask = numpy.abs(position - boxsize) < EDGE_TOL
    position[upper_mask] -= EDGE_TOL

    # Points at lower boundary (within numerical precision)
    lower_mask = numpy.abs(position) < EDGE_TOL
    position[lower_mask] += EDGE_TOL

    # Compute grid indices
    grid_indices = numpy.floor(position / minisize).astype(int)

    # Additional safety check after grid computation
    max_index = n_cells_per_side - 1
    if numpy.any(grid_indices < 0) or numpy.any(grid_indices > max_index):
        raise RuntimeError(
            "Grid index computation resulted in out-of-bounds indices")

    # Compute unique IDs using dot product for efficiency
    ids = numpy.sum(shift * grid_indices, axis=1)

    # Return appropriate type based on input
    if return_scalar:
        return int(ids[0])
    else:
        return ids


def get_adjacent_mini_box_ids(
    mini_box_id: Union[int, numpy.integer],
    boxsize: float,
    minisize: float,
) -> numpy.ndarray:
    """
    Returns all mini-box IDs adjacent to a specified mini-box, including itself.

    This function finds all 27 mini-boxes in the 3×3×3 neighborhood surrounding
    the given mini-box ID in a 3D grid. The neighborhood includes the center box
    itself and all 26 surrounding boxes. Periodic boundary conditions are applied,
    so boxes at the grid edges wrap around to the opposite side.

    Parameters
    ----------
    mini_box_id : int or numpy.integer
        ID of the central mini-box. Must be a valid ID within the grid
        (0 ≤ mini_box_id < total_mini_boxes).
    boxsize : float
        Size of the cubic simulation box. Must be positive.
    minisize : float
        Size of each cubic mini-box. Must be positive and ≤ boxsize.

    Returns
    -------
    numpy.ndarray
        Array of shape (27,) containing all adjacent mini-box IDs, including
        the input ID. The first element is always the input mini_box_id, 
        followed by the 26 neighboring IDs in no particular order.
        Array dtype is numpy.int32.

    Raises
    ------
    TypeError
        If mini_box_id is not an integer type, or if boxsize/minisize are not numeric.
    ValueError
        If minisize > boxsize, or if boxsize/minisize ≤ 0, or if mini_box_id
        is outside the valid range [0, total_mini_boxes-1].

    Notes
    -----
    - Uses periodic boundary conditions: mini-boxes at grid boundaries are
      considered adjacent to mini-boxes on the opposite boundary.
    - The 3D grid uses the mapping: ID = k + j×nx + i×nx², where (i,j,k) are
      grid coordinates and nx is the number of cells per side.
    - For a grid with n cells per side, total mini-boxes = n³.
    - The function always returns exactly 27 IDs regardless of grid size.

    Examples
    --------
    >>> import numpy as np
    >>> # 2×2×2 grid (8 total mini-boxes)
    >>> get_adjacent_mini_box_ids(0, boxsize=2.0, minisize=1.0)
    array([0, 1, 2, 3, 4, 5, 6, 7, ...], dtype=int32)  # All boxes due to wrapping

    >>> # Larger grid - interior box
    >>> ids = get_adjacent_mini_box_ids(111, boxsize=10.0, minisize=1.0)  # 10×10×10 grid
    >>> len(ids)
    27
    >>> ids[0]  # First element is always the input ID
    111

    See Also
    --------
    get_mini_box_id : Convert coordinates to mini-box ID
    """
    # Input validation
    if not isinstance(mini_box_id, (int, numpy.integer)):
        raise TypeError("mini_box_id must be an integer")

    if not isinstance(boxsize, (int, float)) or not isinstance(minisize, (int, float)):
        raise TypeError("boxsize and minisize must be numeric")

    if boxsize <= 0:
        raise ValueError("boxsize must be positive")

    if minisize <= 0:
        raise ValueError("minisize must be positive")

    if minisize > boxsize:
        raise ValueError("minisize cannot be larger than boxsize")

    # Convert to Python int to avoid numpy scalar issues
    mini_box_id = int(mini_box_id)

    # Grid parameters
    cells_per_side = int(numpy.ceil(boxsize / minisize))
    total_mini_boxes = cells_per_side**3
    max_id = total_mini_boxes - 1

    # Validate mini_box_id range
    if mini_box_id < 0:
        raise ValueError(
            f"mini_box_id must be non-negative, got {mini_box_id}")
    if mini_box_id > max_id:
        raise ValueError(
            f"mini_box_id {mini_box_id} exceeds maximum valid ID {max_id} "
            f"for grid with {cells_per_side}³ = {total_mini_boxes} mini-boxes"
        )

    # Convert 1D ID to 3D grid coordinates (i, j, k)
    # Using the mapping: ID = k + j*cells_per_side + i*cells_per_side²
    i = mini_box_id // (cells_per_side**2)
    remainder = mini_box_id % (cells_per_side**2)
    j = remainder // cells_per_side
    k = remainder % cells_per_side

    # Verify the coordinate conversion (safety check)
    reconstructed_id = k + j * cells_per_side + i * (cells_per_side**2)
    if reconstructed_id != mini_box_id:
        raise RuntimeError(
            f"Grid coordinate conversion failed: ID {mini_box_id} -> "
            f"coords ({i},{j},{k}) -> ID {reconstructed_id}"
        )

    # Pre-allocate array for all 27 adjacent IDs
    adjacent_ids = numpy.empty(27, dtype=numpy.int32)

    # First element is always the input ID
    adjacent_ids[0] = mini_box_id
    idx = 1

    # Generate 3×3×3 neighborhood with periodic boundary conditions
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                # Skip the center cell (already added)
                if di == 0 and dj == 0 and dk == 0:
                    continue

                # Apply periodic boundary conditions using modulo
                ni = (i + di) % cells_per_side
                nj = (j + dj) % cells_per_side
                nk = (k + dk) % cells_per_side

                # Convert 3D coordinates back to 1D ID
                neighbor_id = nk + nj * cells_per_side + \
                    ni * (cells_per_side**2)
                adjacent_ids[idx] = neighbor_id
                idx += 1

    return adjacent_ids


def split_simulation_into_mini_boxes(
    positions: numpy.ndarray,
    velocities: numpy.ndarray,
    uid: numpy.ndarray,
    save_path: str,
    boxsize: float,
    minisize: float,
    name: str = None,
    props: Tuple[list, list, list] = None
) -> None:
    """Sorts all items into mini boxes and saves them in disc.

    Parameters
    ----------
    positions : numpy.ndarray
        Cartesian coordinates
    velocities : numpy.ndarray
        Cartesian velocities
    uid : numpy.ndarray
        Unique IDs for each position (e.g. PID, HID)
    save_path : str
        Where to save the IDs
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None
    props : tuple[list(array), list(str), list(dtype)], optional
        Additional arrays to be sorted into mini boxes.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If `chunksize` is too small, the chunk-finding loop cannot properly 
        resolve all the mini box ids within a chunk.
    """
    # Determine number of partitions per side
    cells_per_side = int(numpy.ceil(boxsize / minisize))
    n_cells = cells_per_side**3

    # Compute mini box IDs for all items in chunks with size of the average number
    # of items per mini box to imporve computation speed and reduce memory usage.
    # This is important for large simulations with billions of particles.
    n_items = positions.shape[0]
    chunksize = n_items // n_cells

    uint_dtype = get_min_unit_dtype(n_cells)
    mini_box_ids = numpy.zeros(n_items, dtype=uint_dtype)
    for chunk in tqdm(range(n_cells), desc='Getting IDs', ncols=100, colour='blue'):
        low = chunk * chunksize
        if chunk < n_cells - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        mini_box_ids[low:upp] = get_mini_box_id(
            positions[low:upp], boxsize, minisize)

    # Sort data by mini box id
    mb_order = numpy.argsort(mini_box_ids)
    mini_box_ids = mini_box_ids[mb_order]
    velocities = velocities[mb_order]
    positions = positions[mb_order]
    uid = uid[mb_order]

    if props:
        labels = props[1]
        dtypes = props[2]
        props = props[0]
        for k, item in enumerate(props):
            props[k] = item[mb_order]

    # Get chunk indices finding the left-most occurence of all unique values in
    # a sorted array. This ensures that all items with the same mini box id are
    # in the same chunk and processed at the same time.
    unique_values = numpy.arange(n_cells, dtype=uint_dtype)
    chunk_idx = numpy.searchsorted(mini_box_ids, unique_values, side="left")

    # Get smallest data type to represent IDs
    uint_dtype_pid = get_min_unit_dtype(numpy.max(uid))
    if props:
        labels = ('ID', 'pos', 'vel', *labels)
        dtypes = (uint_dtype_pid, numpy.float32, numpy.float32, *dtypes)
    else:
        labels = ('ID', 'pos', 'vel')
        dtypes = (uint_dtype_pid, numpy.float32, numpy.float32)

    # Create target directory
    save_dir = save_path + f'mini_boxes_nside_{cells_per_side}/'
    ensure_dir_exists(save_dir)

    # For each chunk
    for i, mini_box_id in enumerate(tqdm(unique_values,
                                  desc='Saving mini-boxes',
                                  ncols=100, colour='blue')):
        # Select chunk
        low = chunk_idx[i]
        if i < n_cells - 1:
            upp = chunk_idx[i + 1]
        else:
            upp = None

        # mb_chunk = mini_box_ids[low: upp]
        pos_chunk = positions[low: upp]
        vel_chunk = velocities[low: upp]
        pid_chunk = uid[low: upp]

        if props:
            props_chunks = [None for _ in range(len(props))]
            for k, item in enumerate(props):
                props_chunks[k] = item[low: upp]

        with h5py.File(save_dir + f'{mini_box_id}.hdf5', 'a') as hdf:
            if props:
                data = (pid_chunk, pos_chunk, vel_chunk,
                    *[
                        item_chunk for item_chunk in props_chunks
                    ],
                )
            else:
                data = (pid_chunk, pos_chunk, vel_chunk)

            if (name is not None) and (not name in hdf.keys()):
                prefix = f'{name}/'
            else:
                prefix = ''

            for (label_i, data_i, dtype_i) in zip(labels, data, dtypes):
                hdf.create_dataset(name=prefix+f'{label_i}', data=data_i,
                                    dtype=dtype_i)

    return None


def load_particles(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float = 5.0,
) -> Tuple[numpy.ndarray]:
    """Load particles from a mini box including all particles in adjacent boxes 
    up to the `padding` distance.

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    load_path : str
        Location from where to load the file
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5.

    Returns
    -------
    Tuple[numpy.ndarray]
        Position, velocity, and PID
    """
    # Determine number of partitions per side
    cells_per_side = numpy.int_(numpy.ceil(boxsize / minisize))

    # Generate the IDs and positions of the mini box grid
    grid_ids, grid_pos = generate_mini_box_grid(boxsize, minisize)

    # Get the adjacent mini box IDs
    mini_box_ids = get_adjacent_mini_box_ids(
        mini_box_id=mini_box_id,
        boxsize=boxsize,
        minisize=minisize
    )

    # Create empty lists (containers) to save the data from file for each ID
    pos, vel, pid = ([] for _ in range(3))

    # Load all adjacent boxes
    for i, mini_box in enumerate(mini_box_ids):
        file_name = f'mini_boxes_nside_{cells_per_side}/{mini_box}.hdf5'
        with h5py.File(load_path + file_name, 'r') as hdf:
            pos.append(hdf['part/pos'][()])
            vel.append(hdf['part/vel'][()])
            pid.append(hdf['part/ID'][()])

    # Concatenate into a single array
    pos = numpy.concatenate(pos)
    vel = numpy.concatenate(vel)
    pid = numpy.concatenate(pid)

    # Mask particles within a padding distance of the edge of the box in each
    # direction
    loc_id = grid_ids == mini_box_id
    padded_distance = 0.5 * minisize + padding
    absolute_rel_pos = numpy.abs(
        relative_coordinates(pos, grid_pos[loc_id], boxsize, periodic=True)
    )
    mask = numpy.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

    return pos[mask], vel[mask], pid[mask]


def load_seeds(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float = 5.0,
) -> Tuple[numpy.ndarray]:
    """Load seeds from a mini box

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    load_path : str
        Location from where to load the file
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5

    Returns
    -------
    Tuple[numpy.ndarray]
        Position, velocity, ID, R200b, M200b, Rs and a mask for seeds in the 
        minibox.
    """
    # Determine number of partitions per side
    cells_per_side = numpy.int_(numpy.ceil(boxsize / minisize))

    # Generate the IDs and positions of the mini box grid
    grid_ids, grid_pos = generate_mini_box_grid(boxsize, minisize)

    # Get the adjacent mini box IDs
    mini_box_ids = get_adjacent_mini_box_ids(
        mini_box_id=mini_box_id,
        boxsize=boxsize,
        minisize=minisize
    )

    # Create empty lists (containers) to save the data from file for each ID
    pos, vel, hid, r200, m200, rs, mini_box_mask = ([] for _ in range(7))

    # Load all adjacent boxes
    for i, mini_box in enumerate(mini_box_ids):
        file_name = f'mini_boxes_nside_{cells_per_side}/{mini_box}.hdf5'
        with h5py.File(load_path + file_name, 'r') as hdf:
            pos.append(hdf['seed/pos'][()])
            vel.append(hdf['seed/vel'][()])
            hid.append(hdf['seed/ID'][()])
            r200.append(hdf['seed/R200b'][()])
            m200.append(hdf['seed/M200b'][()])
            rs.append(hdf['seed/Rs'][()])
            n_seeds = len(hdf['seed/ID'][()])
            if mini_box == mini_box_id:
                mini_box_mask.append(numpy.ones(n_seeds, dtype=bool))
            else:
                mini_box_mask.append(numpy.zeros(n_seeds, dtype=bool))

    # Concatenate into a single array
    pos = numpy.concatenate(pos)
    vel = numpy.concatenate(vel)
    hid = numpy.concatenate(hid)
    r200 = numpy.concatenate(r200)
    m200 = numpy.concatenate(m200)
    rs = numpy.concatenate(rs)
    mini_box_mask = numpy.concatenate(mini_box_mask)

    # Mask seeds within a padding distance of the edge of the box in each
    # direction
    loc_id = grid_ids == mini_box_id
    padded_distance = 0.5 * minisize + padding
    absolute_rel_pos = numpy.abs(relative_coordinates(
        pos, grid_pos[loc_id], boxsize, periodic=True,
    ))
    mask = numpy.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

    m200 = m200[mask]
    r200 = r200[mask]
    pos = pos[mask]
    vel = vel[mask]
    hid = hid[mask]
    rs = rs[mask]
    mini_box_mask = mini_box_mask[mask]

    # Sort seeds by M200 (largest first)
    argorder = numpy.argsort(-m200)
    m200 = m200[argorder]
    r200 = r200[argorder]
    pos = pos[argorder]
    vel = vel[argorder]
    hid = hid[argorder]
    rs = rs[argorder]
    mini_box_mask = mini_box_mask[argorder]

    return (pos, vel, hid, r200, m200, rs, mini_box_mask)


###
