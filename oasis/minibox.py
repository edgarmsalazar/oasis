from typing import Tuple

import h5py as h5
import numpy as np
from tqdm import tqdm

from oasis.common import get_np_unit_dtype, mkdir
from oasis.coordinates import relative_coordinates


def generate_mini_box_grid(
    boxsize: float,
    minisize: float,
) -> Tuple[np.ndarray]:
    """Generates a 3D grid of mini boxes.

    Parameters
    ----------
    boxsize : float
        Size of simulation box.
    minisize : float
        Size of mini box.

    Returns
    -------
    Tuple[np.ndarray]
        ID and central coordinate for all mini boxes.
    """
    if minisize > boxsize:
        raise ValueError('Mini box size cannot be larger than box size.')
    
    # Number of mini boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))
    
    # Total number of cells
    n_cells = boxes_per_side**3
    
    # Pre-allocate temporary arrays
    uint_dtype = get_np_unit_dtype(boxes_per_side**2)
    centers_temp = np.zeros((n_cells, 3), dtype=np.float32)
    ids_temp = np.zeros(n_cells, dtype=uint_dtype)
    
    # Pre-compute shift values to avoid repeated calculations
    shift1 = np.array(1, dtype=uint_dtype)
    shift2 = np.array(boxes_per_side, dtype=uint_dtype)  
    shift3 = np.array(boxes_per_side**2, dtype=uint_dtype)
    
    # Generate centers and IDs in a single pass using nested loops
    # This eliminates the need for cartesian_product and intermediate function calls
    index = 0
    for i in range(boxes_per_side):
        for j in range(boxes_per_side):
            for k in range(boxes_per_side):
                # Generate cell centers directly (equivalent to gen_data_pos_regular)
                centers_temp[index, 0] = minisize * (k + 0.5)  # x coordinate
                centers_temp[index, 1] = minisize * (j + 0.5)  # y coordinate  
                centers_temp[index, 2] = minisize * (i + 0.5)  # z coordinate
                
                # Generate unique IDs directly (equivalent to cartesian_product + shifts)
                ids_temp[index] = k * shift1 + j * shift2 + i * shift3
                
                index += 1
    
    # Get the index array that would sort the IDs
    sort_order = np.argsort(ids_temp)
    
    # Sort arrays using the index array (more cache-friendly approach)
    ids = ids_temp[sort_order]
    centres = centers_temp[sort_order]
    
    return ids, centres


def get_mini_box_id(
    position: np.ndarray,
    boxsize: float,
    minisize: float,
) -> int:
    """Returns the mini box ID to which the coordinates `x` fall into

    Parameters
    ----------
    position : np.ndarray
        Position in cartesian coordinates.
    boxsize : float
        Size of simulation box.
    minisize : float
        Size of mini box.

    Returns
    -------
    int
        ID of the mini box.
    """
    EDGE_TOL = 1e-8

    if minisize > boxsize:
        raise ValueError('Mini box size cannot be larger than box size.')
    
    # Ensure position is 2D array with shape (N, 3)
    if position.ndim == 1:
        position = position.reshape(1, 3)
    
    # Pre-compute shift values
    n_cells_per_side = int(np.ceil(boxsize / minisize))
    shift = np.array([1, n_cells_per_side, n_cells_per_side**2])
    
    # Handle edge cases using approximate equality
    # Points at upper boundary
    upper_mask = np.abs(position - boxsize) < 1e-9
    position[upper_mask] -= EDGE_TOL
    
    # Points at lower boundary  
    lower_mask = np.abs(position - 0.0) < 1e-9
    position[lower_mask] += EDGE_TOL
    
    # Compute grid indices
    grid_indices = np.floor(position / minisize).astype(int)
    
    # Compute unique IDs
    ids = np.sum(shift * grid_indices, axis=1)

    return ids


def get_adjacent_mini_box_ids(
    mini_box_id: np.ndarray,
    boxsize: float,
    minisize: float,
) -> np.ndarray:
    """Returns a list of all IDs that are adjacent to the specified mini box ID.
    There are always 27 adjacent boxes in a 3D volume, including the specified 
    ID.

    Parameters
    ----------
    mini_box_id : np.ndarray
        ID of the mini box.
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box

    Returns
    -------
    np.ndarray
        List of mini box IDs adjacent to `id`

    Raises
    ------
    ValueError
        If `id` is not found in the allowed values in `ids`
    """
    if minisize > boxsize:
        raise ValueError('Mini box size cannot be larger than box size.')
    
    # Number of mini boxes per side
    boxes_per_side = int(np.ceil(boxsize / minisize))
    max_id = boxes_per_side**3 - 1
    
    if mini_box_id < 0 or mini_box_id > max_id:
        raise ValueError(f'ID {mini_box_id} is out of bounds (valid range: 0-{max_id})')
    
    # Convert 1D ID to 3D coordinates (i, j, k)
    # ID = k + j*boxes_per_side + i*boxes_per_side^2
    i = mini_box_id // (boxes_per_side**2)
    remainder = mini_box_id % (boxes_per_side**2)
    j = remainder // boxes_per_side
    k = remainder % boxes_per_side
    
    # Start with the input ID as the first element
    adjacent_ids = [mini_box_id]

    # Generate all 27 adjacent box coordinates (3x3x3 neighborhood)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                # Skip the center box (0,0,0) since we already added it
                if di == 0 and dj == 0 and dk == 0:
                    continue

                # Apply periodic boundary conditions
                ni = (i + di) % boxes_per_side
                nj = (j + dj) % boxes_per_side  
                nk = (k + dk) % boxes_per_side
                
                # Convert back to 1D ID
                neighbor_id = nk + nj * boxes_per_side + ni * (boxes_per_side**2)
                adjacent_ids.append(neighbor_id)
    
    return np.array(adjacent_ids, dtype=np.int32)


def generate_mini_box_ids(
    positions: np.ndarray,
    boxsize: float,
    minisize: float,
    chunksize: int = 100_000,
) -> None:
    """Gets the mini box ID for each position

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Where to save the IDs
    chunksize : int, optional
        Number of items to process at a time in chunks, by default 100_000
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None

    Returns
    -------
    None
    """
    n_items = positions.shape[0]
    n_iter = n_items // chunksize

    # Determine data type for integer arrays based on the maximum number of
    # elements
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))
    uint_dtype = get_np_unit_dtype(boxes_per_side**3)

    ids = np.zeros(n_items, dtype=uint_dtype)

    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue'):
        low = chunk * chunksize
        if chunk < n_iter - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        ids[low:upp] = get_mini_box_id(positions[low:upp], boxsize, minisize)

    return ids


def get_chunks(ids: np.ndarray, chunksize: int) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    ids : np.ndarray
        _description_
    chunksize : int
        _description_

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    RuntimeError
        _description_
    """
    n_items = ids.shape[0]
    # Initialize first lower index to zero
    i, upp = 0, 0
    chunks = [i,]
    # Maximum iteration step preventing infinite loop when the chunksize is too 
    # small for the average number of items per ID.
    i_max = np.max([len(np.unique(ids)) + 1, int(1.1 * (n_items // chunksize))])

    while True:
        if i > i_max:
            raise IndexError(f'Maximum iterations reached i_max = {i_max} ' +
                               'Chunk size too small. Please increase it.')
        low = chunks[-1]
        upp = low + chunksize
        if upp < n_items:
            idx = low + np.argmax((ids[low:] - ids[upp])==0)
            chunks.append(idx)
            i += 1
        else:
            idx = -1
            chunks.append(idx)
            break
    
    return np.array(chunks)


def split_box_into_mini_boxes(
    positions: np.ndarray,
    velocities: np.ndarray,
    uid: np.ndarray,
    save_path: str,
    boxsize: float,
    minisize: float,
    chunksize: int = 100_000,
    name: str = None,
    props: Tuple[list, list, list] = None
) -> None:
    """Sorts all items into mini boxes and saves them in disc.

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates
    velocities : np.ndarray
        Cartesian velocities
    uid : np.ndarray
        Unique IDs for each position (e.g. PID, HID)
    save_path : str
        Where to save the IDs
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    chunksize : int, optional
        Number of items to process at a time in chunks, by default 100_000
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
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # If given chunk size is smaller than the number of points given.
    chunksize = np.min([len(positions), chunksize])
    
    # Compute mini box ids
    mini_box_ids = generate_mini_box_ids(
        positions=positions,
        boxsize=boxsize,
        minisize=minisize,
        chunksize=chunksize,
    )

    # Sort data by mini box id
    mb_order = np.argsort(mini_box_ids)
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

    chunk_idx = get_chunks(
        ids=mini_box_ids,
        chunksize=chunksize
    )

    # Get smallest data type to represent IDs
    uint_dtype_pid = get_np_unit_dtype(np.max(uid))
    if props:
        labels = ('ID', 'pos', 'vel', *labels)
        dtypes = (uint_dtype_pid, np.float32, np.float32, *dtypes)
    else:
        labels = ('ID', 'pos', 'vel')
        dtypes = (uint_dtype_pid, np.float32, np.float32)

    # Create target directory
    save_dir = save_path + f'mini_boxes_nside_{boxes_per_side}/'
    mkdir(save_dir)

    # For each chunk
    for chunk_i in tqdm(range(len(chunk_idx)-1), desc='Processing chunks',
                        ncols=100, colour='blue'):
        # Select chunk
        low = chunk_idx[chunk_i]
        upp = chunk_idx[chunk_i + 1]

        mb_chunk = mini_box_ids[low: upp]
        pos_chunk = positions[low: upp]
        vel_chunk = velocities[low: upp]
        pid_chunk = uid[low: upp]

        if props:
            props_chunks = [None for _ in range(len(props))]
            for k, item in enumerate(props):
                props_chunks[k] = item[low: upp]

        # Check which mini box ids are in the chunk
        mb_chunk_low = mb_chunk[0]
        mb_chunk_upp = mb_chunk[-1] + 1

        # Get index (search sorted style) of the first occurence of each
        # distinct mini box id. Append a -1 at the end for completeness.
        indexed_slice = []
        for mb_id in range(mb_chunk_low, mb_chunk_upp):
            indexed_slice.append(np.argmin(mb_chunk - mb_id))
        indexed_slice.append(-1)

        # Save data per slice
        for i, mb_id in enumerate(range(mb_chunk_low, mb_chunk_upp)):
            if props:
                data = (
                    pid_chunk[indexed_slice[i]: indexed_slice[i+1]],
                    pos_chunk[indexed_slice[i]: indexed_slice[i+1]],
                    vel_chunk[indexed_slice[i]: indexed_slice[i+1]],
                    *[
                        item_chunk[indexed_slice[i]: indexed_slice[i+1]]
                        for item_chunk in props_chunks
                    ],
                )
            else:
                data = (
                    pid_chunk[indexed_slice[i]: indexed_slice[i+1]],
                    pos_chunk[indexed_slice[i]: indexed_slice[i+1]],
                    vel_chunk[indexed_slice[i]: indexed_slice[i+1]],
                )

            with h5.File(save_dir + f'{mb_id}.hdf5', 'a') as hdf:
                if (name is not None) and (not name in hdf.keys()):
                    # hdf.create_group(name)
                    prefix = f'{name}/'
                else:
                    prefix=''

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
) -> Tuple[np.ndarray]:
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
    Tuple[np.ndarray]
        Position, velocity, and PID
    """
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

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
        file_name = f'mini_boxes_nside_{boxes_per_side}/{mini_box}.hdf5'
        with h5.File(load_path + file_name, 'r') as hdf:
            pos.append(hdf['part/pos'][()])
            vel.append(hdf['part/vel'][()])
            pid.append(hdf['part/ID'][()])

    # Concatenate into a single array
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    pid = np.concatenate(pid)

    # Mask particles within a padding distance of the edge of the box in each
    # direction
    loc_id = grid_ids == mini_box_id
    padded_distance = 0.5 * minisize + padding
    absolute_rel_pos = np.abs(
        relative_coordinates(pos, grid_pos[loc_id], boxsize, periodic=True)
    )
    mask = np.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

    return pos[mask], vel[mask], pid[mask]


def load_seeds(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float = 5.0,
) -> Tuple[np.ndarray]:
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
    Tuple[np.ndarray]
        Position, velocity, ID, R200b, M200b, Rs and a mask for seeds in the 
        minibox.
    """
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

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
        file_name = f'mini_boxes_nside_{boxes_per_side}/{mini_box}.hdf5'
        with h5.File(load_path + file_name, 'r') as hdf:
            pos.append(hdf['seed/pos'][()])
            vel.append(hdf['seed/vel'][()])
            hid.append(hdf['seed/ID'][()])
            r200.append(hdf['seed/R200b'][()])
            m200.append(hdf['seed/M200b'][()])
            rs.append(hdf['seed/Rs'][()])
            n_seeds = len(hdf['seed/ID'][()])
            if mini_box == mini_box_id:
                mini_box_mask.append(np.ones(n_seeds, dtype=bool))
            else:
                mini_box_mask.append(np.zeros(n_seeds, dtype=bool))

    # Concatenate into a single array
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    hid = np.concatenate(hid)
    r200 = np.concatenate(r200)
    m200 = np.concatenate(m200)
    rs = np.concatenate(rs)
    mini_box_mask = np.concatenate(mini_box_mask)

    # Mask seeds within a padding distance of the edge of the box in each
    # direction
    loc_id = grid_ids == mini_box_id
    padded_distance = 0.5 * minisize + padding
    absolute_rel_pos = np.abs(relative_coordinates(
        pos, grid_pos[loc_id], boxsize, periodic=True,
    ))
    mask = np.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

    m200 = m200[mask]
    r200 = r200[mask]
    pos = pos[mask]
    vel = vel[mask]
    hid = hid[mask]
    rs = rs[mask]
    mini_box_mask = mini_box_mask[mask]

    # Sort seeds by M200 (largest first)
    argorder = np.argsort(-m200)
    m200 = m200[argorder]
    r200 = r200[argorder]
    pos = pos[argorder]
    vel = vel[argorder]
    hid = hid[argorder]
    rs = rs[argorder]
    mini_box_mask = mini_box_mask[argorder]

    return (pos, vel, hid, r200, m200, rs, mini_box_mask)


###
