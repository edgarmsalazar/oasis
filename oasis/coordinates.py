import numpy as np


def relative_coordinates(
    x: np.ndarray,
    x0: np.ndarray,
    boxsize: float,
    periodic: bool = True
) -> float:
    """Returns the coordinates x relative to x0 accounting for periodic boundary
    conditions.

    Parameters
    ----------
    x : np.ndarray
        Position array (N, 3).
    x0 : np.ndarray
        Reference position in cartesian coordinates.
    boxsize : float
        Size of simulation box.
    periodic : bool, optional
        Set to True if the simulation box is periodic, by default True.

    Returns
    -------
    float
        Relative positions.
    """
    if type(x) in [list, tuple]:
        raise TypeError("Input 'x' must be an array (not a list or tuple)")
    if periodic:
        return (x - x0 + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    return x - x0


def velocity_components(
    pos: np.ndarray,
    vel: np.ndarray,
) -> tuple[np.ndarray]:
    """Computes the radial and tangential velocites from cartesian/rectangular 
    coordinates.

    Parameters
    ----------
    pos : np.ndarray
        Cartesian coordinates.
    vel : np.ndarray
        Cartesian velocities.

    Returns
    -------
    tuple[np.ndarray]
        Radial velocity, tangential velocity and magnitude squared of the 
        velocity.
    """
    if np.ndim(pos) < 2:
        raise ValueError(f'Number of dimensions is {np.ndim(pos)}. Please ' +
                         'reshape array to match 2D.')

    # Transform coordinates from cartesian to spherical
    rps = np.sqrt(np.sum(np.square(pos), axis=1))
    rp_hat = pos / rps

    # Compute radial velocity as v dot r_hat
    vr = np.sum(vel * rp_hat, axis=1)

    # Velocity squared
    v2 = np.sum(np.square(vel), axis=1)

    # Compute perpendicular velocity component as v^2 - vr^2
    vt = np.sqrt(v2 - np.square(vr))

    return vr, vt, v2


###
