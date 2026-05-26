import numpy as np
from scipy.optimize import brentq

def MandBP(gfu_vec, dt = None, weights=None, BP=None, MP=False, mass0=None):

    """
    Applies bound-preserving (BP) or mass-preserving (MP) corrections to a vector.

    This function adjusts the input vector `gfu_vec` based on the specified options:
    - If BP is enabled (and MP is False), it clips the vector to the bounds [BP[0], BP[1]].
    - If MP is enabled, it performs a mass-preserving correction by finding a threshold that
      preserves the total mass (weighted sum) while clipping to bounds.

    Parameters:
    - gfu_vec (array-like): The input vector to be corrected.
    - dt (float, optional): Time step or scaling factor used in MP correction. Required if MP=True.
    - weights (array-like, optional): Weights for mass calculation in MP. Required if MP=True.
    - BP (tuple of floats, optional): Bounds [lower, upper] for clipping. Required if BP=True or MP=True.
    - MP (bool, optional): If True, enables mass-preserving correction. Defaults to False.
    - mass0 (float, optional): Target mass to preserve in MP. Required if MP=True.

    Returns:
    - array-like: The corrected vector, clipped to bounds.

    Notes:
    - MP uses ``scipy.optimize.brentq`` to solve for the threshold that preserves mass.
    - Tolerance for convergence in MP is set to 1e-10.
    """

    tol = 1e-10

    if BP and not MP:

        gfu_data = np.clip(gfu_vec, BP[0], BP[1])  

    elif MP:

        def F(xsi):
            temp = gfu_vec + dt * xsi
            return np.sum(weights * np.clip(temp, BP[0], BP[1])) - mass0

        # F is monotone increasing in xsi (for dt > 0).
        # Bracket: at xsi_lo every value clips to BP[0]; at xsi_hi to BP[1].
        xsi_lo = (BP[0] - float(np.max(gfu_vec))) / dt
        xsi_hi = (BP[1] - float(np.min(gfu_vec))) / dt
        xsi2 = brentq(F, xsi_lo, xsi_hi, xtol=tol)

        threshold = dt * xsi2
        gfu_data = np.clip(gfu_vec + threshold, BP[0], BP[1])

    return gfu_data