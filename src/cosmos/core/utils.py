import numpy as np

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
    - MP uses a secant method to solve for the threshold that preserves mass.
    - Tolerance for convergence in MP is set to 1e-10.
    """

    tol = 1e-10

    if BP and not MP:

        gfu_data = np.clip(gfu_vec, BP[0], BP[1])  

    elif MP:

        def F(xsi):

            result = 0

            temp = gfu_vec + dt * xsi

            result -= mass0
            dummy = np.clip(temp, BP[0], BP[1])
            result += np.sum(weights * dummy)

            return result

        xsi_old0 = 0
        xsi_old1 = -dt
        xsi2 = -2

        while abs(F(xsi_old1) - F(xsi_old0))>tol or xsi2==-2:

            if F(xsi_old1) == F(xsi_old0):

                xsi2 = -1

            else:

                F1 = F(xsi_old1)
                F0 = F(xsi_old0)

                xsi2 = xsi_old1-F1*(xsi_old1 - xsi_old0)/(F1 - F0)

                xsi_old0 = xsi_old1
                xsi_old1 = xsi2

        threshold = dt * xsi2
        gfu_data = np.clip(gfu_vec + threshold, BP[0], BP[1]) 

    return gfu_data