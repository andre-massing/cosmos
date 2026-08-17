# Numerical post-processing helpers shared by the ADR and Cahn-Hilliard PDE model
# families, used to enforce bound- and/or mass-preservation on a raw solution vector.

import numpy as np

def MandBP(gfu_vec, dt = None, weights=None, BP=None, MP=False, mass0=None):
    """Mass-and-Bound-Preserving limiter for a finite-element solution vector.

    Clips ``gfu_vec`` to the bounds ``BP = [low, high]``. If ``MP`` (mass-preserving) is
    also requested, a uniform scalar shift is first applied to the raw solution so that,
    after clipping, the discrete total mass (``sum(weights * clipped_values)``) matches
    the target ``mass0`` exactly; the shift is found via the secant method applied to the
    (piecewise-linear, monotone) residual function ``F``.

    Args:
        gfu_vec: Raw solution vector (NumPy array) to be corrected in place logically
            (a new corrected array is returned).
        dt: Current time-step size; only used to scale the shift when ``MP`` is True.
        weights: Quadrature/mass-lumping weights (one per DOF) used to compute the
            discrete mass; only used when ``MP`` is True.
        BP: ``[low, high]`` bounds the solution must respect.
        MP: If True, also enforce mass conservation (requires ``dt``, ``weights`` and
            ``mass0``); if False, only clip to ``BP``.
        mass0: Target total mass to preserve when ``MP`` is True.

    Returns:
        The bound-(and, if requested, mass-)corrected solution vector.
    """

    tol = 1e-10

    if BP and not MP:

        # Bound-preservation only: simple clipping to [low, high].
        gfu_data = np.clip(gfu_vec, BP[0], BP[1])

    elif MP:

        def F(xsi):
            # Residual: discrete mass of the clipped, shifted solution minus the
            # target mass. Its root gives the shift that restores mass conservation.

            result = 0

            temp = gfu_vec + dt * xsi

            result -= mass0
            dummy = np.clip(temp, BP[0], BP[1])
            result += np.sum(weights * dummy)

            return result

        # Secant method iteration to find the root xsi of F.
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

        # Apply the converged shift, then clip to the bounds.
        threshold = dt * xsi2
        gfu_data = np.clip(gfu_vec + threshold, BP[0], BP[1])

    return gfu_data