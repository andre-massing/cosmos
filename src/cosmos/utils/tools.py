from ngsolve import *
from ngsolve.solvers import *


def gradient(f, P):

    m, _ = P.dims
    l = len(f.dims)

    if l == 0:
        # scalar gradient is defined traditionally

        if m == 2:
            output = P * CoefficientFunction((f.Diff(x), f.Diff(y)))
        elif m == 3:
            output = P * CoefficientFunction((f.Diff(x), f.Diff(y), f.Diff(z)))

    elif l == 1:
        # vector gradient is defined component by component
        # and disposed along rows

        if m == 2:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            output = CoefficientFunction((aux1[0], aux2[0], aux1[1], aux2[1]), dims=(m, m)).trans
        elif m == 3:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            aux3 = gradient(f[2], P)
            output = CoefficientFunction(
                (aux1[0], aux2[0], aux3[0], aux1[1], aux2[1], aux3[1], aux1[2], aux2[2], aux3[2]),
                dims=(m, m),
            ).trans

    else:
        raise RuntimeError(
            "Don't know how to take the gradient. Only scalars and vectors are accepted."
        )

    return output
