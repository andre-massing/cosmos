# Small collection of standalone math helpers shared across PDE models: a general
# tangential-gradient operator built from raw symbolic differentiation, and a
# routine to compute a stabilized initial mean-curvature field by solving an
# auxiliary mixed problem (mirrors cosmos.pde.mean_curvature.mean_curvature_boundary_stab_bdf1_model).

from ngsolve import *
from ngsolve .solvers import *
import numpy as np

# Unused/dead code: a generic parameter-dictionary validation helper, left
# commented out.
# def params_check(params = {}, accepted_keys=[], defaults=[]):

#     wrong = [key for key in params.keys() if key not in accepted_keys]

#     if wrong:

#         print("The following keys are unknown parameters:\n", wrong)

#         print("The accepted keys are:\n", accepted_keys)

#         raise ValueError

#     for i, key in enumerate(accepted_keys):

#         params[key] = params.get(key, defaults[i])

def gradient(f,P):
    """Tangential gradient of a scalar or vector NGSolve CoefficientFunction `f`.

    Computed via raw symbolic differentiation (`f.Diff(x)` etc., i.e. the ambient
    gradient) and then projected with the tangential projector `P` (an (m,m)
    CoefficientFunction, e.g. Id - n(x)n). Supports 2D and 3D ambient space.

    Args:
        f: A scalar (0-dim) or vector (1-dim) CoefficientFunction.
        P: The (m,m) tangential projection operator, m = P.dims[0].

    Returns:
        For scalar f: an m-vector CoefficientFunction (the tangential gradient).
        For vector f: an (m,m) CoefficientFunction, the tangential Jacobian, with
        rows/columns arranged so that row i is the tangential gradient of f[i]
        (see the `.trans` transpose used below to fix up the assembly order).
    """

    m, _ = P.dims
    l = len(f.dims)

    if l == 0:
        # Scalar gradient: apply raw ambient differentiation component-wise, then
        # project onto the tangent space with P.

        if m == 2:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y)))
        elif m == 3:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y), f.Diff(z))) 
            
    elif l == 1:
        # Vector gradient: differentiate component-wise (each component's
        # tangential gradient is itself an m-vector, via the scalar case above),
        # then reassemble into an (m,m) matrix. The components are laid out
        # column-by-column (aux1, aux2, ... are consecutive columns) and the final
        # `.trans` transposes so that row i holds the tangential gradient of f[i].

        if m == 2:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            output = CoefficientFunction((aux1[0], aux2[0], \
                aux1[1], aux2[1]), dims = (m,m)).trans
        elif m == 3:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            aux3 = gradient(f[2], P)
            output = CoefficientFunction((aux1[0], aux2[0], aux3[0], \
                aux1[1], aux2[1], aux3[1],\
                    aux1[2], aux2[2], aux3[2]), dims = (m,m)).trans
            
    else:

        raise RuntimeError("Don't know how to take the gradient. Only scalars and vectors are accepted.")
    
    return output


def compute_stab_mc(data, gfu, params):
    """Compute a stabilized INITIAL mean-curvature field by solving an auxiliary
    mixed problem, writing the result into the caller-supplied `gfu`.

    Mirrors the mixed formulation used in
    cosmos.pde.mean_curvature.mean_curvature_boundary_stab_bdf1_model (a scalar
    "mean curvature vector" unknown kappa, tested against a facet-based
    Lagrange-multiplier-like variable), but solved once, standalone, to
    initialize kappa consistently with a stabilized weak Laplace-Beltrami
    identity, rather than as part of a time-stepping scheme.

    Args:
        data: Object exposing `data.mesh` (the SolverMesh/NGSolve mesh to solve on).
        gfu: The (already-allocated) GridFunction to write the mean-curvature
            solution into; only its `.space` is used to build the mixed space.
        params: Dict with required key 'stab' (facet-jump stabilization weight,
            gamma below) and optional key 'clamped_bnd' (bboundary region name(s)
            where a clamped-boundary conormal condition is imposed).
    """

    V2 = gfu.space

    # Facet-based auxiliary space carrying the (tangential) derivative jump used
    # for the stabilization term below: a scalar per-facet field in 2D, a
    # tangential vector field per facet in 3D.
    if data.mesh.dim == 2:

        dV = H1(data.mesh, order=1,\
                    definedon = data.mesh.Boundaries('.*'))

    elif data.mesh.dim == 3:

        dV = VectorFacetSurface(data.mesh, order=1,\
                definedon = data.mesh.Boundaries('.*'))

    h = specialcf.mesh_size
    ns = specialcf.normal(data.mesh.dim)
    tE = specialcf.tangential(data.mesh.dim)
    if data.mesh.dim == 2:
        # In 2D (a curve), the conormal (outward normal to the boundary point,
        # within the curve) coincides with the tangential direction.
        nE = tE
    else:
        # In 3D, the conormal to an edge of the surface is normal to both the
        # surface normal and the edge's tangent.
        nE = Cross(ns, tE)
    Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)

    fes0 = V2*dV
    gfu0 = GridFunction(fes0)

    (kappa0, dkappa0), (eta0, deta0) = fes0.TnT()

    if data.mesh.dim == 2:

        dkappa0 = dkappa0*tE
        deta0 = deta0*tE

        # Jump in the conormal derivative of kappa/eta across the element
        # boundary, comparing the derivative computed from the trace (Deriv())
        # against the independent facet unknown dkappa0/deta0.
        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0)
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0)

    elif data.mesh.dim == 3:

        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0.Trace())
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0.Trace())

    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds  # Adding traces here?
    gamma = params['stab']
    # Facet-jump (interior penalty) stabilization: penalizes the mismatch between
    # the trace-derivative and the independent facet unknown, weighted by the
    # local mesh size and the stabilization parameter gamma.
    A0 += gamma*h*InnerProduct(jump_dkappadn0,jump_detadn0)\
        *ds(element_boundary=True)
    A0.Assemble()

    # Weak Laplace-Beltrami identity: mean-curvature vector kappa satisfies
    # kappa = -Delta_Gamma(X), tested weakly via the tangential projector Ps
    # against the surface gradient of the test function (integration by parts of
    # the tangential gradient of the identity map X).
    F0 += -InnerProduct(Ps, Grad(eta0).Trace())*ds

    if 'clamped_bnd' in params.keys():
        if params['clamped_bnd']:
            # Clamped-boundary conormal condition: add a conormal boundary term
            # concentrated (via the indicator GridFunction gfF) on the named
            # bboundary region, so the natural (Neumann-type) boundary condition
            # from the integration by parts above is replaced there.

            if data.mesh.dim == 3:
                gfF = GridFunction(FacetSurface(data.mesh, order=0))
                gfF.Set(1, definedon=data.mesh.BBoundaries(params['clamped_bnd']))

                F0 += InnerProduct(nE, eta0) * gfF * ds(element_boundary=True)

            elif data.mesh.dim == 2:

                gfF = GridFunction(H1(data.mesh, order =1,\
                        definedon=data.mesh.Boundaries('.*')))
                gfF.Set(1, definedon=data.mesh.BBoundaries(params['clamped_bnd']))

                F0 += InnerProduct(gfF*nE, eta0) \
                    * ds(element_boundary=True)

    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec

    # Only the primary (kappa) component of the mixed solution is written back;
    # the facet-multiplier component is discarded.
    gfu.vec.data = gfu0.components[0].vec.data