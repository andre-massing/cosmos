from ngsolve import *
from ngsolve .solvers import *
import numpy as np

# def params_check(params = {}, accepted_keys=[], defaults=[]):
        
#     wrong = [key for key in params.keys() if key not in accepted_keys]

#     if wrong:

#         print("The following keys are unknown parameters:\n", wrong)

#         print("The accepted keys are:\n", accepted_keys)

#         raise ValueError
    
#     for i, key in enumerate(accepted_keys):

#         params[key] = params.get(key, defaults[i])

def compute_stab_mc(data, gfu, params):

    V2 = gfu.space
        
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
        nE = tE
    else:
        nE = Cross(ns, tE)
    Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)

    fes0 = V2*dV
    gfu0 = GridFunction(fes0)
    
    (kappa0, dkappa0), (eta0, deta0) = fes0.TnT()

    if data.mesh.dim == 2:

        dkappa0 = dkappa0*tE
        deta0 = deta0*tE

        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0)
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0)

    elif data.mesh.dim == 3:

        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0.Trace())
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0.Trace())

    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds  # Adding traces here?
    gamma = params['stab']
    A0 += gamma*h*InnerProduct(jump_dkappadn0,jump_detadn0)\
        *ds(element_boundary=True)
    A0.Assemble()

    F0 += -InnerProduct(Ps, Grad(eta0).Trace())*ds

    if 'clamped_bnd' in params.keys():
        if params['clamped_bnd']:

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

    gfu.vec.data = gfu0.components[0].vec.data