from ngsolve import *
from ngsolve .solvers import *
import numpy as np

def params_check(params = {}, accepted_keys=[], defaults=[]):
        
    wrong = [key for key in params.keys() if key not in accepted_keys]

    if wrong:

        print("The following key are unknown parameters:\n", wrong)

        raise ValueError
    
    for i, key in enumerate(accepted_keys):

        params[key] = params.get(key, defaults[i])

def compute_error(data, gfu, u_ex, norm, domain, VorB):

    ns = specialcf.normal(data['mesh'].dim)
    Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

    if norm == 'L2':

        aux = InnerProduct(gfu-u_ex, gfu-u_ex)

        if VorB == BND:

            err = sqrt(Integrate(aux, mesh = data['mesh'], order = gfu.space.globalorder*2, 
                            VOL_or_BND=BND, 
                            definedon = data['mesh'].Boundaries(domain)))

        else:

            err = sqrt(Integrate(aux, mesh = data['mesh'], order = gfu.space.globalorder*2,
                            definedon = data['mesh'].Materials(domain)))

    else:

        raise ValueError('The norm given is not implemented')
    
    return err

def compute_stab_mc(data, gfu, params):

    V2 = gfu.space
        
    if data['mesh'].dim == 2:
        
        dV = H1(data['mesh'], order=1,\
                    definedon = data['mesh'].Boundaries('.*'))
        
    elif data['mesh'].dim == 3:
        
        dV = VectorFacetSurface(data['mesh'], order=1,\
                definedon = data['mesh'].Boundaries('.*'))

    h = specialcf.mesh_size
    ns = specialcf.normal(data['mesh'].dim)
    tE = specialcf.tangential(data['mesh'].dim)
    if data['mesh'].dim == 2:
        nE = tE
    else:
        nE = Cross(ns, tE)
    Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

    fes0 = V2*dV
    gfu0 = GridFunction(fes0)
    
    (kappa0, dkappa0), (eta0, deta0) = fes0.TnT()

    if data['mesh'].dim == 2:

        dkappa0 = dkappa0*tE
        deta0 = deta0*tE

        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0)
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0)

    elif data['mesh'].dim == 3:

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

    if params['clamped_bnd']:

        if data['mesh'].dim == 3:
            gfF = GridFunction(FacetSurface(data['mesh'], order=0))
            gfF.Set(1, definedon=data['mesh'].BBoundaries(params['clamped_bnd']))

            F0 += InnerProduct(nE, eta0) * gfF * ds(element_boundary=True)

        elif data['mesh'].dim == 2:

            gfF = GridFunction(H1(data['mesh'], order =1,\
                    definedon=data['mesh'].Boundaries('.*')))
            gfF.Set(1, definedon=data['mesh'].BBoundaries(params['clamped_bnd']))

            F0 += InnerProduct(gfF*nE, eta0) \
                * ds(element_boundary=True)

    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec

    gfu.vec.data = gfu0.components[0].vec.data

def compute_mc(data, gfu, params):

    V2 = gfu.space

    h = specialcf.mesh_size
    ns = specialcf.normal(data['mesh'].dim)
    tE = specialcf.tangential(data['mesh'].dim)
    if data['mesh'].dim == 2:
        nE = tE
    else:
        nE = Cross(ns, tE)
    Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

    fes0 = V2
    gfu0 = GridFunction(fes0)
    
    kappa0, eta0 = fes0.TnT()


    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds  # Adding traces here?
    A0.Assemble()

    F0 += -InnerProduct(Ps, Grad(eta0).Trace())*ds

    if params['clamped_bnd']:

        if data['mesh'].dim == 3:
            gfF = GridFunction(FacetSurface(data['mesh'], order=0))
            gfF.Set(1, definedon=data['mesh'].BBoundaries(params['clamped_bnd']))

            F0 += InnerProduct(nE, eta0) * gfF * ds(element_boundary=True)

        elif data['mesh'].dim == 2:

            gfF = GridFunction(H1(data['mesh'], order =1,\
                    definedon=data['mesh'].Boundaries('.*')))
            gfF.Set(1, definedon=data['mesh'].BBoundaries(params['clamped_bnd']))

            F0 += InnerProduct(gfF*nE, eta0) \
                * ds(element_boundary=True)

    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec

    gfu.vec.data = gfu0.vec.data

def compute_displ(data, function, dX, bc = '.*'):

    if data['mesh'].ne == 0:

        fes = VectorH1(data['mesh'], order=dX.space.globalorder)

        gfu = GridFunction(fes)
        gfu.Set(function, definedon = data['mesh'].Boundaries(bc))

        dX.vec.data += gfu.vec.data
    
    else:

        def C(u):
            F = Grad(u) + Grad(u).trans
            return F.trans * F

        def NeoHooke (C):
            return Trace(C)

        fes = VectorH1(data['mesh'], order=dX.space.globalorder, dirichlet='.*')
        u  = fes.TrialFunction()

        a = BilinearForm(fes)
        a += Variation(NeoHooke(C(u)).Compile()*dx)

        gfu = GridFunction(fes)
        gfu.Set(function, definedon = data['mesh'].Boundaries(bc))

        Newton(a, gfu, maxit = 20, printing=False)

        dX.vec.data += gfu.vec.data

def sphere_transformation(A, b, t):

    phi = A*CF((x,y,z)) + b
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y,z)) - b)

    w_phi = A.Diff(t)*inv_phi + b.Diff(t)

    e1 = A[:,0]
    e2 = A[:,1]
    e3 = A[:,2]

    n_ex = Cross(e2, e3)*inv_phi[0]+Cross(e3, e1)*inv_phi[1] + Cross(e1, e2)*inv_phi[2]
    n_ex = n_ex/Norm(n_ex)

    return phi, inv_phi, w_phi, detJ, n_ex

def gradient(f,P):

    m, _ = P.dims
    l = len(f.dims)

    if l == 0:
        # scalar gradient is deifed traditionally

        if m == 2:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y)))
        elif m == 3:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y), f.Diff(z))) 
            
    elif l == 1:
        # vector gradient is defined component by component
        # and disposed along columns

        if m == 2:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            output = CoefficientFunction((aux1[0], aux2[0], \
                aux1[1], aux2[1]), dims = (m,m))
        elif m == 3:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            aux3 = gradient(f[2], P)
            output = CoefficientFunction((aux1[0], aux2[0], aux3[0], \
                aux1[1], aux2[1], aux3[1],\
                    aux1[2], aux2[2], aux3[2]), dims = (m,m))
            
    else:

        raise RuntimeError("Don't know how to take the gradient. Only scalars and vectors are accepted.")
    
    return output