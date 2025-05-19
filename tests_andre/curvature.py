# %%

from cosmos.utils.generate_surface_meshes import generate_torus, generate_sphere
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np

def ComputeMC(mesh, gfu, params):

    if mesh.dim == 2:
        ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ds_lumped = ds(intrules = { SEGM : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir })
    elif mesh.dim == 3:
        ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        ds_lumped = ds(intrules = { TRIG : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { TRIG : ir })

    ns = specialcf.normal(mesh.dim)
    tE = specialcf.tangential(mesh.dim)
    if mesh.dim == 2:
        nE = tE
    else:
        nE = Cross(ns, tE)
    Ps = Id(mesh.dim) - OuterProduct(ns, ns)

    fes0 = gfu.space
    gfu0 = GridFunction(fes0)
    kappa0, eta0 = fes0.TnT()
    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds_lumped
    A0.Assemble()
    F0 += -InnerProduct(Ps, grad(eta0).Trace())*ds

    if params['clamped_bnd']:

        if mesh.dim == 3:
            gfF = GridFunction(FacetSurface(mesh, order=0))
            gfF.Set(1, definedon=mesh.BBoundaries(params['clamped_bnd']))
            F0 += InnerProduct(nE, eta0) * gfF * ds_el_lumped

        elif mesh.dim == 2:
            gfF = GridFunction(H1(mesh, order =1,\
                    definedon=mesh.Boundaries('.*')))
            gfF.Set(1, definedon=mesh.BBoundaries(params['clamped_bnd']))
            F0 += InnerProduct(gfF*nE, eta0) * ds_el_lumped

    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec
    gfu.vec.data = gfu0.vec.data

def ComputeStabMC(mesh, gfu, params):

    if mesh.dim == 2:
        ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ds_lumped = ds(intrules = { SEGM : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir })
    elif mesh.dim == 3:
        ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        ds_lumped = ds(intrules = { TRIG : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { TRIG : ir })

    if mesh.dim == 2:
        dV = H1(mesh, order=1,\
                    definedon = mesh.Boundaries('.*'))
    elif mesh.dim == 3:
        dV = NormalFacetSurface(mesh, order=0,\
                definedon = mesh.Boundaries('.*'))
        
    h = specialcf.mesh_size
    ns = specialcf.normal(mesh.dim)
    tE = specialcf.tangential(mesh.dim)
    if mesh.dim == 2:
        nE = tE
        tEc = CF((-ns[1], ns[0]))
    else:
        nE = Cross(ns, tE)
    Ps = Id(mesh.dim) - OuterProduct(ns, ns)

    Idh = GridFunction(gfu.space)
    Idh.Set(CF((x,y,z)), definedon=mesh.Boundaries(".*"))

    fes0 = gfu.space*dV
    gfu0 = GridFunction(fes0)
    (kappa0, dkappa0), (eta0, deta0) = fes0.TnT()
    if mesh.dim == 2:
        jump_dkappadn0 = (grad(kappa0).Trace()*nE-dkappa0*tEc)
        jump_detadn0 = (grad(eta0).Trace()*nE-deta0*tEc)
    elif mesh.dim == 3:
        jump_dkappadn0 = (grad(kappa0).Trace()*nE-dkappa0.Trace())
        jump_detadn0 = (grad(eta0).Trace()*nE-deta0.Trace())
    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds

    J = specialcf.JacobianMatrix(mesh.dim, mesh.dim-1)
    area = sqrt(Det(J.trans*J))/2
    length = sqrt(area/pi)
    length1 = ComputeH(J)

    Draw(length/length1, mesh)

    A0 += params['stab']*length1*InnerProduct(jump_dkappadn0,jump_detadn0)\
        *ds(element_boundary=True)
    A0.Assemble()
    F0 += -InnerProduct(Ps, grad(eta0).Trace())*ds

    if params['clamped_bnd']:

        if mesh.dim == 3:
            gfF = GridFunction(FacetSurface(mesh, order=0))
            gfF.Set(1, definedon=mesh.BBoundaries(params['clamped_bnd']))
            F0 += InnerProduct(nE, eta0) * gfF * ds_el_lumped

        elif mesh.dim == 2:
            gfF = GridFunction(H1(mesh, order =1,\
                    definedon=mesh.Boundaries('.*')))
            gfF.Set(1, definedon=mesh.BBoundaries(params['clamped_bnd']))
            F0 += InnerProduct(gfF*nE, eta0) * ds_el_lumped

    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec
    gfu.vec.data = gfu0.components[0].vec.data

def ComputeModMC(mesh, gfu, params):

    if mesh.dim == 2:
        ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ds_lumped = ds(intrules = { SEGM : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir })
    elif mesh.dim == 3:
        ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        ds_lumped = ds(intrules = { TRIG : ir })
        ds_el_lumped = ds(element_boundary=True, intrules = { TRIG : ir })

    if mesh.dim == 2:
        dV = H1(mesh, order=1,\
                    definedon = mesh.Boundaries('.*'))
    elif mesh.dim == 3:
        dV = NormalFacetSurface(mesh, order=0,\
                definedon = mesh.Boundaries('.*'))
        
    h = specialcf.mesh_size
    ns = specialcf.normal(mesh.dim)
    tE = specialcf.tangential(mesh.dim)
    if mesh.dim == 2:
        nE = tE
        tEc = CF((-ns[1], ns[0]))
    else:
        nE = Cross(ns, tE)
    Ps = Id(mesh.dim) - OuterProduct(ns, ns)

    Idh = GridFunction(gfu.space)
    Idh.Set(CF((x,y,z)), definedon=mesh.Boundaries(".*"))

    fes0 = gfu.space*dV
    gfu0 = GridFunction(fes0)
    (kappa0, dkappa0), (eta0, deta0) = fes0.TnT()
    if mesh.dim == 2:
        dkappa0 = dkappa0*tE
        deta0 = deta0*tE
        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0*tEc)
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0*tEc)
    elif mesh.dim == 3:
        jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0.Trace())
        jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0.Trace())
    A0 = BilinearForm(fes0)
    F0 = LinearForm(fes0)
    A0 += kappa0*eta0*ds_lumped
    A0 += params['stab']*h*InnerProduct(jump_dkappadn0,jump_detadn0)\
        *ds_el_lumped
    A0.Assemble()
    F0 += -InnerProduct(Ps, grad(eta0).Trace())*ds_lumped

    if params['clamped_bnd']:

        if mesh.dim == 3:
            gfF = GridFunction(FacetSurface(mesh, order=0))
            gfF.Set(1, definedon=mesh.BBoundaries(params['clamped_bnd']))
            F0 += InnerProduct(nE, eta0) * gfF * ds_el_lumped

        elif mesh.dim == 2:
            gfF = GridFunction(H1(mesh, order =1,\
                    definedon=mesh.Boundaries('.*')))
            gfF.Set(1, definedon=mesh.BBoundaries(params['clamped_bnd']))
            F0 += InnerProduct(gfF*nE, eta0) * ds_el_lumped

    F0.Assemble()
    gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec
    gfu.vec.data = gfu0.components[0].vec.data

def ComputeH(J):

    a = Norm(J[:,0])
    b = Norm(J[:,1])
    c = Norm(J[:,0] - J[:,1])

    # max = IfPos(IfPos(a-b, a, b) - c, IfPos(a-b, a, b), c)
    max = IfPos(IfPos(a-b, b, a) - c, c, IfPos(a-b, b, a))

    return max

def ComputeError(mesh, u_h, u_ex):

    ns = specialcf.normal(mesh.dim)

    aux = InnerProduct(ns*(u_h-u_ex), ns*(u_h-u_ex))
    err = sqrt(Integrate(aux, mesh = mesh, order = u_h.space.globalorder*2, 
                    VOL_or_BND=BND))
    
    return err

print('-'*10, 'Torus')
R = sqrt(2)
r =1
n = CF((x,y,z))/r-R*CF((x,0,z))/r/sqrt(x**2+z**2)
cosv = (sqrt(x**2+z**2)-R)/r
H = 2*(R+2*r*cosv)/(2*r*(R+r*cosv))
H_ex = -n*H

power = 2
nref = 3
hs = 0.2/np.power(power, range(nref))
ERRS = np.zeros((len(hs), 3))
for i, h in enumerate(hs):

    mesh, _ = generate_torus(R=R, r=r, maxh = h)
    kappa_h = GridFunction(VectorH1(mesh))

    ComputeMC(mesh, kappa_h, {'clamped_bnd': None})
    err1 = ComputeError(mesh, kappa_h, H_ex)

    ComputeStabMC(mesh, kappa_h, {'stab': 1e-3, 'clamped_bnd': None})
    err2 = ComputeError(mesh, kappa_h, H_ex)

    ComputeModMC(mesh, kappa_h, {'stab': 1e-3, 'clamped_bnd': None})
    err3 = ComputeError(mesh, kappa_h, H_ex)

    print(err1, err2, err3)

    ERRS[i, 0] = err1
    ERRS[i, 1] = err2
    ERRS[i, 2] = err3

print(np.log(ERRS[:-1, :]/ERRS[1:, :])/np.log(power))

print('-'*10, 'Sphere')
R = 1
n = CF((x,y,z))/Norm( CF((x,y,z)))
H_ex = -2*n

power = 2
nref = 4
hs = 0.2/np.power(power, range(nref))
ERRS = np.zeros((len(hs), 3))
for i, h in enumerate(hs):

    mesh, _ = generate_sphere(R=1, maxh = h)
    kappa_h = GridFunction(VectorH1(mesh))

    ComputeMC(mesh, kappa_h, {'clamped_bnd': None})
    err1 = ComputeError(mesh, kappa_h, H_ex)

    ComputeStabMC(mesh, kappa_h, {'stab': 1e-3, 'clamped_bnd': None})
    err2 = ComputeError(mesh, kappa_h, H_ex)

    ComputeModMC(mesh, kappa_h, {'stab': 1e-3, 'clamped_bnd': None})
    err3 = ComputeError(mesh, kappa_h, H_ex)

    print(err1, err2, err3)

    ERRS[i, 0] = err1
    ERRS[i, 1] = err2
    ERRS[i, 2] = err3

print(np.log(ERRS[:-1, :]/ERRS[1:, :])/np.log(power))


