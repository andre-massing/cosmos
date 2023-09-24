# %% Import relevant modules
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw
ngsglobals.msg_level = 2

import sys
sys.path.insert(0, "../../")
from sandbox.generate_surface_meshes import *

# shape = "sphere"
shape = "torus"
order_g = 1
maxh = 0.1

# %% Define geometry and mesh it
if shape == "sphere" :
    R = 1
    sphere = Sphere((0,0,0),R).faces[0]

    # Radius of exact solution at time t
    ex_R = lambda t : sqrt(R**2-4*t)
    # Extinction time
    Tend = R**2/4
    geo = OCCGeometry(sphere)

    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order_g)
else :
    # Mesh geometry with curved elements of order 2
    # Torus mesh
    mesh = generate_torus_mesh(maxh=maxh)
    mesh.Curve(order_g)
    Tend = 0.11

Draw(mesh)

# %% Define mixed function space for parametrization and curvature
order_p = order_g
V = VectorH1(mesh, order=order_p)
Q = H1(mesh, order=order_p)
W = V*Q

# Displacement to deform initial mesh into final mesh
dXh = GridFunction(V)
dXh.vec[:] = 0

# Combined parameter mapping and discrete mean curvature as grid function
Xkappah = GridFunction(W)
Xh, kappah = Xkappah.components
# Setting Initial deformation and mean curvature (latter we don't need )
Xh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# Store identity mapping as grid function,
# to be substracted from parameter mapping to compute 
# displacement of initial mesh
Idh = GridFunction(V)
Idh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# scene = Draw(Xh, mesh, deformation=Xh, vectors=True)
scene = Draw(kappah, mesh, deformation=dXh)
# TODO: vectors=True does not draw vectors on surface mesh?
# scene = Draw(CF((x,y,z)), mesh, deformation=Xh, vectors=True)
# norm_Xh = Norm(Xh)

# %% Define weak form
(X, kappa), (eta, chi) = W.TnT()
nu = specialcf.normal(3)

mass_lumped = True
# time step
# tau = 0.125*maxh**2
tau = 2.0e-4
print(f"Using time step tau = {tau}")

M = BilinearForm(W)
l = LinearForm(W)

if not mass_lumped : 
    print("Using standard inner products ...")
    # TODO: Switch sign of second equation to use SparseCholesky
    M += (InnerProduct(X,nu)*chi - tau*kappa*chi)*ds(deformation=dXh)
    M += tau*kappa*InnerProduct(nu,eta)*ds(deformation=dXh)
    M += tau*InnerProduct(grad(X).Trace(), grad(eta).Trace())*ds(deformation=dXh)
    
    l += InnerProduct(CF((x,y,z)),nu)*chi*ds(deformation=dXh)
else:
    # We follow BGN more closely and define a mass lumped inner product
    # Define simplified quadrature rule for mass lamping
    print("Using mass-lumped inner products ...")
    ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6] )
    ds_lumping = ds(intrules = { TRIG : ir }, deformation=dXh)
    M += (InnerProduct(X,nu)*chi - tau*kappa*chi)*ds_lumping
    M += tau*kappa*InnerProduct(nu,eta)*ds_lumping
    M += tau*(InnerProduct(grad(X).Trace(), grad(eta).Trace()))*ds(deformation=dXh)

    l += InnerProduct(CF((x,y,z)),nu)*chi*ds_lumping

M.Assemble()
Minv = M.mat.Inverse(W.FreeDofs(), inverse="umfpack")
l.Assemble()

#  %% Time loop
i = 0
t = 0
with TaskManager():
    while t <= Tend-tau:
    # while t <= 1*tau:
        print ("\rt=", t, end="")
        # print (f"t={t}\n")
        M.Assemble()
        l.Assemble()
        Minv.Update()
        Xkappah.vec.data = Minv*l.vec
        # dXh.vec.data = Xh.vec - Idh.vec
        dXh.vec.data = Xkappah.components[0].vec - Idh.vec
        
        if i % 1 == 0:
            scene.Redraw()
        
        t += tau
        i += 1
# %%
