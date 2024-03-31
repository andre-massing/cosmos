# %% Import required modules
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw
ngsglobals.msg_level = 2

import sys
sys.path.insert(0, "../../")
from cosmos.utils.generate_surface_meshes import *

shape = "torus"
order_g = 1
maxh = 0.1
mass_lumped = True

# %% Define geometry and mesh it
# TODO: Define better torus geometry!
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
# %% Define relevant function spaces
order_p = order_g
# Function space for displacement
V = VectorH1(mesh, order=order_p)
# Function space for curvature
Q = H1(mesh, order=order_p)
# Mixed space
W = V*Q

# Store identity mapping as grid function,
# to be substracted from parameter mapping to compute 
# displacement of initial mesh
Idh = GridFunction(V)
Idh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# %% Compute mean curvature for start geometry
# Note that we redefine some of these later,
# using mixed space W instead of V ...
# TODO: Put this into a separate utility function
# TODO: Compute mean curvature via stabilized HLL approach
kappavec, eta = V.TnT()
nu = specialcf.normal(3)

ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6] )
M = BilinearForm(V)
if not mass_lumped: 
    print("Using standard inner products ...")
    M += InnerProduct(kappavec, eta)*ds
else:
    print("Using mass-lumped inner products ...")
    ds_lumping = ds(intrules = { TRIG : ir })
    M += InnerProduct(kappavec, eta)*ds_lumping

l = LinearForm(V)
l += InnerProduct(grad(Idh).Trace(), grad(eta).Trace())*ds

M.Assemble()
l.Assemble()

Minv = M.mat.Inverse(V.FreeDofs(), inverse="umfpack")
kappavech = GridFunction(V)
kappavech.vec.data = Minv*l.vec

Draw(kappavech, mesh)

# %% Now postprocess to compute actual mean curvature
kappa, chi = Q.TnT()
M = BilinearForm(Q)
M += kappa*chi*ds

l = LinearForm(Q)
l += InnerProduct(kappavech, nu)*chi*ds

M.Assemble()
l.Assemble()

Minv = M.mat.Inverse(Q.FreeDofs(), inverse="umfpack")
kappah = GridFunction(Q)
kappah.vec.data = Minv*l.vec
Draw(kappah, mesh)

# %% Set solver for geometric evolution problem

# Test and trial functions
(X, kappa), (eta, chi) = W.TnT()

# Displacement to deform initial mesh
dXh = GridFunction(V)
dXh.vec[:] = 0

# Combined parameter mapping and discrete mean curvature as grid f
Xkappah = GridFunction(W)
Xh, kappah = Xkappah.components

# TODO: Just copy values from Idh
Xh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# scene = Draw(Xh, mesh, deformation=Xh, vectors=True)
scene = Draw(kappah, mesh, deformation=dXh)


# %% Define weak form to compute initial mean curvature
