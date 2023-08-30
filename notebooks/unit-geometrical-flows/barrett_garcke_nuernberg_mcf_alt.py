# %% Import relevant modules
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw
ngsglobals.msg_level = 2

# %% Define geometry and mesh it
R = 1
sphere = Sphere((0,0,0),R).faces[0]

# Radius of exact solution at time t
ex_R = lambda t : sqrt(R**2-4*t)
# Extinction time
Tend = R**2/4

# Mesh geometry with curved elements of order 2
order_g = 1
geo = OCCGeometry(sphere)

maxh = 0.1
mesh = Mesh(geo.GenerateMesh(maxh=maxh))
mesh.Curve(order_g)
Draw(mesh)

# %% Define mixed function space for parametrization and curvature
order_p = order_g
V = VectorH1(mesh, order=order_p)

# Displacement to deform initial mesh into final mesh
# TODO: Check whether we can simply take dXh = GridFunction(V)
#       instead of full mixed space W
dXh = GridFunction(V)
dXh.vec[:] = 0

# scene = Draw(Xh, mesh, deformation=Xh, vectors=True)
scene = Draw(Norm(dXh), mesh, deformation=dXh)
# TODO: vectors=True does not draw vectors on surface mesh?
# scene = Draw(CF((x,y,z)), mesh, deformation=Xh, vectors=True)
# norm_Xh = Norm(Xh)

# Combined parameter mapping and discrete mean curvature as grid function
Xh = GridFunction(V)
Xh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# Store identity mapping as grid function,
# to be substracted from parameter mapping to compute 
# displacement of initial mesh
Idh = GridFunction(V)
Idh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# Compute averaged/continuous normal field
nu_weighted = GridFunction(V)

# This is on the reference manifold, but should computed 
# on the deformed manifold...
use_weighted_normal = True
if use_weighted_normal:
    nu.Set(specialcf.normal(3), definedon=mesh.Boundaries(".*"))
else:
    nu = specialcf.normal(3)
# %% Define weak form
X, eta = V.TnT()

mass_lumped = False
# time step
# tau = 0.125*maxh**2
tau = 2e-4

M = BilinearForm(V)
l = LinearForm(V)

if not mass_lumped : 
    print("Using standard inner products ...")
    M += (InnerProduct(X,nu_weighted)*InnerProduct(eta, nu_weighted))*ds(deformation=dXh)
    M += tau*InnerProduct(grad(X).Trace(), grad(eta).Trace())*ds(deformation=dXh)
    
    l += (InnerProduct(CF((x,y,z)),nu_weighted)*InnerProduct(eta, nu_weighted))*ds(deformation=dXh)
else:
    # We follow BGN more closely and define a mass lumped inner product
    # Define simplified quadrature rule for mass lamping
    print("Using mass-lumped inner products ...")
    ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6] )
    ds_lumping = ds(intrules = { TRIG : ir }, deformation=dXh)
    M += (InnerProduct(X,nu_weighted)*InnerProduct(eta, nu_weighted))*ds_lumping
    M += tau*InnerProduct(grad(X).Trace(), grad(eta).Trace())*ds(deformation=dXh)
    
    l += (InnerProduct(CF((x,y,z)),nu_weighted)*InnerProduct(eta, nu_weighted))*ds_lumping

M.Assemble()
Minv = M.mat.Inverse(V.FreeDofs(), inverse="sparsecholesky")
l.Assemble()

#  %% Time loop
i = 0
t = 0
with TaskManager():
    while t <= Tend-tau:
    # while t <= 100*tau:
        print ("\rt=", t, end="")
        M.Assemble()
        l.Assemble()
        Minv.Update()
        Xh.vec.data = Minv*l.vec
        # dXh.vec.data = Xh.vec - Idh.vec
        dXh.vec.data = Xh.vec - Idh.vec
        
        if i % 10  == 0:
            scene.Redraw()
        
        t += tau
        i += 1
# %%
