# %% Import relevant modules
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw
ngsglobals.msg_level = 1

# %% Define geometry and mesh it
R = 1
sphere = Sphere((0,0,0),R).faces[0]

# Radius of exact solution at time t
ex_R = lambda t : sqrt(R**2-4*t)
# Extinction time
Tend = R**2/4
# time step
tau = 2e-4

# Mesh geometry with curved elements of order 2
order_g = 2
geo = OCCGeometry(sphere)

maxh = 0.15
mesh = Mesh(geo.GenerateMesh(maxh=maxh))
mesh.Curve(order_g)

# %% Define function space for parametrization 
order_p = 2
V = VectorH1(mesh, order=order_p)

# Displacement to move from current to next surface
# Note: This is NOT the displacement you apply to the 
#       initial surface you get to the final surface!
dXh = GridFunction(V)
dXh.vec[:] = 0

# Current mapping
Xhold = GridFunction(V)

# New mapping
Xh = GridFunction(V)
Xh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# Store identity mapping as grid function
Idh = GridFunction(V)
# Idh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# scene = Draw(Xh, mesh, deformation=Xh, vectors=True)
scene = Draw(Norm(dXh), mesh, deformation=dXh)
# TODO: vectors=True does not draw vectors on surface mesh?
# scene = Draw(CF((x,y,z)), mesh, deformation=Xh, vectors=True)
# norm_Xh = Norm(Xh)

# %% Define weak form
dX,Y = V.TnT()

M = BilinearForm(V)
M += (dX*Y + tau*InnerProduct(grad(dX).Trace(), grad(Y).Trace()))*ds(deformation=dXh)
M.Assemble()
Minv = M.mat.Inverse(V.FreeDofs(), inverse="sparsecholesky")

l = LinearForm(V)
l += -tau*InnerProduct(grad(Xhold).Trace(), grad(Y).Trace())*ds(deformation=dXh)
l.Assemble()

#  %% Time loop
# import time
# time.sleep(5)
i = 0
t = 0
with TaskManager():
    while t <= Tend-tau:
        print ("\rt=", t, end="")
        # Save old position
        Xhold.vec.data = Xh.vec
        M.Assemble()
        l.Assemble()
        Minv.Update()
        
        # Solve for displacement (w.r.t current Gamma_m) and apply new mesh displacement
        # Compute new displacement
        Xh.vec.data += Minv*l.vec 
        dXh.vec.data += Xh.vec - Xhold.vec
        
        if i % 10 == 0: 
            scene.Redraw()
        
        t += tau
        i += 1