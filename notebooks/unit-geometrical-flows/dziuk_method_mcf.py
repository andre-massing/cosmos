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

# Displacement to apply deform initial mesh to final mesh
dXh = GridFunction(V)
dXh.vec[:] = 0

# Store identity mapping as grid function
Idh = GridFunction(V)
Idh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# Inital deformation
Xh = GridFunction(V)
Xh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

# scene = Draw(Xh, mesh, deformation=Xh, vectors=True)
scene = Draw(Norm(dXh), mesh, deformation=dXh)
# TODO: vectors=True does not draw vectors on surface mesh?
# scene = Draw(CF((x,y,z)), mesh, deformation=Xh, vectors=True)
# norm_Xh = Norm(Xh)

# %% Define weak form
X,Y = V.TnT()

M = BilinearForm(V)
M += (X*Y + tau*InnerProduct(grad(X).Trace(), grad(Y).Trace()))*ds(deformation=dXh)
M.Assemble()
Minv = M.mat.Inverse(V.FreeDofs(), inverse="sparsecholesky")

l = LinearForm(V)
l += CF((x,y,z))*Y*ds(deformation=dXh)
l.Assemble()

#  %% Time loop
i = 0
t = 0
with TaskManager():
    while t <= Tend-tau:
        print ("\rt=", t, end="")
        M.Assemble()
        l.Assemble()
        Minv.Update()
        Xh.vec.data = Minv*l.vec
        dXh.vec.data = Xh.vec - Idh.vec
        
        if i % 10 == 0:
            scene.Redraw()
        
        t += tau
        i += 1