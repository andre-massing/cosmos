# %%

# Case 1 using deformation = dX and skeleton. All skeleton terms are ignored
# and case is solved in original configuration
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.webgui import Draw

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))

order = 1

V = H1(mesh, order=order)
u = V.TrialFunction()
v = V.TestFunction()

dX = GridFunction(VectorH1(mesh))
dX.Set(CF((x, y)))

n = specialcf.normal(2)
h = specialcf.mesh_size
penalty = 10

u_ex = x**2+y**2
f = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)

mesh.SetDeformation(dX)

a = BilinearForm(V, symmetric=True)
a += grad(u)*grad(v)*dx(deformation=dX)
a += (-n*grad(u)*v )*ds(deformation=dX, skeleton=True)
a += (-n*grad(v)*u)*ds(deformation=dX, skeleton=True)
a += (penalty/h*u*v)*ds(deformation=dX)
a.Assemble()

l = LinearForm(V)
l +=  f* v * dx(deformation=dX)
l += ( -n*grad(v)*u_ex)*ds(deformation=dX, skeleton=True)
l += ( penalty/h*u_ex*v)*ds(deformation=dX)
l.Assemble()

u = GridFunction(V)
u.vec.data = a.mat.Inverse() * l.vec

Draw(u)

# %%

# See if the contribution is 0 with deformation flag or just ignored

from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.webgui import Draw

mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))

order = 1

V = H1(mesh, order=order)
u = V.TrialFunction()
v = V.TestFunction()

u_ex = x**2+y**2
f = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)

dX = GridFunction(VectorH1(mesh))
dX.Set(CF((x, y)))
n = specialcf.normal(2)
h = specialcf.mesh_size
penalty = 10

a = BilinearForm(V)
a += (-n*grad(u)*v)*ds(deformation = dX, skeleton=True)
# a += (penalty/h*u*v)*ds(deformation=dX, skeleton=True)
a.Assemble()

print(a.mat)

l = LinearForm(V)
l += (-n*grad(v)*u_ex)*ds(deformation = dX, skeleton=True)
l.Assemble()

print(l.vec)