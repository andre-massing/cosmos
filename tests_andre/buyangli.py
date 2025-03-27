# %%

from ngsolve import *
from cosmos.utils.generate_surface_meshes import generate_sphere
from ngsolve.webgui import Draw

mesh, _ = generate_sphere(maxh = 0.1)

n = specialcf.normal(mesh.dim)
P = Id(mesh.dim) - OuterProduct(n, n)

displ = CF((x, -0.5*y, -0.5*z))

gauss = CF(exp(-4*(x**2+z**2)))
displ = CF((0, gauss*Norm(y), 0))

displ = CF((-y, x, 0))

fes = VectorH1(mesh, definedon=mesh.Boundaries('.*'))
gfu = GridFunction(fes)
Y = GridFunction(fes)

Y.Set(CF((x, y, z)),  definedon=mesh.Boundaries('.*'))

gfu.Set(displ, definedon = mesh.Boundaries('.*'))

Draw(mesh, deformation = gfu)

V1 = VectorH1(mesh, definedon=mesh.Boundaries('.*'))
V2 = H1(mesh, definedon=mesh.Boundaries('.*'))
fes = V1*V2

n_h = GridFunction(V1)

mesh.SetDeformation(gfu)
n_h.Set(n, definedon = mesh.Boundaries('.*'))
mesh.UnsetDeformation()

Jac = Grad(Y).Trace()*Grad(Y).Trace().trans + OuterProduct(n_h, n_h)

(u, p), (v, q) = fes.TnT()

# A = BilinearForm(fes)
# A += InnerProduct(Grad(u).Trace(), Grad(v).Trace())*ds
# A += -p*n_h*v*ds(deformation=gfu)
# A += -q*n_h*u*ds(deformation=gfu)
# F = LinearForm(fes)
# F += (-InnerProduct(Grad(gfu).Trace(), Grad(v).Trace()))*ds

A = BilinearForm(fes)
A += InnerProduct(Grad(u).Trace()*Inv(Jac), Grad(v).Trace())*Det(Jac)*ds(deformation=gfu)
A += -p*n_h*v*Det(Jac)*ds(deformation=gfu)
A += -q*n_h*u*Det(Jac)*ds(deformation=gfu)
F = LinearForm(fes)
F += (-InnerProduct(Grad(gfu).Trace()*Inv(Jac), Grad(v).Trace()))*Det(Jac)*ds(deformation=gfu)

A.Assemble()
F.Assemble()

gfu_t = GridFunction(fes)

gfu_t.vec.data = A.mat.Inverse(freedofs = fes.FreeDofs())*F.vec

mesh.SetDeformation(gfu)
Draw(gfu_t.components[0]*n, mesh)
mesh.UnsetDeformation()

gfu_tot = GridFunction(V1)
gfu_tot.Set(gfu + gfu_t.components[0], definedon = mesh.Boundaries('.*'))

mesh.SetDeformation(gfu_tot)
Draw(mesh)
mesh.UnsetDeformation()

# %%

from ngsolve import *
from cosmos.utils.generate_surface_meshes import generate_box
from ngsolve.webgui import Draw

mesh, _ = generate_box(maxh = 0.05, vol_or_bnd='BND')

n = specialcf.normal(mesh.dim)
P = Id(mesh.dim) - OuterProduct(n, n)

displ = CF((x, -0.5*y, -0.5*z))

# gauss = CF(exp(-4*(x**2+z**2)))
# displ = CF((0, gauss*Norm(y), 0))

# displ = CF((-y, x, 0))

fes = VectorH1(mesh, definedon=mesh.Boundaries('.*'))
gfu = GridFunction(fes)
Y = GridFunction(fes)

Y.Set(CF((x, y, z)),  definedon=mesh.Boundaries('.*'))

gfu.Set(displ, definedon = mesh.Boundaries('.*'))

# Draw(mesh, deformation = gfu)

V1 = VectorH1(mesh, definedon=mesh.Boundaries('.*'))
V2 = H1(mesh, definedon=mesh.Boundaries('.*'))
fes = V1*V2

n_h = GridFunction(V1)

mesh.SetDeformation(gfu)
n_h.Set(n, definedon = mesh.Boundaries('.*'))
mesh.UnsetDeformation()

Jac = Grad(Y).Trace()*Grad(Y).Trace().trans + OuterProduct(n_h, n_h)

(u, p), (v, q) = fes.TnT()

# A = BilinearForm(fes)
# A += InnerProduct(Grad(u).Trace(), Grad(v).Trace())*ds
# A += -p*n_h*v*ds(deformation=gfu)
# A += -q*n_h*u*ds(deformation=gfu)
# F = LinearForm(fes)
# F += (-InnerProduct(Grad(gfu).Trace(), Grad(v).Trace()))*ds

A = BilinearForm(fes)
A += InnerProduct(Grad(u).Trace()*Inv(Jac), Grad(v).Trace())*Det(Jac)*ds(deformation=gfu)
A += -p*n_h*v*Det(Jac)*ds(deformation=gfu)
A += -q*n_h*u*Det(Jac)*ds(deformation=gfu)
F = LinearForm(fes)
F += (-InnerProduct(Grad(gfu).Trace()*Inv(Jac), Grad(v).Trace()))*Det(Jac)*ds(deformation=gfu)

A.Assemble()
F.Assemble()

gfu_t = GridFunction(fes)

gfu_t.vec.data = A.mat.Inverse(freedofs = fes.FreeDofs())*F.vec

mesh.SetDeformation(gfu)
Draw(gfu_t.components[0]*n, mesh)
mesh.UnsetDeformation()

gfu_tot = GridFunction(V1)
gfu_tot.Set(gfu + gfu_t.components[0], definedon = mesh.Boundaries('.*'))

mesh.SetDeformation(gfu_tot)
Draw(mesh)
mesh.UnsetDeformation()