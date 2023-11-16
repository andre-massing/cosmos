#%% 
from netgen.csg import *
from netgen.meshing import MeshingStep
from ngsolve.meshes import MakeStructuredSurfaceMesh
from ngsolve import *
from ngsolve.webgui import Draw
ngsglobals.msg_level = 1
import numpy as np
#%% Define geometry
# Generate mesh and plot it
maxh = 0.5
# mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))

# Define identity embedding for plane
mapping = lambda x,y,z : (x, y, y*(1-y))
mesh = MakeStructuredSurfaceMesh(quads=False, nx=4, ny=4, mapping=mapping)
mesh.Refine(mark_surface_elements=True)

#%% Now curve the mesh to obtain better geometry approximation
#   We use a superparametric approximation to ensure 
#   optimal convergence of uh also in w.r.t to normal component 
order_k = 1
order_g = order_k+1
mesh.Curve(order_g)
Draw(mesh)

# Check convergence of normal vector, mean curvature  (and B itself)
Hhb_errs = []
H_errs = []

def compute_eoc(errs):
    return np.log(np.array(errs)[:-1]/np.array(errs)[1:])/np.log(2)

#%% Define function spaces and variational problem

# Define geometry related object
# Normal vector
ns = specialcf.normal(mesh.dim)
# normal and tangential  projection
Qs = OuterProduct(ns, ns)
Ps = Id(mesh.dim) - Qs
# Weingarten map
Bs = Grad(ns)
# specialcf.Weingarten(mesh.dim)

# Discrete gradient for non-tangential vector fields
# def Grad_h(u):
#     Grad(u)

# V = H1(mesh, order=int(order_k), dims=(3,3))
# V = H1(mesh, order=int(order_k))
# V = H1(mesh, order=int(order_k), dim = 3)
# V = H1(mesh, order=int(order_k), dim=(2,2))
V = VectorH1(mesh, order=int(order_k))
# V = HDivSurface(mesh, order=int(order_k))
# V = H1(mesh, order=int(order_k))
u, v = V.TnT()

Mstar = BilinearForm(V)
# Mstar += InnerProduct(u, v)*dx
Mstar += InnerProduct(Grad(u).Trace(), Grad(v).Trace())*ds
Mstar += InnerProduct(u.Trace(), v.Trace())*ds
# Mstar += InnerProduct(Sym(Grad(u).Trace()), Sym(Grad(v).Trace()))*ds
Mstar.Assemble()
print(Mstar.mat)
# Mstar.mat.COO()
# print(Mstar.mat.height, Mstar.mat.width)
# print(Mstar.mat.FV().NumPy())

#%%
# f = Cof((x, 0, x, y, y, 0, 0,z, z), dims=(3,3))
# f = x*y
f = CF((x, y, z))
fh = GridFunction(V)
fh.Set(f, definedon=mesh.Boundaries(".*"))
l = LinearForm(V)
l += InnerProduct(Grad(fh).Trace(), Grad(v).Trace())*ds
l += InnerProduct(fh.Trace(), v.Trace())*ds
# l += f*v*dx

l.Assemble()
print(l.vec)
print(l.vec.size)

u_h =  GridFunction(V)
u_h.vec.data = Mstar.mat.Inverse(freedofs=V.FreeDofs())*l.vec

vtkfile = VTKOutput(mesh, coefs=[u_h], names=["u_h"], filename="results")
vtkfile.Do(vb=BND) 

# vecfun = CF((-y, sin(x)))
# Draw(vecfun, mesh, vectors=True);

   

# %%
