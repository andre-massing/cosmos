# %%
from manufactured_solution_tools import *

from sympy import *
from sympy.printing import latex, pretty
from sympy.parsing.sympy_parser import parse_expr
import numpy as np

# Level set define Gamma
x,y,z = symbols("x y z") 
xxx = Matrix([x, y, z])

R = 1.0
phi_str = "x*x + y*y + z*z -" + str(R)

# Exact solution
u_str = "sin(x)*sin(y)*sin(z)"

# Compute rhs
f_str, grad_u_str = get_solution_str(u_str, phi_str, xxx)
print(f"f_str is {f_str}")
print(grad_u_str)
type(f_str)

from math import pi
from ngsolve import *

# Choose orders for space and geometry approximation

# Order of finite elemet space
order = 3
# Geometry order
order_g = order
#order_g = 1

# ## 1. Generating the initial surface mesh
from netgen.csg import *
geo = CSGeometry()
geo.Add(Sphere(Pnt(0,0,0), 1).bc("sphere")) 

from netgen.meshing import MeshingParameters, MeshingStep
maxh=0.5
mp = MeshingParameters(maxh=maxh, perfstepsend=MeshingStep.MESHSURFACE)
# TODO: Find a list over all meshing parameters
mesh = Mesh(geo.GenerateMesh(mp=mp))

if order_g > 1:
    mesh.Curve(order_g)
Draw(mesh)

# Compute rhs

# f_str = "sin(x)*sin(y)*sin(z)"
f = eval(f_str)
print(f)

l2_errors = []
h1_errors = []

ref_level_max = 3
for ref_level in range(ref_level_max+1):
    
    # Don't refine the first time
    # N.B. Refining at the end of the loop invalidates u_h
    print(f"Refinement level = {ref_level}")
    if ref_level > 0:
        mesh.Curve(1)
        # Note that surface elements are by default NOT marked. So we need to do a
        mesh.Refine(mark_surface_elements=True)
        mesh.Curve(order_g)
    Redraw()

    print(f"Number of vertices {mesh.nv}")
    print(f"Number of elements {mesh.ne}")

    # ## Surface mesh related quantities
    print("Mesh dim %d" % mesh.dim)

    # ## 2. Discretization

    # ### Surface finite element spaces
    Vh = H1(mesh, order=order)

    u, v = Vh.TnT()

    # ## Surface gradients, co-normal derivatives and jumps
    grad_u = u.Trace().Deriv()
    grad_v = v.Trace().Deriv()

    # ## Bilinear and linear forms
    a = BilinearForm(Vh)
    a += (grad_u * grad_v + u*v)*ds

    l = LinearForm(Vh)
    l += f*v*ds

    # ## Assemble and solve
    a.Assemble()
    l.Assemble()

    u_h = GridFunction(Vh)
    u_h.vec.data = a.mat.Inverse(freedofs=Vh.FreeDofs()) * l.vec
    Draw(u_h, mesh, "u")

    # Compute L2 errors
    u_ex = eval(u_str)
    err_sqr_coefs = (u_ex-u_h)**2
    l2_error = sqrt(Integrate(cf = err_sqr_coefs, mesh=mesh, order=order, VOL_or_BND = BND) )
    print("L2 error: ||u-u_h||_L2 = %e" % l2_error)
    l2_errors.append(l2_error)
    
    # Compute H1 errors by computing a projection into higher-order space
    Vh_ex =H1(mesh, order=order+2)
    
    u_ex_h = GridFunction(Vh_ex)
    # Have to use Set in this way
    u_ex_h.Set(u_ex, definedon=~mesh.Boundaries(''))
    grad_u_ex_h = u_ex_h.Deriv()  # No trace?
    
    u_h_inter = GridFunction(Vh_ex)
    u_h_inter.Set(u_h, definedon=~mesh.Boundaries(''))
    grad_u_h_inter = u_h_inter.Deriv()
    
    h1_error = sqrt(Integrate((grad_u_h_inter-grad_u_ex_h)*(grad_u_h_inter - grad_u_ex_h), mesh, order=order, VOL_or_BND = BND))
    print("H1 error: ||grad(u-u_h)||_L2 = %e" % h1_error)
    h1_errors.append(h1_error)
    

l2_errors = np.array(l2_errors)
l2_eocs = np.log(l2_errors[:-1]/l2_errors[1:])/np.log(2)

print("========================================")
print("Computed L2 EOCs:")
print(l2_eocs)

h1_errors = np.array(h1_errors)
h1_eocs = np.log(h1_errors[:-1]/h1_errors[1:])/np.log(2)

print("========================================")
print("Computed H1 EOCs:")
print(h1_eocs)
