#%% 
from netgen.csg import *
from netgen.meshing import MeshingStep
from ngsolve import *
from ngsolve.webgui import Draw
ngsglobals.msg_level = 1
import numpy as np
#%% Define geometry
geo = CSGeometry()
R = 1 
center = Pnt(0,0,0)
geo.Add(Sphere(center, R))

# Generate mesh and plot it
maxh = 0.5
mesh = Mesh(geo.GenerateMesh(maxh=maxh, perfstepsend=MeshingStep.MESHSURFACE))

#%% Now curve the mesh to obtain better geometry approximation
#   We use a superparametric approximation to ensure 
#   optimal convergence of uh also in w.r.t to normal component 
order_k = 2
order_g = order_k+1
mesh.Curve(order_g)
Draw(mesh)

Hhb_errs = []
H_errs = []

#%% Check convergence of normal vector, mean curvature  (and B itself)

def compute_eoc(errs):
    return np.log(np.array(errs)[:-1]/np.array(errs)[1:])/np.log(2)

num_refs = 4
for ref in range(0,num_refs+1):
    # Define geometry related quantities
    # Surface normal
    nsurf = specialcf.normal(3)

    # Weingarten operator
    # Check sign/orientation
    B = Grad(nsurf)

    # Compute Gaussian and mean curvature
    H = Trace(Grad(nsurf))
    Draw(H, mesh)

    # Define better geometry approximations
    # n_h with better approximation properties
    Vnhb = VectorH1(mesh, order=order_k)
    nhb = GridFunction(Vnhb)
    nhb.Set(nsurf, definedon=mesh.Boundaries(".*"))

    Hhb = Trace(Grad(nhb))
    Draw(Hhb, mesh)

    VH = H1(mesh, order=1)

    H_low = GridFunction(VH)
    H_low.Set(H, definedon=mesh.Boundaries(".*"))
    # Substract exact Gauss curvature
    H_low.vec.FV().NumPy()[:] -= 2
    H_errs.append(np.abs(H_low.vec.FV().NumPy()).max())
    print(f"Maximum error in Gaussian Curvature = {H_errs[-1]}")
    Draw(H_low)

    Hhb_low = GridFunction(VH)
    Hhb_low.Set(Hhb, definedon=mesh.Boundaries(".*"))
    Hhb_low.vec.FV().NumPy()[:] -= 2
    Hhb_errs.append(np.abs(Hhb_low.vec.FV().NumPy()).max())
    print(f"Maximum error in Gaussian Curvature = {Hhb_errs[-1]}")
    
    Draw(Hhb_low)
    
    # Refine  
    mesh.Curve(1)
    mesh.Refine(mark_surface_elements=True)
    mesh.Curve(order_g)

print(compute_eoc(H_errs))
print(compute_eoc(Hhb_errs))


# %%
