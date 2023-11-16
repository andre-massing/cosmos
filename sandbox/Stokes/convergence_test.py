# %% 
# Importing the necessary libraries

import time
# caution: path[0] is reserved for script path (or '' in REPL)
from netgen.occ import *
from netgen.meshing import MeshingStep
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import pandas as pd
from utils import *

from stokes_th import StokesTH
from stokes_hdg import StokesHDG
       
# %% Define manufactured solution
dimension = 3 # dimension of the embedding space
R = 1.0
levelset_str = "x*x + y*y + z*z -" + str(R)

phi = CoefficientFunction(eval(levelset_str))
normal = compute_normal(phi,dimension)
# Tangential projection
Ps = tang_project(normal)

# Define manufactured solutions and corresponding PDE data 
t1 = time.time()

# Manufactured solution
_u = CoefficientFunction((x**3, sin(y), z))
u_ex = Ps*_u
p_ex = CoefficientFunction(sin(x))

nu = Parameter(1.0)
# Right-hand side
f = -nu*Ps*divergence(Sym(gradient(u_ex, Ps)),Ps) + u_ex + gradient(p_ex, Ps)
f = f.Compile()
g = divergence(u_ex, Ps)
g = g.Compile()

t2 = time.time()
print("Time for symbolic computations vector case: {:.2f}".format(t2-t1))

# Define geometry and mesh
geo          = CSGeometry()
sphere       = Sphere(Pnt(0,0,0), 1)
bot          = Plane(Pnt(0,0,0), Vec(0,0,-1))
# finitesphere = sphere * bot

geo.Add(sphere)
# geo.AddSurface(sphere, finitesphere.bc("surface"))
# geo.AddSurface(sphere, finitesphere.bc("surface"))
# geo.NameEdge(sphere,bot, "bottom")

# %% Run convergence study
fes_order_list = [2]
num_refs = 3
ref_fac = 1.5
maxh0 = 0.25

# make directory if it does not exist
results_dir = "results/"
import os
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# solvers = [StokesTH, StokesHDG]
solvers = [StokesHDG]
# solvers = [StokesTH]

for solver in solvers:
    for fes_order in fes_order_list:
        error_list = [[], [], []]
        maxh = maxh0
        for ref in range(num_refs+1):
                mesh = Mesh(geo.GenerateMesh(maxh=maxh, perfstepsend=MeshingStep.MESHSURFACE))
                order_u, order_p = fes_order, fes_order-1
                # order_g = order_u
                order_g = order_u+1
                solver_name = solver.__name__
                file_name = os.path.join(results_dir, f"{solver_name}_{order_u}_{order_p}_ref_{ref}")
                mesh.Curve(order_g)
                u_h, p_h = solver(mesh, 
                                  order_u=order_u,
                                  nu=nu,
                                  f=f, g=g, 
                                  filename=file_name)
                # Compute errors
                errors = compute_errors(mesh, u_ex, u_h, order_u, p_ex, p_h, order_p)
                # Append errors to error list
                for i, error in enumerate(errors):
                    error_list[i].append(error)
                # Compute current eoc tables
                eoc_list = [compute_eoc(errors, ref_fac) for errors in error_list]
                # error_eoc_list = zip(error_list, eoc_list)
                table = pd.DataFrame({'L2_error_u': error_list[0],
                                    'L2_eoc_u'  : eoc_list[0],
                                    'H1_error_u': error_list[1],
                                    'H1_eoc_u'  : eoc_list[1],
                                    'L2_error_p': error_list[2],
                                    'L2_eoc_p'  : eoc_list[2]})
                display(table)
                # print(table.to_string())
                maxh /= ref_fac
                # if ref < num_refs:
                #     mesh.Curve(1)
                #     mesh.Refine(mark_surface_elements=True)
                #     mesh.Curve(order_g)