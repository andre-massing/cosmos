# %%

from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from cosmos.pdes.pde_tools import SaveError, SaveSolution
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import pandas as pd

will_sol = [
    WillmoreDziuk(postprocess = None, time_scheme=BDF1()),
    WillmoreDziukStab(postprocess = None, time_scheme=BDF1())
]

for i, sol in enumerate(will_sol):

    sol_save = SaveSolution(folderpath = './willmore_spine',
                        filename = 'spine' + str(i),
                        sample_rate = 1)
    sol.SaveSol(sol_save)

    t = Parameter(0.0)
    dt = Parameter(1e-5)
    T = 1e-4

    mesh = Mesh('../../data/geometries/spine_closed.vol')

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(sol)
    solver.ale.deformation_field = sol.displacement

    scene = Draw(sol.mean_curvature, mesh, deformation = solver.ale.deformation, min = 0)
    for sol in solver():
        scene.Redraw()
# %%
