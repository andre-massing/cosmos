# %%
from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from netgen.occ import *
from cosmos.utils.generate_surface_meshes import generate_smoothed_box, generate_box
import numpy as np
import logging
from test_helfrich_config import CFG

import os
filename = os.path.splitext(os.path.basename(__file__))[0]

Id = CF((x, y, z))
n = specialcf.normal(3)

maxh = CFG.maxh
mesh, _ = generate_smoothed_box(a = 3, b=1, c=3, maxh = maxh)

logging.getLogger().setLevel(logging.INFO)

dt = CFG.dt
Tend = CFG.Tend
solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)

solvermesh = SolverMesh(mesh)
solver = Solver(solvermesh, solvertime, iter=False,
                name = filename)
solver.output_params(CFG.output_folder, sample_rate=100)

input_params_willmore = {
    "elasticity_modulus": 1,
    "autoupdate": False,
}
willmore = WillmoreBoundaryAPVPBDF1Model(solver, 1, name = 'willmore_boundary_apvp_bdf1', 
                                     input_params = input_params_willmore)

ale = ALEModel(solver, 3)
ale.set_bnd_displacement(willmore.displacement, 'boundary', redistribute=False)

solver.save_model_solution("willmore_boundary_apvp_bdf1", "displacement")
solver.save_model_solution("willmore_boundary_apvp_bdf1", "mean_curvature")

scene1 = Draw(mesh.deformation, mesh)
area0 = Integrate(1, mesh, VOL_or_BND = BND)
volume0 = Integrate(Id*n, mesh, VOL_or_BND = BND)

try:
    for _ in solver():
        pass
    area1 = Integrate(1, mesh, VOL_or_BND = BND)
    volume1 = Integrate(Id*n, mesh, VOL_or_BND = BND)
    with open(CFG.output_folder + '/' + filename + '/eos_file.txt', "w") as f:
        f.write('Simulation terminated correctly\n')
        f.write('Initial area: ' + str(area0) + ' Final area: ' + str(area1) +'\n')
        f.write('Initial volume: ' + str(volume0) + ' Final volume: ' + str(volume1))
except Exception as e:
    with open(CFG.output_folder + '/' + filename + '/error_file.txt', "w") as f:
        f.write('Simulation terminated with error\n')
        f.write('Time: ' +  str(solver.time.t.Get()) +', iter: '+ str(solver.time.iter) + '\n')
        f.write('Cause: ' + str(e))