# %%
from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from netgen.occ import *
import numpy as np
import logging
from test_helfrich_spine_config import CFG

import os
filename = os.path.splitext(os.path.basename(__file__))[0]

Id = CF((x, 0, 0))
n = specialcf.normal(3)

mesh = Mesh('./spine_fine.vol')
# mesh = Mesh('./spine_coarse.vol')

logging.getLogger().setLevel(logging.DEBUG)

dt = CFG.dt
Tend = CFG.Tend
solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)

solvermesh = SolverMesh(mesh)
solver = Solver(solvermesh, solvertime, iter=False,
                name = filename)
solver.output_params(CFG.output_folder, sample_rate=50)

input_params_willmore = {
    "autoupdate": True,
}
willmore = WillmoreBoundaryInexBDF1Model(solver, 1, name = 'willmore_boundary_inex_bdf1', 
                                     input_params = input_params_willmore)

ale = ALEModel(solver, 3)
ale.set_bnd_displacement(willmore.displacement, 'default', redistribute=False)

solver.save_model_solution("willmore_boundary_inex_bdf1", "displacement")
solver.save_model_solution("willmore_boundary_inex_bdf1", "mean_curvature")

scene1 = Draw(mesh.deformation, mesh)
area0 = Integrate(1, mesh, VOL_or_BND = BND)
volume0 = Integrate(Id*n, mesh, VOL_or_BND = BND)

def timestep_f(t):
        return 1e-3 - (1e-3 - dt)*exp(-50*t) 

print(timestep_f(0.05))

scene = Draw(mesh.deformation, mesh)
try:
    with open(CFG.output_folder + filename + '/simulation.txt', "w") as f:
        f.write('Time\tArea\tVolume\tEnergy\n')
        for _ in solver():
            solver.time.input_params["dt"] = timestep_f(solver.time.t.Get())
            scene.Redraw()
            area = Integrate(1, mesh, VOL_or_BND = BND)
            volume = Integrate(Id*n, mesh, VOL_or_BND = BND)
            energy = Integrate(InnerProduct(willmore.mean_curvature, willmore.mean_curvature), mesh, VOL_or_BND = BND)
            f.write(str(solver.time.t.Get()) + '\t' + str(area) + '\t' + str(volume) + '\t' + str(energy) +'\n')
except Exception as e:
    with open(CFG.output_folder + filename + '/error_file.txt', "w") as f:
        f.write('Simulation terminated with error\n')
        f.write('Time: ' +  str(solver.time.t.Get()) +', iter: '+ str(solver.time.iter) + '\n')
        f.write('Cause: ' + str(e))