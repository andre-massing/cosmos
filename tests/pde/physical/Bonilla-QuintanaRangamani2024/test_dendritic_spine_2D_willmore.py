# %%
from ngsolve import *
from ngsolve.webgui import Draw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cosmos import *
from test_dendritic_spine_geom import generate_synapse2d

mesh, _ = generate_synapse2d(maxh=0.02)
mip = mesh(0.3, 0.7)

###################  PARAMETERS  ##################################
# T = 4*60
dt = 1e-5
t = Parameter(0)
T = 1
n = 200
sample_rate = np.maximum(int(abs(T/dt/n)), 1)
dt = Parameter(dt)
# t = Parameter(-4*60)

folderpath = './quintana_2d'

solvermesh = SolverMesh(mesh)
solvertime = SolverTime(dt, t.Get(), T, t_coef=t)
solver = Solver(solvermesh, solvertime, printing=True)

###################  WILLMORE FLOW  ##################################
input_params = {
    'clamped_bnd': 'membrane_bnd',
    'clamped_conormal': CF((0, -1)),
    'autoupdate': True
}
willmore = WillmoreBoundaryBDF1Model(solver, 1, domain = 'membrane', input_params=input_params)

ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig })
ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir_segm, TRIG: ir_trig })
fes_mc = VectorH1(mesh, order = 1, definedon = mesh.Boundaries('membrane'))
gfu_kappa0_vec = GridFunction(fes_mc)
kappa_mc, eta_mc = fes_mc.TnT()
A_mc = BilinearForm(fes_mc, symmetric = True)
A_mc += (kappa_mc*eta_mc)*ds_lumped
A_mc.Assemble()
invA_mc = A_mc.mat.Inverse(freedofs = fes_mc.FreeDofs())
F_mc = LinearForm(fes_mc)
ns = specialcf.normal(2)
Ps = Id(2) - OuterProduct(ns, ns)
F_mc += -InnerProduct(Ps, grad(eta_mc).Trace())*ds
gfBB = GridFunction(H1(mesh, order=1, definedon=mesh.Boundaries('membrane')))
gfBB.Set(1, definedon=mesh.BBoundaries('membrane_bnd'))
F_mc += InnerProduct(input_params["clamped_conormal"], eta_mc)*gfBB*ds_el_lumped
F_mc.Assemble()
gfu_kappa0_vec.vec.data = invA_mc*F_mc.vec

fes_mc_scalar = H1(mesh, order = 1, definedon = mesh.Boundaries('membrane'))
gfu_kappa0_scalar = GridFunction(fes_mc_scalar)
gfu_kappa0_scalar.Set(gfu_kappa0_vec*ns, definedon = mesh.Boundaries('.*'))

gfu_aux = GridFunction(VectorH1(mesh))
gfu_aux.Set(gfu_kappa0_vec, definedon = mesh.Boundaries('.*'))
Draw(gfu_aux)

willmore.set_input_fields({
    'spontaneous_curvature': gfu_kappa0_scalar,
    'elasticity_modulus': 1,
})

###################  ALE  ##################################
ale = ALEModel(solver, 2)
ale.set_bnd_displacement(willmore.displacement, domain = 'membrane', clamped_bnd='membrane_bnd', 
                         redistribute=True)

###################  SIMULATION  ##################################
scene = Draw(mesh.deformation, mesh)
for i, sol in enumerate(solver()):
    scene.Redraw()
# %%

# %%
