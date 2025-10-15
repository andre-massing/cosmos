# %%
import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos import *
from cosmos.utils.generate_meshes import generate_volume_circle

mesh = generate_volume_circle(maxh = 0.05)
solvermesh = SolverMesh(mesh)

dt = 1e-4
T = 1
t0 = 0
solvertime = SolverTime(dt=dt, initial_t=t0, final_t=T)

solver = Solver(solvermesh, solvertime, iter=False, printing = True)

sigma_f = 30
sigma1 = 30
sigma0 = 15

wm_input_params = {
    "autoupdate": True,
}
wm = WillmoreBoundaryVPBDF1Model(solver, 1, input_params=wm_input_params)
ale = ALEModel(solver, 2)
ch_input_params = {
    "mass_preserving": True,
    "bounds" : [0, 1],
    "M": 0.001,
    "epsilon": 0.05,
    "sigma": sigma_f*6*sqrt(2),
    "Neu_bnd_phase": 'boundary',
    "Neu_bnd_potential": 'boundary'
}
ch = CahnHilliardVolumeAlandBDF1Model(solver, 1, input_params=ch_input_params)
ch.phase.Set(IfPos(sqrt(x**2+(y-1)**2)-0.5, 0, 1))
# size = len(ch.phase.vec.data)
# np.random.seed(42)
# ch.phase.vec.data[:] = np.random.uniform(0.2, 0.8, size)

ns = specialcf.normal(2)
Ps = Id(2) - OuterProduct(ns, ns)
Qs = OuterProduct(ns, ns)

def sigma(phi):
    return (sigma1-sigma0)*phi**2*(3-2*phi) + sigma0
def Dsigma(phi):
    return (sigma1-sigma0)*(-6*phi**2+6*phi)

ch.set_input_fields({
    "grad_phase_bnd": lambda: Dsigma(ch.gfu_u_old)/ch_input_params['sigma']/ch_input_params['epsilon']*ns,
    "b": ale.wind
})

K_B = 1e-10
wm.set_input_fields({
    "elasticity_modulus": K_B,
    # "rhs": lambda: sigma(ch.phase)*wm.mean_curvature
})
ale.set_bnd_displacement(lambda: Qs*(wm.displacement), 'boundary', redistribute=True)

# scene = Draw(mesh.deformation, mesh)
scene1 = Draw(ch.phase, mesh)
scene2 = Draw(sigma(ch.phase), mesh)
for _ in solver():
    # scene.Redraw()
    scene1.Redraw()
    scene2.Redraw()
# %%
