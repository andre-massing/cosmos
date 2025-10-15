# %%
import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
from cosmos import *

wp= WorkPlane().RectangleC(4,4).Move(-2) \
    .Circle(0,0,1).Reverse()
face = wp.Face()
face.edges[4].name = 'circle'
face.edges[4].maxh = 0.01

geo = OCCGeometry(face, dim = 2)
mesh = Mesh(geo.GenerateMesh(maxh=0.1))
solvermesh = SolverMesh(mesh)

dt = 1e-3
T = 1
t0 = 0
solvertime = SolverTime(dt=dt, initial_t=t0, final_t=T)

solver = Solver(solvermesh, solvertime, iter=False)

ch_input_params = {
    # "mass_preserving": True,
    # "bounds" : [0, 1],
    "M": 0.001,
    "epsilon": 0.08,
    "sigma": 30,
    "Neu_bnd_phase": 'boundary',
    "Neu_bnd_potential": 'boundary'
}
ch = CahnHilliardVolumeAlandBDF1Model(solver, 1, input_params=ch_input_params)
# np.random.seed(42)
# size = len(ch.phase.vec.data)
# ch.phase.vec.data[:] = np.random.uniform(0, 1, size)
ch.phase.Set(IfPos(sqrt((x-1)**2+(y)**2)-0.5, 0, 1))
aux_vec = ch.phase.vec.Copy()
ch.phase.Set(IfPos(sqrt(x**2+(y-1.25)**2)-0.5, 0, 1))
# ch.phase.vec.data += aux_vec.data

wm_input_params = {
    "autoupdate": True,
}
wm = WillmoreBoundaryInexBDF1Model(solver, 2, input_params=wm_input_params, domain='circle')

ale = ALEModel(solver, 3)

ns = specialcf.normal(2)
Ps = Id(2) - OuterProduct(ns, ns)
Qs = OuterProduct(ns, ns)

sigma1 = 30
sigma0 = 40
def sigma(phi):
    return (sigma1-sigma0)*phi**2*(3-2*phi) + sigma0
def Dsigma(phi):
    return (sigma1-sigma0)*(-6*phi**2+6*phi)

ch.set_input_fields({
    "grad_phase_bnd": lambda: Dsigma(ch.gfu_u_old)/ch_input_params['sigma']/ch_input_params['epsilon']*specialcf.normal(2),
    "b": ale.wind
})

K_B = 0.01
wm.set_input_fields({
    "elasticity_modulus": K_B,
    "rhs": lambda: sigma(ch.phase)*wm.mean_curvature + Dsigma(ch.gfu_u_old)*grad(ch.phase).Trace()
})
ale.set_bnd_displacement(lambda: wm.displacement, 'circle', redistribute=True)

scene = Draw(mesh.deformation, mesh)
scene1 = Draw(ch.phase, mesh)
for _ in solver():
    scene.Redraw()
    scene1.Redraw()