# %%
import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos import *


maxh = 0.05
from ngsolve.webgui import Draw
from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle((0, -1),(1,1),bc="membrane")
geo.SetDomainMaxH(1, maxh)
ngmesh = geo.GenerateMesh()
ngmesh.SetBCName(0, 'bnd')
ngmesh.SetBCName(1, 'bnd')
ngmesh.SetBCName(2, 'bnd')
ngmesh.SetCD2Name(1, 'point')
ngmesh.SetCD2Name(2, 'point')
ngmesh.SetCD2Name(3, 'point')
ngmesh.SetCD2Name(4, 'point')
mesh = Mesh(ngmesh)

bdry_values = {'membrane': 1}
cf = mesh.BoundaryCF(bdry_values, default=0)

g = GridFunction(H1(mesh), name='bdry')
g.Set(cf, definedon=mesh.Boundaries('.*'))
Draw(g);

solvermesh = SolverMesh(mesh)

dt = 1e-3
T = 10
t0 = 0
solvertime = SolverTime(dt=dt, initial_t=t0, final_t=T)

solver = Solver(solvermesh, solvertime, iter=False)

ch_input_params = {
    "mass_preserving": True,
    "bounds" : [0, 1],
    "M": 0.001,
    "epsilon": 0.1,
    "sigma": 30,
    "Neu_bnd_phase": '.*',
    "Neu_bnd_potential": '.*'
}
ch = CahnHilliardVolumeAlandBDF1Model(solver, 1, input_params=ch_input_params)
ch.phase.Set(IfPos(sqrt(x**2+y**2)-0.5, 0, 1))

wm_input_params = {
    "autoupdate": True,
    "clamped_bnd": 'point',
    "clamped_conormal": Normalize(CF((0, y))),
}
wm = WillmoreBoundaryInexBDF1Model(solver, 2, input_params=wm_input_params, domain = 'membrane')

ale = ALEModel(solver, 3)

ns = specialcf.normal(2)
Ps = Id(2) - OuterProduct(ns, ns)
Qs = OuterProduct(ns, ns)

sigma1 = 30
sigma0 = 15
def sigma(phi):
    return (sigma1-sigma0)*phi**2*(3-2*phi) + sigma0
def Dsigma(phi):
    return (sigma1-sigma0)*(-6*phi**2+6*phi)

ch.set_input_fields({
    "grad_phase_bnd": lambda: Dsigma(ch.gfu_u_old)/ch_input_params['sigma']/ch_input_params['epsilon']*specialcf.normal(2),
    "b": ale.wind
})
wm.set_input_fields({
    "elasticity_modulus": 1,
    "rhs": lambda: sigma(ch.phase)*wm.gfu_k_old+ Dsigma(ch.phase)*grad(ch.phase).Trace()
})
ale.set_bnd_displacement(lambda: wm.displacement, 'membrane', redistribute = False, clamped_bnd='point')

scene = Draw(ch.phase, mesh)
scene1 = Draw(mesh.deformation, mesh)
for _ in solver():
    scene.Redraw()
    scene1.Redraw()