# %%

from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_surface_meshes import generate_sphere
import numpy as np
from test_bachini_config import CFG

import os
filename = os.path.splitext(os.path.basename(__file__))[0]

mesh, _ = generate_sphere(R=1, maxh = CFG.maxh)
dt = CFG.dt

Tend = CFG.Tend
solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)

solvermesh = SolverMesh(mesh)
solver = Solver(solvermesh, solvertime, iter=True,
                name = filename)
solver.output_params(CFG.output_folder, sample_rate=1000)

input_params_willmore = {
    "elasticity_modulus": 0.1,
    "autoupdate": True,
}
willmore = WillmoreBoundaryAPBDF1Model(solver, 1, name = 'willmore_boundary_bdf1', 
                                     input_params = input_params_willmore)

input_params_ch = {
    "epsilon": CFG.epsilon,
    "sigma": CFG.sigma,
    "M": CFG.M,
    "fes_order": 2,
    # "bounds": [0, 1], 
    # "mass_preserving": True
}
ch = CahnHilliardBoundaryBachiniBDF1Model(solver, 2, input_params=input_params_ch,
                                          name = 'cahn_hilliard_bachini_bdf1')
import numpy as np
np.random.seed(42)
n = len(ch.phase.vec.data)
epsilon = np.random.uniform(-1, 1, n)
ch.phase.vec.data = 0.9*epsilon

ns = specialcf.normal(2)
Ps = Id(2) - OuterProduct(ns, ns)
V = VectorH1(mesh, definedon = mesh.Boundaries('.*'))
pre_normal = GridFunction(V)
normal = GridFunction(V)

order = mesh.GetCurveOrder()
fes_mc = HDivDivSurface(mesh,order=order-1, definedon = mesh.Boundaries('.*'))
sigma,tau = fes_mc.TnT()
sigma,tau = sigma.Trace(),tau.Trace()

A_mc = BilinearForm(fes_mc, symmetric = True)
A_mc += InnerProduct(sigma,tau)*ds
A_mc.Assemble()
invA_mc = A_mc.mat.Inverse(freedofs = fes_mc.FreeDofs(),inverse="sparsecholesky")

tE = specialcf.tangential(mesh.dim)
ns = specialcf.normal(mesh.dim)
Ps = Id(mesh.dim) - OuterProduct(ns, ns)
nE = Cross(ns, tE)

# Average normal vector
gfF = GridFunction(VectorFacetSurface(mesh,order=order-1))
gfF.Set(ns, dual=True, definedon=mesh.Boundaries(".*"))

F_mc = LinearForm(fes_mc)
F_mc += -1*(InnerProduct(Grad(ns),tau))*ds \
    + -1*((pi/2-acos(Normalize(gfF)*nE))*tau*nE*nE)*ds(element_boundary=True)
F_mc.Assemble()
gfu_mc = GridFunction(fes_mc)
def compute_W():
    A_mc.Assemble()
    invA_mc.Update()
    F_mc.Assemble()
    gfu_mc.vec.data = invA_mc*F_mc.vec

    return gfu_mc

def willmore_rhs():
    var1 = ch.input_params["sigma"]*ch.input_params["epsilon"]/2*Norm(grad(ch.phase).Trace())**2*willmore.gfu_k_old
    var2 = ch.input_params["sigma"]/ch.input_params["epsilon"]*(0.25*(ch.phase**2-1)**2)*willmore.gfu_k_old
    Weingarten = compute_W()
    ns = specialcf.normal(mesh.dim)
    var3 = -1*ch.input_params["sigma"]*ch.input_params["epsilon"]*InnerProduct(grad(ch.phase).Trace(), Weingarten*grad(ch.phase).Trace())*ns
    return var1+var2+var3
willmore.set_input_fields({
    "rhs": willmore_rhs
})

def displ():
    result = willmore.displacement+Ps*ch.potential*grad(ch.phase).Trace()*solver.time.dt
    return result
ale = ALEModel(solver, 3, name = 'ale_model')
ale.set_bnd_displacement(displ, 'boundary', redistribute=True)
ch.set_input_fields({
    "b": ale.wind
})

solver.save_model_solution("ale_model", "displacement")
solver.save_model_solution("willmore_boundary_bdf1", "displacement")
solver.save_model_solution("willmore_boundary_bdf1", "mean_curvature")
solver.save_model_solution('cahn_hilliard_bachini_bdf1', 'phase')
solver.save_model_solution('cahn_hilliard_bachini_bdf1', 'potential')

try:
    for _ in solver():
        pass
except Exception as e:
    with open(CFG.output_folder + '/' + filename + '/error_file.txt', "w") as f:
        f.write('Simulation terminated wit error\n')
        f.write('Time: ' +  str(solver.time.t.Get()) +', iter: '+ str(solver.time.iter) + '\n')
        f.write('Cause: ' + str(e))

