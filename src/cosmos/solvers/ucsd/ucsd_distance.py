# %%
from cosmos.utils.generate_synapse_meshes import *
from cosmos.utils.manufactured_solution_tools import convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# # Only needed for testing
from ngsolve.webgui import Draw
import numpy as np

class DistanceSolver(SteadySolver):

    fields_num = 1

    def __init__(self, mesh, fes_order, bnd_cond=[]):
        
        super().__init__(mesh, bnd_cond)

        self.fes_order = fes_order


    def __call__(self):

        # In the case only Neumann b.c. are applied, the solution with zero mean is
        # computed to give a unique solutioin
        if self.dir_bnd == '':
            
            raise Exception('Computing distance needs the imposition of Dirichlet boundary conditions')

        else: 

            fes = H1(self.mesh, order= self.fes_order, dirichlet = self.dir_bnd)
            fesh = H1(self.mesh, order= self.fes_order+1, dirichlet = self.dir_bnd)
            fesV = VectorH1(self.mesh, order = self.fes_order, dirichlet = self.dir_bnd)
            u,v = fes.TnT()
            uh, vh = fesh.TnT()
            alpha = specialcf.mesh_size

            aux = GridFunction(fes)
            aux.Set(alpha)
            self.dt = np.max(aux.vec.data)**2

            gfuh = GridFunction(fesh)
            gfuh.Set(1, definedon = self.dir_bnd)

            a1 = BilinearForm(fesh)
            a1 += (self.dt*grad(uh)*grad(vh) + uh*vh)*dx
            a1.Assemble()

            l1 = LinearForm(fesh)
            r = l1.vec - a1.mat * gfuh.vec

            gfuh.vec.data += a1.mat.Inverse(freedofs=fesh.FreeDofs()) * r

            X = GridFunction(fesV)
            X.Set(-grad(gfuh)/Norm(grad(gfuh)), dual = True)

            a2 = BilinearForm(fes)
            a2 += (grad(u)*grad(v))*dx
            a2.Assemble()
            
            l2 = LinearForm(fes)
            l2 += X*grad(v)*dx
            l2.Assemble()

            gfu = GridFunction(fes)
            gfu.Set(0, definedon = self.dir_bnd)
            r = l2.vec - a2.mat * gfu.vec

            gfu.vec.data += a2.mat.Inverse(freedofs=fes.FreeDofs()) * r

            gfu.vec.data = -gfu.vec.data

            return gfu

if __name__ == "__main__":

    # fes_order = 2

    # dh = 0.01
    # mesh = generate_synapse2d(maxh = dh)
    # bnd_cond=[['dir', 'membrane|membrane_bnd', CF(0.0)]]
    # solver = DistanceSolver(mesh, fes_order, bnd_cond)

    # gfu = solver()

    # Draw(grad(gfu), mesh)

    fes_order = 2

    dh = 0.01
    mesh, geo = generate_synapse2d(maxh=dh)
    # mesh, geo = generate_circle(maxh = dh)
    bnd_cond=[['dir', 'membrane|membrane_bnd', CF(0.0)]]
    solver = DistanceSolver(mesh, fes_order, bnd_cond)

    gfu = solver()

    Draw(gfu)

    vtkout = VTKOutput(mesh,coefs=[gfu, grad(gfu)],names=["dist", 'grad_d'],filename="./examples/distance/synapse")
    vtkout.Do()

    # V = VectorH1(mesh, order = fes_order)
    # d_s = GridFunction(V)
    # d_s.Set(grad(gfu))

    # Draw(d_s)
    # mip = mesh(0.1, 0.0)
    # funct = Trace(grad(d_s))

    # Draw(funct, mesh)

    # print(funct(mip))

# %%
