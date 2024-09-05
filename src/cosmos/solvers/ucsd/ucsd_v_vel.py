# %%
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.utils.generate_synapse_meshes import *
from cosmos.solvers.solver_base import *
from ngsolve import *

# # Only needed for testing
# from ngsolve.webgui import Draw
import numpy as np
from ngsolve.webgui import Draw

class VelocitySolver(SteadySolver):

    def __init__(self, mesh=None, fes_order=1, bnd_cond=None, verbose = 0, displ = None):
        
        super().__init__(mesh, bnd_cond, verbose=verbose, vectorial=True)

        self.fes_order = fes_order
    
        self.displ = displ
    
    def __call__(self):

        self.__setup__()

        if self.displ == None:
            self.displ = CF((0,)*self.mesh.dim)

        if self.dir_bnd==None:
            raise Exception('Computing distance needs the imposition of Dirichlet boundary conditions')
        
        fes = H1(self.mesh, order= self.fes_order, dirichlet = self.dir_bnd)
        fesh = H1(self.mesh, order= self.fes_order+1, dirichlet = self.dir_bnd)
        fesV = VectorH1(self.mesh, order = self.fes_order)
        u,v = fes.TnT()
        uh, vh = fesh.TnT()
        alpha = specialcf.mesh_size

        aux = GridFunction(fes)
        aux.Set(alpha)
        self.dt = np.max(aux.vec.data)**2

        gfuh = GridFunction(fesh)
        gfuh.Set(1, definedon = self.dir_bnd)

        displ_h = GridFunction(fesV)
        displ_h.Set(self.displ)

        a1 = BilinearForm(fesh)
        a1 += (self.dt*grad(uh)*grad(vh) + uh*vh)*dx(deformation = displ_h)
        a1.Assemble()

        l1 = LinearForm(fesh)
        r = l1.vec - a1.mat * gfuh.vec

        gfuh.vec.data += a1.mat.Inverse(freedofs=fesh.FreeDofs()) * r

        X = GridFunction(fesV)
        X.Set(grad(gfuh)/Norm(grad(gfuh)))

        a2 = BilinearForm(fes)
        a2 += (grad(u)*grad(v))*dx(deformation = displ_h)
        a2.Assemble()
        
        l2 = LinearForm(fes)
        l2 += X*grad(v)*dx(deformation = displ_h)
        l2.Assemble()

        gfu = GridFunction(fes)
        gfu.Set(0, definedon = self.dir_bnd)
        r = l2.vec - a2.mat * gfu.vec

        gfu.vec.data += a2.mat.Inverse(freedofs=fes.FreeDofs()) * r

        gfu.vec.data = -gfu.vec.data

        yield X

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
    solver = VelocitySolver(mesh, fes_order, bnd_cond)

    gfu = solver()

    gfuv = next(gfu)

    Draw(gfuv, mesh, vectors = True)

# %%
