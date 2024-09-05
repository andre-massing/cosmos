# %%
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# # Only needed for testing
# from ngsolve.webgui import Draw
import numpy as np

class PoissonSolver(SteadySolver):

    def __init__(self, mesh=None, fes_order=1, bnd_cond=None, rhs=CF(0.0), verbose = 0):
        
        super().__init__(mesh, bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.rhs = rhs
    
    def __call__(self):

        self.__setup__()

        if self.dir_bnd!=None:
            V = H1(self.mesh, order= self.fes_order, dirichlet = self.dir_bnd)

            fes = V
            u, v = fes.TnT()
        else:
            # In case of fully Neumann boundary conditions
            # a zero-mean solution is enforced through a lagrange multiplier
            V = H1(self.mesh, order= self.fes_order)
            Q = NumberSpace(self.mesh) # Lagrange multiplier for zero mean solution
            fes = V*Q
            (u, lam), (v, mu) = fes.TnT()

        a = BilinearForm(fes)
        a += grad(u)*grad(v)*dx
        if self.dir_bnd == None:
            a += (lam*v-mu*u)*dx
        f = LinearForm(fes)
        f += self.rhs*v*dx

        if self.neu_bnd != None:
            n = specialcf.normal(self.mesh.dim)
            f += self.neu_cf*n*v*ds(definedon = self.mesh.Boundaries(self.neu_bnd))

        # gfu is the grid function used for computations and containing all
        # the variables
        # u_h is a pointer to the function I want to pass as output and on which I want the boundary conditions to be applied
        gfu = GridFunction(fes)
        if self.dir_bnd == None:
            u_h = gfu.components[0]
        else:
            u_h = gfu

        res = f.vec.CreateVector()    
        with TaskManager():
            a.Assemble()
            f.Assemble()

            res = f.vec
            if self.dir_bnd != None: 
                u_h.Set(self.dir_cf, definedon = self.mesh.Boundaries(self.dir_bnd))
                res.data += - a.mat * gfu.vec

                gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * res

            else:

                gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

        yield u_h

if __name__ == "__main__":

    ##### 2D CASE #####

    u_ex = cos(4*pi*x)*cos(6*pi*y)
    grad_u = CF((u_ex.Diff(x), u_ex.Diff(y)))

    rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)

    dh = 0.1
    geo = unit_square

    ## Case with mixed boundary conditions
    bnd_cond=[['neu', 'right|left', grad_u],
              ['dir', 'top|bottom', u_ex]]
    fes_order = 2
    solver = PoissonSolver(fes_order=fes_order, bnd_cond=bnd_cond, rhs=rhs,  verbose = 0)

    conv = Convergence(geom=geo, dh = 0.1, power=2, n_refinements=3)
    order = conv(solver=solver, exact_sol=u_ex)
    print(order)

    ## Case with neumann boundary conditions only
    # The exact solution already has zero mean
    bnd_cond=[['neu', 'right|left|top|bottom', grad_u]]
    solver.bnd_cond = bnd_cond

    conv.n_ref = 2
    order = conv(solver=solver, exact_sol=u_ex)
    print(order)

    ## Case with dirichlet boundary conditions only
    bnd_cond=[['dir', 'right|left|top|bottom', u_ex]]
    solver.bnd_cond = bnd_cond

    conv.n_ref = 3
    conv.power = 1.5
    order = conv(solver=solver, exact_sol=u_ex)
    print(order)

    ##### 3D CASE #####

    u_ex = cos(4*pi*x)*cos(6*pi*y)*cos(5*pi*z)
    grad_u = CF((u_ex.Diff(x), u_ex.Diff(y), u_ex.Diff(z)))

    rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y) - u_ex.Diff(z).Diff(z)

    geo = unit_cube

    ## Case with mixed boundary conditions
    bnd_cond=[['neu', 'right|left|back', grad_u],
              ['dir', 'top|bottom|front', u_ex]]
    fes_order = 1
    solver = PoissonSolver(fes_order=fes_order, bnd_cond=bnd_cond, rhs=rhs,  verbose = 0)

    conv = Convergence(geom=geo, dh = 0.1, power=1.5, n_refinements=3)
    order = conv(solver=solver, exact_sol=u_ex)
    print(order)

    ## Case with neumann boundary conditions only
    # The exact solution already has zero mean
    bnd_cond=[['neu', 'right|left|top|bottom|front|back', grad_u]]
    solver.bnd_cond = bnd_cond

    conv.n_ref = 2
    order = conv(solver=solver, exact_sol=u_ex)
    print(order)

    ## Case with dirichlet boundary conditions only
    bnd_cond=[['dir', 'right|left|top|bottom|front|back', u_ex]]
    solver.bnd_cond = bnd_cond

    conv.n_ref = 2
    conv.power = 2
    order = conv(solver=solver, exact_sol=u_ex)
    print(order)

# %%
