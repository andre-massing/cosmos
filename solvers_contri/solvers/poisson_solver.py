# %%
import sys
sys.path.insert(0, '../solvers/')
from solver_base import SteadySolver
from ngsolve import *

# # Only needed for testing
# sys.path.insert(0, '../geometries/')
# from generate_meshes import *
# from ngsolve.webgui import Draw

class PoissonSolver(SteadySolver):

    fields_num = 1
    dir_bnd = ''
    neu_bnd = ''

    def __init__(self, mesh, fes_order, bnd_cond=[["dir", "default", CF(0.0)]], rhs=CF(0.0)):
        
        super().__init__(mesh, bnd_cond)

        self.fes_order = fes_order
        self.rhs = rhs


    def __call__(self):

        # In the case only Neumann b.c. are applied, the solution with zero mean is
        # computed to give a unique solutioin
        if self.dir_bnd == '':
            V = H1(self.mesh, order= self.fes_order)
            Q = NumberSpace(self.mesh) # Lagrange multiplier for zero mean
            fes = V*Q

            (u, lam), (v, mu) = fes.TnT()

            a = BilinearForm(fes)
            a += grad(u)*grad(v)*dx
            a += (lam*v-mu*u)*dx
            a.Assemble()
            f = LinearForm(fes)
            f += self.rhs*v*dx
            f += self.neu_cf*v*ds(definedon = self.mesh.Boundaries(self.neu_bnd))
            f.Assemble()

            # gfu is the grid function used for computations
            # u_h is the function I want to pass as output
            gfu = GridFunction(fes)
            u_h = gfu.components[0]
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

            return u_h

        else: 
            fes = H1(self.mesh, order= self.fes_order, dirichlet = self.dir_bnd)
            u,v = fes.TnT()

            a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
            f = LinearForm(fes)
            f += self.rhs*v*dx
            f += self.neu_cf*v*ds(definedon = self.mesh.Boundaries(self.neu_bnd))
            f.Assemble()

            gfu = GridFunction(fes)
            res = f.vec.CreateVector()
            gfu.Set(self.dir_cf, definedon = self.mesh.Boundaries(self.dir_bnd))
            res.data = f.vec - a.mat * gfu.vec
            gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * res

            return gfu

if __name__ == "__main__":

    pass

    # mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    # mesh.Curve(3)
    # fes_order = 1

    # u_ex = cos(4*x)*cos(6*y)
    # mean_u = Integrate(u_ex, mesh)
    # u_ex = u_ex-mean_u

    # rhs = -u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)
    # n = specialcf.normal(mesh.dim)
    # grad_u_n = InnerProduct(CF((u_ex.Diff(x), u_ex.Diff(y))), n)
    # bnd_cond=[['dir', 'right|left', u_ex],
    #           ["neu", "top|bottom", grad_u_n]]
    # solver1 = PoissonSolver(mesh, fes_order, bnd_cond, rhs)

    # gfu1 = solver1()
    # err1 = Integrate((gfu1-u_ex)*(gfu1-u_ex) ,mesh, order = solver1.fes_order+2)

    # mesh.Refine()
    # gfu2 = solver1()
    # err2 = Integrate((gfu2-u_ex)*(gfu2-u_ex) ,mesh, order = solver1.fes_order+2)

    # print(err1/err2)

# %%
