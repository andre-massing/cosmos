# %%
import sys
sys.path.insert(0, '../solvers/')
from solver_base import UnsteadySolver
from ngsolve import *

# Only needed for testing
sys.path.insert(0, '../geometries/')
from generate_meshes import *
from ngsolve.webgui import Draw
import time as time

class HeatSolver(UnsteadySolver):

    fields_num = 1
    dir_bnd = ''
    neu_bnd = ''

    def __init__(self, mesh, fes_order, dt, t = Parameter(0.0), T=1.0, bnd_cond=[["dir", "default", CF(0.0)]], u0 = None, rhs = CF(0.0)):
        
        super().__init__(mesh, dt, t, T, bnd_cond)

        self.fes_order = fes_order
        self.rhs = rhs
        self.u0 = u0

        self.setup()

    def setup(self):

        if self.dir_bnd == '':
            V = H1(self.mesh, order= self.fes_order)
            Q = NumberSpace(self.mesh)
            fes = V*Q

            (u, lam), (v, mu) = fes.TnT()

            self.a = BilinearForm(fes, symmetric = True)
            self.a += grad(u)*grad(v)*dx
            self.a += 1/self.dt*u*v*dx
            self.a += (lam*v-mu*u)*dx

            self.a.Assemble()
            self.a_inv = self.a.mat.Inverse(freedofs = fes.FreeDofs())

            self.m = BilinearForm(fes, symmetric = True)
            self.m += 1/self.dt*u*v*dx
            self.m.Assemble()

            self.f = LinearForm(fes)
            self.f += self.rhs*v*dx
            self.f += self.neu_cf*v*ds(definedon = self.mesh.Boundaries(self.neu_bnd))

            self.gfu = GridFunction(fes)
            self.u_h = self.gfu.components[0]

            self.gfu_old = GridFunction(fes)

            self.u_h.Set(self.u0)

        else: 

            fes = H1(self.mesh, order= self.fes_order, dirichlet = self.dir_bnd)
            u,v = fes.TnT()

            self.a = BilinearForm(fes, symmetric = True)
            self.a += grad(u)*grad(v)*dx
            self.a += 1/self.dt*u*v*dx

            self.a.Assemble()
            self.a_inv = self.a.mat.Inverse(freedofs = fes.FreeDofs())

            self.m = BilinearForm(fes, symmetric = True)
            self.m += 1/self.dt*u*v*dx
            self.m.Assemble()

            self.f = LinearForm(fes)
            self.f += self.rhs*v*dx
            self.f += self.neu_cf*v*ds(definedon = self.mesh.Boundaries(self.neu_bnd))

            self.gfu = GridFunction(fes)

            self.gfu_old = GridFunction(fes)

            self.u_h = self.gfu
            self.u_h.Set(self.u0)

    def __call__(self):

        res = self.f.vec.CreateVector()
        self.uh_t = GridFunction(self.u_h.space,multidim=0)
        self.uh_t.AddMultiDimComponent(self.u_h.vec)

        while self.t.Get()<self.T- 0.5 * self.dt:

            self.update()

            res.data = self.f.vec \
                - self.a.mat * self.gfu.vec \
                + self.m.mat*self.gfu_old.vec
            self.gfu.vec.data += self.a_inv * res

            self.uh_t.AddMultiDimComponent(self.u_h.vec)

            yield self.u_h, self.uh_t


    def update(self):

        aux = self.t.Get() + self.dt
        self.t.Set(aux)

        self.gfu_old.vec.data = self.gfu.vec.data

        self.u_h.Set(self.dir_cf, definedon = self.mesh.Boundaries(self.dir_bnd))

        # self.a.Assemble() # not needed in this case
        self.f.Assemble()
        # self.a_inv.Update() # not needed in this case

        

if __name__ == "__main__":

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    mesh.Curve(3)
    fes_order = 1

    t = Parameter(0.0)

    u_ex = cos(4*x)*cos(6*y)*sin(10*t)

    rhs = u_ex.Diff(t)-u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)
    n = specialcf.normal(mesh.dim)
    grad_u_n = InnerProduct(CF((u_ex.Diff(x), u_ex.Diff(y))), n)
    bnd_cond=[["neu", "top|bottom", grad_u_n],
              ["dir", "right|left", u_ex]]
    t0 = 0
    T = 1.0
    dt = 0.01

    solver1 = HeatSolver(mesh, fes_order, dt, t, T, bnd_cond, u_ex, rhs)

    try:
        while True:
            gfu, gfut = next(solver1())
    except StopIteration:
        print("Simulation has reached final time successfully")
    except Exception as E:
        print("Something went wrong during the simulation. Exception:")
        print(E)
        
# %%
