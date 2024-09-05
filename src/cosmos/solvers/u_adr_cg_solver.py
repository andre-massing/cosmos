# %%
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# Only needed for testing
from ngsolve.webgui import Draw
import time as time

class uADR_CGSolver(UnsteadySolver):

    def __init__(self, mesh=None, fes_order=1, b=None, c=None, d=None, dt=0.1, t = Parameter(0.0), T=1.0, bnd_cond=None, u0 = None, rhs = CF(0.0), verbose = 0):
        
        super().__init__(mesh=mesh, dt = dt, t = t, T=T, bnd_cond=bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.rhs = rhs
        self.u0 = u0
        self.b = b # Coefficient function/ for convection term
        self.c = c # Coefficient function/scalar for reaction term
        self.d = d # Scalar for diffusion term (no matrix allowed for now)

    def __build_Ab__(self):

        if self.neu_bnd != None:
            
            raise Exception("Neumann boundary conditions are not implemented yet")

        else: 

            fes = H1(self.mesh, order=self.fes_order, dgjumps = True)
            u,v = fes.TnT()

            h = specialcf.mesh_size
            n = specialcf.normal(self.mesh.dim)
            bn_minus = 0.5*(Norm(self.b*n)-self.b*n)
            bn_plus = 0.5*(Norm(self.b*n)+self.b*n)

            # trial and test function gradeint jump across elements
            jump_u = n*(grad(u) - (grad(u)).Other())
            jump_v = n*(grad(v) - (grad(v)).Other())

            self.a = BilinearForm(fes)

            penalty = self.d*(self.fes_order+1)*(self.fes_order+self.mesh.dim)/self.mesh.dim/h
            diffusion = self.d*grad(u)*grad(v)*dx - self.d*grad(u).Trace()*n*v*ds(skeleton=True) \
                - self.d*grad(v).Trace()*n*u*ds(skeleton=True) + penalty*u*v*ds(skeleton=True)
            
            reaction = self.c * u*v * dx

            # CIP stabilization
            S_int = 0.5*Norm(self.b*n) # Parameter corresponding to upwind stabilization
            convection = -self.b*grad(v) * u *dx \
                + bn_plus*u*v*ds(skeleton=True) + \
                h**2*S_int*jump_u*jump_v*dx(skeleton=True)
            
            time_form = 1/self.dt*u*v*dx

            if (self.d==0):
                self.a += time_form + reaction + convection
            elif (Integrate(Norm(self.b), self.mesh)<1e-14):
                self.a +=  time_form + reaction + diffusion
            else:
                self.a +=  time_form + reaction + convection + diffusion


            self.m = BilinearForm(fes, symmetric = True)
            self.m += 1/self.dt*u*v*dx

            with TaskManager():
                self.a.Assemble()
                self.a_inv = self.a.mat.Inverse(freedofs = fes.FreeDofs())
                self.m.Assemble()
                

            self.f = LinearForm(fes)
            # rhs taking care of boundary conditions
            f_diff = penalty*self.dir_cf*v*ds(skeleton=True) - self.d*grad(v).Trace()*n*self.dir_cf*ds(skeleton=True)
            f_conv = bn_minus*self.dir_cf*v * ds(skeleton=True)

            if (self.d==0):
                self.f += self.rhs*v * dx + f_conv
            elif (Integrate(Norm(self.b), self.mesh)<1e-14):
                self.f += self.rhs*v * dx + f_diff 
            else:
                self.f += self.rhs*v * dx + f_diff + f_conv

            self.gfu = GridFunction(fes)

            self.gfu_old = GridFunction(fes)

            self.u_h = self.gfu
            self.u_h.Set(self.u0)

    def __call__(self):

        self.__setup__()

        self.__build_Ab__()

        res = self.f.vec.CreateVector()

        while self.t.Get()<self.T- 0.5 * self.dt:

            self.__update__()

            res.data = self.f.vec \
                - self.a.mat * self.gfu.vec \
                + self.m.mat*self.gfu_old.vec
            self.gfu.vec.data += self.a_inv * res

            yield self.u_h


    def __update__(self):

        aux = self.t.Get() + self.dt
        self.t.Set(aux)

        with TaskManager():

            self.gfu_old.vec.data = self.gfu.vec.data

            self.u_h.Set(self.dir_cf, definedon = self.mesh.Boundaries(self.dir_bnd))

            self.a.Assemble()
            self.a_inv.Update()
            self.f.Assemble()

        

if __name__ == "__main__":

    t = Parameter(0.0)

    u_ex = cos(4*x)*cos(6*y)*sin(10*t)

    b = CF((sin(6*t),x**3))
    d = 1 + x**2
    c = cos(10*t)

    rhs = u_ex.Diff(t) +(-(d*u_ex.Diff(x)).Diff(x) - (d*u_ex.Diff(y)).Diff(y)) \
        + (u_ex*b[0]).Diff(x) + (b[1]*u_ex).Diff(y) \
        + c*u_ex
    
    fes_order = 2
    t0 = 0
    T = 1.0
    dt = 0.1

    geo = unit_square
    bnd_cond=[['dir', 'right|left|top|bottom', u_ex]]

    solver = uADR_CGSolver(fes_order=fes_order, b=b, c=c, d=d, dt=dt, t=t, T=T, bnd_cond=bnd_cond, u0=u_ex, rhs=rhs)

    conv = Convergence(geom=geo, dh = 0.1, power=1.5, n_refinements=3, time_adapt=True)

    order = conv(solver=solver, exact_sol=u_ex)
    print(order)
        
# %%
