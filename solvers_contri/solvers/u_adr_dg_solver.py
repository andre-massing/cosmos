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

class uADR_DGSolver(UnsteadySolver):

    fields_num = 1
    dir_bnd = ''
    neu_bnd = ''

    def __init__(self, mesh, fes_order, b, c, d, dt, t = Parameter(0.0), T=1.0, bnd_cond=[["dir", "default", CF(0.0)]], u0 = None, rhs = CF(0.0)):
        
        super().__init__(mesh, dt, t, T, bnd_cond)

        self.fes_order = fes_order
        self.rhs = rhs
        self.u0 = u0
        self.b = b # Coefficient function/ for convection term
        self.c = c # Coefficient function/scalar for reaction term
        self.d = d # Scalar for diffusion term (no matrix allowed for now)

        self.setup()

    def setup(self):

        if self.dir_bnd == '':
            
            pass

        else: 

            fes = L2(self.mesh, order=self.fes_order, dgjumps = True)
            u,v = fes.TnT()

            # trial and test function jump across elements
            jump_u = u-u.Other()
            jump_v = v-v.Other()
            n = specialcf.normal(self.mesh.dim)
            # trial an test function mean gradient across elements
            mean_dudn = 0.5*n * (self.d*grad(u)+self.d*grad(u.Other()))
            mean_dvdn = 0.5*n * (self.d*grad(v)+self.d*grad(v.Other()))

            alpha = 4*self.d  # Paramter for the stabilization of the diffusive term
            h = specialcf.mesh_size
            self.a = BilinearForm(fes)
            
            diffusion = self.d*grad(u)*grad(v) * dx \
                +alpha*self.fes_order**2/h*jump_u*jump_v * dx(skeleton=True) \
                +(-mean_dudn*jump_v-mean_dvdn*jump_u) * dx(skeleton=True) \
                +alpha*self.fes_order**2/h*u*v * ds(skeleton=True) \
                +self.d*(-n*grad(u)*v-n*grad(v)*u)* ds(skeleton=True)


            reaction = self.c * u*v * dx 

            bn_plus = IfPos(self.b*n, self.b*n, 0)
            bn_minus = IfPos(self.b*n, 0, -self.b*n)
            mean_u = 0.5*(u+u.Other())

            S_int = 0.5*Norm(self.b*n) # Parameter corresponding to upwind stabilization
            convection = - u*(self.b*grad(v))*dx + bn_plus*u*v * ds(skeleton=True)\
                + self.b*n*mean_u*jump_v * dx(skeleton=True) \
                +S_int*jump_u*jump_v * dx(skeleton=True)
            
            time_form = 1/self.dt*u*v*dx

            if (self.d==0):
                self.a += time_form + reaction + convection
            elif (Integrate(Norm(self.b), self.mesh)<1e-14):
                self.a +=  time_form + reaction + diffusion
            else:
                self.a +=  time_form + reaction + convection + diffusion

            self.a.Assemble()
            self.a_inv = self.a.mat.Inverse(freedofs = fes.FreeDofs())

            self.m = BilinearForm(fes, symmetric = True)
            self.m += 1/self.dt*u*v*dx
            self.m.Assemble()

            self.f = LinearForm(fes)
            # rhs taking care of boundary conditions
            f_diff = - self.d*grad(v)*n*self.dir_cf * ds(skeleton=True) \
                    + alpha*self.fes_order**2/h*self.dir_cf*v * ds(skeleton=True)
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

        res = self.f.vec.CreateVector()
        self.uh_t = GridFunction(self.u_h.space,multidim=0)
        self.uh_t.AddMultiDimComponent(self.u_h.vec)

        while self.t.Get()<self.T- 0.5 * self.dt:

            self.update()

            res.data = self.f.vec \
                + self.m.mat*self.gfu_old.vec
            self.gfu.vec.data = self.a_inv * res

            self.uh_t.AddMultiDimComponent(self.u_h.vec)

            yield self.u_h, self.uh_t


    def update(self):

        aux = self.t.Get() + self.dt
        self.t.Set(aux)

        self.gfu_old.vec.data = self.gfu.vec.data

        self.a.Assemble() 
        self.a_inv.Update()
        self.f.Assemble()

        

if __name__ == "__main__":

    # pass

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    mesh.Curve(1)
    fes_order = 2

    t = Parameter(0.0)
    u_ex = cos(4*x)*cos(6*y)*sin(10*t)

    b = CF((sin(6*t),x**3))
    d = 2
    c = cos(10*t)
    mean_u = Integrate(u_ex, mesh)
    # u_ex = u_ex-mean_u

    rhs = u_ex.Diff(t) +d*(-u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)) \
        + (u_ex*b[0]).Diff(x) + (b[1]*u_ex).Diff(y) \
        + c*u_ex
    n = specialcf.normal(mesh.dim)
    grad_u_n = InnerProduct(CF((u_ex.Diff(x), u_ex.Diff(y))), n)
    bnd_cond=[['dir', 'right|left|top|bottom', u_ex]]
    t0 = 0
    T = 0.1
    dt = 0.01

    solver1 = uADR_DGSolver(mesh, fes_order, b, c, d, dt, t, T, bnd_cond, u_ex, rhs)

    ERR1 = 0
    try:
        while True:
            gfu, gfut = next(solver1())
            err = Integrate((u_ex-gfu)*(u_ex-gfu), mesh, order = fes_order +2)
            ERR1 += dt*err
    except StopIteration:
        print("Simulation has reached final time successfully")
    except Exception as E:
        print("Something went wrong during the simulation. Exception:")
        print(E)
    ERR1 = sqrt(ERR1)
    print(ERR1)

    mesh.Refine()
    dt = dt*2**(-(fes_order+1)) # scaling so that the time convergence cannot be seen 
    t.Set(0.0)
    solver2 = uADR_DGSolver(mesh, fes_order, b, c, d, dt, t, T, bnd_cond, u_ex, rhs)
    ERR2 = 0
    try:
        while True:
            gfu, gfut = next(solver2())
            err = Integrate((u_ex-gfu)*(u_ex-gfu), mesh, order = fes_order +2)
            ERR2 += dt*err
    except StopIteration:
        print("Simulation has reached final time successfully")
    except Exception as E:
        print("Something went wrong during the simulation. Exception:")
        print(E)
    ERR2 = sqrt(ERR2)
    print(ERR2)

    print(ERR1/ERR2)
        
# %%
