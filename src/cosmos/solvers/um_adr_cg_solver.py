# %%
from cosmos.utils.generate_synapse_meshes import *
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# Only needed for testing
from ngsolve.webgui import Draw
import time as time
import numpy as np

class umADR_CGSolver(UnsteadySolver):

    def __init__(self, mesh=None, fes_order=1, geo_order = 1, b=None, c=None, d=None, dt=0.1, t = Parameter(0.0), T=1.0, bnd_cond=None, u0 = None, rhs = CF(0.0), displ_ex = None, verbose = 0):
        
        super().__init__(mesh=mesh, dt = dt, t = t, T=T, bnd_cond=bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.geo_order = geo_order
        self.rhs = rhs
        self.u0 = u0
        self.b = b # Coefficient function/ for convection term
        self.c = c # Coefficient function/scalar for reaction term
        self.d = d # Scalar for diffusion term (no matrix allowed for now)

        self.displ_ex = displ_ex

    def __build_Ab__(self):

        if self.neu_bnd == '' or self.dir_bnd == '':
            
            raise Exception('Only conditions on the total flux are allowed. Use ''dir'' boundry conditions for the convection term and ''neu'' for the diffusion term')

        fes = H1(self.mesh, order=self.fes_order, dgjumps = True)
        V = VectorH1(self.mesh, order=self.fes_order)
        u,v = fes.TnT()

        self.displ_h = GridFunction(V)

        self.a = BilinearForm(fes)

        n = specialcf.normal(self.mesh.dim)
        h = specialcf.mesh_size
        
        diffusion = self.d*grad(u)*grad(v) * dx(deformation = self.displ_h)
        reaction = self.c * u*v * dx(deformation = self.displ_h) 
        time_form = 1/self.dt*u*v*dx(deformation = self.displ_h)

        # CIP stabilization
        S_int = 0.5*Norm(self.b*n) # Parameter corresponding to upwind stabilization
        jump_u = n*(grad(u) - (grad(u)).Other())
        jump_v = n*(grad(v) - (grad(v)).Other())
        bn_plus = 0.5*(Norm(self.b*n)+self.b*n)
        convection = -self.b*grad(v) * u *dx(deformation = self.displ_h) \
            + bn_plus*u*v*ds(skeleton=True, deformation = self.displ_h) + \
            h**2*S_int*jump_u*jump_v*dx(skeleton=True, deformation = self.displ_h)

        if (self.d==0):
            self.a += time_form + reaction + convection
        elif (Integrate(Norm(self.b), self.mesh)<1e-12):
            self.a += time_form + reaction + diffusion
        else:
            self.a += time_form + reaction + convection + diffusion


        self.m = BilinearForm(fes, symmetric = True)
        self.m += 1/self.dt*u*v*dx(deformation = self.displ_h)

        with TaskManager():
            self.a.Assemble()
            self.a_inv = self.a.mat.Inverse(freedofs = fes.FreeDofs())
            self.m.Assemble()

        self.f = LinearForm(fes)

        self.f += self.rhs*v*dx(deformation = self.displ_h)
        # Flux b.c.
        self.f += -self.neu_cf*n*v*ds(definedon = self.mesh.Boundaries(self.neu_bnd), deformation = self.displ_h)
        self.f += -IfPos(self.b*n, 0, 1)*self.dir_cf*n*v*ds(definedon = self.mesh.Boundaries(self.dir_bnd), deformation = self.displ_h)

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
                + self.m.mat*self.gfu_old.vec
            self.gfu.vec.data = self.a_inv * res

            yield self.u_h


    def __update__(self):

        self.gfu_old.vec.data = self.gfu.vec.data

        with TaskManager():

            self.displ_h.Set(self.displ_ex)
            self.m.Assemble()

            self.t.Set(self.t.Get() + self.dt)
            self.displ_h.Set(self.displ_ex)

            self.a.Assemble() 
            self.a_inv.Update()

            self.f.Assemble()

            

def gradient(f,P):

    m, _ = P.dims
    l = len(f.dims)

    if l == 0:
        # scalar gradient is deifed traditionally

        if m == 2:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y)))
        elif m == 3:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y), f.Diff(z))) 
            
    elif l == 1:
        # vector gradient is defined component by component
        # and disposed along columns

        if m == 2:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            output = CoefficientFunction((aux1[0], aux2[0], \
                aux1[1], aux2[1]), dims = (m,m))
        elif m == 3:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            aux3 = gradient(f[2], P)
            output = CoefficientFunction((aux1[0], aux2[0], aux3[0], \
                aux1[1], aux2[1], aux3[1],\
                    aux1[2], aux2[2], aux3[2]), dims = (m,m))
            
    else:

        raise RuntimeError("Don't know how to take the gradient. Only scalars and vectors are accepted.")
    
    return output

        

if __name__ == "__main__":

    t = Parameter(0.0)

    # General affine transformation
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t),\
                sin(t), cos(t)), dims = (2,2))
    b = CF((0.2*t, 0.1*t))

    def compute_transformation(A, b):

        phi = A*CF((x,y)) + b
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y)) - b)
        w_phi = A.Diff(t)*inv_phi + b.Diff(t)

        return phi, inv_phi, w_phi
    
    phi, inv_phi, w_phi = compute_transformation(A, b)

    displ_ex = phi - CF((x,y))

    b = CF((sin(t), exp(x)))
    d = 1+x**2
    c = cos(x*y)

    # I want a stationary solution on the moving mesh
    u_ex = cos(x)*sin(y)*cos(t)

    # u_ex = exp(-20*((x-0.1)**2+y**2))
    flux1 = (b*u_ex).Compile()
    flux2 = (- d*CF((u_ex.Diff(x), u_ex.Diff(y)))).Compile()
    rel_flux = (w_phi*u_ex).Compile()
    flux = flux1 + flux2 + rel_flux
    rhs = (u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c*u_ex).Compile()

    fes_order = 1
    t0 = 0
    T = 1
    dt = 0.1

    _, geo = generate_circle(maxh=0.2, order_g = fes_order)
    bnd_cond=[['neu', 'membrane', flux2],
              ['dir', 'membrane', flux1]]
    
    conv = Convergence(geom=geo, dh = 0.1, power=1.5, n_refinements=3, time_adapt=True)
    
    solver = umADR_CGSolver(fes_order=fes_order, b=b, c=c, d=d, dt=dt, t=t, T=T, bnd_cond=bnd_cond, u0=u_ex, rhs=rhs, displ_ex=displ_ex)

    order = conv(solver=solver, exact_sol=u_ex)
    print(order)

# %%
