# %%
from cosmos.utils.generate_synapse_meshes import *
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# Only needed for testing
from ngsolve.webgui import Draw
import time as time
import numpy as np

class s_umADR_CGSolver(UnsteadySolver):

    def __init__(self, mesh=None, fes_order=1, geo_order = 1, b=None, c=None, d=None, dt=0.1, t = Parameter(0.0), T=1.0, bnd_cond=None, u0 = None, rhs = CF(0.0), displ_ex = None, verbose = 0):
        
        super().__init__(mesh=mesh, dt = dt, t = t, T=T, bnd_cond=bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.geo_order = geo_order
        self.rhs = rhs
        self.u0 = u0
        self.b = b
        self.c = c
        self.d = d

        self.displ_ex = displ_ex

    def __build_Ab__(self):

        fes = H1(self.mesh, order=self.fes_order)
        V = VectorH1(self.mesh, order=self.fes_order)
        u,v = fes.TnT()

        self.displ_h = GridFunction(V)

        self.a = BilinearForm(fes)
        
        diffusion = self.d*grad(u).Trace()*grad(v).Trace() * ds(deformation = self.displ_h)
        time_form = 1/self.dt*u*v*ds(deformation = self.displ_h)
        reaction = self.c*u*v*ds(deformation = self.displ_h)
        convection = -self.b*grad(v).Trace() * u *ds(deformation = self.displ_h)

        self.a += time_form + diffusion + reaction + convection

        self.m = BilinearForm(fes, symmetric = True)
        self.m += 1/self.dt*u*v*ds(deformation = self.displ_h)

        with TaskManager():
            self.a.Assemble()
            self.a_inv = self.a.mat.Inverse(freedofs = fes.FreeDofs())

        self.f = LinearForm(fes)

        self.f += self.rhs*v*ds(deformation = self.displ_h)

        self.gfu = GridFunction(fes)

        self.gfu_old = GridFunction(fes)

        self.u_h = self.gfu
        self.u_h.Set(self.u0, definedon=self.mesh.Boundaries(".*"))

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

            self.displ_h.Set(self.displ_ex, definedon=self.mesh.Boundaries(".*"))
            self.m.Assemble()

            self.t.Set(self.t.Get() + self.dt)
            self.displ_h.Set(self.displ_ex, definedon=self.mesh.Boundaries(".*"))

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

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    b = CF((0.2*t, 0.1*t, 0))


    def compute_transformation(A, b):

        phi = A*CF((x,y,z)) + b
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y,z)) - b)
        w_phi = A.Diff(t)*inv_phi + b.Diff(t)

        e1 = A[:,0]
        e2 = A[:,1]
        e3 = A[:,2]

        n_ex = Cross(e2, e3)*inv_phi[0]+Cross(e3, e1)*inv_phi[1] + Cross(e1, e2)*inv_phi[2]
        n_ex = n_ex/Norm(n_ex)

        return phi, inv_phi, w_phi, detJ, n_ex
    
    phi, inv_phi, w_phi, detJ, n_ex = compute_transformation(A, b)
    P_ex = Id(3)-OuterProduct(n_ex, n_ex)

    displ_ex = phi - CF((x,y,z))
    
    d = 1 + x**2
    c = cos(x*y)
    b = P_ex*CF((2,1,0))

    # I want a stationary solution on the moving mesh
    u_ex = cos(x)*sin(y)*cos(t)

    u_ex = exp(-10*((x-0.7)**2+y**2+(z-0.7)**2))

    flux1 = b*u_ex
    flux2 = - d*gradient(u_ex, P_ex)
    rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
    flux = flux1 + flux2
    rhs = (u_ex.Diff(t) + rel_flux + Trace(gradient(flux, P_ex)) + c*u_ex).Compile()

    fes_order = 2
    T = 1
    dt = 0.1
    _, geo = generate_sphere(maxh=0.2, R = 1, order_g = fes_order)

    conv = Convergence(geom=geo, dh = 0.1, power=1.5, n_refinements=2, time_adapt=True, vol_or_bnd='BND')
    
    solver = s_umADR_CGSolver(fes_order=fes_order, b=b, c=c, d=d, dt=dt, t=t, T=T, u0=u_ex, rhs=rhs, displ_ex=displ_ex)

    order = conv(solver=solver, exact_sol=u_ex, vol_or_bnd_err='BND')
    print(order)

# %%
