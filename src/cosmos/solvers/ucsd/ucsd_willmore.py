# %%
from cosmos.utils.generate_synapse_meshes import *
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# Only needed for testing
from ngsolve.webgui import Draw
import time as time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class WillmoreSolver(UnsteadySolver):

    def __init__(self, mesh=None, fes_order=1, dt=0.1, t = Parameter(0.0), T=1.0, bnd_cond=None, u0 = None, rhs=None, barbed_ends = CF(0.0), verbose = 0):

        super().__init__(mesh=mesh, dt = dt, t = t, T=T, bnd_cond=bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.rhs = rhs
        self.u0 = u0
        self.barbed_ends = barbed_ends
        
    def __build_Ab__(self):

        V1 = VectorH1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))
        V2 = H1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))

        Vaux = VectorH1(self.mesh, order=self.fes_order)
        self.dummy = GridFunction(Vaux)

        self.displ_h = GridFunction(V1)
        self.displ_h_old = GridFunction(V1)

        if self.mesh.dim == 3:
            Ident = CF((x,y,z))
        elif self.mesh.dim == 2:
            Ident = CF((x,y))

        self.X_m = GridFunction(V1)
        self.X_m.Set(Ident, definedon=self.mesh.Boundaries(".*"))

        self.X0 = GridFunction(V1)
        self.X0.Set(Ident, definedon=self.mesh.Boundaries(".*"))

        self.u_h = GridFunction(V1)

        self.nu = specialcf.normal(self.mesh.dim)
        self.Ps = Id(self.mesh.dim) - OuterProduct(self.nu, self.nu)

        def D_s(chi):
            sym = 0.5*(Grad(chi).Trace()+Grad(chi).Trace().trans)
            return sym
        
        X = V1*V1
        (dX, kappa), (chi, eta) = X.TnT()

        self.dXYk_h = GridFunction(X)
        self.dX_h, self.kappa_h = self.dXYk_h.components
        
        (kappa0, eta0) = V1.TnT()
        M0 = BilinearForm(V1)
        l0 = LinearForm(V1)
        M0 += kappa0*eta0*ds
        M0.Assemble()
        l0 += -InnerProduct(self.Ps, Grad(eta0).Trace())*ds
        l0.Assemble()

        self.kappa_h.vec.data = M0.mat.Inverse(V1.FreeDofs())*l0.vec

        # wein = self.Lifting()
        # self.kappa_h.Set(Trace(wein)*self.nu, definedon = self.mesh.Boundaries(".*"))

        self.dt_var = Parameter(self.dt)

        self.M = BilinearForm(X)
        self.l = LinearForm(X)

        self.M += InnerProduct(dX, chi)/self.dt_var*ds(deformation = self.displ_h)
        self.M += -InnerProduct(Grad(kappa).Trace(), Grad(chi).Trace())*ds(deformation = self.displ_h)

        self.l += InnerProduct(Trace(Grad(self.kappa_h).Trace()),Trace(Grad(chi).Trace()))*ds(deformation = self.displ_h)
        self.l += -2*InnerProduct(Grad(self.kappa_h).Trace().trans, D_s(chi)*Grad(self.X_m).Trace().trans)*ds(deformation = self.displ_h)
        self.l += 0.5*InnerProduct((Norm(self.kappa_h)**2)*Grad(self.X_m).Trace(),Grad(chi).Trace())*ds(deformation = self.displ_h)

        self.M += (kappa*eta+InnerProduct(Grad(dX).Trace(), Grad(eta).Trace()))*ds(deformation = self.displ_h)

        self.l += -InnerProduct(Grad(self.X_m).Trace(), Grad(eta).Trace())*ds(deformation = self.displ_h)

        # X = V1*V1*V1
        # (dX, Y, kappa), (chi, zeta, eta) = X.TnT()
        # self.M = BilinearForm(X)
        # self.l = LinearForm(X)

        # self.dXYk_h = GridFunction(X)
        # self.dX_h, self.Y_h, self.kappa_h = self.dXYk_h.components
        
        # (kappa0, eta0) = V1.TnT()
        # M0 = BilinearForm(V1)
        # l0 = LinearForm(V1)
        # M0 += kappa0*eta0*ds
        # M0.Assemble()
        # l0 += -InnerProduct(Grad(self.X0).Trace(), Grad(eta0).Trace())*ds
        # l0.Assemble()

        # self.kappa_h.vec.data = M0.mat.Inverse(V1.FreeDofs())*l0.vec
        # self.Y_h.vec.data = self.kappa_h.vec.data

        # self.M += InnerProduct(dX, chi)/self.dt_var*ds(deformation = self.displ_h)
        # self.M += -InnerProduct(Grad(Y).Trace(), Grad(chi).Trace())*ds(deformation = self.displ_h)

        # self.l += InnerProduct(Trace(Grad(self.Y_h).Trace()),Trace(Grad(chi).Trace()))*ds(deformation = self.displ_h)
        # self.l += -2*InnerProduct(Grad(self.Y_h).Trace(), D_s(chi))*ds(deformation = self.displ_h)
        # self.l += -0.5*InnerProduct((Norm(self.kappa_h)**2),Trace(Grad(chi).Trace()))*ds(deformation = self.displ_h)
        # self.l += InnerProduct(InnerProduct(self.kappa_h,self.Y_h),Trace(Grad(chi).Trace()))*ds(deformation = self.displ_h)

        # self.M += (kappa*zeta - Y*zeta)*ds(deformation = self.displ_h)

        # self.M += (kappa*eta+InnerProduct(Grad(dX).Trace(), Grad(eta).Trace()))*ds(deformation = self.displ_h)

        # self.l += -InnerProduct(Grad(self.X_m).Trace(), Grad(eta).Trace())*ds(deformation = self.displ_h)

        # EXTRA COMPONENT DUE TO SURFACE REACTANT
        beta = 10*self.barbed_ends
        self.l += beta*self.nu*chi*ds(deformation = self.displ_h)

        self.M.Assemble()
        self.Minv = self.M.mat.Inverse(X.FreeDofs(), inverse="umfpack")


        ## DUAN LI

        self.invJ, self.detJ = self._compute_jacobian()

        W = V1*V2
        (u2, p),(v2, q) = W.TnT()

        self.M2 = BilinearForm(W)
        self.M2 += (InnerProduct(Grad(u2).Trace(), Grad(v2).Trace()) \
                                - p*self.nu*v2\
                                    - q*self.nu*u2)*ds
        self.M2.Assemble()
        self.M2inv = self.M2.mat.Inverse(W.FreeDofs())

        self.l2 = LinearForm(W)
        self.l2 += (-InnerProduct(Grad(self.X_m).Trace(), \
                                            Grad(v2).Trace()))*ds
        
        # self.M2 = BilinearForm(W)
        # self.M2 += (InnerProduct(Grad(u2).Trace()*self.invJ, Grad(v2).Trace()*sqrt(self.detJ)) \
        #                 - p*self.nu*v2*sqrt(self.detJ)\
        #                     - q*self.nu*u2*sqrt(self.detJ))*ds(deformation = self.displ_h)
        # self.M2.Assemble()
        # self.M2inv = self.M2.mat.Inverse(W.FreeDofs())

        # self.l2 = LinearForm(W)
        # self.l2 += (-InnerProduct(Grad(self.X_m).Trace()*self.invJ, \
        #         Grad(v2).Trace()*sqrt(self.detJ)))*ds(deformation = self.displ_h)
        
        self.dXkappa_h = GridFunction(W)
        
    def __call__(self):

        self.__setup__()

        self.__build_Ab__()

        while self.t.Get()<self.T- 0.5 * self.dt_var.Get():

            self.displ_h_old.vec.data = self.displ_h.vec.data

            self.__update1__()

            self.dXYk_h.vec.data = self.Minv*self.l.vec
            self.displ_h.vec.data += self.dX_h.vec
            self.X_m.vec.data += self.dX_h.vec

            self.__update2__()

            self.dXkappa_h.vec.data = self.M2inv*self.l2.vec
            self.displ_h.vec.data += self.dXkappa_h.components[0].vec
            self.X_m.vec.data += self.dXkappa_h.components[0].vec

            self.dummy.Set(self.dX_h+self.dXkappa_h.components[0], definedon = self.mesh.Boundaries('.*'))

            yield self.dummy
            
    def __update1__(self):

        with TaskManager():

            self.dt_var.Set(self.dt)

            self.M.Assemble() 
            self.Minv.Update()

            self.l.Assemble()

    def __update2__(self):

        with TaskManager():

            self.invJ, self.detJ = self._compute_jacobian()

            self.M2.Assemble() 
            self.M2inv.Update()

            self.l2.Assemble()

            self.t.Set(self.t.Get() + self.dt_var.Get())
    
    def _compute_jacobian(self):

        J = Grad(self.X0).Trace()*Grad(self.X0).Trace().trans+OuterProduct(self.nu, self.nu)
        invJ = Inv(J)
        detJ = Det(J)

        return invJ, detJ

    def Lifting(self):

        n = specialcf.normal(self.mesh.dim)
        t = specialcf.tangential(self.mesh.dim)
        mu = Cross(n,t)
        
        # Average normal vector
        gfF = GridFunction(VectorFacetSurface(self.mesh, order=self.fes_order-1))
        gfF.Set(n, definedon=self.mesh.Boundaries(".*"), dual = True)
        
        fes = HDivDivSurface(self.mesh,order=self.fes_order-1)
        sigma,tau = fes.TnT()
        sigma,tau = sigma.Trace(),tau.Trace()
        
        a = BilinearForm(fes, symmetric=True)
        a += InnerProduct(sigma,tau)*ds
        
        wein = specialcf.Weingarten(self.mesh.dim)
        f = LinearForm(fes)
        f += -InnerProduct(wein,tau)*ds \
                - (pi/2-acos(Normalize(gfF)*mu))*tau*mu*mu*ds(element_boundary=True)
        
        gflift = GridFunction(fes)
        
        with TaskManager():
            a.Assemble()
            f.Assemble()
            gflift.vec.data = a.mat.Inverse(fes.FreeDofs(),inverse="sparsecholesky")*f.vec
            
        return gflift


if __name__ == "__main__":

    ## CONVERGENCE 1

    # t = Parameter(0.0)

    # T = 0.2
    # dt = 0.005
    # fes_order = 1

    # # _, geo = generate_sphere(maxh=0.1, R = 1, order_g = fes_order)
    # _, geo = generate_torus(maxh=0.5, R = sqrt(2), r=1.0, order_g = fes_order)

    # conv = Convergence(geom=geo, dh = 0.2, power=1.5, n_refinements=2, time_adapt=True, vol_or_bnd='BND')

    # solver = WillmoreSolver(fes_order=fes_order, dt=dt, t=t, T=T)

    # order = conv(solver=solver, exact_sol=CF((0,0,0)), vol_or_bnd_err='BND')
    # print(order)

    # EXAMPLE

    t = Parameter(0.0)

    T = 1.0
    dt = 0.001
    fes_order = 1

    # mesh, geo = generate_box(maxh=0.1, order_g=fes_order, a=1, b=5)
    mesh, geo = generate_cube_g5(maxh=0.15, order_g = fes_order, R=3.0)
    # mesh, geo = generate_torus(maxh=0.2, R = sqrt(2), r=1.0, order_g = fes_order)
    # mesh, geo = generate_n_torus(maxh=0.1, order_g = fes_order, char_len=1, n=3)
    # mesh, geo = generate_sphere(maxh=0.05, R = 0.5, order_g = fes_order)
    # mesh, geo = generate_ball(maxh=0.1, R = 1, order_g = fes_order)
    # mesh, geo = generate_circle(maxh=0.1, R = 1, order_g = fes_order)

    solver = WillmoreSolver(mesh=mesh, fes_order=fes_order, dt=dt, t=t, T=T)

    i=0
    out_int = int(((T-0.0)/dt)//20)

    for sol in solver():

        # Draw(CF((x,y,z)), solver.mesh, deformation=solver.displ_h)
        
        if i%(out_int+1)==0:
            clipping = { "function" : True,  "pnt" : (0,0,0), "vec" : (0,1,0) }
            Draw(solver.displ_h, solver.mesh, deformation=solver.displ_h)
        i+=1

# %%
