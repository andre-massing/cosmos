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
        self.dX_h = GridFunction(V1)

        self.u_h = GridFunction(V1)

        self.nu = specialcf.normal(self.mesh.dim)
        self.Ps = Id(self.mesh.dim) - OuterProduct(self.nu, self.nu)

        ## BALAZS

        X = V2*V2*V1*V1
        (H, Vi, nu, zeta), (phi_H, phi_Vi, phi_nu, phi_zeta) = X.TnT()

        self.HVnz_h = GridFunction(X)
        self.H_h, self.Vi_h, self.nu_h, self.zeta_h = self.HVnz_h.components

        self.nu_h.Set(self.nu,  definedon=self.mesh.Boundaries(".*"))
        self.H_h.Set(self.MeanC(), definedon=self.mesh.Boundaries(".*"))
        self.A_h = 0.5*(Grad(self.nu_h).Trace() + Grad(self.nu_h).Trace().trans)
        self.Q_h = -0.5*self.H_h**3 + Norm(self.A_h)**2*self.H_h

        self.M = BilinearForm(X)
        self.l = LinearForm(X)

        """
        Part for H
        """
        self.M += (H*phi_H)*ds(deformation = self.displ_h)
        self.M += -dt*InnerProduct(grad(Vi).Trace(), grad(phi_H).Trace())*ds(deformation = self.displ_h)

        #! Try substitute Weingarten with appropriate one
        self.l += -dt*InnerProduct(Norm(self.A_h)**2*self.Vi_h, phi_H)*ds(deformation = self.displ_h)
        self.l += (self.H_h*phi_H)*ds(deformation = self.displ_h)

        """
        Part for Vi
        """
        self.M += (Vi*phi_Vi)*ds(deformation = self.displ_h)
        self.M += InnerProduct(grad(H).Trace(), grad(phi_Vi).Trace())*ds(deformation = self.displ_h)

        self.l += self.Q_h*phi_Vi*ds(deformation = self.displ_h)

        """
        Part for nu
        """
        self.M += (nu*phi_nu)*ds(deformation = self.displ_h)
        # May be necessary change in .trans
        self.M += -dt*InnerProduct(Grad(zeta).Trace(), Grad(phi_nu).Trace())*ds(deformation = self.displ_h)

        #! Try substitute Weingarten and normal with appropriate one
        self.l += dt*InnerProduct((self.H_h*self.A_h-self.A_h*self.A_h)*self.zeta_h, phi_nu)*ds(deformation = self.displ_h)
        self.l += dt*InnerProduct(Norm(grad(self.H_h).Trace())**2*self.nu_h + self.A_h*self.A_h*grad(self.H_h).Trace(), phi_nu)*ds(deformation = self.displ_h)
        self.l += dt*2*InnerProduct(self.A_h*grad(self.H_h).Trace(), Grad(phi_nu).Trace()*self.nu_h)*ds(deformation = self.displ_h)
        self.l += dt*InnerProduct(self.Q_h, Trace(Grad(phi_nu).Trace()))*ds(deformation = self.displ_h)
        self.l += -dt*InnerProduct(self.Q_h*self.H_h*self.nu_h, phi_nu)*ds(deformation = self.displ_h)
        self.M += (self.nu_h*phi_nu)*ds(deformation = self.displ_h)

        """
        Part for zeta
        """
        self.M += (zeta*phi_zeta)*ds(deformation = self.displ_h)
        self.M += InnerProduct(Grad(nu).Trace(), Grad(phi_zeta).Trace())*ds(deformation = self.displ_h)

        #! Try substitute Weingarten and normal with appropriate one
        self.l += InnerProduct(Norm(self.A_h)**2*self.nu_h, phi_zeta)*ds(deformation = self.displ_h)

        # Draw(self.nu_h, self.mesh)
        # Draw(self.H_h, self.mesh)

        self.M.Assemble()
        self.Minv = self.M.mat.Inverse(X.FreeDofs(), inverse="umfpack")


        ## DUAN LI

        self.invJ, self.detJ = self._compute_jacobian()

        W = V1*V2
        (u2, p),(v2, q) = W.TnT()
        
        self.M2 = BilinearForm(W)
        self.M2 += (InnerProduct(Grad(u2).Trace()*self.invJ, Grad(v2).Trace()*sqrt(self.detJ)) \
                        - p*self.nu*v2*sqrt(self.detJ)\
                            - q*self.nu*u2*sqrt(self.detJ))*ds(deformation = self.displ_h)
        self.M2.Assemble()
        self.M2inv = self.M2.mat.Inverse(W.FreeDofs())

        self.l2 = LinearForm(W)
        self.l2 += (-InnerProduct(Grad(self.X_m).Trace()*self.invJ, \
                Grad(v2).Trace()*sqrt(self.detJ)))*ds(deformation = self.displ_h)
        
        self.dXkappa_h = GridFunction(W)
        
    def __call__(self):

        self.__setup__()

        self.__build_Ab__()

        while self.t.Get()<self.T- 0.5 * self.dt:

            self.displ_h_old.vec.data = self.displ_h.vec.data

            self.__update1__()

            self.HVnz_h.vec.data = self.Minv*self.l.vec
            Draw(self.H_h, self.mesh)
            Draw(self.Vi_h, self.mesh)
            Draw(self.nu_h[0], self.mesh)
            Draw(self.zeta_h, self.mesh)

            self.nu_h.Set(self.nu,  definedon=self.mesh.Boundaries(".*"))
            self.H_h.Set(self.MeanC(), definedon=self.mesh.Boundaries(".*"))
            self.dX_h.Set(self.Vi_h*self.dt*self.nu_h, definedon=self.mesh.Boundaries(".*"))
            self.displ_h.vec.data += self.dX_h.vec
            self.X_m.vec.data += self.dX_h.vec

            self.__update2__()

            # self.dXkappa_h.vec.data = self.M2inv*self.l2.vec
            # self.displ_h.vec.data += self.dXkappa_h.components[0].vec
            # self.X_m.vec.data += self.dXkappa_h.components[0].vec

            self.dummy.vec.data = self.dX_h.vec.data
            # self.dummy.vec.data += self.dXkappa_h.components[0].vec.data

            yield self.dummy
            
    def __update1__(self):

        with TaskManager():

            self.M.Assemble() 
            self.Minv.Update()

            self.l.Assemble()

    def __update2__(self):

        with TaskManager():

            # self.invJ, self.detJ = self._compute_jacobian()

            # self.M2.Assemble() 
            # self.M2inv.Update()

            # self.l2.Assemble()

            self.t.Set(self.t.Get() + self.dt)
    
    def _compute_jacobian(self):

        J = Grad(self.X_m).Trace()*Grad(self.X_m).Trace().trans+OuterProduct(self.nu, self.nu)
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
        f += InnerProduct(wein,tau)*ds \
                + (pi/2-acos(Normalize(gfF)*mu))*tau*mu*mu*ds(element_boundary=True)
        
        gflift = GridFunction(fes)
        
        with TaskManager():
            a.Assemble()
            f.Assemble()
            gflift.vec.data = a.mat.Inverse(fes.FreeDofs(),inverse="sparsecholesky")*f.vec
            
        return gflift
    
    def Wein(self):

        weingarten = self.Lifting()

        return weingarten 
    
    def MeanC(self):

        result = Trace(self.Wein())

        return result 


if __name__ == "__main__":

    ## Example

    t = Parameter(0.0)

    T = 0.001
    dt = 0.001
    fes_order = 1

    mesh, geo = generate_box(maxh=0.08, order_g=fes_order, a=1, b=1)
    # mesh, geo = generate_cube_g5(maxh=0.2, order_g = fes_order, R=3.0)
    # mesh, geo = generate_n_torus(maxh=0.1, order_g = fes_order, char_len=1, n=3)
    # mesh, geo = generate_sphere(maxh=0.1, R = 1, order_g = fes_order)
    # mesh, geo = generate_ball(maxh=0.1, R = 1, order_g = fes_order)
    # mesh, geo = generate_torus(maxh=0.5, R = sqrt(2), r=1, order_g = fes_order)

    solver = WillmoreSolver(mesh=mesh, fes_order=fes_order, dt=dt, t=t, T=T)

    i=0
    out_int = int(((T-0.0)/dt)//10)

    for sol in solver():

        # Draw(CF((x,y,z)), solver.mesh, deformation=solver.displ_h)
        
        if i%(out_int+1)==0:
            Draw(solver.displ_h, solver.mesh, deformation=solver.displ_h)
        i+=1

    ## CONVERGENCE 1

    # t = Parameter(0.0)

    # T = 0.1
    # dt = 0.001
    # fes_order = 1

    # # _, geo = generate_sphere(maxh=0.2, R = 1, order_g = fes_order)
    # _, geo = generate_torus(maxh=0.5, R = sqrt(2), r=1, order_g = fes_order)

    # conv = Convergence(geom=geo, dh = 0.2, power=1.5, n_refinements=1, time_adapt=True, vol_or_bnd='BND')

    # solver = WillmoreSolver(fes_order=fes_order, dt=dt, t=t, T=T)

    # order = conv(solver=solver, exact_sol=CF((0,0,0)), vol_or_bnd_err='BND')
    # print(order)

    ## EXAMPLE

    # t = Parameter(0.0)

    # T = 1.0
    # dt = 0.001
    # fes_order = 1

    # # mesh, geo = generate_box(maxh=0.15, order_g=fes_order, a=1, b=5)
    # # mesh, geo = generate_cube_g5(maxh=0.2, order_g = fes_order, R=3.0)
    # # mesh, geo = generate_n_torus(maxh=0.1, order_g = fes_order, char_len=1, n=3)
    # # mesh, geo = generate_sphere(maxh=0.1, R = 1, order_g = fes_order)
    # mesh, geo = generate_ball(maxh=0.1, R = 1, order_g = fes_order)

    # solver = WillmoreSolver(mesh=mesh, fes_order=fes_order, dt=dt, t=t, T=T)

    # i=0
    # out_int = int(((T-0.0)/dt)//10)

    # for sol in solver():

    #     # Draw(CF((x,y,z)), solver.mesh, deformation=solver.displ_h)
        
    #     if i%(out_int+1)==0:
    #         Draw(solver.displ_h, solver.mesh, deformation=solver.displ_h)
    #     i+=1

# %%
