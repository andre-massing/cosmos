# %%
from cosmos.utils.generate_synapse_meshes import *
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# Only needed for testing
from ngsolve.webgui import Draw
import time as time
import numpy as np
import pandas as pd

class WillmoreSolverOpen(UnsteadySolver):

    def __init__(self, mesh=None, fes_order=1, dt=0.1, t = Parameter(0.0), T=1.0, bnd_cond=None, u0 = None, rhs=None, barbed_ends = CF(0.0), bnd_name = "default", verbose = 0):

        super().__init__(mesh=mesh, dt = dt, t = t, T=T, bnd_cond=bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.rhs = rhs
        self.u0 = u0
        self.barbed_ends = barbed_ends
        self.bnd_name = bnd_name

        self.setup = False
        if mesh != None:
            self.setup = True

            self.__setup__()

            self.__build_Ab__()
        
    def __build_Ab__(self):

        V1_0 = VectorH1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'), dirichlet_bbnd = self.mesh.BBoundaries(self.bnd_name))
        V1 = VectorH1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))
        V2 = H1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))

        self.displ_h = GridFunction(V1_0)
        self.displ_h_old = GridFunction(V1_0)
        self.dXtot_h = GridFunction(V1_0)

        if self.mesh.dim == 3:
            Ident = CF((x,y,z))
        elif self.mesh.dim == 2:
            Ident = CF((x,y))

        self.X_m = GridFunction(V1_0)
        self.X_m.Set(Ident, definedon = self.mesh.Boundaries(".*"))

        self.X0 = GridFunction(V1_0)
        self.X0.Set(Ident, definedon = self.mesh.Boundaries(".*"))

        self.u_h = GridFunction(V1)

        self.nu = specialcf.normal(self.mesh.dim)
        self.Ps = Id(self.mesh.dim) - OuterProduct(self.nu, self.nu)

        def gradient(phi):

            return Grad(phi).Trace().trans
        
        def D_s(chi):
            sym = 0.5*(Grad(chi).Trace()+Grad(chi).Trace().trans)
            return sym
        
        X = V1_0*V1
        (dX, kappa), (chi, eta) = X.TnT()

        self.dXk1_h = GridFunction(X)
        self.dX1_h, self.kappa1_h = self.dXk1_h.components

        rho = pi
        xsi = CF((x,y,z))*sin(rho) + CF((0,0,1))*cos(rho)
        
        (kappa0, eta0) = V1.TnT()
        M0 = BilinearForm(V1)
        l0 = LinearForm(V1)
        M0 += kappa0*eta0*ds
        M0.Assemble()
        l0 += -InnerProduct(self.Ps, Grad(eta0).Trace())*ds
        l0 += InnerProduct(xsi, eta0)*ds(definedon=self.mesh.BBoundaries(self.bnd_name))
        l0.Assemble()

        self.kappa1_h.vec.data = M0.mat.Inverse(V1.FreeDofs())*l0.vec

        Draw(self.kappa1_h)

        self.dt_var = Parameter(self.dt)

        self.M = BilinearForm(X)
        self.l = LinearForm(X)

        self.M += InnerProduct(dX, chi)/self.dt_var*ds(deformation = self.displ_h)
        self.M += -InnerProduct(gradient(kappa), gradient(chi))*ds(deformation = self.displ_h)

        self.l += InnerProduct(Trace(gradient(self.kappa1_h)),Trace(gradient(chi)))*ds(deformation = self.displ_h)
        self.l += -2*InnerProduct(gradient(self.kappa1_h), D_s(chi)*gradient(self.X_m))*ds(deformation = self.displ_h)
        self.l += 0.5*InnerProduct((Norm(self.kappa1_h)**2)*gradient(self.X_m), gradient(chi))*ds(deformation = self.displ_h)

        self.M += (kappa*eta+InnerProduct(gradient(dX), gradient(eta)))*ds(deformation = self.displ_h)

        self.l += -InnerProduct(gradient(self.X_m),gradient(eta))*ds(deformation = self.displ_h)

        ## Addition for the open boundary
        rho = pi
        xsi = CF((x,y,z))*sin(rho) + CF((0,0,1))*cos(rho)
        self.l += InnerProduct(xsi, eta)*ds(deformation = self.displ_h, definedon=mesh.BBoundaries(self.bnd_name))


        # self.M += InnerProduct(dX, chi)/self.dt_var*ds(deformation = self.displ_h)
        # self.M += -InnerProduct(gradient(kappa), gradient(chi))*ds(deformation = self.displ_h)

        # self.M += -InnerProduct(Trace(gradient(kappa)),Trace(gradient(chi)))*ds(deformation = self.displ_h)
        # self.M += 2*InnerProduct(gradient(kappa), D_s(chi)*gradient(self.X_m))*ds(deformation = self.displ_h)

        # self.M += -0.5*InnerProduct((Norm(self.kappa1_h)**2)*gradient(dX), gradient(chi))*ds(deformation = self.displ_h)
        # self.l += 0.5*InnerProduct((Norm(self.kappa1_h)**2)*gradient(self.X_m), gradient(chi))*ds(deformation = self.displ_h)

        # self.M += (kappa*eta+InnerProduct(gradient(dX), gradient(eta)))*ds(deformation = self.displ_h)

        # self.l += -InnerProduct(gradient(self.X_m),gradient(eta))*ds(deformation = self.displ_h)

        # X = V1*V1*V1
        # (dX, Y, kappa), (chi, zeta, eta) = X.TnT()

        # self.dXk1_h = GridFunction(X)
        # self.dX1_h, self.Y1_h, self.kappa1_h = self.dXk1_h.components
        
        # (kappa0, eta0) = V1.TnT()
        # M0 = BilinearForm(V1)
        # l0 = LinearForm(V1)
        # M0 += kappa0*eta0*ds
        # M0.Assemble()
        # l0 += -InnerProduct(self.Ps, Grad(eta0).Trace())*ds
        # l0.Assemble()

        # self.kappa1_h.vec.data = M0.mat.Inverse(V1.FreeDofs())*l0.vec
        # self.Y1_h.vec.data = self.kappa1_h.vec.data

        # self.dt_var = Parameter(self.dt)

        # self.M = BilinearForm(X)
        # self.l = LinearForm(X)

        # self.M += InnerProduct(dX, chi)/self.dt_var*ds(deformation = self.displ_h)
        # self.M += -InnerProduct(gradient(Y), gradient(chi))*ds(deformation = self.displ_h)

        # self.l += InnerProduct(Trace(gradient(self.Y1_h)),Trace(gradient(chi)))*ds(deformation = self.displ_h)
        # self.l += -2*InnerProduct(gradient(self.Y1_h), D_s(chi))*ds(deformation = self.displ_h)
        # self.l += -0.5*InnerProduct((Norm(self.kappa1_h)**2),Trace(gradient(chi)))*ds(deformation = self.displ_h)
        # self.l += InnerProduct(self.kappa1_h*self.Y1_h,Trace(gradient(chi)))*ds(deformation = self.displ_h)

        # self.M += (kappa*zeta - Y*zeta)*ds(deformation = self.displ_h)

        # self.M += (kappa*eta+InnerProduct(gradient(dX), gradient(eta)))*ds(deformation = self.displ_h)

        # self.l += -InnerProduct(gradient(self.X_m), gradient(eta))*ds(deformation = self.displ_h)


        # EXTRA COMPONENT DUE TO SURFACE REACTANT
        beta = 10*self.barbed_ends
        self.l += beta*self.nu*chi*ds(deformation = self.displ_h)

        self.M.Assemble()
        self.Minv = self.M.mat.Inverse(X.FreeDofs(), inverse="umfpack")

        ## DUAN LI

        W = V1_0*V2

        (u2, p),(v2, q) = W.TnT()

        self.dXk2_h = GridFunction(W)
        self.dX2_h, self.kappa2_h = self.dXk2_h.components


        self.M2 = BilinearForm(W)
        self.M2 += InnerProduct(gradient(u2), gradient(v2))*ds
        self.M2 += - p*self.nu*v2*ds(deformation = self.displ_h)
        self.M2 += - q*self.nu*u2*ds(deformation = self.displ_h)
        
        self.M2.Assemble()
        self.M2inv = self.M2.mat.Inverse(W.FreeDofs())

        self.l2 = LinearForm(W)
        self.l2 += (-InnerProduct(gradient(self.X_m), \
                                            gradient(v2)))*ds
        
        # self.M2 = BilinearForm(W)
        # self.M2 += (InnerProduct(gradient(u2)*Inv(gradient(self.X0)*gradient(self.X0).trans+OuterProduct(self.nu, self.nu)), gradient(v2)*sqrt(Det(gradient(self.X0)*gradient(self.X0).trans+OuterProduct(self.nu, self.nu)))) \
        #                 - p*self.nu*v2*sqrt(Det(gradient(self.X0)*gradient(self.X0).trans+OuterProduct(self.nu, self.nu)))\
        #                     - q*self.nu*u2*sqrt(Det(gradient(self.X0)*gradient(self.X0).trans+OuterProduct(self.nu, self.nu))))*ds(deformation = self.displ_h)
        # self.M2.Assemble()
        # self.M2inv = self.M2.mat.Inverse()

        # self.l2 = LinearForm(W)
        # self.l2 += (-InnerProduct(gradient(self.X_m)*Inv(gradient(self.X0)*gradient(self.X0).trans+OuterProduct(self.nu, self.nu)), \
        #         gradient(v2)*sqrt(Det(gradient(self.X0)*gradient(self.X0).trans+OuterProduct(self.nu, self.nu)))))*ds(deformation = self.displ_h)
        
        
        
    def __call__(self):

        if self.setup == False:

            self.__setup__()

            self.__build_Ab__()

        while self.t.Get()<self.T- 0.5 * self.dt_var.Get():

            self.displ_h_old.vec.data = self.displ_h.vec.data

            self.__update1__()

            self.dXk1_h.vec.data = self.Minv*self.l.vec
            self.displ_h.vec.data += self.dX1_h.vec
            self.X_m.vec.data += self.dX1_h.vec

            self.__update2__()

            self.dXk2_h.vec.data = self.M2inv*self.l2.vec
            self.displ_h.vec.data += self.dX2_h.vec
            self.X_m.vec.data += self.dX2_h.vec

            self.dXtot_h.vec.data = self.dX1_h.vec.data + self.dX2_h.vec.data

            yield self.displ_h*self.nu
            
    def __update1__(self):

        with TaskManager():

            self.dt_var.Set(self.dt)

            self.M.Assemble() 
            self.Minv.Update()

            self.l.Assemble()

    def __update2__(self):

        with TaskManager():

            self.M2.Assemble() 
            self.M2inv.Update()

            self.l2.Assemble()

            self.t.Set(self.t.Get() + self.dt_var.Get())

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

    # ## CONVERGENCE SPHERE

    # t = Parameter(0.0)
    # T = 1.0
    # dt0 = 0.01
    # t_refinements = 3
    # t_power = 1.5
    # dt_vals = dt0/np.power(1.5, np.arange(t_refinements+1))

    # h_refinements = 2
    # fes_order = 1
    # dh0 = 0.2
    # h_power = 1.5
    # dh_vals = dh0/np.power(1.5, np.arange(h_refinements+1))

    # _, geo = generate_sphere(maxh=0.2, R = 1, order_g = fes_order)

    # ERR = np.zeros((len(dt_vals), len(dh_vals)))

    # for i, dt in enumerate(dt_vals):

    #     conv = Convergence(geom=geo, dh = dh0, power=h_power, n_refinements=h_refinements, time_adapt=False, vol_or_bnd='BND')

    #     solver = WillmoreSolver(fes_order=fes_order, dt=dt, t=t, T=T)

    #     err_dt = conv(solver=solver, exact_sol=CF(0), vol_or_bnd_err='BND')

    #     ERR[i,:] = err_dt

    # name = "convergence/willmore/sphere_time_k" + str(fes_order) + ".dat"

    # space_labels =  [f'{x:.2e}' for x in dh_vals]
    # space_labels = ["dt"] + space_labels
    # time_output = np.column_stack((dt_vals, ERR))

    # df = pd.DataFrame(time_output, columns=space_labels)
    # df.to_csv(name, sep='\t', index=False)

    # name = "convergence/willmore/sphere_space_k" + str(fes_order) + ".dat"

    # time_labels =  [f'{x:.2e}' for x in dt_vals]
    # time_labels = ["dh"] + time_labels
    # space_output = np.column_stack((dh_vals, ERR.T))

    # df = pd.DataFrame(space_output, columns=time_labels)
    # df.to_csv(name, sep='\t', index=False)

    ## CONVERGENCE TORUS

    # t = Parameter(0.0)
    # T = 1.0
    # dt0 = 0.01
    # t_refinements = 2
    # t_power = 1.5
    # dt_vals = dt0/np.power(1.5, np.arange(t_refinements+1))

    # h_refinements = 2
    # fes_order = 1
    # dh0 = 0.4
    # h_power = 1.5
    # dh_vals = dh0/np.power(1.5, np.arange(h_refinements+1))

    # _, geo = generate_torus(maxh=0.1, R = sqrt(2), r = 1, order_g = fes_order, vol_or_bnd='BND')

    # ERR = np.zeros((len(dt_vals), len(dh_vals)))

    # for i, dt in enumerate(dt_vals):

    #     conv = Convergence(geom=geo, dh = dh0, power=h_power, n_refinements=h_refinements, time_adapt=False, vol_or_bnd='BND')

    #     solver = WillmoreSolver(fes_order=fes_order, dt=dt, t=t, T=T)

    #     err_dt = conv(solver=solver, exact_sol=CF(0), vol_or_bnd_err='BND')

    #     ERR[i,:] = err_dt

    # name = "convergence/willmore/torus_time_k" + str(fes_order) + ".dat"

    # space_labels =  [f'{x:.2e}' for x in dh_vals]
    # space_labels = ["dt"] + space_labels
    # time_output = np.column_stack((dt_vals, ERR))

    # df = pd.DataFrame(time_output, columns=space_labels)
    # df.to_csv(name, sep='\t', index=False)

    # name = "convergence/willmore/torus_space_k" + str(fes_order) + ".dat"

    # time_labels =  [f'{x:.2e}' for x in dt_vals]
    # time_labels = ["dh"] + time_labels
    # space_output = np.column_stack((dh_vals, ERR.T))

    # df = pd.DataFrame(space_output, columns=time_labels)
    # df.to_csv(name, sep='\t', index=False)


    # EXAMPLE

    # mesh, _ = generate_open_sphere(maxh=0.1, R = 1, order_g = 1)
    # Draw(mesh)

    # t = Parameter(0.0)

    # T = 0.1
    # dt = 0.001
    # fes_order = 1

    # solver = WillmoreSolverOpen(mesh=mesh, fes_order=fes_order, dt=dt, t=t, T=T, bnd_name = "boundary")

    # solver_generator = solver()

    # vtkout = VTKOutput(mesh,coefs=[solver.displ_h], names=["displ"],filename="./examples/willmore_open/vtk/sphere")
    # vtkout.Do(time = solver.t.Get(), vb = BND)

    # i=0
    # out_int = int(((T-0.0)/dt)//10)

    # nsteps = int((T-0.0)/dt)
    # ramp_steps = int(nsteps*0.1)
    
    # exp0 = -3
    # exp1 = int(np.log10(dt))
    # ramp_vals = np.logspace(exp0, exp1, num=ramp_steps)

    # time_vals = np.concatenate((ramp_vals, np.ones(nsteps)*dt))

    # settings={"camera": {"transformations": [{"type": "rotateX", "angle": -45}]}}
    # scene = Draw(solver.displ_h, solver.mesh, deformation=solver.displ_h, settings= settings)

    # for i, dt_i in enumerate(time_vals):

    #     solver.dt = dt_i

    #     _ = next(solver_generator)

    #     if i%(out_int+1)==0:
    #         vtkout.Do(time = solver.t.Get(), vb = BND)
    #         scene.Redraw()
    #     i+=1

    # EXAMPLE 2

    mesh, _ = generate_open_torus(maxh=0.15, R = 2, r = 1, order_g = 1)
    Draw(mesh)

    t = Parameter(0.0)

    T = 0.5
    dt = 0.001
    fes_order = 1

    solver = WillmoreSolverOpen(mesh=mesh, fes_order=fes_order, dt=dt, t=t, T=T, bnd_name = "boundary")

    solver_generator = solver()

    vtkout = VTKOutput(mesh,coefs=[solver.displ_h], names=["displ"],filename="./examples/willmore_open/vtk/torus")
    vtkout.Do(time = solver.t.Get(), vb = BND)

    i=0
    out_int = int(((T-0.0)/dt)//10)

    nsteps = int((T-0.0)/dt)
    ramp_steps = int(nsteps*0.1)
    
    exp0 = -3
    exp1 = int(np.log10(dt))
    ramp_vals = np.logspace(exp0, exp1, num=ramp_steps)

    time_vals = np.concatenate((ramp_vals, np.ones(nsteps)*dt))

    settings={"camera": {"transformations": [{"type": "rotateX", "angle": -45}]}}
    scene = Draw(solver.displ_h, solver.mesh, deformation=solver.displ_h, settings= settings)

    for i, dt_i in enumerate(time_vals):

        solver.dt = dt_i

        _ = next(solver_generator)

        if i%(out_int+1)==0:
            vtkout.Do(time = solver.t.Get(), vb = BND)
            scene.Redraw()
        i+=1
# %%
