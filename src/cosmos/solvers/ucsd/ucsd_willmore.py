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

class WillmoreSolver(UnsteadySolver):

    def __init__(self, mesh=None, fes_order=1, dt=0.1, t = Parameter(0.0), T=1.0, bnd_cond=None, u0 = None, rhs=None, barbed_ends = CF(0.0), verbose = 0):

        super().__init__(mesh=mesh, dt = dt, t = t, T=T, bnd_cond=bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.rhs = rhs
        self.u0 = u0
        self.barbed_ends = barbed_ends

        self.setup = False
        if mesh != None:
            self.setup = True

            self.__setup__()

            self.__build_Ab__()
        
    def __build_Ab__(self):

        V1 = VectorH1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))
        V2 = H1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))

        self.displ_h = GridFunction(V1)
        self.displ_h_old = GridFunction(V1)
        self.dXtot_h = GridFunction(V1)

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

        self.dXk1_h = GridFunction(X)
        self.dX1_h, self.kappa1_h = self.dXk1_h.components
        
        (kappa0, eta0) = V1.TnT()
        M0 = BilinearForm(V1)
        l0 = LinearForm(V1)
        M0 += kappa0*eta0*ds
        M0.Assemble()
        l0 += -InnerProduct(self.Ps, Grad(eta0).Trace())*ds
        l0.Assemble()

        self.kappa1_h.vec.data = M0.mat.Inverse(V1.FreeDofs())*l0.vec

        self.dt_var = Parameter(self.dt)

        self.M = BilinearForm(X)
        self.l = LinearForm(X)

        self.M += InnerProduct(dX, chi)/self.dt_var*ds(deformation = self.displ_h)
        self.M += -InnerProduct(Grad(kappa).Trace(), Grad(chi).Trace())*ds(deformation = self.displ_h)

        self.l += InnerProduct(Trace(Grad(self.kappa1_h).Trace()),Trace(Grad(chi).Trace()))*ds(deformation = self.displ_h)
        self.l += -2*InnerProduct(Grad(self.kappa1_h).Trace().trans, D_s(chi)*Grad(self.X_m).Trace().trans)*ds(deformation = self.displ_h)
        self.l += 0.5*InnerProduct((Norm(self.kappa1_h)**2)*Grad(self.X_m).Trace(),Grad(chi).Trace())*ds(deformation = self.displ_h)

        self.M += (kappa*eta+InnerProduct(Grad(dX).Trace(), Grad(eta).Trace()))*ds(deformation = self.displ_h)

        self.l += -InnerProduct(Grad(self.X_m).Trace(), Grad(eta).Trace())*ds(deformation = self.displ_h)

        # EXTRA COMPONENT DUE TO SURFACE REACTANT
        # beta = 10*self.barbed_ends
        # self.l += beta*self.nu*chi*ds(deformation = self.displ_h)

        self.M.Assemble()
        self.Minv = self.M.mat.Inverse(X.FreeDofs(), inverse="umfpack")

        ## DUAN LI

        W = V1*V2
        (u2, p),(v2, q) = W.TnT()

        self.dXk2_h = GridFunction(W)
        self.dX2_h, self.kappa2_h = self.dXk2_h.components

        self.M2 = BilinearForm(W)
        self.M2 += (InnerProduct(Grad(u2).Trace(), Grad(v2).Trace()) \
                                - p*self.nu*v2\
                                    - q*self.nu*u2)*ds
        self.M2.Assemble()
        self.M2inv = self.M2.mat.Inverse(W.FreeDofs())

        self.l2 = LinearForm(W)
        self.l2 += (-InnerProduct(Grad(self.X_m).Trace(), \
                                            Grad(v2).Trace()))*ds
        
        
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



if __name__ == "__main__":

    # ## CONVERGENCE SPHERE

    # t = Parameter(0.0)
    # T = 1.0
    # dt0 = 0.01
    # t_refinements = 1
    # t_power = 1.5
    # dt_vals = dt0/np.power(1.5, np.arange(t_refinements+1))

    # h_refinements = 1
    # fes_order = 1
    # dh0 = 0.2
    # h_power = 1.5
    # dh_vals = dh0/np.power(1.5, np.arange(h_refinements+1))

    # _, geo = generate_sphere(maxh=0.1, R = 1, order_g = fes_order)

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

    # ## CONVERGENCE TORUS

    # t = Parameter(0.0)
    # T = 1.0
    # dt0 = 0.01
    # t_refinements = 1
    # t_power = 1.5
    # dt_vals = dt0/np.power(1.5, np.arange(t_refinements+1))

    # h_refinements = 1
    # fes_order = 1
    # dh0 = 0.2
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

    t = Parameter(0.0)

    T = 1.0
    dt = 0.001
    fes_order = 1

    mesh, geo = generate_torus(maxh=0.2, R = 2, r=1.0, order_g = fes_order, vol_or_bnd='BND')

    solver = WillmoreSolver(mesh=mesh, fes_order=fes_order, dt=dt, t=t, T=T)

    vtkout = VTKOutput(mesh,coefs=[solver.displ_h], names=["displ"],filename="./examples/willmore/vtk/torus")
    vtkout.Do(time = solver.t.Get(), vb = BND)

    i=0
    out_int = int(((T-0.0)/dt)//20)

    for sol in solver():
        
        if i%(out_int+1)==0:
            vtkout.Do(time = solver.t.Get(), vb = BND)
            Draw(solver.displ_h, solver.mesh, deformation=solver.displ_h)
        i+=1

# %%
