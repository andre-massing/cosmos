# %%
from cosmos.utils.generate_synapse_meshes import *
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from cosmos.solvers.ucsd.ucsd_um_bs_adr_cg import *
from cosmos.solvers.ucsd.ucsd_willmore import *
from cosmos.solvers.ucsd.ucsd_v_poisson import *
from ngsolve import *
import netgen.occ as occ

# Only needed for testing
from ngsolve.webgui import Draw
import time as time
import numpy as np

class ucsd_total(UnsteadySolver):

    def __init__(self, mesh=None, fes_order = 1, dt = 0.1, T = 1.0):

        super().__init__(mesh=mesh, dt = dt, T=T)

        self.fes_order = fes_order

        self.u0_v = exp(-5*((x-0.5)**2+y**2+z**2))
        self.u0_s = exp(-5*((x-0.5)**2+y**2+z**2))
        self.displ = CF((0,)*self.mesh.dim)
        
        self.__setup_willmore__()

        self.__setup_um__()

        self.__setup_vp__()

    def __call__(self):

        nsteps = int((self.T-0.0)/self.dt)
        ramp_steps = int(nsteps*0.1)
        
        k=0
        n_out = 100
        out_int = int(nsteps//n_out)
        exp0 = -7
        exp1 = int(np.log10(self.dt))
        ramp_vals = np.logspace(exp0, exp1, num=ramp_steps)

        time_vals = np.concatenate((ramp_vals, np.ones(nsteps)*self.dt))


        for i, dt_i in enumerate(time_vals):

            self.solver_w.dt = dt_i
            self.solver_um.dt = dt_i

            sol1 = next(self.solver_w_generator)

            sol2 = self.__compute_d__(sol1, self.solver_um.displ)

            self.solver_um.displ = sol2

            sol3 = next(self.solver_um_generator)

            self.solver_w.barbed_ends = sol3

            # if k%(out_int+1)==0:
            #     Draw(self.solver_um.dummy, self.solver_w.mesh, deformation=self.solver_w.displ_h)

            if k==0:
                vtktmp = VTKOutput(self.mesh, coefs=[self.solver_um.displ_h, self.solver_um.dummy, self.solver_um.u_h, sol2], names = ["displ", "v", "u", "displ_w"], filename = "./tmp/sol")
                vtktmp.Do(time = self.solver_um.t.Get())
            if k%(out_int+1)==0:
                vtktmp.Do(time = self.solver_um.t.Get())

            #     u_tot = Integrate(self.solver_um.u_h, self.mesh, order = self.fes_order+2)
            #     v_tot =  Integrate(self.solver_um.v_h, self.mesh, order = self.fes_order+2, VOL_or_BND=BND)
            #     print('u+v: ', u_tot+v_tot)
            k+=1

            print(self.solver_w.t.Get())

            yield sol3


    def __setup_willmore__(self):

        t2 = Parameter(0.0)

        self.solver_w = WillmoreSolver(mesh=self.mesh, fes_order=self.fes_order, dt=self.dt, t=t2, T=self.T, barbed_ends=self.u0_s)

        self.solver_w_generator = self.solver_w()


    def __setup_um__(self):

        d_v = 0.001
        c_v = 0.0
        b_v = CF((1,0,0))

        d_s = 0.001
        c_s = 0.0
        b_s = CF((0,)*self.mesh.dim)

        rhs_v = CF(0.0)

        rhs_s = CF(0.0)

        alpha = 1.0
        beta = 1.0

        bnd_cond=[['dir', 'membrane', CF((0,)*self.mesh.dim)]]
        
        t1 = Parameter(0.0)
        
        self.solver_um = um_bs_ADR_CGSolver(mesh = self.mesh, fes_order=self.fes_order, b=[b_v, b_s], c=[c_v, c_s], d=[d_v, d_s], dt=self.dt, t=t1, T=self.T, u0=[self.u0_v, self.u0_s], rhs=[rhs_v, rhs_s], bnd_cond = bnd_cond, coupling = [alpha, beta], displ=self.displ)

        self.solver_um_generator = self.solver_um()

    def __setup_vp__(self):

        bnd_cond = [['dir', 'membrane', CF((0,)*self.mesh.dim)]]
        self.vp_solver = VectorPoissonSolver(mesh = self.mesh, fes_order=self.fes_order, bnd_cond=bnd_cond, rhs=CF((0,)*self.mesh.dim), displ=self.displ)

    def __compute_d__(self, u0, displ=None):

        if displ==None:
            displ = CF((0,)*self.mesh.dim)

        self.vp_solver.bnd_cond = [['dir', 'membrane', u0]]
        self.vp_solver.displ = displ
        solver_generator = self.vp_solver()

        res = next(solver_generator)

        return res


        

if __name__ == "__main__":

    # mesh, _ = generate_ball(maxh=0.05, R = 0.5, order_g =1)
    # mesh, _ = generate_cube(maxh=0.03, R = 0.5, order_g =1)
    mesh, geo = generate_cube(maxh=0.05, order_g=1, R=1, b=1)

    # mesh, _ = generate_circle(maxh=0.2, R = 1.0, order_g =2)

    # R = 1.0
    # body = occ.Box(occ.Pnt(-R/2,-R/2,-R/2), occ.Pnt(R/2, R/2, R/2))
    # body.faces.name = "membrane"

    # geo= occ.OCCGeometry(body)
    # mesh = geo.GenerateMesh(maxh=0.1)
    # mesh = Mesh(mesh)

    dt = 1e-3
    T = 1.0

    solver = ucsd_total(mesh = mesh, dt=dt, T = T)

    i=0
    out_int = int(((T-0.0)/dt)//100)

    for u in solver():

        j=0
        # Draw(solver.solver_w.barbed_ends, mesh, deformation = solver.solver_um.displ_h)

        # if i%(out_int+1)==0:
        #     Draw(solver.solver_w.barbed_ends, mesh, deformation = solver.solver_um.displ_h)
        # i+=1

# %%
