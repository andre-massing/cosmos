from ngsolve import *
from cosmos.pdes.pde_ch_base import BaseCH
from cosmos.pdes.pde_tools import compute_error, MandBP
import numpy as np
import scipy.sparse as sp
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady
from ngsolve.webgui import Draw

class CahnHilliardBnd(BaseCH):

    def __init__(self, theta = None, gamma = None, sigma = None, c0 = None, phi0 = None, rhs_c = None,
                 rhs_p = None, domain:str = '.*', name = ['phase', 'potential'], periodic:bool = False,
                 MP:bool = False, BP:bool = None, time_scheme = BDF1()):

        super().__init__()

        if theta:
            self.theta = Field(theta)
        else:
            self.theta = Field(1.0)
        if gamma:
            self.gamma = Field(gamma)
        else:
            self.gamma = Field(1.0)
        if sigma:
            self.sigma = Field(sigma)
        else:
            self.sigma = Field(1.0)
        self.c0 = Field(c0)
        self.phi0 = Field(phi0)
        self.rhs_c = Field(rhs_c)
        self.rhs_p = Field(rhs_p)
        self.domain = domain
        self.name = name
        self.periodic = periodic
        self.MP = MP
        self.BP = BP
        self.time_scheme = time_scheme

        self.fields = 2
        self.nonlinear = True

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Boundaries(self.domain)
        
        if self.periodic:
            V = Compress(Periodic(H1(solverdata.mesh, order = self.fes_order, 
                                            definedon = self.domain)))
        else:
            V = Compress(H1(solverdata.mesh, order = self.fes_order, 
                                   definedon = self.domain))
        self.fes = V*V
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)
        self.phase, self.potential = self.gfu.components
        self.gfu_save = list(GridFunction(Compress(H1(solverdata.mesh, order = self.fes_order))**2).components)

        if self.c0():
            self.phase.Set(self.c0(), definedon = self.domain)
        if self.phi0():
            self.potential.Set(self.phi0(), definedon = self.domain)

        self.gfu_save[0].Set(self.phase, definedon = self.domain)
        self.gfu_save[1].Set(self.potential, definedon = self.domain)

        if self.MP:

            if solverdata.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir })
            elif solverdata.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir })
            A = BilinearForm(self.phase.space, symmetric = True)
            u, v = self.phase.space.TnT()
            A += u*v*ds_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.phase.vec.Copy().FV().NumPy()
            self.mass0 = np.sum(weights*gfu_vec)

        for save in self.save_error:
            save.Initialize(solverdata, self)

        for save in self.save_solution:
            save.Initialize(solverdata, self)

        ns = specialcf.normal(solverdata.mesh.dim)
        self.Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)        

    def GetLHS(self, solverdata, trial, test):

        lhs = self.theta()*grad(trial[1]).Trace()*grad(test[0]).Trace()*ds(deformation = solverdata.ale.deformation)
        lhs += trial[1]*test[1]*ds(deformation = solverdata.ale.deformation)
        lhs += -1*self.sigma()*self.gamma()*grad(trial[0]).Trace()*grad(test[1]).Trace()*ds(deformation = solverdata.ale.deformation)
        
        lhs += (solverdata.ale.material_velocity - solverdata.ale.velocity)*grad(trial[0]).Trace()*test[0]*ds(deformation = solverdata.ale.deformation)
                    
        return lhs

    def GetRHS(self, solverdata, test):

        rhs = CF(0)*test[0]*ds(deformation = solverdata.ale.deformation)
        if self.rhs_c():
            rhs += self.rhs_c()*test[0]*ds(deformation = solverdata.ale.deformation)
        if self.rhs_p():
            rhs += self.rhs_p()*test[1]*ds(deformation = solverdata.ale.deformation)
            
        return rhs

    def GetMass(self, solverdata, trial, test):

        mass = 1/solverdata.dt*trial[0]*test[0]*ds(deformation = solverdata.ale.deformation)
        return mass

    def GetNL(self, solverdata, trial, test):

        nonlin = -self.sigma()/self.gamma()*(trial[0]**3 - trial[0])*test[1]*ds(deformation = solverdata.ale.deformation)
                    
        return nonlin
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        if self.BP and not self.MP:

            gfu_vec = self.phase.vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.BP)
            self.phase.vec.data = gfu_new

        elif self.MP:

            if hasattr(solverdata, 'dt'):
                dt = solverdata.dt.Get()
            else:
                raise Exception('A time-dependent simulation is needed to impose conservative mass!')

            if solverdata.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir }, deformation=solverdata.ale.deformation)
            elif solverdata.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir }, deformation=solverdata.ale.deformation)
            A = BilinearForm(self.phase.space, symmetric = True)
            u, v = self.phase.space.TnT()
            A += u*v*ds_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.phase.vec.Copy().FV().NumPy()

            if self.BP:
                BP = self.BP
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.MP, mass0 =self.mass0, dt = dt)

            self.phase.vec.data = gfu_new

    def get_error(self, solverdata, ex_sol, norm):

        err1 = compute_error(data=solverdata, gfu = self.phase, u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        err2 = compute_error(data=solverdata, gfu = self.gfu_comp[1], u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err1, err2]

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              Diffusion-dominant problems are supposed to be simulated.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')