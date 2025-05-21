from ngsolve import *
from cosmos.pdes.pde_adr_base import BaseADR
from cosmos.pdes.pde_tools import compute_error, MandBP
import numpy as np
import scipy.sparse as sp
from ngsolve.webgui import Draw
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady
from cosmos.solvers.fields import Field

class VolADR(BaseADR):

    def __init__(self, b = Field(), c = Field(), d = Field(), u0 = Field(), rhs = Field(),
                 neu_d = {}, neu_b = {}, dir_d = {}, dir_b = {}, Fneu_b = {},
                 domain:str = '.*', name:str = 'volume_adr', periodic :bool = False,
                 MP:bool = False, BP:bool = False, time_scheme = BDF1()):

        super().__init__()

        self.b = b
        self.c = c
        self.d = d
        self.u0 = u0
        self.rhs = rhs
        self.neu_d = neu_d
        self.neu_b = neu_b
        self.dir_d = dir_d
        self.dir_b = dir_b
        self.Fneu_b = Fneu_b
        self.MP = MP
        self.BP = BP
        self.periodic = periodic
        self.name = [name]
        self.domain = domain
        self.time_scheme = time_scheme

        self.nfields = 1
        self.solute = self.gfu

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Materials(self.domain)

        if solverdata.mesh.ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + self.name + ' cannot be initialized')
        
        if self.periodic:
            self.fes = Compress(Periodic(H1(solverdata.mesh, order = self.fes_order, 
                                            definedon = self.domain)))
        else:
            self.fes = Compress(H1(solverdata.mesh, order = self.fes_order, 
                                   definedon = self.domain))
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)
        self.solute = self.gfu

        if self.u0():
            self.gfu.Set(self.u0())

        self.gfu_save = [self.gfu]

        if self.MP:
            if solverdata.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            elif solverdata.mesh.dim == 3:
                raise Exception('Not implemented yet!')
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            A = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            A += u*v*dx_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu0_vec = self.gfu.vec.Copy().FV().NumPy()
            self.mass0 = np.sum(weights*gfu0_vec)

        for save in self.save_error:
            save.Initialize(solverdata, self)

        for save in self.save_solution:
            save.Initialize(solverdata, self)

    def GetLHS(self, solverdata, trial, test):

        n = specialcf.normal(solverdata.mesh.dim)
        h = specialcf.mesh_size

        if self.c():
            lhs = self.c()*trial[0]*test[0]*dx(deformation = solverdata.ale.deformation)
        else:
            lhs =  CF(0)*trial[0]*test[0]*dx(deformation = solverdata.ale.deformation)

        if self.d():
            lhs += self.d()*grad(trial[0])*grad(test[0])*dx(deformation = solverdata.ale.deformation)
                    
            if self.dir_d:
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for key, field in self.dir_d.items():
                    lhs += - self.d()*InnerProduct(n, grad(trial[0]))*test[0]*ds(definedon = key, skeleton=True, deformation = solverdata.ale.deformation) \
                        - self.d()*InnerProduct(n, grad(test[0]))*trial[0]*ds(definedon = key, skeleton=True, deformation = solverdata.ale.deformation)\
                        + self.d()*alpha/h*trial[0]*test[0]*ds(definedon = key, skeleton = True, deformation = solverdata.ale.deformation)\
            
        if self.b():
            lhs += -self.b()*grad(test[0]) * trial[0]*dx(deformation = solverdata.ale.deformation)

            if self.neu_b:
                for key, field in self.neu_b.items():
                    lhs += IfPos(self.b()*n, self.b()*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = key, deformation = solverdata.ale.deformation)

            if self.dir_b:
                for key, field in self.dir_b.items():
                    lhs += IfPos(self.b()*n, self.b()*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = key, deformation = solverdata.ale.deformation)
                    
            if self.Fneu_b:
                for key, field in self.Fneu_b.items():
                    lhs += IfPos(self.b()*n, self.b()*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = solverdata.mesh.Boundaries('.*') - solverdata.mesh.Boundaries(key), deformation = solverdata.ale.deformation)
            
        return lhs

    def GetRHS(self, solverdata, test):

        n = specialcf.normal(solverdata.mesh.dim)
        h = specialcf.mesh_size

        if self.rhs():
            rhs = self.rhs()*test[0]*dx(deformation = solverdata.ale.deformation)
        else:
            rhs = CF(0)*test[0]*dx(deformation = solverdata.ale.deformation)
        
        if self.dir_d:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            for key, field in self.dir_d.items():
                rhs += self.d()*alpha/h*field()*test[0]*ds(definedon = key, skeleton = True, deformation = solverdata.ale.deformation)\
                    - self.d()*InnerProduct(n, grad(test[0]))*field()*ds(definedon = key, skeleton=True, deformation = solverdata.ale.deformation)
        if self.neu_d:
            for key, field in self.neu_d.items():
                rhs += -field()*n*test[0]*ds(definedon = key, deformation = solverdata.ale.deformation)

        if self.dir_b:
            for key, field in self.dir_b.items():
                rhs += -IfPos(self.b()*n, CF(0), self.b()*n*field())*test[0]*ds(definedon = key, deformation = solverdata.ale.deformation)
        if self.neu_b:
            for key, field in self.neu_b.items():
                rhs += -IfPos(self.b()*n, CF(0), field()*n)*test[0]*ds(definedon = key, deformation = solverdata.ale.deformation)

        if self.Fneu_b:
            for key, field in self.Fneu_b.items():
                rhs += -field()*n*test[0]*ds(definedon = solverdata.mesh.Boundaries(key), deformation = solverdata.ale.deformation)
            
        return rhs

    def GetMass(self, solverdata, trial, test):

        mass = 1/solverdata.dt*trial[0]*test[0]*dx(deformation = solverdata.ale.deformation)
        return mass
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        if self.BP and not self.MP:

            gfu_vec = self.gfu.vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.BP)
            self.gfu.vec.data = gfu_new

        elif self.MP:

            if hasattr(solverdata, 'dt'):
                dt = solverdata.dt.Get()
            else:
                raise Exception('A time-dependent simulation is needed to impose conservative mass!')

            if solverdata.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir }, deformation=solverdata.ale.deformation)
            elif solverdata.mesh.dim == 3:
                raise Exception('Not yet implemented!')
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir }, deformation=solverdata.ale.deformation)
            A = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            A += u*v*dx_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu.vec.Copy().FV().NumPy()

            if self.BP:
                BP = self.BP
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.MP, mass0 =self.mass0, dt = dt)

            self.gfu.vec.data = gfu_new

    def get_error(self, solverdata, ex_sol, norm):

        err = compute_error(data=solverdata, gfu = self.gfu, u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        return [err]

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              Diffusion-dominant problems are supposed to be simulated.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, field in self.params.items():
        #     print('  -', key, '- with field ', str(field))

        # print(60*'-', '\n')