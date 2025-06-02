from ngsolve import *
from cosmos.pdes.pde_mc_base import BaseMC
from cosmos.pdes.pde_tools import compute_error
from ngsolve.webgui import Draw
import numpy as np
from cosmos.solvers.fields import Field
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady

class MCDziuk(BaseMC):

    def __init__(self, rhs = None, domain:str = '.*', name:str = 'displacement',
                 time_scheme = BDF1()):

        super().__init__()

        self.rhs = Field(rhs)
        self.domain = domain
        self.name = [name]
        self.time_scheme = time_scheme

        self.nfields = 1
        self.displacement = self.gfu

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Boundaries(self.domain)
        
        self.fes = Compress(VectorH1(solverdata.mesh, order=self.fes_order,
                    definedon=self.domain))

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.displacement = self.gfu
        self.X0 = GridFunction(self.fes)
        if solverdata.mesh.dim == 2:
            self.X0.Set(CF((x,y)), definedon=self.domain)
        elif solverdata.mesh.dim == 3:
            self.X0.Set(CF((x, y, z)), definedon=self.domain)
            
        V_vol = VectorH1(solverdata.mesh, order = self.fes_order)
        self.gfu_save = [GridFunction(Compress(V_vol))]

        for save in self.save_error:
            save.Initialize(solverdata, self)

        for save in self.save_solution:
            save.Initialize(solverdata, self)
            
    def GetLHS(self, solverdata, trial, test):

        lhs = (InnerProduct(grad(trial[0]).Trace(), grad(test[0]).Trace()))*ds(deformation=solverdata.ale.deformation)
        
        return lhs
        
    def GetRHS(self, solverdata, test):

        rhs = -1*InnerProduct(grad(self.X0).Trace(), grad(test[0]).Trace())*ds(deformation=solverdata.ale.deformation)
        if self.rhs():
            rhs += self.rhs()*test[0]*ds(deformation = solverdata.ale.deformation)

        return rhs
    
    def GetMass(self, solverdata, trial, test):

        mass = trial[0]*test[0]/solverdata.dt*ds(deformation=solverdata.ale.deformation)

        return mass
    
    def Update(self, solverdata):

        self.gfu_save[0].Set(self.displacement, definedon = self.domain)

        for save in self.save_error:
            save.Save(solverdata, self)

        for save in self.save_solution:
            save.Save(solverdata, self)

        solverdata.mesh.UnsetDeformation()

    def get_error(self, solverdata, ex_sol, norm):

        err = compute_error(data=solverdata, gfu = self.displacement, u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err]
    
    def set_solution(self, value):

        self.displacement.Set(value, definedon=self.domain)

    def get_solution(self):

        return self.displacement

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Mean Curvature flow of Dziuk')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')