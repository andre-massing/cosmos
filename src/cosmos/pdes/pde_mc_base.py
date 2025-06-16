from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.ale import ale
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady
from cosmos.pdes.pde_tools import compute_error
from ngsolve.webgui import Draw
import numpy as np

class BaseMC(BasePDE):

    def Solve(self, solverdata):

        self.PreProcess(solverdata)
        self.SystemSolve(solverdata)
        self.PostProcess(solverdata)

    def SystemSolve(self, solverdata):

        import time

        if isinstance(self.time_scheme, CN):
            raise Exception('Crack-Nicholson scheme not implemented for this solver')
        
        if len(solverdata.prev_def) == 1 or isinstance(self.time_scheme, BDF1):

            t1 = time.time()

            ale_curr = ale(solverdata)
            ale_curr.deformation.vec.data = solverdata.ale.deformation.vec.data
            ale_curr.velocity.vec.data = solverdata.ale.velocity.vec.data

            solverdata.ale.deformation.vec.data = solverdata.prev_def[-1].data
            solverdata.ale.velocity.vec.data = solverdata.prev_vel[-1].data

            t2 = time.time()

            M = BilinearForm(self.fes)
            M += self.GetMass(solverdata, self.get_trial(), self.get_test())
            M.Assemble()
            resMold = M.mat*self.prev_gfu[-1]

            solverdata.t.Set(solverdata.t.Get() + solverdata.dt.Get())

            t3 = time.time()

            A = BilinearForm(self.fes)
            F = LinearForm(self.fes)
            F += self.GetRHS(solverdata, self.get_test())
            F.Assemble()

            t4 = time.time()

            A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
            res = F.vec
            res += resMold
            A += self.GetMass(solverdata, self.get_trial(), self.get_test())
            
        elif isinstance(self.time_scheme, BDF2):

            ale_curr = ale(solverdata)
            ale_curr.deformation.vec.data = solverdata.ale.deformation.vec.data
            ale_curr.velocity.vec.data = solverdata.ale.velocity.vec.data

            solverdata.ale.deformation.vec.data = 2*solverdata.prev_def[-1].data - 1*solverdata.prev_def[-2].data
            solverdata.ale.velocity.vec.data = 2*solverdata.prev_vel[-1].data - 1*solverdata.prev_vel[-2].data

            M = BilinearForm(self.fes)
            M += self.GetMass(solverdata, self.get_trial(), self.get_test())
            M.Assemble()
            resMoldold = M.mat*self.prev_gfu[-2]
            resMold = M.mat*self.prev_gfu[-1]

            solverdata.t.Set(solverdata.t.Get() + solverdata.dt.Get())

            A = BilinearForm(self.fes, condense = True)
            F = LinearForm(self.fes)
            F += self.GetRHS(solverdata, self.get_test())
            F.Assemble()

            A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
            res = F.vec
            res += -0.5*resMoldold
            res += 2*resMold
            A += 1.5*self.GetMass(solverdata, self.get_trial(), self.get_test())

        elif isinstance(self.time_scheme, Steady):
            raise Exception('Steady solver not yet implemented for this solver')

        else:
            raise Exception('Time integration not known for this solver')
        
        A.Assemble()

        t5 = time.time()

        self.gfu.vec.data = A.mat.Inverse(freedofs = self.fes.FreeDofs())*res

        t6 = time.time()

        solverdata.t.Set(solverdata.t.Get() - solverdata.dt.Get())
        solverdata.ale.deformation.vec.data = ale_curr.deformation.vec.data
        solverdata.ale.velocity.vec.data = ale_curr.velocity.vec.data