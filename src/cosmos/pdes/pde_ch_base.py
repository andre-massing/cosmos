from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.ale import ale
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady
from cosmos.pdes.pde_tools import compute_error
from ngsolve.webgui import Draw
import numpy as np

def SimpleNewtonSolve(data, a, f, gfu, tol=1e-7, maxits=20):
    res = gfu.vec.CreateVector()
    du = gfu.vec.CreateVector()
    fes = gfu.space
    for it in range(maxits):
        a.Apply(gfu.vec, res)
        res += -1*f
        a.AssembleLinearization(gfu.vec)
        du.data = a.mat.Inverse(fes.FreeDofs()) * res
        gfu.vec.data -= du
        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        if stopcritval < tol:
            break

    if it == maxits-1:
        print(stopcritval)
        raise Exception('Newton solver has not converged!')

class BaseCH(BasePDE):

    def Solve(self, solverdata):

        self.PreProcess(solverdata)
        self.SystemSolve(solverdata)
        self.PostProcess(solverdata)

    def SystemSolve(self, solverdata):

        if isinstance(self.time_scheme, CN):
            raise Exception('Crack-Nicholson scheme not implemented for this solver')
        
        if len(solverdata.prev_def) == 1 or isinstance(self.time_scheme, BDF1):

            ale_curr = ale(solverdata)
            ale_curr.deformation.vec.data = solverdata.ale.deformation.vec.data
            ale_curr.velocity.vec.data = solverdata.ale.velocity.vec.data
            ale_curr.material_velocity.vec.data = solverdata.ale.material_velocity.vec.data

            solverdata.ale.deformation.vec.data = solverdata.prev_def[-1].data
            solverdata.ale.velocity.vec.data = solverdata.prev_vel[-1].data
            solverdata.ale.material_velocity.vec.data = solverdata.prev_mat_vel[-1].data

            M = BilinearForm(self.fes)
            M += self.GetMass(solverdata, self.get_trial(), self.get_test())
            M.Assemble()
            resMold = M.mat*self.prev_gfu[-1]

            solverdata.t.Set(solverdata.t.Get() + solverdata.dt.Get())

            A = BilinearForm(self.fes)
            F = LinearForm(self.fes)
            F += self.GetRHS(solverdata, self.get_test())
            F.Assemble()

            A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
            A += self.GetNL(solverdata, self.get_trial(), self.get_test())
            res = F.vec
            res += resMold
            A += self.GetMass(solverdata, self.get_trial(), self.get_test())         

        elif isinstance(self.time_scheme, BDF2):

            ale_curr = ale(solverdata)
            ale_curr.deformation.vec.data = solverdata.ale.deformation.vec.data
            ale_curr.velocity.vec.data = solverdata.ale.velocity.vec.data
            ale_curr.material_velocity.vec.data = solverdata.ale.material_velocity.vec.data

            solverdata.ale.deformation.vec.data = 2*solverdata.prev_def[-1].data - 1*solverdata.prev_def[-2].data
            solverdata.ale.velocity.vec.data = 2*solverdata.prev_vel[-1].data - 1*solverdata.prev_vel[-2].data
            solverdata.ale.material_velocity.vec.data = 2*solverdata.prev_mat_vel[-1].data - 1*solverdata.prev_mat_vel[-2].data

            M = BilinearForm(self.fes)
            M += self.GetMass(solverdata, self.get_trial(), self.get_test())
            M.Assemble()
            resMoldold = M.mat*self.prev_gfu[-2]
            resMold = M.mat*self.prev_gfu[-1]

            solverdata.t.Set(solverdata.t.Get() + solverdata.dt.Get())

            A = BilinearForm(self.fes)
            F = LinearForm(self.fes)
            F += self.GetRHS(solverdata, self.get_test())
            F.Assemble()

            A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
            A += self.GetNL(solverdata, self.get_trial(), self.get_test())
            res = F.vec
            res += -0.5*resMoldold
            res += 2*resMold
            A += 1.5*self.GetMass(solverdata, self.get_trial(), self.get_test())

        elif isinstance(self.time_scheme, Steady):
            raise Exception('Steady solver not yet implemented for this solver')

        else:
            raise Exception('Time integration not known for this solver')
        
        SimpleNewtonSolve(solverdata, A, res, self.gfu)

        solverdata.t.Set(solverdata.t.Get() - solverdata.dt.Get())
        solverdata.ale.deformation.vec.data = ale_curr.deformation.vec.data
        solverdata.ale.velocity.vec.data = ale_curr.velocity.vec.data
        solverdata.ale.material_velocity.vec.data = ale_curr.velocity.vec.data

    def Update(self, solverdata):

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)

        self.gfu_save[0].Set(self.phase, definedon = self.domain)
        self.gfu_save[1].Set(self.potential, definedon = self.domain)

        for save in self.save_error:
            save.Save(solverdata, self)

        for save in self.save_solution:
            save.Save(solverdata, self)

        solverdata.mesh.UnsetDeformation()

    def set_solution(self, value):

        self.phase.Set(value[0], definedon=self.domain)
        self.potential.Set(value[1], definedon=self.domain)

    def get_solution(self):

        return self.phase, self.potential