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

        A = BilinearForm(self.fes)
        F = LinearForm(self.fes)

        if isinstance(self.time_scheme, Steady):

            raise Exception('Steady scheme not implemented for this Cahn-Hilliard solver')
        
        else:

            M = BilinearForm(self.fes)
            M += self.GetMass(solverdata, self.get_trial(), self.get_test())
            M.Assemble()

            if isinstance(self.time_scheme, BDF2) and len(self.prev_gfu)==1 or \
                isinstance(self.time_scheme, BDF1):

                A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
                A += self.GetMass(solverdata, self.get_trial(), self.get_test())
                F += self.GetRHS(solverdata, self.get_test())
                F.Assemble()
                res = F.vec
                res += M.mat*self.prev_gfu[-1]

            elif isinstance(self.time_scheme, BDF2) and len(self.prev_gfu)>1:

                A += self.GetLHS(solverdata, self.get_trial(), self.get_test())
                A += 1.5*self.GetMass(solverdata, self.get_trial(), self.get_test())
                F += self.GetRHS(solverdata, self.get_test())
                F.Assemble()
                res = F.vec
                res += 2*M.mat*self.prev_gfu[-1]
                res += -0.5*M.mat*self.prev_gfu[-2]

            else:

                raise Exception('Time scheme not implemented for this Cahn-Hilliard solver')

        A.Assemble()
        self.gfu.vec.data = A.mat.Inverse(freedofs = self.fes.FreeDofs())*res

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