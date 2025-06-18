# Import necessary libraries
from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.ale import ale
from ngsolve.webgui import Draw
from cosmos.solvers.time_schemes import BDF1, BDF2, CN, Steady

class WeakCoupling(BasePDE):
    
    def __init__(self, tol = 1e-10, type = 'implicit'):

        super().__init__()
        self.tol = tol
        self.type = type
        
        self.PDEs = []

    def AddPDEs(self, *args):

        for pde in args:
            if not isinstance(pde, BasePDE):
                raise TypeError("The added PDE must inherit from BasePDE")
            self.PDEs.append(pde)

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        if len(self.PDEs)<2:
            raise Exception('Coupling class is meant for more than only 1 PDE!')

        for pde in self.PDEs:
            pde.Initialize(solverdata)

    def Solve(self, solverdata):

        dim = len(self.PDEs)
        errors = [1e10]*dim
        gfus_old = [None]*dim
        gfus_new = [None]*dim

        for i, pde in enumerate(self.PDEs):
            pde.Solve(solverdata)
            solverdata.ale.compute_new_def()
            gfus_old[i] = pde.gfu.vec.Copy()

        if self.type == 'implicit':
            while max(errors)>self.tol:
                for i, pde in enumerate(self.PDEs):
                    pde.SystemSolve(solverdata)
                    pde.PostProcess(solverdata)
                    solverdata.ale.compute_new_def()
                    gfus_new[i] = pde.gfu.vec.Copy()
                    errors[i] = Norm(gfus_new[i] - gfus_old[i])/Norm(gfus_old[i])
                    gfus_old[i] = gfus_new[i]
        elif self.type != 'explicit':
            raise Exception('Weak coupling type ', self.type, ' not implemented')

    def Update(self, solverdata):

        for pde in self.PDEs:
            pde.Update(solverdata)

    def get_error(self, data, ex_sol, norm):

        raise Exception('Coupling class doesn''t have get_error method, use the single PDE one' )
    
    def set_solution(self, value):

        raise Exception('Coupling class doesn''t have set_solution method, use the single PDE one' )

    def get_solution(self):

        raise Exception('Coupling class doesn''t have get_solution method, use the single PDE one' )
    
    def print_info(self):

        for pde in self.PDEs:
            pde.print_info()