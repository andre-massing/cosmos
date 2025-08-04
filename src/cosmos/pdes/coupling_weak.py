# Import necessary libraries
from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
import time

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

        # if len(self.PDEs)<2:
        #     raise Exception('Coupling class is meant for more than only 1 PDE!')

        for pde in self.PDEs:
            pde.Initialize(solverdata)

    def Solve(self, solverdata):

        dim = len(self.PDEs)
        errors = [1e10]*dim

        for i, pde in enumerate(self.PDEs):
            pde.Solve(solverdata)
            pde.PostProcess(solverdata)
            solverdata.ale.compute_new_def()

        iter = 0
        max_iter = 30
        if self.type == 'implicit':
            while max(errors)>self.tol and iter<max_iter:

                iter += 1
                for i, pde in enumerate(self.PDEs):

                    vec1 = pde.gfu.vec.Copy()
                    t1 = time.time()
                    pde.SystemSolve(solverdata)
                    t2 = time.time()
                    pde.PostProcess(solverdata)
                    t3 = time.time()
                    solverdata.ale.compute_new_def()
                    t4 = time.time()
                    vec2 = pde.gfu.vec.Copy()
                    errors[i] = Norm(vec2 - vec1)/Norm(vec2)
                    t5 = time.time()

                    # print('Solving time:', t2-t1)
                    # print('Postprocess time:', t3-t2)
                    # print('Deformation update', t4-t3)
                    # print('Norm computation', t5-t4)

            if iter == max_iter:
                raise Exception('Reached '+ str(max_iter) + ' iterations without converging!')

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