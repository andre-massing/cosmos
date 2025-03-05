# %%
from ngsolve import *
from cosmos.solvers.base import Base

class BaseSolver(Base):

    def __init__(self, fes_order=1, **kwargs):
        
        self.fes_order = fes_order
        self.params = kwargs
        self.sol = []
        self.problem = None

    def initialize(self):
        
        raise NotImplementedError
    
    def solve_step(self):
        
        raise NotImplementedError
    
    def update(self):
        
        raise NotImplementedError
    
    def get_solution(self):
        
        raise NotImplementedError
    
    def set_solution(self, **kwargs):
        
        raise NotImplementedError

    def draw_solution(self, **kwargs):
        
        raise NotImplementedError
    
    def ComputeError(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['ex_sol', 'norm', 'filename', 'folderpath']
        defaults = [None, 'L2', 'results.csv', '.']  
        self.params_check(params, accepted_keys, defaults)

        self.error_params = params

    def SaveSolution(self, **kwargs):

        params = kwargs
        accepted_keys = ['filename', 'subdivision', 'n_steps', 'folderpath']
        defaults = ['sol', 1, 100, './']  
        self.params_check(params, accepted_keys, defaults)

        self.save_params = params

    def print_info(self):
        
        raise NotImplementedError

