# %%
from ngsolve import *
from cosmos.solvers.tools import params_check

class BasePDE():

    def __init__(self, fes_order=1):
        
        self.fes_order = fes_order
        self.dim = 2
        self.fields = None
        self.fes = None
        self.gfu = None
        self.gfu_old = None
        self.params = None
        self.error_params = None
        self.save_params = None

    def Initialize(self):
        
        raise NotImplementedError
    
    def GetLHS(self):
        
        raise NotImplementedError
    
    def GetRHS(self):
        
        raise NotImplementedError
    
    def GetMass(self):
        
        raise NotImplementedError
    
    def Update(self):
        
        raise NotImplementedError
    
    def get_error(self):

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
        params_check(params, accepted_keys, defaults)

        self.error_params = params

    def SaveSolution(self, **kwargs):

        params = kwargs
        accepted_keys = ['filename', 'subdivision', 'n_samples', 'folderpath']
        defaults = ['sol', 1, 100, './']  
        params_check(params, accepted_keys, defaults)

        self.save_params = params

    def print_info(self):
        
        raise NotImplementedError

