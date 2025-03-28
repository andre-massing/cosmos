# %%
from ngsolve import *
from cosmos.pdes.pde_tools import SaveError, SaveSolution

class BasePDE():

    def __init__(self):

        self.name = None
        self.fes_order = 1
        self.dim = None
        self.fields = None
        self.fes = None
        self.gfu = None
        self.gfu_old = None
        self.gfu_save = None
        self.params = None
        self.save_error = []
        self.save_solution = []

    def Initialize(self, *args, **kwargs):
        
        raise NotImplementedError
    
    def GetLHS(self, *args, **kwargs):
        
        raise NotImplementedError
    
    def GetRHS(self, *args, **kwargs):
        
        raise NotImplementedError
    
    def GetMass(self, *args, **kwargs):
        
        raise NotImplementedError
    
    def Update(self, *args, **kwargs):
        
        raise NotImplementedError
    
    def SaveSol(self, save):

        if not isinstance(save, SaveSolution):
            raise TypeError("The added saving parameters must be a SaveSolution object")

        self.save_solution.append(save)

    def SaveErr(self, err):

        if not isinstance(err, SaveError):
            raise TypeError("The added saving parameters must be a SaveError object")
        
        self.save_error.append(err)
    
    def get_error(self, *args, **kwargs):

        raise NotImplementedError
    
    def get_solution(self, *args, **kwargs):
        
        raise NotImplementedError
    
    def set_solution(self, *args, **kwargs):
        
        raise NotImplementedError

    def print_info(self, *args, **kwargs):
        
        raise NotImplementedError

