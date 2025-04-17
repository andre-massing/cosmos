# %%
from ngsolve import *
from cosmos.pdes.pde_tools import SaveError, SaveSolution

class BasePDE():

    def __init__(self):

        self.name = None
        self.fes_order = 1
        self.nfields = None
        self.fes = None
        self.trial = None
        self.test = None
        self.gfu = None
        self.prev_gfu = []
        self.gfu_save = None
        self.params = None
        self.save_error = []
        self.save_solution = []
        self.nonlinear = False
        self.initialized = False

    def Initialize(self, data):
        
        raise NotImplementedError
    
    def GetLHS(self, data, trial, test, dX = None):
        
        raise NotImplementedError
    
    def GetRHS(self, data, test, dX = None):
        
        raise NotImplementedError
    
    def GetMass(self, data, trial, test, dX = None):
        
        raise NotImplementedError
    
    def PreProcess(self, data, dX = None):
        
        self.prev_gfu.append(self.gfu.vec.Copy())    
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)
    
    def PostProcess(self, data, dX = None):
        
        pass
    
    def Update(self, data):
        
        pass
    
    def SaveSol(self, save):

        if not isinstance(save, SaveSolution):
            raise TypeError("The added saving parameters must be a SaveSolution object")

        self.save_solution.append(save)

    def SaveErr(self, err):

        if not isinstance(err, SaveError):
            raise TypeError("The added saving parameters must be a SaveError object")
        
        self.save_error.append(err)
    
    def get_trial(self):

        if isinstance(self.trial, list):
            return self.trial
        else:
            return [self.trial]
        
    def get_test(self):

        if isinstance(self.test, list):
            return self.test
        else:
            return [self.test]
    
    def get_error(self, data, ex_sol, norm):

        raise NotImplementedError
    
    def set_solution(self, value):
        
        raise NotImplementedError
    
    def get_solution(self):
        
        raise NotImplementedError

    def print_info(self):
        
        raise NotImplementedError

