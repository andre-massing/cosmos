# Import necessary libraries
from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.ale import ALE
from cosmos.utils.tools import params_check
from collections import Counter
from tqdm import tqdm
import numpy as np

class StrongCoupling(BasePDE):
    
    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        # Initialize the parameters
        accepted_keys = ['MP', 'BP']
        defaults = [False, None]
        params_check(self.params, accepted_keys, defaults)
        
        self.PDEs = []
        self.pntrs = [0]
        self.cpl = []

    def AddPDEs(self, *args):

        for pde in args:
            if not isinstance(pde, BasePDE):
                raise TypeError("The added PDE must inherit from BasePDE")
            self.PDEs.append(pde)

    def AddCoupling(self, **kwargs):

        self.nonlinear = True

        params = kwargs
        # Initialize the parameters
        accepted_keys = ['f']
        defaults = [None]  
        params_check(params, accepted_keys, defaults)
        self.cpl.append(params)

    def Initialize(self, data):

        if self.initialized:
            return
        else:
            self.initialized = True

        if len(self.PDEs)<2:
            raise Exception('Coupling class is meant for more than only 1 PDE!')

        self.PDEs[0].Initialize(data)
        self.fes = self.PDEs[0].fes
        
        self.pntrs.append(self.PDEs[0].nfields)
        i = self.PDEs[0].nfields
        for pde in self.PDEs[1:]:

            pde.Initialize(data)
            i += pde.nfields
            self.pntrs.append(i)
            self.fes = self.fes*pde.fes
            
        self.gfu = GridFunction(self.fes)

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

    def GetLHS(self, data, trial, test, dX = None):

        lhs = self.PDEs[0].GetLHS(data, trial[self.pntrs[0]:self.pntrs[1]],
                                test[self.pntrs[0]:self.pntrs[1]], dX = dX)
        for i, pde in enumerate(self.PDEs[1:]):
            lhs += pde.GetLHS(data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                            test[self.pntrs[i+1]:self.pntrs[i+2]], dX = dX)       
            
        return lhs
    
    def GetRHS(self, data, test, dX = None):

        rhs = self.PDEs[0].GetRHS(data, test[self.pntrs[0]:self.pntrs[1]], dX = dX)
        for i, pde in enumerate(self.PDEs[1:]):
            rhs += pde.GetRHS(data, test[self.pntrs[i+1]:self.pntrs[i+2]], dX = dX)
            
        return rhs
    
    def GetNL(self, data, trial, test, dX = None):

        cpl = self.cpl[0]['f'](data, trial, test, dX)
        for cpl_i in self.cpl[1:]:
            cpl += cpl_i['f'](data, trial, test, dX)

        return cpl
    
    def GetMass(self, data, trial, test, dX = None):

        mass = self.PDEs[0].GetMass(data, trial[self.pntrs[0]:self.pntrs[1]],
                                    test[self.pntrs[0]:self.pntrs[1]], dX = dX)

        for i, pde in enumerate(self.PDEs[1:]):
            mass_i  = pde.GetMass(data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                            test[self.pntrs[i+1]:self.pntrs[i+2]], dX = dX)
            mass += mass_i
            
        return mass
    
    def PreProcess(self, data, dX = None):

        for i, pde in enumerate(self.PDEs):
            pde.PreProcess(data, dX)
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):
                if pde.nfields == 1:
                    self.gfu.components[k].vec.data = pde.gfu.vec.data
                else:
                    self.gfu.components[k].vec.data = pde.gfu.components[j].vec.data

        super().PreProcess(data)
    
    def PostProcess(self, data, dX = None):

        super().PostProcess(data)

        for i, pde in enumerate(self.PDEs):
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):
                if pde.nfields == 1:
                    pde.gfu.vec.data = self.gfu.components[k].vec.data
                else:
                    pde.gfu.components[j].vec.data = self.gfu.components[k].vec.data
            pde.PostProcess(data, dX)

    def Update(self, data):

        for pde in self.PDEs:
            pde.Update(data)

    def get_error(self, data, ex_sol, norm):

        raise Exception('Coupling class doesn''t have get_error method, use the single PDE one' )
    
    def set_solution(self, value):

        raise Exception('Coupling class doesn''t have set_solution method, use the single PDE one' )

    def get_solution(self):

        raise Exception('Coupling class doesn''t have get_solution method, use the single PDE one' )
    
    def print_info(self):

        for pde in self.PDEs:
            pde.print_info()