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
        accepted_keys = ['ale', 'MP', 'BP']
        defaults = [False, False, None]
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
        
        if self.params['ale']:
            self.aux_ale = auxALE()
            self.ale = ALE()
            self.PDEs.append(self.ale)

        self.PDEs[0].Initialize(data)
        self.fes = self.PDEs[0].fes
        if self.PDEs[0].nonlinear:
            self.nonlinear = True
        
        self.pntrs.append(self.PDEs[0].nfields)
        i = self.PDEs[0].nfields
        for pde in self.PDEs[1:]:
            i += pde.nfields
            self.pntrs.append(i)
            pde.Initialize(data)
            self.fes = self.fes*pde.fes
            if pde.nonlinear:
                self.nonlinear = True
            
        self.gfu = GridFunction(self.fes)

        if self.params['ale']:
            self.aux_ale.deformation = self.gfu.components[-2]
            self.aux_ale.velocity = self.gfu.components[-1]

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

    def GetLHS(self, data, trial, test, ale):

        if self.params['ale']:
            ale_aux = self.aux_ale
        else:
            ale_aux = ale

        lhs = self.PDEs[0].GetLHS(data, trial[self.pntrs[0]:self.pntrs[1]],
                                test[self.pntrs[0]:self.pntrs[1]], ale_aux)
        for i, pde in enumerate(self.PDEs[1:]):
            lhs += pde.GetLHS(data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                            test[self.pntrs[i+1]:self.pntrs[i+2]], ale_aux)       
            
        return lhs
    
    def GetRHS(self, data, test, ale):

        if self.params['ale']:
            ale_aux = self.aux_ale
        else:
            ale_aux = ale

        rhs = self.PDEs[0].GetRHS(data, test[self.pntrs[0]:self.pntrs[1]], ale_aux)
        for i, pde in enumerate(self.PDEs[1:]):
            rhs += pde.GetRHS(data, test[self.pntrs[i+1]:self.pntrs[i+2]], ale_aux)
            
        return rhs
    
    def GetNL(self, data, trial, test, ale):

        if self.params['ale']:
            ale_aux = self.aux_ale
        else:
            ale_aux = ale

        if self.cpl:
            cpl = self.cpl[0]['f'](data, trial, test, ale_aux)
            for cpl_i in self.cpl[1:]:
                cpl += cpl_i['f'](data, trial, test, ale_aux)
        else:
            cpl = CF(0)*ds
        for i, pde in enumerate(self.PDEs):
            if pde.nonlinear:
                cpl += pde.GetNL(data, trial[self.pntrs[i]:self.pntrs[i+1]],
                            test[self.pntrs[i]:self.pntrs[i+1]], ale_aux)

        return cpl
    
    def GetMass(self, data, trial, test, ale):

        if self.params['ale']:
            ale_aux = self.aux_ale
        else:
            ale_aux = ale

        mass = self.PDEs[0].GetMass(data, trial[self.pntrs[0]:self.pntrs[1]],
                                    test[self.pntrs[0]:self.pntrs[1]], ale_aux)

        for i, pde in enumerate(self.PDEs[1:]):
            mass_i  = pde.GetMass(data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                            test[self.pntrs[i+1]:self.pntrs[i+2]], ale_aux)
            mass += mass_i
            
        return mass
    
    def PreProcess(self, data, ale):

        if self.params['ale']:
            ale_aux = self.aux_ale
        else:
            ale_aux = ale

        for i, pde in enumerate(self.PDEs):
            pde.PreProcess(data, ale_aux)
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):
                if pde.nfields == 1:
                    self.gfu.components[k].vec.data = pde.gfu.vec.data
                else:
                    self.gfu.components[k].vec.data = pde.gfu.components[j].vec.data

        super().PreProcess(data, ale_aux)
    
    def PostProcess(self, data, ale):

        if self.params['ale']:
            ale_aux = self.aux_ale
        else:
            ale_aux = ale

        super().PostProcess(data, ale_aux)

        for i, pde in enumerate(self.PDEs):
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):
                if pde.nfields == 1:
                    pde.gfu.vec.data = self.gfu.components[k].vec.data
                else:
                    pde.gfu.components[j].vec.data = self.gfu.components[k].vec.data
            pde.PostProcess(data, ale_aux)

    def Update(self, data, ale):

        if self.params['ale']:
            ale_aux = self.aux_ale
        else:
            ale_aux = ale

        for pde in self.PDEs:
            pde.Update(data, ale_aux)

    def get_error(self, data, ex_sol, norm):

        raise Exception('Coupling class doesn''t have get_error method, use the single PDE one' )
    
    def set_solution(self, value):

        raise Exception('Coupling class doesn''t have set_solution method, use the single PDE one' )

    def get_solution(self):

        raise Exception('Coupling class doesn''t have get_solution method, use the single PDE one' )
    
    def print_info(self):

        for pde in self.PDEs:
            pde.print_info()


class auxALE():

    def __init__(self):

        self.deformation = None
        self.velocity = None