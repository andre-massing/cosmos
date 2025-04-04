# Import necessary libraries
from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from collections import Counter
from tqdm import tqdm
from ngsolve.solvers import *
import numpy as np

class Container(BasePDE):
    
    def __init__(self, **kwargs):

        self.params = kwargs

        # Initialize the parameters
        accepted_keys = ['MP', 'BP']
        defaults = [False, [-np.inf, np.inf]]
        params_check(self.params, accepted_keys, defaults)
        
        self.PDEs = []
        self.pntrs = [0]
        self.nl = []
        self.fes = None
        self.gfu = None
        self.gfu_old = None

    def AddPDEs(self, *args):

        for pde in args:
            if not isinstance(pde, BasePDE):
                raise TypeError("The added PDE must inherit from BasePDE")
            self.PDEs.append(pde)

    def AddNonlinearity(self, **kwargs):

        params = kwargs
        # Initialize the parameters
        accepted_keys = ['marker0', 'markers', 'f', 'VorB', 'domain', 'grad']
        defaults = [0, 0, {}, VOL, '.*', False]  
        params_check(params, accepted_keys, defaults)
        self.nl.append(params)

    def Initialize(self, data):

        if len(self.PDEs)<2:
            raise Exception('Container class is meant for more than only 1 PDE!')

        self.PDEs[0].Initialize(data)
        self.fes = self.PDEs[0].fes
        
        self.pntrs.append(self.PDEs[0].fields)
        i = self.PDEs[0].fields
        for pde in self.PDEs[1:]:

            i += pde.fields
            self.pntrs.append(i)

            pde.Initialize(data)
            self.fes = self.fes*pde.fes
            
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        self.gfu_comp = self.gfu.components
        self.gfu_old_comp = self.gfu_old.components

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        for i, pde in enumerate(self.PDEs):
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):
                if pde.fields == 1:
                    self.gfu.components[k].vec.data = pde.gfu.vec.data
                    self.gfu_old.components[k].vec.data = pde.gfu_old.vec.data
                else:
                    self.gfu.components[k].vec.data = pde.gfu.components[j].vec.data
                    self.gfu_old.components[k].vec.data = pde.gfu_old.components[j].vec.data

    def GetLHS(self, data, trial, test):

        if len(trial[self.pntrs[0]:self.pntrs[1]]) == 1:
            lhs = self.PDEs[0].GetLHS(data, trial[self.pntrs[0]:self.pntrs[1]][0],
                                    test[self.pntrs[0]:self.pntrs[1]][0])
        else:
            lhs = self.PDEs[0].GetLHS(data, trial[self.pntrs[0]:self.pntrs[1]],
                                    test[self.pntrs[0]:self.pntrs[1]])

        for i, pde in enumerate(self.PDEs[1:]):

            if len(trial[self.pntrs[i+1]:self.pntrs[i+2]]) == 1:
                lhs += pde.GetLHS(data, trial[self.pntrs[i+1]:self.pntrs[i+2]][0],
                                test[self.pntrs[i+1]:self.pntrs[i+2]][0])
            else:
                lhs += pde.GetLHS(data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                                test[self.pntrs[i+1]:self.pntrs[i+2]])
            
        return lhs
    
    def GetRHS(self, data, test):

        if len(test[self.pntrs[0]:self.pntrs[1]]) == 1:
            rhs = self.PDEs[0].GetRHS(data, test[self.pntrs[0]:self.pntrs[1]][0])
        else:
            rhs = self.PDEs[0].GetRHS(data, test[self.pntrs[0]:self.pntrs[1]])

        for i, pde in enumerate(self.PDEs[1:]):

            if len(test[self.pntrs[i+1]:self.pntrs[i+2]]) == 1:
                rhs += pde.GetRHS(data, test[self.pntrs[i+1]:self.pntrs[i+2]][0])
            else:
                rhs += pde.GetRHS(data, test[self.pntrs[i+1]:self.pntrs[i+2]])
            
        return rhs
    
    def GetMass(self, data, trial, test):

        if len(trial[self.pntrs[0]:self.pntrs[1]]) == 1:
            mass = self.PDEs[0].GetMass(data, trial[self.pntrs[0]:self.pntrs[1]][0],
                                    test[self.pntrs[0]:self.pntrs[1]][0])
        else:
            mass = self.PDEs[0].GetMass(data, trial[self.pntrs[0]:self.pntrs[1]],
                                        test[self.pntrs[0]:self.pntrs[1]])

        for i, pde in enumerate(self.PDEs[1:]):

            if len(trial[self.pntrs[i+1]:self.pntrs[i+2]]) == 1:
                mass_i = pde.GetMass(data, trial[self.pntrs[i+1]:self.pntrs[i+2]][0],
                                test[self.pntrs[i+1]:self.pntrs[i+2]][0])
            else:
                mass_i  = pde.GetMass(data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                                test[self.pntrs[i+1]:self.pntrs[i+2]])
            
            mass += mass_i
            
        return mass
    
    def GetMassOld(self, data, trial, test):

        if len(trial[self.pntrs[0]:self.pntrs[1]]) == 1:
            mass = self.PDEs[0].GetMassOld(data, trial[self.pntrs[0]:self.pntrs[1]][0],
                                    test[self.pntrs[0]:self.pntrs[1]][0])
        else:
            mass = self.PDEs[0].GetMassOld(data, trial[self.pntrs[0]:self.pntrs[1]],
                                        test[self.pntrs[0]:self.pntrs[1]])

        for i, pde in enumerate(self.PDEs[1:]):

            if len(trial[self.pntrs[i+1]:self.pntrs[i+2]]) == 1:
                mass_i = pde.GetMassOld(data, trial[self.pntrs[i+1]:self.pntrs[i+2]][0],
                                test[self.pntrs[i+1]:self.pntrs[i+2]][0])
            else:
                mass_i  = pde.GetMassOld(data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                                test[self.pntrs[i+1]:self.pntrs[i+2]])
            
            mass += mass_i
            
        return mass
    
    def GetNL(self, data, trial, test):

        nl_i = self.nl[0]
        trials_i = [trial[i] for i in nl_i['markers']]
        test_i = test[nl_i['marker0']]

        if nl_i['VorB'] == BND:
            if nl_i['grad']:
                nonlin = nl_i['f'](trials_i)*grad(test_i)*ds(definedon=nl_i['domain'])
            else:
                nonlin = nl_i['f'](trials_i)*test_i*ds(definedon=nl_i['domain'])
        else:
            if nl_i['grad']:
                nonlin = nl_i['f'](trials_i)*grad(test_i)*dx(definedon=nl_i['domain'])
            else:
                nonlin = nl_i['f'](trials_i)*test_i*dx(definedon=nl_i['domain']) 

        for nl_i in self.nl[1:]:

            trials_i = [trial[i] for i in nl_i['markers']]
            test_i = test[nl_i['marker0']]

            if nl_i['VorB'] == BND:
                if nl_i['grad']:
                    nonlin += nl_i['f'](trials_i)*grad(test_i)*ds(definedon=nl_i['domain'])
                else:
                    nonlin += nl_i['f'](trials_i)*test_i*ds(definedon=nl_i['domain'])
            else:
                if nl_i['grad']:
                    nonlin += nl_i['f'](trials_i)*grad(test_i)*dx(definedon=nl_i['domain'])
                else:
                    nonlin += nl_i['f'](trials_i)*test_i*dx(definedon=nl_i['domain'])
                    
        return nonlin
    
    def PreProcess(self, data):

        for pde in self.PDEs:

            pde.PreProcess(data)
    
    def PostProcess(self, data):

        for pde in self.PDEs:

            pde.PostProcess(data)

    def Update(self, data):

        self.gfu_old.vec.data = self.gfu.vec.data

        for i, pde in enumerate(self.PDEs):
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):

                if pde.fields == 1:
                    pde.gfu.vec.data = self.gfu.components[k].vec.data
                else:
                    pde.gfu.components[j].vec.data = self.gfu.components[k].vec.data

            pde.Update(data)

    def get_error(self, data, ex_sol, norm):

        raise Exception('Container class doesn''t have get_error method, use the single PDE one' )
    
    def set_solution(self, value):

        raise Exception('Container class doesn''t have set_solution method, use the single PDE one' )

    def get_solution(self):

        raise Exception('Container class doesn''t have get_solution method, use the single PDE one' )
    
    def print_info(self):

        for pde in self.PDEs:
            pde.print_info()