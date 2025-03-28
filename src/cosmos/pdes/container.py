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

    def Initialize(self, mesh_data):

        if len(self.PDEs)<2:
            raise Exception('Container class is meant for more than only 1 PDE!')

        self.PDEs[0].Initialize(mesh_data)
        self.fes = self.PDEs[0].fes
        
        self.pntrs.append(self.PDEs[0].fields)
        i = self.PDEs[0].fields
        for pde in self.PDEs[1:]:

            i += pde.fields
            self.pntrs.append(i)

            pde.Initialize(mesh_data)
            self.fes = self.fes*pde.fes
            
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

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

    def GetLHS(self, mesh_data, trial, test, dX):

        if len(trial[self.pntrs[0]:self.pntrs[1]]) == 1:
            lhs = self.PDEs[0].GetLHS(mesh_data, trial[self.pntrs[0]:self.pntrs[1]][0],
                                    test[self.pntrs[0]:self.pntrs[1]][0], dX)
        else:
            lhs = self.PDEs[0].GetLHS(mesh_data, trial[self.pntrs[0]:self.pntrs[1]],
                                    test[self.pntrs[0]:self.pntrs[1]], dX)

        for i, pde in enumerate(self.PDEs[1:]):

            if len(trial[self.pntrs[i+1]:self.pntrs[i+2]]) == 1:
                lhs += pde.GetLHS(mesh_data, trial[self.pntrs[i+1]:self.pntrs[i+2]][0],
                                test[self.pntrs[i+1]:self.pntrs[i+2]][0], dX)
            else:
                lhs += pde.GetLHS(mesh_data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                                test[self.pntrs[i+1]:self.pntrs[i+2]], dX)
            
        return lhs
    
    def GetRHS(self, mesh_data, test, dX):

        if len(test[self.pntrs[0]:self.pntrs[1]]) == 1:
            rhs = self.PDEs[0].GetRHS(mesh_data, test[self.pntrs[0]:self.pntrs[1]][0], dX)
        else:
            rhs = self.PDEs[0].GetRHS(mesh_data, test[self.pntrs[0]:self.pntrs[1]], dX)

        for i, pde in enumerate(self.PDEs[1:]):

            if len(test[self.pntrs[i+1]:self.pntrs[i+2]]) == 1:
                rhs += pde.GetRHS(mesh_data, test[self.pntrs[i+1]:self.pntrs[i+2]][0], dX)
            else:
                rhs += pde.GetRHS(mesh_data, test[self.pntrs[i+1]:self.pntrs[i+2]], dX)
            
        return rhs
    
    def GetMass(self, mesh_data, trial, test, dX, gfu):

        if len(trial[self.pntrs[0]:self.pntrs[1]]) == 1:
            mass, mass_gfu = self.PDEs[0].GetMass(mesh_data, trial[self.pntrs[0]:self.pntrs[1]][0],
                                    test[self.pntrs[0]:self.pntrs[1]][0], dX,
                                    gfu.components[self.pntrs[0]:self.pntrs[1]][0])
        else:
            mass, mass_gfu = self.PDEs[0].GetMass(mesh_data, trial[self.pntrs[0]:self.pntrs[1]],
                                        test[self.pntrs[0]:self.pntrs[1]], dX,
                                        gfu.components[self.pntrs[0]:self.pntrs[1]])

        for i, pde in enumerate(self.PDEs[1:]):

            if len(trial[self.pntrs[i+1]:self.pntrs[i+2]]) == 1:
                mass_i, mass_gfu_i = pde.GetMass(mesh_data, trial[self.pntrs[i+1]:self.pntrs[i+2]][0],
                                test[self.pntrs[i+1]:self.pntrs[i+2]][0], dX,
                                gfu.components[self.pntrs[i+1]:self.pntrs[i+2]][0])
            else:
                mass_i, mass_gfu_i = pde.GetMass(mesh_data, trial[self.pntrs[i+1]:self.pntrs[i+2]],
                                test[self.pntrs[i+1]:self.pntrs[i+2]], dX,
                                gfu.components[self.pntrs[i+1]:self.pntrs[i+2]])
            
            mass += mass_i
            mass_gfu += mass_gfu_i
            
        return mass, mass_gfu
    
    def GetNL(self, mesh_data, trial, test, dX):

        nl_i = self.nl[0]
        trials_i = [trial[i] for i in nl_i['markers']]
        test_i = test[nl_i['marker0']]

        if nl_i['VorB'] == BND:
            if nl_i['grad']:
                nonlin = nl_i['f'](trials_i)*grad(test_i)*ds(definedon=nl_i['domain'],
                                                                deformation = dX)
            else:
                nonlin = nl_i['f'](trials_i)*test_i*ds(definedon=nl_i['domain'],
                                                        deformation = dX)
        else:
            if nl_i['grad']:
                nonlin = nl_i['f'](trials_i)*grad(test_i)*dx(definedon=nl_i['domain'],
                                                                deformation = dX)
            else:
                nonlin = nl_i['f'](trials_i)*test_i*dx(definedon=nl_i['domain'],
                                                        deformation = dX) 

        for nl_i in self.nl[1:]:

            trials_i = [trial[i] for i in nl_i['markers']]
            test_i = test[nl_i['marker0']]

            if nl_i['VorB'] == BND:
                if nl_i['grad']:
                    nonlin += nl_i['f'](trials_i)*grad(test_i)*ds(definedon=nl_i['domain'],
                                                                  deformation = dX)
                else:
                    nonlin += nl_i['f'](trials_i)*test_i*ds(definedon=nl_i['domain'],
                                                            deformation = dX)
            else:
                if nl_i['grad']:
                    nonlin += nl_i['f'](trials_i)*grad(test_i)*dx(definedon=nl_i['domain'],
                                                                  deformation = dX)
                else:
                    nonlin += nl_i['f'](trials_i)*test_i*dx(definedon=nl_i['domain'],
                                                            deformation = dX)
                    
        return nonlin
    
    def MP_and_BP(self, mesh_data, dX, gfu):

         for i, pde in enumerate(self.PDEs):
            if 'MP' in pde.params or 'BP' in pde.params:
                if len(gfu.components[self.pntrs[i]:self.pntrs[i+1]]) == 1:
                    pde.MP_and_BP(mesh_data, dX, gfu.components[self.pntrs[i]:self.pntrs[i+1]][0])
                else:
                    pde.MP_and_BP(mesh_data, dX, gfu.components[self.pntrs[i]:self.pntrs[i+1]])


    def Update(self, mesh_data):

        self.gfu_old.vec.data = self.gfu.vec.data

        for i, pde in enumerate(self.PDEs):
            for j, k in enumerate(range(self.pntrs[i], self.pntrs[i+1])):

                if pde.fields == 1:
                    pde.gfu.vec.data = self.gfu.components[k].vec.data
                else:
                    pde.gfu.components[j].vec.data = self.gfu.components[k].vec.data

            pde.Update(mesh_data)

    def get_error(self, mesh_data):

        ERR = []

        for pde in self.PDEs:

            ERR.append(pde.get_error(mesh_data))

        return ERR
    
    def print_info(self):

        for pde in self.PDEs:
            pde.print_info()