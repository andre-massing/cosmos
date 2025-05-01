from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
from ngsolve.webgui import Draw

class Multiplier(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        accepted_keys = ['space', 'name']
        defaults = [None, 'multiplier']
        
        if kwargs:
            params_check(kwargs, accepted_keys, defaults)
            self.params = kwargs
        else:
            self.params = {}
            params_check(self.params, accepted_keys, defaults)
        self.nfields = 1
        self.nl = []
        self.rhs = []
        self.lhs = []

    def Initialize(self, data):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.name = [self.params['name']] 
        
        self.fes = self.params['space']

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)

        self.gfu_save = self.gfu

        # for save in self.save_error:
        #     save.Initialize(data, self)

        # for save in self.save_solution:
        #     save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test, ale):

        if self.lhs:
            lhs_i = self.lhs[0]
            lhs = lhs_i['f'](data, trial, test, ale)
            for lhs_i in self.rhs[1:]:
                lhs += lhs_i['f'](data, trial, test, ale)
        else:
            lhs = CF(0)*trial[0]*test[0]*ds

        return lhs 
        
    def GetRHS(self, data, test, ale):

        if self.rhs:
            rhs_i = self.rhs[0]
            rhs = rhs_i['f'](data, test, ale)
            for rhs_i in self.rhs[1:]:
                rhs += rhs_i['f'](data, test, ale)
        else:
            rhs = CF(0)*ds

        return rhs
    
    def GetMass(self, data, trial, test, ale):

        mass = CF(0)*ds

        return mass
    
    def GetNL(self, data, trial, test, ale):

        nl_i = self.nl[0]
        nonlin = nl_i['f'](data, trial, test, ale)
        for nl_i in self.nl[1:]:
            nonlin += nl_i['f'](data, trial, test, ale)

        return nonlin
    
    def AddCoupling(self, **kwargs):

        self.nonlinear = True

        params = kwargs
        # Initialize the parameters
        accepted_keys = ['f']
        defaults = [None]  
        params_check(params, accepted_keys, defaults)
        self.nl.append(params)

    def AddRHS(self, **kwargs):

        params = kwargs
        # Initialize the parameters
        accepted_keys = ['f']
        defaults = [None]  
        params_check(params, accepted_keys, defaults)
        self.rhs.append(params)

    def AddLHS(self, **kwargs):

        params = kwargs
        # Initialize the parameters
        accepted_keys = ['f']
        defaults = [None]  
        params_check(params, accepted_keys, defaults)
        self.lhs.append(params)
    
    def PreProcess(self, data, ale):

        pass
    
    def PostProcess(self, data, ale):

        pass
        
    def Update(self, data, ale):

        pass

        # for save in self.save_error:
        #     save.Save(data, self)

        # for save in self.save_solution:
        #     save.Save(data, self)

    # def get_error(self, data, ex_sol, norm):

    #     err0 = compute_error(data=data, gfu = self.dX, u_ex=ex_sol,
    #                         norm = norm, domain = self.domain, 
    #                         VorB = VOL)
        
    #     return [err0]
    
    # def set_solution(self, value):

    #     self.dX.Set(value, definedon=self.domain)

    # def get_solution(self):

    #     return self.gfu

    def print_info(self):

        print(60*'-')

        print('This is a ALE auxiliary pde')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')