from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
from ngsolve.webgui import Draw

class ALE(BasePDE):

    def __init__(self):

        super().__init__()

        self.name = ['deformation', 'velocity']
        self.nfields = 2
        self.nonlinear = False
        self.nl = []
        self.rhs = []
        self.lhs = []
        self.mass = []

    def Initialize(self, data):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.dim = data.mesh.dim
        if data.mesh.ne == 0:
            self.domain = data.mesh.Boundaries('.*')
        else:
            self.domain = data.mesh.Materials('.*')

        V = VectorH1(data.mesh, order=self.fes_order,
                    definedon=self.domain)
        
        self.fes = CompressCompound(V*V)

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu = GridFunction(self.fes)
        self.deformation, self.velocity = self.gfu.components

        self.deformation0 = GridFunction(Compress(V))
        if data.mesh.dim == 2:
            self.deformation0.Set(CF((x, y)), definedon=self.domain)
        elif data.mesh.dim == 3:
            self.deformation0.Set(CF((x, y, z)), definedon=self.domain)
        self.gfu_save = self.gfu

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)
            
    def GetLHS(self, data, trial, test, ale):

        if data.mesh.ne == 0:
            lhs = trial[0]*test[0]*ds
            lhs += trial[1]*test[1]*ds
        else:
            lhs = trial[0]*test[0]*dx
            lhs += trial[1]*test[1]*dx

        if self.lhs:
            lhs_i = self.lhs[0]
            lhs = lhs_i['f'](data, trial, test, ale)
            for lhs_i in self.rhs[1:]:
                lhs += lhs_i['f'](data, trial, test, ale)   

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

        if self.mass:
            mass_i = self.mass[0]
            mass = mass_i['f'](data, trial, test, ale)
            for mass_i in self.mass[1:]:
                mass += mass_i['f'](data, trial, test, ale)
        else:
            mass = CF(0)*ds

        return mass
    
    def GetNL(self, data, trial, test, ale):

        nl_i = self.nl[0]
        nonlin = nl_i['f'](data, trial, test, ale)
        for nl_i in self.nl[1:]:
            nonlin += nl_i['f'](data, trial, test, ale)

        return nonlin
    
    def __add_forms__(self, kwargs):

        params = kwargs
        # Initialize the parameters
        accepted_keys = ['f']
        defaults = [None]  
        params_check(params, accepted_keys, defaults)

        return params
    
    def AddCoupling(self, **kwargs):

        self.nonlinear = True
        params = self.__add_forms__(kwargs)
        self.nl.append(params)

    def AddRHS(self, **kwargs):

        params = self.__add_forms__(kwargs)
        self.rhs.append(params)

    def AddLHS(self, **kwargs):

        params = self.__add_forms__(kwargs)
        self.lhs.append(params)

    def AddMass(self, **kwargs):

        params = self.__add_forms__(kwargs)
        self.mass.append(params)
    
    def PreProcess(self, data, ale):

        self.prev_gfu.append(self.gfu.vec.Copy())      
        if len(self.prev_gfu)>6:
            self.prev_gfu.pop(0)
    
    def PostProcess(self, data, ale):

        super().PostProcess(data, ale)
        
    def Update(self, data, ale):

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

    def get_error(self, data, ex_sol, norm):

        err0 = compute_error(data=data, gfu = self.deformation, u_ex=ex_sol[0],
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        err1 = compute_error(data=data, gfu = self.velocity, u_ex=ex_sol[1],
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        return [err0]
    
    def set_solution(self, value):

        self.dX.Set(value, definedon=self.domain)

    def get_solution(self):

        return self.gfu

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