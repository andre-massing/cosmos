from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.pdes.ale import ALE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error, MandBP
import numpy as np
import scipy.sparse as sp
from ngsolve.webgui import Draw

class VolADR(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        self.nfields = 1

        # Initialize the parameters
        accepted_keys = ['b', 'c', 'd', 'u0', 'rhs',
                         'neu_d', 'neu_b', 'dir_d', 'dir_b', 'Fneu_b',
                         'domain', 'name', 'periodic',
                         'MP', 'BP']
        defaults = [None, None, None, None, None, 
                    {}, {}, {}, {}, {},
                    '.*', "volume_adr", False,
                    False, None]

        if kwargs:
            params_check(kwargs, accepted_keys, defaults)
            self.params = kwargs
            for i, (key, value) in enumerate(self.params.items()):
                if i<5 and isinstance(self.params[key], (int, float)):
                    self.params[key] = CF(self.params[key])
        else:
            self.params = {}
            params_check(self.params, accepted_keys, defaults)

    def Initialize(self, data):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = data.mesh.Materials(self.params['domain'])
        self.name = [self.params['name']]

        if data.mesh.ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + self.name + ' cannot be initialized')
        
        if self.params['periodic']:
            self.fes = Compress(Periodic(H1(data.mesh, order = self.fes_order, 
                                            definedon = self.params['domain'])))
        else:
            self.fes = Compress(H1(data.mesh, order = self.fes_order, 
                                   definedon = self.params['domain']))
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)

        if self.params['u0']:
            self.gfu.Set(self.params['u0'])

        self.gfu_save = [self.gfu]

        if self.params['MP']:
            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            elif data.mesh.dim == 3:
                raise Exception('Note implemented yet!')
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            A = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            A += u*v*dx_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu0_vec = self.gfu.vec.Copy().FV().NumPy()
            self.mass0 = np.sum(weights*gfu0_vec)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)

    def GetLHS(self, data, trial, test, ale):

        n = specialcf.normal(data.mesh.dim)
        h = specialcf.mesh_size

        if self.params['c']:
            lhs = self.params['c']*trial[0]*test[0]*dx(deformation = ale.deformation)
        else:
            lhs =  CF(0)*trial[0]*test[0]*dx(deformation = ale.deformation)

        if self.params['d']:
            lhs += self.params['d']*grad(trial[0])*grad(test[0])*dx(deformation = ale.deformation)
                    
            if self.params['dir_d']:
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for key, value in self.params['dir_d'].items():
                    lhs += - self.params['d']*InnerProduct(n, grad(trial[0]))*test[0]*ds(definedon = key, skeleton=True, deformation = ale.deformation) \
                        - self.params['d']*InnerProduct(n, grad(test[0]))*trial[0]*ds(definedon = key, skeleton=True, deformation = ale.deformation)\
                        + self.params['d']*alpha/h*trial[0]*test[0]*ds(definedon = key, skeleton = True, deformation = ale.deformation)\
            
        if self.params['b']:
            lhs += -self.params['b']*grad(test[0]) * trial[0]*dx(deformation = ale.deformation)

            if self.params['neu_b']:
                for key, value in self.params['neu_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = key, deformation = ale.deformation)

            if self.params['dir_b']:
                for key, value in self.params['dir_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = key, deformation = ale.deformation)
                    
            if self.params['Fneu_b']:
                for key, value in self.params['Fneu_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = data.mesh.Boundaries('.*') - data.mesh.Boundaries(key), deformation = ale.deformation)
            
        return lhs

    def GetRHS(self, data, test, ale):

        n = specialcf.normal(data.mesh.dim)
        h = specialcf.mesh_size

        if self.params['rhs']:
            rhs = self.params['rhs']*test[0]*dx(deformation = ale.deformation)
        else:
            rhs = CF(0)*test[0]*dx(deformation = ale.deformation)
        
        if self.params['dir_d']:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            for key, value in self.params['dir_d'].items():
                rhs += self.params['d']*alpha/h*value*test[0]*ds(definedon = key, skeleton = True, deformation = ale.deformation)\
                    - self.params['d']*InnerProduct(n, grad(test[0]))*value*ds(definedon = key, skeleton=True, deformation = ale.deformation)
        if self.params['neu_d']:
            for key, value in self.params['neu_d'].items():
                rhs += -value*n*test[0]*ds(definedon = key, deformation = ale.deformation)

        if self.params['dir_b']:
            for key, value in self.params['dir_b'].items():
                rhs += -IfPos(self.params['b']*n, CF(0), self.params['b']*n*value)*test[0]*ds(definedon = key, deformation = ale.deformation)
        if self.params['neu_b']:
            for key, value in self.params['neu_b'].items():
                rhs += -IfPos(self.params['b']*n, CF(0), value*n)*test[0]*ds(definedon = key, deformation = ale.deformation)

        if self.params['Fneu_b']:
            for key, value in self.params['Fneu_b'].items():
                rhs += -value*n*test[0]*ds(definedon = data.mesh.Boundaries(key), deformation = ale.deformation)
            
        return rhs

    def GetMass(self, data, trial, test, ale):

        mass = 1/data.dt*trial[0]*test[0]*dx(deformation = ale.deformation)
        return mass
    
    def PostProcess(self, data, ale):

        super().PostProcess(data, ale)

        if self.params['BP'] and not self.params['MP']:

            gfu_vec = self.gfu.vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.params['BP'])
            self.gfu.vec.data = gfu_new

        elif self.params['MP']:

            if hasattr(data, 'dt'):
                dt = data.dt.Get()
            else:
                raise Exception('A time-dependent simulation is needed to impose conservative mass!')

            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir }, deformation=data.dX)
            elif data.mesh.dim == 3:
                raise Exception('Not yet implemented!')
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir }, deformation=data.dX)
            A = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            A += u*v*dx_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu.vec.Copy().FV().NumPy()

            if self.params['BP']:
                BP = self.params['BP']
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.params['MP'], mass0 =self.mass0, dt = dt)

            self.gfu.vec.data = gfu_new

    def Update(self, data, ale):

        data.mesh.SetDeformation(ale.deformation)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

        data.mesh.UnsetDeformation()

    def get_error(self, data, ex_sol, norm):

        err = compute_error(data=data, gfu = self.gfu, u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = VOL)
        
        return [err]

    def set_solution(self, value):

        self.gfu.Set(value, definedon=self.domain)

    def get_solution(self):

        return self.gfu

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              Diffusion-dominant problems are supposed to be simulated.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')