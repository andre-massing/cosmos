from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error, MandBP
import numpy as np
import scipy.sparse as sp

class BndADR(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.fields = 1

        # Initialize the parameters
        accepted_keys = ['b', 'c', 'd', 'u0', 'rhs',
                         'neu_d', 'neu_b', 'dir_d', 'dir_b', 'Fneu_b',
                         'domain', 'name', 'periodic',
                         'MP', 'BP']
        defaults = [None, None, None, None, None, 
                    {}, {}, {}, {}, {},
                    '.*', "surface_adr", False,
                    False, [-np.inf, np.inf],
                    None, None]
        
        if kwargs:
            params_check(kwargs, accepted_keys, defaults)
            self.params = kwargs
            for i, (key, value) in enumerate(self.params.items()):
                if i<5 and isinstance(self.params[key], (int, float)):
                    self.params[key] = CF(self.params[key])
        else:
            self.params = {}
            params_check(self.params, accepted_keys, defaults) 

        self.gfu = self.params['u0']

    def Initialize(self, data):

        self.dim = data.mesh.dim
        self.domain = data.mesh.Boundaries(self.params['domain'])
        self.name = [self.params['name']]
        
        if self.params['periodic']:
            self.fes = Compress(Periodic(H1(data.mesh, order = self.fes_order, 
                definedon=self.domain)))
        else:
            self.fes = Compress(H1(data.mesh, order = self.fes_order,
                definedon=self.domain))
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        if self.params['u0']:
            self.gfu.Set(self.params['u0'], 
                        definedon=self.domain)
        self.gfu_old.vec.data = self.gfu.vec.data

        if self.params['MP']:
            self.params['gfu0'] = self.gfu.vec.Copy()

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu_save = [GridFunction(Compress(H1(data.mesh, order = self.fes_order)))]
        self.gfu_save[0].Set(self.gfu, definedon = self.domain)

        self.gfu_comp = self.gfu
        self.gfu_old_comp = self.gfu_old

        if self.params['MP']:
            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir })
            elif data.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir })
            A = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            A += u*v*ds_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu0_vec = self.gfu.vec.Copy().FV().NumPy()
            self.mass0 = np.sum(weights*gfu0_vec)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)

    def GetLHS(self, data, trial, test):

        ns = specialcf.normal(data.mesh.dim)
        Ps = Id(self.dim) - OuterProduct(ns, ns) 
        tE = specialcf.tangential(data.mesh.dim)
        h = specialcf.mesh_size
        if data.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if data.mesh.dim == 2:
            facet_space = self.fes
        else:
            facet_space = FacetSurface(data.mesh, order = 0)

        if self.params['c']:
            lhs = self.params['c']*trial*test*ds
        else:
            lhs = CF(0)*trial*test*ds

        if self.params['d']:
            lhs += self.params['d']*grad(trial).Trace()*grad(test).Trace()*ds

            if self.params['dir_d']:

                dir_d = {}
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for i, (key, value) in enumerate(self.params['dir_d'].items()):

                    dir_d[str(i)] = GridFunction(facet_space)
                    dir_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += - self.params['d']*InnerProduct(nE, grad(trial).Trace())*dir_d[str(i)]*test*ds(element_boundary=True) \
                            - self.params['d']*InnerProduct(nE, grad(test).Trace())*dir_d[str(i)]*trial*ds(element_boundary=True)\
                            + self.params['d']*alpha/h*trial*test*dir_d[str(i)]*ds(element_boundary=True)
                            
        if self.params['b']:
            lhs += -self.params['b']*grad(test).Trace() * trial *ds

            if self.params['neu_b']:

                neu_b = {}
                for i, (key, value) in enumerate(self.params['neu_b'].items()):
                    neu_b[str(i)] = GridFunction(facet_space)
                    neu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial, 0)\
                                        *neu_b[str(i)]*test*ds(element_boundary=True)
            
            if self.params['Fneu_b']:

                Fneu_b = {}
                for i, (key, value) in enumerate(self.params['Fneu_b'].items()):
                    Fneu_b[str(i)] = GridFunction(facet_space)
                    Fneu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries('.*') - data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial, CF(0))\
                                        *Fneu_b[str(i)]*test*ds(element_boundary=True)
                    
            if self.params['dir_b']:

                dir_b = {}

                for i, (key, value) in enumerate(self.params['dir_b'].items()):
                    dir_b[str(i)] = GridFunction(facet_space)
                    dir_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial, 0)\
                                        *dir_b[str(i)]*test*ds(element_boundary=True)
                
        return lhs

    def GetRHS(self, data, test):

        ns = specialcf.normal(data.mesh.dim)
        Ps = Id(self.dim) - OuterProduct(ns, ns) 
        tE = specialcf.tangential(data.mesh.dim)
        h = specialcf.mesh_size
        if data.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if data.mesh.dim == 2:
            facet_space = self.fes
        else:
            facet_space = FacetSurface(data.mesh, order = 0)

        if self.params['rhs']:
            rhs =  self.params['rhs']*test*ds
        else:
            rhs =  CF(0)*test*ds

        if self.params['neu_d']:

            neu_d = {}

            for i, (key, value) in enumerate(self.params['neu_d'].items()):                
                neu_d[str(i)] = GridFunction(facet_space)
                neu_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -InnerProduct(nE, value)*neu_d[str(i)]*test*ds(element_boundary=True)

        if self.params['neu_b']:

            neu_b = {}
            for i, (key, value) in enumerate(self.params['neu_b'].items()):
                neu_b[str(i)] = GridFunction(facet_space)
                neu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, self.params['b']), 0,
                            InnerProduct(nE, value))*neu_b[str(i)]*test\
                                *ds(element_boundary=True)
            
        if self.params['Fneu_b']:

            Fneu_b = {}

            for i, (key, value) in enumerate(self.params['Fneu_b'].items()):
                Fneu_b[str(i)] = GridFunction(facet_space)
                Fneu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -InnerProduct(nE, value)*Fneu_b[str(i)]*test\
                                *ds(element_boundary=True)
                
        if self.params['dir_d']:

            alpha = 5 * self.fes_order * (self.fes_order+1)

            dir_d = {}
            for i, (key, value) in enumerate(self.params['dir_d'].items()):
                dir_d[str(i)] = GridFunction(facet_space)
                dir_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += self.params['d']*alpha/h*value*test*dir_d[str(i)]*ds(element_boundary=True) \
                    - self.params['d']*InnerProduct(nE, grad(test).Trace())*dir_d[str(i)]*value*ds(element_boundary=True)
                
                
        if self.params['dir_b']:

            dir_b = {}
            for i, (key, value) in enumerate(self.params['dir_b'].items()):
                dir_b[str(i)] = GridFunction(facet_space)
                dir_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, self.params['b']), 0,
                            InnerProduct(nE, self.params['b']*value))*dir_b[str(i)]*test\
                                *ds(element_boundary=True)         
                
        return rhs

    def GetMass(self, data, trial, test):
        mass = 1/data.dt*trial*test*ds
        return mass
    
    def GetMassOld(self, data, trial, test):
        return self.GetMass(data, trial, test)
    
    def PreProcess(self, data):

        pass
    
    def PostProcess(self, data):

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
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir })
            elif data.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir })
            A = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            A += u*v*ds_lumped
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
    
    def Update(self, data):

        self.gfu_old.vec.data = self.gfu.vec.data
        self.gfu_save[0].Set(self.gfu, definedon = self.domain)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

    def get_error(self, data, ex_sol, norm):

        err = compute_error(data=data, gfu = self.gfu, u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err]
    
    def set_solution(self, value):

        self.gfu.Set(value, definedon=self.domain)
        self.gfu_old.Set(value, definedon=self.domain)

    def get_solution(self):

        return self.gfu     

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              It allows to solve surface diffusion-dominant equations.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')