from ngsolve import *
from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.pde_tools import compute_error
import os
from ngsolve.webgui import Draw
import numpy as np
import scipy.sparse as scipy

class AdBndADR(BndADR):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.fields = 2

    def Initialize(self, data):

        self.dim = data.mesh.dim
        self.domain = data.mesh.Boundaries(self.params['domain'])
        self.name = [self.params['name']]
        
        if self.params['periodic']:
            V = Periodic(H1(data.mesh, order = self.fes_order,
                            definedon=self.domain))
        else:
            V = H1(data.mesh, order = self.fes_order,
                definedon=self.domain)
            
        if data.mesh.dim == 2:
            dV = H1(data.mesh, order = 1,
                definedon=self.domain)
        else:
            dV = NormalFacetSurface(data.mesh, order = 0,
                definedon=self.domain)
            
        self.fes = CompressCompound(V*dV)
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        if self.params['u0']:
            self.gfu.components[0].Set(self.params['u0'], 
                        definedon=self.domain)
        self.gfu_old.vec.data = self.gfu.vec.data

        if self.params['MP']:
            self.params['gfu0'] = self.gfu.components[0].vec.Copy()

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu_save = [GridFunction(Compress(H1(data.mesh, order = self.fes_order)))]
        self.gfu_save[0].Set(self.gfu.components[0], definedon = self.domain)

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
            facet_space = H1(data.mesh, order = 1, definedon=self.domain)
        else:
            facet_space = FacetSurface(data.mesh, order = 0)

        if self.params['c']:
            lhs = self.params['c']*trial[0]*test[0]*ds
        else:
            lhs = CF(0)*trial[0]*test[0]*ds

        if self.params['d']:
            lhs += self.params['d']*grad(trial[0]).Trace()*grad(test[0]).Trace()*ds
            
            if self.params['dir_d']:

                dir_d = {}

                alpha = 5 * self.fes_order * (self.fes_order+1)
                for i, (key, value) in enumerate(self.params['dir_d'].items()):

                    dir_d[str(i)] = GridFunction(facet_space)
                    dir_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += - self.params['d']*InnerProduct(nE, grad(trial[0]).Trace())*dir_d[str(i)]*test[0]*ds(element_boundary=True) \
                            - self.params['d']*InnerProduct(nE, grad(test[0]).Trace())*dir_d[str(i)]*trial[0]*ds(element_boundary=True)\
                            + self.params['d']*alpha/h*trial[0]*test[0]*dir_d[str(i)]*ds(element_boundary=True)

        if self.params['b']:
            b = Ps*self.params['b']
            lhs += -b*grad(test[0]).Trace() * trial[0] *ds

            if self.params['neu_b']:

                neu_b = {}
                for i, (key, value) in enumerate(self.params['neu_b'].items()):
                    neu_b[str(i)] = GridFunction(facet_space)
                    neu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], 0)\
                                        *neu_b[str(i)]*test[0]*ds(element_boundary=True)
            
            if self.params['Fneu_b']:

                Fneu_b = {}
                for i, (key, value) in enumerate(self.params['Fneu_b'].items()):
                    Fneu_b[str(i)] = GridFunction(facet_space)
                    Fneu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries('.*') - data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], CF(0))\
                                        *Fneu_b[str(i)]*test[0]*ds(element_boundary=True)
                    
            if self.params['dir_b']:

                dir_b = {}

                for i, (key, value) in enumerate(self.params['dir_b'].items()):
                    dir_b[str(i)] = GridFunction(facet_space)
                    dir_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], 0)\
                                        *dir_b[str(i)]*test[0]*ds(element_boundary=True)
                    
            stab = Norm(self.params['b'])*h
            if self.params['d']:
                stab += Norm(self.params['d'])
            if self.params['c']:
                stab += Norm(self.params['c'])*h**2
            if data.mesh.dim == 2:
                tEc = CF((-ns[1], ns[0]))
                jump_dudn = (trial[0].Trace().Deriv() - trial[1]*tEc)*nE
                jump_dvdn = (test[0].Trace().Deriv() - test[1]*tEc)*nE
            elif data.mesh.dim == 3:
                jump_dudn = (trial[0].Trace().Deriv() - trial[1].Trace())*nE
                jump_dvdn = (test[0].Trace().Deriv() - test[1].Trace())*nE

            gfFone = GridFunction(facet_space)
            gfFone.Set(1, definedon=self.domain, dual = True)
            gfFBB = GridFunction(facet_space)
            gfFBB.Set(1, definedon=data.mesh.BBoundaries('.*'))
            lhs +=  h**3/stab*(gfFone - gfFBB)*InnerProduct(jump_dudn,jump_dvdn)\
                *ds(element_boundary=True)
            
            if data.mesh.dim == 2:
                lhs +=  gfFBB*InnerProduct(trial[1],test[1])\
                *ds(element_boundary=True)
            elif data.mesh.dim == 3:
                lhs +=  gfFBB*trial[1].Trace()*test[1].Trace()*ds(element_boundary=True)
                    
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
            facet_space = H1(data.mesh, order = 1, definedon=self.domain)
        else:
            facet_space = FacetSurface(data.mesh, order = 0)

        if self.params['rhs']:
            rhs =  self.params['rhs']*test[0]*ds
        else:
            rhs =  CF(0)*test[0]*ds

        if self.params['neu_d']:

            neu_d = {}

            for i, (key, value) in enumerate(self.params['neu_d'].items()):                
                neu_d[str(i)] = GridFunction(facet_space)
                neu_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -InnerProduct(nE, value)*neu_d[str(i)]*test[0]*ds(element_boundary=True)

        if self.params['neu_b']:

            b = Ps*self.params['b']

            neu_b = {}

            for i, (key, value) in enumerate(self.params['neu_b'].items()):
                neu_b[str(i)] = GridFunction(facet_space)
                neu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, b), 0,
                            InnerProduct(nE, value))*neu_b[str(i)]*test[0]\
                                *ds(element_boundary=True)
            
        if self.params['Fneu_b']:

            Fneu_b = {}

            for i, (key, value) in enumerate(self.params['Fneu_b'].items()):
                Fneu_b[str(i)] = GridFunction(facet_space)
                Fneu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -InnerProduct(nE, value)*Fneu_b[str(i)]*test[0]\
                                *ds(element_boundary=True)
                
        if self.params['dir_d']:

            alpha = 5 * self.fes_order * (self.fes_order+1)

            dir_d = {}
            for i, (key, value) in enumerate(self.params['dir_d'].items()):
                dir_d[str(i)] = GridFunction(facet_space)
                dir_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += self.params['d']*alpha/h*value*test[0]*dir_d[str(i)]*ds(element_boundary=True) \
                    - self.params['d']*InnerProduct(nE, grad(test[0]).Trace())*dir_d[str(i)]*value*ds(element_boundary=True)
                
                
        if self.params['dir_b']:

            b = Ps*self.params['b']

            dir_b = {}
            for i, (key, value) in enumerate(self.params['dir_b'].items()):
                dir_b[str(i)] = GridFunction(facet_space)
                dir_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, b), 0,
                            InnerProduct(nE, b*value))*dir_b[str(i)]*test[0]\
                                *ds(element_boundary=True)         
                
        return rhs

    def GetMass(self, data, trial, test):

        mass = 1/data['dt']*trial[0]*test[0]*ds
                
        return mass

    def Update(self, data):

        self.gfu_old.vec.data = self.gfu.vec.data
        self.gfu_save[0].Set(self.gfu.components[0], definedon = self.domain)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

    def get_error(self, data, ex_sol, norm):

        err = compute_error(data=data, gfu = self.gfu.components[0], u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err]
    
    def set_solution(self, value):

        self.gfu.components[0].Set(value, definedon=self.domain)
        self.gfu_old.components[0].Set(value, definedon=self.domain)

    def get_solution(self):

        return self.gfu.components[0]     

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              It allows to solve surface advection dominant equations.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')