from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error, MandBP
import os
from ngsolve.webgui import Draw
import numpy as np
import scipy.sparse as sp

class AdBndADR(BasePDE):

    def __init__(self, b = None, c = None, d = None, u0 = None, rhs = None,
                 neu_d = {}, neu_b = {}, dir_d = {}, dir_b = {}, Fneu_b = {},
                 domain:str = '.*', name:str = 'surface_adr', periodic :bool = False,
                 MP:bool = False, BP:bool = False):

        super().__init__()

        self.b = b
        self.c = c
        self.d = d
        self.u0 = u0
        self.rhs = rhs
        self.neu_d = neu_d
        self.neu_b = neu_b
        self.dir_d = dir_d
        self.dir_b = dir_b
        self.Fneu_b = Fneu_b
        self.MP = MP
        self.BP = BP
        self.periodic = periodic
        self.name = [name]
        self.domain = domain

        self.nfields = 2

    def Initialize(self, data):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = data.mesh.Boundaries(self.domain)
        
        if self.periodic:
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

        if self.u0():
            self.gfu.components[0].Set(self.u0(), 
                        definedon=self.domain)

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu_save = [GridFunction(Compress(H1(data.mesh, order = self.fes_order)))]
        self.gfu_save[0].Set(self.gfu.components[0], definedon = self.domain)

        if self.MP:
            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir })
            elif data.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir })
            A = BilinearForm(self.gfu.components[0].space, symmetric = True)
            u, v = self.gfu.components[0].space.TnT()
            A += u*v*ds_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu0_vec = self.gfu.components[0].vec.Copy().FV().NumPy()
            self.mass0 = np.sum(weights*gfu0_vec)

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)

    def GetLHS(self, data, trial, test, ale):

        ns = specialcf.normal(data.mesh.dim)
        Ps = Id(data.mesh.dim) - OuterProduct(ns, ns) 
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

        if self.c():
            lhs = self.c()*trial[0]*test[0]*ds(deformation = ale.deformation)
        else:
            lhs = CF(0)*trial[0]*test[0]*ds(deformation = ale.deformation)

        if self.d():
            lhs += self.d()*grad(trial[0]).Trace()*grad(test[0]).Trace()*ds(deformation = ale.deformation)
            
            if self.dir_d:

                dir_d = {}

                alpha = 5 * self.fes_order * (self.fes_order+1)
                for i, (key, field) in enumerate(self.dir_d.items()):

                    dir_d[str(i)] = GridFunction(facet_space)
                    dir_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += - self.d()*InnerProduct(nE, grad(trial[0]).Trace())*dir_d[str(i)]*test[0]*ds(element_boundary=True, deformation = ale.deformation) \
                            - self.d()*InnerProduct(nE, grad(test[0]).Trace())*dir_d[str(i)]*trial[0]*ds(element_boundary=True, deformation = ale.deformation)\
                            + self.d()*alpha/h*trial[0]*test[0]*dir_d[str(i)]*ds(element_boundary=True, deformation = ale.deformation)

        if self.b():
            b = Ps*self.b()
            lhs += -b*grad(test[0]).Trace() * trial[0] *ds(deformation = ale.deformation)

            if self.neu_b:

                neu_b = {}
                for i, (key, field) in enumerate(self.neu_b.items()):
                    neu_b[str(i)] = GridFunction(facet_space)
                    neu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], 0)\
                                        *neu_b[str(i)]*test[0]*ds(element_boundary=True, deformation = ale.deformation)
            
            if self.Fneu_b:

                Fneu_b = {}
                for i, (key, field) in enumerate(self.Fneu_b.items()):
                    Fneu_b[str(i)] = GridFunction(facet_space)
                    Fneu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries('.*') - data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], CF(0))\
                                        *Fneu_b[str(i)]*test[0]*ds(element_boundary=True, deformation = ale.deformation)
                    
            if self.dir_b:

                dir_b = {}
                for i, (key, field) in enumerate(self.dir_b.items()):
                    dir_b[str(i)] = GridFunction(facet_space)
                    dir_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], 0)\
                                        *dir_b[str(i)]*test[0]*ds(element_boundary=True, deformation = ale.deformation)
                    
            stab = Norm(b)*h
            if self.d():
                stab += Norm(self.d())
            if self.c():
                stab += Norm(self.c())*h**2
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
                *ds(element_boundary=True, deformation = ale.deformation)
            
            if data.mesh.dim == 2:
                lhs +=  gfFBB*InnerProduct(trial[1],test[1])\
                         *ds(element_boundary=True, deformation = ale.deformation)
            elif data.mesh.dim == 3:
                lhs +=  gfFBB*trial[1].Trace()*test[1].Trace()*ds(element_boundary=True, deformation = ale.deformation)
                    
        return lhs

    def GetRHS(self, data, test, ale):

        ns = specialcf.normal(data.mesh.dim)
        Ps = Id(data.mesh.dim) - OuterProduct(ns, ns)
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

        if self.rhs():
            rhs =  self.rhs()*test[0]*ds(deformation = ale.deformation)
        else:
            rhs =  CF(0)*test[0]*ds(deformation = ale.deformation)

        if self.neu_d:

            neu_d = {}
            for i, (key, field) in enumerate(self.neu_d.items()):                
                neu_d[str(i)] = GridFunction(facet_space)
                neu_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -InnerProduct(nE, field())*neu_d[str(i)]*test[0]*ds(element_boundary=True, deformation = ale.deformation)

        if self.neu_b:

            b = Ps*self.b()
            neu_b = {}
            for i, (key, field) in enumerate(self.neu_b.items()):
                neu_b[str(i)] = GridFunction(facet_space)
                neu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, b), 0,
                            InnerProduct(nE, field()))*neu_b[str(i)]*test[0]\
                                *ds(element_boundary=True, deformation = ale.deformation)
            
        if self.Fneu_b:

            Fneu_b = {}
            for i, (key, field) in enumerate(self.Fneu_b.items()):
                Fneu_b[str(i)] = GridFunction(facet_space)
                Fneu_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -InnerProduct(nE, field())*Fneu_b[str(i)]*test[0]\
                                *ds(element_boundary=True, deformation = ale.deformation)
                
        if self.dir_d:

            alpha = 5 * self.fes_order * (self.fes_order+1)
            dir_d = {}
            for i, (key, field) in enumerate(self.dir_d.items()):
                dir_d[str(i)] = GridFunction(facet_space)
                dir_d[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += self.d()*alpha/h*field()*test[0]*dir_d[str(i)]*ds(element_boundary=True, deformation = ale.deformation) \
                    - self.d()*InnerProduct(nE, grad(test[0]).Trace())*dir_d[str(i)]*field()*ds(element_boundary=True, deformation = ale.deformation)
                
                
        if self.dir_b:

            b = Ps*self.b()
            dir_b = {}
            for i, (key, field) in enumerate(self.dir_b.items()):
                dir_b[str(i)] = GridFunction(facet_space)
                dir_b[str(i)].Set(1, definedon=data.mesh.BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, b), 0,
                            InnerProduct(nE, b*field()))*dir_b[str(i)]*test[0]\
                                *ds(element_boundary=True, deformation = ale.deformation)         
                
        return rhs

    def GetMass(self, data, trial, test, ale):
        mass = 1/data.dt*trial[0]*test[0]*ds(deformation = ale.deformation) 
        return mass
    
    def PostProcess(self, data, ale):

        super().PostProcess(data, ale)

        if self.BP and not self.MP:

            gfu_vec = self.gfu.components[0].vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.BP)
            self.gfu.components[0].vec.data = gfu_new

        elif self.MP:

            if hasattr(data, 'dt'):
                dt = data.dt.Get()
            else:
                raise Exception('A time-dependent simulation is needed to impose conservative mass!')

            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir }, deformation=ale.deformation)
            elif data.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir }, deformation=ale.deformation)
            A = BilinearForm(self.gfu.components[0].space, symmetric = True)
            u, v = self.gfu.components[0].space.TnT()
            A += u*v*ds_lumped
            A.Assemble()
            rows,cols,vals = A.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu.components[0].vec.Copy().FV().NumPy()

            if self.BP:
                BP = self.BP
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.MP, mass0 =self.mass0, dt = dt)

            self.gfu.components[0].vec.data = gfu_new

    def Update(self, data, ale):

        data.mesh.SetDeformation(ale.deformation)

        self.gfu_save[0].Set(self.gfu.components[0], definedon = self.domain)

        for save in self.save_error:
            save.Save(data, self)

        for save in self.save_solution:
            save.Save(data, self)

        data.mesh.UnsetDeformation()

    def get_error(self, data, ex_sol, norm):

        err = compute_error(data=data, gfu = self.gfu.components[0], u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err]
    
    def set_solution(self, field):

        self.gfu.components[0].Set(field, definedon=self.domain)

    def get_solution(self):

        return self.gfu.components[0]     

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              It allows to solve surface advection dominant equations.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, field in self.params.items():
        #     print('  -', key, '- with field ', str(field))

        # print(60*'-', '\n')