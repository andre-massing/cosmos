from ngsolve import *
from cosmos.pdes.pde_adr_base import BaseADR
from cosmos.pdes.pde_tools import compute_error, MandBP
from cosmos.solvers.time_schemes import BDF1, BDF2, CN
from ngsolve.webgui import Draw
import numpy as np
import scipy.sparse as sp
from cosmos.solvers.fields import Field

class AdBndADR(BaseADR):

    def __init__(self, b = Field(), c = Field(), d = Field(), u0 = Field(), rhs = Field(),
                 neu_d = {}, neu_b = {}, dir_d = {}, dir_b = {}, Fneu_b = {},
                 domain:str = '.*', name:str = 'surface_adr', periodic :bool = False,
                 MP:bool = False, BP:bool = False, time_scheme = BDF1()):

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
        self.time_scheme = time_scheme

        self.nfields = 2
        self.solute = self.gfu

    def Initialize(self, solverdata):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = solverdata.mesh.Boundaries(self.domain)
        
        if self.periodic:
            V = Periodic(H1(solverdata.mesh, order = self.fes_order,
                            definedon=self.domain))
        else:
            V = H1(solverdata.mesh, order = self.fes_order,
                definedon=self.domain)
            
        if solverdata.mesh.dim == 2:
            dV = H1(solverdata.mesh, order = 1,
                definedon=self.domain)
        else:
            dV = NormalFacetSurface(solverdata.mesh, order = 0,
                definedon=self.domain)
            
        self.fes = CompressCompound(V*dV)
        
        self.gfu = GridFunction(self.fes)

        if self.u0():
            self.gfu.components[0].Set(self.u0(), 
                        definedon=self.domain)
        self.solute = self.gfu.components[0]

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu_save = [GridFunction(Compress(H1(solverdata.mesh, order = self.fes_order)))]
        self.gfu_save[0].Set(self.gfu.components[0], definedon = self.domain)

        if self.MP:
            if solverdata.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir })
            elif solverdata.mesh.dim == 3:
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
            save.Initialize(solverdata, self)

        for save in self.save_solution:
            save.Initialize(solverdata, self)

    def GetLHS(self, solverdata, trial, test):

        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns) 
        tE = specialcf.tangential(solverdata.mesh.dim)
        h = specialcf.mesh_size
        if solverdata.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if solverdata.mesh.dim == 2:
            facet_space = H1(solverdata.mesh, order = 1, definedon=self.domain)
        else:
            facet_space = FacetSurface(solverdata.mesh, order = 0)

        if self.c():
            lhs = self.c()*trial[0]*test[0]*ds(deformation = solverdata.ale.deformation)
        else:
            lhs = CF(0)*trial[0]*test[0]*ds(deformation = solverdata.ale.deformation)

        if self.d():
            lhs += self.d()*grad(trial[0]).Trace()*grad(test[0]).Trace()*ds(deformation = solverdata.ale.deformation)
            
            if self.dir_d:

                dir_d = {}

                alpha = 5 * self.fes_order * (self.fes_order+1)
                for i, (key, field) in enumerate(self.dir_d.items()):

                    dir_d[str(i)] = GridFunction(facet_space)
                    dir_d[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries(key))
                    lhs += - self.d()*InnerProduct(nE, grad(trial[0]).Trace())*dir_d[str(i)]*test[0]*ds(element_boundary=True, deformation = solverdata.ale.deformation) \
                            - self.d()*InnerProduct(nE, grad(test[0]).Trace())*dir_d[str(i)]*trial[0]*ds(element_boundary=True, deformation = solverdata.ale.deformation)\
                            + self.d()*alpha/h*trial[0]*test[0]*dir_d[str(i)]*ds(element_boundary=True, deformation = solverdata.ale.deformation)

        if self.b():
            b = Ps*self.b()
            lhs += -b*grad(test[0]).Trace() * trial[0] *ds(deformation = solverdata.ale.deformation)

            if self.neu_b:

                neu_b = {}
                for i, (key, field) in enumerate(self.neu_b.items()):
                    neu_b[str(i)] = GridFunction(facet_space)
                    neu_b[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], 0)\
                                        *neu_b[str(i)]*test[0]*ds(element_boundary=True, deformation = solverdata.ale.deformation)
            
            if self.Fneu_b:

                Fneu_b = {}
                for i, (key, field) in enumerate(self.Fneu_b.items()):
                    Fneu_b[str(i)] = GridFunction(facet_space)
                    Fneu_b[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries('.*') - solverdata.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], CF(0))\
                                        *Fneu_b[str(i)]*test[0]*ds(element_boundary=True, deformation = solverdata.ale.deformation)
                    
            if self.dir_b:

                dir_b = {}
                for i, (key, field) in enumerate(self.dir_b.items()):
                    dir_b[str(i)] = GridFunction(facet_space)
                    dir_b[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, b), 
                                    InnerProduct(nE, b)*trial[0], 0)\
                                        *dir_b[str(i)]*test[0]*ds(element_boundary=True, deformation = solverdata.ale.deformation)
                    
            stab = IfPos(Norm(self.b())-1, Norm(self.b()), 1)
            if solverdata.mesh.dim == 2:
                tEc = CF((-ns[1], ns[0]))
                jump_dudn = (trial[0].Trace().Deriv() - trial[1]*tEc)*nE
                jump_dvdn = (test[0].Trace().Deriv() - test[1]*tEc)*nE
            elif solverdata.mesh.dim == 3:
                jump_dudn = (trial[0].Trace().Deriv() - trial[1].Trace())*nE
                jump_dvdn = (test[0].Trace().Deriv() - test[1].Trace())*nE

            gfFone = GridFunction(facet_space)
            gfFone.Set(1, definedon=self.domain, dual = True)
            gfFBB = GridFunction(facet_space)
            gfFBB.Set(1, definedon=solverdata.mesh.BBoundaries('.*'))
            lhs +=  h**2/stab*(gfFone - gfFBB)*InnerProduct(jump_dudn,jump_dvdn)\
                *ds(element_boundary=True, deformation = solverdata.ale.deformation)
            
            if solverdata.mesh.dim == 2:
                lhs +=  gfFBB*InnerProduct(trial[1],test[1])\
                         *ds(element_boundary=True, deformation = solverdata.ale.deformation)
            elif solverdata.mesh.dim == 3:
                lhs +=  gfFBB*trial[1].Trace()*test[1].Trace()*ds(element_boundary=True, deformation = solverdata.ale.deformation)
                    
        return lhs

    def GetRHS(self, solverdata, test):

        ns = specialcf.normal(solverdata.mesh.dim)
        Ps = Id(solverdata.mesh.dim) - OuterProduct(ns, ns)
        tE = specialcf.tangential(solverdata.mesh.dim)
        h = specialcf.mesh_size
        if solverdata.mesh.dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if solverdata.mesh.dim == 2:
            facet_space = H1(solverdata.mesh, order = 1, definedon=self.domain)
        else:
            facet_space = FacetSurface(solverdata.mesh, order = 0)

        if self.rhs():
            rhs =  self.rhs()*test[0]*ds(deformation = solverdata.ale.deformation)
        else:
            rhs =  CF(0)*test[0]*ds(deformation = solverdata.ale.deformation)

        if self.neu_d:

            neu_d = {}
            for i, (key, field) in enumerate(self.neu_d.items()):                
                neu_d[str(i)] = GridFunction(facet_space)
                neu_d[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries(key))
                rhs += -InnerProduct(nE, field())*neu_d[str(i)]*test[0]*ds(element_boundary=True, deformation = solverdata.ale.deformation)

        if self.neu_b:

            b = Ps*self.b()
            neu_b = {}
            for i, (key, field) in enumerate(self.neu_b.items()):
                neu_b[str(i)] = GridFunction(facet_space)
                neu_b[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, b), 0,
                            InnerProduct(nE, field()))*neu_b[str(i)]*test[0]\
                                *ds(element_boundary=True, deformation = solverdata.ale.deformation)
            
        if self.Fneu_b:

            Fneu_b = {}
            for i, (key, field) in enumerate(self.Fneu_b.items()):
                Fneu_b[str(i)] = GridFunction(facet_space)
                Fneu_b[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries(key))
                rhs += -InnerProduct(nE, field())*Fneu_b[str(i)]*test[0]\
                                *ds(element_boundary=True, deformation = solverdata.ale.deformation)
                
        if self.dir_d:

            alpha = 5 * self.fes_order * (self.fes_order+1)
            dir_d = {}
            for i, (key, field) in enumerate(self.dir_d.items()):
                dir_d[str(i)] = GridFunction(facet_space)
                dir_d[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries(key))
                rhs += self.d()*alpha/h*field()*test[0]*dir_d[str(i)]*ds(element_boundary=True, deformation = solverdata.ale.deformation) \
                    - self.d()*InnerProduct(nE, grad(test[0]).Trace())*dir_d[str(i)]*field()*ds(element_boundary=True, deformation = solverdata.ale.deformation)
                
                
        if self.dir_b:

            b = Ps*self.b()
            dir_b = {}
            for i, (key, field) in enumerate(self.dir_b.items()):
                dir_b[str(i)] = GridFunction(facet_space)
                dir_b[str(i)].Set(1, definedon=solverdata.mesh.BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, b), 0,
                            InnerProduct(nE, b*field()))*dir_b[str(i)]*test[0]\
                                *ds(element_boundary=True, deformation = solverdata.ale.deformation)         
                
        return rhs

    def GetMass(self, solverdata, trial, test):
        mass = 1/solverdata.dt*trial[0]*test[0]*ds(deformation = solverdata.ale.deformation) 
        return mass
    
    def PostProcess(self, solverdata):

        super().PostProcess(solverdata)

        if self.BP and not self.MP:

            gfu_vec = self.gfu.components[0].vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.BP)
            self.gfu.components[0].vec.data = gfu_new

        elif self.MP:

            if hasattr(solverdata, 'dt'):
                dt = solverdata.dt.Get()
            else:
                raise Exception('A time-dependent simulation is needed to impose conservative mass!')

            if solverdata.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
                ds_lumped = ds(intrules = { SEGM : ir }, deformation=solverdata.ale.deformation)
            elif solverdata.mesh.dim == 3:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                ds_lumped = ds(intrules = { TRIG : ir }, deformation=solverdata.ale.deformation)
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

    def Update(self, solverdata):

        solverdata.mesh.SetDeformation(solverdata.ale.deformation)

        self.gfu_save[0].Set(self.gfu.components[0], definedon = self.domain)

        for save in self.save_error:
            save.Save(solverdata, self)

        for save in self.save_solution:
            save.Save(solverdata, self)

        solverdata.mesh.UnsetDeformation()

    def get_error(self, solverdata, ex_sol, norm):

        err = compute_error(data=solverdata, gfu = self.solute, u_ex=ex_sol,
                            norm = norm, domain = self.domain, 
                            VorB = BND)
        
        return [err]  

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              It allows to solve surface advection dominant equations.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, field in self.params.items():
        #     print('  -', key, '- with field ', str(field))

        # print(60*'-', '\n')