from ngsolve import *
from cosmos.pdes.pde_adr_vol import VolADR
import numpy as np
import scipy.sparse as sp

class AdVolADR(VolADR):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def Initialize(self, data):

        if self.initialized:
            return
        else:
            self.initialized = True

        self.domain = data.mesh.Materials(self.domain)

        if data.mesh.ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + str(self.name) + ' cannot be initialized')
        
        if self.periodic:
            self.fes = Compress(Periodic(H1(data.mesh, order = self.fes_order, 
                                            dgjumps = True, definedon = self.domain)))
        else:
            self.fes = Compress(H1(data.mesh, order = self.fes_order, 
                                   dgjumps = True, definedon = self.domain))
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)

        if self.u0():
            self.gfu.Set(self.u0())

        self.gfu_save = [self.gfu]

        if self.MP:
            if data.mesh.dim == 2:
                ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
                dx_lumped = dx(intrules = { TRIG : ir })
            elif data.mesh.dim == 3:
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

        if self.c():
            lhs = self.c()*trial[0]*test[0]*dx(deformation = ale.deformation)
        else:
            lhs =  CF(0)*trial[0]*test[0]*dx(deformation = ale.deformation)

        if self.d():
            lhs += self.d()*grad(trial[0])*grad(test[0])*dx(deformation = ale.deformation)
                    
            if self.dir_d:
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for key, field in self.dir_d.items():
                    lhs += - self.d()*InnerProduct(n, grad(trial[0]))*test[0]*ds(definedon = key, skeleton=True, deformation = ale.deformation) \
                        - self.d()*InnerProduct(n, grad(test[0]))*trial[0]*ds(definedon = key, skeleton=True, deformation = ale.deformation)\
                        + self.d()*alpha/h*trial[0]*test[0]*ds(definedon = key, skeleton = True, deformation = ale.deformation)\
            
        if self.b():
            stab = Norm(self.b())*h
            if self.d():
                stab += Norm(self.d())
            if self.c:
                stab += Norm(self.c())*h**2
            jump_u = n*(grad(trial[0]) - (grad(trial[0])).Other())
            jump_v = n*(grad(test[0]) - (grad(test[0])).Other())
            lhs += -self.b()*grad(test[0]) * trial[0]*dx(deformation = ale.deformation)\
                    + h**3/stab*jump_u*jump_v*dx(skeleton=True, deformation = ale.deformation)

            if self.neu_b:
                for key, field in self.neu_b.items():
                    lhs += IfPos(self.b()*n, self.b()*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = key, deformation = ale.deformation)

            if self.dir_b:
                for key, field in self.dir_b.items():
                    lhs += IfPos(self.b()*n, self.b()*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = key, deformation = ale.deformation)
                    
            if self.Fneu_b:
                for key, field in self.Fneu_b.items():
                    lhs += IfPos(self.b()*n, self.b()*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = data.mesh.Boundaries('.*') - data.mesh.Boundaries(key), deformation = ale.deformation)
            
        return lhs

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              Diffusion-dominant problems are supposed to be simulated.\n \
              It uses conforming H-1 elements of order 1.')

        # print('Its parameters are')
        # for key, value in self.params.items():
        #     print('  -', key, '- with value ', str(value))

        # print(60*'-', '\n')