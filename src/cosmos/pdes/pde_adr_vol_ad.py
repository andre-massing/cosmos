from ngsolve import *
from cosmos.pdes.pde_adr_vol import VolADR
from cosmos.pdes.pde_tools import MandBP
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

        self.domain = data.mesh.Materials(self.params['domain'])
        self.name = [self.params['name']]

        if data.mesh.ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + str(self.name) + ' cannot be initialized')
        
        if self.params['periodic']:
            self.fes = Compress(Periodic(H1(data.mesh, order = self.fes_order, 
                                            dgjumps = True, definedon = self.domain)))
        else:
            self.fes = Compress(H1(data.mesh, order = self.fes_order, 
                                   dgjumps = True, definedon = self.domain))
            
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

    def GetLHS(self, data, trial, test, dX = None):

        n = specialcf.normal(data.mesh.dim)
        h = specialcf.mesh_size

        if self.params['c']:
            lhs = self.params['c']*trial[0]*test[0]*dx(deformation = dX)
        else:
            lhs =  CF(0)*trial[0]*test[0]*dx(deformation = dX)

        if self.params['d']:
            lhs += self.params['d']*grad(trial[0])*grad(test[0])*dx(deformation = dX)
                    
            if self.params['dir_d']:
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for key, value in self.params['dir_d'].items():
                    lhs += - self.params['d']*InnerProduct(n, grad(trial[0]))*test[0]*ds(definedon = key, skeleton=True, deformation = dX) \
                        - self.params['d']*InnerProduct(n, grad(test[0]))*trial[0]*ds(definedon = key, skeleton=True, deformation = dX)\
                        + self.params['d']*alpha/h*trial[0]*test[0]*ds(definedon = key, skeleton = True, deformation = dX)\
            
        if self.params['b']:
            stab = Norm(self.params['b'])*h
            if self.params['d']:
                stab += Norm(self.params['d'])
            if self.params['c']:
                stab += Norm(self.params['c'])*h**2
            jump_u = n*(grad(trial[0]) - (grad(trial[0])).Other())
            jump_v = n*(grad(test[0]) - (grad(test[0])).Other())
            lhs += -self.params['b']*grad(test[0]) * trial[0]*dx(deformation = dX)\
                    + h**3/stab*jump_u*jump_v*dx(skeleton=True, deformation = dX)

            if self.params['neu_b']:
                for key, value in self.params['neu_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = key, deformation = dX)

            if self.params['dir_b']:
                for key, value in self.params['dir_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = key, deformation = dX)
                    
            if self.params['Fneu_b']:
                for key, value in self.params['Fneu_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial[0], CF(0))*test[0]\
                        *ds(definedon = data.mesh.Boundaries('.*') - data.mesh.Boundaries(key), deformation = dX)
            
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