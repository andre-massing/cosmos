from ngsolve import *
from cosmos.pdes.pde_adr_vol import VolADR
from cosmos.utils.tools import params_check
from cosmos.pdes.pde_tools import compute_error
import numpy as np

class AdVolADR(VolADR):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def Initialize(self, data):

        self.dim = data.mesh.dim
        self.domain = data.mesh.Materials(self.params['domain'])
        self.name = [self.params['name']]

        if data.mesh.ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + self.name + ' cannot be initialized')
        
        if self.params['periodic']:
            self.fes = Compress(Periodic(H1(data.mesh, order = self.fes_order, 
                                            dgjumps = True, definedon = self.domain)))
        else:
            self.fes = Compress(H1(data.mesh, order = self.fes_order, 
                                   dgjumps = True, definedon = self.domain))
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        if self.params['u0']:
            self.gfu.Set(self.params['u0'])
        self.gfu_old.vec.data = self.gfu.vec.data

        self.gfu_save = [self.gfu]

        if self.params['MP']:
            self.params['gfu0'] = self.gfu.vec.Copy()

        for save in self.save_error:
            save.Initialize(data, self)

        for save in self.save_solution:
            save.Initialize(data, self)

    def GetLHS(self, data, trial, test):

        n = specialcf.normal(data.mesh.dim)
        h = specialcf.mesh_size

        if self.params['c']:
            lhs = self.params['c']*trial*test*dx
        else:
            lhs =  CF(0)*trial*test*dx

        if self.params['d']:
            lhs += self.params['d']*grad(trial)*grad(test)*dx
                    
            if self.params['dir_d']:
                alpha = 5 * self.fes_order * (self.fes_order+1)
                for key, value in self.params['dir_d'].items():
                    lhs += - self.params['d']*InnerProduct(n, grad(trial))*test*ds(definedon = key, skeleton=True) \
                        - self.params['d']*InnerProduct(n, grad(test))*trial*ds(definedon = key, skeleton=True)\
                        + self.params['d']*alpha/h*trial*test*ds(definedon = key, skeleton = True)\
            
        if self.params['b']:
            stab = Norm(self.params['b'])*h
            if self.params['d']:
                stab += Norm(self.params['d'])
            if self.params['c']:
                stab += Norm(self.params['c'])*h**2
            jump_u = n*(grad(trial) - (grad(trial)).Other())
            jump_v = n*(grad(test) - (grad(test)).Other())
            lhs += -self.params['b']*grad(test) * trial*dx\
                    + h**3/stab*jump_u*jump_v*dx(skeleton=True)

            if self.params['neu_b']:
                for key, value in self.params['neu_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial, CF(0))*test\
                        *ds(definedon = key)

            if self.params['dir_b']:
                for key, value in self.params['dir_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial, CF(0))*test\
                        *ds(definedon = key)
                    
            if self.params['Fneu_b']:
                for key, value in self.params['Fneu_b'].items():
                    lhs += IfPos(self.params['b']*n, self.params['b']*n*trial, CF(0))*test\
                        *ds(definedon = data.mesh.Boundaries('.*') - data.mesh.Boundaries(key))
            
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