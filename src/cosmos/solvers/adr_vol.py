from ngsolve import *
from cosmos.solvers.base_pde import BasePDE
from cosmos.solvers.tools import compute_error, params_check
import os
import csv
from ngsolve.webgui import Draw
import numpy as np
import scipy.sparse as scipy

class VolADR(BasePDE):

    def __init__(self, **kwargs):

        super().__init__()

        self.params = kwargs

        self.fields = 1

        # Initialize the parameters
        accepted_keys = ['b', 'c', 'd', 'u0', 'rhs',
                         'neu_d', 'neu_b', 'dir_d', 'dir_b', 'Fneu_b',
                         'domain', 'name', 'periodic',
                         'MP', 'BP', 'stationary']
        defaults = [None, None, None, None, None, 
                    {}, {}, {}, {}, {},
                    '.*', 'volume_adr', False,
                    False, [-np.inf, np.inf], False]

        params_check(self.params, accepted_keys, defaults)
        for i, (key, value) in enumerate(self.params.items()):
            if i<5 and isinstance(self.params[key], (int, float)):
                self.params[key] = CF(self.params[key])

        self.gfu = self.params['u0']

    def Initialize(self, mesh_data):

        self.dim = mesh_data['mesh'].dim

        if mesh_data['mesh'].ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + self.params['name'] + ' cannot be initialized')
        
        if self.params['periodic']:
            self.fes = Compress(Periodic(H1(mesh_data['mesh'], order = self.fes_order, 
                                            definedon = self.params['domain'])))
        else:
            self.fes = Compress(H1(mesh_data['mesh'], order = self.fes_order, 
                                   definedon = self.params['domain']))
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        if self.params['u0']:
            self.gfu.Set(self.params['u0'])
        self.gfu_old.vec.data = self.gfu.vec.data

        if self.params['MP']:
            self.params['gfu0'] = self.gfu.vec.Copy()

        if self.error_params:

            file_path = os.path.join(self.error_params['folderpath'], self.error_params['filename'])
            os.makedirs(self.error_params['folderpath'], exist_ok=True)

        if self.save_params:

            if mesh_data['mesh'].ne == 0:
                self.save_params['VorB'] = BND
            else:
                self.save_params['VorB'] = VOL

            file_path = os.path.join(self.save_params['folderpath'], self.save_params['filename'])
            os.makedirs(self.save_params['folderpath'], exist_ok=True)

            self.save_params['vtk'] = VTKOutput(mesh_data['mesh'], coefs= [self.gfu],
                        names=[self.params['name']], filename = file_path,
                        subdivision = self.save_params['subdivision'])

    def GetLHS(self, mesh_data, trial, test):

        n = specialcf.normal(mesh_data['mesh'].dim)
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
            lhs += -self.params['b']*grad(test) * trial*dx

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
                        *ds(definedon = mesh_data['mesh'].Boundaries('.*') - mesh_data['mesh'].Boundaries(key))
            
        return lhs

    def GetRHS(self, mesh_data, test):

        n = specialcf.normal(mesh_data['mesh'].dim)
        h = specialcf.mesh_size

        rhs = self.params['rhs']*test*dx
        
        if self.params['dir_d']:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            for key, value in self.params['dir_d'].items():
                rhs += self.params['d']*alpha/h*value*test*ds(definedon = key, skeleton = True)\
                    - self.params['d']*InnerProduct(n, grad(test))*value*ds(definedon = key, skeleton=True)
        if self.params['neu_d']:
            for key, value in self.params['neu_d'].items():
                rhs += -value*n*test*ds(definedon = key)

        if self.params['dir_b']:
            for key, value in self.params['dir_b'].items():
                rhs += -IfPos(self.params['b']*n, CF(0), self.params['b']*n*value)*test*ds(definedon = key)
        if self.params['neu_b']:
            for key, value in self.params['neu_b'].items():
                rhs += -IfPos(self.params['b']*n, CF(0), value*n)*test*ds(definedon = key)

        if self.params['Fneu_b']:
            for key, value in self.params['Fneu_b'].items():
                rhs += -value*n*test*ds(definedon = mesh_data['mesh'].Boundaries(key))
            
        return rhs

    def GetMass(self, mesh_data, trial, test, gfu):

        if self.params['stationary']:
            factor = 0
        else:
            factor = 1
        
        mass = factor/mesh_data['dt']*trial*test*dx
        mass_gfu = factor/mesh_data['dt']*self.gfu*test*dx
            
        return mass, mass_gfu
    
    def MP_and_BP(self, mesh_data, gfu):

        dt = mesh_data['dt'].Get()
        tol = 1e-10

        gfu_vec = gfu.vec.FV().NumPy()[:]

        if self.params['BP'] and not self.params['MP']:

            gfu_data = np.minimum(np.maximum(gfu_vec, self.params['BP'][0]), self.params['BP'][1])  
            gfu.vec.data[:] = gfu_data

        elif self.params['MP']:
            
            gfu0 = self.params['gfu0'].FV().NumPy()[:]

            u, v = gfu.space.TnT()
            if mesh_data['mesh'].dim == 2:
                ir = IntegrationRule([(0,0), (1,0),(0,1)], [1/6, 1/6, 1/6])
            else:
                raise Exception('Not yet implemented!')
            A = BilinearForm(gfu.space, symmetric = True)
            A += u*v*dx(intrules={TRIG: ir})
            A.Assemble()
            A_old = BilinearForm(gfu.space, symmetric = True)
            A_old += u*v*dx(intrules={TRIG: ir})
            A_old.Assemble()

            rows,cols,vals = A.mat.COO()
            Adiag = scipy.csr_matrix((vals,(rows,cols))).diagonal()
            rows,cols,vals = A_old.mat.COO()
            Adiag_old = scipy.csr_matrix((vals,(rows,cols))).diagonal()

            prod0 = np.dot(gfu0, Adiag_old)

            def F(xsi):

                result = 0

                temp = gfu_vec + dt * xsi

                result -= prod0
                dummy = np.minimum(np.maximum(temp, self.params['BP'][0]), 
                                   self.params['BP'][1])
                result += np.sum(Adiag * dummy)

                return result

            xsi_old0 = 0
            xsi_old1 = dt
            xsi2 = 1e100

            while abs(F(xsi_old1) - F(xsi_old0))>tol:

                F1 = F(xsi_old1)
                F0 = F(xsi_old0)

                xsi2 = xsi_old1-F1*(xsi_old1 - xsi_old0)/(F1 - F0)

                xsi_old0 = xsi_old1
                xsi_old1 = xsi2

            threshold = dt * xsi2
            gfu_data = np.minimum(np.maximum(gfu_vec + threshold, 
                                             self.params['BP'][0]), self.params['BP'][1]) 
            gfu.vec.data[:] = gfu_data

    def Update(self, mesh_data):

        self.gfu_old.vec.data = self.gfu.vec.data

    def get_error(self, mesh_data):

        err = compute_error(data=mesh_data, gfu = self.gfu, u_ex=self.error_params['ex_sol'],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = VOL)
        
        return err

    def set_solution(self, mesh_data, value):

        self.gfu.Set(value, definedon=mesh_data['mesh'].Materials(self.params['domain']))
        self.gfu_old.Set(value, definedon=mesh_data['mesh'].Materials(self.params['domain']))

    def get_solution(self, **kwargs):

        return self.gfu
        
    def draw_solution(self, mesh_data, **kwargs):

        Draw(self.gfu)

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              Diffusion-dominant problems are supposed to be simulated.\n \
              It uses conforming H-1 elements of order 1.')

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with value ', str(value))

        print(60*'-', '\n')

class AdVolADR(VolADR):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def Initialize(self, mesh_data):

        self.dim = mesh_data['mesh'].dim

        if mesh_data['mesh'].ne == 0:
            raise Exception('The mesh has no volume elements! The PDE ' 
                            + self.params['name'] + ' cannot be initialized')
        
        if self.params['periodic']:
            self.fes = Compress(Periodic(H1(mesh_data['mesh'], order = self.fes_order, 
                                            dgjumps = True, definedon = self.params['domain'])))
        else:
            self.fes = Compress(H1(mesh_data['mesh'], order = self.fes_order, 
                                   dgjumps = True, definedon = self.params['domain']))
            
        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        if self.params['u0']:
            self.gfu.Set(self.params['u0'])
        self.gfu_old.vec.data = self.gfu.vec.data

        if self.params['MP']:
            self.params['gfu0'] = self.gfu.vec.Copy()

        if self.error_params:

            file_path = os.path.join(self.error_params['folderpath'], self.error_params['filename'])
            os.makedirs(self.error_params['folderpath'], exist_ok=True)

        if self.save_params:

            if mesh_data['mesh'].ne == 0:
                self.save_params['VorB'] = BND
            else:
                self.save_params['VorB'] = VOL

            file_path = os.path.join(self.save_params['folderpath'], self.save_params['filename'])
            os.makedirs(self.save_params['folderpath'], exist_ok=True)

            self.save_params['vtk'] = VTKOutput(mesh_data['mesh'], coefs= [self.gfu],
                        names=[self.params['name']], filename = file_path,
                        subdivision = self.save_params['subdivision'])

    def GetLHS(self, mesh_data, trial, test):

        n = specialcf.normal(mesh_data['mesh'].dim)
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
                        - self.params['d']*InnerProduct(n, grad(test))*trial*ds(definedon = key, skeleton=True) \
                        + self.params['d']*alpha/h*trial*test*ds(definedon = key, skeleton = True)
            
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
                        *ds(definedon = mesh_data['mesh'].Boundaries('.*') - mesh_data['mesh'].Boundaries(key)) 
                        
        return lhs

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              It allows for the solution of advection-dominant problems.\n \
              It uses conforming H-1 elements of order 1.')

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with value ', str(value))

        print(60*'-', '\n')