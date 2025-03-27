from ngsolve import *
from cosmos.solvers.base_pde import BasePDE
from cosmos.solvers.tools import compute_error, params_check
import os
import csv
from ngsolve.webgui import Draw
import numpy as np
import scipy.sparse as scipy

class BndADR(BasePDE):

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
                    '.*', 'surface_adr', False,
                    False, [-np.inf, np.inf], False]

        params_check(self.params, accepted_keys, defaults)
        for i, (key, value) in enumerate(self.params.items()):
            if i<5 and isinstance(self.params[key], (int, float)):
                self.params[key] = CF(self.params[key])

        self.gfu = self.params['u0']

    def Initialize(self, mesh_data):

        self.dim = mesh_data['mesh'].dim
        
        if self.params['periodic']:
            self.fes = Compress(Periodic(H1(mesh_data['mesh'], order = self.fes_order, dgjumps = True, 
                definedon=mesh_data['mesh'].Boundaries(self.params['domain']))))
        else:
            self.fes = Compress(H1(mesh_data['mesh'], order = self.fes_order, dgjumps = True, 
                definedon=mesh_data['mesh'].Boundaries(self.params['domain'])))
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        if self.params['u0']:
            self.gfu.Set(self.params['u0'], 
                        definedon=mesh_data['mesh'].Boundaries('.*'))
        self.gfu_old.vec.data = self.gfu.vec.data

        if self.params['MP']:
            self.params['gfu0'] = self.gfu.vec.Copy()

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu_save = GridFunction(Compress(H1(mesh_data['mesh'], order = self.fes_order)))
        self.gfu_save.Set(self.gfu, definedon = mesh_data['mesh'].Boundaries(self.params['domain']))

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

            self.save_params['vtk'] = VTKOutput(mesh_data['mesh'], coefs= [self.gfu_save],
                        names=[self.params['name']], filename = file_path,
                        subdivision = self.save_params['subdivision'])

    def GetLHS(self, mesh_data, trial, test, dX):

        ns = specialcf.normal(mesh_data['mesh'].dim)
        tE = specialcf.tangential(mesh_data['mesh'].dim)
        h = specialcf.mesh_size
        if mesh_data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if mesh_data['mesh'].dim == 2:
            gfF = GridFunction(self.fes)
        else:
            gfF = GridFunction(FacetSurface(mesh_data['mesh'], order = 0))

        if self.params['c']:
            lhs = self.params['c']*trial*test*ds(deformation = dX)
        else:
            lhs = CF(0)*trial*test*ds(deformation = dX)

        if self.params['d']:
            lhs += self.params['d']*grad(trial).Trace()*grad(test).Trace()\
            *ds(deformation = dX)

            if self.params['dir_d']:

                alpha = 5 * self.fes_order * (self.fes_order+1)
                for key, value in self.params['dir_d'].items():
                    gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                    lhs += - self.params['d']*InnerProduct(nE, grad(trial).Trace())*gfF*test*ds(deformation = dX, element_boundary=True) \
                            - self.params['d']*InnerProduct(nE, grad(test).Trace())*gfF*trial*ds(deformation = dX, element_boundary=True)\
                            + self.params['d']*alpha/h*trial*test*gfF*ds(deformation = dX, element_boundary=True)
                            

        if self.params['b']:
            lhs += -self.params['b']*grad(test).Trace() * trial *ds(deformation = dX)

            if self.params['neu_b']:

                for key, value in self.params['neu_b'].items():
                    gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial, 0)\
                                        *gfF*test*ds(deformation = dX, element_boundary=True)
            
            if self.params['Fneu_b']:

                for key, value in self.params['neu_b'].items():
                    gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries('.*') - mesh_data['mesh'].BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial, CF(0))\
                                        *gfF*test*ds(deformation = dX, element_boundary=True)
                    
            if self.params['dir_b']:

                for key, value in self.params['dir_b'].items():
                    gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial, 0)\
                                        *gfF*test*ds(deformation = dX, element_boundary=True)
                
        return lhs

    def GetRHS(self, mesh_data, test, dX):

        ns = specialcf.normal(mesh_data['mesh'].dim)
        tE = specialcf.tangential(mesh_data['mesh'].dim)
        h = specialcf.mesh_size
        if mesh_data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if mesh_data['mesh'].dim == 2:
            gfF = GridFunction(self.fes)
        else:
            gfF = GridFunction(FacetSurface(mesh_data['mesh'], order = 0))

        rhs =  self.params['rhs']*test*ds(deformation = dX)

        if self.params['neu_d']:

            for key, value in self.params['neu_d'].items():                
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += -InnerProduct(nE, value)*gfF*test*ds(deformation = dX, element_boundary=True)

        if self.params['neu_b']:

            for key, value in self.params['neu_b'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, self.params['b']), 0,
                            InnerProduct(nE, value))*gfF*test\
                                *ds(deformation = dX, element_boundary=True)
            
        if self.params['Fneu_b']:

            for key, value in self.params['neu_b'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += -InnerProduct(nE, value)*gfF*test\
                                *ds(deformation = dX, element_boundary=True)
                
        if self.params['dir_d']:

            alpha = 5 * self.fes_order * (self.fes_order+1)
            for key, value in self.params['dir_d'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += self.params['d']*alpha/h*value*test*gfF*ds(deformation = dX, element_boundary=True) \
                    - self.params['d']*InnerProduct(nE, grad(test).Trace())*gfF*value*ds(deformation = dX, element_boundary=True)
                
        if self.params['dir_b']:

            for key, value in self.params['dir_b'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, self.params['b']), 0,
                            InnerProduct(nE, self.params['b']*value))*gfF*test\
                                *ds(deformation = dX, element_boundary=True)
                
        return rhs

    def GetMass(self, mesh_data, trial, test, dX, gfu):

        if self.params['stationary']:
            factor = 0
        else:
            factor = 1

        mass = factor/mesh_data['dt']*trial*test*ds(deformation = dX)
        mass_gfu = factor/mesh_data['dt']*self.gfu*test*ds(deformation = dX)
                
        return mass, mass_gfu
    
    def MP_and_BP(self, mesh_data, dX, gfu):

        dt = mesh_data['dt'].Get()
        tol = 1e-10

        gfu_vec = gfu.vec.FV().NumPy()[:]

        if self.params['BP'] and not self.params['MP']:

            gfu_data = np.minimum(np.maximum(gfu_vec, self.params['BP'][0]), self.params['BP'][1])  
            gfu.vec.data[:] = gfu_data

        elif self.params['MP']:
            
            gfu0 = self.params['gfu0'].FV().NumPy()[:]

            u, v = gfu.space.TnT()
            if mesh_data['mesh'].dim == 3:
                ir = IntegrationRule([(0,0), (1,0),(0,1)], [1/6, 1/6, 1/6])
            else:
                raise Exception('Not yet implemented!')
            A = BilinearForm(gfu.space, symmetric = True)
            A += u*v*ds(intrules={TRIG: ir}, deformation = dX)
            A.Assemble()
            A_old = BilinearForm(gfu.space, symmetric = True)
            A_old += u*v*ds(intrules={TRIG: ir})
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
            xsi_old1 = -dt
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
        self.gfu_save.Set(self.gfu, definedon = mesh_data['mesh'].Boundaries(self.params['domain']))

    def get_error(self, mesh_data):

        err = compute_error(data=mesh_data, gfu = self.gfu, u_ex=self.error_params['ex_sol'],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = BND)
        
        return err
    
    def set_solution(self, mesh_data, value):

        self.gfu.Set(value, definedon=mesh_data['mesh'].Boundaries(self.params['domain']))
        self.gfu_old.Set(value, definedon=mesh_data['mesh'].Boundaries(self.params['domain']))

    def get_solution(self, **kwargs):

        return self.gfu
        
    def draw_solution(self, mesh_data, **kwargs):

        if 'dX' in mesh_data:
            dX = mesh_data['dX']
        else:
            dX = GridFunction(VectorH1(mesh_data['mesh']))

        Draw(self.gfu_save, deformation = dX)       

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              It allows to solve surface diffusion-dominant equations.\n \
              It uses conforming H-1 elements of order 1.')

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with value ', str(value))

        print(60*'-', '\n')

class AdBndADR(BndADR):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.fields = 2

    def Initialize(self, mesh_data):

        self.dim = mesh_data['mesh'].dim
        
        if self.params['periodic']:
            V = Periodic(H1(mesh_data['mesh'], order = self.fes_order,
                            definedon=mesh_data['mesh'].Boundaries(self.params['domain'])))
        else:
            V = H1(mesh_data['mesh'], order = self.fes_order,
                definedon=mesh_data['mesh'].Boundaries(self.params['domain']))
            
        if mesh_data['mesh'].dim == 2:
            dV = H1(mesh_data['mesh'], order = 1,
                definedon=mesh_data['mesh'].Boundaries(self.params['domain']))
        else:
            dV = NormalFacetSurface(mesh_data['mesh'], order = 0,
                definedon=mesh_data['mesh'].Boundaries(self.params['domain']))
            
        self.fes = CompressCompound(V*dV)
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        if self.params['u0']:
            self.gfu.components[0].Set(self.params['u0'], 
                        definedon=mesh_data['mesh'].Boundaries('.*'))
        self.gfu_old.vec.data = self.gfu.vec.data

        if self.params['MP']:
            self.params['gfu0'] = self.gfu.components[0].vec.Copy()

        self.trial = self.fes.TrialFunction()
        self.test = self.fes.TestFunction()

        self.gfu_save = GridFunction(Compress(H1(mesh_data['mesh'], order = self.fes_order)))
        self.gfu_save.Set(self.gfu.components[0], definedon = mesh_data['mesh'].Boundaries(self.params['domain']))

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

            self.save_params['vtk'] = VTKOutput(mesh_data['mesh'], coefs= [self.gfu_save],
                        names=[self.params['name']], filename = file_path,
                        subdivision = self.save_params['subdivision'])

    def GetLHS(self, mesh_data, trial, test, dX):

        ns = specialcf.normal(mesh_data['mesh'].dim)
        tE = specialcf.tangential(mesh_data['mesh'].dim)
        h = specialcf.mesh_size
        if mesh_data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if mesh_data['mesh'].dim == 2:
            gfF = GridFunction(H1(mesh_data['mesh'], order = 1))
        else:
            gfF = GridFunction(FacetSurface(mesh_data['mesh'], order = 0))

        if self.params['c']:
            lhs = self.params['c']*trial[0]*test[0]*ds(deformation = dX)
        else:
            lhs = CF(0)*trial[0]*test[0]*ds(deformation = dX)

        if self.params['d']:
            lhs += self.params['d']*grad(trial[0]).Trace()*grad(test[0]).Trace()\
                *ds(deformation = dX)
            
            if self.params['dir_d']:

                alpha = 5 * self.fes_order * (self.fes_order+1)

                for key, value in self.params['dir_d'].items():

                    gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                    lhs += - self.params['d']*InnerProduct(nE, grad(trial[0]).Trace())*gfF*test[0]*ds(deformation = dX, element_boundary=True) \
                            - self.params['d']*InnerProduct(nE, grad(test[0]).Trace())*gfF*trial[0]*ds(deformation = dX, element_boundary=True)\
                            + self.params['d']*alpha/h*trial[0]*test[0]*gfF*ds(deformation = dX, element_boundary=True)

        if self.params['b']:
            lhs += -self.params['b']*grad(test[0]).Trace() * trial[0] *ds(deformation = dX)

            stab = Norm(self.params['b'])*h
            if self.params['d']:
                stab += Norm(self.params['d'])
            if self.params['c']:
                stab += Norm(self.params['c'])*h**2
            if mesh_data['mesh'].dim == 2:
                tEc = CF((-ns[1], ns[0]))
                jump_dudn = (trial[0].Trace().Deriv() - trial[1]*tEc)*nE
                jump_dvdn = (test[0].Trace().Deriv() - test[1]*tEc)*nE
            elif mesh_data['mesh'].dim == 3:
                jump_dudn = (trial[0].Trace().Deriv() - trial[1].Trace())*nE
                jump_dvdn = (test[0].Trace().Deriv() - test[1].Trace())*nE
            lhs +=  h**3/stab*InnerProduct(jump_dudn,jump_dvdn)\
                *ds(deformation = dX, element_boundary=True)

            if self.params['neu_b']:
                for key, value in self.params['neu_b'].items():
                    gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial[0], 0)\
                                        *gfF*test[0]*ds(deformation = dX, element_boundary=True)
                
            if self.params['Fneu_b']:
                for key, value in self.params['neu_b'].items():
                    gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries('.*') - mesh_data['mesh'].BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial[0], CF(0))\
                                        *gfF*test[0]*ds(deformation = dX, element_boundary=True)
                    
            if self.params['dir_b']:
                for key, value in self.params['dir_b'].items():
                    gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                    lhs += IfPos(InnerProduct(nE, self.params['b']), 
                                    InnerProduct(nE, self.params['b'])*trial[0], 0)\
                                        *gfF*test[0]*ds(deformation = dX, element_boundary=True)
                    
        return lhs

    def GetRHS(self, mesh_data, test, dX):

        ns = specialcf.normal(mesh_data['mesh'].dim)
        tE = specialcf.tangential(mesh_data['mesh'].dim)
        h = specialcf.mesh_size
        if mesh_data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        rhs =  self.params['rhs']*test[0]*ds(deformation = dX)

        if mesh_data['mesh'].dim == 2:
            gfF = GridFunction(H1(mesh_data['mesh'], order = 1,\
                                    definedon=mesh_data['mesh'].Boundaries('.*')))
        else:
            gfF = GridFunction(FacetSurface(mesh_data['mesh'], order = 0))

        if self.params['neu_d']:
            for key, value in self.params['neu_d'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += -InnerProduct(nE, value)*gfF*test[0]*ds(deformation = dX, element_boundary=True)

        if self.params['neu_b']:
            for key, value in self.params['neu_b'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, self.params['b']), 0,
                            InnerProduct(nE, value))*gfF*test[0]\
                                *ds(deformation = dX, element_boundary=True)
            
        if self.params['Fneu_b']:
            for key, value in self.params['neu_b'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += -InnerProduct(nE, value)*gfF*test[0]\
                                *ds(deformation = dX, element_boundary=True)
                
        if self.params['dir_d']:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            for key, value in self.params['dir_d'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += self.params['d']*alpha/h*value*test[0]*gfF*ds(deformation = dX, element_boundary=True)\
                    - self.params['d']*InnerProduct(nE, grad(test[0]).Trace())*gfF*value*ds(deformation = dX, element_boundary=True) 
                
        if self.params['dir_b']:
            for key, value in self.params['dir_b'].items():
                gfF.Set(1, definedon=mesh_data['mesh'].BBoundaries(key))
                rhs += -IfPos(InnerProduct(nE, self.params['b']), 0,
                            InnerProduct(nE, self.params['b']*value))*gfF*test[0]\
                                *ds(deformation = dX, element_boundary=True)
                
        return rhs

    def GetMass(self, mesh_data, trial, test, dX, gfu):

        if self.params['stationary']:
            factor = 0
        else:
            factor = 1

        mass = factor/mesh_data['dt']*trial[0]*test[0]*ds(deformation = dX)
        mass_gfu = factor/mesh_data['dt']*self.gfu.components[0]*test[0]*ds(deformation = dX)
                
        return mass, mass_gfu
    
    def MP_and_BP(self, mesh_data, dX, gfu):

        dt = mesh_data['dt'].Get()
        tol = 1e-10

        gfu_vec = gfu.components[0].vec.FV().NumPy()[:]

        if self.params['BP'] and not self.params['MP']:

            gfu_data = np.minimum(np.maximum(gfu_vec, self.params['BP'][0]), self.params['BP'][1])  
            gfu.components[0].vec.data[:] = gfu_data

        elif self.params['MP']:
            
            gfu0 = self.params['gfu0'].FV().NumPy()[:]

            u, v = gfu.components[0].space.TnT()
            if mesh_data['mesh'].dim == 3:
                ir = IntegrationRule([(0,0), (1,0),(0,1)], [1/6, 1/6, 1/6])
            else:
                raise Exception('Not yet implemented!')
            A = BilinearForm(gfu.components[0].space, symmetric = True)
            A += u*v*ds(intrules={TRIG: ir}, deformation = dX)
            A.Assemble()
            A_old = BilinearForm(gfu.components[0].space, symmetric = True)
            A_old += u*v*ds(intrules={TRIG: ir})
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
            xsi_old1 = -dt
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
            gfu.components[0].vec.data[:] = gfu_data

    def Update(self, mesh_data):

        self.gfu_old.vec.data = self.gfu.vec.data
        self.gfu_save.Set(self.gfu.components[0], definedon = mesh_data['mesh'].Boundaries(self.params['domain']))

    def get_error(self, mesh_data):

        err = compute_error(data=mesh_data, gfu = self.gfu.components[0], u_ex=self.error_params['ex_sol'],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = BND)
        
        return err
    
    def set_solution(self, mesh_data, value):

        self.gfu.components[0].Set(value, definedon=mesh_data['mesh'].Boundaries(self.params['domain']))
        self.gfu_old.components[0].Set(value, definedon=mesh_data['mesh'].Boundaries(self.params['domain']))

    def get_solution(self, **kwargs):

        return self.gfu.components[0]
        
    def draw_solution(self, mesh_data, **kwargs):

        if 'dX' in mesh_data:
            dX = mesh_data['dX']
        else:
            dX = GridFunction(VectorH1(mesh_data['mesh']))

        Draw(self.gfu_save, deformation = dX)       

    def print_info(self):

        print(60*'-')

        print('This is a general solver for a Advection-Diffusion-Reaction problem.\n \
              It allows to solve surface advection dominant equations.\n \
              It uses conforming H-1 elements of order 1.')

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with value ', str(value))

        print(60*'-', '\n')