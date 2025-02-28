from ngsolve import *
from cosmos.solvers.base_solver import BaseSolver
from cosmos.utils.manufactured_solution_tools import gradient
from ngsolve.webgui import Draw
from ngsolve.solvers import *
import time
import numpy as np
import scipy.sparse as scipy

class ADRSolver(BaseSolver):

    def __init__(self, **kwargs):

        super().__init__(fes_order=1, **kwargs)

        if len(kwargs)>1:
            raise Exception('Base Advection-Diffusion-Reaction simulations do \
                            not accept additional keywords as contructor')
        
        accepted_keys = ['linear']
        defaults = [False]
        self.params_check(self.params, accepted_keys, defaults)

        self.species = []
        self.couplings = []

    def AddSpecie(self, **kwargs):

        data = self.problem.data

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['VorB', 'rhs', 'advection', 'diffusion', 'reaction',
                         'u0', 'neu_d', 'neu_b', 'dir_d', 'dir_b', 'Fneu_b',
                         'MP', 'PP', 'domain', 'stationary']
        defaults = [VOL, CF(0.0), CF((0,)*data['mesh'].dim), CF(0.0), CF(0.0),
                    CF(0.0), {}, {}, {}, {}, {},
                    False, False, '.*', False]
        self.params_check(params, accepted_keys, defaults)

        if data['mesh'].ne == 0 and params['VorB'] == VOL:
            raise Exception('The mesh has no volume elements! The specie is not added')

        params['marker'] = len(self.species)

        if params['VorB'] == BND:
            params['fes'] = Compress(H1(data['mesh'], order = self.fes_order, dgjumps = True, 
                 definedon=data['mesh'].Boundaries(params['domain'])))
        elif params['VorB'] == VOL:
            params['fes'] = Compress(H1(data['mesh'], order = self.fes_order, dgjumps = True,
                          definedon = params['domain']))

        self.species.append(params)

    def AddNonlinearity(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['marker0', 'markers', 'f', 'VorB', 'domain']
        defaults = [0, 0, {}, VOL, '.*']  
        self.params_check(params, accepted_keys, defaults)

        self.couplings.append(params)

    def initialize(self):

        data = self.problem.data

        if data['mesh'].ne == 0:
            V = H1(data['mesh'], order = self.fes_order,
                               definedon=data['mesh'].Boundaries('.*'))
        else:
            V = H1(data['mesh'], order = self.fes_order)
       
        for i, sp in enumerate(self.species):

            if i == 0:
                self.fes = sp['fes']
                self.fes_save = V
            else:
                self.fes = self.fes*sp['fes']
                self.fes_save = self.fes_save*V

        if len(self.species)<2:
            self.fes = self.fes*NumberSpace(data['mesh'])
            self.fes_save = self.fes_save*NumberSpace(data['mesh'])
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.gfu_save = GridFunction(self.fes_save)

        list = []
        for i, sp in enumerate(self.species):

            if sp['VorB'] == BND:
                self.gfu.components[i].Set(sp['u0'], 
                        definedon=data['mesh'].Boundaries('.*'))
            elif sp['VorB'] == VOL:
                self.gfu.components[i].Set(sp['u0'])

            list.append(self.gfu.components[i].vec.Copy())

        self.sol=[list]
        self.gfu_old.vec.data = self.gfu.vec.data

    def solve_step(self):

        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)

        self.u, self.v = self.fes.TnT()

        for sp in self.species:

            if sp['VorB'] == BND:

                lhs, rhs, mass, mass_old = self.__add_surface_forms__(sp)

            else:

                lhs, rhs, mass, mass_old = self.__add_bulk_forms__(sp)

            if self.params['linear']:

                if sp['stationary']:

                    self.A += lhs
                    self.F += rhs

                else:

                    self.A += lhs + mass
                    self.F += rhs + mass_old

            else:

                if sp['stationary']:

                    self.A += lhs - rhs

                else:

                    self.A += lhs + mass - rhs - mass_old
        
        if len(self.species)<2:
            self.A += self.u[-1]*self.v[-1]*ds

        for cp in self.couplings:

            self.__add_nonlinear_forms__(cp)

        with TaskManager():

            if self.params['linear']:

                self.A.Assemble()
                self.F.Assemble()

                self.gfu.vec.data = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())*self.F.vec

            else:

                Newton(self.A, self.gfu, maxit=20, printing = False)

        for sp in self.species:

            if sp['MP'] or sp['PP']:

                self.__mp__(sp)

    def __add_surface_forms__(self, sp):

        data = self.problem.data

        sp_up = self.params_update(sp)

        i = sp['marker']

        ns = specialcf.normal(data['mesh'].dim)
        tE = specialcf.tangential(data['mesh'].dim)
        h = specialcf.mesh_size
        if data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        if ('dX' in data) and ('dX_old' in data):
            dX = data['dX']
            dX_old = data['dX_old']
        else:
            dX = GridFunction(VectorH1(data['mesh']))
            dX_old = GridFunction(VectorH1(data['mesh']))

        lhs = sp_up['diffusion']*grad(self.u[i]).Trace()*grad(self.v[i]).Trace()\
            *ds(deformation = dX)
        lhs += sp_up['reaction']*self.u[i]*self.v[i]*ds(deformation = dX)
        # CIP stabilization for convection part
        S_int = 0.5*Norm(sp_up['advection']*nE) # Parameter corresponding to upwind stabilization
        jump_u = nE*(grad(self.u[i]).Trace() - (grad(self.u[i])).Other().Trace())
        jump_v = nE*(grad(self.v[i]).Trace() - (grad(self.v[i])).Other().Trace())
        lhs += -sp_up['advection']*grad(self.v[i]).Trace() * self.u[i] *ds(deformation = dX)\

        rhs =  sp_up['rhs']*self.v[i]*ds(deformation = dX)
        mass = 1/data['dt']*self.u[i]*self.v[i]*ds(deformation = dX)
        mass_old = 1/data['dt']*self.gfu_old.components[i]*self.v[i]*ds(deformation = dX_old)

        if sp_up['neu_d']:

            for key, value in sp_up['neu_d'].items():

                if data['mesh'].dim == 2:
                    gfF = GridFunction(H1(data['mesh'], order = 1,\
                                            definedon=data['mesh'].Boundaries('.*')))
                else:
                    gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(key))

                rhs += -InnerProduct(nE, value)*gfF*self.v[i]*ds(deformation = dX, element_boundary=True)

        if sp_up['neu_b']:

            for key, value in sp_up['neu_b'].items():

                if data['mesh'].dim == 2:
                    gfF = GridFunction(sp_up['fes'])
                else:
                    gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(key))

                lhs += IfPos(InnerProduct(nE, sp_up['advection']), 
                                InnerProduct(nE, sp_up['advection'])*self.u[i], 0)\
                                    *gfF*self.v[i]*ds(deformation = dX, element_boundary=True)
                rhs += -IfPos(InnerProduct(nE, sp_up['advection']), 0,
                            InnerProduct(nE, value))*gfF*self.v[i]\
                                *ds(deformation = dX, element_boundary=True)
            
        if sp_up['Fneu_b']:

            for key, value in sp_up['neu_b'].items():

                if data['mesh'].dim == 2:
                    gfF = GridFunction(H1(data['mesh'], order = 1,\
                                            definedon=data['mesh'].Boundaries('.*')))
                    gfF1 = GridFunction(H1(data['mesh'], order = 1,\
                                            definedon=data['mesh'].Boundaries('.*')))
                else:
                    gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                    gfF1 = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(key))
                gfF1.Set(1, definedon=data['mesh'].BBoundaries('.*') - data['mesh'].BBoundaries(key))

                lhs += IfPos(InnerProduct(nE, sp_up['advection']), 
                                InnerProduct(nE, sp_up['advection'])*self.u[i], CF(0))\
                                    *gfF1*self.v[i]*ds(deformation = dX, element_boundary=True)
                rhs += -InnerProduct(nE, value)*gfF*self.v[i]\
                                *ds(deformation = dX, element_boundary=True)
                
        if sp_up['dir_d']:

            alpha = sqrt((self.fes_order +1)*(self.fes_order + data['mesh'].dim)/data['mesh'].dim)
            alpha = 5 * self.fes_order * (self.fes_order+1)

            for key, value in sp_up['dir_d'].items():

                if data['mesh'].dim == 2:
                    gfF = GridFunction(sp_up['fes'])
                else:
                    gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(key))
                lhs += - sp_up['diffusion']*InnerProduct(nE, grad(self.u[i]).Trace())*gfF*self.v[i]*ds(deformation = dX, element_boundary=True) \
                        - sp_up['diffusion']*InnerProduct(nE, grad(self.v[i]).Trace())*gfF*self.u[i]*ds(deformation = dX, element_boundary=True)\
                        + sp_up['diffusion']*alpha/h*self.u[i]*self.v[i]*ds(deformation = dX, definedon = data['mesh'].BBoundaries(key))
                rhs += - sp_up['diffusion']*InnerProduct(nE, grad(self.v[i]).Trace())*gfF*value*ds(deformation = dX, element_boundary=True)\
                    + sp_up['diffusion']*alpha/h*value*self.v[i]*ds(deformation = dX, definedon = data['mesh'].BBoundaries(key))
                
        if sp_up['dir_b']:

            for key, value in sp_up['dir_b'].items():

                if data['mesh'].dim == 2:
                    gfF = GridFunction(H1(data['mesh'], order = 1,\
                                            definedon=data['mesh'].Boundaries('.*')))
                else:
                    gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(key))

                lhs += IfPos(InnerProduct(nE, sp_up['advection']), 
                                InnerProduct(nE, sp_up['advection'])*self.u[i], 0)\
                                    *gfF*self.v[i]*ds(deformation = dX, element_boundary=True)
                rhs += -IfPos(InnerProduct(nE, sp_up['advection']), 0,
                            InnerProduct(nE, sp_up['advection']*value))*gfF*self.v[i]\
                                *ds(deformation = dX, element_boundary=True)
                
        return lhs, rhs, mass, mass_old

    def __add_bulk_forms__(self, sp):

        data = self.problem.data

        sp_up = self.params_update(sp)

        i = sp['marker']

        n = specialcf.normal(data['mesh'].dim)
        h = specialcf.mesh_size

        if ('dX' in data) and ('dX_old' in data):
            dX = data['dX']
            dX_old = data['dX_old']
        else:
            dX = GridFunction(VectorH1(data['mesh']))
            dX_old = GridFunction(VectorH1(data['mesh']))

        lhs = sp_up['diffusion']*grad(self.u[i])*grad(self.v[i])\
            *dx(deformation = dX)
        lhs += sp_up['reaction']*self.u[i]*self.v[i]\
            *dx(deformation = dX)
        # CIP stabilization for convection part
        S_int = 0.5*Norm(sp_up['advection']*n) # Parameter corresponding to upwind stabilization
        jump_u = n*(grad(self.u[i]) - (grad(self.u[i])).Other())
        jump_v = n*(grad(self.v[i]) - (grad(self.v[i])).Other())
        lhs += -sp_up['advection']*grad(self.v[i]) * self.u[i] \
            *dx(deformation = dX)\
                    + h**2*S_int*jump_u*jump_v\
            *dx(deformation = dX, skeleton=True)
        
        mass = 1/data['dt']*self.u[i]*self.v[i]\
            *dx(deformation = dX)
        mass_old =  1/data['dt']*self.gfu_old.components[i]*self.v[i]\
            *dx(deformation = dX_old)

        rhs = sp_up['rhs']*self.v[i]*dx(deformation = dX)
        
        if sp_up['neu_d']:

            for key, value in sp_up['neu_d'].items():
                rhs += -value*n*self.v[i]*ds(deformation = dX, definedon = key)

        if sp_up['neu_b']:

            for key, value in sp_up['neu_b'].items():
                lhs += IfPos(sp_up['advection']*n, sp_up['advection']*n*self.u[i], CF(0))*self.v[i]\
                    *ds(deformation = dX, definedon = key)
                rhs += -IfPos(sp_up['advection']*n, CF(0), value*n)*self.v[i]*ds(deformation = dX, definedon = key)

        if sp_up['Fneu_b']:

            for key, value in sp_up['Fneu_b'].items():
                lhs += IfPos(sp_up['advection']*n, sp_up['advection']*n*self.u[i], CF(0))*self.v[i]\
                    *ds(deformation = dX, definedon = data['mesh'].Boundaries('.*') - data['mesh'].Boundaries(key))
                rhs += -value*n*self.v[i]*ds(deformation = dX, definedon = data['mesh'].Boundaries(key))

        if sp_up['dir_d']:

            alpha = (self.fes_order +1)*(self.fes_order + data['mesh'].dim)/data['mesh'].dim
            alpha = 5 * self.fes_order * (self.fes_order+1)

            for key, value in sp_up['dir_d'].items():
                lhs += - sp_up['diffusion']*InnerProduct(n, grad(self.u[i]))*self.v[i]*ds(deformation = dX, definedon = key, skeleton=True) \
                    - sp_up['diffusion']*InnerProduct(n, grad(self.v[i]))*self.u[i]*ds(deformation = dX, definedon = key, skeleton=True) \
                    + sp_up['diffusion']*alpha/h*self.u[i]*self.v[i]*ds(deformation = dX, definedon = key)
                rhs += - sp_up['diffusion']*InnerProduct(n, grad(self.v[i]))*value*ds(deformation = dX, definedon = key, skeleton=True) \
                    + sp_up['diffusion']*alpha/h*value*self.v[i]*ds(deformation = dX, definedon = key)

        if sp_up['dir_b']:

            for key, value in sp_up['dir_b'].items():
                lhs += IfPos(sp_up['advection']*n, sp_up['advection']*n*self.u[i], CF(0))*self.v[i]\
                    *ds(deformation = dX, definedon = key)
                rhs += -IfPos(sp_up['advection']*n, CF(0), sp_up['advection']*value*n)*self.v[i]*ds(deformation = dX, definedon = key)
            
        return lhs, rhs, mass, mass_old

    def __add_nonlinear_forms__(self, cp):

        trial = [self.u[i] for i in cp['markers']]
        test = self.v[cp['marker0']]

        if cp['VorB'] == BND:

            self.A += cp['f'](trial)*test*ds(definedon=cp['domain'])

        else:

            self.A += cp['f'](trial)*test*dx(definedon=cp['domain'])

    def __mp__(self, sp):

        i = sp['marker']

        data = self.problem.data
        dt = data['dt'].Get()
        tol = 1e-10

        gfu = self.gfu.components[i].vec.FV().NumPy()[:]

        if 'dX' in data and 'dX_old' in data:

            dX = data['dX']

            if sp['PP']:
                gfu0 = self.gfu.components[i].vec.FV().NumPy()[:]
                dX_old = dX
            else:
                gfu0 = self.sol[0][i].FV().NumPy()[:]
                dX_old = GridFunction(VectorH1(data['mesh']))

        else:
            gfu0 = gfu
            dX = GridFunction(VectorH1(data['mesh']))
            dX_old = dX

        if sp['VorB'] == BND:
            u, v = sp['fes'].TnT()
            ir = IntegrationRule([(0,0), (1,0),(0,1)], [1/6, 1/6, 1/6])
            A = BilinearForm(sp['fes'], symmetric = True)
            A += u*v*ds(deformation = dX, intrules={TRIG:ir})
            A.Assemble()
            A_old = BilinearForm(sp['fes'], symmetric = True)
            A_old += u*v*ds(deformation = dX_old, intrules={TRIG:ir})
            A_old.Assemble()
        else:
            u, v = sp['fes'].TnT()
            ir = IntegrationRule([(0,0), (1,0),(0,1)], [1/6, 1/6, 1/6])
            A = BilinearForm(sp['fes'], symmetric = True)
            A += u*v*dx(intrules={TRIG: ir}, deformation = dX)
            A.Assemble()
            A_old = BilinearForm(sp['fes'], symmetric = True)
            A_old += u*v*dx(intrules={TRIG: ir}, deformation = dX_old)
            A_old.Assemble()

        rows,cols,vals = A.mat.COO()
        Adiag = scipy.csr_matrix((vals,(rows,cols))).diagonal()
        rows,cols,vals = A_old.mat.COO()
        Adiag_old = scipy.csr_matrix((vals,(rows,cols))).diagonal()

        prod0 = np.dot(gfu0, Adiag_old)

        def F(xsi):

            result = 0

            temp = gfu + dt * xsi

            result -= prod0
            result += np.sum(Adiag[temp > 0] * temp[temp > 0])

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
        gfu_data = np.maximum(gfu + threshold, 0) 
        self.gfu.components[i].vec.data[:] = gfu_data

    def update(self):

        self.gfu_old.vec.data = self.gfu.vec.data

        list = []
        for i in range(len(self.species)):
            list.append(self.gfu.components[i].vec.Copy())
        self.sol.append(list)

    def set_solution(self, values, **kwargs):

        data = self.problem.data

        for i, sp in enumerate(self.species):

            if sp['VorB'] == BND:
                self.gfu.components[i].Set(values[i], definedon=data['mesh'].Boundaries(sp['domain']))
                self.gfu_old.components[i].Set(values[i], definedon=data['mesh'].Boundaries(sp['domain']))
            else:
                self.gfu.components[i].Set(values[i], definedon=data['mesh'].Materials(sp['domain']))
                self.gfu_old.components[i].Set(values[i], definedon=data['mesh'].Materials(sp['domain']))

    def get_solution(self, **kwargs):

        data = self.problem.data

        for i, sp in enumerate(self.species):

            if sp['VorB'] == BND:
                self.gfu_save.components[i].Set(self.gfu.components[i],
                        definedon=data['mesh'].Boundaries(sp['domain']))
            else:
                self.gfu_save.components[i].Set(self.gfu.components[i],
                        definedon=data['mesh'].Materials(sp['domain'])) 

        return [self.gfu_save.components[i] for i in range(len(self.species))]
        
    def draw_solution(self, **kwargs):

        data = self.problem.data

        if 'dX' in data:
            dX = data['dX']
        else:
            dX = GridFunction(VectorH1(data['mesh']))

        sol = self.get_solution()

        for i, sp in enumerate(self.species):

            Draw(sol[i], deformation = dX)

    def save_solution(self, **kwargs):

        data = self.problem.data

        params = kwargs
        accepted_keys = ['filename', 'subdivision', 'n_steps']
        defaults = ['sol', 1, 100]  
        self.params_check(params, accepted_keys, defaults)

        vtk = VTKOutput(data['mesh'],
                        coefs= [self.gfu_save.components[i] for i in range(len(self.species))],
                        names=["adr_sol" + str(i) for i in range(len(self.species))],
                        filename=params['filename'],
                        subdivision=params['subdivision'])
        
        tot_steps = len(self.sol)
        jump = max(int(tot_steps/params['n_steps']), 1)

        for i in range(0, tot_steps, jump):

            sol_i = self.sol[i]

            for j, sp in enumerate(self.species):

                self.gfu.components[j].vec.data = sol_i[j].data

            if 'dX' in data:
                data['dX'].vec.data = data['dX_array'][i].data
                dX = data['dX']
            else:
                dX = GridFunction(VectorH1(data['mesh']))
     
            data['mesh'].SetDeformation(dX)
            sol = self.get_solution()
            vtk.Do(time=data['t_array'][i])
            data['mesh'].UnsetDeformation()

    def compute_error(self, u_ex, norm):

        data = self.problem.data

        err = [[] for i in range(len(self.species))]
        ns = specialcf.normal(data['mesh'].dim)
        Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

        if norm == 'L2':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])
                if 'dX' in data:
                    data['dX'].vec.data = data['dX_array'][i].data
                    dX = data['dX']
                else:
                    dX = GridFunction(VectorH1(data['mesh']))
                data['mesh'].SetDeformation(dX)

                for j in range(len(self.species)):

                    self.gfu.components[j].vec.data = sol_i[j].data

                    aux = InnerProduct(self.gfu.components[j]-u_ex[j], self.gfu.components[j]-u_ex[j])

                    if self.species[j]['VorB'] == BND:

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, 
                                                     VOL_or_BND=BND, 
                                                     definedon = data['mesh'].Boundaries(self.species[j]['domain']))))

                    else:

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2,
                                        definedon = data['mesh'].Materials(self.species[j]['domain']))))

                data['mesh'].UnsetDeformation()

        else:

            raise ValueError('The norm given is not implemented')
        
        if len(self.species) >1:
            return err
        else:

            return err[0]

    def print_info(self):

        print(60*'-')

        print('This is a solver for a coupled Bulk-Surface Advection-Diffusion-Reaction problem')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with type ', type(value))

        print(60*'-', '\n')