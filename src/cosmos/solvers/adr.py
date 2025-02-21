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
        # accepted_keys = []
        # defaults = []
        # self.params_check(self.params, accepted_keys, defaults)

        self.species = []
        self.couplings = []

    def AddSpecie(self, **kwargs):

        data = self.problem.data

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['VorB', 'rhs', 'advection', 'diffusion', 'reaction',
                         'u0', 'dir_bc', 'flux_c_bc', 'flux_d_bc', 'Fflux_c_bc',
                         'MP', 'PP', 'domain']
        defaults = [VOL, CF(0.0), CF((0,)*data['mesh'].dim), CF(0.0), CF(0.0),
                    CF(0.0), {}, {}, {}, {}, False, False, '.*']
        self.params_check(params, accepted_keys, defaults)

        if data['mesh'].ne == 0 and params['VorB'] == VOL:
            raise Exception('The mesh has no volume elements! The specie is not added')

        params['marker'] = len(self.species)

        if params['VorB'] == BND:
            params['fes'] = Compress(H1(data['mesh'], order = self.fes_order, dgjumps = True, 
                 definedon=data['mesh'].Boundaries(params['domain'])))
        elif params['VorB'] == VOL:
            params['fes'] = Compress(H1(data['mesh'], order = self.fes_order, dgjumps = True, 
                 definedon=data['mesh'].Materials(params['domain'])))

        self.species.append(params)

    def AddCoupling(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['marker1', 'marker2', 'f1', 'f2']
        defaults = [0, 1, {}, {}]  
        self.params_check(params, accepted_keys, defaults)

        self.couplings.append(params)

    def initialize(self):

        data = self.problem.data

        V_s = H1(data['mesh'], order = self.fes_order, dgjumps = True, 
                 definedon=data['mesh'].Boundaries('.*'))
        V_v = H1(data['mesh'], order = self.fes_order, dgjumps = True)

        self.fes = self.species[0]['fes']

        if data['mesh'].ne == 0:
            self.fes_save = V_s
        else:
            self.fes_save = V_v

        if len(self.species)>1:        
            for sp in self.species[1:]:

                self.fes = self.fes*sp['fes']
                self.fes_save = self.fes_save*self.V_v
        else:
            self.fes = self.fes*NumberSpace(data['mesh'])
            self.fes_save = self.fes_save*NumberSpace(data['mesh'])
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.gfu_save = GridFunction(self.fes_save)

        list = []
        for i, sp in enumerate(self.species):

            if sp['VorB'] == BND:
                self.gfu.components[i].Set(sp['u0'], 
                                           definedon=data['mesh'].Boundaries(sp['domain']))
            else:
                self.gfu.components[i].Set(sp['u0'],
                                           definedon=data['mesh'].Materials(sp['domain']))

            list.append(self.gfu.components[i].vec.Copy())

        self.sol=[list]
        self.gfu_old.vec.data = self.gfu.vec.data

    def solve_step(self):

        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)

        self.u, self.v = self.fes.TnT()

        for sp in self.species:

            if sp['VorB'] == BND:

                self.__add_surface_forms__(sp)

            else:

                self.__add_bulk_forms__(sp)
        
        if len(self.species)<2:
            self.A += self.u[-1]*self.v[-1]*ds

        for cp in self.couplings:

            self.__add_coupling_forms__(cp)

        with TaskManager():

            if len(self.couplings)>0:

                Newton(self.A, self.gfu, maxit=100, printing = False)

            else:

                self.A.Assemble()
                self.F.Assemble()

                self.gfu.vec.data = self.A.mat.Inverse(self.fes.FreeDofs())*self.F.vec

        for sp in self.species:

            if sp['MP'] or sp['PP']:

                self.__mp__(sp)

    def __add_surface_forms__(self, sp):

        data = self.problem.data

        sp_up = self.params_update(sp)

        i = sp['marker']

        separator = '|'
        dir_bnd = separator.join(list(sp_up['dir_bc'].keys()))
        flux_c_bnd = separator.join(list(sp_up['flux_c_bc'].keys()))
        Fflux_c_bnd = separator.join(list(sp_up['Fflux_c_bc'].keys()))
        flux_d_bnd = separator.join(list(sp_up['flux_d_bc'].keys()))

        if sp_up['dir_bc']:
            dir_cf = sp_up['dir_bc'][dir_bnd]
        if sp_up['flux_c_bc']:
            flux_c_cf = sp_up['flux_c_bc'][flux_c_bnd]
        if sp_up['Fflux_c_bc']:
            Fflux_c_cf = sp_up['Fflux_c_bc'][Fflux_c_bnd]
        if sp_up['flux_d_bc']:
            flux_d_cf = sp_up['flux_d_bc'][flux_d_bnd]

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

        diffusion = sp_up['diffusion']*grad(self.u[i]).Trace()*grad(self.v[i]).Trace()\
            *ds(deformation = dX, definedon=data['mesh'].Boundaries(sp['domain']))
        reaction = sp_up['reaction']*self.u[i]*self.v[i]*ds(deformation = dX, 
                definedon=data['mesh'].Boundaries(sp['domain']))
        # CIP stabilization for convection part
        advection = -sp_up['advection']*grad(self.v[i]).Trace() * self.u[i] *ds(deformation = dX, 
                definedon=data['mesh'].Boundaries(sp['domain']))

        rhs =  sp_up['rhs']*self.v[i]*ds(deformation = dX,
                definedon=data['mesh'].Boundaries(sp['domain']))
        mass = 1/data['dt']*self.u[i]*self.v[i]*ds(deformation = dX, 
                definedon=data['mesh'].Boundaries(sp['domain']))
        mass_old = 1/data['dt']*self.gfu_old.components[i]*self.v[i]*ds(deformation = dX_old,
                definedon=data['mesh'].Boundaries(sp['domain']))

        if sp_up['flux_d_bc']:
            if data['mesh'].dim == 2:
                gfF = GridFunction(H1(data['mesh'], order = 1,\
                                        definedon=data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(flux_d_bnd))
            else:
                gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(flux_d_bnd))
            rhs += -InnerProduct(nE, flux_d_cf)*gfF*self.v[i]*ds(deformation = dX, element_boundary=True)
        if sp_up['flux_c_bc']:
            if data['mesh'].dim == 2:
                gfF = GridFunction(H1(data['mesh'], order = 1,\
                                        definedon=data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(flux_c_bnd))
            else:
                gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(flux_c_bnd))
            advection += IfPos(InnerProduct(nE, sp_up['advection']), 
                            InnerProduct(nE, sp_up['advection'])*self.u[i], 0)\
                                *gfF*self.v[i]*ds(deformation = dX, element_boundary=True)
            rhs += -IfPos(InnerProduct(nE, sp_up['advection']), 0,
                         InnerProduct(nE, flux_c_cf))*gfF*self.v[i]\
                            *ds(deformation = dX, element_boundary=True)
        if sp_up['Fflux_c_bc']:
            if data['mesh'].dim == 2:
                gfF = GridFunction(H1(data['mesh'], order = 1,\
                                        definedon=data['mesh'].Boundaries('.*')))
                gfF1 = GridFunction(H1(data['mesh'], order = 1,\
                                        definedon=data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(Fflux_c_bnd))
                gfF1.Set(1, definedon=data['mesh'].BBoundaries('.*') - data['mesh'].BBoundaries(Fflux_c_bnd))
            else:
                gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF1 = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(Fflux_c_bnd))
                gfF1.Set(1, definedon=data['mesh'].BBoundaries('.*') - data['mesh'].BBoundaries(Fflux_c_bnd))
            advection += IfPos(InnerProduct(nE, sp_up['advection']), 
                            InnerProduct(nE, sp_up['advection'])*self.u[i], CF(0))\
                                *gfF1*self.v[i]*ds(deformation = dX, element_boundary=True)
            rhs += -InnerProduct(nE, Fflux_c_cf)*gfF*self.v[i]\
                            *ds(deformation = dX, element_boundary=True)
        if sp_up['dir_bc']:
            if data['mesh'].dim == 2:
                gfF = GridFunction(H1(data['mesh'], order = 1,\
                                        definedon=data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(dir_bnd))
                alpha = 5 * self.fes_order * (self.fes_order+1)
                diffusion += - InnerProduct(nE, grad(self.u[i]).Trace())*gfF*self.v[i]*ds(deformation = dX, element_boundary=True) \
                        - InnerProduct(nE, grad(self.v[i]).Trace())*gfF*self.u[i]*ds(deformation = dX, element_boundary=True) \
                        + alpha/h*self.u[i]*self.v[i]*gfF*ds(deformation = dX, element_boundary=True)
                rhs += - InnerProduct(nE, grad(self.v[i]).Trace())*gfF*dir_cf*ds(deformation = dX, element_boundary=True) \
                        + alpha/h*dir_cf*self.v[i]*gfF*ds(deformation = dX, element_boundary=True)
            else:
                gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(dir_bnd))
                alpha = 5 * self.fes_order * (self.fes_order+1)
                diffusion += - InnerProduct(nE, grad(self.u[i]).Trace())*gfF*self.v[i]*ds(deformation = dX, element_boundary=True) \
                        - InnerProduct(nE, grad(self.v[i]).Trace())*gfF*self.u[i]*ds(deformation = dX, element_boundary=True) \
                        + alpha/h*self.u[i]*self.v[i]*ds(deformation = dX, definedon = data['mesh'].BBoundaries(dir_bnd))
                rhs += - InnerProduct(nE, grad(self.v[i]).Trace())*gfF*dir_cf*ds(deformation = dX, element_boundary=True) \
                        + alpha/h*dir_cf*self.v[i]*ds(deformation = dX, definedon = data['mesh'].BBoundaries(dir_bnd))

        if len(self.couplings)>0:       
            self.A += mass + reaction + diffusion + advection - rhs - mass_old
        else:
            self.A += mass + reaction + diffusion + advection
            self.F += rhs + mass_old

    def __add_bulk_forms__(self, sp):

        data = self.problem.data

        sp_up = self.params_update(sp)

        i = sp['marker']

        dir_cf = data['mesh'].BoundaryCF(sp_up['dir_bc'], default=0)
        flux_c_cf = data['mesh'].BoundaryCF(sp_up['flux_c_bc'], default=CF((0,) * data['mesh'].dim))
        Fflux_c_cf = data['mesh'].BoundaryCF(sp_up['Fflux_c_bc'], default=CF((0,) * data['mesh'].dim))
        flux_d_cf = data['mesh'].BoundaryCF(sp_up['flux_d_bc'], default=CF((0,) * data['mesh'].dim))

        separator = '|'
        dir_bnd = separator.join(list(sp_up['dir_bc'].keys()))
        flux_c_bnd = separator.join(list(sp_up['flux_c_bc'].keys()))
        Fflux_c_bnd = separator.join(list(sp_up['Fflux_c_bc'].keys()))
        flux_d_bnd = separator.join(list(sp_up['flux_d_bc'].keys()))

        n = specialcf.normal(data['mesh'].dim)
        h = specialcf.mesh_size

        if ('dX' in data) and ('dX_old' in data):
            dX = data['dX']
            dX_old = data['dX_old']
        else:
            dX = GridFunction(VectorH1(data['mesh']))
            dX_old = GridFunction(VectorH1(data['mesh']))

        diffusion = sp_up['diffusion']*grad(self.u[i])*grad(self.v[i])\
            *dx(deformation = dX)
        reaction = sp_up['reaction']*self.u[i]*self.v[i]\
            *dx(deformation = dX)
        # CIP stabilization for convection part
        S_int = 0.5*Norm(sp_up['advection']*n) # Parameter corresponding to upwind stabilization
        jump_u = n*(grad(self.u[i]) - (grad(self.u[i])).Other())
        jump_v = n*(grad(self.v[i]) - (grad(self.v[i])).Other())
        advection = -sp_up['advection']*grad(self.v[i]) * self.u[i] \
            *dx(deformation = dX)\
                    + h**2*S_int*jump_u*jump_v\
            *dx(deformation = dX, skeleton=True)
        
        mass = 1/data['dt']*self.u[i]*self.v[i]\
            *dx(deformation = dX)
        mass_old =  1/data['dt']*self.gfu_old.components[i]*self.v[i]\
            *dx(deformation = dX_old)

        rhs = sp_up['rhs']*self.v[i]*dx(deformation = dX)
        
        if sp_up['flux_d_bc']:
            rhs += -flux_d_cf*n*self.v[i]*ds(deformation = dX, definedon = data['mesh'].Boundaries(flux_d_bnd))
        if sp_up['flux_c_bc']:
            advection += IfPos(sp_up['advection']*n, sp_up['advection']*n*self.u[i], CF(0))*self.v[i]\
                *ds(deformation = dX, definedon = data['mesh'].Boundaries(flux_c_bnd))
            rhs += -IfPos(sp_up['advection']*n, CF(0), flux_c_cf*n)*self.v[i]*ds(deformation = dX, definedon = data['mesh'].Boundaries(flux_c_bnd))
        if sp_up['Fflux_c_bc']:
            advection += IfPos(sp_up['advection']*n, sp_up['advection']*n*self.u[i], CF(0))*self.v[i]\
                *ds(deformation = dX, definedon = data['mesh'].Boundaries('.*') - data['mesh'].Boundaries(Fflux_c_bnd))
            rhs += -Fflux_c_cf*n*self.v[i]*ds(deformation = dX, definedon = data['mesh'].Boundaries(Fflux_c_bnd))
        if sp_up['dir_bc']:
            alpha = 5 * self.fes_order * (self.fes_order+1)
            diffusion += - InnerProduct(n, grad(self.u[i]))*self.v[i]*ds(deformation = dX, definedon = data['mesh'].Boundaries(dir_bnd), skeleton=True) \
                - InnerProduct(n, grad(self.v[i]))*self.u[i]*ds(deformation = dX, definedon = data['mesh'].Boundaries(dir_bnd), skeleton=True) \
                + alpha/h*self.u[i]*self.v[i]*ds(deformation = dX, definedon = data['mesh'].Boundaries(dir_bnd))
            rhs += - dir_cf*InnerProduct(n, grad(self.v[i]))*ds(deformation = dX, definedon = data['mesh'].Boundaries(dir_bnd), skeleton=True) \
                + alpha/h*dir_cf*self.v[i]*ds(deformation = dX, definedon = data['mesh'].Boundaries(dir_bnd))
            
        if len(self.couplings)>0:       
            self.A += mass + reaction + diffusion + advection - rhs - mass_old
        else:
            self.A += mass + reaction + diffusion + advection
            self.F += rhs + mass_old

    def __add_coupling_forms__(self, cp):

        u1 = self.u[cp['marker1']]
        v1 = self.v[cp['marker1']]
        u2 = self.u[cp['marker2']]
        v2 = self.v[cp['marker2']]

        if self.species[cp['marker1']]['VorB'] == VOL and self.species[cp['marker2']]['VorB'] == VOL:

            if cp['f1']:

                self.A += cp['f1'](u1, u2)*v1*dx

            if cp['f2']:

                self.A += cp['f2'](u1, u2)*v2*dx

        else:
            
            if cp['f1']:

                self.A += cp['f1'](u1, u2)*v1*ds

            if cp['f2']:

                self.A += cp['f2'](u1, u2)*v2*ds

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
            sol = self.get_solution(data = data)
            vtk.Do(time=data['t_array'][i])
            data['mesh'].UnsetDeformation()

    def compute_error(self, u_ex, norm):

        data = self.problem.data

        err = [[] for i in range(len(self.species))]
        ns = specialcf.normal(data['mesh'].dim)
        Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

        if norm == 'H1':

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

                    if self.species[j]['VorB'] == BND:

                        aux = InnerProduct(grad(self.gfu.components[j]).Trace()-gradient(u_ex[j], Ps), \
                                            grad(self.gfu.components[j]).Trace()-gradient(u_ex[j], Ps))

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                    else:

                        aux = InnerProduct(grad(self.gfu.components[j])-gradient(u_ex[j], Id(data['mesh'].dim)), \
                                            grad(self.gfu.components[j])-gradient(u_ex[j], Id(data['mesh'].dim)))

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                data['mesh'].UnsetDeformation()

        elif norm == 'L2':

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

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                    else:

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2)))

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