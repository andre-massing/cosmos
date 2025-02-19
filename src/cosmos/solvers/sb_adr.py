from ngsolve import *
from cosmos.solvers.base_solver import BaseSolver
from cosmos.utils.manufactured_solution_tools import gradient
from ngsolve.webgui import Draw
from ngsolve.solvers import *
import time

class SurfaceBulkADRSolver(BaseSolver):

    def __init__(self, fes_order=1, **kwargs):

        super().__init__(fes_order=fes_order, **kwargs)

        if len(kwargs)>1:
            raise Exception('SurfaceBulk simulations do not accept additional\
                            keywords as contructor')
        # accepted_keys = []
        # defaults = []
        # self.params_check(self.params, accepted_keys, defaults)

        self.species = []
        self.couplings = []

    def attach_bulk_adr(self, **kwargs):

        data = self.problem.data

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['rhs', 'advection', 'diffusion', 'reaction', 'u0', 'flux_c_bc', 'flux_d_bc']
        defaults = [CF(0.0), CF((0,)*data['mesh'].dim), CF(0.0), CF(0.0), CF(0.0), {}, {}]  
        self.params_check(params, accepted_keys, defaults)

        params['type'] = VOL

        self.species.append(params)

    def attach_surface_adr(self, **kwargs):

        data = self.problem.data

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['rhs', 'advection', 'diffusion', 'reaction', 'u0']
        defaults = [CF(0.0), CF((0,)*data['mesh'].dim), CF(0.0), CF(0.0), CF(0.0)]  
        self.params_check(params, accepted_keys, defaults)

        params['type'] = BND

        self.species.append(params)

    def add_coupling(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['mrk1', 'mrk2', 'f1', 'f2']
        defaults = [0, 1, {}, {}]  
        self.params_check(params, accepted_keys, defaults)

        self.couplings.append(params)

    def initialize(self):

        data = self.problem.data

        if len(self.species)<2:
            raise Exception('Being a coupled simulation at least two species must be defined using the methods -attach_surface_adr- and -attach_bulk_adr-')

        V_s = H1(data['mesh'], order = self.fes_order, dgjumps = True, definedon=data['mesh'].Boundaries('.*'))
        V_vol = H1(data['mesh'], order = self.fes_order, dgjumps = True)
        V = H1(data['mesh'], order = self.fes_order)

        if self.species[0]['type'] == BND:
            self.fes = V_s
        else:
            self.fes = V_vol
        self.fes_vol = V
        
        for sp in self.species[1:]:

            if sp['type'] == BND:
                self.fes = self.fes*V_s
            else:
                self.fes = self.fes*V_vol

            self.fes_vol = self.fes_vol*V
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.gfu_vol = GridFunction(self.fes_vol)

        list = []
        for i, sp in enumerate(self.species):

            if sp['type'] == BND:
                self.gfu.components[i].Set(sp['u0'], definedon=data['mesh'].Boundaries('.*'))
            else:
                self.gfu.components[i].Set(sp['u0'])

            list.append(self.gfu.components[i].vec.Copy())

        self.sol=[list]
        self.gfu_old.vec.data = self.gfu.vec.data

    def solve_step(self):

        self.A = BilinearForm(self.fes)

        self.u, self.v = self.fes.TnT()

        for i, sp in enumerate(self.species):

            if sp['type'] == BND:

                self.__add_surface_adr_forms__(i, sp)

            else:

                self.__add_bulk_adr_forms__(i, sp)

        for i, cp in enumerate(self.couplings):

            self.__add_coupling_forms__(cp)

        with TaskManager():
                
                Newton(self.A, self.gfu, maxit=100, printing = False)

    def __add_surface_adr_forms__(self, i, sp):

        data = self.problem.data

        sp_up = self.params_update(sp)

        diffusion = sp_up['diffusion']*grad(self.u[i]).Trace()*grad(self.v[i]).Trace()*ds
        reaction = sp_up['reaction']*self.u[i]*self.v[i]*ds
        # CIP stabilization for convection part
        advection = -sp_up['advection']*grad(self.v[i]).Trace() * self.u[i] *ds
        rhs =  - sp_up['rhs']*self.v[i]*ds


        mass = 1/data['dt']*self.u[i]*self.v[i]*ds
        mass_old = - 1/data['dt']*self.gfu_old.components[i]*self.v[i]*ds
        
        self.A += reaction + diffusion + advection + rhs + mass + mass_old

    def __add_bulk_adr_forms__(self, i, sp):

        data = self.problem.data

        sp_up = self.params_update(sp)

        flux_c_cf = data['mesh'].BoundaryCF(sp_up['flux_c_bc'], default=CF((0,) * data['mesh'].dim))
        flux_d_cf = data['mesh'].BoundaryCF(sp_up['flux_d_bc'], default=CF((0,) * data['mesh'].dim))

        separator = '|'
        flux_c_bnd = separator.join(list(sp_up['flux_c_bc'].keys()))
        flux_d_bnd = separator.join(list(sp_up['flux_d_bc'].keys()))

        n = specialcf.normal(data['mesh'].dim)
        h = specialcf.mesh_size

        diffusion = sp_up['diffusion']*grad(self.u[i])*grad(self.v[i])*dx
        reaction = sp_up['reaction']*self.u[i]*self.v[i]*dx
        # CIP stabilization for convection part
        S_int = 0.5*Norm(sp_up['advection']*n) # Parameter corresponding to upwind stabilization
        jump_u = n*(grad(self.u[i]) - (grad(self.u[i])).Other())
        jump_v = n*(grad(self.v[i]) - (grad(self.v[i])).Other())
        advection = -sp_up['advection']*grad(self.v[i]) * self.u[i] *dx \
            + IfPos(sp_up['advection']*n, sp_up['advection']*n*self.u[i], CF(0))*self.v[i]*ds \
            + h**2*S_int*jump_u*jump_v*dx(skeleton=True)
        
        rhs = - sp_up['rhs']*self.v[i]*dx
        if sp_up['flux_d_bc']:
            rhs += flux_d_cf*n*self.v[i]*ds(definedon = data['mesh'].Boundaries(flux_d_bnd))
        if sp_up['flux_c_bc']:
            rhs += IfPos(sp_up['advection']*n, CF((0,) * data['mesh'].dim), flux_c_cf)\
            *n*self.v[i]*ds(definedon = data['mesh'].Boundaries(flux_c_bnd))
        mass = 1/data['dt']*self.u[i]*self.v[i]*dx
        mass_old =  - 1/data['dt']*self.gfu_old.components[i]*self.v[i]*dx
        
        self.A += reaction + diffusion + advection + rhs + mass + mass_old

    def __add_coupling_forms__(self, cp):

        u1 = self.u[cp['mrk1']]
        v1 = self.v[cp['mrk1']]
        u2 = self.u[cp['mrk2']]
        v2 = self.v[cp['mrk2']]

        if self.species[cp['mrk1']]['type'] == VOL and self.species[cp['mrk2']]['type'] == VOL:

            if cp['f1']:

                self.A += cp['f1'](u1, u2)*v1*dx

            if cp['f2']:

                self.A += cp['f2'](u1, u2)*v2*dx

        else:
            
            if cp['f1']:

                self.A += cp['f1'](u1, u2)*v1*ds

            if cp['f2']:

                self.A += cp['f2'](u1, u2)*v2*ds

    def update(self):

        self.gfu_old.vec.data = self.gfu.vec.data

        list = []
        for i in range(len(self.species)):
            list.append(self.gfu.components[i].vec.Copy())
        self.sol.append(list)

    def set_solution(self, values, **kwargs):

        data = self.problem.data

        for i, sp in enumerate(self.species):

            if sp['type'] == BND:
                self.gfu.components[i].Set(values[i], definedon=data['mesh'].Boundaries('.*'))
                self.gfu_old.components[i].Set(values[i], definedon=data['mesh'].Boundaries('.*'))
            else:
                self.gfu.components[i].Set(values[i])
                self.gfu_old.components[i].Set(values[i])

    def get_solution(self):

        data = self.problem.data

        if data['mesh'].ne == 0:
            return self.gfu.components
        else:
            for i, sp in enumerate(self.species):
                if sp['type'] == BND:
                    self.gfu_vol.components[i].Set(self.gfu.components[i], definedon=data['mesh'].Boundaries('.*'))
                else:
                    self.gfu_vol.components[i].Set(self.gfu.components[i]) 

            return self.gfu_vol.components
        
    def draw_solution(self, **kwargs):

        sol = self.get_solution()

        for i, sp in enumerate(self.species):
            
            Draw(sol[i])

    def save_solution(self, **kwargs):

        data = self.problem.data

        params = kwargs
        accepted_keys = ['filename', 'subdivision', 'n_steps']
        defaults = ['sol', 1, 100]  
        self.params_check(params, accepted_keys, defaults)

        vtk = VTKOutput(data['mesh'],
                        coefs= [self.gfu_vol.components[i] for i in range(len(self.species))],
                        names=["adr_sol" + str(i) for i in range(len(self.species))],
                        filename=params['filename'],
                        subdivision=params['subdivision'])
        
        tot_steps = len(self.sol)
        jump = max(int(tot_steps/params['n_steps']), 1)

        for i in range(0, tot_steps, jump):

            sol_i = self.sol[i]

            for j, sp in enumerate(self.species):

                self.gfu.components[j].vec.data = sol_i[j].data

            sol = self.get_solution()
            vtk.Do(time=data['t_array'][i])

    def compute_error(self, u_ex, norm):

        data = self.problem.data

        err = [[] for i in range(len(self.species))]
        ns = specialcf.normal(data['mesh'].dim)
        Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

        if norm == 'H1':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])

                for j in range(len(self.species)):

                    self.gfu.components[j].vec.data = sol_i[j].data

                    if self.species[j]['type'] == BND:

                        aux = InnerProduct(grad(self.gfu.components[j]).Trace()-gradient(u_ex[j], Ps), \
                                            grad(self.gfu.components[j]).Trace()-gradient(u_ex[j], Ps))

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                    else:

                        aux = InnerProduct(grad(self.gfu.components[j])-gradient(u_ex[j], Id(data['mesh'].dim)), \
                                            grad(self.gfu.components[j])-gradient(u_ex[j], Id(data['mesh'].dim)))

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

        elif norm == 'L2':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])

                for j in range(len(self.species)):

                    self.gfu.components[j].vec.data = sol_i[j].data

                    aux = InnerProduct(self.gfu.components[j]-u_ex[j], self.gfu.components[j]-u_ex[j])

                    if self.species[j]['type'] == BND:

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                    else:

                        err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2)))


        else:

            raise ValueError('The norm given is not implemented')

        return err

    def print_info(self):

        print(60*'-')

        print('This is a solver for a coupled Bulk-Surface Advection-Diffusion-Reaction problem')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with type ', type(value))

        print(60*'-', '\n')