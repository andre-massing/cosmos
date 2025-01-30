from ngsolve import *
from cosmos.solvers.base_solver import BaseSolver
from cosmos.utils.manufactured_solution_tools import gradient
from ngsolve.webgui import Draw

class BulkADRSolver(BaseSolver):

    def __init__(self, fes_order=1, **kwargs):

        super().__init__(fes_order=fes_order, **kwargs)

        accepted_keys = ['rhs', 'advection', 'diffusion', 'reaction',
                         'u0', 'dir_bc', 'flux_c_bc', 'flux_d_bc']
        defaults = [CF(0.0), CF(0.0), CF(0.0), CF(0.0),
                    CF(0.0), {}, {}, {}]
        self.params_check(self.params, accepted_keys, defaults)

    def initialize(self):

        data = self.problem.data

        for bc in ['dir_bc', 'flux_c_bc', 'flux_d_bc']:

            if not set(list(self.params[bc].keys())) <= set(data['boundary_mrk'] + ['.*']):
                raise ValueError('Boundary conditions are imposed on\
                                boundary regions which name is not in the \
                                mesh boundary list')

        separator = '|'
        dir_bnd = separator.join(list(self.params['dir_bc'].keys()))

        if self.params['dir_bc']:
            V = H1(data['mesh'], order = self.fes_order, dirichlet = dir_bnd, dgjumps = True)
        else:
            V = H1(data['mesh'], order = self.fes_order, dgjumps = True)
        self.fes = V
       
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.gfu.Set(self.params['u0'])
        self.sol=[self.gfu.vec.Copy()]
        self.gfu_old.vec.data = self.gfu.vec.data
    
    def solve_step(self):

        data = self.problem.data

        params = self.params_update(self.params)
        if params['advection'].dim == 1:
            params['advection'] = CF((0,)*data['mesh'].dim)

        dir_cf = data['mesh'].BoundaryCF(params['dir_bc'], default=0)
        flux_c_cf = data['mesh'].BoundaryCF(params['flux_c_bc'], default=CF((0,) * data['mesh'].dim))
        flux_d_cf = data['mesh'].BoundaryCF(params['flux_d_bc'], default=CF((0,) * data['mesh'].dim))

        separator = '|'
        dir_bnd = separator.join(list(params['dir_bc'].keys()))
        flux_c_bnd = separator.join(list(params['flux_c_bc'].keys()))
        flux_d_bnd = separator.join(list(params['flux_d_bc'].keys()))

        u, v = self.fes.TnT()

        n = specialcf.normal(data['mesh'].dim)
        h = specialcf.mesh_size

        A = BilinearForm(self.fes)
        diffusion = params['diffusion']*grad(u)*grad(v)*dx
        reaction = params['reaction']*u*v*dx
        # CIP stabilization for convection part
        S_int = 0.5*Norm(params['advection']*n) # Parameter corresponding to upwind stabilization
        jump_u = n*(grad(u) - (grad(u)).Other())
        jump_v = n*(grad(v) - (grad(v)).Other())
        advection = -params['advection']*grad(v) * u *dx \
            + IfPos(params['advection']*n, params['advection']*n*u, CF(0))*v*ds(definedon = data['mesh'].Boundaries(flux_c_bnd)) \
            + h**2*S_int*jump_u*jump_v*dx(skeleton=True) 
        
        A += reaction + diffusion + advection
        
        F = LinearForm(self.fes)
        F += params['rhs']*v*dx
        F += -flux_d_cf*n*v*ds(definedon = data['mesh'].Boundaries(flux_d_bnd))
        F += -IfPos(params['advection']*n, CF(0), flux_c_cf*n)*v*ds(definedon = data['mesh'].Boundaries(flux_c_bnd))

        M = BilinearForm(self.fes, symmetric = True)
        M += 1/data['dt']*u*v*dx

        with TaskManager():
            A.Assemble()
            F.Assemble()
            M.Assemble()

            res = F.vec
            res += M.mat*self.gfu_old.vec

            Mstar = A.mat.CreateMatrix()
            Mstar.AsVector().data = M.mat.AsVector() + A.mat.AsVector()
            if params['dir_bc']: 
                self.gfu.Set(dir_cf, definedon = data['mesh'].Boundaries(dir_bnd))

                res += - Mstar * self.gfu.vec

                self.gfu.vec.data += Mstar.Inverse(freedofs=self.fes.FreeDofs()) * res

            else:

                self.gfu.vec.data = Mstar.Inverse(freedofs=self.fes.FreeDofs()) * res

    def update(self):

        self.gfu_old.vec.data = self.gfu.vec.data
        self.sol.append(self.gfu.vec.Copy())

    def set_solution(self, value = CF(0), **kwargs):

        self.gfu.Set(value)
    
    def get_solution(self):

        return self.gfu
    
    def draw_solution(self, **kwargs):
        
        Draw(self.get_solution())
    
    def save_solution(self, **kwargs):

        data = self.problem.data

        params = kwargs
        accepted_keys = ['filename', 'subdivision', 'n_steps']
        defaults = ['sol', 1, 100]  
        self.params_check(params, accepted_keys, defaults)
        
        vtk = VTKOutput(data['mesh'],
                        coefs=[self.get_solution()],
                        names=["adr_sol"],
                        filename=params['filename'],
                        subdivision=params['subdivision'])
        
        tot_steps = len(self.sol)
        jump = max(int(tot_steps/params['n_steps']), 1)

        for i in range(0, tot_steps, jump):

            sol_i = self.sol[i]

            self.gfu.vec.data = sol_i.data
            vtk.Do(time=data['t_array'][i])

    def compute_error(self, u_ex, norm):

        data = self.problem.data

        err = []

        if norm == 'H1':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])
                self.gfu.vec.data = sol_i.data

                aux = InnerProduct(grad(self.gfu)-gradient(u_ex, Id(data['mesh'].dim)), \
                                    grad(self.gfu)-gradient(u_ex, Id(data['mesh'].dim)))

                err.append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2)))

        elif norm == 'L2':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])
                self.gfu.vec.data = sol_i.data

                aux = InnerProduct(self.gfu-u_ex, self.gfu-u_ex)

                err.append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2)))

        else:

            raise ValueError('The norm given is not implemented')

        return err

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Advection-Diffusion-Reaction problem')
        print('This is meant to be a bulk simulation and not a surface one (see SurfaceADRSolver)')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with type ', type(value))

        if self.params['dir_bc']:

            print('Dirichlet boundary conditions are applied to')
            for key, value in self.params['dir_bc'].items():
                print('  -', key, '- with type ', type(value))

        if self.params['flux_c_bc']:

            print('Neumann boundary conditions of convective type are applied to')
            for key, value in self.params['flux_c_bc'].items():
                print('  -', key, '- with type ', type(value))

        if self.params['flux_d_bc']:

            print('Neumann boundary conditions of diffusive type are applied to')
            for key, value in self.params['flux_d_bc'].items():
                print('  -', key, '- with type ', type(value))

        print(60*'-', '\n')