from ngsolve import *
from cosmos.solvers.base_solver import BaseSolver
from cosmos.utils.manufactured_solution_tools import gradient

from ngsolve.webgui import Draw

class PoissonSolver(BaseSolver):

    def __init__(self, fes_order=1, **kwargs):

        super().__init__(fes_order, **kwargs)

        # Initialize the parameters
        accepted_keys = ['rhs', 'dir_bc', 'neu_bc']
        defaults = [CF(0.0), {}, {}] 
        self.params_check(self.params, accepted_keys, defaults)

    def initialize(self):

        data = self.problem.data

        params = self.params_update(self.params)

        if params['dir_bc']:

            if not set(list(params['dir_bc'].keys())) <= set(data['boundary_mrk']):
                raise ValueError('Dirichlet boundary conditions are imposed \
                              on boundary regions which name is not in \
                             the mesh boundary list')
            separator = '|'
            dir_bnd = separator.join(list(params['dir_bc'].keys()))

            V = H1(data['mesh'], order = self.fes_order, dirichlet = dir_bnd)
            self.fes = V
        else:
            V = H1(data['mesh'], order= self.fes_order)
            Q = NumberSpace(data['mesh'])
            self.fes = V*Q

        self.gfu = GridFunction(self.fes)
    
    def solve_step(self):

        data = self.problem.data

        params = self.params_update(self.params)

        if params['dir_bc']:
            u, v = self.fes.TnT()
            separator = '|'
            dir_bnd = separator.join(list(params['dir_bc'].keys()))
            dir_bc = self.params_update(params['dir_bc'])
            dir_cf = data['mesh'].BoundaryCF(dir_bc, default=0)
            u, v = self.fes.TnT()
        else:
            (u, lam), (v, mu) = self.fes.TnT()

        A = BilinearForm(self.fes)
        A += grad(u)*grad(v)*dx
        if not params['dir_bc']:
            A += (lam*v-mu*u)*dx
        F = LinearForm(self.fes)
        F += params['rhs']*v*dx

        if params['neu_bc']:
            n = specialcf.normal(data['mesh'].dim)
            neu_bc = self.params_update(params['neu_bc'])
            neu_cf = data['mesh'].BoundaryCF(neu_bc, default=CF((0,) * data['mesh'].dim))
            separator = '|' 
            neu_bnd = separator.join(list(neu_bc.keys()))
            F += neu_cf*n*v*ds(definedon = data['mesh'].Boundaries(neu_bnd))

        res = F.vec.CreateVector()    
        with TaskManager():
            A.Assemble()
            F.Assemble()

            res = F.vec
            if params['dir_bc']: 
                print(self.fes.__dict__)
                self.gfu.Set(dir_cf, definedon = data['mesh'].Boundaries(dir_bnd))
                res.data += - A.mat * self.gfu.vec

                self.gfu.vec.data += A.mat.Inverse(freedofs=self.fes.FreeDofs()) * res

            else:

                self.gfu.vec.data = A.mat.Inverse(freedofs=self.fes.FreeDofs()) * F.vec

    def update(self):

        self.sol.append(self.get_solution())

    def set_solution(self, **kwargs):
        
        raise Exception('No setter is available for a stationary solver')

    def get_solution(self):
        
        if not self.params['dir_bc']:
            return self.gfu.components[0]
        else:
            return self.gfu

    def draw_solution(self, **kwargs):
        
        Draw(self.get_solution())
    
    def save_solution(self, **kwargs):

        data = self.problem.data

        params = kwargs
        accepted_keys = ['filename', 'subdivision']
        defaults = ['sol', 1]  
        self.params_check(params, accepted_keys, defaults)
        
        vtk = VTKOutput(data['mesh'],
                        coefs=[self.get_solution()],
                        names=["poisson_sol"],
                        filename=params['filename'],
                        subdivision=params['subdivision'])
        vtk.Do()

    def compute_error(self, u_ex, norm):

        data = self.problem.data

        if norm == 'H1':

            aux = InnerProduct(grad(self.sol[0])-gradient(u_ex, Id(data['mesh'].dim)), \
                                grad(self.sol[0])-gradient(u_ex, Id(data['mesh'].dim)))

            err = sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2))

        elif norm == 'L2':

            aux = InnerProduct(self.sol[0]-u_ex, self.sol[0]-u_ex)

            err = sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2))

        else:

            raise ValueError('The norm given is not implemented')

        return err

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Poisson problem')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with type ', type(value))

        if self.params['dir_bc']:
            print('Dirichlet boundary conditions are applied to')
            for key, value in self.params['dir_bc'].items():
                print('  -', key, '- with type ', type(value))

        if self.params['neu_bc']:
            print('Neumann boundary conditions are applied to')
            for key, value in self.params['neu_bc'].items():
                print('  -', key, '- with type ', type(value))

        print(60*'-', '\n')
