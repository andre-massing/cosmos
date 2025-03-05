from ngsolve import *
from cosmos.solvers.base_solver import BaseSolver

from ngsolve.webgui import Draw

class HarmonicMMSolver(BaseSolver):

    def __init__(self, **kwargs):

        super().__init__(fes_order = 1, **kwargs)

        # Initialize the parameters
        accepted_keys = ['velocity']
        defaults = [None] 
        self.params_check(self.params, accepted_keys, defaults)

    def initialize(self):

        data = self.problem.data

        self.fes_order = data['mesh'].GetCurveOrder()

        V1_s = VectorH1(data['mesh'], order=self.fes_order,\
                        definedon=data['mesh'].Boundaries('.*'))
        V2_s = H1(data['mesh'], order=self.fes_order, definedon=data['mesh'].Boundaries('.*'))
        self.fes_s = V1_s*V2_s
        self.dX_m = GridFunction(V1_s)
        self.X_m = GridFunction(V1_s)

        self.fes_v = VectorH1(data['mesh'], order= self.fes_order, dirichlet = data['mesh'].Boundaries('.*'))
        self.gfu_v = GridFunction(self.fes_v)
    
    def solve_step(self):

        data = self.problem.data

        params = self.params_update(self.params)

        (u, p),(v, q) = self.fes_s.TnT()
        self.gfu_s = GridFunction(self.fes_s)

        n = specialcf.normal(data['mesh'].dim)
        P = Id(data['mesh'].dim) - OuterProduct(n, n)

        data['mesh'].SetDeformation(data['dX_old'])

        
        self.dX_m.Set(params['velocity']*data['dt'], definedon=data['mesh'].Boundaries('.*'))

        self.X_m.Set(data['dX_old'] + self.dX_m, definedon=data['mesh'].Boundaries('.*'))

        data['mesh'].UnsetDeformation()

        ds_old = ds(deformation=data['dX_old'])
        ds_new = ds(deformation=self.X_m)

        A_s = BilinearForm(self.fes_s)
        A_s += InnerProduct(Grad(u).Trace(), Grad(v).Trace())*ds_old
        A_s += -p*n*v*ds_new
        A_s += -q*n*u*ds_new
        F_s = LinearForm(self.fes_s)
        F_s += (-InnerProduct(Grad(self.X_m).Trace(), Grad(v).Trace()))*ds_old

        with TaskManager():
            A_s.Assemble()

            A_s_inv = A_s.mat.Inverse(freedofs=self.fes_s.FreeDofs())
            F_s.Assemble()

            self.gfu_s.vec.data = A_s_inv*F_s.vec

        if data['mesh'].ne != 0:

            dx_old = dx(deformation=data['dX_old'])

            
            u, v = self.fes_v.TnT()

            A = BilinearForm(self.fes_v)
            A += InnerProduct(Grad(u), Grad(v))*dx_old

            self.gfu_v.Set(self.dX_m + self.gfu_s.components[0], definedon = data['mesh'].Boundaries('.*'))
    
            with TaskManager():
                A.Assemble()
                   
                res = - A.mat * self.gfu_v.vec

                self.gfu_v.vec.data += A.mat.Inverse(freedofs=self.fes_v.FreeDofs()) * res

                self.X_m = GridFunction(VectorH1(data['mesh'], order= self.fes_order))
                self.X_m.Set(data['dX_old'] + self.gfu_v)
                data['dX'].Set(self.X_m)

        else:

            data['dX'].Set(data['dX_old'] + self.dX_m + self.gfu_s.components[0], definedon = data['mesh'].Boundaries('.*'))


    def update(self):

        data = self.problem.data

        self.sol.append(self.get_solution())

    def set_solution(self, **kwargs):
        
        raise Exception('No setter is available for a stationary solver')

    def get_solution(self):

        data = self.problem.data

        if data['mesh'].ne != 0:
            return self.gfu_v
        else:  
            return self.gfu_s.components[0]

    def draw_solution(self, **kwargs):
        
        Draw(self.get_solution())
    
    def save_solution(self, **kwargs):

        data = self.problem.data

        params = kwargs
        accepted_keys = ['filename', 'subdivision']
        defaults = ['sol', 1]  
        self.params_check(params, accepted_keys, defaults)

        if data['mesh'].ne != 0:
            v_or_b = VOL
        else:  
            v_or_b = BND
        
        vtk = VTKOutput(data['mesh'],
                        coefs=[self.get_solution()],
                        names=["mm_sol"],
                        filename=params['filename'],
                        subdivision=params['subdivision'])
        vtk.Do(vb = v_or_b)

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

        print('This is a solver for the moving the mesh by solving')
        print('a vector Poisson problem.')
        print('It works for both surface and bulk displacements.')
        print('It corrects the surface displacement first and then,')
        print('if present, it moves the bulk mesh too accordingly.')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with type ', type(value))

        print(60*'-', '\n')


# class ElasticMMSolver(BaseSolver):

#     def __init__(self, fes_order=1, params = None, velocity = None):

#         super().__init__(fes_order, params)

#         self.velocity = velocity

#     def initialize(self, data):

#         self.fes_order = data['mesh'].GetCurveOrder()
    
#     def solve_step(self, data):

#         # Use the mesh from the shared state to solve diffusion
#         self.solve_elastic_mm(data)
    
#     def solve_elastic_mm(self, data):

#         V1_s = VectorH1(data['mesh'], order=self.fes_order,\
#                         definedon=data['mesh'].Boundaries('.*'))
#         V2_s = H1(data['mesh'], order=self.fes_order, definedon=data['mesh'].Boundaries('.*'))

#         self.fes_s = V1_s*V2_s
#         (u, p),(v, q) = self.fes_s.TnT()
#         self.gfu_s = GridFunction(self.fes_s)

#         n = specialcf.normal(data['mesh'].dim)
#         P = Id(data['mesh'].dim) - OuterProduct(n, n)

#         data['mesh'].SetDeformation(data['dX_old'])

#         dX_m = GridFunction(V1_s)
#         dX_m.Set(self.velocity()*data['dt'], definedon=data['mesh'].Boundaries('.*'))

#         X_m = GridFunction(V1_s)
#         X_m.Set(data['dX_old'] + dX_m, definedon=data['mesh'].Boundaries('.*'))

#         data['mesh'].UnsetDeformation()

#         ds_old = ds(deformation=data['dX_old'])
#         ds_new = ds(deformation=X_m)

#         def E(u):
#             return Grad(u).Trace() + Grad(u).Trace().trans

#         A_s = BilinearForm(self.fes_s)
#         A_s += InnerProduct(E(u), E(v))*ds_old
#         A_s += -p*n*v*ds_new
#         A_s += -q*n*u*ds_new
#         F_s = LinearForm(self.fes_s)
#         F_s += (-InnerProduct(E(dX_m), E(v)))*ds_old

#         with TaskManager():
#             A_s.Assemble()

#             A_s_inv = A_s.mat.Inverse(freedofs=self.fes_s.FreeDofs())
#             F_s.Assemble()

#             self.gfu_s.vec.data = A_s_inv*F_s.vec

#         if data['mesh'].ne != 0:

#             dx_old = dx(deformation=data['dX_old'])

#             self.fes_v = VectorH1(data['mesh'], order= self.fes_order, dirichlet = data['mesh'].Boundaries('.*'))
#             u, v = self.fes_v.TnT()

#             A = BilinearForm(self.fes_v)
#             A += InnerProduct(Grad(u), Grad(v))*dx_old

#             self.gfu_v = GridFunction(self.fes_v)

#             self.gfu_v.Set(dX_m + self.gfu_s.components[0], definedon = data['mesh'].Boundaries('.*'))
    
#             with TaskManager():
#                 A.Assemble()
                   
#                 res = - A.mat * self.gfu_v.vec

#                 self.gfu_v.vec.data += A.mat.Inverse(freedofs=self.fes_v.FreeDofs()) * res

#                 X_m = GridFunction(VectorH1(data['mesh'], order= self.fes_order))
#                 X_m.Set(data['dX_old'] + self.gfu_v)
#                 data['dX'].Set(X_m)

#             pass

#         else:

#             data['dX'].Set(data['dX_old'] + dX_m + self.gfu_s.components[0], definedon = data['mesh'].Boundaries('.*'))


#     def update(self, data):

#         self.sol.append(self.gfu_s.components[0])

#     def compute_error(self, data, u_ex, norm):

#         if norm == 'H1':

#             aux = InnerProduct(grad(self.sol[0])-gradient(u_ex, Id(data['mesh'].dim)), \
#                                 grad(self.sol[0])-gradient(u_ex, Id(data['mesh'].dim)))

#             err = sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2))

#         elif norm == 'L2':

#             aux = InnerProduct(self.sol[0]-u_ex, self.sol[0]-u_ex)

#             err = sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2))

#         else:

#             raise ValueError('The norm given is not implemented')

#         return err


#     def print_solver_info(self):

#         print(60*'-')

#         print('This is a solver for the moving the mesh by solving')
#         print('a vector Poisson problem.')
#         print('It works for both surface and bulk displacements.')
#         print('It corrects the surface displacement first and then,')
#         print('if present, it moves the bulk mesh too accordingly.')
#         print('It uses conforming H-1 elements of order ', self.fes_order)

#         print('Its parameters are')
#         for key, value in self.params.items():
#             print('  -', key, '- with type ', type(value))

#         if self.dir_bc:

#             print('Dirichlet boundary conditions are applied to')
#             for key, value in self.dir_bc.items():
#                 print('  -', key, '- with type ', type(value))

#         print(60*'-', '\n')
