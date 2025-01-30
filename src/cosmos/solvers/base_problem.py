# Import necessary libraries
from ngsolve import *
from collections import Counter
from cosmos.solvers.base import Base
from cosmos.solvers.base_solver import BaseSolver
from tqdm import tqdm

class SteadyProblem(Base):
    
    def __init__(self, mesh, **kwargs):
        
        self.mesh = mesh
        self.params = kwargs
        self.solvers = []
        self.params_check(params = self.params, 
                             accepted_keys=['verbose'], 
                             defaults=[0])
        
        if self.mesh.ne == 0:

            self.domain_markers = list(Counter(self.mesh.GetBoundaries()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBBoundaries()).keys())

        else:

            self.domain_markers = list(Counter(self.mesh.GetMaterials()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBoundaries()).keys())
        
        self.data = {"mesh": self.mesh,
                     "verbosity": self.params['verbose'],
                     "solvers": self.solvers,
                     "domain_mrk": self.domain_markers,
                     "boundary_mrk": self.boundary_markers}
        
        if self.data['verbosity'] > 0:
            self.print_mesh_info()

    def set_verbosity(self, value):

        if type(value) != type(1):
            raise ValueError("The given value must be an integer number!")
        else:
            self.data['verbosity'] = value

    def print_mesh_info(self):
        """_summary_
        """

        print(60*'-')

        print('The mesh is a ', str(self.mesh.dim) + 'D mesh')

        if self.mesh.ne == 0:
            print('! The mesh is also an hypersurface embedded in ', str(self.mesh.dim) + 'D !')

        print('The number of:')
        print(5*' '+'vertices is: ', self.mesh.nv)
        print(5*' '+'edges is: ', self.mesh.nedge)
        print(5*' '+'faces is: ', self.mesh.nface)
        print(5*' '+'facets is: ', self.mesh.nfacet)
        print(5*' '+'elements is: ', self.mesh.ne, '( = 0 if hypersurface)')

        regions = [self.mesh.GetMaterials(),
                    self.mesh.GetBoundaries(), 
                    self.mesh.GetBBoundaries(), 
                    self.mesh.GetBBBoundaries()]

        for i in range(self.mesh.dim+1):

            if regions[i]:

                cnt = Counter(regions[i])

                print('Regions of co-dimension ', i,':')
                for key, value in cnt.items():
                    print('  - There is(are) ', value, 'region(s) called ', key)

        print(60*'-', '\n')

    def attach_solver(self, solver):

        if not isinstance(solver, BaseSolver):
            raise TypeError("Solver must inherit from BaseSolver")
        solver.problem = self
        self.solvers.append(solver)

        if self.data['verbosity'] > 0:
            solver.print_info()

    def initialize(self):

        for solver in self.solvers:
            solver.initialize()

    def solve_step(self):

        for solver in self.solvers:
            solver.solve_step()
        
        for solver in self.solvers:
            solver.update()

    def run(self):

        self.initialize()
        
        self.solve_step()

        self.post_process()

    def post_process(self):

        pass

class UnsteadyProblem(SteadyProblem):
    
    def __init__(self, mesh, dt, t, T, **kwargs):

        super().__init__(mesh=mesh, **kwargs)

        self.dt = dt
        self.T = T
        self.t = t
        self.data["t"] = self.t # Shared state
        self.data["dt"] = self.dt
        self.data["t_array"] = [self.t.Get()]

    def _generator(self):

        max_steps = int(self.T/self.dt.Get())

        self.initialize()

        yield

        for step in tqdm(range(max_steps), desc="Running Simulation...", 
               ascii=False, ncols=75):

            self.t.Set(self.t.Get() + self.dt.Get())
            # print(f"Time step {step + 1}/{max_steps}, Time: {self.t.Get():.3f}")
            self.solve_step()
            self.post_process()

            yield

    def __call__(self):
        return self._generator()
    
    def run(self):

        for step in self(): 
            pass

    def post_process(self):

        self.data["t_array"].append(self.t.Get())

class MovingProblem(UnsteadyProblem):
    
    def __init__(self, mesh, dt, t, T, **kwargs):

        super().__init__(mesh=mesh, dt=dt, t=t, T=T, **kwargs)

        if self.mesh.ne == 0:
            fes = VectorH1(self.mesh, order = self.mesh.GetCurveOrder(), definedon = self.mesh.Boundaries('.*'))
        else:
            fes = VectorH1(self.mesh, order = self.mesh.GetCurveOrder())
        self.displ = GridFunction(fes)
        self.displ_old = GridFunction(fes)
        self.data["dX"] = self.displ
        self.data["dX_old"] = self.displ_old
        self.data["dX_array"] = [self.displ.vec.Copy()]

        self.displ_solvers = []
        def f():
            return CF((0,)*self.mesh.dim)
        self.data["f_dX"] = f

        self.mm_solvers = []

    def initialize(self):

        for solver in self.displ_solvers:
            solver.initialize()
        for solver in self.mm_solvers:
            solver.initialize()
        for solver in self.solvers:
            solver.initialize()

    def _generator(self):

        max_steps = int(self.T/self.dt.Get())

        self.initialize()

        yield

        for step in tqdm(range(max_steps), desc="Running Simulation...", 
               ascii=False, ncols=75):

            self.t.Set(self.t.Get() + self.dt.Get())

            self.displ_step()
            self.mm_step()
            self.solve_step()

            self.post_process()

            yield

    def __call__(self):
        return self._generator()
    
    def run(self):

        for step in self(): 
            pass

    def post_process(self):

        self.data["t_array"].append(self.t.Get())
        self.data['dX_old'].vec.data = self.data['dX'].vec.data
        self.data["dX_array"].append(self.data["dX"].vec.Copy())

    ## TOOLS TO PRESCRIBE THE DISPLACEMENT

    def set_displacement(self, new_function):
        """
        Update the user-defined function.

        Parameters:
        new_function (callable): The new function to use.
        """
        if not callable(new_function):
            raise ValueError("The provided new_function must be callable.")
        self.data['f_dX'] = new_function

    def displ_step(self):
        """
        Solve one time step for all solvers in sequence.
        """
        for solver in self.displ_solvers:
            solver.solve_step()
        
        for solver in self.displ_solvers:
            solver.update()

        if self.mesh.ne == 0:
            self.data['dX'].Set(self.data['f_dX'](), definedon = self.data["mesh"].Boundaries('.*'))
        else:
            self.data['dX'].Set(self.data['f_dX']())

    def attach_displ_solver(self, solver):
        """
        Attach a new solver to the coupled problem.
        """
        if not isinstance(solver, BaseSolver):
            raise TypeError("Solver must inherit from BaseSolver")
        solver.problem = self
        self.displ_solvers.append(solver)

        if self.data['verbosity'] > 0:
            solver.print_info()

    ## TOOLS TO CORRECT OR IMPOSE THE MESH MOVEMENT

    def mm_step(self):

        for solver in self.mm_solvers:
            solver.solve_step()

        for solver in self.mm_solvers:
            solver.update()

    def attach_mm_solver(self, solver):

        if not isinstance(solver, BaseSolver):
            raise TypeError("Solver must inherit from BaseSolver")
        solver.problem = self
        self.mm_solvers.append(solver)

        if self.data['verbosity'] > 0:
            solver.print_info()


