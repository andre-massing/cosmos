# Import necessary libraries
from ngsolve import *
from collections import Counter
from cosmos.solvers.base import Base
from cosmos.solvers.base_solver import BaseSolver
from tqdm import tqdm
from cosmos.solvers.tools import params_update, compute_displ

import cProfile
import pstats

class Simulation(Base):
    
    def __init__(self, mesh, **kwargs):
        
        self.mesh = mesh
        self.params = kwargs
        self.solvers = []
        accepted_keys = ['verbose', 'steady', 't', 'dt', 'T', 'moving']
        defaults = [0, False, Parameter(0), Parameter(0.1), 0.1, False]
        self.params_check(params = self.params, 
                             accepted_keys = accepted_keys, 
                             defaults = defaults)
        
        if self.mesh.ne == 0:

            self.domain_markers = list(Counter(self.mesh.GetBoundaries()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBBoundaries()).keys())

        else:

            self.domain_markers = list(Counter(self.mesh.GetMaterials()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBoundaries()).keys())
        
        self.data = {"mesh": self.mesh,
                     "t": self.params['t'],
                     "dt": self.params['dt'],
                     "T": self.params['T'],
                     'steady': self.params['steady'],
                     'moving': self.params['moving'],
                     "verbosity": self.params['verbose'],
                     "solvers": self.solvers,
                     "iter": 0,
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
                    print(' N. ', value, 'region(s) called ', key)

        print(60*'-', '\n')

    def AddSolver(self, solver):

        if not isinstance(solver, BaseSolver):
            raise TypeError("Solver must inherit from BaseSolver")
        solver.problem = self
        self.solvers.append(solver)

        if self.data['verbosity'] > 0:
            solver.print_info()
    
    def __generator__(self):

        max_steps = int(self.params['T']/self.params['dt'].Get())

        if self.data['verbosity']>0:
            print('Initializing...')

        self.__initialize__()

        if self.data['verbosity']>0:
            print('Initialization successful \n\
                    N.', len(self.solvers), ' solvers have been initialized correctly')

        yield

        if self.data['verbosity']>0:
            print('-'*10, '\nStarting the simulation...')

        if self.data['verbosity']>0:

            for step in tqdm(range(max_steps), desc="\t Running Simulation...", 
                ascii=False, ncols=75):

                self.params['t'].Set(self.params['t'].Get() + self.params['dt'].Get())
                self.data['iter'] += 1

                self.__solve_step__()
                self.__post_process__()

                yield

        else:

            for step in range(max_steps):

                self.params['t'].Set(self.params['t'].Get() + self.params['dt'].Get())
                self.data['iter'] += 1

                self.__solve_step__()
                self.__post_process__()

                yield

        if self.data['verbosity']>0:
            print('Simulation concluded successfully')
            print('-'*10)

    def __call__(self):
        return self.__generator__()

    def __initialize__(self):

        for solver in self.solvers:
            
            solver.initialize()  # Direct execution
            

    def __solve_step__(self):

        for solver in self.solvers:
            solver.solve_step()
        
        for solver in self.solvers:
            solver.update()

    def run(self):

        if self.params['steady']:

            self.__initialize__()
            
            self.__solve_step__()

            self.__post_process__()

        else:

            for step in self(): 
                pass

    def __post_process__(self):

        pass

class MovingSimulation(Simulation):
    
    def __init__(self, mesh, **kwargs):

        super().__init__(mesh=mesh, **kwargs)

        if self.mesh.ne == 0:
            fes = VectorH1(self.mesh, order = self.mesh.GetCurveOrder(), 
                           definedon = self.mesh.Boundaries('.*'))
        else:
            fes = VectorH1(self.mesh, order = self.mesh.GetCurveOrder())
        self.data["dX"] = GridFunction(fes)
        self.data["dX_old"] = GridFunction(fes)
        self.data["V"] = GridFunction(fes)
        
        self.motion_solvers = []
        self.motions = []

    def __initialize__(self):

        for solver in self.motion_solvers:
            solver.initialize()

        for solver in self.solvers:
            solver.initialize()

    def __generator__(self):

        max_steps = int(self.params['T']/self.params['dt'].Get())

        if self.data['verbosity']>0:
            print('Initializing...')

        self.__initialize__()

        if self.data['verbosity']>0:
            print('Initialization successful \n\
                    N.', len(self.solvers) +  len(self.motion_solvers), ' solvers have been initialized correctly')

        yield

        if self.data['verbosity']>0:
            print('-'*10, '\nStarting the simulation...')

        if self.data['verbosity']>0:

            for step in tqdm(range(max_steps), desc="Running Simulation...", 
                ascii=False, ncols=75):

                self.params['t'].Set(self.params['t'].Get() + self.params['dt'].Get())
                self.data['iter'] += 1

                self.__update_motion__()

                self.__solve_step__()

                self.__post_process__()

                yield

        else:

            for step in range(max_steps):

                self.params['t'].Set(self.params['t'].Get() + self.params['dt'].Get())
                self.data['iter'] += 1

                self.__update_motion__()

                self.__solve_step__()

                self.__post_process__()

                yield


        if self.data['verbosity']>0:
            print('Simulation concluded successfully')
            print('-'*10)

    def __call__(self):
        return self.__generator__()
    
    def run(self):

        for step in self(): 
            pass

    def __post_process__(self):

        self.data['dX_old'].vec.data = self.data['dX'].vec.data

    ## TOOLS TO PRESCRIBE THE DISPLACEMENT

    def AddMotionSolver(self, solver):
        """
        Attach a new solver to the coupled problem.
        """
        if not isinstance(solver, BaseSolver):
            raise TypeError("Solver must inherit from BaseSolver")
        
        solver.problem = self
        self.motion_solvers.append(solver)

        if self.data['verbosity'] > 0:
            solver.print_info()

    def AddMotion(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['function', 'domain']
        defaults = [CF((0,)*self.mesh.dim), '.*']  
        self.params_check(params, accepted_keys, defaults)

        params['type'] = 'everywhere'

        self.motions.append(params)

    def AddNormalMotion(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['function', 'domain']
        defaults = [CF(0), '.*']
        self.params_check(params, accepted_keys, defaults)

        params['type'] = 'normal'

        self.motions.append(params)

    def AddTangentialMotion(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['function', 'domain']
        defaults = [CF(0), '.*']
        self.params_check(params, accepted_keys, defaults)

        params['type'] = 'tangential'

        self.motions.append(params)

    def __update_motion__(self):

        for solver in self.motion_solvers:
            solver.solve_step()

        if self.mesh.ne !=0:

            for params in self.motions:

                up_p = params_update(params)

                match up_p['type']:

                    case 'everywhere':
                        self.data['dX'].Set(up_p['function'])

                    case 'normal':
                        compute_displ(self.data, up_p['function'], self.data['dX'], bc = up_p['domain'])

        else:

            for params in self.motions:

                up_p = params_update(params)

                match up_p['type']:

                    case 'everywhere':
                        self.data['dX'].Set(up_p['function'],
                            definedon = self.mesh.Boundaries(up_p['domain']))
                    
                    case 'normal':
                        compute_displ(self.data, up_p['function'], self.data['dX'], bc = up_p['domain'])

                    
        for solver in self.motion_solvers:
            solver.update()





        


