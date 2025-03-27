# Import necessary libraries
from ngsolve import *
from ngsolve.solvers import Newton
from collections import Counter
from cosmos.solvers.base_pde import BasePDE
from cosmos.solvers.tools import params_check
from cosmos.solvers.container import Container
from tqdm import tqdm
import csv

class Steady():

    def __init__(self, mesh, **kwargs):

        self.mesh = mesh
        self.mesh_data = kwargs
        accepted_keys = ['verbose', 'linear']
        defaults = [0, False]
        self.PDEs = []
        params_check(params = self.mesh_data, 
                    accepted_keys = accepted_keys, 
                    defaults = defaults)
        
        if self.mesh.ne == 0:

            self.domain_markers = list(Counter(self.mesh.GetBoundaries()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBBoundaries()).keys())

        else:

            self.domain_markers = list(Counter(self.mesh.GetMaterials()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBoundaries()).keys())
        
        self.mesh_data["mesh"] = self.mesh
        self.mesh_data["domain_mrk"] = self.domain_markers
        self.mesh_data["boundary_mrk"] = self.boundary_markers

        if self.mesh_data['verbose'] > 0:
            self.print_mesh_info()

    def AddPDEs(self, *args):

        for pde in args:

            if not isinstance(pde, BasePDE):
                raise TypeError("The added PDE must inherit from BasePDE")
            self.PDEs.append(pde)

            if self.mesh_data['verbose'] > 0:
                pde.print_info()

    def Solve(self):

        with TaskManager():

            self.PreProcess()

            for pde in self.PDEs:

                pde.Initialize(self.mesh_data)

                A = BilinearForm(pde.fes)
                F = LinearForm(pde.fes)

                lhs = pde.GetLHS(self.mesh_data, pde.trial, pde.test)
                rhs = pde.GetRHS(self.mesh_data, pde.test)

                if hasattr(pde, 'nl') and not self.mesh_data['linear']:

                    if pde.nl:
                        A += pde.GetNL(self.mesh_data, pde.trial, pde.test)
                    A += lhs - rhs

                    Newton(A, pde.gfu, maxit=100, printing = False)

                else:

                    A += lhs
                    F += rhs

                    A.Assemble()
                    F.Assemble()

                    pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

                self.PostProcess()
                    
    def PreProcess(self):

        pass

    def PostProcess(self):

        for pde in self.PDEs:

            pde.Update(self.mesh_data)

            if isinstance(pde, Container):
                for pde_i in pde.PDEs:
                    if pde_i.error_params:
                        self.print_error(pde_i)
                    if pde_i.save_params:
                        pde_i.save_params['vtk'].Do(vb = pde_i.save_params['VorB'])
            else:
                if pde.error_params:
                    self.print_error(pde)
                if pde.save_params:
                    pde.save_params['vtk'].Do(vb = pde.save_params['VorB'])

    def print_error(self, pde):

        file_path = os.path.join(pde.error_params['folderpath'], pde.error_params['filename'])
        err = pde.get_error(self.mesh_data)

        if type(err) is list:
            towrite = [self.mesh_data['t'].Get()] + err
            columns = pde.params['name']
        else:
            towrite = [err]
            columns = [pde.params['name']]

        with open(file_path, mode='w', newline='') as file:  # 'w' mode ensures overwriting
            writer = csv.writer(file)
            writer.writerow(columns)  # Write header row

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(towrite)

    
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

class BackwardEuler(Steady):

    def __init__(self, mesh, **kwargs):

        params = kwargs

        accepted_keys = ['verbose', 'linear', 't', 'dt', 'T']
        defaults = [0, False, Parameter(0), Parameter(0.1), 0.1]
        params_check(params = params, 
                    accepted_keys = accepted_keys, 
                    defaults = defaults)
        
        super().__init__(mesh, verbose = params['verbose'], linear = params['linear'])
        
        self.mesh_data['t'] = params['t']
        self.mesh_data['dt'] = params['dt']
        self.mesh_data['T'] = params['T']
        self.mesh_data['iter'] = 0

    def SolveStep(self):

        for pde in self.PDEs:

            A = BilinearForm(pde.fes)
            F = LinearForm(pde.fes)

            lhs = pde.GetLHS(self.mesh_data, pde.trial, pde.test)
            rhs = pde.GetRHS(self.mesh_data, pde.test)
            mass, mass_gfu = pde.GetMass(self.mesh_data, pde.trial, pde.test, pde.gfu_old)

            if hasattr(pde, 'nl') and not self.mesh_data['linear']:

                if pde.nl:
                    A += pde.GetNL(self.mesh_data, pde.trial, pde.test)
                A += lhs + mass - rhs - mass_gfu

                Newton(A, pde.gfu, maxit=100, printing = False)

            else:

                A += lhs + mass
                F += rhs + mass_gfu

                A.Assemble()
                F.Assemble()

                pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec
                
    def print_error0(self, pde):

        file_path = os.path.join(pde.error_params['folderpath'], pde.error_params['filename'])
        err = pde.get_error(self.mesh_data)

        if type(err) is list:
            towrite = [self.mesh_data['t'].Get()] + err
            columns = ['Time'] + pde.params['name']
        else:
            towrite = [self.mesh_data['t'].Get()] + [err]
            columns = ['Time'] + [pde.params['name']]

        with open(file_path, mode='w', newline='') as file:  # 'w' mode ensures overwriting
            writer = csv.writer(file)
            writer.writerow(columns)  # Write header row

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(towrite)
    
    def print_error(self, pde):

        file_path = os.path.join(pde.error_params['folderpath'], pde.error_params['filename'])
        err = pde.get_error(self.mesh_data)

        if type(err) is list:
            towrite = [self.mesh_data['t'].Get()] + err
        else:
            towrite = [self.mesh_data['t'].Get()] + [err]

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(towrite)

    def __generator__(self):

        with TaskManager():

            max_steps = int(self.mesh_data['T']/self.mesh_data['dt'].Get())

            if self.mesh_data['verbose']>0:
                print('Initializing...')

            self.Initialize()

            if self.mesh_data['verbose']>0:
                print('Initialization successful \n\
                        N.', len(self.PDEs), ' PDEs have been initialized correctly')

            yield

            if self.mesh_data['verbose']>0:
                print('-'*10, '\nStarting the simulation...')

            if self.mesh_data['verbose']>0:

                for step in tqdm(range(max_steps), desc="\t Running Simulation...", 
                    ascii=False, ncols=75):

                    self.mesh_data['t'].Set(self.mesh_data['t'].Get() + self.mesh_data['dt'].Get())
                    self.mesh_data['iter'] += 1

                    self.PreProcess()
                    self.SolveStep()
                    self.PostProcess()

                    yield

            else:

                for step in range(max_steps):

                    self.mesh_data['t'].Set(self.mesh_data['t'].Get() + self.mesh_data['dt'].Get())
                    self.mesh_data['iter'] += 1

                    self.PreProcess()
                    self.SolveStep()
                    self.PostProcess()

                    yield

            if self.mesh_data['verbose']>0:
                print('Simulation concluded successfully')
                print('-'*10)

    def __call__(self):
        return self.__generator__()

    def Initialize(self):

        for pde in self.PDEs:

            pde.Initialize(self.mesh_data)

            if isinstance(pde, Container):
                for pde_i in pde.PDEs:
                    if pde_i.error_params:
                        self.print_error0(pde_i)
                    if pde_i.save_params:
                        pde_i.save_params['vtk'].Do(time = self.mesh_data['t'].Get(), vb = pde_i.save_params['VorB'])
            else:
                if pde.error_params:
                    self.print_error0(pde)
                if pde.save_params:
                    pde.save_params['vtk'].Do(time = self.mesh_data['t'].Get(), vb = pde.save_params['VorB'])

    def Solve(self):

        for step in self(): 
                pass
        
    def PostProcess(self):
        
        for pde in self.PDEs:

            if isinstance(pde, Container):
                pde.MP_and_BP(self.mesh_data, pde.gfu)
            else:
                if 'MP' in pde.params or 'BP' in pde.params:
                    pde.MP_and_BP(self.mesh_data, pde.gfu)

            pde.Update(self.mesh_data)

            if isinstance(pde, Container):
                for pde_i in pde.PDEs:
                    if pde_i.error_params:
                        self.print_error(pde_i)
                    if pde_i.save_params:
                        pde_i.save_params['vtk'].Do(time = self.mesh_data['t'].Get(), vb = pde_i.save_params['VorB'])
            else:
                if pde.error_params:
                    self.print_error(pde)
                if pde.save_params:
                    pde.save_params['vtk'].Do(time = self.mesh_data['t'].Get(), vb = pde.save_params['VorB'])