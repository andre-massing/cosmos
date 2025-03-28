# Import necessary libraries
from ngsolve import *
from ngsolve.solvers import Newton
from ngsolve.webgui import Draw
from cosmos.solvers.solvers import BackwardEuler
from cosmos.pdes.base_pde import BasePDE
from cosmos.utils.tools import params_check
from cosmos.pdes.container import Container
from tqdm import tqdm
import csv

class BackwardEulerM(BackwardEuler):

    def __init__(self, mesh, **kwargs):
        
        super().__init__(mesh, **kwargs)

        if self.mesh.ne == 0:
            fes = VectorH1(self.mesh, order = self.mesh.GetCurveOrder(), 
                           definedon = self.mesh.Boundaries('.*'))
        else:
            fes = VectorH1(self.mesh, order = self.mesh.GetCurveOrder())
        
        self.mesh_data['dX'] = GridFunction(fes)
        self.mesh_data['dX_old'] = GridFunction(fes)
        self.mesh_data['V'] = GridFunction(fes)

        self.M_PDEs = []
        self.M = []

    def Initialize(self):

        super().Initialize()

        for pde in self.M_PDEs:

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


    def SolveStep(self):

        for pde in self.PDEs:

            A = BilinearForm(pde.fes)
            F = LinearForm(pde.fes)

            lhs = pde.GetLHS(self.mesh_data, pde.trial, pde.test, self.mesh_data['dX'])
            rhs = pde.GetRHS(self.mesh_data, pde.test, self.mesh_data['dX'])

            _, mass_gfu = pde.GetMass(self.mesh_data, pde.trial, pde.test, self.mesh_data['dX_old'], pde.gfu_old)
            mass, _ = pde.GetMass(self.mesh_data, pde.trial, pde.test, self.mesh_data['dX'], pde.gfu_old)

            if hasattr(pde, 'nl') and not self.mesh_data['linear']:

                if pde.nl:
                    A += pde.GetNL(self.mesh_data, pde.trial, pde.test, self.mesh_data['dX'])
                A += lhs + mass - rhs - mass_gfu

                Newton(A, pde.gfu, maxit=100, printing = False)

            else:

                A += lhs + mass
                F += rhs + mass_gfu

                A.Assemble()
                F.Assemble()

                pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

    def PreProcess(self):
        
        self.SolveStepM()

        dX_vol = GridFunction(self.mesh_data['dX'].space)
        dX_bnd = GridFunction(self.mesh_data['dX'].space)

        for move in self.M:

            if move['type'] == BND or move['type'] == 'normal':

                dX_bnd.vec.data += self.__add_bnd_motion__(move).vec.data

            elif move['type'] ==  VOL:

                dX_vol.vec.data += self.__add_vol_motion__(move).vec.data

        self.mesh_data['dX'].vec.data += dX_vol.vec.data
        self.mesh_data['dX'].vec.data += dX_bnd.vec.data

        self.mesh_data['V'].Set(self.mesh_data['dX'] - self.mesh_data['dX_old'])

    def PostProcess(self):

        self.mesh_data['dX_old'].vec.data = self.mesh_data['dX'].vec.data

        for pde in self.M_PDEs:

            self.mesh_data['mesh'].SetDeformation(self.mesh_data['dX'])

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

            self.mesh_data['mesh'].UnsetDeformation()

        for pde in self.PDEs:

            if isinstance(pde, Container):
                pde.MP_and_BP(self.mesh_data, self.mesh_data['dX'], pde.gfu)
            else:
                if pde.params['MP'] or pde.params['BP']:
                    pde.MP_and_BP(self.mesh_data, self.mesh_data['dX'], pde.gfu)

            pde.Update(self.mesh_data)

            self.mesh_data['mesh'].SetDeformation(self.mesh_data['dX'])

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

            self.mesh_data['mesh'].UnsetDeformation()

    ## TOOLS TO PRESCRIBE THE DISPLACEMENT

    def AddMotionSolver(self, pde):
        """
        Attach a new solver to the coupled problem.
        """
        if not isinstance(pde, BasePDE):
            raise TypeError("Solver must inherit from BaseSolver")
        
        self.M_PDEs.append(pde)

        if self.mesh_data['verbose'] > 0:
            pde.print_info()

    def AddVolMotion(self, **kwargs):

        params = kwargs

        if self.mesh.ne ==0:
            raise Exception('No volume elements! The motion is not added to the simulation')

        # Initialize the parameters
        accepted_keys = ['function', 'domain', 'total']
        defaults = [CF((0,)*self.mesh.dim), '.*', False]  
        params_check(params, accepted_keys, defaults)

        params['type'] = VOL

        self.M.append(params)

    def AddBndMotion(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['function', 'domain', 'total']
        defaults = [CF(0), '.*', False]
        params_check(params, accepted_keys, defaults)

        params['type'] = BND

        self.M.append(params)

    def AddNormalMotion(self, **kwargs):

        params = kwargs

        # Initialize the parameters
        accepted_keys = ['function', 'domain', 'total']
        defaults = [CF(0), '.*', False]
        params_check(params, accepted_keys, defaults)

        params['type'] = 'normal'

        self.M.append(params)

    def SolveStepM(self):

        for pde in self.M_PDEs:

            A = BilinearForm(pde.fes)
            F = LinearForm(pde.fes)

            lhs = pde.GetLHS(self.mesh_data, pde.trial, pde.test, self.mesh_data['dX_old'])
            rhs = pde.GetRHS(self.mesh_data, pde.test, self.mesh_data['dX_old'])

            mass, mass_gfu = pde.GetMass(self.mesh_data, pde.trial, pde.test, self.mesh_data['dX_old'], pde.gfu_old)

            if hasattr(pde, 'nl') and not self.mesh_data['linear']:

                if pde.nl:
                    A += pde.GetNL(self.mesh_data, pde.trial, pde.test, self.mesh_data['dX_old'])
                A += lhs + mass - rhs - mass_gfu

                Newton(A, pde.gfu, maxit=100, printing = False)

            else:

                A += lhs + mass
                F += rhs + mass_gfu

                A.Assemble()
                F.Assemble()

                pde.gfu.vec.data = A.mat.Inverse(freedofs = pde.fes.FreeDofs())*F.vec

            if isinstance(pde, Container):
                pde.MP_and_BP(self.mesh_data, self.mesh_data['dX_old'], pde.gfu)
            else:
                if 'MP' in pde.params or 'BP' in pde.params:
                    pde.MP_and_BP(self.mesh_data, self.mesh_data['dX_old'], pde.gfu)

            pde.Update(self.mesh_data, self.mesh_data['dX_old'])

    def __add_vol_motion__(self, move):

        dX = GridFunction(self.mesh_data['dX'].space)

        if move['total']: 

            dX.Set(move['function']() - self.mesh_data['dX_old'],
                    definedon = self.mesh.Materials(move['domain']))
            
        else:

            self.mesh_data['mesh'].SetDeformation(self.mesh_data['dX_old'])
            dX.Set(move['function'](),
                    definedon = self.mesh.Materials(move['domain']))
            self.mesh_data['mesh'].UnsetDeformation()
        
        return dX
        
    def __add_bnd_motion__(self, move):

        dX = GridFunction(self.mesh_data['dX'].space)
        n = specialcf.normal(self.mesh_data['mesh'].dim)

        if move['total']:

            if move['type'] == BND:
                    
                dX.Set(move['function']() - self.mesh_data['dX_old'],
                        definedon = self.mesh.Boundaries(move['domain']))

            else:

                dX.Set(move['function']()*n - self.mesh_data['dX_old'],
                        definedon = self.mesh.Boundaries(move['domain']))
                
            # if self.mesh_data['mesh'].ne != 0:

            #     gfu = elastic_motion(self.mesh_data['mesh'], dX, move['domain'])
            #     dX.vec.data = gfu.vec.data

        else:

            self.mesh_data['mesh'].SetDeformation(self.mesh_data['dX_old'])

            if move['type'] == BND:

                dX.Set(move['function'](),
                        definedon = self.mesh.Boundaries(move['domain']))

            else:

                dX.Set(move['function']()*n,
                            definedon = self.mesh.Boundaries(move['domain']))
                
            # if self.mesh_data['mesh'].ne != 0:

            #     gfu = elastic_motion(self.mesh_data['mesh'], dX, move['domain'])
            #     dX.vec.data = gfu.vec.data

            self.mesh_data['mesh'].UnsetDeformation()

        
        return dX
