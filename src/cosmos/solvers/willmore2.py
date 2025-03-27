from ngsolve import *
from cosmos.solvers.base_solver import BaseSolver
from cosmos.solvers.tools import params_check, compute_error, compute_mc, compute_stab_mc
from ngsolve.webgui import Draw
import csv

class StabWillmoreSolver(BaseSolver):

    def __init__(self, **kwargs):

        super().__init__(fes_order=1, **kwargs)

        accepted_keys = ['rhs', 'stab', 'sp_curv', 'clamped_bnd', 'domain']
        defaults = [CF(0), CF(1e-3), CF(0.0), {}, '.*']
        self.params_check(self.params, accepted_keys, defaults)

        self.error_params = None
        self.save_params = None

    def initialize(self):

        data = self.problem.data

        if self.fes_order>1:
            raise Exception("Lumped mass not defined for fes_order>1 !")

        
        if self.params['clamped_bnd']:
            V1 = Compress(VectorH1(data['mesh'], order=self.fes_order,
                        definedon=data['mesh'].Boundaries(self.params['domain']),
                        dirichlet_bbnd = data['mesh'].BBoundaries(self.params['clamped_bnd'])))
        else:
            V1 = Compress(VectorH1(data['mesh'], order=self.fes_order,
                        definedon=data['mesh'].Boundaries(self.params['domain'])))
            
        V2 = Compress(VectorH1(data['mesh'], order=self.fes_order,
                    definedon=data['mesh'].Boundaries(self.params['domain'])))
        
        if data['mesh'].dim == 2:
            
            dV = Compress(H1(data['mesh'], order=1,\
                     definedon = data['mesh'].Boundaries(self.params['domain'])))
        elif data['mesh'].dim == 3:
            
            dV = Compress(VectorFacetSurface(data['mesh'], order=1,\
                    definedon = data['mesh'].Boundaries(self.params['domain'])))
            
        V_vol = VectorH1(data['mesh'], order = self.fes_order)

        if data['mesh'].ne == 0:
            self.dX_save = GridFunction(V1)
            self.kappa_save = GridFunction(V2)
        else:
            self.dX_save = GridFunction(V_vol)
            self.kappa_save = GridFunction(V_vol)
        
        self.fes = V1*V2*dV
        self.gfu = GridFunction(self.fes)
        self.dX_h, self.Y_h, _ = self.gfu.components
        self.kappa_h = GridFunction(V2)

        compute_stab_mc(data, self.kappa_h, self.params)
        data['mesh'].SetDeformation(data['dX'])
        ns = specialcf.normal(data['mesh'].dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                        definedon = data['mesh'].Boundaries('.*'))  
        data['mesh'].UnsetDeformation()

        if self.error_params:
            err = []

            for i, el in enumerate( [self.dX_h, self.kappa_h]):

                err.append(compute_error(data=data, gfu = el, u_ex=self.error_params['ex_sol'][i],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = BND))

            file_path = os.path.join(self.error_params['folderpath'], self.error_params['filename'])
            os.makedirs(self.error_params['folderpath'], exist_ok=True)

            towrite = [data['t'].Get()] + err
            

            # Define column names
            columns = ['Time', 'displacement', 'mean_curv']

            # Check if file exists
            with open(file_path, mode='w', newline='') as file:  # 'w' mode ensures overwriting
                writer = csv.writer(file)
                writer.writerow(columns)  # Write header row

            # Write the first row
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(towrite)

        if self.save_params:

            file_path = os.path.join(self.save_params['folderpath'], self.save_params['filename'])
            os.makedirs(self.save_params['folderpath'], exist_ok=True)
        
            tot_steps = int(data['T']/data['dt'].Get())
            self.save_params['jump'] = max(int(tot_steps/self.save_params['n_steps']), 1)

            self.save_params['vtk'] = VTKOutput(data['mesh'],
                        coefs = [self.dX_save, self.kappa_save],
                        names = ["displacement", 'mean_curvature'],
                        filename = file_path,
                        subdivision = self.save_params['subdivision'])
            
            if data['mesh'].ne == 0:
                self.save_params['VorB'] = BND
            else:
                self.save_params['VorB'] = VOL
            
            _ = self.get_solution()

            if 'dX' in data:
                data['mesh'].SetDeformation(data['dX'])
            
            self.save_params['vtk'].Do(time=data['t'].Get(), vb = self.save_params['VorB'])
            data['mesh'].UnsetDeformation()
        
    
    def solve_step(self):

        data = self.problem.data

        data['mesh'].SetDeformation(data['dX'])

        compute_stab_mc(data, self.kappa_h, self.params)
        
        ns = specialcf.normal(data['mesh'].dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                        definedon = data['mesh'].Boundaries('.*'))  
        
        data['mesh'].UnsetDeformation()

        if data['mesh'].dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = data['dX_old'])
        elif data['mesh'].dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = data['dX_old'])

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym

        # Compute mesh size, normal and tangential vectors 
        h = specialcf.mesh_size
        ns = specialcf.normal(data['mesh'].dim)
        tE = specialcf.tangential(data['mesh'].dim)
        if data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

        (dX, Y, dkappa), (chi, eta, deta) = self.fes.TnT()

        if data['mesh'].dim == 2:

            dkappa = dkappa*tE
            deta = deta*tE

            jump_dkappadn = (Y.Trace().Deriv()*nE-dkappa)
            jump_detadn = (eta.Trace().Deriv()*nE-deta)

        elif data['mesh'].dim == 3:

            jump_dkappadn = (Y.Trace().Deriv()*nE-dkappa.Trace())
            jump_detadn = (eta.Trace().Deriv()*nE-deta.Trace())

        A = BilinearForm(self.fes)
        A += InnerProduct(dX, chi)/data['dt']*ds_lumped
        A += -InnerProduct(grad(Y).Trace(), grad(chi).Trace())*ds(deformation = data['dX_old'])
        A += InnerProduct(Y, eta)*ds(deformation = data['dX_old'])
        A += (InnerProduct(grad(dX).Trace(), grad(eta).Trace()))*ds(deformation = data['dX_old'])

        gamma = self.params['stab']
        A += gamma*h*InnerProduct(jump_dkappadn,jump_detadn)\
            *ds(deformation = data['dX_old'], element_boundary=True)

        F = LinearForm(self.fes)
        if data['mesh'].dim == 3:
            F += InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(chi).Trace()))*ds(deformation = data['dX_old'])
            F += -2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(chi, Ps)*Ps.trans)*ds(deformation = data['dX_old'])
            F += -self.params['sp_curv']*InnerProduct(self.kappa_h, grad(chi).Trace().trans*ns)*ds_lumped
            F += -0.5*InnerProduct((Norm(self.kappa_h - self.params['sp_curv']*ns)**2)*Ps,grad(chi).Trace())*ds_lumped
            F += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,grad(chi).Trace())*ds_lumped
        elif data['mesh'].dim == 2 :
            F += InnerProduct((Norm(self.Y_h)**2)*Ps,grad(chi).Trace())*ds_lumped

        F += -InnerProduct(Ps, grad(eta).Trace())*ds(deformation = data['dX_old'])
        F += -self.params['sp_curv']*InnerProduct(ns, eta)*ds(deformation = data['dX_old'])
        F += InnerProduct(self.params['rhs'], eta)*ds(deformation = data['dX_old'])

        ## Addition for the open boundary
        if self.params['clamped_bnd']:

            if data['mesh'].dim == 3:
                
                gfF = GridFunction(FacetSurface(data['mesh'], order=0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bnd']))

                F += InnerProduct(nE, eta) * gfF * ds(element_boundary=True)

            elif data['mesh'].dim == 2:

                gfF = GridFunction(H1(data['mesh'], order =1, definedon=data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bnd']))

                F += InnerProduct(gfF*nE, eta) * ds(element_boundary=True)

        with TaskManager():
            A.Assemble()
            F.Assemble()
            Ainv = A.mat.Inverse(self.fes.FreeDofs())

            self.gfu.vec.data = Ainv*F.vec
        
    def update(self):

        data = self.problem.data

        data['mesh'].SetDeformation(data['dX_old'])
        self.kappa_h.Set(self.Y_h + self.params['sp_curv']*specialcf.normal(data['mesh'].dim),
                         definedon=data['mesh'].Boundaries('.*'))
        data['mesh'].UnsetDeformation()

        if self.error_params:

            err = []

            for i, el in enumerate( [self.dX_h, self.kappa_h]):

                err.append(compute_error(data=data, gfu = el, u_ex=self.error_params['ex_sol'][i],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = BND))
            
            towrite = [data['t'].Get()] + err
        
            file_path = os.path.join(self.error_params['folderpath'], self.error_params['filename'])

            # Write the first row
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(towrite)

        if self.save_params:

            if data['iter']%self.save_params['jump'] == 0:

                _ = self.get_solution()

                if 'dX' in data:
                    data['mesh'].SetDeformation(data['dX'])
                
                self.save_params['vtk'].Do(time=data['t'].Get(), vb = self.save_params['VorB'])
                data['mesh'].UnsetDeformation()

    
    def set_solution(self, value = [], **kwargs):

        data = self.problem.data
        self.dX_h.Set(value[0], definedon=data['mesh'].Boundaries(self.params['domain']))
        self.kappa_h.Set(value[1], definedon=data['mesh'].Boundaries(self.params['domain']))

    def get_solution(self):

        data = self.problem.data

        self.dX_save.Set(self.dX_h, definedon = data['mesh'].Boundaries('.*'))
        self.kappa_save.Set(self.kappa_h, definedon = data['mesh'].Boundaries('.*'))

        return [self.dX_save, self.kappa_save]
    
    def draw_solution(self, **kwargs):

        data = self.problem.data
            
        Draw(self.get_solution()[0], deformation = data['dX'])
        Draw(self.get_solution()[1], deformation = data['dX'])

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Willmore flow')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with value ', str(value))

        print(60*'-', '\n')


class WillmoreSolver(BaseSolver):

    def __init__(self, **kwargs):

        super().__init__(fes_order=1, **kwargs)

        accepted_keys = ['rhs', 'sp_curv', 'clamped_bnd', 'domain']
        defaults = [CF(0), CF(0.0), {}, '.*']
        self.params_check(self.params, accepted_keys, defaults)

        self.error_params = None
        self.save_params = None

    def initialize(self):

        data = self.problem.data

        if self.fes_order>1:
            raise Exception("Lumped mass not defined for fes_order>1 !")

        
        if self.params['clamped_bnd']:
            V1 = Compress(VectorH1(data['mesh'], order=self.fes_order,
                        definedon=data['mesh'].Boundaries(self.params['domain']),
                        dirichlet_bbnd = data['mesh'].BBoundaries(self.params['clamped_bnd'])))
        else:
            V1 = Compress(VectorH1(data['mesh'], order=self.fes_order,
                        definedon=data['mesh'].Boundaries(self.params['domain'])))
            
        V2 = Compress(VectorH1(data['mesh'], order=self.fes_order,
                    definedon=data['mesh'].Boundaries(self.params['domain'])))

            
        V_vol = VectorH1(data['mesh'], order = self.fes_order)

        if data['mesh'].ne == 0:
            self.dX_save = GridFunction(V1)
            self.kappa_save = GridFunction(V2)
        else:
            self.dX_save = GridFunction(V_vol)
            self.kappa_save = GridFunction(V_vol)
        
        self.fes = V1*V2
        self.gfu = GridFunction(self.fes)
        self.dX_h, self.Y_h = self.gfu.components
        self.kappa_h = GridFunction(V2)

        compute_mc(data, self.kappa_h, self.params)
        data['mesh'].SetDeformation(data['dX'])
        ns = specialcf.normal(data['mesh'].dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                        definedon = data['mesh'].Boundaries('.*'))  
        data['mesh'].UnsetDeformation()

        if self.error_params:
            err = []

            for i, el in enumerate( [self.dX_h, self.kappa_h]):

                err.append(compute_error(data=data, gfu = el, u_ex=self.error_params['ex_sol'][i],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = BND))

            file_path = os.path.join(self.error_params['folderpath'], self.error_params['filename'])
            os.makedirs(self.error_params['folderpath'], exist_ok=True)

            towrite = [data['t'].Get()] + err
            

            # Define column names
            columns = ['Time', 'displacement', 'mean_curv']

            # Check if file exists
            with open(file_path, mode='w', newline='') as file:  # 'w' mode ensures overwriting
                writer = csv.writer(file)
                writer.writerow(columns)  # Write header row

            # Write the first row
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(towrite)

        if self.save_params:

            file_path = os.path.join(self.save_params['folderpath'], self.save_params['filename'])
            os.makedirs(self.save_params['folderpath'], exist_ok=True)
        
            tot_steps = int(data['T']/data['dt'].Get())
            self.save_params['jump'] = max(int(tot_steps/self.save_params['n_steps']), 1)

            self.save_params['vtk'] = VTKOutput(data['mesh'],
                        coefs = [self.dX_save, self.kappa_save],
                        names = ["displacement", 'mean_curvature'],
                        filename = file_path,
                        subdivision = self.save_params['subdivision'])
            
            if data['mesh'].ne == 0:
                self.save_params['VorB'] = BND
            else:
                self.save_params['VorB'] = VOL
            
            _ = self.get_solution()

            if 'dX' in data:
                data['mesh'].SetDeformation(data['dX'])
            
            self.save_params['vtk'].Do(time=data['t'].Get(), vb = self.save_params['VorB'])
            data['mesh'].UnsetDeformation()
        
    
    def solve_step(self):

        data = self.problem.data

        data['mesh'].SetDeformation(data['dX'])

        compute_mc(data, self.kappa_h, self.params)
        
        ns = specialcf.normal(data['mesh'].dim)
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                        definedon = data['mesh'].Boundaries('.*'))  
        
        data['mesh'].UnsetDeformation()

        if data['mesh'].dim == 2:
            ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ds_lumped = ds(intrules = { SEGM : ir }, deformation = data['dX_old'])
        elif data['mesh'].dim == 3:
            ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = { TRIG : ir }, deformation = data['dX_old'])

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym

        # Compute mesh size, normal and tangential vectors 
        h = specialcf.mesh_size
        ns = specialcf.normal(data['mesh'].dim)
        tE = specialcf.tangential(data['mesh'].dim)
        if data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

        (dX, Y), (chi, eta) = self.fes.TnT()

        A = BilinearForm(self.fes)
        A += InnerProduct(dX, chi)/data['dt']*ds_lumped
        A += -InnerProduct(grad(Y).Trace(), grad(chi).Trace())*ds(deformation = data['dX_old'])
        A += InnerProduct(Y, eta)*ds(deformation = data['dX_old'])
        A += (InnerProduct(grad(dX).Trace(), grad(eta).Trace()))*ds(deformation = data['dX_old'])

        F = LinearForm(self.fes)
        if data['mesh'].dim == 3:
            F += InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(chi).Trace()))*ds(deformation = data['dX_old'])
            F += -2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(chi, Ps)*Ps.trans)*ds(deformation = data['dX_old'])
            F += -self.params['sp_curv']*InnerProduct(self.kappa_h, grad(chi).Trace().trans*ns)*ds_lumped
            F += -0.5*InnerProduct((Norm(self.kappa_h - self.params['sp_curv']*ns)**2)*Ps,grad(chi).Trace())*ds_lumped
            F += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,grad(chi).Trace())*ds_lumped
        elif data['mesh'].dim == 2 :
            F += InnerProduct((Norm(self.Y_h)**2)*Ps,grad(chi).Trace())*ds_lumped

        F += -InnerProduct(Ps, grad(eta).Trace())*ds(deformation = data['dX_old'])
        F += -self.params['sp_curv']*InnerProduct(ns, eta)*ds(deformation = data['dX_old'])
        F += InnerProduct(self.params['rhs'], eta)*ds(deformation = data['dX_old'])

        ## Addition for the open boundary
        if self.params['clamped_bnd']:

            if data['mesh'].dim == 3:
                
                gfF = GridFunction(FacetSurface(data['mesh'], order=0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bnd']))

                F += InnerProduct(nE, eta) * gfF * ds(element_boundary=True)

            elif data['mesh'].dim == 2:

                gfF = GridFunction(H1(data['mesh'], order =1, definedon=data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bnd']))

                F += InnerProduct(gfF*nE, eta) * ds(element_boundary=True)

        with TaskManager():
            A.Assemble()
            F.Assemble()
            Ainv = A.mat.Inverse(self.fes.FreeDofs())

            self.gfu.vec.data = Ainv*F.vec
        
    def update(self):

        data = self.problem.data

        data['mesh'].SetDeformation(data['dX_old'])
        self.kappa_h.Set(self.Y_h + self.params['sp_curv']*specialcf.normal(data['mesh'].dim),
                         definedon=data['mesh'].Boundaries('.*'))
        data['mesh'].UnsetDeformation()

        if self.error_params:

            err = []

            for i, el in enumerate( [self.dX_h, self.kappa_h]):

                err.append(compute_error(data=data, gfu = el, u_ex=self.error_params['ex_sol'][i],
                            norm = self.error_params['norm'], domain = self.params['domain'], 
                            VorB = BND))
            
            towrite = [data['t'].Get()] + err
        
            file_path = os.path.join(self.error_params['folderpath'], self.error_params['filename'])

            # Write the first row
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(towrite)

        if self.save_params:

            if data['iter']%self.save_params['jump'] == 0:

                _ = self.get_solution()

                if 'dX' in data:
                    data['mesh'].SetDeformation(data['dX'])
                
                self.save_params['vtk'].Do(time=data['t'].Get(), vb = self.save_params['VorB'])
                data['mesh'].UnsetDeformation()

    
    def set_solution(self, value = [], **kwargs):

        data = self.problem.data
        self.dX_h.Set(value[0], definedon=data['mesh'].Boundaries('.*'))
        self.kappa_h.Set(value[1], definedon=data['mesh'].Boundaries('.*'))

    def get_solution(self):

        data = self.problem.data

        self.dX_save.Set(self.dX_h, definedon = data['mesh'].Boundaries('.*'))
        self.kappa_save.Set(self.kappa_h, definedon = data['mesh'].Boundaries('.*'))

        return [self.dX_save, self.kappa_save]
    
    def draw_solution(self, **kwargs):

        data = self.problem.data
            
        Draw(self.get_solution()[0], deformation = data['dX'])
        Draw(self.get_solution()[1], deformation = data['dX'])

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Willmore flow')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with value ', str(value))

        print(60*'-', '\n')