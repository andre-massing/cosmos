from ngsolve import *
from glapypack.solvers.base_solver import BaseSolver
from glapypack.utils.test_tools import gradient
from ngsolve.webgui import Draw

class MovingSurfaceADRSolver(BaseSolver):

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

            if len(self.params[bc])>1:
                raise ValueError('For surface simulations boundary\
                                conditions must be applied to one\
                                boundary marker only \
                                (possibly composed as in - right|left - )')
            
            if not set(list(self.params[bc].keys())) <= set(data['boundary_mrk'] + ['.*']):
                raise ValueError('Boundary conditions are imposed on\
                                boundary regions which name is not in the \
                                mesh boundary list')

        separator = '|'
        dir_bnd = separator.join(list(self.params['dir_bc'].keys()))

        if self.params['dir_bc']:
            V = H1(data['mesh'], order = self.fes_order,
                   dirichlet_bbnd = dir_bnd, dgjumps = True,
                   definedon=data['mesh'].Boundaries('.*'))
        else:
            V = H1(data['mesh'], order = self.fes_order, dgjumps = True,
                   definedon=data['mesh'].Boundaries('.*'))
        self.fes = V

        if data['mesh'].ne != 0:
            V_vol = H1(data['mesh'], order = self.fes_order)
            self.gfu_vol = GridFunction(V_vol)
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.gfu.Set(self.params['u0'], definedon=data['mesh'].Boundaries('.*'))
        self.sol=[self.gfu.vec.Copy()]
        self.gfu_old.vec.data = self.gfu.vec.data
    
    def solve_step(self):

        data = self.problem.data

        params = self.params_update(self.params)
        if params['advection'].dim == 1:
            params['advection'] = CF((0,)*data['mesh'].dim)

        separator = '|'
        dir_bnd = separator.join(list(params['dir_bc'].keys()))
        flux_c_bnd = separator.join(list(params['flux_c_bc'].keys()))
        flux_d_bnd = separator.join(list(params['flux_d_bc'].keys()))

        if params['dir_bc']:
            dir_cf = params['dir_bc'][dir_bnd]
        if params['flux_c_bc']:
            flux_c_cf = params['flux_c_bc'][flux_c_bnd]
        if params['flux_d_bc']:
            flux_d_cf = params['flux_d_bc'][flux_d_bnd]

        u, v = self.fes.TnT()

        ns = specialcf.normal(data['mesh'].dim)
        tE = specialcf.tangential(data['mesh'].dim)
        h = specialcf.mesh_size
        if data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)

        ds_new = ds(deformation = data['dX'])
        ds_old = ds(deformation = data['dX_old'])

        A = BilinearForm(self.fes)
        diffusion = params['diffusion']*grad(u).Trace()*grad(v).Trace()*ds_new
        reaction = params['reaction']*u*v*ds_new
        # CIP stabilization for convection part
        S_int = 0.5*Norm(params['advection']*nE) # Parameter corresponding to upwind stabilization
        jump_u = nE*(grad(u).Trace() - (grad(u).Trace()).Other())
        jump_v = nE*(grad(v).Trace() - (grad(v).Trace()).Other())
        if data['mesh'].dim == 2:
            gfF = GridFunction(H1(data['mesh'], order = 1,\
                                    definedon=data['mesh'].Boundaries('.*')))
            gfF.Set(1, definedon=data['mesh'].BBoundaries(flux_c_bnd))
        else:
            gfF = GridFunction(FacetSurface(data['mesh'], order = 0))
            gfF.Set(1, definedon=data['mesh'].BBoundaries(flux_c_bnd))
        advection = -params['advection']*grad(v).Trace() * u *ds_new\
                    + IfPos(InnerProduct(nE, params['advection']), 
                            InnerProduct(nE, params['advection'])*u, 0)\
                                *gfF*v*ds_new(element_boundary=True)
        
        A += reaction + diffusion + advection
        
        F = LinearForm(self.fes)
        F += params['rhs']*v*ds_new
        if params['flux_c_bc']:
            F += -IfPos(InnerProduct(nE, params['advection']), 0,
                         InnerProduct(nE, flux_c_cf))*gfF*v*ds_new(element_boundary=True)
        if params['flux_d_bc']:
            F += -InnerProduct(nE, flux_d_cf)*gfF*v*ds_new(element_boundary=True)

        M = BilinearForm(self.fes, symmetric = True)
        M += 1/data['dt']*u*v*ds_new

        M_old = BilinearForm(self.fes, symmetric = True)
        M_old += 1/data['dt']*u*v*ds_old

        with TaskManager():
            A.Assemble()
            F.Assemble()
            M.Assemble()
            M_old.Assemble()

            res = F.vec
            res += M_old.mat*self.gfu_old.vec

            Mstar = A.mat.CreateMatrix()
            Mstar.AsVector().data = M.mat.AsVector() + A.mat.AsVector()
            if params['dir_bc']: 

                data['mesh'].SetDeformation(data['dX'])
                self.gfu.Set(dir_cf, definedon = data['mesh'].BBoundaries(dir_bnd))
                data['mesh'].UnsetDeformation()

                res.data += - Mstar * self.gfu.vec

                self.gfu.vec.data += Mstar.Inverse(freedofs=self.fes.FreeDofs()) * res

            else:

                self.gfu.vec.data = Mstar.Inverse(freedofs=self.fes.FreeDofs()) * res

    def update(self):

        self.gfu_old.vec.data = self.gfu.vec.data
        self.sol.append(self.gfu.vec.Copy())

    def set_solution(self, value = CF(0), **kwargs):

        data = self.problem.data
        self.gfu.Set(value, definedon=data['mesh'].Boundaries('.*'))

    def get_solution(self):

        data = self.problem.data

        if data['mesh'].ne == 0:
            return self.gfu
        else:
            self.gfu_vol.Set(self.gfu, definedon = data['mesh'].Boundaries('.*'))

            return self.gfu_vol
    
    def draw_solution(self, **kwargs):

        data = self.problem.data
            
        Draw(self.get_solution(), deformation = data['dX'])

    def save_solution(self, **kwargs):

        params = kwargs
        accepted_keys = ['filename', 'subdivision']
        defaults = ['sol', 1]  
        self.params_check(params, accepted_keys, defaults)

        data = self.problem.data
        if data['mesh'].ne != 0:
            vtk = VTKOutput(data['mesh'],
                        coefs=[self.gfu_vol],
                        names=["adr_sol"],
                        filename=params['filename'],
                        subdivision=params['subdivision'])
            
            v_or_b = VOL
        else: 
            vtk = VTKOutput(data['mesh'],
                        coefs=[self.gfu],
                        names=["adr_sol"],
                        filename=params['filename'],
                        subdivision=params['subdivision']) 
            v_or_b = BND
        
        for i, sol_i in enumerate(self.sol):

            data['dX'].vec.data = data['dX_array'][i].data
            data['mesh'].SetDeformation(data['dX'])
            self.gfu.vec.data = sol_i.data
            _ = self.get_solution()
            vtk.Do(time=data['t_array'][i], vb = v_or_b)
            data['mesh'].UnsetDeformation()

    def compute_error(self, u_ex, norm):

        data = self.problem.data

        err = []
        ns = specialcf.normal(data['mesh'].dim)
        Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

        if norm == 'H1':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])
                data['dX'].vec.data = data['dX_array'][i].data
                data['mesh'].SetDeformation(data['dX'])
                self.gfu.vec.data = sol_i.data

                aux = InnerProduct(grad(self.gfu).Trace()-gradient(u_ex, Ps), \
                                    grad(self.gfu).Trace()-gradient(u_ex, Ps))

                err.append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                data['mesh'].UnsetDeformation()

        elif norm == 'L2':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])
                data['dX'].vec.data = data['dX_array'][i].data
                data['mesh'].SetDeformation(data['dX'])
                self.gfu.vec.data = sol_i.data

                aux = InnerProduct(self.gfu-u_ex, self.gfu-u_ex)

                err.append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                data['mesh'].UnsetDeformation()

        else:

            raise ValueError('The norm given is not implemented')

        return err

    def print_info(self):

        print(60*'-')

        print('This is a solver for the Advection-Diffusion-Reaction problem')
        print('This meant to be a moving surface simulation')
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
