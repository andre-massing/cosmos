from ngsolve import *
from cosmos.solvers.base_solver import BaseSolver
from cosmos.utils.manufactured_solution_tools import gradient
from ngsolve.webgui import Draw

class StabWillmoreSolver(BaseSolver):

    def __init__(self, fes_order=1, **kwargs):

        super().__init__(fes_order=fes_order, **kwargs)

        accepted_keys = ['rhs', 'stab', 'sp_curv', 'clamped_bc']
        defaults = [CF(0), CF(1e-3), CF(0.0), {}]
        self.params_check(self.params, accepted_keys, defaults)

    def initialize(self):

        data = self.problem.data

        if self.fes_order>1:
            raise Exception("Lumped mass not defined for fes_order>1 !")
        
        if self.params['clamped_bc']:
            if not set([self.params['clamped_bc']]) <= set(data['boundary_mrk'] + ['.*']):

                raise ValueError('Dirichlet boundary conditions are imposed on boundary regions which name is not in the mesh boundary list')
        
        if self.params['clamped_bc']:
            V1 = VectorH1(data['mesh'], order=self.fes_order,
                        definedon=data['mesh'].Boundaries('.*'),
                        dirichlet_bbnd = data['mesh'].BBoundaries(self.params['clamped_bc']))
        else:
            V1 = VectorH1(data['mesh'], order=self.fes_order,
                        definedon=data['mesh'].Boundaries('.*'))
            
        V2 = VectorH1(data['mesh'], order=self.fes_order,
                    definedon=data['mesh'].Boundaries('.*'))
        
        if data['mesh'].dim == 2:
            
            dV = H1(data['mesh'], order=1,\
                     definedon = data['mesh'].Boundaries('.*'))
        elif data['mesh'].dim == 3:
            
            dV = VectorFacetSurface(data['mesh'], order=1,\
                    definedon = data['mesh'].Boundaries('.*'))
            
        if data['mesh'].ne != 0:
            V_vol = VectorH1(data['mesh'], order = self.fes_order)
            self.dX_vol = GridFunction(V_vol)
            self.kappa_vol = GridFunction(V_vol)
        
        self.fes = V1*V2*dV
        self.gfu = GridFunction(self.fes)
        self.dX_h, self.Y_h, _ = self.gfu.components
        self.kappa_h = GridFunction(V2)

        self.initialize_curv(data)

        self.sol.append([self.dX_h.vec.Copy(), self.kappa_h.vec.Copy()])

    def initialize_curv(self, data):

        V2 = VectorH1(data['mesh'], order=self.fes_order,
                    definedon=data['mesh'].Boundaries('.*'))
        
        if data['mesh'].dim == 2:
            
            dV = H1(data['mesh'], order=1,\
                     definedon = data['mesh'].Boundaries('.*'))
        elif data['mesh'].dim == 3:
            
            dV = VectorFacetSurface(data['mesh'], order=1,\
                    definedon = data['mesh'].Boundaries('.*'))

        ds_def = ds(deformation = data['dX_old'])

        h = specialcf.mesh_size
        ns = specialcf.normal(data['mesh'].dim)
        tE = specialcf.tangential(data['mesh'].dim)
        if data['mesh'].dim == 2:
            nE = tE
        else:
            nE = Cross(ns, tE)
        Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

        fes0 = V2*dV
        gfu0 = GridFunction(fes0)
        
        (kappa0, dkappa0), (eta0, deta0) = fes0.TnT()

        if data['mesh'].dim == 2:

            dkappa0 = dkappa0*tE
            deta0 = deta0*tE

            jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0)
            jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0)

        elif data['mesh'].dim == 3:

            jump_dkappadn0 = (kappa0.Trace().Deriv()*nE-dkappa0.Trace())
            jump_detadn0 = (eta0.Trace().Deriv()*nE-deta0.Trace())

        A0 = BilinearForm(fes0)
        F0 = LinearForm(fes0)
        A0 += kappa0*eta0*ds_def  # Adding traces here?
        gamma = self.params['stab']
        A0 += gamma*h*InnerProduct(jump_dkappadn0,jump_detadn0)\
            *ds_def(element_boundary=True)
        A0.Assemble()

        F0 += -InnerProduct(Ps, Grad(eta0).Trace())*ds_def

        if self.params['clamped_bc']:

            if data['mesh'].dim == 3:
                gfF = GridFunction(FacetSurface(data['mesh'], order=0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bc']))

                F0 += InnerProduct(nE, eta0) * gfF * ds_def(element_boundary=True)

            elif data['mesh'].dim == 2:

                gfF = GridFunction(H1(data['mesh'], order =1,\
                        definedon=data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bc']))

                F0 += InnerProduct(gfF*nE, eta0) \
                    * ds_def(element_boundary=True)

        F0.Assemble()
        gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec

        self.kappa_h.vec.data = gfu0.components[0].vec.data
        data['mesh'].SetDeformation(data['dX_old'])
        self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns,
                     definedon = data['mesh'].Boundaries('.*'))  
        data['mesh'].UnsetDeformation()
    
    def solve_step(self):

        data = self.problem.data

        # self.initialize_curv(data)

        ds_def = ds(deformation = data['dX_old'])
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
        A += -InnerProduct(grad(Y).Trace(), grad(chi).Trace())*ds_def
        A += InnerProduct(Y, eta)*ds_def
        A += (InnerProduct(grad(dX).Trace(), grad(eta).Trace()))*ds_def

        gamma = self.params['stab']
        A += gamma*h*InnerProduct(jump_dkappadn,jump_detadn)\
            *ds_def(element_boundary=True)

        F = LinearForm(self.fes)
        if data['mesh'].dim == 3:
            F += InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(chi).Trace()))*ds_def
            F += -2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(chi, Ps)*Ps.trans)*ds_def
            F += -self.params['sp_curv']*InnerProduct(self.kappa_h, grad(chi).Trace().trans*ns)*ds_lumped
            F += -0.5*InnerProduct((Norm(self.kappa_h - self.params['sp_curv']*ns)**2)*Ps,grad(chi).Trace())*ds_lumped
            F += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,grad(chi).Trace())*ds_lumped
        elif data['mesh'].dim == 2 :
            F += InnerProduct((Norm(self.Y_h)**2)*Ps,grad(chi).Trace())*ds_lumped

        F += -InnerProduct(Ps, grad(eta).Trace())*ds_def
        F += -self.params['sp_curv']*InnerProduct(ns, eta)*ds_def
        F += InnerProduct(self.params['rhs'], eta)*ds_def

        ## Addition for the open boundary
        if self.params['clamped_bc']:

            if data['mesh'].dim == 3:
                
                gfF = GridFunction(FacetSurface(data['mesh'], order=0))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bc']))

                F += InnerProduct(nE, eta) * gfF * ds(element_boundary=True)

            elif data['mesh'].dim == 2:

                gfF = GridFunction(H1(data['mesh'], order =1, definedon=data['mesh'].Boundaries('.*')))
                gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bc']))

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

        self.sol.append([self.dX_h.vec.Copy(), self.kappa_h.vec.Copy()])

    
    def set_solution(self, value = [], **kwargs):

        data = self.problem.data
        self.dX_h.Set(value[0], definedon=data['mesh'].Boundaries('.*'))
        self.kappa_h.Set(value[1], definedon=data['mesh'].Boundaries('.*'))

    def get_solution(self):

        data = self.problem.data

        if data['mesh'].ne == 0:
            return [self.dX_h, self.kappa_h]
        else:
            self.dX_vol.Set(self.dX_h, definedon = data['mesh'].Boundaries('.*'))
            self.kappa_vol.Set(self.kappa_h, definedon = data['mesh'].Boundaries('.*'))

            return [self.dX_vol, self.kappa_vol]

        return 
    
    def draw_solution(self, **kwargs):

        data = self.problem.data
            
        Draw(self.get_solution()[0], deformation = data['dX'])
        Draw(self.get_solution()[1], deformation = data['dX'])

    def save_solution(self, **kwargs):

        params = kwargs
        accepted_keys = ['filename', 'subdivision']
        defaults = ['sol', 1]  
        self.params_check(params, accepted_keys, defaults)

        data = self.problem.data
        if data['mesh'].ne != 0:
            vtk = VTKOutput(data['mesh'],
                        coefs=[self.dX_vol, self.kappa_vol],
                        names=["displacement", "mean_curvature"],
                        filename=params['filename'],
                        subdivision=params['subdivision']) 
            
            v_or_b = VOL
        else: 
            vtk = VTKOutput(data['mesh'],
                        coefs=[self.dX_h, self.kappa_h],
                        names=["displacement", "mean_curvature"],
                        filename=params['filename'],
                        subdivision=params['subdivision']) 
            v_or_b = BND

        
        for i, sol_i in enumerate(self.sol):

            data['dX'].vec.data = data['dX_array'][i].data
            data['mesh'].SetDeformation(data['dX'])
            for j in range(2):
                self.gfu.components[j].vec.data = sol_i[j].data
            _ = self.get_solution()
            vtk.Do(time=data['t_array'][i], vb = v_or_b)
            data['mesh'].UnsetDeformation()

    def compute_error(self, u_ex, norm):

        data = self.problem.data

        err = [[], []]
        ns = specialcf.normal(data['mesh'].dim)
        Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

        if norm == 'H1':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])
                data['dX'].vec.data = data['dX_array'][i].data
                data['mesh'].SetDeformation(data['dX'])

                for j in range(2):

                    self.gfu.components[j].vec.data = sol_i[j].data

                    aux = InnerProduct(grad(self.gfu.components[j]).Trace()-gradient(u_ex[j], Ps), \
                                        grad(self.gfu.components[j]).Trace()-gradient(u_ex[j], Ps))

                    err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                data['mesh'].UnsetDeformation()

        elif norm == 'L2':

            for i, sol_i in enumerate(self.sol):

                data['t'].Set(data['t_array'][i])
                data['dX'].vec.data = data['dX_array'][i].data
                data['mesh'].SetDeformation(data['dX'])

                for j in range(2):

                    self.gfu.components[j].vec.data = sol_i[j].data

                    aux = InnerProduct(self.gfu.components[j]-u_ex[j], self.gfu.components[j]-u_ex[j])

                    err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

                data['mesh'].UnsetDeformation()

        else:

            raise ValueError('The norm given is not implemented')

        return err

    def print_solver_info(self):

        print(60*'-')

        print('This is a solver for the Willmore flow')
        print('It computes one step of it given the mesh')
        print('The algorithm is described in the article:')
        print('Stabilization for the mean curvature is implemented as in the article:')
        print('It uses conforming H-1 elements of order ', self.fes_order)

        print('Its parameters are')
        for key, value in self.params.items():
            print('  -', key, '- with type ', type(value))

        print(60*'-', '\n')


# class WillmoreSolver(BaseSolver):

#     def __init__(self, fes_order=1, params = None, clamped_bc = ''):

#         super().__init__(fes_order=fes_order, params=params)

#         self.params['clamped_bc'] = clamped_bc

#     def initialize(self):

#         data = self.problem.data

#         # Initialize the parameters
#         accepted_keys = ['rhs', 'sp_curv']
#         defaults = [CF((0,)*data['mesh'].dim), CF(0.0)]  
#         self.__params_setup__(self.params, accepted_keys, defaults)

#         if self.fes_order>1:
#             raise Exception("Lumped mass not defined for fes_order>1 !")
        
#         if self.params['clamped_bc']:
#             if not set([self.params['clamped_bc']]) <= set(data['boundary_mrk'] + ['.*']):

#                 raise ValueError('Dirichlet boundary conditions are imposed on boundary regions which name is not in the mesh boundary list')
        
#         if self.params['clamped_bc']:
#             V1 = VectorH1(data['mesh'], order=self.fes_order,
#                         definedon=data['mesh'].Boundaries('.*'),
#                         dirichlet_bbnd = data['mesh'].BBoundaries(self.params['clamped_bc']))
#         else:
#             V1 = VectorH1(data['mesh'], order=self.fes_order,
#                         definedon=data['mesh'].Boundaries('.*'))
            
#         V2 = VectorH1(data['mesh'], order=self.fes_order,
#                     definedon=data['mesh'].Boundaries('.*'))
        
        
#         self.fes = V1*V2
#         self.gfu = GridFunction(self.fes)
#         self.dX_h, self.Y_h = self.gfu.components
#         self.kappa_h = GridFunction(V2)

#         self.initialize_curv(data)

#         self.sol.append([self.dX_h.vec.Copy(), self.kappa_h.vec.Copy()])

#     def initialize_curv(self, data):

#         V2 = VectorH1(data['mesh'], order=self.fes_order,
#                     definedon=data['mesh'].Boundaries('.*'))

#         ds_def = ds(deformation = data['dX_old'])

#         h = specialcf.mesh_size
#         ns = specialcf.normal(data['mesh'].dim)
#         tE = specialcf.tangential(data['mesh'].dim)
#         if data['mesh'].dim == 2:
#             nE = tE
#         else:
#             nE = Cross(ns, tE)
#         Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

#         fes0 = V2
#         gfu0 = GridFunction(fes0)
        
#         kappa0, eta0 = fes0.TnT()

#         A0 = BilinearForm(fes0)
#         F0 = LinearForm(fes0)
#         A0 += kappa0*eta0*ds_def  # Adding traces here?
#         A0.Assemble()

#         F0 += -InnerProduct(Ps, Grad(eta0).Trace())*ds_def

#         if self.params['clamped_bc']:

#             if data['mesh'].dim == 3:
#                 gfF = GridFunction(FacetSurface(data['mesh'], order=0))
#                 gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bc']))

#                 F0 += InnerProduct(nE, eta0) * gfF * ds_def(element_boundary=True)

#             elif data['mesh'].dim == 2:

#                 gfF = GridFunction(H1(data['mesh'], order =1,\
#                         definedon=data['mesh'].Boundaries('.*')))
#                 gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bc']))

#                 F0 += InnerProduct(gfF*nE, eta0) \
#                     * ds_def(element_boundary=True)

#         F0.Assemble()
#         gfu0.vec.data = A0.mat.Inverse(fes0.FreeDofs())*F0.vec

#         self.kappa_h.vec.data = gfu0.vec.data
#         data['mesh'].SetDeformation(data['dX_old'])
#         self.Y_h.Set(self.kappa_h - self.params['sp_curv']*ns, definedon = data['mesh'].Boundaries('.*'))  
#         data['mesh'].UnsetDeformation()
          
    
#     def solve_step(self):

#         data = self.problem.data

#         self.initialize_curv(data)

#         ds_def = ds(deformation = data['dX_old'])
#         if data['mesh'].dim == 2:
#             ir = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
#             ds_lumped = ds(intrules = { SEGM : ir }, deformation = data['dX_old'])
#         elif data['mesh'].dim == 3:
#             ir = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
#             ds_lumped = ds(intrules = { TRIG : ir }, deformation = data['dX_old'])

#         def D_s(chi, Ps):
#             sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
#             return sym

#         # Compute mesh size, normal and tangential vectors 
#         h = specialcf.mesh_size
#         ns = specialcf.normal(data['mesh'].dim)
#         tE = specialcf.tangential(data['mesh'].dim)
#         if data['mesh'].dim == 2:
#             nE = tE
#         else:
#             nE = Cross(ns, tE)
#         Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

#         (dX, Y), (chi, eta) = self.fes.TnT()


#         A = BilinearForm(self.fes)
#         A += InnerProduct(dX, chi)/data['dt']*ds_lumped
#         A += -InnerProduct(grad(Y).Trace(), grad(chi).Trace())*ds_def
#         A += InnerProduct(Y, eta)*ds_def
#         A += (InnerProduct(grad(dX).Trace(), grad(eta).Trace()))*ds_def

#         F = LinearForm(self.fes)
#         if data['mesh'].dim == 3:
#             F += InnerProduct(Trace(grad(self.Y_h).Trace()),Trace(grad(chi).Trace()))*ds_def
#             F += -2*InnerProduct(grad(self.Y_h).Trace().trans, D_s(chi, Ps)*Ps.trans)*ds_def
#             F += -self.params['sp_curv']*InnerProduct(self.kappa_h, grad(chi).Trace().trans*ns)*ds_lumped
#             F += -0.5*InnerProduct((Norm(self.kappa_h - self.params['sp_curv']*ns)**2)*Ps,grad(chi).Trace())*ds_lumped
#             F += InnerProduct(InnerProduct(self.Y_h, self.kappa_h)*Ps,grad(chi).Trace())*ds_lumped
#         elif data['mesh'].dim == 2 :
#             F += InnerProduct((Norm(self.Y_h)**2)*Ps,grad(chi).Trace())*ds_lumped

#         F += -InnerProduct(Ps, grad(eta).Trace())*ds_def
#         F += -self.params['sp_curv']*InnerProduct(ns, eta)*ds_def
#         F += InnerProduct(self.params['rhs'], eta)*ds_def

#         ## Addition for the open boundary
#         if self.params['clamped_bc']:

#             if data['mesh'].dim == 3:
                
#                 gfF = GridFunction(FacetSurface(data['mesh'], order=0))
#                 gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bc']))

#                 F += InnerProduct(nE, eta) * gfF * ds(element_boundary=True)

#             elif data['mesh'].dim == 2:

#                 gfF = GridFunction(H1(data['mesh'], order =1, definedon=data['mesh'].Boundaries('.*')))
#                 gfF.Set(1, definedon=data['mesh'].BBoundaries(self.params['clamped_bc']))

#                 F += InnerProduct(gfF*nE, eta) * ds(element_boundary=True)

#         with TaskManager():
#             A.Assemble()
#             F.Assemble()
#             Ainv = A.mat.Inverse(self.fes.FreeDofs())

#             self.gfu.vec.data = Ainv*F.vec
        
#     def update(self):

#         data = self.problem.data

#         data['mesh'].SetDeformation(data['dX_old'])
#         self.kappa_h.Set(self.Y_h + self.params['sp_curv']*specialcf.normal(data['mesh'].dim),
#                          definedon=data['mesh'].Boundaries('.*'))
#         data['mesh'].UnsetDeformation()

#         self.sol.append([self.dX_h.vec.Copy(), self.kappa_h.vec.Copy()])

#     def get_solution(self):

#         return [self.dX_h, self.kappa_h]

#     def compute_error(self, u_ex, norm):

#         data = self.problem.data

#         err = [[], []]
#         ns = specialcf.normal(data['mesh'].dim)
#         Ps = Id(data['mesh'].dim) - OuterProduct(ns, ns)

#         if norm == 'H1':

#             for i, sol_i in enumerate(self.sol):

#                 data['t'].Set(data['t_array'][i])
#                 data['dX'].vec.data = data['dX_array'][i].data
#                 data['mesh'].SetDeformation(data['dX'])

#                 for j in range(2):

#                     self.gfu.components[j].vec.data = sol_i[j].data

#                     aux = InnerProduct(grad(self.gfu.components[j]).Trace()-gradient(u_ex[j], Ps), \
#                                         grad(self.gfu.components[j]).Trace()-gradient(u_ex[j], Ps))

#                     err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

#                 data['mesh'].UnsetDeformation()

#         elif norm == 'L2':

#             for i, sol_i in enumerate(self.sol):

#                 data['t'].Set(data['t_array'][i])
#                 data['dX'].vec.data = data['dX_array'][i].data
#                 data['mesh'].SetDeformation(data['dX'])

#                 for j in range(2):

#                     self.gfu.components[j].vec.data = sol_i[j].data

#                     aux = InnerProduct(self.gfu.components[j]-u_ex[j], self.gfu.components[j]-u_ex[j])

#                     err[j].append(sqrt(Integrate(aux, mesh = data['mesh'], order = self.fes_order*2, VOL_or_BND=BND)))

#                 data['mesh'].UnsetDeformation()

#         else:

#             raise ValueError('The norm given is not implemented')

#         return err
    
#     def draw_solution(self):

#         data = self.problem.data

#         import time

#         if data['mesh'].ne == 0:
#             V = VectorH1(data['mesh'], order = self.fes_order, definedon = data['mesh'].Boundaries('.*'))
#             gfu = GridFunction(V)

#             # Plotting the displacement
#             gfu.vec.data = self.sol[0][0].data
#             data['dX'].vec.data = data['dX_array'][0].data

#             scene = Draw(gfu, deformation = data['dX'])

#             for i, sol_i in enumerate(self.sol[1:]):

#                 time.sleep(0.1)

#                 gfu.vec.data = sol_i[0].data
#                 data['dX'].vec.data = data['dX_array'][i].data

#                 scene.Redraw()

#             # Plotting the mean curvature
#             gfu.vec.data = self.sol[0][1].data
#             data['dX'].vec.data = data['dX_array'][0].data

#             scene = Draw(gfu, deformation = data['dX'])

#             for i, sol_i in enumerate(self.sol[1:]):

#                 time.sleep(0.1)

#                 gfu.vec.data = sol_i[1].data
#                 data['dX'].vec.data = data['dX_array'][i].data

#                 scene.Redraw()
#         else:

#             V_v = VectorH1(data['mesh'], order = self.fes_order)
#             gfu_v = GridFunction(V_v)
#             V_s = VectorH1(data['mesh'], order = self.fes_order, definedon = data['mesh'].Boundaries('.*'))
#             gfu_s = GridFunction(V_s)

#             # Plotting the displacement
#             gfu_s.vec.data = self.sol[0][0].data
#             gfu_v.Set(gfu_s, definedon = data['mesh'].Boundaries('.*'))

#             for i, sol_i in enumerate(self.sol[1:]):


#                 gfu_s.vec.data = sol_i[0].data
#                 gfu_v.Set(gfu_s, definedon = data['mesh'].Boundaries('.*'))

#                 scene.Redraw()

#             # Plotting the mean curvature
#             gfu_s.vec.data = self.sol[0][1].data
#             gfu_v.Set(gfu_s, definedon = data['mesh'].Boundaries('.*'))

#             for i, sol_i in enumerate(self.sol[1:]):

#                 gfu_s.vec.data = sol_i[1].data
#                 gfu_v.Set(gfu_s, definedon = data['mesh'].Boundaries('.*'))

#                 scene.Redraw()

#     def print_solver_info(self):

#         print(60*'-')

#         print('This is a solver for the Willmore flow')
#         print('It computes one step of it given the mesh')
#         print('The algorithm is described in the article:')
#         print('Stabilization for the mean curvature is implemented as in the article:')
#         print('It uses conforming H-1 elements of order ', self.fes_order)

#         print('Its parameters are')
#         for key, value in self.params.items():
#             print('  -', key, '- with type ', type(value))

#         print(60*'-', '\n')