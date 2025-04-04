from ngsolve import *
import time

class CouplingScheme():

    def __init__(self):
        pass

    def Solve(self, *args, **kwargs):

        raise Exception('Base class method is called! Empty method!')
    
class DisplacementCoupling(CouplingScheme):

    def __init__(self, type = 'Explicit'):
        
        self.type = type
        self.displacements = []

    def SolveStep(self, dX_data, dX_pdes, data, pdes):

        if self.type == 'Explicit':
            self.SolveExplicitStep(dX_data, dX_pdes, data, pdes)
        elif self.type == 'Staggered':
            self.SolveStaggeredStep(dX_data, dX_pdes, data, pdes)

    def AddDisplacement(self, function, domain, v_or_b, total):

        self.displacements.append([function, domain, v_or_b, total])

    def ComputeDisplacement(self, dX_data, data):

        dX_loc = GridFunction(dX_data.dX.space)

        dX_data.dX.vec.data = dX_data.dX_old.vec.data
        for f, domain, v_or_b, total in self.displacements:

            if data.mesh.ne != 0 and v_or_b == BND:
                # raise Exception('Do not know how to extend the displacement to the volume part!')
                pass
            
            if v_or_b == BND:
                if total:
                    data.mesh.UnsetDeformation()
                    dX_loc.Set(f() - dX_data.dX_old, definedon = data.mesh.Boundaries(domain))
                    data.mesh.SetDeformation(dX_data.dX_old)
                else:
                    dX_loc.Set(f(), definedon = data.mesh.Boundaries(domain))
            elif v_or_b == VOL:
                if total:
                    data.mesh.UnsetDeformation()
                    dX_loc.Set(f() - dX_data.dX_old, definedon = data.mesh.Materials(domain))
                    data.mesh.SetDeformation(dX_data.dX_old)
                else:
                    dX_loc.Set(f(), definedon = data.mesh.Materials(domain))
                    
            dX_data.dX.vec.data += dX_loc.vec.data

    def UpdateDisplacement(self, dX_data, data):

        dX_data.dX_old.vec.data = dX_data.dX.vec.data
        data.mesh.SetDeformation(dX_data.dX)

    def SolveExplicitStep(self, dX_data, dX_pdes, data, pdes):

        for pde, scheme in dX_pdes:
            scheme.Solve(data, pde, dX_data)
        self.ComputeDisplacement(dX_data, data)
        for pde, scheme in pdes:
            scheme.Solve(data, pde, dX_data)
        self.UpdateDisplacement(dX_data, data)

    def SolveStaggeredStep(self, dX_data, dX_pdes, data, pdes):

        converged = False
        tol = 1e-2

        while not converged:
            for pde, scheme in dX_pdes:
                scheme.Solve(data, pde, dX_data)
            self.ComputeDisplacement(dX_data, data)
            for pde, scheme in pdes:
                scheme.Solve(data, pde, dX_data)
            self.UpdateDisplacement(dX_data, data)
            converged = True
            for pde, scheme in dX_pdes:
                if Norm(pde.gfu.vec - pde.gfu_old.vec)/Norm(pde.gfu_old.vec)>tol:
                    converged = False
                pde.gfu_old.vec.data = pde.gfu.vec.data
            for pde, scheme in pdes:
                if Norm(pde.gfu.vec - pde.gfu_old.vec)/Norm(pde.gfu_old.vec)>tol:
                    converged = False
                pde.gfu_old.vec.data = pde.gfu.vec.data

        # self.UpdateDisplacement(dX_data, data)


