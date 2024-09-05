# %%
from cosmos.utils.generate_synapse_meshes import *
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# Only needed for testing
from ngsolve.webgui import Draw
import time as time
import numpy as np

class bs_ADR_CGSolver(SteadySolver):

    def __init__(self, mesh=None, fes_order=1, b=None, c=None, d=None, bnd_cond=None, rhs = CF(0.0), coupling = [CF(1.0), CF(1.0)], verbose = 0):
        
        super().__init__(mesh=mesh, bnd_cond=bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.rhs_v = rhs[0]
        self.b_v = b[0]
        self.c_v = c[0]
        self.d_v = d[0]

        self.rhs_s = rhs[1]
        self.b_s = b[1]
        self.c_s = c[1]
        self.d_s = d[1] # Scalar for diffusion term (no matrix allowed for now)

        self.alpha = coupling[0]
        self.beta = coupling[1]

    def __call__(self):

        self.__setup__()

        fes_v = H1(self.mesh, order=self.fes_order, dgjumps = True)
        fes_s = H1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))
        fes = fes_v*fes_s
        V = VectorH1(self.mesh, order=self.fes_order)
        (u_v, u_s), (v_v, v_s) = fes.TnT()

        dummy = GridFunction(fes_v)

        self.displ_h = GridFunction(V)

        self.a = BilinearForm(fes)
        self.f = LinearForm(fes)

        ## Volume part

        n = specialcf.normal(self.mesh.dim)
        h = specialcf.mesh_size

        diffusion_v = self.d_v*grad(u_v)*grad(v_v) * dx
        reaction_v = self.c_v * u_v*v_v * dx

        # CIP stabilization
        S_int = 0.5*Norm(self.b_v*n) # Parameter corresponding to upwind stabilization
        jump_u = n*(grad(u_v) - (grad(u_v)).Other())
        jump_v = n*(grad(v_v) - (grad(v_v)).Other())
        convection_v = -self.b_v*grad(v_v) * u_v *dx \
            + h**2*S_int*jump_u*jump_v*dx(skeleton=True)
        
        coupling_v = (self.alpha*u_v - self.beta*u_s)*v_v*ds

        self.a += diffusion_v + reaction_v + convection_v + coupling_v

        self.f += self.rhs_v*v_v*dx

        ## Surface part
        
        diffusion_s = self.d_s*grad(u_s).Trace()*grad(v_s).Trace() * ds
        reaction_s = self.c_s*u_s*v_s*ds
        convection_s = -self.b_s*grad(v_s).Trace() * u_s *ds
        coupling_s = -(self.alpha*u_v - self.beta*u_s)*v_s*ds

        self.a += diffusion_s + reaction_s + convection_s + coupling_s

        self.f += self.rhs_s*v_s*ds

        self.gfu = GridFunction(fes)

        self.u_h = self.gfu.components[0]
        self.v_h = self.gfu.components[1]

        with TaskManager():
            self.a.Assemble()
            self.f.Assemble()
            self.a_inv = self.a.mat.Inverse(freedofs = fes.FreeDofs())

            self.gfu.vec.data = self.a_inv * self.f.vec

            dummy.Set(self.v_h, definedon = self.mesh.Boundaries('.*'))

        yield self.u_h

def gradient(f,P):

    m, _ = P.dims
    l = len(f.dims)

    if l == 0:
        # scalar gradient is deifed traditionally

        if m == 2:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y)))
        elif m == 3:
            output = P*CoefficientFunction((f.Diff(x), f.Diff(y), f.Diff(z))) 
            
    elif l == 1:
        # vector gradient is defined component by component
        # and disposed along columns

        if m == 2:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            output = CoefficientFunction((aux1[0], aux2[0], \
                aux1[1], aux2[1]), dims = (m,m))
        elif m == 3:
            aux1 = gradient(f[0], P)
            aux2 = gradient(f[1], P)
            aux3 = gradient(f[2], P)
            output = CoefficientFunction((aux1[0], aux2[0], aux3[0], \
                aux1[1], aux2[1], aux3[1],\
                    aux1[2], aux2[2], aux3[2]), dims = (m,m))
            
    else:

        raise RuntimeError("Don't know how to take the gradient. Only scalars and vectors are accepted.")
    
    return output
        

if __name__ == "__main__":

    n_ex = CF((x, y, z))/Norm(CF((x,y,z)))
    P_ex = Id(3)-OuterProduct(n_ex,n_ex)
    
    d_v = 1 + x**2
    c_v = cos(x)
    b_v = CF((2,1,0))

    d_s = 1 + y**2
    c_s = cos(y)
    b_s = P_ex*CF((0,0,0))

    alpha = 1.0
    beta = 1.0

    # I want a stationary solution on the moving mesh
    u_ex = exp(-x*(x-1)*y*(y-1))
    v_ex = ((d_v*gradient(u_ex, Id(3)) - b_v*u_ex)*n_ex + alpha*u_ex)/beta

    # u_ex = cos(pi*x)*sin(3*y)
    # v_ex = z*cos(5*x)

    flux1_v = b_v*u_ex
    flux2_v = - d_v*gradient(u_ex, Id(3))
    flux_v = flux1_v + flux2_v
    rhs_v = Trace(gradient(flux_v, Id(3))) + c_v*u_ex

    bnd_cond=[['neu', 'membrane', flux2_v],
              ['dir', 'membrane', flux1_v]]

    flux1_s = b_s*v_ex
    flux2_s = - d_s*gradient(v_ex, P_ex)
    flux_s = flux1_s + flux2_s
    rhs_s = Trace(gradient(flux_s, P_ex)) + c_s*v_ex + beta*v_ex - alpha*u_ex

    fes_order = 1
    R0 = 1.0
    _, geo = generate_ball(maxh=0.2, R = R0, order_g =2)
    
    solver = bs_ADR_CGSolver(fes_order=fes_order, b=[b_v, b_s], c=[c_v, c_s], d=[d_v, d_s], rhs=[rhs_v, rhs_s], bnd_cond = bnd_cond, coupling = [alpha, beta])

    conv = Convergence(geom=geo, dh = 0.2, power=1.5, n_refinements=3, time_adapt=True)

    # Convergence order of the bulk solution
    order = conv(solver=solver, exact_sol=u_ex, vol_or_bnd_err='VOL')
    print(order)

    # Convergence order of the surface solution. Change in the algorithm above, function __call__ the output to: 
    # yield self.dummy
    # order = conv(solver=solver, exact_sol=v_ex, vol_or_bnd_err='BND')
    # print(order)

# %%
