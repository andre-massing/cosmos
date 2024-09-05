# %%
from cosmos.utils.generate_synapse_meshes import *
from cosmos.utils.manufactured_solution_tools import Convergence
from cosmos.solvers.solver_base import *
from ngsolve import *

# Only needed for testing
from ngsolve.webgui import Draw
import time as time
import numpy as np

class um_bs_ADR_CGSolver(UnsteadySolver):

    def __init__(self, mesh=None, fes_order=1, b=None, c=None, d=None, dt=0.1, t=Parameter(0.0), T=1.0, u0=None, bnd_cond=None, rhs = CF(0.0), coupling = [CF(1.0), CF(1.0)], displ = None, verbose = 0):
        
        super().__init__(mesh=mesh, dt = dt, t = t, T=T, bnd_cond=bnd_cond, verbose=verbose)

        self.fes_order = fes_order
        self.rhs_v = rhs[0]
        self.b_v = b[0]
        self.c_v = c[0]
        self.d_v = d[0]
        self.u0 = u0[0]

        self.rhs_s = rhs[1]
        self.b_s = b[1]
        self.c_s = c[1]
        self.d_s = d[1]
        self.v0 = u0[1]

        self.displ = displ

        self.alpha = coupling[0]
        self.beta = coupling[1]

    def __build_Ab__(self):

        fes_v = H1(self.mesh, order=self.fes_order, dgjumps = True)
        fes_s = H1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))
        self.fes = fes_v*fes_s
        V = VectorH1(self.mesh, order=self.fes_order)
        Vs = VectorH1(self.mesh, order=self.fes_order, definedon=self.mesh.Boundaries('.*'))
        (u_v, u_s), (v_v, v_s) = self.fes.TnT()

        self.dummy = GridFunction(fes_v)

        self.displ_h = GridFunction(V)
        self.displ_old_h = GridFunction(V)

        self.dt_var = Parameter(self.dt)

        self.b_v_h = GridFunction(V)
        self.b_s_h = GridFunction(Vs)
        self.b_v_h.Set(self.b_v)
        self.b_s_h.Set(self.b_s, definedon = self.mesh.Boundaries('.*'))

        self.a = BilinearForm(self.fes)
        self.m = BilinearForm(self.fes, symmetric = True)
        self.f = LinearForm(self.fes)

        ## Volume part

        n = specialcf.normal(self.mesh.dim)
        h = specialcf.mesh_size

        diffusion_v = self.d_v*grad(u_v)*grad(v_v) * dx(deformation = self.displ_h)
        reaction_v = self.c_v * u_v*v_v * dx(deformation = self.displ_h)
        time_form_v = 1/self.dt_var*u_v*v_v*dx(deformation = self.displ_h)

        # CIP stabilization
        S_int = 0.5*Norm(self.b_v_h*n) # Parameter corresponding to upwind stabilization
        jump_u = n*(grad(u_v) - (grad(u_v)).Other())
        jump_v = n*(grad(v_v) - (grad(v_v)).Other())
        convection_v = -self.b_v_h*grad(v_v) * u_v *dx(deformation = self.displ_h) \
            + h**2*S_int*jump_u*jump_v*dx(skeleton=True, deformation = self.displ_h)
        
        coupling_v = (self.alpha*u_v - self.beta*u_s)*v_v*ds(deformation = self.displ_h)

        self.a += diffusion_v + reaction_v + time_form_v + convection_v + coupling_v

        self.m += 1/self.dt_var*u_v*v_v*dx(deformation = self.displ_h)

        self.f += self.rhs_v*v_v*dx(deformation = self.displ_h)

        ## Surface part
        
        diffusion_s = self.d_s*grad(u_s).Trace()*grad(v_s).Trace() * ds(deformation = self.displ_h)
        reaction_s = self.c_s*u_s*v_s*ds(deformation = self.displ_h)
        time_form_s = 1/self.dt_var*u_s*v_s*ds(deformation = self.displ_h)
        convection_s = -self.b_s_h*grad(v_s).Trace() * u_s *ds(deformation = self.displ_h)
        coupling_s = -(self.alpha*u_v - self.beta*u_s)*v_s*ds(deformation = self.displ_h)

        self.a += diffusion_s + reaction_s + time_form_s + convection_s + coupling_s

        self.m += 1/self.dt_var*u_s*v_s*ds(deformation = self.displ_h)

        self.f += self.rhs_s*v_s*ds(deformation = self.displ_h)

        with TaskManager():
            self.a.Assemble()
            self.a_inv = self.a.mat.Inverse(freedofs = self.fes.FreeDofs())

        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        self.u_h = self.gfu.components[0]
        self.v_h = self.gfu.components[1]

        self.u_h.Set(self.u0)
        self.v_h.Set(self.v0, definedon = self.mesh.Boundaries(".*"))
        self.dummy.Set(self.v_h, definedon = self.mesh.Boundaries('.*'))

    def __call__(self):

        self.__setup__()

        self.__build_Ab__()

        res = self.f.vec.CreateVector()
        self.gfu_old.vec.data = self.gfu.vec.data

        while self.t.Get()<self.T- 0.5 * self.dt_var.Get():

            self.__update__()

            res.data = self.f.vec \
                + self.m.mat*self.gfu_old.vec
            self.gfu.vec.data = self.a_inv * res

            self.dummy.Set(self.v_h, definedon = self.mesh.Boundaries('.*'))

            self.__finalize__()

            yield self.dummy


    def __update__(self):

        with TaskManager():

            self.dt_var.Set(self.dt)

            self.m.Assemble()

            self.t.Set(self.t.Get() + self.dt_var.Get())
            self.displ_h.Set(self.displ_old_h+self.displ)
            # self.displ_h.Set(self.displ)
            self.b_v_h.Set(self.b_v)
            self.b_s_h.Set(self.b_s, definedon = self.mesh.Boundaries('.*'))

            self.a.Assemble() 
            self.a_inv.Update()

            self.f.Assemble()

    def __finalize__(self):

        self.gfu_old.vec.data = self.gfu.vec.data
        self.displ_old_h.vec.data = self.displ_h.vec.data




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

    t = Parameter(0.0)

    # General affine transformation (sphere to ellipse)
    A = CF(((1+0.25*sin(t))*cos(t), -sin(t), 0,\
                sin(t), (1-0.25*sin(t))*cos(t), 0,\
                    0, 0, 1), dims = (3,3))
    b = CF((0.2*t, 0.1*t, 0))


    def compute_transformation(A, b):

        phi = A*CF((x,y,z)) + b
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y,z)) - b)
        w_phi = A.Diff(t)*inv_phi + b.Diff(t)

        e1 = A[:,0]
        e2 = A[:,1]
        e3 = A[:,2]

        n_ex = Cross(e2, e3)*inv_phi[0]+Cross(e3, e1)*inv_phi[1] + Cross(e1, e2)*inv_phi[2]
        n_ex = n_ex/Norm(n_ex)

        return phi, inv_phi, w_phi, detJ, n_ex

    phi, inv_phi, w_phi, detJ, n_ex = compute_transformation(A, b)
    P_ex = Id(3)-OuterProduct(n_ex, n_ex)

    displ_ex = phi - CF((x,y,z))
    
    d_v = 1 + x**2
    c_v = cos(x)
    b_v = CF((2,1,0))

    d_s = 1 + y**2
    c_s = cos(y)
    b_s = P_ex*CF((0,0,0))

    alpha = 1.0
    beta = 1.0

    # I want a stationary solution on the moving mesh
    u_ex = cos(pi*x)*sin(3*y)*cos(t)
    v_ex = ((d_v*gradient(u_ex, Id(3)) - b_v*u_ex)*n_ex + alpha*u_ex)/beta

    # u_ex = cos(pi*x)*sin(3*y)
    # v_ex = z*cos(5*x)

    flux1_v = (b_v*u_ex).Compile()
    flux2_v = (- d_v*gradient(u_ex, Id(3))).Compile()
    rel_flux_v = (w_phi*u_ex).Compile()
    flux_v = flux1_v + flux2_v + rel_flux_v
    rhs_v = (u_ex.Diff(t) + Trace(gradient(flux_v, Id(3))) + c_v*u_ex).Compile()

    bnd_cond=[['neu', 'membrane', flux2_v],
              ['dir', 'membrane', flux1_v]]

    flux1_s = b_s*v_ex
    flux2_s = - d_s*gradient(v_ex, P_ex)
    rel_flux_s = v_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(v_ex, Id(3))
    flux_s = flux1_s + flux2_s
    rhs_s = (v_ex.Diff(t) + rel_flux_s + Trace(gradient(flux_s, P_ex)) + c_s*v_ex + beta*v_ex - alpha*u_ex).Compile()

    fes_order = 1
    R0 = 1.0
    T = 1.0
    dt = 0.2
    _, geo = generate_ball(maxh=0.2, R = R0, order_g =2)
    
    solver = um_bs_ADR_CGSolver(fes_order=fes_order, b=[b_v, b_s], c=[c_v, c_s], d=[d_v, d_s], dt=dt, t=t, T=T, u0=[u_ex, v_ex], rhs=[rhs_v, rhs_s], bnd_cond = bnd_cond, coupling = [alpha, beta], displ=displ_ex)

    conv = Convergence(geom=geo, dh = 0.2, power=1.5, n_refinements=2, time_adapt=True)

    # Convergence order of the bulk solution
    # order = conv(solver=solver, exact_sol=u_ex, vol_or_bnd_err='VOL')
    # print(order)

    # # Convergence order of the surface solution. Change in the algorithm above, function __call__ the output to: 
    # # yield self.dummy
    order = conv(solver=solver, exact_sol=v_ex, vol_or_bnd_err='BND')
    print(order)

# %%
