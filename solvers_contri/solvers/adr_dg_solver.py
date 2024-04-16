# %%
import sys
sys.path.insert(0, '../solvers/')
from solver_base import SteadySolver
from ngsolve import *

# # Only needed for testing
# sys.path.insert(0, '../geometries/')
# from generate_meshes import *
from ngsolve.webgui import Draw

class ADR_DGSolver(SteadySolver):

    fields_num = 1
    dir_bnd = ''
    neu_bnd = ''

    def __init__(self, mesh, fes_order, b, c, d, bnd_cond=[["dir", "default", CF(0.0)]], rhs=CF(0.0)):
        
        super().__init__(mesh, bnd_cond)

        self.fes_order = fes_order
        self.rhs = rhs
        self.b = b # Coefficient function/ for convection term
        self.c = c # Coefficient function/scalar for reaction term
        self.d = d # Scalar for diffusion term (no matrix allowed for now)


    def __call__(self):

        if self.dir_bnd == '':

            # What happens if I only have Neumann boundary conditions?

            pass

        else: 
            fes = L2(self.mesh, order=int(self.fes_order), dgjumps = True)
            u,v = fes.TnT()

            # trial and test function jump across elements
            jump_u = u-u.Other()
            jump_v = v-v.Other()
            n = specialcf.normal(self.mesh.dim)
            # trial an test function mean gradient across elements
            mean_dudn = 0.5*n * (self.d*grad(u)+self.d*grad(u.Other()))
            mean_dvdn = 0.5*n * (self.d*grad(v)+self.d*grad(v.Other()))

            alpha = 4*self.d  # Paramter for the stabilization of the diffusive term
            h = specialcf.mesh_size
            a = BilinearForm(fes)
            
            diffusion = self.d*grad(u)*grad(v) * dx \
                +alpha*self.fes_order**2/h*jump_u*jump_v * dx(skeleton=True) \
                +(-mean_dudn*jump_v-mean_dvdn*jump_u) * dx(skeleton=True) \
                +alpha*self.fes_order**2/h*u*v * ds(skeleton=True) \
                +self.d*(-n*grad(u)*v-n*grad(v)*u)* ds(skeleton=True)


            reaction = self.c * u*v * dx 

            bn_plus = IfPos(self.b*n, self.b*n, 0)
            bn_minus = IfPos(self.b*n, 0, -self.b*n)
            mean_u = 0.5*(u+u.Other())

            S_int = 0.5*Norm(self.b*n) # Parameter corresponding to upwind stabilization
            convection = - u*(self.b*grad(v))*dx + bn_plus*u*v * ds(skeleton=True)\
                + self.b*n*mean_u*jump_v * dx(skeleton=True) \
                +S_int*jump_u*jump_v * dx(skeleton=True)

            if (self.d==0):
                a += reaction + convection
            elif (Integrate(Norm(self.b), self.mesh)<1e-14):
                a += reaction + diffusion
            else:
                a += reaction + convection + diffusion
            
            a.Assemble()

            f = LinearForm(fes)
            # rhs taking care of boundary conditions
            f_diff = - self.d*grad(v)*n*self.dir_cf * ds(skeleton=True) \
                    + alpha*self.fes_order**2/h*self.dir_cf*v * ds(skeleton=True)
            f_conv = bn_minus*self.dir_cf*v * ds(skeleton=True)

            if (self.d==0):
                f += self.rhs*v * dx + f_conv
            elif (Integrate(Norm(self.b), self.mesh)<1e-14):
                f += self.rhs*v * dx + f_diff    
            else:
                f += self.rhs*v * dx + f_diff + f_conv      
            
            f.Assemble()

            gfu = GridFunction(fes)
            gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

            return gfu

if __name__ == "__main__":

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    mesh.Curve(1)
    fes_order = 3

    u_ex = cos(4*x)*cos(6*y)
    b = CF((x**2,sin(10*y)))
    d = 0.0
    c = 1.0
    mean_u = Integrate(u_ex, mesh)
    # u_ex = u_ex-mean_u

    rhs = d*(-u_ex.Diff(x).Diff(x) - u_ex.Diff(y).Diff(y)) \
        + (u_ex*b[0]).Diff(x) + (b[1]*u_ex).Diff(y) \
        + c*u_ex
    n = specialcf.normal(mesh.dim)
    grad_u_n = InnerProduct(CF((u_ex.Diff(x), u_ex.Diff(y))), n)
    bnd_cond=[['dir', 'right|left|top|bottom', u_ex]]
    solver1 = ADR_DGSolver(mesh, fes_order, b,c,d, bnd_cond, rhs)

    gfu1 = solver1()
    err1 = Integrate((gfu1-u_ex)*(gfu1-u_ex), mesh, order = solver1.fes_order+2)

    Draw(u_ex-gfu1, mesh)

    mesh.Refine()
    gfu2 = solver1()
    err2 = Integrate((gfu2-u_ex)*(gfu2-u_ex), mesh, order = solver1.fes_order+2)

    Draw(u_ex-gfu2, mesh)

    print(err1/err2)
# %%

