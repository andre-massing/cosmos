# %% 
# Importing the necessary libraries

import time as time
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import matplotlib.pyplot as plt
# Output files path
path = "./results/"
import sys
sys.path.append("../")
from cosmos.utils.generate_surface_meshes import generate_half_sphere_mesh

# %% 
# Definition of exact geometry and symbolic operators

R = 1.0
levelset_str = "x*x + y*y + z*z -" + str(R)
dimension = 3 # dimension of the embedding space

phi = CoefficientFunction(eval(levelset_str))

def compute_normal(phi, dim):
    if dim == 2:
        normal = CoefficientFunction((phi.Diff(x), phi.Diff(y)))
    elif dim ==3:
        normal = CoefficientFunction((phi.Diff(x), phi.Diff(y), phi.Diff(z)))
    else:
        raise RuntimeError("Don't know how to compute normal for dim not equal to 2 or 3")
    
    return normal/Norm(normal)

normal = -compute_normal(phi,dimension)

P = Id(normal.dim) - OuterProduct(normal, normal)
Q = Id(normal.dim) - P

def gradient(f,P):

    m, _ = P.dims
    l = len(f.dims)

    if l == 0:

        if m == 2:
            gradient = P*CoefficientFunction((f.Diff(x), f.Diff(y)))
        elif m == 3:
            gradient = P*CoefficientFunction((f.Diff(x), f.Diff(y), f.Diff(z))) 
            
    elif l == 1:

        if m == 2:
            gradient = P*CoefficientFunction((f[0].Diff(x), f[0].Diff(y), \
                f[1].Diff(x), f[1].Diff(y)), dims = (m,m))*P
        elif m == 3:
            gradient = P*CoefficientFunction((f[0].Diff(x), f[0].Diff(y), f[0].Diff(z), \
                f[1].Diff(x), f[1].Diff(y), f[1].Diff(z),\
                    f[2].Diff(x), f[2].Diff(y), f[2].Diff(z)), dims = (m,m))*P
            
    else:

        raise RuntimeError("Don't know how to take the gradient. Only scalars and vectors are accepted.")
    
    return gradient

# %% 
# Creation of manufactured solutions and exact auxiliary funcitons (rhs etc)

# Including time
t = Parameter(0.0)

# Computation of exact manufactured solution and right-had side
t1 = time.time()

# Exact velocity solution
_u = CoefficientFunction((x**3*cos(3*t), sin(y), z))
u_ex = (P*_u).Compile()

# Time Derivative
u_t = (P*u_ex.Diff(t)).Compile()

# Gradient
grad_u = gradient(u_ex,P).Compile()

# Divergence
div_u = Trace(grad_u).Compile()

# Symgrad
surface_strain = Sym(grad_u)
div_sym_grad_1 = Trace(gradient(surface_strain[0,:],P))
div_sym_grad_2 = Trace(gradient(surface_strain[1,:],P))
div_sym_grad_3 = Trace(gradient(surface_strain[2,:],P))

div_sym_grad = (P*CoefficientFunction((div_sym_grad_1, div_sym_grad_2, div_sym_grad_3))).Compile()

# Exact pressure solution
p_ex = CoefficientFunction(sin(x)*cos(5*t))

grad_p = gradient(p_ex,P)

nu = Parameter(1.0)

# Exact rhs to impose manufactured solution
f = -nu*div_sym_grad + u_t + grad_p + grad_u.trans*u_ex
g = div_u

t2 = time.time()

print("Time for symbolic computations vector case: {:.2f}".format(t2-t1))

# %% 
# Definition of the solving algorithm

def NavierStokes(mesh, fes_order, times, f, g, m, n):

    dt = times[1]-times[0] 

    # Spaces definition
    V = VectorH1(mesh, order=int(fes_order), dirichlet_bbnd="bottom")
    Q = H1(mesh, order=int(fes_order-1))
    N = NumberSpace(mesh)
    W = V*Q*Q*N

    (u, p, lam1, lam2), (v, q, mu1, mu2) = W.TnT()

    x_h = GridFunction(W)
    u_h = x_h.components[0]
    p_h = x_h.components[1]

    # Setting up output
    vtkout = VTKOutput(mesh,coefs=[u_h, p_h],names=["u", "p"],filename=path + "unsteady_navierstokes",subdivision=2)
    vtkout_sol = VTKOutput(mesh,coefs=[u_ex, p_ex],names=["u", "p"],filename=path + "unsteady_navierstokes_sol",subdivision=2)

    x0_h, bdf_coef, factor = Initialize(mesh, x_h, times, vtkout, vtkout_sol)

    # Mesh size
    h = specialcf.mesh_size

    # Normal vector
    ns = specialcf.normal(mesh.dim)
    # normal and tangential  projection
    Qs = OuterProduct(ns, ns)
    Ps = Id(mesh.dim) - Qs
    # Weingarten map
    Bs = Grad(ns)

    # Covariant Derivative
    def Cov_h(u):
        return Ps*Grad(u.Trace())*Ps

    # Corrected Covariant Derivative
    def Cov_th(u):
        return Cov_h(u) - InnerProduct(u, ns)*Bs
    
    # Strain tensor
    def E_h(u):
        return Sym(Ps*Grad(u).Trace()*Ps)

    # Corrected strain rate tensor for non-tangential 
    def E_th(u):
        return E_h(u) - InnerProduct(u, ns)*Bs

    A = BilinearForm(W, symmetric = True)

    A += nu*InnerProduct(E_th(u), E_th(v))*ds
    
    # Add b_h contributions 
    A += -InnerProduct(p, Trace(Grad(v).Trace()))*ds
    A += -InnerProduct(q, Trace(Grad(u).Trace()))*ds

    # Lagrangian multiplier penalization of normal component
    A += lam1*InnerProduct(v,ns)*ds
    A += mu1*InnerProduct(u,ns)*ds

    # Add convection for IMEX scheme
    A += InnerProduct(Cov_h(u).trans*u_h.Trace(), Ps*v)*ds

    # Constraint on zero pressure mean
    A += (p*mu2 + q*lam2)*ds

    # Mass matrix for time-stepping
    mass = BilinearForm(W, symmetric = True)
    mass += InnerProduct(Ps*u, Ps*v)*ds
    mass.Assemble()

    # Imposing manufactured solutions
    l = LinearForm(W)
    l += ( InnerProduct(Ps*f, Ps*v) - g*q )* ds 

    res = x_h.vec.CreateVector()
    mstar = mass.mat.CreateMatrix()
    for k, loc_time in enumerate(times[bdf_order:]):

        t.Set(times[k+bdf_order])

        with TaskManager():

            Extrapolate(x_h, x0_h)

            A.Assemble()
            mstar.AsVector().data = factor/dt * mass.mat.AsVector() + A.mat.AsVector()
            invmstar = mstar.Inverse(W.FreeDofs())

            l.Assemble()
            res.data = l.vec
            for j in range(bdf_order):
                res.data -= bdf_coef[bdf_order-j-1]/dt * mass.mat * x0_h[j].data

            u_h.Set(u_ex, definedon=mesh.BBoundaries("bottom"))
            res.data -= mstar*x_h.vec

            x_h.vec.data += invmstar * res.data

        err_u = InnerProduct(u_ex-u_h, u_ex-u_h)
        l2u_error[m,n] = max(l2u_error[m,n], sqrt(Integrate(cf = err_u, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)))
        err_p = InnerProduct(p_ex-p_h, p_ex-p_h)
        l2p_error[m,n] += dt*Integrate(cf = err_p, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)
        err_u_grad = InnerProduct(grad_u-Ps*Grad(u_h)*Ps, grad_u-Ps*Grad(u_h)*Ps)
        h1u_error[m,n] += dt*Integrate(cf = err_u_grad, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)

        vtkout.Do(vb=BND, time = t.Get())
        vtkout_sol.Do(vb=BND, time = t.Get())

        print("Simulation time: {:.3f}".format(t.Get()), end = "\r")

        for k in range(bdf_order-1):
            x0_h[k].data = x0_h[k+1].data
        
        x0_h[bdf_order-1].data = x_h.vec

    l2p_error[m,n] = sqrt(l2p_error[m,n])
    h1u_error[m,n] = sqrt(h1u_error[m,n])

# %%
# Initialization function
def Initialize(mesh, x_h, times, vtkout, vtkout_sol):

    if len(times)<=bdf_order:

        raise RuntimeWarning("Too few steps for the BDF order given (ex. BDF-5 and 3 time-steps)")
    
    if (bdf_order==1):
        bdf_coef=np.array([-1])
        factor = 1
    elif (bdf_order==2):
        bdf_coef=np.array([-2, 0.5])
        factor = 1.5
    elif (bdf_order==3):
        bdf_coef=np.array([-3, 1.5, -1/3])
        factor = 11/6
    elif (bdf_order==4):
        bdf_coef=np.array([-4, 3, -4/3, 1/4])
        factor = 25/12
    elif (bdf_order==5):
        bdf_coef=np.array([-5, 5, -10/3, 5/4, -1/5])
        factor = 137/60
    elif (bdf_order==6):
        bdf_coef=np.array([-6, 15/2, -20/3, 15/4, -6/5, 1/6])
        factor = 147/60
    else:
        raise Exception("Order "+str(bdf_order)+" for BDF methods is not stable")
    
    x0_h = [x_h.vec.CreateVector() for i in range(bdf_order)]
    for k in range(bdf_order):

        t.Set(times[k])
        x_h.components[0].Set(u_ex, definedon=mesh.Boundaries(".*"))

        vtkout.Do(vb=BND, time = t.Get())
        vtkout_sol.Do(vb=BND, time = t.Get())

        x0_h[k].data = x_h.vec.data

        print("Simulation time: {:.3f}".format(t.Get()), end = "\r")

    return x0_h, bdf_coef, factor

# %%
# Extrapolation auxiliary function
def Extrapolate(X_h, X0_h):

    if (bdf_order==1):
        bdf_coef=np.array([1])
    elif (bdf_order==2):
        bdf_coef=np.array([2, -1])
    elif (bdf_order==3):
        bdf_coef=np.array([3, -3, 1])
    elif (bdf_order==4):
        bdf_coef=np.array([4, -6, 4, -1])
    elif (bdf_order==5):
        bdf_coef=np.array([5, -10, 10, -5, 1])
    elif (bdf_order==6):
        bdf_coef=np.array([6, -15, 20, -15, 6, -1])

    X_h.vec[:] = 0
    for k in range(bdf_order):
        X_h.vec.data += bdf_coef[bdf_order-k-1]*X0_h[k].data

# %% 
# Definiton of the convergence parameters and arrays

refinements = 1
powerlaw = 1.5
h_list = [0.25]
Nstep_list = [1/0.125]
for i in range(refinements):
    h_list.append(h_list[-1]/powerlaw)
    Nstep_list.append(Nstep_list[-1]*powerlaw)

fes_order_list = [2, 3]
initial_t = 0
final_t = 1.0
bdf_order = 4

l2u_error = np.zeros((len(fes_order_list), len(h_list)))
l2p_error = np.zeros((len(fes_order_list), len(h_list)))
h1u_error = np.zeros((len(fes_order_list), len(h_list))) 

# %% 
# Performing the convergence study

def convergence(h_list, fes_order_list, Nstep_list):

    for i,k in enumerate(fes_order_list):

        for j,h in enumerate(h_list):

            t1 = time.time()

            mesh = generate_half_sphere_mesh(h, order_g = k+1, r = R)

            times = np.linspace(initial_t, final_t, int(Nstep_list[j]))

            NavierStokes(mesh, k, times, f, g,  i, j)

            t2 = time.time()

            print("Time for FEM solution: {:.2f}".format(t2-t1))


convergence(h_list, fes_order_list, Nstep_list)

# %% 
# #Plotting and printing of the convergence study

ul2_eoc = np.log(l2u_error[:,:-1]/l2u_error[:,1:])/np.log(powerlaw)
print(r"Velocity convergence $L^\infty L^2$")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", ul2_eoc[:,-1])

uh1_eoc = np.log(h1u_error[:,:-1]/h1u_error[:,1:])/np.log(powerlaw)
print(r"Velocity convergence $L^2H^1$")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", uh1_eoc[:,-1])

pl2_eoc = np.log(l2p_error[:,:-1]/l2p_error[:,1:])/np.log(powerlaw)
print(r"Pressure convergence L^2L^2")
print("Space order: ", np.array(fes_order_list)-1)
print("EOC: ", pl2_eoc[:,-1])

import matplotlib.pyplot as plt

fig, axs = plt.subplots(len(fes_order_list),1, figsize = (10,20))

for i,k in enumerate(fes_order_list):

    m1,_ = np.polyfit(np.log(h_list), np.log(l2u_error[i,:]), 1)
    axs[i].loglog(h_list, l2u_error[i,:], '-ro' , label = r"u $L^\infty L^2$ norm - m={:.2f}".format(m1))
    m2,_ = np.polyfit(np.log(h_list), np.log(l2p_error[i,:]), 1)
    axs[i].loglog(h_list, l2p_error[i,:], '-g+' , label = r"p $L^2L^2$ norm - m={:.2f}".format(m2))
    m3,_ = np.polyfit(np.log(h_list), np.log(h1u_error[i,:]), 1)
    axs[i].loglog(h_list, h1u_error[i,:], '-b*' , label = r"u $L^2H^1$ norm - m={:.2f}".format(m3))

    axs[i].legend(loc='lower right')
    axs[i].set_title("Finite element space order {:d}".format(k))
    axs[i].grid('on')

# %%
## Comments/problems/updates

'''
- The code is limited to low Reynolds numbers as it is, no stabilization for the convective term is implemented
- An IMEX scheme is used to deal with the convection, the nonlinear term
- Convergence rates are suboptimal. The L2H1 gradient error seems to have optimal converge rates and equal to the one in L\inftyL2. Pressure error seems to be independent on the BDF order
- Augmenting the fes degree of the lagrange multiplier seems to help raising the convergence rates but it actually makes the code unstable, thus we sticked to the choices made in the paper and it shouldn't be changed
'''
##

# %%
