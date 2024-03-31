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

# Convective term
conv_ex = OuterProduct(u_ex, u_ex)
div_conv_1 = Trace(gradient(conv_ex[0,:],P))
div_conv_2 = Trace(gradient(conv_ex[1,:],P))
div_conv_3 = Trace(gradient(conv_ex[2,:],P))

div_conv = (P*CoefficientFunction((div_conv_1, div_conv_2, div_conv_3))).Compile()

# Exact pressure solution
p_ex = CoefficientFunction(sin(x)*cos(5*t))

grad_p = gradient(p_ex,P)


'''
AAA: The use of HDG should allow me to use way lower Reynolds numbers, but it does not hold
here!
'''
##
nu = Parameter(0.00001)
##

# Exact rhs to impose manufactured solution
f = -nu*div_sym_grad + u_t + grad_p + div_conv
g = div_u

t2 = time.time()

print("Time for symbolic computations vector case: {:.2f}".format(t2-t1))

# %% 
# Definition of the solving algorithm

def NavierStokes(mesh, fes_order, times, f, g, m, n):

    dt = times[1]-times[0] 
    bonus_intorder = 0

    # Spaces definition
    VDivSurf = HDivSurface(mesh, order=int(fes_order), dirichlet_bbnd = "bottom")
    VHat = HCurl(mesh, order=int(fes_order), orderface = 0, dirichlet_bbnd = "bottom")
    Q = SurfaceL2(mesh, order=int(fes_order-1))
    N = NumberSpace(mesh)

    W = VDivSurf*VHat*Q*N

    (u, uhat, p, lam), (v, vhat, q, mu) = W.TnT()
    vhat = vhat.Trace()
    uhat = uhat.Trace()

    x_h = GridFunction(W)
    u_h = x_h.components[0]
    uhat_h = x_h.components[1]
    p_h = x_h.components[2]

    # Setting up output
    vtkout = VTKOutput(mesh,coefs=[u_h, p_h],names=["u", "p"],filename=path + "unsteady_navierstokes",subdivision=2)
    vtkout_sol = VTKOutput(mesh,coefs=[u_ex, p_ex],names=["u", "p"],filename=path + "unsteady_navierstokes_sol",subdivision=2)

    x0_h, bdf_coef, factor = Initialize(mesh, x_h, times, vtkout, vtkout_sol)

    # Mesh size
    h = specialcf.mesh_size

    # Normal, tangential, conormal vectors
    ns = specialcf.normal(mesh.dim)
    ts = specialcf.tangential(mesh.dim)
    nn = Cross(ns,ts)
    # normal and tangential  projection
    Qs = OuterProduct(ns, ns)
    Ps = Id(mesh.dim) - Qs
    # Weingarten map
    Bs = Grad(ns)
    
    # elimination of co-normal component
    def tang(vec):
        return vec - (vec*nn)*nn
    epsu = Sym(Ps*Grad(u)*Ps)
    epsv = Sym(Ps*Grad(v)*Ps)

    A = BilinearForm(W)
    A += nu*InnerProduct(epsu, epsv)*ds
    A += nu* InnerProduct ( epsu * nn,  tang(vhat-v.Trace()) )*ds(element_boundary=True)
    A += nu*InnerProduct ( epsv * nn,  tang(uhat-u.Trace()) )*ds(element_boundary=True)
    A += nu* 4*fes_order*fes_order/h * InnerProduct ( tang(vhat-v.Trace()),  tang(uhat-u.Trace()) )*ds(element_boundary=True)
    A += (-div(u.Trace()) *q.Trace() - div(v.Trace()) *p.Trace() -lam*q-mu*p)*ds

    # Convective term
    A += -InnerProduct(Ps*Grad(v)*Ps*u_h.Trace(), u.Trace())*ds(bonus_intorder = bonus_intorder)
    u_Other = (u.Trace()*nn)*nn + tang(uhat)
    A += IfPos(u_h.Trace() * nn, u_h.Trace()*nn*u.Trace()*v.Trace(), u_h.Trace()*nn*u_Other*v.Trace())*ds(bonus_intorder = bonus_intorder, element_boundary=True)

    #A.Assemble()

    # Mass matrix for time-stepping
    mass = BilinearForm(W, symmetric = True)
    mass += u.Trace()*v.Trace()*ds
    mass.Assemble()

    # Imposing manufactured solutions
    l = LinearForm(W)
    l += ( InnerProduct(Ps*f, v.Trace()) - g*q )* ds 

    res = x_h.vec.CreateVector()
    mstar = mass.mat.CreateMatrix()
    for k, loc_time in enumerate(times[bdf_order:]):

        t.Set(times[k+bdf_order])

        res = l.vec.CreateVector()

        with TaskManager():

            Extrapolate(x_h, x0_h)

            A.Assemble()
            mstar.AsVector().data = factor/dt * mass.mat.AsVector() + A.mat.AsVector()
            invmstar = mstar.Inverse(W.FreeDofs())

            l.Assemble()
            res.data = l.vec
            for j in range(bdf_order):
                res.data -= bdf_coef[bdf_order-j-1]/dt * mass.mat * x0_h[j].data

            x_h.components[1].Set(u_ex, definedon=mesh.BBoundaries("bottom"))
            res.data -= mstar*x_h.vec

            x_h.vec.data += invmstar * res.data

        err_u = InnerProduct(u_ex-u_h, u_ex-u_h)
        l2u_error[m,n] = max(l2u_error[m,n], sqrt(Integrate(cf = err_u, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)))
        err_p = InnerProduct(p_ex-p_h, p_ex-p_h)
        l2p_error[m,n] += dt*Integrate(cf = err_p, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)
        #err_u_grad = InnerProduct(grad_u-Ps*Grad(u_h)*Ps, grad_u-Ps*Grad(u_h)*Ps)
        #h1u_error[m,n] += dt*Integrate(cf = err_u_grad, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)

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
        x_h.components[1].Set(u_ex, definedon=mesh.Boundaries(".*"))

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
- The convective term is expressed in conservative form (see exact solution) contrary to the other implementations
- An IMEX scheme is used to deal with the convection, the nonlinear term
- Exact continuity is traded for exact tangentiality
- Convergence is suboptimal
- The warning on the "orderface" flag must be solved. It could also be Hcurl was just a temporary solution. VectorFacetSurface is maybe a solution?
- Need to find a way to compute the H1 norm of U (see source code from article)
'''
##

# %%
