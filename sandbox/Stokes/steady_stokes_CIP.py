# %% 
# Importing the necessary libraries

import time
# caution: path[0] is reserved for script path (or '' in REPL)
from netgen.occ import *
from netgen import meshing
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw

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

geo          = CSGeometry()
sphere       = Sphere(Pnt(0,0,0), 1)
bot          = Plane(Pnt(0,0,0), Vec(0,0,-1))
finitesphere = sphere * bot

geo.AddSurface(sphere, finitesphere.bc("surface"))
geo.NameEdge(sphere,bot, "bottom")

MESHSURF = meshing.MeshingStep.MESHSURFACE

# %% 
# Creation of manufactured solutions and exact auxiliary funcitons (rhs etc)

# Computation of exact manufactured solution and right-had side
t1 = time.time()

# Exact velocity solution
_u = CoefficientFunction((x**3, sin(y), z))
u_ex = (P*_u).Compile()

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
p_ex = CoefficientFunction(sin(x))

grad_p = gradient(p_ex,P)

nu = Parameter(1.0)

# Exact rhs to impose manufactured solution
f = -nu*div_sym_grad + u_ex + grad_p # + grad_u*u_ex
g = div_u

t2 = time.time()

print("Time for symbolic computations vector case: {:.2f}".format(t2-t1))

# %% 
# Definition of the solving algorithm

def Stokes(mesh, fes_order, f, g, m, n):

    # Spaces definition
    V = VectorH1(mesh, order=int(fes_order), dirichlet_bbnd="bottom")
    Q = H1(mesh, order=int(fes_order))
    N = NumberSpace(mesh)
    W = V*Q*N

    (u, p, lam), (v, q, mu) = W.TnT()

    x_h = GridFunction(W)
    u_h = x_h.components[0]
    p_h = x_h.components[1]

    # Setting up output
    path = "./results/"
    vtkout = VTKOutput(mesh,coefs=[u_h, p_h],names=["u", "p"],filename=path + "steady_navierstokes",subdivision=2)
    vtkout_sol = VTKOutput(mesh,coefs=[u_ex, p_ex],names=["u", "p"],filename=path + "steady_navierstokes_sol",subdivision=2)

    # Mesh size
    h = specialcf.mesh_size

    # Normal vector
    ns = specialcf.normal(mesh.dim)
    ts = specialcf.tangential(mesh.dim)
    nn = Cross(ns,ts)
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

    A = BilinearForm(W)

    A += nu*InnerProduct(E_th(u), E_th(v))*ds
    # Penalization of normal component through bilinear form  
    # beta = Penalty parameter to weakly enforce tangential condition
    beta = 1.0
    A += nu*beta*h**(-2.0)*InnerProduct(ns*u, ns*v)*ds
    
    # Add b_h contributions 
    A += InnerProduct(grad(q).Trace(), u)*ds
    A += InnerProduct(grad(p).Trace(), v)*ds

    jump_dpdn = (grad(p).Trace()-grad(p.Other()).Trace())*nn
    jump_dqdn = (grad(q).Trace()-grad(q.Other()).Trace())*nn

    A += -h**3/nu*InnerProduct(jump_dpdn,jump_dqdn)*ds(element_boundary=True)

    # Constraint on zero presure mean
    A += (p*mu + q*lam)*ds

    # Mass term
    A += InnerProduct(Ps*u, Ps*v)*ds

    A.Assemble()

    # Imposing manufactured solutions
    l = LinearForm(W)
    l += ( InnerProduct(Ps*f, Ps*v) - g*q )* ds 
    l += InnerProduct(q, u_ex*nn)*ds(definedon=mesh.BBoundaries("bottom"))

    res = x_h.vec.CreateVector()

    # matrix for actual system inversion
    with TaskManager():
        l.Assemble()
        res.data = l.vec

        u_h.Set(u_ex, definedon=mesh.BBoundaries("bottom"))
        res.data -= A.mat*x_h.vec

        x_h.vec.data += A.mat.Inverse(freedofs = W.FreeDofs()) * res.data

    err_u = InnerProduct(u_ex-u_h, u_ex-u_h)
    l2u_error[m,n] = sqrt(Integrate(cf = err_u, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
    err_p = InnerProduct(p_ex-p_h, p_ex-p_h)
    l2p_error[m,n] = sqrt(Integrate(cf = err_p, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
    err_u_grad = InnerProduct(grad_u-Ps*Grad(u_h)*Ps, grad_u-Ps*Grad(u_h)*Ps)
    h1u_error[m,n] = sqrt(Integrate(cf = err_u_grad, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))

    vtkout.Do(vb=BND)
    vtkout_sol.Do(vb=BND)

# %% 
# Definiton of the convergence parameters and arrays

import numpy as np

refinements = 3
powerlaw = 1.5
h_list = [0.25]
for i in range(refinements):
    h_list.append(h_list[-1]/powerlaw)
fes_order_list = [1, 2]

l2u_error = np.zeros((len(fes_order_list), len(h_list)))
l2p_error = np.zeros((len(fes_order_list), len(h_list)))
h1u_error = np.zeros((len(fes_order_list), len(h_list))) 

# %% 
# Performing the convergence study

def convergence(h_list, fes_order_list):

    for i,k in enumerate(fes_order_list):

        for j,h in enumerate(h_list):

            t1 = time.time()

            spmesh = Mesh(geo.GenerateMesh(maxh=h))
            spmesh.Curve(k+1);

            Stokes(spmesh, k, f, g,  i, j)

            t2 = time.time()

            print("Time for FEM solution: {:.2f}".format(t2-t1))


convergence(h_list, fes_order_list)

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
    axs[i].loglog(h_list, l2u_error[i,:], '-ro' , label = r"u $L^2$ norm - m={:.2f}".format(m1))
    m2,_ = np.polyfit(np.log(h_list), np.log(l2p_error[i,:]), 1)
    axs[i].loglog(h_list, l2p_error[i,:], '-g+' , label = r"p $L^2$ norm - m={:.2f}".format(m2))
    m3,_ = np.polyfit(np.log(h_list), np.log(h1u_error[i,:]), 1)
    axs[i].loglog(h_list, h1u_error[i,:], '-b*' , label = r"u $H^1$ norm - m={:.2f}".format(m3))

    axs[i].legend(loc='lower right')
    axs[i].set_title("Finite element space order {:d}".format(k))
    axs[i].grid('on')

# %%
## Comments/problems/updates

'''
- The code is limited to low Reynolds numbers as it is, no stabilization for the convective term is implemented
- The convective term is treated completely explicitly (with BDF high order extrapolation) for now, possible improvement could be passing to implicit-explicit handling
- Convective term is for now grad_u*u, but it can be changed to the appropriate one if needed (grad_u.trans*u)
- Convergence rates are optimal
- Spatial errors of L\inftyL^2 of velocity seems to not be influenced by BDF order 
- The use of grad(q)*u for the incompressibility constraint makes the system nonsymmetric (in case an open surface is considered as in this case)
'''

print(uh1_eoc)

##

# %%
