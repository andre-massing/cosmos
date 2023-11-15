# %% 
# Importing the necessary libraries

import time
# caution: path[0] is reserved for script path (or '' in REPL)
from netgen.occ import *
from netgen.meshing import MeshingStep
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import pandas as pd

# %% 
# Definition of exact geometry and symbolic operators
def compute_normal(phi, dim):
    if dim == 2:
        normal = CoefficientFunction((phi.Diff(x), phi.Diff(y)))
    elif dim ==3:
        normal = CoefficientFunction((phi.Diff(x), phi.Diff(y), phi.Diff(z)))
    else:
        raise RuntimeError("Don't know how to compute normal for dim not equal to 2 or 3")
    normal = normal/Norm(normal)
    return normal.Compile()

def tang_project(normal):
    return Id(normal.dim) - OuterProduct(normal, normal)
    
def gradient(f, Ps):
    m, _ = Ps.dims
    l = len(f.dims)
    if l == 0:
        if m == 2:
            gradient = Ps*CoefficientFunction((f.Diff(x), f.Diff(y)))
        elif m == 3:
            gradient = Ps*CoefficientFunction((f.Diff(x), f.Diff(y), f.Diff(z))) 
    elif l == 1:
        if m == 2:
            gradient = Ps*CoefficientFunction(
                (f[0].Diff(x), f[0].Diff(y), 
                 f[1].Diff(x), f[1].Diff(y)), dims = (m,m))*Ps
        elif m == 3:
            gradient = Ps*CoefficientFunction(
                (f[0].Diff(x), f[0].Diff(y), f[0].Diff(z),
                 f[1].Diff(x), f[1].Diff(y), f[1].Diff(z),
                 f[2].Diff(x), f[2].Diff(y), f[2].Diff(z)), dims = (m,m))*Ps
    else:
        raise RuntimeError("Don't know how to take the gradient. Only scalars and vectors are accepted.")
    
    return gradient

def divergence(f, Ps):
    l = len(f.dims)
    # Vector
    if l == 1:
       div_f = Trace(gradient(f, Ps))
    # Square matrix?
    elif l == 2 and f.dims[0] == f.dims[1]:
        div_f = CoefficientFunction(tuple(Trace(gradient(f[:, i] , Ps)) for i in range(f.dims[0])))
    else:
        raise RuntimeError("Don't know how to take the divergence. Only vector and square matrices are accepted.")
    return div_f

# %% Define error computation
def compute_errors(mesh, 
                   u_ex, u_h, order_u,
                   p_ex, p_h, order_p):
    err_u = InnerProduct(u_ex-u_h, u_ex-u_h)
    err_u_grad = InnerProduct(grad_u-Ps*Grad(u_h)*Ps, grad_u-Ps*Grad(u_h)*Ps)
    err_p = InnerProduct(p_ex-p_h, p_ex-p_h)
    l2u_error = sqrt(Integrate(cf = err_u, mesh=mesh, order = order_u+2, VOL_or_BND = BND))
    h1u_error = sqrt(Integrate(cf = err_u_grad, mesh=mesh, order = order_u+2, VOL_or_BND = BND))
    l2p_error = sqrt(Integrate(cf = err_p, mesh=mesh, order = order_p+2, VOL_or_BND = BND))
    return l2u_error, h1u_error, l2p_error

def compute_eoc(errs, fac=2):
    eocs = np.log(np.array(errs)[:-1]/np.array(errs)[1:])/np.log(fac)
    eocs = np.insert(eocs, 0, np.Inf)
    return eocs 

# %% Define Stokes CIP solver
def StokesCIP(mesh, *, order_u, order_p, f, g, filename="results/stokes_cip_sol"):
    # Function spaces
    V = VectorH1(mesh, order=int(order_u))
    Q = H1(mesh, order=int(order_p))
    N = NumberSpace(mesh)
    
    # TODO: Only needed if no boundary is included
    W = V*Q*N
    (u, p, lam), (v, q, mu) = W.TnT()

    x_h = GridFunction(W)
    u_h = x_h.components[0]
    p_h = x_h.components[1]

    # Setting up output
    vtkout = VTKOutput(mesh,
                       coefs=[u_h, p_h],
                       names=["u_h", "p_h"],
                       filename=filename, subdivision=2)

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
    # def E_th(u):
        # return E_h(Ps*u)
    
    A = BilinearForm(W)

    A += nu*InnerProduct(E_th(u), E_th(v))*ds
    # Mass term
    A += InnerProduct(Ps*u, Ps*v)*ds
    # Add b_h contributions 
    A += InnerProduct(grad(q).Trace(), u)*ds
    A += InnerProduct(grad(p).Trace(), v)*ds
    
    # TODO: Only needed if no boundary is included
    # Constraint to enforce zero presure mean for boundary less surfaces
    A += (p*mu + q*lam)*ds
    
    # Penalization of normal component through bilinear form  
    # beta = Penalty parameter to weakly enforce tangential condition
    beta = 1.0
    # A += nu*beta*h**(-2.0)*InnerProduct(ns*u, ns*v)*ds
    A += nu*beta/(h*h)*InnerProduct(ns*u, ns*v)*ds
    # A += nu*beta/h*InnerProduct(ns*u, ns*v)*ds

    # Pressure stabilization
    # jump_dpdn = (grad(p).Trace()-grad(p.Other()).Trace())*nn
    # jump_dqdn = (grad(q).Trace()-grad(q.Other()).Trace())*nn
    # A += -h**3/nu*InnerProduct(jump_dpdn,jump_dqdn)*ds(element_boundary=True)
    A.Assemble()

    # Imposing manufactured solutions
    l = LinearForm(W)
    l += ( InnerProduct(Ps*f, Ps*v) - g*q)* ds 
    # l += InnerProduct(q, u_ex*nn)*ds(definedon=mesh.BBoundaries("bottom"))

    res = x_h.vec.CreateVector()

    # matrix for actual system inversion
    with TaskManager():
        l.Assemble()
        res.data = l.vec

        # u_h.Set(u_ex, definedon=mesh.BBoundaries("bottom"))
        res.data -= A.mat*x_h.vec

        x_h.vec.data += A.mat.Inverse(freedofs = W.FreeDofs()) * res.data
        
    vtkout.Do(vb=BND)
    
    return u_h, p_h

       
# %% Define manufactured solution
dimension = 3 # dimension of the embedding space
R = 1.0
levelset_str = "x*x + y*y + z*z -" + str(R)

phi = CoefficientFunction(eval(levelset_str))
normal = compute_normal(phi,dimension)
# Tangential projection
Ps = tang_project(normal)

# Define manufactured solutions and corresponding PDE data 
t1 = time.time()

# Manufactured solution
_u = CoefficientFunction((x**3, sin(y), z))
u_ex = Ps*_u
p_ex = CoefficientFunction(sin(x))

nu = Parameter(1.0)
# Right-hand side
f = -nu*Ps*divergence(Sym(gradient(u_ex, Ps)),Ps) + u_ex + gradient(p_ex, Ps)
f = f.Compile()
g = divergence(u_ex, Ps)
g = g.Compile()

# Gradient of u (for H1 error computation)
grad_u = gradient(u_ex, Ps)

t2 = time.time()
print("Time for symbolic computations vector case: {:.2f}".format(t2-t1))

# Define geometry and mesh
geo          = CSGeometry()
sphere       = Sphere(Pnt(0,0,0), 1)
bot          = Plane(Pnt(0,0,0), Vec(0,0,-1))
# finitesphere = sphere * bot

geo.Add(sphere)
# geo.AddSurface(sphere, finitesphere.bc("surface"))
# geo.AddSurface(sphere, finitesphere.bc("surface"))
# geo.NameEdge(sphere,bot, "bottom")

# %% Run convergence study
fes_order_list = [(2,1)]
num_refs = 3
maxh0 = 0.2

# make directory if it does not exist
results_dir = "results/"
import os
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

error_list = [[], [], []]
for fes_order in fes_order_list:
    maxh = maxh0
    for ref in range(num_refs+1):
        
            mesh = Mesh(geo.GenerateMesh(maxh=maxh, perfstepsend=MeshingStep.MESHSURFACE))
            order_u, order_p = fes_order
            # order_g = order_u
            order_g = order_u+1
            file_name = os.path.join(results_dir, f"stokes_cip_sol_fesorders_{order_u}_{order_p}_ref_{ref}")
            mesh.Curve(order_g)
            u_h, p_h = StokesCIP(mesh, 
                                 order_u=order_u, order_p=order_p, 
                                 f=f, g=g, 
                                 filename=file_name)
            # Compute errors
            errors = compute_errors(mesh, u_ex, u_h, order_u, p_ex, p_h, order_p)
            # Append errors to error list
            for i, error in enumerate(errors):
                error_list[i].append(error)
            # Compute current eoc tables
            eoc_list = [compute_eoc(errors) for errors in error_list]
            # error_eoc_list = zip(error_list, eoc_list)
            table = pd.DataFrame({'L2_error_u': error_list[0],
                                  'L2_eoc_u'  : eoc_list[0],
                                  'H1_error_u': error_list[1],
                                  'H1_eoc_u'  : eoc_list[1],
                                  'L2_error_p': error_list[2],
                                  'L2_eoc_p'  : eoc_list[2]})
            display(table)
            # print(table.to_string())
            maxh /= 2
            # if ref < num_refs:
            #     mesh.Curve(1)
            #     mesh.Refine(mark_surface_elements=True)
            #     mesh.Curve(order_g)
    
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(len(fes_order_list),1, figsize = (10,20))

# for i,k in enumerate(fes_order_list):
#     m1,_ = np.polyfit(np.log(h_list), np.log(l2u_error[i,:]), 1)
#     axs[i].loglog(h_list, l2u_error[i,:], '-ro' , label = r"u $L^2$ norm - m={:.2f}".format(m1))
#     m2,_ = np.polyfit(np.log(h_list), np.log(l2p_error[i,:]), 1)
#     axs[i].loglog(h_list, l2p_error[i,:], '-g+' , label = r"p $L^2$ norm - m={:.2f}".format(m2))
#     m3,_ = np.polyfit(np.log(h_list), np.log(h1u_error[i,:]), 1)
#     axs[i].loglog(h_list, h1u_error[i,:], '-b*' , label = r"u $H^1$ norm - m={:.2f}".format(m3))

#     axs[i].legend(loc='lower right')
#     axs[i].set_title("Finite element space order {:d}".format(k))
#     axs[i].grid('on')