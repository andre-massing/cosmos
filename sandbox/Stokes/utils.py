# %% 
# Importing the necessary libraries
from ngsolve import *
import numpy as np

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
    
    # Compute gradient of u (for H1 error computation)
    mesh = u_h.space.mesh
    ns = specialcf.normal(mesh.dim)
    Qs = OuterProduct(ns, ns)
    Ps = Id(mesh.dim) - Qs
    grad_u = gradient(u_ex, Ps)
    
    # Compute errors
    err_u = InnerProduct(u_ex-u_h, u_ex-u_h)
    # TODO: Next line does not work universally for all velocity spaces
    #       This works now for Hdiv spaces 
    err_u_grad = InnerProduct(grad_u-Ps*u_h.Operator("grad", BND)*Ps, 
                              grad_u-Ps*u_h.Operator("grad", BND)*Ps)
    err_p = InnerProduct(p_ex-p_h, p_ex-p_h)
    l2u_error = sqrt(Integrate(cf = err_u, mesh=mesh, order = order_u+2, VOL_or_BND = BND))
    h1u_error = sqrt(Integrate(cf = err_u_grad, mesh=mesh, order = order_u+2, VOL_or_BND = BND))
    l2p_error = sqrt(Integrate(cf = err_p, mesh=mesh, order = order_p+2, VOL_or_BND = BND))
    return l2u_error, h1u_error, l2p_error

def compute_eoc(errs, fac=2):
    eocs = np.log(np.array(errs)[:-1]/np.array(errs)[1:])/np.log(fac)
    eocs = np.insert(eocs, 0, np.Inf)
    return eocs 

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