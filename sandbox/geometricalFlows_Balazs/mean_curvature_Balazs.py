# %%
## Importing the necessary libraries

import time as time
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import matplotlib.pyplot as plt
# Output files path
path = "../results/"
import sys
sys.path.append("../")
from cosmos.utils.generate_surface_meshes import generate_sphere_mesh

# %%
# Definition of exact geometry and related differential operators
# This follow the notation present in Balazs

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

normal = compute_normal(phi,dimension)

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
# Definiton of exact solution
# No rexact rhs is needed since the solution is just the natural evolution

t = Parameter(0.0)

n_ex = CF((x, y, z))/Norm(CF((x, y, z))).Compile()
X_ex = (n_ex*sqrt(R**2-4*t)).Compile()
displ_ex = (X_ex - CF((x, y, z))).Compile()
H_ex = (2.0/sqrt(R**2-4*t)).Compile()
gradH = gradient(H_ex, Id(dimension)-OuterProduct(n_ex,n_ex)).Compile()
gradn = gradient(n_ex, Id(dimension)).Compile()
v_ex = X_ex.Diff(t)

# %%
# Solving algorithms

def meanCurvFLow(mesh, fes_order, times, m_index, n_index):

    dt = times[1]-times[0]

    V = VectorH1(mesh, order=int(fes_order))
    Q = H1(mesh,order=int(fes_order))
    U = V*Q
    (n,H), (n_t,H_t) = U.TnT()
    (v, v_t) = V.TnT()


    # Grid funciton for displacement
    displ = GridFunction(V)
    # Grid function with initial configuration
    Identity = GridFunction(V)
    Identity.Set(CF((x, y, z)), definedon=mesh.Boundaries(".*"))

    U_h = GridFunction(U)
    n_h = U_h.components[0]
    H_h = U_h.components[1]
    X_h = GridFunction(V)

    vtkout = VTKOutput(mesh,coefs=[U_h.components[0], U_h.components[1], displ],names=["n", "H", "displ"],filename=path + "mean_curvature",subdivision=2)
    vtkout_sol = VTKOutput(mesh,coefs=[n_ex, H_ex, displ_ex],names=["n", "H", "displ"],filename=path + "mean_curvature_sol",subdivision=2)

    X0_h, U0_h, bdf_coef, factor = Initialize(mesh, X_h, U_h, times, vtkout)

    K_of_v =  BilinearForm(V)
    K_of_v += ( v*v_t + InnerProduct(grad(v).Trace().trans,grad(v_t).Trace().trans) ) * ds(deformation = displ)
    K_of_v.Assemble()
    inv_v = K_of_v.mat.Inverse(V.FreeDofs())

    K_of_U = BilinearForm(U)
    K_of_U += (factor/dt*(n*n_t+H*H_t) + InnerProduct(grad(n).Trace().trans,grad(n_t).Trace().trans) + InnerProduct(grad(H).Trace(),grad(H_t).Trace())) * ds(deformation=displ)
    K_of_U.Assemble()
    inv_U = K_of_U.mat.Inverse(U.FreeDofs())

    M_of_U = BilinearForm(U)
    M_of_U +=  (n*n_t+H*H_t) * ds(deformation=displ)
    M_of_U.Assemble()

    rhs_U = BilinearForm(U)
    alpha = InnerProduct(grad(n).Trace().trans, grad(n).Trace().trans)
    rhs_U += alpha*(n*n_t + H*H_t)*ds(deformation=displ)
    rhs_U.Assemble()

    rhs_v = LinearForm(V)
    rhs_v += (-H_h*n_h*v_t-InnerProduct(H_h*grad(n_h).Trace().trans+OuterProduct(grad(H_h).Trace(), n_h.Trace()), grad(v_t).Trace()))*ds(deformation=displ)
    rhs_v.Assemble()

    l = LinearForm(U)

    res = U_h.vec.CreateVector()
    aux = l.vec.CreateVector()
    for k, loc_time in enumerate(times[bdf_order:]):

        t.Set(times[k+bdf_order])

        with TaskManager():

            Extrapolate(U_h, U0_h)
            Extrapolate(X_h, X0_h)
            displ.vec.data = X_h.vec - Identity.vec

            K_of_U.Assemble()
            M_of_U.Assemble()
            inv_U.Update()
            rhs_U.Assemble()
            rhs_U.Apply(U_h.vec, aux)
            res.data = aux
            for j in range(bdf_order):
                res.data -= bdf_coef[bdf_order-j-1]/dt * M_of_U.mat * U0_h[j].data

            U_h.vec.data = inv_U * res.data

            K_of_v.Assemble()
            inv_v.Update()
            rhs_v.Assemble()
            X_h.vec.data = inv_v*rhs_v.vec

            for j in range(bdf_order):
                X_h.vec.data -= bdf_coef[bdf_order-j-1]/dt * X0_h[j].data
            X_h.vec.data = dt/factor*X_h.vec.data

        err_X = InnerProduct(X_h-X_ex, X_h-X_ex)
        l2X_error[m_index,n_index] = max(l2X_error[m_index, n_index], sqrt(Integrate(cf = err_X, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)))
        err_n = InnerProduct(U_h.components[0]-n_ex, U_h.components[0]-n_ex)
        l2n_error[m_index,n_index] = max(l2n_error[m_index,n_index], sqrt(Integrate(cf = err_n, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)))
        err_H = InnerProduct(U_h.components[1]-H_ex, U_h.components[1]-H_ex)
        l2H_error[m_index,n_index] = max(l2H_error[m_index,n_index], sqrt(Integrate(cf = err_H, mesh=mesh, order = fes_order+2, VOL_or_BND = BND)))


        vtkout.Do(vb=BND, time = t.Get())
        vtkout_sol.Do(vb=BND, time = t.Get())


        print("Simulation time: {:.3f}".format(t.Get()), end = "\r")

        for k in range(bdf_order-1):
            X0_h[k].data = X0_h[k+1].data
            U0_h[k].data = U0_h[k+1].data
        
        X0_h[bdf_order-1].data = X_h.vec.data
        U0_h[bdf_order-1].data = U_h.vec.data

def Initialize(mesh, X_h, U_h, times, vtkout):

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
        raise Exception("Order "+str(bdf_order)+" for BDF fes_orderethods is not stable")
    
    X0_h = [X_h.vec.CreateVector() for i in range(bdf_order)]
    U0_h = [U_h.vec.CreateVector() for i in range(bdf_order)]
    for k in range(bdf_order):

        t.Set(times[k])
        U_h.components[0].Set(n_ex, definedon=mesh.Boundaries(".*"))
        U_h.components[1].Set(H_ex, definedon=mesh.Boundaries(".*"))
        X_h.Set(X_ex, definedon=mesh.Boundaries(".*"))

        vtkout.Do(vb=BND, time = t.Get())

        U0_h[k].data = U_h.vec.data
        X0_h[k].data = X_h.vec.data

        print("Simulation time: {:.3f}".format(t.Get()), end = "\r")

    return X0_h, U0_h, bdf_coef, factor

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

    X_h.vec[:]=0
    for k in range(bdf_order):
        X_h.vec.data += bdf_coef[bdf_order-k-1]*X0_h[k].data
    
# %%
# Definiton of convergence study parameters

refinements = 3
powerlaw = 1.5
h_list = [0.25]
Nstep_list = [1/0.0625]
for i in range(refinements):
    h_list.append(h_list[-1]/powerlaw)
    Nstep_list.append(Nstep_list[-1]*powerlaw)

fes_order_list = [1, 2, 3]
initial_t = 0
final_t = 0.1875
bdf_order = 4

l2X_error = np.zeros((len(fes_order_list), len(h_list)))
l2n_error = np.zeros((len(fes_order_list), len(h_list)))
l2H_error = np.zeros((len(fes_order_list), len(h_list)))

# %%
# Performing the converge study

def convergence(h_list, fes_order_list, Nstep_list):

    for i,k in enumerate(fes_order_list):

        for j,h in enumerate(h_list):

            t1 = time.time()

            mesh = generate_sphere_mesh(h, order_g = k+1, r = R)

            times = np.linspace(initial_t, final_t, int(Nstep_list[j]))

            meanCurvFLow(mesh, k, times, i, j)

            t2 = time.time()

            print("Time for FEM solution: {:.2f}".format(t2-t1))

convergence(h_list, fes_order_list, Nstep_list)

# %%
# Plotting and printing of the convergence study

X_eoc = np.log(l2X_error[:,:-1]/l2X_error[:,1:])/np.log(powerlaw)
print(r"Position convergence $L^\inftyL^2$")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", X_eoc[:,-1])

n_eoc = np.log(l2n_error[:,:-1]/l2n_error[:,1:])/np.log(powerlaw)
print(r"Normal convergence $L^\inftyL^2$")
print("Space order: ", np.array(fes_order_list)-1)
print("EOC: ", n_eoc[:,-1])

H_eoc = np.log(l2H_error[:,:-1]/l2H_error[:,1:])/np.log(powerlaw)
print(r"Mean curv. convergence $L^\inftyL^2$")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", H_eoc[:,-1])

fig, axs = plt.subplots(len(fes_order_list),1, figsize = (10,20))

for i,k in enumerate(fes_order_list):

    m1,_ = np.polyfit(np.log(h_list), np.log(l2X_error[i,:]), 1)
    axs[i].loglog(h_list, l2X_error[i,:], '-ro' , label = r"X $L^\infty L^2$ tangent norm - m={:.2f}".format(m1))
    m2,_ = np.polyfit(np.log(h_list), np.log(l2n_error[i,:]), 1)
    axs[i].loglog(h_list, l2n_error[i,:], '-g+' , label = r"n $L^\infty L^2$ tangent norm - m={:.2f}".format(m2))
    m3,_ = np.polyfit(np.log(h_list), np.log(l2H_error[i,:]), 1)
    axs[i].loglog(h_list, l2H_error[i,:], '-bx' , label = r"H $L^\infty L^2$ tangent norm - m={:.2f}".format(m3))

    axs[i].legend(loc='lower right')
    axs[i].set_title("Finite element space order {:d}".format(k))
    axs[i].grid('on')

# %%
## Comments/problems/updates

'''

PROBLEMS:

COMMENTS:

'''
##
# %%
