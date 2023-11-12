# %%
## Importing the necessary libraries

import time as time
import matplotlib.pyplot as plt
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
path = "./results/"
import sys
sys.path.append("../")
from generate_surface_meshes import generate_torus_mesh


# %%
# Definition of exact geometry and related differential operators
# This follow the notation present in Balazs

r = 1.0
R = r*sqrt(2)
levelset_str = "(sqrt(x*x+y*y) - " + str(R) + ")**2 + z*z -" + str(r**2)
dimension = 3 # dimension of the embedding space

dimension = 3 # dimension of the embedding space

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

# %%
# Definiton of exact solution
# For this converge case, the Clifford torus should be stationary

with TaskManager():
    t = Parameter(0.0)

    phi = CoefficientFunction(eval(levelset_str))

    n_ex = gradient(phi,Id(3))
    n_ex = (n_ex/Norm(n_ex)).Compile()

    V_ex = CF(0.0)
    P_ex = (Id(3)-OuterProduct(n_ex,n_ex)).Compile()

    u = atan2(y, x)
    v = atan2(z, sqrt(x**2+y**2)-R)

    X_ex = CF(((R+r*cos(v))*cos(u), (R+r*cos(v))*sin(u), r*sin(v))).Compile()

    grad_n = gradient(n_ex,P_ex).Compile()

    H_ex = Trace(grad_n).Compile()

    z_ex = gradient(H_ex, P_ex).Compile()

    displ_ex = CF((0,0,0))

# %%
# Solving algorithms

def WillmoreFLow(mesh, fes_order, times, m_index, n_index):

    dt = times[1]-times[0]
    Q = H1(mesh,order=int(fes_order))
    V = VectorH1(mesh, order=int(fes_order))
    W = Q*V*Q*V
    (H, n, Vnorm, gradH),(H_t, n_t, Vnorm_t, gradH_t) = W.TnT()

    # Grid funciton for displacement
    X_h = GridFunction(V)
    displ = GridFunction(V)
    # Grid funciton for displacement
    Identity = GridFunction(V)
    Identity.Set(X_ex, definedon=mesh.Boundaries(".*"))

    W_h = GridFunction(W)
    H_h = W_h.components[0]
    n_h = W_h.components[1]
    Vnorm_h = W_h.components[2]
    gradH_h = W_h.components[3]

    err_array = np.zeros((6, len(times)))

    vtkout = VTKOutput(mesh,coefs=[H_h, n_h, Vnorm_h, gradH_h, displ],names=["H", "n", "V", "z", "displ"],filename=path + "WillmoreFlow",subdivision=2)
    vtkout_sol = VTKOutput(mesh,coefs=[H_ex, n_ex, V_ex, z_ex, displ_ex],names=["H", "n", "V", "z", "displ"],filename=path + "WillmoreFlow_sol",subdivision=2)

    X0_h, W0_h, bdf_coef, factor = Initialize(mesh, X_h, W_h, displ, times, vtkout, vtkout_sol, fes_order, err_array)

    #ns = specialcf.normal(mesh.dim)
    ns = n_h
    Ps = Id(mesh.dim) - OuterProduct(ns,ns)

    K = BilinearForm(W)
    # Mass part
    K += (factor/dt*H*H_t)*ds
    K += (factor/dt*n*n_t)*ds
    K += (Vnorm*Vnorm_t)*ds
    K += (gradH*gradH_t)*ds

    # Mixed part
    K += (-InnerProduct(grad(Vnorm).Trace(), grad(H_t).Trace()))*ds
    K += (-InnerProduct(grad(gradH).Trace().trans, Grad(n_t).Trace().trans))*ds
    K += (InnerProduct(grad(H).Trace(), grad(Vnorm_t).Trace()))*ds
    K += (InnerProduct(grad(n).Trace().trans, grad(gradH_t).Trace().trans))*ds

    K.Assemble()
    invK = K.mat.Inverse(freedofs = W.FreeDofs())

    M = BilinearForm(W)
    M += (H*H_t)*ds
    M += (n*n_t)*ds

    rhs = LinearForm(W)
    A_h = Sym(Ps*grad(ns))
    A_h_2n = Trace(A_h*A_h.trans)
    Q_h = -0.5*H_h*H_h*H_h+A_h_2n*H_h
    rhs += (InnerProduct(grad(H_h).Trace(), grad(H_h).Trace())*ns+A_h*A_h*grad(H_h).Trace())*n_t*ds
    rhs += 2*(InnerProduct(A_h*grad(H_h).Trace(), Grad(n_t).Trace().trans*ns))*ds
    rhs += (Q_h*Trace(Grad(n_t).Trace().trans))*ds
    rhs += -InnerProduct(Q_h*H_h*ns, n_t)*ds

    rhs += Q_h*Vnorm_t*ds
    rhs += A_h_2n*InnerProduct(ns, gradH_t)*ds

    F = BilinearForm(W)
    F += (-A_h_2n*Vnorm*H_t)*ds
    F += (H_h*A_h-A_h*A_h)*gradH*n_t*ds

    res = W_h.vec.CreateVector()

    for k, loc_time in enumerate(times[bdf_order:]):

        t.Set(times[k+bdf_order])

        Extrapolate(X_h, X0_h)
        Extrapolate(W_h, W0_h)
        displ.vec.data = X_h.vec - Identity.vec

        mesh.SetDeformation(displ)

        with TaskManager():

            K.Assemble()
            invK.Update()
            M.Assemble()
            rhs.Assemble()
            F.Assemble()
            res.data = rhs.vec
            for j in range(bdf_order):
                res.data -= bdf_coef[bdf_order-j-1]/dt*M.mat*W0_h[j].data
            res.data += F.mat*W_h.vec

            W_h.vec.data = invK*res

            #n_h.Set(n_ex, definedon=mesh.Boundaries(".*"))
            #n_h = MakeUnitary(n_h, mesh, fes_order)

            #gradH_h.Set(z_ex, definedon=mesh.Boundaries(".*"))
            #gradH_h = MakeTangent(gradH_h, ns, mesh, fes_order)

            X_h.Set(Vnorm_h*n_h*dt/factor, definedon=mesh.Boundaries(".*"), dual = True)
            for j in range(bdf_order):
                X_h.vec.data -= bdf_coef[bdf_order-j-1]/factor*IdentityMatrix(V.ndof, complex=False)*X0_h[j].data
        
        err_X = InnerProduct(X_h-X_ex, X_h-X_ex)
        err_array[0,k+bdf_order] = sqrt(Integrate(cf = err_X, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_n = InnerProduct(n_h-n_ex,n_h-n_ex)
        err_array[1,k+bdf_order] = sqrt(Integrate(cf = err_n, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_H = InnerProduct(H_h-H_ex,H_h-H_ex)
        err_array[2,k+bdf_order] = sqrt(Integrate(cf = err_H, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_V = InnerProduct(Vnorm_h-V_ex, Vnorm_h-V_ex)
        err_array[3,k+bdf_order] = sqrt(Integrate(cf = err_V, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_z = InnerProduct(gradH_h-z_ex,gradH_h-z_ex)
        err_array[4,k+bdf_order] = sqrt(Integrate(cf = err_z, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_array[5,k+bdf_order] = Integrate(cf = InnerProduct(H_h, H_h), mesh=mesh, order = fes_order+2, VOL_or_BND = BND)/2.0

        l2X_error[m_index,n_index] = max(l2X_error[m_index, n_index], err_array[0,k])
        l2n_error[m_index,n_index] = max(l2n_error[m_index,n_index], err_array[1,k])
        l2H_error[m_index,n_index] = max(l2H_error[m_index,n_index], err_array[2,k])
        l2V_error[m_index,n_index] = max(l2V_error[m_index,n_index], err_array[3,k])
        l2z_error[m_index,n_index] = max(l2z_error[m_index,n_index], err_array[4,k])

        mesh.UnsetDeformation()

        #vtkout.Do(vb=BND, time = t.Get())
        #vtkout_sol.Do(vb=BND, time = t.Get())

        print("Simulation time: {:.3f}".format(t.Get()), end = "\r")

        for k in range(bdf_order-1):
            X0_h[k].data = X0_h[k+1].data
            W0_h[k].data = W0_h[k+1].data
        
        X0_h[bdf_order-1].data = X_h.vec.data
        W0_h[bdf_order-1].data = W_h.vec.data

    return err_array

def Initialize(mesh, X_h, W_h, displ, times, vtkout, vtkout_sol, fes_order, err_array):

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
        raise Exception("Order "+str(bdf_order)+" for BDF fes_order is not stable")

    X0_h = [X_h.vec.CreateVector() for i in range(bdf_order)] 
    W0_h = [W_h.vec.CreateVector() for i in range(bdf_order)]
    for k in range(bdf_order):

        t.Set(times[k])
        displ.Set(displ_ex, definedon=mesh.Boundaries(".*"))

        mesh.SetDeformation(displ)

        X_h.Set(X_ex, definedon=mesh.Boundaries(".*"))
        W_h.components[0].Set(H_ex, definedon=mesh.Boundaries(".*"))
        W_h.components[1].Set(n_ex, definedon=mesh.Boundaries(".*"))
        W_h.components[2].Set(V_ex, definedon=mesh.Boundaries(".*"))
        W_h.components[3].Set(z_ex, definedon=mesh.Boundaries(".*"))
        
        X0_h[k].data = X_h.vec
        W0_h[k].data = W_h.vec

        err_X = InnerProduct(X_h-X_ex, X_h-X_ex)
        err_array[0,k] = sqrt(Integrate(cf = err_X, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_n = InnerProduct(W_h.components[1]-n_ex, W_h.components[1]-n_ex)
        err_array[1,k] = sqrt(Integrate(cf = err_n, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_H = InnerProduct(W_h.components[0]-H_ex, W_h.components[0]-H_ex)
        err_array[2,k] = sqrt(Integrate(cf = err_H, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_V = InnerProduct(W_h.components[2]-V_ex, W_h.components[2]-V_ex)
        err_array[3,k] = sqrt(Integrate(cf = err_V, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_z = InnerProduct(W_h.components[3]-z_ex, W_h.components[3]-z_ex)
        err_array[4,k] = sqrt(Integrate(cf = err_z, mesh=mesh, order = fes_order+2, VOL_or_BND = BND))
        err_array[5,k] = Integrate(cf = InnerProduct(W_h.components[0], W_h.components[0]), mesh=mesh, order = fes_order+2, VOL_or_BND = BND)/2.0

        mesh.UnsetDeformation()

        #vtkout.Do(vb=BND, time = t.Get())
        #vtkout_sol.Do(vb=BND, time = t.Get())

        print("Simulation time: {:.3f}".format(t.Get()), end = "\r")

    return X0_h, W0_h, bdf_coef, factor

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
# Auxiliary functions for improved performance

def MakeUnitary(n_h, mesh, fes_order):

    V = VectorH1(mesh, order = fes_order)
    u,v = V.TnT()
    gfu = GridFunction(V)

    a = BilinearForm(V)
    a += Norm(n_h)*u*v*ds
    a.Assemble()
    inv = a.mat.Inverse()

    f = LinearForm(V)
    f += n_h*v*ds
    f.Assemble()

    gfu.vec.data = inv*f.vec

    return gfu

def MakeTangent(gradH_h, n_h, mesh, fes_order):

    V = VectorH1(mesh, order = fes_order)
    u,v = V.TnT()
    gfu = GridFunction(V)

    P_h = Id(mesh.dim) - OuterProduct(n_h, n_h)

    a = BilinearForm(V)
    a += u*v*ds
    a.Assemble()
    inv = a.mat.Inverse()

    f = LinearForm(V)
    f += (P_h*gradH_h)*v*ds
    f.Assemble()

    gfu.vec.data = inv*f.vec

    return gfu

def SetVelocity(Vnorm_h, n_h, mesh, fes_order, dt, factor):

    V = VectorH1(mesh, order = fes_order)
    u,v = V.TnT()
    gfu = GridFunction(V)

    a = BilinearForm(V)
    a += u*v*ds
    a.Assemble()
    inv = a.mat.Inverse()

    f = LinearForm(V)
    f += (Vnorm_h*n_h*dt/factor)*v*ds
    f.Assemble()

    gfu.vec.data = inv*f.vec

    return gfu

# %%
# Definiton of convergence study parameters

refinements = 3
powerlaw = 1.5
h_list = [0.25]
Nstep_list = [1/0.0625]
for i in range(refinements):
    h_list.append(h_list[-1]/powerlaw)
    Nstep_list.append(Nstep_list[-1]*powerlaw)
fes_order_list = [2, 3]
initial_t = 0
final_t = 1.0
bdf_order = 2

l2X_error = np.zeros((len(fes_order_list), len(h_list)))
l2n_error = np.zeros((len(fes_order_list), len(h_list)))
l2H_error = np.zeros((len(fes_order_list), len(h_list)))
l2V_error = np.zeros((len(fes_order_list), len(h_list)))
l2z_error = np.zeros((len(fes_order_list), len(h_list)))

# %%
# Performing the converge study

def convergence(h_list, fes_order_list, Nstep_list):

    write = True

    for i,k in enumerate(fes_order_list):

        for j,h in enumerate(h_list):

            times = np.linspace(initial_t, final_t, int(Nstep_list[j]))

            mesh = generate_torus_mesh(maxh = h, order_g = k+2, R=R, r=r)

            t1 = time.time()

            err_array = WillmoreFLow(mesh, k, times, i, j)

            plt.semilogy(times, err_array[0,:], label = 'X')
            plt.semilogy(times, err_array[1,:], label = 'n')
            plt.semilogy(times, err_array[2,:], label = 'H')
            plt.semilogy(times, err_array[3,:], label = 'V')
            plt.semilogy(times, err_array[4,:], label = 'z')

            plt.legend()
            plt.show()

            plt.plot(times, err_array[5,:], label = 'Computed')
            plt.plot(times, np.ones(len(err_array[5,:]))*4*np.pi**2, label = 'Exact')
            plt.legend()
            plt.show()

            t2 = time.time()

            print("\n Time for FEM solution with h={:.2e}, dt={:.2e}: {:.2f}".format(h, times[1]-times[0], t2-t1))


convergence(h_list, fes_order_list, Nstep_list)

# %%
# Plotting and printing of the convergence study

X_eoc = np.log(l2X_error[:,:-1]/l2X_error[:,1:])/np.log(powerlaw)
print("Position convergence")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", X_eoc[:,-1])

n_eoc = np.log(l2n_error[:,:-1]/l2n_error[:,1:])/np.log(powerlaw)
print("normal convergence")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", n_eoc[:,-1])

H_eoc = np.log(l2H_error[:,:-1]/l2H_error[:,1:])/np.log(powerlaw)
print("Mean curv. convergence")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", H_eoc[:,-1])

V_eoc = np.log(l2V_error[:,:-1]/l2V_error[:,1:])/np.log(powerlaw)
print("Concentration convergence")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", V_eoc[:,-1])

z_eoc = np.log(l2z_error[:,:-1]/l2z_error[:,1:])/np.log(powerlaw)
print("Concentration convergence")
print("Space order: ", np.array(fes_order_list))
print("EOC: ", z_eoc[:,-1])

import matplotlib.pyplot as plt

fig, axs = plt.subplots(len(fes_order_list),1, figsize = (10,20))

for i,k in enumerate(fes_order_list):

    m1,_ = np.polyfit(np.log(h_list), np.log(l2X_error[i,:]), 1)
    axs[i].loglog(h_list, l2X_error[i,:], '-ro' , label = "X L^2 norm - m={:.2f}".format(m1))
    m2,_ = np.polyfit(np.log(h_list), np.log(l2n_error[i,:]), 1)
    axs[i].loglog(h_list, l2n_error[i,:], '-g+' , label = "n L^2 norm - m={:.2f}".format(m2))
    m3,_ = np.polyfit(np.log(h_list), np.log(l2H_error[i,:]), 1)
    axs[i].loglog(h_list, l2H_error[i,:], '-bx' , label = "H L^2 norm - m={:.2f}".format(m3))
    m4,_ = np.polyfit(np.log(h_list), np.log(l2V_error[i,:]), 1)
    axs[i].loglog(h_list, l2V_error[i,:], '-k.' , label = "V L^2 norm - m={:.2f}".format(m4))
    m5,_ = np.polyfit(np.log(h_list), np.log(l2z_error[i,:]), 1)
    axs[i].loglog(h_list, l2z_error[i,:], '-k.' , label = "z L^2 norm - m={:.2f}".format(m5))

    axs[i].legend(loc='lower right')
    axs[i].set_title("Finite element space order {:d}".format(k))
    axs[i].grid('on')
    

# %%
## Comments/problems/updates

'''

- Balazs uses the discrete n_h for the contruction of the rhs, while I get convergence only if I use the ns deriving from the surface X. I thus obtain optimal convergence only if the geimetry order is +2 wrt the FEM space order
- Why not dealing with part of the rhs in an IMEX fashion for stability?
- Seems like a refiniemen of the internal part of the torus is quite imporant for convergence 
'''
##
# %%