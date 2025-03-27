# %%

from cosmos.utils.generate_surface_meshes import generate_half_sphere
from ngsolve import *
from cosmos.solvers.tools import gradient
from ngsolve.webgui import Draw
import pandas as pd
import matplotlib.pyplot as plt

mesh, _ = generate_half_sphere(maxh = 0.1)

from cosmos.solvers.tools import gradient

n_ex = CF((x,y,z))/Norm(CF((x,y,z)))
P = Id(3) - OuterProduct(n_ex, n_ex) 

u_ex = cos(pi*x)*sin(pi*y) # Exact solution
d = CF(0) # diffusion coefficient
c = 1 + y**2 # reaction coefficient
b = P*CF((-z, 0, x))
flux_d = (- d*gradient(u_ex, P))
flux_b = b*u_ex
rhs = ( Trace(gradient(flux_d + flux_b, P)) + c*u_ex).Compile() # manufactured solution right-hand side

dim = mesh.dim
fes_order = 1
domain = '.*'
periodic = False
params = {
    'd' : d,
    'c' : c,
    'b' : b,
    'rhs': rhs,
    'Fneu_b': {},
    'neu_b': {},
    'dir_b': {},
    'dir_d': {},
    'neu_d': {}
}
dX = GridFunction(VectorH1(mesh=mesh))

bnd = 'dir'

if bnd == 'dir':
    params['dir_b'] = {'.*': u_ex}
    params['dir_d'] = {'.*': u_ex}

else:
    params['neu_d'] = {'.*': flux_d} 
    params['neu_b'] = {'.*': flux_b} 
        
if periodic:
    V = Periodic(H1(mesh, order = fes_order,
                    definedon=mesh.Boundaries(domain)))
else:
    V = H1(mesh, order = fes_order,
        definedon=mesh.Boundaries(domain))
    
if mesh.dim == 2:
    dV = VectorH1(mesh, order = 1,
        definedon=mesh.Boundaries(domain))
else:
    dV = NormalFacetSurface(mesh, order = 0,
        definedon=mesh.Boundaries(domain))


print(V.ndof)
print(dV.ndof)
fes = CompressCompound(V*dV)

gfu = GridFunction(fes)
gfu_old = GridFunction(fes)

gfu.components[0].Set(u_ex, 
            definedon=mesh.Boundaries('.*'))
gfu_old.vec.data = gfu.vec.data

trial = fes.TrialFunction()
test = fes.TestFunction()

ns = specialcf.normal(mesh.dim)
tE = specialcf.tangential(mesh.dim)
h = specialcf.mesh_size
if mesh.dim == 2:
    nE = tE
else:
    nE = Cross(ns, tE)

if mesh.dim == 2:
    gfF = GridFunction(H1(mesh, order = 1,\
                            definedon=mesh.Boundaries('.*')))
else:
    gfF = GridFunction(FacetSurface(mesh, order = 0))

lhs = params['d']*grad(trial[0]).Trace()*grad(test[0]).Trace()\
    *ds(deformation = dX)
lhs += params['c']*trial[0]*test[0]*ds(deformation = dX)
lhs += -params['b']*grad(test[0]).Trace() * trial[0] *ds(deformation = dX)

jump_dudn = (trial[0].Trace().Deriv()*nE - trial[1].Trace()*nE)
jump_dvdn = (test[0].Trace().Deriv()*nE - test[1].Trace()*nE)

S_int = h**2/(Norm(params['d']) + Norm(params['b'])*h + Norm(params['c'])*h**2)

lhs +=  h*S_int*InnerProduct(jump_dudn,jump_dvdn)\
    *ds(deformation = dX, element_boundary=True)

# lhs += trial[1].Trace()*test[1].Trace()*ds(element_boundary=True)

if params['neu_b']:

    for key, value in params['neu_b'].items():
        gfF.Set(1, definedon=mesh.BBoundaries(key))
        lhs += IfPos(InnerProduct(nE, params['b']), 
                        InnerProduct(nE, params['b'])*trial[0], 0)\
                            *gfF*test[0]*ds(deformation = dX, element_boundary=True)
    
if params['Fneu_b']:

    for key, value in params['neu_b'].items():
        gfF.Set(1, definedon=mesh.BBoundaries('.*') - mesh.BBoundaries(key))
        lhs += IfPos(InnerProduct(nE, params['b']), 
                        InnerProduct(nE, params['b'])*trial[0], CF(0))\
                            *gfF*test[0]*ds(deformation = dX, element_boundary=True)
        
if params['dir_d']:

    alpha = 5 * fes_order * (fes_order+1)

    for key, value in params['dir_d'].items():

        gfF.Set(1, definedon=mesh.BBoundaries(key))
        lhs += - params['d']*InnerProduct(nE, grad(trial[0]).Trace())*gfF*test[0]*ds(deformation = dX, element_boundary=True) \
                + params['d']*alpha/h*trial[0]*test[0]*ds(deformation = dX, definedon = mesh.BBoundaries(key))
        # - params['d']*InnerProduct(nE, grad(test[0]).Trace())*gfF*trial[0]*ds(deformation = dX, element_boundary=True)\
        
if params['dir_b']:

    for key, value in params['dir_b'].items():

        gfF.Set(1, definedon=mesh.BBoundaries(key))

        lhs += IfPos(InnerProduct(nE, params['b']), 
                        InnerProduct(nE, params['b'])*trial[0], 0)\
                            *gfF*test[0]*ds(deformation = dX, element_boundary=True)

rhs =  params['rhs']*test[0]*ds(deformation = dX)

if params['neu_d']:

    for key, value in params['neu_d'].items():

        gfF.Set(1, definedon=mesh.BBoundaries(key))

        rhs += -InnerProduct(nE, value)*gfF*test[0]*ds(deformation = dX, element_boundary=True)

if params['neu_b']:

    for key, value in params['neu_b'].items():

        gfF.Set(1, definedon=mesh.BBoundaries(key))

        rhs += -IfPos(InnerProduct(nE, params['b']), 0,
                    InnerProduct(nE, value))*gfF*test[0]\
                        *ds(deformation = dX, element_boundary=True)
    
if params['Fneu_b']:

    for key, value in params['neu_b'].items():

        gfF.Set(1, definedon=mesh.BBoundaries(key))

        rhs += -InnerProduct(nE, value)*gfF*test[0]\
                        *ds(deformation = dX, element_boundary=True)
        
if params['dir_d']:

    alpha = 5 * fes_order * (fes_order+1)

    for key, value in params['dir_d'].items():

        gfF.Set(1, definedon=mesh.BBoundaries(key))
        rhs += params['d']*alpha/h*value*test[0]*ds(deformation = dX, definedon = mesh.BBoundaries(key))
            # - params['d']*InnerProduct(nE, grad(test[0]).Trace())*gfF*value*ds(deformation = dX, element_boundary=True)\ 
        
if params['dir_b']:

    for key, value in params['dir_b'].items():

        gfF.Set(1, definedon=mesh.BBoundaries(key))

        rhs += -IfPos(InnerProduct(nE, params['b']), 0,
                    InnerProduct(nE, params['b']*value))*gfF*test[0]\
                        *ds(deformation = dX, element_boundary=True)
        
A = BilinearForm(fes)
A += lhs

F = LinearForm(fes)
F += rhs

A.Assemble()
F.Assemble()

import scipy.sparse as sp
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = (12, 12)
A_scipy = sp.csr_matrix(A.mat.CSR())
fig = plt.figure(); ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
ax1.set_xlabel("numerically non-zero"); ax1.spy(A_scipy)
ax2.set_xlabel("reserved entries (potentially non-zero)"); ax2.spy(A_scipy,precision=-1)
plt.show()

gfu.vec.data = A.mat.Inverse(freedofs = fes.FreeDofs())*F.vec

gfu_save = GridFunction(H1(mesh))
gfu_save.Set(gfu.components[0], definedon = mesh.Boundaries('.*'))
Draw(gfu_save)
# %%

# %%
