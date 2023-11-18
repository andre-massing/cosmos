# %% 
# Importing the necessary libraries

import time
import sys
from netgen.occ import *
from netgen.meshing import MeshingStep
from netgen.csg import *
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import pandas as pd
from utils import *

# %% Define Stokes CIP solver
def StokesHDG(mesh, *, order_u, nu, f, g, filename="results/stokes_hdg_sol"):
    # Function spaces
    order_p = order_u - 1
    VDiv = HDivSurface(mesh, order=int(order_u))
    VDiv = Compress(VDiv)
    VHat = HCurl(mesh, order=int(order_u), orderface=0)
    VHat = Compress(VHat)
    Q    = SurfaceL2(mesh, order=int(order_p))
    N    = NumberSpace(mesh)
    
    # TODO: Only needed if no boundary is included
    W = VDiv*VHat*Q*N
    (u, uhat, p, lam), (v, vhat, q, mu) = W.TnT()
    
    # Redefine them to be trace by default
    # TODO: Check if we can do this also for u
    uhat, vhat, p, q = uhat.Trace(), vhat.Trace(), p.Trace(), q.Trace() 

    x_h = GridFunction(W)
    u_h = x_h.components[0]
    uhat_h = x_h.components[1]
    p_h = x_h.components[2]

    # Setting up output
    vtkout = VTKOutput(mesh,
                       coefs=[u_h, p_h],
                       names=["u_h", "p_h"],
                       filename=filename, subdivision=2)

    # Mesh size
    h = specialcf.mesh_size

    # Surface normal vector and corresponding projections
    ns = specialcf.normal(mesh.dim)
    Qs = OuterProduct(ns, ns)
    Ps = Id(mesh.dim) - Qs
    
    # Edge tangential and co-normal vector and corresponding projection
    tE = specialcf.tangential(mesh.dim)
    nE = Cross(ns,tE)
    # QE = OuterProduct(nE, nE)
    def  QE(vec):
        return vec - (vec*nE)*nE
    
    # Strain tensor
    # TODO: Check if we need the projection to the right in the strain tensor
    def E_th(u):
        return Sym(Ps*Grad(u)*Ps)
    
    A = BilinearForm(W, symmetric=True)
    A += nu*InnerProduct(E_th(u), E_th(v))*ds
    # Consistency term (Note the positive sign due to the swapping the order of vhat and v.Trace())
    A += nu*InnerProduct(E_th(u)*nE, QE(vhat-v.Trace()))*ds(element_boundary=True)
    # Symmetry term
    A += nu*InnerProduct(E_th(v)*nE, QE(uhat-u.Trace()))*ds(element_boundary=True)
    
    alpha = 5.0 
    A += nu*alpha*order_u*order_u/h*InnerProduct(QE(vhat-v.Trace()), QE(uhat-u.Trace()))*ds(element_boundary=True)
    
    # Mass term
    A += InnerProduct(u.Trace(), v.Trace())*ds
    
    # Add b_h contributions 
    A += -div(u.Trace())*q*ds
    A += -div(v.Trace())*p*ds
    
    # TODO: Only needed if no boundary is included
    # Constraint to enforce zero presure mean for boundary less surfaces
    A += (p*mu + q*lam)*ds

    # Imposing manufactured solutions
    l = LinearForm(W)
    l += (InnerProduct(Ps*f, v.Trace()) - g*q)*ds

    res = x_h.vec.CreateVector()

    # matrix for actual system inversion
    # SetNumThreads(1)
    SetNumThreads(1)
    with TaskManager():
        A.Assemble()
        l.Assemble()
        res.data = l.vec

        res.data -= A.mat*x_h.vec
        x_h.vec.data += A.mat.Inverse(freedofs = W.FreeDofs()) * res.data
        
    vtkout.Do(vb=BND)
    
    return u_h, p_h
