# %%

from ngsolve import*
from cosmos.solvers.solvers import BackwardEuler
from cosmos.solvers.container import Container
from cosmos.solvers.adr_vol import VolADR
from netgen.occ import *
import numpy as np

shape = Rectangle(1,1).Face()
right=shape.edges.Max(X)
right.name="right"
shape.edges.Min(X).Identify(right,name="left")
top=shape.edges.Max(Y)
top.name="top"
shape.edges.Min(Y).Identify(top,name="bottom")
geom = OCCGeometry(shape, dim=2)
mesh = Mesh(geom.GenerateMesh(maxh=0.05))

M = 1
gamma = 1
epsilon = 1e-3

dt = Parameter(1e-2)
t = Parameter(0.0)
T = 1

adr1 = VolADR(u0 = 0.5*sin(1e7*(x+y*y)) + 0.3, periodic = True,
              MP = True, BP = [-1, 1])
adr2 = VolADR(c = 1, periodic = True, stationary = True)

adr1.SaveSolution(folderpath = './ch_results/', filename = 'ch_2a', n_samples = 1)
adr2.SaveSolution(folderpath = './ch_results/', filename = 'ch_2b', n_samples = 1)

cnt = Container()
cnt.AddPDEs(adr1, adr2)

def f1(trials):
    return M*grad(trials[1])
cnt.AddNonlinearity(marker0 = 0, markers = [0, 1], f = f1, grad = True)

def f2_1(trials):
    return - epsilon*grad(trials[0])
cnt.AddNonlinearity(marker0 = 1, markers = [0, 1], f = f2_1, grad = True)

def f2_2(trials):
    return - gamma*(4*trials[0]**3 - 6*trials[0]**2 + 2*trials[0])
cnt.AddNonlinearity(marker0 = 1, markers = [0, 1], f = f2_2)

solver = BackwardEuler(t = t, dt = dt, T = T, mesh = mesh, verbose = 1)
solver.AddPDEs(cnt)

solver.Solve()
print('\n Computed solution')
adr1.draw_solution(solver.mesh_data)
