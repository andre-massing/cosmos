# %%

from ngsolve import *
from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_ch_bnd import CahnHilliardBnd
import numpy as np
from cosmos.utils.generate_surface_meshes import generate_half_sphere
from ngsolve.webgui import Draw

mesh, _ = generate_half_sphere(maxh = 0.1)

theta = 0.001
gamma = 0.02

dt = Parameter(1e-2)
t = Parameter(0.0)
T = 10

ch = CahnHilliardBnd(c0 = sin(1e7*(x+y*y)), theta = theta, gamma=gamma,
                     time_scheme = BDF2(), MP = True, BP = [-1, 1])

solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
solver.AddPDE(ch)

scene = Draw(ch.phase, mesh)
for sol in solver():
    scene.Redraw()

# %%

from ngsolve import *
from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_ch_vol import CahnHilliardVol
import numpy as np
from ngsolve.webgui import Draw

# mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))

from netgen.geom2d import *
periodic = SplineGeometry()
pnts = [ (0,0), (1,0), (1,1), (0,1) ]
pnums = [periodic.AppendPoint(*p) for p in pnts]

periodic.Append ( ["line", pnums[0], pnums[1]],bc="outer")
# This should be our master edge so we need to save its number.
lright = periodic.Append ( ["line", pnums[1], pnums[2]], bc="periodic")
periodic.Append ( ["line", pnums[2], pnums[3]], bc="outer")
# Minion boundaries must be defined in the same direction as master ones,
# this is why the the point numbers of this spline are defined in the reverse direction,
# leftdomain and rightdomain must therefore be switched as well!
# We use the master number as the copy argument to create a slave edge.
periodic.Append ( ["line", pnums[0], pnums[3]], leftdomain=0, rightdomain=1, copy=lright, bc="periodic")

mesh = Mesh(periodic.GenerateMesh(maxh=0.05))

theta = 1
gamma = 1e-2

dt = Parameter(1e-2)
t = Parameter(0.0)
T = 1

ch = CahnHilliardVol(c0 = sin(1e7*(x+y*y)), theta = theta, gamma=gamma,
                     time_scheme = BDF2())

solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
solver.AddPDE(ch)

scene = Draw(ch.phase, mesh)
for sol in solver():
    scene.Redraw()
