# %%

from cosmos.solvers.moving_sb_adr import MovingSurfaceBulkADRSolver
from cosmos.solvers.sb_adr import SurfaceBulkADRSolver
from cosmos.solvers.base_problem import MovingProblem
from cosmos.solvers.mesh_moving import HarmonicMMSolver
from cosmos.utils.manufactured_solution_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw
    
from netgen.occ import *

mesh, _ = generate_circle(R=1)
Area0 = Integrate(CF(1), mesh, VOL_or_BND=BND)

T = 12
dt = Parameter(0.01)
t = Parameter(0)

simulation = MovingProblem(mesh=mesh, dt=dt, t=t, T=T)

'''
Defining the coupled solver
'''
fes_order = 1
sb_sol = MovingSurfaceBulkADRSolver(fes_order=fes_order)

'''
Adding the solver to the simulation
'''
simulation.attach_solver(sb_sol)


'''
Defining the coefficients of the species on the surface
'''
def velocity_n():

    mesh.SetDeformation(simulation.data['dX'])
    Area = Integrate(CF(1), mesh, VOL_or_BND=BND)
    mesh.UnsetDeformation()
    stretch = Area**(s+1)/(Area**s + Area0**s)

    A, B = sb_sol.get_solution()

    V = (A**m/(A**m + stretch**m) - B**m/(B**m + 1**m))

    return V
def velocity():
    mesh.SetDeformation(simulation.data['dX'])
    V = velocity_n()*specialcf.normal(mesh.dim)
    mesh.UnsetDeformation()
    return V

def reaction1():
    cf = -1-0.4*velocity_n()
    return cf
def reaction2():
    cf = -1+0.4*velocity_n() 
    return cf

'''
Adding the species to the solver
'''
angle = 160
## Species 1
sb_sol.attach_surface_adr(diffusion = CF(1),
                          reaction = reaction1,
                          u0 = CF(1))
## Species 2
sb_sol.attach_surface_adr(diffusion = CF(1),
                          reaction = reaction2,
                          u0 = IfPos(x/Norm(CF((x,y))) - cos(angle/180*pi), 0.1, 1))
# sb_sol.attach_surface_adr(diffusion = CF(0.1),
#                           reaction = reaction2,
#                           u0 = CF(1))

'''
Add the non-linear coupling between the species
'''
## Non-linear coupling
a12 = 1
a21 = 1
def f1(u1, u2):
    return u1**2+a12*u1*u2
def f2(u1, u2):
    return u2**2+a21*u1*u2
sb_sol.add_coupling(mrk1 = 0, mrk2 = 1, f1 = f1, f2 = f2)

'''
Prescribing the displacement and the mesh-motion
'''
m = 4 
s = 8
harmonic_mm_sol = HarmonicMMSolver(velocity=velocity)
simulation.attach_mm_solver(harmonic_mm_sol)

'''
Running the simulation
'''
modified = False

for step in simulation():

    # if simulation.t.Get() > 10 and not modified:

    #     A, B = sb_sol.get_solution()

    #     sb_sol.set_solution([A, IfPos(x/Norm(CF((x,y))) - cos(angle/180*pi), 0.1*B, B)])

    #     modified = True

    pass

sb_sol.draw_solution()

'''
Plotting the solution
'''

# sb_sol.save_solution(filename = './results/cell_polarity', n_steps = 200)

# %%
