# %%

from cosmos.solvers.moving_sb_adr import MovingSurfaceBulkADRSolver
from cosmos.solvers.base_problem import MovingProblem
from cosmos.solvers.mesh_moving import HarmonicMMSolver
from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
import numpy as np
from ngsolve.webgui import Draw
import matplotlib.pyplot as plt

'''
Article: https://pubmed.ncbi.nlm.nih.gov/26414403/
Problems:
- The simulation as of now does not reach a stationary state
- I fear it's related to mass conservation, I will read more in depth 
    the article and try to see if that's what is missing
'''

'''
Creating of the initial mesh and computation of the initial circumference
'''
mesh, _ = generate_circle(R=1)
Area0 = Integrate(CF(1), mesh, VOL_or_BND=BND)

'''
Definition of the time parameters
'''
T = 10
dt = Parameter(0.01)
t =Parameter(0)

'''
Definition of the simulation
'''
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
# Below the normal velocity (scalar), which is a function of various parameters
# as described in the supplements of:
# https://pubmed.ncbi.nlm.nih.gov/26414403/
m = 4 
s = 8
def velocity_n():

    mesh.SetDeformation(simulation.data['dX'])
    Area = Integrate(CF(1), mesh, VOL_or_BND=BND)
    mesh.UnsetDeformation()
    stretch = Area**(s+1)/(Area**s + Area0**s)

    A, B = sb_sol.get_solution()

    V = (A**m/(A**m + stretch**m) - B**m/(B**m + 1**m))

    return V
# Here I define the velocity as vector
# TO DO: double check if this is actually what I want
# Maybe better to create a GridFunction?
def velocity():
    mesh.SetDeformation(simulation.data['dX'])
    V = velocity_n()*specialcf.normal(mesh.dim)
    mesh.UnsetDeformation()
    return V
# Reaction parameters
def reaction1():
    cf = -1-0.4*velocity_n()
    return cf
def reaction2():
    cf = -1+0.4*velocity_n() 
    return cf

'''
Adding the species to the solver
'''
angle = 120
## Species 1
sb_sol.attach_surface_adr(diffusion = CF(1),
                          reaction = CF(-1),
                          u0 = CF(1))
## Species 2
sb_sol.attach_surface_adr(diffusion = CF(1),
                          reaction = CF(-1),
                        #   u0 = CF(1))
                         u0 = CF(1))

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
harmonic_mm_sol = HarmonicMMSolver(velocity=velocity)
simulation.attach_mm_solver(harmonic_mm_sol)

'''
Running the simulation
'''
modified = False
mass = []

simulation.initialize()

scene = Draw(mesh, deformation = simulation.data['dX'])

sA = []
sB = []

for step in simulation():

    mesh.SetDeformation(simulation.data['dX'])
    sA.append( Integrate(sb_sol.get_solution()[0], mesh, BND) )
    sB.append( Integrate(sb_sol.get_solution()[1], mesh, BND) )
    mesh.UnsetDeformation()

    scene.Redraw()

    if simulation.t.Get() > 5 and not modified:

        print('Entered the loop')

        A, B = sb_sol.get_solution()

        sb_sol.set_solution([A, IfPos(x/Norm(CF((x,y))) - cos(angle/180*pi), 0.1*B, B)])

        modified = True

plt.plot(simulation.data['t_array'], np.array(sA), label = 'A')
plt.plot(simulation.data['t_array'], np.array(sB), label = 'B')
plt.plot(simulation.data['t_array'], np.array(sA)+np.array(sB), label = 'A+B')
plt.legend()
plt.show()
# Drawing the final state
# sb_sol.draw_solution()

'''
Saving the solution
'''
sb_sol.save_solution(filename = './cell_polarity/cell_polarity', n_steps = 200)

# %%
