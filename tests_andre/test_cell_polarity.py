# %%

from cosmos.solvers.adr import ADRSolver
from cosmos.solvers.simulation import MovingSimulation
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
T = 40
dt = Parameter(0.01)
t =Parameter(0)

'''
Definition of the simulation
'''
simulation = MovingSimulation(mesh=mesh, dt=dt, t=t, T=T)

'''
Defining the coupled solver
'''
sb_sol = ADRSolver()

'''
Adding the solver to the simulation
'''
simulation.AddSolver(sb_sol)


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
def normal_displacement():
    V = velocity_n()*simulation.data['dt']
    return V
# Reaction parameters
def reaction1():
    cf = -1-0.4*velocity_n()
    return cf
def reaction2():
    cf = -1+0.4*velocity_n() 
    return cf

simulation.AddNormalMotion(function = normal_displacement, domain = '.*')

'''
Adding the species to the solver
'''
angle = 30
## Species 1
sb_sol.AddSpecie(VorB = BND,
                 diffusion = CF(0.2),
                reaction = reaction1,
                u0 = CF(1))
## Species 2
sb_sol.AddSpecie(VorB = BND,
                 diffusion = CF(0.2),
                reaction = reaction2,
                u0 = CF(1))

'''
Add the non-linear coupling between the species
'''
## Non-linear coupling
a12 = 1
a21 = 1
def f1(trial):
    return trial[0]**2+a12*trial[0]*trial[1]
def f2(trial):
    return trial[1]**2+a21*trial[0]*trial[1]
sb_sol.AddNonlinearity(marker0 = 0, markers = [0, 1], f = f1, VorB = BND)
sb_sol.AddNonlinearity(marker0 = 1, markers = [0, 1], f = f2, VorB = BND)

'''
Running the simulation
'''
modified = False
mass = []

sA = []
sB = []

for i, step in enumerate(simulation()):

    if i == 0:
        _ = sb_sol.get_solution()
        scene1 = Draw(sb_sol.gfu_save.components[0], deformation = simulation.data['dX'])
        scene2 = Draw(sb_sol.gfu_save.components[1], deformation = simulation.data['dX'])

    mesh.SetDeformation(simulation.data['dX'])
    sA.append( Integrate(sb_sol.get_solution()[0], mesh, BND) )
    sB.append( Integrate(sb_sol.get_solution()[1], mesh, BND) )
    mesh.UnsetDeformation()

    if simulation.data['t'].Get() > 3 and not modified:

        A, B = sb_sol.get_solution()

        sb_sol.set_solution([A, IfPos(x/Norm(CF((x,y))) - cos(angle/180*pi), 0.01*B, B)])

        modified = True

    _ = sb_sol.get_solution()
    scene1.Redraw()
    scene2.Redraw()

plt.plot(np.array(sA), label = 'A')
plt.plot(np.array(sB), label = 'B')
plt.plot(np.array(sA)+np.array(sB), label = 'A+B')
plt.legend()
plt.show()
# Drawing the final state
# sb_sol.draw_solution()


# %%
