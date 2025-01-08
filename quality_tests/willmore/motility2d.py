# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.solvers.um_bs_adr_cg_solver import um_sb_ADR_CGSolver
from cosmos.solvers.vector_poisson_solver import VectorPoissonSolver
from cosmos.solvers.distance_solver import DistanceSolver
from cosmos.utils.generate_surface_meshes import generate_circle
import time
import numpy as np
import pandas as pd
import os
directory = "./motility_tests"
os.makedirs(directory, exist_ok=True)

# mesh = Mesh(unit_square.GenerateMesh(maxh = 0.05))
mesh, _ = generate_circle(maxh=0.1, order_g=1)

initial_mc_type = [1]
postprocessing_type = ['harmonic_step']
stabilized_mc = [True]

fes_order = 1
T = 100
dt0 = 0.01
dh0 = 0.2
A0 = 1
B0 = 1*IfPos(x+0.1, 0.1, 1)

dt1 = Parameter(dt0)
t1 = Parameter(0.0)
B0 = IfPos(t1-2, IfPos(x, 0.1, 1), 1)
params_sol1 = {'k_diffusion_s': CF(0.1),
              'initial_c_s': A0,
              'coupling_coefs': [CF(0), CF(0)]}
    
solver1 = um_sb_ADR_CGSolver(mesh, fes_order=fes_order, dt=dt1, t=t1, T=T, params = params_sol1)
A_generator = solver1()

dt2 = Parameter(dt0)
t2 = Parameter(0.0)
params_sol2 = {'k_diffusion_s': CF(0.1),
              'initial_c_s': B0,
              'coupling_coefs': [CF(0), CF(0)]}
solver2 = um_sb_ADR_CGSolver(mesh, fes_order=fes_order, dt=dt2, t=t2, T=T, params = params_sol2)
B_generator = solver2()

bnd_cond={'dirichlet': [None]}
v_p_params = {'boundary_c': bnd_cond}
v_poisson_solver = VectorPoissonSolver(mesh, fes_order=fes_order, params = v_p_params)

distance_params = {'zero_bnd': '.*'}
distance_solver = DistanceSolver(mesh, fes_order=fes_order, params = distance_params)

# Grid-functions for printing purposes
fes2 = VectorH1(mesh, order = 1)
fes3 = H1(mesh, order = 1)

gfu_mesh_moving = GridFunction(fes2)
gfu_velocity = GridFunction(fes2)
gfu_displacement = GridFunction(fes2)
gfu_distance = GridFunction(fes3)
gfu_A = GridFunction(fes3)
gfu_B = GridFunction(fes3)
gfu_A.Set(solver1.sol_h[1], definedon=mesh.Boundaries('.*'))
gfu_B.Set(solver2.sol_h[1], definedon=mesh.Boundaries('.*'))

aux = GridFunction(fes3)

velocity = GridFunction(fes3)
displacement = GridFunction(fes3)

kappa_1 = 0.4
kappa_2 = 0.4
a_12 = 1
a_21 = 1
m = 4
s = 8
Area0 = Integrate(1, mesh, VOL_or_BND=BND, order=1)

for i, ic in enumerate(initial_mc_type):

    for j, stab in enumerate(stabilized_mc):

        for k, pp in enumerate(postprocessing_type):

            # Creating sub-directory for results
            sub_dir = directory + '/' + str(i) + str(j) + str(k)
            os.makedirs(sub_dir, exist_ok=True)

            solver1.t.Set(0.0)
            solver2.t.Set(0.0)
            
            surface_area = []
            t_vec = []

            vtkout_v = VTKOutput(mesh, coefs=[gfu_A, gfu_B, gfu_mesh_moving, gfu_distance, gfu_velocity],
                                    names=['A', 'B', 'volume_displacement', 'distance_function',
                                        'distance_f_gradient'], filename=sub_dir + '/results_vol' )
            
            scene1 = Draw(gfu_A, mesh, deformation = gfu_mesh_moving)
            scene2 = Draw(gfu_B, mesh, deformation = gfu_mesh_moving)
            scene3 = Draw(velocity, mesh, deformation = gfu_mesh_moving)

            try:

                out_step = int(int(solver1.T/solver1.dt.Get())/200)+1
                count = 0

                dummy = False

                while True:

                    count+=1

                    next(A_generator)
                    next(B_generator)

                    mesh.SetDeformation(solver1.displ_h)
                    Area = Integrate(1, mesh, order = 2*fes_order, VOL_or_BND=BND)
                    stretch = Area**(s+1)/(Area**s+Area0**s)
                    surface_area.append(Area)
                    gfu_displacement.Set(solver1.displ_h + displacement*specialcf.normal(mesh.dim), definedon=mesh.Boundaries('.*'))
                    mesh.UnsetDeformation()

                    velocity.Set(solver1.sol_h[1]**m/(solver1.sol_h[1]**m+1*stretch**m) \
                                - solver2.sol_h[1]**m/(solver2.sol_h[1]**m+1**m), definedon=mesh.Boundaries('.*'))
                    displacement.Set(velocity*solver1.dt.Get(), definedon=mesh.Boundaries('.*'))

                    v_poisson_solver.params['boundary_c']['dirichlet'] =[['.*', gfu_displacement]]
                    next(v_poisson_solver())

                    distance_solver.params['displacement'] = v_poisson_solver.sol_h[0]
                    next(distance_solver())
                    
                    solver1.params['k_reaction_s'] = -(1+kappa_1*velocity)
                    solver1.params['rhs_s'] = -solver1.sol_h[1]**2 - a_12*solver1.sol_h[1]*solver2.sol_h[1]
                    solver1.params['displacement'] = v_poisson_solver.sol_h[0]

                    solver2.params['k_reaction_s'] = -(1-kappa_2*velocity)
                    solver2.params['rhs_s'] = -solver2.sol_h[1]**2 - a_21*solver1.sol_h[1]*solver2.sol_h[1]
                    solver2.params['displacement'] = v_poisson_solver.sol_h[0]

                    t_vec.append(solver1.t.Get())

                    gfu_mesh_moving.Set(v_poisson_solver.sol_h[0])
                    gfu_distance.Set(distance_solver.sol_h[0])
                    gfu_velocity.Set(distance_solver.sol_h[1])
                    gfu_A.Set(solver1.sol_h[1], definedon=mesh.Boundaries('.*'))
                    gfu_B.Set(solver2.sol_h[1], definedon=mesh.Boundaries('.*'))

                    if count%out_step == 0:
                        vtkout_v.Do(time=solver1.t.Get(), vb=VOL)

                    scene1.Redraw()
                    scene2.Redraw()
                    scene3.Redraw()

                    print('time:', solver1.t.Get(), ' stretch: ', stretch, end="\r")

                    if t1.Get()>2 and dummy == False:

                        solver2.sol_h[1].Set(solver2.sol_h[1], definedon=mesh.Boundaries('.*'))

            except Exception as e:
                
                print(f'caught {type(e)}:', e)

            # Printing the results
            
            name = sub_dir + "/results.dat"

            horizontal_labels = ['name-dt', 'surface_area']
            vertical_labels =  [f'{x:.2e}' for x in t_vec]
            output = np.column_stack((vertical_labels, np.column_stack([surface_area])))

            df = pd.DataFrame(output, columns=horizontal_labels)
            df.to_csv(name, sep='\t', index=False)

       
# %%