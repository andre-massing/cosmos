# Run file via 
# mpirun -np <NUM_PROC> python poisson_mpi_yield.py 
from mpi4py import MPI
from ngsolve import *
from netgen.geom2d import unit_square

comm = MPI.COMM_WORLD

if comm.rank == 0:
    ngmesh = unit_square.GenerateMesh(maxh=0.1).Distribute(comm)
else:
    ngmesh = netgen.meshing.Mesh.Receive(comm)

for l in range(3):
    ngmesh.Refine()
mesh = Mesh(ngmesh)

def solve_fem(mesh):
    fes = H1(mesh, order=3, dirichlet=".*")
    u,v = fes.TnT()

    a = BilinearForm(grad(u)*grad(v)*dx)
    pre = Preconditioner(a, "local")
    a.Assemble()

    f = LinearForm(1*v*dx).Assemble()
    gfu = GridFunction(fes)

    inv = CGSolver(a.mat, pre.mat)
    gfu.vec.data = inv*f.vec

    print("First call ...")
    yield (gfu, f)
    
    f = LinearForm(10*v*dx).Assemble()
    gfu.vec.data = inv*f.vec
    
    print("Second call ...")
    yield (gfu,f)
    

solver = solve_fem(mesh)

for (gfu, f) in solver:
    ip = InnerProduct(gfu.vec, f.vec)
    if comm.rank == 0:
        print ("(u,f) =", ip)
    # (u,f) = 0.03514425357822445

# Parallel write out via vtk
vtk = VTKOutput(mesh, coefs=[gfu], names=["sol"], filename="vtk_mpi_example", subdivision=2) 
vtk.Do()

# Write out data via pickle
import pickle
netgen.meshing.SetParallelPickling(True)
pickle.dump(gfu, open("solution.pickle"+str(comm.rank), "wb"))

# Find out how to load data via pickling 
# This will be important for checkpointing in large simulations





    
