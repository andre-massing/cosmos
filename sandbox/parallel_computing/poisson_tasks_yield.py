#%%
from ngsolve import *
from ngsolve.webgui  import Draw

#%%
def solve_fem():
    mesh = Mesh(unit_cube.GenerateMesh(maxh=0.2))
    fes = H1(mesh, order=2, dirichlet="left|bottom")
    u,v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx)
    c = Preconditioner(a, "local")

    SetNumThreads(4)
    with TaskManager():
        for l in range(3):
            mesh.Refine()
            fes.Update()
            a.Assemble()
            f = LinearForm(1*v*dx).Assemble()
            gfu = GridFunction(fes)
            inv = CGSolver(a.mat, c)
            gfu.vec.data = inv * f.vec
            Draw(gfu)
            yield gfu
            
#%%
# solve_fem()
for u_h in solve_fem():
    print("Solving again ...")