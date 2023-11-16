from netgen.csg import *
from ngsolve.internal import visoptions
from netgen.meshing import *
from ngsolve import *

from math import pi

from HDG import SolveHDG
from HdivHDG import SolveHdivHDG
from H1_Lag import SolveH1Lagrange
from H1_Pen import SolveH1Penalty
from PseudoDG import SolvePseudoDG
from PseudoHdivDG import SolvePseudoHdivDG

geo = CSGeometry()

r = 1
sphere = Sphere(Pnt(0,0,0),r).bc("dir")
geo.Add(sphere)
mesh = Mesh(geo.GenerateMesh(perfstepsend=MeshingStep.MESHSURFACE,optsteps2d=3,maxh=20))

Draw(mesh)

normal = specialcf.normal(3)
nmat = CoefficientFunction(normal, dims =(3,1))
proj = Id(3) - nmat*nmat.trans

force = CoefficientFunction(((-56 * z ** 2 + 36 * z) * x ** 8 / 2 + 18 * x ** 7 * y ** 2 + (-2 - 168 * z ** 4 + 108 * z ** 3 + (-168 * y ** 2 + 228) * z ** 2 + (108 * y ** 2 - 140) * z) * x ** 6 / 2 + (108 * y ** 4 + 108 * y ** 2 * z ** 2 - 140 * y ** 2 + 2) * x ** 5 / 2 + (-168 * z ** 6 + 108 * z ** 5 + (-336 * y ** 2 + 456) * z ** 4 + (216 * y ** 2 - 280) * z ** 3 + (-168 * y ** 4 + 456 * y ** 2 - 342) * z ** 2 + (108 * y ** 4 - 280 * y ** 2 + 198) * z - 4 * y ** 2 + 6) * x ** 4 / 2 + (108 * y ** 2 * z ** 4 + (216 * y ** 4 - 280 * y ** 2 + 4) * z ** 2 + 108 * y ** 6 - 280 * y ** 4 + 198 * y ** 2 - 6) * x ** 3 / 2 + (-56 * z ** 8 + 36 * z ** 7 + (-168 * y ** 2 + 228) * z ** 6 + (108 * y ** 2 - 140) * z ** 5 + (-168 * y ** 4 + 456 * y ** 2 - 345) * z ** 4 + (108 * y ** 4 - 280 * y ** 2 + 202) * z ** 3 + (-56 * y ** 6 + 228 * y ** 4 - 347 * y ** 2 + 208) * z ** 2 + (36 * y ** 6 - 140 * y ** 4 + 202 * y ** 2 - 113) * z - 2 * y ** 4 + 6 * y ** 2 - 6) * x ** 2 / 2 + (36 * y ** 2 * z ** 6 + (108 * y ** 4 - 140 * y ** 2 + 2) * z ** 4 + (108 * y ** 6 - 280 * y ** 4 + 198 * y ** 2 - 6) * z ** 2 + 36 * y ** 8 - 140 * y ** 6 + 196 * y ** 4 - 108 * y ** 2 + 4) * x / 2 + 1 - 0.5e1 / 0.2e1 * z ** 6 + 2 * z ** 5 + (-10 * y ** 2 + 17) * z ** 4 / 2 + (8 * y ** 2 - 11) * z ** 3 / 2 + (-5 * y ** 4 + 17 * y ** 2 - 24) * z ** 2 / 2 + (4 * y ** 4 - 11 * y ** 2 + 12) * z / 2,-28 * y * ((z ** 2 - 0.9e1 / 0.14e2 * z) * x ** 7 - 0.9e1 / 0.14e2 * x ** 6 * y ** 2 + (0.1e1 / 0.28e2 + 3 * z ** 4 - 0.27e2 / 0.14e2 * z ** 3 + (-0.57e2 / 0.14e2 + 3 * y ** 2) * z ** 2 + (0.5e1 / 0.2e1 - 0.27e2 / 0.14e2 * y ** 2) * z) * x ** 5 + (-0.27e2 / 0.14e2 * y ** 4 - 0.5e1 / 0.28e2 + 0.5e1 / 0.2e1 * y ** 2 - 0.27e2 / 0.14e2 * y ** 2 * z ** 2) * x ** 4 + (3 * z ** 6 - 0.27e2 / 0.14e2 * z ** 5 + (-0.57e2 / 0.7e1 + 6 * y ** 2) * z ** 4 + (5 - 0.27e2 / 0.7e1 * y ** 2) * z ** 3 + (0.337e3 / 0.56e2 - 0.57e2 / 0.7e1 * y ** 2 + 3 * y ** 4) * z ** 2 + (-0.97e2 / 0.28e2 + 5 * y ** 2 - 0.27e2 / 0.14e2 * y ** 4) * z + y ** 2 / 14 - 0.3e1 / 0.28e2) * x ** 3 + (-0.27e2 / 0.14e2 * y ** 2 * z ** 4 + (5 * y ** 2 - 0.5e1 / 0.14e2 - 0.27e2 / 0.7e1 * y ** 4) * z ** 2 - 0.27e2 / 0.14e2 * y ** 6 - 0.107e3 / 0.28e2 * y ** 2 + 0.1e1 / 0.2e1 + 5 * y ** 4) * x ** 2 + (z ** 8 - 0.9e1 / 0.14e2 * z ** 7 + (-0.57e2 / 0.14e2 + 3 * y ** 2) * z ** 6 + (0.5e1 / 0.2e1 - 0.27e2 / 0.14e2 * y ** 2) * z ** 5 + (0.335e3 / 0.56e2 - 0.57e2 / 0.7e1 * y ** 2 + 3 * y ** 4) * z ** 4 + (-0.97e2 / 0.28e2 + 5 * y ** 2 - 0.27e2 / 0.14e2 * y ** 4) * z ** 3 + (-0.191e3 / 0.56e2 - 0.57e2 / 0.14e2 * y ** 4 + 0.337e3 / 0.56e2 * y ** 2 + y ** 6) * z ** 2 + (0.51e2 / 0.28e2 + 0.5e1 / 0.2e1 * y ** 4 - 0.97e2 / 0.28e2 * y ** 2 - 0.9e1 / 0.14e2 * y ** 6) * z - 0.3e1 / 0.28e2 * y ** 2 + 0.3e1 / 0.28e2 + y ** 4 / 28) * x - 0.9e1 / 0.14e2 * y ** 2 * z ** 6 + (-0.27e2 / 0.14e2 * y ** 4 - 0.5e1 / 0.28e2 + 0.5e1 / 0.2e1 * y ** 2) * z ** 4 + (-0.27e2 / 0.14e2 * y ** 6 - 0.107e3 / 0.28e2 * y ** 2 + 0.1e1 / 0.2e1 + 5 * y ** 4) * z ** 2 - 0.15e2 / 0.28e2 - 0.9e1 / 0.14e2 * y ** 8 + 0.65e2 / 0.28e2 * y ** 2 + 0.5e1 / 0.2e1 * y ** 6 - 0.51e2 / 0.14e2 * y ** 4),-28 * z ** 9 * x + 18 * z ** 8 * x + (-168 * x ** 3 - 168 * x * y ** 2 + 36 * y ** 2 + 228 * x) * z ** 7 / 2 + (108 * x ** 3 + 108 * x * y ** 2 - 140 * x) * z ** 6 / 2 + (-168 * x ** 5 + (-336 * y ** 2 + 456) * x ** 3 + 108 * x ** 2 * y ** 2 + (-168 * y ** 4 + 456 * y ** 2 - 345) * x + 108 * y ** 4 - 140 * y ** 2 + 2) * z ** 5 / 2 + (108 * x ** 5 + (216 * y ** 2 - 280) * x ** 3 + (108 * y ** 4 - 280 * y ** 2 + 198) * x) * z ** 4 / 2 + (-56 * x ** 7 + (-168 * y ** 2 + 228) * x ** 5 + 108 * x ** 4 * y ** 2 + (-168 * y ** 4 + 456 * y ** 2 - 357) * x ** 3 + (216 * y ** 4 - 280 * y ** 2 + 4) * x ** 2 + (-56 * y ** 6 + 228 * y ** 4 - 357 * y ** 2 + 219) * x + 108 * y ** 6 - 280 * y ** 4 + 198 * y ** 2 - 6) * z ** 3 / 2 + 18 * x * (x ** 6 + (3 * y ** 2 - 0.35e2 / 0.9e1) * x ** 4 + (3 * y ** 4 - 0.70e2 / 0.9e1 * y ** 2 + 0.101e3 / 0.18e2) * x ** 2 + y ** 6 - 0.35e2 / 0.9e1 * y ** 4 + 0.101e3 / 0.18e2 * y ** 2 - 0.113e3 / 0.36e2) * z ** 2 + (36 * x ** 6 * y ** 2 - 12 * x ** 5 + (108 * y ** 4 - 140 * y ** 2 + 2) * x ** 4 + (-24 * y ** 2 + 34) * x ** 3 + (108 * y ** 6 - 280 * y ** 4 + 198 * y ** 2 - 6) * x ** 2 + (-12 * y ** 4 + 34 * y ** 2 - 36) * x + 36 * y ** 8 - 140 * y ** 6 + 196 * y ** 4 - 108 * y ** 2 + 4) * z / 2 + 2 * x ** 5 + (8 * y ** 2 - 11) * x ** 3 / 2 + (4 * y ** 4 - 11 * y ** 2 + 14) * x / 2)).Compile(True)

dexu = CoefficientFunction((((-x ** 2 + 1) * (2 * z ** 2 * x - 2 * x * z - y ** 2) - x * y * (y * z ** 2 - y * z) - x * z * (z ** 3 - z ** 2 + 1)) * (-x ** 2 + 1) - (-2 * (-x ** 2 + 1) * x * y - x * y * (z ** 2 * x - x * z - 3 * y ** 2 + 1) + 2 * y * z ** 2 * x) * x * y - ((-x ** 2 + 1) * (-2 * (-x ** 2 + 1) * z - x ** 2) - x * y * (2 * x * y * z - x * y) - x * z * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * x * z,-((-x ** 2 + 1) * (2 * z ** 2 * x - 2 * x * z - y ** 2) - x * y * (y * z ** 2 - y * z) - x * z * (z ** 3 - z ** 2 + 1)) * x * y + (-2 * (-x ** 2 + 1) * x * y - x * y * (z ** 2 * x - x * z - 3 * y ** 2 + 1) + 2 * y * z ** 2 * x) * (-y ** 2 + 1) - ((-x ** 2 + 1) * (-2 * (-x ** 2 + 1) * z - x ** 2) - x * y * (2 * x * y * z - x * y) - x * z * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * y * z,-((-x ** 2 + 1) * (2 * z ** 2 * x - 2 * x * z - y ** 2) - x * y * (y * z ** 2 - y * z) - x * z * (z ** 3 - z ** 2 + 1)) * x * z - (-2 * (-x ** 2 + 1) * x * y - x * y * (z ** 2 * x - x * z - 3 * y ** 2 + 1) + 2 * y * z ** 2 * x) * y * z + ((-x ** 2 + 1) * (-2 * (-x ** 2 + 1) * z - x ** 2) - x * y * (2 * x * y * z - x * y) - x * z * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * (-z ** 2 + 1),(-x * y * (2 * z ** 2 * x - 2 * x * z - y ** 2) + (-y ** 2 + 1) * (y * z ** 2 - y * z) - y * z * (z ** 3 - z ** 2 + 1)) * (-x ** 2 + 1) - (2 * x ** 2 * y ** 2 + (-y ** 2 + 1) * (z ** 2 * x - x * z - 3 * y ** 2 + 1) + 2 * y ** 2 * z ** 2) * x * y - (-x * y * (-2 * (-x ** 2 + 1) * z - x ** 2) + (-y ** 2 + 1) * (2 * x * y * z - x * y) - y * z * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * x * z,-(-x * y * (2 * z ** 2 * x - 2 * x * z - y ** 2) + (-y ** 2 + 1) * (y * z ** 2 - y * z) - y * z * (z ** 3 - z ** 2 + 1)) * x * y + (2 * x ** 2 * y ** 2 + (-y ** 2 + 1) * (z ** 2 * x - x * z - 3 * y ** 2 + 1) + 2 * y ** 2 * z ** 2) * (-y ** 2 + 1) - (-x * y * (-2 * (-x ** 2 + 1) * z - x ** 2) + (-y ** 2 + 1) * (2 * x * y * z - x * y) - y * z * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * y * z,-(-x * y * (2 * z ** 2 * x - 2 * x * z - y ** 2) + (-y ** 2 + 1) * (y * z ** 2 - y * z) - y * z * (z ** 3 - z ** 2 + 1)) * x * z - (2 * x ** 2 * y ** 2 + (-y ** 2 + 1) * (z ** 2 * x - x * z - 3 * y ** 2 + 1) + 2 * y ** 2 * z ** 2) * y * z + (-x * y * (-2 * (-x ** 2 + 1) * z - x ** 2) + (-y ** 2 + 1) * (2 * x * y * z - x * y) - y * z * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * (-z ** 2 + 1),(-x * z * (2 * z ** 2 * x - 2 * x * z - y ** 2) - y * z * (y * z ** 2 - y * z) + (-z ** 2 + 1) * (z ** 3 - z ** 2 + 1)) * (-x ** 2 + 1) - (2 * x ** 2 * z * y - y * z * (z ** 2 * x - x * z - 3 * y ** 2 + 1) - 2 * (-z ** 2 + 1) * y * z) * x * y - (-x * z * (-2 * (-x ** 2 + 1) * z - x ** 2) - y * z * (2 * x * y * z - x * y) + (-z ** 2 + 1) * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * x * z,-(-x * z * (2 * z ** 2 * x - 2 * x * z - y ** 2) - y * z * (y * z ** 2 - y * z) + (-z ** 2 + 1) * (z ** 3 - z ** 2 + 1)) * x * y + (2 * x ** 2 * z * y - y * z * (z ** 2 * x - x * z - 3 * y ** 2 + 1) - 2 * (-z ** 2 + 1) * y * z) * (-y ** 2 + 1) - (-x * z * (-2 * (-x ** 2 + 1) * z - x ** 2) - y * z * (2 * x * y * z - x * y) + (-z ** 2 + 1) * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * y * z,-(-x * z * (2 * z ** 2 * x - 2 * x * z - y ** 2) - y * z * (y * z ** 2 - y * z) + (-z ** 2 + 1) * (z ** 3 - z ** 2 + 1)) * x * z - (2 * x ** 2 * z * y - y * z * (z ** 2 * x - x * z - 3 * y ** 2 + 1) - 2 * (-z ** 2 + 1) * y * z) * y * z + (-x * z * (-2 * (-x ** 2 + 1) * z - x ** 2) - y * z * (2 * x * y * z - x * y) + (-z ** 2 + 1) * (3 * z ** 2 * x - 2 * x * z - y ** 2)) * (-z ** 2 + 1)), dims = (3,3)).Compile(True)

exu = CoefficientFunction((-z*z,y,x))

Draw(cf = proj*exu, mesh = mesh, name = "ex_u")
Draw(cf =force, mesh = mesh, name = "force")

quantities = ["ndof", "ncdof", "nze", "L2Error_tang", "L2Error_normal", "H1Error"]
methods = ["PseudoDG", "PseudoHdivDG", "HDG", "HdivHDG", "H1Lag", "H1Pen"]

nref = 5

eval_quantities = dict()
for method in methods:
    eval_quantities[method] = dict()
    for error in quantities:
        eval_quantities[method][error] = [ None for i in range(nref)]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ku", help="order", type=int)
parser.add_argument("--kg", help="geom_order", type=int)
args = parser.parse_args()
# kwargs = dict(order

if args.ku == None:
    args.ku = 1
    print("Setting ku to", args.ku)

if args.kg == None:
    args.kg = args.ku + 1
    print("Setting kg to", args.kg)

condense = True
order = args.ku
geomorder = args.kg

print(eval_quantities)
for ref in range(nref):
    if ref != 0:
        mesh.Refine()
        
    mesh.Curve(geomorder)
    
    h = specialcf.mesh_size
    
    Draw(mesh)
    local_errs = SolveHDG( mesh, exu = exu, ex_du = dexu, force = force, order = order, condense=condense, alpha = 10, label="HDG")
    for i, error in enumerate(quantities):
        eval_quantities["HDG"][error][ref] = local_errs[i]
    local_errs = SolveHdivHDG( mesh, exu = exu, ex_du = dexu, force = force, order = order, condense=condense, alpha = 10, label="HdivDG")
    for i, error in enumerate(quantities):
        eval_quantities["HdivHDG"][error][ref] = local_errs[i]
    local_errs = SolveH1Lagrange( mesh, exu = exu, ex_du = dexu, force = force, order = order, condense=condense, label="H1Lag")
    for i, error in enumerate(quantities):
        eval_quantities["H1Lag"][error][ref] = local_errs[i]
    local_errs = SolveH1Penalty( mesh, exu = exu, ex_du = dexu, force = force, order = order, condense=condense,
                                 penalty = 10*h**(-order-1),
                                 # penalty = h**(-2),
                                 label="H1Pen")
    for i, error in enumerate(quantities):
        eval_quantities["H1Pen"][error][ref] = local_errs[i]

    local_errs = SolvePseudoDG( mesh, order = order, label="PseudoDG")
    for i, error in enumerate(quantities):
        eval_quantities["PseudoDG"][error][ref] = local_errs[i]

    local_errs = SolvePseudoHdivDG( mesh, order = order, label="PseudoHdivDG")
    for i, error in enumerate(quantities):
        eval_quantities["PseudoHdivDG"][error][ref] = local_errs[i]

    print(eval_quantities)    



def write(string):
    outfile.write(string)
    print(string,end="")
        
for quantity in quantities:
    outfile = open("data_sphere/comparison_"+quantity+"_Pk"+str(order)+"_Pg"+str(geomorder)+".dat","w")
    write("#"+quantity+"\n")
    write("#ref"+"\t")
    for method in methods:
        write(method+"\t")
    write("\n")
    for ref in range(nref):
        write(str(ref)+"\t")
        for method in methods:
            write(str(eval_quantities[method][quantity][ref])+"\t")
        write("\n")
    write("\n")

    outfile.close()    
    
