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

from geometries.kink import MakeKinkMesh, RefineKinkMesh




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ku", help="order", type=int)
parser.add_argument("--H", help="height", type=float)
args = parser.parse_args()
# kwargs = dict(order

if args.ku == None:
    args.ku = 1
    print("Setting ku to", args.ku)

if args.H == None:
    args.H = 1
    print("Setting H to", args.H)

H=args.H
#sqrt(0.75)
# H=0.01
#H=0
globalh=0.2
mesh = MakeKinkMesh(H,maxh=globalh)
W = sqrt(1-H**2)

# Draw(x,mesh,"x")
# gft = GridFunction(H1(mesh,order=1))
# gft.Set(x,definedon = mesh.Boundaries(".*"))
# Draw(gft,mesh,"x2")

# input("")
# exit()

xi = x/W
eta = y
tau1 = CoefficientFunction((W,0,IfPos(x-W,-H,H)))
tau2 = CoefficientFunction((0,1,0))

# Draw(mesh)

normal = specialcf.normal(3)
nmat = CoefficientFunction(normal, dims =(3,1))
proj = Id(3) - nmat*nmat.trans

exu = (cos(pi*xi)+sin(pi*eta))*tau1 + (sin(pi*xi)+cos(pi*eta))*tau2

m_div_eps_u = pi**2*(cos(pi*xi)+0.5*sin(pi*eta))*tau1 + pi**2*(cos(pi*eta)+0.5*sin(pi*xi))*tau2

force = m_div_eps_u + exu

dexu = -pi * sin(pi*xi) * OuterProduct(tau1,tau1) + pi * cos(pi*eta) * OuterProduct(tau1,tau2) + pi * cos(pi*xi) * OuterProduct(tau2,tau1) - pi * sin(pi*eta) * OuterProduct(tau2,tau2)

# CoefficientFunction((0,0,0,0,0,0,0,0,0),dims=(3,3))


Draw(cf = proj*exu, mesh = mesh, name = "ex_u")
Draw(cf =force, mesh = mesh, name = "force")

quantities = ["ndof", "ncdof", "nze", "L2Error_tang", "L2Error_normal", "H1Error"]
methods = ["HDG", "HdivHDG", "H1Lag", "H1Pen"]

nref = 5

eval_quantities = dict()
for method in methods:
    eval_quantities[method] = dict()
    for error in quantities:
        eval_quantities[method][error] = [ None for i in range(nref)]

# kwargs = dict(order 
condense = True


condense = True
order = args.ku

# geomorder = order+1
dirichlet = "left|right|topleft|topright|bottomleft|bottomright"

print(eval_quantities)
for ref in range(nref):
    if ref != 0:
        RefineKinkMesh(mesh, H)
        # globalh /= 2
        # mesh = MakeKinkMesh(H,maxh=globalh)
        # mesh.Refine()
        
    # mesh.Curve(geomorder)
    
    h = specialcf.mesh_size
    
    Draw(mesh)
    local_errs = SolveHDG( mesh, exu = exu, ex_du = dexu, force = force, order = order, condense=condense, alpha = 10,dirichlet=dirichlet, label="HDG")
    for i, error in enumerate(quantities):
        eval_quantities["HDG"][error][ref] = local_errs[i]
    local_errs = SolveHdivHDG( mesh, exu = exu, ex_du = dexu, force = force, order = order, condense=condense, alpha = 10, label="HdivDG")
    for i, error in enumerate(quantities):
        eval_quantities["HdivHDG"][error][ref] = local_errs[i]
    local_errs = SolveH1Lagrange( mesh, exu = exu, ex_du = dexu, force = force, order = order, condense=condense, label="H1Lag")
    for i, error in enumerate(quantities):
        eval_quantities["H1Lag"][error][ref] = local_errs[i]
    local_errs = SolveH1Penalty( mesh, exu = exu, ex_du = dexu, force = force, order = order, condense=condense, penalty = 10* h**(-order-1), label="H1Pen")
    for i, error in enumerate(quantities):
        eval_quantities["H1Pen"][error][ref] = local_errs[i]

    print(eval_quantities)    



def write(string):
    outfile.write(string)
    print(string,end="")
        
for quantity in quantities:
    outfile = open("data_kink/comparison_"+quantity+"_P"+str(order)+"_H"+str(H)+".dat","w")
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
    
