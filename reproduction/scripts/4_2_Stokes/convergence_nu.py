from ngsolve import *
from netgen.csg import *
from ngsolve.internal import visoptions

from netgen import meshing
from math import pi


geo = CSGeometry()

from HdivHDG import SolveHdivHDG_stokes
from HDG import SolveHDG_stokes

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ku", help="order", type=int)
parser.add_argument("--kg", help="geom_order", type=int)
parser.add_argument("--maxh", help="maxh", type=float)
parser.add_argument("--nref", help="number of refinement levels", type=int)
parser.add_argument("--nthreads", help="number of threads", type=int)

args = parser.parse_args()

if args.ku == None:
    args.ku = 2
    print("Setting ku to", args.ku)

if args.kg == None:
    args.kg = args.ku + 2
    print("Setting kg to", args.kg)

if args.maxh == None:
    args.maxh = 1
    print("Setting maxh to", args.maxh)

if args.nref == None:
    args.nref = 8
    print("Setting maxh to", args.nref)
    
if args.nthreads != None:
    SetNumThreads(args.nthreads)    

####shell######
r = 1/pi
cyl = Cylinder(Pnt(0,0,0), Pnt(1,0,0),r)
top = Plane(Pnt(0,0,0),Vec(0,0,-1)) #.bc("dir")
left = Plane(Pnt(0,0,0),Vec(-1,0,0))#.bc("dir")
right =Plane(Pnt(1,0,0),Vec(1,0,0))#.bc("dir")

shell = (cyl * left*right*top)

geo.AddSurface(cyl, shell)
################

order = args.ku
geomorder = args.kg

mesh = Mesh(geo.GenerateMesh(maxh=args.maxh))
mesh.Curve(geomorder)
Draw(mesh)

nmat = CoefficientFunction((0,y,z), dims =(3,1)) * pi
proj = Id(3) - nmat*nmat.trans

xi = x
eta =  asin(pi*y)/pi+1/2

tau_1 = CoefficientFunction((1, 0,0))
tau_2 = CoefficientFunction((0,z,-y)) * pi

u_ex = xi*xi*(1-xi)*(1-xi) *(2*eta*(1-eta)*(1-eta) - 2* eta*eta*(1-eta)) * tau_1 - eta*eta*(1-eta)*(1-eta) *(2*xi*(1-xi)*(1-xi) - 2*xi*xi*(1-xi)) * tau_2

D_uex = proj * CoefficientFunction((u_ex[0].Diff(x), u_ex[0].Diff(y),u_ex[0].Diff(z),
                             u_ex[1].Diff(x), u_ex[1].Diff(y),u_ex[1].Diff(z),
                             u_ex[2].Diff(x), u_ex[2].Diff(y),u_ex[2].Diff(z)), dims=(3,3)) * proj

eps_uex = 1/2 * (D_uex + D_uex.trans)
p_ex = xi**5+eta**5-1/3


quantities = ["ndof", "H1err_u", "L2err_u", "L2err_divu", "L2err_p"]
methods = ["HDG","HdivHDG"]

nref = args.nref
condense = True

eval_quantities = dict()
for method in methods:
    eval_quantities[method] = dict()
    for error in quantities:
        eval_quantities[method][error] = [ None for i in range(nref)]

mesh.Curve(geomorder)
with TaskManager():
 for ref in range(nref):
    nu = 10**(-ref)
  
    f_1 =  -12 * (eta - 1/2) * (2*nu) * (xi ** 4 - 2 * xi ** 3 + (2 * eta ** 2 - 2 * eta + 1) * xi ** 2 + (-2 * eta ** 2 + 2 * eta) * xi + eta ** 2 / 3 - eta / 3) + 5 * xi*xi*xi*xi
    f_2 = 24 *(2*nu)* ((eta ** 2 - eta + 1/6) * xi ** 2 + (-eta ** 2 + eta - 1/6) * xi + eta ** 2 * (eta - 1) ** 2 / 2) * (xi - 1/2) + 5 * eta*eta*eta*eta

    force = f_1 * tau_1 + f_2 * tau_2
        
    Draw(mesh)
    local_errs = SolveHDG_stokes( mesh, exu = u_ex, ex_du = eps_uex, exp = p_ex, force = force, nu = 2* nu, order = order, alpha = 10, condense = True , inverse = "umfpack")
    for i, error in enumerate(quantities):
        eval_quantities["HDG"][error][ref] = local_errs[i]

    local_errs = SolveHdivHDG_stokes( mesh, exu = u_ex, ex_du = eps_uex, exp = p_ex, force = force, nu = 2*nu, order = order, alpha = 10, condense = condense , inverse = "umfpack")
    for i, error in enumerate(quantities):
        eval_quantities["HdivHDG"][error][ref] = local_errs[i]

def write(string):
    outfile.write(string)
    print(string,end="")
        
for quantity in quantities:
    outfile = open("data_stokes/nu_comparison_"+quantity+"_ku_"+str(order)+"_kg_"+str(geomorder)+".dat","w")
    write("#"+quantity+"\n")
    write("#ref"+"\t")
    for method in methods:
        write(method+"\t")
    write("\n")
    for ref in range(nref):
        nu = 10**(-ref)
        write(str(nu)+"\t")
        for method in methods:
            write(str(eval_quantities[method][quantity][ref])+"\t")
        write("\n")
    write("\n")

    outfile.close()    
    
