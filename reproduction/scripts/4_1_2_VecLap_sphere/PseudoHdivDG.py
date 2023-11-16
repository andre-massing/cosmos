from netgen.csg import *
from ngsolve.internal import visoptions
from netgen.meshing import *
from ngsolve import *

from math import pi

def SolvePseudoHdivDG(mesh, order=1, label= "PseudoHdivDG"):
  V = SurfaceL2(mesh, order = 0)
  print(V.ndof)
  V = Compress(V)
  print(V.ndof)
  nel = V.ndof
  V = Compress(HDivSurface(mesh, order = 0))
  nf = V.ndof
  
  # asdf
  ndof_el = (order+1) * (order-1)
  ndof_f = order+1
  ndof = nf * ndof_f + nel * ndof_el
  nze = nf * (ndof_f) * (13*ndof_f+6*ndof_el) + nel * ndof_el * ( 4 * ndof_el + 9 * ndof_f)
  return ndof, ndof, nze, 0 ,0 ,0
  
