from netgen.csg import *
from ngsolve.internal import visoptions
from netgen.meshing import *
from ngsolve import *

from math import pi

def SolvePseudoDG(mesh, order=1, label= "PseudoDG"):
  V = SurfaceL2(mesh, order = 0)
  print(V.ndof)
  V = Compress(V)
  print(V.ndof)
  # asdf
  ndof_el = (order+1) * (order+2)
  ndof = V.ndof * ndof_el
  nze = ndof * 4 * ndof_el
  return ndof, ndof, nze, 0 ,0 ,0
  
