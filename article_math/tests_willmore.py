# %%

import sys
import os

# Add the 'tests_adr' folder to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), './tests_willmore'))
from convergence_clifford_torus_scalar import test_clifford_torus_scalar
from convergence_clifford_torus import test_clifford_torus
from convergence_sphere_scalar import test_sphere_scalar
from convergence_sphere import test_sphere
from convergence_sphere_sp_curv import test_sphere_sp_curv
from sigar311 import test_sigar311
from sigar511 import test_sigar511
from torus21meshlab import test_torus21meshlab
from torus21ngsolve import test_torus21ngsolve

from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from cosmos.pdes.pde_willmore_walker import WillmoreWalker
from cosmos.pdes.pde_willmore_walker_dl import WillmoreWalkerDL
from cosmos.solvers.time_schemes import BDF1

results_folder = "./results_willmore"
time_scheme = BDF1()
solvers = [
    (results_folder ,"WillmoreDziuk", WillmoreDziuk, {"time_scheme": time_scheme}),
    (results_folder ,"WillmoreDziukStab", WillmoreDziukStab, {"time_scheme": time_scheme}),
    (results_folder ,"WillmoreDziukPP", WillmoreDziuk, {"time_scheme": time_scheme, "postprocess": True}),
    (results_folder ,"WillmoreDziukStabPP", WillmoreDziukStab, {"time_scheme": time_scheme, "postprocess": True}),
]

# Run all tests
for results, name, constructor, kwargs in solvers:
    test_clifford_torus(results, name, constructor, **kwargs)
    test_sphere(results, name, constructor, **kwargs)
    test_sphere_sp_curv(results, name, constructor, **kwargs)
    test_sigar311(results, name, constructor, **kwargs)
    test_sigar511(results, name, constructor, **kwargs)
    test_torus21meshlab(results, name, constructor, **kwargs)
    test_torus21ngsolve(results, name, constructor, **kwargs)

solvers = [
    (results_folder ,"WillmoreWalker", WillmoreWalker, { "time_scheme": time_scheme}),
    (results_folder ,"WillmoreWalkerPP", WillmoreWalker, { "time_scheme": time_scheme, "postprocess": True}),
    (results_folder ,"WillmoreWalkerDL", WillmoreWalkerDL, { "time_scheme": time_scheme}),
]

# Run all tests
for results, name, constructor, kwargs in solvers:
    test_clifford_torus_scalar(results, name, constructor, **kwargs)
    test_sphere_scalar(results, name, constructor, **kwargs)
    test_sphere_sp_curv(results, name, constructor, **kwargs)
    test_sigar311(results, name, constructor, **kwargs)
    test_sigar511(results, name, constructor, **kwargs)
    test_torus21meshlab(results, name, constructor, **kwargs)
    test_torus21ngsolve(results, name, constructor, **kwargs)