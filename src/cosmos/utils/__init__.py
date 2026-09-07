"""Re-exports the generic, simulation-independent geometry/mesh helpers.

Everything here (``cosmos.utils.tools``, ``.generate_meshes``,
``.mesh_fixing``) has no dependency on ``cosmos.core``/``cosmos.pde`` and is
usable on its own, outside of any ``CosmosModel``.
"""

from cosmos.utils.tools import gradient
from cosmos.utils.mesh_fixing import CosmosAliasMesh, fill_mesh
from cosmos.utils.generate_meshes import (
    generate_boundary_arc,
    generate_boundary_1D_circle,
    generate_boundary_plane,
    generate_boundary_circle,
    generate_volume_circle,
    generate_boundary_sphere,
    generate_boundary_ellipse,
    generate_boundary_cigar,
    generate_boundary_half_sphere,
    generate_volume_ball,
    generate_boundary_cylinder,
    generate_boundary_torus,
    generate_boundary_half_torus,
    generate_boundary_box,
    generate_boundary_smoothed_box,
    import_stl_mesh,
)

__all__ = [
    "gradient",
    "CosmosAliasMesh",
    "fill_mesh",
    "generate_boundary_arc",
    "generate_boundary_1D_circle",
    "generate_boundary_plane",
    "generate_boundary_circle",
    "generate_volume_circle",
    "generate_boundary_sphere",
    "generate_boundary_ellipse",
    "generate_boundary_cigar",
    "generate_boundary_half_sphere",
    "generate_volume_ball",
    "generate_boundary_cylinder",
    "generate_boundary_torus",
    "generate_boundary_half_torus",
    "generate_boundary_box",
    "generate_boundary_smoothed_box",
    "import_stl_mesh",
]
