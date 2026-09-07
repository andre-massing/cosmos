# %%
"""Repairing real (segmented/imported) meshes for use in ``CosmosModel``.

Unlike ``generate_meshes.py`` (which builds clean meshes from analytic
geometry), this module edits an existing Netgen ``.vol`` file's raw text
sections directly -- fixing inconsistent surface-triangle winding, deriving
a surface from a volume mesh, or copying boundary tags from one mesh onto a
geometrically-matching one. It exists because meshes produced outside
Cosmos (e.g. from image segmentation) commonly fail assumptions
``CosmosModel``/NGSolve rely on (a consistent outward normal in particular)
that hand-built geometries from ``generate_meshes.py`` never violate. See
``article/vol/README.md`` for a worked example of the full repair pipeline
this module is used for.
"""

import copy
import math
from collections import Counter, defaultdict, deque

import netgen as ngen
import numpy as np
from ngsolve import *

SECTION_HEADERS = [
    "",
    "",
    "# surfnr	domin	domout	tlosurf	bcprop",
    "# surfnr    bcnr   domin  domout      np      p1      p2      p3",
    "# surfnr    bcnr   domin  domout      np      p1      p2      p3",
    "#  matnr      np      p1      p2      p3      p4",
    "# surfid  0   p1   p2   trignum1    trignum2   domin/surfnr1    domout/surfnr2   ednr1   dist1   ednr2   dist2",
    "#          X             Y             Z",
    "#          pnum             index",
    "",
    "",
    "",
    "",
    "#   Surfnr     Red     Green     Blue",
    "",
]

SECTION_NAMES = [
    "dimension",
    "geomtype",
    "facedescriptors",
    "surfaceelementsuv",
    "surfaceelements",
    "volumeelements",
    "edgesegmentsgi2",
    "points",
    "pointelements",
    "materials",
    "bcnames",
    "cd2names",
    "cd3names",
    "face_colours",
    "face_transparencies",
]


class CosmosAliasMeshSection:
    """Represents a single named section of a Netgen mesh file.

    Stores the section name, its optional header comment, the element count
    (``dim``), and the list of data rows (``entries``). Provides helpers to
    print, add, or replace entries and to copy state from another section.
    """

    def __init__(self, name="", header="", dim=0, entries=None):

        self.name = name
        self.header = header
        self.dim = dim
        self.entries = [] if entries is None else list(entries)

    def print(self, f=None):

        if not f:
            f = sys.stdout

        print(self.header, file=f)
        print(self.name, file=f)
        print(self.dim, file=f)
        if self.entries:
            for line in self.entries:
                print("\t".join(str(num) for num in line), file=f)
        print("\n", file=f)

    def add_entries(self, entries):
        self.entries.append(entries)

    def replace_entries(self, entries):
        self.entries = list(entries)

    def set(self, section):
        self.dim = section.dim
        self.entries = list(section.entries)

    def get(self):
        return self.dim, self.entries


class CosmosAliasMesh:
    """Reads, parses, and exposes a Netgen ``.vol`` mesh file for programmatic editing.

    Loads the mesh both as an NGSolve ``Mesh`` object and as a collection of
    :class:`CosmosAliasMeshSection` objects, allowing individual sections
    (points, surface elements, volume elements, boundary names, etc.) to be
    inspected and rewritten before exporting a modified ``.vol`` file.
    """

    def __init__(self, filename: str):

        self.filename = filename
        self.sections = {}
        for section_name, section_header in zip(SECTION_NAMES, SECTION_HEADERS):
            self.sections[section_name] = CosmosAliasMeshSection(
                name=section_name, header=section_header
            )

        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            self.text_lines = f.readlines()
        self.mesh = Mesh(self.filename)
        self.ngmesh = self.mesh.ngmesh

        self._parse_sections()

    def _parse_sections(self):

        for i, line in enumerate(self.text_lines):
            name = line.strip()
            if name in self.sections.keys():
                dim = int(self.text_lines[i + 1].strip())
                self.sections[name].dim = dim
                if name not in {"dimension", "geomtype"}:
                    self.sections[name].entries = [
                        str(self.text_lines[i + 2 + j]).strip().split() for j in range(dim)
                    ]
                else:
                    pass

    def print_text(self, f=None):

        if not f:
            f = sys.stdout

        print("mesh3d\n", file=f)
        for section in self.sections.values():
            if section.name == "cd3names" and self.sections["dimension"].dim == 2:
                pass
            else:
                section.print(f=f)
        print("endmesh\n", file=f)

    def get_points(self):

        points = []
        try:
            for points_vals in self.sections["points"].entries:
                points.append((float(points_vals[0]), float(points_vals[1]), float(points_vals[2])))
        except Exception as e:
            print("Mesh does not have points!")
            print(e)

        return points

    def get_surface_tris(self):

        tris = []
        if self.sections["geomtype"].dim == 0:
            for surface_vals in self.sections["surfaceelements"].entries:
                tris.append((int(surface_vals[5]), int(surface_vals[6]), int(surface_vals[7])))
        elif self.sections["geomtype"].dim == 12:
            for surface_vals in self.sections["surfaceelementsuv"].entries:
                tris.append((int(surface_vals[5]), int(surface_vals[6]), int(surface_vals[7])))

        return tris

    def set_surface_tris(self, tris):

        if self.sections["geomtype"].dim == 0:
            for i, surface_vals in enumerate(self.sections["surfaceelements"].entries):
                surface_vals[5] = tris[i][0]
                surface_vals[6] = tris[i][1]
                surface_vals[7] = tris[i][2]
        elif self.sections["geomtype"].dim == 12:
            for i, surface_vals in enumerate(self.sections["surfaceelementsuv"].entries):
                surface_vals[5] = tris[i][0]
                surface_vals[6] = tris[i][1]
                surface_vals[7] = tris[i][2]

        return tris

    def reorient_surface_triangles_consistently(self, flip=False):

        tris = self.get_surface_tris()
        edge_to_tris = defaultdict(list)
        for ti, t in enumerate(tris):
            for e in self._tri_edges_oriented(t):
                edge_to_tris[self._undirected(e)].append((ti, e))
        n = len(tris)
        flipped = [False] * n
        visited = [False] * n

        for seed in range(n):
            if visited[seed]:
                continue
            q = deque([seed])
            visited[seed] = True
            while q:
                u = q.popleft()
                tu = tris[u]
                if flipped[u]:
                    tu = (tu[0], tu[2], tu[1])
                for e in self._tri_edges_oriented(tu):
                    ue = self._undirected(e)
                    neighs = edge_to_tris[ue]
                    for v, ve in neighs:
                        if v == u:
                            continue
                        ui, uj = e
                        vi, vj = ve
                        same_dir = ui == vi and uj == vj
                        if not visited[v]:
                            visited[v] = True
                            if same_dir:
                                flipped[v] = not flipped[v]
                            q.append(v)
        oriented = []
        for i, t in enumerate(tris):
            if flip:
                oriented.append((t[0], t[2], t[1]) if not flipped[i] else t)
            else:
                oriented.append((t[0], t[2], t[1]) if flipped[i] else t)
        self.set_surface_tris(oriented)

    def _tri_edges_oriented(self, t):
        a, b, c = t
        return ((a, b), (b, c), (c, a))

    def _undirected(self, e):
        i, j = e
        return (i, j) if i < j else (j, i)

    def _compute_area_normal(self, points, tris):
        nsum = np.zeros(3)
        for a, b, c in tris:
            pa = np.array(points[a - 1])
            pb = np.array(points[b - 1])
            pc = np.array(points[c - 1])
            nsum += np.cross(pb - pa, pc - pa)
        return nsum

    def _pca_third_axis(self, points):
        P = np.array(points, dtype=float)
        c = P.mean(axis=0)
        X = P - c
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt[2]

    def globally_align_sign(self, points, tris):
        nsum = self._compute_area_normal(points, tris)
        axis = self._pca_third_axis(points)
        if np.dot(nsum, axis) < 0:
            tris = [(a, c, b) for (a, b, c) in tris]
        return tris

    def flip_normal_orientation(self):

        tris = self.get_surface_tris()
        tris = [(a, c, b) for (a, b, c) in tris]
        self.set_surface_tris(tris)

    def export_mesh(self, name):

        name = name.replace(".vol", "")
        filename = name + ".vol"
        with open(filename, "w") as f:
            self.print_text(f=f)

    def build_boundary_edges(self, tris):
        cnt = Counter()

        def ek(i, j):
            return (i, j) if i < j else (j, i)

        for a, b, c in tris:
            cnt[ek(a, b)] += 1
            cnt[ek(b, c)] += 1
            cnt[ek(c, a)] += 1
        return [e for e, n in cnt.items() if n == 1]

    def build_boundary_loops(self, points, boundary_edges):

        adj = defaultdict(list)
        for u, v in boundary_edges:
            adj[u].append(v)
            adj[v].append(u)
        visited_v = set()
        loops = []
        for start in list(adj.keys()):
            if start in visited_v:
                continue
            comp = set()
            dq = deque([start])
            visited_v.add(start)
            comp.add(start)
            while dq:
                x = dq.popleft()
                for y in adj[x]:
                    if y not in visited_v:
                        visited_v.add(y)
                        comp.add(y)
                        dq.append(y)
            compE = [(u, v) for (u, v) in boundary_edges if u in comp and v in comp]
            if not compE:
                continue
            ladj = defaultdict(list)
            for u, v in compE:
                ladj[u].append(v)
                ladj[v].append(u)
            endpoints = [v for v in comp if len(ladj[v]) == 1]
            start_v = endpoints[0] if endpoints else next(iter(comp))
            order = [start_v]
            used = set()
            cur = start_v
            prev = None
            while True:
                nxt = None
                for nb in ladj[cur]:
                    e = (min(cur, nb), max(cur, nb))
                    if e in used:
                        continue
                    if nb != prev:
                        nxt = nb
                        break
                if nxt is None:
                    break
                used.add((min(cur, nxt), max(cur, nxt)))
                order.append(nxt)
                prev, cur = cur, nxt
                if not endpoints and cur == start_v:
                    break
            if not endpoints and order[0] != order[-1]:
                order.append(order[0])
            loops.append(order)

        # distances
        def P(i):
            return points[i - 1]

        loop_edges_params = []
        for order in loops:
            total = 0.0
            segs = []
            for i in range(len(order) - 1):
                u, v = order[i], order[i + 1]
                du = P(u)
                dv = P(v)
                d = math.dist(du, dv)
                segs.append(((u, v), d))
                total += d
            s = 0.0
            entries = []
            for (u, v), d in segs:
                s0 = 0.0 if total == 0 else s / total
                s += d
                s1 = 1.0 if total == 0 else s / total
                entries.append((u, v, s0, s1))
            loop_edges_params.append(entries)

        return loops, loop_edges_params

    def fix_dim2_boundary(self, override=False):

        if self.sections["edgesegmentsgi2"].dim != 0 and not override:
            raise ValueError("BBoundary list is non-empty, use the flag override to add them")
        elif self.sections["edgesegmentsgi2"].dim != 0 and override:
            self.sections["edgesegmentsgi2"].dim = 0
            self.sections["edgesegmentsgi2"].entries = []
            self.sections["cd2names"].dim = 0
            self.sections["cd2names"].entries = []

        tris = self.get_surface_tris()
        points = self.get_points()
        boundary_edges = self.build_boundary_edges(tris)
        loops, loop_edges_params = self.build_boundary_loops(points, boundary_edges)
        counter = 0

        for loop_id, entries in enumerate(loop_edges_params, start=1):
            for u, v, s0, s1 in entries:
                counter += 1
                self.sections["edgesegmentsgi2"].entries.append(
                    [
                        str(loop_id),
                        str(0),
                        str(u),
                        str(v),
                        str(-1),
                        str(-1),
                        str(0),
                        str(0),
                        str(loop_id),
                        str(s0),
                        str(loop_id),
                        str(s1),
                    ]
                )
        self.sections["edgesegmentsgi2"].dim = counter

        for i in range(loop_id):
            self.sections["cd2names"].entries.append(
                [str(int(i + 1)), "bboundary" + str(int(i + 1))]
            )
        self.sections["cd2names"].dim = loop_id

    def mark_cd_elements(self, mesh_to_mark):
        """
        Marks surface elements and edge segments from mesh_to_mark onto this mesh,
        adding a new boundary condition to the matched surface triangles and
        remapping edge segments to the new mesh's point indices.
        """

        # --- 1. Build point correspondence: new mesh index -> old mesh index ---
        pts_new = np.array(
            [[float(p[k]) for k in range(3)] for p in self.sections["points"].entries]
        )
        pts_old = np.array(
            [[float(p[k]) for k in range(3)] for p in mesh_to_mark.sections["points"].entries]
        )

        pnts_new_to_old = []
        unmatched = []

        for i, point_new in enumerate(pts_new):
            # Fast exact match first
            diffs = np.abs(pts_old - point_new)
            exact = np.where(np.all(diffs < 1e-10, axis=1))[0]
            if exact.size > 0:
                pnts_new_to_old.append(int(exact[0]) + 1)  # 1-based
            else:
                # Nearest-neighbour fallback for floating point edge cases
                dists = np.linalg.norm(pts_old - point_new, axis=1)
                nearest_j = int(np.argmin(dists))
                nearest_dist = dists[nearest_j]
                if nearest_dist < 1e-6:
                    pnts_new_to_old.append(nearest_j + 1)
                    unmatched.append((i + 1, nearest_dist))
                else:
                    pnts_new_to_old.append(0)

        n_matched = sum(1 for x in pnts_new_to_old if x != 0)
        print(f"Matched {n_matched} / {len(pnts_new_to_old)} points.")
        if unmatched:
            print(f"  {len(unmatched)} used nearest-neighbour fallback (all distances negligible):")
            for pnum, dist in unmatched:
                print(f"    point {pnum}: dist = {dist:.2e}")
        if n_matched == 0:
            raise RuntimeError(
                "No points matched — are the two meshes in the same coordinate space?"
            )

        # Reverse map: old index -> new index (for remapping edge segments)
        old_to_new = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(pnts_new_to_old, start=1)
            if old_idx != 0
        }

        # --- 2. Mark surface elements that match mesh_to_mark's surface triangles ---
        bnd_mark_id = max(self.sections["bcnames"].dim + 1, 2)
        bnd_mark_name = f"boundary{bnd_mark_id}"

        # Build a set of canonical triangle keys from mesh_to_mark for O(1) lookup
        old_surf_keys = {
            frozenset([int(e[5]), int(e[6]), int(e[7])])
            for e in mesh_to_mark.sections["surfaceelements"].entries
        }

        n_marked = 0
        for surfel_new in self.sections["surfaceelements"].entries:
            old_indices = {pnts_new_to_old[int(surfel_new[k]) - 1] for k in [5, 6, 7]}
            if old_indices in [fs for fs in old_surf_keys if fs == old_indices]:
                surfel_new[1] = str(bnd_mark_id)
                n_marked += 1
        print(f"Marked {n_marked} surface elements with bcnr={bnd_mark_id}.")

        # Update bcnames
        if self.sections["bcnames"].dim == 0:
            self.sections["bcnames"].entries = [["1", "default"], ["2", "boundary2"]]
            self.sections["bcnames"].dim = 2
        else:
            self.sections["bcnames"].entries.append([str(bnd_mark_id), bnd_mark_name])
            self.sections["bcnames"].dim += 1

        # --- 3. Remap edge segments from mesh_to_mark into this mesh ---
        remapped_edges = []
        skipped_edges = []

        for j, edge_old in enumerate(mesh_to_mark.sections["edgesegmentsgi2"].entries, start=1):
            old_p2, old_p3 = int(edge_old[2]), int(edge_old[3])
            if old_p2 not in old_to_new or old_p3 not in old_to_new:
                skipped_edges.append((j, old_p2, old_p3))
                continue
            edge_new = list(edge_old)
            edge_new[2] = old_to_new[old_p2]
            edge_new[3] = old_to_new[old_p3]
            remapped_edges.append(edge_new)

        if skipped_edges:
            print(f"Warning: {len(skipped_edges)} edge segments skipped (unmapped points):")
            for j, p2, p3 in skipped_edges:
                print(f"  segment {j}: old points ({p2}, {p3})")

        self.sections["edgesegmentsgi2"].entries = remapped_edges
        self.sections["edgesegmentsgi2"].dim = len(remapped_edges)
        self.sections["cd2names"].dim = copy.deepcopy(mesh_to_mark.sections["cd2names"].dim)
        self.sections["cd2names"].entries = copy.deepcopy(mesh_to_mark.sections["cd2names"].entries)

        print(f"Remapped {len(remapped_edges)} edge segments.")

    def build_surface_from_volume(self, surfnr=1, bcnr=1, domin=1, domout=0):
        """
        Extracts boundary triangles from tetrahedral volume elements and populates
        the surfaceelements section. Boundary faces are those belonging to exactly
        one tetrahedron (not shared between two).

        Args:
            surfnr: surface number / face descriptor index (default 1)
            bcnr:   boundary condition number (default 1)
            domin:  domain index inside the surface (default 1)
            domout: domain index outside the surface (default 0)
        """
        vol_entries = self.sections["volumeelements"].entries
        if not vol_entries:
            raise RuntimeError("No volume elements found in mesh.")

        # Count how many tets share each face.
        # A tet entry format is: matnr  np  p1  p2  p3  p4
        # Faces are canonical (sorted) tuples so orientation doesn't affect counting.
        face_count = Counter()
        face_owners = {}  # canonical face -> original (p1,p2,p3) from first owning tet

        for entry in vol_entries:
            # entry: [matnr, np, p1, p2, p3, p4]
            p = [int(entry[2]), int(entry[3]), int(entry[4]), int(entry[5])]
            # The 4 faces of a tetrahedron (outward-facing winding kept as-is for now)
            tet_faces = [
                (p[0], p[1], p[2]),
                (p[0], p[1], p[3]),
                (p[0], p[2], p[3]),
                (p[1], p[2], p[3]),
            ]
            for face in tet_faces:
                canonical = tuple(sorted(face))
                face_count[canonical] += 1
                if canonical not in face_owners:
                    face_owners[canonical] = face  # preserve original winding

        # Boundary faces appear exactly once
        boundary_faces = [
            face_owners[canonical] for canonical, count in face_count.items() if count == 1
        ]

        if not boundary_faces:
            raise RuntimeError("No boundary faces found — are volume elements tetrahedral?")

        # Populate surfaceelements section
        self.sections["surfaceelements"].entries = []
        for p1, p2, p3 in boundary_faces:
            self.sections["surfaceelements"].entries.append(
                [
                    str(surfnr),
                    str(bcnr),
                    str(domin),
                    str(domout),
                    str(3),  # np = 3 (triangle)
                    str(p1),
                    str(p2),
                    str(p3),
                ]
            )
        self.sections["surfaceelements"].dim = len(boundary_faces)


def fill_mesh(old_mesh, maxh):

    new_mesh = ngen.meshing.Mesh()
    fd_outside = new_mesh.Add(ngen.meshing.FaceDescriptor(bc=1, domin=1, surfnr=1))

    pmap1 = {}
    m1 = old_mesh.ngmesh
    for e in m1.Elements2D():
        for v in e.vertices:
            if v not in pmap1:
                pmap1[v] = new_mesh.Add(m1[v])

    for e in m1.Elements2D():
        new_mesh.Add(ngen.meshing.Element2D(fd_outside, [pmap1[v] for v in e.vertices]))

    new_mesh.GenerateVolumeMesh(maxh=maxh, grading=0.7)
    new_mesh = Mesh(new_mesh)

    return new_mesh
