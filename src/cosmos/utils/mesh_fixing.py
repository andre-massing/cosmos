# %%

from ngsolve import *
from ngsolve.webgui import Draw
import netgen as ngen
from collections import defaultdict, deque, Counter
import numpy as np
import re
import math
import copy

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
    
    def __init__(self, name = '', header = '', dim = 0, entries = None):

        self.name = name
        self.header = header
        self.dim = dim
        self.entries = [] if entries is None else list(entries)

    def print(self, f = None):

        if not f:
            f = sys.stdout

        print(self.header, file = f) 
        print(self.name, file = f)
        print(self.dim, file = f)
        if self.entries:
            for line in self.entries:
                print("\t".join(str(num) for num in line), file = f)
        print('\n', file = f)

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

    def __init__(self, filename: str):

        self.filename = filename
        self.sections = {}
        for section_name, section_header in zip(SECTION_NAMES, SECTION_HEADERS):
            self.sections[section_name] = CosmosAliasMeshSection(name = section_name, 
                                                                 header = section_header)

        try:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                self.text_lines=f.readlines()
            self.mesh = Mesh(self.filename)
            self.ngmesh = self.mesh.ngmesh

            self._parse_sections()
        except Exception as e:
            print('Error:', e)
            raise Exception('Impossible to generate NgSolve mesh from given filename')

    def _parse_sections(self):

        for i, line in enumerate(self.text_lines):
            name = line.strip()
            if name in self.sections.keys():
                dim = int(self.text_lines[i+1].strip())
                self.sections[name].dim = dim
                if name not in {"dimension", "geomtype"}:
                    self.sections[name].entries = [str(self.text_lines[i+2+j]).strip().split() for j in range(dim)]
                else:
                    pass

    def print_text(self, f = None):

        if not f:
            f = sys.stdout

        print('mesh3d\n', file = f)
        for section in self.sections.values():
            if section.name == 'cd3names' and self.sections['dimension'].dim == 2:
                pass
            else:
                section.print(f = f)
        print('endmesh\n', file = f)

    def get_points(self):

        points = []
        try:
            for points_vals in self.sections['points'].entries:
                points.append((float(points_vals[0]), float(points_vals[1]), float(points_vals[2])))
        except Exception as e:
            print('Mesh does not have points!')
            print(e)

        return points

    def get_surface_tris(self):

        tris = []
        if self.sections['geomtype'].dim == 0:
            for surface_vals in self.sections['surfaceelements'].entries:
                tris.append((int(surface_vals[5]), int(surface_vals[6]), int(surface_vals[7])))
        elif self.sections['geomtype'].dim == 12:
            for surface_vals in self.sections['surfaceelementsuv'].entries:
                tris.append((int(surface_vals[5]), int(surface_vals[6]), int(surface_vals[7])))

        return tris
    
    def set_surface_tris(self, tris):

        if self.sections['geomtype'].dim == 0:
            for i, surface_vals in enumerate(self.sections['surfaceelements'].entries):
                surface_vals[5] = tris[i][0]
                surface_vals[6] = tris[i][1]
                surface_vals[7] = tris[i][2]
        elif self.sections['geomtype'].dim == 12:
            for i, surface_vals in enumerate(self.sections['surfaceelementsuv'].entries):
                surface_vals[5] = tris[i][0]
                surface_vals[6] = tris[i][1]
                surface_vals[7] = tris[i][2]

        return tris

    def reorient_surface_triangles_consistently(self, flip = False):

        tris = self.get_surface_tris()
        edge_to_tris = defaultdict(list)
        for ti, t in enumerate(tris):
            for e in self._tri_edges_oriented(t):
                edge_to_tris[self._undirected(e)].append((ti, e))
        n = len(tris)
        flipped = [False]*n
        visited = [False]*n

        for seed in range(n):
            if visited[seed]: continue
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
                    for (v, ve) in neighs:
                        if v == u: 
                            continue
                        ui, uj = e
                        vi, vj = ve
                        same_dir = (ui==vi and uj==vj)
                        if not visited[v]:
                            visited[v] = True
                            if same_dir:
                                flipped[v] = not flipped[v]
                            q.append(v)
        oriented = []
        for i,t in enumerate(tris):
            if flip:
                oriented.append((t[0], t[2], t[1]) if not flipped[i] else t)
            else:
                oriented.append((t[0], t[2], t[1]) if flipped[i] else t)
        self.set_surface_tris(oriented)
    
    def _tri_edges_oriented(self, t):
        a,b,c = t
        return ((a,b),(b,c),(c,a))

    def _undirected(self, e):
        i,j = e
        return (i,j) if i<j else (j,i)
    
    def _compute_area_normal(self, points, tris):
        nsum = np.zeros(3)
        for a,b,c in tris:
            pa = np.array(points[a-1]); pb=np.array(points[b-1]); pc=np.array(points[c-1])
            nsum += np.cross(pb-pa, pc-pa)
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
            tris = [(a,c,b) for (a,b,c) in tris]
        return tris
    
    def flip_normal_orientation(self):

        tris = self.get_surface_tris()
        tris = [(a,c,b) for (a,b,c) in tris]
        self.set_surface_tris(tris)

    def export_mesh(self, name):

        name = name.replace('.vol', '')
        filename = name + '.vol'
        with open(filename, 'w') as f:
            self.print_text(f=f)

    def build_boundary_edges(self, tris):
        cnt = Counter()
        def ek(i,j):
            return (i,j) if i<j else (j,i)
        for a,b,c in tris:
            cnt[ek(a,b)] += 1
            cnt[ek(b,c)] += 1
            cnt[ek(c,a)] += 1
        return [e for e,n in cnt.items() if n==1]

    def build_boundary_loops(self, points, boundary_edges):

        adj = defaultdict(list)
        for u,v in boundary_edges:
            adj[u].append(v)
            adj[v].append(u)
        visited_v = set()
        loops = []
        for start in list(adj.keys()):
            if start in visited_v:
                continue
            comp = set(); dq = deque([start]); visited_v.add(start); comp.add(start)
            while dq:
                x = dq.popleft()
                for y in adj[x]:
                    if y not in visited_v:
                        visited_v.add(y); comp.add(y); dq.append(y)
            compE = [(u,v) for (u,v) in boundary_edges if u in comp and v in comp]
            if not compE: continue
            ladj = defaultdict(list)
            for u,v in compE:
                ladj[u].append(v); ladj[v].append(u)
            endpoints = [v for v in comp if len(ladj[v])==1]
            start_v = endpoints[0] if endpoints else next(iter(comp))
            order=[start_v]; used=set(); cur=start_v; prev=None
            while True:
                nxt=None
                for nb in ladj[cur]:
                    e=(min(cur,nb),max(cur,nb))
                    if e in used: continue
                    if nb!=prev: nxt=nb; break
                if nxt is None: break
                used.add((min(cur,nxt),max(cur,nxt)))
                order.append(nxt)
                prev,cur=cur,nxt
                if not endpoints and cur==start_v:
                    break
            if not endpoints and order[0]!=order[-1]:
                order.append(order[0])
            loops.append(order)
        # distances
        def P(i): return points[i-1]
        loop_edges_params=[]
        for order in loops:
            total=0.0
            segs=[]
            for i in range(len(order)-1):
                u,v=order[i],order[i+1]
                du=P(u); dv=P(v)
                d=math.dist(du,dv)
                segs.append(((u,v),d)); total+=d
            s=0.0; entries=[]
            for (u,v),d in segs:
                s0=0.0 if total==0 else s/total
                s+=d
                s1=1.0 if total==0 else s/total
                entries.append((u,v,s0,s1))
            loop_edges_params.append(entries)

        return loops, loop_edges_params
    
    def fix_dim2_boundary(self, override = False):

        if self.sections["edgesegmentsgi2"].dim != 0 and not override:
            raise Exception('BBoundary list is non-empty, use the flag override to add them')
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

        for loop_id,entries in enumerate(loop_edges_params, start=1):
            for (u,v,s0,s1) in entries:
                counter += 1
                self.sections["edgesegmentsgi2"].entries.append([str(loop_id), str(0), str(u), str(v), str(-1), str(-1), str(0), str(0), str(loop_id), str(s0), str(loop_id), str(s1)])
        self.sections["edgesegmentsgi2"].dim = counter
        
        for i in range(loop_id):
            self.sections["cd2names"].entries.append([str(int(i+1)), 'bboundary' + str(int(i+1))])
        self.sections["cd2names"].dim = loop_id

    def mark_cd_elements(self, mesh_to_mark):

        pnts_old_to_new = []
        for i, point_new in enumerate(self.sections['points'].entries, start = 1):
            found = False
            for j, point_old in enumerate(mesh_to_mark.sections['points'].entries, start = 1):
                if all(any(math.isclose(float(m), float(n)) for m in point_old) for n in point_new):
                    found = True
                    pnts_old_to_new.append(j)
            if not found:
                pnts_old_to_new.append(0)

        bnd_mark_id = max(self.sections['bcnames'].dim+1, 2)
        bnd_mark_name = 'boundary' + str(bnd_mark_id)

        bbnd_mark_id = max(self.sections['cd2names'].dim+1, 2)
        bbnd_mark_name = 'bboundary' + str(bbnd_mark_id)

        for i, surfel_new in enumerate(self.sections['surfaceelements'].entries, start = 1):
            for j, surfel_old in enumerate(mesh_to_mark.sections['surfaceelements'].entries, start = 1):

                set1 = (int(surfel_old[5]), int(surfel_old[6]), int(surfel_old[7]))
                set2 = (pnts_old_to_new[int(surfel_new[5])-1], pnts_old_to_new[int(surfel_new[6])-1], pnts_old_to_new[int(surfel_new[7])-1])
                if all(el in set2 for el in set1):
                    surfel_new[1] = str(bnd_mark_id)

        if self.sections['bcnames'].dim == 0:
            self.sections['bcnames'].dim = 2
            self.sections['bcnames'].entries.append(['1', 'default'])
            self.sections['bcnames'].entries.append(['2', 'boundary2'])
        else:
            self.sections['bcnames'].dim += 1
            self.sections['bcnames'].entries.append([str(bnd_mark_id),  bnd_mark_name])

        self.sections['edgesegmentsgi2'].entries = []
        for j, edge_el_old in enumerate(mesh_to_mark.sections['edgesegmentsgi2'].entries, start = 1):
            edge_el_new = [cmp for cmp in edge_el_old]
            edge_el_new[2] = pnts_old_to_new.index(int(edge_el_old[2]))+1
            edge_el_new[3] = pnts_old_to_new.index(int(edge_el_old[3]))+1
            self.sections['edgesegmentsgi2'].entries.append(edge_el_new)
        self.sections['edgesegmentsgi2'].dim = mesh_to_mark.sections['edgesegmentsgi2'].dim
        self.sections['cd2names'].dim = copy.deepcopy(mesh_to_mark.sections['cd2names'].dim)
        self.sections['cd2names'].entries = copy.deepcopy(mesh_to_mark.sections['cd2names'].entries)

def fill_mesh(old_mesh):

    new_mesh = ngen.meshing.Mesh()
    fd_outside = new_mesh.Add (ngen.meshing.FaceDescriptor(bc=1,domin=1,surfnr=1))

    pmap1 = { }
    m1 = old_mesh.ngmesh
    for e in m1.Elements2D():
        for v in e.vertices:
            if (v not in pmap1):
                pmap1[v] = new_mesh.Add (m1[v])

    for e in m1.Elements2D():
        new_mesh.Add (ngen.meshing.Element2D (fd_outside, [pmap1[v] for v in e.vertices]))

    new_mesh.GenerateVolumeMesh(maxh = 0.1)
    new_mesh = Mesh(new_mesh)

    return new_mesh