# %%

import re
import argparse
from collections import defaultdict, deque
import math

SECTION_NAMES = {
    "points",
    "surfaceelements",
    "surfaceelementsuv",
    "edgesegments",
    "edgesegmentsgi2",
    "volumeelements",
    "tetrahedra",
    "hexahedra",
    "prisms",
    "pyramids",
    "pointelements",
    "cd2names",
}

VOL_INT = r"[-+]?\d+"
VOL_FLOAT = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"

def find_section(lines, name):
    name = name.lower()
    for i, ln in enumerate(lines):
        if ln.strip().lower() == name:
            return i
    return -1

def find_sections(lines):
    secs = {}
    for i, ln in enumerate(lines):
        key = ln.strip().lower()
        if key in SECTION_NAMES:
            secs[key] = i
    return secs

def parse_points(lines):
    header = "#          X             Y             Z"
    hidx = next((i for i,l in enumerate(lines) if l.strip()==header), -1)
    pidx = next((i for i,l in enumerate(lines) if l.strip().lower()=="points"), -1)
    if pidx < 0:
        raise RuntimeError("points section not found")
    n = None
    if pidx+1 < len(lines) and re.match(r"^\s*\d+\s*$", lines[pidx+1]):
        n = int(lines[pidx+1].strip())
        start = pidx+2
        pts = []
        for k in range(n):
            s = lines[start+k].strip()
            if not s: continue
            vals = re.findall(VOL_FLOAT, s)
            if len(vals) < 3:
                raise RuntimeError(f"Malformed point line: '{s}'")
            x,y,z = map(float, vals[:3])
            pts.append((x,y,z))
        return pts, pidx, pidx+1, start, start+n
    else:
        start = pidx+1
        pts = []
        j = start
        while j < len(lines) and lines[j].strip().lower() not in SECTION_NAMES:
            s = lines[j].strip()
            if s and not s.startswith("#"):
                vals = re.findall(VOL_FLOAT, s)
                if len(vals) >= 3:
                    pts.append(tuple(map(float, vals[:3])))
            j += 1
        return pts, pidx, None, start, j

def parse_surfaceelements(lines):
    sidx = find_section(lines, "surfaceelements")
    if sidx < 0:
        raise RuntimeError("surfaceelements not found")
    body_start = sidx+1
    count = None
    if body_start < len(lines) and re.match(r"^\s*\d+\s*$", lines[body_start]):
        count = int(lines[body_start].strip())
        body_start += 1
        body_end = body_start + count
    else:
        end_marker = "#  matnr      np      p1      p2      p3      p4"
        body_end = body_start
        while body_end < len(lines):
            s = lines[body_end].strip()
            if s == "" or s.startswith("#"):
                if s.startswith("#") and "matnr" in s:
                    break
                body_end += 1
                continue
            if s.lower() in SECTION_NAMES:
                break
            body_end += 1
    tris = []
    for k in range(body_start, body_end):
        s = lines[k].strip()
        if not s or s.startswith("#"): 
            continue
        ints = re.findall(VOL_INT, s)
        if len(ints) >= 3:
            a,b,c = map(int, ints[-3:])
            tris.append((a,b,c))
    return tris, sidx, body_start, body_end

def tri_edges_oriented(t):
    a,b,c = t
    return ((a,b),(b,c),(c,a))

def undirected(e):
    i,j = e
    return (i,j) if i<j else (j,i)

def reorient_tris_consistent(tris):
    edge_to_tris = defaultdict(list)
    for ti, t in enumerate(tris):
        for e in tri_edges_oriented(t):
            edge_to_tris[undirected(e)].append((ti, e))
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
            for e in tri_edges_oriented(tu):
                ue = undirected(e)
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
        oriented.append((t[0], t[2], t[1]) if flipped[i] else t)
    return oriented, flipped

def compute_area_normal(points, tris):
    import numpy as np
    nsum = np.zeros(3)
    for a,b,c in tris:
        pa = np.array(points[a-1]); pb=np.array(points[b-1]); pc=np.array(points[c-1])
        nsum += np.cross(pb-pa, pc-pa)
    return nsum

def pca_third_axis(points):
    import numpy as np
    P = np.array(points, dtype=float)
    c = P.mean(axis=0)
    X = P - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[2]

def globally_align_sign(points, tris):
    import numpy as np
    nsum = compute_area_normal(points, tris)
    axis = pca_third_axis(points)
    if np.dot(nsum, axis) < 0:
        tris = [(a,c,b) for (a,b,c) in tris]
    return tris

def rewrite_surfaceelements(lines, tris):
    sidx = find_section(lines, "surfaceelements")
    if sidx < 0:
        block = ["surfaceelements\n", f"{len(tris)}\n"] + [f"1 1 0 0 3 {a} {b} {c}\n" for (a,b,c) in tris]
        lines.extend(block)
        return lines
    next_idx = len(lines)
    for j in range(sidx+1, len(lines)):
        if lines[j].strip().lower() in SECTION_NAMES:
            next_idx = j
            break
    block = ["surfaceelements\n", f"{len(tris)}\n"] + [f"1 1 0 0 3 {a} {b} {c}\n" for (a,b,c) in tris]
    return lines[:sidx] + block + lines[next_idx:]

def read_surfaceelements(lines):
    tris, sidx, bstart, bend = parse_surfaceelements(lines)
    return tris

def build_boundary_edges(tris):
    from collections import Counter
    cnt = Counter()
    def ek(i,j):
        return (i,j) if i<j else (j,i)
    for a,b,c in tris:
        cnt[ek(a,b)] += 1
        cnt[ek(b,c)] += 1
        cnt[ek(c,a)] += 1
    return [e for e,n in cnt.items() if n==1]

def build_boundary_loops(points, boundary_edges):
    from collections import defaultdict, deque
    import math
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

def write_edgesegmentsgi2(lines, loops, loop_edges_params):
    edge_lines=[]
    for loop_id,entries in enumerate(loop_edges_params, start=1):
        for (u,v,s0,s1) in entries:
            edge_lines.append(f"{loop_id}       0       {u}       {v}       -1       -1        0        0        {loop_id} {s0:.16g}        {loop_id} {s1:.16g}")
    header="edgesegmentsgi2"
    hdr_idx=find_section(lines, header)
    block=[header+"\n", f"{len(edge_lines)}\n"]+[ln+"\n" for ln in edge_lines]+["\n"]
    if hdr_idx<0:
        lines.extend(block); return lines
    next_idx=len(lines)
    for j in range(hdr_idx+1,len(lines)):
        if lines[j].strip().lower() in SECTION_NAMES:
            next_idx=j; break
    return lines[:hdr_idx]+block+lines[next_idx:]

def find_section_block(lines, header):
    hdr_idx=find_section(lines, header)
    if hdr_idx<0: return None,None,None
    body_start=hdr_idx+1
    if body_start<len(lines) and re.match(r"^\s*\d+\s*$", lines[body_start]):
        n=int(lines[body_start].strip()); body_start+=1
        body_end=min(body_start+n, len(lines))
        return hdr_idx, body_start, body_end
    j=body_start
    while j<len(lines) and lines[j].strip().lower() not in SECTION_NAMES:
        j+=1
    return hdr_idx, body_start, j

def write_cd2names_after_pointelements(lines, loop_count, base_name="bboundary", header="cd2names"):
    hdr_idx, body_start, body_end = find_section_block(lines, "pointelements")
    insert_at = len(lines)
    if hdr_idx is not None:
        insert_at = body_end
        k = insert_at
        while k < len(lines) and lines[k].strip()=="":
            k += 1
        del lines[insert_at:k]
        lines[insert_at:insert_at] = ["\n","\n"]
        insert_at += 2
    else:
        while len(lines)>0 and lines[-1].strip()=="":
            lines.pop()
        lines.extend(["\n","\n"])
        insert_at = len(lines)
    block=[header+"\n", f"{loop_count}\n"]
    for i in range(1, loop_count+1):
        block.append(f"{i}\t{base_name}{i}\n")
    lines[insert_at:insert_at] = block
    return lines

def ensure_points_header(lines):
    coord_header = "#          X             Y             Z\n"
    pidx = find_section(lines, "points")
    if pidx>=0:
        if pidx==0 or lines[pidx-1].strip()!=coord_header.strip():
            lines.insert(pidx, coord_header)
    return lines

input = "../../../data/geometries/spine_whole_closed_fixed_fine.vol"
output = "../../../data/geometries/dummy.vol"

with open(input, "r", encoding="utf-8", errors="ignore") as f:
    lines=f.readlines()

points, p_hdr_idx, p_cnt_idx, p_start, p_end = parse_points(lines)
tris_raw, sidx, sbegin, send = parse_surfaceelements(lines)

tris_cons, flipped = reorient_tris_consistent(tris_raw)
tris_cons = globally_align_sign(points, tris_cons)
flip = True
if flip: 
    tris_cons = [(a,c,b) for (a,b,c) in tris_cons]

lines = rewrite_surfaceelements(lines, tris_cons)

# boundary edges and loops
boundary_edges = build_boundary_edges(tris_cons)
loops, loop_edges_params = build_boundary_loops(points, boundary_edges)

lines = write_edgesegmentsgi2(lines, loops, loop_edges_params)
lines = ensure_points_header(lines)
lines = write_cd2names_after_pointelements(lines, loop_count=len(loops), base_name="bboundary", header="cd2names")

outp = output or input
with open(outp, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Triangles: {len(tris_raw)} -> oriented: {len(tris_cons)}. Loops: {len(loops)}. Output: {outp}")

