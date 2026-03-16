import logging
logger = logging.getLogger(__name__)

import os
import time
from ngsolve import *
import numpy as np
from contextlib import redirect_stdout
import meshio

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Sequence, Optional, Union

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel

class CosmosIOManager:

    def __init__(self, kwargs):

        self.params = kwargs
        self.root = None
        self.sample_rate = None
        self.samples = None
        self.output_step_data = None
        self.output_step_headers = ['iter', 'time', 'dt', 'step_elapsed', 'ale_elapsed']

    def initialize(self, model: "CosmosModel"):

        if 'root' in self.params.keys():
            self.root = os.path.join(self.params['root'], model.name)
            os.makedirs(self.root, exist_ok=True)
            if 'sample_rate' in self.params.keys():
                self.sample_rate = self.params['sample_rate']
            elif 'samples' in self.params.keys():
                self.samples = self.params['samples']
                self.N = 0
            else:
                raise Exception(f'Sample rate (or n. of samples) not set for output in folder {self.root}')

            output_model_data = os.path.join(self.root, 'model_data.txt')
            with open(output_model_data, "w") as f:
                self.print_model_data(model, file=f)

            self.output_step_data = os.path.join(self.root, 'output_step_data.txt')
            if 'output_callables' in self.params:
                if isinstance(self.params['output_callables'], dict):
                    for key in self.params['output_callables'].keys():
                        self.output_step_headers.append(key)
                else:
                    raise Exception('output_callables has to be a dictionary of callables')

            with open(self.output_step_data, "w") as f:
                print(*[x for x in self.output_step_headers], sep="\t", file=f)

            vol_output_vtk_folder = os.path.join(self.root, 'vol_pdes')
            vol_output_vtk_name = os.path.join(vol_output_vtk_folder, 'vol_pdes')
            os.makedirs(vol_output_vtk_folder, exist_ok=True)
            vol_vtk_gfu = []
            self.vol_vtk_names = []
            for c in model.compartments:
                if c.is_vol:
                    vol_vtk_gfu += c.vtk_gfu
                    self.vol_vtk_names += c.vtk_names
            for p in model.pdes:
                if p.is_vol and p.params['printing']:
                    vol_vtk_gfu += p.vtk_gfu
                    self.vol_vtk_names += p.vtk_names

            bnd_output_vtk_folder = os.path.join(self.root, 'bnd_pdes')
            self.bnd_output_vtk_name = os.path.join(bnd_output_vtk_folder, 'bnd_pdes')
            os.makedirs(bnd_output_vtk_folder, exist_ok=True)
            self.bnd_vtk_gfu = []
            self.bnd_vtk_names = []
            for c in model.compartments:
                if c.is_bnd:
                    self.bnd_vtk_gfu += c.vtk_gfu
                    self.bnd_vtk_names += c.vtk_names
            for p in model.pdes:
                if p.is_bnd and p.params['printing']:
                    self.bnd_vtk_gfu += p.vtk_gfu
                    self.bnd_vtk_names += p.vtk_names

            if model.is_vol:
                vol_vtk_gfu += model.ale.vtk_gfu
                self.vol_vtk_names += model.ale.vtk_names
            elif model.is_bnd:
                self.bnd_vtk_gfu += model.ale.vtk_gfu
                self.bnd_vtk_names += model.ale.vtk_names
                
            if model.dim == 2 and model.is_bnd: 
                pass
            else:
                self.vol_vtk = VTKOutput(model.parentmesh,
                                    coefs=vol_vtk_gfu,
                                    names =self.vol_vtk_names,
                                    filename= vol_output_vtk_name)
                self.bnd_vtk = VTKOutput(model.parentmesh,
                                    coefs=self.bnd_vtk_gfu,
                                    names =self.bnd_vtk_names,
                                    filename= self.bnd_output_vtk_name)
                
            self.num_saved = 0
            self.t_saved = []

    def save_step_data(self, model: "CosmosModel"):

        if self.root != None:
            # if model.time.iter % self.sample_rate == 0:

            dt_save = (model.time.t1-model.time.t0)/self.samples
            if (model.time.t.Get()-model.time.t0)//dt_save>=self.N:

                print('PRINTED!')

                with open(self.output_step_data, "a") as f:
                    self.print_step_data(model, file=f)

                if model.dim == 2 and model.is_bnd: 
                    npoints = round(len(model.ale.X.vec)/2)
                    coords = model.ale.X.vec.FV().NumPy()
                    points_xy = coords.reshape((npoints, 2), order = 'F')
                    pdata_s = {}
                    pdata_v = {}
                    for gfu_name, gfu in zip(self.bnd_vtk_names, self.bnd_vtk_gfu):
                        if gfu.space.type == 'h1ho':
                            pdata_s[gfu_name] = gfu.vec.FV().NumPy()
                        elif gfu.space.type == 'VectorH1':
                            pdata_v[gfu_name] = gfu.vec.FV().NumPy().reshape((npoints, 2), order = 'F')
                        else:
                            raise Exception('Unknown format for 2D BND output')
                    write_curve_meshio(
                        self.bnd_output_vtk_name + f"_step{round(model.time.iter/self.sample_rate):05d}.vtu",
                        points_xy=points_xy,
                        point_scalars=pdata_s,
                        point_vectors=pdata_v
                    )
                    
                else:
                    if self.vol_vtk_names:
                        self.vol_vtk.Do(time = model.time.t.Get(), vb = VOL)
                    if self.bnd_vtk_names:
                        if model.dim == 2:
                            self.bnd_vtk.Do(time = model.time.t.Get(), vb = VOL)
                        else:
                            self.bnd_vtk.Do(time = model.time.t.Get(), vb = BND)

                self.num_saved += 1
                self.t_saved.append(model.t.Get())

                if model.dim == 2 and model.is_bnd:
                    files = [f"bnd_pdes_step{i:05d}.vtu" for i in range(self.num_saved)]
                    # Optional: supply physical simulation times for each file
                    write_pvd(self.bnd_output_vtk_name + '.pvd', files, timesteps=self.t_saved)

                self.N += 1

    def finalize(self, model: "CosmosModel"):

        pass


    def print_model_data(self, model: "CosmosModel", file = None):

        if file is None:
            file = sys.stdout

        with redirect_stdout(file):
            print(f'The mesh has dimension: {model.dim}')
            print(f'The mesh geometry order is: {model.geo_order}')
            print('The mesh has the following codimension-0 domains:')
            print(f'{model.vol_ids}')
            print('The mesh has the following codimension-1 domains:')
            print(f'{model.bnd_ids}')
            print('The mesh has the following codimension-2 domains:')
            print(f'{model.bbnd_ids}')
            print('Mesh characteristics are:')
            print(f'\t - number of elements: {model.num_elem}')
            print(f'\t - number of vertices: {model.num_vertices}')
            print(f'\t - number of facets: {model.num_facets}')
            print(f'\t - number of faces: {model.num_faces}')
            print(f'\t - number of edges: {model.num_edges}')
            print()

            print(f'There are N.{len(model.compartments)} compartments\n')

            for i, comp in enumerate(model.compartments):
                print(f'Compartment N.{i}')
                comp.print_compartment_info()

    def print_step_data(self, model: "CosmosModel", file = None):

        if file is None:
            file = sys.stdout
            print(self.output_step_headers)

        if model.time.iter == 0: 
            dt = None
        else:
            dt = model.time.prev_dt[-2]

        numbers = [model.time.iter, model.time.t.Get(), 
                   dt, model.step.step_elasped_time,
                   model.ale.ale_elapsed_time]
        
        if 'output_callables' in self.params:
            if isinstance(self.params['output_callables'], dict):
                for value in self.params['output_callables'].values():
                    numbers.append(value())
            else:
                raise Exception('output_callables has to be a dictionary of callables')

        with redirect_stdout(file):
            print(*[x for x in numbers], sep="\t", file=file)

import numpy as np

import numpy as np
import meshio

def write_curve_vtp(filename, points_xy, closed=False, point_data=None, cell_data=None):
    pts2 = np.asarray(points_xy, float)
    N = pts2.shape[0]
    points = np.column_stack([pts2, np.zeros(N)])

    if closed:
        lines = np.column_stack([np.arange(N), np.roll(np.arange(N), -1)]).astype(np.int64)
    else:
        lines = np.column_stack([np.arange(N-1), np.arange(1, N)]).astype(np.int64) if N > 1 else np.empty((0,2), np.int64)

    point_data = point_data or {}
    cell_data  = cell_data  or {}

    # meshio wants cell_data as lists (one per cell block)
    cell_data_listed = {k: [np.asarray(v)] for k, v in cell_data.items()}

    mesh = meshio.Mesh(
        points=points,
        cells=[("line", lines)],
        point_data={k: np.asarray(v) for k, v in point_data.items()},
        cell_data=cell_data_listed,
    )
    meshio.write(filename, mesh)   # filename ".vtp" -> XML PolyData


def write_polyline_vtk(
    filename,
    points_xy,
    closed=False,
    point_data=None,   # dict: name -> (N,) or (N,2) or (N,3)
    cell_data=None,    # dict: name -> (M,) or (M,2) or (M,3)
):
    """
    Write a 2D polyline curve to legacy VTK (POLYDATA) for ParaView.

    points_xy: (N,2) array-like
    closed: if True, connect last->first
    point_data: per-point scalars/vectors
    cell_data: per-segment scalars/vectors (segments = lines cells)
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be shape (N,2)")
    n = pts.shape[0]

    # Build line connectivity as individual segments
    if closed:
        segs = np.column_stack([np.arange(n, dtype=int), np.roll(np.arange(n, dtype=int), -1)])
    else:
        if n < 2:
            segs = np.empty((0, 2), dtype=int)
        else:
            segs = np.column_stack([np.arange(n - 1, dtype=int), np.arange(1, n, dtype=int)])
    m = segs.shape[0]

    point_data = point_data or {}
    cell_data  = cell_data  or {}

    # Basic validation
    for k, v in point_data.items():
        arr = np.asarray(v)
        if arr.shape[0] != n:
            raise ValueError(f"point_data['{k}'] first dim must be N={n}, got {arr.shape}")
    for k, v in cell_data.items():
        arr = np.asarray(v)
        if arr.shape[0] != m:
            raise ValueError(f"cell_data['{k}'] first dim must be M={m}, got {arr.shape}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("2D curve polyline\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # POINTS (write as 3D with z=0)
        f.write(f"POINTS {n} float\n")
        for x, y in pts:
            f.write(f"{x:.9g} {y:.9g} 0\n")

        # LINES: each segment is a 2-vertex polyline cell
        # Format: LINES <numLines> <totalIndicesCount>
        # Each line contributes: "2 i j" -> 3 integers
        total = m * 3
        f.write(f"LINES {m} {total}\n")
        for i, j in segs:
            f.write(f"2 {int(i)} {int(j)}\n")

        # Helper writers
        def write_data_block(kind, data_dict, expected_len):
            if not data_dict:
                return
            f.write(f"{kind} {expected_len}\n")
            for name, values in data_dict.items():
                arr = np.asarray(values)
                if arr.ndim == 1:
                    # scalar
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for val in arr:
                        f.write(f"{float(val):.9g}\n")
                elif arr.ndim == 2 and arr.shape[1] in (2, 3):
                    # vector (pad 2D -> 3D)
                    f.write(f"VECTORS {name} float\n")
                    if arr.shape[1] == 2:
                        for vx, vy in arr:
                            f.write(f"{float(vx):.9g} {float(vy):.9g} 0\n")
                    else:
                        for vx, vy, vz in arr:
                            f.write(f"{float(vx):.9g} {float(vy):.9g} {float(vz):.9g}\n")
                else:
                    raise ValueError(
                        f"Data '{name}' must be (N,) scalar or (N,2)/(N,3) vector; got {arr.shape}"
                    )

        # Per-point functions
        write_data_block("POINT_DATA", point_data, n)

        # Per-segment functions
        write_data_block("CELL_DATA", cell_data, m)


def write_pvd(
    out_path: Union[str, Path],
    file_paths: Sequence[Union[str, Path]],
    timesteps: Optional[Sequence[float]] = None,
    byte_order: str = "LittleEndian",
    version: str = "0.1",
):
    """
    Write a .pvd collection file that ParaView can open as a time series.

    Parameters
    ----------
    out_path : str or Path
        Path to write the .pvd file (e.g. "curve.pvd").
    file_paths : sequence of str or Path
        Ordered list of filenames (relative or absolute) to include in the collection.
    timesteps : sequence of float, optional
        Same length as file_paths. If None, uses 0,1,2,...
    byte_order : "LittleEndian" or "BigEndian" (default "LittleEndian")
    version : xml version attribute (default "0.1")
    """
    file_paths = [Path(p) for p in file_paths]
    n = len(file_paths)
    if timesteps is None:
        timesteps = list(range(n))
    else:
        if len(timesteps) != n:
            raise ValueError("timesteps must have same length as file_paths")

    # Root element
    vtkfile = ET.Element(
        "VTKFile",
        attrib={"type": "Collection", "version": version, "byte_order": byte_order},
    )
    collection = ET.SubElement(vtkfile, "Collection")

    # Add DataSet entries
    for t, p in zip(timesteps, file_paths):
        # Use posix path (ParaView likes forward slashes); keep relative paths if provided
        file_attr = str(p)
        ET.SubElement(
            collection,
            "DataSet",
            attrib={
                "timestep": str(float(t)),
                "group": "",
                "part": "0",
                "file": file_attr,
            },
        )

    # Pretty-print and write to disk (with XML declaration)
    rough = ET.tostring(vtkfile, "utf-8")
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent="  ", encoding="utf-8")

    out_path = Path(out_path)
    out_path.write_bytes(pretty)
    # print(f"Wrote {out_path} with {n} dataset entries.")

def write_curve_meshio(
    filename,
    points_xy,                 # (N,2)
    closed=True,
    point_scalars=None,        # dict: name -> (N,)
    point_vectors=None,        # dict: name -> (N,2) or (N,3)
    cell_scalars=None,         # dict: name -> (M,)
    cell_vectors=None,         # dict: name -> (M,2) or (M,3)
):
    pts2 = np.asarray(points_xy, dtype=float)
    if pts2.ndim != 2 or pts2.shape[1] != 2:
        raise ValueError("points_xy must be shape (N,2)")
    N = pts2.shape[0]

    # meshio expects 3D points for many formats; store z=0
    points = np.column_stack([pts2, np.zeros(N)])

    # Build segment connectivity (M,2)
    if closed:
        lines = np.column_stack([np.arange(N), np.roll(np.arange(N), -1)]).astype(np.int64)
    else:
        if N < 2:
            lines = np.empty((0, 2), dtype=np.int64)
        else:
            lines = np.column_stack([np.arange(N - 1), np.arange(1, N)]).astype(np.int64)
    M = lines.shape[0]

    # Assemble point_data
    point_data = {}
    point_scalars = point_scalars or {}
    point_vectors = point_vectors or {}

    for name, arr in point_scalars.items():
        a = np.asarray(arr)
        if a.shape != (N,):
            raise ValueError(f"point scalar '{name}' must be (N,), got {a.shape}")
        point_data[name] = a

    for name, arr in point_vectors.items():
        a = np.asarray(arr)
        if a.shape[0] != N or a.ndim != 2 or a.shape[1] not in (2, 3):
            raise ValueError(f"point vector '{name}' must be (N,2) or (N,3), got {a.shape}")
        if a.shape[1] == 2:
            a = np.column_stack([a, np.zeros(N)])
        point_data[name] = a

    # Assemble cell_data (meshio requires list per cell-block)
    cell_data = {}
    cell_scalars = cell_scalars or {}
    cell_vectors = cell_vectors or {}

    for name, arr in cell_scalars.items():
        a = np.asarray(arr)
        if a.shape != (M,):
            raise ValueError(f"cell scalar '{name}' must be (M,), got {a.shape}")
        cell_data[name] = [a]  # list aligns with cells=[("line", lines)]

    for name, arr in cell_vectors.items():
        a = np.asarray(arr)
        if a.shape[0] != M or a.ndim != 2 or a.shape[1] not in (2, 3):
            raise ValueError(f"cell vector '{name}' must be (M,2) or (M,3), got {a.shape}")
        if a.shape[1] == 2:
            a = np.column_stack([a, np.zeros(M)])
        cell_data[name] = [a]

    mesh = meshio.Mesh(
        points=points,
        cells=[("line", lines)],
        point_data=point_data,
        cell_data=cell_data,
    )

    # Good ParaView choices:
    # - .vtu (Unstructured Grid) robust for data arrays
    # - .vtk legacy also works
    meshio.write(filename, mesh)

