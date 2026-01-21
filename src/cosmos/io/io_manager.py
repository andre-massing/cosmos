import logging
logger = logging.getLogger(__name__)

import os
import time
from ngsolve import *
import numpy as np
from contextlib import redirect_stdout

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel

class CosmosIOManager:

    def __init__(self, kwargs):

        self.params = kwargs
        self.root = None
        self.sample_rate = None
        self.output_step_data = None
        self.output_step_headers = ['iter', 'time', 'dt', 'step_elapsed', 'ale_elapsed']

    def initialize(self, model: "CosmosModel"):

        if 'root' in self.params.keys():
            self.root = os.path.join(self.params['root'], model.name)
            os.makedirs(self.root, exist_ok=True)
            if 'sample_rate' in self.params.keys():
                self.sample_rate = self.params['sample_rate']
            else:
                raise Exception(f'Sample rate not set for output in folder {self.root}')

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

            self.dX_store = GridFunction(model.ale.dX.space)

    def save_step_data(self, model: "CosmosModel"):

        if self.root != None:
            if model.time.iter % self.sample_rate == 0:
                with open(self.output_step_data, "a") as f:
                    self.print_step_data(model, file=f)

                for pde in model.pdes:
                    if pde.params['printing']:
                        if pde.is_bnd:
                            if model.dim == 2:
                                pde.vtk.Do(time = model.time.t.Get(), vb = VOL)
                            else:
                                pde.vtk.Do(time = model.time.t.Get(), vb = BND)
                        else:
                            pde.vtk.Do(time = model.time.t.Get(), vb = VOL)

                if model.is_bnd:
                    model.ale.vtk.Do(time = model.time.t.Get(), vb = BND)
                else:
                    model.ale.vtk.Do(time = model.time.t.Get(), vb = VOL)

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

        numbers = [model.time.iter, model.time.t.Get(), 
                   model.time.dt.Get(), model.step.step_elasped_time,
                   model.ale.ale_elapsed_time]
        
        if 'output_callables' in self.params:
            if isinstance(self.params['output_callables'], dict):
                for value in self.params['output_callables'].values():
                    numbers.append(value())
            else:
                raise Exception('output_callables has to be a dictionary of callables')

        with redirect_stdout(file):
            print(*[x for x in numbers], sep="\t", file=file)