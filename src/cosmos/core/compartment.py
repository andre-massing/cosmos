import logging
logger = logging.getLogger(__name__)

from ngsolve import *
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel

class CosmosCompartment:
    """Represents a computational sub-domain (compartment) within a CosmosModel.

    A compartment wraps either a volume domain—identified by a material name and
    its bounding surface—or a surface domain—identified by a boundary name and
    its co-dimension-2 boundary. It stores a list of PDE models assigned to it
    and an optional ALE field.
    """

    def __init__(self, name = '', model:"CosmosModel" = None, **kwargs):

        self.name = name
        self.model = model
        self.dim_emd = model.dim
        self.parentmesh = model.parentmesh
        self.pdes = []
        self.ale = None

        if {'material', 'boundary'} <= kwargs.keys():

            if all(x in self.model.vol_ids for x in kwargs['material'].split('|')):
                self.domain_id = kwargs['material']
                self.domain = self.model.parentmesh.Materials(kwargs['material'])
            else:
                raise ValueError(f'Material {kwargs["material"]} for compartment {self.name} not present in given Model')
            if all(x in self.model.bnd_ids for x in kwargs['boundary'].split('|')):
                self.boundary_id = kwargs['boundary']
                self.boundary = self.model.parentmesh.Boundaries(kwargs['boundary'])
            else:
                raise ValueError(f'Boundary {kwargs["boundary"]} for compartment {self.name} not present in given Model')
            self.is_bnd = False
            self.is_vol = True
            self.dim = self.model.dim
            

        elif {'boundary', 'bboundary'} <= kwargs.keys():
            if all(x in self.model.bnd_ids for x in kwargs['boundary'].split('|')):
                self.domain_id = kwargs['boundary']
                self.domain = self.model.parentmesh.Boundaries(kwargs['boundary'])
            else:
                raise ValueError(f'Boundary {kwargs["boundary"]} for compartment {self.name} not present in given Model')
            if all(x in self.model.bbnd_ids for x in kwargs['bboundary'].split('|')) or kwargs['bboundary']=='':
                self.boundary_id = kwargs['bboundary']
                self.boundary = self.model.parentmesh.BBoundaries(kwargs['bboundary'])
            else:
                raise ValueError(f'BBoundary {kwargs["bboundary"]} for compartment {self.name} not present in given Model')
            self.is_bnd = True
            self.is_vol = False
            self.dim = self.model.dim-1

            if {'clamped_bbnd'} <= kwargs.keys():
                self.clamped_bbnd = kwargs['clamped_bbnd']
            else:
                self.clamped_bbnd = ''
            if {'navier_bbnd'} <= kwargs.keys():
                self.navier_bbnd = kwargs['navier_bbnd']
            else:
                self.navier_bbnd = ''
        else:
            raise ValueError(f'Compartment {self.name} has to be initialized with the couple: \n {{material, boundary}} or {{boundary, bboundary}}')
        
        gfu = GridFunction(H1(model.parentmesh, order = model.geo_order))
        gfu.Set(1, definedon = self.domain)
        self.vtk_gfu = [gfu]
        self.vtk_names = [self.name]
        
    def print_compartment_info(self):

        print(f'The compartment name is: {self.name}')
        print('with:')
        print(f'\t - Dimension: {self.dim}')
        print(f'\t - Domain name: {self.domain_id}')
        print(f'\t - Boundary name:  {self.boundary_id} \n')
    