import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import InputField, OutputField, Field
from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel
from cosmos.core.utils import MandBP
from ngsolve.webgui import Draw
import numbers
import time

class DistanceBoundaryModel(BasePDEModel):

    def __init__(self, name:str = 'DistanceBoundaryModel', model:CosmosModel = None, compartment:CosmosCompartment = None, **kwargs):
        
        super().__init__(name=name, model = model, compartment=compartment)
        
        self.is_bnd = True
        self.is_vol = False
        self.name = name
        self.model = model
        self.compartment = compartment

        if 'zero_bbnd' not in kwargs.keys():
            raise Exception('The argument -zero_bbnd- has to be passed at initialization for DistanceBoundaryModel')
        else:
            self.params["zero_bbnd"] = kwargs['zero_bbnd']

        self.params["fes_order"] = 2
        self.params["subdivision"] = 0

        self.fes = H1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain,
                                dirichlet_bbnd = self.model.parentmesh.BBoundaries(self.params['zero_bbnd']))
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.sol = [self.gfu]
        self.output_fields["distance"] = OutputField(self.gfu, "distance", BND)

    def Initialize(self):

        if self.params['printing']:
            output_vtk_folder = os.path.join(self.model.io.root, self.name)
            output_vtk_name = os.path.join(output_vtk_folder, self.name)
            os.makedirs(output_vtk_folder, exist_ok=True)
            self.vtk = VTKOutput(self.model.parentmesh,
                                coefs=[self.output_fields["distance"]._coef],
                                names =['distance'],
                                filename= output_vtk_name, 
                                subdivision = self.params['subdivision'])

    def PreProcess(self):
        
        self.gfu.vec.data[:] = 0

    def Solve(self):

        dt = specialcf.mesh_size**2

        fes1 = H1(self.model.parentmesh, order = self.params["fes_order"], 
                                definedon = self.compartment.domain,
                                dirichlet_bbnd = self.model.parentmesh.BBoundaries(self.params['zero_bbnd']))
        
        u1, v1 = fes1.TnT()
        A1 = BilinearForm(fes1)
        A1 += (u1*v1 + dt*grad(u1).Trace()*grad(v1).Trace())*ds(deformation = self.model.dX)
        A1.Assemble()
        gfu1 = GridFunction(fes1)
        gfu1.Set(1, definedon =  self.model.parentmesh.BBoundaries(self.params['zero_bbnd']))
        res1 = -1*A1.mat*gfu1.vec
        gfu1.vec.data += A1.mat.Inverse(freedofs = fes1.FreeDofs())*res1 

        fes2 = VectorH1(self.model.parentmesh, order = self.params["fes_order"]-1, 
                                definedon = self.compartment.domain)
        u2, v2 = fes2.TnT()
        A2 = BilinearForm(fes2)
        A2 += u2*v2*ds(deformation = self.model.dX)
        A2.Assemble()
        F2 = LinearForm(fes2)
        F2 += -1*Normalize(grad(gfu1).Trace())*v2*ds(deformation = self.model.dX)
        F2.Assemble()
        gfu2 = GridFunction(fes2)
        gfu2.vec.data = A2.mat.Inverse(freedofs = fes2.FreeDofs())*F2.vec

        u3, v3 = self.fes.TnT()
        A3 = BilinearForm(self.fes)
        A3 += grad(u3).Trace()*grad(v3).Trace()*ds(deformation = self.model.dX)
        A3.Assemble()
        F3 = LinearForm(self.fes)
        F3 += -1*Trace(Grad(gfu2).Trace())*v3*ds(deformation = self.model.dX)
        F3.Assemble()
        self.gfu.vec.data += A3.mat.Inverse(freedofs = self.fes.FreeDofs())*F3.vec

    def PostProcess(self):

        pass