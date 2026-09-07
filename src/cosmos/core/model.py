"""``CosmosModel``: the single object every Cosmos script constructs.

Everything else in ``cosmos.core`` exists to be owned by a ``CosmosModel``:
it builds one ``CosmosTimeManager``, one ``CosmosStepManager``, one
``CosmosALEManager``, and one ``CosmosIOManager`` in its own constructor
(see below) and never expects a second model to share them. PDE models
(``cosmos.pde``) and ALE fields, by contrast, are created *through* this
model (``create_pde``/``create_ale``) and attached to a
``CosmosCompartment`` (``create_compartment``) rather than constructed
directly by user code -- that indirection is what lets ``create_pde``
validate a PDE's ``is_bnd``/``is_vol`` against its compartment before
anything is assembled.
"""

import logging

logger = logging.getLogger(__name__)

from collections import Counter
from ngsolve import *

from cosmos.core.time_manager import CosmosTimeManager
from cosmos.core.step_manager import CosmosStepManager
from cosmos.core.ale_manager import CosmosALEManager, CosmosBndALEField, CosmosVolALEField
from cosmos.core.io_manager import CosmosIOManager
from cosmos.core.compartment import CosmosCompartment

from typing import Any, Generator, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos.pde.base import BasePDEModel
    from cosmos.core.ale_manager import CosmosBndALEField, CosmosVolALEField


class CosmosModel:
    """Top-level orchestrator for a time-dependent PDE simulation on an NGSolve mesh.

    Collects the parent mesh together with compartments, PDE models, ALE fields,
    a time manager, a step manager, and an I/O manager. The simulation is driven
    step-by-step via the generator returned by ``__call__`` (use ``yield`` to
    interleave custom logic between steps) or can be run to completion with
    :meth:`run`.
    """

    def __init__(self, name: str, parentmesh: Mesh, **kwargs: Any) -> None:

        self.name = name
        self.parentmesh = parentmesh
        if kwargs:
            self.params = kwargs
        else:
            self.params = {}

        allowed_keys = {
            "t0",
            "t1",
            "dt",
            "t",
            "coupling_type",
            "adaptive_timestep",
            "redistribute",
            "root",
            "samples",
            "volume_ALE",
            "surface_ALE",
            "output_callables",
        }

        for key in self.params.keys():
            if key not in allowed_keys:
                raise ValueError(
                    f"Parameter {key} is not a valid parameter for CosmosModel. Allowed parameters are: {allowed_keys}"
                )

        # mesh-related properties
        self.dim = self.parentmesh.dim
        self.geo_order = self.parentmesh.GetCurveOrder()
        self.num_elem = self.parentmesh.ne
        self.num_faces = self.parentmesh.nface
        self.num_facets = self.parentmesh.nfacet
        self.num_edges = self.parentmesh.nedge
        self.num_vertices = self.parentmesh.nv
        self.is_bnd = self.num_elem == 0
        self.is_vol = self.num_elem != 0
        self.vol_ids = set(Counter(self.parentmesh.GetMaterials()).keys())
        self.bnd_ids = set(Counter(self.parentmesh.GetBoundaries()).keys())
        self.bbnd_ids = set(Counter(self.parentmesh.GetBBoundaries()).keys())

        # model components
        self.compartments: List[CosmosCompartment] = []
        self.pdes: List[BasePDEModel] = []
        # These three lists are populated by create_pde's `ale_type` argument
        # and read back by CosmosStepManager.solve_step: pdes_init are solved
        # once at t0 only, pdes_pre before the ALE mesh update, pdes_post
        # after it -- this ordering is what lets a PDE see either the old or
        # the newly-moved geometry within the same time step.
        self.pdes_init: List[BasePDEModel] = []
        self.pdes_pre: List[BasePDEModel] = []
        self.pdes_post: List[BasePDEModel] = []
        self.ales = []
        # Every manager below is owned exclusively by this model and stores
        # a back-reference to it (self.model), so none of their methods need
        # `model` passed in again once constructed.
        self.time = CosmosTimeManager(self.params)
        self.step = CosmosStepManager(self, self.params)
        self.ale = CosmosALEManager(self, self.params)
        self.io = CosmosIOManager(self, self.params)

        self.dt = self.time.dt
        self.t = self.time.t

    def initialize(self) -> None:
        self.time.initialize()
        self.ale.initialize()
        self.step.initialize()
        self.io.initialize()

        logger.info(
            f"[Cosmos] '{self.name}': starting simulation — dim={self.dim}, "
            f"{len(self.compartments)} compartment(s), {len(self.pdes)} PDE(s), "
            f"t in [{self.time.t0}, {self.time.t1}], dt0={self.time.dt0}, "
            f"coupling='{self.step.coupling_type}'"
        )

    def __call__(self) -> Generator:
        return self._generator()

    def _generator(self) -> Generator:
        # The whole simulation lifecycle in one place: initialize every
        # manager and PDE, yield once per completed step (including the very
        # first, pre-solve state) so callers can drive it with a for-loop
        # (via __call__) or exhaust it in one call (via run()), and finalize
        # once the requested time range is covered. step.solve_step() is
        # where CosmosStepManager actually dispatches to the PDEs' four-phase
        # lifecycle and the ALE mesh update; ale.finalize() then commits that
        # step's mesh motion before the next one begins.
        with TaskManager():
            self.initialize()
            self.io.save_step_data()

            yield

            while self.t.Get() <= self.time.t1:
                self.step.solve_step()
                self.time.next()
                self.ale.finalize()

                self.io.save_step_data()

                logger.info(
                    f"[Cosmos] '{self.name}' step {self.time.iter}: "
                    f"t={self.t.Get():.4g}, dt={self.dt.Get():.4g}"
                )

                yield

            logger.info(
                f"[Cosmos] '{self.name}' finished after {self.time.iter} step(s) "
                f"at t={self.t.Get():.4g}"
            )
            self.io.finalize()

    def run(self) -> None:
        for _ in self():
            pass

    def set_params(self, force: bool = False, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key in self.params.keys() and not force:
                raise ValueError(
                    f"Parameter {key} already set in the model, use flag force = True to force the behavior"
                )
            elif key in self.params.keys() and force:
                self.params[key] = value
            else:
                self.params[key] = value

    def create_compartment(self, name: str, **kwargs: Any) -> CosmosCompartment:
        if any(name == x.name for x in self.compartments):
            raise ValueError(
                f"Unique names must be assigned to compartments, name {name} already used"
            )
        compartment = CosmosCompartment(name=name, model=self, **kwargs)
        self.compartments.append(compartment)
        return compartment

    def create_pde(
        self,
        name: str,
        pde_model: Type[BasePDEModel],
        compartment: CosmosCompartment,
        ale_type: int,
        **kwargs: Any,
    ) -> BasePDEModel:

        # pde_model is any concrete cosmos.pde subclass of BasePDEModel; its
        # is_bnd/is_vol class attributes (set once per PDE type, not per
        # instance -- see pde/base.py) are cross-checked here against the
        # compartment's own is_bnd/is_vol so a surface-only model can never
        # silently end up assembled on a volume compartment or vice versa.
        pde = pde_model(name, self, compartment, **kwargs)
        if pde.is_bnd and compartment.is_vol:
            raise ValueError(
                f"Boundary PDE {pde_model} is trying to be imposed on a non-boundary domain {compartment.name} or the opposite"
            )
        elif pde.is_vol and compartment.is_bnd:
            raise ValueError(
                f"Volume PDE {pde_model} is trying to be imposed on a boundary domain {compartment.name} or the opposite"
            )
        else:
            compartment.pdes.append(pde)
            # ale_type slots this PDE into one of the three lists CosmosStepManager
            # walks in solve_step(): -1 = init-only (e.g. a one-shot distance
            # function), 0 = solved before the ALE mesh update, 1 = solved after.
            if ale_type == -1:
                self.pdes_init.append(pde)
            elif ale_type == 0:
                self.pdes_pre.append(pde)
            elif ale_type == 1:
                self.pdes_post.append(pde)
            else:
                raise ValueError("Parameter pde-type must be in the values {-1 ,0, 1}")
            self.pdes.append(pde)
        return pde

    def create_ale(
        self,
        name: str,
        compartment: CosmosCompartment,
        **kwargs: Any,
    ):

        if any(
            x == y
            for z in self.ales
            for x in compartment.domain_id.split("|")
            for y in z.compartment.boundary_id.split("|")
        ) or any(
            x == y
            for z in self.ales
            for x in compartment.domain_id.split("|")
            for y in z.compartment.domain_id.split("|")
        ):
            raise ValueError(
                f"ALE motion for domain {compartment.domain_id} (or part of it) has already been set"
            )
        else:
            if any(name == x.name for x in self.ales):
                raise ValueError(
                    f"Name {name} for ALE motion has already been used. Names must be unique"
                )
            else:
                # Which ALE field class gets built follows the same
                # is_bnd/is_vol split used throughout: a surface compartment
                # gets a CosmosBndALEField (normal + tangential velocity),
                # a volume compartment a CosmosVolALEField (a single domain
                # velocity) -- both live in core/ale_manager.py and report
                # back to the one CosmosALEManager the model already owns.
                if compartment.is_bnd:
                    ale = CosmosBndALEField(name, self, compartment)
                else:
                    ale = CosmosVolALEField(name, self, compartment)
                self.ales.append(ale)
                compartment.ale = ale
        return ale

    def print_model_data(self) -> None:
        self.io.print_model_data()

    def print_step_data(self) -> None:
        self.io.print_step_data()
