# Arbitrary Lagrangian-Eulerian (ALE) mesh-motion coupling. ALEField wraps a single
# prescribed-displacement contribution (on a boundary or volume region) and,
# optionally, a tangential mesh-redistribution ("wind") correction that reparametrizes
# surface nodes tangentially to preserve element quality without altering the shape.
# ALEModel aggregates all registered ALEFields, turns the prescribed boundary
# displacement into a smooth interior mesh motion via a discrete harmonic extension,
# and commits the resulting mesh deformation at the end of each time step.

import logging
logger = logging.getLogger(__name__)

import numpy as np
from ngsolve import *
from cosmos.config.parameters import get_config
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import OutputField, Field
from cosmos.core.solver import Solver
from ngsolve.webgui import Draw
import numbers

class ALEField(Field):
    """A single prescribed-displacement contribution to the mesh motion, defined
    on one boundary or volume region.

    Beyond wrapping the coefficient (via the base ``Field`` class), this optionally
    solves an auxiliary mixed problem to compute a tangential "wind" correction that
    redistributes surface mesh nodes along the manifold (independent of the normal
    motion that changes the shape), keeping element quality high over long
    simulations. Two redistribution schemes are supported, selected via
    ``redistribute_type``:
      - "DuanLi": solves the mixed harmonic-map-style problem on the FIXED reference
        configuration.
      - "MDR" (time-weighted variant): solves the same style of problem on the last
        COMMITTED deformation, scaled by 1/dt.
    """

    def __init__(self, solver:Solver, coef: CoefficientFunction, domain:str, VorB=None,  clamped_bnd:str='', redistribute = False, redistribute_type = 'DuanLi'):
        """
        Args:
            solver: The Solver owning the mesh this field displaces.
            coef: The prescribed displacement CoefficientFunction (or callable).
            domain: Boundary or material region name(s) this field applies to.
            VorB: ``BND`` or ``VOL``, selecting whether ``domain`` names a boundary
                or a material region.
            clamped_bnd: Optional bboundary (edge) region name(s) where the
                redistribution wind field is clamped to zero (Dirichlet on bboundary).
            redistribute: If True, additionally solve for a tangential
                redistribution "wind" field each ``update()``.
            redistribute_type: "DuanLi" or "MDR" (see class docstring).
        """

        super().__init__(coef)

        self._solver = solver
        self.dims = self().dims
        self.name = domain
        self.domain = domain
        self.clamped_bnd = clamped_bnd
        self.redistribute = redistribute
        self.redistribute_type = redistribute_type
        self.VorB = VorB

        # Displacement space for this field: vector H1 on the named boundary/volume
        # region, optionally clamped (Dirichlet) on a bboundary (edge) region.
        if self.VorB == BND:
            V = VectorH1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(), 
                         definedon = self._solver.ngsmesh.Boundaries(domain),
                         dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(clamped_bnd))
            self.ale_displ = GridFunction(V)
        elif self.VorB == VOL:
            V = VectorH1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(), 
                         definedon = self._solver.ngsmesh.Materials(domain))
            self.ale_displ = GridFunction(V)

        if self.redistribute:

            if self.redistribute_type == 'DuanLi':

                # Reference (fixed) configuration on which the reparametrization
                # problem is posed, and the "material" (actually-deformed)
                # configuration used to express the tangential constraint.
                deformation0 = GridFunction(self._solver.mesh.curr_deformation.space)
                self.mat_def = GridFunction(self._solver.mesh.curr_deformation.space)

                self.ale_displ = GridFunction(V)
                V2 = H1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(),
                        definedon = self._solver.ngsmesh.Boundaries(domain),
                         dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(clamped_bnd))

                # Mixed space: w is the tangential "wind" reparametrization field,
                # kappa is the Lagrange-multiplier-like scalar enforcing that w stays
                # tangential to the (deformed) surface.
                fes_pp = V*V2
                self.A_pp = BilinearForm(fes_pp, symmetric = True)
                self.F_pp = LinearForm(fes_pp)
                self.duanli = GridFunction(fes_pp)
                self.wind = self.duanli.components[0]

                (w, kappa), (eta, mu) = fes_pp.TnT()
                ns = specialcf.normal(self._solver.ngsmesh.dim)
                Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(ns, ns)

                # Harmonic-map-style bilinear form: surface-gradient stiffness for w
                # on the fixed reference configuration, saddle-point coupling with
                # the multiplier kappa enforcing tangentiality on the deformed mesh.
                self.A_pp += (InnerProduct(grad(w).Trace(), grad(eta).Trace()))*ds(deformation = deformation0)
                self.A_pp += (-1*InnerProduct(eta*ns, kappa))*ds(deformation=self.mat_def)
                self.A_pp += (-1*InnerProduct(w*ns, mu))*ds(deformation=self.mat_def)
                self.A_pp.Assemble()
                self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

                # Right-hand side driving w towards the tangential projection of the
                # identity map minus the current material deformation (i.e. towards
                # an equi-distributed reparametrization of the reference mesh).
                self.F_pp += (-1*InnerProduct(Ps, grad(eta).Trace()))*ds(deformation = deformation0)
                self.F_pp += (-1*InnerProduct(grad(self.mat_def).Trace(), grad(eta).Trace()))*ds(deformation = deformation0)
                self.F_pp += (InnerProduct(grad(deformation0).Trace(), grad(eta).Trace()))*ds(deformation = deformation0)

            elif self.redistribute_type == 'MDR':

                # Mesh-Distribution/time-weighted variant: same mixed structure as
                # DuanLi, but posed on the last COMMITTED deformation and scaled by
                # 1/dt, so `wind` here is directly a velocity-like quantity.
                self.mat_def = GridFunction(self._solver.mesh.curr_deformation.space)
                deform = self._solver.mesh.prev_deformation[-1]

                self.ale_displ = GridFunction(V)
                V2 = H1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(),
                        definedon = self._solver.ngsmesh.Boundaries(domain),
                        dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(clamped_bnd))

                fes_pp = V*V2
                self.A_pp = BilinearForm(fes_pp, symmetric = True)
                self.F_pp = LinearForm(fes_pp)
                self.duanli = GridFunction(fes_pp)
                self.wind = self.duanli.components[0]

                (w, kappa), (eta, mu) = fes_pp.TnT()
                ns = specialcf.normal(self._solver.ngsmesh.dim)
                Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(ns, ns)

                # Same saddle-point structure as DuanLi, but the stiffness term is
                # scaled by 1/dt (time-weighted) and posed on the last committed
                # deformation rather than the fixed reference configuration.
                self.A_pp += (InnerProduct(grad(w).Trace()/self._solver.time.dt, grad(eta).Trace()))*ds(deformation = deform)
                self.A_pp += (-1*InnerProduct(eta*ns, kappa))*ds(deformation = self.mat_def)
                self.A_pp += (-1*InnerProduct(w*ns, mu))*ds(deformation = self.mat_def)
                self.A_pp.Assemble()
                self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

                # RHS driven by the (scaled) surface gradient of the prescribed
                # displacement itself, rather than by a fixed-reference identity map.
                self.F_pp += (-1*InnerProduct(grad(self.ale_displ).Trace()/self._solver.time.dt, grad(eta).Trace()))*ds(deformation = deform)

            else:
                raise Exception('Mesh redistribution_type not recognized, use DuanLi or MDR as options')

    @property
    def cf(self):
        """The evaluated CoefficientFunction currently backing this field."""
        return self._eval()

    @cf.setter
    def cf(self, new_coef):
        """Replace the underlying prescribed-displacement coefficient (number,
        CoefficientFunction, GridFunction, or callable)."""
        if isinstance(new_coef, GridFunction) or isinstance(new_coef, CoefficientFunction):
            self._coef = new_coef
        elif isinstance(new_coef, numbers.Number):
            self._coef = CF(new_coef)
        elif callable(new_coef):
            self._coef = new_coef
        else:
            raise Exception("Unsupported type for Field " +  self.name)

    def update(self):
        """Refresh the prescribed displacement GridFunction from the current
        coefficient and, if redistribution is enabled, (re-)solve the auxiliary
        mixed problem and add the resulting tangential "wind" correction."""

        if self.VorB == BND:
            self.ale_displ.Set(self(), definedon = self._solver.ngsmesh.Boundaries(self.domain), dual = True)
        elif self.VorB == VOL:
            self.ale_displ.Set(self(), definedon = self.domain, dual = True)

        if self.redistribute:
            # The material (actually deformed) configuration is the last committed
            # mesh state plus this field's own prescribed displacement.
            self.mat_def.Set(self.ale_displ + self._solver.mesh.prev_deformation[-1], dual=True, definedon=self._solver.ngsmesh.Boundaries(self.domain))
            self.A_pp.Assemble()
            self.invA_pp.Update()
            self.F_pp.Assemble()
            self.duanli.vec.data = self.invA_pp*self.F_pp.vec
            # Add the tangential wind correction on top of the normal/prescribed
            # displacement, so ale_displ now carries both contributions.
            self.ale_displ.vec.data += self.wind.vec.data


class ALEModel(BasePDEModel):
    """Aggregates all registered ALEField contributions and computes the actual
    mesh deformation increment applied at each time step.

    Boundary and volume prescribed displacements are mutually exclusive per model
    instance (see `set_bnd_displacement`/`set_vol_displacement`). On a bulk mesh,
    the summed boundary displacement is extended into the mesh interior via a
    discrete harmonic (Laplace) extension (`self.A`/`self.invA`, assembled once in
    `__init__`), so the interior elements move smoothly with the prescribed
    boundary motion. `Solve()` produces a TENTATIVE new deformation
    (`solver.mesh.curr_deformation`); `PostProcess()` is where that tentative
    deformation is actually COMMITTED as the mesh's new accepted state.
    """

    def __init__(self, solver:Solver, model_order:int, name:str ='ALEModel'):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            name: Unique name for this model instance within the solver.
        """

        super().__init__()

        self.name = name
        self._solver = solver
        self.model_order = model_order

        solver._attach_model(self, self.model_order)
        self.domain = self._solver.mesh.domain        
        self.gfu = GridFunction(self._solver.mesh.curr_deformation.space)
        self.aux_gfu = GridFunction(self._solver.mesh.curr_deformation.space)
        self.tot_wind = GridFunction(self._solver.mesh.curr_deformation.space)
        self.aux_tot_wind = GridFunction(self._solver.mesh.curr_deformation.space)

        if self._solver.ngsmesh.ne != 0:
            # Bulk mesh: the prescribed boundary displacement needs to be extended
            # into the volume. Build (once) the discrete harmonic-extension
            # operator: a vector Laplace problem with the extension's own Dirichlet
            # value plugged in as a residual correction at Solve()-time (see below).
            self.VorB = VOL
            self.output_fields["displacement"] = OutputField(self.gfu, "displacement", VOL)
            vol_space = VectorH1(self._solver.ngsmesh, order=self._solver.ngsmesh.GetCurveOrder(),
                                definedon = self._solver.ngsmesh.Materials('.*'), dirichlet = self._solver.ngsmesh.Boundaries('.*'))
            u, v = vol_space.TnT()
            self.A = BilinearForm(vol_space, symmetric = True)

            self.A += InnerProduct(grad(u), grad(v))*dx
            self.A.Assemble()
            self.invA = self.A.mat.Inverse(freedofs = vol_space.FreeDofs())
        else:
            # Pure surface mesh: there is no interior to extend into, the
            # displacement is simply the sum of the registered boundary fields.
            self.VorB = BND
            self.output_fields["displacement"] = OutputField(self.gfu, "displacement", BND)

        self.bnd_fields = {}
        self.bnd_is_prescribed = False
        self.vol_fields = {}
        self.vol_is_prescribed = False

    def PreProcess(self):
        """No state to snapshot before Solve() for this model."""

        pass

    def Solve(self):
        """Sum the (possibly redistributed) displacement contributions from all
        registered fields, extend a prescribed boundary displacement harmonically
        into the mesh interior if this is a bulk mesh, and set
        `solver.mesh.curr_deformation` to the previous committed deformation plus
        the new increment. This deformation is TENTATIVE until `PostProcess()`
        commits it."""

        self.gfu.vec.data[:] = 0
        self.tot_wind.vec.data[:] = 0

        if self.bnd_is_prescribed:

            redistribute = False

            for field in self.bnd_fields.values():
                field.update()
                # Scatter this field's local boundary displacement into the
                # model-wide GridFunction (defined over the whole boundary/mesh),
                # then accumulate.
                self.aux_gfu.Set(field.ale_displ, definedon = self._solver.ngsmesh.Boundaries(field.domain), dual = True)
                self.gfu.vec.data += self.aux_gfu.vec.data

                if field.redistribute:
                    # Accumulate the tangential redistribution wind separately (note
                    # the sign flip: the wind here is subtracted, i.e. tot_wind ends
                    # up storing the correction needed to counteract the
                    # redistribution already folded into ale_displ).
                    self.aux_tot_wind.Set(field.wind, definedon = self._solver.ngsmesh.Boundaries(field.domain), dual = True)
                    self.tot_wind.vec.data += -1*self.aux_tot_wind.vec.data
                    redistribute = True

            if self._solver.ngsmesh.ne != 0:
                # Discrete harmonic extension: solve A*gfu = -A*gfu_bnd for the
                # interior correction (self.gfu currently holds only the boundary
                # values with zero interior; freedofs excludes the boundary so this
                # only updates interior dofs), then add it to the boundary data
                # already stored in self.gfu, yielding a smooth extension.
                self.A.Assemble()
                self.invA.Update()
                res = -1*self.A.mat*self.gfu.vec
                self.gfu.vec.data += self.invA*res

                if redistribute:
                    # Same harmonic-extension trick applied to the accumulated
                    # tangential wind field.
                    res = -1*self.A.mat*self.tot_wind.vec
                    self.tot_wind.vec.data += self.invA*res

        else:
            # Volume-prescribed case: no interior extension needed, just sum the
            # per-region volume displacement fields directly.
            for field in self.vol_fields.values():
                field.update()
                self.aux_gfu.Set(field.ale_def, definedon = field.domain)
                self.gfu.vec.data += self.aux_gfu.vec.data

        # New tentative deformation = last committed deformation + this step's
        # increment; not yet applied to the mesh (see advance_mesh/reset_mesh).
        self._solver.mesh.curr_deformation.vec.data = self._solver.mesh.prev_deformation[-1].vec.data + self.gfu.vec.data

    def PostProcess(self):
        """Commit the tentative deformation computed in Solve() as the mesh's new
        accepted state; this is the point in the simulation where the mesh
        actually moves (see SolverMesh.update_state)."""

        self._solver.mesh.update_state()

    def set_bnd_displacement(self, coef, domain, clamped_bnd='', redistribute=False, redistribute_type = 'DuanLi'):
        """Register a prescribed-displacement field on a boundary region.

        Mutually exclusive with `set_vol_displacement` on this model instance.

        Args:
            coef: The prescribed displacement CoefficientFunction (or callable).
            domain: Boundary region name(s) (matching mesh boundary markers).
            clamped_bnd: Optional bboundary (edge) region name(s) where the
                redistribution wind field is clamped to zero.
            redistribute: If True, enable tangential mesh redistribution for this
                field (see ALEField).
            redistribute_type: "DuanLi" or "MDR" (see ALEField).
        """

        if self.vol_is_prescribed:
            raise Exception('Cannot prescribe boundary displacement if volume displacement is already prescribed')
        else:
            self.bnd_is_prescribed = True

        if not all(word in self._solver.mesh.bnd_markers for word in domain.split('|')):
            raise Exception('Boundary not present in list of boundary for the mesh')
        if clamped_bnd:
            if not all(word in self._solver.mesh.bbnd_markers for word in clamped_bnd.split('|')):
                raise Exception('Clamped boundary not present in list of boundaries for the mesh')
        
        self.bnd_fields[domain] = ALEField(self._solver, coef, domain, VorB=BND, clamped_bnd=clamped_bnd,
                                           redistribute=redistribute, redistribute_type = redistribute_type)

    def set_vol_displacement(self, coef, domain):
        """Register a prescribed-displacement field on a volume (material) region.

        Mutually exclusive with `set_bnd_displacement` on this model instance.

        Args:
            coef: The prescribed displacement CoefficientFunction (or callable).
            domain: Material region name(s) (matching mesh material markers).
        """

        if self.bnd_is_prescribed:
            raise Exception('Cannot prescribe volume displacement if boundary displacement is already prescribed')
        else:
            self.vol_is_prescribed = True
        
        if not all(word in self._solver.mesh.vol_markers for word in domain.split('|')) and self._solver.ngsmesh.ne == 0:
            raise Exception('The mesh is a surface mesh, use set_bnd_displacement instead')
        if not all(word in self._solver.mesh.vol_markers for word in domain.split('|')) and not self._solver.ngsmesh.ne == 0:
            raise Exception('Material not present in list of materials for the mesh')
        
        self.vol_fields[domain] = ALEField(self._solver, coef, domain, VorB=VOL)

    @property
    def displacement(self):
        """The raw mesh displacement increment computed for this step."""
        return self.gfu

    @property
    def ale_vel(self):
        """ALE grid velocity: displacement increment divided by dt."""
        return self.gfu/self._solver.time.dt

    @property
    def mat_vel(self):
        """Material point velocity, including the tangential redistribution
        contribution: (tot_wind + displacement) / dt."""
        return (self.tot_wind+self.gfu)/self._solver.time.dt

    @property
    def wind(self):
        """Tangential redistribution velocity: tot_wind / dt."""
        return self.tot_wind/self._solver.time.dt