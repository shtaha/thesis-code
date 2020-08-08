import numpy as np
import pyomo.environ as pyo

from .standard import StandardDCOPF
from ..parameters import MultistepTopologyParameters


class MultistepTopologyDCOPF(StandardDCOPF):
    def __init__(
        self,
        name,
        grid,
        grid_backend=None,
        forecasts=None,
        params=MultistepTopologyParameters(),
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            grid=grid,
            grid_backend=grid_backend,
            params=params,
            verbose=verbose,
            **kwargs,
        )

        self.forecasts = forecasts

        # Optimal switching status
        self.x_gen = None
        self.x_load = None
        self.x_line_or_1 = None
        self.x_line_or_2 = None
        self.x_line_ex_1 = None
        self.x_line_ex_2 = None

        # Auxiliary
        self.x_line_status_switch = None
        self.x_substation_topology_switch = None

    """
        INDEXED SETS
    """

    def _build_indexed_sets(self):
        self._build_indexed_sets_standard()
        self._build_indexed_sets_substation()
        self._build_indexed_sets_time()

    def _build_indexed_sets_substation(self):
        self.model.sub_set = pyo.Set(
            initialize=self.sub.index, within=pyo.NonNegativeIntegers,
        )
        self.model.sub_bus_set = pyo.Set(
            initialize=[1, 2], within=pyo.NonNegativeIntegers,
        )

    def _build_indexed_sets_time(self):
        self.model.time_set = pyo.Set(
            initialize=np.arange(self.params.horizon), within=pyo.NonNegativeIntegers,
        )

    """
        PARAMETERS.
    """

    def _build_parameters_injections(self):
        init_value = (
            self.forecasts.load_p
            if self.forecasts
            else np.tile(self.load.p_pu, (self.params.horizon, 1))
        )
        self.model.load_p = pyo.Param(
            self.model.time_set,
            self.model.load_set,
            initialize=self._create_map_dual_ids_to_values(
                self.forecasts.time_steps, self.load.index, init_value
            ),
            within=pyo.Reals,
        )

        if self.params.lin_gen_penalty or self.params.quad_gen_penalty:
            init_value = (
                self.forecasts.prod_p
                if self.forecasts
                else np.tile(self.load.p_pu, (self.params.horizon, 1))
            )
            self.model.gen_p_ref = pyo.Param(
                self.model.time_set,
                self.model.gen_set,
                initialize=self._create_map_dual_ids_to_values(
                    self.forecasts.time_steps, self.gen.index, init_value
                ),
                within=pyo.Reals,
            )

    def _build_parameters_topology(self):
        self.model.sub_ids_to_bus_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.bus),
            within=self.model.bus_set * self.model.bus_set,
        )

        self.model.bus_ids_to_sub_bus_ids = pyo.Param(
            self.model.bus_set,
            initialize=self._create_map_ids_to_values(self.bus.index, self.bus.sub_bus),
        )

        self.model.bus_ids_to_sub_ids = pyo.Param(
            self.model.bus_set,
            initialize=self._create_map_ids_to_values(self.bus.index, self.bus["sub"]),
            within=self.model.sub_set,
        )

        if len(self.ext_grid.index):
            self.model.bus_ids_to_ext_grid_ids = pyo.Param(
                self.model.bus_set,
                initialize=self._create_map_ids_to_values(
                    self.bus.index, self.bus.ext_grid
                ),
                within=pyo.Any,
            )

        self.model.line_ids_to_sub_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index.values,
                self._dataframe_to_list_of_tuples(self.line[["sub_or", "sub_ex"]]),
            ),
            within=self.model.sub_set * self.model.sub_set,
        )

        # Line statuses
        self.model.line_status = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.status
            ),
            within=pyo.Boolean,
        )

        # Substation grid elements
        self.model.sub_ids_to_gen_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.gen),
            within=pyo.Any,
        )
        self.model.sub_ids_to_load_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.load),
            within=pyo.Any,
        )
        self.model.sub_ids_to_line_or_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.line_or),
            within=pyo.Any,
        )
        self.model.sub_ids_to_line_ex_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.line_ex),
            within=pyo.Any,
        )
        self.model.sub_n_elements = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(
                self.sub.index, self.sub.n_elements
            ),
            within=pyo.NonNegativeIntegers,
        )

        # Bus within a substation
        self.model.gen_ids_to_sub_bus_ids = pyo.Param(
            self.model.gen_set,
            initialize=self._create_map_ids_to_values(self.gen.index, self.gen.sub_bus),
            within=self.model.sub_bus_set,
        )
        self.model.load_ids_to_sub_bus_ids = pyo.Param(
            self.model.load_set,
            initialize=self._create_map_ids_to_values(
                self.load.index, self.load.sub_bus
            ),
            within=self.model.sub_bus_set,
        )
        self.model.line_or_ids_to_sub_bus_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.sub_bus_or
            ),
            within=self.model.sub_bus_set,
        )
        self.model.line_ex_ids_to_sub_bus_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.sub_bus_ex
            ),
            within=self.model.sub_bus_set,
        )

        if len(self.ext_grid.index):
            self.model.ext_grid_ids_to_sub_bus_ids = pyo.Param(
                self.model.ext_grid_set,
                initialize=self._create_map_ids_to_values(
                    self.ext_grid.index, self.ext_grid.sub_bus
                ),
                within=self.model.sub_bus_set,
            )

    """
        VARIABLES.
    """

    def _build_variables(self):
        pass

    #     self._build_variables_delta()  # Bus voltage angles with bounds
    #
    #     self._build_variables_generators()  # Generator productions with bounds
    #     if len(self.ext_grid.index):
    #         self._build_variables_ext_grids()  # External grid productions with bounds
    #
    #     self._build_variables_lines()  # Power line flows without bounds
    #
    #     # Indicator variables for bus configuration of power lines, generators, and loads
    #     self._build_variables_bus_configuration()
    #
    #     # Auxiliary variables indicating line status changes and substation topology reconfigurations
    #     self._build_variables_changes()
    #
    #     #
    #     # if self.lin_line_margins:
    #     #     self._build_variable_mu()
    #     #
    #     # if self.lin_gen_penalty:
    #     #     self._build_variable_mu_gen()
    #     #
    #     #
    #
    # def _build_variables_delta(self):
    #     # Bus voltage angle
    #     def _bounds_delta(model, t, bus_id):
    #         if bus_id == pyo.value(model.slack_bus_id):
    #             return 0.0, 0.0
    #         else:
    #             return -model.delta_max, model.delta_max
    #
    #     self.model.delta = pyo.Var(
    #         self.model.time_set,
    #         self.model.bus_set,
    #         domain=pyo.Reals,
    #         bounds=_bounds_delta,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.bus.index,
    #             np.zeros((self.params.horizon, len(self.bus.index))),
    #         ),
    #     )
    #
    # def _build_variables_generators(self):
    #     def _bounds_gen_p(model, t, gen_id):
    #         return model.gen_p_min[gen_id], model.gen_p_max[gen_id]
    #
    #     self.model.gen_p = pyo.Var(
    #         self.model.time_set,
    #         self.model.gen_set,
    #         domain=pyo.NonNegativeReals,
    #         bounds=_bounds_gen_p,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps, self.gen.index, self.forecasts.prod_p
    #         ),
    #     )
    #
    # def _build_variables_ext_grids(self):
    #     def _bounds_ext_grid_p(model, t, ext_grid_id):
    #         return model.ext_grid_p_min[ext_grid_id], model.ext_grid_p_max[ext_grid_id]
    #
    #     self.model.ext_grid_p = pyo.Var(
    #         self.model.time_set,
    #         self.model.ext_grid_set,
    #         domain=pyo.Reals,
    #         bounds=_bounds_ext_grid_p,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.ext_grid.index,
    #             np.zeros((self.params.horizon, len(self.ext_grid.index))),
    #         ),
    #     )
    #
    # def _build_variables_lines(self):
    #     self.model.line_flow = pyo.Var(
    #         self.model.time_set,
    #         self.model.line_set,
    #         domain=pyo.Reals,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.line.index,
    #             np.tile(self.line.p_pu, (self.params.horizon, 1)),
    #         ),
    #     )
    #
    # def _build_variables_bus_configuration(self):
    #     """
    #     Creates indicator variables corresponding to bus switching of each grid element over the whole horizon.
    #     Variables are initialized to a non-modified input grid.
    #     """
    #     # Power line bus switching
    #     init_conf = np.equal(
    #         self.bus.sub_bus.values[self.line.bus_or.values], 1
    #     ).astype(int)
    #     self.model.x_line_or_1 = pyo.Var(
    #         self.model.time_set,
    #         self.model.line_set,
    #         domain=pyo.Binary,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.line.index,
    #             np.tile(init_conf, (self.params.horizon, 1)),
    #         ),
    #     )
    #     init_conf = np.equal(
    #         self.bus.sub_bus.values[self.line.bus_or.values], 2
    #     ).astype(int)
    #     self.model.x_line_or_2 = pyo.Var(
    #         self.model.time_set,
    #         self.model.line_set,
    #         domain=pyo.Binary,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.line.index,
    #             np.tile(init_conf, (self.params.horizon, 1)),
    #         ),
    #     )
    #
    #     init_conf = np.equal(
    #         self.bus.sub_bus.values[self.line.bus_ex.values], 1
    #     ).astype(int)
    #     self.model.x_line_ex_1 = pyo.Var(
    #         self.model.time_set,
    #         self.model.line_set,
    #         domain=pyo.Binary,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.line.index,
    #             np.tile(init_conf, (self.params.horizon, 1)),
    #         ),
    #     )
    #     init_conf = np.equal(
    #         self.bus.sub_bus.values[self.line.bus_ex.values], 2
    #     ).astype(int)
    #     self.model.x_line_ex_2 = pyo.Var(
    #         self.model.time_set,
    #         self.model.line_set,
    #         domain=pyo.Binary,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.line.index,
    #             np.tile(init_conf, (self.params.horizon, 1)),
    #         ),
    #     )
    #
    #     # Generator bus switching
    #     init_conf = np.equal(self.bus.sub_bus.values[self.gen.bus.values], 2).astype(
    #         int
    #     )
    #     self.model.x_gen = pyo.Var(
    #         self.model.time_set,
    #         self.model.gen_set,
    #         domain=pyo.Binary,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.gen.index,
    #             np.tile(init_conf, (self.params.horizon, 1)),
    #         ),
    #     )
    #
    #     # Load switching
    #     init_conf = (
    #         np.equal(self.bus.sub_bus.values[self.load.bus.values], 2).astype(int),
    #     )
    #     self.model.x_load = pyo.Var(
    #         self.model.time_set,
    #         self.model.load_set,
    #         domain=pyo.Binary,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.load.index,
    #             np.tile(init_conf, (self.params.horizon, 1)),
    #         ),
    #     )
    #
    # def _build_variables_changes(self):
    #     self.model.x_line_status_switch = pyo.Var(
    #         self.model.time_set,
    #         self.model.line_set,
    #         domain=pyo.Binary,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.line.index,
    #             np.zeros((self.params.horizon, len(self.line.index))),
    #         ),
    #     )
    #
    #     self.model.x_substation_topology_switch = pyo.Var(
    #         self.model.time_set,
    #         self.model.sub_set,
    #         domain=pyo.Binary,
    #         initialize=self._create_map_dual_ids_to_values(
    #             self.forecasts.time_steps,
    #             self.sub.index,
    #             np.zeros((self.params.horizon, len(self.sub.index))),
    #         ),
    #     )
    #
    # def _build_variables_bus_requirements(self):
    #     if self.params.requirement_at_least_two:
    #         init_conf = np.greater(self.bus.n_elements, 1.0).astype(int)
    #         self.model.w_bus_activation = pyo.Var(
    #             self.model.time_set,
    #             self.model.bus_set,
    #             domain=pyo.Binary,
    #             initialize=self._create_map_dual_ids_to_values(
    #                 self.forecasts.time_steps,
    #                 self.bus.index,
    #                 np.tile(init_conf, (self.params.horizon, 1)),
    #             ),
    #         )
    #
    #     if self.params.requirement_balance:
    #         self.model.w_bus_balance = pyo.Var(
    #             self.model.time_set,
    #             self.model.bus_set,
    #             domain=pyo.Binary,
    #             initialize=self._create_map_dual_ids_to_values(
    #                 self.forecasts.time_steps,
    #                 self.bus.index,
    #                 [
    #                     np.greater(
    #                         len(self.bus.gen[bus_id]) + len(self.bus.load[bus_id]), 0
    #                     ).astype(int)
    #                     for bus_id in self.bus.index
    #                 ],
    #             ),
    #         )
    #
    """
        CONSTRAINTS.
    """

    def _build_constraints(self):
        pass

    #     self._build_constraint_line_flows()  # Power flow definition
    #     self._build_constraint_bus_balance()  # Bus power balance
    #
    #     self._build_constraint_line_or()
    #     self._build_constraint_line_ex()
    #     #
    #     # if not allow_onesided_disconnection:
    #     #     self._build_constraint_onesided_line_disconnection()
    #     #
    #     # if not allow_implicit_diconnection:
    #     #     self._build_constraint_implicit_line_disconnection()
    #     #
    #     # if not allow_onesided_reconnection:
    #     #     self._build_constraint_onesided_line_reconnection()
    #     #
    #     # # Constraints to eliminate symmetric topologies
    #     # if symmmetry:
    #     #     self._build_constraint_symmetry()
    #     #
    #     # if gen_load_bus_balance:
    #     #     self._build_constraint_gen_load_bus_balance()
    #     #
    #     # if switching_limits:
    #     #     self._build_constraint_line_status_switch()
    #     #     self._build_constraint_substation_topology_switch()
    #     #
    #     # if cooldown:
    #     #     self._build_constraint_cooldown()
    #     #
    #     # if unitary_action:
    #     #     self._build_constraint_unitary_action()
    #     #
    #     # if lin_line_margins:
    #     #     self._build_constraint_lin_line_margins()
    #     #
    #     # if lin_gen_penalty:
    #     #     self._build_constraint_lin_gen_penalty()
    #
    # def _build_constraint_line_flows(self):
    #     # Power flow equation with topology switching
    #     def _constraint_line_flow(model, t, line_id):
    #         sub_or, sub_ex = model.line_ids_to_sub_ids[line_id]
    #         bus_or_1, bus_or_2 = model.sub_ids_to_bus_ids[sub_or]
    #         bus_ex_1, bus_ex_2 = model.sub_ids_to_bus_ids[sub_ex]
    #
    #         return model.line_flow[t, line_id] == model.line_b[line_id] * (
    #             (
    #                 model.delta[t, bus_or_1] * model.x_line_or_1[t, line_id]
    #                 + model.delta[t, bus_or_2] * model.x_line_or_2[t, line_id]
    #             )
    #             - (
    #                 model.delta[t, bus_ex_1] * model.x_line_ex_1[t, line_id]
    #                 + model.delta[t, bus_ex_2] * model.x_line_ex_2[t, line_id]
    #             )
    #         )
    #
    #     self.model.constraint_line_flow = pyo.Constraint(
    #         self.model.time_set, self.model.line_set, rule=_constraint_line_flow
    #     )
    #
    # def _build_constraint_bus_balance(self):
    #     # Bus power balance constraints
    #     def _constraint_bus_balance(model, t, bus_id):
    #         sub_id = model.bus_ids_to_sub_ids[bus_id]
    #
    #         # Generator bus injections
    #         bus_gen_ids = model.sub_ids_to_gen_ids[sub_id]
    #         if len(bus_gen_ids):
    #             bus_gen_p = [
    #                 model.gen_p[t, gen_id] * (1 - model.x_gen[t, gen_id])
    #                 if model.bus_ids_to_sub_bus_ids[bus_id] == 1
    #                 else model.gen_p[t, gen_id] * model.x_gen[t, gen_id]
    #                 for gen_id in bus_gen_ids
    #             ]
    #             sum_gen_p = sum(bus_gen_p)
    #         else:
    #             sum_gen_p = 0.0
    #
    #         if len(self.ext_grid.index):
    #             bus_ext_grid_ids = model.bus_ids_to_ext_grid_ids[bus_id]
    #             bus_ext_grids_p = [
    #                 model.ext_grid_p[t, ext_grid_id] for ext_grid_id in bus_ext_grid_ids
    #             ]
    #             sum_gen_p = sum_gen_p + sum(bus_ext_grids_p)
    #
    #         # Load bus injections
    #         bus_load_ids = model.sub_ids_to_load_ids[sub_id]
    #         if len(bus_load_ids):
    #             bus_load_p = [
    #                 model.load_p[t, load_id] * (1 - model.x_load[t, load_id])
    #                 if model.bus_ids_to_sub_bus_ids[bus_id] == 1
    #                 else model.load_p[t, load_id] * model.x_load[t, load_id]
    #                 for load_id in bus_load_ids
    #             ]
    #             sum_load_p = sum(bus_load_p)
    #         else:
    #             sum_load_p = 0.0
    #
    #         # Power line flows
    #         flows_out = [
    #             model.line_flow[t, line_id] * model.x_line_or_1[t, line_id]
    #             if model.bus_ids_to_sub_bus_ids[bus_id] == 1
    #             else model.line_flow[t, line_id] * model.x_line_or_2[t, line_id]
    #             for line_id in model.line_set
    #             if sub_id == model.line_ids_to_sub_ids[line_id][0]
    #         ]
    #
    #         flows_in = [
    #             model.line_flow[t, line_id] * model.x_line_ex_1[t, line_id]
    #             if model.bus_ids_to_sub_bus_ids[bus_id] == 1
    #             else model.line_flow[t, line_id] * model.x_line_ex_2[t, line_id]
    #             for line_id in model.line_set
    #             if sub_id == model.line_ids_to_sub_ids[line_id][1]
    #         ]
    #
    #         if len(flows_in) == 0 and len(flows_out) == 0:
    #             return pyo.Constraint.Skip
    #
    #         return sum_gen_p - sum_load_p == sum(flows_out) - sum(flows_in)
    #
    #     self.model.constraint_bus_balance = pyo.Constraint(
    #         self.model.time_set, self.model.bus_set, rule=_constraint_bus_balance
    #     )
    #
    # def _build_constraint_line_or(self):
    #     def _constraint_line_or(model, t, line_id):
    #         return model.x_line_or_1[t, line_id] + model.x_line_or_2[t, line_id] <= 1
    #
    #     self.model.constraint_line_or = pyo.Constraint(
    #         self.model.time_set, self.model.line_set, rule=_constraint_line_or
    #     )
    #
    # def _build_constraint_line_ex(self):
    #     def _constraint_line_ex(model, t, line_id):
    #         return model.x_line_ex_1[t, line_id] + model.x_line_ex_2[t, line_id] <= 1
    #
    #     self.model.constraint_line_ex = pyo.Constraint(
    #         self.model.time_set, self.model.line_set, rule=_constraint_line_ex
    #     )
    #
    """
        OBJECTIVE
    """

    def _build_objective(self):
        pass

    # TODO:
    # Line power flows
    # def _bounds_flow_max(model, line_id):
    #     if model.line_status[line_id]:
    #         return -model.line_flow_max[line_id], model.line_flow_max[line_id]
    #     else:
    #         return 0.0, 0.0
