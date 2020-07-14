import numpy as np
import pyomo.environ as pyo

from .standard import StandardDCOPF
from ...data_utils import parse_gurobi_log


class TopologyOptimizationDCOPF(StandardDCOPF):
    def __init__(
        self,
        name,
        grid,
        grid_backend,
        n_line_status_switch=1,
        n_substation_topology_switch=1,
        solver_name="gurobi",
        verbose=False,
        **kwargs,
    ):
        super().__init__(name, grid, grid_backend, solver_name, verbose, **kwargs)
        self.n_line_status_switch = n_line_status_switch
        self.n_substation_topology_switch = n_substation_topology_switch

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

    def build_model(
        self,
        line_disconnection=True,
        symmetry=True,
        switching_limits=True,
        cooldown=True,
        gen_cost=False,
        line_margin=False,
        min_rho=True,
    ):
        # Model
        self.model = pyo.ConcreteModel(f"{self.name}")

        # Indexed sets
        self._build_indexed_sets()  # Indexing over buses, lines, generators, and loads

        # Substation set
        self.model.sub_set = pyo.Set(
            initialize=self.sub.index, within=pyo.NonNegativeIntegers,
        )
        self.model.sub_bus_set = pyo.Set(
            initialize=[1, 2], within=pyo.NonNegativeIntegers,
        )

        # Parameters
        self._build_parameters()

        # Variables
        self._build_variables(min_rho=min_rho)

        # Constraints
        self._build_constraints(
            line_disconnection=line_disconnection,
            symmmetry=symmetry,
            switching_limits=switching_limits,
            cooldown=cooldown,
            min_rho=min_rho,
        )

        # Objective
        self._build_objective(
            gen_cost=gen_cost, line_margin=line_margin, min_rho=min_rho
        )  # Objective to be optimized.

    """
        PARAMETERS.
    """

    def _build_parameters(self):
        self._build_parameters_delta()  # Bus voltage angle bounds and reference node
        self._build_parameters_generators()  # Bounds on generator production
        self._build_parameters_lines()  # Power line thermal limit and susceptance
        self._build_parameters_objective()  # Objective parameters

        if len(self.ext_grid.index):
            self._build_parameters_ext_grids()  # External grid power limits

        self._build_parameters_topology()  # Topology of generators and power lines
        self._build_parameters_loads()  # Bus load injections

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

        self.model.line_ids_to_sub_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index.values,
                self._dataframe_to_list_of_tuples(self.line[["sub_or", "sub_ex"]]),
            ),
            within=self.model.sub_set * self.model.sub_set,
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
            self.model.bus_ids_to_ext_grid_ids = pyo.Param(
                self.model.bus_set,
                initialize=self._create_map_ids_to_values(
                    self.bus.index, self.bus.ext_grid
                ),
                within=pyo.Any,
            )
            self.model.ext_grid_ids_to_sub_bus_ids = pyo.Param(
                self.model.ext_grid_set,
                initialize=self._create_map_ids_to_values(
                    self.ext_grid.index, self.ext_grid.sub_bus
                ),
                within=self.model.sub_bus_set,
            )

    def _build_parameters_loads(self):
        self.model.load_p = pyo.Param(
            self.model.load_set,
            initialize=self._create_map_ids_to_values(self.load.index, self.load.p_pu),
            within=pyo.Reals,
        )

    """
        VARIABLES.
    """

    def _build_variables(self, min_rho=True):
        self._build_variables_standard_generators()  # Generator productions and bounds
        self._build_variables_standard_delta()  # Bus voltage angles and bounds
        self._build_variables_standard_line()  # Power line flows

        if len(self.ext_grid.index):
            self._build_variables_standard_ext_grids()

        if min_rho:
            self._build_variable_min_rho()

        # Power line OR bus switching
        self.model.x_line_or_1 = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.equal(self.bus.sub_bus.values[self.line.bus_or.values], 1).astype(
                    int
                ),
            ),
        )
        self.model.x_line_or_2 = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.equal(self.bus.sub_bus.values[self.line.bus_or.values], 2).astype(
                    int
                ),
            ),
        )

        # Power line EX bus switching
        self.model.x_line_ex_1 = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.equal(self.bus.sub_bus.values[self.line.bus_ex.values], 1).astype(
                    int
                ),
            ),
        )
        self.model.x_line_ex_2 = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.equal(self.bus.sub_bus.values[self.line.bus_ex.values], 2).astype(
                    int
                ),
            ),
        )

        # Generator switching
        self.model.x_gen = pyo.Var(
            self.model.gen_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.gen.index,
                np.equal(self.bus.sub_bus.values[self.gen.bus.values], 2),
            ),
        )

        # Load switching
        self.model.x_load = pyo.Var(
            self.model.load_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.load.index,
                np.equal(self.bus.sub_bus.values[self.load.bus.values], 2),
            ),
        )

        # Auxiliary variables for counting the number of line status changes and substation topology reconfigurations
        self.model.x_line_status_switch = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index, np.zeros_like(self.line.index),
            ),
        )

        self.model.x_substation_topology_switch = pyo.Var(
            self.model.sub_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.sub.index, np.zeros_like(self.sub.index),
            ),
        )

    def _build_variable_min_rho(self):
        self.model.min_rho = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0.0, 1.0),
            initialize=np.max(np.abs(self.line.p_pu) / self.line.max_p_pu),
        )

    """
        CONSTRAINTS.
    """

    def _build_constraints(
        self,
        line_disconnection=True,
        symmmetry=True,
        switching_limits=True,
        cooldown=True,
        min_rho=True,
    ):
        self._build_constraint_line_flows()  # Power flow definition
        self._build_constraint_bus_balance()  # Bus power balance
        self._build_constraint_line_or()
        self._build_constraint_line_ex()

        if line_disconnection:
            self._build_constraint_line_disconnection()

        # Constraints to eliminate symmetric topologies
        if symmmetry:
            self._build_constraint_symmetry()

        if switching_limits:
            self._build_constraint_line_status_switch()
            self._build_constraint_substation_topology_switch()

        if cooldown:
            pass

        if min_rho:
            self._build_constraint_min_rho()

    def _build_constraint_line_flows(self):
        # Power flow equation with topology switching
        def _constraint_line_flow(model, line_id):
            sub_or, sub_ex = model.line_ids_to_sub_ids[line_id]
            bus_or_1, bus_or_2 = model.sub_ids_to_bus_ids[sub_or]
            bus_ex_1, bus_ex_2 = model.sub_ids_to_bus_ids[sub_ex]

            return model.line_flow[line_id] == model.line_b[line_id] * (
                (
                    model.delta[bus_or_1] * model.x_line_or_1[line_id]
                    + model.delta[bus_or_2] * model.x_line_or_2[line_id]
                )
                - (
                    model.delta[bus_ex_1] * model.x_line_ex_1[line_id]
                    + model.delta[bus_ex_2] * model.x_line_ex_2[line_id]
                )
            )

        self.model.constraint_line_flow = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_flow
        )

    def _build_constraint_bus_balance(self):
        # Bus power balance constraints
        def _constraint_bus_balance(model, bus_id):
            sub_id = model.bus_ids_to_sub_ids[bus_id]

            # Generator bus injections
            bus_gen_ids = model.sub_ids_to_gen_ids[sub_id]
            if len(bus_gen_ids):
                bus_gen_p = [
                    model.gen_p[gen_id] * (1 - model.x_gen[gen_id])
                    if model.bus_ids_to_sub_bus_ids[bus_id] == 1
                    else model.gen_p[gen_id] * model.x_gen[gen_id]
                    for gen_id in bus_gen_ids
                ]
                sum_gen_p = sum(bus_gen_p)
            else:
                sum_gen_p = 0.0

            if len(self.ext_grid.index):
                bus_ext_grid_ids = model.bus_ids_to_ext_grid_ids[bus_id]
                bus_ext_grids_p = [
                    model.ext_grid_p[ext_grid_id] for ext_grid_id in bus_ext_grid_ids
                ]
                sum_gen_p = sum_gen_p + sum(bus_ext_grids_p)

            # Load bus injections
            bus_load_ids = model.sub_ids_to_load_ids[sub_id]
            if len(bus_load_ids):
                bus_load_p = [
                    model.load_p[load_id] * (1 - model.x_load[load_id])
                    if model.bus_ids_to_sub_bus_ids[bus_id] == 1
                    else model.load_p[load_id] * model.x_load[load_id]
                    for load_id in bus_load_ids
                ]
                sum_load_p = sum(bus_load_p)
            else:
                sum_load_p = 0.0

            # Power line flows
            flows_out = [
                model.line_flow[line_id] * model.x_line_or_1[line_id]
                if model.bus_ids_to_sub_bus_ids[bus_id] == 1
                else model.line_flow[line_id] * model.x_line_or_2[line_id]
                for line_id in model.line_set
                if sub_id == model.line_ids_to_sub_ids[line_id][0]
            ]

            flows_in = [
                model.line_flow[line_id] * model.x_line_ex_1[line_id]
                if model.bus_ids_to_sub_bus_ids[bus_id] == 1
                else model.line_flow[line_id] * model.x_line_ex_2[line_id]
                for line_id in model.line_set
                if sub_id == model.line_ids_to_sub_ids[line_id][1]
            ]

            if len(flows_in) == 0 and len(flows_out) == 0:
                return pyo.Constraint.Skip

            return sum_gen_p - sum_load_p == sum(flows_out) - sum(flows_in)

        self.model.constraint_bus_balance = pyo.Constraint(
            self.model.bus_set, rule=_constraint_bus_balance
        )

    def _build_constraint_line_or(self):
        def _constraint_line_or(model, line_id):
            return model.x_line_or_1[line_id] + model.x_line_or_2[line_id] <= 1

        self.model.constraint_line_or = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_or
        )

    def _build_constraint_line_ex(self):
        def _constraint_line_ex(model, line_id):
            return model.x_line_ex_1[line_id] + model.x_line_ex_2[line_id] <= 1

        self.model.constraint_line_ex = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_ex
        )

    def _build_constraint_line_disconnection(self):
        def _constraint_line_disconnection(model, line_id):
            return (
                model.x_line_or_1[line_id] + model.x_line_or_2[line_id]
                == model.x_line_ex_1[line_id] + model.x_line_ex_2[line_id]
            )

        self.model.constraint_line_disconnection = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_disconnection
        )

    def _build_constraint_symmetry(self):
        for sub_id in self.grid.fixed_elements.index:
            line_or = self.grid.fixed_elements.line_or[sub_id]
            line_ex = self.grid.fixed_elements.line_ex[sub_id]

            if len(line_or):
                line_id = line_or[0]
                self.model.x_line_or_2[line_id].value = 0
                self.model.x_line_or_2[line_id].fixed = True
            if len(line_ex):
                line_id = line_ex[0]
                self.model.x_line_ex_2[line_id].value = 0
                self.model.x_line_ex_2[line_id].fixed = True

    def _build_constraint_line_status_switch(self):
        def _constraint_line_status_switch(model, line_id):
            if model.line_status[line_id]:
                x_line_status = 1 - model.x_line_status_switch[line_id]
            else:
                x_line_status = model.x_line_status_switch[line_id]

            return (
                model.x_line_or_1[line_id]
                + model.x_line_or_2[line_id]
                + model.x_line_ex_1[line_id]
                + model.x_line_ex_2[line_id]
                <= 2 * x_line_status
            )

        def _constraint_max_line_status_switch(model):
            return (
                sum([model.x_line_status_switch[line_id] for line_id in model.line_set])
                <= self.n_line_status_switch
            )

        # Auxiliary constraint for checking line status switch
        self.model.constraint_line_status_switch = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_status_switch
        )

        # Limit the number of line status switches
        self.model.constraint_max_line_status_switch = pyo.Constraint(
            rule=_constraint_max_line_status_switch
        )

    def _build_constraint_substation_topology_switch(self):
        def _constraint_substation_topology_switch(model, sub_id):
            sub_gen_ids = model.sub_ids_to_gen_ids[sub_id]
            sub_load_ids = model.sub_ids_to_load_ids[sub_id]
            sub_line_or_ids = model.sub_ids_to_line_or_ids[sub_id]
            sub_line_ex_ids = model.sub_ids_to_line_ex_ids[sub_id]

            x_sub_gen_switch = [
                model.x_gen[gen_id]
                if model.gen_ids_to_sub_bus_ids[gen_id] == 1
                else 1 - model.x_gen[gen_id]
                for gen_id in sub_gen_ids
            ]
            x_sub_load_switch = [
                model.x_load[load_id]
                if model.load_ids_to_sub_bus_ids[load_id] == 1
                else 1 - model.x_load[load_id]
                for load_id in sub_load_ids
            ]

            x_sub_line_or_switch = []
            for line_id in sub_line_or_ids:
                if model.line_status[line_id]:
                    if model.line_or_ids_to_sub_bus_ids[line_id] == 1:
                        x_sub_line_or_switch.append(model.x_line_or_2[line_id])
                    elif model.line_or_ids_to_sub_bus_ids[line_id] == 2:
                        x_sub_line_or_switch.append(model.x_line_or_1[line_id])
                    else:
                        raise ValueError("No such substation bus.")
                else:
                    # If line is reconnected, then skip
                    pass

            x_sub_line_ex_switch = []
            for line_id in sub_line_ex_ids:
                if model.line_status[line_id]:
                    if model.line_ex_ids_to_sub_bus_ids[line_id] == 1:
                        x_sub_line_ex_switch.append(model.x_line_ex_2[line_id])
                    elif model.line_ex_ids_to_sub_bus_ids[line_id] == 2:
                        x_sub_line_ex_switch.append(model.x_line_ex_1[line_id])
                    else:
                        raise ValueError("No such substation bus.")
                else:
                    # If line is reconnected, then skip
                    pass

            return (
                sum(x_sub_gen_switch)
                + sum(x_sub_load_switch)
                + sum(x_sub_line_or_switch)
                + sum(x_sub_line_ex_switch)
                <= model.sub_n_elements[sub_id]
                * model.x_substation_topology_switch[sub_id]
            )

        def _constraint_max_substation_topology_switch(model):
            return (
                sum(
                    [
                        model.x_substation_topology_switch[sub_id]
                        for sub_id in model.sub_set
                    ]
                )
                <= self.n_substation_topology_switch
            )

        # Auxiliary constraint for checking substation topology reconfigurations
        self.model.constraint_substation_topology_switch = pyo.Constraint(
            self.model.sub_set, rule=_constraint_substation_topology_switch
        )

        # Limit the number of substation topology reconfigurations
        self.model.constraint_max_substation_topology_switch = pyo.Constraint(
            rule=_constraint_max_substation_topology_switch
        )

    def _build_constraint_min_rho(self):
        def _constraint_min_rho_upper(model, line_id):
            return (
                model.line_flow[line_id] <= model.line_flow_max[line_id] * model.min_rho
            )

        def _constraint_min_rho_lower(model, line_id):
            return (
                -model.line_flow_max[line_id] * model.min_rho
                <= model.line_flow[line_id]
            )

        self.model.constraint_min_rho_upper = pyo.Constraint(
            self.model.line_set, rule=_constraint_min_rho_upper
        )

        self.model.constraint_min_rho_lower = pyo.Constraint(
            self.model.line_set, rule=_constraint_min_rho_lower
        )

    """
        OBJECTIVE.
    """

    def _build_objective(self, gen_cost=False, line_margin=False, min_rho=True):
        # Minimize generator costs
        def _objective_gen_p(model):
            return sum(
                [
                    model.gen_p[gen_id] * model.gen_costs[gen_id]
                    for gen_id in model.gen_set
                ]
            )

        # Maximize line margins
        def _objective_line_margin(model):
            return sum(
                [
                    model.line_flow[line_id] ** 2 / model.line_flow_max[line_id] ** 2
                    for line_id in model.line_set
                ]
            )

        # Maximize line margins
        def _objective_min_rho(model):
            return model.min_rho

        def _objective(model):
            if min_rho:
                return _objective_min_rho(model)
            else:
                if gen_cost and line_margin:
                    return _objective_gen_p(model) + _objective_line_margin(model)
                elif line_margin:
                    return _objective_line_margin(model)
                else:
                    return _objective_gen_p(model)

        self.model.objective = pyo.Objective(rule=_objective, sense=pyo.minimize)

    """
        SOLVE FUNCTIONS.
    """

    def solve(self, verbose=False, tol=1e-9, time_limit=20):
        self._solve(verbose=verbose, tol=tol, time_limit=20)

        # Parse Gurobi log for additional information
        gap = parse_gurobi_log(self.solver._log)["gap"]
        if gap < 1e-6:
            gap = 1e-6

        # Save standard DC-OPF variable results
        self._solve_save()

        # Save line status variable
        self.x_gen = self._round_solution(self._access_pyomo_variable(self.model.x_gen))
        self.x_load = self._round_solution(
            self._access_pyomo_variable(self.model.x_load)
        )
        self.x_line_or_1 = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_or_1)
        )
        self.x_line_or_2 = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_or_2)
        )
        self.x_line_ex_1 = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_ex_1)
        )
        self.x_line_ex_2 = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_ex_2)
        )

        self.x_line_status_switch = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_status_switch)
        )
        self.x_substation_topology_switch = self._round_solution(
            self._access_pyomo_variable(self.model.x_substation_topology_switch)
        )

        if verbose:
            self.model.display()

        result = {
            "res_cost": self.res_cost,
            "res_bus": self.res_bus,
            "res_line": self.res_line,
            "res_gen": self.res_gen,
            "res_x": np.concatenate(
                (
                    self.x_gen,
                    self.x_load,
                    self.x_line_or_1,
                    self.x_line_or_2,
                    self.x_line_ex_1,
                    self.x_line_ex_2,
                )
            ),
            "res_x_gen": self.x_gen,
            "res_x_load": self.x_load,
            "res_x_line_or_1": self.x_line_or_1,
            "res_x_line_or_2": self.x_line_or_2,
            "res_x_line_ex_1": self.x_line_ex_1,
            "res_x_line_ex_2": self.x_line_ex_2,
            "res_x_line_status_switch": self.x_line_status_switch,
            "res_x_substation_topology_switch": self.x_substation_topology_switch,
            "res_gap": gap,
        }

        return result
