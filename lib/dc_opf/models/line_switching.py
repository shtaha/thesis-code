import pyomo.environ as pyo

from lib.data_utils import parse_gurobi_log
from .standard import StandardDCOPF


class LineSwitchingDCOPF(StandardDCOPF):
    def __init__(
        self,
        name,
        grid,
        grid_backend,
        n_line_status_changes=1,
        solver_name="gurobi",
        verbose=False,
        **kwargs,
    ):
        super().__init__(name, grid, grid_backend, solver_name, verbose, **kwargs)

        # Limit on the number of line status changes
        self.n_line_status_changes = n_line_status_changes

        # Optimal line status
        self.x = None

    def build_model(self, big_m=True, gen_cost=True, line_margin=True):
        # Model
        self.model = pyo.ConcreteModel(f"{self.name}")

        # Indexed sets
        self._build_indexed_sets()  # Indexing over buses, lines, generators, and loads

        # Parameters
        self._build_parameters()

        # Variables
        self._build_variables()

        # Constraints
        self._build_constraints(big_m=big_m)

        # Objective
        self._build_objective(
            gen_cost=gen_cost, line_margin=line_margin
        )  # Objective to be optimized.

    """
            VARIABLES.
        """

    def _build_variables(self):
        self._build_variables_standard_generators()  # Generator productions and bounds

        self._build_variable_standard_delta()  # Bus voltage angles and bounds

        if len(self.ext_grid.index):
            self._build_variables_standard_ext_grids()  # External grid power injections

        # Power line flows
        self.model.line_flow = pyo.Var(
            self.model.line_set,
            domain=pyo.Reals,
            initialize=self._create_map_ids_to_values(self.line.index, self.line.p_pu),
        )

        # Power line status
        # x = 0: Line is disconnected.
        # x = 1: Line is disconnected.
        self.model.x = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.status.values.astype(int)
            ),
        )

    """
        CONSTRAINTS.
    """

    def _build_constraints(self, big_m=True):
        # Power line flow definition
        # big_m = False: F_l = F_ij = b_ij * (delta_i - delta_j) * x_l
        # big_m = True: -M_l (1 - x_l) <= F_ij - b_ij * (delta_i - delta_j) <= M_l * (1 - x_l)
        # M_l = b_l * (pi/2 - (- pi/2)) = b_l * pi
        self._build_constraint_line_flows(big_m=big_m)  # Power flow definition

        # Indicator constraints on power line flow
        # -F_l^max * x_l <= F_l <= F_l^max * x_l
        self._build_constraint_line_max_flow()

        self._build_constraint_bus_balance()  # Bus power balance

        self._build_constraint_max_line_status_changes()

    def _build_constraint_line_max_flow(self):
        def _constraint_max_flow_lower(model, line_id):
            return (
                -model.line_flow_max[line_id] * model.x[line_id]
                <= model.line_flow[line_id]
            )

        def _constraint_max_flow_upper(model, line_id):
            return (
                model.line_flow[line_id]
                <= model.line_flow_max[line_id] * model.x[line_id]
            )

        self.model.constraint_line_max_flow_lower = pyo.Constraint(
            self.model.line_set, rule=_constraint_max_flow_lower
        )
        self.model.constraint_line_max_flow_upper = pyo.Constraint(
            self.model.line_set, rule=_constraint_max_flow_upper
        )

    def _build_constraint_line_flows(self, big_m=True):
        if big_m:
            self.model.big_m = pyo.Param(
                self.model.line_set,
                initialize=self._create_map_ids_to_values(
                    self.line.index, self.line.b_pu * 2 * self.grid.delta_max
                ),
                within=pyo.PositiveReals,
            )

            # -M_l(1 - x_l) <= F_ij - b_ij * (delta_i - delta_j) <= M_l * (1 - x_l)
            def _constraint_line_flow_upper(model, line_id):
                return model.line_flow[line_id] - model.line_b[line_id] * (
                    model.delta[model.line_ids_to_bus_ids[line_id][0]]
                    - model.delta[model.line_ids_to_bus_ids[line_id][1]]
                ) <= model.big_m[line_id] * (1 - model.x[line_id])

            def _constraint_line_flow_lower(model, line_id):
                return -model.big_m[line_id] * (
                    1 - model.x[line_id]
                ) <= model.line_flow[line_id] - model.line_b[line_id] * (
                    model.delta[model.line_ids_to_bus_ids[line_id][0]]
                    - model.delta[model.line_ids_to_bus_ids[line_id][1]]
                )

            self.model.constraint_line_flow_upper = pyo.Constraint(
                self.model.line_set, rule=_constraint_line_flow_upper
            )

            self.model.constraint_line_flow_lower = pyo.Constraint(
                self.model.line_set, rule=_constraint_line_flow_lower
            )
        else:

            def _constraint_line_flow(model, line_id):
                return (
                    model.line_flow[line_id]
                    == model.line_b[line_id]
                    * (
                        model.delta[model.line_ids_to_bus_ids[line_id][0]]
                        - model.delta[model.line_ids_to_bus_ids[line_id][1]]
                    )
                    * model.x[line_id]
                )

            self.model.constraint_line_flow = pyo.Constraint(
                self.model.line_set, rule=_constraint_line_flow
            )

    def _build_constraint_max_line_status_changes(self):
        def _constraint_max_line_status_change(model):
            line_status_change = [
                1 - model.x[line_id] if model.line_status[line_id] else model.x[line_id]
                for line_id in model.line_set
            ]

            return sum(line_status_change) <= self.n_line_status_changes

        self.model.constraint_max_line_status_changes = pyo.Constraint(
            rule=_constraint_max_line_status_change
        )

    """
        SOLVE FUNCTIONS.
    """

    def solve(self, verbose=False, tol=1e-9, time_limit=20):
        self._solve(verbose=verbose, tol=tol, time_limit=20)

        # Parse Gurobi log for additional information
        gap = parse_gurobi_log(self.solver._log)["gap"]
        if gap < 1e-4:
            gap = 1e-4

        # Save standard DC-OPF variable results
        self._solve_save()

        # Save line status variable
        self.x = self._round_solution(self._access_pyomo_variable(self.model.x))
        self.res_line["status"] = self.x[~self.line.trafo]
        self.res_trafo["status"] = self.x[self.line.trafo]

        if verbose:
            self.model.display()

        result = {
            "res_cost": self.res_cost,
            "res_bus": self.res_bus,
            "res_line": self.res_line,
            "res_gen": self.res_gen,
            "res_x": self.x,
            "res_gap": gap,
        }
        return result
