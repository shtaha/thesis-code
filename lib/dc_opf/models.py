import numpy as np
import pandapower as pp
import pandas as pd
import pyomo.environ as pyo
import pyomo.opt as pyo_opt

from lib.data_utils import parse_gurobi_log


class UnitConverter:
    def __init__(self, base_unit_p=1e6, base_unit_v=1e5):
        # Base units
        self.base_unit_p = base_unit_p  # Base unit is 1 MVA or 1 MW or 1 MVar
        self.base_unit_v = base_unit_v  # Base unit is 100 kV
        self.base_unit_i = self.base_unit_p / self.base_unit_v  # Base unit is 1 A
        self.base_unit_z = self.base_unit_v ** 2 / self.base_unit_p

        self.unit_p_mw = 1e6
        self.unit_a_ka = 1e3
        self.unit_v_kv = 1e3

    def print_base_units(self):
        print("base units")
        print("{:<10}{} W".format("unit p", self.base_unit_p))
        print("{:<10}{} V".format("unit v", self.base_unit_v))
        print("{:<10}{} A".format("unit a", self.base_unit_i))
        print("{:<10}{} Ohm".format("unit z", self.base_unit_z))

    def convert_mw_to_per_unit(self, p_mw):
        p_pu = p_mw * self.unit_p_mw / self.base_unit_p
        return p_pu

    def convert_ohm_to_per_unit(self, z_ohm):
        z_pu = z_ohm / self.base_unit_z
        return z_pu

    def convert_a_to_per_unit(self, i_a):
        i_pu = i_a / self.base_unit_i
        return i_pu

    def convert_ka_to_per_unit(self, i_ka):
        i_pu = i_ka * self.unit_a_ka / self.base_unit_i
        return i_pu

    def convert_kv_to_per_unit(self, v_kv):
        v_pu = v_kv * self.unit_v_kv / self.base_unit_v
        return v_pu

    def convert_per_unit_to_mw(self, p_pu):
        p_mw = p_pu * self.base_unit_p / self.unit_p_mw
        return p_mw

    def convert_per_unit_to_ka(self, i_pu):
        i_ka = i_pu * self.base_unit_i / self.unit_a_ka
        return i_ka

    @staticmethod
    def convert_degree_to_rad(deg):
        rad = deg / 180.0 * np.pi
        return rad

    @staticmethod
    def convert_rad_to_deg(rad):
        deg = rad / np.pi * 180.0
        return deg


class PyomoMixin:
    @staticmethod
    def _dataframe_to_list_of_tuples(df):
        return [tuple(row) for row in df.to_numpy()]

    @staticmethod
    def _reverse_index_map(indices_1, indices_2):
        reverse_map = [list(np.equal(indices_1, idx).nonzero()[0]) for idx in indices_2]
        return reverse_map

    @staticmethod
    def _create_map_ids_to_values_sum(ids, sum_ids, values):
        return {idx: values[sum_ids[idx]].sum() for idx in ids}

    @staticmethod
    def _create_map_ids_to_values(ids, values):
        return {idx: value for idx, value in zip(ids, values)}

    @staticmethod
    def _access_pyomo_variable(var):
        return np.array([pyo.value(var[idx]) for idx in var])


class StandardDCOPF(UnitConverter, PyomoMixin):
    def __init__(self, name, grid, solver="gurobi", verbose=False, **kwargs):
        UnitConverter.__init__(self, **kwargs)
        if verbose:
            self.print_base_units()

        self.name = name
        self.grid = grid

        self.bus = grid.bus
        self.line = grid.line
        self.gen = grid.gen
        self.load = grid.load

        self.model = None
        self.solver = pyo_opt.SolverFactory(solver)

        # Results
        self.res_cost = None
        self.res_bus = None
        self.res_line = None
        self.res_gen = None

        # DC-OPF Costs
        self.gen["cost_pu"] = np.ones_like(self.gen.index)

    def set_gen_cost(self, gen_costs):
        gen_costs = np.array(gen_costs).flatten()
        assert gen_costs.size == self.gen["cost_pu"].size  # Check dimensions

        self.gen["cost_pu"] = gen_costs

    def _build_per_unit_grid(self):
        """
        Note: DataFrames are passed-by-reference. self.grid.line == self.line
        """
        # Buses
        self.bus["vn_pu"] = self.convert_kv_to_per_unit(self.bus["vn_kv"])

        # Power lines
        if "max_loading_percent" not in self.line.columns:
            self.line["max_loading_percent"] = 100 * np.ones((self.line.shape[0],))

        self.line["x_pu"] = self.convert_ohm_to_per_unit(
            self.line["x_ohm_per_km"] * self.line["length_km"] / self.line["parallel"]
        )
        self.line["b_pu"] = 1 / self.line["x_pu"]
        self.line["max_i_pu"] = self.convert_ka_to_per_unit(self.line["max_i_ka"])

        # Assume a line connects buses with same voltage level
        self.line["max_p_pu"] = (
            self.line["max_i_pu"] * self.bus["vn_pu"][self.line["from_bus"]].values
        )

        # Generators
        self.gen["p_pu"] = self.convert_mw_to_per_unit(self.gen["p_mw"])
        self.grid.gen["max_p_pu"] = self.convert_mw_to_per_unit(self.gen["max_p_mw"])
        self.grid.gen["min_p_pu"] = self.convert_mw_to_per_unit(self.gen["min_p_mw"])

        # Loads
        self.load["p_pu"] = self.convert_mw_to_per_unit(self.load["p_mw"])

    def print_per_unit_grid(self):
        print("\nGRID\n")
        print("BUS\n" + self.bus[["name", "vn_pu"]].to_string())
        print(
            "LINE\n"
            + self.line[
                [
                    "name",
                    "from_bus",
                    "to_bus",
                    "b_pu",
                    "max_i_pu",
                    "max_p_pu",
                    "max_loading_percent",
                ]
            ].to_string()
        )
        print(
            "GEN\n"
            + self.gen[
                ["name", "bus", "p_pu", "min_p_pu", "max_p_pu", "cost_pu"]
            ].to_string()
        )
        print("LOAD\n" + self.load[["name", "bus", "p_pu"]].to_string())

    def print_model(self):
        print(self.model.pprint())

    def build_model(self):
        self._build_per_unit_grid()

        self.model = pyo.ConcreteModel("Standard DC-OPF Model")

        """
            DC-OPF INDEXED SETS.
        """
        self._build_indexed_sets()  # Indexing over buses, lines, generators, and loads

        """
            DC-OPF PARAMETERS - FIXED.
        """
        self._build_parameters_generators()  # Bounds on generator production
        self._build_parameters_lines()  # Power line thermal limit and susceptance
        self._build_parameters_objective()  # Objective parameters

        """
            DC-OPF PARAMETERS - VARIABLE.
        """
        self._build_parameters_topology()  # Topology of generators and power lines
        self._build_parameters_delta()  # Bus voltage angle bounds and reference node
        self._build_parameters_loads()  # Bus load injections

        """
            DC-OPF VARIABLES.
        """
        self._build_variables_standard_generators()  # Generator productions and bounds
        self._build_variable_standard_delta()  # Bus voltage angles and bounds
        self._build_variable_standard_line()  # Power line flows and bounds

        """
            DC-OPF CONSTRAINTS.            
        """
        self._build_constraint_line_flows()  # Power flow definition
        self._build_constraint_bus_balance()  # Bus power balance

        """
            DC-OPF OBJECTIVE.
        """
        self._build_objective()  # Objective to be optimized.

    def _build_constraint_bus_balance(self):
        # Bus power balance constraints
        def _constraint_bus_balance(model, bus_id):
            bus_gen_ids = model.bus_ids_to_gen_ids[bus_id]
            bus_gen_p = [model.gen_p[gen_id] for gen_id in bus_gen_ids]

            # Injections
            sum_gen_p = 0
            if len(bus_gen_p):
                sum_gen_p = sum(bus_gen_p)

            sum_load_p = float(model.bus_load_p[bus_id])

            # Power line flows
            flows_out = [
                model.line_flow[line_id]
                for line_id in model.line_set
                if bus_id == model.line_ids_to_bus_ids[line_id][0]
            ]

            flows_in = [
                model.line_flow[line_id]
                for line_id in model.line_set
                if bus_id == model.line_ids_to_bus_ids[line_id][1]
            ]

            if len(flows_in) == 0 and len(flows_out) == 0:
                return pyo.Constraint.Skip

            return sum_gen_p - sum_load_p == sum(flows_out) - sum(flows_in)

        self.model.constraint_bus_balance = pyo.Constraint(
            self.model.bus_set, rule=_constraint_bus_balance
        )

    def _build_constraint_line_flows(self):
        # Power flow equation
        def _constraint_line_flow(model, line_id):
            return model.line_flow[line_id] == model.line_b[line_id] * (
                model.delta[model.line_ids_to_bus_ids[line_id][0]]
                - model.delta[model.line_ids_to_bus_ids[line_id][1]]
            )

        self.model.constraint_line_flow = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_flow
        )

    def _build_variable_standard_line(self):
        # Line power flows
        def _bounds_flow_max(model, line_id):
            return -model.line_flow_max[line_id], model.line_flow_max[line_id]

        self.model.line_flow = pyo.Var(
            self.model.line_set, domain=pyo.Reals, bounds=_bounds_flow_max
        )

    def _build_variable_standard_delta(self):
        # Bus voltage angle
        def _bounds_delta(model, bus_id):
            if bus_id == pyo.value(model.slack_bus_id):
                return 0.0, 0.0
            else:
                return -model.delta_max, model.delta_max

        self.model.delta = pyo.Var(
            self.model.bus_set,
            domain=pyo.Reals,
            bounds=_bounds_delta,
            initialize=self._create_map_ids_to_values(
                self.bus.index.values, np.zeros_like(self.bus.index.values)
            ),
        )

    def _build_variables_standard_generators(self):
        def _bounds_gen_p(model, gen_id):
            return model.gen_p_min[gen_id], model.gen_p_max[gen_id]

        self.model.gen_p = pyo.Var(
            self.model.gen_set,
            domain=pyo.NonNegativeReals,
            bounds=_bounds_gen_p,
            initialize=self._create_map_ids_to_values(
                self.gen.index.values, self.gen["p_pu"]
            ),
        )

    def _build_parameters_objective(self):
        self.model.gen_costs = pyo.Param(
            self.model.gen_set,
            initialize=self._create_map_ids_to_values(
                self.gen.index.values, self.gen["cost_pu"]
            ),
            within=pyo.NonNegativeReals,
        )

    def _build_parameters_loads(self):
        # Load bus injections
        self.model.bus_load_p = pyo.Param(
            self.model.bus_set,
            initialize=self._create_map_ids_to_values_sum(
                self.bus.index.values,
                self._reverse_index_map(self.load["bus"].values, self.bus.index.values),
                self.load["p_pu"],
            ),
            within=pyo.NonNegativeReals,
        )

    def _build_parameters_delta(self):
        # Bus voltage angles
        self.model.delta_max = pyo.Param(
            initialize=np.pi / 2, within=pyo.NonNegativeReals
        )

        # Slack bus index
        self.model.slack_bus_id = pyo.Param(
            initialize=np.where(self.gen["slack"])[0][0].astype(int),
            within=self.model.bus_set,
        )

    def _build_parameters_topology(self):
        self.model.line_ids_to_bus_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index.values,
                self._dataframe_to_list_of_tuples(self.line[["from_bus", "to_bus"]]),
            ),
            within=self.model.bus_set * self.model.bus_set,
        )

        self.model.bus_ids_to_gen_ids = pyo.Param(
            self.model.bus_set,
            initialize=self._create_map_ids_to_values(
                self.bus.index.values,
                self._reverse_index_map(self.gen["bus"].values, self.bus.index.values),
            ),
            within=pyo.Any,
        )

    def _build_parameters_lines(self):
        self.model.line_flow_max = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index.values, self.line["max_p_pu"]
            ),
            within=pyo.NonNegativeReals,
        )
        self.model.line_b = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index.values, self.line["b_pu"]
            ),
            within=pyo.NonNegativeReals,
        )

    def _build_parameters_generators(self):
        self.model.gen_p_max = pyo.Param(
            self.model.gen_set,
            initialize=self._create_map_ids_to_values(
                self.gen.index.values, self.gen["max_p_pu"]
            ),
            within=pyo.NonNegativeReals,
        )

        # Set minimum generator to 0 if negative value
        self.model.gen_p_min = pyo.Param(
            self.model.gen_set,
            initialize=self._create_map_ids_to_values(
                self.gen.index.values, np.maximum(0.0, self.gen["min_p_pu"].values)
            ),
            within=pyo.NonNegativeReals,
        )

    def _build_indexed_sets(self):
        self.model.bus_set = pyo.Set(
            initialize=self.bus.index.values, within=pyo.NonNegativeIntegers
        )
        self.model.line_set = pyo.Set(
            initialize=self.line.index.values, within=pyo.NonNegativeIntegers
        )
        self.model.load_set = pyo.Set(
            initialize=self.load.index.values, within=pyo.NonNegativeIntegers
        )
        self.model.gen_set = pyo.Set(
            initialize=self.gen.index.values, within=pyo.NonNegativeIntegers
        )

    def _build_objective(self):
        def _objective_gen_p(model):
            return sum(
                [
                    model.gen_p[gen_id] * model.gen_costs[gen_id]
                    for gen_id in model.gen_set
                ]
            )

        self.model.objective = pyo.Objective(rule=_objective_gen_p, sense=pyo.minimize)

    def _solve_save(self):
        self.res_cost = pyo.value(self.model.objective)

        self.res_bus = self.bus[["name", "vn_pu"]].copy()
        self.res_bus["delta_pu"] = self._access_pyomo_variable(self.model.delta)
        self.res_bus["delta_deg"] = self.convert_rad_to_deg(
            self._access_pyomo_variable(self.model.delta)
        )

        self.res_gen = self.gen[["name", "min_p_pu", "max_p_pu", "cost_pu"]].copy()
        self.res_gen["p_pu"] = self._access_pyomo_variable(self.model.gen_p)

        self.res_line = self.line[
            ["name", "from_bus", "to_bus", "max_i_pu", "max_p_pu"]
        ].copy()
        self.res_line["p_pu"] = self._access_pyomo_variable(self.model.line_flow)
        self.res_line["loading_percent"] = np.abs(
            self.res_line["p_pu"] / self.line["max_p_pu"] * 100
        )

    def _solve(self, verbose=False, tol=1e-9):
        """
        Set options to solver and solve the MIP.
        Gurobi parameters: https://www.gurobi.com/documentation/9.0/refman/parameters.html
        """
        options = {
            "OptimalityTol": tol,
            "MIPGap": tol,
        }

        self.solver.solve(self.model, tee=verbose, options=options)

    def solve(self, verbose=False, tol=1e-9):
        self._solve(verbose=verbose, tol=tol)

        # Save standard DC-OPF variable results
        self._solve_save()

        if verbose:
            self.model.display()

        result = {
            "res_cost": self.res_cost,
            "res_bus": self.res_bus,
            "res_line": self.res_line,
            "res_gen": self.res_gen,
        }
        return result

    def solve_backend(self, verbose=False):
        for gen_id in self.gen.index.values:
            pp.create_poly_cost(
                self.grid,
                gen_id,
                "gen",
                cp1_eur_per_mw=self.convert_per_unit_to_mw(self.gen["cost_pu"][gen_id]),
            )

        try:
            pp.rundcopp(self.grid, verbose=verbose)
            valid = True
        except pp.optimal_powerflow.OPFNotConverged as e:
            valid = False
            print(e)

        # Convert NaNs of inactive buses to 0
        self.grid.res_bus = self.grid.res_bus.fillna(0)
        self.grid.res_line = self.grid.res_line.fillna(0)
        self.grid.res_gen = self.grid.res_gen.fillna(0)

        self.grid.res_bus["delta_pu"] = self.convert_degree_to_rad(
            self.grid.res_bus["va_degree"]
        )
        self.grid.res_line["p_pu"] = self.convert_mw_to_per_unit(
            self.grid.res_line["p_from_mw"]
        )
        self.grid.res_line["i_pu"] = self.convert_ka_to_per_unit(
            self.grid.res_line["i_from_ka"]
        )
        self.grid.res_gen["p_pu"] = self.convert_mw_to_per_unit(
            self.grid.res_gen["p_mw"]
        )
        self.grid.res_gen["cost_pu"] = self.gen["cost_pu"]

        result = {
            "res_cost": self.grid.res_cost,
            "res_bus": self.grid.res_bus,
            "res_line": self.grid.res_line,
            "res_gen": self.grid.res_gen,
            "valid": valid,
        }
        return result

    def print_results(self):
        print("\nRESULTS\n")
        print("{:<10}{}".format("OBJECTIVE", self.res_cost))
        print("RES BUS\n" + self.res_bus.to_string())
        print("RES LINE\n" + self.res_line.to_string())
        print("RES GEN\n" + self.res_gen.to_string())

    def print_results_backend(self):
        print("\nRESULTS BACKEND\n")
        print("{:<10}{}".format("OBJECTIVE", self.grid.res_cost))
        print("RES BUS\n" + self.grid.res_bus[["delta_pu"]].to_string())
        print(
            "RES LINE\n"
            + self.grid.res_line[["p_pu", "i_pu", "loading_percent"]].to_string()
        )
        print("RES GEN\n" + self.grid.res_gen[["p_pu"]].to_string())

    def solve_and_compare(self, verbose=False, tol=1e-9):
        result = self.solve(verbose=verbose, tol=tol)
        result_backend = self.solve_backend()

        res_cost = pd.DataFrame(
            {
                "objective": [result["res_cost"]],
                "b_objective": [result_backend["res_cost"]],
                "diff": np.abs(result["res_cost"] - result_backend["res_cost"]),
            }
        )

        res_bus = pd.DataFrame(
            {
                "delta_pu": result["res_bus"]["delta_pu"],
                "b_delta_pu": result_backend["res_bus"]["delta_pu"],
                "diff": np.abs(
                    result["res_bus"]["delta_pu"]
                    - result_backend["res_bus"]["delta_pu"]
                ),
            }
        )

        res_line = pd.DataFrame(
            {
                "p_pu": result["res_line"]["p_pu"],
                "b_p_pu": result_backend["res_line"]["p_pu"],
                "diff": np.abs(
                    result["res_line"]["p_pu"] - result_backend["res_line"]["p_pu"]
                ),
            }
        )

        res_gen = pd.DataFrame(
            {
                "gen_pu": result["res_gen"]["p_pu"],
                "b_gen_pu": result_backend["res_gen"]["p_pu"],
                "diff": np.abs(
                    result["res_gen"]["p_pu"] - result_backend["res_gen"]["p_pu"]
                ),
                "gen_cost_pu": self.gen["cost_pu"],
            }
        )

        if verbose:
            print(res_cost.to_string())
            print(res_bus.to_string())
            print(res_line.to_string())
            print(res_gen.to_string())

        result = {
            "res_cost": res_cost,
            "res_bus": res_bus,
            "res_line": res_line,
            "res_gen": res_gen,
        }
        return result


class LineSwitchingDCOPF(StandardDCOPF):
    def __init__(
        self,
        name,
        grid,
        n_line_status_changes=1,
        solver="gurobi",
        verbose=False,
        **kwargs
    ):
        super().__init__(name, grid, solver, verbose, **kwargs)

        # Limit on the number of line status changes
        self.n_line_status_changes = n_line_status_changes

        # Optimal line status
        self.x = None

    def build_model(self, big_m=True):
        self._build_per_unit_grid()

        self.model = pyo.ConcreteModel("DC-OPF Model with line switching")

        """
            DC-OPF INDEXED SETS.
        """
        self._build_indexed_sets()  # Indexing over buses, lines, generators, and loads

        """
            DC-OPF PARAMETERS - FIXED.
        """
        self._build_parameters_generators()  # Bounds on generator production
        self._build_parameters_lines()  # Power line thermal limit and susceptance
        self._build_parameters_objective()  # Objective parameters

        """
            DC-OPF PARAMETERS - VARIABLE.
        """
        self._build_parameters_topology()  # Topology of generators and power lines
        self._build_parameters_delta()  # Bus voltage angle bounds and reference node
        self._build_parameters_loads()  # Bus load injections

        """
            DC-OPF VARIABLES.
        """
        self._build_variables_standard_generators()  # Generator productions and bounds
        self._build_variable_standard_delta()  # Bus voltage angles and bounds

        # Line status
        # x = 0: Line is disconnected.
        # x = 1: Line is disconnected.
        self.model.x = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index.values, self.line["in_service"].values.astype(int)
            ),
        )

        self._build_variable_line()

        """
            DC-OPF CONSTRAINTS.            
        """
        # Power line flow definition
        # big_m = False: F_l = F_ij = b_ij * (delta_i - delta_j) * x_l
        # big_m = True: -M_l (1 - x_l) <= F_ij - b_ij * (delta_i - delta_j) <= M_l * (1 - x_l)
        # M_l = b_l * (pi/2 - (- pi/2)) = b_l * pi
        self._build_constraint_line_flows(big_m=big_m)  # Power flow definition

        # Indicator constraints on power line flow
        # -F_l^max * x_l <= F_l <= F_l^max * x_l
        self._build_constraint_line_max_flow()

        self._build_constraint_bus_balance()  # Bus power balance

        # Limit number of line status changes
        self._build_constraint_max_line_status_changes()

        """
            DC-OPF OBJECTIVE.
        """
        self._build_objective()  # Objective to be optimized.

    def _build_variable_line(self):
        self.model.line_flow = pyo.Var(self.model.line_set, domain=pyo.Reals)

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
            if big_m:
                self.model.big_m = pyo.Param(
                    self.model.line_set,
                    initialize=self._create_map_ids_to_values(
                        self.line.index.values, self.line["b_pu"] * np.pi
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
                1 - model.x[line_id]
                if self.line["in_service"][line_id]
                else model.x[line_id]
                for line_id in model.line_set
            ]

            return sum(line_status_change) <= self.n_line_status_changes

        self.model.constraint_max_line_status_changes = pyo.Constraint(
            rule=_constraint_max_line_status_change
        )

    def _build_objective(self):
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

        def _objective(model):
            return _objective_gen_p(model) + _objective_line_margin(model)

        self.model.objective = pyo.Objective(rule=_objective, sense=pyo.minimize)

    def solve(self, verbose=False, tol=1e-9):
        self._solve(verbose=verbose, tol=tol)

        # Parse Gurobi log for additional information
        gap = parse_gurobi_log(self.solver._log)["gap"]
        if gap < 1e-6:
            gap = 1e-6

        # Save standard DC-OPF variable results
        self._solve_save()

        # Save line status variable
        self.x = self._round_solution(self._access_pyomo_variable(self.model.x))
        self.res_line["line_status"] = self.x

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

    @staticmethod
    def _round_solution(x):
        x = np.round(x)
        return x


class TopologyOptimizationDCOPF(StandardDCOPF):
    def __init__(
        self,
        name,
        grid,
        solver="gurobi",
        verbose=False,
        **kwargs
    ):
        super().__init__(name, grid, solver, verbose, **kwargs)

        # Optimal switching status
        self.x_line_or = None
        self.x_line_ex = None
        self.x_gen = None
        self.x_load = None

    def build_model(self, big_m=True):
        self._build_per_unit_grid()

        self.model = pyo.ConcreteModel("DC-OPF Model with topology optimization")

        """
            DC-OPF INDEXED SETS.
        """
        self._build_indexed_sets()  # Indexing over buses, lines, generators, and loads

        sub_ids = sorted(self.grid.bus["sub_id"].unique())
        self.model.sub_set = self.model.bus_set = pyo.Set(
            initialize=sub_ids, within=pyo.NonNegativeIntegers
        )

        # """
        #     DC-OPF PARAMETERS - FIXED.
        # """
        # self._build_parameters_generators()  # Bounds on generator production
        # self._build_parameters_lines()  # Power line thermal limit and susceptance
        # self._build_parameters_objective()  # Objective parameters
        #
        # """
        #     DC-OPF PARAMETERS - VARIABLE.
        # """
        # self._build_parameters_topology()  # Topology of generators and power lines
        # self._build_parameters_delta()  # Bus voltage angle bounds and reference node
        # self._build_parameters_loads()  # Bus load injections
        #
        # """
        #     DC-OPF VARIABLES.
        # """
        # self._build_variables_standard_generators()  # Generator productions and bounds
        # self._build_variable_standard_delta()  # Bus voltage angles and bounds
        #
        # # Line status
        # # x = 0: Line is disconnected.
        # # x = 1: Line is disconnected.
        # self.model.x = pyo.Var(
        #     self.model.line_set,
        #     domain=pyo.Binary,
        #     initialize=self._create_map_ids_to_values(
        #         self.line.index.values, self.line["in_service"].values.astype(int)
        #     ),
        # )
        #
        # self._build_variable_line()
        #
        # """
        #     DC-OPF CONSTRAINTS.
        # """
        # # Power line flow definition
        # # big_m = False: F_l = F_ij = b_ij * (delta_i - delta_j) * x_l
        # # big_m = True: -M_l (1 - x_l) <= F_ij - b_ij * (delta_i - delta_j) <= M_l * (1 - x_l)
        # # M_l = b_l * (pi/2 - (- pi/2)) = b_l * pi
        # self._build_constraint_line_flows(big_m=big_m)  # Power flow definition
        #
        # # Indicator constraints on power line flow
        # # -F_l^max * x_l <= F_l <= F_l^max * x_l
        # self._build_constraint_line_max_flow()
        #
        # self._build_constraint_bus_balance()  # Bus power balance
        #
        # # Limit number of line status changes
        # self._build_constraint_max_line_status_changes()
        #
        # """
        #     DC-OPF OBJECTIVE.
        # """
        # self._build_objective()  # Objective to be optimized.

