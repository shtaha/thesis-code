import numpy as np
import pandapower as pp
import pandas as pd
import pyomo.environ as pyo
import pyomo.opt as pyo_opt


class PyomoMixin:
    pass


class UnitConverter:

    def __init__(self, base_unit_p=1e6, base_unit_v=1e5):
        # Base units
        self.base_unit_p = base_unit_p  # Base unit is 1 MVA or 1 MW or 1 MVar
        self.base_unit_v = base_unit_v  # Base unit is 100 kV
        self.base_unit_a = self.base_unit_p / self.base_unit_v  # Base unit is 1 A
        self.base_unit_z = self.base_unit_v ** 2 / self.base_unit_p

        self.unit_p_mw = 1e6
        self.unit_a_ka = 1e3
        self.unit_v_kv = 1e3

    def print_converted(self):
        print("base units")
        print("{:<10}{}".format("unit p", self.base_unit_p))
        print("{:<10}{}".format("unit v", self.base_unit_v))
        print("{:<10}{}".format("unit a", self.base_unit_a))
        print("{:<10}{}".format("unit z", self.base_unit_z))

    def convert_mw_to_per_unit(self, p_mw):
        p_pu = p_mw * self.unit_p_mw / self.base_unit_p
        return p_pu

    def convert_ohm_to_per_unit(self, z_ohm):
        z_pu = z_ohm / self.base_unit_z
        return z_pu

    def convert_a_to_per_unit(self, i_a):
        i_pu = i_a / self.base_unit_a
        return i_pu

    def convert_ka_to_per_unit(self, i_ka):
        i_pu = i_ka * self.unit_a_ka / self.base_unit_a
        return i_pu

    def convert_kv_to_per_unit(self, v_kv):
        v_pu = v_kv * self.unit_v_kv / self.base_unit_v
        return v_pu


class DCOptimalPowerFlow(object):
    """
    Constructor for DC-OPF given the environment and the observation.
    """

    def __init__(self, env, solver="gurobi", verbose=False):
        """
        Initialize DC-OPF Parameters given an environment: information on topology, generators, loads, and lines.
        """
        self.env = env
        self.backend = env.backend
        self.grid = env.backend._grid
        self.unit_converter = UnitConverter(base_unit_v=self.grid.bus["vn_kv"].values[0] * 1000)  # TODO: Hack

        self.bus = self.grid.bus
        self.gen = self.grid.gen
        self.load = self.grid.load
        self.line = self.grid.line

        # Grid parameters
        self.n_bus = None
        self.bus_ids = None
        self.n_sub = None
        self.sub_ids = None
        self.bus_ids_to_sub_ids = None
        self.bus_ids_to_sub_bus_ids = None

        # Generator parameters
        self.n_gen = None
        self.gen_ids = None
        self.gen_p_max = None  # In p.u.
        self.gen_p_min = None  # In p.u.
        self.gen_ids_to_bus_ids = None  # VARIABLE
        self.bus_ids_to_gen_ids = None  # VARIABLE
        self.gen_costs = None  # TEST VARIABLE

        # Load parameters
        self.n_load = None
        self.load_ids = None
        self.load_ids_to_bus_ids = None  # VARIABLE
        self.load_p = None  # In p.u. # VARIABLE
        self.bus_ids_to_load_ids = None  # VARIABLE

        # Line parameters
        self.n_line = None
        self.line_ids = None
        self.line_resistance = None  # In p.u.
        self.line_reactance = None  # In p.u.
        self.line_inverse_reactance = None  # In p.u.
        self.line_i_max = None  # In p.u.
        self.line_p_max = None  # In p.u.
        self.line_ids_to_bus_ids = None  # VARIABLE
        self.line_ids_to_or_bus_ids = None  # VARIABLE
        self.line_ids_to_ex_bus_ids = None  # VARIABLE

        # Initialize parameter values
        self.initialize_grid(verbose=verbose)
        self.initialize_generators(verbose=verbose)
        self.initialize_loads(verbose=verbose)
        self.initialize_lines(verbose=verbose)

        # DC-OPF model
        self.model = None

        # Initialize DC-OPF model and solver
        self.initialize_model()
        self.solver = pyo_opt.SolverFactory(solver)

        self.print_model(verbose=verbose)

    def print_model(self, verbose=False):
        if verbose:
            print(self.model.pprint())

    def initialize_model(self):
        self.model = pyo.ConcreteModel("DC-OPF Model")
        self.model.bus_set = pyo.Set(initialize=self.bus_ids, within=pyo.NonNegativeIntegers)
        self.model.line_set = pyo.Set(initialize=self.line_ids, within=pyo.NonNegativeIntegers)
        self.model.load_set = pyo.Set(initialize=self.load_ids, within=pyo.NonNegativeIntegers)
        self.model.gen_set = pyo.Set(initialize=self.gen_ids, within=pyo.NonNegativeIntegers)

        # FIXED Model parameters
        self.model.gen_p_max = pyo.Param(self.model.gen_set,
                                         initialize=self._create_map_ids_to_values(self.gen_ids, self.gen_p_max),
                                         within=pyo.NonNegativeReals)
        self.model.gen_p_min = pyo.Param(self.model.gen_set,
                                         initialize=self._create_map_ids_to_values(self.gen_ids, self.gen_p_min),
                                         within=pyo.NonNegativeReals)

        self.model.gen_costs = pyo.Param(self.model.gen_set,
                                         initialize=self._create_map_ids_to_values(
                                             self.gen_ids, self.gen_costs),
                                         within=pyo.NonNegativeReals)

        self.model.line_p_max = pyo.Param(self.model.line_set,
                                          initialize=self._create_map_ids_to_values(self.line_ids, self.line_p_max),
                                          within=pyo.NonNegativeReals)
        self.model.line_inverse_reactance = pyo.Param(self.model.line_set,
                                                      initialize=self._create_map_ids_to_values(self.line_ids,
                                                                                                self.line_inverse_reactance),
                                                      within=pyo.NonNegativeReals)

        # VARIABLE Model parameters
        self.model.line_ids_to_bus_ids = pyo.Param(self.model.line_set,
                                                   initialize=self._create_map_ids_to_values(self.line_ids,
                                                                                             self.line_ids_to_bus_ids),
                                                   within=self.model.bus_set * self.model.bus_set)
        self.model.bus_ids_to_gen_ids = pyo.Param(self.model.bus_set,
                                                  initialize=self._create_map_ids_to_values(self.bus_ids,
                                                                                            self.bus_ids_to_gen_ids),
                                                  within=pyo.Any)
        # Load bus injections
        self.model.bus_load_p = pyo.Param(self.model.bus_set,
                                          initialize=self._create_map_ids_to_values_sum(self.bus_ids,
                                                                                        self.bus_ids_to_load_ids,
                                                                                        self.load_p),
                                          within=pyo.NonNegativeReals)

        # Variables
        self.model.gen_p = pyo.Var(self.model.gen_set, domain=pyo.PositiveReals)
        self.model.delta = pyo.Var(self.model.bus_set, domain=pyo.Reals)  # Voltage angle
        self.model.line_p = pyo.Var(self.model.line_set, domain=pyo.Reals)  # Line flow
        self.model.bus_gen_p = pyo.Var(self.model.bus_set, within=pyo.NonNegativeReals)  # Generator bus injections

        # Constraints
        # Bound generator output
        self.model.constraint_gen_p = pyo.Constraint(self.model.gen_set, rule=self._constraint_gen_p)

        # Bound line power flow
        self.model.constraint_line_p = pyo.Constraint(self.model.line_set, rule=self._constraint_line_p)

        # Define line power flow
        self.model.constraint_line_p_def = pyo.Constraint(self.model.line_set, rule=self._constraint_line_p_def)

        # Set delta[0] = 0
        self.model.constraint_delta = pyo.Constraint(self.model.bus_set, rule=self._constraint_delta)

        # Define bus generator injections
        self.model.constrain_bus_gen_p = pyo.Constraint(self.model.bus_set, rule=self._constraint_bus_gen_p)

        # Bus power balance constraints
        self.model.constraint_bus_balance_p = pyo.Constraint(self.model.bus_set, rule=self._constraint_bus_balance_p)

        # Objectives
        self.model.objective = pyo.Objective(rule=self._objective_gen_p, sense=pyo.minimize)

    def update(self, obs):
        """
        Update DC-OPF parameters given a new observation.

        That is: grid topology,
        """
        pass

    def set_gen_cost(self, gen_costs):
        gen_costs = np.array(gen_costs).flatten()
        assert gen_costs.size == self.gen_costs.size  # Check dimensions

        self.gen_costs = gen_costs

    def solve_dc_opf_backend(self, verbose=False, **kwargs):
        for gen_id in self.gen_ids:
            pp.create_poly_cost(self.grid, gen_id, "gen", cp1_eur_per_mw=self.gen_costs[gen_id])

        pp.rundcopp(self.grid, verbose=verbose, **kwargs)

        if not verbose:
            print("\nDC-OPF solved by backend")
            print("{:<10}{}".format("cost", self.grid.res_cost))
            self._print_res_gen(self.grid.res_gen["p_mw"], self.gen_costs)

    def solve_dc_opf(self, verbose=False):
        _ = self.solver.solve(self.model, tee=verbose)

        if verbose:
            print(self.model.display())
        else:
            print("\nDC-OPF solved")
            print("{:<10}{}".format("cost", pyo.value(self.model.objective)))
            self._print_res_gen(self._access_pyomo_variable(self.model.gen_p), self.gen_costs)

    def initialize_grid(self, verbose=False):
        self.n_bus = self.bus.shape[0]
        self.bus_ids = self.bus.index.values  # [0, 1, ..., n_bus-1] (n_bus, )

        # self.n_sub = self.env.sub_info.shape[0]
        self.n_sub = self.env.n_sub
        self.sub_ids = np.arange(0, self.n_sub)  # [0, 1, ..., n_sub-1] (n_sub, )
        self.bus_ids_to_sub_ids = np.tile(self.sub_ids, 2)  # [0, 1, ..., n_sub-1, 0, 1, ..., n_sub-1] (n_bus, )
        self.bus_ids_to_sub_bus_ids = np.tile([1, 2], self.n_sub)  # [1, 2, 1, 2, ..., 1, 2] (n_sub, )

        if verbose:
            print("initializing grid ...")
            print("{:<30}{}\t{}".format("sub", self.sub_ids, self.n_sub))
            print("{:<30}{}\t{}".format("bus", self.bus_ids, self.n_bus))
            print("{:<30}{}".format("bus_ids_to_sub_ids", self.bus_ids_to_sub_ids))
            print("{:<30}{}\n".format("bus_ids_to_sub_bus_ids", self.bus_ids_to_sub_bus_ids))

        assert self.n_bus == 2 * self.n_sub  # Each substation has 2 buses

    def initialize_generators(self, verbose=False):
        self.n_gen = self.env.n_gen
        self.gen_ids = self.gen.index.values

        self.gen_p_max = self.unit_converter.convert_mw_to_per_unit(self.env.gen_pmax)  # In p.u.
        self.gen_p_min = self.unit_converter.convert_mw_to_per_unit(self.env.gen_pmin)  # In p.u.

        self.gen_ids_to_bus_ids = self.gen["bus"].values

        self.bus_ids_to_gen_ids = [list(np.equal(self.gen_ids_to_bus_ids, bus_id).nonzero()[0]) for bus_id in
                                   self.bus_ids]

        if verbose:
            print("initializing generators ...")
            print("{:<30}{}\t{}".format("gen", self.gen_ids, self.n_gen))
            print("{:<30}{} p.u.".format("gen_p_max", self.gen_p_max))
            print("{:<30}{} p.u.".format("gen_p_min", self.gen_p_min))
            print("{:<30}{}".format("gen_ids_to_bus_ids", self.gen_ids_to_bus_ids))
            print("{:<30}{}\n".format("bus_ids_to_gen_ids", self.bus_ids_to_gen_ids))

        # Initial value
        self.gen_costs = np.ones_like(self.gen_ids)

        assert self.n_gen == self.gen.shape[0]  # Check number of generators
        # assert np.equal(self.unit_converter.convert_mw_to_per_unit(self.gen["max_p_mw"]),
        #                 self.gen_p_max).all()  # Check if grid.json and prods_char.csv match
        # assert np.equal(self.unit_converter.convert_mw_to_per_unit(self.gen["min_p_mw"]),
        #                 self.gen_p_min).all()  # Check if grid.json and prods_char.csv match

    def initialize_loads(self, verbose=False):
        self.n_load = self.env.n_load
        self.load_ids = self.load.index.values

        self.load_p = self.unit_converter.convert_mw_to_per_unit(self.load["p_mw"].values)  # In p.u.

        self.load_ids_to_bus_ids = self.load["bus"].values

        self.bus_ids_to_load_ids = [list(np.equal(self.load_ids_to_bus_ids, bus_id).nonzero()[0]) for bus_id in
                                    self.bus_ids]

        if verbose:
            print("initializing loads ...")
            print("{:<30}{}\t{}".format("load", self.load_ids, self.n_load))
            print("{:<30}{}".format("load_p", self.load_p))
            print("{:<30}{}".format("load_ids_to_bus_ids", self.load_ids_to_bus_ids))
            print("{:<30}{}\n".format("bus_ids_to_load_ids", self.bus_ids_to_load_ids))

        assert self.n_load == self.load.shape[0]  # Check number of loads
        # assert np.equal(self.unit_converter.convert_mw_to_per_unit(self.env.get_obs().load_p),
        #                 self.load_p).all()  # Check consistency between new obs and env

    def initialize_lines(self, verbose=False):
        self.n_line = self.env.n_line
        self.line_ids = self.line.index.values

        # [(bus_or, bus_ex), ..., (bus_or, bus_ex)] (n_line, )
        self.line_ids_to_bus_ids = list(zip(self.line["from_bus"], self.line["to_bus"]))
        self.line_ids_to_or_bus_ids = self.line["from_bus"].values
        self.line_ids_to_ex_bus_ids = self.line["to_bus"].values

        line_length = self.line["length_km"].values
        line_parallel = self.line["parallel"].values

        if "max_loading_percent" in self.line.columns:
            line_max_loading = self.line["max_loading_percent"].values / 100.0
        else:
            line_max_loading = np.ones((self.n_line,))

        self.line_resistance = np.divide(np.multiply(self.line["r_ohm_per_km"].values, line_length),
                                         line_parallel)  # In Ohms
        self.line_resistance = self.unit_converter.convert_ohm_to_per_unit(self.line_resistance)

        self.line_reactance = np.divide(np.multiply(self.line["x_ohm_per_km"].values, line_length),
                                        line_parallel)  # Reactance in Ohms
        self.line_reactance = self.unit_converter.convert_ohm_to_per_unit(self.line_reactance)  # In p.u.

        self.line_inverse_reactance = np.divide(1, self.line_reactance)  # Negative susceptance in p.u.

        self.line_i_max = np.multiply(self.env.get_thermal_limit(), line_max_loading)  # Line current limit in Amperes
        self.line_i_max = self.unit_converter.convert_a_to_per_unit(self.line_i_max)  # In p.u.

        # TODO: HACK ONLY LINE OR!
        self.line_p_max = self.line_i_max * self.unit_converter.convert_kv_to_per_unit(
            self.bus["vn_kv"].values[self.line_ids_to_or_bus_ids])  # In p.u.

        if verbose:
            print("initializing lines ...")
            print("{:<30}{}\t{}".format("line", self.line_ids, self.n_line))
            print("{:<30}{}".format("line_resistance", self.line_resistance))
            print("{:<30}{}".format("line_reactance", self.line_reactance))
            print("{:<30}{}".format("line_inverse_reactance", self.line_inverse_reactance))
            print("{:<30}{}".format("line_i_max", self.line_i_max))
            print("{:<30}{}".format("line_p_max", self.line_p_max))
            print("{:<30}{}\n".format("line_ids_to_bus_ids", self.line_ids_to_bus_ids))

        assert self.n_line == self.line.shape[0]  # Check number of lines
        # assert np.equal(self.unit_converter.convert_ka_to_per_unit(self.line["max_i_ka"]),
        #                 self.line_i_max).all()  # Check consistency of thermal limits

    """
        Pyomo modeling functions.
    """

    @staticmethod
    def _bounds_gen_p(model, gen_id):
        return model.gen_p_min[gen_id], model.gen_p_max[gen_id]

    @staticmethod
    def _constraint_gen_p(model, gen_id):
        return pyo.inequality(model.gen_p_min[gen_id], model.gen_p[gen_id], model.gen_p_max[gen_id])

    @staticmethod
    def _constraint_line_p(model, line_id):
        return pyo.inequality(-model.line_p_max[line_id], model.line_p[line_id], model.line_p_max[line_id])

    @staticmethod
    def _constraint_line_p_def(model, line_id):
        return model.line_p[line_id] == model.line_inverse_reactance[line_id] * (
                model.delta[model.line_ids_to_bus_ids[line_id][0]] - model.delta[model.line_ids_to_bus_ids[line_id][1]])

    @staticmethod
    def _constraint_delta(model, bus_id):
        if bus_id == 0:
            return model.delta[bus_id] == 0
        else:
            return pyo.Constraint.Skip

    @staticmethod
    def _constraint_bus_gen_p(model, bus_id):
        bus_gen_ids = model.bus_ids_to_gen_ids[bus_id]
        bus_gen_p = [model.gen_p[gen_id] for gen_id in bus_gen_ids]
        if len(bus_gen_p):
            return model.bus_gen_p[bus_id] == sum(bus_gen_p)
        else:
            # If no generator bus injections
            return model.bus_gen_p[bus_id] == 0

    @staticmethod
    def _constraint_bus_balance_p(model, bus_id):
        return model.bus_gen_p[bus_id] - model.bus_load_p[bus_id] == \
               sum([model.line_p[line_id] for line_id in model.line_set if
                    bus_id == model.line_ids_to_bus_ids[line_id][0]]) - \
               sum([model.line_p[line_id] for line_id in model.line_set if
                    bus_id == model.line_ids_to_bus_ids[line_id][1]])

    @staticmethod
    def _objective_gen_p(model):
        return sum([model.gen_p[gen_id] * model.gen_costs[gen_id] for gen_id in model.gen_set])

    """
        Pyomo helper functions.
    """

    @staticmethod
    def _create_map_ids_to_values_sum(ids, sum_ids, values):
        return {idx: values[sum_ids[idx]].sum() for idx in ids}

    @staticmethod
    def _create_map_ids_to_values(ids, values):
        return {idx: value for idx, value in zip(ids, values)}

    @staticmethod
    def _access_pyomo_variable(var):
        return np.array([pyo.value(var[idx]) for idx in var])

    @staticmethod
    def _print_res_gen(gen_p, gen_costs):
        res_gen = pd.DataFrame()
        res_gen["gen_p_mw"] = gen_p
        res_gen["gen_cost_per_mw"] = gen_costs
        print(res_gen.to_string())


class StandardDCOPF(object):
    pass
