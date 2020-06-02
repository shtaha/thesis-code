import numpy as np
import pandapower as pp
import pyomo.environ as pyo
import pyomo.opt as pyo_opt


class DCOptimalPowerFlow(object):
    """
    Constructor for DC-OPF given the environment and the observation.
    """

    def __init__(self, env, solver="gurobi"):
        """
        Initialize DC-OPF Parameters given an environment: information on topology, generators, loads, and lines.
        """
        self.env = env
        self.backend = env.backend
        self.grid = env.backend._grid

        self.bus = self.grid.bus
        self.gen = self.grid.gen
        self.load = self.grid.load
        self.line = self.grid.line

        # Base units
        self.unit_p = 1000000  # Base unit is 1 MVA or 1 MW or 1 MVar
        self.unit_a = 1.0  # Base unit is 1 A
        self.unit_v = 1000  # Base unit is 1 kV

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
        self.gen_p_max = None
        self.gen_p_min = None
        self.gen_ids_to_bus_ids = None  # VARIABLE
        self.bus_ids_to_gen_ids = None  # VARIABLE

        # Load parameters
        self.n_load = None
        self.load_ids = None
        self.load_ids_to_bus_ids = None  # VARIABLE
        self.load_p = None  # VARIABLE
        self.bus_ids_to_load_ids = None  # VARIABLE

        # Line parameters
        self.n_line = None
        self.line_ids = None
        self.line_resistance = None
        self.line_reactance = None
        self.line_inverse_reactance = None
        self.line_i_max = None
        self.line_p_max = None
        self.line_ids_to_bus_ids = None  # VARIABLE

        # Initialize parameter values
        self.initialize_grid()
        self.initialize_generators()
        self.initialize_loads()
        self.initialize_lines()

        # DC-OPF model
        self.model = None

        # Initialize DC-OPF model and solver
        self.initialize_model()
        self.solver = pyo_opt.SolverFactory(solver)

        self.print_model()

    def print_model(self):
        # print(self.model.display())
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
        self.model.constraint_line_p = pyo.Constraint(self.model.gen_set, rule=self._constraint_line_p)

        # Define line power flow
        self.model.constraint_line_p_def = pyo.Constraint(self.model.gen_set, rule=self._constraint_line_p_def)

        # Set delta[0] = 0
        self.model.constraint_delta = pyo.Constraint(self.model.bus_set, rule=self._constraint_delta)

        # Define bus generator injections
        self.model.constrain_bus_gen_p = pyo.Constraint(self.model.bus_set, rule=self._constraint_bus_gen_p)

        # Bus power balance constraints
        self.model.constraint_bus_balance_p = pyo.Constraint(self.model.bus_set, rule=self._constraint_bus_balance_p)

    def update(self, obs):
        """
        Update DC-OPF parameters given a new observation.

        That is: grid topology,
        """
        pass

    def dc_opf_backend(self, **kwargs):
        pp.rundcopp(self.grid, **kwargs)

    def initialize_grid(self):
        self.n_bus = self.bus.shape[0]
        self.bus_ids = self.bus.index.values  # [0, 1, ..., n_bus-1] (n_bus, )

        # self.n_sub = self.env.sub_info.shape[0]
        self.n_sub = self.env.n_sub
        self.sub_ids = np.arange(0, self.n_sub)  # [0, 1, ..., n_sub-1] (n_sub, )
        self.bus_ids_to_sub_ids = np.tile(self.sub_ids, 2)  # [0, 1, ..., n_sub-1, 0, 1, ..., n_sub-1] (n_bus, )
        self.bus_ids_to_sub_bus_ids = np.tile([1, 2], self.n_sub)  # [1, 2, 1, 2, ..., 1, 2] (n_sub, )

    def initialize_generators(self):
        self.n_gen = self.env.n_gen
        self.gen_ids = self.gen.index.values

        self.gen_p_max = self.env.gen_pmax
        self.gen_p_min = self.env.gen_pmin

        self.gen_ids_to_bus_ids = self.gen["bus"].values

        self.bus_ids_to_gen_ids = [list(np.equal(self.gen_ids_to_bus_ids, bus_id).nonzero()[0]) for bus_id in
                                   self.bus_ids]

    def initialize_loads(self):
        self.n_load = self.env.n_load
        self.load_ids = self.load.index.values

        self.load_p = self.load["p_mw"].values

        self.load_ids_to_bus_ids = self.load["bus"].values

        self.bus_ids_to_load_ids = [list(np.equal(self.load_ids_to_bus_ids, bus_id).nonzero()[0]) for bus_id in
                                    self.bus_ids]

    def initialize_lines(self):
        self.n_line = self.env.n_line
        self.line_ids = self.line.index.values

        self.line_resistance = np.divide(np.multiply(self.line["r_ohm_per_km"].values, self.line["length_km"]),
                                         self.line["parallel"])

        self.line_reactance = np.divide(np.multiply(self.line["x_ohm_per_km"].values, self.line["length_km"]),
                                        self.line["parallel"])  # Reactance in Ohms
        self.line_inverse_reactance = np.divide(1, self.line_reactance)  # Negative susceptance in Siemens/Mhos

        self.line_i_max = self.env.get_thermal_limit() / self.unit_a  # Line current limit in Amperes
        self.line_p_max = np.multiply(np.square(self.line_i_max * self.unit_a),
                                      self.line_resistance) / self.unit_p  # Line thermal limit in MW

        # [(bus_or, bus_ex), ..., (bus_or, bus_ex)] (n_line, )
        self.line_ids_to_bus_ids = list(zip(self.line["from_bus"], self.line["to_bus"]))

    """
        Helpers.
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
            return pyo.Constraint.Skip

    @staticmethod
    def _constraint_bus_balance_p(model, bus_id):
        return model.bus_gen_p[bus_id] - model.bus_load_p[bus_id] == \
            sum([model.line_p[line_id] for line_id in model.line_set if bus_id == model.line_ids_to_bus_ids[line_id][0]]) + \
            sum([model.line_p[line_id] for line_id in model.line_set if bus_id == model.line_ids_to_bus_ids[line_id][1]])

    @staticmethod
    def _create_map_ids_to_values_sum(ids, sum_ids, values):
        return {idx: values[sum_ids[idx]].sum() for idx in ids}

    @staticmethod
    def _create_map_ids_to_values(ids, values):
        return {idx: value for idx, value in zip(ids, values)}


def update_backend(env, verbose=False):
    """
    Update backend grid with missing data.
    """
    grid = env.backend._grid

    if env.name == "rte_case5_example":
        # Loads
        grid.load["controllable"] = False

        # Generators
        grid.gen["controllable"] = True
        grid.gen["min_p_mw"] = env.gen_pmin
        grid.gen["max_p_mw"] = env.gen_pmax

        # Additional data
        # Not used for the time being
        # grid.gen["type"] = env.gen_type
        # grid.gen["gen_redispatchable"] = env.gen_redispatchable
        # grid.gen["gen_max_ramp_up"] = env.gen_max_ramp_up
        # grid.gen["gen_max_ramp_down"] = env.gen_max_ramp_down
        # grid.gen["gen_min_uptime"] = env.gen_min_uptime
        # grid.gen["gen_min_downtime"] = env.gen_min_downtime

    if verbose:
        print(env.name.upper())
        print("bus\n" + grid.bus.to_string())
        print("gen\n" + grid.gen.to_string())
        print("load\n" + grid.load.to_string())
        print("line\n" + grid.line.to_string())


