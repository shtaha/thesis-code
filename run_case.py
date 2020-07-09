import numpy as np
import pandas as pd

from lib.dc_opf import StandardDCOPF, GridDCOPF, load_case
from lib.visualizer import describe_environment

case = load_case("l2rpn2020")

"""
    LOADED ENVIRONMENT.
    -> case.grid_org (not used in computations)
"""
print("\n\nLOADED\n\n")
describe_environment(case.env)
print(case.grid_org)
print("BUS\n" + case.grid_org.bus.to_string())
print("GEN\n" + case.grid_org.gen.to_string())
print("LOAD\n" + case.grid_org.load.to_string())
print("LINE\n" + case.grid_org.line.to_string())
print("EXT GRID\n" + case.grid_org.ext_grid.to_string())
print("TRAFO\n" + case.grid_org.trafo.to_string())
print("SHUNT\n" + case.grid_org.shunt.to_string())
print("BUS GEODATA\n" + case.grid_org.bus_geodata.to_string())

"""
    BACKEND COMPUTATION ENVIRONMENT.
    -> case.grid_backend
    With updated:
        - bus: name
        - gen: type, max_p_mw, min_p_mw, max_ramp_up, min_ramp_down, redispatchable, min_uptime, min_downtime
        - line: max_i_ka
        - trafo: b_pu, max_p_pu
"""
print("\n\nBACKEND\n\n")
print(case.grid_backend)
print("BUS\n" + case.grid_backend.bus.to_string())
print("GEN\n" + case.grid_backend.gen.to_string())
print("LOAD\n" + case.grid_backend.load.to_string())
print("LINE\n" + case.grid_backend.line.to_string())
print("EXT GRID\n" + case.grid_backend.ext_grid.to_string())
print("TRAFO\n" + case.grid_backend.trafo.to_string())
print("SHUNT\n" + case.grid_backend.shunt.to_string())
print("BUS GEODATA\n" + case.grid_backend.bus_geodata.to_string())


# Check
for line_id in case.grid_backend.line.index:
    bus_or = case.grid_backend.line["from_bus"][line_id]
    bus_ex = case.grid_backend.line["to_bus"][line_id]

    vn_or = case.grid_backend.bus["vn_kv"][bus_or]
    vn_ex = case.grid_backend.bus["vn_kv"][bus_ex]

    if vn_or != vn_ex:
        raise EnvironmentError("Power line connecting buses with different nominal voltages.")

"""
    CONSTRUCT DC-OPF GRID.
    All quantities are given per unit.
    Constructed from BACKEND GRID.
"""

grid_pu = GridDCOPF(case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p)
grid_pu.print_grid()

"""
    CONSTRUCT DC-OPF OPTIMIZATION PROBLEM.
"""
model = StandardDCOPF(
    f"{case.name} Standard DC OPF",
    grid=grid_pu,
    grid_backend=case.grid_backend,
    base_unit_p=case.base_unit_p,
    base_unit_v=case.base_unit_v,
)

np.random.seed(0)
model.grid.gen["cost_pu"] = np.random.uniform(0.5, 5.0, model.grid.gen.shape[0])
model.grid.line["b_pu"][[45, 46, 47]] = [5000.0, 1014.19878296, 3311.25827815]
model.build_model()
model.print_model()

"""
    SOLVE OPTIMIZATION PROBLEM.
"""

model.solve_and_compare(verbose=True)

print(model.grid_backend)
# BUS - PARAMETERS + RESULTS
print(model.grid_backend.bus.to_string())
print(model.grid.bus.to_string())
print(model.grid_backend.res_bus.to_string())
print(model.res_bus.to_string())

# GEN - PARAMETERS + RESULTS
print(model.grid_backend.gen.to_string())
print(model.grid.gen.to_string())
print(model.grid_backend.res_gen.to_string())
print(model.res_gen.to_string())

# LOAD - PARAMETERS + RESULTS
print(model.grid_backend.load.to_string())
print(model.grid.load.to_string())
print(model.grid_backend.res_load.to_string())

# LINE
print(model.grid_backend.line.to_string())
print(model.grid.line.to_string())
print(model.grid_backend.res_line.to_string())
print(model.res_line.to_string())

# EXT GRID
print(model.grid_backend.ext_grid.to_string())
print(model.grid.ext_grid.to_string())
print(model.grid_backend.res_ext_grid.to_string())
print(model.res_ext_grid.to_string())

# TRAFO
print(model.grid_backend.ext_grid.to_string())
print(model.grid.trafo.to_string())
print(model.grid_backend.res_trafo.to_string())
print(model.res_trafo.to_string())

# SHUNT - PARAMETERS (OUT OF SERVICE) + RESULTS
print(model.grid_backend.shunt.to_string())
print(model.grid_backend.res_shunt.to_string())

"""
    COMPARE PARAMETERS.
        - bus: v_pu -> OK
        - gen: max_p_pu, min_p_pu, cost_pu, p_pu -> OK
        - load: p_pu -> OK
        - line: b_pu, max_p_pu, status, p_pu
            - b_pu -> OK: 
                x = length_km * x_ohm_per_km / parallel
                x_pu = ohm_to_pu(x)
                b_pu = 1 / x_pu
            - max_p_pu -> OK:
                max_i_pu = ka_to_pu(max_i_ka)
                max_p_pu = sqrt(3) * max_i_pu * v_pu_from_bus
            - p_pu  -> OK
            - status  -> OK
        - ext_grid: p_pu, min_p_pu, max_p_pu -> OK
        - trafo: b_pu, max_p_pu, status, p_pu
                - b_pu -> OK: 
                    x = length_km * x_ohm_per_km / parallel
                    x_pu = ohm_to_pu(x)
                    b_pu = 1 / x_pu
                - max_p_pu -> OK:
                    max_i_pu = ka_to_pu(max_i_ka)
                    max_p_pu = sqrt(3) * max_i_pu * v_pu_from_bus
                - p_pu  -> OK
                - status  -> OK
"""

model.solve_backend()

"""
    POWER LINES.
"""
bus_or = model.grid_backend.line["from_bus"].values
bus_ex = model.grid_backend.line["to_bus"].values

delta_or = model.convert_degree_to_rad(model.grid_backend.res_bus["va_degree"][bus_or]).values
delta_ex = model.convert_degree_to_rad(model.grid_backend.res_bus["va_degree"][bus_ex]).values
p_pu = model.convert_mw_to_per_unit(model.grid_backend.res_line["p_from_mw"]).values
max_p_pu = np.abs(p_pu) / (model.grid_backend.res_line["loading_percent"].values / 100.0)

# p_pu = b_pu * (delta_or - delta_ex)
diff_line = pd.DataFrame()
diff_line["b_pu"] = model.grid.line["b_pu"][~model.grid.line.trafo]
diff_line["b_b_pu"] = p_pu / (delta_or - delta_ex)
diff_line["diff_b_pu"] = diff_line["b_pu"] - diff_line["b_b_pu"]
diff_line["max_p_pu"] = model.grid.line["max_p_pu"][~model.grid.line.trafo]
diff_line["b_max_p_pu"] = max_p_pu
diff_line["diff_max_p_pu"] = diff_line["max_p_pu"] - diff_line["b_max_p_pu"]
print(diff_line.to_string())

"""
    TRANSFORMERS.
"""
bus_or = model.grid_backend.trafo["hv_bus"].values
bus_ex = model.grid_backend.trafo["lv_bus"].values

delta_or = model.convert_degree_to_rad(model.grid_backend.res_bus["va_degree"][bus_or]).values
delta_ex = model.convert_degree_to_rad(model.grid_backend.res_bus["va_degree"][bus_ex]).values
p_pu = model.convert_mw_to_per_unit(model.grid_backend.res_trafo["p_hv_mw"]).values
max_p_pu = np.abs(p_pu) / np.abs(model.grid_backend.res_trafo["loading_percent"].values / 100.0)

# p_pu = b_pu * (delta_or - delta_ex)
diff_trafo = pd.DataFrame()
diff_trafo["b_pu"] = model.grid.trafo["b_pu"]
diff_trafo["b_b_pu"] = p_pu / (delta_or - delta_ex)
diff_trafo["diff"] = diff_trafo["b_pu"] - diff_trafo["b_b_pu"]
diff_trafo["max_p_pu"] = model.grid.trafo["max_p_pu"]
diff_trafo["b_max_p_pu"] = max_p_pu
diff_trafo["diff_max_p_pu"] = diff_trafo["max_p_pu"] - diff_trafo["b_max_p_pu"]
print(diff_trafo.to_string())
