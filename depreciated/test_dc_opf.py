# import itertools
# import time
# import unittest
#
# import numpy as np
# import pandapower as pp
# import pandas as pd
#
# from lib.data_utils import indices_to_hot, hot_to_indices
# from lib.dc_opf import OPFCase3, OPFCase6, OPFRTECase5
# from lib.dc_opf import (
#     # StandardDCOPF,
#     LineSwitchingDCOPF,
#     TopologyOptimizationDCOPF,
# )
#
# from lib.dc_opf import StandardDCOPF
#
#
# class TestStandardDCOPF(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         print("\nStandard DC-OPF tests.\n")
#
#     def runner_opf(self, model, n_tests=20, eps=1e-4, verbose=False, tol=1e-9):
#         conditions = list()
#         for i in range(n_tests):
#             np.random.seed(i)
#             gen_cost = np.random.uniform(1.0, 5.0, (model.grid.gen.shape[0],))
#             model.set_gen_cost(gen_cost)
#             model.build_model()
#             result = model.solve_and_compare(verbose=verbose, tol=tol)
#
#             conditions.append(
#                 {
#                     "cost": np.less_equal(result["res_cost"]["diff"], eps).all(),
#                     "bus": np.less_equal(result["res_bus"]["diff"], eps).all(),
#                     "line": np.less_equal(result["res_line"]["diff"], eps).all(),
#                     "gen": np.less_equal(result["res_gen"]["diff"], eps).all(),
#                 }
#             )
#
#         conditions = pd.DataFrame(conditions)
#         conditions["passed"] = np.all(conditions.values, axis=-1)
#
#         print(f"\n\n{model.name}\n")
#         print(conditions.to_string())
#
#         time.sleep(0.1)
#         # Test DC Power Flow
#         self.assertTrue(conditions["passed"].values.all())
#
#     """
#     Test standard DC-OPF implementation.
#     """
#
#     def test_case3(self):
#         case = OPFCase3()
#         model = StandardDCOPF(
#             case.name,
#             case.grid,
#             base_unit_p=case.base_unit_p,
#             base_unit_v=case.base_unit_v,
#         )
#
#         self.runner_opf(model, verbose=True)
#
#     def test_case6(self):
#         case = OPFCase6()
#         model = StandardDCOPF(
#             case.name,
#             case.grid,
#             base_unit_p=case.base_unit_p,
#             base_unit_v=case.base_unit_v,
#         )
#
#         self.runner_opf(model)
#
#     def test_case3_by_value(self):
#         """
#         Test for power flow computation.
#         """
#         case = OPFCase3()
#         model = StandardDCOPF(
#             f"{case.name} BY VALUE",
#             case.grid,
#             base_unit_p=case.base_unit_p,
#             base_unit_v=case.base_unit_v,
#         )
#
#         model.set_gen_cost(np.array([1.0]))
#         model.build_model()
#
#         result = model.solve(verbose=True, tol=1e-9)
#         model.print_results()
#
#         time.sleep(0.1)
#         # Test DC Power Flow
#         self.assertTrue(
#             np.equal(
#                 result["res_bus"]["delta_pu"].values,
#                 np.array([0.0, -0.250, -0.375, 0.0, 0.0, 0.0]),
#             ).all()
#         )
#
#     def test_rte_case5(self):
#         case = OPFRTECase5()
#         model = StandardDCOPF(
#             case.name,
#             case.grid,
#             base_unit_p=case.base_unit_p,
#             base_unit_v=case.base_unit_v,
#         )
#
#         self.runner_opf(model, n_tests=5)
#
#     """
#         Infeasible problem.
#     """
#
#     # TODO: Resolve.
#     #
#     # def test_l2rpn2019(self):
#     #     case = OPFL2RPN2019()
#     #
#     #     model = StandardDCOPF(
#     #             case.name,
#     #             case.grid,
#     #             base_unit_p=case.base_unit_p,
#     #             base_unit_v=case.base_unit_v,
#     #         )
#     #
#     #     self.runner_opf(model, n_tests=5)
#
#
# class TestLineSwitchingDCOPF(unittest.TestCase):
#     """
#         Test DC-OPF with line status switching implementation.
#     """
#
#     @classmethod
#     def setUpClass(cls):
#         print("\nDC-OPF with line switching tests.\n")
#
#     def runner_opf_line_switching(
#         self, model, grid, n_max_line_status_changed, verbose=False, tol=1e-9
#     ):
#         np.random.seed(0)
#         gen_cost = np.random.uniform(1.0, 5.0, (model.grid.gen.shape[0],))
#         model.set_gen_cost(gen_cost)
#         model.build_model()
#
#         if verbose:
#             model.print_model()
#
#         # Construct all possible configurations
#         line_statuses = list()
#         for i in range(
#             n_max_line_status_changed + 1
#         ):  # Number of line disconnection 0, 1, ..., n
#             line_statuses.extend(
#                 [
#                     ~indices_to_hot(
#                         list(line_status), length=grid.line.shape[0], dtype=np.bool,
#                     )
#                     for line_status in itertools.combinations(grid.line.index, i)
#                 ]
#             )
#
#         results_backend = pd.DataFrame(
#             columns=["line_status", "objective", "loads_p", "generators_p", "valid"]
#         )
#         for line_status in line_statuses:
#             model.grid.line["in_service"] = line_status
#             result_backend = model.solve_backend()
#
#             objective = result_backend["res_cost"]
#
#             if model.solver_name == "gurobi":
#                 objective = (
#                     objective
#                     + np.square(
#                         result_backend["res_line"]["p_pu"] / model.line["max_p_pu"]
#                     ).sum()
#                 )
#
#             loads_p = grid.load["p_pu"].sum()
#             generators_p = result_backend["res_gen"]["p_pu"].sum()
#             valid = generators_p > loads_p - 1e-6 and result_backend["valid"]
#
#             results_backend = results_backend.append(
#                 {
#                     "line_status": tuple(line_status),
#                     "objective": np.round(objective, 2),
#                     "loads_p": loads_p,
#                     "generators_p": generators_p,
#                     "valid": valid,
#                 },
#                 ignore_index=True,
#             )
#
#         # Solve for optimal line status configuration
#         result = model.solve(verbose=verbose, tol=tol)
#         result_status = result["res_x"]
#         result_objective = result["res_cost"]
#         result_gap = result["res_gap"]  # Gap for finding the optimal configuration
#
#         if verbose:
#             model.print_results()
#
#         # Check with brute force solution
#         objective_brute = results_backend["objective"][results_backend["valid"]].min()
#         hot_brute = (
#             np.abs(results_backend["objective"].values - objective_brute) < result_gap
#         )
#         indices_brute = hot_to_indices(hot_brute)
#         status_brute = results_backend["line_status"][indices_brute]
#
#         match_idx = [
#             idx
#             for idx, line_status in zip(indices_brute, status_brute)
#             if np.equal(line_status, result_status).all()
#         ]
#
#         # Compare
#         results_backend["candidates"] = hot_brute
#         results_backend["result_objective"] = np.nan
#         results_backend["result_objective"][match_idx] = np.round(result_objective, 2)
#
#         results_backend["line_status"] = [
#             " ".join(np.array(line_status).astype(int).astype(str))
#             for line_status in results_backend["line_status"]
#         ]
#
#         print(f"\n{model.name}\n")
#         print(results_backend.to_string())
#
#         time.sleep(0.1)
#         self.assertTrue(bool(match_idx))
#
#     def test_case3_line_switching(self):
#         n_max_line_status_changed = 2
#
#         case = OPFCase3()
#         model = LineSwitchingDCOPF(
#             f"{case.name} Line Switching",
#             case.grid,
#             n_max_line_status_changed=n_max_line_status_changed,
#             base_unit_p=case.base_unit_p,
#             base_unit_v=case.base_unit_v,
#         )
#
#         self.runner_opf_line_switching(
#             model, case.grid, n_max_line_status_changed, verbose=True
#         )
#
#     def test_case6_line_switching(self):
#         n_max_line_status_changed = 2
#
#         case = OPFCase6()
#         model = LineSwitchingDCOPF(
#             f"{case.name} Line Switching",
#             case.grid,
#             n_max_line_status_changed=n_max_line_status_changed,
#             base_unit_p=case.base_unit_p,
#             base_unit_v=case.base_unit_v,
#         )
#
#         self.runner_opf_line_switching(
#             model, case.grid, n_max_line_status_changed, verbose=True
#         )
#
#     def test_rte_case5_line_switching(self):
#         n_max_line_status_changed = 3
#
#         case = OPFRTECase5()
#         model = LineSwitchingDCOPF(
#             f"{case.name} Line Switching",
#             case.grid,
#             n_max_line_status_changed=n_max_line_status_changed,
#             base_unit_p=case.base_unit_p,
#             base_unit_v=case.base_unit_v,
#         )
#
#         self.runner_opf_line_switching(
#             model, case.grid, n_max_line_status_changed, verbose=True,
#         )
#
#     """
#     Infeasible problem.
#     """
#
#     # TODO: Resolve.
#     #
#     # def test_l2rpn2019_line_switching(self):
#     #     n_max_line_status_changed = 2
#     #
#     #     case = OPFL2RPN2019()
#     #     model = LineSwitchingDCOPF(
#     #             f"{case.name} Line Switching",
#     #             case.grid,
#     #             n_max_line_status_changed=n_max_line_status_changed,
#     #             base_unit_p=case.base_unit_p,
#     #             base_unit_v=case.base_unit_v,
#     #     )
#     #
#     #     self.runner_opf_line_switching(
#     #         model, case.grid, n_max_line_status_changed, verbose=True
#     #     )
#
#
# class TestTopologyOptimizationDCOPF(unittest.TestCase):
#     """
#         Test DC-OPF with line status switching implementation.
#     """
#
#     @classmethod
#     def setUpClass(cls):
#         print("\nDC-OPF with topology optimization tests.\n")
#
#     @staticmethod
#     def topology_to_parts(x_topology, n_gen, n_load, n_line):
#         x_gen = x_topology[:n_gen]
#         x_load = x_topology[n_gen : (n_gen + n_load)]
#         x_line_or_1 = x_topology[(n_gen + n_load) : (n_gen + n_load + n_line)]
#         x_line_or_2 = x_topology[
#             (n_gen + n_load + n_line) : (n_gen + n_load + 2 * n_line)
#         ]
#         x_line_ex_1 = x_topology[
#             (n_gen + n_load + 2 * n_line) : (n_gen + n_load + 3 * n_line)
#         ]
#         x_line_ex_2 = x_topology[(n_gen + n_load + 3 * n_line) :]
#
#         return x_gen, x_load, x_line_or_1, x_line_or_2, x_line_ex_1, x_line_ex_2
#
#     def is_valid_topology(self, x_topology, n_gen, n_load, n_line):
#         (
#             x_gen,
#             x_load,
#             x_line_or_1,
#             x_line_or_2,
#             x_line_ex_1,
#             x_line_ex_2,
#         ) = self.topology_to_parts(x_topology, n_gen, n_load, n_line)
#
#         cond_line_or = np.less_equal(x_line_or_1 + x_line_or_2, 1).all()
#         cond_line_ex = np.less_equal(x_line_ex_1 + x_line_ex_2, 1).all()
#         cond_line_disconnected = np.equal(
#             x_line_or_1 + x_line_or_2, x_line_ex_1 + x_line_ex_2
#         ).all()
#
#         return cond_line_or and cond_line_ex and cond_line_disconnected
#
#     def runner_opf_topology_optimization(self, model, case, verbose=False, tol=1e-2):
#         """
#             SOLUTION.
#         """
#         np.random.seed(0)
#         gen_cost = np.random.uniform(1.0, 5.0, (model.grid.gen.shape[0],))
#         model.set_gen_cost(gen_cost)
#         model.build_model()
#
#         if verbose:
#             model.print_model()
#
#         if model.solver_name != "gurobi":
#             print("Solver does not support bilinear or quadratic terms.")
#             self.assertTrue(True)
#             return
#
#         result = model.solve(verbose=verbose, tol=tol)
#         result_x = result["res_x"]
#         result_objective = result["res_cost"]
#         result_gap = result["res_gap"]  # Gap for finding the optimal configuration
#
#         if verbose:
#             model.print_results()
#
#         n_gen = model.grid.gen.shape[0]
#         n_load = model.grid.load.shape[0]
#         n_line = model.grid.line.shape[0]
#
#         """
#             BACKEND BRUTE FORCE.
#         """
#
#         results_backend = []
#         for idx, x_topology in enumerate(
#             itertools.product([0, 1], repeat=n_gen + n_load + 4 * n_line)
#         ):
#             # x_topology = [x_gen, x_load, x_line_or_1, x_line_or_2, x_line_ex_1, x_line_ex_2]
#             x_topology = np.array(x_topology, dtype=np.int)
#
#             # Initialization of variables
#             (
#                 x_gen,
#                 x_load,
#                 x_line_or_1,
#                 x_line_or_2,
#                 x_line_ex_1,
#                 x_line_ex_2,
#             ) = self.topology_to_parts(x_topology, n_gen, n_load, n_line)
#
#             # Check valid topology
#             if self.is_valid_topology(x_topology, n_gen, n_load, n_line):
#                 # Generator bus
#                 gen_sub_bus = np.ones_like(x_gen)
#                 gen_sub_bus[x_gen.astype(np.bool)] = 2
#                 gen_bus = [
#                     case.grid.sub["bus"][sub_id][
#                         list(case.grid.sub["sub_bus"][sub_id]).index(sub_bus)
#                     ]
#                     for sub_bus, sub_id in zip(gen_sub_bus, case.grid.gen["sub"])
#                 ]
#
#                 # Load bus
#                 load_sub_bus = np.ones_like(x_load)
#                 load_sub_bus[x_load.astype(np.bool)] = 2
#                 load_bus = [
#                     case.grid.sub["bus"][sub_id][
#                         list(case.grid.sub["sub_bus"][sub_id]).index(sub_bus)
#                     ]
#                     for sub_bus, sub_id in zip(load_sub_bus, case.grid.load["sub"])
#                 ]
#
#                 # Power line status
#                 line_status = np.logical_and(
#                     np.logical_or(x_line_or_1, x_line_or_2),
#                     np.logical_or(x_line_ex_1, x_line_ex_2),
#                 )
#
#                 # Power line - Origin bus
#                 line_or_sub_bus = -np.ones_like(x_line_or_1)
#                 line_or_sub_bus[x_line_or_1.astype(np.bool)] = 1
#                 line_or_sub_bus[x_line_or_2.astype(np.bool)] = 2
#                 line_or_bus = [
#                     case.grid.sub["bus"][sub_id][
#                         list(case.grid.sub["sub_bus"][sub_id]).index(sub_bus)
#                     ]
#                     if sub_bus != -1
#                     else case.grid.sub["bus"][sub_id][0]
#                     for sub_bus, sub_id in zip(
#                         line_or_sub_bus, case.grid.line["from_sub"]
#                     )
#                 ]
#
#                 # Power line - Extremity bus
#                 line_ex_sub_bus = -np.ones_like(x_line_ex_1)
#                 line_ex_sub_bus[x_line_ex_1.astype(np.bool)] = 1
#                 line_ex_sub_bus[x_line_ex_2.astype(np.bool)] = 2
#                 line_ex_bus = [
#                     case.grid.sub["bus"][sub_id][
#                         list(case.grid.sub["sub_bus"][sub_id]).index(sub_bus)
#                     ]
#                     if sub_bus != -1
#                     else case.grid.sub["bus"][sub_id][0]
#                     for sub_bus, sub_id in zip(
#                         line_ex_sub_bus, case.grid.line["to_sub"]
#                     )
#                 ]
#
#                 # Construct grid for backend
#                 grid_tmp = case.grid.deepcopy()
#                 grid_tmp.line["in_service"] = line_status
#                 grid_tmp.gen["bus"] = gen_bus
#                 grid_tmp.load["bus"] = load_bus
#                 grid_tmp.line["from_bus"] = line_or_bus
#                 grid_tmp.line["to_bus"] = line_ex_bus
#
#                 for gen_id in grid_tmp.gen.index.values:
#                     pp.create_poly_cost(
#                         grid_tmp,
#                         gen_id,
#                         "gen",
#                         cp1_eur_per_mw=case.convert_per_unit_to_mw(
#                             grid_tmp.gen["cost_pu"][gen_id]
#                         ),
#                     )
#
#                 print(f"{len(results_backend) + 1}/{idx}: Running DC-OPF ...")
#                 try:
#                     pp.rundcopp(grid_tmp)
#                     valid = True
#                 except (pp.optimal_powerflow.OPFNotConverged, IndexError) as e:
#                     grid_tmp.res_cost = 0.0
#                     print(e)
#                     continue
#
#                 load_p = case.convert_mw_to_per_unit(grid_tmp.load["p_mw"].sum())
#                 gen_p = case.convert_mw_to_per_unit(grid_tmp.res_gen["p_mw"].sum())
#                 valid = valid and np.abs(gen_p - load_p) < 1e-6
#
#                 objective = grid_tmp.res_cost
#
#                 if model.solver_name == "gurobi":
#                     objective = (
#                         objective
#                         + +np.square(
#                             case.convert_mw_to_per_unit(grid_tmp.res_line["p_from_mw"])
#                             / grid_tmp.line["max_p_pu"]
#                         ).sum()
#                     )
#
#                 results_backend.append(
#                     {
#                         "x": np.concatenate(
#                             (
#                                 x_gen,
#                                 x_load,
#                                 x_line_or_1,
#                                 x_line_or_2,
#                                 x_line_ex_1,
#                                 x_line_ex_2,
#                             )
#                         ),
#                         "gen_bus": gen_bus,
#                         "load_bus": load_bus,
#                         "line_or_bus": line_or_bus,
#                         "line_ex_bus": line_ex_bus,
#                         "line_status": line_status.astype(int),
#                         "valid": valid,
#                         "objective": np.round(objective, 2),
#                         "load_p": load_p,
#                         "gen_p": np.round(gen_p, 2),
#                     }
#                 )
#
#         results_backend = pd.DataFrame(results_backend)
#
#         # Check with brute force solution
#         objective_brute = results_backend["objective"][results_backend["valid"]].min()
#         hot_brute = (
#             np.abs(results_backend["objective"].values - objective_brute) < result_gap
#         )
#         indices_brute = hot_to_indices(hot_brute)
#         status_brute = results_backend["x"][indices_brute]
#
#         match_idx = [
#             idx
#             for idx, line_status in zip(indices_brute, status_brute)
#             if np.equal(line_status, result_x).all()
#         ]
#
#         # Compare
#         results_backend["candidates"] = hot_brute
#         results_backend["result_objective"] = np.nan
#         results_backend["result_objective"][match_idx] = np.round(result_objective, 2)
#
#         if verbose:
#             model.print_results()
#
#         print(
#             results_backend[
#                 [
#                     "gen_bus",
#                     "load_bus",
#                     "line_or_bus",
#                     "line_ex_bus",
#                     "line_status",
#                     "load_p",
#                     "gen_p",
#                     "valid",
#                     "candidates",
#                     "objective",
#                     "result_objective",
#                 ]
#             ][results_backend["candidates"]].to_string()
#         )
#
#         time.sleep(0.1)
#         self.assertTrue(bool(match_idx))
#
#     def test_case3_topology(self):
#         case = OPFCase3()
#         model = TopologyOptimizationDCOPF(
#             f"{case.name} Topology Optimization",
#             case.grid,
#             base_unit_p=case.base_unit_p,
#             base_unit_v=case.base_unit_v,
#         )
#
#         self.runner_opf_topology_optimization(model, case)
#
#     """
#         INTRACTABLE.
#     """
#
#     # def test_case6_topology(self):
#     #     case = OPFCase6()
#     #     model = TopologyOptimizationDCOPF(
#     #         f"{case.name} Topology Optimization",
#     #         case.grid,
#     #         base_unit_p=case.base_unit_p,
#     #         base_unit_v=case.base_unit_v,
#     #     )
#     #
#     #     self.runner_opf_topology_optimization(model, case, verbose=True)
#     #
#     # def test_rte_case5_topology(self):
#     #     case = OPFRTECase5()
#     #     model = TopologyOptimizationDCOPF(
#     #         f"{case.name} Topology Optimization",
#     #         case.grid,
#     #         base_unit_p=case.base_unit_p,
#     #         base_unit_v=case.base_unit_v,
#     #     )
#     #
#     #     self.runner_opf_topology_optimization(model, case, verbose=True)
