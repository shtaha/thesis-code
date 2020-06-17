import numpy as np
import grid2op
from lib.dc_opf.cases import OPFCase3, OPFCase6
from lib.dc_opf.models import StandardDCOPF, LineSwitchingDCOPF
from lib.data_utils import update_backend


if __name__ == "__main__":
    """
        CASE 3
    """

    case3 = OPFCase3()
    test_opf = StandardDCOPF(
        "CASE 3",
        case3.grid,
        base_unit_p=case3.base_unit_p,
        base_unit_v=case3.base_unit_v,
    )

    # Set generator costs
    gen_cost = np.random.uniform(1.0, 1.5, (case3.grid.gen.shape[0],))
    test_opf.set_gen_cost(gen_cost)

    test_opf.build_model()

    print(f"\n\n{test_opf.name}\n")
    test_opf.print_per_unit_grid()
    test_opf.print_model()

    # Solve OPFs
    test_opf.solve()
    test_opf.solve_backend()

    # Print results
    test_opf.print_results()
    test_opf.print_results_backend()

    """
    CASE 6
    """

    case6 = OPFCase6()
    test_opf = LineSwitchingDCOPF(
        "CASE 3",
        case6.grid,
        n_line_status_changes=0,
        base_unit_p=case6.base_unit_p,
        base_unit_v=case6.base_unit_v,
    )

    # Set generator costs
    gen_cost = np.random.uniform(1.0, 1.5, (case6.grid.gen.shape[0],))
    test_opf.set_gen_cost(gen_cost)

    test_opf.build_model(big_m=True)

    print(f"\n\n{test_opf.name}\n")
    test_opf.print_per_unit_grid()
    test_opf.print_model()

    # Solve OPFs
    test_opf.solve()
    test_opf.solve_backend()

    # Print results
    test_opf.print_results()
    test_opf.print_results_backend()

    # Compare with backend
    test_opf.solve_and_compare(verbose=True)

    """
        RTE CASE 5
    """
    env = grid2op.make(dataset="rte_case5_example")
    update_backend(env)
    grid = env.backend._grid

    model_opf = LineSwitchingDCOPF(
        "RTE CASE 5 Line Switching",
        grid,
        n_line_status_changes=0,
        base_unit_p=1e6,
        base_unit_v=1e5,
    )

    # Set generator costs
    gen_cost = np.random.uniform(1.0, 1.5, (case6.grid.gen.shape[0],))
    test_opf.set_gen_cost(gen_cost)

    test_opf.build_model(big_m=True)

    print(f"\n\n{model_opf.name}\n")
    test_opf.print_per_unit_grid()
    test_opf.print_model()

    # Solve OPFs
    test_opf.solve()
    test_opf.solve_backend()

    # Print results
    test_opf.print_results()
    test_opf.print_results_backend()

    # Compare with backend
    test_opf.solve_and_compare(verbose=True)

    # TODO: DEVELOPMENT
    # TODO: CORRECT GRID -> gen_p_min >= 0
    # env = grid2op.make(dataset="l2rpn_2019")
    # env = grid2op.make(dataset="rte_case5_example")
    # grid = env.backend._grid
    # print_environment_attributes(env)
    #
    # update_backend(env, verbose=True)

    # test_opf = StandardDCOPF("L2RPN 2019", env.backend._grid, base_unit_p=1e6, base_unit_v=100000.0)
    # test_opf.build_model()
    # test_opf.print_per_unit_grid()
    #
    # test_opf.solve_backend()
    # test_opf.print_results_backend()
