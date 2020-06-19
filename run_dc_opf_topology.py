from lib.dc_opf.cases import OPFCase6
from lib.dc_opf.models import TopologyOptimizationDCOPF

if __name__ == "__main__":
    # env = grid2op.make(dataset="rte_case5_example")
    # update_backend(env)
    # grid = env.backend._grid
    #
    # print(grid.bus.to_string())
    # print(grid.line.to_string())
    #
    # model_opf = TopologyOptimizationDCOPF(
    #     "RTE CASE 5 Topology Optimization",
    #     grid,
    #     base_unit_p=1e6,
    #     base_unit_v=1e5,
    # )
    case6 = OPFCase6()
    model_opf = TopologyOptimizationDCOPF(
        "CASE 6",
        case6.grid,
        base_unit_p=case6.base_unit_p,
        base_unit_v=case6.base_unit_v,
    )
    model_opf.build_model()
    model_opf.print_model()
