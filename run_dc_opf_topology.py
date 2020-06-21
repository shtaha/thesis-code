from lib.dc_opf.cases import OPFCase3
from lib.dc_opf.models import TopologyOptimizationDCOPF

if __name__ == "__main__":
    case = OPFCase3()
    model_opf = TopologyOptimizationDCOPF(
        f"{case.name} with topology optimization",
        case.grid,
        base_unit_p=case.base_unit_p,
        base_unit_v=case.base_unit_v,
    )
    model_opf.build_model()
    model_opf.print_model()
