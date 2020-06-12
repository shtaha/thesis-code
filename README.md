# MSc Thesis by Rok Šikonja

## Install
    
    pip install grid2op
    pip install l2rpn-baselines
    pip install imageio  # Visualization
    
    pip install numpy matplotlib tensorflow jupyter pygame
    pip install pyomo
    
    pip freeze > requirements.txt

## Units

[Units and conversion](https://pandapower.readthedocs.io/en/v2.2.2/elements/line.html)

    self._info_to_units = {
                "rho": "%",
                "a": "A", (a_or, a_ex)
                "p": "MW",
                "q": "MVar",
                "v":"kV"
    }

    S_N = 1 MVA
    V_N is the nominal voltage at a bus in kV
    
    Z_N = V_N^2/S_N
    