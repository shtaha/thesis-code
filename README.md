# MSc Thesis by Rok Šikonja

## Install
    
    pip install grid2op
    pip install l2rpn-baselines
    pip install imageio  # Visualization
    
    pip install numpy matplotlib tensorflow jupyter pygame
    pip install pyomo
    
    pip freeze > requirements-win-10.txt
    pip freeze > requirements-ubuntu-20-04-LTS.txt

## WSL
    
    git clone https://github.com/roksikonja/thesis-code.git


## Units

[Units and conversion](https://pandapower.readthedocs.io/en/v2.2.2/elements/line.html)

## Tests
    
    python -m unittest discover <directory>
    python -m unittest discover tests