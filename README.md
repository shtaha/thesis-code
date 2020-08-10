# MSc Thesis by Rok Šikonja

    git clone https://github.com/roksikonja/thesis-code.git

## Install
    
    pip install numpy numba matplotlib pandas seaborn  # Basics     
    pip install grid2op

    pip install pyomo
    pip install glpk mosek  # Solvers
    
    pip tensorflow jupyter pygame
    
    pip install l2rpn-baselines
    pip install imageio
        
    pip freeze > requirements-win-10.txt
    pip freeze > requirements-ubuntu-20-04-LTS.txt

    
## Units

[Units and conversion](https://pandapower.readthedocs.io/en/v2.2.2/elements/line.html)


## Tests
    black . --exclude="./venv"
    
    python -m unittest discover <directory>
    python -m unittest discover tests
    
## GLPK

### Windows
    
    # Install Microsoft Visual Studio C++ 14.0
    # http://www.osemosys.org/uploads/1/8/5/0/18504136/glpk_installation_guide_for_windows10_-_201702.pdf
    # http://winglpk.sourceforge.net/
    # Download latest 
    # https://sourceforge.net/projects/winglpk/files/latest/download
    # https://sourceforge.net/projects/winglpk/
    # Unzip and copy to C:\glpk-X.Y
    # Add C:\glpk-X.Y\w64 to System PATH
    glpsol --help  # Check
    
### Linux WSL
    
    # Fix issue - Build manually
    sudo add-apt-repository ppa:rafaeldtinoco/lp1871129
    sudo apt update
    sudo apt install libc6=2.31-0ubuntu8+lp1871129~1 libc6-dev=2.31-0ubuntu8+lp1871129~1 libc-dev-bin=2.31-0ubuntu8+lp1871129~1 -y --allow-downgrades
    sudo apt-mark hold libc6
    
    # Ubuntu
    sudo apt-get install glpk-utils
    glpsol --version  # Check
   