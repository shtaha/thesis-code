# MSc Thesis by Rok Šikonja

    git clone https://github.com/roksikonja/thesis-code.git

## Install
    
    python -m pip install --upgrade pip
    
    pip install numpy numba matplotlib pandas seaborn PyPDF2 # Basics     
    pip install grid2op

    pip install pyomo
    pip install glpk mosek gurobi # Solvers
    
    pip jupyter pygame
    pip install tensorflow-addons
    pip install graph_nets "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability --use-feature=2020-resolver
    
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
    sudo apt-get install liblzma-dev
    sudo apt-mark hold libc6
    
    # Ubuntu
    sudo apt-get install glpk-utils
    glpsol --version  # Check
   
    cd /mnt/c/Users/chrosik/data_grid2op
    cp -R rte_case5_example ~/data_grid2op/rte_case5_example
    cp -R l2rpn_2019 ~/data_grid2op/l2rpn_2019
   
    cd /mnt/c/Users/chrosik
    cp -R -v data_grid2op ~/

    
# Deep Mind Graph Networks

    pip install grid2op matplotlib numba
    pip install jupyter
    pip install graph_nets "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability --use-feature=2020-resolver


# Microsoft GNNs

    https://github.com/microsoft/tf2-gnn

    pip install tf2_gnn
    
    python -m pip install --upgrade pip
    pip check  # Check dependecy conflicts
    
    pip install <package> --use-feature=2020-resolver
    
# TensorBoard

    tensorboard --logdir=/full_path_to_your_logs
    
# GPU
    High scale NNs
    Look ahead multiple steps ahead
    
    15 minutes-generator schedules
    
    # S
    
# Colab

    https://github.com/roksikonja/thesis-code
    
- JP
    - Request.
        - Document ML approach: even failure attempts, assess techniques.
            - Concise, -- try to handle readability.
        - Final presentation: virtual.
            - MILP: anaylsis, do extensive.
        - Next student start of november.
            - Make a future work section.
            - Minimal set of files for MIP oracle.
            - Set up a simulator.
        - Hitachi ABB
        - AI ETH Research Center: Applications for PhD fellowships.


        None :   True : minimize : (f_line[0] + line_flow[0]**2/33809.32542390213 + f_line[1] + line_flow[1]**2/5441.558780921856 + f_line[2] + line_flow[2]**2/5605.348233562487 + f_line[3] + line_flow[3]**2/4281.296599656867 + f_line[4] + line_flow[4]**2/1463.5078578307875 + f_line[5] + line_flow[5]**2/6801.8938170537 + f_line[6] + line_flow[6]**2/2734.3080266859033 + f_line[7] + line_flow[7]**2/2986.2069856597955 + f_line[8] + line_flow[8]**2/675.0000191299623 + f_line[9] + line_flow[9]**2/2000.0173745090287 + f_line[10] + line_flow[10]**2/516.4283297001966 + f_line[11] + line_flow[11]**2/323.2331965413323 + f_line[12] + line_flow[12]**2/1410.5640067731874 + f_line[13] + line_flow[13]**2/4574.707089369562 + f_line[14] + line_flow[14]**2/3753.1103585995734 + f_line[15] + line_flow[15]**2/1345.7770108751865 + f_line[16] + line_flow[16]**2/919.8002564977505 + f_line[17] + line_flow[17]**2/783.4368839892013 + f_line[18] + line_flow[18]**2/300.0000305263711 + f_line[19] + line_flow[19]**2/723.5427679801942)/20 + 100.0*mu_gen + 0.05*(x_line_status_switch[0] + x_line_status_switch[1] + x_line_status_switch[2] + x_line_status_switch[3] + x_line_status_switch[4] + x_line_status_switch[5] + x_line_status_switch[6] + x_line_status_switch[7] + x_line_status_switch[8] + x_line_status_switch[9] + x_line_status_switch[10] + x_line_status_switch[11] + x_line_status_switch[12] + x_line_status_switch[13] + x_line_status_switch[14] + x_line_status_switch[15] + x_line_status_switch[16] + x_line_status_switch[17] + x_line_status_switch[18] + x_line_status_switch[19] + x_substation_topology_switch[0] + x_substation_topology_switch[1] + x_substation_topology_switch[2] + x_substation_topology_switch[3] + x_substation_topology_switch[4] + x_substation_topology_switch[5] + x_substation_topology_switch[6] + x_substation_topology_switch[7] + x_substation_topology_switch[8] + x_substation_topology_switch[9] + x_substation_topology_switch[10] + x_substation_topology_switch[11] + x_substation_topology_switch[12] + x_substation_topology_switch[13])

        None :   True : minimize : mu_max + 100.0*mu_gen + 0.05*(x_line_status_switch[0] + x_line_status_switch[1] + x_line_status_switch[2] + x_line_status_switch[3] + x_line_status_switch[4] + x_line_status_switch[5] + x_line_status_switch[6] + x_line_status_switch[7] + x_line_status_switch[8] + x_line_status_switch[9] + x_line_status_switch[10] + x_line_status_switch[11] + x_line_status_switch[12] + x_line_status_switch[13] + x_line_status_switch[14] + x_line_status_switch[15] + x_line_status_switch[16] + x_line_status_switch[17] + x_line_status_switch[18] + x_line_status_switch[19] + x_substation_topology_switch[0] + x_substation_topology_switch[1] + x_substation_topology_switch[2] + x_substation_topology_switch[3] + x_substation_topology_switch[4] + x_substation_topology_switch[5] + x_substation_topology_switch[6] + x_substation_topology_switch[7] + x_substation_topology_switch[8] + x_substation_topology_switch[9] + x_substation_topology_switch[10] + x_substation_topology_switch[11] + x_substation_topology_switch[12] + x_substation_topology_switch[13])
