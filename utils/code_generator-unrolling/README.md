# Automaic code generator for Tiramisu

Generates random Tiramisu codes. The program currently generates two types of codes :
* Sequence of computations which can be : simple assignments, assignments with other computations or stencils.
* Sequence of convolution layers. With two types of padding : same padding (adding 0 padding so that the input and the output layers have the same height and width), and valid padding (no padding).

# Running the generator
## Generator parameters
```
cd cmake-build-debug

#edit the "inputs.txt" file as needed
```
## Creating random samples
```
make
./restructured
```
# Running the programs on Lanka
## Compiling all programs
```
cd /data/scratch/henni-mohammed/tiramisu
screen                                                    #it's better to use screen

srun -N 1 -n 1 --exclusive -p lanka-v3 --pty bash -i
source ../anaconda3/bin/activate                          #activate the anaconda virtual environment
python3 compile.py --tiramisu-root $(pwd) --data-path ../data/ compile_all
```
# Running all programs
```
python3 compile.py --tiramisu-root $(pwd) --data-path ../data/ execute_all --timeout=150
```
