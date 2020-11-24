# Benchmark Autotuner
## Description
Autotuning is a process which consists of finding "automatically" the best parameters for a certain task.

In our case, we want to find the best optimization parameters for our code. This allows to have the best execution time possible for a certain hardware configuration.

Manually tuning, consists of :
1. Executing the benchmark code many times using different optimization parameter values at each execution.
2. Recording the execution time for each of the parameter values.
3. Choosing the best parameter values for the benchmark according to the execution time.

This process takes a huge amount of time if done manually. It can be very easily automated through scripts, this is what we propose here.

## How the autotuner works
1. **The Benchmark Developer** specifies in an `autotune.json` file what parameters need to be tuned.
2. The autotuner reads the `autotune.json` file to determine what parameters the developer wants to tune then generates a tuning script `tuning_script.sh`.
4. The tuning script can then be used by **The Benchmark User** to tune the parameters for their hardware configuration.

## For Developers
Our benchmarks are generally composed of :
1. A generator file : which contains tiramisu code.
2. A wrapper file : which calls the generated tiramisu function and compares results to the reference.
3. A header file : `configure.h` to store benchmark parameters and optimization parameters.

We propose a way to specify what parameters to tune, and what values to try for each of the parameters. This is done by specifying an `autotune.json` file, which declares each of the parameters we want to tune, and the values we want to try (ranges).

To use the autotuner in your benchmark, you need to follow the following steps:
### Identify tuning constants (#define)
Go through your benchmark parameters (configure.h), and decide what parameters you want the benchmark user to tune. You will need to define :
1. Constant names : for example, in the case : `#define X_BL 14`, the constant name is `X_BL`.
2. Ranges of values to explore for each constant : for exemple, I want to try all the `X_BL` values between 1 and 100.
3. Divisibility constraints on each constant : for example, I want `X_BL` to be a divisor of *224*.
### Identify the generator and the wrapper
1. Find the generator file and add `#define __TIRAMISU_GENERATOR__` at the top of the file
2. Find the generator file and add `#define __TIRAMISU_WRAPPER__` at the top of the file
3. Add at the bottom of the `configure.h` file this piece of code which permits the tuner to overwrite parameter values.
```c
#if defined(__TIRAMISU_WRAPPER__) || defined(__TIRAMISU_GENERATOR__)
	#if TUNE_PARAMETERS
		#include "param_tuning.h"
	#endif
#endif
```
### Copy the `autotune_generator.py` file to the benchmark's folder (same folder as `configure.h` and `autotune.json`)
### `autotune.json` creation
Create an autotune.json file (or copy the example you can find in this folder).

The `autotune.json` file has a specific format. Here, you need to recall the constant names and the values you want them to take. Here is an example of an `autotune.json` file, we will explain each of the fields.
```json
{
  "tuning_header_file_dir":".",
  "tuning_header_file_name":"param_tuning.h",
  "executable_dir":".",
  "executable_name":"wrapper_nn_block_vgg",
  "compile_path":"../../../../../../build/benchmarks/DNN/blocks/vggBlock/cpu/dense",
  "compile_options":"",
  "output_file_path":"./log.txt",
  "parameters_to_tune":[
    {
      "name":"X1_BLOCKING",
      "values":"4:16",
      "divisor_of":112
    },
    {
      "name":"X2_BLOCKING",
      "values":"3,5,9"
    }
  ],
  "tuning_script_file_dir": ".",
  "tuning_script_file_name":"tuning_script.sh"
}
```
The fields are explained in this table, the ones that are important to change are highlighted in bold, the other ones can be left as is if you plan on working inside the benchmark folder.
| Field                   |    Description                                                                       |
|-------------------------|--------------------------------------------------------------------------------------|
| tuning_header_file_dir  |  Specifies where you want the autotuning header to be stored (generally .)           |
| tuning_header_file_name |  Specifies the autotuning header file's name (leave it as it is for the general case)|
| executable_dir          |  Specifies where your code's executable is generated (generally same directory as the autotuning script) |
| **executable_name**     |  Specifies the executable's name, this is generally the wrapper's executable (depends on the code you're tuning, it can be for example wrapper_nn_block_vgg)|
| **compile_path**        |  Specifies the code's compilation directory (the folder where you have to execute make)|
| compile_options         |  You can add here any compilation options to pass to make (generally empty)|
| output_file_path        |  Specifies the file you want to store in the log of all the executions performed by the tuning script|
| **parameters_to_tune**  |  Specifies one by one, each of the parameters you want to tune by their name, then each of the values they can take, and what division constraints you want on them |
| tuning_script_file_dir  |  Specifies the directory where you want the tuning_script.sh tuning shell script to be generated |
| tuning_script_file_name |  Specifies the tuning script name (you can leave it as it is) |

Note that the values for the parameters to take, can be specified in three ways :

1.`"values":"4:16"` : Indicates we want to try all the values between 4 and 16.

2.`"values":"4:16:2"` : Indicates we want to try all the values between 4 and 16, by making 2 steps (starts at 4 and ends at 16 (included))

3.`"values":"2,10,24"` : Indicates we want to try all the listed values

Adding a `"divisor_of":112` statement, will limit the tested values to divisors of 112.
For example, the following declaration states that we want to test all the divisors of 112 that are between 4 and 16  (4 and 16 included).
```json
   {
      "name":"X1_BLOCKING",
      "values":"4:16",
      "divisor_of":112
    }
```

### Run the autotune script generator
Execute the following shell command :
```
python autotune_generator.py
```
This script generates a `tuning_script.sh` that can be used by the developer or the user to tune the code.
Notice that an end user only needs the `tuning_script.sh` file to tune their code.

### See results in the log.txt file and pick the best parameters.
There is a version which automatically picks the best case, but this requires some important changes on the benchmark to tune. You can find an example of this in the `Conv-ReLU-Conv-ReLU` and the `vggBlock` benchmarks (found in `benchmarks/DNN/blocks/`.


## For End Users
### Run the `tuning_script.sh` file
Execute the following command :
```
./tuning_script.sh
```
This script can be used to tune the optimization parameters for your hardware configuration.

### See results in the log.txt file and pick the best parameters.
Pick the optimization parameters with the lowest execution time and set the corresponding parameters in the `configure.h` file.
