# Autoscheduler tutorial

There are two modes to run the autoscheduler: first, by using the cost model to predict the speed up from applying a sequence of transformations on an input program, and second, by executing each point in the exploration and obtaining the real speed up. These two modes can be interchanged using the ```EXPLORE_BY_EXECUTION```environment variable.
# Cost model environment
To be able to use the model, you will need to install a conda environment and activate it. To do this, use the following commands from the `tutorials_Atuoscheduler` directory:
```bash
conda env create -f ./model/environment.yml
```

This should create an environment called `cost_model_env`. There is no need to activate this environment since we will specify the path to its python version in the `function_autoscheduler.cpp` file.
# Program file strcuture
To run the autoscheduler on an input program, please respect the following file structure. Assuming **function** is the name of the program:

```
├── tutorials_autoscheduler
│   ├── function_autoscheduler.cpp
│   ├── function_generator.cpp
│   ├── function_wrapper.cpp
│   ├── function_wrapper.h
│   ├── autoschedule.sh
│   ├──...
```

## function_generator.cpp

The program generator contains a tiramisu program with a call to the `codegen` function that generates an object file. For more information on writing Tiramisu programs, please refer to the [Tiramisu documentation](http://tiramisu-compiler.org/).

## function_autoscheduler.cpp

This is the main file that will launch the autoscheduler. It has the same function definition as the generator file of our program. Afterward, we record the dependencies of the original program to be able to test the legality of transformations during the exploration and define the search parameters.

```c++
// Record program dependencies
prepare_schedules_for_legality_checks();
performe_full_dependency_analysis();

// Define search parameters
const int beam_size = get_beam_size();
declare_memory_usage();
```

We instantiate the schedule generator (candidate generation algorithm) and point out the object file that should be used to execute the programs if we choose to run by execution.

```c++
auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_Q, &b_R}, "function_gramschmidt.o", "./function_gramschmidt_wrapper");
```

We define our search method (currently, only beam search is supported). 
```c++
auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, exec_eval, scheds_gen);
``` 
If we choose to run by model, we change the `exec_eval` instance with an instantiation of the model evaluation class: 
```c++
auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_mode(py_cmd_path, {py_interface_path});
auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, model_eval, scheds_gen);

```
We instantiate the autoscheduler object and call the sample search space function, which starts the autoschedling and saves the exploration trace and explored schedules in a JSON file. Again, we replace the execution evaluator with the model evaluator if the exploration is by model. 

```c++
auto_scheduler::auto_scheduler as(bs, exec_eval);
as.set_exec_evaluator(exec_eval);
as.sample_search_space("./function_gramschmidt_explored_schedules.json", true);
```

## function_wrapper.cpp
We use the wrapper function to get the real measurements of the original program and then for each sequence of transformations applied to the original program. To do this, we create and initialize as many buffers as our program needs. Then, we measure the time it takes for one call of the function. Sometimes, we measure multiple instances of the execution time to avoid noisy measurements.


## function_wrapper.h
Helper functions and headers for the wrapper file.

# Running the provided example
Given this structure, you can run the autoschedling for a program using the ```autoschedule.sh``` script. The script will compile and run all the necessary files and output the autoschedling process in the cmd.

Some parameters that are set in this script include:
* `TIRAMISU_ROOT`: Absolute path to the Tiramisu directory
* `BEAM_SIZE`: This is a search method parameter specific to beam search. It determines how many of the best partial solutions to continue exploring. 
* `EXPLORE_BY_EXECUTION`: Measure the real execution time of explored schedules. If set to 0, the exploration will be done using the cost model.
* `AS_VERBOSE`: Output varies steps in the autoscheduling process to stdout.
* `NB_EXEC`: Number of times to measure the execution time of a program. Added to avoid noisy measurements.
* `SAVE_BEST_SCHED_IN_FILE`: Set to yes if you want to save the best schedule from the exploration in CSV format. Requires defining `LOG_FILE_PATH`.
* `LOG_FILE_PATH`: Path where to save exploration results.
* `EXECUTE_BEST_AND_INITIAL_SCHED`: Measure the execution time of the initial and best schedule in the exploration regardless of the exploration mode (model or execution). Useful when running the exploration by model to verify the performance of the best-predicted schedule.

To run the autoscheduler:
1. Open a terminal in this directory (`tutorials_Atuoscheduler`).  
2. Change the `TIRAMISU_ROOT` environment variable in the `autoschedule.sh` script. 
3. Specify the path to Python in the autoscheduler file. Please use the version installed in the first step. The path should end with `anaconda3/envs/cost_model_env/bin/python3.10`. You can use the command `whereis python` to find the path of all your installed Python versions.
4. Call the script using the function name. (`function_gramschmidt` in this case)

```bash
bash autoschedule.sh function_gramschmidt
```

The exploration JSON will be saved in the build file.

# Testing your own Tiramisu program
Assuming you have your own Tiramisu program in the form of a `generator.cpp` file and you want to change the example file in this repo, You would need to:
* Change the program definition in the `autoscheduler.cpp` file
* Change the function name and call in the `wrapper.h` file (in case the number of buffers has changed)
* Change the function name and call in the `wrapper.cpp` file (in case the number of buffers has changed)
* Change the buffer initialization in the `wrapper.cpp` file in the case where the buffer sizes have changed. Otherwise, you will probably get a segmentation fault for an out-of-bounds memory access
* Make sure the function name if consistent across all four files

# Dataset description 
Please check the `dataset_description.md` file in this directory for a general description of the dataset the cost model was trained on. This would be helpful in the case where a user wants to use the cost model without using the Tiramisu autoscheduler.  