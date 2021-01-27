# FlexNLP Documentation
## How to add a new tiramisu flexnlp function
### Step I : Create the C/C++ function
This step consists on creating the C/C++ wrapper function that will call the FlexNLP API.
1. Go to the tiramisu/src/tiramisu_flexnlp_wrappers.cpp file.
2. Add your C/C++ function, you can include any header files you need.
3. Each device has a device_id, you can retrieve the corresponding PETop object (PETop is a temporary name, maybe we will have an abstract class that PETop heritates from to support many types of accelerators, not only PE)
4. You can set the availability of the retrieved PETop object using set_accelerator_availability (check tiramisu/src/FlexNLP/tiramisu_flexnlp.h for the details), then start using it (for e.g call an LSTMCell)

### Step II : Create the tiramisu function

1. Go to the tiramisu/src/tiramisu_expr.cpp file.
2. Add the tiramisu function corresponding to the C/C++ function added in step I. The difference is that, this function will use tiramisu expr and buffer types instead of C/C++ pointers and types. This tiramisu function will call the C/C++ function previously created. The created function can then be called in any Tiramisu expression (check tiramisu/benchmarks/FlexNLP/LSTM/flexnlp_tiramisu_generator.cpp for an example of a function call in a Tiramisu program)
3. Add the tiramisu function's header at the bottom of the tiramisu/include/tiramisu/expr.h file.
4. The function can now be called in any Tiramisu program.

> For a simple example, you can check the `flexnlp_initialize` Tiramisu function (found in tiramisu/src/tiramisu_expr.cpp) corresponding to the tiramisu_flexnlp_initialize C/C++ function (found in tiramisu/src/tiramisu_flexnlp_wrappers.cpp).

### How Tiramisu function calls work
In tiramisu, it's possible to call a Tiramisu function (Tiramisu functions are implemented in tiramisu_expr.cpp with headers in expr.h).
A function is called in a computation's expression e.g :
```
// Runs the LSTM cell function
computation run_lstm("run_lstm", {l},
      flexnlp_lstm_cell(*input_cpu_input.get_buffer(), *input_cpu_W.get_buffer(),
                        *input_cpu_output.get_buffer(), *input_cpu_h_out.get_buffer(),
                        l)
);
```
Notice the function call `flexnlp_lstm_cell(...)`.
At the code generation stage, this function call will be replaced by the corresponding wrapper `tiramisu_flexnlp_lstm_cell(...)` which itself contains calls to the FlexNLP API.
So a Tiramisu-FlexNLP function is defined on 3 levels :
1. `Tiramisu function`: these are implemented in the tiramisu/src/tiramisu_expr.cpp file. These calls the corresponding Tiramisu wrapper function.
2. `Tiramisu wrapper function`: these are implemented in the tiramisu/src/tiramisu_flexnlp_wrappers.cpp as extern "C" functions. The wrapper function uses the FlexNLP API function to perform the task.
3. `FlexNLP function (API)`: which is are provided through the FlexNLP API header file (here, we use the behavioral interface)

## Writing a Tiramisu-FlexNLP program
The process is the same as any tiramisu program, the only differences are that you have to :
1. Call the `flexnlp_initialize` Tiramisu function, by giving it the number of FlexNLP devices to use. (This requires creating a computation that will call flexnlp_initialize(num_devices))
2. Call any FlexNLP functions (some need the device_id)
3. Call the `flexnlp_finalize` Tiramisu function, it doesn't take any parameter. (This requires you to create a computation that will call flexnlp_finalize(), the function will return an integer)
4. Generate code by calling the tiramisu::codegen function and giving to it the tiramisu::hardware_architecture_t::arch_flexnlp flag e.g :
```
tiramisu::codegen({
        input_cpu_input.get_buffer(),
        input_cpu_W.get_buffer(),
        input_cpu_output.get_buffer(),
        input_cpu_h_out.get_buffer()
    },"generated_flexnlp_test.o", tiramisu::hardware_architecture_t::arch_flexnlp);
```

## Available FlexNLP-Tiramisu functions
### Initialization
- `flexnlp_initialize`: This function instanciates the FlexNLPContext global variable, it contains an object for each of the accelerators, it has one parameter which is the number of accelerators that will be used in the Tiramisu-FlexNLP program. (Look at **tiramisu/benchmarks/FlexNLP/LSTM/flexnlp_tiramisu_generator.cpp** for an example).
- `flexnlp_finalize`: This function destroys the FlexNLPContext global variable, you have to run it at the end of the Tiramisu-FlexNLP program (Look at **tiramisu/benchmarks/FlexNLP/LSTM/flexnlp_tiramisu_generator.cpp** for an example).
### Data Copy Functions
- `flexnlp_load_weights`: This function loads the weights to the FlexNLP device (in the weights SRAM), it has as parameters : the weights buffer, the offset according to the weights buffer, and the number of elements to copy. (Look at **tiramisu/benchmarks/FlexNLP/LSTM_manual_data_copy/flexnlp_tiramisu_generator.cpp** for an example).
- `flexnlp_load_input`: This function is the same as `flexnlp_load_weights`, but for the Input.
- `flexnlp_store_output`: This function is the same as `flexnlp_load_weights`, but for the Output.

### Supported DL Operators
#### LSTM
##### One Accelerator (PE ?)
- `flexnlp_lstm_cell`: This function represents a single LSTM cell on one PE with automatic data copy.
- `flexnlp_lstm_cell_manual`: This function represents a single LSTM cell on one PE with manual data copy.

##### Multiple Accelerators (PEs ?)
- `flexnlp_lstm_cell_partitioned`: This function splits a single LSTM cell execution on the same accelerator (Not enough memory to fit the weights in the FlexNLP memory)
- `flexnlp_lstm_cell_partitioned_multi_accelerator`: This function splits a single LSTM cell execution on HIDDEN_SIZE/OUTPUT_SIZE accelerators in parallel (Not enough memory to fit the weights in the FlexNLP memory). Note that : parallel execution isn't supported yet as we don't have wait functions in the Behavioral Interface yet.

## Added (or modified) files and folders

### Modified
- `tiramisu/include/tiramisu/expr.h`: for adding the different Tiramisu-FlexNLP functions' headers (the ones cited in **Available FlexNLP-Tiramisu functions)
- `tiramisu/src/tiramisu_expr.cpp`: for adding the implementation of the different Tiramisu-FlexNLP functions' implementation (the ones in expr.h)
- `tiramisu/src/tiramisu_function.cpp`: for adding support for the FlexNLP architecture in Tiramisu.
- `tiramisu/benchmarks/CMakeLists.txt`: for adding the 4 FlexNLP benchmarks (LSTM, LSTM manual data copy, LSTM split on a single accelerator, and LSTM split on multiple accelerators)

### Added
- `tiramisu/benchmarks/FlexNLP`: added the different multilayer LSTM benchmarks (LSTM, LSTM_manual_data_copy, LSTM_partitioned_same_accelerator, LSTM_partitioned_multi_accelerators)
- `tiramisu/src/FlexNLP/pe_int8_top.h`: behavioral interface provided by Daniel. Contains the different FlexNLP functions.
- `tiramisu/src/FlexNLP/tiramisu_flexnlp.h`: contains the FlexNLPContext class which is used to manage the Accelerator objects (PETop objects), it allows us to have one FlexNLP context containing all of the accelerators that we can call only by specifying a device ID. (This allows us to work with multiple accelerators)
- `tiramisu/src/tiramisu_flexnlp_wrappers`: these are wrappers for the Tiramisu-FlexNLP functions declared in expr.h (and implemented in tiramisu_expr.cpp), these wrappers call the FlexNLP API through the PETop class (check pe_int8_top.h for the implementation)
