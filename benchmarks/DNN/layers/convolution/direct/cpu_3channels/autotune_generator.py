import json
import math
import sys
import os

'''
This script is used for generating tuning scripts for the benchmarks:
How to generate a tuning script ?
=================================
1. Create a file named autotune.json in the benchmark's directory
2. Fill the file according to the example autotune.json format.
    a. Specify tuning header name
    b. Specify executable's directory
    c. Specify the generated executable's name
    d. Specify the compilation (build) path (where to execute make)
    e. Specify compile options, these are appended  to make, for example, if
    you set compile_options to run_lstm, the following command will be executed
    make run_lstm
    f. Specify the output log file of all the values and their corresponding time.
    d. Specify the parameters to tune and their values ranges. (Ranges are inclusive, both extremities)
3. Note that it'd be better if you use absolute paths, but you can specify paths
relative to the directory you'll execute the tuning script in.
4. Execute this script python autotune_generator.py to generate the output tuning script file

'''
def getDivisors(n):
    divisors = [1, n]
    nSQRT = int(math.sqrt(n))
    if n%2 == 0:
        for i in range(2, nSQRT):
            if(n%i == 0):
                divisors.append(i)
                divisors.append(n//i)
    else:
        for i in range(3, nSQRT, 2):
            if(n%i == 0):
                divisors.append(i)
                divisors.append(n//i)
    if nSQRT * nSQRT == n:
        divisors.append(nSQRT)
    divisors.sort()
    return divisors

# read file
with open('autotune.json', 'r') as autotune_file:
    config = autotune_file.read()

# parse file
autotune_config = json.loads(config)

################ Get relative paths
path_script_to_tuning_header=os.path.relpath(autotune_config["tuning_header_file_dir"], autotune_config["tuning_script_file_dir"])
path_script_to_output_file = os.path.relpath(autotune_config["output_file_path"], autotune_config["tuning_script_file_dir"])
path_script_to_compile_dir=os.path.relpath(autotune_config["compile_path"], autotune_config["tuning_script_file_dir"])
path_compile_dir_to_executable_dir=os.path.relpath(autotune_config["executable_dir"], autotune_config["compile_path"])
path_executable_to_tuning_script=os.path.relpath(autotune_config["tuning_header_file_dir"], autotune_config["executable_dir"])
path_executable_to_output_file=os.path.relpath(autotune_config["output_file_path"], autotune_config["executable_dir"])
tune_parameters_script = "printf \"\" > "+path_script_to_output_file+"\n"
param_number=1
tabbing=""
for param in autotune_config["parameters_to_tune"]:
    # Create values range
    if "," in param["values"]: # case of a list of values to try
        values = [int(val) for val in param["values"].split(",")]
    elif ":" in param["values"]: # case of a range of values to try
        range_values = param["values"].split(":")
        if len(range_values)==3: # case of range with step
            values = [*(range(int(range_values[0]), int(range_values[1]) + 1, int(range_values[2])))]
        elif len(range_values)==2: # case of range with step=1
            values = [*(range(int(range_values[0]), int(range_values[1]) + 1))]
    elif param["values"].isdigit(): # case of a single integer
        values=[int(param["values"])]
    else :
        print("Error : values field in parameter number %d must be an integer, a start:end:step range, or a list of comma separated integers i1,i2,i3,i4"%param_number)
        sys.exit(1)
    if(len(values)==0):
        print("Error, there are no values represented by parameter %d's values field"%param_number)
        sys.exit(2)

    # Consider divisor_of parameter
    if "divisor_of" in param:
        divisors = getDivisors(int(param["divisor_of"]))
        values = [val for val in values if val in divisors]

    #Error if there are no values left after taking off non divisors
    if(len(values) == 0):
        print("Error : There are no divisors of %d for parameter number %d's values"% (param["divisor_of"], param_number))
        sys.exit(3)
    vals_string = " ".join(str(val) for val in values)
    tune_parameters_script+= tabbing + "for "+param["name"]+" in "+ vals_string + "; do\n"
    tabbing += "\t"
    param_number+=1

header_file = path_script_to_tuning_header +"/"+ autotune_config["tuning_header_file_name"]
param_number=1
for param in autotune_config["parameters_to_tune"]:
    tune_parameters_script+= tabbing+"printf \"#ifdef "+param["name"]+"\\n\t"
    if param_number == 1:
        tune_parameters_script+= tabbing+"#undef "+param["name"]+"\\n#endif\\n\" > "+header_file+"\n"
    else:
        tune_parameters_script+= tabbing+"#undef "+param["name"]+"\\n#endif\\n\" >> "+header_file+"\n"
    tune_parameters_script+= tabbing+"printf \"#define "+param["name"]+" $"+param["name"]+"\\n\" >> "+header_file+"\n"
    param_number+=1
tune_parameters_script+= tabbing+"printf \"\" >> "+header_file+"\n"
tune_parameters_script+="\n"

#Logging and showing progress
printf_string = "printf \""
for param in autotune_config["parameters_to_tune"]:
    printf_string+=param["name"]+"=$"+param["name"]+", "

tune_parameters_script+=tabbing+printf_string+"\";\n"+tabbing+printf_string+"\""+" >> "+ path_script_to_output_file+";\n"

#Building part
tune_parameters_script+= tabbing+"cd "+path_script_to_compile_dir+"\n"
tune_parameters_script+= tabbing+"make "+autotune_config["compile_options"]+" > /dev/null 2>&1;\n"
tune_parameters_script+= tabbing+"cd "+path_compile_dir_to_executable_dir+";\n"
tune_parameters_script+= tabbing+"./"+autotune_config["executable_name"]+" |tee -a "+path_executable_to_output_file+";\n"
tune_parameters_script+= tabbing+"./clean.sh;\n"
tune_parameters_script+= tabbing+"cd "+path_executable_to_tuning_script+";\n"

#at this stage param_number = number of parameters + 1
param_number-=2
for param in autotune_config["parameters_to_tune"]:
    tune_parameters_script+= param_number *"\t"+"done\n"
    param_number-=1

with open(autotune_config["tuning_script_file_dir"]+"/"+autotune_config["tuning_script_file_name"], "w") as f:
    f.write(tune_parameters_script)
print(tune_parameters_script)
