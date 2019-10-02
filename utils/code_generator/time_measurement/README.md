## Time measurement

Here's a description of this folder (there are scripts that generate python lists, we use pickle to save them to disk) :

### create_progs_list.py

The first script to call : it generates a list of programs that will be executed. This list will be a more convenient way to distribute tasks across multiple nodes.

Each entry of the list is of format : (function_id, schedule_id).

**IMPORTANT :** There are some directories that contain only a single program, with no schedule. Those programs won't be included in the final list as they don't need to be executed : they just have a speedup of 1, so if you want them in the final dataset, just include them with a speedup of 1.

### Phase 1 : generate Tiramisu object files
Compile the Tiramisu codes and generate object files. This is where you must begin if your don't have Tiramisu object files.

### Phase 2 : compile wrappers and execute them
Compile the wrappers and do time measurement. You must have your Tiramisu object files before.

### IMPORTANT :

Before submitting the jobs to SLURM, it's better to execute the scripts manually and see if they work fine.
