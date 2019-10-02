## Phase 1 : generate Tiramisu object files

Here's a description of each script :

### compile_tiramisu_code.py

Compile the codes written with Tiramisu and generate the object files.

### generate_job_files.py

This script generates the job files that will be submitted to SLURM. See the script comments for more information.

### How to work with the scripts

Before executing a script, you need to configure some parameters. See each script for more details. Note that to execute those scripts, you need to have generated the list of programs with `create_progs_list.py`.

First, configure `compile_tiramisu_code.py` and then execute `generate_job_files.py`. After that, submit the jobs to SLURM with `submit_compile_jobs.sh`.
