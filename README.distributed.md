This document shows how to run the test suite with MPI enabled. The same rules would apply for the tutorials with MPI enabled as well.
It first outlines directions for running on the Lanka cluster (good for benchmarking), then gives directions for just running on your local machine (good for development/debugging).

## Running Distributed Tests on the Lanka Cluster

For tiramisu, we have been testing with OpenMPI, but the scheduler on Lanka, slurm, is built with MVAPICH support, so we need to bypass it.
These instructions give all the necessary steps for setting up and running distributed tiramisu with MPI.

Note: Make sure you have installed OpenMPI on lanka somewhere on the /data/scratch filesystem. That way, all of the
lanka nodes can see the binaries.

#### 1. Allocate nodes from Slurm
You need to have slurm grant you nodes. But you want to do this from within a compute node, not the login node.
- Run

   `$ srun -N 1 -n 1 --exclusive --pty bash -i`
   
    To get into a compute node
- Then run

   `$ salloc --exclusive -N <#_Nodes>`
   
    To get Slurm to give you exclusive access to #_Nodes. All of the distributed tests use 10 nodes, so if you
want to run the MPI tests, get at least 10 nodes. You can see this if you look at the file `tests/test_list.txt` and look for `mpi`
- Run

   `$ squeue`
To verify that you were granted all the nodes you requested.

#### 2. Update the configuration.cmake file
**Things like emacs don't work very well on the compute nodes, so you probably want to do this from another tab on the lanka login node**
- Set `USE_MPI` to `TRUE`.
- Set `MPI_BUILD_DIR` to the location of the directory you built OpenMPI in. This path will be prepended in CMake
to find the bin, include, and lib directories (for example `${MPI_BUILD_DIR}/bin`)
- Set `MPI_NODES` to be a comma-delimited list of all the lanka nodes that you got from the `salloc` command.
squeue will show you the names.

#### 3. Build and Run tiramisu
**Make sure you compile and run from the same compute node that you ran `salloc` on!** Sometimes slurm seems to complain if you try to run from a different node.    
- Build tiramisu as usual

    `$ mkdir build && cd build`
    
    `$ cmake ..`
    
    `$ make tiramisu`
    
- Do the normal `$ make test` to run the full test suite, or just use `$ ctest -R <test_#>` to run a single test. You should be able to successfully
run the MPI tests.

#### 4. Remove allocated nodes
- When you are done, free up the nodes that slurm gave you so others can use them. You can always allocate them again :)
To do that, run

    `$ scancel <job_number>`
    
You can get the job number from running `squeue`. 

## Running Distributed Tests on your Local Machine

Most likely, you don't have slurm on your local machine, nor do you have multiple nodes, so the steps are a little different.
You still need to have OpenMPI installed on your local machine.
Note that performance will likely be terrible, not just because you only have one node, but because OpenMPI doesn't particularly
like when you oversubscribe a node (i.e. run more MPI processes than the machine has processors). But since you're not benchmarking on
your local machine, and the tests are short, it shouldn't be a problem.

#### 1. Update the configuration.cmake file
**Things like emacs don't work very well on the compute nodes, so you probably want to do this from another tab on the lanka login node**
- Set `USE_MPI` to `TRUE`.
- Set `MPI_BUILD_DIR` to the location of the directory you built OpenMPI in. This path will be prepended in CMake
to find the bin, include, and lib directories (for example `${MPI_BUILD_DIR}/bin`)
- Set `MPI_NODES` to be just be the following:

     `set(MPI_NODES "localhost")`

#### 2. Build and Run tiramisu
- Build tiramisu as usual

    `$ mkdir build && cd build`
    
    `$ cmake ..`
    
    `$ make tiramisu`
    
- Do the normal `$ make test` to run the full test suite, or just use `$ ctest -R <test_#>` to run a single test. You should be able to successfully
run the MPI tests.