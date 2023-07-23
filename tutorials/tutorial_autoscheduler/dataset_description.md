# General description
The dataset is a set of functions. For each function, we save the explored schedules and their execution times. You can find an example of how the function is stored in the `function_gramschmidt_explored_schedules.json` file.  

# Function structure 
First, we save some information about the exploration of this function that will not be used for training but is useful for data management:
* `file_name`: The path of the JSON file for this function
* `node_name`: The HPC node used to explore this function
* `parameters`: Exploration parameters like the beam size

We also save the `initial_execution_time` for this function. This field will be used in data loading to get the speed up of each schedule explored for this function. 


## Program annotations
The `program_annotations` dictionary is the representation of the original program. It contains the following keys:
* `memory_size`: The sum of all buffer sizes in megabytes
* `iterators`: All the loops in the program represented as a dictionary. Each entry has as a key the iterator name and contains a dictionary with the following keys: 
    * `upper_bound`: Upper bound of this loop as a string (to support the case for non-rectangular loop nests)
    * `lower_bound`: Lower bound of this loop as a string (to support the case for non-rectangular loop nests)
    * `parent_iterator`: Parent loop iterator. `null` if this iterator is the root iterator (outermost loop in a loop nest)
    * `child_iterators`: List of child loop names directly included in this iterator.
    * `computations_list`: List of computations that are directly contained in this iterator.
* `computations`: All the computations in the program as a dictionary with the computation name as key and the following features for each computation:
    * `absolute_order`: Order of execution of this computation in the function
    * `iterators`: The list of iterator names that contain this computation
    * `comp_is_reduction`: Whether this computation represents a reduction operation
    * `write_access_relation`: The buffer and placement of write access for this computation
    * `write_buffer_id`: The ID of the buffer where the computation writes its result
    * `data_type`: Datatype of the buffer that the computation writes to and the type for the result of the computation.  
    * `data_type_size`: The size of the data type in bits. Not yet used for training.
    * `accesses`: list of read accesses for this computation. Each access is a dictionary with the following keys:
        * `access_is_reduction`: Whether this computation is used in its own assignment
        * `buffer_id`: ID of the buffer being accessed
        * `access_matrix`: Representation of this access in matrix format
    * `expression_representation`: tree representation of the expression of the computation. Check [this document](https://docs.google.com/document/d/1VehfCpFWR0dqRxBtDbn0KrO7yeGeIItUEcvyym0Mzrk/edit?usp=sharing) for an explanation of how the expression tree is represented.

## Schedule list
A list of all the explored schedules for this function and their execution time. Each schedule is a dictionary with the following keys:
* `execution_times`: A list of execution for this schedule. We take the minimum of this list as the representative.
* `exploration_method`: We plan on augmenting the dataset through different exploration techniques (like reinforcement learning, for example). This tag will indicate which exploration technique was used to obtain this schedule. 
* `legality_check`: The legality of this schedule. At the moment, all the schedules in the dataset are legal, but for the sake of generality, we can add illegal schedules as well to remove the legality check for points we have already explored and save time during the exploration.
* `sched_str`: A string representation of what transformations were applied in this schedule. Used for debugging and data analysis.  

We represent a schedule as all the transformations that were applied for each computation. Each computation name is thus a key in the schedule dictionary. For each computation, we save the following attributes: 

* `shiftings`: A list of applied loop shiftings. Each entry is itself a list containing the following elements in this order:
    * The iterator that is to be shifted
    * Shifting factor
* `tiling`: A dictionary containing information about the tiling that was applied:
    * `tiling_depth`: 
        * 2 for 2-dimensional tiling 
        * 3 for 3-dimensional tiling
    * `tiling_dims`: names of the iterators that were tiled
    * `tiling_factors`: list of tiling factors.

* `unrolling_factor`: the unrolling factor of the loop. At the moment, we only unroll the innermost loop for each computation, so the factor is the only missing information. If the factor is null, unrolling wasn’t applied in this schedule. 

* `parallelized_dim`: the name of the loop that was parallelized (if parallelization was applied)  

* `transformations_list`: a list of affine transformations represented as tags. Each entry in the list represents an affine transformation that is either **interchange**, **reversal**, or **skewing**. The order of the elements of the list 
is the order of application of the transformations. The representation for each transformation is:
         
         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'third_skewing_loop', 'skew_parameter_1', 'skew_parameter_2', 'skew_parameter_3', 'skew_parameter_4', 'skew_parameter_5', 'skew_parameter_6', 'skew_parameter_7', 'skew_parameter_8', 'skew_parameter_9']
         
     * Where the `type_of_transformation` tag is:
         - `0` for no transformation being applied
         - `1` for loop interchange
         - `2` for loop reversal
         - `3` for loop skewing
    - In the case of skewing, we are specifying the new values for the transformed submatrix of the affine transformation

In addition to the representation of all the transformations that were applied for each computation, we also represent whether fusion was applied between computations:
* `fusions`: A list of applied fusion. Each entry is itself a list containing the following elements in this order:
    * The first computation
    * The second computation
    * The level at which the two were fused

    Please note that in Tiramisu, we explore fusion by moving one computation from a loop nest to another. The application of multiple instances of this transformation can achieve the conventional loop fusion transformation. 
* `tree_structure`: A tree representation of the function. This tree is a dictionary that contains the key `roots`, which contains a list of dictionaries where each entry in the list represents a root of the program. A root is the outermost loop in a block of nested loops. We represent each block of loops recursively through a  tree representation. Each node in the tree has the following attribute:
    * `loop_name`: The name of the loop represented by this node
    * `computations_list`: A list of computation names directly included in this loop
    * `child_list`: A list of nodes that are children of this node
    
    Please note that we only reflect the application of fusion on the `tree_structure`. Other transformations, like tiling, that change the structure by adding loops or changing the loop bounds are not reflected. This is a conceptual decision that could be changed in the future. 


