import copy
import sympy
import re
import numpy as np
import torch

from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from torch.nn.utils.rnn import pad_sequence

# An exception to limit the maximum number of allowed transformations 
class NbTranformationException(Exception):
    pass

class RandomMatrix(Exception):
    pass

# An exception to limit the maximum number of read-write accesses. 
class NbAccessException(Exception):
    pass

# An exception to limit the maximum number of nested loops. Currently set to 5.
class LoopsDepthException(Exception):
    pass

# Maximum sequence of transformations (reversal, interchange and skewing) allowed. Currently set to 4 
MAX_NUM_TRANSFORMATIONS = 4

# Maximum size of the tags vector representing each transformation
MAX_TAGS = 16

# Maximum depth of a loop nest for each computation
MAX_DEPTH = 5

# Maximum length of expressions in the dataset
MAX_EXPR_LEN = 66

device = "cpu"
train_device = torch.device("cpu")


class NbMatricesException(Exception):
    pass


class NbAccessException(Exception):
    pass


class LoopsDepthException(Exception):
    pass



global_dioph_sols_dict = dict()

# Separate a computation vector into 3 parts where the middle part is the transformation vectors
def seperate_vector(
    X: torch.Tensor, num_transformations: int = 4, pad: bool = True, pad_amount: int = 5
) -> torch.Tensor:
    batch_size, _ = X.shape
    first_part = X[:, :33]
    second_part = X[:, 33 : 33 + MAX_TAGS * num_transformations]
    third_part = X[:, 33 + MAX_TAGS * num_transformations :]
    vectors = []
    for i in range(num_transformations):
        vector = second_part[:, MAX_TAGS * i : MAX_TAGS * (i + 1)].reshape(batch_size, 1, -1)
        vectors.append(vector)

    if pad:
        for i in range(pad_amount):
            vector = torch.zeros_like(vector)
            vectors.append(vector)
    return (first_part, torch.cat(vectors[0:], dim=1), third_part)

def get_representation_template(program_json, no_sched_json):
    # Set the max and min number of accesses allowed 
    max_accesses = 15
    min_accesses = 0

    comps_repr_templates_list = []
    comps_indices_dict = dict()
    comps_placeholders_indices_dict = dict()

    
    # Get the computations (program statements) dictionary and order them according to the absolute_order attribute
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )
    # For each computation in the program
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        # Check if the computation accesses conform to the minimum and maximum allowed
        if len(comp_dict["accesses"]) > max_accesses:
            raise NbAccessException
        
        if len(comp_dict["accesses"]) < min_accesses:
            raise NbAccessException
        
        # Check if the number of iterators for this computation doesn't surpass the maximum allowed
        if len(comp_dict["iterators"]) > MAX_DEPTH:
            raise LoopsDepthException
    
        comp_repr_template = []
        comp_repr_template.append(+comp_dict["comp_is_reduction"])

        iterators_repr = []
        
        # Add a representation of each loop of this computation
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            iterator_dict = program_json["iterators"][iterator_name]
            # Create a unique code for each loop
            c_code = "C" + str(comp_index)
            l_code = c_code + "-L" + str(iter_i)
            
            # Add a placeholder for transformations applied to this loop
            iterators_repr.extend(
                [
                    l_code + "Parallelized",
                    l_code + "Tiled",
                    l_code + "TileFactor",
                    l_code + "Fused",
                    l_code + "Shifted",
                    l_code + "ShiftFactor",
                ]
            )
        
        iterator_repr_size = int(len(iterators_repr) / len(comp_dict["iterators"]))
        
        # Add padding incase the number of loops is lower than the max
        iterators_repr.extend(
            [0] * iterator_repr_size * (MAX_DEPTH - len(comp_dict["iterators"]))
        )
        
        # Add two tags for whether unrolling was applied and the unrolling factor
        iterators_repr.extend([c_code + "-Unrolled", c_code + "-UnrollFactor"])

        # Add a placeholder for the other transformations to be applied (skewing, reversal and interchage)
        iterators_repr.append(c_code + "-TransformationTagsStart")
        iterators_repr.extend(["M"] * (MAX_TAGS * MAX_NUM_TRANSFORMATIONS - 2))
        iterators_repr.append(c_code + "-TransformationTagsEnd")
        
        # Adding initial constraint matrix
        iterators_repr.append(c_code+'-OgConstraintMatrixStart')
        iterators_repr.extend(['OgC']*((MAX_DEPTH*MAX_DEPTH*2)-2))
        iterators_repr.append(c_code+'-OgConstraintMatrixEnd')
        
        # Adding initial constraint vector
        iterators_repr.append(c_code+'-OgConstraintVectorStart')
        iterators_repr.extend(['V']*(MAX_DEPTH*2-2))
        iterators_repr.append(c_code+'-OgConstraintVectorEnd')
        
        # Adding transformed constraint matrix
        iterators_repr.append(c_code+'-ConstraintMatrixStart')
        iterators_repr.extend(['C']*((MAX_DEPTH*MAX_DEPTH*2)-2))
        iterators_repr.append(c_code+'-ConstraintMatrixEnd')
                              
        # Add the loop representation to the computation vector 
        comp_repr_template.extend(iterators_repr)
        
        # Pad the write access matrix and add it to the representation
        padded_write_matrix = pad_access_matrix(
            isl_to_write_matrix(comp_dict["write_access_relation"])
        )
        write_access_repr = [
            comp_dict["write_buffer_id"] + 1
        ] + padded_write_matrix.flatten().tolist()
        comp_repr_template.extend(write_access_repr)

        # Pad the read access matrix and add it to the representation 
        read_accesses_repr = []
        for read_access_dict in comp_dict["accesses"]:
            read_access_matrix = pad_access_matrix(
                read_access_dict["access_matrix"]
            )
            read_access_repr = (
                [+read_access_dict["access_is_reduction"]]
                + [read_access_dict["buffer_id"] + 1]
                + read_access_matrix.flatten().tolist()
            )
            read_accesses_repr.extend(read_access_repr)
        access_repr_len = (MAX_DEPTH + 1) * (MAX_DEPTH + 2) + 1 + 1
        read_accesses_repr.extend(
            [0] * access_repr_len * (max_accesses - len(comp_dict["accesses"]))
        )
        comp_repr_template.extend(read_accesses_repr)
        
        # Add the representation of this computation to the list of containing all computations
        comps_repr_templates_list.append(comp_repr_template)
        
        # Create a mapping between the features and their position in the representation
        comps_indices_dict[comp_name] = comp_index
        for j, element in enumerate(comp_repr_template):
            if isinstance(element, str):
                comps_placeholders_indices_dict[element] = (comp_index, j)
            
        
    # Create a representation of the loops independantly from the computations
    loops_repr_templates_list = []
    loops_indices_dict = dict()
    loops_placeholders_indices_dict = dict()

    for loop_index, loop_name in enumerate(program_json["iterators"]):
        # Create a unique code for each loop
        loop_repr_template = []
        l_code = "L" + loop_name
        
        # Add a placeholder for transformations applied to this loop
        loop_repr_template.extend(
            [
                l_code + "Parallelized",
                l_code + "Tiled",
                l_code + "TileFactor",
                l_code + "Fused",
                l_code + "Unrolled",
                l_code + "UnrollFactor",
                l_code + "Shifted",
                l_code + "ShiftFactor",
            ]
        )
        
        # Create a mapping between the features and their position in the representation
        loops_repr_templates_list.append(loop_repr_template)
        loops_indices_dict[loop_name] = loop_index
        for j, element in enumerate(loop_repr_template):
            if isinstance(element, str):
                loops_placeholders_indices_dict[element] = (loop_index, j)
    
    
    # Make sure no fusion was applied on this version and get the original tree structure 
    assert "fusions" not in no_sched_json or no_sched_json["fusions"] == None
    orig_tree_structure = no_sched_json["tree_structure"]
    tree_annotation = copy.deepcopy(orig_tree_structure)
    
    # Add necessary attributes to the tree_structure
    prog_tree = update_tree_atributes(tree_annotation, loops_indices_dict, comps_indices_dict, train_device="cpu")
    
    return (
        prog_tree,
        comps_repr_templates_list,
        loops_repr_templates_list,
        comps_placeholders_indices_dict,
        loops_placeholders_indices_dict
    )

# Change the structure of the tree annotations to contain a uinque index for each loop and a has_comps boolean
# This is used to prepare for the recusive embedding of the program during the training
def update_tree_atributes(node, loops_indices_dict, comps_indices_dict, train_device="cpu"):
        if "roots" in node :
            for root in node["roots"]:
                update_tree_atributes(root, loops_indices_dict, comps_indices_dict, train_device=train_device)
            return node

        node["loop_index"] = torch.tensor(loops_indices_dict[node["loop_name"]]).to(
            train_device
        )
        if node["computations_list"] != []:
            node["computations_indices"] = torch.tensor(
                [
                    comps_indices_dict[comp_name]
                    for comp_name in node["computations_list"]
                ]
            ).to(train_device)
            node["has_comps"] = True
        else:
            node["has_comps"] = False
        for child_node in node["child_list"]:
            update_tree_atributes(child_node, loops_indices_dict, comps_indices_dict, train_device=train_device)
        return node

def get_schedule_representation(
    program_json,
    schedule_json,
    comps_repr_templates_list,
    loops_repr_templates_list,
    comps_placeholders_indices_dict,
    loops_placeholders_indices_dict,   
):
    
    # Create a copy of the templates to avoid modifying the values for other schedules
    comps_repr = copy.deepcopy(comps_repr_templates_list)
    loops_repr = copy.deepcopy(loops_repr_templates_list)
    comps_expr_repr = []
    
    # Get an ordered list of computations from the program JSON
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )
    
    # For each computation
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = program_json["computations"][comp_name]
        comp_schedule_dict = schedule_json[comp_name]
        
        # Get the computation expression representation
        expr_dict = comp_dict["expression_representation"]
        comp_type = comp_dict["data_type"]
        expression_representation = get_tree_expr_repr(expr_dict, comp_type)
        
        # Padd the expression representtaion
        expression_representation.extend([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * (MAX_EXPR_LEN - len(expression_representation)))
        
        # Add the expression representation for this computation to the output
        comps_expr_repr.append(expression_representation)
        
        fused_levels = []
        
        # If fusion was applied, save which two loops were fused together
        if "fusions" in schedule_json and schedule_json["fusions"]:
            for fusion in schedule_json["fusions"]:
                if comp_name in fusion:
                    fused_levels.append(fusion[2])
        
        c_code = "C" + str(comp_index)
        # Loop representation for this computation
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            l_code = c_code + "-L" + str(iter_i)
            
            # Check whether parallelization was applied and put the tag in its corresponding position in the computation representation
            parallelized = 0
            if iterator_name == comp_schedule_dict["parallelized_dim"]:
                parallelized = 1
            p_index = comps_placeholders_indices_dict[l_code + "Parallelized"]
            comps_repr[p_index[0]][p_index[1]] = parallelized
            
            # Check whether tiling was applied and put the tags in their corresponding position in the computation representation
            tiled = 0
            tile_factor = 0
            if comp_schedule_dict["tiling"] and (
                iterator_name in comp_schedule_dict["tiling"]["tiling_dims"]
            ):
                tiled = 1
                tile_factor_index = comp_schedule_dict["tiling"]["tiling_dims"].index(
                    iterator_name
                )
                tile_factor = int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tile_factor_index]
                )
            p_index = comps_placeholders_indices_dict[l_code + "Tiled"]
            comps_repr[p_index[0]][p_index[1]] = tiled
            p_index = comps_placeholders_indices_dict[l_code + "TileFactor"]
            comps_repr[p_index[0]][p_index[1]] = tile_factor

            # Check whether fusion was applied and put the tag in its corresponding position in the computation representation
            fused = 0
            if iter_i in fused_levels:
                fused = 1
            p_index = comps_placeholders_indices_dict[l_code + "Fused"]
            comps_repr[p_index[0]][p_index[1]] = fused
            
            shifted = 0
            shifting_factor = 0
            if comp_schedule_dict['shiftings']:
                for shifting in comp_schedule_dict['shiftings']: 
                    if iterator_name.startswith(shifting[0]): # loof if the current loop is being shifted
                        shifted=1
                        shifting_factor = shifting[1]
                        break
            p_index = comps_placeholders_indices_dict[l_code + "Shifted"]
            comps_repr[p_index[0]][p_index[1]] = shifted
            p_index = comps_placeholders_indices_dict[l_code + "ShiftFactor"]
            comps_repr[p_index[0]][p_index[1]] = shifting_factor
        # Check whether unrolling was applied and put the tags in their corresponding position in the computation representation
        unrolled = 0
        unroll_factor = 0
        if comp_schedule_dict["unrolling_factor"]:
            unrolled = 1
            unroll_factor = int(comp_schedule_dict["unrolling_factor"])
            
        p_index = comps_placeholders_indices_dict[c_code + "-Unrolled"]
        comps_repr[p_index[0]][p_index[1]] = unrolled
        
        p_index = comps_placeholders_indices_dict[c_code + "-UnrollFactor"]
        comps_repr[p_index[0]][p_index[1]] = unroll_factor
        
        # Check which transformations (interchange, reversal and skweing) were applied and add the padded vector representation to their corresponding position
        padded_tags = get_padded_transformation_tags(
            program_json, schedule_json, comp_name
        )
        
        tags_start = comps_placeholders_indices_dict[ c_code + "-TransformationTagsStart" ]
        
        tags_end = comps_placeholders_indices_dict[c_code + "-TransformationTagsEnd"]
        
        nb_tags_elements = tags_end[1] - tags_start[1] + 1
        
        assert len(padded_tags) == nb_tags_elements
        
        comps_repr[tags_start[0]][tags_start[1] : tags_end[1] + 1] = padded_tags
        
        # Add the padded original constraints matrix to the representation
        ogc_start = comps_placeholders_indices_dict[c_code+'-OgConstraintMatrixStart']
        
        ogc_end = comps_placeholders_indices_dict[c_code+'-OgConstraintMatrixEnd']
        
        nb_mat_elements = ogc_end[1] - ogc_start[1] + 1
        
        assert(MAX_DEPTH*MAX_DEPTH*2 == nb_mat_elements)
        
        padded_coeff_mat, padded_constants_col = get_padded_initial_iteration_domain(program_json,comp_name, pad=True)
        
        comps_repr[ogc_start[0]][ogc_start[1] : ogc_end[1] + 1 ] = padded_coeff_mat.flatten().tolist()
        
        
        # Add the padded original constraints vector to the representation
        ogv_start = comps_placeholders_indices_dict[c_code+'-OgConstraintVectorStart']
        
        ogv_end = comps_placeholders_indices_dict[c_code+'-OgConstraintVectorEnd']
        
        nb_mat_elements = ogv_end[1] - ogv_start[1] + 1
        
        comps_repr[ogv_start[0]][ogv_start[1] : ogv_end[1] + 1 ] = padded_constants_col.tolist()
        
        # Add the padded transformed constraints vector to the representation
        c_start = comps_placeholders_indices_dict[c_code+'-ConstraintMatrixStart']
        
        c_end = comps_placeholders_indices_dict[c_code+'-ConstraintMatrixEnd']
        
        nb_mat_elements = c_end[1] - c_start[1] + 1

        assert(MAX_DEPTH*MAX_DEPTH*2 == nb_mat_elements)
        
        comps_repr[c_start[0]][ c_start[1] : c_end[1] + 1 ] = get_padded_transformed_iteration_domain(program_json, schedule_json, comp_name).flatten().tolist()
        

    # Fill the loop representation
    # Initialization
    loop_schedules_dict = dict()
    for loop_name in program_json["iterators"]:
        loop_schedules_dict[loop_name] = dict()
        loop_schedules_dict[loop_name]["tiled"] = 0
        loop_schedules_dict[loop_name]["tile_factor"] = 0
        loop_schedules_dict[loop_name]["unrolled"] = 0
        loop_schedules_dict[loop_name]["unroll_factor"] = 0
        loop_schedules_dict[loop_name]["shifted"] = 0
        loop_schedules_dict[loop_name]["shift_factor"] = 0
        loop_schedules_dict[loop_name]["parallelized"] = 0
        loop_schedules_dict[loop_name]["fused"] = 0

    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_schedule_dict = schedule_json[comp_name]
        # If this computation was moved to a different loop nest through the application of fusion,
        # we know that it will use the same iterators as the computations it was moved to.
        # The two computations thus share the same schedule (Since we only apply loop transformations and they share the same loops)
        if ("fusions" in schedule_json and schedule_json["fusions"]):
            for fusion in schedule_json["fusions"]:
                if comp_name in fusion:
                    iterator_comp_name = fusion[0]
                    transf_loop_nest = program_json["computations"][iterator_comp_name]["iterators"].copy()
                    comp_schedule_dict = schedule_json[iterator_comp_name]
        
        # Check whether tiling was applied 
        if comp_schedule_dict["tiling"]:
            for tiled_loop_index, tiled_loop in enumerate(
                comp_schedule_dict["tiling"]["tiling_dims"]
            ):
                loop_schedules_dict[tiled_loop]["tiled"] = 1
                
                # Make sure this loop either hasn't yet been tiled or has beeen tiled by the same factor
                assert(loop_schedules_dict[tiled_loop]["tile_factor"] == 0 or loop_schedules_dict[tiled_loop]["tile_factor"] == int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tiled_loop_index]))
                loop_schedules_dict[tiled_loop]["tile_factor"] = int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tiled_loop_index]
                )
                
        # Check whether unrolling was applied 
        if comp_schedule_dict["unrolling_factor"]:
            
            comp_innermost_loop = get_comp_iterators_from_tree_struct(schedule_json, comp_name)[-1]
            loop_schedules_dict[comp_innermost_loop]["unrolled"] = 1
            
            # Make sure this loop either hasn't yet been unrolled or has beeen unrolled by the same factor
            assert (loop_schedules_dict[comp_innermost_loop]["unroll_factor"] == 0 or loop_schedules_dict[comp_innermost_loop]["unroll_factor"] == int(comp_schedule_dict["unrolling_factor"]))
            
            loop_schedules_dict[comp_innermost_loop]["unroll_factor"] = int(comp_schedule_dict["unrolling_factor"])
            
        # Check whether parallelization was applied 
        if comp_schedule_dict["parallelized_dim"]:
            loop_schedules_dict[comp_schedule_dict["parallelized_dim"]]["parallelized"] = 1
        
        
        if comp_schedule_dict['shiftings']:
            for shifting in comp_schedule_dict['shiftings']: 
                loop_schedules_dict[shifting[0]]["shifted"] = 1
                loop_schedules_dict[shifting[0]]["shift_factor"] = shifting[1]
        
    # Check whether fusion was applied 
    if "fusions" in schedule_json and schedule_json["fusions"]:
        for fusion in schedule_json["fusions"]:
            fused_loop1 = computations_dict[fusion[0]]["iterators"][fusion[2]]
            fused_loop2 = computations_dict[fusion[1]]["iterators"][fusion[2]]
            loop_schedules_dict[fused_loop1]["fused"] = 1
            loop_schedules_dict[fused_loop2]["fused"] = 1
            
    
    # Get the index of each feature in the loop representation and replace it with the the information obtained from the schedule
    for loop_name in program_json["iterators"]:
        l_code = "L" + loop_name

        p_index = loops_placeholders_indices_dict[l_code + "Parallelized"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
            "parallelized"
        ]

        p_index = loops_placeholders_indices_dict[l_code + "Tiled"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["tiled"]
        p_index = loops_placeholders_indices_dict[l_code + "TileFactor"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
            "tile_factor"
        ]

        p_index = loops_placeholders_indices_dict[l_code + "Unrolled"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["unrolled"]
        p_index = loops_placeholders_indices_dict[l_code + "UnrollFactor"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
            "unroll_factor"
        ]
        
        p_index = loops_placeholders_indices_dict[l_code + "Shifted"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["shifted"]
        p_index = loops_placeholders_indices_dict[l_code + "ShiftFactor"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
            "shift_factor"
        ]
        
        p_index = loops_placeholders_indices_dict[l_code + "Fused"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["fused"]
    comp_transformed_iterators = get_comp_iterators_from_tree_struct(schedule_json, comp_name)    
    # Check if any iterators were removed because of fusion
    if (len(program_json["iterators"])>len(comp_transformed_iterators)):
        # If this is the case, add the missing vectors with zeros in all the transformations
        removed_iterators = list(set(program_json["iterators"]) - set(comp_transformed_iterators))
        for loop_name in removed_iterators:
            l_code = "L" + loop_name

            p_index = loops_placeholders_indices_dict[l_code + "Parallelized"]
            loops_repr[p_index[0]][p_index[1]] = 0

            p_index = loops_placeholders_indices_dict[l_code + "Tiled"]
            loops_repr[p_index[0]][p_index[1]] = 0
            p_index = loops_placeholders_indices_dict[l_code + "TileFactor"]
            loops_repr[p_index[0]][p_index[1]] = 0

            p_index = loops_placeholders_indices_dict[l_code + "Unrolled"]
            loops_repr[p_index[0]][p_index[1]] = 0
            p_index = loops_placeholders_indices_dict[l_code + "UnrollFactor"]
            loops_repr[p_index[0]][p_index[1]] = 0
            
            p_index = loops_placeholders_indices_dict[l_code + "Shifted"]
            loops_repr[p_index[0]][p_index[1]] = 0
            p_index = loops_placeholders_indices_dict[l_code + "ShiftFactor"]
            loops_repr[p_index[0]][p_index[1]] = 0
            
            p_index = loops_placeholders_indices_dict[l_code + "Fused"]
            loops_repr[p_index[0]][p_index[1]] = 0

    
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr), 0)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr), 0)
    comps_expr_repr = torch.tensor([comps_expr_repr]).float()
    
    return computations_tensor, loops_tensor, comps_expr_repr

# Get the representation of the whole expression recursively
def get_tree_expr_repr(node, comp_type):
        expr_tensor = []
        if node["children"] != []:
            for child_node in node["children"]:
                expr_tensor.extend(get_tree_expr_repr(child_node, comp_type))
        expr_tensor.append(get_expr_repr(node["expr_type"], comp_type))

        return expr_tensor

# One-hot encoding for expressions and their datatypes
def get_expr_repr(expr, comp_type):
        expr_vector = []
        if(expr == "add"):
            expr_vector = [1, 0, 0, 0, 0, 0, 0, 0]
        elif(expr == "sub"):
            expr_vector = [0, 1, 0, 0, 0, 0, 0, 0]
        elif(expr == "mul"):
            expr_vector = [0, 0, 1, 0, 0, 0, 0, 0]
        elif(expr == "div"):
            expr_vector = [0, 0, 0, 1, 0, 0, 0, 0]
        elif(expr == "sqrt"):
            expr_vector = [0, 0, 0, 0, 1, 0, 0, 0]
        elif(expr == "min"):
            expr_vector = [0, 0, 0, 0, 0, 1, 0, 0]
        elif(expr == "max"):
            expr_vector = [0, 0, 0, 0, 0, 0, 1, 0]
        else:
            expr_vector = [0, 0, 0, 0, 0, 0, 0, 1]
        
        comp_type_vector = []
        if(comp_type == "int32"):
            comp_type_vector = [1, 0, 0]
        elif(comp_type == "float32"):
            comp_type_vector = [0, 1, 0]
        elif(comp_type == "float64"):
            comp_type_vector = [0, 0, 1]
            
        return expr_vector + comp_type_vector

# Add padding to the read/write access matrices
def pad_access_matrix(access_matrix):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    padded_access_matrix = np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 2))
    padded_access_matrix[
        : access_matrix.shape[0], : access_matrix.shape[1] - 1
    ] = access_matrix[:, :-1]
    padded_access_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]

    return padded_access_matrix


# Solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
# Used to get skewing parameters
def linear_diophantine_default(f_i, f_j):
    n1 = abs(f_i)
    n2 = abs(f_j)
    
    while(n1 != n2):
        if(n1 > n2):
            n1 -=  n2
        else:
            n2 -=  n1
            
    # Update f_i and f_j to equivalent but prime between themselfs value
    f_i = f_i / n1
    f_j = f_j / n1
    
    found = False
    gamma = 0
    sigma = 1
    
    if (f_j == 1) or (f_i == 1):
        gamma = f_i - 1
        sigma = 1
        # Since sigma = 1  then
        # f_i - gamma * f_j = 1 & using the previous condition :
        #  - f_i = 1 : then gamma = 0 (f_i-1) is enough
        #  - f_j = 1 : then gamma = f_i -1  
    else:
        if (f_j == -1) and (f_i > 1):
            gamma = 1
            sigma = 0
        else:
            # General case : solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
            i = 0
            while (i < 100) and (not found):
                if ((sigma * f_i) % abs(f_j)) == 1:
                    found = True
                else:
                    sigma += 1
                    i += 1
            if not found:
                # Detect infinite loop and prevent it in case where f_i and f_j are not prime between themselfs
                print("Error cannof find solution to diophantine equation")
                return
            gamma = ((sigma * f_i) - 1) / f_j
    return gamma, sigma


# Tranfrom the access relations to matrices
def isl_to_write_matrix(isl_map):
    comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
    for i, buf_iter in enumerate(buf_iter_names):
        for j, comp_iter in enumerate(comp_iter_names):
            if buf_iter == comp_iter:
                matrix[i, j] = 1
                break
    return matrix


def isl_to_write_dims(
    isl_map,
):
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    return buf_iter_names

# Get the involved computations from a specific node 
def get_involved_comps(node):
        result = []
        if(len(node)==0): 
            return result
        for comp in node["computations_list"]:
            result.append(comp)
        for child in node["child_list"]:
            for comp in get_involved_comps(child):
                result.append(comp)
        return result

# Retrieve the iterators that involve this computation from the schedule tree_structure
def get_comp_iterators_from_tree_struct(schedule_json, comp_name):
    tree = schedule_json["tree_structure"]
    level = tree
    iterators = []
    to_explore = []
    # only add the root that contains the computation we are looking for
    for root in tree["roots"]:
        if (comp_name in get_involved_comps(root)):
            to_explore.append(root)
    
    while(to_explore):
        level = to_explore.pop(0)
        if(comp_name in get_involved_comps(level)):
            iterators.append(level['loop_name'])
            
        for element in level["child_list"]:
            to_explore.append(element)
    
    return iterators

# Helper function to return lines from the constraint matrix
def format_bound(iterator_name, bound, iterators_list, is_lower):
    output = []
    for i in iterators_list:
        if i == iterator_name:
            if is_lower :
                output.append(-1)
            else:
                output.append(1)
        elif (i == bound):
            if is_lower :
                output.append(1)
            else:
                output.append(-1)
        else:
            output.append(0)
    return output

# A constraint matrix is the set of linear inequalities that describes the iteration domain.
# Example:
# if the iteration domain D is the follwoing
#     {i >= 0
#      i < 128
# D =  j >= 0
#      j < 32
#      k >= 0
#      k < 64}
# The iterator vector is 
# x = [i, 
#      j, 
#      k]
# The coeffcients matrix A would be:
#     [-1,   0,   0,
#       1,   0,   0,
# A=    0,  -1,   0,
#       0,   1,   0,
#       0,   0,  -1,
#       0,   0,   1]
# The second hand side of the equation b (constants vector) is the vector:
#     b= [0,
#         127,
#         0,
#         31,
#         0,
#         63]
# Since:
#    D = Ax<=b
# Get the matrix describing the initial constraints for this program
# EDIT: Padding is done with zeros, this idea is no longer in use. ignore the following comment. --- For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner. The constants vector is padded with zeros
def get_padded_initial_iteration_domain(program_json,comp_name, pad=True):
    #supports bounds of type ± i ± j ±...± cst and max(cst, iter)
    comp_dict = program_json['computations'][comp_name] 
    nb_dims = len(comp_dict['iterators'])
    
    coeff_mat = np.zeros((nb_dims*2,nb_dims),int)
    constants_col = np.zeros((nb_dims*2),int)
    
    for i,iterator_name_row in enumerate(comp_dict['iterators']):# rows loop
        
        # set the diagonal-ish values
        coeff_mat[i*2,i]=-1
        coeff_mat[i*2+1,i]=1
        
        iterator_row_dict = program_json['iterators'][iterator_name_row]
        
        #get the simplified expression of the bounds
        upper_bound = str(sympy.simplify(iterator_row_dict['upper_bound'])).replace(' ','')
        lower_bound = str(sympy.simplify(iterator_row_dict['lower_bound'])).replace(' ','')
        
        # in some special cases, the lower bound of i appears as max(cst, iter) when iter<=i. The following is a non-general solution
        if 'max' in lower_bound or 'Max' in lower_bound:
            lower_bound = re.findall('[mM]ax\(.+,(.+)\)',lower_bound)[0]
        
        # find the iterator names and constants used in the bounds
        iterators_in_upper = re.findall('[a-zA-Z]\w*', upper_bound)
        constants_in_upper = re.findall('(?:^|\+|-)\d+', upper_bound)
        iterators_in_lower = re.findall('[a-zA-Z]\w*', lower_bound)
        constants_in_lower = re.findall('(?:^|\+|-)\d+', lower_bound)
        
        #if no constants used, set to 0
        if not constants_in_upper:
            constants_in_upper = [0]
        else:
            assert len(constants_in_upper)==1
       
        if not constants_in_lower:
            constants_in_lower = [0]
        else:
            assert len(constants_in_lower)==1
        
        #for each iterator in the bounds expression, set the corresponding values in the matrix
        for iter_name in iterators_in_upper:
            if iter_name not in comp_dict["iterators"]:
                # replace iter_name with the first iterator that starts with iter_name in the list of iterators
                for iterator in comp_dict["iterators"]:
                    if iterator.startswith(iter_name):
                        iter_name = iterator
                        break
            col_idx = comp_dict['iterators'].index(iter_name)
            if '-'+iter_name in upper_bound:
                coeff_mat[i*2+1,col_idx]=1
            else:
                coeff_mat[i*2+1,col_idx]=-1
        constants_col[i*2+1]= int(constants_in_upper[0])-1 #adding a -1 because we are representing a non-strict inequality
        
        for iter_name in iterators_in_lower:
            if iter_name not in comp_dict["iterators"]:
                # replace iter_name with the first iterator that starts with iter_name in the list of iterators
                for iterator in comp_dict["iterators"]:
                    if iterator.startswith(iter_name):
                        iter_name = iterator
                        break
            col_idx = comp_dict['iterators'].index(iter_name)
            if '-'+iter_name in upper_bound:
                coeff_mat[i*2,col_idx]=-1
            else:
                coeff_mat[i*2,col_idx]=+1
        constants_col[i*2]= -int(constants_in_lower[0])
                
#     constants_col = constants_col.reshape(-1,1)
    
    
    # Add padding if requested
    if pad:
        padded_coeff_mat = np.pad(coeff_mat, [(0,MAX_DEPTH*2-nb_dims*2),(0,MAX_DEPTH-nb_dims)], mode='constant', constant_values=0)
        
#         #Edit: this idea has been dropped.!! For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner
#         for i in range(nb_dims,MAX_DEPTH):
#             padded_coeff_mat[i*2,i]=-1
#             padded_coeff_mat[i*2+1,i]=1
            
        padded_constants_col = np.pad(constants_col, [(0,MAX_DEPTH*2-nb_dims*2)], mode='constant', constant_values=0)
        return padded_coeff_mat, padded_constants_col
    else:
        return coeff_mat, constants_col

# Get the matrix describing the iteration domain after applying a sequence of affine transformations
# The transformed constraint matrix is: the original constraint matrix multiplied by the inverse of the transformation matrix
def get_padded_transformed_iteration_domain(program_json, schedule_json, comp_name, pad=True):
    # Extract the transformations matrix for this schedule
    transformation_matrix = get_transformation_matrix(program_json, schedule_json, comp_name)
    
    # Create the initial constraint matrix without any padding
    A,b = get_padded_initial_iteration_domain(program_json,comp_name, pad=False)
    nb_dims = A.shape[1]
        
    # Get the inverse of the transformation matrix
    inverse = np.linalg.inv(transformation_matrix)
    
    # Multiply thw two to gte the transformed constraint matrix
    result = np.matmul(A, inverse)
    result = np.array(result)
    
    if pad:
        result = np.pad(result, [(0, (MAX_DEPTH)*2 - result.shape[0]), (0, MAX_DEPTH - result.shape[1])],
        mode="constant",
        constant_values=0)
         #EDIT: Padding is done with zeros, this idea is no longer in use. ignore the following comment. ---   For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner
#         for i in range(nb_dims,MAX_DEPTH):
#             result[i*2,i]=-1
#             result[i*2+1,i]=1
        
    return result

# Convert a tags vector describing an affine transfromation (Reversal, Skewing, Interchange) into a matrix that represents the same transformation
def get_trasnformation_matrix_from_vector(transformation, matrix_size):
    matrix = np.identity(matrix_size)
    assert(len(transformation) == MAX_TAGS)
    if (transformation[0] == 1):
        # Interchange
        assert(transformation[1] < matrix_size and transformation[2] < matrix_size)
        matrix[transformation[1], transformation[2]] = 1
        matrix[transformation[1], transformation[1]] = 0
        matrix[transformation[2], transformation[1]] = 1
        matrix[transformation[2], transformation[2]] = 0

    elif (transformation[0] == 2):
        # Reversal
        assert(transformation[3] < matrix_size)
        matrix[transformation[3], transformation[3]] = -1

    elif transformation[0] == 3:
        # 2D Skewing
        if transformation[6] == 0:
            
            assert(transformation[4] < matrix_size and transformation[5] < matrix_size)
            matrix[transformation[4], transformation[4]] = transformation[7]
            matrix[transformation[4], transformation[5]] = transformation[8]
            matrix[transformation[5], transformation[4]] = transformation[9]
            matrix[transformation[5], transformation[5]] = transformation[10]
        if transformation[6] > 0:
            # 3D skeweing
            assert(transformation[4] < matrix_size and transformation[5] < matrix_size and transformation[6] < matrix_size)
            matrix[transformation[4], transformation[4]] = transformation[7]
            matrix[transformation[4], transformation[5]] = transformation[8]
            matrix[transformation[4], transformation[6]] = transformation[9]
            matrix[transformation[5], transformation[4]] = transformation[10]
            matrix[transformation[5], transformation[5]] = transformation[11]
            matrix[transformation[5], transformation[6]] = transformation[12]
            matrix[transformation[6], transformation[4]] = transformation[13]
            matrix[transformation[6], transformation[5]] = transformation[14]
            matrix[transformation[6], transformation[6]] = transformation[15]
        
    return matrix

# Transform a sequence of transformation vectors into a single transfromation matrix that represents the whole sequence
def get_transformation_matrix(
    program_json, schedule_json, comp_name
):
    nb_iterators = len(program_json["computations"][comp_name]["iterators"])
    final_transformation = np.identity(nb_iterators)
    for transformation in schedule_json[comp_name]["transformations_list"]:
        matrix = get_trasnformation_matrix_from_vector(transformation, nb_iterators)
        final_transformation = np.matmul(matrix, final_transformation)
    return final_transformation


# Check whether the string contains an integer and return true if so
def is_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

# Returns a vector that represents the right hand sise of teh constraint matrix inequalities
# (The vector b from the previous example)
def get_padded_second_side_of_the_constraint_equations_original(program_json, schedule_json, comp_name):
    iterators_list = program_json["computations"][comp_name]["iterators"]
    result = []
    for it in iterators_list:
        if(is_int(program_json["iterators"][it]["lower_bound"])):
            result.append(int(program_json["iterators"][it]["lower_bound"]))
        else:
#             shift = extract_shift_from_bound(program_json["iterators"][it]["lower_bound"])
            result.append(0)
        if(is_int(program_json["iterators"][it]["upper_bound"])):
            result.append(int(program_json["iterators"][it]["upper_bound"]))
        else:
#             shift = extract_shift_from_bound(program_json["iterators"][it]["upper_bound"])
            result.append(0)
    result = result + [0]*(MAX_DEPTH*2-len(result))
    return result

# A function to extract the transformations applied on a spesific computation in the form of a vector of tags
# Padding is added if the number of transformations is less than the maximum value of MAX_NUM_TRANSFORMATIONS
# The tag representation is as follows:
#         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'third_skewing_loop', 'skew_parameter_1', 'skew_parameter_2', 'skew_parameter_3', 'skew_parameter_4', 'skew_parameter_5', 'skew_parameter_6', 'skew_parameter_7', 'skew_parameter_8', 'skew_parameter_9']
#     Where the type_of_transformation tag is:
#         - 0 for no transformation being applied
#         - 1 for loop interchange
#         - 2 for loop reversal
#         - 3 for loop skewing
# In the case for skewing we are specifying the new values for the transformed submatrix
def get_padded_transformation_tags(
    program_json, schedule_json, comp_name
):
    # Extract information about the computation and the transformations that were applied from the json input
    comp_dict = program_json["computations"][comp_name]
    comp_schedule_dict = schedule_json[comp_name]
    nb_iterators = len(comp_dict["iterators"])
    loop_nest = comp_dict["iterators"][:]
    
    # Create an identity vector that represents that no transformation was applied
    identity = np.zeros((nb_iterators, nb_iterators), int)
    np.fill_diagonal(identity, 1)
    identity_tags = np.zeros((1,MAX_TAGS), dtype=np.int32)
    
    tag_factors = []
    for transformation in comp_schedule_dict['transformations_list']:
        tag_factors.append(transformation)
    
    
    # Add padding to the sequence of vectors in case the number of transformations is less than MAX_NUM_TRANSFORMATIONS+1
    tags_list = [item for sublist in tag_factors for item in sublist]
    tags_list += [0]*(MAX_NUM_TRANSFORMATIONS*MAX_TAGS - len(tags_list)) 
    
    return tags_list