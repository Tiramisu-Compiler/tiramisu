import copy
import json
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from torch.nn.utils.rnn import pad_sequence


device = "cpu"
train_device = torch.device("cpu")


class NbMatricesException(Exception):
    pass


class NbAccessException(Exception):
    pass


class LoopsDepthException(Exception):
    pass



global_dioph_sols_dict = dict()
MAX_MATRICES = 5


def get_representation_template(program_json, no_sched_json, max_depth, train_device="cpu"):
    max_accesses = 15
    min_accesses = 1

    comps_repr_templates_list = []
    comps_expr_repr_templates_list = []
    comps_indices_dict = dict()
    comps_placeholders_indices_dict = dict()

    # program_json = program_dict["program_annotation"]
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )
    
    def get_expr_repr(expr):
        if(expr == "add"):
            return [1, 0, 0, 0, 0]
        elif(expr == "sub"):
            return [0, 1, 0, 0, 0]
        elif(expr == "mul"):
            return [0, 0, 1, 0, 0]
        elif(expr == "div"):
            return [0, 0, 0, 1, 0]
        # elif(expr == "value"):
        #     return [0, 0, 0, 0, 1, 0, 0, 0]
        # elif(expr == "access"):
        #     return [0, 0, 0, 0, 0, 1, 0, 0]
        # elif(expr == "buffer"):
        #     return [0, 0, 0, 0, 0, 0, 1, 0]
        else:
            return [0, 0, 0, 0, 1]

    def get_tree_expr_repr(node):
        expr_tensor = []
        if node["children"] != []:
            for child_node in node["children"]:
                expr_tensor.extend(get_tree_expr_repr(child_node))
        expr_tensor.append(get_expr_repr(node["expr_type"]))

        return expr_tensor
    
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        # to get the expression representation
        expr_dict = comp_dict["expression_representation"]
        comps_expr_repr_templates_list.append(get_tree_expr_repr(expr_dict))
        
        if len(comp_dict["accesses"]) > max_accesses:
            raise NbAccessException
        # if len(comp_dict["accesses"]) < min_accesses:
        #     raise NbAccessException
        if len(comp_dict["iterators"]) > max_depth:
            raise LoopsDepthException

        comp_repr_template = []

        comp_repr_template.append(+comp_dict["comp_is_reduction"])

        iterators_repr = []
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            iterator_dict = program_json["iterators"][iterator_name]
            # iterators_repr.extend(
            #     [iterator_dict["lower_bound"], iterator_dict["upper_bound"]]
            # )
            try:
                iterators_repr.append(int(iterator_dict["lower_bound"]))
            except:
                iterators_repr.append(0)
            try:
                iterators_repr.append(int(iterator_dict["upper_bound"]))
            except:
                iterators_repr.append(0)

            c_code = "C" + str(comp_index)
            l_code = c_code + "-L" + str(iter_i)
            iterators_repr.extend(
                [
                    l_code + "Parallelized",
                    l_code + "Tiled",
                    l_code + "TileFactor",
#                     l_code + "Fused",
                ]
            )

        iterator_repr_size = int(len(iterators_repr) / len(comp_dict["iterators"]))
        iterators_repr.extend(
            [0] * iterator_repr_size * (max_depth - len(comp_dict["iterators"]))
        )

        iterators_repr.extend([c_code + "-Unrolled", c_code + "-UnrollFactor"])

        iterators_repr.append(c_code + "-TransformationMatrixStart")
        # MAX_MATRICES is the max number of matrices
        iterators_repr.extend(["M"] * (((2*max_depth+2 + 1) ** 2) * MAX_MATRICES - 2))
        iterators_repr.append(c_code + "-TransformationMatrixEnd")
        
        # Adding initial constraint matrix
        iterators_repr.append(c_code+'-OgConstraintMatrixStart')
        iterators_repr.extend(['C']*((max_depth+1)*((max_depth+1)*2)-2))
        iterators_repr.append(c_code+'-OgConstraintMatrixEnd')
        # Adding transformed constraint matrix
        iterators_repr.append(c_code+'-ConstraintMatrixStart')
        iterators_repr.extend(['C']*((max_depth+1)*((max_depth+1)*2)-2))
        iterators_repr.append(c_code+'-ConstraintMatrixEnd')
        
        # Adding static dimensions for the iterators
        iterators_repr.append(c_code+'-OgStaticDimensionsStart')
        iterators_repr.extend(['S']*((max_depth+1)-2))
        iterators_repr.append(c_code+'-OgStaticDimensionsEnd')    
         # Adding static dimensions for the iterators
        iterators_repr.append(c_code+'-StaticDimensionsStart')
        iterators_repr.extend(['S']*((max_depth+1)-2))
        iterators_repr.append(c_code+'-StaticDimensionsEnd')
        
        
        comp_repr_template.extend(iterators_repr)

        padded_write_matrix = pad_access_matrix(
            isl_to_write_matrix(comp_dict["write_access_relation"]), max_depth
        )
        write_access_repr = [
            comp_dict["write_buffer_id"] + 1
        ] + padded_write_matrix.flatten().tolist()

        comp_repr_template.extend(write_access_repr)

        read_accesses_repr = []
        for read_access_dict in comp_dict["accesses"]:
            read_access_matrix = pad_access_matrix(
                read_access_dict["access_matrix"], max_depth
            )
            read_access_repr = (
                [+read_access_dict["access_is_reduction"]]
                + [read_access_dict["buffer_id"] + 1]
                + read_access_matrix.flatten().tolist()
            )
            read_accesses_repr.extend(read_access_repr)

        access_repr_len = (max_depth + 1) * (max_depth + 2) + 1 + 1
        read_accesses_repr.extend(
            [0] * access_repr_len * (max_accesses - len(comp_dict["accesses"]))
        )

        comp_repr_template.extend(read_accesses_repr)

        # comp_repr_template.append(comp_dict["number_of_additions"])
        # comp_repr_template.append(comp_dict["number_of_subtraction"])
        # comp_repr_template.append(comp_dict["number_of_multiplication"])
        # comp_repr_template.append(comp_dict["number_of_division"])

        comps_repr_templates_list.append(comp_repr_template)
        comps_indices_dict[comp_name] = comp_index
        for j, element in enumerate(comp_repr_template):
            if isinstance(element, str):
                comps_placeholders_indices_dict[element] = (comp_index, j)

    loops_repr_templates_list = []
    loops_indices_dict = dict()
    loops_placeholders_indices_dict = dict()

    for loop_index, loop_name in enumerate(program_json["iterators"]):
        loop_repr_template = []
        l_code = "L" + loop_name

        # loop_repr_template.extend(
        #     [
        #         program_json["iterators"][loop_name]["lower_bound"],
        #         program_json["iterators"][loop_name]["upper_bound"],
        #     ]
        # )
        try:
            loop_repr_template.append(int(program_json["iterators"][loop_name]["lower_bound"]))
        except:
            loop_repr_template.append(0)
        try:
            loop_repr_template.append(int(program_json["iterators"][loop_name]["upper_bound"]))
        except:
            loop_repr_template.append(0)
        loop_repr_template.extend(
            [
                l_code + "Parallelized",
                l_code + "Tiled",
                l_code + "TileFactor",
#                 l_code + "Fused",
                l_code + "Unrolled",
                l_code + "UnrollFactor",
            ]
        )
        loop_repr_template.extend(
            [l_code + "TransfMatRowStart"]
            + ["M"] * ((2*max_depth+2 + 1) - 2)
            + [l_code + "TransfMatRowEnd"]
        )
        loop_repr_template.extend(
            [l_code + "TransfMatColStart"]
            + ["M"] * (2*max_depth+2 - 2 + 1)
            + [l_code + "TransfMatColEnd"]
        )

        loops_repr_templates_list.append(loop_repr_template)
        loops_indices_dict[loop_name] = loop_index

        for j, element in enumerate(loop_repr_template):
            if isinstance(element, str):
                loops_placeholders_indices_dict[element] = (loop_index, j)

    def update_tree_atributes(node, train_device="cpu"):
        if "roots" in node :
            for root in node["roots"]:
                update_tree_atributes(root, train_device=train_device)
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
            update_tree_atributes(child_node, train_device=train_device)
        return node

    assert "fusions" not in no_sched_json or no_sched_json["fusions"] == None
    orig_tree_structure = no_sched_json["tree_structure"]
    tree_annotation = copy.deepcopy(orig_tree_structure)
    prog_tree = update_tree_atributes(tree_annotation, train_device=train_device)
    
    # adding padding to the expressions to get same size expressions
    max_exprs = 0
    max_exprs = max([len(comp) for comp in comps_expr_repr_templates_list])
    lengths = []
    for j in range(len(comps_expr_repr_templates_list)):
        lengths.append(len(comps_expr_repr_templates_list[j]))
        comps_expr_repr_templates_list[j].extend(
            [[0, 0, 0, 0, 0]] * (max_exprs - len(comps_expr_repr_templates_list[j])))
    comps_expr_lengths = torch.tensor(lengths)
    comps_expr_repr_templates_list = torch.tensor([comps_expr_repr_templates_list])
    return (
        prog_tree,
        comps_repr_templates_list,
        loops_repr_templates_list,
        comps_placeholders_indices_dict,
        loops_placeholders_indices_dict,
        comps_expr_repr_templates_list,
        comps_expr_lengths,
    )


def pad_access_matrix(access_matrix, max_depth):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
    padded_access_matrix[
        : access_matrix.shape[0], : access_matrix.shape[1] - 1
    ] = access_matrix[:, :-1]
    padded_access_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]

    return padded_access_matrix


def linear_diophantine_default(f_i, f_j):
    found = False
    gamma = 0
    sigma = 1
    if (f_j == 1) or (f_i == 1):
        gamma = f_i - 1
        sigma = 1
    else:
        if (f_j == -1) and (f_i > 1):
            gamma = 1
            sigma = 0
        else:
            i = 0
            while (i < 100) and (not found):
                if ((sigma * f_i) % abs(f_j)) == 1:
                    found = True
                else:
                    sigma += 1
                    i += 1
            if not found:
                print("Error cannof find solution to diophantine equation")
                return
            gamma = ((sigma * f_i) - 1) / f_j

    return gamma, sigma


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


def get_original_static_dims(program_json):
    iterators = program_json['iterators']
    result = []
    nb_comps = len(program_json['computations'])
    for i in range(nb_comps):
        result.append([])
    
    level = iterators
    to_explore = []
    to_explore.append(list(iterators.keys())[0]) # why 0 ? 
    while(to_explore):
        it_name = to_explore.pop(0)
        iterator = iterators[it_name]
        if (len(iterator["child_iterators"])>1):
            static_dim = 0
            for it in iterator["child_iterators"]:
                involved_comps = get_involved_comps_from_iterator(it, program_json )
                for i in involved_comps:
                    comp_index = program_json['computations'][i]['absolute_order']-1
                    result[comp_index].append(static_dim)
                static_dim = static_dim + 10
        else:
            if(len(iterator["child_iterators"]) == 1):
                involved_comps = get_involved_comps_from_iterator(it_name, program_json)
                for i in involved_comps:
                    
                    comp_index = program_json['computations'][i]['absolute_order']-1
                    result[comp_index].append(0)
            else:
                if(len(iterator["computations_list"])>0):
                    involved_comps = get_involved_comps_from_iterator(it_name, program_json)
                    static_dim = 0
                    for i in involved_comps:
                        
                        comp_index = program_json['computations'][i]['absolute_order']-1
                        result[comp_index].append(static_dim)
                        static_dim = static_dim + 10
        for element in iterator["child_iterators"]:
            to_explore.append(element)
    return result
def get_static_dims(schedule_json, program_json):
    tree_structure = schedule_json['tree_structure']
    result = []
    nb_comps = len(program_json['computations'])
    
    for i in range(nb_comps):
        result.append([])
    
    level = tree_structure
    to_explore = [root for root in tree_structure["roots"]]
    # to_explore.append(tree_structure)
    while(to_explore):
        level = to_explore.pop(0)
        if (len(level["child_list"])>1):
            static_dim = 0
            for sub_level in level["child_list"]:
                involved_comps = get_involved_comps(sub_level)
                for i in involved_comps:
                    
                    comp_index = program_json['computations'][i]['absolute_order']-1
                    result[comp_index].append(static_dim)
                static_dim = static_dim + 10
        else:
            if(len(level["child_list"]) == 1):
                involved_comps = get_involved_comps(level)
                for i in involved_comps:
                    
                    comp_index = program_json['computations'][i]['absolute_order']-1
                    result[comp_index].append(0)
            else:
                if(len(level["computations_list"])>0):
                    involved_comps = get_involved_comps(level)
                    static_dim = 0
                    for i in involved_comps:
                        
                        comp_index = program_json['computations'][i]['absolute_order']-1
                        result[comp_index].append(static_dim)
                        static_dim = static_dim + 10
        for element in level["child_list"]:
            to_explore.append(element)
    return result
def get_involved_comps_from_iterator(iterator, program_json):
        result = []
        node = program_json['iterators'][iterator]
        if(len(node)==0): 
            return result
        for comp in node["computations_list"]:
            result.append(comp)
        for child in node["child_iterators"]:
            for comp in get_involved_comps_from_iterator(child, program_json):
                result.append(comp)
        return result
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
def add_static_dims(matrix, static_dims):
    size = len(matrix)*2 + 2
    gen_matrix = np.zeros((size, size), int)
    np.fill_diagonal(gen_matrix, 1)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            gen_matrix[2*i+1][2*j+1] = matrix[i][j]
    for i in range(len(static_dims)):
        gen_matrix[2*i+2][size-1] = static_dims[i]
        gen_matrix[2*i+2][2*i+2] = 0
    return gen_matrix
def get_comp_iterators_from_tree_struct(schedule_json, comp_name):
    tree = schedule_json["tree_structure"]
    level = tree
    iterators = []
    to_explore = [root for root in tree["roots"]]
    # to_explore.append(tree)
    while(to_explore):
        level = to_explore.pop(0)
        if(comp_name in get_involved_comps(level)):
            iterators.append(level['loop_name'])
               
        for element in level["child_list"]:
            to_explore.append(element)
    
    return iterators
def get_iterators_from_tree_struct(schedule_json):
    tree = schedule_json["tree_structure"]
    level = tree
    iterators = []
    to_explore = []
    to_explore.append(tree)
    while(to_explore):
        level = to_explore.pop(0)
        
        iterators.append(level['loop_name'])
               
        for element in level["child_list"]:
            to_explore.append(element)
    
    return iterators
def format_bound(id_rank, size, is_lower):
    output = []
    for i in range(size):
        if i == id_rank:
            if is_lower :
                output.append(-1)
            else:
                output.append(1)
        else:
            output.append(0)
    return output
def get_padded_initial_constrain_matrix(nb_iterators, max_depth):
    result = []
    for i in range(nb_iterators):
        for j in range(2):
            if j == 0 :
                result.append(format_bound(i, nb_iterators, True))
            else:
                result.append(format_bound(i, nb_iterators, False))
#     print(result)
#     print(len(result))
    result = np.c_[np.ones(len(result)), result]
    result = np.r_[[np.ones(len(result[0]))], result]
    result = np.pad(
        result,
        [
            (0, (max_depth + 1)*2 - result.shape[0]),
            (0, max_depth + 1 - result.shape[1]),
        ],
        mode="constant",
        constant_values=0,
    )
    return result
def get_padded_transformed_constrain_matrix(nb_iterators, max_depth, transformation_matrix):
    result = []
    for i in range(nb_iterators):
        for j in range(2):
            if j == 0 :
                result.append(format_bound(i, nb_iterators, True))
            else:
                result.append(format_bound(i, nb_iterators, False))
    inverse = np.linalg.inv(transformation_matrix)
    result = np.matmul(result, inverse)
    
    result = np.c_[np.ones(len(result)), result]
    result = np.r_[[np.ones(len(result[0]))], result]
    result = np.pad(
        result,
        [
            (0, (max_depth + 1)*2 - result.shape[0]),
            (0, max_depth + 1 - result.shape[1]),
        ],
        mode="constant",
        constant_values=0,
    )
#     print(transformation_matrix)
#     print(result)
    return result
def get_transformation_matrix(
    program_json, schedule_json, comp_name, max_depth=None
):

#     comp_name = list(program_json["computations"].keys())[0]
    comp_dict = program_json["computations"][comp_name]
    comp_schedule_dict = schedule_json[comp_name]
    nb_iterators = len(comp_dict["iterators"])
    loop_nest = comp_dict["iterators"][:]
    identity = np.zeros((nb_iterators, nb_iterators), int)
    np.fill_diagonal(identity, 1)

    if "transformation_matrices" in comp_schedule_dict:
        # torch.concat( list(X.view(-1,3,3)), dim=1)
        if comp_schedule_dict["transformation_matrices"] != []:
            if ("transformation_matrix" in comp_schedule_dict) and (
                comp_schedule_dict["transformation_matrix"]
            ):
                # print("transformation_matrix@2",comp_schedule_dict["transformation_matrix"])
                final_transformation_matrix = np.array(
                    list(map(int, comp_schedule_dict["transformation_matrix"]))
                ).reshape(nb_iterators, nb_iterators)
            else:
                final_transformation_matrix = identity.copy()
            final_mat = final_transformation_matrix

        else:
            final_mat = identity.copy()
            final_transformation_matrix = final_mat

    else:
        interchange_matrix = identity.copy()
        if comp_schedule_dict["interchange_dims"]:
            first_iter_index = loop_nest.index(
                comp_schedule_dict["interchange_dims"][0]
            )
            second_iter_index = loop_nest.index(
                comp_schedule_dict["interchange_dims"][1]
            )
            interchange_matrix[first_iter_index, first_iter_index] = 0
            interchange_matrix[second_iter_index, second_iter_index] = 0
            interchange_matrix[first_iter_index, second_iter_index] = 1
            interchange_matrix[second_iter_index, first_iter_index] = 1
            loop_nest[first_iter_index], loop_nest[second_iter_index] = (
                loop_nest[second_iter_index],
                loop_nest[first_iter_index],
            )

        skewing_matrix = identity.copy()
        if comp_schedule_dict["skewing"]:
            first_iter_index = loop_nest.index(
                comp_schedule_dict["skewing"]["skewed_dims"][0]
            )
            second_iter_index = loop_nest.index(
                comp_schedule_dict["skewing"]["skewed_dims"][1]
            )
            first_factor = int(comp_schedule_dict["skewing"]["skewing_factors"][0])
            second_factor = int(comp_schedule_dict["skewing"]["skewing_factors"][1])

            if (first_factor, second_factor) in global_dioph_sols_dict:
                a, b = global_dioph_sols_dict[(first_factor, second_factor)]
            else:
                a, b = linear_diophantine_default(first_factor, second_factor)
            skewing_matrix[first_iter_index, first_iter_index] = first_factor
            skewing_matrix[first_iter_index, second_iter_index] = second_factor
            skewing_matrix[second_iter_index, first_iter_index] = a
            skewing_matrix[second_iter_index, second_iter_index] = b

        final_mat = skewing_matrix @ interchange_matrix


    padded_mat = final_mat


    return padded_mat
def get_schedule_representation(
    program_json,
    no_sched_json,
    schedule_json,
    comps_repr_templates_list,
    loops_repr_templates_list,
    comps_placeholders_indices_dict,
    loops_placeholders_indices_dict,
    max_depth,
):

    comps_repr = copy.deepcopy(comps_repr_templates_list)
    loops_repr = copy.deepcopy(loops_repr_templates_list)

    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )

    padded_tranf_mat_per_comp = dict()

    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = program_json["computations"][comp_name]
        comp_schedule_dict = schedule_json[comp_name]
        c_code = "C" + str(comp_index)

#         fused_levels = []
#         if "fusions" in schedule_json and schedule_json["fusions"]:
#             for fusion in schedule_json["fusions"]:

#                 if comp_name in fusion:
#                     fused_levels.append(fusion[2])

        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):

            l_code = c_code + "-L" + str(iter_i)

            parallelized = 0
            if iterator_name == comp_schedule_dict["parallelized_dim"]:
                parallelized = 1
            p_index = comps_placeholders_indices_dict[l_code + "Parallelized"]
            comps_repr[p_index[0]][p_index[1]] = parallelized

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

#             fused = 0
#             if iter_i in fused_levels:
#                 fused = 1
#             p_index = comps_placeholders_indices_dict[l_code + "Fused"]
#             comps_repr[p_index[0]][p_index[1]] = fused

        unrolled = 0
        unroll_factor = 0
        if comp_schedule_dict["unrolling_factor"]:
            unrolled = 1
            unroll_factor = int(comp_schedule_dict["unrolling_factor"])
        p_index = comps_placeholders_indices_dict[c_code + "-Unrolled"]
        comps_repr[p_index[0]][p_index[1]] = unrolled
        p_index = comps_placeholders_indices_dict[c_code + "-UnrollFactor"]
        comps_repr[p_index[0]][p_index[1]] = unroll_factor

        mat_start = comps_placeholders_indices_dict[
            c_code + "-TransformationMatrixStart"
        ]
        mat_end = comps_placeholders_indices_dict[c_code + "-TransformationMatrixEnd"]
        nb_mat_elements = mat_end[1] - mat_start[1] + 1
        max_depth = int(np.sqrt(nb_mat_elements / MAX_MATRICES)) - 1
        padded_matrix = get_padded_transformation_matrix(
            program_json, schedule_json, comp_name, max_depth
        )
        assert len(padded_matrix.flatten().tolist()) == nb_mat_elements

        # print("padded_matrix.flatten().tolist()=",padded_matrix.flatten().tolist())
        # print("mat_start[0]",mat_start[0])
        # print("mat_start[1]",mat_start[1])
        # print("mat_end[1] + 1",mat_end[1] + 1)
        ogc_start = comps_placeholders_indices_dict[c_code+'-OgConstraintMatrixStart']
        ogc_end = comps_placeholders_indices_dict[c_code+'-OgConstraintMatrixEnd']
        nb_mat_elements = ogc_end[1] - ogc_start[1] + 1
        max_depth_it = int(np.sqrt(nb_mat_elements / 2)) - 1
        comps_repr[ogc_start[0]][
            ogc_start[1] : ogc_end[1] + 1
        ] = get_padded_initial_constrain_matrix(len(comp_dict["iterators"]), max_depth_it).flatten().tolist()
        c_start = comps_placeholders_indices_dict[c_code+'-ConstraintMatrixStart']
        c_end = comps_placeholders_indices_dict[c_code+'-ConstraintMatrixEnd']
        nb_mat_elements = c_end[1] - c_start[1] + 1
        max_depth_it = int(np.sqrt(nb_mat_elements / 2)) - 1
#         print(nb_mat_elements)
#         print(len(get_padded_transformed_constrain_matrix(len(comp_dict["iterators"]), max_depth_it, get_transformation_matrix(program_json, schedule_json, comp_name, max_depth)).flatten().tolist()))
        comps_repr[c_start[0]][
            c_start[1] : c_end[1] + 1
        ] = get_padded_transformed_constrain_matrix(len(comp_dict["iterators"]), max_depth_it, get_transformation_matrix(program_json, schedule_json, comp_name, max_depth)).flatten().tolist()
        s_start = comps_placeholders_indices_dict[c_code+'-OgStaticDimensionsStart']
        s_end = comps_placeholders_indices_dict[c_code+'-OgStaticDimensionsEnd']
        nb_mat_elements = s_end[1] - s_start[1] + 1
        comps_repr[mat_start[0]][
            mat_start[1] : mat_end[1] + 1
        ] = padded_matrix.flatten().tolist()
        # adding the original transformed static dimensions
        s_start = comps_placeholders_indices_dict[c_code+'-OgStaticDimensionsStart']
        s_end = comps_placeholders_indices_dict[c_code+'-OgStaticDimensionsEnd']
        nb_mat_elements = s_end[1] - s_start[1] + 1
        all_static_dims = get_original_static_dims(program_json)
        comp_index = program_json['computations'][comp_name]['absolute_order']-1
        static_dims = all_static_dims[comp_index]
        for i in range(nb_mat_elements-len(static_dims)):
            static_dims.append(0)

        comps_repr[s_start[0]][s_start[1]:s_end[1]+1] = static_dims
        # adding the transformed static dimensions
        s_start = comps_placeholders_indices_dict[c_code+'-StaticDimensionsStart']
        s_end = comps_placeholders_indices_dict[c_code+'-StaticDimensionsEnd']
        nb_mat_elements = s_end[1] - s_start[1] + 1
        all_static_dims = get_static_dims(schedule_json, program_json)
        comp_index = program_json['computations'][comp_name]['absolute_order']-1
        static_dims = all_static_dims[comp_index]
        for i in range(nb_mat_elements-len(static_dims)):
            static_dims.append(0)

        comps_repr[s_start[0]][s_start[1]:s_end[1]+1] = static_dims
        # print("padded_matrix[0,:].reshape(max_depth + 1,max_depth + 1)=",padded_matrix[0,:].reshape(max_depth + 1,max_depth + 1))
        padded_tranf_mat_per_comp[comp_name] = padded_matrix[0, :].reshape(
            max_depth + 1, max_depth + 1
        )

    loop_schedules_dict = dict()
    for loop_name in program_json["iterators"]:
        loop_schedules_dict[loop_name] = dict()
        loop_schedules_dict[loop_name]["TransformationMatrixCol"] = []
        loop_schedules_dict[loop_name]["TransformationMatrixRow"] = []
        loop_schedules_dict[loop_name]["tiled"] = 0
        loop_schedules_dict[loop_name]["tile_factor"] = 0
        loop_schedules_dict[loop_name]["unrolled"] = 0
        loop_schedules_dict[loop_name]["unroll_factor"] = 0
        loop_schedules_dict[loop_name]["parallelized"] = 0
#         loop_schedules_dict[loop_name]["fused"] = 0

    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_schedule_dict = schedule_json[comp_name]
        comp_iterators_from_tree_struct = get_comp_iterators_from_tree_struct(no_sched_json, comp_name)
        if comp_schedule_dict["tiling"]:
            for tiled_loop_index, tiled_loop in enumerate(
                comp_schedule_dict["tiling"]["tiling_dims"]
            ):
                loop_schedules_dict[tiled_loop]["tiled"] = 1
                assert loop_schedules_dict[tiled_loop][
                    "tile_factor"
                ] == 0 or loop_schedules_dict[tiled_loop]["tile_factor"] == int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tiled_loop_index]
                )
                loop_schedules_dict[tiled_loop]["tile_factor"] = int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tiled_loop_index]
                )
        if comp_schedule_dict["unrolling_factor"]:
            comp_innermost_loop = computations_dict[comp_name]["iterators"][-1]
            loop_schedules_dict[comp_innermost_loop]["unrolled"] = 1
            assert loop_schedules_dict[comp_innermost_loop][
                "unroll_factor"
            ] == 0 or loop_schedules_dict[comp_innermost_loop]["unroll_factor"] == int(
                comp_schedule_dict["unrolling_factor"]
            )
            loop_schedules_dict[comp_innermost_loop]["unroll_factor"] = int(
                comp_schedule_dict["unrolling_factor"]
            )
        if comp_schedule_dict["parallelized_dim"]:
            loop_schedules_dict[comp_schedule_dict["parallelized_dim"]][
                "parallelized"
            ] = 1

        assert padded_tranf_mat_per_comp[comp_name].shape == (
            (max_depth + 1),
            (max_depth + 1),
        )
        # print(comp_iterators_from_tree_struct)
        for iter_i, loop_name in enumerate(comp_iterators_from_tree_struct):
            
            if len(loop_schedules_dict[loop_name]["TransformationMatrixCol"]) > 0:
                assert (
                    loop_schedules_dict[loop_name]["TransformationMatrixCol"]
                    == padded_tranf_mat_per_comp[comp_name][:, iter_i + 1]
                ).all()
            else:
                loop_schedules_dict[loop_name][
                    "TransformationMatrixCol"
                ] = padded_tranf_mat_per_comp[comp_name][:, iter_i + 1]
            if len(loop_schedules_dict[loop_name]["TransformationMatrixRow"]) > 0:
                assert (
                    loop_schedules_dict[loop_name]["TransformationMatrixRow"]
                    == padded_tranf_mat_per_comp[comp_name][iter_i + 1, :]
                ).all()
            else:
                loop_schedules_dict[loop_name][
                    "TransformationMatrixRow"
                ] = padded_tranf_mat_per_comp[comp_name][iter_i + 1, :]

#     if "fusions" in schedule_json and schedule_json["fusions"]:
#         for fusion in schedule_json["fusions"]:
#             fused_loop1 = computations_dict[fusion[0]]["iterators"][fusion[2]]
#             fused_loop2 = computations_dict[fusion[1]]["iterators"][fusion[2]]
#             loop_schedules_dict[fused_loop1]["fused"] = 1
#             loop_schedules_dict[fused_loop2]["fused"] = 1
        
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

#         p_index = loops_placeholders_indices_dict[l_code + "Fused"]
#         loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["fused"]

        row_start = loops_placeholders_indices_dict[l_code + "TransfMatRowStart"]
        row_end = loops_placeholders_indices_dict[l_code + "TransfMatRowEnd"]
        nb_row_elements = row_end[1] - row_start[1] + 1
        
        assert (
            len(loop_schedules_dict[loop_name]["TransformationMatrixRow"])
            == nb_row_elements
        )
        loops_repr[row_start[0]][row_start[1] : row_end[1] + 1] = loop_schedules_dict[
            loop_name
        ]["TransformationMatrixRow"]

        col_start = loops_placeholders_indices_dict[l_code + "TransfMatColStart"]
        col_end = loops_placeholders_indices_dict[l_code + "TransfMatColEnd"]
        nb_col_elements = col_end[1] - col_start[1] + 1
        assert (
            len(loop_schedules_dict[loop_name]["TransformationMatrixCol"])
            == nb_col_elements
        )
        loops_repr[col_start[0]][col_start[1] : col_end[1] + 1] = loop_schedules_dict[
            loop_name
        ]["TransformationMatrixCol"]

    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr), 0)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr), 0)

    return computations_tensor, loops_tensor


def get_padded_transformation_matrix(
    program_json, schedule_json, comp_name, max_depth=None
):

#     comp_name = list(program_json["computations"].keys())[0]
    comp_dict = program_json["computations"][comp_name]
    comp_schedule_dict = schedule_json[comp_name]
    nb_iterators = len(comp_dict["iterators"])
    loop_nest = comp_dict["iterators"][:]
    identity = np.zeros((nb_iterators, nb_iterators), int)
    np.fill_diagonal(identity, 1)
    all_static_dims = get_static_dims(schedule_json, program_json)
    comp_index = program_json['computations'][comp_name]['absolute_order']-1
    static_dims = all_static_dims[comp_index]
    all_original_static_dims = get_original_static_dims(program_json)
    comp_index = program_json['computations'][comp_name]['absolute_order']-1
    original_static_dims  = all_original_static_dims[comp_index]
    if "transformation_matrices" in comp_schedule_dict:
        # torch.concat( list(X.view(-1,3,3)), dim=1)
        if comp_schedule_dict["transformation_matrices"] != []:
            if ("transformation_matrix" in comp_schedule_dict) and (
                comp_schedule_dict["transformation_matrix"]
            ):
                # print("transformation_matrix@2",comp_schedule_dict["transformation_matrix"])
                final_transformation_matrix = np.array(
                    list(map(int, comp_schedule_dict["transformation_matrix"]))
                ).reshape(nb_iterators, nb_iterators)
            else:
                final_transformation_matrix = identity.copy()
            
            final_mat = add_static_dims(final_transformation_matrix, static_dims)
            final_mat = np.c_[np.ones(final_mat.shape[0]), final_mat]
            final_mat = np.r_[[np.ones(final_mat.shape[1])], final_mat]
            final_mat = np.pad(
                final_mat,
                [
                    (0, max_depth + 1 - final_mat.shape[0]),
                    (0, max_depth + 1 - final_mat.shape[1]),
                ],
                mode="constant",
                constant_values=0,
            )
            final_mat_factors = [final_mat.reshape(1, -1)]
            for matrix in comp_schedule_dict["transformation_matrices"][::-1]:
                assert np.sqrt(len(matrix)) == nb_iterators
                transformation_matrix = np.array(list(map(int, matrix))).reshape(
                    nb_iterators, nb_iterators
                )
                if (transformation_matrix == identity).all():
                    transformation_matrix = add_static_dims(transformation_matrix, original_static_dims)
                else:
                    zeros = [0] * nb_iterators
                    transformation_matrix = add_static_dims(transformation_matrix, zeros)    
                # print(transformation_matrix)
                transformation_matrix = np.c_[
                    np.ones(transformation_matrix.shape[0]), transformation_matrix
                ]
                transformation_matrix = np.r_[
                    [np.ones(transformation_matrix.shape[1])], transformation_matrix
                ]
                transformation_matrix = np.pad(
                    transformation_matrix,
                    [
                        (0, max_depth + 1 - transformation_matrix.shape[0]),
                        (0, max_depth + 1 - transformation_matrix.shape[1]),
                    ],
                    mode="constant",
                    constant_values=0,
                )
                final_mat_factors.append(transformation_matrix.reshape(1, -1))
            if len(final_mat_factors) > MAX_MATRICES:
                # print("length exceeded = ", len(final_mat_factors))
                # raise NbMatricesException
                final_mat_factors = final_mat_factors[:MAX_MATRICES]
            final_mat = (
                np.concatenate(final_mat_factors, axis=0)
                if final_mat_factors
                else identity.copy()
            )
        else:
            final_mat = identity.copy()
            final_transformation_matrix = final_mat
            final_mat = add_static_dims(final_mat, original_static_dims)
            final_mat = np.c_[np.ones(final_mat.shape[0]), final_mat]
            final_mat = np.r_[[np.ones(final_mat.shape[1])], final_mat]
            final_mat = np.pad(
                final_mat,
                [
                    (0, max_depth + 1 - final_mat.shape[0]),
                    (0, max_depth + 1 - final_mat.shape[1]),
                ],
                mode="constant",
                constant_values=0,
            ).reshape(1, -1)

        comparison_matrix = identity.copy()
        for mat in comp_schedule_dict["transformation_matrices"][::-1]:
            comparison_matrix = comparison_matrix @ np.array(
                list(map(int, mat))
            ).reshape(nb_iterators, nb_iterators)
        # print(comparison_matrix,final_transformation_matrix)
        assert (comparison_matrix == final_transformation_matrix).all()
    else:
        interchange_matrix = identity.copy()
        if comp_schedule_dict["interchange_dims"]:
            first_iter_index = loop_nest.index(
                comp_schedule_dict["interchange_dims"][0]
            )
            second_iter_index = loop_nest.index(
                comp_schedule_dict["interchange_dims"][1]
            )
            interchange_matrix[first_iter_index, first_iter_index] = 0
            interchange_matrix[second_iter_index, second_iter_index] = 0
            interchange_matrix[first_iter_index, second_iter_index] = 1
            interchange_matrix[second_iter_index, first_iter_index] = 1
            loop_nest[first_iter_index], loop_nest[second_iter_index] = (
                loop_nest[second_iter_index],
                loop_nest[first_iter_index],
            )

        skewing_matrix = identity.copy()
        if comp_schedule_dict["skewing"]:
            first_iter_index = loop_nest.index(
                comp_schedule_dict["skewing"]["skewed_dims"][0]
            )
            second_iter_index = loop_nest.index(
                comp_schedule_dict["skewing"]["skewed_dims"][1]
            )
            first_factor = int(comp_schedule_dict["skewing"]["skewing_factors"][0])
            second_factor = int(comp_schedule_dict["skewing"]["skewing_factors"][1])

            if (first_factor, second_factor) in global_dioph_sols_dict:
                a, b = global_dioph_sols_dict[(first_factor, second_factor)]
            else:
                a, b = linear_diophantine_default(first_factor, second_factor)
            skewing_matrix[first_iter_index, first_iter_index] = first_factor
            skewing_matrix[first_iter_index, second_iter_index] = second_factor
            skewing_matrix[second_iter_index, first_iter_index] = a
            skewing_matrix[second_iter_index, second_iter_index] = b

        final_mat = skewing_matrix @ interchange_matrix
        zeros = [0] * nb_iterators
        final_mat_factors = []
        for matrix in [final_mat, skewing_matrix, interchange_matrix]:
            matrix = add_static_dims(matrix, zeros)
            matrix = np.c_[np.ones(matrix.shape[0]), matrix]
            matrix = np.r_[[np.ones(matrix.shape[1])], matrix]
            matrix = np.pad(
                matrix,
                [
                    (0, max_depth + 1 - matrix.shape[0]),
                    (0, max_depth + 1 - matrix.shape[1]),
                ],
                mode="constant",
                constant_values=0,
            )
            final_mat_factors.append(matrix.reshape(1, -1))
        final_mat = (
            np.concatenate(final_mat_factors, axis=0)
            if final_mat_factors
            else identity.copy()
        )

    padded_mat = final_mat

    if max_depth != None:
        #    padded_mat = np.c_[np.ones(padded_mat.shape[0]), padded_mat]
        #    padded_mat = np.r_[[np.ones(padded_mat.shape[1])], padded_mat]
        padding_ranges = [
            (0, MAX_MATRICES - final_mat.shape[0]),
            (0, (max_depth + 1) ** 2 - final_mat.shape[1]),
        ]
        try:
            padded_mat = np.pad(
                final_mat,
                padding_ranges,
                mode="constant",
                constant_values=0,
            )
        except ValueError:
            print("ValueError")
            print(final_mat.shape)
            print(padding_ranges)
    # print("padded_mat",padded_mat.reshape(6,6))
    return padded_mat