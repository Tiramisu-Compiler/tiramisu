import copy
import json
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils.config import *
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


class NbMatricesException(Exception):
    pass


class NbAccessException(Exception):
    pass


class LoopsDepthException(Exception):
    pass


global_dioph_sols_dict = dict()
MAX_MATRICES = 5


def get_tree_footprint(tree):
    footprint = "<L" + str(int(tree["loop_index"])) + ">"
    if tree["has_comps"]:
        footprint += "["
        for idx in tree["computations_indices"]:
            footprint += "C" + str(int(idx))
        footprint += "]"
    for child in tree["child_list"]:
        footprint += get_tree_footprint(child)
    footprint += "</L" + str(int(tree["loop_index"])) + ">"
    return footprint


def load_data(
    train_val_dataset_file,
    split_ratio=None,
    max_batch_size=2048,
    drop_sched_func=None,
    drop_prog_func=None,
    default_eval=None,
    speedups_clip_func=None,
    store_device="cpu",
    train_device="cpu",
):
    print("loading batches from: " + train_val_dataset_file)
    dataset = Dataset(
        train_val_dataset_file,
        max_batch_size,
        drop_sched_func,
        drop_prog_func,
        default_eval,
        speedups_clip_func,
        store_device=store_device,
        train_device=train_device,
    )
    if split_ratio == None:
        split_ratio = 0.2
    if split_ratio > 1:
        validation_size = split_ratio
    else:
        validation_size = int(split_ratio * len(dataset))
    indices = list(range(len(dataset)))

    val_batches_indices, train_batches_indices = (
        indices[:validation_size],
        indices[validation_size:],
    )
    val_batches_list = []
    train_batches_list = []
    for i in val_batches_indices:
        val_batches_list.append(dataset[i])
    for i in train_batches_indices:
        train_batches_list.append(dataset[i])
    print("Data loaded")
    print(
        "Sizes: " + str((len(val_batches_list), len(train_batches_list))) + " batches"
    )
    return (
        dataset,
        val_batches_list,
        val_batches_indices,
        train_batches_list,
        train_batches_indices,
    )


def get_representation_template(program_dict, max_depth, train_device="cpu"):
    max_accesses = 15
    min_accesses = 1

    comps_repr_templates_list = []
    comps_indices_dict = dict()
    comps_placeholders_indices_dict = dict()

    program_json = program_dict["program_annotation"]
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )

    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        if len(comp_dict["accesses"]) > max_accesses:
            raise NbAccessException
        if len(comp_dict["accesses"]) < min_accesses:
            raise NbAccessException
        if len(comp_dict["iterators"]) > max_depth:
            raise LoopsDepthException

        comp_repr_template = []

        comp_repr_template.append(+comp_dict["comp_is_reduction"])

        iterators_repr = []
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            iterator_dict = program_json["iterators"][iterator_name]
            iterators_repr.extend(
                [iterator_dict["lower_bound"], iterator_dict["upper_bound"]]
            )

            c_code = "C" + str(comp_index)
            l_code = c_code + "-L" + str(iter_i)
            iterators_repr.extend(
                [
                    l_code + "Parallelized",
                    l_code + "Tiled",
                    l_code + "TileFactor",
                    l_code + "Fused",
                ]
            )

        iterator_repr_size = int(len(iterators_repr) / len(comp_dict["iterators"]))
        iterators_repr.extend(
            [0] * iterator_repr_size * (max_depth - len(comp_dict["iterators"]))
        )

        iterators_repr.extend([c_code + "-Unrolled", c_code + "-UnrollFactor"])

        iterators_repr.append(c_code + "-TransformationMatrixStart")
        # MAX_MATRICES is the max number of matrices
        iterators_repr.extend(["M"] * (((max_depth + 1) ** 2) * MAX_MATRICES - 2))
        iterators_repr.append(c_code + "-TransformationMatrixEnd")

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

        comp_repr_template.append(comp_dict["number_of_additions"])
        comp_repr_template.append(comp_dict["number_of_subtraction"])
        comp_repr_template.append(comp_dict["number_of_multiplication"])
        comp_repr_template.append(comp_dict["number_of_division"])

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

        loop_repr_template.extend(
            [
                program_json["iterators"][loop_name]["lower_bound"],
                program_json["iterators"][loop_name]["upper_bound"],
            ]
        )
        loop_repr_template.extend(
            [
                l_code + "Parallelized",
                l_code + "Tiled",
                l_code + "TileFactor",
                l_code + "Fused",
                l_code + "Unrolled",
                l_code + "UnrollFactor",
            ]
        )
        loop_repr_template.extend(
            [l_code + "TransfMatRowStart"]
            + ["M"] * ((max_depth + 1) - 2)
            + [l_code + "TransfMatRowEnd"]
        )
        loop_repr_template.extend(
            [l_code + "TransfMatColStart"]
            + ["M"] * (max_depth - 2 + 1)
            + [l_code + "TransfMatColEnd"]
        )

        loops_repr_templates_list.append(loop_repr_template)
        loops_indices_dict[loop_name] = loop_index

        for j, element in enumerate(loop_repr_template):
            if isinstance(element, str):
                loops_placeholders_indices_dict[element] = (loop_index, j)

    def update_tree_atributes(node, train_device="cpu"):
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

    no_sched_json = program_dict["schedules_list"][0]
    assert "fusions" not in no_sched_json or no_sched_json["fusions"] == None
    orig_tree_structure = no_sched_json["tree_structure"]
    tree_annotation = copy.deepcopy(orig_tree_structure)
    prog_tree = update_tree_atributes(tree_annotation, train_device=train_device)

    return (
        prog_tree,
        comps_repr_templates_list,
        loops_repr_templates_list,
        comps_placeholders_indices_dict,
        loops_placeholders_indices_dict,
    )


def get_schedule_representation(
    program_json,
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

        fused_levels = []
        if "fusions" in schedule_json and schedule_json["fusions"]:
            for fusion in schedule_json["fusions"]:

                if comp_name in fusion:
                    fused_levels.append(fusion[2])

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

            fused = 0
            if iter_i in fused_levels:
                fused = 1
            p_index = comps_placeholders_indices_dict[l_code + "Fused"]
            comps_repr[p_index[0]][p_index[1]] = fused

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
        comps_repr[mat_start[0]][
            mat_start[1] : mat_end[1] + 1
        ] = padded_matrix.flatten().tolist()

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
        loop_schedules_dict[loop_name]["fused"] = 0

    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_schedule_dict = schedule_json[comp_name]
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
        for iter_i, loop_name in enumerate(computations_dict[comp_name]["iterators"]):
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

    if "fusions" in schedule_json and schedule_json["fusions"]:
        for fusion in schedule_json["fusions"]:
            fused_loop1 = computations_dict[fusion[0]]["iterators"][fusion[2]]
            fused_loop2 = computations_dict[fusion[1]]["iterators"][fusion[2]]
            loop_schedules_dict[fused_loop1]["fused"] = 1
            loop_schedules_dict[fused_loop2]["fused"] = 1

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

        p_index = loops_placeholders_indices_dict[l_code + "Fused"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["fused"]

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
    # print(program_json["computations"],schedule_json[comp_name])

    comp_name = list(program_json["computations"].keys())[0]
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
                    continue
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
                raise NbMatricesException
                # final_mat_factors = final_mat_factors[:MAX_MATRICES]
            final_mat = (
                np.concatenate(final_mat_factors, axis=0)
                if final_mat_factors
                else identity.copy()
            )
        else:
            final_mat = identity.copy()
            final_transformation_matrix = final_mat
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

        final_mat_factors = []
        for matrix in [final_mat, skewing_matrix, interchange_matrix]:
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


def get_datapoint_attributes(func_name, program_dict, schedule_index, tree_footprint):
    schedule_json = program_dict["schedules_list"][schedule_index]
    sched_id = str(schedule_index).zfill(4)
    sched_str = sched_json_to_sched_str(schedule_json)
    exec_time = np.min(schedule_json["execution_times"])
    memory_use = program_dict["program_annotation"]["memory_size"]
    node_name = program_dict["node_name"] if "node_name" in program_dict else "unknown"
    speedup = program_dict["initial_execution_time"] / exec_time

    return (
        func_name,
        sched_id,
        sched_str,
        exec_time,
        memory_use,
        node_name,
        tree_footprint,
        speedup,
    )


def sched_json_to_sched_str(sched_json):

    if "sched_str" in sched_json:
        return sched_json["sched_str"]

    orig_loop_nest = []
    orig_loop_nest.append(sched_json["tree_structure"]["loop_name"])
    child_list = sched_json["tree_structure"]["child_list"]
    while len(child_list) > 0:
        child_loop = child_list[0]
        orig_loop_nest.append(child_loop["loop_name"])
        child_list = child_loop["child_list"]

    comp_name = [
        n
        for n in sched_json.keys()
        if not n in ["unfuse_iterators", "tree_structure", "execution_times"]
    ][0]
    schedule = sched_json[comp_name]
    transf_loop_nest = orig_loop_nest
    sched_str = ""

    if "Transformation Matrix" in schedule:
        if schedule["Transformation Matrix"]:
            sched_str += "M(" + ",".join(schedule["Transformation Matrix"]) + ")"
    elif "transformation_matrix" in schedule:
        if schedule["transformation_matrix"]:
            sched_str += "M(" + ",".join(schedule["transformation_matrix"]) + ")"
    if schedule["interchange_dims"]:
        first_dim_index = transf_loop_nest.index(schedule["interchange_dims"][0])
        second_dim_index = transf_loop_nest.index(schedule["interchange_dims"][1])
        sched_str += "I(L" + str(first_dim_index) + ",L" + str(second_dim_index) + ")"
        transf_loop_nest[first_dim_index], transf_loop_nest[second_dim_index] = (
            transf_loop_nest[second_dim_index],
            transf_loop_nest[first_dim_index],
        )
    if schedule["skewing"]:
        first_dim_index = transf_loop_nest.index(schedule["skewing"]["skewed_dims"][0])
        second_dim_index = transf_loop_nest.index(schedule["skewing"]["skewed_dims"][1])
        first_factor = schedule["skewing"]["skewing_factors"][0]
        second_factor = schedule["skewing"]["skewing_factors"][1]
        sched_str += (
            "S(L"
            + str(first_dim_index)
            + ",L"
            + str(second_dim_index)
            + ","
            + str(first_factor)
            + ","
            + str(second_factor)
            + ")"
        )
    if schedule["parallelized_dim"]:
        dim_index = transf_loop_nest.index(schedule["parallelized_dim"])
        sched_str += "P(L" + str(dim_index) + ")"
    if schedule["tiling"]:
        if schedule["tiling"]["tiling_depth"] == 2:
            first_dim = schedule["tiling"]["tiling_dims"][0]
            second_dim = schedule["tiling"]["tiling_dims"][1]
            first_dim_index = transf_loop_nest.index(first_dim)
            second_dim_index = transf_loop_nest.index(second_dim)
            first_factor = schedule["tiling"]["tiling_factors"][0]
            second_factor = schedule["tiling"]["tiling_factors"][1]
            sched_str += (
                "T2(L"
                + str(first_dim_index)
                + ",L"
                + str(second_dim_index)
                + ","
                + str(first_factor)
                + ","
                + str(second_factor)
                + ")"
            )
            i = transf_loop_nest.index(first_dim)
            transf_loop_nest[i : i + 1] = first_dim + "_outer", second_dim + "_outer"
            i = transf_loop_nest.index(second_dim)
            transf_loop_nest[i : i + 1] = first_dim + "_inner", second_dim + "_inner"
        else:
            first_dim = schedule["tiling"]["tiling_dims"][0]
            second_dim = schedule["tiling"]["tiling_dims"][1]
            third_dim = schedule["tiling"]["tiling_dims"][2]
            first_dim_index = transf_loop_nest.index(first_dim)
            second_dim_index = transf_loop_nest.index(second_dim)
            third_dim_index = transf_loop_nest.index(third_dim)
            first_factor = schedule["tiling"]["tiling_factors"][0]
            second_factor = schedule["tiling"]["tiling_factors"][1]
            third_factor = schedule["tiling"]["tiling_factors"][2]
            sched_str += (
                "T3(L"
                + str(first_dim_index)
                + ",L"
                + str(second_dim_index)
                + ",L"
                + str(third_dim_index)
                + ","
                + str(first_factor)
                + ","
                + str(second_factor)
                + ","
                + str(third_factor)
                + ")"
            )
            i = transf_loop_nest.index(first_dim)
            transf_loop_nest[i : i + 1] = (
                first_dim + "_outer",
                second_dim + "_outer",
                third_dim + "_outer",
            )
            i = transf_loop_nest.index(second_dim)
            transf_loop_nest[i : i + 1] = (
                first_dim + "_inner",
                second_dim + "_inner",
                third_dim + "_inner",
            )
            transf_loop_nest.remove(third_dim)
    if schedule["unrolling_factor"]:
        dim_index = len(transf_loop_nest) - 1
        dim_name = transf_loop_nest[-1]
        sched_str += "U(L" + str(dim_index) + "," + schedule["unrolling_factor"] + ")"
        transf_loop_nest[dim_index : dim_index + 1] = (
            dim_name + "_Uouter",
            dim_name + "_Uinner",
        )

    return sched_str


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


def get_results_df(
    dataset, batches_list, indices, model, log=False, train_device="cpu"
):
    df = pd.DataFrame()
    model.eval()
    torch.set_grad_enabled(False)
    all_outputs = []
    all_labels = []
    prog_names = []
    sched_names = []
    exec_times = []
    sched_strs = []
    memory_uses = []
    node_names = []
    tree_footprints = []

    for k, (inputs, labels) in tqdm(list(enumerate(batches_list))):
        original_device = labels.device
        inputs = (inputs[0], inputs[1].to(train_device), inputs[2].to(train_device))
        labels = labels.to(train_device)
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        all_outputs.append(outputs)
        all_labels.append(labels)

        assert len(outputs) == len(dataset.batched_datapoint_attributes[indices[k]])
        zipped_attributes = list(zip(*dataset.batched_datapoint_attributes[indices[k]]))
        prog_names.extend(zipped_attributes[0])
        sched_names.extend(zipped_attributes[1])
        sched_strs.extend(zipped_attributes[2])
        exec_times.extend(zipped_attributes[3])
        memory_uses.extend(zipped_attributes[4])
        node_names.extend(zipped_attributes[5])
        tree_footprints.extend(zipped_attributes[6])
        inputs = (
            inputs[0],
            inputs[1].to(original_device),
            inputs[2].to(original_device),
        )
        labels = labels.to(original_device)
    preds = torch.cat(all_outputs)
    targets = torch.cat(all_labels)
    preds = preds.cpu().detach().numpy().reshape((-1,))
    preds = np.around(preds, decimals=6)
    targets = np.around(targets.cpu().detach().numpy().reshape((-1,)), decimals=6)

    assert preds.shape == targets.shape
    df["name"] = prog_names
    df["tree_struct"] = tree_footprints
    df["sched_name"] = sched_names
    df["sched_str"] = sched_strs
    df["exec_time"] = exec_times
    df["memory_use"] = list(map(float, memory_uses))
    df["node_name"] = node_names
    df["prediction"] = np.array(preds)
    df["target"] = np.array(targets)

    df["APE"] = np.abs(df.target - df.prediction) / df.target * 100
    df["sched_str"] = df["sched_str"].apply(lambda x: simplify_sched_str(x))

    return df


def function_wise_ndcg_1(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG_1=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=1)
    return pd.Series(dict(nDCG_1=score))


def function_wise_ndcg_5(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG_5=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=5)
    return pd.Series(dict(nDCG_5=score))


def function_wise_ndcg_full(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=None)
    return pd.Series(dict(nDCG=score))


def function_wise_spearman(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(Spearman_r=np.nan))
    score = spearmanr(g["target"], g["prediction"])[0]
    return pd.Series(dict(Spearman_r=score))


def function_wise_ape(g):
    score = np.mean(g["APE"])
    return pd.Series(dict(MAPE=score))


def get_scores(df):
    with tqdm(total=6) as pbar:
        df_spearman = df.groupby("name").apply(function_wise_spearman).reset_index()
        pbar.update(1)
        df_mape = df.groupby("name").apply(function_wise_ape).reset_index()
        pbar.update(1)
        df_ndcg = df.groupby("name").apply(function_wise_ndcg_full).reset_index()
        pbar.update(1)
        df_ndcg1 = df.groupby("name").apply(function_wise_ndcg_1).reset_index()
        pbar.update(1)
        df_ndcg5 = df.groupby("name").apply(function_wise_ndcg_5).reset_index()
        pbar.update(1)
        df_count = df.groupby("name").agg("count").reset_index()[["name", "sched_name"]]
        df_count.columns = ["name", "count"]
        pbar.update(1)

    scores_df = (
        df_count.merge(df_ndcg, on="name")
        .merge(df_ndcg5, on="name")
        .merge(df_ndcg1, on="name")
        .merge(df_spearman, on="name")
        .merge(df_mape, on="name")
    )
    return scores_df


def simplify_sched_str(
    sched_str,
):

    if sched_str.count("M") == 1:
        return sched_str
    comps = re.findall("C\d+", sched_str)
    comps = set(comps)

    mats = set(re.findall(r"M\({[\dC\,]+},([\d\,\-]+)", sched_str))
    comps_per_mat = {mat: [] for mat in mats}
    new_mats_str = ""
    for mat in comps_per_mat:
        for mat_part in re.findall("M\({[C\d\,]+}," + mat, sched_str):
            comps_per_mat[mat].extend(re.findall("C\d+", mat_part))
        new_mats_str += "M({" + ",".join(sorted(comps_per_mat[mat])) + "}," + mat + ")"
    return re.sub("(M\({[\dC\,]+},[\d\,\-]+\))+", new_mats_str, sched_str)


class Dataset:
    def __init__(
        self,
        dataset_filename,
        max_batch_size,
        drop_sched_func=None,
        drop_prog_func=None,
        can_set_default_eval=None,
        speedups_clip_func=None,
        store_device="cpu",
        train_device="cpu",
    ):

        if dataset_filename.endswith("json"):
            with open(dataset_filename, "r") as f:
                dataset_str = f.read()
            self.programs_dict = json.loads(dataset_str)
        elif dataset_filename.endswith("pkl"):
            with open(dataset_filename, "rb") as f:
                self.programs_dict = pickle.load(f)

        self.batched_X = []
        self.batched_Y = []
        self.batches_dict = dict()
        self.max_depth = 5
        self.nb_dropped = 0
        self.nb_pruned = 0
        self.dropped_funcs = []
        self.function_names_map = dict()
        self.batched_datapoint_attributes = []
        self.nb_datapoints = 0

        if drop_sched_func == None:

            def drop_sched_func(x, y):
                return False

        if drop_prog_func == None:

            def drop_prog_func(x):
                return False

        if speedups_clip_func == None:

            def speedups_clip_func(x):
                return x

        if can_set_default_eval == None:

            def can_set_default_eval(x, y):
                return 0

        functions_list = list(self.programs_dict.keys())
        random.Random(42).shuffle(functions_list)
        for index, function_name in enumerate(tqdm(functions_list)):
            if drop_prog_func(self.programs_dict[function_name]):
                self.nb_dropped += len(
                    self.programs_dict[function_name]["schedules_list"]
                )
                self.dropped_funcs.append(function_name)
                continue

            program_json = self.programs_dict[function_name]["program_annotation"]

            try:
                (
                    prog_tree,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                ) = get_representation_template(
                    self.programs_dict[function_name],
                    max_depth=self.max_depth,
                    train_device=train_device,
                )
            except (NbAccessException, LoopsDepthException):
                self.nb_dropped += len(
                    self.programs_dict[function_name]["schedules_list"]
                )
                continue
            program_exec_time = self.programs_dict[function_name][
                "initial_execution_time"
            ]
            tree_footprint = get_tree_footprint(prog_tree)
            self.batches_dict[tree_footprint] = self.batches_dict.get(
                tree_footprint,
                {
                    "tree": prog_tree,
                    "comps_tensor_list": [],
                    "loops_tensor_list": [],
                    "datapoint_attributes_list": [],
                    "speedups_list": [],
                    "exec_time_list": [],
                    "func_id": [],
                },
            )

            for schedule_index in range(
                len(self.programs_dict[function_name]["schedules_list"])
            ):
                schedule_json = self.programs_dict[function_name]["schedules_list"][
                    schedule_index
                ]
                sched_exec_time = np.min(schedule_json["execution_times"])
                if drop_sched_func(
                    self.programs_dict[function_name], schedule_index
                ) or (not sched_exec_time):
                    self.nb_dropped += 1
                    self.nb_pruned += 1
                    continue

                sched_speedup = program_exec_time / sched_exec_time

                def_sp = can_set_default_eval(
                    self.programs_dict[function_name], schedule_index
                )
                if def_sp > 0:
                    sched_speedup = def_sp

                sched_speedup = speedups_clip_func(sched_speedup)

                try:
                    comps_tensor, loops_tensor = get_schedule_representation(
                        program_json,
                        schedule_json,
                        comps_repr_templates_list,
                        loops_repr_templates_list,
                        comps_placeholders_indices_dict,
                        loops_placeholders_indices_dict,
                        self.max_depth,
                    )
                except (NbMatricesException, AssertionError):
                    # print(function_name, schedule_json)
                    continue

                datapoint_attributes = get_datapoint_attributes(
                    function_name,
                    self.programs_dict[function_name],
                    schedule_index,
                    tree_footprint,
                )

                self.batches_dict[tree_footprint]["func_id"].append(index)
                self.batches_dict[tree_footprint]["comps_tensor_list"].append(
                    comps_tensor
                )
                self.batches_dict[tree_footprint]["loops_tensor_list"].append(
                    loops_tensor
                )
                self.batches_dict[tree_footprint]["datapoint_attributes_list"].append(
                    datapoint_attributes
                )
                self.batches_dict[tree_footprint]["speedups_list"].append(sched_speedup)
                self.nb_datapoints += 1

        storing_device = torch.device(store_device)
        print("Batching ...")
        for tree_footprint in tqdm(self.batches_dict):

            # shuffling the lists inside each footprint to avoid having batches with very low program diversity
            zipped = list(
                zip(
                    self.batches_dict[tree_footprint]["datapoint_attributes_list"],
                    self.batches_dict[tree_footprint]["comps_tensor_list"],
                    self.batches_dict[tree_footprint]["loops_tensor_list"],
                    self.batches_dict[tree_footprint]["speedups_list"],
                    self.batches_dict[tree_footprint]["func_id"],
                )
            )

            random.shuffle(zipped)
            (
                self.batches_dict[tree_footprint]["datapoint_attributes_list"],
                self.batches_dict[tree_footprint]["comps_tensor_list"],
                self.batches_dict[tree_footprint]["loops_tensor_list"],
                self.batches_dict[tree_footprint]["speedups_list"],
                self.batches_dict[tree_footprint]["func_id"],
            ) = zip(*zipped)

            for chunk in range(
                0,
                len(self.batches_dict[tree_footprint]["speedups_list"]),
                max_batch_size,
            ):
                if (
                    storing_device.type == "cuda"
                    and (
                        torch.cuda.memory_allocated(storing_device.index)
                        / torch.cuda.get_device_properties(
                            storing_device.index
                        ).total_memory
                    )
                    > 0.80
                ):  # Check GPU memory in order to avoid Out of memory error
                    print(
                        "GPU memory on "
                        + str(storing_device)
                        + " nearly full, switching to CPU memory"
                    )
                    storing_device = torch.device("cpu")
                self.batched_datapoint_attributes.append(
                    self.batches_dict[tree_footprint]["datapoint_attributes_list"][
                        chunk : chunk + max_batch_size
                    ]
                )
                self.batched_X.append(
                    (
                        self.batches_dict[tree_footprint]["tree"],
                        torch.cat(
                            self.batches_dict[tree_footprint]["comps_tensor_list"][
                                chunk : chunk + max_batch_size
                            ],
                            0,
                        ).to(storing_device),
                        torch.cat(
                            self.batches_dict[tree_footprint]["loops_tensor_list"][
                                chunk : chunk + max_batch_size
                            ],
                            0,
                        ).to(storing_device),
                    )
                )
                self.batched_Y.append(
                    torch.FloatTensor(
                        self.batches_dict[tree_footprint]["speedups_list"][
                            chunk : chunk + max_batch_size
                        ]
                    ).to(storing_device)
                )

        # shuffling batches to avoid having the same footprint in consecutive batches
        zipped = list(
            zip(
                self.batched_X,
                self.batched_Y,
                self.batched_datapoint_attributes,
            )
        )
        random.shuffle(zipped)
        (
            self.batched_X,
            self.batched_Y,
            self.batched_datapoint_attributes,
        ) = zip(*zipped)

        print(
            f"Number of datapoints {self.nb_datapoints} Number of batches {len(self.batched_Y)}"
        )
        del self.programs_dict
        del self.batches_dict

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            return (self.batched_X[index], self.batched_Y[index])

    def __len__(self):
        return len(self.batched_Y)


def has_skippable_loop_1comp(
    prog_dict,
):

    program_json = prog_dict["program_annotation"]
    if not len(program_json["computations"]) == 1:
        return False
    comp_name = list(program_json["computations"].keys())[0]
    comp_dict = program_json["computations"][comp_name]
    write_buffer_id = comp_dict["write_buffer_id"]
    iterators = comp_dict["iterators"]
    write_dims = isl_to_write_dims(comp_dict["write_access_relation"])
    read_buffer_ids = [e["buffer_id"] for e in comp_dict["accesses"]]

    if len(write_dims) == len(iterators):

        if (
            len(read_buffer_ids) == 1
            and read_buffer_ids[0] == write_buffer_id
            and comp_dict["number_of_additions"] == 0
            and comp_dict["number_of_subtraction"] == 0
            and comp_dict["number_of_multiplication"] == 0
            and comp_dict["number_of_division"] == 0
        ):
            return True
        return False

    if not write_buffer_id in read_buffer_ids:
        return True

    found = False
    for access in comp_dict["accesses"]:
        if access["buffer_id"] == write_buffer_id and not access_is_stencil(access):
            found = True
            break
    if not found:
        if write_dims[-1] != iterators[-1]:
            return True

    for access in comp_dict["accesses"]:
        if access["buffer_id"] == write_buffer_id and access_is_stencil(access):
            return False

    read_dims_bools = []
    for access in comp_dict["accesses"]:
        read_dims_bools.append(np.any(access["access_matrix"], axis=0))
    read_dims_bools = np.any(read_dims_bools, axis=0)
    read_iterators = [
        iterators[i]
        for i, is_used in enumerate(read_dims_bools[:-1])
        if is_used == True
    ]
    used_iterators = set(write_dims + read_iterators)
    if len(used_iterators) == len(iterators):
        return False

    if iterators[-1] in used_iterators:
        if len(comp_dict["accesses"]) > 2:
            return False

    return True


def sched_is_prunable_1comp(schedule_str, prog_depth):
    if re.search("P\(L2\)U\(L3,\d+\)", schedule_str):
        return True
    if prog_depth == 2:
        if re.search("P\(L1\)(?:[^T]|$)", schedule_str):
            return True
    if prog_depth == 3:
        if re.search("P\(L2\)(?:[^T]|$|T2\(L0,L1)", schedule_str):
            return True
    return False


def can_set_default_eval_1comp(schedule_str, prog_depth):
    def_sp = 0

    if prog_depth == 2:
        if re.search("P\(L1\)$", schedule_str):
            def_sp = 0.001
    if prog_depth == 3:
        if re.search("P\(L2\)$", schedule_str):
            def_sp = 0.001
    return def_sp


def access_is_stencil(access):
    return np.any(access["access_matrix"], axis=0)[-1]


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


def wrongly_pruned_schedule(program_dict, schedule_index):
    schedule_dict = program_dict["schedules_list"][schedule_index]
    if not "sched_str" in schedule_dict:
        return False
    sched_str = schedule_dict["sched_str"]
    if schedule_dict["execution_times"] == None:
        return False
    if len(schedule_dict["execution_times"]) == 1:
        depths = []
        target = (
            program_dict["initial_execution_time"] / schedule_dict["execution_times"][0]
        )
        if target > 0.00097 and target < 0.00103:
            for depth in program_dict["program_annotation"]["computations"]:
                depths.append(
                    len(
                        program_dict["program_annotation"]["computations"][depth][
                            "iterators"
                        ]
                    )
                )
            reg_str = ""
            for j in reversed(range(len(depths))):
                for i in range(depths[j] - 1):
                    reg_str += (
                        ".*P\(\{(C[0-9],)*C"
                        + str(j)
                        + "(,C[0-9])*\},L"
                        + str(i)
                        + "\)$|"
                    )
            reg_str = reg_str[:-1]
            if re.search(reg_str, sched_str):

                return True
    return False

def speedup_clip(speedup):
    if speedup < 0.01:
        speedup = 0.01
    return speedup


def drop_program(prog_dict):
    if len(prog_dict["schedules_list"]) < 2:
        return True
    if has_skippable_loop_1comp(prog_dict):
        return True
    if (
        "node_name" in prog_dict and prog_dict["node_name"] == "lanka24"
    ):  # drop if we the program is run by lanka24 (because its measurements are inacurate)
        return True
    return False


def drop_schedule(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    schedule_str = sched_json_to_sched_str(schedule_json)
    program_depth = len(prog_dict["program_annotation"]["iterators"])
    if (not schedule_json["execution_times"]) or min(
        schedule_json["execution_times"]
    ) < 0:  # exec time is set to -1 on datapoints that are deemed noisy, or if list empty
        return True
    if (
        len(prog_dict["program_annotation"]["computations"]) == 1
    ):  # this function works only on single comp programs
        if sched_is_prunable_1comp(schedule_str, program_depth):
            return True
    if wrongly_pruned_schedule(prog_dict, schedule_index):
        return True

    return False


def default_eval(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    schedule_str = sched_json_to_sched_str(schedule_json)
    program_depth = len(prog_dict["program_annotation"]["iterators"])
    if (
        len(prog_dict["program_annotation"]["computations"]) == 1
    ):  # this function works only on single comp programs
        return can_set_default_eval_1comp(schedule_str, program_depth)
    else:
        return 0

