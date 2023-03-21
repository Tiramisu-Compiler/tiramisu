import copy
import re

import numpy as np
import torch



device = "cpu"
train_device = torch.device("cpu")


class NbMatricesException(Exception):
    pass


class NbAccessException(Exception):
    pass


class LoopsDepthException(Exception):
    pass


train_dataset_file = "/data/mm12191/datasets/dataset_batch760000-780130_train.pkl"
val_dataset_file = "/data/mm12191/datasets/dataset_batch760000-780130_val.pkl"
global_dioph_sols_dict = dict()
MAX_MATRICES = 5


def get_sched_rep(program_json, sched_json, max_depth):
    max_accesses = 15
    min_accesses = 1
    #     max_depth = 5

    comps_repr_templates_list = []
    comps_indices_dict = dict()
    comps_placeholders_indices_dict = dict()

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
        # Is this computation a reduction
        comp_repr_template.append(+comp_dict["comp_is_reduction"])
        #         iterators representation + tiling and interchage
        iterators_repr = []
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            iterator_dict = program_json["iterators"][iterator_name]
            iterators_repr.extend(
                [iterator_dict["lower_bound"], iterator_dict["upper_bound"]]
            )

            # transformations placeholders
            c_code = "C" + str(comp_index)
            l_code = c_code + "-L" + str(iter_i)
            iterators_repr.extend(
                [
                    l_code + "Parallelized",
                    l_code + "Tiled",
                    l_code + "TileFactor",
                    l_code + "Fused",
                ]
            )  # unrolling is skipped since it is only applied on innermost loop
        # Adding padding
        iterator_repr_size = int(len(iterators_repr) / len(comp_dict["iterators"]))
        # adding iterators padding
        iterators_repr.extend(
            [0] * iterator_repr_size * (max_depth - len(comp_dict["iterators"]))
        )
        # Adding unrolling placeholder since unrolling can only be applied to the innermost loop
        iterators_repr.extend([c_code + "-Unrolled", c_code + "-UnrollFactor"])

        # Adding transformation matrix place holder
        iterators_repr.append(c_code + "-TransformationMatrixStart")
        iterators_repr.extend(["M"] * (((max_depth + 1) ** 2) * MAX_MATRICES - 2))
        iterators_repr.append(c_code + "-TransformationMatrixEnd")

        # Adding the iterators representation to computation vector
        comp_repr_template.extend(iterators_repr)
        #  Write access representation to computation vector
        padded_write_matrix = pad_access_matrix(
            isl_to_write_matrix(comp_dict["write_access_relation"]), max_depth
        )
        # buffer_id + flattened access matrix
        write_access_repr = [
            comp_dict["write_buffer_id"] + 1
        ] + padded_write_matrix.flatten().tolist()

        # Adding write access representation to computation vector
        comp_repr_template.extend(write_access_repr)
        # Read Access representation
        read_accesses_repr = []
        for read_access_dict in comp_dict["accesses"]:
            read_access_matrix = pad_access_matrix(
                read_access_dict["access_matrix"], max_depth
            )
            read_access_repr = (
                [+read_access_dict["access_is_reduction"]]
                + [read_access_dict["buffer_id"] + 1]
                + read_access_matrix.flatten().tolist()
            )  # buffer_id + flattened access matrix
            read_accesses_repr.extend(read_access_repr)
        # access matrix size +1 for buffer id +1 for is_access_reduction
        access_repr_len = (max_depth + 1) * (max_depth + 2) + 1 + 1
        # adding accesses padding
        read_accesses_repr.extend(
            [0] * access_repr_len * (max_accesses - len(comp_dict["accesses"]))
        )

        comp_repr_template.extend(read_accesses_repr)
        # Adding Operations count to computation vector
        comp_repr_template.append(comp_dict["number_of_additions"])
        comp_repr_template.append(comp_dict["number_of_subtraction"])
        comp_repr_template.append(comp_dict["number_of_multiplication"])
        comp_repr_template.append(comp_dict["number_of_division"])

        # adding log(x+1) of the representation
        #         log_rep = list(np.log1p(comp_representation))
        #         comp_representation.extend(log_rep)

        comps_repr_templates_list.append(comp_repr_template)
        comps_indices_dict[comp_name] = comp_index
        for j, element in enumerate(comp_repr_template):
            if isinstance(element, str):
                comps_placeholders_indices_dict[element] = (comp_index, j)

    # building loop representation template

    loops_repr_templates_list = []
    loops_indices_dict = dict()
    loops_placeholders_indices_dict = dict()
    #     assert len(program_json['iterators'])==len(set(program_json['iterators'])) #just to make sure that loop names are not duplicates, but this can't happen because it's a dict
    # !! is the order in this list fix? can't we get new indices during schedule repr !!! should we use loop name in plchldrs instead of index ? !! #Edit: now it's using the name, so this issue shouldn't occure
    for loop_index, loop_name in enumerate(program_json["iterators"]):
        loop_repr_template = []
        l_code = "L" + loop_name
        # upper and lower bound
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
            + ["M"] * (max_depth - 2 + 1)
            + [l_code + "TransfMatRowEnd"]
        )  # +1 for the frame
        loop_repr_template.extend(
            [l_code + "TransfMatColStart"]
            + ["M"] * (max_depth - 2 + 1)
            + [l_code + "TransfMatColEnd"]
        )
        # adding log(x+1) of the loop representation
        loops_repr_templates_list.append(loop_repr_template)
        loops_indices_dict[loop_name] = loop_index

        for j, element in enumerate(loop_repr_template):
            if isinstance(element, str):
                loops_placeholders_indices_dict[element] = (loop_index, j)

    def update_tree_atributes(node):
        node["loop_index"] = loops_indices_dict[node["loop_name"]]
        if node["computations_list"] != []:
            node["computations_indices"] = [
                    comps_indices_dict[comp_name]
                    for comp_name in node["computations_list"]
                ]
            node["has_comps"] = True
        else:
            node["has_comps"] = False
        for child_node in node["child_list"]:
            update_tree_atributes(child_node)
        return node

    # getting the original tree structure
    no_sched_json = sched_json
    assert "fusions" not in no_sched_json or no_sched_json["fusions"] == None
    orig_tree_structure = no_sched_json["tree_structure"]
    # to avoid altering the original tree from the json
    tree_annotation = copy.deepcopy(orig_tree_structure)
    prog_tree = update_tree_atributes(tree_annotation)

    #     loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr_templates_list),0)#.to(device)
    #     computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr_templates_list),0)#.to(device)
    return (
        prog_tree,
        comps_repr_templates_list,
        loops_repr_templates_list,
        comps_placeholders_indices_dict,
        loops_placeholders_indices_dict,
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


def isl_to_write_matrix(isl_map):  # for now this function only support reductions
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
):  # return the buffer iterator that defines the write buffer
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    return buf_iter_names


def get_schedule_matrix_interchange(first_loop, second_loop, dim):
    matrix = np.identity(dim).tolist()
    matrix[first_loop][second_loop] = 1
    matrix[second_loop][first_loop] = 1
    matrix[second_loop][second_loop] = 0
    matrix[first_loop][first_loop] = 0
    return matrix


def get_schedule_matrix_skewing(first_loop, second_loop, first_skew, second_skew, dim):
    matrix = np.identity(dim).tolist()
    matrix[first_loop][second_loop] = second_skew
    matrix[first_loop][first_loop] = first_skew
    return matrix


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
                # print("-------tiled-------", comp_schedule_dict["tiling"])
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
        padded_matrix = get_padded_transformation_matrix(
            program_json, schedule_json, comp_name, max_depth
        )
        # print("padded_matrix.shape=",padded_matrix.shape)
        comps_repr[mat_start[0]][
            mat_start[1] : mat_end[1] + 1
        ] = padded_matrix.flatten().tolist()
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
            iterator = fusion[-1]
            for loop in fusion[:-1]:
                fused_loop = computations_dict[loop]["iterators"][iterator]
            # fused_loop2 = computations_dict[fusion[1]]["iterators"][fusion[2]]
                loop_schedules_dict[fused_loop]["fused"] = 1
            # loop_schedules_dict[fused_loop2]["fused"] = 1
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
        # print(len(loop_schedules_dict[loop_name]["TransformationMatrixRow"]),nb_row_elements)
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
    comp_name = list(program_json["computations"].keys())[0]
    comp_dict = program_json["computations"][comp_name]
    comp_schedule_dict = schedule_json[comp_name]
    nb_iterators = len(comp_dict["iterators"])
    loop_nest = comp_dict["iterators"][:]
    identity = np.zeros((nb_iterators, nb_iterators), int)
    np.fill_diagonal(identity, 1)
    if "transformation_matrices" in comp_schedule_dict:
        if comp_schedule_dict["transformation_matrices"] != []:
            if ("transformation_matrix" in comp_schedule_dict) and (
                comp_schedule_dict["transformation_matrix"] is not None 
            ):
                final_transformation_matrix = comp_schedule_dict["transformation_matrix"].reshape(nb_iterators, nb_iterators)
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
                assert len(matrix.shape) == 2
                assert matrix.shape[0] == matrix.shape[1]
                assert matrix.shape[0] == nb_iterators
                transformation_matrix = matrix.reshape(nb_iterators, nb_iterators)
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
                raise NbMatricesException
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
            comparison_matrix = comparison_matrix @ mat.reshape(nb_iterators, nb_iterators)
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
    return padded_mat

def nest_iterators(root_iterator, iterators):
    if root_iterator['child_iterators'] == []:
        return {'loop_name': root_iterator["loop_name"],
                 'computations_list': root_iterator['computations_list'],
                 'child_list': []}
    subtrees = []
    for loop_name in root_iterator['child_iterators']:
        child_iterator = iterators[loop_name]
        child_iterator["loop_name"] = loop_name
        sub_tree = nest_iterators(child_iterator, iterators)
        subtrees.append(sub_tree)
    return {'loop_name': root_iterator["loop_name"],
             'computations_list': root_iterator['computations_list'],
             'child_list': subtrees}

def get_tree_structure(prog_dict):
    iterators = prog_dict["iterators"]

    mentionned = []
    for loop, content in iterators.items():
        mentionned.extend(content['child_iterators'])

    possible_root  = [loop for loop in iterators if loop not in mentionned]
    assert len(possible_root) == 1
    root_loop_name = possible_root[0]

    root_iterator = prog_dict["iterators"][root_loop_name]
    root_iterator["loop_name"] = root_loop_name
    return nest_iterators(root_iterator, iterators)

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