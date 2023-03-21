import json
import re

import numpy as np


class LargeAccessMatices(Exception):
    pass


class NbAccessException(Exception):
    pass


class LoopsDepthException(Exception):
    pass


class TimeOutException(Exception):
    pass


class LoopExtentException(Exception):
    pass


class RepresentationLengthException(Exception):
    pass


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        return json.JSONEncoder.default(self, obj)


class LCException(Exception):
    pass


class SkewParamsException(Exception):
    pass


class IsTiledException(Exception):
    pass


class IsInterchangedException(Exception):
    pass


class IsSkewedException(Exception):
    pass


class IsUnrolledException(Exception):
    pass


class IsParallelizedException(Exception):
    pass


class IsReversedException(Exception):
    pass


class SkewUnrollException(Exception):
    pass


class ScheduleUtils:

    @classmethod
    def linear_diophantine_default(cls, f_i, f_j):
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

    @classmethod
    def pad_access_matrix(cls, access_matrix, max_depth):
        access_matrix = np.array(access_matrix)
        access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
        access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
        padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
        padded_access_matrix[:access_matrix.shape[0], :access_matrix.shape[1] -
                             1] = access_matrix[:, :-1]
        padded_access_matrix[:access_matrix.shape[0], -1] = access_matrix[:,
                                                                          -1]

        return padded_access_matrix

    @classmethod
    def isl_to_write_matrix(cls, isl_map):
        comp_iterators_str = re.findall(r'\[(.*)\]\s*->', isl_map)[0]
        buffer_iterators_str = re.findall(r'->\s*\w*\[(.*)\]', isl_map)[0]
        buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
        comp_iter_names = re.findall(r'(?:\s*(\w+))+', comp_iterators_str)
        buf_iter_names = re.findall(r'(?:\s*(\w+))+', buffer_iterators_str)
        matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
        for i, buf_iter in enumerate(buf_iter_names):
            for j, comp_iter in enumerate(comp_iter_names):
                if buf_iter == comp_iter:
                    matrix[i, j] = 1
                    break
        return matrix

    @classmethod
    def sched_json_to_sched_str(cls, sched_json, prog_it):
        orig_loop_nest = []
        orig_loop_nest.append(list(prog_it.keys())[0])
        child_list = prog_it[list(prog_it.keys())[0]]['child_iterators']
        while len(child_list) > 0:
            child_loop = prog_it[child_list[0]]
            orig_loop_nest.append(child_list[0])
            child_list = child_loop['child_iterators']

        comp_name = [
            n for n in sched_json.keys() if
            not n in ['unfuse_iterators', 'tree_structure', 'execution_times']
        ][0]
        schedule = sched_json[comp_name]
        transf_loop_nest = orig_loop_nest
        sched_str = ''

        if schedule['interchange_dims']:
            first_dim_index = transf_loop_nest.index(
                schedule['interchange_dims'][0])
            second_dim_index = transf_loop_nest.index(
                schedule['interchange_dims'][1])
            sched_str += 'I(L' + str(first_dim_index) + ',L' + str(
                second_dim_index) + ')'
            transf_loop_nest[first_dim_index], transf_loop_nest[
                second_dim_index] = transf_loop_nest[
                    second_dim_index], transf_loop_nest[first_dim_index]
        if schedule['skewing']['skewed_dims']:
            first_dim_index = transf_loop_nest.index(
                schedule['skewing']['skewed_dims'][0])
            second_dim_index = transf_loop_nest.index(
                schedule['skewing']['skewed_dims'][1])
            first_factor = schedule['skewing']['skewing_factors'][0]
            second_factor = schedule['skewing']['skewing_factors'][1]
            sched_str += 'S(L' + str(first_dim_index) + ',L' + str(
                second_dim_index) + ',' + str(first_factor) + ',' + str(
                    second_factor) + ')'
        if schedule['parallelized_dim']:
            dim_index = transf_loop_nest.index(schedule['parallelized_dim'])
            sched_str += 'P(L' + str(dim_index) + ')'
        if schedule['tiling']['tiling_dims']:
            if schedule['tiling']['tiling_depth'] == 2:
                first_dim = schedule['tiling']['tiling_dims'][0]
                second_dim = schedule['tiling']['tiling_dims'][1]

                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                first_factor = schedule['tiling']['tiling_factors'][0]
                second_factor = schedule['tiling']['tiling_factors'][1]
                sched_str += 'T2(L' + str(first_dim_index) + ',L' + str(
                    second_dim_index) + ',' + str(first_factor) + ',' + str(
                        second_factor) + ')'
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[
                    i:i + 1] = first_dim + '_outer', second_dim + '_outer'
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[
                    i:i + 1] = first_dim + '_inner', second_dim + '_inner'
            else:
                first_dim = schedule['tiling']['tiling_dims'][0]
                second_dim = schedule['tiling']['tiling_dims'][1]
                third_dim = schedule['tiling']['tiling_dims'][2]
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                third_dim_index = transf_loop_nest.index(third_dim)
                first_factor = schedule['tiling']['tiling_factors'][0]
                second_factor = schedule['tiling']['tiling_factors'][1]
                third_factor = schedule['tiling']['tiling_factors'][2]
                sched_str += 'T3(L' + str(first_dim_index) + ',L' + str(
                    second_dim_index) + ',L' + str(
                        third_dim_index) + ',' + str(first_factor) + ',' + str(
                            second_factor) + ',' + str(third_factor) + ')'
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[
                    i:i +
                    1] = first_dim + '_outer', second_dim + '_outer', third_dim + '_outer'
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[
                    i:i +
                    1] = first_dim + '_inner', second_dim + '_inner', third_dim + '_inner'
                transf_loop_nest.remove(third_dim)
        if schedule['unrolling_factor']:
            dim_index = len(transf_loop_nest) - 1
            dim_name = transf_loop_nest[-1]
            sched_str += 'U(L' + str(dim_index) + ',' + str(
                schedule['unrolling_factor']) + ')'
            transf_loop_nest[dim_index:dim_index +
                             1] = dim_name + '_Uouter', dim_name + '_Uinner'
        if schedule["reversed_dim"]:
            dim_index = transf_loop_nest.index(schedule["reversed_dim"])
            sched_str += 'R(L' + str(dim_index) + ')'

        return sched_str

    @classmethod
    def get_schedules_str(cls, programs_list, programs_dict):
        if programs_dict != {}:

            functions_set = {}

            for fun in programs_list:

                if 'schedules_list' in programs_dict[fun].keys():
                    schedules = programs_dict[fun]['schedules_list']

                    schedules_set = {}

                    for schedule in schedules:

                        comp = list(schedule.keys())[0]
                        schedule_str = schedule[comp]["schedule_str"]
                        schedules_set[schedule_str] = schedule[comp][
                            "execution_times"]

                    functions_set[fun] = schedules_set

            return functions_set
        else:
            return {}

    @classmethod
    def get_representation(cls, program_annot):
        max_dims = 7
        max_depth = 5
        max_accesses = 21
        program_representation = []
        indices_dict = dict()
        computations_dict = program_annot['computations']
        ordered_comp_list = sorted(
            list(computations_dict.keys()),
            key=lambda x: computations_dict[x]['absolute_order'])

        placeholders_comp = {}

        for index, comp_name in enumerate(ordered_comp_list):
            comp_dict = computations_dict[comp_name]
            comp_representation = []

            iterators_repr = []
            for iter_i, iterator_name in enumerate(comp_dict['iterators']):
                iterator_dict = program_annot['iterators'][iterator_name]
                iterators_repr.extend([
                    iterator_dict['lower_bound'], iterator_dict['upper_bound']
                ])

                l_code = 'L' + iterator_name
                iterators_repr.extend([
                    l_code + 'Interchanged', l_code + 'Skewed',
                    l_code + 'SkewFactor', l_code + 'Parallelized',
                    l_code + 'Tiled', l_code + 'TileFactor',
                    l_code + 'Reversed', l_code + 'Fused', 0, 0,
                    l_code + "_1" + 'Interchanged', l_code + "_1" + 'Skewed',
                    l_code + "_1" + 'SkewFactor',
                    l_code + "_1" + 'Parallelized', l_code + "_1" + 'Tiled',
                    l_code + "_1" + 'TileFactor', l_code + "_1" + 'Reversed',
                    l_code + "_1" + 'Fused'
                ])

            iterator_repr_size = int(
                len(iterators_repr) / (2 * len(comp_dict['iterators'])))
            iterators_repr.extend([0] * iterator_repr_size * 2 *
                                  (max_depth - len(comp_dict['iterators'])))

            iterators_repr.extend(['Unrolled', 'UnrollFactor'])

            comp_representation.extend(iterators_repr)

            padded_write_matrix = cls.pad_access_matrix(
                cls.isl_to_write_matrix(comp_dict['write_access_relation']),
                max_depth)
            write_access_repr = [comp_dict['write_buffer_id'] + 1
                                 ] + padded_write_matrix.flatten().tolist()

            comp_representation.extend(write_access_repr)

            read_accesses_repr = []
            for read_access_dict in comp_dict['accesses']:
                read_access_matrix = cls.pad_access_matrix(
                    read_access_dict['access_matrix'], max_depth)
                read_access_repr = [read_access_dict['buffer_id'] + 1
                                    ] + read_access_matrix.flatten().tolist()
                read_accesses_repr.extend(read_access_repr)

            access_repr_len = (max_depth + 1) * (max_depth + 2) + 1
            read_accesses_repr.extend(
                [0] * access_repr_len *
                (max_accesses - len(comp_dict['accesses'])))

            comp_representation.extend(read_accesses_repr)

            comp_representation.append(comp_dict['number_of_additions'])
            comp_representation.append(comp_dict['number_of_subtraction'])
            comp_representation.append(comp_dict['number_of_multiplication'])
            comp_representation.append(comp_dict['number_of_division'])

            placeholders_indices_dict = {}
            for i, element in enumerate(comp_representation):
                if isinstance(element, str):
                    placeholders_indices_dict[element] = i
                    comp_representation[i] = 0
            placeholders_comp[comp_name] = placeholders_indices_dict

            program_representation.append(comp_representation)
            indices_dict[comp_name] = index

        return program_representation, placeholders_comp, indices_dict

    @classmethod
    def get_representation_template(cls, program_annot):

        max_accesses = 15
        min_accesses = 1
        max_depth = 5

        comp_name = list(program_annot['computations'].keys())[0]
        comp_dict = program_annot['computations'][comp_name]

        if len(comp_dict['accesses']) > max_accesses:
            raise NbAccessException
        if len(comp_dict['accesses']) < min_accesses:
            raise NbAccessException
        if len(comp_dict['iterators']) > max_depth:
            raise LoopsDepthException

        comp_repr_template = []

        iterators_repr = []
        for iter_i, iterator_name in enumerate(comp_dict['iterators']):
            iterator_dict = program_annot['iterators'][iterator_name]
            iterators_repr.extend(
                [iterator_dict['lower_bound'], iterator_dict['upper_bound']])

            l_code = 'L' + iterator_name
            iterators_repr.extend([
                l_code + 'Interchanged', l_code + 'Skewed',
                l_code + 'SkewFactor', l_code + 'Parallelized',
                l_code + 'Tiled', l_code + 'TileFactor', l_code + 'Reversed',
                0, 0, l_code + "_1" + 'Interchanged', l_code + "_1" + 'Skewed',
                l_code + "_1" + 'SkewFactor', l_code + "_1" + 'Parallelized',
                l_code + "_1" + 'Tiled', l_code + 'TileFactor',
                l_code + "_1" + 'Reversed'
            ])

        iterator_repr_size = int(
            len(iterators_repr) / (2 * len(comp_dict['iterators'])))
        iterators_repr.extend([0] * iterator_repr_size * 2 *
                              (max_depth - len(comp_dict['iterators'])))

        iterators_repr.extend(['Unrolled', 'UnrollFactor'])

        comp_repr_template.extend(iterators_repr)

        padded_write_matrix = cls.pad_access_matrix(
            cls.isl_to_write_matrix(comp_dict['write_access_relation']),
            max_depth)
        write_access_repr = [comp_dict['write_buffer_id'] + 1
                             ] + padded_write_matrix.flatten().tolist()

        comp_repr_template.extend(write_access_repr)

        read_accesses_repr = []
        for read_access_dict in comp_dict['accesses']:
            read_access_matrix = cls.pad_access_matrix(
                read_access_dict['access_matrix'], max_depth)
            read_access_repr = [read_access_dict['buffer_id'] + 1
                                ] + read_access_matrix.flatten().tolist()
            read_accesses_repr.extend(read_access_repr)

        access_repr_len = (max_depth + 1) * (max_depth + 2) + 1
        read_accesses_repr.extend([0] * access_repr_len *
                                  (max_accesses - len(comp_dict['accesses'])))

        comp_repr_template.extend(read_accesses_repr)

        comp_repr_template.append(comp_dict['number_of_additions'])
        comp_repr_template.append(comp_dict['number_of_subtraction'])
        comp_repr_template.append(comp_dict['number_of_multiplication'])
        comp_repr_template.append(comp_dict['number_of_division'])

        placeholders_indices_dict = {}
        for i, element in enumerate(comp_repr_template):
            if isinstance(element, str):
                placeholders_indices_dict[element] = i
                comp_repr_template[i] = 0

        return comp_repr_template, placeholders_indices_dict

    @classmethod
    def sched_str(cls, sched_str, id, params, comp_indic):
        if id in range(28):
            sched_str += 'I(L' + str(params["first_dim_index"]) + ',L' + str(
                params['second_dim_index']) + ')'
        else:
            if id in range(28, 41):
                if params["tiling_depth"] == 2:
                    sched_str += 'T2(L' + str(
                        params["first_dim_index"]) + ',L' + str(
                            params['second_dim_index']) + ',' + str(
                                params["first_factor"]) + ',' + str(
                                    params["second_factor"]) + ')'
                else:
                    sched_str += 'T3(L' + str(
                        params["first_dim_index"]) + ',L' + str(
                            params['second_dim_index']) + ',L' + str(
                                params["third_dim_index"]) + ',' + str(
                                    params["first_factor"]) + ',' + str(
                                        params["second_factor"]) + ',' + str(
                                            params["third_factor"]) + ')'
            else:
                if id in range(41, 44):
                    for comp in params:
                        sched_str += 'U(L' + str(
                            params[comp]["dim_index"]) + ',' + str(
                                params[comp]['unrolling_factor']) + ",C" + str(
                                    comp_indic[comp]) + ')'
                else:
                    if id in range(44, 46):
                        sched_str += 'S(L' + str(
                            params["first_dim_index"]) + ',L' + str(
                                params['second_dim_index']) + ',' + str(
                                    params["first_factor"]) + ',' + str(
                                        params["second_factor"]) + ')'
                    else:
                        if id in range(46, 48):
                            sched_str += 'P(L' + str(params["dim_index"]) + ')'
                        else:
                            if id in range(48, 56):
                                sched_str += 'R(L' + str(
                                    params["dim_index"]) + ')'
                            else:
                                if id in range(56, 61):
                                    sched_str += 'F(L' + str(
                                        params["dim_index"]) + ')'

        return sched_str

    @classmethod
    def get_orig_tree_struct(cls, program_json, root_iterator):
        tree_struct = {
            'loop_name':
            root_iterator,
            'computations_list':
            program_json['iterators'][root_iterator]['computations_list'][:],
            'child_list': []
        }
        for child_iterator in program_json['iterators'][root_iterator][
                'child_iterators']:
            tree_struct['child_list'].append(
                self.get_orig_tree_struct(program_json, child_iterator))
        return tree_struct

    @classmethod
    def update_iterators(cls, id, it_list, action_params, added_iterators,
                         comp_indic_dict):
        for comp in it_list:
            if id in range(28):
                tmp = it_list[comp][action_params["first_dim_index"]]
                it_list[comp][
                    action_params["first_dim_index"]] = it_list[comp].pop(
                        action_params["second_dim_index"])
                it_list[comp][action_params["second_dim_index"]] = tmp

            if id in range(28, 41):
                depth_1 = action_params["first_dim_index"]
                depth_2 = action_params["second_dim_index"]

                keys = list(it_list[comp].keys())

                i = len(keys) - 1

                if action_params["tiling_depth"] == 2:
                    while i > depth_2:
                        if action_params["tiling_loop_1"] and action_params[
                                "tiling_loop_2"]:
                            it_list[comp][i + 2] = it_list[comp][i]
                        elif action_params["tiling_loop_1"] or action_params[
                                "tiling_loop_2"]:
                            it_list[comp][i + 1] = it_list[comp][i]
                        i -= 1

                else:
                    if action_params["tiling_depth"] == 3:
                        depth_3 = action_params["third_dim_index"]

                        while i > depth_3:
                            if action_params["tiling_loop_1"] and action_params[
                                    "tiling_loop_2"] and action_params[
                                        "tiling_loop_3"]:
                                it_list[comp][i + 3] = it_list[comp][i]
                            else:
                                booleans = [
                                    action_params["tiling_loop_1"],
                                    action_params["tiling_loop_2"],
                                    action_params["tiling_loop_3"]
                                ]
                                if booleans.count(True) == 2:
                                    it_list[comp][i + 2] = it_list[comp][i]
                                elif booleans.count(True) == 1:
                                    it_list[comp][i + 1] = it_list[comp][i]
                            i -= 1

                if action_params["tiling_depth"] == 2:
                    if action_params["tiling_loop_1"] and action_params[
                            "tiling_loop_2"]:

                        it_list[comp][depth_1][
                            'upper_bound'] = it_list[comp][depth_1][
                                'upper_bound'] / action_params["first_factor"]
                        it_list[comp][depth_1 + 2] = {}
                        it_list[comp][depth_1 + 2]['iterator'] = "{}_1".format(
                            it_list[comp][depth_1]['iterator'])
                        it_list[comp][depth_1 + 2]['lower_bound'] = it_list[
                            comp][depth_1]['lower_bound']
                        it_list[comp][
                            depth_1 +
                            2]['upper_bound'] = action_params["first_factor"]

                        added_iterators.append(it_list[comp][depth_1 +
                                                             2]['iterator'])

                        it_list[comp][depth_2][
                            'upper_bound'] = it_list[comp][depth_2][
                                'upper_bound'] / action_params["second_factor"]
                        it_list[comp][depth_2 + 2] = {}
                        it_list[comp][depth_2 + 2]['iterator'] = "{}_1".format(
                            it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2 + 2]['lower_bound'] = it_list[
                            comp][depth_2]['lower_bound']
                        it_list[comp][
                            depth_2 +
                            2]['upper_bound'] = action_params["second_factor"]

                        added_iterators.append(it_list[comp][depth_2 +
                                                             2]['iterator'])

                    else:
                        if action_params["tiling_loop_1"]:

                            it_list[comp][depth_1]['upper_bound'] = it_list[
                                comp][depth_1]['upper_bound'] / action_params[
                                    "first_factor"]
                            it_list[comp][depth_1 + 2] = {}
                            it_list[comp][
                                depth_1 + 2]['iterator'] = "{}_1".format(
                                    it_list[comp][depth_1]['iterator'])
                            it_list[comp][depth_1 +
                                          2]['lower_bound'] = it_list[comp][
                                              depth_1]['lower_bound']
                            it_list[comp][depth_1 + 2][
                                'upper_bound'] = action_params["first_factor"]

                            added_iterators.append(
                                it_list[comp][depth_1 + 2]['iterator'])

                        elif action_params["tiling_loop_2"]:

                            it_list[comp][depth_2]['upper_bound'] = it_list[
                                comp][depth_2]['upper_bound'] / action_params[
                                    "second_factor"]
                            it_list[comp][depth_2 + 1] = {}
                            it_list[comp][
                                depth_2 + 1]['iterator'] = "{}_1".format(
                                    it_list[comp][depth_2]['iterator'])
                            it_list[comp][depth_2 +
                                          1]['lower_bound'] = it_list[comp][
                                              depth_2]['lower_bound']
                            it_list[comp][depth_2 + 1][
                                'upper_bound'] = action_params["second_factor"]

                            added_iterators.append(
                                it_list[comp][depth_2 + 1]['iterator'])

                elif action_params["tiling_depth"] == 3:

                    if action_params["tiling_loop_1"] and action_params[
                            "tiling_loop_2"] and action_params["tiling_loop_3"]:

                        it_list[comp][depth_1][
                            'upper_bound'] = it_list[comp][depth_1][
                                'upper_bound'] / action_params["first_factor"]
                        it_list[comp][depth_1 + 3] = {}
                        it_list[comp][depth_1 + 3]['iterator'] = "{}_1".format(
                            it_list[comp][depth_1]['iterator'])
                        it_list[comp][depth_1 + 3]['lower_bound'] = it_list[
                            comp][depth_1]['lower_bound']
                        it_list[comp][
                            depth_1 +
                            3]['upper_bound'] = action_params["first_factor"]

                        added_iterators.append(it_list[comp][depth_1 +
                                                             3]['iterator'])

                        it_list[comp][depth_2][
                            'upper_bound'] = it_list[comp][depth_2][
                                'upper_bound'] / action_params["second_factor"]
                        it_list[comp][depth_2 + 3] = {}
                        it_list[comp][depth_2 + 3]['iterator'] = "{}_1".format(
                            it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2 + 3]['lower_bound'] = it_list[
                            comp][depth_2]['lower_bound']
                        it_list[comp][
                            depth_2 +
                            3]['upper_bound'] = action_params["second_factor"]

                        added_iterators.append(it_list[comp][depth_2 +
                                                             3]['iterator'])

                        it_list[comp][depth_3][
                            'upper_bound'] = it_list[comp][depth_3][
                                'upper_bound'] / action_params["third_factor"]
                        it_list[comp][depth_3 + 3] = {}
                        it_list[comp][depth_3 + 3]['iterator'] = "{}_1".format(
                            it_list[comp][depth_3]['iterator'])
                        it_list[comp][depth_3 + 3]['lower_bound'] = it_list[
                            comp][depth_3]['lower_bound']
                        it_list[comp][
                            depth_3 +
                            3]['upper_bound'] = action_params["third_factor"]

                        added_iterators.append(it_list[comp][depth_3 +
                                                             3]['iterator'])

                    elif action_params["tiling_loop_1"] and action_params[
                            "tiling_loop_2"]:

                        it_list[comp][depth_1][
                            'upper_bound'] = it_list[comp][depth_1][
                                'upper_bound'] / action_params["first_factor"]
                        it_list[comp][depth_1 + 3] = {}
                        it_list[comp][depth_1 + 3]['iterator'] = "{}_1".format(
                            it_list[comp][depth_1]['iterator'])
                        it_list[comp][depth_1 + 3]['lower_bound'] = it_list[
                            comp][depth_1]['lower_bound']
                        it_list[comp][
                            depth_1 +
                            3]['upper_bound'] = action_params["first_factor"]

                        added_iterators.append(it_list[comp][depth_1 +
                                                             3]['iterator'])

                        it_list[comp][depth_2][
                            'upper_bound'] = it_list[comp][depth_2][
                                'upper_bound'] / action_params["second_factor"]
                        it_list[comp][depth_2 + 3] = {}
                        it_list[comp][depth_2 + 3]['iterator'] = "{}_1".format(
                            it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2 + 3]['lower_bound'] = it_list[
                            comp][depth_2]['lower_bound']
                        it_list[comp][
                            depth_2 +
                            3]['upper_bound'] = action_params["second_factor"]

                        added_iterators.append(it_list[comp][depth_2 +
                                                             3]['iterator'])

                    elif action_params["tiling_loop_2"] and action_params[
                            "tiling_loop_3"]:

                        it_list[comp][depth_2][
                            'upper_bound'] = it_list[comp][depth_2][
                                'upper_bound'] / action_params["second_factor"]
                        it_list[comp][depth_2 + 2] = {}
                        it_list[comp][depth_2 + 2]['iterator'] = "{}_1".format(
                            it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2 + 2]['lower_bound'] = it_list[
                            comp][depth_2]['lower_bound']
                        it_list[comp][
                            depth_2 +
                            2]['upper_bound'] = action_params["second_factor"]

                        added_iterators.append(it_list[comp][depth_2 +
                                                             2]['iterator'])

                        it_list[comp][depth_3][
                            'upper_bound'] = it_list[comp][depth_3][
                                'upper_bound'] / action_params["third_factor"]
                        it_list[comp][depth_3 + 2] = {}
                        it_list[comp][depth_3 + 2]['iterator'] = "{}_1".format(
                            it_list[comp][depth_3]['iterator'])
                        it_list[comp][depth_3 + 2]['lower_bound'] = it_list[
                            comp][depth_3]['lower_bound']
                        it_list[comp][
                            depth_3 +
                            2]['upper_bound'] = action_params["third_factor"]

                        added_iterators.append(it_list[comp][depth_3 +
                                                             2]['iterator'])

                    elif action_params["tiling_loop_1"] and action_params[
                            "tiling_loop_3"]:

                        it_list[comp][depth_1][
                            'upper_bound'] = it_list[comp][depth_1][
                                'upper_bound'] / action_params["first_factor"]
                        it_list[comp][depth_1 + 3] = {}
                        it_list[comp][depth_1 + 3]['iterator'] = "{}_1".format(
                            it_list[comp][depth_1]['iterator'])
                        it_list[comp][depth_1 + 3]['lower_bound'] = it_list[
                            comp][depth_1]['lower_bound']
                        it_list[comp][
                            depth_1 +
                            3]['upper_bound'] = action_params["first_factor"]

                        added_iterators.append(it_list[comp][depth_1 +
                                                             3]['iterator'])

                        it_list[comp][depth_3][
                            'upper_bound'] = it_list[comp][depth_3][
                                'upper_bound'] / action_params["third_factor"]
                        it_list[comp][depth_3 + 2] = {}
                        it_list[comp][depth_3 + 2]['iterator'] = "{}_1".format(
                            it_list[comp][depth_3]['iterator'])
                        it_list[comp][depth_3 + 2]['lower_bound'] = it_list[
                            comp][depth_3]['lower_bound']
                        it_list[comp][
                            depth_3 +
                            2]['upper_bound'] = action_params["third_factor"]

                        added_iterators.append(it_list[comp][depth_3 +
                                                             2]['iterator'])
                    else:
                        if action_params["tiling_loop_1"]:

                            it_list[comp][depth_1]['upper_bound'] = it_list[
                                comp][depth_1]['upper_bound'] / action_params[
                                    "first_factor"]
                            it_list[comp][depth_1 + 3] = {}
                            it_list[comp][
                                depth_1 + 3]['iterator'] = "{}_1".format(
                                    it_list[comp][depth_1]['iterator'])
                            it_list[comp][depth_1 +
                                          3]['lower_bound'] = it_list[comp][
                                              depth_1]['lower_bound']
                            it_list[comp][depth_1 + 3][
                                'upper_bound'] = action_params["first_factor"]

                            added_iterators.append(
                                it_list[comp][depth_1 + 3]['iterator'])

                        elif action_params["tiling_loop_2"]:

                            it_list[comp][depth_2]['upper_bound'] = it_list[
                                comp][depth_2]['upper_bound'] / action_params[
                                    "second_factor"]
                            it_list[comp][depth_2 + 2] = {}
                            it_list[comp][
                                depth_2 + 2]['iterator'] = "{}_1".format(
                                    it_list[comp][depth_2]['iterator'])
                            it_list[comp][depth_2 +
                                          2]['lower_bound'] = it_list[comp][
                                              depth_2]['lower_bound']
                            it_list[comp][depth_2 + 2][
                                'upper_bound'] = action_params["second_factor"]

                            added_iterators.append(
                                it_list[comp][depth_2 + 2]['iterator'])

                        elif action_params["tiling_loop_3"]:

                            it_list[comp][depth_3]['upper_bound'] = it_list[
                                comp][depth_3]['upper_bound'] / action_params[
                                    "third_factor"]
                            it_list[comp][depth_3 + 1] = {}
                            it_list[comp][
                                depth_3 + 1]['iterator'] = "{}_1".format(
                                    it_list[comp][depth_3]['iterator'])
                            it_list[comp][depth_3 +
                                          1]['lower_bound'] = it_list[comp][
                                              depth_3]['lower_bound']
                            it_list[comp][depth_3 + 1][
                                'upper_bound'] = action_params["third_factor"]

                            added_iterators.append(
                                it_list[comp][depth_3 + 1]['iterator'])

            elif id in range(41, 44):
                it_list[comp][action_params["dim_index"]][
                    'upper_bound'] = it_list[comp][action_params["dim_index"]][
                        'upper_bound'] / action_params['unrolling_factor']

            elif id in range(44, 46):
                depth_1 = action_params["first_dim_index"]
                depth_2 = action_params["second_dim_index"]

                l1_lower_bound = it_list[comp][depth_1]["lower_bound"]
                l1_upper_bound = it_list[comp][depth_1]["upper_bound"]
                l2_lower_bound = it_list[comp][depth_2]["lower_bound"]
                l2_upper_bound = it_list[comp][depth_2]["upper_bound"]

                l1_extent = abs(l1_upper_bound - l1_lower_bound)
                l2_extent = abs(l2_upper_bound - l2_lower_bound)

                l2_lower_bound = 0
                l1_lower_bound = abs(
                    action_params["first_factor"]) * l1_lower_bound
                l1_upper_bound = l1_lower_bound + abs(
                    action_params["first_factor"]) * l1_extent + abs(
                        action_params["second_factor"]) * l2_extent
                l2_upper_bound = ((l1_extent * l2_extent) /
                                  (l1_upper_bound - l1_lower_bound)) + 1

                it_list[comp][depth_1]["lower_bound"] = l1_lower_bound
                it_list[comp][depth_1]["upper_bound"] = l1_upper_bound
                it_list[comp][depth_2]["lower_bound"] = l2_lower_bound
                it_list[comp][depth_2]["upper_bound"] = l2_upper_bound

            elif id in range(48, 56):
                tmp = it_list[comp][action_params["dim_index"]]['lower_bound']
                it_list[comp][
                    action_params["dim_index"]]['lower_bound'] = it_list[comp][
                        action_params["dim_index"]]['upper_bound']
                it_list[comp][action_params["dim_index"]]['upper_bound'] = tmp

        it_list = dict(sorted(it_list.items()))

        return it_list
