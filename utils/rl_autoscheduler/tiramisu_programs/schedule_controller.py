import copy
import sys
import time
import traceback
from typing import List

import torch
from rl_interface.action import Action

from tiramisu_programs.optimization import OptimizationCommand
from tiramisu_programs.schedule import Schedule
from tiramisu_programs.schedule_utils import *
from tiramisu_programs.surrogate_model_utils.json_to_tensor import \
    get_schedule_representation
from tiramisu_programs.surrogate_model_utils.modeling import \
    Model_Recursive_LSTM_v2

global_dioph_sols_dict = dict()


class ScheduleController:

    def __init__(self,
                 schedule: Schedule = None,
                 nb_executions=5,
                 scheds=None,
                 config=None):
        self.depth = 0
        self.schedule = []
        self.schedule_object = schedule
        self.scheds = scheds
        self.nb_executions = nb_executions
        self.speedup = 1.0
        self.steps = 0
        self.new_scheds = {}
        self.search_time = time.time()
        self.config = config
        if self.config.tiramisu.env_type == "cpu":
            self.measurement_env = self.schedule_object.prog.evaluate_schedule
        else:
            self.measurement_env = self.get_exec_time_by_model
        self.lc_total_time = 0
        self.lc_data = []
        self.schedule_list_model = []
        self.model = Model_Recursive_LSTM_v2()
        self.model.load_state_dict(
            torch.load(config.tiramisu.model_checkpoint, map_location="cpu"))

    def apply_action(self, action):

        # Initialize variables
        exit = False
        done = False
        info = {}
        self.steps += 1
        first_comp = self.schedule_object.comps[0]
        saved_legality = self.get_legality(action=action)

        if not action.id in range(44, 46):  # If the action is skewing
            action_params = action.parameter()
        else:
            comp = list(self.schedule_object.it_dict.keys())[0]
            action_params = action.parameter(comp, self.schedule_object.prog)

        if action.id in range(28):  # Interchange
            if not self.schedule_object.is_interchaged:
                params = [
                    int(action_params["first_dim_index"]),
                    int(action_params["second_dim_index"])
                ]

                optim1 = OptimizationCommand("Interchange", params,
                                             self.schedule_object.comps)
                self.schedule.append(optim1)

                if self.schedule_object.is_unrolled:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, self.non_skewed_comps, first_comp) if saved_legality is None else saved_legality
                else:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, first_comp=first_comp) if saved_legality is None else saved_legality

                if lc_check == -1:
                    print("X: The action produced an error.")
                    self.pop_schedule(action=action)
                    raise LCException
                if lc_check == 0:
                    print("X: Illegal action")
                    self.pop_schedule(action=action)
                    info = {"illegal_action": True}
                    done = False
                    return self.schedule_object.repr, 1.0, done, info
                self.schedule_object.apply_interchange(action_params)
                print("O: Interchange applied")
                self.schedule_object.is_interchaged = True

            else:
                print("X: Interchange already applied execption")
                raise IsInterchangedException

        if action.id in range(28, 41):  # Tiling
            if not self.schedule_object.is_tiled:
                params = [
                    int(action_params["first_dim_index"]),
                    int(action_params["second_dim_index"])
                ]
                params.append(action_params["first_factor"])
                params.append(action_params["second_factor"])

                if action_params["tiling_depth"] == 3:
                    params.insert(2, action_params["third_dim_index"])
                    params.append(action_params["third_factor"])

                optim2 = OptimizationCommand("Tiling", params,
                                             self.schedule_object.comps)

                self.schedule.append(optim2)

                if self.schedule_object.is_unrolled:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, self.non_skewed_comps, first_comp) if saved_legality is None else saved_legality
                else:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, first_comp=first_comp) if saved_legality is None else saved_legality
                if lc_check == -1:
                    print("X: This action produces an error")
                    self.pop_schedule(action=action)
                    raise LCException
                if lc_check == 0:
                    print("X: Illegal action")
                    self.pop_schedule(action=action)
                    info = {"illegal_action": True}
                    done = False
                    return self.schedule_object.repr, 1.0, done, info

                self.schedule_object.apply_tiling(action_params)
                print("O: Tiling applied")

                self.schedule_object.is_tiled = True

                done = True
                exit = True
                self.schedule_object.schedule_str = ScheduleUtils.sched_str(
                    self.schedule_object.schedule_str, action.id,
                    action_params, self.schedule_object.comp_indic_dict)
            else:
                print("X: Tiling already applied exception")
                raise IsTiledException

        if action.id in range(41, 44):  # Unrolling
            params = {}
            if not self.schedule_object.is_unrolled:
                self.non_skewed_comps = []
                for comp in self.schedule_object.comps:
                    it_skewed = "L" + self.schedule_object.it_dict[comp][
                        action_params[comp]
                        ["dim_index"]]["iterator"] + "Skewed"
                    if self.schedule_object.repr["representation"][
                            self.schedule_object.comp_indic_dict[comp]][
                                self.schedule_object.placeholders[comp]
                                [it_skewed]] != 1:
                        self.non_skewed_comps.append(comp)
                for comp in self.non_skewed_comps:
                    params[comp] = [
                        int(action_params[comp]["dim_index"]),
                        int(action_params[comp]["unrolling_factor"])
                    ]
                if self.non_skewed_comps != []:
                    optim3 = OptimizationCommand("Unrolling", params,
                                                 self.non_skewed_comps)
                    self.schedule.append(optim3)
                    start_time = time.time()
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, self.non_skewed_comps, first_comp) if saved_legality is None else saved_legality
                    l_time = time.time() - start_time
                    self.lc_total_time += l_time

                    if lc_check == -1:
                        print("X: This action produces an error")
                        self.pop_schedule(action=action)
                        raise LCException

                    if lc_check == 0:
                        print("X: Illegal action")
                        self.pop_schedule(action=action)
                        info = {"illegal_action": True}
                        done = False
                        return self.schedule_object.repr, 1.0, done, info

                    self.schedule_object.apply_unrolling(action_params)
                    print("O: Unrolling applied")
                    for i in range(41, 44):
                        self.schedule_object.repr["action_mask"][i] = 0
                    self.schedule_object.is_unrolled = True
                else:
                    lc_check = 0
                    info[
                        'error'] = "trying to apply unrolling after skewing in one of the computations"

            else:
                print("X: Unrolling is already applied")
                raise IsUnrolledException

        if action.id in range(44, 46):  # Skewing

            if not self.schedule_object.is_skewed:

                if (action_params["first_factor"] != None
                        and action_params["second_factor"] != None):
                    non_inner_comps = []
                    for comp in self.schedule_object.comps:
                        if (action_params["first_dim_index"] !=
                                len(self.schedule_object.it_dict[comp]) - 1
                                and action_params["second_dim_index"] !=
                                len(self.schedule_object.it_dict[comp]) - 1
                            ) or (
                                (action_params["first_dim_index"]
                                 == len(self.schedule_object.it_dict[comp]) - 1
                                 or action_params["second_dim_index"]
                                 == len(self.schedule_object.it_dict[comp]) - 1
                                 and not self.schedule_object.is_unrolled)):
                            non_inner_comps.append(comp)

                    if non_inner_comps != []:

                        params = [
                            int(action_params["first_dim_index"]),
                            int(action_params["second_dim_index"])
                        ]
                        params.append(action_params["first_factor"])
                        params.append(action_params["second_factor"])

                        optim4 = OptimizationCommand("Skewing", params,
                                                     non_inner_comps)

                        self.schedule.append(optim4)

                        start_time = time.time()
                        if self.schedule_object.is_unrolled:
                            lc_check = self.schedule_object.prog.check_legality_of_schedule(
                                self.schedule, self.non_skewed_comps,
                                first_comp) if saved_legality is None else saved_legality
                        else:
                            lc_check = self.schedule_object.prog.check_legality_of_schedule(
                                self.schedule, first_comp=first_comp) if saved_legality is None else saved_legality
                        l_time = time.time() - start_time
                        self.lc_total_time += l_time

                        if lc_check == -1:
                            print("X: This action produces an error")
                            self.pop_schedule(action=action)
                            raise LCException
                        if lc_check == 0:
                            print("X: Illegal action")
                            self.pop_schedule(action=action)
                            info = {"illegal_action": True}
                            done = False
                            return self.schedule_object.repr, 1.0, done, info

                        self.schedule_object.apply_skewing(action_params)
                        print("O: Skewing is applied")
                        self.schedule_object.is_skewed = True

                    else:
                        raise SkewUnrollException

                else:
                    print("X: Skewing prams are null")
                    raise SkewParamsException

            else:
                print("X: Skewing is already applied")
                raise IsSkewedException

        if action.id in range(46, 48):  # Parallelization
            if not self.schedule_object.is_parallelized:
                params = [int(action_params["dim_index"])]

                optim5 = OptimizationCommand("Parallelization", params,
                                             self.schedule_object.comps)
                self.schedule.append(optim5)
                start_time = time.time()
                if self.schedule_object.is_unrolled:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, self.non_skewed_comps, first_comp) if saved_legality is None else saved_legality
                else:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, first_comp=first_comp) if saved_legality is None else saved_legality

                l_time = time.time() - start_time
                self.lc_total_time += l_time
                if lc_check == -1:
                    print("X: This action produces an error")
                    self.pop_schedule(action=action)
                    raise LCException

                if lc_check == 0:
                    print("X: Illegal action")
                    self.pop_schedule(action=action)
                    info = {"illegal_action": True}
                    done = False
                    return self.schedule_object.repr, 1.0, done, info

                self.schedule_object.apply_parallelization(action_params)
                print("O: Parallelisation applied")
                self.schedule_object.is_parallelized = True
            else:
                print("X: Parallelisation is already applied")
                raise IsParallelizedException

        if action.id in range(48, 56):  # Reversal

            if not self.schedule_object.is_reversed:
                params = [int(action_params["dim_index"])]
                optim6 = OptimizationCommand("Reversal", params,
                                             self.schedule_object.comps)
                self.schedule.append(optim6)
                start_time = time.time()
                if self.schedule_object.is_unrolled:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, self.non_skewed_comps, first_comp=first_comp) if saved_legality is None else saved_legality
                else:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, first_comp=first_comp) if saved_legality is None else saved_legality
                l_time = time.time() - start_time
                self.lc_total_time += l_time
                if lc_check == -1:
                    print("X: This action produces am error")
                    self.pop_schedule(action=action)
                    raise LCException
                if lc_check == 0:
                    print("X: Illegal action")
                    self.pop_schedule(action=action)
                    info = {"illegal_action": True}
                    done = False
                    return self.schedule_object.repr, 1.0, done, info

                self.schedule_object.apply_reversal(action_params)

                print("O: Loop reversal applied")
                self.schedule_object.is_reversed = True
            else:
                print("X: Loop reversal already applied")
                raise IsReversedException

        if action.id in range(56, 61):  # Fusion
            params = [
                int(action_params["dim_index"]), action_params["fuse_comps"]
            ]
            if action_params["fuse_comps"] != [] and len(
                    action_params["fuse_comps"]) != 1:

                optim7 = OptimizationCommand("Fusion", params,
                                             action_params["fuse_comps"])

                self.schedule.append(optim7)

                start_time = time.time()

                if self.schedule_object.is_unrolled:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, self.non_skewed_comps, first_comp) if saved_legality is None else saved_legality
                else:
                    lc_check = self.schedule_object.prog.check_legality_of_schedule(
                        self.schedule, first_comp=first_comp) if saved_legality is None else saved_legality

                l_time = time.time() - start_time
                self.lc_total_time += l_time

                if lc_check == -1:
                    print("X: This action produces an error")
                    self.pop_schedule(action=action)
                    raise LCException

                if lc_check == 0:
                    print("X: Illegal action")
                    self.pop_schedule(action=action)
                    info = {"illegal_action": True}
                    done = False
                    return self.schedule_object.repr, 1.0, done, info

                self.schedule_object.apply_fusion(action_params)
                print("O: Loop fusion applied")
                self.schedule_object.is_fused = True
            else:
                lc_check = 0
                print("X: Unable to fuse")

        if action.id == Action.EXIT:
            done = True
            exit = True

        if (not exit and lc_check != 0) and not (action.id in range(
                41, 44) and self.schedule_object.is_skewed):
            self.schedule_object.schedule_str = ScheduleUtils.sched_str(
                self.schedule_object.schedule_str, action.id, action_params,
                self.schedule_object.comp_indic_dict)
            if not action.id in range(41, 44):
                self.schedule_object.it_dict = ScheduleUtils.update_iterators(
                    action.id, self.schedule_object.it_dict, action_params,
                    self.schedule_object.added_iterators,
                    self.schedule_object.comp_indic_dict)

            self.depth += 1
            return self.schedule_object.repr, 1.0, done, info
        elif exit:
            return self.schedule_object.repr, 1.0, done, info
        elif lc_check == 0:
            return self.schedule_object.repr, 1.0, done, info
        else:
            return self.schedule_object.repr, 1.0, done, info

    def pop_schedule(self, action):
        self.schedule_object.repr["action_mask"][action.id] = 0
        self.schedule.pop()

    def get_final_score(self):
        exec_time = 0
        exec_time = self.get_exec_time()
        speedup = 1.0
        if exec_time != 0:
            speedup = (self.schedule_object.prog.initial_execution_time /
                       exec_time)
        return speedup

    def test_additional_actions(self, training=True):
        info = dict()
        if training:
            print(
                "This operation alters the training and, therefore, it won't be executed")
            try:
                exec_time = 0
                exec_time = self.get_exec_time()
            except:
                pass
        else:
            if self.schedule_object.is_unrolled:
                for optim in self.schedule:
                    if optim.type == "Unrolling":
                        unroll_optimisation = optim

                new_unrolling_params = {}
                new_unrolling_optim_params = {}
                for comp in self.non_skewed_comps:
                    unroll_factor = unroll_optimisation.params_list[comp][1]
                    new_unrolling_params[comp] = {
                        "dim_index": len(self.schedule_object.it_dict[comp]) - 1,
                        "unrolling_factor": unroll_factor
                    }
                    new_unrolling_optim_params[comp] = [
                        len(self.schedule_object.it_dict[comp]
                            ) - 1, unroll_factor
                    ]

                new_unrolling_optim = OptimizationCommand(
                    "Unrolling", new_unrolling_optim_params, self.non_skewed_comps)
                new_unrolling_str = ""
                unrolling_str = ""

                for comp in self.non_skewed_comps:
                    unroll_factor = unroll_optimisation.params_list[comp][1]
                    new_unrolling_str += "U(L" + str(
                        len(self.schedule_object.it_dict[comp]) -
                        1) + "," + str(unroll_factor) + ",C" + str(
                            self.schedule_object.comp_indic_dict[comp]) + ")"
                    unrolling_str += "U(L" + str(
                        unroll_optimisation.params_list[comp][0]) + "," + str(
                            unroll_factor) + ",C" + str(
                                self.schedule_object.comp_indic_dict[comp]) + ")"
                self.schedule_object.schedule_str = self.schedule_object.schedule_str.replace(
                    unrolling_str, "") + new_unrolling_str
                self.schedule.remove(unroll_optimisation)
                self.schedule.append(new_unrolling_optim)
                self.schedule_object.apply_unrolling(new_unrolling_params)

            self.search_time = time.time() - self.search_time

            try:
                exec_time = 0
                exec_time = self.get_exec_time()

                if not self.schedule_object.is_parallelized:
                    print("Testing if parallelization improves the performance...")
                    action = Action(Action.PARALLELIZATION0,
                                    self.schedule_object.it_dict,
                                    self.schedule_object.common_it)
                    action_params = action.parameter()

                    params = [int(action_params["dim_index"])]

                    optim5 = OptimizationCommand("Parallelization", params,
                                                 self.schedule_object.comps)
                    first_comp = list(self.schedule_object.it_dict.keys())[0]
                    iterator = self.schedule_object.it_dict[first_comp][
                        action_params["dim_index"]]['iterator']
                    self.schedule_object.schedule_dict[first_comp][
                        "parallelized_dim"] = iterator

                    self.schedule.append(optim5)

                    try:

                        self.schedule_object.schedule_str = ScheduleUtils.sched_str(
                            self.schedule_object.schedule_str, action.id,
                            action_params, self.schedule_object.comp_indic_dict)
                        parallelized_exec_time = self.get_exec_time()
                        parallelization_str = 'P(L' + str(
                            action_params["dim_index"]) + ')'
                    except:
                        print("X: Illegal action")
                        self.schedule.remove(optim5)
                        self.schedule_object.schedule_str = self.schedule_object.schedule_str.replace(
                            parallelization_str, "")

                    if parallelized_exec_time < exec_time and parallelized_exec_time != 0:
                        exec_time = parallelized_exec_time

                        self.schedule_object.apply_parallelization(
                            action_params)
                        print("O: Parallelization improves the performance.")

                    else:
                        self.schedule.remove(optim5)
                        self.new_scheds[self.schedule_object.prog.name].pop(
                            self.schedule_object.schedule_str)
                        self.schedule_object.schedule_str = self.schedule_object.schedule_str.replace(
                            parallelization_str, "")
                        self.schedule_object.schedule_dict[first_comp][
                            "parallelized_dim"] = None
                        print("X: Parallelization improves the performance")

            except:

                print("X: Error while measuring performance")
                print(f"failed to save schedule",
                      traceback.format_exc(),
                      flush=True)
                info = {"Internal execution error": True}
                return self.schedule_object.repr, self.speedup, True, info

        if exec_time != 0:
            print("\nThe final schedule is ",
                  self.schedule_object.schedule_str)
            self.speedup = (
                self.schedule_object.prog.initial_execution_time / exec_time)
            print("The speedup is: ", self.speedup)
            start_time = time.time()
        info["depth"] = self.depth
        return self.schedule_object.repr, self.speedup, True, info

    def get_exec_time_by_model(self, optims_list, cmd_type, nb_executions,
                               initial_exec_time):
        self.schedule_list_model.append({
            "schedule_str":
            self.schedule_object.schedule_str,
            "schedule_dict":
            self.schedule_object.schedule_dict
        })
        stat = dict()
        try:
            computations_tensor, loops_tensor = get_schedule_representation(
                self.schedule_object.annotations,
                self.schedule_object.schedule_dict,
                self.schedule_object.templates["comps_repr_templates_list"],
                self.schedule_object.templates["loops_repr_templates_list"],
                self.schedule_object.
                templates["comps_placeholders_indices_dict"],
                self.schedule_object.
                templates["loops_placeholders_indices_dict"],
                max_depth=self.schedule_object.MAX_DEPTH - 1)
            tree_tensors = (self.schedule_object.templates["prog_tree"],
                            computations_tensor, loops_tensor)
            with torch.no_grad():
                predicted_speedup = self.model(
                    tree_tensors,
                    num_matrices=self.schedule_object.MAX_DEPTH - 1).item()
                stat[
                    "initial_execution_time"] = self.schedule_object.prog.initial_execution_time
                stat["predicted_speedup"] = predicted_speedup
                print(f"The predicted speedup is {predicted_speedup}")
                stat[
                    "predicted_execution_time"] = self.schedule_object.prog.initial_execution_time / predicted_speedup
        except Exception:
            print("ERROR_MODEL", traceback.format_exc())
            print(sys.exc_info()[2])

        return stat["predicted_execution_time"]

    def get_exec_time(self):
        prog_name = self.schedule_object.prog.name
        execution_time = 0
        if self.schedule_object.schedule_str != "" and self.schedule != []:
            if prog_name in self.scheds.keys():
                if self.schedule_object.schedule_str in self.scheds[prog_name]:
                    execution_time = self.scheds[prog_name][
                        self.schedule_object.schedule_str][0]
                else:
                    if prog_name in self.new_scheds.keys(
                    ) and self.schedule_object.schedule_str in self.new_scheds[
                            prog_name].keys():
                        execution_time = self.new_scheds[prog_name][
                            self.schedule_object.schedule_str][1]
                    else:
                        curr_sched = copy.deepcopy(self.schedule)
                        self.new_scheds[prog_name] = {}
                        execution_time = self.measurement_env(
                            self.schedule, 'sched_eval', self.nb_executions,
                            self.schedule_object.prog.initial_execution_time)
                        self.new_scheds[prog_name][
                            self.schedule_object.schedule_str] = (
                                curr_sched, execution_time, 0)
            else:
                if prog_name in self.new_scheds.keys():
                    if self.schedule_object.schedule_str in self.new_scheds[
                            prog_name].keys():
                        execution_time = self.new_scheds[prog_name][
                            self.schedule_object.schedule_str][1]
                    else:
                        curr_sched = copy.deepcopy(self.schedule)
                        execution_time = self.measurement_env(
                            self.schedule, 'sched_eval', self.nb_executions,
                            self.schedule_object.prog.initial_execution_time)
                        self.new_scheds[prog_name][
                            self.schedule_object.schedule_str] = (
                                curr_sched, execution_time, 0)
                else:
                    curr_sched = copy.deepcopy(self.schedule)
                    self.new_scheds[prog_name] = {}
                    start_time = time.time()
                    execution_time = self.measurement_env(
                        self.schedule, 'sched_eval', self.nb_executions,
                        self.schedule_object.prog.initial_execution_time)
                    sched_time = time.time() - start_time
                    self.new_scheds[prog_name][
                        self.schedule_object.schedule_str] = (curr_sched,
                                                              execution_time,
                                                              0)
        else:
            execution_time = self.schedule_object.prog.initial_execution_time
        return execution_time

    def save_legality_data(self, action, lc_check):
        key = f"{self.schedule_object.prog.name}@{self.schedule_object.schedule_str}@{action}"
        self.lc_data.append(
            [
                key,
                lc_check
            ]
        )

    def get_legality(self, action):
        key = f"{self.schedule_object.prog.name}@{self.schedule_object.schedule_str}@{action}"
        values = [v for (k, v) in self.lc_data if k == key]
        return values[0] if len(values) else None

    def get_legality_data(self):
        return self.lc_data

    def load_legality_data(self, lc_data: List) -> None:
        self.lc_data = lc_data
