import random as r
from random import random, randint, uniform, randrange, choice, shuffle, randint
from function import Function
from itertools import combinations, product
from copy import deepcopy
from functools import reduce

def copy(d):
    new = d.copy()
    new["tiling"] = d["tiling"].copy()
    return new


def add_variables(l, old, variable_counter):
    id = variable_counter
    a = l.index(old[0])
    new = []
    for _ in old:
        new.append(id)
        new.append(id + 1)
        id = id + 2
    res = l[:]
    res[a:a + len(old)] = new
    return res





class ScheduledFunction:
    TILE_SIZES = [32, 64, 128]
    TILE_DEPTHS = [2, 3, ]
    UNROLL_SIZES = [4, 8, 16, ]
#  for 2 iterators, this class will generate up to 128 schedules.
    def __init__(self, seed, batch_name = '', benchmark=False, sizes=[], output_dir=''):
        self.benchmark = benchmark
        self.batch_name = batch_name
        self.output_dir = output_dir
        self.f = Function(seed, batch_name= self.batch_name, benchmark=self.benchmark, sizes=sizes, output_dir=self.output_dir)
        self.backup_original_variables = self.f.variables[:]
        self.schedules = []
        schedulable_variable = list(filter(lambda v: v.schedulable, self.f.variables))
        variable_ids = [vv.id for vv in schedulable_variable]
        self.TILE_DEPTHS = list(filter(lambda d: d <= len(variable_ids), self.TILE_DEPTHS))
        for comp_ordering in self.f.fusion_orderings:
            schedule = {'comp_ordering':comp_ordering}
            variable_counter = len(self.f.variables)  # for new variables from tilling
            combination_interchange = [list(i) for i in combinations(variable_ids, 2)]
            for i, j in combination_interchange:
                schedule_inter = deepcopy(schedule)
                ids_interchanged = variable_ids[:]
                a, b = variable_ids.index(i), variable_ids.index(j)
                ids_interchanged[b], ids_interchanged[a] = i, j
                schedule_inter["variables"] = ids_interchanged
                schedule_inter["interchange_dims"] = [i, j]
                schedule_inter["tiling"] = None
                schedule_inter["unrolling_factor"] = None
                self.schedules.append(schedule_inter.copy())
                for tile_depth in self.TILE_DEPTHS:
                    schedule_inter_dp = schedule_inter.copy()
                    schedule_inter_dp["tiling"] = {}
                    variables_id_copy = schedule_inter["variables"][:]
                    # tile_combinations = [variables_id_copy[kk:kk + tile_depth] for kk in
                    #                      range(len(variables_id_copy) - tile_depth)]
                    tile_combinations = [variables_id_copy[kk:kk + tile_depth] for kk in
                                         range(len(variables_id_copy) - tile_depth + 1)]
                    # print("TileCombinations",tile_combinations)

                    for variables_to_tile in tile_combinations:

                        schedule_inter_dp_t = schedule_inter_dp.copy()
                        schedule_inter_dp_t["variables"] = add_variables(schedule_inter_dp_t["variables"],
                                                                         variables_to_tile, variable_counter)
                        tiling_values = [list(_) for _ in product(self.TILE_SIZES, repeat=tile_depth)]
                        # print("variables_to_tile",variables_to_tile)
                        # print("tiling_values",tiling_values)
                        # exit(255)
                        tiling_values = self.filter_tiling_values(variables_to_tile, tiling_values)
                        isTiled = False
                        #print("tiling_values", tiling_values)
                        for tiling_value in tiling_values:
                            isTiled = True
                            schedule_inter_dp_t["tiling"]["tiling_depth"] = tile_depth
                            schedule_inter_dp_t["tiling"]["tiling_dims"] = variables_to_tile[:]
                            schedule_inter_dp_t["tiling"]["tiling_factors"] = tiling_value[:]
                            self.schedules.append(copy(schedule_inter_dp_t))
                            for unrolling_size in self.UNROLL_SIZES:
                                schedule_inter_dp_un = copy(schedule_inter_dp_t)
                                schedule_inter_dp_un["unrolling_factor"] = unrolling_size
                                self.schedules.append(schedule_inter_dp_un.copy())
                        if not isTiled :
                            for unrolling_size in self.UNROLL_SIZES:
                                schedule_inter_dp_un = copy(schedule_inter_dp)
                                schedule_inter_dp_un["unrolling_factor"] = unrolling_size
                                schedule_inter_dp_un["tiling"] = None
                                self.schedules.append(schedule_inter_dp_un.copy())

            # no interchange :
            schedule_inter = deepcopy(schedule)
            schedule_inter["variables"]=variable_ids
            schedule_inter["interchange_dims"] = []
            schedule_inter["tiling"] = None
            schedule_inter["unrolling_factor"] = None
            for tile_depth in self.TILE_DEPTHS:
                schedule_inter_dp = schedule_inter.copy()
                schedule_inter_dp["tiling"] = {}
                variables_id_copy = schedule_inter["variables"][:]
                # tile_combinations = [variables_id_copy[kk:kk + tile_depth] for kk in
                #                      range(len(variables_id_copy) - tile_depth)]
                tile_combinations = [variables_id_copy[kk:kk + tile_depth] for kk in
                                     range(len(variables_id_copy) - tile_depth + 1)]
                # print("TileCombinations",tile_combinations)

                for variables_to_tile in tile_combinations:

                    schedule_inter_dp_t = schedule_inter_dp.copy()
                    schedule_inter_dp_t["variables"] = add_variables(schedule_inter_dp_t["variables"],
                                                                     variables_to_tile, variable_counter)
                    tiling_values = [list(_) for _ in product(self.TILE_SIZES, repeat=tile_depth)]
                    # print("variables_to_tile",variables_to_tile)
                    # print("tiling_values",tiling_values)
                    # exit(255)
                    tiling_values = self.filter_tiling_values(variables_to_tile, tiling_values)
                    isTiled = False
                    # print("tiling_values", tiling_values)
                    for tiling_value in tiling_values:
                        isTiled = True
                        schedule_inter_dp_t["tiling"]["tiling_depth"] = tile_depth
                        schedule_inter_dp_t["tiling"]["tiling_dims"] = variables_to_tile[:]
                        schedule_inter_dp_t["tiling"]["tiling_factors"] = tiling_value[:]
                        self.schedules.append(deepcopy(schedule_inter_dp_t))
                        for unrolling_size in self.UNROLL_SIZES:
                            schedule_inter_dp_un = deepcopy(schedule_inter_dp_t)
                            schedule_inter_dp_un["unrolling_factor"] = unrolling_size
                            self.schedules.append(schedule_inter_dp_un.copy())
                    if not isTiled:

                        for unrolling_size in self.UNROLL_SIZES: 
                            schedule_inter_dp_un = deepcopy(schedule_inter_dp)
                            schedule_inter_dp_un["unrolling_factor"] = unrolling_size
                            schedule_inter_dp_un["tiling"] = None
                            self.schedules.append(schedule_inter_dp_un.copy())

    def update_function(self, seed):
        self.f.variables = self.backup_original_variables[:]
        #self.f = Function(seed, benchmark=self.benchmark)

        # This code generates errors :
        # for v in self.f.variables:
        #     if v.sup is None:
        #         self.f.variables.remove(v)

    def filter_tiling_values(self,variable_id_to_tile, tiling_values):
        # print("#################################################################")
        # print()
        new_tiling_values=[]
        variable_sup_values=[]
        for id in variable_id_to_tile:
            for var in self.f.variables:
                if var.id == id:
                    variable_sup_values.append((var.sup.value))
                    break

        for tiling_value in tiling_values :
            legal = True
            for variable_sup_value, t_value in zip(variable_sup_values, tiling_value):
                if variable_sup_value <= t_value:
                    legal = False
                    break
            if legal:
                new_tiling_values.append(tiling_value)

        # print("variable_sup_values",variable_sup_values)
        # print("tiling_values", tiling_values)
        # print("new_tiling_values", new_tiling_values)
        return new_tiling_values


    def old__init__(self, seed):
        self.f = Function(seed)
        self.schedules = []
        schedulable_variable = list(filter(lambda v: v.schedulable, self.f.variables))
        variable_ids = [vv.id for vv in schedulable_variable]
        min_size = min([vv.sup.value for vv in schedulable_variable])

        self.TILE_SIZES = list(filter(lambda d: d <= min_size, self.TILE_SIZES))
        self.UNROLL_SIZES = list(filter(lambda d: d< min_size, self.UNROLL_SIZES))
        self.TILE_DEPTHS = list(filter(lambda d: d<= len(variable_ids), self.TILE_DEPTHS))
        print("TTILE_SIZES : ",self.TILE_SIZES)
        print("TILE_DEPTHS : ", self.TILE_DEPTHS)
        print("UNROLL_SIZES : ", self.UNROLL_SIZES)
        print(variable_ids)

        variable_counter = len(self.f.variables)  # for new variables from tilling
        combination_interchange = [list(i) for i in combinations(variable_ids, 2)]
        for i, j in combination_interchange:
            schedule_inter = {}
            ids_interchanged = variable_ids[:]
            a, b = variable_ids.index(i), variable_ids.index(j)
            ids_interchanged[b], ids_interchanged[a] = i, j
            schedule_inter["variables"] = ids_interchanged
            schedule_inter["interchange_dims"] = [i, j]
            schedule_inter["tiling"] = None
            schedule_inter["unrolling_factor"] = None
            self.schedules.append(schedule_inter.copy())
            for tile_depth in self.TILE_DEPTHS:
                schedule_inter_dp = schedule_inter.copy()
                schedule_inter_dp["tiling"] = {}
                variables_id_copy = schedule_inter["variables"][:]
                # tile_combinations = [variables_id_copy[kk:kk + tile_depth] for kk in
                #                      range(len(variables_id_copy) - tile_depth)]
                tile_combinations = [variables_id_copy[kk:kk + tile_depth] for kk in
                                     range(len(variables_id_copy) - tile_depth +1)]
                #print("tile_combinations ", tile_combinations)
                for variables_to_tile in tile_combinations:
                    schedule_inter_dp_t = schedule_inter_dp.copy()
                    schedule_inter_dp_t["variables"] = add_variables(schedule_inter_dp_t["variables"],
                                                                     variables_to_tile, variable_counter)
                    tiling_values = [list(_) for _ in product(self.TILE_SIZES, repeat=tile_depth)]
                    for tiling_value in tiling_values:
                        schedule_inter_dp_t["tiling"]["tiling_depth"] = tile_depth
                        schedule_inter_dp_t["tiling"]["tiling_dims"] = variables_to_tile[:]
                        schedule_inter_dp_t["tiling"]["tiling_factors"] = tiling_value[:]
                        self.schedules.append(copy(schedule_inter_dp_t))
                        for unrolling_size in self.UNROLL_SIZES:
                            schedule_inter_dp_un = copy(schedule_inter_dp_t)
                            schedule_inter_dp_un["unrolling_factor"] = unrolling_size
                            self.schedules.append(schedule_inter_dp_un.copy())

                    # for interchanged_vars in combination_interchange:
        #     schedule = {}
        #     # schedule["interchange_dims"] = interchanged_vars
        #     for depth in self.TILE_DEPTH:
        #         for tiled_vars in [list(i) for i in combinations(self.f.variables, depth)]:
        #             for tiled_var in tiled_vars:
        #                 for tile_size in self.TILE_SIZES:
        #                     for unrolled_var in self.f.variables:
        #                         pass