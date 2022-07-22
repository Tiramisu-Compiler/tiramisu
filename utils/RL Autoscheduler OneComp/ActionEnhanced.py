import ujson as json
from tqdm.notebook import tqdm
import numpy as np
import random 
import pandas as pd
import time
import copy
import math
import re

import time

from utilsEnhanced import LoopExtentException



class Action():
    """"
    Action class to store and standardize the action for the environment.
    """
    def __init__(self, id_, it_dict, common_it):
        """"
        Initialization of an action.
        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        #iterators list dict with depth as keys
        self.it_dict=it_dict
        self.common_it= common_it



    def parameter(self,comp=None, prog=None):
        print("in action params function")
        """"
        Property method to return the parameter related to the action selected.
        Returns:
            The parameter related to this action_id
        """
        
        if self.id==0:#INTERCHANGE01
            return {
                "first_dim_index":0,
                "second_dim_index":1
            }

        elif self.id==1:#INTERCHANGE02
            return {
                "first_dim_index":0,
                "second_dim_index":2
            }
        
        elif self.id==2:#INTERCHANGE03
            return {
                "first_dim_index":0,
                "second_dim_index":3
            }

        elif self.id==3:#INTERCHANGE04
            return {
                "first_dim_index":0,
                "second_dim_index":4
            }
        elif self.id==4:#INTERCHANGE05
            return {
                "first_dim_index":0,
                "second_dim_index":5
            }
        elif self.id==5:#INTERCHANGE06
            return {
                "first_dim_index":0,
                "second_dim_index":6
            }
        elif self.id==6:#INTERCHANGE07
            return {
                "first_dim_index":0,
                "second_dim_index":7
            }

        elif self.id==7:#INTERCHANGE12
            return {
                "first_dim_index":1,
                "second_dim_index":2
            }

        elif self.id==8:#INTERCHANGE13
            return {
                "first_dim_index":1,
                "second_dim_index":3
            }

        elif self.id==9:#INTERCHANGE14
            return {
                "first_dim_index":1,
                "second_dim_index":4
            }
        elif self.id==10:#INTERCHANGE15
            return {
                "first_dim_index":1,
                "second_dim_index":5
            }
        elif self.id==11:#INTERCHANGE16
            return {
                "first_dim_index":1,
                "second_dim_index":6
            }
        elif self.id==12:#INTERCHANGE17
            return {
                "first_dim_index":1,
                "second_dim_index":7
            }

        elif self.id==13:#INTERCHANGE23
            return {
                "first_dim_index":2,
                "second_dim_index":3
            }

        elif self.id==14:#INTERCHANGE24
            return {
                "first_dim_index":2,
                "second_dim_index":4
            }
        elif self.id==15:#INTERCHANGE25
            return {
                "first_dim_index":2,
                "second_dim_index":5
            }
        elif self.id==16:#INTERCHANGE26
            return {
                "first_dim_index":2,
                "second_dim_index":6
            }
        elif self.id==17:#INTERCHANGE27
            return {
                "first_dim_index":2,
                "second_dim_index":7
            }

        elif self.id==18:#INTERCHANGE34
            return {
                "first_dim_index":3,
                "second_dim_index":4
            }
        elif self.id==19:#INTERCHANGE35
            return {
                "first_dim_index":3,
                "second_dim_index":5
            }
        elif self.id==20:#INTERCHANGE36
            return {
                "first_dim_index":3,
                "second_dim_index":6
            }
        elif self.id==21:#INTERCHANGE37
            return {
                "first_dim_index":3,
                "second_dim_index":7
            }
        elif self.id==22:#INTERCHANGE45
            return {
                "first_dim_index":4,
                "second_dim_index":5
            }
        elif self.id==23:#INTERCHANGE46
            return {
                "first_dim_index":4,
                "second_dim_index":6
            }
        elif self.id==24:#INTERCHANGE47
            return {
                "first_dim_index":4,
                "second_dim_index":7
            }
        elif self.id==25:#INTERCHANGE56
            return {
                "first_dim_index":5,
                "second_dim_index":6
            }
        elif self.id==26:#INTERCHANGE57
            return {
                "first_dim_index":5,
                "second_dim_index":7
            }
        elif self.id==27:#INTERCHANGE67
            return {
                "first_dim_index":6,
                "second_dim_index":7
            }

        elif self.id in range(28,41):#TILING
            
            tiling_flag_1=True
            tiling_flag_2=True
            tiling_flag_3=True

            if self.id in range(28,35):
                depth=2
            else:
                depth=3

            #Tiling 2D
            if depth==2:
                if self.id == 28:#TILING2D01
                    first_it=0
                    second_it=1
                elif self.id == 29:#TILING2D12
                    first_it=1
                    second_it=2
                elif self.id == 30:#TILING2D23
                    first_it=2
                    second_it=3
                elif self.id == 31:#TILING2D34
                    first_it=3
                    second_it=4
                elif self.id == 32:#TILING2D45
                    first_it=4
                    second_it=5
                elif self.id == 33:#TILING2D56
                    first_it=5
                    second_it=6
                elif self.id == 34:#TILING2D67
                    first_it=6
                    second_it=7
                
                # The factors are chosen randomly from a set of factors that usually give good results, this can be extended
                first_fact=random.choice([32, 64, 128])
                second_fact=random.choice([32, 64, 128])
                print("after choosing first and second params and factors")

                #calculate the loop extent to see if we should create new iterators or not
                #since it's applicable on the common iterators, we retrieve the information from the first computation
                loop_extent_1=abs(self.it_dict[0][first_it]['upper_bound']-self.it_dict[0][first_it]['lower_bound'])
                print("\n first loop extent is ", loop_extent_1)
                print("first factor is", first_fact)
                if loop_extent_1==first_fact:
                    tiling_flag_1=False
                    print("tiling flag 1 false, loopextent == factor")
                elif loop_extent_1<first_fact:
                    print("exceeeption, loop extent 1 smaller than factor")
                    raise LoopExtentException

                loop_extent_2=abs(self.it_dict[0][second_it]['upper_bound']-self.it_dict[0][second_it]['lower_bound'])
                print("\n second loop extent is ", loop_extent_2)
                print("second factor is", second_fact)
                if loop_extent_2==second_fact:
                    tiling_flag_2=False
                    print("tiling flag 2 false, loopextent == factor")
                elif loop_extent_2<second_fact:
                    print("exceeeption, loop extent 2 smaller than factor")
                    raise LoopExtentException

            
                
                return {
                    "tiling_depth": 2,
                    "first_dim_index":first_it,
                    "tiling_loop_1":tiling_flag_1,
                    "second_dim_index":second_it,
                    "tiling_loop_2":tiling_flag_2,
                    "first_factor":first_fact,
                    "second_factor":second_fact,
                    "tiling_loop_3":None,
                    "third_factor":None,
                    "third_factor":None
                }

            # Tiling 3D
            elif depth==3:
                if self.id == 35:#TILING3D012
                    first_it=0
                    second_it=1
                    third_it=2
                elif self.id == 36:#TILING3D123
                    first_it=1
                    second_it=2
                    third_it=3
                elif self.id == 37:#TILING3D234
                    first_it=2
                    second_it=3
                    third_it=4
                elif self.id == 38:#TILING3D345
                    first_it=3
                    second_it=4
                    third_it=5
                elif self.id == 39:#TILING3D456
                    first_it=4
                    second_it=5
                    third_it=6
                elif self.id == 40:#TILING3D567
                    first_it=5
                    second_it=6
                    third_it=7
                    
                first_fact=random.choice([32, 64, 128])
                second_fact=random.choice([32, 64, 128])
                third_fact=random.choice([32, 64, 128])
                 #calculate the loop extent to see if we should create new iterators or not
                loop_extent_1=abs(self.it_dict[0][first_it]['upper_bound']-self.it_dict[0][first_it]['lower_bound'])
                print("\n first loop extent is ", loop_extent_1)
                print("first factor is", first_fact)
                if loop_extent_1==first_fact:
                    tiling_flag_1=False
                    print("tiling flag 1 false, loopextent == factor")
                elif loop_extent_1<first_fact:
                    print("exceeeption, loop extent 1 smaller than factor")
                    raise LoopExtentException

                loop_extent_2=abs(self.it_dict[0][second_it]['upper_bound']-self.it_dict[0][second_it]['lower_bound'])
                print("\n second loop extent is ", loop_extent_2)
                print("second factor is", second_fact)
                if loop_extent_2==second_fact:
                    tiling_flag_2=False
                    print("tiling flag 2 false, loopextent == factor")
                elif loop_extent_2<second_fact:
                    print("exceeeption, loop extent 2 smaller than factor")
                    raise LoopExtentException

                loop_extent_3=abs(self.it_dict[0][third_it]['upper_bound']-self.it_dict[0][third_it]['lower_bound'])
                print("\n third loop extent is ", loop_extent_3)
                print("third factor is", third_fact)
                if loop_extent_3==third_fact:
                    tiling_flag_3=False
                    print("tiling flag 3 false, loopextent == factor")
                elif loop_extent_3<third_fact:
                    print("exceeeption, loop extent 3 smaller than factor")
                    raise LoopExtentException

                return {
                    "tiling_depth": 3,
                    "first_dim_index":first_it,
                    "tiling_loop_1":tiling_flag_1,
                    "second_dim_index":second_it,
                    "tiling_loop_2":tiling_flag_2,
                    "third_dim_index":third_it,
                    "tiling_loop_3":tiling_flag_3,
                    "first_factor":first_fact,
                    "second_factor":second_fact,
                    "third_factor":third_fact
                }

        elif self.id==41:#UNROLLING
            params = {}
            for comp in self.it_dict:
                it=len(self.it_dict[comp])-1
                unrolling_fact=random.choice([4, 8, 16])
                params[comp]={
                "dim_index":it,
                "unrolling_factor":unrolling_fact
                }
                
            return params



        elif self.id==42:#SKEWING

            # The loop levels are randm
            it_len = len(self.common_it)
            first_it=random.randint(0,it_len-1)
            if first_it == it_len-1:
                second_it = first_it
                first_it= first_it-1
            else:
                second_it=first_it+1

            it_dict={"first_dim_index":first_it,"second_dim_index":second_it}
 
            solver_res = prog.call_solver(comp, it_dict)
           
            if solver_res == None or solver_res== '-1':
                return {
                "first_dim_index":first_it,
                "second_dim_index":second_it,
                "first_factor":None,
                "second_factor":None,
            }
            else:
                print("solver results are:",solver_res)
                return {
                    "first_dim_index":first_it,
                    "second_dim_index":second_it,
                    "first_factor":int(solver_res[0]),
                    "second_factor":int(solver_res[1]),
                }


        elif self.id==43: #PARALLELIZATION
            # Parallelization is better applied on the two outermost loops, we make a random choice
            # we iterate over the iterators list and look for the outermost loop, the one without a parent
            if len(self.common_it) >1:
                it=random.choice([len(self.it)-2, len(self.it)-1])
            else:
                it=len(self.common_it)
            return {
                "dim_index":it,
            } 
        
        elif self.id in range(44,52): 
            if self.id==44:#LOOP REVERSAL0
                it= 0
            elif self.id==45:#LOOP REVERSAL1
                it= 1
            elif self.id==46:#LOOP REVERSAL2
                it= 2
            elif self.id==47:#LOOP REVERSAL3
                it= 3
            elif self.id==48:#LOOP REVERSAL4
                it= 4
            elif self.id==49:#LOOP REVERSAL5
                it= 5
            elif self.id==50:#LOOP REVERSAL6
                it= 6
            elif self.id==51:#LOOP REVERSAL7
                it= 7
            return {
                "dim_index":it,
            } 
        