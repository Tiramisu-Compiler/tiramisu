import sys
from ActionEnhanced import Action

from pyfiglet import Figlet

from Tiramisu_ProgramLocal import Tiramisu_Program
from utilsEnhanced import (
    TimeOutException,
    get_cpp_file,
    get_dataset,
    get_representation_template,
    get_schedules_str,
    sched_str,
    update_iterators,
)
from Optim_cmd import optimization_command
import gym
import numpy as np

import ujson as json

import numpy as np
np.seterr(invalid='raise')

import random
import time
import copy
import time

class RepresentationLengthException(Exception):
    pass

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

class SearchSpaceSparseEnhanced(gym.Env):

    INTERCHANGE01 = 0
    INTERCHANGE02 = 1
    INTERCHANGE03 = 2
    INTERCHANGE04 = 3
    INTERCHANGE05 = 4
    INTERCHANGE06 = 5
    INTERCHANGE07 = 6
    INTERCHANGE12 = 7
    INTERCHANGE13 = 8
    INTERCHANGE14 = 9
    INTERCHANGE15 = 10
    INTERCHANGE16 = 11
    INTERCHANGE17 = 12
    INTERCHANGE23 = 13
    INTERCHANGE24 = 14
    INTERCHANGE25 = 15
    INTERCHANGE26 = 16
    INTERCHANGE27 = 17
    INTERCHANGE34 = 18
    INTERCHANGE35 = 19
    INTERCHANGE36 = 20
    INTERCHANGE37 = 21
    INTERCHANGE45 = 22
    INTERCHANGE46 = 23
    INTERCHANGE47 = 24
    INTERCHANGE56 = 25
    INTERCHANGE57 = 26
    INTERCHANGE67 = 27

    TILING2D01 = 28
    TILING2D12 = 29
    TILING2D23 = 30
    TILING2D34 = 31
    TILING2D45 = 32
    TILING2D56 = 33
    TILING2D67 = 34
    TILING3D012 = 35
    TILING3D123 = 36
    TILING3D234 = 37
    TILING3D345 = 38
    TILING3D456 = 39
    TILING3D567 = 40

    UNROLLING = 41

    SKEWING = 42

    PARALLELIZATION = 43

    REVERSAL0=44
    REVERSAL1=45
    REVERSAL2=46
    REVERSAL3=47
    REVERSAL4=48
    REVERSAL5=49
    REVERSAL6=50
    REVERSAL7=51

    EXIT=52

    MAX_DEPTH = 6

    ACTIONS_ARRAY=[ 'INTERCHANGE01', 'INTERCHANGE02', 'INTERCHANGE03', 'INTERCHANGE04', 'INTERCHANGE05', 'INTERCHANGE06', 'INTERCHANGE07',
    'INTERCHANGE12', 'INTERCHANGE13', 'INTERCHANGE14', 'INTERCHANGE15', 'INTERCHANGE16' , 'INTERCHANGE17', 'INTERCHANGE23', 'INTERCHANGE24',
    'INTERCHANGE25', 'INTERCHANGE26', 'INTERCHANGE27', 'INTERCHANGE34', 'INTERCHANGE35', 'INTERCHANGE36' , 'INTERCHANGE37', 'INTERCHANGE45',
    'INTERCHANGE46', 'INTERCHANGE47', 'INTERCHANGE56', 'INTERCHANGE57', 'INTERCHANGE67',  'TILING2D01', 'TILING2D12', 'TILING2D23', 'TILING2D34',
    'TILING2D45', 'TILING2D56', 'TILING2D67', 'TILING3D012', 'TILING3D123', 'TILING3D234', 'TILING3D345', 'TILING3D456', 'TILING3D567', 'UNROLLING',
    'SKEWING', 'PARALLELIZATION', 'REVERSAL0', 'REVERSAL1', 'REVERSAL2', 'REVERSAL3', 'REVERSAL4', 'REVERSAL5', 'REVERSAL6', 'REVERSAL7', 'EXIT']

    def __init__(self, programs_file, dataset_path):
        
        f = Figlet(font='banner3-D')
        print(f.renderText("Tiramisu"))
        print("Initialisation de l'environnement")

        #print("\nInitialisation de l'environnement")
        self.placeholders = []
        self.speedup = 0
        self.schedule = []

        print("Récupération des données depuis {} \n".format(dataset_path))
        #print("En train de charger le dataset...")

        #load already saved data
        self.progs_list=get_dataset(dataset_path)
        self.programs_file=programs_file


        self.tiramisu_progs=[]
        self.progs_annot={}
        

        f = open(self.programs_file)
        self.progs_dict=json.load(f)
        print("Dataset chargé!\n")

        self.scheds = get_schedules_str(
            list(self.progs_dict.keys()), self.progs_dict
        )  # to use it to get the execution time

        self.action_space = gym.spaces.Discrete(53)
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(794,))
        self.observation_space = gym.spaces.Dict({
            "representation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(784,)),
            "action_mask":gym.spaces.Box(low=0, high=1, shape=(53,))
        })

        self.dataset_path=dataset_path
        self.depth = 0
        self.nb_executions=5
        self.episode_total_time=0
        self.lc_total_time=0
        self.codegen_total_time=0
        self.prog_ind=0


    def reset(self, file=None):
        print("\nRéinitialisation de l'environnement\n")

        self.episode_total_time=time.time()
        self.lc_total_time=0
        self.codegen_total_time=0
        print("Choix d'un programme Tiramisu...\n")
        while True:
            try:

                # init_indc = random.randint(0, len(self.progs_list) - 1)

<<<<<<< HEAD
                file = get_cpp_file(self.dataset_path,self.progs_list[init_indc])
                
                #file='../benchmarks_sources/function_heat2d_MEDIUM/function_heat2d_MEDIUM_generator.cpp'
                #file='../benchmarks_sources/function_matmul_SMALL/function_matmul_SMALL_generator.cpp'
=======
                # file = get_cpp_file(self.dataset_path,self.progs_list[init_indc])

                file='../benchmarks_sources/function_heat2d_SMALL/function_heat2d_SMALL_generator.cpp'
>>>>>>> 6405030e15f319c97eea94c2480d0f5e219b84f9

                # self.prog contains the tiramisu prog from the RL interface
                self.prog = Tiramisu_Program(file)
                #print("Le programme numéro {} de la liste, nommé {} est choisi \n".format(init_indc, self.prog.name))

                self.annotations=self.prog.get_program_annotations()

                rep_temp = get_representation_template(self.annotations)

                print("\n *-*-*- Le code source -*-*-* \n")
                print(self.prog.original_str)


                if len(rep_temp[0]) != 784:
                    raise RepresentationLengthException

                if self.progs_dict == {} or self.prog.name not in self.progs_dict.keys():
                    try: 
                        print("\nC'est un nouveau programme, il sera éxecuté pour obtenir le temps d'execution initial\n")
                        start_time=time.time()
                        self.initial_execution_time=self.prog.evaluate_schedule([],'initial_exec', self.nb_executions)
                        cg_time=time.time()-start_time 
                        #print("\nLe temps d'execution initial de ce programme est", self.initial_execution_time)
                        self.codegen_total_time +=cg_time
                    except TimeOutException:
                        print("Désolé, je dois changer le programme choisi car son exécution a pris beaucoup de temps...\n ")
                        continue
                    self.progs_dict[self.prog.name]={}
                    self.progs_dict[self.prog.name]["initial_execution_time"]=self.initial_execution_time

                else:
                    print("Le temps d'exécution de ce programme existe dans les données générées.\n")
                    self.initial_execution_time=self.progs_dict[self.prog.name]["initial_execution_time"]

                print("Le temps d'exécution initial récupéré est: {}".format(self.initial_execution_time))

            except:
                continue

            self.placeholders = rep_temp[1]
            self.comp=self.prog.comp_name
            iterators=list(self.annotations["iterators"].keys())
            self.it_dict={}
            print("\nLes niveaux de boucles de ce programme sont:")
            for i in range (len(self.annotations["iterators"])):
                  self.it_dict[i]={}
                  self.it_dict[i]['iterator']=iterators[i]
                  self.it_dict[i]['lower_bound']=self.annotations['iterators'][iterators[i]]['lower_bound']
                  self.it_dict[i]['upper_bound']=self.annotations['iterators'][iterators[i]]['upper_bound']
                  print("\n{} : {}".format(i, self.it_dict[i]))

            
            
            

            self.obs={}
            self.obs["representation"] = np.array(rep_temp[0],dtype=np.float32)
            print("\nLa représentation vectorielle initiale de ce programme est:", self.obs["representation"] )

            if len(self.annotations["iterators"]) == 5:
                self.obs["action_mask"] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
            else:
                if len(self.annotations["iterators"]) == 4:
                    self.obs["action_mask"] = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=np.float32)
                else: 
                    if len(self.annotations["iterators"]) == 3:
                        self.obs["action_mask"] = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                    else: 
                        if len(self.annotations["iterators"]) == 2:
                            self.obs["action_mask"] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                        else:
                            if len(self.annotations["iterators"]) == 1:
                                self.obs["action_mask"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            
            print("\nLe masque inital des actions est: ",self.obs["action_mask"])
            self.depth = 0
            self.schedule = []
            self.schedule_str = ""
            self.speedup = 0
            self.is_interchaged=False
            self.is_tiled=False
            self.is_unrolled=False
            self.is_skewed=False
            self.is_parallelized=False
            self.is_reversed=False
            self.steps=0
            self.new_scheds={}

            self.search_time=time.time()

            return self.obs

    def step(self, raw_action):
        print("\nDébut d'un timestep")

        action_name=self.ACTIONS_ARRAY[raw_action]
        print("\nL'action {} est choisie".format(action_name))

        exit=False
        done=False
        info={}
        applied_exception=False
        skew_params_exception=False
        skew_unroll=False
        reward=0
        self.steps+=1

        try:
            
            action = Action(raw_action, self.it_dict)

            self.obs = copy.deepcopy(self.obs) # get current observation
            try:
                if action.id != self.SKEWING:
                    action_params = action.parameter()

                else:
                    action_params=action.parameter(self.comp, self.prog)
            except:
                print("\nOoops... Il n'existe pas de facteur convenable.")
                info = {"parameter exception": True}
                done = False
                #return self.obs, reward, done, info
                return self.obs, reward, done, info

                
            if action.id in range(28):
                print("\nLes paramètres sont:")
                print("\nLe premier niveau de boucle:", action_params["first_dim_index"])
                print("\nLe deuxième niveau de boucle:", action_params["second_dim_index"])

                if not self.is_interchaged:
                   
                    params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]

                    optim1 = optimization_command(self.comp, "Interchange", params)
                    self.schedule.append(optim1)
           
                    start_time = time.time()
                    lc_check = self.prog.check_legality_of_schedule(self.schedule)
                    l_time = time.time() - start_time

                    self.lc_total_time+=l_time                  


                    if lc_check == -1: 
                        print("\nCette action a généré une erreur")
                        self.obs["action_mask"][action.id]=0    
                        raise LCException

                    if lc_check == 0:
                        print("\nCette action est illégale")
                        self.schedule.pop()
                        
                        info = {"illegal_action": True}
                        done = False
                        return self.obs, reward, done, info
                    
                    print("\nCette action est légale")
                    self.apply_interchange(action_params)

                    self.is_interchaged=True
                    
                else:
                    print("interchange already applied exception")
                    applied_exception=True
                    raise IsInterchangedException
                    #to expierment with the reward in this case
                    
            if action.id in range(28,41):
                print("\nLes paramètres sont:")
                print("\nLe premier niveau de boucle:", action_params["first_dim_index"])
                print("\nLe deuxième niveau de boucle:", action_params["second_dim_index"])
                print("\nLe premier facteur:", action_params["first_factor"])
                print("\nLe deuxième facteur:", action_params["second_factor"])

                if not self.is_tiled:
                    params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]
                    params.append(action_params["first_factor"])
                    params.append(action_params["second_factor"])
                    
                    if action_params["tiling_depth"] == 3:
                        params.insert(2, action_params["third_dim_index"])
                        params.append(action_params["third_factor"])

                    if action_params["tiling_depth"] == 3:
                        print(action_params["tiling_loop_3"])

                    optim2 = optimization_command(self.comp, "Tiling", params)

                    self.schedule.append(optim2)

                    start_time = time.time()
                    lc_check = self.prog.check_legality_of_schedule(self.schedule)
                    l_time = time.time() - start_time
                    
                    self.lc_total_time+=l_time

                    
                    if lc_check == -1:  
                        print("\nCette action a généré une erreur")
 
                        raise LCException

                    if lc_check == 0:
                        print("\nCette action est illégale")
                        self.schedule.pop()
                        #reward = -1
                        info = {"illegal_action": True}
                        done = False
                        return self.obs, reward, done, info

                    print("\nCette action est légale")
                    self.apply_tiling(action_params)

                    self.is_tiled=True
                else:
                    print("\n tiling already applied exception")
                    applied_exception=True
                    raise IsTiledException

            if action.id == self.UNROLLING:
                if not self.is_unrolled:
                    print("\nLes paramètres sont:")
                    print("\nLe niveau de boucle:", action_params["dim_index"])
                    print("\nLe facteur:", action_params["unrolling_factor"])

                    params=[int(action_params["dim_index"]), int(action_params["unrolling_factor"])]

                    #we don't apply unrolling on a level that's skewed, we get the tag to see if it's skewed or not
                    if not self.is_skewed and self.obs["representation"][params[0]*9+3]!=1:

                        optim3 = optimization_command(self.comp, "Unrolling", params)
                        
                        self.schedule.append(optim3)

                        start_time = time.time()
                        lc_check = self.prog.check_legality_of_schedule(self.schedule)
                        l_time = time.time() - start_time
                        self.lc_total_time+=l_time
                        
                    
                        if lc_check == -1:  
                            print("\nCette action a généré une erreur")                         
                            raise LCException

                        if lc_check == 0:
                            print("\nCette action est illégale")
                            self.schedule.pop()
                            #reward = -1
                            info = {"illegal_action": True}
                            done = False
                            return self.obs, reward, done, info

                        print("\nCette action est légale")
                        self.obs["action_mask"][41]=0
                        self.is_unrolled=True

                    else:
                        lc_check=0
                        info['error']="trying to apply unrolling after skewing"
                    
                else:
                    applied_exception=True
                    print("\n unrolling is already applied")

                    raise IsUnrolledException

            if action.id == self.SKEWING:

                print("\nLes paramètres sont:")
                print("\nLe premier niveau de boucle:", action_params["first_dim_index"])
                print("\nLe deuxième niveau de boucle:", action_params["second_dim_index"])
                print("\nLe premier facteur:", action_params["first_factor"])
                print("\nLe deuxième facteur:", action_params["second_factor"])

                if not self.is_skewed:

                
                    if (action_params["first_factor"] != None and action_params["second_factor"] != None):
                        if (action_params["first_dim_index"] != len(self.it_dict)-1 and action_params["second_dim_index"] != len(self.it_dict)-1) or ( (action_params["first_dim_index"] == len(self.it_dict)-1 or action_params["second_dim_index"] == len(self.it_dict)-1 and not self.is_unrolled )) :

                            params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]
                            params.append(action_params["first_factor"])
                            params.append(action_params["second_factor"])

                            optim4 = optimization_command(self.comp, "Skewing", params)

                            self.schedule.append(optim4)

                            start_time = time.time()
                            lc_check = self.prog.check_legality_of_schedule(self.schedule)
                            l_time = time.time() - start_time
                            self.lc_total_time+=l_time

                        
                            if lc_check == -1:  
                                print("\nCette action a généré une erreur")   
                                raise LCException

                            if lc_check == 0:
                                print("\nCette action est illégale")
                                self.schedule.pop()
                                info = {"illegal_action": True}
                                done = False
                                return self.obs, reward, done, info

                            print("\nCette action est légale")

                            self.apply_skewing(action_params)
                            self.is_skewed=True

                        else:
                            skew_unroll=True
                            raise SkewUnrollException

                    else:
                        print("\n Pas de paramètres de Skewing pour ces deux niveaux de boucles!")
                        skew_params_exception=True
                        raise SkewParamsException

                
                
                else:
                    print("\n sekwing is already applied")
                    applied_exception=True
                    raise IsSkewedException

            if action.id == self.PARALLELIZATION:

                print("\nLes paramètres sont:")
                print("\nLe niveau de boucle:", action_params["dim_index"])

                if not self.is_parallelized:
                    params=[int(action_params["dim_index"])]

                    optim5 = optimization_command(self.comp, "Parallelization", params)
                
                    self.schedule.append(optim5)

                    start_time = time.time()
                    lc_check = self.prog.check_legality_of_schedule(self.schedule)
                    l_time = time.time() - start_time
                    self.lc_total_time+=l_time

                    if lc_check == -1:    
                        print("\nCette action a généré une erreur")      
                        raise LCException

                    if lc_check == 0:
                        print("\nCette action est illégale")
                        self.schedule.pop()
                        info = {"illegal_action": True}
                        done = False
                        return self.obs, reward, done, info

                    print("\nCette action est légale")
                    self.apply_parallelization(action_params)
                    
                    self.is_parallelized=True
                else:
                    applied_exception=True
                    print("\n parallelisation is already applied")
                    raise IsParallelizedException

            if action.id in range(44,52):
                print("\nLes paramètres sont:")
                print("\nLe niveau de boucle:", action_params["dim_index"])
                
                if not self.is_reversed:

                    params=[int(action_params["dim_index"])]

                    optim6 = optimization_command(self.comp, "Reversal", params)

                    self.schedule.append(optim6)
                    
                    start_time=time.time()
                    lc_check = self.prog.check_legality_of_schedule(self.schedule)
                    l_time = time.time() - start_time
                    self.lc_total_time+=l_time

                    if lc_check == -1: 
                        print("\nCette action a généré une erreur")   
                        raise LCException

                    if lc_check == 0:
                        print("\nCette action est illégale")
                        self.schedule.pop()
                        info = {"illegal_action": True}
                        done = False
                        return self.obs, reward, done, info


                    print("\nCette action est légale")

                    self.apply_reversal(action_params)
                   
                    self.is_reversed=True
                else:
                    applied_exception=True

                    print("\n loop reversal already applied")

                    raise IsReversedException
     
            if action.id==self.EXIT:
                done=True
                exit=True
                
            if (not exit and lc_check!=0) and not (action.id == 41 and self.is_skewed):
                self.schedule_str = sched_str(self.schedule_str, action.id, action_params)
                if not action.id == 41:
                    self.it_dict=update_iterators(action.id, self.it_dict, action_params)
                    print("\nLe nouveau dictionnaire des niveaux de boucles:")
                    for i in range(len(self.it_dict)):
                        print("\n {}: {}".format(i, self.it_dict[i]))

                self.depth += 1
            

        except Exception as e:
            print(e.__class__.__name__)
            if applied_exception:
                print("applied exception")
                info = {"more than one time": True}
                done = False
                return self.obs, reward, done, info

            else:
                ex_type, ex_value, ex_traceback = sys.exc_info()
                if self.schedule != [] and not skew_params_exception and not skew_unroll:
                    self.schedule.pop()
                print("\nCette action a généré une erreur, elle ne sera pas appliquée.")
                #print("else exception", ex_type.__name__, ex_value, ex_traceback)
                done = False
                info = {
                    "depth": self.depth,
                    "error": "ended with error in the step function",
                }

                return self.obs, reward, done, info
        finally:


            if(self.depth == self.MAX_DEPTH) or (self.steps >=20):
                done=True


            if done: 
                print("\nFin de l'épisode")

                if self.is_unrolled:       
                    for optim in self.schedule:
                        #print(optim.type)
                        if optim.type == "Unrolling":

                            unroll_optimisation=optim

                    
                    new_unrolling_params={"dim_index":len(self.it_dict)-1,"unrolling_factor":unroll_optimisation.params_list[1]}
                    new_unrolling_optim_params=[len(self.it_dict)-1,unroll_optimisation.params_list[1]]
                    new_unrolling_optim=optimization_command(self.comp, "Unrolling", new_unrolling_optim_params)
                    new_unrolling_str="U(L"+str(len(self.it_dict)-1)+","+str(unroll_optimisation.params_list[1])+")"

                    unrolling_str="U(L"+str(unroll_optimisation.params_list[0])+","+str(unroll_optimisation.params_list[1])+")" 

                    self.schedule.remove(unroll_optimisation)      

                    self.schedule_str=self.schedule_str.replace(unrolling_str, "") + new_unrolling_str
                    self.schedule.append(new_unrolling_optim)
                    
                    self.apply_unrolling(new_unrolling_params)
                    



                self.search_time= time.time()-self.search_time
                
                try:
                    exec_time=0
                    writing_time=0
                    exec_time = self.get_exec_time()
                    print("\nTester si la parallélisation apporte un meilleur speedup...")

                    if not self.is_parallelized:
                        #print("inside parallelization in done")
                        action = Action(self.PARALLELIZATION, self.it_dict)
                        action_params = action.parameter()
                        params=[int(action_params["dim_index"])]

                        optim5 = optimization_command(self.comp, "Parallelization", params)
                        
                        self.schedule.append(optim5)
            
                        try:

                            self.schedule_str = sched_str(self.schedule_str, action.id, action_params)
                            parallelized_exec_time=self.get_exec_time()
                            parallelization_str='P(L'+str(action_params["dim_index"])+')'
                            
                            # print("the exec time with parallelization is", parallelized_exec_time)
                            # print("the exec time without parallelization is", exec_time)
                        except:
                            self.schedule.remove(optim5)
                            self.schedule_str=self.schedule_str.replace(parallelization_str, "")
                            
                        
                        if parallelized_exec_time < exec_time:
                            exec_time = parallelized_exec_time
                            
                            self.apply_parallelization(action_params)
                            print("La parallélisation améliore le temps d'exécution donc elle est appliquée.")

                        else:
                            self.schedule.remove(optim5)
                            self.new_scheds[self.prog.name].pop(self.schedule_str)
                            self.schedule_str=self.schedule_str.replace(parallelization_str, "")
                            print("La parallélisation n'améliore pas le temps d'exécution, alors elle n'est pas appliquée.")
                           
                            
                except:
                    info = {"Internal execution error": True}

                    print("\nErreur lors de la mesure du temps d'exécution.")                    
                    return self.obs, reward, done, info

                print("\nLa représentation finale est", self.obs["representation"])

                if exec_time!=0:
                    print("\nLe schedule final trouvé est: ",self.schedule_str)
                    print("\nLe nouveau temps d'exécution est: {} s".format(exec_time))
                    if self.initial_execution_time >=  exec_time:
                        self.speedup = (self.initial_execution_time / exec_time)

                    else:
                        self.speedup = -(exec_time / self.initial_execution_time )

                    reward=self.speedup
                    print("\nLe speedup résultant est", (self.initial_execution_time / exec_time))
                
                    start_time=time.time()
                    try:
                        self.save_sched_to_dataset()
                        self.write_data()
                        writing_time=time.time()-start_time
                        #print("Data saved in ",writing_time)
                        print("\nSchedule sauvegardé avec succès!")
                    except:
                        print("\nErreur lors de la sauvegarde du schedule")

                self.episode_total_time= time.time()-self.episode_total_time
                # print("CODE GEN :",self.codegen_total_time)
                # print("LC : ",self.lc_total_time)
                # print("\nEPISODE TOTAL TIME : {}\nLEGALITY CHECK TIME RATIO : {}\nCODE GENERATION TIME RATIO : {}\nWRITING TIME RATIO : {}\n".format(self.episode_total_time, self.lc_total_time/self.episode_total_time, self.codegen_total_time/self.episode_total_time, writing_time/self.episode_total_time))

            info["depth"] =  self.depth

            if done == False:
                print("\nLa récompense retournée est ",reward)
            return self.obs, reward, done, info

        


    def apply_interchange(self, action_params):
        l_code = "L" + self.it_dict[action_params["first_dim_index"]]['iterator']
        self.obs["representation"][self.placeholders[l_code + "Interchanged"]] = 1
        l_code = "L" + self.it_dict[action_params["second_dim_index"]]['iterator']
        self.obs["representation"][self.placeholders[l_code + "Interchanged"]] = 1
        for i in range(28):
            self.obs["action_mask"][i]=0


    def apply_tiling(self, action_params):
       
        first_dim_index=action_params["first_dim_index"]
        second_dim_index=action_params["second_dim_index"]

        l_code = "L" + self.it_dict[first_dim_index]['iterator']
        self.obs["representation"][self.placeholders[l_code + "Tiled"]] = 1
        self.obs["representation"][self.placeholders[l_code + "TileFactor"]] = action_params[
            "first_factor"
        ]

        #update the loop bounds if tiling is applied on loop 1
        if action_params["tiling_loop_1"]:
            #print("inside loop tiling 1")
            new_upper_bound_1=self.obs["representation"][first_dim_index*18+1]/action_params["first_factor"]
            self.obs["representation"][first_dim_index*18+1]=new_upper_bound_1
            new_inner_upper_bound_1=action_params["first_factor"]
            self.obs["representation"][first_dim_index*18+10]=new_inner_upper_bound_1
            #print("after loop tiling 1")

        l_code = "L" + self.it_dict[second_dim_index]['iterator']
        self.obs["representation"][self.placeholders[l_code + "Tiled"]] = 1
        self.obs["representation"][self.placeholders[l_code + "TileFactor"]] = action_params[
            "second_factor"
        ]
        #update the loop bounds if tiling is applied on loop 2
        if action_params["tiling_loop_2"]:
            #print("inside loop tiling 2")
            new_upper_bound_2=self.obs["representation"][second_dim_index*18+1]/action_params["second_factor"]
            self.obs["representation"][second_dim_index*18+1]=new_upper_bound_2
            new_inner_upper_bound_2=action_params["second_factor"]
            self.obs["representation"][second_dim_index*18+10]=new_inner_upper_bound_2
            #print("after loop tiling 2")

        if action_params["tiling_depth"] == 3:
            third_dim_index=action_params["third_dim_index"]
            l_code = "L" + self.it_dict[third_dim_index]['iterator']
            self.obs["representation"][self.placeholders[l_code + "Tiled"]] = 1
            self.obs["representation"][self.placeholders[l_code + "TileFactor"]] = action_params[
                "third_factor"
            ]
            #update the loop bounds if tiling is applied on loop 3
            if action_params["tiling_loop_3"]:
                #print("inside loop tiling 3")
                new_upper_bound_3=self.obs["representation"][third_dim_index*18+1]/action_params["third_factor"]
                self.obs["representation"][third_dim_index*18+1]=new_upper_bound_3
                new_inner_upper_bound_3=action_params["third_factor"]
                self.obs["representation"][third_dim_index*18+10]=new_inner_upper_bound_3
                #print("after loop tiling 3")
        
        
        if self.is_interchaged == False:

            if len(self.annotations["iterators"]) == 5:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE05, self.INTERCHANGE06, self.INTERCHANGE07, self.INTERCHANGE15, self.INTERCHANGE16, self.INTERCHANGE17, 
                    self.INTERCHANGE25, self.INTERCHANGE26, self.INTERCHANGE27, self.INTERCHANGE35, self.INTERCHANGE36, self.INTERCHANGE37, 
                    self.INTERCHANGE45, self.INTERCHANGE46, self.INTERCHANGE47,self.INTERCHANGE56,self.INTERCHANGE57, self.INTERCHANGE67]]=1
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE05, self.INTERCHANGE06, self.INTERCHANGE15, self.INTERCHANGE16, self.INTERCHANGE25, self.INTERCHANGE26, 
                    self.INTERCHANGE35, self.INTERCHANGE36, self.INTERCHANGE45, self.INTERCHANGE46, self.INTERCHANGE56]]=1
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][[self.INTERCHANGE05, self.INTERCHANGE15, self.INTERCHANGE25,self.INTERCHANGE35, self.INTERCHANGE45]]=1

            if len(self.annotations["iterators"]) == 4:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE04, self.INTERCHANGE05, self.INTERCHANGE06, self.INTERCHANGE14, self.INTERCHANGE15, self.INTERCHANGE16, 
                    self.INTERCHANGE24, self.INTERCHANGE25, self.INTERCHANGE26, self.INTERCHANGE34, self.INTERCHANGE35, self.INTERCHANGE36, 
                    self.INTERCHANGE45, self.INTERCHANGE46, self.INTERCHANGE56]]=1
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE04, self.INTERCHANGE05, self.INTERCHANGE14, self.INTERCHANGE15,
                    self.INTERCHANGE24, self.INTERCHANGE25, self.INTERCHANGE34, self.INTERCHANGE35, self.INTERCHANGE45]]=1
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][[self.INTERCHANGE04, self.INTERCHANGE14, self.INTERCHANGE24, self.INTERCHANGE34]]=1    

            if len(self.annotations["iterators"]) == 3:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE03, self.INTERCHANGE04, self.INTERCHANGE05, self.INTERCHANGE13, self.INTERCHANGE14, self.INTERCHANGE15, 
                    self.INTERCHANGE23, self.INTERCHANGE24, self.INTERCHANGE25, self.INTERCHANGE34, self.INTERCHANGE35, 
                    self.INTERCHANGE45]]=1    
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE03, self.INTERCHANGE04, self.INTERCHANGE13, self.INTERCHANGE14,
                    self.INTERCHANGE23, self.INTERCHANGE24, self.INTERCHANGE34]]=1 
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][[self.INTERCHANGE03, self.INTERCHANGE13, self.INTERCHANGE23]]=1 
            
            if len(self.annotations["iterators"]) == 2:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE02, self.INTERCHANGE03, self.INTERCHANGE04, self.INTERCHANGE12, self.INTERCHANGE13, self.INTERCHANGE14, 
                    self.INTERCHANGE23, self.INTERCHANGE24, self.INTERCHANGE34]]=1    
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE02, self.INTERCHANGE03, self.INTERCHANGE12, self.INTERCHANGE13, self.INTERCHANGE23]]=1 
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][[self.INTERCHANGE02, self.INTERCHANGE12]]=1 

            if len(self.annotations["iterators"]) == 1:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE01, self.INTERCHANGE02, self.INTERCHANGE03, self.INTERCHANGE12, self.INTERCHANGE13, self.INTERCHANGE23]]=1    
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.INTERCHANGE01, self.INTERCHANGE02, self.INTERCHANGE12, self.INTERCHANGE13]]=1    
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][[self.INTERCHANGE01]]=1  

        if self.is_reversed == False:
            if len(self.annotations["iterators"]) == 5:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL5,self.REVERSAL6, self.REVERSAL7]]=1
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL5,self.REVERSAL6]]=1
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][self.REVERSAL5]=1

            elif len(self.annotations["iterators"]) == 4:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL4,self.REVERSAL5, self.REVERSAL6]]=1
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL4,self.REVERSAL5]]=1
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][self.REVERSAL4]=1

            elif len(self.annotations["iterators"]) == 3:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL3,self.REVERSAL4, self.REVERSAL5]]=1
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL3,self.REVERSAL4]]=1
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][self.REVERSAL3]=1

            elif len(self.annotations["iterators"]) == 2:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL2,self.REVERSAL3, self.REVERSAL4]]=1
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL2,self.REVERSAL3]]=1
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][self.REVERSAL2]=1

            elif len(self.annotations["iterators"]) == 1:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL1,self.REVERSAL2, self.REVERSAL3]]=1
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    self.obs["action_mask"][[self.REVERSAL1,self.REVERSAL2]]=1
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                    self.obs["action_mask"][self.REVERSAL1]=1
        
        for i in range(28,41):
            self.obs["action_mask"][i]=0

            

    def apply_unrolling(self, action_params):

        self.obs["representation"][self.placeholders["Unrolled"]] = 1
        self.obs["representation"][self.placeholders["UnrollFactor"]] = action_params["unrolling_factor"]

        l_code = "L" + self.it_dict[action_params["dim_index"]]['iterator']
        index_upper_bound=self.placeholders[l_code+'Interchanged']-1
        self.obs["representation"][index_upper_bound]=self.obs["representation"][index_upper_bound]/action_params["unrolling_factor"]

        

    def apply_skewing(self, action_params):

        dim_1=action_params["first_dim_index"]
        dim_2=action_params["second_dim_index"]

        l1_code = "L" + self.it_dict[dim_1]['iterator']
        l2_code = "L" + self.it_dict[dim_2]['iterator']

        #to get the start of the iterator in the representation template (just after the bounds)
        index1_upper_bound=self.placeholders[l1_code+'Interchanged']-1
        index1_lower_bound=self.placeholders[l1_code+'Interchanged']-2
        index2_upper_bound=self.placeholders[l2_code+'Interchanged']-1
        index2_lower_bound=self.placeholders[l2_code+'Interchanged']-2

        l1_lower_bound=self.obs["representation"][index1_lower_bound]
        l1_upper_bound=self.obs["representation"][index1_upper_bound]
        l2_lower_bound=self.obs["representation"][index2_lower_bound]
        l2_upper_bound=self.obs["representation"][index2_upper_bound]

        l1_extent = l1_upper_bound - l1_lower_bound
        l2_extent = l2_upper_bound - l2_lower_bound

        skew_factor = action_params["first_factor"]
        self.obs["representation"][self.placeholders[l1_code + "Skewed"]] = 1
        self.obs["representation"][self.placeholders[l1_code + "SkewFactor"]] = skew_factor
        self.obs["representation"][index1_lower_bound]= abs(action_params["first_factor"]) * l1_lower_bound
        self.obs["representation"][index1_upper_bound]= l1_lower_bound + abs(action_params["first_factor"]) * l1_extent + abs(action_params["second_factor"]) * l2_extent

        skew_factor = action_params["second_factor"]
        self.obs["representation"][self.placeholders[l2_code + "Skewed"]] = 1
        self.obs["representation"][self.placeholders[l2_code + "SkewFactor"]] = skew_factor
        self.obs["representation"][index2_lower_bound]= 0
        self.obs["representation"][index2_upper_bound]=(l2_extent) + 1

        self.obs["action_mask"][42]=0

    def apply_parallelization(self, action_params):
        l_code = "L" + self.it_dict[action_params["dim_index"]]['iterator']

        self.obs["representation"][self.placeholders[l_code + "Parallelized"]] = 1
        self.obs["action_mask"][43]=0

    def apply_reversal(self, action_params):
        l_code = "L" + self.it_dict[action_params["dim_index"]]['iterator']

        index_upper_bound=self.placeholders[l_code+'Interchanged']-1
        index_lower_bound=self.placeholders[l_code+'Interchanged']-2

        self.obs["representation"][self.placeholders[l_code + "Reversed"]] = 1

        tmp=self.obs["representation"][index_lower_bound]
        self.obs["representation"][index_lower_bound]=self.obs["representation"][index_upper_bound]
        self.obs["representation"][index_upper_bound]=tmp 

        for i in range(44,52):
            self.obs["action_mask"][i]=0




    
    #add new_scheds to the original schedules list
    def save_sched_to_dataset(self):
        for func in self.new_scheds.keys():
            for schedule_str in self.new_scheds[func].keys():#schedule_str represents the key, for example: 'Interchange Unrolling Tiling', the value is a tuple(schedule,execution_time)

                schedule=self.new_scheds[func][schedule_str][0]#here we get the self.obs["schedule"] containing the omtimizations list
                exec_time=self.new_scheds[func][schedule_str][1]
                search_time=self.new_scheds[func][schedule_str][2]
                comp=self.comp

                #Initialize an empty dict
                sched_dict={
                comp: {
                "schedule_str":schedule_str,
                "search_time":search_time,
                "interchange_dims": [],
                "tiling": {
                    "tiling_depth":None,
                    "tiling_dims":[],
                    "tiling_factors":[]
                },
                "unrolling_factor": None,
                "parallelized_dim": None,
                "reversed_dim": None,
                "skewing": {    
                    "skewed_dims": [],
                    "skewing_factors": [],
                    "average_skewed_extents": [],
                    "transformed_accesses": []
                            },
                "unfuse_iterators": [],
                "tree_structure": {},
                "execution_times": []}
                }

                for optim in schedule:
                    if optim.type == 'Interchange':
                        sched_dict[comp]["interchange_dims"]=[self.it_dict[optim.params_list[0]]['iterator'], self.it_dict[optim.params_list[1]]['iterator']]

                    elif optim.type == 'Skewing':
                        first_dim_index=self.it_dict[optim.params_list[0]]['iterator']
                        second_dim_index= self.it_dict[optim.params_list[1]]['iterator']
                        first_factor=optim.params_list[2]
                        second_factor=optim.params_list[3]

                        sched_dict[comp]["skewing"]["skewed_dims"]=[first_dim_index,second_dim_index]
                        sched_dict[comp]["skewing"]["skewing_factors"]=[first_factor,second_factor]

                    elif optim.type == 'Parallelization':
                         sched_dict[comp]["parallelized_dim"]=self.it_dict[optim.params_list[0]]['iterator']

                    elif optim.type == 'Tiling':
                        #Tiling 2D
                        if len(optim.params_list)==4:
                            sched_dict[comp]["tiling"]["tiling_depth"]=2
                            sched_dict[comp]["tiling"]["tiling_dims"]=[self.it_dict[optim.params_list[0]]['iterator'],self.it_dict[optim.params_list[1]]['iterator']]
                            sched_dict[comp]["tiling"]["tiling_factors"]=[optim.params_list[2],optim.params_list[3]]

                        #Tiling 3D
                        elif len(optim.params_list)==6:
                            sched_dict[comp]["tiling"]["tiling_depth"]=3
                            sched_dict[comp]["tiling"]["tiling_dims"]=[self.it_dict[optim.params_list[0]]['iterator'],self.it_dict[optim.params_list[1]]['iterator'],self.it_dict[optim.params_list[2]]['iterator']]
                            sched_dict[comp]["tiling"]["tiling_factors"]=[optim.params_list[3],optim.params_list[4],optim.params_list[5]]

                    elif optim.type == 'Unrolling':
                         sched_dict[comp]["unrolling_factor"]=optim.params_list[0]

                    elif optim.type == 'Reversal':
                         sched_dict[comp]["reversed_dim"]=self.it_dict[optim.params_list[0]]['iterator']

                sched_dict[comp]["execution_times"].append(exec_time)
                if not "schedules_list" in self.progs_dict[func].keys():
                    self.progs_dict[func]["schedules_list"]=[sched_dict]
                else:
                    self.progs_dict[func]["schedules_list"].append(sched_dict)
            
       

    def write_data(self):
        print("\nSauvegarde de données")
        with open(self.programs_file, 'w') as f:
            json.dump(self.progs_dict, f)
        f.close()



    def get_exec_time(self):
        prog_name= self.prog.name
        execution_time=0
        schedule_str = copy.deepcopy(self.schedule_str)
        if schedule_str != "" and self.schedule != []:
            if prog_name in self.scheds.keys():
               
                if schedule_str in self.scheds[prog_name]:
                   
                   
                    execution_time=self.scheds[prog_name][schedule_str][0]
                  

                else:  
                    if prog_name in self.new_scheds.keys():
                        if schedule_str in self.new_scheds[prog_name].keys():
                            
                            execution_time=self.new_scheds[prog_name][schedule_str][1]
                           
                    else:
                        curr_sched=copy.deepcopy(self.schedule)
                        self.new_scheds[prog_name]={}
                        execution_time=self.prog.evaluate_schedule(self.schedule,'sched_eval',self.nb_executions, self.initial_execution_time)
                        self.new_scheds[prog_name][schedule_str]=(curr_sched,execution_time,0)

                    
            else:

                if prog_name in self.new_scheds.keys():
                    if schedule_str in self.new_scheds[prog_name].keys():
                        execution_time=self.new_scheds[prog_name][schedule_str][1]
                        

                    else:
                        curr_sched=copy.deepcopy(self.schedule)
                        execution_time=self.prog.evaluate_schedule(self.schedule,'sched_eval',self.nb_executions, self.initial_execution_time)
                        self.new_scheds[prog_name][schedule_str]=(curr_sched,execution_time,0)
                        

                else:
                    curr_sched=copy.deepcopy(self.schedule)
                    self.new_scheds[prog_name]={}
                    start_time=time.time()
                    execution_time=self.prog.evaluate_schedule(self.schedule,'sched_eval',self.nb_executions, self.initial_execution_time)
                    sched_time=time.time()-start_time
                    self.codegen_total_time+=sched_time

                    self.new_scheds[prog_name][schedule_str]=(curr_sched,execution_time,0)

        else:
            execution_time=self.initial_execution_time

        if self.schedule_str == "":
            print("\nAucun schedule à appliquer")
        else:
            print("\nLe temps de l'execution du programme {} en appliquant le schedule {} est {} s".format(self.prog.name, self.schedule_str,execution_time))
        return execution_time
