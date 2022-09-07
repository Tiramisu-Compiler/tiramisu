import random
import sys
#from pyfiglet import Figlet
from ActionEnhanced import Action
from Tiramisu_Program import Tiramisu_Program
from Optim_cmd import optimization_command
import gym
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import ujson as json
np.seterr(invalid='raise')
import time
import copy
import time
from utilsEnhanced import (
    TimeOutException,
    get_cpp_file,
    get_dataset,
    get_representation,
    get_schedules_str,
    sched_str,
    update_iterators,
)

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

class SearchSpaceSparseEnhancedMult(gym.Env):

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

    UNROLLING4 = 41
    UNROLLING8 = 42
    UNROLLING16 = 43

    SKEWING01 = 44
    SKEWING12 = 45

    PARALLELIZATION0 = 46
    PARALLELIZATION1 = 47

    REVERSAL0=48
    REVERSAL1=49
    REVERSAL2=50
    REVERSAL3=51
    REVERSAL4=52
    REVERSAL5=53
    REVERSAL6=54
    REVERSAL7=55

    FUSION0=56
    FUSION1=57
    FUSION2=58
    FUSION3=59
    FUSION4=60

    EXIT=61
    

    MAX_DEPTH = 7

    ACTIONS_ARRAY=[ 'INTERCHANGE01', 'INTERCHANGE02', 'INTERCHANGE03', 'INTERCHANGE04', 'INTERCHANGE05', 'INTERCHANGE06', 'INTERCHANGE07',
    'INTERCHANGE12', 'INTERCHANGE13', 'INTERCHANGE14', 'INTERCHANGE15', 'INTERCHANGE16' , 'INTERCHANGE17', 'INTERCHANGE23', 'INTERCHANGE24',
    'INTERCHANGE25', 'INTERCHANGE26', 'INTERCHANGE27', 'INTERCHANGE34', 'INTERCHANGE35', 'INTERCHANGE36' , 'INTERCHANGE37', 'INTERCHANGE45',
    'INTERCHANGE46', 'INTERCHANGE47', 'INTERCHANGE56', 'INTERCHANGE57', 'INTERCHANGE67',  'TILING2D01', 'TILING2D12', 'TILING2D23', 'TILING2D34',
    'TILING2D45', 'TILING2D56', 'TILING2D67', 'TILING3D012', 'TILING3D123', 'TILING3D234', 'TILING3D345', 'TILING3D456', 'TILING3D567', 'UNROLLING4', 'UNROLLING8', 'UNROLLING16',
    'SKEWING01', 'SKEWING01', 'PARALLELIZATION0', 'PARALLELIZATION1', 'REVERSAL0', 'REVERSAL1', 'REVERSAL2', 'REVERSAL3', 'REVERSAL4', 'REVERSAL5', 'REVERSAL6', 'REVERSAL7', 'FUSION0',
    'FUSION1', 'FUSION2', 'FUSION3', 'FUSION4','EXIT']


    def __init__(self, programs_file, dataset_path):

        # f = Figlet(font='banner3-D')
        # print(f.renderText("Tiramisu"))
        print("Initialisation de l'environnement")
        
        self.placeholders = []
        self.speedup = 0
        self.schedule = []

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

        self.action_space = gym.spaces.Discrete(62)
        self.observation_space = gym.spaces.Dict({
            "representation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,1052)),
            "action_mask":gym.spaces.Box(low=0, high=1, shape=(62,)),
            "loops_representation":gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,26)),
            "child_list":gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,11)),
            "has_comps":gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,)),
            "computations_indices":gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,5)),
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
        while True:
            try:
                init_indc = random.randint(0, len(self.progs_list) - 1)

                file = get_cpp_file(self.dataset_path,self.progs_list[init_indc])
                # file = get_cpp_file(self.dataset_path,file) 
                #file = "../../Dataset_Multi/function000719/function000719_generator.cpp"
                
                # self.prog contains the tiramisu prog from the RL interface
                self.prog = Tiramisu_Program(file)
                #print("Le programme numéro {} de la liste, nommé {} est choisi \n".format(init_indc, self.prog.name))
                print("\n *-*-*- Le code source -*-*-* \n")
                print(self.prog.original_str)

                self.comps = list(self.prog.comp_name)

                self.annotations=self.prog.get_program_annotations()

                prog_rep, comps_placeholders, self.comp_indic_dict = get_representation(self.annotations)

                print("the length is", len(prog_rep[0]))

                for comp_rep in prog_rep:
                    if len(comp_rep) != 1052:
                        raise RepresentationLengthException
                
                
                if len(self.comps)!= 1:
                    print("more than one comp")
                    self.comps_it = []
                    for comp in self.comps:
                        self.comps_it.append(self.annotations["computations"][comp]["iterators"])
                    
                    #print("got the comp it", self.comps_it)

                    self.common_it = self.comps_it[0]

                    for comp_it in self.comps_it[1:]:
                        #print("common it is ", self.common_it)
                        self.common_it = [it for it in comp_it if it in self.common_it]

                    print("the common iterators are", self.common_it)

                elif len(self.comps)>5: # To avoid IndexError in self.obs["representation"]
                    continue

                else:
                    print("one comp, no need for common iterators")
                    self.common_it= self.annotations["computations"][self.comps[0]]["iterators"]

                if self.progs_dict == {} or self.prog.name not in self.progs_dict.keys():
                    try: 
                        print("getting the intitial exe time by execution")
                        start_time=time.time()
                        self.initial_execution_time=self.prog.evaluate_schedule([],'initial_exec', self.nb_executions)
                        cg_time=time.time()-start_time 
                        #print("After getting initial exec time:",cg_time, "initial exec time is :", self.initial_execution_time)
                        self.codegen_total_time +=cg_time
                    except TimeOutException:
                        continue
                    self.progs_dict[self.prog.name]={}
                    self.progs_dict[self.prog.name]["initial_execution_time"]=self.initial_execution_time

                else:
                    print("the initial execution time exists")
                    self.initial_execution_time=self.progs_dict[self.prog.name]["initial_execution_time"]

                print("The initial execution time is", self.initial_execution_time)

            except:
                continue

            self.placeholders = comps_placeholders
            self.added_iterators=[]   

            self.obs={}
            self.obs["representation"] = np.empty((0,1052),np.float32)
            self.obs["loops_representation"]=np.empty((0,26),np.float32)
            self.obs['child_list']=np.empty((0,11),np.float32)
            self.obs['has_comps']=np.empty((0,12),np.float32)
            self.obs['computations_indices']=np.empty((0,5),np.float32)

            for i in range (5):
                if i>=len(prog_rep):
                    self.obs["representation"]=np.vstack([self.obs["representation"], np.zeros(1052)])
                else:
                    self.obs["representation"]=np.vstack([self.obs["representation"], np.array([prog_rep[i]],dtype=np.float32)])

            #print("\nLa représentation vectorielle initiale de ce programme est:", self.obs["representation"] )
            
            print("\nLes niveaux de boucles de ce programme sont:")
            self.it_dict={}
            for comp in self.comps:        
                comp_it_dict={}
                iterators=list(self.annotations["computations"][comp]["iterators"])
                
                for i in range (len(iterators)):
                    comp_it_dict[i]={}
                    comp_it_dict[i]['iterator']=iterators[i]
                    comp_it_dict[i]['lower_bound']=self.annotations['iterators'][iterators[i]]['lower_bound']
                    comp_it_dict[i]['upper_bound']=self.annotations['iterators'][iterators[i]]['upper_bound']

                self.it_dict[comp]=comp_it_dict
            print(self.it_dict)

            iterators=list(self.annotations["iterators"].keys())

            for i in range(len(iterators)):
           
                loop_repr=[]
                loop_repr.append(self.annotations['iterators'][iterators[i]]['lower_bound'])
                loop_repr.append(self.annotations['iterators'][iterators[i]]['upper_bound'])
                loop_repr.extend([0,0,0,0,0,0,0,0,0,0,0])
                loop_log_rep = list(np.log1p(loop_repr))
                loop_repr.extend(loop_log_rep)
                self.obs["loops_representation"]=np.vstack([self.obs["loops_representation"],np.array([loop_repr])])

                childs_indexes=[iterators.index(child) for child in self.annotations['iterators'][iterators[i]]['child_iterators']]
                if len(childs_indexes)!=11:
                    for j in range(11-len(childs_indexes)):
                        childs_indexes.append(-1)
                self.obs["child_list"]=np.vstack([self.obs["child_list"], np.array([childs_indexes])])
                
                if self.annotations['iterators'][iterators[i]]['computations_list']!=[]:
                    self.obs['has_comps']=np.append(self.obs['has_comps'],1)
                else:
                    self.obs['has_comps']=np.append(self.obs['has_comps'],0)

                computations_list=list(self.annotations['computations'].keys())
                loop_comps=[computations_list.index(comp) for comp in self.annotations['iterators'][iterators[i]]['computations_list']]
                if len(loop_comps)!=5:
                    for j in range(5-len(loop_comps)):
                        loop_comps.append(-1)
                self.obs["computations_indices"]=np.vstack([self.obs["computations_indices"],np.array([loop_comps])])
            

            #Add null vectors if needed to avoid mismatching error of env.observation's type and reset_obs's type              
            for i in range(15-len(self.annotations["iterators"])):
                loop_repr=np.full(26,-1)
                self.obs["loops_representation"]=np.vstack([self.obs["loops_representation"],loop_repr])
            
            for i in range(12-len(self.annotations["iterators"])):
                self.obs["child_list"]=np.vstack([self.obs["child_list"], np.full(11,-1)])
                self.obs['has_comps']=np.append(self.obs['has_comps'],0)
                self.obs["computations_indices"]=np.vstack([self.obs["computations_indices"],np.full(5,-1)])

            
            if len(self.common_it) == 5:
                self.obs["action_mask"] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
            else:
                if len(self.common_it) == 4:
                    self.obs["action_mask"] = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
                else: 
                    if len(self.common_it) == 3:
                        self.obs["action_mask"] = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
                    else: 
                        if len(self.common_it) == 2:
                            self.obs["action_mask"] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
                        else:
                            if len(self.common_it) == 1:
                                self.obs["action_mask"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        
            if len(self.comps)==1:
                np.put(self.obs["action_mask"],[56,57,58,59,60],[0, 0, 0, 0, 0])   
        
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

            print("the rep in reset is",self.obs["representation"])

            return self.obs

    def step(self, raw_action):
        print("in step function")
        action_name=self.ACTIONS_ARRAY[raw_action]
        print("\nL'action {} est choisie".format(action_name))
        print("the curr schedule is: ",len(self.schedule),self.schedule_str)
        exit=False
        done=False
        info={}
        applied_exception=False
        skew_params_exception=False
        skew_unroll=False
        reward=0
        self.steps+=1
        first_comp=self.comps[0]

        try:
            action = Action(raw_action, self.it_dict, self.common_it)
            print("after creating the action")
            self.obs = copy.deepcopy(self.obs) # get current observation
            try:
                if not action.id in range(44,46):
                    action_params = action.parameter()
                    print("action params first are", action_params)
                else:
                    comp=list(self.it_dict.keys())[0]
                    action_params=action.parameter(comp, self.prog)
            except:
                print("\nOoops... Il n'existe pas de facteur convenable.")
                info = {"parameter exception": True}
                done = False
                #return self.obs, reward, done, info
                return self.obs, reward, done, info

            #print("\n The chosen action is: ", action.id)
                
            if action.id in range(28):
                
                if not self.is_interchaged:
                   
                    params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]

                    optim1 = optimization_command("Interchange", params, self.comps)
                    print("got the optim cmd")
                    self.schedule.append(optim1)
           
                    
                    if self.is_unrolled:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                    else:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                    
                    print("\n in interchange,  lc res: {}".format(lc_check))
                                


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
                    
                    self.apply_interchange(action_params)
                    print("interchange applied")
                    self.is_interchaged=True
                    
                else:
                    print("interchange already applied execption")
                    applied_exception=True
                    raise IsInterchangedException
                    #to expierment with the reward in this case
                    
            if action.id in range(28,41):
                if not self.is_tiled:
                    params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]
                    params.append(action_params["first_factor"])
                    params.append(action_params["second_factor"])
                    
                    if action_params["tiling_depth"] == 3:
                        params.insert(2, action_params["third_dim_index"])
                        params.append(action_params["third_factor"])

                    
                    optim2 = optimization_command("Tiling", params, self.comps)

                    self.schedule.append(optim2)


                    if self.is_unrolled:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                    else:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)

                    
                    print("\n in tiling,  lc res: {}".format(lc_check))
                    
                    
                    if lc_check == -1:   
                        print("\nCette action a généré une erreur")
                        raise LCException

                    if lc_check == 0:
                        print("\nCette action est illégale")
                        self.schedule.pop()
                        info = {"illegal_action": True}
                        done = False
                        return self.obs, reward, done, info

                    self.apply_tiling(action_params)
                    print("\n tiling applied")

                    self.is_tiled=True
                else:
                    print("\n tiling already applied execption")
                    applied_exception=True
                    raise IsTiledException

            if action.id in range(41,44):
                params = {}
                if not self.is_unrolled:
                    # print("action params of unrolling", action_params["dim_index"])
                    # print("action params of unrolling", action_params["unrolling_factor"])

                    #we don't apply unrolling on a level that's skewed, we get the tag to see if it's skewed or not
                    self.non_skewed_comps = []
                    for comp in self.comps:
                        it_skewed="L"+self.it_dict[comp][action_params[comp]["dim_index"]]["iterator"]+"Skewed"
                        if self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][it_skewed]]!=1:
                            self.non_skewed_comps.append(comp)
                    
                    #for mult comps, unrolling returns a dict of parameters, each for each comp
                    for comp in self.non_skewed_comps:
                        params[comp]=[int(action_params[comp]["dim_index"]), int(action_params[comp]["unrolling_factor"])]
                    print("\nLes paramètres sont:",params)

                    if self.non_skewed_comps != []:
                        print("it's not skewed")

                        optim3 = optimization_command( "Unrolling", params, self.non_skewed_comps)
                        
                        self.schedule.append(optim3)

                        start_time = time.time()
                        
                        lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                        l_time = time.time() - start_time
                        print("\n unrollling lc check {} ".format(lc_check))
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

                        #self.apply_unrolling(action_params)
                        print("\n unrolling applied")
                        for i in range(41,44):
                            self.obs["action_mask"][i]=0
                        self.is_unrolled=True
                    else:
                        #reward=-1
                        lc_check=0
                        info['error']="trying to apply unrolling after skewing in one of the computations"
                    
                else:
                    applied_exception=True
                    print("\n unrolling is already applied")

                    raise IsUnrolledException

            if action.id in range(44,46):

                if not self.is_skewed:

                
                    if (action_params["first_factor"] != None and action_params["second_factor"] != None):
                        
                        print("\nLes paramètres sont:")
                        print("\nLe premier niveau de boucle:", action_params["first_dim_index"])
                        print("\nLe deuxième niveau de boucle:", action_params["second_dim_index"])
                        print("\nLe premier facteur:", action_params["first_factor"])
                        print("\nLe deuxième facteur:", action_params["second_factor"])
                        non_inner_comps = []
                        for comp in self.comps:
                            if (action_params["first_dim_index"] != len(self.it_dict[comp])-1 and action_params["second_dim_index"] != len(self.it_dict[comp])-1) or ( (action_params["first_dim_index"] == len(self.it_dict[comp])-1 or action_params["second_dim_index"] == len(self.it_dict[comp])-1 and not self.is_unrolled )) :
                                non_inner_comps.append(comp)


                        if non_inner_comps != []:

                            params=[int(action_params["first_dim_index"]), int(action_params["second_dim_index"])]
                            params.append(action_params["first_factor"])
                            params.append(action_params["second_factor"])

                            optim4 = optimization_command("Skewing", params, non_inner_comps)

                            self.schedule.append(optim4)

                            start_time = time.time()
                            if self.is_unrolled:
                                lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                            else:
                                lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                            l_time = time.time() - start_time
                            print("\n skewing lc check res {} ".format(lc_check))
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

                            self.apply_skewing(action_params)
                            print("\n skewing is applied")
                            self.is_skewed=True

                        else:
                            skew_unroll=True
                            raise SkewUnrollException

                    else:
                        print("\n skewing prams are null")
                        skew_params_exception=True
                        raise SkewParamsException

                
                
                else:
                    print("\n sekwing is already applied")
                    applied_exception=True
                    raise IsSkewedException

            if action.id in range(46,48):
                if not self.is_parallelized:
                    print("\nLes paramètres sont:")
                    print("\nLe niveau de boucle:", action_params["dim_index"])

                    params=[int(action_params["dim_index"])]

                    optim5 = optimization_command("Parallelization", params, self.comps)
                
                    self.schedule.append(optim5)

                    start_time = time.time()
                    if self.is_unrolled:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                    else:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                    
                    l_time = time.time() - start_time
                    print("\n parallelzation lc check {}".format(lc_check))
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

                    self.apply_parallelization(action_params)
                    print("\n parallelisation applied")
                    self.is_parallelized=True
                else:
                    applied_exception=True
                    print("\n parallelisation is already applied")
                    raise IsParallelizedException

            if action.id in range(48,56):
                
                if not self.is_reversed:
                    print("\nLes paramètres sont:")
                    print("\nLe niveau de boucle:", action_params["dim_index"])

                    params=[int(action_params["dim_index"])]

                    optim6 = optimization_command( "Reversal", params, self.comps)

                    self.schedule.append(optim6)
                    
                    start_time=time.time()
                    if self.is_unrolled:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                    else:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                    l_time = time.time() - start_time
                    print("loop reversal lc check {}".format(lc_check))
                    self.lc_total_time+=l_time

                    if lc_check == -1: 
                        print("\nCette action a généré une erreur")
                        self.obs["action_mask"][action.id]=0
                        raise LCException

                    if lc_check == 0:
                        print("\nCette action est illégale")
                        self.schedule.pop()
                        #self.obs["action_mask"][action.id]=0
                        #reward = -1
                        info = {"illegal_action": True}
                        done = False
                        return self.obs, reward, done, info

                    self.apply_reversal(action_params)
                    print("\n loop reversal applied")
                    self.is_reversed=True
                else:
                    applied_exception=True

                    print("\n loop reversal already applied")

                    raise IsReversedException

            if action.id in range(56,61):
                params=[int(action_params["dim_index"]), action_params["fuse_comps"]]

                print("fuse params are", action_params["dim_index"], '\n', action_params["fuse_comps"])

                if action_params["fuse_comps"] != [] and len(action_params["fuse_comps"])!=1:

                    optim7 = optimization_command( "Fusion", params, action_params["fuse_comps"])

                    print("fusion optim created")

                    self.schedule.append(optim7)
                    
                    start_time=time.time()

                    if self.is_unrolled:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule, self.non_skewed_comps,first_comp)
                    else:
                        lc_check = self.prog.check_legality_of_schedule(self.schedule,first_comp=first_comp)
                        
                    l_time = time.time() - start_time
                    print("loop fusion lc check {}".format(lc_check))
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

                    self.apply_fusion(action_params)
                    print("\n loop fusion applied")
                    self.is_fused=True
                else:
                    lc_check=0
                    print("unable to fuse")
                    #reward=-1

     
            if action.id==self.EXIT:
                print("\n **** it's an exit action ****")
                done=True
                exit=True
                
            if (not exit and lc_check!=0) and not (action.id in range(41,44) and self.is_skewed):
                print("in the long cond after actions")
                self.schedule_str = sched_str(self.schedule_str, action.id, action_params, self.comp_indic_dict)
                # print("the original iterators were:", self.it_dict)
                if not action.id in range(41,44):
                    self.it_dict=update_iterators(action.id, self.it_dict, action_params, self.added_iterators, self.comp_indic_dict)
                    print("after update iterators with the schedule", self.schedule_str, "it is", self.it_dict)

                self.depth += 1
            

        except Exception as e:
            print(e.__class__.__name__)
            if applied_exception:
                #reward = -1 
                print("applied exception")
                info = {"more than one time": True}
                done = False
                return self.obs, reward, done, info

            else:
                ex_type, ex_value, ex_traceback = sys.exc_info()
                if self.schedule != [] and not skew_params_exception and not skew_unroll:
                    self.schedule.pop()
                #reward = -1
                print("\nCette action a généré une erreur, elle ne sera pas appliquée.")
                print("else exception", ex_type.__name__, ex_value, ex_traceback)
                done = False
                info = {
                    "depth": self.depth,
                    "error": "ended with error in the step function",
                }

                return self.obs, reward, done, info
        finally:


            if(self.depth == self.MAX_DEPTH) or (self.steps >=20):
                done=True

            # print("--- done is ----", done)

            if done: 
                print("\nFin de l'épisode")
                if self.is_unrolled:       
                    for optim in self.schedule:
                        print(optim.type)
                        if optim.type == "Unrolling":
                            unroll_optimisation=optim

                    new_unrolling_params={}
                    new_unrolling_optim_params={}
                    for comp in self.non_skewed_comps:
                        unroll_factor=unroll_optimisation.params_list[comp][1]
                        new_unrolling_params[comp]={"dim_index":len(self.it_dict[comp])-1,"unrolling_factor":unroll_factor}
                        new_unrolling_optim_params[comp]=[len(self.it_dict[comp])-1, unroll_factor]
                        
                    new_unrolling_optim=optimization_command("Unrolling", new_unrolling_optim_params, self.non_skewed_comps)
                    
                    new_unrolling_str=""
                    unrolling_str=""

                    for comp in self.non_skewed_comps: 
                        unroll_factor=unroll_optimisation.params_list[comp][1]
                        # print("comp", comp)  
                        # print("unroll_factor", unroll_factor)
                        new_unrolling_str+="U(L"+str(len(self.it_dict[comp])-1)+","+str(unroll_factor)+",C"+str(self.comp_indic_dict[comp]) +")"
                        #print("new_unrolling_str",new_unrolling_str)
                        unrolling_str+="U(L"+str(unroll_optimisation.params_list[comp][0])+","+str(unroll_factor)+",C"+str(self.comp_indic_dict[comp]) +")" 
                        #print("unrolling_str", unrolling_str)

                    self.schedule_str=self.schedule_str.replace(unrolling_str, "") + new_unrolling_str
                    
                    self.schedule.remove(unroll_optimisation)      
                    self.schedule.append(new_unrolling_optim)
                    self.apply_unrolling(new_unrolling_params)
                    #no need to update the iterators list because it's the end of the episode



                self.search_time= time.time()-self.search_time
                
                try:
                    exec_time=0
                    writing_time=0
                    exec_time = self.get_exec_time()

                    if not self.is_parallelized:
                        #print("inside parallelization in done")
                        print("Tester si la parallélisation apporte un meilleur speedup...")
                        action = Action(self.PARALLELIZATION0, self.it_dict, self.common_it)
                        action_params = action.parameter()

                        params=[int(action_params["dim_index"])]

                        optim5 = optimization_command("Parallelization", params, self.comps)
                    
                        self.schedule.append(optim5)

            
                        try:

                            self.schedule_str = sched_str(self.schedule_str, action.id, action_params, self.comp_indic_dict)
                            parallelized_exec_time=self.get_exec_time()
                            parallelization_str='P(L'+str(action_params["dim_index"])+')'
                            print("exec time with parallelization: ", parallelized_exec_time)
                            
                            # print("the exec time with parallelization is", parallelized_exec_time)
                            # print("the exec time without parallelization is", exec_time)
                        except:
                            print("\nCette action est illégale")
                            self.schedule.remove(optim5)
                            self.schedule_str=self.schedule_str.replace(parallelization_str, "")
                            
                        
                        if parallelized_exec_time < exec_time and parallelized_exec_time!=0:
                            exec_time = parallelized_exec_time
                            
                            self.apply_parallelization(action_params)
                            print("La parallélisation améliore le temps d'exécution donc elle est appliquée.")

                        else:
                            self.schedule.remove(optim5)
                            self.new_scheds[self.prog.name].pop(self.schedule_str)
                            self.schedule_str=self.schedule_str.replace(parallelization_str, "")
                            print("La parallélisation n'améliore pas le temps d'exécution, alors elle n'est pas appliquée.")
                    

                except:
                    print("\nErreur lors de la mesure du temps d'exécution.") 
                    info = {"Internal execution error": True}
                    #reward=-1

                    print("error with get execution time, going out")
                    return self.obs, reward, done, info

                if exec_time!=0:
                    print("\nLe schedule final trouvé est: ",self.schedule_str)
                    print("The new execution time is ", exec_time)
                    #self.speedup = (self.initial_execution_time - exec_time)/self.initial_execution_time
                    if self.initial_execution_time >=  exec_time:
                        
                        self.speedup = (self.initial_execution_time / exec_time)
                    else:
                        self.speedup = -(exec_time / self.initial_execution_time )
                    
                    print("the speedup is: ", self.speedup)
                    reward=self.speedup
                    print('the new scheds are', self.new_scheds)
                    start_time=time.time()
                    try:
                        self.save_sched_to_dataset()
                        self.write_data()
                        writing_time=time.time()-start_time
                        print("Data saved in ",writing_time)
                    except:
                        print("failed to save schedule")

                self.episode_total_time= time.time()-self.episode_total_time
                print("CODE GEN :",self.codegen_total_time)
                print("LC : ",self.lc_total_time)
                print("\nEPISODE TOTAL TIME : {}\nLEGALITY CHECK TIME RATIO : {}\nCODE GENERATION TIME RATIO : {}\nWRITING TIME RATIO : {}\n".format(self.episode_total_time, self.lc_total_time/self.episode_total_time, self.codegen_total_time/self.episode_total_time, writing_time/self.episode_total_time))

            info["depth"] =  self.depth

            
            print("the reward is",reward)
            print("the rep out of the step fct is",self.obs["representation"])
            return self.obs, reward, done, info

        


    def apply_interchange(self, action_params):
        for comp in self.comps:
            l_code = "L" + self.it_dict[comp][action_params["first_dim_index"]]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Interchanged"]] = 1
            l_code = "L" + self.it_dict[comp][action_params["second_dim_index"]]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Interchanged"]] = 1

        #Update the loops representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[comp][action_params["first_dim_index"]]['iterator'] in iterators:
            loop_1=iterators.index(self.it_dict[comp][action_params["first_dim_index"]]['iterator'])
        elif self.it_dict[comp][action_params["first_dim_index"]]['iterator'] in self.added_iterators:
            loop_1=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params["first_dim_index"]]['iterator'])  
        self.obs["loops_representation"][loop_1][2]=1
        
        if self.it_dict[comp][action_params["second_dim_index"]]['iterator'] in iterators:
            loop_2=iterators.index(self.it_dict[comp][action_params["second_dim_index"]]['iterator'])
        elif self.it_dict[comp][action_params["second_dim_index"]]['iterator'] in self.added_iterators:
            loop_2=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params["second_dim_index"]]['iterator'])  
        self.obs["loops_representation"][loop_2][2]=1

        for i in range(28):
            self.obs["action_mask"][i]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0



    def apply_tiling(self, action_params):

        for comp in self.comps:
            comp_index=self.comp_indic_dict[comp]
       
            first_dim_index=action_params["first_dim_index"]
            second_dim_index=action_params["second_dim_index"]

            l_code = "L" + self.it_dict[comp][first_dim_index]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Tiled"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "TileFactor"]] = action_params[
                "first_factor"
            ]

            #update the loop bounds if tiling is applied on loop 1
            if action_params["tiling_loop_1"]:
                print("inside loop tiling 1")
                new_upper_bound_1=self.obs["representation"][self.comp_indic_dict[comp]][first_dim_index*20+1]/action_params["first_factor"]
                self.obs["representation"][self.comp_indic_dict[comp]][first_dim_index*20+1]=new_upper_bound_1
                new_inner_upper_bound_1=action_params["first_factor"]
                self.obs["representation"][self.comp_indic_dict[comp]][first_dim_index*20+10]=new_inner_upper_bound_1
                print("after loop tiling 1")
                #Add the loop representation of the newly added iterator
                loop_added="{}_1".format(self.it_dict[comp][first_dim_index]['iterator'])
                self.added_iterators.append(loop_added)
                loop_index=len(self.annotations['iterators']) + self.added_iterators.index(loop_added)
                #Initialize lower and upper bounds
                loop_repr=[]
                if self.obs["representation"][comp_index][self.placeholders[comp][l_code + "Reversed"]]==1:
                    lower_bound=self.obs["representation"][comp_index][second_dim_index*20+1]
                else:
                    lower_bound=self.obs["representation"][comp_index][second_dim_index*20]                
                loop_repr.extend([lower_bound, action_params["first_factor"]])
                #Initialize the different tags
                loop_repr.extend([0,0,0,0,0,0,0,0,0,0,0])
                loop_log_rep = list(np.log1p(loop_repr))
                loop_repr.extend(loop_log_rep)
                self.obs["loops_representation"][loop_index]=loop_repr

            l_code = "L" + self.it_dict[comp][second_dim_index]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Tiled"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "TileFactor"]] = action_params[
                "second_factor"
            ]
            #update the loop bounds if tiling is applied on loop 2
            if action_params["tiling_loop_2"]:
                print("inside loop tiling 2")
                new_upper_bound_2=self.obs["representation"][self.comp_indic_dict[comp]][second_dim_index*20+1]/action_params["second_factor"]
                self.obs["representation"][self.comp_indic_dict[comp]][second_dim_index*20+1]=new_upper_bound_2
                new_inner_upper_bound_2=action_params["second_factor"]
                self.obs["representation"][self.comp_indic_dict[comp]][second_dim_index*20+10]=new_inner_upper_bound_2
                print("after loop tiling 2")

                #Add the loop representation of the newly added iterator
                loop_added="{}_1".format(self.it_dict[comp][second_dim_index]['iterator'])
                self.added_iterators.append(loop_added)
                loop_index=len(self.annotations['iterators']) + self.added_iterators.index(loop_added)
                #Initialize lower and upper bounds
                loop_repr=[]

                if self.obs["representation"][comp_index][self.placeholders[comp][l_code + "Reversed"]]==1:
                    lower_bound=self.obs["representation"][comp_index][second_dim_index*20+1]
                else:
                    lower_bound=self.obs["representation"][comp_index][second_dim_index*20]
                loop_repr.extend([lower_bound, action_params["second_factor"]])

                #Initialize the different tags
                loop_repr.extend([0,0,0,0,0,0,0,0,0,0,0])
                loop_log_rep = list(np.log1p(loop_repr))
                loop_repr.extend(loop_log_rep)
                self.obs["loops_representation"][loop_index]=loop_repr

            if action_params["tiling_depth"] == 3:
                third_dim_index=action_params["third_dim_index"]
                l_code = "L" + self.it_dict[comp][third_dim_index]['iterator']
                self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Tiled"]] = 1
                self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "TileFactor"]] = action_params[
                    "third_factor"
                ]
                #update the loop bounds if tiling is applied on loop 3
                if action_params["tiling_loop_3"]:
                    print("inside loop tiling 3")
                    new_upper_bound_3=self.obs["representation"][self.comp_indic_dict[comp]][third_dim_index*20+1]/action_params["third_factor"]
                    self.obs["representation"][self.comp_indic_dict[comp]][third_dim_index*20+1]=new_upper_bound_3
                    new_inner_upper_bound_3=action_params["third_factor"]
                    self.obs["representation"][self.comp_indic_dict[comp]][third_dim_index*20+10]=new_inner_upper_bound_3
                    print("after loop tiling 3")

                    #Add the loop representation of the newly added iterator
                    loop_added="{}_1".format(self.it_dict[comp][third_dim_index]['iterator'])
                    self.added_iterators.append(loop_added)
                    loop_index=len(self.annotations['iterators']) + self.added_iterators.index(loop_added)
                    #Initialize lower and upper bounds
                    loop_repr=[]
                    if self.obs["representation"][comp_index][self.placeholders[comp][l_code + "Reversed"]]==1:
                        lower_bound=self.obs["representation"][comp_index][third_dim_index*20+1]
                    else:
                        lower_bound=self.obs["representation"][comp_index][third_dim_index*20]

                    loop_repr.extend([lower_bound,action_params["third_factor"]])
                    #Initialize the different tags
                    loop_repr.extend([0,0,0,0,0,0,0,0,0,0,0])
                    loop_log_rep = list(np.log1p(loop_repr))
                    loop_repr.extend(loop_log_rep)
                    self.obs["loops_representation"][loop_index]=loop_repr

        #Update the loops representation
        iterators=list(self.annotations["iterators"].keys())

        if self.it_dict[comp][first_dim_index]['iterator'] in iterators:
            loop_1=iterators.index(self.it_dict[comp][first_dim_index]['iterator'])
        elif self.it_dict[comp][first_dim_index]['iterator'] in self.added_iterators:
            loop_1=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][first_dim_index]['iterator'])

        self.obs["loops_representation"][loop_1][3]=1
        self.obs["loops_representation"][loop_1][4]=action_params['first_factor']

        if self.it_dict[comp][second_dim_index]['iterator'] in iterators:
            loop_2=iterators.index(self.it_dict[comp][second_dim_index]['iterator'])
        elif self.it_dict[comp][second_dim_index]['iterator'] in self.added_iterators:
            loop_2=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][second_dim_index]['iterator'])  

        self.obs["loops_representation"][loop_2][3]=1
        self.obs["loops_representation"][loop_2][4]=action_params['second_factor']

        #Update the loop representation
        if action_params["tiling_depth"] == 3:

            if self.it_dict[comp][third_dim_index]['iterator'] in iterators:
                loop_3=iterators.index(self.it_dict[comp][third_dim_index]['iterator'])
            elif self.it_dict[comp][third_dim_index]['iterator'] in self.added_iterators:
                loop_3=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][third_dim_index]['iterator'])  

            self.obs["loops_representation"][loop_3][3]=1
            self.obs["loops_representation"][loop_3][4]=action_params['third_factor']
            
            
            if self.is_interchaged == False:

                if len(self.common_it) == 5:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE05, self.INTERCHANGE06, self.INTERCHANGE07, self.INTERCHANGE15, self.INTERCHANGE16, self.INTERCHANGE17, 
                        self.INTERCHANGE25, self.INTERCHANGE26, self.INTERCHANGE27, self.INTERCHANGE35, self.INTERCHANGE36, self.INTERCHANGE37, 
                        self.INTERCHANGE45, self.INTERCHANGE46, self.INTERCHANGE47,self.INTERCHANGE56,self.INTERCHANGE57, self.INTERCHANGE67]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE05, self.INTERCHANGE06, self.INTERCHANGE15, self.INTERCHANGE16, self.INTERCHANGE25, self.INTERCHANGE26, 
                        self.INTERCHANGE35, self.INTERCHANGE36, self.INTERCHANGE45, self.INTERCHANGE46, self.INTERCHANGE56]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[self.INTERCHANGE05, self.INTERCHANGE15, self.INTERCHANGE25,self.INTERCHANGE35, self.INTERCHANGE45]]=1

                if len(self.common_it) == 4:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE04, self.INTERCHANGE05, self.INTERCHANGE06, self.INTERCHANGE14, self.INTERCHANGE15, self.INTERCHANGE16, 
                        self.INTERCHANGE24, self.INTERCHANGE25, self.INTERCHANGE26, self.INTERCHANGE34, self.INTERCHANGE35, self.INTERCHANGE36, 
                        self.INTERCHANGE45, self.INTERCHANGE46, self.INTERCHANGE56]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE04, self.INTERCHANGE05, self.INTERCHANGE14, self.INTERCHANGE15,
                        self.INTERCHANGE24, self.INTERCHANGE25, self.INTERCHANGE34, self.INTERCHANGE35, self.INTERCHANGE45]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[self.INTERCHANGE04, self.INTERCHANGE14, self.INTERCHANGE24, self.INTERCHANGE34]]=1    

                if len(self.common_it) == 3:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE03, self.INTERCHANGE04, self.INTERCHANGE05, self.INTERCHANGE13, self.INTERCHANGE14, self.INTERCHANGE15, 
                        self.INTERCHANGE23, self.INTERCHANGE24, self.INTERCHANGE25, self.INTERCHANGE34, self.INTERCHANGE35, 
                        self.INTERCHANGE45]]=1    
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE03, self.INTERCHANGE04, self.INTERCHANGE13, self.INTERCHANGE14,
                        self.INTERCHANGE23, self.INTERCHANGE24, self.INTERCHANGE34]]=1 
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[self.INTERCHANGE03, self.INTERCHANGE13, self.INTERCHANGE23]]=1 
                
                if len(self.common_it) == 2:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE02, self.INTERCHANGE03, self.INTERCHANGE04, self.INTERCHANGE12, self.INTERCHANGE13, self.INTERCHANGE14, 
                        self.INTERCHANGE23, self.INTERCHANGE24, self.INTERCHANGE34]]=1    
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE02, self.INTERCHANGE03, self.INTERCHANGE12, self.INTERCHANGE13, self.INTERCHANGE23]]=1 
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[self.INTERCHANGE02, self.INTERCHANGE12]]=1 

                if len(self.common_it) == 1:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE01, self.INTERCHANGE02, self.INTERCHANGE03, self.INTERCHANGE12, self.INTERCHANGE13, self.INTERCHANGE23]]=1    
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.INTERCHANGE01, self.INTERCHANGE02, self.INTERCHANGE12, self.INTERCHANGE13]]=1    
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][[self.INTERCHANGE01]]=1  

            if self.is_reversed == False:
                if len(self.common_it) == 5:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL5,self.REVERSAL6, self.REVERSAL7]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL5,self.REVERSAL6]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][self.REVERSAL5]=1

                elif len(self.common_it) == 4:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL4,self.REVERSAL5, self.REVERSAL6]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL4,self.REVERSAL5]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][self.REVERSAL4]=1

                elif len(self.common_it) == 3:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL3,self.REVERSAL4, self.REVERSAL5]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL3,self.REVERSAL4]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][self.REVERSAL3]=1

                elif len(self.common_it) == 2:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL2,self.REVERSAL3, self.REVERSAL4]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL2,self.REVERSAL3]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][self.REVERSAL2]=1

                elif len(self.common_it) == 1:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL1,self.REVERSAL2, self.REVERSAL3]]=1
                    elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"] or action_params["tiling_loop_2"] and action_params["tiling_loop_3"] or action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                        self.obs["action_mask"][[self.REVERSAL1,self.REVERSAL2]]=1
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"] or action_params["tiling_loop_3"] :
                        self.obs["action_mask"][self.REVERSAL1]=1
        
        for i in range(28,41):
            self.obs["action_mask"][i]=0

        for i in range(56,61):
            self.obs["action_mask"][i]=0

            

    def apply_unrolling(self, action_params):

        for comp in self.comps:

            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp]["Unrolled"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp]["UnrollFactor"]] = action_params[comp]["unrolling_factor"]

            l_code = "L" + self.it_dict[comp][action_params[comp]["dim_index"]]['iterator']
            index_upper_bound=self.placeholders[comp][l_code+'Interchanged']-1
            self.obs["representation"][self.comp_indic_dict[comp]][index_upper_bound]=self.obs["representation"][self.comp_indic_dict[comp]][index_upper_bound]/action_params[comp]["unrolling_factor"]

            #Update the loop representation
            iterators=list(self.annotations["iterators"].keys())
            if self.it_dict[comp][action_params[comp]["dim_index"]]['iterator'] in iterators:
                loop_index=iterators.index(self.it_dict[comp][action_params[comp]["dim_index"]]['iterator'])
            elif self.it_dict[comp][action_params[comp]["dim_index"]]['iterator'] in self.added_iterators:
                loop_index=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params[comp]["dim_index"]]['iterator'])           
            self.obs["loops_representation"][loop_index][5]=1
            self.obs["loops_representation"][loop_index][6]=action_params[comp]['unrolling_factor']

        for i in range(41,44):
            self.obs["action_mask"][i]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0

    def apply_skewing(self, action_params):

        dim_1=action_params["first_dim_index"]
        dim_2=action_params["second_dim_index"]

        for comp in self.comps:

            l1_code = "L" + self.it_dict[comp][dim_1]['iterator']
            l2_code = "L" + self.it_dict[comp][dim_2]['iterator']

            #to get the start of the iterator in the representation template (just after the bounds)
            index1_upper_bound=self.placeholders[comp][l1_code+'Interchanged']-1
            index1_lower_bound=self.placeholders[comp][l1_code+'Interchanged']-2
            index2_upper_bound=self.placeholders[comp][l2_code+'Interchanged']-1
            index2_lower_bound=self.placeholders[comp][l2_code+'Interchanged']-2

            l1_lower_bound=self.obs["representation"][self.comp_indic_dict[comp]][index1_lower_bound]
            l1_upper_bound=self.obs["representation"][self.comp_indic_dict[comp]][index1_upper_bound]
            l2_lower_bound=self.obs["representation"][self.comp_indic_dict[comp]][index2_lower_bound]
            l2_upper_bound=self.obs["representation"][self.comp_indic_dict[comp]][index2_upper_bound]

            l1_extent = l1_upper_bound - l1_lower_bound
            l2_extent = l2_upper_bound - l2_lower_bound

            skew_factor = action_params["first_factor"]
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l1_code + "Skewed"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l1_code + "SkewFactor"]] = skew_factor
            self.obs["representation"][self.comp_indic_dict[comp]][index1_lower_bound]= abs(action_params["first_factor"]) * l1_lower_bound
            self.obs["representation"][self.comp_indic_dict[comp]][index1_upper_bound]= l1_lower_bound + abs(action_params["first_factor"]) * l1_extent + abs(action_params["second_factor"]) * l2_extent

            skew_factor = action_params["second_factor"]
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l2_code + "Skewed"]] = 1
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l2_code + "SkewFactor"]] = skew_factor
            self.obs["representation"][self.comp_indic_dict[comp]][index2_lower_bound]= 0
            self.obs["representation"][self.comp_indic_dict[comp]][index2_upper_bound]=(l2_extent) + 1

        #Update the loop representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[comp][dim_1]['iterator'] in iterators:
            loop_1=iterators.index(self.it_dict[comp][dim_1]['iterator'])
        elif self.it_dict[comp][dim_1]['iterator'] in self.added_iterators:
            loop_1=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][dim_1]['iterator'])        
        self.obs["loops_representation"][loop_1][7]=1
        self.obs["loops_representation"][loop_1][8]=action_params['first_factor']
        #Skewing is applied on common loop levels so loop bounds are equal for all computations
        self.obs["loops_representation"][loop_1][9]=self.obs["representation"][0][index1_upper_bound]-self.obs["representation"][0][index1_lower_bound]

        if self.it_dict[comp][dim_2]['iterator'] in iterators:
            loop_2=iterators.index(self.it_dict[comp][dim_2]['iterator'])
        elif self.it_dict[comp][dim_2]['iterator'] in self.added_iterators:
            loop_2=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][dim_2]['iterator']) 
        self.obs["loops_representation"][loop_2][7]=1
        self.obs["loops_representation"][loop_2][8]=action_params['second_factor']
        self.obs["loops_representation"][loop_2][9]=self.obs["representation"][0][index2_upper_bound]-self.obs["representation"][0][index2_lower_bound]

        self.obs["action_mask"][44]=0
        self.obs["action_mask"][45]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0

    def apply_parallelization(self, action_params):
        first_comp=list(self.it_dict.keys())[0]
        l_code = "L" + self.it_dict[first_comp][action_params["dim_index"]]['iterator']

        self.obs["representation"][0][self.placeholders[first_comp][l_code + "Parallelized"]] = 1

        #Update the loop representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[first_comp][action_params["dim_index"]]['iterator'] in iterators:
            loop_index=iterators.index(self.it_dict[first_comp][action_params["dim_index"]]['iterator'])
        elif self.it_dict[first_comp][action_params["dim_index"]]['iterator'] in self.added_iterators:
            loop_index=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[first_comp][action_params["dim_index"]]['iterator'])
        self.obs["loops_representation"][loop_index][10]=1
        #Update the action mask
        self.obs["action_mask"][46]=0
        self.obs["action_mask"][47]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0

    def apply_reversal(self, action_params):
        for comp in self.comps:
            l_code = "L" + self.it_dict[comp][action_params["dim_index"]]['iterator']

            index_upper_bound=self.placeholders[comp][l_code+'Interchanged']-1
            index_lower_bound=self.placeholders[comp][l_code+'Interchanged']-2

            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Reversed"]] = 1

            tmp=self.obs["representation"][self.comp_indic_dict[comp]][index_lower_bound]
            self.obs["representation"][self.comp_indic_dict[comp]][index_lower_bound]=self.obs["representation"][self.comp_indic_dict[comp]][index_upper_bound]
            self.obs["representation"][self.comp_indic_dict[comp]][index_upper_bound]=tmp 

        #Update the loop representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[comp][action_params["dim_index"]]['iterator'] in iterators:
            loop_index=iterators.index(self.it_dict[comp][action_params["dim_index"]]['iterator'])
        elif self.it_dict[comp][action_params["dim_index"]]['iterator'] in self.added_iterators:
            loop_index=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params["dim_index"]]['iterator'])        
        self.obs["loops_representation"][loop_index][11]=1

        for i in range(48,56):
            self.obs["action_mask"][i]=0
        for i in range(56,61):
            self.obs["action_mask"][i]=0
    
    def apply_fusion(self, action_params):
        for comp in action_params["fuse_comps"]:
            l_code = "L" + self.it_dict[comp][action_params["dim_index"]]['iterator']
            self.obs["representation"][self.comp_indic_dict[comp]][self.placeholders[comp][l_code + "Fused"]] = 1

        #Update the loop representation
        iterators=list(self.annotations["iterators"].keys())
        if self.it_dict[comp][action_params["dim_index"]]['iterator'] in iterators:
            loop_index=iterators.index(self.it_dict[comp][action_params["dim_index"]]['iterator'])
        elif self.it_dict[comp][action_params["dim_index"]]['iterator'] in self.added_iterators:
            loop_index=len(self.annotations['iterators'])+ self.added_iterators.index(self.it_dict[comp][action_params["dim_index"]]['iterator'])        
        self.obs["loops_representation"][loop_index][12]=1

        for i in range(56,61):
            self.obs["action_mask"][i]=0




    
    #add new_scheds to the original schedules list
    def save_sched_to_dataset(self):
        for func in self.new_scheds.keys():
            for schedule_str in self.new_scheds[func].keys():#schedule_str represents the key, for example: 'Interchange Unrolling Tiling', the value is a tuple(schedule,execution_time)

                schedule=self.new_scheds[func][schedule_str][0]#here we get the self.obs["schedule"] containing the omtimizations list
                exec_time=self.new_scheds[func][schedule_str][1]
                search_time=self.new_scheds[func][schedule_str][2]
                for comp in self.comps:

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
                            sched_dict[comp]["interchange_dims"]=[self.it_dict[comp][optim.params_list[0]]['iterator'], self.it_dict[comp][optim.params_list[1]]['iterator']]

                        elif optim.type == 'Skewing':
                            first_dim_index=self.it_dict[comp][optim.params_list[0]]['iterator']
                            second_dim_index= self.it_dict[comp][optim.params_list[1]]['iterator']
                            first_factor=optim.params_list[2]
                            second_factor=optim.params_list[3]

                            sched_dict[comp]["skewing"]["skewed_dims"]=[first_dim_index,second_dim_index]
                            sched_dict[comp]["skewing"]["skewing_factors"]=[first_factor,second_factor]

                        elif optim.type == 'Parallelization':
                            sched_dict[comp]["parallelized_dim"]=self.it_dict[comp][optim.params_list[0]]['iterator']

                        elif optim.type == 'Tiling':
                            #Tiling 2D
                            if len(optim.params_list)==4:
                                sched_dict[comp]["tiling"]["tiling_depth"]=2
                                sched_dict[comp]["tiling"]["tiling_dims"]=[self.it_dict[comp][optim.params_list[0]]['iterator'],self.it_dict[comp][optim.params_list[1]]['iterator']]
                                sched_dict[comp]["tiling"]["tiling_factors"]=[optim.params_list[2],optim.params_list[3]]

                            #Tiling 3D
                            elif len(optim.params_list)==6:
                                sched_dict[comp]["tiling"]["tiling_depth"]=3
                                sched_dict[comp]["tiling"]["tiling_dims"]=[self.it_dict[comp][optim.params_list[0]]['iterator'],self.it_dict[comp][optim.params_list[1]]['iterator'],self.it_dict[comp][optim.params_list[2]]['iterator']]
                                sched_dict[comp]["tiling"]["tiling_factors"]=[optim.params_list[3],optim.params_list[4],optim.params_list[5]]

                        elif optim.type == 'Unrolling':
                            sched_dict[comp]["unrolling_factor"]=optim.params_list[comp][1]

                        elif optim.type == 'Reversal':
                            sched_dict[comp]["reversed_dim"]=self.it_dict[comp][optim.params_list[0]]['iterator']
                        
                        elif optim.type == 'Fusion':
                            pass

                sched_dict[comp]["execution_times"].append(exec_time)
                if not "schedules_list" in self.progs_dict[func].keys():
                    self.progs_dict[func]["schedules_list"]=[sched_dict]
                else:
                    self.progs_dict[func]["schedules_list"].append(sched_dict)
            
       

    def write_data(self):
        print("in write data")
        with open(self.programs_file, 'w') as f:
            json.dump(self.progs_dict, f)
        print("done writing data")
        f.close()



    def get_exec_time(self):
        print("in get_exec_time")

        prog_name= self.prog.name
        execution_time=0
        if self.schedule_str != "" and self.schedule != []:
            if prog_name in self.scheds.keys():
                #print("Am in 1")

                if self.schedule_str in self.scheds[prog_name]:
                    #print("Am in 1.1")
                    print("Prog in sched: True, sched in scheds: True")
                    execution_time=self.scheds[prog_name][self.schedule_str][0]
                    print("**out of ** Prog in sched: True, sched in scheds: False")

                else:  
                    #print("Am in 1.2")
                    
                    if prog_name in self.new_scheds.keys() and self.schedule_str in self.new_scheds[prog_name].keys():
                        #print("Am in 1.2.1")
                        print("Prog in sched: True, sched in scheds: False, shced in new_scheds: True")
                        execution_time=self.new_scheds[prog_name][self.schedule_str][1]
                        print("**out of **Prog in sched: True, sched in scheds: False, shced in new_scheds: True")
                    else:
                        #print("Am in 1.2.2")
                        curr_sched=copy.deepcopy(self.schedule)
                        print("Prog in sched: True, sched in scheds: False, shced in new_scheds: False")
                        self.new_scheds[prog_name]={}
                        execution_time=self.prog.evaluate_schedule(self.schedule,'sched_eval',self.nb_executions, self.initial_execution_time)
                        self.new_scheds[prog_name][self.schedule_str]=(curr_sched,execution_time,0)
                        print("**out of **Prog in sched: True, sched in scheds: False, shced in new_scheds: False")

                    
            else:

                #print("Am in 2")
                if prog_name in self.new_scheds.keys():
                    #print("Am in 2.1")

                    if self.schedule_str in self.new_scheds[prog_name].keys():
                        #print("Am in 2.1.1")
                        print("Prog in sched: False, sched in scheds: False Prog in new sched: True, sched in new scheds: True")
                        execution_time=self.new_scheds[prog_name][self.schedule_str][1]
                        print("** out of** Prog in sched: False, sched in scheds: False Prog in new sched: True, sched in new scheds: True")
                        

                    else:
                        #print("Am in 2.1.2")
                        curr_sched=copy.deepcopy(self.schedule)
                        print("Prog in sched: False, sched in scheds: False Prog in new sched: True, sched in new scheds: False")
                        execution_time=self.prog.evaluate_schedule(self.schedule,'sched_eval',self.nb_executions, self.initial_execution_time)
                        self.new_scheds[prog_name][self.schedule_str]=(curr_sched,execution_time,0)
                        print("** out of** Prog in sched: False, sched in scheds: False Prog in new sched: True, sched in new scheds: False")
                        

                else:
                    #print("Am in 2.2")
                    curr_sched=copy.deepcopy(self.schedule)
                    print("Prog in sched: False, sched in scheds: False Prog in new sched: False")
                    self.new_scheds[prog_name]={}
                    start_time=time.time()
                    execution_time=self.prog.evaluate_schedule(self.schedule,'sched_eval',self.nb_executions, self.initial_execution_time)
                    sched_time=time.time()-start_time
                    self.codegen_total_time+=sched_time

                    self.new_scheds[prog_name][self.schedule_str]=(curr_sched,execution_time,0)
                    print("**out of **Prog in sched: True, sched in scheds: False, shced in new_scheds: False")

        else:
            execution_time=self.initial_execution_time
                    
        print("get_exec_time returned {} for the function {}".format(execution_time,self.prog.name))
        return execution_time