
class optimization_command():
    def __init__(self, optim_type, params_list, comps):
        print("init optim cmd")
        assert optim_type in ['Interchange', 'Skewing', 'Parallelization', 'Tiling', 'Unrolling', 'Reversal', 'Fusion'], 'Unknown transformation: '+ optim_type
        self.type = optim_type
        self.params_list = params_list
        self.comps = comps
        self.tiramisu_optim_str = self.get_tiramisu_optim_str()
        
        
    def get_tiramisu_optim_str(self):
        if self.type == 'Interchange':
            # format of params_list must be [firts_loop, second_loop]
            assert len(self.params_list)==2
            interchange_str = '.interchange('+','.join([str(p) for p in self.params_list])+');'
            optim_str = ""
            for comp in self.comps:
                optim_str += "\n\t {}".format(comp)+interchange_str
            return optim_str
        elif self.type == 'Skewing':
            # format of params_list must be [firts_loop, second_loop, first_factor, second_factor]
            assert len(self.params_list)==4
            skewing_str = '.skew('+','.join([str(p) for p in self.params_list])+');'
            optim_str = ""
            for comp in self.comps:
                optim_str += "\n\t {}".format(comp)+ skewing_str
            return optim_str
        elif self.type == 'Parallelization':
            # format of params_list must be [loop]
            assert len(self.params_list)==1
            return '\t'+self.comps[0]+'.tag_parallel_level('+str(self.params_list[0])+');'
        elif self.type == 'Tiling':
             # format of params_list must be [firts_loop, second_loop, first_factor, second_factor] in the case of tiling 2D
             # or [firts_loop, second_loop, third_loop, first_factor, second_factor, third_factor] in the case of tiling 3D
            assert len(self.params_list)==4 or len(self.params_list)==6
            print("in tiling, optim str")
            tiling_str = '.tile('+','.join([str(p) for p in self.params_list])+');'
            optim_str = ""
            for comp in self.comps:
                optim_str += "\n\t {}".format(comp)+ tiling_str
            return optim_str
        elif self.type == 'Unrolling': 
            optim_str = ""
            for comp in self.comps:
                unrolling_str = '.unroll('+','.join([str(p) for p in self.params_list[comp]])+');'
                optim_str += "\n\t {}".format(comp)+ unrolling_str
            return optim_str
        elif self.type == 'Reversal':
             # format of params_list must be [firts_loop, second_loop, first_factor, second_factor] in the case of tiling 2D
             # or [firts_loop, second_loop, third_loop, first_factor, second_factor, third_factor] in the case of tiling 3D
            assert len(self.params_list)==1 
            reversal_str = '.loop_reversal('+str(self.params_list[0])+');'
            optim_str = ""
            for comp in self.comps:
                optim_str += "\n\t {}".format(comp)+ reversal_str
            return optim_str
        elif self.type == 'Fusion':
            optim_str = ""
            prev_comp=self.comps[0]
            for comp in self.comps[1:]:
                optim_str += "\n\t {}".format(prev_comp)+ '.then('+ str(comp) + ',' + str(self.params_list[0])+');'
                prev_comp = comp
            return optim_str

