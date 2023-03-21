import re
import shutil
import sys, os, subprocess
import json
from pathlib import Path
from datetime import datetime
import random


class optimization_command():
    def __init__(self, comp_name, optim_type, params_list):
        self.comp_name = comp_name
        assert optim_type in ['Interchange', 'Skewing', 'Parallelization', 'Tiling', 'Unrolling', 'Reversal'], 'Unknown transformation: '+ optim_type
        self.type = optim_type
        self.params_list = params_list
        self.tiramisu_optim_str = self.get_tiramisu_optim_str()
        
    def get_tiramisu_optim_str(self):
        if self.type == 'Interchange':
            # format of params_list must be [firts_loop, second_loop]
            assert len(self.params_list)==2
            return '\t'+self.comp_name+'.interchange('+','.join([str(p) for p in self.params_list])+');'
        elif self.type == 'Skewing':
            # format of params_list must be [firts_loop, second_loop, first_factor, second_factor]
            assert len(self.params_list)==4
            return '\t'+self.comp_name+'.skew('+','.join([str(p) for p in self.params_list])+');'
        elif self.type == 'Parallelization':
            # format of params_list must be [loop]
            assert len(self.params_list)==1
            return '\t'+self.comp_name+'.tag_parallel_level('+str(self.params_list[0])+');'
        elif self.type == 'Tiling':
             # format of params_list must be [firts_loop, second_loop, first_factor, second_factor] in the case of tiling 2D
             # or [firts_loop, second_loop, third_loop, first_factor, second_factor, third_factor] in the case of tiling 3D
            assert len(self.params_list)==4 or len(self.params_list)==6
            return '\t'+self.comp_name+'.tile('+','.join([str(p) for p in self.params_list])+');'
        elif self.type == 'Unrolling':
            # format of params_list must be [loop, factor]
            assert len(self.params_list)==2
            return '\t'+self.comp_name+'.unroll('+','.join([str(p) for p in self.params_list])+');'
        elif self.type == 'Reversal':
             # format of params_list must be [firts_loop, second_loop, first_factor, second_factor] in the case of tiling 2D
             # or [firts_loop, second_loop, third_loop, first_factor, second_factor, third_factor] in the case of tiling 3D
            assert len(self.params_list)==1 
            return '\t'+self.comp_name+'.loop_reversal('+str(self.params_list[0])+');'