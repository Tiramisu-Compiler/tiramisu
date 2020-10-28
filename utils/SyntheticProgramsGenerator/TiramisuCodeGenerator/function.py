from builtins import Exception
import random as r
from random import random, randint, uniform, randrange, choice, shuffle, randint
from typing import List
from classes import *
from functools import reduce
from functools import cmp_to_key
from itertools import combinations, product

# ROOT_DIR_EXEC = "/data/scratch/mhleghettas/data0/programs/"
from classes import Variable
ROOT_DIR_EXEC = "./Sample/"
BIG_SIZES = [128 * i for i in range(1, 25)] + [128 * i for i in range(1, 17)] + \
                    [128 * i for i in range(1, 13)] + [128 * i for i in range(1, 9)]
MEDIUM_SIZES = [8, 32, 32, 64, 64, 64, 96, ]
SMALL_SIZES = [3, 3, 3, 4, 5, 6, 7, 8, 9, 10]

shuffle(BIG_SIZES)
shuffle(MEDIUM_SIZES)
shuffle(SMALL_SIZES)




def get_comps_ordering(comp_list):
    '''
    returns the computation ordering using the .then() statement as a string

    takes as input a list of Computation objects, a Computation object must have a '.name' attribute and a '.variables'
    attribute which is a list of Variable objects, a Variable object must have a '.id' (int or string) attribute and a '.name'
    attribute (the .id attribute can be replaced with the .name attribute)
    '''

    sorted_comp_list = sorted(comp_list, key=cmp_to_key(compare_by_vars))
    has_predecessor = [False for comp in comp_list]
    fused_comps_list = []
    fused_comps_list.append([sorted_comp_list[0]])
    code_buffer = '    '

    if (len(comp_list) < 2):  # if the program contains one computation no need for ordering
        return ''

    for i in range(1, len(sorted_comp_list)):
        for j in range(0, len(fused_comps_list)):
            if (sorted_comp_list[i].variables[-1].id == fused_comps_list[j][-1].variables[-1].id):
                fused_comps_list[j].append(sorted_comp_list[i])
                has_predecessor[i] = True
                break

        if (not has_predecessor[i]):
            fused_comps_list.append([sorted_comp_list[i]])

    code_buffer += sorted_comp_list[0].name

    for i in range(0, len(fused_comps_list)):
        for j in range(1, len(fused_comps_list[i])):
            if (not ((i == 0) and (j == 1))):
                code_buffer += '\n           '

            code_buffer += ".then(" + fused_comps_list[i][j].name + ", " + fused_comps_list[i][j].variables[
                -1].name + ")"

        if (i < len(fused_comps_list) - 1):
            k = 0
            while (fused_comps_list[i][-1].variables[k].id == fused_comps_list[i + 1][0].variables[k].id):
                k += 1
                if ((k == len(fused_comps_list[i][-1].variables)) or (k == len(
                        fused_comps_list[i + 1][
                            0].variables))):  # all vars of one of the comps are contained in the other
                    break

            code_buffer += '\n           '
            code_buffer += ".then(" + fused_comps_list[i + 1][0].name + ", " + fused_comps_list[i][-1].variables[
                k - 1].name + ")"

    code_buffer += ";\n"

    return code_buffer


def compare_by_vars(comp1, comp2):
    k = 0
    while ((k < len(comp1.variables)) and (k < len(comp2.variables))):
        if (comp1.variables[k].name < comp2.variables[k].name):
            return 1
        if (comp1.variables[k].name > comp2.variables[k].name):
            return -1
        k += 1

    if (len(comp1.variables) < len(comp2.variables)):
        return 1
    if (len(comp1.variables) > len(comp2.variables)):
        return -1

    return 1


# scheduling order : interchange tile , unroll the innermost, paralleling the outermost


class Function:
    MAX_MEMORY = 5 * 1024 * 1024 * 1024
    MAX_INPUTS_EACH_TIME = 4
    MAX_NUMBER_COMPUTATIONS_EACH_TIME = 2
    MAX_SIZE_SPLIT = 3  # not used
    MIN_NUMBER_FIRST_VARIABLES = 2  # must be >=2 (at least one interchange !)
    MAX_NUMBER_FIRST_VARIABLES = 4
    MAX_NUMBER_VARIABLES_BRANCH = 3
    NUMBER_OF_BRANCHES = 2
    MAX_NUMBER_VARIABLES = MAX_NUMBER_FIRST_VARIABLES + MAX_NUMBER_VARIABLES_BRANCH * NUMBER_OF_BRANCHES

    INPUT_ASSIGNMENT_PROBABILITY = 0.25
    STENSILE_PROBABILITY = 0.1
    CONVOLUTION_PROBABILITY = 0.1
    REDUCTION_PROBABILITY = 0.5
    SIMPLE_EXPRESSION_PROBABILITY = 1 - INPUT_ASSIGNMENT_PROBABILITY - STENSILE_PROBABILITY - CONVOLUTION_PROBABILITY
    tensor_id = 0

    def __init__(self, id, batch_name='', benchmark=False, sizes=[], output_dir=''):
        self.batch_name = batch_name
        self.output_dir = output_dir
        self.benchmark = benchmark
#         print('-----------------------{}-----------------------'.format(id))
        if not self.benchmark:
            r.seed(id)
            self.represenattion = {}
            self.id = id
            self.schedule_string = ""
            self.core_function()
            self.set_iterator_sizes()
            
            self.fusion_orderings = self.get_fusion_orderings()
            if len(self.fusion_orderings) == 0:
                self.fusion_orderings = [get_comps_ordering(self.computations)]
            self.comps_ordering = self.fusion_orderings[-1]

            self.create_buffers()
            # self.reduce_sizes()
        else:
            self.id = id
            self.schedule_string = ""
            self.comps_ordering = ""
            self.correction_store_in = ""
            self.buffers = []
            self.represenattion = {}
            self.id = id
            if self.id.startswith("_matmul"): 
                self.matmul_function(sizes=sizes)
            elif self.id.startswith("_convolution"): 
                self.convolution_function(sizes=sizes)
            elif self.id.startswith("_heat2d_noinit"): 
                self.heat2d_noinit_function(sizes=sizes)
            elif self.id.startswith("_heat2d"): 
                self.heat2d_function(sizes=sizes)
            elif self.id.startswith("_heat3d"): 
                self.heat3d_function(sizes=sizes)
            elif self.id.startswith("_cvtcolor_orig"): 
                self.cvtcolor_orig_function(sizes=sizes)
            elif self.id.startswith("_cvtcolor"): 
                self.cvtcolor_function(sizes=sizes)
            elif self.id.startswith("_conv_relu"): 
                self.conv_relu_function(sizes=sizes)
            elif self.id.startswith("_blur_i"): 
                self.blur_i_function(sizes=sizes)
            elif self.id.startswith("_blur"): 
                self.blur_function(sizes=sizes)

#             elif self.id == "_jacobi1d":
#                 self.jacobi1d_function()
#             elif self.id == "_jacobi2d":
#                 self.jacobi2d_function()
#             elif self.id == "_jacobi2d_small":
#                 self.jacobi2d_function(size = 128 )
#             elif self.id == "_jacobi2d_big":
#                 self.jacobi2d_function(size =1024)
            elif self.id.startswith("_jacobi1d_r"): 
                self.jacobi1d_r_function(sizes=sizes)
            elif self.id.startswith("_jacobi2d_r"): 
                self.jacobi2d_r_function(sizes=sizes)
            elif self.id.startswith("_seidel2d"): 
                self.seidel2d_function(sizes=sizes)
#             elif self.id == "_doitgen":
#                 self.doitgen_function()
            elif self.id.startswith("_doitgen_r"): 
                self.doitgen_r_function(sizes=sizes)
            elif self.id.startswith("_atax"): 
                self.atax_function(sizes=sizes)
            elif self.id.startswith("_bicg"): 
                self.bicg_function(sizes=sizes)
            elif self.id.startswith("_gemm"): 
                self.gemm_function(sizes=sizes)
            elif self.id.startswith("_gemver"): 
                self.gemver_function(sizes=sizes)
            elif self.id.startswith("_mvt"): 
                self.mvt_function(sizes=sizes)
                
            self.resize_buffers_benchmark()
            self.fusion_orderings = [self.comps_ordering]

    def mvt_function(self,sizes=[256]):
        self.constants = [Constant(0),]
        self.constants[0].value = sizes[0]

        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[0], True),
                          ]
        self.inputs = [Input(0, [self.variables[0], self.variables[1]])] + \
                      [Input(i, [self.variables[0]]) for i in range(1, 3)]

        self.computations = []

        self.computations.append(
            Computation(3, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[1], ],
                        expression="comp03(i00, i01) + input00(i00, i01) * input01(i01)"
                        ))

        self.computations.append(
            Computation(4, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[1], ],
                        expression="comp04(i00, i01) + input00(i01, i00) * input02(i01)"
                        ))

        self.comps_ordering = "    comp03.then(comp04, i01);\n"

        self.create_buffers()
        # self.buffers.remove(self.buffers[2])
        # self.correction_store_in = "    comp09.store_in(&buf08, {i00});\n"
    def gesummv_function(self, size=256):
        pass
    #     for (i = 0; i < _PB_N; i++)
    #         {
    #             tmp[i] = 0;
    #         y[i] = 0;
    #         for (j = 0; j < _PB_N; j++)
    #         {
    #             tmp[i] = A[i][j] * x[j] + tmp[i];
    #         y[i] = B[i][j] * x[j] + y[i];
    #         }
    #         y[i] = alpha * tmp[i] + beta * y[i];
    #         }
    #
    # tmp[inj] = A[i][j] * x[j] + tmp[i, j];
    # y[i, j] = B[i][j] * x[j] + y[i, j];
    #
    # y'[i] = alpha * tmp[i,0] + beta * y[i,0];

    def gemver_function(self, sizes=[256]):
        self.constants = [Constant(0),]
        self.constants[0].value = sizes[0]

        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[0], False),
                          ]
        self.inputs = [Input(0, [self.variables[0], self.variables[1]])] + \
                      [Input(i, [self.variables[0]]) for i in range(1, 7)]

        self.computations = []
        self.computations.append(Computation(7, self.variables[:], None,
                                             expression="input00(i00, i01) + input01(i00) * input02(i01) + input03(i00) * input04(i01)"))
        self.computations.append(
            Computation(8, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[1], ],
                        expression="comp08(i00, i01) + 6 * input00(i00, i01) * input05(i01)"
                        ))
        self.computations.append(
            Computation(9, [self.variables[0]], None,
                        simple_expression=True,
                        expression="comp08(i00, 0) + input06(i00)"
                        ))
        self.computations.append(
            Computation(10, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[1], ],
                        expression="comp10(i00, i01) + 5 * input00(i00, i01) * comp09(i01)"
                        ))

        self.comps_ordering = "    comp07.then(comp08, i01)\n         .then(comp09, i00)\n         .then(comp10, i00);\n"

        self.create_buffers()
        self.buffers.remove(self.buffers[2])
        self.correction_store_in = "    comp09.store_in(&buf08, {i00});\n"

    def gemm_function(self, sizes=[256, 1024, 256]):
        self.constants = [Constant(0), Constant(1), Constant(2)]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.constants[2].value = sizes[2]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], False),
                          ]
        self.inputs = [Input(0, [self.variables[0], self.variables[2]]),
                       Input(1, [self.variables[2], self.variables[1]]),
                       Input(2, [self.variables[0], self.variables[1]])]

        self.computations = []
#         self.computations.append(Computation(3, [self.variables[0], self.variables[1]], None,
#                                              expression=" 6 * input02(i00, i01)"))
        self.computations.append(
            Computation(4, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[2] ],
                        expression="comp04(i00, i01, i02) + 5 * (input00(i00, i02) * input01(i02, i01))"
                        ))
        self.computations.append(Computation(5, [self.variables[0], self.variables[1]], None,
                                             expression="comp04(i00, i01, 0) + 6 * input02(i00, i01)"))
        self.comps_ordering = "    comp04.then(comp05, i01);\n"

        self.create_buffers()
#         self.buffers.remove(self.buffers[1])
#         self.correction_store_in = "    comp03.store_in(&buf03, {i00, i01});\n"

    def bicg_function(self, sizes=[256,256]):
        self.constants = [Constant(0), Constant(1), ]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          ]
        self.inputs = [Input(0, [self.variables[0], self.variables[1]]),
                       Input(1, [self.variables[0]]),
                       Input(2, [self.variables[1]]), ]

        self.computations = []
        # self.computations.append(Computation(2, [self.variables[0], self.variables[1]], None,
        #                                      expression=" 0"))
        self.computations.append(
            Computation(3, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[0], ],
                        expression="comp03(i00, i01) + input00(i00, i01) * input01(i00)"
                        ))
        self.computations.append(
            Computation(4, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[1], ],
                        expression="comp04(i00, i01) + input00(i00, i01) * input02(i01)"
                        ))
        self.comps_ordering = "    comp03.then(comp04, i01);\n"

        self.create_buffers()
        # self.buffers.remove(self.buffers[1])
        # self.correction_store_in = "    comp02.store_in(&buf02, {i00, i01});\n"

    def atax_function(self, sizes=[256,256]):
        self.constants = [Constant(0), Constant(1), ]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          ]
        self.inputs = [Input(0, [self.variables[0], self.variables[1]]),
                       Input(1, [self.variables[1]]), ]

        self.computations = []
        # self.computations.append(Computation(2, [self.variables[0], self.variables[1]], None,
        #                                      expression=" 0"))
        self.computations.append(
            Computation(2, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[1], ],
                        expression="comp02(i00, i01) + input00(i00, i01) * input01(i01)"
                        ))
        self.computations.append(
            Computation(3, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[0], ],
                        expression="comp03(i00, i01) + input00(i00, i01) * comp02(i00,0)"
                        ))
        self.comps_ordering = "    comp02.then(comp03, i01);\n"

        self.create_buffers()
        # self.buffers.remove(self.buffers[1])
        # self.correction_store_in = "    comp02.store_in(&buf02, {i00, i01});\n"

#     def doitgen_function(self, sizes=[64, 512, 1024]):
#         self.constants = [Constant(0), Constant(1), Constant(2)]

#         self.constants[0].value = sizes[0]
#         self.constants[1].value = sizes[1]
#         self.constants[2].value = sizes[2]
#         self.variables = [Variable(0, -1, self.constants[0], True),
#                           Variable(1, 0, self.constants[1], True),
#                           Variable(2, 1, self.constants[2], True),
#                           Variable(3, 2, self.constants[2], False)
#                           ]
#         self.inputs = [Input(0, [self.variables[0], self.variables[1], self.variables[2]]),
#                        Input(1, [self.variables[3], self.variables[2]]), ]

#         self.computations = []
#         self.computations.append(Computation(2, [self.variables[0], self.variables[1], self.variables[2], ], None,
#                                              reduction=True, reduction_axe=[self.variables[0], self.variables[1]],
#                                              expression=" 0"))
#         self.computations.append(
#             Computation(3, self.variables[:], None,
#                         reduction=True, reduction_axe=[self.variables[0], self.variables[1],self.variables[3], ],
#                         expression="comp03(i00, i01, i02, i03) + input00(i00, i01, i02) * input01(i03, i02)"
#                         ))
#         self.computations.append(
#             Computation(4, self.variables[:-1], None,
#                         input_assignment=True,
#                         expression="comp03(i00, i01, i02, 0)"
#                         ))
#         self.comps_ordering = "    comp02.then(comp03, i02).then(comp04, i02);\n"

#         self.create_buffers()
#         self.buffers.remove(self.buffers[1])
#         self.buffers.remove(self.buffers[1])
#         self.correction_store_in = "    comp03.store_in(&buf02, {i02});\n    comp04.store_in(&buf00);\n"
    
    def doitgen_r_function(self, sizes=[256, 256, 256]):
        self.constants = [Constant(0), Constant(1), Constant(2)]

        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.constants[2].value = sizes[2]
        self.variables = [Variable(0, -1, self.constants[0], True), #r
                          Variable(1, 0, self.constants[1], True),  #q
                          Variable(2, 1, self.constants[2], True),  #p
                          Variable(3, 2, self.constants[2], False)  #s
                          ]
        self.inputs = [Input(0, [self.variables[0], self.variables[1], self.variables[3]]),
                       Input(1, [self.variables[2], self.variables[3]]), ]

        self.computations = []
#         self.computations.append(Computation(2, [self.variables[0], self.variables[1], self.variables[2], ], None,
#                                              reduction=True, reduction_axe=[self.variables[0], self.variables[1]],
#                                              expression=" 0"))
        self.computations.append(
            Computation(3, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[3]],
                        expression="comp03(i00, i01, i02, i03) + input00(i00, i01, i03) * input01(i02, i03)"
                        ))
#         self.computations.append(
#             Computation(4, self.variables[:-1], None,
#                         input_assignment=True,
#                         expression="comp03(i00, i01, i02, 0)"
#                         ))
#         self.comps_ordering = "    comp02.then(comp03, i02).then(comp04, i02);\n"

        self.create_buffers()
#         self.buffers.remove(self.buffers[1])
#         self.buffers.remove(self.buffers[1])
#         self.correction_store_in = "    comp03.store_in(&buf02, {i02});\n    comp04.store_in(&buf00);\n"

#     def jacobi1d_function(self, sizes=[10,256]):
#         self.constants = [Constant(0), Constant(1)]
#         self.constants[0].value = sizes[0]
#         self.constants[1].value = sizes[1]
#         self.variables = [Variable(0, -1, self.constants[0], True),
#                           Variable(1, 0, self.constants[1], True),]

#         self.variables[1].correct(1)

#         self.inputs = []
#         self.computations = []

#         self.computations.append(
#             Computation(0, self.variables[:], self.inputs[:], reduction=True, reduction_axe=[self.variables[0]],
#                         expression="3 * (comp01(0, i01) + comp01(0, i01 + 1) + comp01(0, i01 + 2))"
#                         )
#         )
#         self.computations.append(
#             Computation(1, self.variables[:], self.inputs[:], reduction=True, reduction_axe=[self.variables[0]],
#                         expression="3 * (comp00(0, i01) + comp00(0, i01 + 1) + comp00(0, i01 + 2))"
#                         )
#         )
#         self.create_buffers()
#         for buf in self.buffers:
#             buf.ignore_correction = True
#         self.comps_ordering = "    comp00.then(comp01, i01);\n"
        
    def jacobi1d_r_function(self, sizes = [96]):
        self.constants = [Constant(0)]
        self.constants[0].value = sizes[0]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          ]

        self.variables[0].correct(1)

        self.inputs = [Input(0, [self.variables[0]])]
        self.computations = []

        self.computations.append(
            Computation(1, self.variables[:], self.inputs[:], stencil=True,
                        expression="3 * (input00(i00) + input00(i00 + 1) + input00(i00 + 2))"
                        )
        )

        self.create_buffers()


#     def jacobi2d_function(self,size=128):
#         self.constants = [Constant(0)]
#         self.constants[0].value = size

#         self.variables = [Variable(0, -1, self.constants[0], True),
#                           Variable(1, 0, self.constants[0], True),
#                           ]

#         self.variables[0].correct(1)
#         self.variables[1].correct(1)


#         self.inputs = []
#         self.computations = []

#         self.computations.append(
#             Computation(0, self.variables[:], self.inputs[:], reduction=True, reduction_axe=[],
#                         expression="3 * (comp01(i00 + 1, i01 + 1) + comp01(i00 + 1,i01) + comp01(i00 + 1, i01 + 2) + comp01(i00 + 2, i01 + 1) + comp01(i00 + 2, i01 + 1))"
#                         )
#         )
#         self.computations.append(
#             Computation(1, self.variables[:], self.inputs[:],  reduction=True, reduction_axe=[],
#                         expression="3 * (comp00(i00 + 1, i01 + 1) + comp00(i00 + 1,i01) + comp00(i00 + 1, i01 + 2) + comp00(i00 + 2, i01 + 1) + comp00(i00 + 2, i01 + 1))"
#                         )
#         )
#         self.create_buffers()
#         self.comps_ordering = "    comp00.then(comp01, i01);\n"
    
    def jacobi2d_r_function(self,sizes=[96,96]):
        self.constants = [Constant(0), Constant(1)]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]

        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          ]

        self.variables[0].correct(1)
        self.variables[1].correct(1)


        self.inputs = [Input(0, [self.variables[0],self.variables[1]])]
        self.computations = []

        self.computations.append(
            Computation(1, self.variables[:], self.inputs[:], stencil=True, 
                        expression="3 * (input00(i00 + 1, i01 + 1) + input00(i00 + 1,i01) + input00(i00 + 1, i01 + 2) + input00(i00 + 2, i01 + 1) + input00(i00, i01 + 1))"
                        )
        )

        self.create_buffers()

    def seidel2d_function(self, sizes=[256]):
        self.constants = [Constant(0),]
        self.constants[0].value = sizes[0]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[0], True),
                          ]

        for v in self.variables[:]:
            v.correct(1)

        self.inputs = [Input(0, [self.variables[0], self.variables[1]])]
        self.computations = []

        self.computations.append(
            Computation(1, self.variables[:], self.inputs[:], reduction=True, reduction_axe=[],
                        expression="9 * (input00(i00, i01) + input00(i00, i01 + 1) + input00(i00, i01 + 2) + input00(i00 + 1, i01) + input00(i00 + 1, i01 + 1) + input00(i00 + 1, i01 + 2) + input00(i00 + 2, i01) + input00(i00 + 2, i01 + 1) + input00(i00 + 2, i01 + 2))")

        )
        self.create_buffers()


    def blur_function(self, sizes=[1024, 1024, 3]):
        self.constants = [Constant(0), Constant(1), Constant(2)]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.constants[2].value = sizes[2]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], True),
                          ]
        self.variables[0].correct(1)
        self.variables[1].correct(1)

        self.inputs = [Input(0, [self.variables[0], self.variables[1], self.variables[2]]),
                       ]

        self.computations = []
        self.computations.append(
            Computation(1, self.variables[:], None,
                        stencil=True,
                        expression="input00(i00, i01, i02) + input00(i00 + 1, i01, i02) + input00(i00 + 2, i01, i02) + input00(i00, i01 + 1, i02) + input00(i00 + 1, i01 + 1, i02) + input00(i00 + 2, i01 + 1 , i02) + input00(i00, i01 +2 , i02) + input00(i00 + 1, i01 +2 , i02) + input00(i00 + 2, i01 + 2, i02) "
                        ))
        # self.computations.append(
        #     Computation(2, self.variables[:], None,
        #                 stencil=True,
        #                 expression="comp01(i00, i01, i02) + comp01(i00, i01 + 1, i02) + comp01(i00, i01 + 2, i02)"
        #                 ))

        self.create_buffers()
        # self.comps_ordering = "    comp01.then(comp02, i00);\n"
    
    def blur_i_function(self, sizes=[3, 1024, 1024]): #Trying to interchange the dims
        self.constants = [Constant(0), Constant(1), Constant(2)]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.constants[2].value = sizes[2]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], True),
                          ]
        self.variables[1].correct(1)
        self.variables[2].correct(1)

        self.inputs = [Input(0, [self.variables[0], self.variables[1], self.variables[2]]),
                       ]

        self.computations = []
#         self.computations.append(
#             Computation(1, self.variables[:], None,
#                         stencil=True,
#                         expression="input00(i00, i01, i02) + input00(i00 + 1, i01, i02) + input00(i00 + 2, i01, i02) + input00(i00, i01 + 1, i02) + input00(i00 + 1, i01 + 1, i02) + input00(i00 + 2, i01 + 1 , i02) + input00(i00, i01 +2 , i02) + input00(i00 + 1, i01 +2 , i02) + input00(i00 + 2, i01 + 2, i02) "
#                         ))
        self.computations.append(
            Computation(1, self.variables[:], None,
                        stencil=True,
                        expression="input00(i00, i01, i02) + input00(i00, i01 + 1, i02) + input00(i00, i01 + 2, i02) + input00(i00, i01, i02 + 1) + input00(i00, i01 + 1, i02 + 1) + input00(i00, i01 + 2, i02 + 1) + input00(i00, i01, i02 + 2) + input00(i00, i01 + 1, i02 + 2) + input00(i00, i01 + 2, i02 + 2)"
                        ))
        # self.computations.append(
        #     Computation(2, self.variables[:], None,
        #                 stencil=True,
        #                 expression="comp01(i00, i01, i02) + comp01(i00, i01 + 1, i02) + comp01(i00, i01 + 2, i02)"
        #                 ))

        self.create_buffers()
        # self.comps_ordering = "    comp01.then(comp02, i00);\n"

    def matmul_function(self, sizes=[256, 1024, 256]):
        self.constants = [Constant(0), Constant(1), Constant(2)]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.constants[2].value = sizes[2]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], False),
                          ]
        self.inputs = [Input(0, [self.variables[0], self.variables[2]]),
                       Input(1, [self.variables[2], self.variables[1]]), ]

        self.computations = []
        # self.computations.append(Computation(2, [self.variables[0], self.variables[1]], None,
        #                                      expression=" 0"))
        self.computations.append(
            Computation(2, self.variables[:], None,
                        reduction=True, reduction_axe=[self.variables[2], ],
                        expression="comp02(i00, i01, i02) + input00(i00, i02) * input01(i02, i01)"
                        ))
        # self.comps_ordering = "    comp02.then(comp03, i01);\n"

        self.create_buffers()
        # self.buffers.remove(self.buffers[1])
#         self.correction_store_in = "    comp02.store_in(&buf02, {i00, i01});\n"

    def convolution_function(self, sizes=[32, 2, 1022, 1022, 3, 3, 3]):
        self.constants = [Constant(i) for i in range(7)]
        self.constants[0].value = sizes[0] # 32, 8
        self.constants[1].value = sizes[1]
        self.constants[2].value = sizes[2]
        self.constants[3].value = sizes[3]
        self.constants[4].value = sizes[4]
        self.constants[5].value = sizes[5]
        self.constants[6].value = sizes[6]

        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], True),
                          Variable(3, 2, self.constants[3], True),
                          Variable(4, 3, self.constants[4], False),
                          Variable(5, 4, self.constants[5], False),
                          Variable(6, 5, self.constants[6], False),
                          ]
        variables_to_computation = self.variables[:4]
        self.inputs = [Input(0, [self.variables[1]]), ]
        self.computations = [
            Computation(1, variables_to_computation, self.inputs[:],
                        input_assignment=True), ]
        variables_to_computation = self.variables[:]
        v_id = len(self.variables)
        c = Constant(len(self.constants))
        self.constants.append(c)

        variables_input_1 = [self.variables[0], self.variables[4]]
        reference_variables1 = [self.variables[2], self.variables[5]]
        reference_variables2 = [self.variables[3], self.variables[6]]
        i_y = Variable(v_id, v_id - 1, c, False,
                       reference_variables=reference_variables1)
        variables_input_1.append(i_y)
        self.variables.append(i_y)
        v_id = len(self.variables)
        c = Constant(len(self.constants))
        self.constants.append(c)
        i_x = Variable(v_id, v_id - 1, c, False,
                       reference_variables=reference_variables2)
        self.variables.append(i_x)
        variables_input_1.append(i_x)
        variables_input_2 = [self.variables[1], self.variables[4],
                             self.variables[5], self.variables[6],
                             ]
        self.inputs.append(Input(2, variables_input_1))
        self.inputs.append(Input(3, variables_input_2))

        inputs_to_computation = self.inputs[-2:]
        self.computations.append(
            Computation(4, variables_to_computation, inputs_to_computation,
                        convolution=True,
                        expression="comp04(i00, i01, i02, i03, i04, i05, i06) + input02(i00, i04, i02 + i05, i03 + i06) * input03(i01, i04, i05, i06)"))
        self.comps_ordering = "    comp01.then(comp04,i03);\n"
        self.create_buffers()
        self.buffers.remove(self.buffers[1])
        self.correction_store_in = "    comp04.store_in(&buf01, {i00, i01, i02, i03});\n"
        tensors = self.inputs[:] + self.computations[:]
        memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
#         print("memory space by convolution is  : ", memory_size / 1024 / 1024 / 1024 / 4, "GB")
#         print([(vv.sup.value, vv.schedulable) for vv in self.variables])


    def conv_relu_function(self, sizes=[32, 2, 1022, 1022, 3, 3, 3]):
        self.constants = [Constant(i) for i in range(7)]
        self.constants[0].value = sizes[0] # 32, 8
        self.constants[1].value = sizes[1]
        self.constants[2].value = sizes[2]
        self.constants[3].value = sizes[3]
        self.constants[4].value = sizes[4]
        self.constants[5].value = sizes[5]
        self.constants[6].value = sizes[6]


        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], True),
                          Variable(3, 2, self.constants[3], True),
                          Variable(4, 3, self.constants[4], False),
                          Variable(5, 4, self.constants[5], False),
                          Variable(6, 5, self.constants[6], False),
                          ]
        variables_to_computation = self.variables[:4]
        self.inputs = [Input(0, [self.variables[1]]), ]
        self.computations = [
            Computation(1, variables_to_computation, self.inputs[:],
                        input_assignment=True), ]
        variables_to_computation = self.variables[:]
        v_id = len(self.variables)
        c = Constant(len(self.constants))
        self.constants.append(c)

        variables_input_1 = [self.variables[0], self.variables[4]]
        reference_variables1 = [self.variables[2], self.variables[5]]
        reference_variables2 = [self.variables[3], self.variables[6]]
        i_y = Variable(v_id, v_id - 1, c, False,
                       reference_variables=reference_variables1)
        variables_input_1.append(i_y)
        self.variables.append(i_y)
        v_id = len(self.variables)
        c = Constant(len(self.constants))
        self.constants.append(c)
        i_x = Variable(v_id, v_id - 1, c, False,
                       reference_variables=reference_variables2)
        self.variables.append(i_x)
        variables_input_1.append(i_x)
        variables_input_2 = [self.variables[1], self.variables[4],
                             self.variables[5], self.variables[6],
                             ]
        self.inputs.append(Input(2, variables_input_1))
        self.inputs.append(Input(3, variables_input_2))

        inputs_to_computation = self.inputs[-2:]
        self.computations.append(
            Computation(4, variables_to_computation, inputs_to_computation,
                        convolution=True,
                        expression="comp04(i00, i01, i02, i03, i04, i05, i06) + input02(i00, i04, i02 + i05, i03 + i06) * input03(i01, i04, i05, i06)"))

        variables_to_computation = self.variables[:4]
        self.computations.append(
            Computation(5, variables_to_computation, None, input_assignment=True,
                        expression="expr(o_max, comp04(i00, i01, i02, i03, 0, 0, 0), 0)"
                        )

        )
        # self.computations.append(
        #     Computation(6, variables_to_computation, None, reduction=True,
        #                 expression="expr(o_max, comp06(i00, i01, i02, i03),  comp05(i00, i01, i02, i03))"
        #                 )
        #
        # )



        self.comps_ordering = "    comp01.then(comp04,i03)\n          .then(comp05,i03);\n"
        self.create_buffers()
        self.buffers.remove(self.buffers[1])
        self.correction_store_in = "    comp04.store_in(&buf01, {i00, i01, i02, i03});\n"






        tensors = self.inputs[:] + self.computations[:]
        memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
        #print("memory space by convolution is  : ", memory_size / 1024 / 1024 / 1024 / 4, "GB")
        #print([(vv.sup.value, vv.schedulable) for vv in self.variables])


    def heat2d_function(self,sizes=[1024,1024]):
        self.constants = [Constant(i) for i in range(2)]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          ]
        for v in self.variables:
            v.correct(1)

        self.inputs = [Input(0, self.variables[:])]
        self.computations = []

        self.computations.append(
            Computation(1, self.variables[:], None, simple_expression=True,
                        expression=" 0")
        )
        self.computations.append(
            Computation(2, self.variables[:], self.inputs[-1], stencil=True,
                        expression="3 * input00(i00 + 1, i01 + 1) +   4 * (input00(i00 + 1, i01 + 2) + input00(i00 + 1, i01) + input00(i00 + 2, i01 + 1) + input00(i00, i01 + 1))"
                        )
        )

        self.comps_ordering = "    comp01.then(comp02,i01);\n"

        self.create_buffers()
        self.buffers.remove(self.buffers[1])
        self.correction_store_in = "    comp02.store_in(&buf01, {i00, i01});\n"
        
    
    def heat2d_noinit_function(self,sizes=[1024,1024]):        
        self.constants = [Constant(i) for i in range(2)]
        self.constants[0].value = sizes[0]
        self.constants[1].value = sizes[1]
        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          ]
        for v in self.variables:
            v.correct(1)

        self.inputs = [Input(0, self.variables[:])]
        self.computations = []

        self.computations.append(
            Computation(2, self.variables[:], self.inputs[-1], stencil=True,
                        expression="3 * input00(i00 + 1, i01 + 1) +   4 * (input00(i00 + 1, i01 + 2) + input00(i00 + 1, i01) + input00(i00 + 2, i01 + 1) + input00(i00, i01 + 1))"
                        )
        )


        self.create_buffers()

    def heat3d_function(self,sizes=[960,1120,1214]):
        self.constants = [Constant(i) for i in range(3)]
        self.constants[0].value = sizes[0]  
        self.constants[1].value = sizes[1]
        self.constants[2].value = sizes[2]
        

        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], True)
                          ]
        for v in self.variables[:]:
            v.correct(1)

        # variables_to_input = self.variables[1:]
        # self.inputs = [Input(0, variables_to_input)]

        # inputs_to_computation = self.inputs[:]
        variables_to_computation = self.variables[:]
        # self.computations = [
        #     Computation(1, variables_to_computation, inputs_to_computation,
        #                 input_assignment=True,
        #                 )]
        # inputs_to_computation = self.inputs[-1]
        self.inputs = [Input(0, self.variables[:])]
        self.computations = []
        self.computations.append(
            Computation(2, variables_to_computation, self.inputs[0],
                        stencil=True,
                        expression="input00(i00, i01 + 1 ,i02 + 1) + 6 * input00(i00 + 1, i01 + 1 ,i02 + 1) -  input00(i00 + 2, i01 + 1 ,i02 + 1) + input00(i00 + 1, i01 ,i02 + 1) -  input00(i00 + 1, i01 + 2 ,i02 + 1) + input00(i00 + 1, i01 + 1, i02) -  input00(i00 + 1, i01 + 1 ,i02 + 2)"
                        #expression="comp02(i00, i01 - 1, i02, i03) + comp02(i00, i01, i02, i03) + comp02(i00, i01 + 1, i02, i03) - comp02(i00, i01, i02 - 1, i03) + comp02(i00, i01, i02, i03) + comp02(i00, i01, i02 + 1, i03) + comp02(i00, i01, i02, i03 - 1) + comp02(i00, i01, i02, i03) + comp02(i00, i01, i02, i03 + 1)"
                        )
        )

        # self.comps_ordering = "    comp01.then(comp02,i00);\n"

        self.create_buffers()
        # self.buffers.remove(self.buffers[1])
        # self.correction_store_in = "    comp02.store_in(&buf01);\n"

    def rgbyuv420_function(self):
        self.constants = [Constant(i) for i in range(5)]
        self.constants[0].value = 3
        self.constants[1].value = 2112
        self.constants[2].value = 3520
        self.constants[3].value = int(self.constants[1].value / 2)
        self.constants[4].value = int(self.constants[2].value / 2)

        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], True),
                          Variable(3, 2, self.constants[3], True),
                          ]
        # TODO: make this variables shared

    def cvtcolor_orig_function(self,sizes=[1024,1024,3]):
        self.constants = [Constant(i) for i in range(3)]
        self.constants[0].value = sizes[0]  # y
        self.constants[1].value = sizes[1]  # x
        self.constants[2].value = sizes[2]  # c

        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], False),
                          ]
        self.inputs = [
            Input(0, [self.variables[0], self.variables[1],self.variables[2]]),
                       ]
        self.computations = [
            Computation(2, [self.variables[0], self.variables[1]], self.inputs,
                        reduction=False,
                        expression="2 * input00(i00, i01, 0) + 3 * input00(i00, i01, 1) + 4 * input00(i00, i01, 2)"
#                         expression="input00(i00, i01, i02) * input01(i02)"
                        #expression="expr(o_cast, p_int_32, (((((expr(o_cast, p_uint32, input000(2, y, x)) * expr((uint32_t)1868)) + (expr(o_cast, p_uint32, input000(1, y, x)) * expr((uint32_t)9617))) + (expr(o_cast, p_uint32, input000(0, y, x)) * expr((uint32_t)4899))) + expr((uint32_t)8192)) / expr((uint32_t)16384)))"
                        )
        ]
        self.create_buffers()
        
    def cvtcolor_function(self,sizes=[1024,1024,3]):
        self.constants = [Constant(i) for i in range(3)]
        self.constants[0].value = sizes[0]  # y
        self.constants[1].value = sizes[1]  # x
        self.constants[2].value = sizes[2]  # c

        self.variables = [Variable(0, -1, self.constants[0], True),
                          Variable(1, 0, self.constants[1], True),
                          Variable(2, 1, self.constants[2], False),
                          ]
        self.inputs = [
            Input(0, [self.variables[0], self.variables[1],self.variables[2]]),
            Input(1, [self.variables[2]]),
                       ]
        self.computations = [
            Computation(2, [self.variables[0], self.variables[1], self.variables[2]], self.inputs,
                        reduction=True, reduction_axe=[self.variables[2] ],
                        expression="comp02(i00, i01, i02) + input00(i00, i01, i02) * input01(i02)")
        ]
        self.create_buffers()

    def resize_buffers_benchmark(self):
        min_big = 128
        tensors = self.inputs[:] + self.computations[:]
        variables = list(filter(lambda v: not v.sup.relative, self.variables))
        memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
        i = -1
        j = -1
#         print()
#         print("memory space was : ", memory_size / 1024 / 1024 / 1024, "GB")
#         print([(vv.sup.value, vv.schedulable) for vv in self.variables])
        while memory_size > self.MAX_MEMORY:
#             print('-----> variables :', [v.sup.value for v in self.variables])
            i = i + 1
            j = (j + 1) % len(variables)
            if i > 30000 * len(variables):
#                 print('variables :', [v.sup.value for v in variables])
                raise Exception("'set_iterator_sizes' not working at function :  " + self.id)
            v = variables[j]
            if v.sup.value >= min_big:
                v.sup.value = v.sup.value - 32
            else:
                variables.remove(v)
            if len(variables) == 0:
                variables = list(filter(lambda v: not v.sup.relative, self.variables))
                min_big = 64
            memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
#         print("memory updated to : ", memory_size / 1024 / 1024 / 1024, "GB")
#         print([(vv.sup.value, vv.schedulable) for vv in self.variables])
#         print(self.id)

    def set_random_branches(self):
        # self.NUMBER_VARIABLES_PROBABILITIES = [0.17,0.3,0.3  , 0.03, 0.03, 0.17]
        # assert reduce(lambda a, b: a + b, self.NUMBER_VARIABLES_PROBABILITIES) == 1
        self.NUMBER_VARIABLES_PROBABILITIES = [0.17, 0.47, 0.77, 0.8, 0.83, 1]
        self.number_computations_each_branch = []
        self.number_first_variables = 0
        self.branch_lengths = []
        two_computations_p = 0.5
        p = uniform(0, 1)
        if p < 0.5:
            number_of_branches = 2
        else:
            number_of_branches = 1

        for branch in range(number_of_branches):
            p = uniform(0, 1)
            if p < self.NUMBER_VARIABLES_PROBABILITIES[0]:
                self.branch_lengths.append(2)
            elif p < self.NUMBER_VARIABLES_PROBABILITIES[1]:
                self.branch_lengths.append(3)
            elif p < self.NUMBER_VARIABLES_PROBABILITIES[2]:
                self.branch_lengths.append(4)
            elif p < self.NUMBER_VARIABLES_PROBABILITIES[3]:
                self.branch_lengths.append(5)
            elif p < self.NUMBER_VARIABLES_PROBABILITIES[4]:
                self.branch_lengths.append(6)
            elif p < self.NUMBER_VARIABLES_PROBABILITIES[5]:
                self.branch_lengths.append(7)

            if self.branch_lengths[-1] > 5:
                # self.NUMBER_VARIABLES_PROBABILITIES = [0.17,0.3,0.3  , 0.03, 0.03, 0.17]
                self.NUMBER_VARIABLES_PROBABILITIES = [0, 0, 0.9, 1, 1, 1]
            elif self.branch_lengths[-1] == 2:
                # self.NUMBER_VARIABLES_PROBABILITIES = [0.34 , 0.3, 0.3 , 0.03, 0.03, 0]
                self.NUMBER_VARIABLES_PROBABILITIES = [0.24, 0.64, 0.94, 0.97, 1, 1]
            elif self.branch_lengths[-1] == 3:
                # self.NUMBER_VARIABLES_PROBABILITIES = [0.24, 0.35, 0.35 , 0.03, 0.03, 0]
                self.NUMBER_VARIABLES_PROBABILITIES = [0.34, 0.59, 0.94, 0.97, 1, 0]

            if self.branch_lengths[-1] > 4 or len(self.branch_lengths)>1:
                two_computations_p = 0.1
            else:
                two_computations_p = 0.6

            p = uniform(0, 1)
            if p < two_computations_p:
                self.number_computations_each_branch.append(2)
            else:
                self.number_computations_each_branch.append(1)
        shuffle(self.branch_lengths)
        self.number_first_variables = min(self.branch_lengths)
        self.number_first_variables = min(4, self.number_first_variables)
        self.branch_lengths = [n - self.number_first_variables for n in self.branch_lengths]

    def update_comp_type_probability(self, length):
        self.COMP_TYPE_PROBABILITY = {}
        if length == 2:
            self.COMP_TYPE_PROBABILITY['simple_expression'] = 0.33
            self.COMP_TYPE_PROBABILITY['stencil'] = 0.33
            self.COMP_TYPE_PROBABILITY['input_assignment'] = 0.34
            self.COMP_TYPE_PROBABILITY['reduction'] = 0
            self.COMP_TYPE_PROBABILITY['convolution'] = 0
        elif length == 3:
            self.COMP_TYPE_PROBABILITY['simple_expression'] = 0.23
            self.COMP_TYPE_PROBABILITY['stencil'] = 0.27
            self.COMP_TYPE_PROBABILITY['input_assignment'] = 0.23
            self.COMP_TYPE_PROBABILITY['reduction'] = 0.27
            self.COMP_TYPE_PROBABILITY['convolution'] = 0
        elif length == 4:
            self.COMP_TYPE_PROBABILITY['simple_expression'] = 0.05
            self.COMP_TYPE_PROBABILITY['stencil'] = 0.35
            self.COMP_TYPE_PROBABILITY['input_assignment'] = 0.26
            self.COMP_TYPE_PROBABILITY['reduction'] = 0.34
            self.COMP_TYPE_PROBABILITY['convolution'] = 0

        elif length == 5:
            self.COMP_TYPE_PROBABILITY['simple_expression'] = 0
            self.COMP_TYPE_PROBABILITY['stencil'] = 0.3
            self.COMP_TYPE_PROBABILITY['input_assignment'] = 0.1
            self.COMP_TYPE_PROBABILITY['reduction'] = 0.25
            self.COMP_TYPE_PROBABILITY['convolution'] = 0.35

        elif length == 6:
            self.COMP_TYPE_PROBABILITY['simple_expression'] = 0
            self.COMP_TYPE_PROBABILITY['stencil'] = 0.1
            self.COMP_TYPE_PROBABILITY['input_assignment'] = 0.1
            self.COMP_TYPE_PROBABILITY['reduction'] = 0.1
            self.COMP_TYPE_PROBABILITY['convolution'] = 0.7

        elif length == 7:
            self.COMP_TYPE_PROBABILITY['simple_expression'] = 0
            self.COMP_TYPE_PROBABILITY['stencil'] = 0.02
            self.COMP_TYPE_PROBABILITY['input_assignment'] = 0.02
            self.COMP_TYPE_PROBABILITY['reduction'] = 0.02
            self.COMP_TYPE_PROBABILITY['convolution'] = 0.94
        one = reduce(lambda a, b: a + b, list(self.COMP_TYPE_PROBABILITY.values()))
        assert one == 1

    def core_function(self):
        self.tensor_id = 0
        self.variables_in_branches = []
        self.set_random_branches()
        self.constants = [Constant(i) for i in range(self.number_first_variables)]
        self.variables = [Variable(self.constants.index(c), self.constants.index(c) - 1, c, True) for c in
                          self.constants]
        first_variables = self.variables[:]
        self.first_variables = self.variables[:]
        self.inputs = []
        self.computations = []

        for branch_length, number_computations in zip(self.branch_lengths, self.number_computations_each_branch):
            variables_in_branch = []
            for level in range(branch_length):
                c = Constant(len(self.constants))
                self.constants.append(c)
                new_variable_id = len(self.variables)
                if level == 0:
                    new_v = Variable(new_variable_id, self.number_first_variables - 1, c, False)
                    self.variables.append(new_v)
                    variables_in_branch.append(new_v)
                else:
                    new_v = Variable(new_variable_id, new_variable_id - 1, c, False)
                    self.variables.append(new_v)
                    variables_in_branch.append(new_v)
            self.variables_in_branches.append(variables_in_branch[:])
            variables_to_computation = first_variables[:] + variables_in_branch[:]
            self.update_comp_type_probability(len(variables_to_computation))
            for k_ in range(number_computations):
                simple_expression = False
                stencil = False
                input_assignment = False
                reduction = False
                convolution = False

                probability = uniform(0, 1)
                if probability < self.COMP_TYPE_PROBABILITY['simple_expression']:
                    simple_expression = True
                elif probability < self.COMP_TYPE_PROBABILITY['simple_expression'] + \
                        self.COMP_TYPE_PROBABILITY['input_assignment']:
                    input_assignment = True
                elif probability < self.COMP_TYPE_PROBABILITY['simple_expression'] + \
                        self.COMP_TYPE_PROBABILITY['input_assignment'] + \
                        self.COMP_TYPE_PROBABILITY['stencil']:
                    stencil = True
                elif probability < self.COMP_TYPE_PROBABILITY['simple_expression'] + \
                        self.COMP_TYPE_PROBABILITY['input_assignment'] + \
                        self.COMP_TYPE_PROBABILITY['stencil'] + \
                        self.COMP_TYPE_PROBABILITY['reduction']:
                    reduction = True
                else:
                    convolution = True

                if simple_expression:
                    self.computations.append(Computation(self.tensor_id, variables_to_computation, None,
                                                         simple_expression=True))
                    self.tensor_id += 1
                elif stencil:
                    variables_to_input = variables_to_computation[:]
                    self.inputs.append(Input(self.tensor_id, variables_to_input))
                    self.tensor_id += 1
                    self.computations.append(
                        Computation(self.tensor_id, variables_to_computation, self.inputs[-1],
                                    stencil=True))
                    self.tensor_id += 1
                elif input_assignment:
                    input_dimension = randint(1, 3)
                    order = (uniform(0, 1) < 0.85)
                    variables_to_input = choose_elements_from_list(variables_to_computation, input_dimension,
                                                                   order=order)
                    self.inputs.append(Input(self.tensor_id, variables_to_input))
                    self.tensor_id += 1
                    inputs_to_computation = self.inputs[-1:]
                    self.computations.append(
                        Computation(self.tensor_id, variables_to_computation, inputs_to_computation,
                                    input_assignment=True))
                    self.tensor_id += 1
                elif reduction:
                    axe, l, r = split_for_reduction(variables_to_computation)
                    self.inputs.append(Input(self.tensor_id, r))
                    self.tensor_id += 1
                    self.inputs.append(Input(self.tensor_id, l))
                    self.tensor_id += 1
                    inputs_to_computation = self.inputs[-2:]
                    self.computations.append(
                        Computation(self.tensor_id, variables_to_computation, inputs_to_computation,
                                    reduction=True, reduction_axe=axe,
                                    ))
                    self.tensor_id += 1
                else:
                    v_id = len(self.variables)
                    c = Constant(len(self.constants))
                    self.constants.append(c)
                    p1 = choose_elements_from_list(first_variables, 3)
                    p2 = choose_elements_from_list(variables_in_branch, 3)
                    variables_input_1 = [p1[0], p2[0]]
                    reference_variables1 = [p1[-2], p2[-2]]
                    reference_variables2 = [p1[-1], p2[-1]]
                    i_y = Variable(v_id, - 1, c, False,
                                   reference_variables=reference_variables1)
                    variables_input_1.append(i_y)
                    self.variables.append(i_y)
                    v_id = len(self.variables)
                    c = Constant(len(self.constants))
                    self.constants.append(c)
                    i_x = Variable(v_id, - 1, c, False,
                                   reference_variables=reference_variables2)
                    self.variables.append(i_x)
                    variables_input_1.append(i_x)
                    variables_input_2 = choose_elements_from_list(first_variables, 1) + variables_in_branch[:]
                    self.inputs.append(Input(self.tensor_id, variables_input_1))
                    self.tensor_id += 1
                    self.inputs.append(Input(self.tensor_id, variables_input_2))
                    self.tensor_id += 1
                    inputs_to_computation = self.inputs[-2:]
                    self.computations.append(
                        Computation(self.tensor_id, variables_to_computation, inputs_to_computation,
                                    convolution=True))
                    self.tensor_id += 1

    def get_variable_level(self, v):
        if v.id == 0:
            return 0
        else:
            for w in self.variables:
                if w.id == v.parent:
                    p = w
                    break
            return 1 + self.get_variable_level(p)

    def set_iterator_sizes(self):
        min_small = 3

        min_big = 128

        max_small = max(SMALL_SIZES)
        max_big = max(BIG_SIZES)

        for v in self.variables:
            if not v.sup.relative:
                v.level = self.get_variable_level(v)
                v.sup.value = choice(SMALL_SIZES)
            else:
                v.sup.value = 0
        max_level = max([v.level for v in self.variables])
        if max_level >= 4:
            for v in self.variables:
                if v.level == 0:
                    v.sup.value = choice(MEDIUM_SIZES)
                elif v.level == 1:
                    v.sup.value = choice(SMALL_SIZES)
                elif v.level == 2 or v.level == 3:
                    v.sup.value = choice(BIG_SIZES)
        else:
            for v in self.variables:
                if v.level >= max_level - 2:
                    v.sup.value = choice(BIG_SIZES)
                if v.level == 0:
                    p = uniform(0, 1)
                    if p < 0.5:
                        v.sup.value = choice(SMALL_SIZES)

        tensors = self.inputs[:] + self.computations[:]
        variables = list(filter(lambda v: not v.sup.relative, self.variables))
        memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
        i = -1
        j = -1

#         print()
#         print("memory space was : ", memory_size / 1024 / 1024 / 1024, "GB")
#         print([(vv.sup.value, vv.schedulable) for vv in self.variables])

        while memory_size > self.MAX_MEMORY:
            i = i + 1
            j = (j + 1) % len(variables)
            if i > 300 * len(variables):
                raise Exception("'set_iterator_sizes' not working at function :  "+self.id)
            v = variables[j]
            if v.sup.value >= min_big:
                v.sup.value = v.sup.value - 32
            else:
                variables.remove(v)
            if len(variables) == 0:
                variables = list(filter(lambda v: not v.sup.relative, self.variables))
                min_big = 64


            memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
#         print("memory updated to : ", memory_size / 1024 / 1024 / 1024, "GB")
#         print([(vv.sup.value, vv.schedulable) for vv in self.variables])
#         print(self.id)


    def get_fusion_orderings(self):
        sorted_comp_list = sorted(self.computations[:], key=cmp_to_key(compare_by_vars))
#         print([comp.name for comp in sorted_comp_list])
        variables = list(filter(lambda v: not v.sup.relative, self.variables))
        for v in variables:
            v.level = self.get_variable_level(v)
        shared_variables = list(filter(lambda v: v.schedulable, variables))
        max_shared_level = max([v.level for v in shared_variables])
        assert max_shared_level+1 == len(shared_variables)

        branches_comps = []
        orders_in_branches = []

        branch_heads = ['', ] + list(filter(lambda v: v.level == max_shared_level+1, variables))
        for i in branch_heads:
            group_comps = []
            if i != '':
                current_variables = shared_variables[:] + [i]
                for j in variables:
                    if j.parent == current_variables[-1].id:
                        current_variables.append(j)
            else :
                current_variables = shared_variables[:]
            current_variables_set = set([w.id for w in current_variables])
            for comp in sorted_comp_list:
                selected = False
                comp_variables = list(filter(lambda v: not v.sup.relative, comp.variables))
                comp_variables_set = set([w.id for w in  comp_variables])

                # if current_variables_set.issubset(comp_variables_set) and \
                #         current_variables_set.isuperset(comp_variables_set):
                #     selected = True
                # elif i!= '' and current_variables_set.isuperset(comp_variables_set) and len(shared_variables)<len(comp_variables_set):
                #
                selected = current_variables_set.issuperset(comp_variables_set) and (current_variables_set.issubset(comp_variables_set) or len(shared_variables)<len(comp_variables_set))
                if selected:
                    group_comps.append(comp)

            if len(group_comps) < 1:
                continue
            code_buff = group_comps[0].name
            for comp in group_comps[1:]:
                fusion_variable = list(filter(lambda v: not v.sup.relative, comp.variables))[-1]
                code_buff += '.then(' + comp.name + ', ' + fusion_variable.name + ')'
            # code_buff += ';\n'

            branches_comps.append(group_comps)
            if len(group_comps) < 2:
                code_buff = ''
            orders_in_branches.append(code_buff)

        fusion_orderings = []
        for b_i in range(len(branches_comps)-1):
            for v in shared_variables:
                code_buff = branches_comps[b_i][-1].name + '.then(' + branches_comps[b_i+1][0].name + ', ' + v.name + ')'
                fusion_orderings.append(orders_in_branches+[code_buff])
#         print([(v.name, v.level, v.schedulable) for v in variables])
#         print(orders_in_branches)
#         print([[comp.name for comp in branches_comp] for branches_comp in branches_comps])
#         print(fusion_orderings)
        res = []
        for v,orders in zip(shared_variables,fusion_orderings):
            x = ''
            for order in orders:
                if order == '':
                    continue
                x += '    ' + order + ';\n'
            x += '    // fusion variable :{}\n'.format(v.name)
            res.append(x)
        return res



#old code --------------------------------------------------




#old code --------------------------------------------------







































    def correct_buffer_sizes_old(self):
        BIG_SIZES = [128 * i for i in range(1, 17)]
        MEDIUM_SIZES = [8, 32, 64, 96, 128]
        SMALL_SIZES = [2, 3, 4, 6, 9]
        f = self.first_variables
        if len(self.variables_in_branches[0]) > len(self.variables_in_branches[1]):
            b = self.variables_in_branches[0][:]
            s = self.variables_in_branches[1][:]
        else:
            b = self.variables_in_branches[1][:]
            s = self.variables_in_branches[0][:]

        for v in b:
            v.sup.value = choice(SMALL_SIZES)
        for v in s:
            v.sup.value = choice(SMALL_SIZES)

        if len(f) + len(b) >= 5:  # conv
            f[-1].sup.value = choice(BIG_SIZES)
            f[-2].sup.value = choice(BIG_SIZES)
            f[-3].sup.value = choice(SMALL_SIZES)
            if len(f) > 3:
                f[0].sup.value = choice(MEDIUM_SIZES)
            for v in b:
                v.sup.value = choice(SMALL_SIZES)
            for v in s:
                v.sup.value = choice(SMALL_SIZES)
        else:
            f[-1].sup.value = choice(BIG_SIZES)
            f[-2].sup.value = choice(BIG_SIZES)
            if len(f) > 3:
                f[-3].sup.value = choice(BIG_SIZES)
                f[-4].sup.value = choice(SMALL_SIZES)
            else:
                if len(f) > 2:
                    f[-3].sup.value = choice(BIG_SIZES)
                else:
                    if len(b) > 0:
                        b[0].sup.value = choice(BIG_SIZES)
                    if len(s) > 0:
                        s[0].sup.value = choice(BIG_SIZES)

    def create_variables_computations_inputs_last_old(self):
        self.tensor_id = 0
        numberFirstVariables = randint(self.MIN_NUMBER_FIRST_VARIABLES, self.MAX_NUMBER_FIRST_VARIABLES)
        self.constants = [Constant(i) for i in range(numberFirstVariables)]
        self.variables = [Variable(self.constants.index(c), self.constants.index(c) - 1, c, True) for c in
                          self.constants]
        firstVariables = self.variables[:]
        self.inputs = []
        self.computations = []

        for branch in range(self.NUMBER_OF_BRANCHES):
            numberVaribalesInBranch = randint(1, self.MAX_NUMBER_VARIABLES_BRANCH)
            numberComputations = randint(1, self.MAX_NUMBER_COMPUTATIONS_EACH_TIME)
            variablesInBranch = []
            for ii in range(numberVaribalesInBranch):
                c = Constant(len(self.constants))
                self.constants.append(c)
                newVariableId = len(self.variables)
                if ii == 0:
                    self.variables.append(Variable(newVariableId, numberFirstVariables - 1, c, False))
                    variablesInBranch.append(Variable(newVariableId, numberFirstVariables - 1, c, False))
                else:
                    self.variables.append(Variable(newVariableId, newVariableId - 1, c, False))
                    variablesInBranch.append(Variable(newVariableId, newVariableId - 1, c, False))

                if ii == numberVaribalesInBranch - 1:
                    n = numberComputations
                else:
                    n = randint(0, numberComputations - 1)
                    numberComputations = numberComputations - n

                variablesToComputation = firstVariables + variablesInBranch[:]
                # shuffle(variablesToComputation)
                for k_ in range(n):
                    probability = uniform(0, 1)
                    if probability < self.SIMPLE_EXPRESSION_PROBABILITY:
                        self.computations.append(Computation(self.tensor_id, variablesToComputation, None, 0))
                        self.tensor_id += 1
                        if len(firstVariables) == 2 and len(variablesInBranch) == 1:
                            l = [firstVariables[0], variablesInBranch[0]]
                            r = [variablesInBranch[0], firstVariables[1]]
                            self.inputs.append(Input(self.tensor_id, r))
                            self.tensor_id += 1
                            self.inputs.append(Input(self.tensor_id, l))
                            self.tensor_id += 1
                            self.computations.append(
                                Computation(self.tensor_id, firstVariables + [variablesInBranch[0], ], self.inputs[:],
                                            2,
                                            reduction=True, reductionAxe=[variablesInBranch[0], ],
                                            ))
                            self.tensor_id += 1
                    elif probability < self.SIMPLE_EXPRESSION_PROBABILITY + self.STENSILE_PROBABILITY:
                        self.inputs.append(Input(self.tensor_id, variablesToComputation))
                        self.tensor_id += 1
                        self.computations.append(
                            Computation(self.tensor_id, variablesToComputation, self.inputs[-1], 1))
                        self.tensor_id += 1
                    elif probability < self.SIMPLE_EXPRESSION_PROBABILITY + self.STENSILE_PROBABILITY + self.INPUT_ASSIGNMENT_PROBABILITY:
                        numberInputs = randint(2, self.MAX_INPUTS_EACH_TIME)
                        for kk in range(numberInputs):
                            inputDimention = randint(0, len(variablesToComputation))
                            # variables_input = variablesToComputation[:]
                            # p = uniform(0, 1)
                            # if p < 0.5:
                            #     shuffle(variables_input)
                            variables = firstVariables[:]
                            p = uniform(0, 1)
                            if p < 0.5:
                                shuffle(variables)
                            variables = variables + [choice(variablesInBranch) for _ in range(inputDimention)]
                            p = uniform(0, 1)
                            if p < 0.5:
                                shuffle(variables)
                            self.inputs.append(Input(self.tensor_id, variables))
                            self.tensor_id += 1
                        self.computations.append(
                            Computation(self.tensor_id, variablesToComputation, self.inputs[:], numberInputs))
                        self.tensor_id += 1
                    else:

                        variables_input_1 = [choice(firstVariables), choice(firstVariables), ]
                        v_id = len(self.variables)
                        c = Constant(len(self.constants))
                        self.constants.append(c)
                        i_y = Variable(v_id, v_id - 1, c, False,
                                       reference_variables=[choice(firstVariables), choice(variablesInBranch)])
                        variables_input_1.append(i_y)
                        self.variables.append(i_y)
                        v_id = len(self.variables)
                        c = Constant(len(self.constants))
                        self.constants.append(c)
                        i_x = Variable(v_id, v_id - 1, c, False,
                                       reference_variables=[choice(firstVariables), choice(variablesInBranch)])
                        self.variables.append(i_x)
                        variables_input_1.append(i_x)
                        variables_input_2 = [choice(firstVariables), choice(firstVariables),
                                             choice(variablesInBranch), choice(variablesInBranch)]
                        self.inputs.append(Input(self.tensor_id, variables_input_1))
                        self.tensor_id += 1
                        self.inputs.append(Input(self.tensor_id, variables_input_2))
                        self.tensor_id += 1
                        self.computations.append(
                            Computation(self.tensor_id, variablesToComputation, self.inputs[:], 2, conv=True))
                        self.tensor_id += 1

    def reduce_sizes_old(self):
        tensors = self.inputs[:] + self.computations[:]
        schedulable_variable = list(filter(lambda v: v.schedulable, self.variables))
        unschedulable_variable = list(filter(lambda v: not v.schedulable and not v.sup.relative, self.variables))
        memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
        # print("memory space was : ", memory_size / 1024 / 1024 / 1024, "GB")
        # print([(vv.sup.value, vv.schedulable) for vv in self.variables])
        reduce_schedulable = False
        while (memory_size > self.MAX_MEMORY):
            if reduce_schedulable:
                v = choice(schedulable_variable)
                if v.sup.value > 32:
                    v.sup.value = int(v.sup.value / 2)
                else:
                    for vv in schedulable_variable:
                        if vv.sup.value == max([v.sup.value for v in schedulable_variable]):
                            vv.sup.value = int(vv.sup.value / 2)
                            break
            elif len(unschedulable_variable) > 0:
                v = choice(unschedulable_variable)
                if v.sup.value > 16:
                    v.sup.value = randint(3, 16)
                else:
                    for vv in unschedulable_variable:
                        if vv.sup.value == max([v.sup.value for v in unschedulable_variable]):
                            v.sup.value = randint(3, 16)
                            break

            reduce_schedulable = True
            for v in unschedulable_variable:
                if v.sup.value > 16:
                    reduce_schedulable = False
            if reduce_schedulable:
                schedulable_variable[1].sup.value = randint(5, 10)

            memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
        # print("Memory after reduction: ", memory_size / 1024 / 1024 / 1024, "GB")
        # print([(vv.sup.value, vv.schedulable) for vv in self.variables])
        # print(max([vv.sup.value for vv in self.variables]))

    def reduce_sizes_old0(self):
        tensors = self.inputs[:] + self.computations[:]

        def eval(t):
            return t.get_volume()

        tensors.sort(key=eval, reverse=True)
        if len(tensors) == 0:
            print("ééééééééééééé")
        memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])
#         print("Memory space was : ", memory_size / 1024 / 1024 / 1024 / 4, "GB")

        while (memory_size > self.MAX_MEMORY):
            max_size_larget_tensor = max([vv.sup.value for vv in tensors[0].variables])
            if max_size_larget_tensor > 64:
                for cons in [vv.sup for vv in tensors[0].variables]:
                    if cons.value == max_size_larget_tensor:
                        cons.value = int(cons.value / 2)
                        #print(cons.value * 2, " **-----> ", cons.value)
                        continue
            else:
                max_size = max([vv.sup.value for vv in self.variables])
                for cons in [vv.sup for vv in self.variables]:
                    if cons.value == max_size:
                        if cons.value / 64 > 64:
                            cons.value = int(cons.value / 64)
                        elif cons.value / 32 > 64:
                            cons.value = int(cons.value / 32)
                        elif cons.value / 16 > 64:
                            cons.value = int(cons.value / 16)
                        elif cons.value / 8 > 64:
                            cons.value = int(cons.value / 8)
                        elif cons.value / 4 > 64:
                            cons.value = int(cons.value / 4)
                        else:
                            cons.value = int(cons.value / 2)
                        #print(cons.value * 2, " -----> ", cons.value)
                        continue
            memory_size = functools.reduce(lambda a, b: a + b, [t.get_volume() for t in tensors])

#         print("Memory after reduction: ", memory_size / 1024 / 1024 / 1024 / 4, "GB")
#         print([vv.sup.value for vv in self.variables])
#         print(max([vv.sup.value for vv in self.variables]))


    def create_variables_computations_inputs(self):
        self.tensor_id = 0
        numberFirstVariables = randint(self.MIN_NUMBER_FIRST_VARIABLES, self.MAX_NUMBER_FIRST_VARIABLES)
        self.constants = [Constant(i) for i in range(numberFirstVariables)]
        self.variables = [Variable(self.constants.index(c), self.constants.index(c) - 1, c, True) for c in
                          self.constants]

        # self.inputs = [Input(0, self.variables[:]), ]
        # We have decided to generate them just before each computation
        numberSecondVaribales = randint(1, self.MAX_NUMBER_SECOND_VARIABLES)
        self.inputs = []
        self.computations = []
        while (numberSecondVaribales > 0):
            n = randint(1, numberSecondVaribales)
            numberSecondVaribales = numberSecondVaribales - n
            for _ in range(n):
                c = Constant(len(self.constants))
                self.constants.append(c)
                self.variables.append(Variable(len(self.variables), len(self.variables) - 1, c, False))

            numberComputations = randint(1, self.MAX_NUMBER_COMPUTATIONS_EACH_TIME)
            for __ in range(numberComputations):
                probability = uniform(0, 1)
                if probability < self.SIMPLE_EXPRESSION_PROBABILITY:
                    self.computations.append(Computation(self.tensor_id, self.variables[:], None, 0))
                    self.tensor_id += 1
                elif probability < self.SIMPLE_EXPRESSION_PROBABILITY + self.STENSILE_PROBABILITY:
                    self.inputs.append(Input(self.tensor_id, self.variables[:]))
                    self.tensor_id += 1
                    self.computations.append(Computation(self.tensor_id, self.variables[:], self.inputs[-1], 1))
                    self.tensor_id += 1
                else:
                    numberInputs = randint(2, self.MAX_INPUTS_EACH_TIME)

                    for kk in range(numberInputs):
                        # numberVariablesForCreation = randint(1,len(self.variables)-1)
                        inputDimention = randint(1, len(self.variables) - 1)
                        variables = [choice(self.variables) for i in range(inputDimention)]
                        self.inputs.append(Input(self.tensor_id, variables))
                        # self.inputs.append(Input(self.tensorId, self.variables[:]))
                        self.tensor_id += 1
                    # self.computations.append(
                    #     Computation(self.tensorId, self.variables[:], self.inputs[:], numberInputs))
                    self.computations.append(
                        Computation(self.tensor_id, self.variables[:], self.inputs[:], numberInputs))
                    self.tensor_id += 1

    def create_buffers(self):
        self.buffers = [Buffer(c) for c in self.computations]
        for inp in self.inputs:
            self.buffers.append(Buffer(inp))

    def get_program_cpp(self, schedule_number, comps_ordering):
        res = """#include <tiramisu/tiramisu.h>
using namespace tiramisu;

int main(int argc, char **argv){
    tiramisu::init("function""" + str(self.id) + schedule_number + '");' + "\n\n"

        res = res + """    constant """
        for i in self.constants:
            res = res + str(i) + ", "
        res = res[:-2] + ";\n\n"

        res = res + """    var """
        for i in self.variables:
            res = res + str(i) + ", "
        res = res[:-2] + ";\n\n"

        for i in self.inputs:
            res = res + "    " + str(i) + "\n"
        res = res + "\n"

        for i in self.computations:
            res = res + "    " + str(i) + "\n"
        for i in self.computations:
            res = res + "    " + i.get_setexpression()
        res = res + "\n"

        res += comps_ordering + "\n"

        res = res + self.schedule_string + "\n"

        for i in self.buffers:
            res = res + "    " + str(i) + "\n"
        res = res + "\n"

        for i in self.buffers:
            res = res + """    """ + i.write_store() + "\n"
        if self.benchmark:
            res = res + self.correction_store_in

        res = res + "\n    tiramisu::codegen({"
        for i in self.buffers:
            res = res + "&" + i.name + ", "
        # res = res[:-2] + '},' + '"' + ROOT_DIR_EXEC + 'function' + str(self.id) + '/function' + str(self.id) + \
        #       schedule_number + '/function' + str(self.id) + schedule_number + '.o");\n\n'
#         res = res[:-2] + '},' + '"' + '../data/'+self.batch_name+'/programs/function'+self.id+'/function' + str(self.id)\
#             + schedule_number + '/function' + str(self.id) + schedule_number + '.o");\n\n'
        res = res[:-2] + '},' + '"' + self.output_dir+'/function'+self.id+'/function' + str(self.id)\
            + schedule_number + '/function' + str(self.id) + schedule_number + '.o");\n\n'

        res = res + "\n" + """    return 0;
}"""

        return res

    def get_wrapper_cpp(self, schedule_number):
        # TODO : what about function.o.h or function.o.h
        res = '#include "Halide.h"\n' + '#include "function' + str(self.id) + schedule_number + '_wrapper.h"\n'

        res = res + """#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

#define MAX_RAND 200

using namespace std::chrono;
using namespace std;

int main(int, char **argv){\n"""

        for i in self.buffers:
            res = res + "    " + i.write_for_wrapper() + "\n\n\n"
        res = res + "    int nb_tests = atoi(argv[1]);\n\n"
        if "no" in schedule_number:
            res = res + "    nb_tests = 1.5 * (nb_tests-1) + 1;\n\n"
        res = res + "    duration < double > diff;\n"
        res = res + "    for (int i = 0; i < 2; ++i) {\n"
        res = res + "    auto t0 = std::chrono::high_resolution_clock::now();\n\n"
        res = res + "    function" + str(self.id) + schedule_number + "("
        for i in self.buffers:
            res = res + i.name + ".raw_buffer()" + ", "
        res = res[:-2] + ");\n\n" + "    auto t1 = std::chrono::high_resolution_clock::now();\n\n"
        res = res + "    diff = t1 - t0;\n"
        res = res + '    std::cout << duration_cast<microseconds>(t1 - t0).count() << " ";\n'
        res = res + "    }\n"
        res = res + "    if ((diff.count() < 2) && (nb_tests < 15))\n"
        res = res + "        nb_tests += 5;\n"
        res = res + "    for (int i = 0; i < nb_tests - 2; ++i) {\n"
        res = res + "    auto t0 = std::chrono::high_resolution_clock::now();\n\n"
        res = res + "    function" + str(self.id) + schedule_number + "("
        for i in self.buffers:
            res = res + i.name + ".raw_buffer()" + ", "
        res = res[:-2] + ");\n\n" + "    auto t1 = std::chrono::high_resolution_clock::now();\n\n"
        res = res + '    std::cout << duration_cast<microseconds>(t1 - t0).count() << " ";\n'
        res = res + "    }\n"
        res = res + "    return 0;\n"
        res = res + "}"

        return res

    def get_wrapper_h(self, schedule_number):
        res = "#ifndef HALIDE__generated_function" + str(self.id) + schedule_number + "_h\n" + \
              "#define HALIDE__generated_function" + str(self.id) + schedule_number + "_h\n"

        res = res + """#include <tiramisu/utils.h>

#define NB_THREAD_INIT 48
struct args {
    int *buf;
    unsigned long long int part_start;
    unsigned long long int part_end;
    int value;
};

void *init_part(void *params)
{
   int *buffer = ((struct args*) params)->buf;
   unsigned long long int start = ((struct args*) params)->part_start;
   unsigned long long int end = ((struct args*) params)->part_end;
   int val = ((struct args*) params)->value;
   for (unsigned long long int k = start; k < end; k++){
       buffer[k]=val;
   }
   pthread_exit(NULL);
}

void parallel_init_buffer(int* buf, unsigned long long int size, int value){
    pthread_t threads[NB_THREAD_INIT]; 
    struct args params[NB_THREAD_INIT];
    for (int i = 0; i < NB_THREAD_INIT; i++) {
        unsigned long long int start = i*size/NB_THREAD_INIT;
        unsigned long long int end = std::min((i+1)*size/NB_THREAD_INIT, size);
        params[i] = (struct args){buf, start, end, value};
        pthread_create(&threads[i], NULL, init_part, (void*)&(params[i])); 
    }
    for (int i = 0; i < NB_THREAD_INIT; i++) 
        pthread_join(threads[i], NULL); 
    return;
}
        


#ifdef __cplusplus
extern "C" {
#endif\n\n
"""
        res = res + "int function" + str(self.id) + schedule_number + "("
        for i in self.buffers:
            res = res + "halide_buffer_t *" + i.name + ", "
        res = res[:-2] + ");\n\n"
        res = res + """#ifdef __cplusplus
}  // extern "C"
#endif
#endif"""
        return res

    def get_representation(self):
        self.represenattion["seed"] = self.id
        self.represenattion["type"] = 0
        self.represenattion["loops"] = {}
        self.represenattion["loops"]["n"] = len(self.variables)
        loops_array = []

        for v in self.variables:
            loop_dict = {}
            loop_dict["id"] = v.id
            loop_dict["parent"] = v.parent
            loop_dict["position"] = 0  # TODO : We do not know what is that
            loop_dict["loop_it"] = v.id
            loop_dict["assignments"] = {}
            assignments_array = []
            for c in self.computations:
                for var in c.variables:
                    if var.id == v.id:
                        if v in c.variables:
                            assignments_array.append({"id": c.id, "position": c.variables.index(v)})
            loop_dict["assignments"]["n"] = len(assignments_array)
            loop_dict["assignments"]["assignments_array"] = assignments_array
            loops_array.append(loop_dict)

        self.represenattion["loops"]["loops_array"] = loops_array

        self.represenattion["computations"] = {}
        self.represenattion["computations"]["n"] = len(self.computations)
        computations_array = []
        for c in self.computations:
            computation_dict = {}
            computation_dict["comp_id"] = c.id
            computation_dict["lhs_data_type"] = c.dataType
            computation_dict["loop_iterators_ids"] = [v.id for v in c.variables]
            computation_dict["operations_histogram"] = [[0], ]  # TODO : We do not know what it is !
            computation_dict["rhs_accesses"] = {}
            accesses = []
            for id, m in c.accessMatrices:
                accesses.append({"comp_id": id, "access": m})

            computation_dict["rhs_accesses"]["n"] = len(accesses)
            computation_dict["rhs_accesses"]["accesses"] = accesses
            computations_array.append(computation_dict)

        self.represenattion["computations"]["computations_array"] = computations_array

        self.represenattion["inputs"] = {}
        self.represenattion["inputs"]["n"] = len(self.inputs)
        inputs_array = []
        for inp in self.inputs:
            inputs_array.append({"input_id": inp.id, "data_type": inp.dataType,
                                 "loop_iterators_ids": [v.id for v in inp.variables]})

        self.represenattion["inputs"]["inputs_array"] = inputs_array

        self.represenattion["iterators"] = {}
        self.represenattion["iterators"]["n"] = len(self.variables)
        self.represenattion["iterators"]["iterators_array"] = [{"it_id": v.id, "lower_bound": v.inf,
                                                                "upper_bound": v.sup.value} for v in self.variables]

        return self.represenattion

    def apply_schedule(self, schedule):
        if schedule is None:
            self.schedule_string = ""
            for c in self.computations:
                self.schedule_string = self.schedule_string + "    " + c.name + ".parallelize(" + self.variables[
                    0].name + ");\n"

            self.schedule_string = self.schedule_string + "\n"
        else:
#             print("********************")
#             print(schedule['variables'])

            schedule['repeated'] = False
            id = schedule["variables"][-1]
            for v in self.variables:
                if v.id == id:
                    last_shared_variable = v
            self.schedule_string = ""
            newVaribales = []
            for id in schedule["variables"]:
                missing = True
                for v in self.variables:
                    if v.id == id:
                        missing = False
                if missing:
                    new_v = Variable(id, None, None, False)
                    self.variables.append(new_v)
                    newVaribales.append(new_v)
            #         print(new_v,end=', ')
            # print(schedule["variables"])
            # print()
            if len(schedule["interchange_dims"]) > 1:
                id1, id2 = schedule["interchange_dims"]
                for v in self.variables:
                    if v.id == id1:
                        v1 = v
                    if v.id == id2:
                        v2 = v
                for c in self.computations:
                    if v1 in c.variables and v2 in c.variables:
                        self.schedule_string = self.schedule_string + "    " + c.name + ".interchange(" + v1.name + ", " + v2.name + ");\n"

            tiling = schedule["tiling"]
            if tiling is not None:
                ids = tiling["tiling_dims"]
                tilingFactors = tiling["tiling_factors"]
                tiledVariables = []

                for id in ids:
                    for v in self.variables:
                        if id == v.id:
                            v.tiled = True
                            tiledVariables.append(v)
                for c in self.computations:
                    valid = True
                    for ti in tiledVariables:
                        if ti not in c.variables:
                            valid = False
                    if valid:
                        self.schedule_string = self.schedule_string + "    " + c.name + ".tile("
                        for v in tiledVariables:
                            self.schedule_string = self.schedule_string + v.name + ", "

                        for factor in tilingFactors:
                            self.schedule_string = self.schedule_string + str(factor) + ", "

                        for v in newVaribales:
                            self.schedule_string = self.schedule_string + v.name + ", "

                        self.schedule_string = self.schedule_string[:-2] + ");\n"
            if schedule["unrolling_factor"] is not None:
                schedule['repeated'] = True
                id = schedule["variables"][-1]
                for v in self.variables:
                    if v.id == id:
                        vUnroll = v
                for c in self.computations:
                    if c.variables[-1].tiled:
                        self.schedule_string = self.schedule_string + "    " + c.name + ".unroll(" + vUnroll.name + \
                                               ", " + str(schedule["unrolling_factor"]) + ");\n"
                        schedule['repeated'] = False
                    elif c.variables[-1].schedulable:
                        if int(schedule["unrolling_factor"]) <= c.variables[-1].sup.value:
                            self.schedule_string = self.schedule_string + "    " + c.name + ".unroll(" + vUnroll.name + \
                                                   ", " + str(schedule["unrolling_factor"]) + ");\n"
                            schedule['repeated'] = False
                    else:  # we are sure that the tiling value is > unrolling factor
                        self.schedule_string = self.schedule_string + "    " + c.name + ".unroll(" + c.variables[-1].name + \
                                               ", " + str(schedule["unrolling_factor"]) + ");\n"
                        schedule['repeated'] = False
            id = schedule["variables"][0]
            for v in self.variables:
                if v.id == id:
                    vParallelize = v
            for c in self.computations:
                self.schedule_string = self.schedule_string + "    " + c.name + ".parallelize(" + vParallelize.name + ");\n"











    def apply_schedule_old(self, schedule):

        if schedule is None:
            self.schedule_string = ""
            for c in self.computations:
                self.schedule_string = self.schedule_string + "    " + c.name + ".parallelize(" + self.variables[
                    0].name + ");\n"

            self.schedule_string = self.schedule_string + "\n"
        else:
            self.schedule_string = ""
            self.new_variables = []
            for id in schedule["variables"]:
                missing = True
                for v in self.variables:
                    if v.id == id:
                        missing = False
                if missing:
                    new_v = Variable(id, None, None, True)
                    self.variables.append(new_v)
                    self.new_variables.append(new_v)

            id1, id2 = schedule["interchange_dims"]
            for v in self.variables:
                if v.id == id1:
                    v1 = v
                if v.id == id2:
                    v2 = v
            for c in self.computations:
                self.schedule_string = self.schedule_string + "    " + c.name + ".interchange(" + v1.name + ", " + v2.name + ");\n"

            tiling = schedule["tiling"]
            if tiling is not None:
                ids = tiling["tiling_dims"]
                tilingFactors = tiling["tiling_factors"]
                tiledVariables = []

                for id in ids:
                    for v in self.variables:
                        if id == v.id:
                            tiledVariables.append(v)
                for c in self.computations:
                    self.schedule_string = self.schedule_string + "    " + c.name + ".tile("
                    for v in tiledVariables:
                        self.schedule_string = self.schedule_string + v.name + ", "

                    for factor in tilingFactors:
                        self.schedule_string = self.schedule_string + str(factor) + ", "

                    for new_v in self.new_variables:
                        self.schedule_string = self.schedule_string + str(len(self.new_variables)) + ", "

                    self.schedule_string = self.schedule_string[:-2] + ");\n"
            if schedule["unrolling_factor"] is not None:
                id = schedule["variables"][-1]
                for v in self.variables:
                    if v.id == id:
                        vUnroll = v
                for c in self.computations:
                    self.schedule_string = self.schedule_string + "    " + c.name + ".unroll(" + vUnroll.name + \
                                           ", " + str(schedule["unrolling_factor"]) + ");\n"

            id = schedule["variables"][0]
            for v in self.variables:
                if v.id == id:
                    vParallelize = v
            for c in self.computations:
                self.schedule_string = self.schedule_string + "    " + c.name + ".parallelize(" + vParallelize.name + ");\n"
