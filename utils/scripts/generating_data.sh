#!/bin/bash
# This script allow generating confugurated Data_set using code_generator-unrolling 
# Number of functions for each data_type (int or float )
# bach size for each compute node 
export DATA_DIRECTORY=/data/scratch/b_asma/tiramisu_Extraction/tiramisu/utils/code_generator-unrolling/cmake-build-debug/
#int baches
echo "****** INT BACHES ******"
export INT_BACH_NUMBER=5
export INT_BACH_SIZE=25
export INT_TYPE_NAME_GENERATOR=p_int32
export INT_TYPE_NAME_WRAPPER=int32_t
let CODE_FROM=-${INT_BACH_SIZE}
cd ${DATA_DIRECTORY}
echo "nb_codes : \"${INT_BACH_SIZE}\"
nb_stages (number of computations) : \"1\"
default_type_tiramisu : \"${INT_TYPE_NAME_GENERATOR}\"
default_type_warpper : \"${INT_TYPE_NAME_WRAPPER}\"
assignment_prob : \"0.2\"
assignment_input_prob : \"0.6\"
max_nb_dims (number of nested loops) : \"4\"
max_nb_inputs : \"10\"
max_offset (stencils) : \"2\"
convolutions_prob : \"0\"
same_padding_prob (convolutions): \"0.1\"
tile_sizes : \"32 64 128\"
unrolling_factors : \"all\"
scheduling_commands : \"interchanging tiling unrolling\"
shedules (all, random) : \"all\"
nb_rand_schedules : \"10\"" &>'inputs.txt'
for (( i=0; i<$INT_BACH_NUMBER; i++ ))
do
 let CODE_FROM=CODE_FROM+INT_BACH_SIZE
 echo ${CODE_FROM} |./restructured 
 echo "${INT_BACH_SIZE} Codes was generated from the Function${CODE_FROM} "
 mv samples samples${CODE_FROM}
 done
 let CODE_FROM=CODE_FROM+INT_BACH_SIZE
#float baches 
echo "****** FLOAT BACHES ******"
export FLOAT_BACH_NUMBER=5
export FLOAT_BACH_SIZE=25
export FLOAT_TYPE_NAME_GENERATOR=p_float32
export FLOAT_TYPE_NAME_WRAPPER=float
let CODE_FROM=$CODE_FROM-${FLOAT_BACH_SIZE}
cd ${DATA_DIRECTORY}
echo "nb_codes : \"${FLOAT_BACH_SIZE}\"
nb_stages (number of computations) : \"1\"
default_type_tiramisu : \"${FLOAT_TYPE_NAME_GENERATOR}\"
default_type_warpper : \"${FLOAT_TYPE_NAME_WRAPPER}\"
assignment_prob : \"0.2\"
assignment_input_prob : \"0.6\"
max_nb_dims (number of nested loops) : \"4\"
max_nb_inputs : \"10\"
max_offset (stencils) : \"2\"
convolutions_prob : \"0\"
same_padding_prob (convolutions): \"0.1\"
tile_sizes : \"32 64 128\"
unrolling_factors : \"all\"
scheduling_commands : \"interchanging tiling unrolling\"
shedules (all, random) : \"all\"
nb_rand_schedules : \"10\"" &>'inputs.txt'
for (( i=0; i<$FLOAT_BACH_NUMBER; i++ ))
do
 let CODE_FROM=CODE_FROM+FLOAT_BACH_SIZE
 echo ${CODE_FROM} |./restructured 
 echo "${FLOAT_BACH_SIZE} Codes was generated from the Function${CODE_FROM} "
 mv samples samples${CODE_FROM}
 done


#float baches 