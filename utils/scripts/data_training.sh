#!/bin/bash
 export DATA_DIRECTORY=/data/scratch/b_asma/tiramisu_extract/tiramisu/utils/code_generator-unrolling/cmake-build-debug/samples/
 export TIRAMISU_ROOT_EXECUT=/data/scratch/b_asma/tiramisu
 export TIRAMISU_ROOT_EXTRACT=/data/scratch/b_asma/tiramisu_extract/tiramisu
 FUNCTION_DIRECTORY=function
 SCHEDULE_DIRECTORY=schedule
 UNROLL_DIRECTORY=unroll
 FILE_NAME_EXEC='exec_times.txt'
 FILE_NAME_UNROLL_FACTORS=${DATA_DIRECTORY}"/unroll_factors.csv"
 filename_out=${DATA_DIRECTORY}"/output.txt"
 let nb_f=-1
 let nb_s=-1
 let nb_unr=0
 let best_unroll=0
 let best_time=99999999999

for f in `ls ${DATA_DIRECTORY}  | sort -V `; do  
    if [ -d "${DATA_DIRECTORY}/$f" ]; then
       cd  ${DATA_DIRECTORY}/$f
       echo "The function: "${PWD}
       let nb_f++
       let nb_s=-1
       for s in `ls ${DATA_DIRECTORY}/$f | sort -V` ; do
         if [ -d "${DATA_DIRECTORY}/$f/$s" ]; then
           cd ${DATA_DIRECTORY}/$f/$s
           echo "The schedule: " ${PWD}
           let nb_s++
           let nb_unr=0
           for u in `ls ${DATA_DIRECTORY}/$f/$s | sort -V `; do
               if [ -d "${DATA_DIRECTORY}/$f/$s/$u" ]; then
                   cd ${DATA_DIRECTORY}/$f/$s/$u
                   echo "Unrolling factor Exploration in the directory "${PWD}
                   export CLEANINR_DIRECTORY=${PWD}/cleaner
                   export ACTUAL_PROGRAM=${FUNCTION_DIRECTORY}${nb_f}_${SCHEDULE_DIRECTORY}_${nb_s}_${UNROLL_DIRECTORY}_${nb_unr}
                    for i in {1..3}; do
                       mkdir cleaner      
                       if [[ ("$nb_unr" = "0") && ("$i" = "1" ) ]];then
                          echo " Extracting Features of the function " ${DATA_DIRECTORY}/$f "cpt" $i
                          TIRAMISU_ROOT=${TIRAMISU_ROOT_EXTRACT}
                       else 
                          echo " Executing " ${ACTUAL_PROGRAM}  "cpt" $i
                          TIRAMISU_ROOT=${TIRAMISU_ROOT_EXECUT}
                        fi
                       # compile the generator.cpp 
                       g++ -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o ${CLEANINR_DIRECTORY}/${ACTUAL_PROGRAM}_generator -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ${ACTUAL_PROGRAM}.cpp -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm  &>${filename_out}
                       # generating the file ${ACTUAL_PROGRAM}.o
                       cd ${CLEANINR_DIRECTORY}
                      ./${ACTUAL_PROGRAM}_generator &>${filename_out}
                       # compile the wrapper code and link it to the generated object file.
                       g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o ${ACTUAL_PROGRAM}_wrapper -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build  ../${ACTUAL_PROGRAM}_wrapper.cpp  ${CLEANINR_DIRECTORY}/${ACTUAL_PROGRAM}.o -ltiramisu -lHalide -ldl -lpthread -lz -lm &>${filename_out}
                       # Run the wrapper
                       ./${ACTUAL_PROGRAM}_wrapper &>${filename_out}
                       cd ..
                       rm -rf ${CLEANINR_DIRECTORY}
                    done
                    let n=0
                    let sum=0
                    while read line; do
                    # reading each line
                    echo ${line}
                    sum=$(echo $sum + $line | bc -l) 
                    echo $sum
                    let n++
                    done < $FILE_NAME_EXEC
                    #rm $FILE_NAME_EXEC
                    sum=$(echo $sum / $n | bc -l) 
                    echo "Moyenne" $sum
                    if (( $(echo "${best_time} > $sum" |bc -l) )); then
                       best_time=${sum} 
                       best_unroll=${nb_unr}
                    fi
                   let nb_unr++
                fi            
            done
            echo "write the best unroll factor for the the function" ${nb_f} "_the schedule_ " ${nb_s}
            echo ${best_unroll} >>${FILE_NAME_UNROLL_FACTORS}
            let best_unroll=0
            let best_time=99999999999
          fi  
       done
     fi  
done

