printf "" > log.txt
LAST_BEST=1000000000
echo "Running MKLDNN Version : "
./compile_and_run_mkldnn.sh
echo "Tuning Hyperparameters : "
for X_BL1 in 7 14 28; do
	for X_BL2 in 7 14 28; do
		for Y_BL1 in 2 7; do
			for Y_BL2 in 2 7; do
				for FOUT_BL in 2 4 8 16 32 64; do
					printf "#ifdef X_BL1\n	#undef X_BL1\n#endif\n" > ./param_tuning.h
					printf "#define X_BL1 $X_BL1\n" >> ./param_tuning.h
					printf "#ifdef X_BL2\n	#undef X_BL2\n#endif\n" >> ./param_tuning.h
					printf "#define X_BL2 $X_BL2\n" >> ./param_tuning.h
					printf "#ifdef Y_BL1\n	#undef Y_BL1\n#endif\n" >> ./param_tuning.h
					printf "#define Y_BL1 $Y_BL1\n" >> ./param_tuning.h
					printf "#ifdef Y_BL2\n	#undef Y_BL2\n#endif\n" >> ./param_tuning.h
					printf "#define Y_BL2 $Y_BL2\n" >> ./param_tuning.h
					printf "#ifdef FOUT_BL\n	#undef FOUT_BL\n#endif\n" >> ./param_tuning.h
					printf "#define FOUT_BL $FOUT_BL\n" >> ./param_tuning.h
					printf "" >> ./param_tuning.h

					printf "X_BL1=$X_BL1, X_BL2=$X_BL2, Y_BL1=$Y_BL1, Y_BL2=$Y_BL2, FOUT_BL=$FOUT_BL, " >> log.txt;
					cd ../../../../../../build/benchmarks/DNN/blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_256_512_28
					make  > /dev/null 2>&1;
					cd ../../../../../../../benchmarks/DNN/blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_256_512_28;
					execution_output=$(./wrapper_sparse_conv_relu_conv_relu_256_512_28);

					execution_time=$(echo $execution_output | cut -f2 -d,)
					correctness=$(echo $execution_output | cut -f3 -d,)
					# Check if the output is correct and if the execution time is better
					if [ $correctness -eq "1" ]; then
					  LAST_BEST=$(echo $execution_time $LAST_BEST | awk '{if ($1 < $2) print $1; else print $2}')
						if [ $LAST_BEST == $execution_time ]; then
							cp param_tuning.h param_best.h
							echo "=====+>BETTER EXECUTION TIME FOUND FOR PARAMETERS : "
							printf "X_BL1=$X_BL1, X_BL2=$X_BL2, Y_BL1=$Y_BL1, Y_BL2=$Y_BL2, FOUT_BL=$FOUT_BL ||| ";
							echo "$LAST_BEST ms"
							echo ""
						fi
					fi

					cd .;
				done
			done
		done
	done
done
cp ./param_best.h ./param_tuning.h
# Compile code with best parameters
cd ../../../../../../build/benchmarks/DNN/blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_256_512_28
make  > /dev/null 2>&1;
cd ../../../../../../../benchmarks/DNN/blocks/Conv-ReLU-Conv-ReLU/cpu/sparse_32channels_256_512_28;
rm ./param_best.h
