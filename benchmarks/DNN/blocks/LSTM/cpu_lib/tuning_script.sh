printf "" > log.txt
for GEMM_BATCH in 1 2 4 5 10 20 25 50 100; do
	for VEC_LEN in 2 4 8 16 32; do
		printf "#ifdef GEMM_BATCH\n			#undef GEMM_BATCH\n#endif\n" > ./param_tuning.h
		printf "#define GEMM_BATCH $GEMM_BATCH\n" >> ./param_tuning.h
		printf "#ifdef VEC_LEN\n			#undef VEC_LEN\n#endif\n" >> ./param_tuning.h
		printf "#define VEC_LEN $VEC_LEN\n" >> ./param_tuning.h
		printf "" >> ./param_tuning.h

		printf "GEMM_BATCH=$GEMM_BATCH, VEC_LEN=$VEC_LEN, ";
		printf "GEMM_BATCH=$GEMM_BATCH, VEC_LEN=$VEC_LEN, " >> log.txt;
		cd ../../../../../build/benchmarks/DNN/blocks/LSTM/cpu_lib
		make  > /dev/null 2>&1;
		cd ../../../../../../benchmarks/DNN/blocks/LSTM/cpu_lib;
		./wrapper_lstm |tee -a log.txt;
		./clean.sh;
		cd .;
	done
done
