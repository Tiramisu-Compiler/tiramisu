printf "" > log.txt
for X1_BLOCKING in 4 7 8 14 16; do
	for X2_BLOCKING in 3 5 9; do
		printf "#ifdef X1_BLOCKING\n			#undef X1_BLOCKING\n#endif\n" > ./param_tuning.h
		printf "#define X1_BLOCKING $X1_BLOCKING\n" >> ./param_tuning.h
		printf "#ifdef X2_BLOCKING\n			#undef X2_BLOCKING\n#endif\n" >> ./param_tuning.h
		printf "#define X2_BLOCKING $X2_BLOCKING\n" >> ./param_tuning.h
		printf "" >> ./param_tuning.h

		printf "X1_BLOCKING=$X1_BLOCKING, X2_BLOCKING=$X2_BLOCKING, ";
		printf "X1_BLOCKING=$X1_BLOCKING, X2_BLOCKING=$X2_BLOCKING, " >> log.txt;
		cd ../../../../../../build/benchmarks/DNN/blocks/vggBlock/cpu/dense
		make  > /dev/null 2>&1;
		cd ../../../../../../../benchmarks/DNN/blocks/vggBlock/cpu/dense;
		./wrapper_nn_block_vgg |tee -a log.txt;
		./clean.sh;
		cd .;
	done
done
