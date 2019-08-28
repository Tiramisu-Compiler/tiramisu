printf "" > log.txt
for FIN_BLOCKING in 2 4 8 16 32; do
	for FOUT_BLOCKING in 2 4 8 16 32; do
		for X_BLOCKING in 3 5 9; do
			printf "#ifdef FIN_BLOCKING\n				#undef FIN_BLOCKING\n#endif\n" > ./param_tuning.h
			printf "#define FIN_BLOCKING $FIN_BLOCKING\n" >> ./param_tuning.h
			printf "#ifdef FOUT_BLOCKING\n				#undef FOUT_BLOCKING\n#endif\n" >> ./param_tuning.h
			printf "#define FOUT_BLOCKING $FOUT_BLOCKING\n" >> ./param_tuning.h
			printf "#ifdef X_BLOCKING\n				#undef X_BLOCKING\n#endif\n" >> ./param_tuning.h
			printf "#define X_BLOCKING $X_BLOCKING\n" >> ./param_tuning.h
			printf "" >> ./param_tuning.h

			printf "FIN_BLOCKING=$FIN_BLOCKING, FOUT_BLOCKING=$FOUT_BLOCKING, X_BLOCKING=$X_BLOCKING, ";
			printf "FIN_BLOCKING=$FIN_BLOCKING, FOUT_BLOCKING=$FOUT_BLOCKING, X_BLOCKING=$X_BLOCKING, " >> log.txt;
			cd ../../../../../../build/benchmarks/DNN/blocks/fusedresNet_inference/cpu/dense
			make  > /dev/null 2>&1;
			cd ../../../../../../../benchmarks/DNN/blocks/fusedresNet_inference/cpu/dense;
			./wrapper_nn_block_fused_resnet_inference |tee -a log.txt;
			./clean.sh;
			cd .;
		done
	done
done
