printf "" > log.txt
for FOUT_BLOCKING in 2 4 8 16; do
	for X_BLOCKING in 2 4 8 16; do
		printf "#ifdef FOUT_BLOCKING\n			#undef FOUT_BLOCKING\n#endif\n" > ./param_tuning.h
		printf "#define FOUT_BLOCKING $FOUT_BLOCKING\n" >> ./param_tuning.h
		printf "#ifdef X_BLOCKING\n			#undef X_BLOCKING\n#endif\n" >> ./param_tuning.h
		printf "#define X_BLOCKING $X_BLOCKING\n" >> ./param_tuning.h
		printf "" >> ./param_tuning.h

		printf "FOUT_BLOCKING=$FOUT_BLOCKING, X_BLOCKING=$X_BLOCKING, ";
		printf "FOUT_BLOCKING=$FOUT_BLOCKING, X_BLOCKING=$X_BLOCKING, " >> log.txt;
		cd ../../../../../../build/benchmarks/DNN/layers/convolution/direct/cpu_3channels
		make  > /dev/null 2>&1;
		cd ../../../../../../../benchmarks/DNN/layers/convolution/direct/cpu_3channels;
		./wrapper_conv_layer_3channels |tee -a log.txt;
		./clean.sh;
		cd .;
	done
done
