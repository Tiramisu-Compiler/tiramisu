CUDA_ROOT = /usr/local/cuda
PPCG = ../../software/ppcg/ppcg

run_sgemm: sgemm
	./sgemm

sgemm: sgemm_host.cu sgemm_kernel.o
	${CUDA_ROOT}/bin/nvcc sgemm_host.cu sgemm_kernel.o -o sgemm

sgemm_kernel.o: sgemm_kernel.hu sgemm_kernel.cu
	${CUDA_ROOT}/bin/nvcc sgemm_kernel.cu -c sgemm_kernel.o

sgemm_host.cu sgemm_kernel.hu sgemm_kernel.cu: sgemm.c
	${PPCG} --target=cuda sgemm.c

clean:
	-rm sgemm_host.cu sgemm_kernel.hu sgemm_kernel.cu sgemm_kernel.o sgemm
