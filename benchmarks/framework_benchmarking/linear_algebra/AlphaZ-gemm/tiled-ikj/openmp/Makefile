include ../../../configure.sh

CFLAGS=-O3 -I/usr/include/malloc/ -I${OPENMP_DIR}/include/libiomp/ -L${OPENMP_DIR}/lib/ -l${OPENMP_LIB}
LIBRARIES=
OBJS = gemm.o 
all: plain check

debug: CFLAGS =-DDEBUG -g -Wall -Wextra -std=c99 -I/usr/include/malloc/
debug: all
		
plain: $(OBJS)
	$(CC) gemm-wrapper.c -o gemm $(OBJS) $(CFLAGS) $(LIBRARIES)

check: $(OBJS)
	$(CC) gemm-wrapper.c -o gemm.check $(OBJS) $(CFLAGS) $(LIBRARIES) -DCHECKING 

verify: $(OBJS) gemm_verify.o
	$(CC) gemm-wrapper.c -o gemm.verify $(OBJS) gemm_verify.o $(CFLAGS) $(LIBRARIES) -DVERIFY

verify-rand: $(OBJS) gemm_verify.o
	$(CC) gemm-wrapper.c -o gemm.verify-rand $(OBJS) gemm_verify.o $(CFLAGS) $(LIBRARIES) -DVERIFY -DRANDOM

gemm.o : gemm.c
	$(CC) gemm.c -o gemm.o $(CFLAGS) $(LIBRARIES) -c

gemm_verify.o : gemm_verify.c
	$(CC) gemm_verify.c -o gemm_verify.o $(CFLAGS) $(LIBRARIES) -c

clean:
	rm -f *.o gemm gemm.check gemm.verify gemm.verify-rand
