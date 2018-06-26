#! /usr/bin/python

import os
import sys

SCHEDULE_FILE="linear_algebra/blas/level3/sgemm/SCHEDULE.h"
COMPILATION_COMMAND="./compile_and_run_benchmarks.sh linear_algebra/blas/level3/sgemm/ sgemm"

for B0 in [32,64]: #128
    for B1 in [32,64]: #128
	for B2 in [32,64]:
	    for L3_B0 in [2,4,8,16]:
		for L3_B1 in [2,4,8,16]:
		    for L3_B2 in [2,4,8,16]:
			for U1 in [16,32,64]: #128
				try:
					file = open(SCHEDULE_FILE,"w") 
					file.write("#define B0 " + str(B0) + "\n")
					file.write("#define B1 " + str(B1) + "\n")
					file.write("#define B2 " + str(B2) + "\n")
					file.write("#define L3_B0 " + str(L3_B0) + "\n")
					file.write("#define L3_B1 " + str(L3_B1) + "\n")
					file.write("#define L3_B2 " + str(L3_B2) + "\n")
					file.write("#define U1 " + str(U1) + "\n")
					file.close()
					sizes = "B0 = " + str(B0) + ", B1 = " + str(B1) + ", B2 = " + str(B2) + ", "
					sizes += "L3_B0 = " + str(L3_B0) + ", L3_B1 = " + str(L3_B1) + ", L3_B2 = " + str(L3_B2) + ", "
					sizes += "U1 = " + str(U1)
					os.system("")
					os.system("echo " + sizes)
					os.system(COMPILATION_COMMAND)
					os.system("echo ------------------------------------------------------- ")
				except KeyboardInterrupt:
					print "Ctrl-c pressed ..."
				        sys.exit(1)
					
