#! /usr/bin/python

import os

for B0 in [8,16,32,64,128,256]:
    for B1 in [8,16,32,64,128,256]:
	for B2 in [8,16,32,64,128,256]:
	    for L3_B0 in [2]: #,4,8,16]:
		for L3_B1 in [2]: #,4,8,16]:
		    for L3_B2 in [2]: #,4,8,16]:
			    file = open("sgemm/SCHEDULE.h","w") 
			    file.write("#define B0 " + str(B0) + "\n")
			    file.write("#define B1 " + str(B1) + "\n")
		    	    file.write("#define B2 " + str(B2) + "\n")
			    file.write("#define L3_B0 " + str(L3_B0) + "\n")
			    file.write("#define L3_B1 " + str(L3_B1) + "\n")
		    	    file.write("#define L3_B2 " + str(L3_B2) + "\n")
			    file.close()
			    sizes = "B0 = " + str(B0) + ", B1 = " + str(B1) + ", B2 = " + str(B2) + ", "
			    sizes += "L3_B0 = " + str(L3_B0) + ", L3_B1 = " + str(L3_B1) + ", L3_B2 = " + str(L3_B2)
			    os.system("")
			    os.system("echo " + sizes)
			    os.system("./compile_and_run_benchmarks.sh sgemm")
			    os.system("echo ------------------------------------------------------- ")
