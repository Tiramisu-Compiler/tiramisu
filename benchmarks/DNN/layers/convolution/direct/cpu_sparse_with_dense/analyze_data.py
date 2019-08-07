#!/usr/bin/python

import numpy as np
import sys


def is_zero(s):
    is_zeroo = True
    for i in range(0,3):
	for j in range(0,3):
	    if s[i, j] != 0:
		is_zeroo = False
    return is_zeroo;


def array_has_pattern(s, pattern):
    is_pat = True
    for i in range(0,3):
	for j in range(0,3):
	    if (pattern[i,j] == -1):
		continue;
	    if (pattern[i,j] == 0):
		if (s[i,j] != 0):
		    is_pat = False
	    if (pattern[i,j] != 0):
		if (s[i,j] == 0):
		    is_pat = False
    return is_pat;


def count_pattern(d, pattern, excluded_patterns):
    nb_pat = 0
    for i in range(0,32):
	for j in range(0,32):
	    s = d[i,j]
	    if array_has_pattern(s, pattern) == True:
		if (len(excluded_patterns) == 0):
			nb_pat += 1;
		else:
		    for excluded in excluded_patterns:
			if array_has_pattern(s, excluded) == False:
			    nb_pat += 1;
    print("Pattern:\n" + np.array2string(pattern))
    print("Count = " + str(nb_pat) + "/" + str(32*32) + " = " + str((nb_pat*100)/(32*32)) + "%\n")
    return nb_pat;


def count_pattern_union(d, patterns, excluded_patterns):
    print("------ Counting number of occurences of a union of patterns ------ \n")
    nb_pat = 0
    for p in patterns:
	nb_pat += count_pattern(d, p, excluded_patterns)
    print("Number of occurences of a union of patterns:\n" + str(nb_pat) + "/" + str(32*32) + " = " + str((nb_pat*100)/(32*32)) + "%\n")


def main():
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    if len(sys.argv) > 1:
	data_file = sys.argv[1]
    else:
	data_file = 'resnet_10.npy'

    d = np.load(data_file)

    print("Analyzing file: " + data_file)
    print("-------------------------------")

    count_pattern(d, np.array([[0,0,0],
			       [0,0,0],
			       [0,0,0]]), [])

    count_pattern_union(d, [np.array([[0,0,0],
				      [0,0,0],
				      [0,0,1]]),
			    np.array([[0,0,0],
				      [0,0,0],
				      [0,1,0]]),
			    np.array([[0,0,0],
				     [0,0,0],
				     [1,0,0]]),
	                    np.array([[0,0,0],
				      [0,0,1],
				      [0,0,0]]),
			    np.array([[0,0,0],
				      [0,1,0],
				      [0,0,0]]),
			    np.array([[0,0,0],
				      [1,0,0],
				      [0,0,0]]),
		            np.array([[0,0,1],
				      [0,0,0],
				      [0,0,0]]),
			    np.array([[0,1,0],
				      [0,0,0],
				      [0,0,0]]),
			    np.array([[1,0,0],
				      [0,0,0],
				      [0,0,0]])],
			    [])

    count_pattern(d, np.array([[-1,-1,-1],
			       [ 0, 0, 0],
			       [ 0, 0, 0]]),
		    [np.array([[ 0, 0, 0],
			       [ 0, 0, 0],
			       [ 0, 0, 0]]),
		     np.array([[ 1, 0, 0],
			       [ 0, 0, 0],
			       [ 0, 0, 0]]),
		     np.array([[ 0, 1, 0],
			       [ 0, 0, 0],
			       [ 0, 0, 0]]),
		     np.array([[ 0, 0, 1],
			       [ 0, 0, 0],
			       [ 0, 0, 0]])
		    ])

    count_pattern(d, np.array([[0,0,0],
			       [-1,-1,-1],
			       [0,0,0]]),
		    [])

    count_pattern(d, np.array([[0,0,0],
			       [0,0,0],
			       [-1,-1,-1]]),
		    [])

main()
#print(d)
