#!/usr/bin/python

import numpy as np
import sys

RARE_CASES=True
ANALYSIS_1=False
ANALYSIS_2=True

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


def count_pattern(d, pattern):
    nb_pat = 0
    for i in range(0,32):
	for j in range(0,32):
	    s = d[i,j]
	    if array_has_pattern(s, pattern) == True:
		nb_pat += 1;
    return nb_pat;

def count_pattern_per_output_channel(d, pattern):
    nb_pat = 32
    for i in range(0,32):
	old_nb_pat = nb_pat;
	new_nb_pat = 0;
	for j in range(0,32):
	    s = d[i,j]
	    if array_has_pattern(s, pattern) == True:
		new_nb_pat += 1;
	print("Patterns in output channel " + str(i) + " : " + str(new_nb_pat));
	nb_pat = min(old_nb_pat, new_nb_pat)
    return nb_pat;

def count_patterns(d, patterns, excluded_patterns):
    print("------ Counting patterns ------ \n")
    nb_pat = 0
    for p in patterns:
	nb_pat += count_pattern(d, p)
    for excluded in excluded_patterns:
	nb_pat -= count_pattern(d, excluded)
    print("Patterns included:\n")
    for p in patterns:
	print(np.array2string(p))
    print("\nPatterns excluded:\n")
    for e in excluded_patterns:
	print(np.array2string(e))
    print("\nNumber of patterns: " + str(nb_pat) + "/" + str(32*32) + " = " + str((nb_pat*100)/(32*32)) + "%\n")

def count_patterns_per_output_channel(d, patterns, excluded_patterns):
    print("------ Counting patterns per output channel ------ \n")
    print("Patterns included:\n")
    for p in patterns:
	print(np.array2string(p))
    print("\nPatterns excluded:\n")
    for e in excluded_patterns:
	print(np.array2string(e))
    nb_pat = 0
    for p in patterns:
	nb_pat += count_pattern_per_output_channel(d, p)
    for excluded in excluded_patterns:
	nb_pat -= count_pattern_per_output_channel(d, excluded)
    print("\nNumber of patterns per output channel: " + str(nb_pat) + "/" + str(32) + " = " + str((nb_pat*100)/(32)) + "%\n")


def main():
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    if len(sys.argv) > 1:
	data_file = sys.argv[1]
    else:
	data_file = 'resnet_10.npy'

    d = np.load(data_file)

    print("--------------------------------------------------------------------------------")
    print("Analyzing file: " + data_file)
    print("--------------------------------------------------------------------------------")

    if (ANALYSIS_1):
	count_patterns(d, [np.array([[0,0,0],
				   [0,0,0],
				   [0,0,0]])],
			  [])

	count_patterns(d, [np.array([[-1,-1,-1],
				     [ 0, 0, 0],
				     [ 0, 0, 0]])],
			  [np.array([[0, 0, 0],
				     [0, 0, 0],
				     [0, 0, 0]])])

	count_patterns(d, [np.array([[ 0, 0, 0],
				     [-1,-1,-1],
				     [ 0, 0, 0]])],
			  [np.array([[0, 0, 0],
				     [0, 0, 0],
				     [0, 0, 0]])])

	count_patterns(d, [np.array([[ 0, 0, 0],
				     [ 0, 0, 0],
				     [-1,-1,-1]])],
			  [np.array([[0, 0, 0],
				     [0, 0, 0],
				     [0, 0, 0]])])

	if (RARE_CASES):
	    count_patterns(d, [np.array([[1,0,0],
					 [0,0,0],
					 [0,0,0]])],
				    [])
	    count_patterns(d, [np.array([[0,1,0],
					 [0,0,0],
					 [0,0,0]])],
				    [])
	    count_patterns(d, [np.array([[0,0,1],
					 [0,0,0],
					 [0,0,0]])],
				    [])
	    count_patterns(d, [np.array([[0,0,0],
					 [1,0,0],
					 [0,0,0]])],
				    [])
	    count_patterns(d, [np.array([[0,0,0],
					 [0,1,0],
					 [0,0,0]])],
				    [])
	    count_patterns(d, [np.array([[0,0,0],
					 [0,0,1],
					 [0,0,0]])],
				    [])
	    count_patterns(d, [np.array([[0,0,0],
					 [0,0,0],
					 [1,0,0]])],
				    [])
	    count_patterns(d, [np.array([[0,0,0],
					 [0,0,0],
					 [0,1,0]])],
				    [])
	    count_patterns(d, [np.array([[0,0,0],
					 [0,0,0],
					 [0,0,1]])],
				    [])

	    count_patterns(d, [np.array([[0,0,0],
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

	    count_patterns(d, [np.array([[-1,-1,-1],
					 [ 0, 0, 0],
					 [ 0, 0, 0]])],
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

	    count_patterns(d, [np.array([[ 0, 0, 0],
					 [-1,-1,-1],
					 [ 0, 0, 0]])],
			      [np.array([[ 0, 0, 0],
					 [ 1, 0, 0],
					 [ 0, 0, 0]]),
			       np.array([[ 0, 0, 0],
					 [ 0, 1, 0],
					 [ 0, 0, 0]]),
			       np.array([[ 0, 0, 0],
					 [ 0, 0, 1],
					 [ 0, 0, 0]]),
			       np.array([[ 0, 0, 0],
					 [ 0, 0, 0],
					 [ 0, 0, 0]])
			       ])

	    count_patterns(d, [np.array([[ 0, 0, 0],
					 [ 0, 0, 0],
					 [-1,-1,-1]])],
			      [np.array([[ 0, 0, 0],
					 [ 0, 0, 0],
					 [ 1, 0, 0]]),
			       np.array([[ 0, 0, 0],
					 [ 0, 0, 0],
					 [ 0, 1, 0]]),
			       np.array([[ 0, 0, 0],
					 [ 0, 0, 0],
					 [ 0, 0, 1]]),
			       np.array([[ 0, 0, 0],
					 [ 0, 0, 0],
					 [ 0, 0, 0]])
			       ])



    if (ANALYSIS_2):
	count_patterns_per_output_channel(d, [np.array([[0,0,0],
							[0,0,0],
							[0,0,0]])],
					     [])

	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[ 0, 0, 0],
							[ 0, 0, 0]])],
					     [np.array([[0, 0, 0],
							[0, 0, 0],
							[0, 0, 0]])])

	count_patterns_per_output_channel(d, [np.array([[ 0, 0, 0],
							[-1,-1,-1],
							[ 0, 0, 0]])],
					     [np.array([[0, 0, 0],
							[0, 0, 0],
							[0, 0, 0]])])

	count_patterns_per_output_channel(d, [np.array([[ 0, 0, 0],
							[ 0, 0, 0],
							[-1,-1,-1]])],
					    [np.array([[0, 0, 0],
						       [0, 0, 0],
						       [0, 0, 0]])])


main()
#print(d)
