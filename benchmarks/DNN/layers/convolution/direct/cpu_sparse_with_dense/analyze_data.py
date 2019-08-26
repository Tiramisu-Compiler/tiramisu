#!/usr/bin/python

import numpy as np
import sys

Channels = 32

RARE_CASES=False
ANALYSIS_0=True
ANALYSIS_1=False
ANALYSIS_2=False
ANALYSIS_3=False
ANALYSIS_4=False
ANALYSIS_5=False
ANALYSIS_6=False
ANALYSIS_7=False
ANALYSIS_8=False
ANALYSIS_9=False
ANALYSIS_10=False
ANALYSIS_11=False

# Format is [OutChannels, InChannels, 3, 3]
PRODUCE_CSV_FRIENDLY_OUTPUT=True

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
    for i in range(0,Channels):
	for j in range(0,Channels):
	    s = d[i,j]
	    if array_has_pattern(s, pattern) == True:
		nb_pat += 1;
    return nb_pat;

def count_pattern_per_output_channel(d, pattern):
    nb_pat = Channels
    final_histogram_of_patterns_per_input_channel = [0] * Channels
    for i in range(0,Channels): # for each output channel
	histogram_of_patterns_per_input_channel = [0] * Channels
	old_nb_pat = nb_pat;
	new_nb_pat = 0;
	for j in range(0,Channels): # for each input channel
	    s = d[i,j]
	    if array_has_pattern(s, pattern) == True:
		new_nb_pat += 1;
		histogram_of_patterns_per_input_channel[j] += 1;
		final_histogram_of_patterns_per_input_channel[j] += 1;
	if (PRODUCE_CSV_FRIENDLY_OUTPUT == False):
	    print("Patterns in output channel " + str(i) + " : " + str(new_nb_pat));
	nb_pat = min(old_nb_pat, new_nb_pat)
	if (PRODUCE_CSV_FRIENDLY_OUTPUT == False):
	    print("Histogram of patterns for output channel " + str(i) + ": ");
	for j in range(0,Channels):
	    print(str(histogram_of_patterns_per_input_channel[j]) + ";"),
	print("")
    print("\nFinal histogram of patterns for all input channels: ");
    for j in range(0,Channels):
	    print(str(final_histogram_of_patterns_per_input_channel[j]) + " "),
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
    print("\nNumber of patterns: " + str(nb_pat) + "/" + str(Channels*Channels) + " = " + str((nb_pat*100)/(Channels*Channels)) + "%\n")

def count_patterns_per_output_channel(d, patterns, excluded_patterns):
    print("------ Counting patterns per output channel ------ \n")
    print("Patterns included:\n")
    for p in patterns:
	print(np.array2string(p))
    print("\nPatterns excluded:\n")
    for e in excluded_patterns:
	print(np.array2string(e))
    nb_pat = 0
    print("\nPattern CSV:\n")
    for p in patterns:
	nb_pat += count_pattern_per_output_channel(d, p)
    for excluded in excluded_patterns:
	nb_pat -= count_pattern_per_output_channel(d, excluded)
    print("\nNumber of patterns per output channel: " + str(nb_pat) + "/" + str(Channels) + " = " + str((nb_pat*100)/(Channels)) + "%\n")


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

    if (ANALYSIS_0):
	count_patterns_per_output_channel(d, [np.array([[0,0,0],
						        [0,0,0],
						        [0,0,0]])],
						       [])

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
    if (ANALYSIS_3):
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[ 0, 0, 0],
							[ 0, 0, 0]])],
					     [])

    if (ANALYSIS_4):
	count_patterns_per_output_channel(d, [np.array([[ 0, 0, 0],
							[-1,-1,-1],
							[ 0, 0, 0]])],
					     [])

    if (ANALYSIS_5):
	count_patterns_per_output_channel(d, [np.array([[ 0, 0, 0],
							[ 0, 0, 0],
							[-1,-1,-1]])],
					     [])

    if (ANALYSIS_6):
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[-1,-1,-1],
							[ 0, 0, 0]])],
					     [])

    if (ANALYSIS_7):
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[ 0, 0, 0],
							[-1,-1,-1]])],
					     [])

    if (ANALYSIS_8):
	count_patterns_per_output_channel(d, [np.array([[ 0, 0, 0],
							[-1,-1,-1],
							[-1,-1,-1]])],
					     [])

    if (ANALYSIS_9):
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
    if (ANALYSIS_10):
	count_patterns_per_output_channel(d, [np.array([[-1, 0, 0],
							[-1,-1,-1],
							[-1,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[0, -1, 0],
							[-1,-1,-1],
							[-1,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[ 0, 0,-1],
							[-1,-1,-1],
							[-1,-1,-1]])],
					     [])
    if (ANALYSIS_11):
	count_patterns_per_output_channel(d, [np.array([[-1, 0,-1],
							[-1,-1,-1],
							[-1,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[-1,-1, 0],
							[-1,-1,-1],
							[-1,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[ 0,-1,-1],
							[-1,-1,-1],
							[-1,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[-1, 0,-1],
							[-1,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[ 0,-1,-1],
							[-1,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[-1,-1, 0],
							[-1,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[-1,-1,-1],
							[ 0,-1,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[-1,-1,-1],
							[-1, 0,-1]])],
					     [])
	count_patterns_per_output_channel(d, [np.array([[-1,-1,-1],
							[-1,-1,-1],
							[-1,-1, 0]])],
					     [])








main()
#print(d)
