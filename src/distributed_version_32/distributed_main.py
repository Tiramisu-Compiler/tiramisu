#!/usr/bin/python

import islpy as isl
import distributed_module as dist


def mat_mul_2D():
	S0_dom = isl.Set("[N]->{S0[i, j, k]: 0 <= i < N and 0 <= j < N and 0 <= k < N}");
	Access_A = isl.Map("{S0[i, j, k]->A[i, k]}")
	Access_B = isl.Map("{S0[i, j, k]->B[k, j]}")
	Access_C = isl.Map("{S0[i, j, k]->C[i, j, k]}")
	A = isl.Set("[N]->{A[i,j]: 0<=i<N and 0<=j<N}")
	B = isl.Set("[N]->{B[i,j]: 0<=i<N and 0<=j<N}")
	C = isl.Set("[N]->{C[i,j,k]: 0<=i<N and 0<=j<N and 0<=k<N}")
	return {'S0_dom':S0_dom, 'Access_A':Access_A, 'Access_B':Access_B, 'Access_C':Access_C, 'A':A, 'B':B, 'C':C}

def mat_mul_2D_fixed():
	S0_dom = isl.Set("{S0[i, j, k]: 0 <= i < 12 and 0 <= j < 12 and 0 <= k < 12}");
	Access_A = isl.Map("{S0[i, j, k]->A[i, k]}")
	Access_B = isl.Map("{S0[i, j, k]->B[k, j]}")
	Access_C = isl.Map("{S0[i, j, k]->C[i, j, k]}")
	A = isl.Set("{A[i,j]: 0<=i<12 and 0<=j<12}")
	B = isl.Set("{B[i,j]: 0<=i<12 and 0<=j<12}")
	C = isl.Set("{C[i,j,k]: 0<=i<12 and 0<=j<12 and 0<=k<12}")
	return {'S0_dom':S0_dom, 'Access_A':Access_A, 'Access_B':Access_B, 'Access_C':Access_C, 'A':A, 'B':B, 'C':C}


def vec_add():
	S0_dom = isl.Set("[N]->{S0[i]: 0 <= i <= N}");
	Access_A = isl.Map("{S0[i]->A[i]}")
	Access_B = isl.Map("{S0[i]->B[i]}")
	Access_C = isl.Map("{S0[i]->C[i]}")
	A = isl.Set("[N]->{A[i]: 0<=i<N}")
	B = isl.Set("[N]->{B[i]: 0<=i<N}")
	C = isl.Set("[N]->{C[i]: 0<=i<N}")
	return {'S0_dom':S0_dom, 'Access_A':Access_A, 'Access_B':Access_B, 'Access_C':Access_C, 'A':A, 'B':B, 'C':C}

def matrix_matrix_add_fixed():
	S0_dom = isl.Set("{S0[i,k]: 0 <= i < 9 and 0 <= k < 9}");
	Access_A = isl.Map("{S0[i,k]->A[i,k]}")
	Access_B = isl.Map("{S0[i,k]->B[i,k]}")
	Access_C = isl.Map("{S0[i,k]->C[i]}")
	A = isl.Set("{A[i,k]: 0<=i<9 and 0<=k<9}")
	B = isl.Set("{B[i,k]: 0<=i<9 and 0<=k<9}")
	C = isl.Set("{C[i]: 0<=i<9}")
	return {'S0_dom':S0_dom, 'Access_A':Access_A, 'Access_B':Access_B, 'Access_C':Access_C, 'A':A, 'B':B, 'C':C}


def data_1():
	program = vec_add()
	partitioners_A = dist.partition(program['A'], [0], [4], [16], "row")
	partitioners_B = dist.partition(program['B'], [0], [4], [16], "row")
	partitioners_C = dist.partition(program['C'], [0], [4], [16], "row")
	str_schedule = "{Send_A[p0, i1]->[0, p0, i1]; Send_B[p0, i1]->[1, p0, i1]; Receive_A[p0, i1]->[2, p0, i1]; Receive_B[p0, i1]->[3, p0, i1]; S0[i]->[4, 0, i]}"
	context = isl.Set("[r0]->{:}")
	return {'S0_dom':program['S0_dom'], 'A':program['A'], 'B':program['B'], 'C':program['C'], 'Access_A':program['Access_A'], 'Access_B':program['Access_B'], 'Access_C':program['Access_C'], 'pr_partitioner_A':partitioners_A['pr'], 'ps_partitioner_A':partitioners_A['ps'], 'pr_partitioner_B':partitioners_B['pr'], 'ps_partitioner_B':partitioners_B['ps'], 'pr_partitioner_C':partitioners_C['pr'], 'ps_partitioner_C':partitioners_C['ps'], 'str_schedule':str_schedule, 'context':context}


def data_3():
	program = mat_mul_2D()
	partitioners_A = dist.partition(program['A'], [0], [3], [12], "row")
	partitioners_B = dist.partition(program['B'], [0], [3], [12], "row")
	partitioners_C = dist.partition(program['C'], [0], [3], [12], "row")
	str_schedule = "{Send_A[p0, i1, i2]->[0, p0, i1, i2]; Send_B[p0, i1, i2]->[1, p0, i1, i2]; Receive_A[p0, i1, i2]->[2, p0, i1, i2]; Receive_B[p0, i1, i2]->[3, p0, i1, i2]; S0[i, j, k]->[4, 0, i, j]}"
	context = isl.Set("[r0, r1]->{:}")
	return {'S0_dom':program['S0_dom'], 'A':program['A'], 'B':program['B'], 'C':program['C'], 'Access_A':program['Access_A'], 'Access_B':program['Access_B'], 'Access_C':program['Access_C'], 'pr_partitioner_A':partitioners_A['pr'], 'ps_partitioner_A':partitioners_A['ps'], 'pr_partitioner_B':partitioners_B['pr'], 'ps_partitioner_B':partitioners_B['ps'], 'pr_partitioner_C':partitioners_C['pr'], 'ps_partitioner_C':partitioners_C['ps'], 'str_schedule':str_schedule, 'context':context}


def data_4():
	program = mat_mul_2D_fixed()
	partitioners_A = dist.partition(program['A'], [0,1], [3,3], [12,12], "row")
	partitioners_B = dist.partition(program['B'], [0,1], [3,3], [12,12], "row")
	partitioners_C = dist.partition(program['C'], [0,1], [3,3], [12,12], "row")
	str_schedule = "[r]->{Send_A[p0, p1, i1, i2]->[0, p0, p1, i1, i2]; Send_B[p0, p1, i1, i2]->[1, p0, p1, i1, i2]; Receive_A[p0, p1, i1, i2]->[2, p0, p1, i1, i2]; Receive_B[p0, p1, i1, i2]->[3, p0, p1, i1, i2]; S0[i, j, k]->[4, 0, 0, i, j]}"
	context = isl.Set("[r0, r1]->{:}")
	return {'S0_dom':program['S0_dom'], 'A':program['A'], 'B':program['B'], 'C':program['C'], 'Access_A':program['Access_A'], 'Access_B':program['Access_B'], 'Access_C':program['Access_C'], 'pr_partitioner_A':partitioners_A['pr'], 'ps_partitioner_A':partitioners_A['ps'], 'pr_partitioner_B':partitioners_B['pr'], 'ps_partitioner_B':partitioners_B['ps'], 'pr_partitioner_C':partitioners_C['pr'], 'ps_partitioner_C':partitioners_C['ps'], 'str_schedule':str_schedule, 'context':context}


def data_5():
	program = mat_mul_2D_fixed()
	partitioners_A = dist.partition(program['A'], [0,1], [4,3], [12,12], "col")
	partitioners_B = dist.partition(program['B'], [0,1], [3,4], [12,12], "row")
	partitioners_C = dist.partition(program['C'], [0,1], [3,4], [12,12], "row")

	str_schedule = "[r0, r1]->{Send_A[p0, p1, i1, i2]->[0, p0, p1, i1, i2]; Send_B[p0, p1, i1, i2]->[1, p0, p1, i1, i2]; Receive_A[p0, p1, i1, i2]->[2, p0, p1, i1, i2]; Receive_B[p0, p1, i1, i2]->[3, p0, p1, i1, i2]; S0[i, j, k]->[4, 0, 0, i, j]}"
#	context = isl.Set("[r0, r1]->{:r0=1 and r1=1}")  #Correct
#	context = isl.Set("[r0, r1]->{:r0=0 and r1=0}")  #Correct
#	context = isl.Set("[r0, r1]->{:r0=0 and r1=1}")  #Correct
	context = isl.Set("[r0, r1]->{:}")
	return {'S0_dom':program['S0_dom'], 'A':program['A'], 'B':program['B'], 'C':program['C'], 'Access_A':program['Access_A'], 'Access_B':program['Access_B'], 'Access_C':program['Access_C'], 'pr_partitioner_A':partitioners_A['pr'], 'ps_partitioner_A':partitioners_A['ps'], 'pr_partitioner_B':partitioners_B['pr'], 'ps_partitioner_B':partitioners_B['ps'], 'pr_partitioner_C':partitioners_C['pr'], 'ps_partitioner_C':partitioners_C['ps'], 'str_schedule':str_schedule, 'context':context}


def data_6():
	# C(i) = A(i,k) + B(i,k)
	program = matrix_matrix_add_fixed()
	partitioners_A = dist.partition(program['A'], [0,1], [3,3], [9,9], "row")
	partitioners_B = dist.partition(program['B'], [0,1], [3,3], [9,9], "row")
	partitioners_C = dist.partition(program['C'], [0],   [9],  [9],    "row")

	str_schedule = "[r0, r1]->{Send_A[p0, p1, i1, i2]->[0, p0, p1, i1, i2]; Send_B[p0, p1, i1, i2]->[1, p0, p1, i1, i2]; Receive_A[p0, p1, i1, i2]->[2, p0, p1, i1, i2]; Receive_B[p0, p1, i1, i2]->[3, p0, p1, i1, i2]; S0[i, k]->[4, 0, 0, i, k]}"
#	context = isl.Set("[r0, r1]->{:r0=1 and r1=1}")
	context = isl.Set("[r0, r1]->{:r0=0 and r1=0}")
#	context = isl.Set("[r0, r1]->{:}")
	return {'S0_dom':program['S0_dom'], 'A':program['A'], 'B':program['B'], 'C':program['C'], 'Access_A':program['Access_A'], 'Access_B':program['Access_B'], 'Access_C':program['Access_C'], 'pr_partitioner_A':partitioners_A['pr'], 'ps_partitioner_A':partitioners_A['ps'], 'pr_partitioner_B':partitioners_B['pr'], 'ps_partitioner_B':partitioners_B['ps'], 'pr_partitioner_C':partitioners_C['pr'], 'ps_partitioner_C':partitioners_C['ps'], 'str_schedule':str_schedule, 'context':context}


############################################################################

#result = data_1();
result = data_3();
#result = data_4();
#result = data_5();
#result = data_6();

dist.compute_communication(result['S0_dom'], result['Access_A'], result['Access_B'], result['Access_C'], result['str_schedule'], result['A'], result['B'], result['C'], result['pr_partitioner_A'], result['ps_partitioner_A'], result['pr_partitioner_B'], result['ps_partitioner_B'], result['pr_partitioner_C'], result['ps_partitioner_C'], result['context'])



# TODO
# - Test the algorithm very well.
# - Support different stationary matrices.
# - Implement all the examples that they provided.
# - Support reductions.
# - Separate my implementation in a library, return intervals.
# - Support general number of arrays as input (currently, a hardcoded set of arrays is supported).
