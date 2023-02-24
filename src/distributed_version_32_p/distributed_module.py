import islpy as isl

simple_send_receive = False

def print_set(name, s):
	print name, " := ",
	if (s.is_empty()):
		print "Empty"
	else:
		print s, ";"


# The set S is assumed to be blocked.  The size of each multidimensional
# block is indicated using the "block" list.
# Each block is partitioned into partitions.
# Each processors of the list "processors" will run a partition.
# "dims" indicates the dimensions of "S" that are divided into blocks.
# "dims", "processors" and "block" are all integer lists.
# This function returns a map that partitions the set S.
# The name of the processor in this map is cpu_name. It can be either "r" or "rp".
# orientation: "row" or "col". "row" means that processors are distributed
# in row major order, while "col" means that they are distributed
# in column major order.
def generate_partitionning_map(S, dims, processors, block, orientation, cpu_name):
	name = S.get_tuple_name();
	dim_number = S.dim(isl.dim_type.set)

	# suffix is the name of the buffer that is being partitioned. If the suffix = A, then
	# processors that partition this array will be called rA0, rA1, ...
	# This is necessary because the partitionning of the array A may be different from
	# the partitionning of the array B.
	suffix = name

	NP = 1;
	for i in range(0,len(processors)):
		NP = NP * processors[i];

	dims_str = ""
	for i in range(0,dim_number):
		dims_str += "i" + str(i)
		if (i < dim_number-1):
			dims_str += ", "

	params = ""
	for i in range(0,len(dims)):
		params += cpu_name + str(i) + ", "
	for i in range(0,len(dims)):
		params += cpu_name + suffix + str(i)
		if (i < len(dims)-1):
			params += ", "

	processor_definitions = "" 
	for i in range(0,len(dims)):
		processor_definitions += "0 <= " + cpu_name + str(i) + " < " + str(NP) + " and "
		processor_definitions += cpu_name + suffix + str(i) + " = " + cpu_name + ((str(i)) if (orientation == "row") else (str(len(dims)-1-i)))
		if (i < len(dims)):
			processor_definitions += " and "

	partitions_str = "[" + params + "]->{" + name + "[" + dims_str + "]->" + name + "[" + dims_str + "]: "
	partitions_str += processor_definitions
	for i in range(0, len(dims)):
		partitions_str += str(block[i]/processors[i]) + cpu_name + suffix + str(i) + "<=i" + str(dims[i]) + "<" + str(block[i]/processors[i]) + "*(" + cpu_name + suffix + str(i) + "+1)"
		if (i < len(dims) - 1):
			partitions_str += " and "

	partitions_str += "}"
	print "Processor definitions = ", processor_definitions
	print "Partitions: ", partitions_str 
	partitions = isl.Map(partitions_str)
	# The parameters have the following form : [r0, r1, ..., rA0, rA1, ...].
	# The first set of parameters are always (r0, r1, ...) or (rp1, rp1, ...).  We need to project
	# all the parameters except the rX or rpX parameters so that we get every thing in function of rX or rpX only
	# instead of rA0, rA1, ... This works because we have already provided constraints
	# that indicate the relation between rA0, rA1, ... and (rX or rpX). So projecting rA0, rA1, ...
	# would leave the map in function of rX or rpX only which is what we want.
	partitions = partitions.project_out(isl.dim_type.param, len(dims), partitions.dim(isl.dim_type.param)-len(dims)).coalesce()
	return partitions

# The set S is assumed to be blocked.  The size of each multidimensional
# block is indicated using the "block" list.
# Each block is partitioned into partitions.
# Each processors of the list "processors" will run a partition.
# "dims" indicates the dimensions of "S" that are divided into blocks.
# "dims", "processors" and "block" are all integer lists.
# This function returns two maps, each map partitions the set into
# the selected partitions but one of the map is in function of the
# processor pr while the other is in function of the processor ps.
# orientation: "row" or "col". "row" means that processors are distributed
# in row major order, while "col" means that they are distributed
# in column major order.
def partition(S, dims, processors, block, orientation):
	rp_2D_partitions_map = generate_partitionning_map(S, dims, processors, block, orientation, "rp")
	r_2D_partitions_map =  generate_partitionning_map(S, dims, processors, block, orientation, "r")

	return {'ps':rp_2D_partitions_map, 'pr':r_2D_partitions_map}



# The set r_S is defined in function of r and rp.  in the send and receive sets
# r represents the rank of this processor while rp (r prime) is the rank of
# the processor that will send data to this processor or that will receive
# data from this processor. That is, it represents the processor ID from which
# this processor receives data or to which it sends its data.
# We want to represent rp using a loop so that we can send/receive from all
# the processors that are identified using the loop iterator.
# To do that, we create a variable "p" that is equal to "rp", then
# we project the parameter "rp".
def augment_set_with_processor(r_S, nb_processors):
	name = r_S.get_tuple_name();
	dim_number = r_S.dim(isl.dim_type.set)
	dims_str = ""
	for i in range(0, dim_number):
		dims_str += "i" + str(i)
		if (i < dim_number-1):
			dims_str += ", "
	params = ""
	processor_dims = ""
	processor_constraints = ""
	for i in range(0, nb_processors):
		params += "r" + str(i) + ", " 
		processor_dims += "p" + str(i)
		processor_constraints += " p" + str(i) + " = rp" + str(i)
		if (i < nb_processors-1):
			processor_dims += ","
			processor_constraints += " and "
	for i in range(0, nb_processors):
		params += "rp" + str(i) 
		if (i < nb_processors-1):
			params += ","


	map_str = "["+ params + "]->{" + name + "[" + dims_str + "]->" + name + "[" + processor_dims + "," + dims_str + "]: " + processor_constraints + "}"
	print "Augmentation map (adds news dimensions) : ", map_str
	# Add a new dimension p0 to the set.
	augmentation_map = isl.Map(map_str)
	augmented_r_S = r_S.apply(augmentation_map)

	print "Projecting out 1 dimension of the parameters starting from the dimension 1. This dimension represents the parameter rp."
	augmented_r_S = augmented_r_S.project_out(isl.dim_type.param, nb_processors, nb_processors).coalesce()

	return augmented_r_S



# r is my_rank().
def compute_communication(S0_dom, Access_A, Access_B, Access_C, str_schedule, A, B, C, r_partitioner_A, rp_partitioner_A, r_partitioner_B, rp_partitioner_B, r_partitioner_C, rp_partitioner_C, context):
	# 0 - Print all inputs
	##########################################@
	print "S0_dom = ", S0_dom
	print "Access_A = ", Access_A
	print "Access_B = ", Access_B
	print "Access_C = ", Access_C
	print "Schedule = ", str_schedule
	print "A = ", A
	print "B = ", B
	print "C = ", C
	print "\n\n"
	print "r_partitioner_A = ", r_partitioner_A
	print "rp_partitioner_A = ", rp_partitioner_A
	print "r_partitioner_B = ", r_partitioner_B
	print "rp_partitioner_B = ", rp_partitioner_B
	print "r_partitioner_C = ", r_partitioner_C
	print "rp_partitioner_C = ", rp_partitioner_C
	print "\n\n"


	# I - Compute the "Have" sets for r and rp.
	############################################
	r_have_A = A.apply(r_partitioner_A)
	rp_have_A = A.apply(rp_partitioner_A)
	r_have_B = B.apply(r_partitioner_B)
	rp_have_B = B.apply(rp_partitioner_B)
	r_have_C = C.apply(r_partitioner_C)
	rp_have_C = C.apply(rp_partitioner_C)
	print "r_have_A = ", r_have_A;
	print "r_have_B = ", r_have_B
	print "r_have_C = ", r_have_C
	print "rp_have_A = ", rp_have_A
	print "rp_have_B = ", rp_have_B
	print "rp_have_C = ", rp_have_C


	# II - Compute the "Need" parts of A, B, and C
	#	II.1- Reverse access to C
	#	II.2- Apply the have_C to the reverse of C to get the corresponding domain of S0
	#	II.2- Apply the domain of S0 to Access_A and Access_B to derive the needed parts
	# 	      of A and B
	################################################
	reverse_Access_C = Access_C.reverse()
	S0_corresponding_to_p_owned_C = r_have_C.apply(reverse_Access_C)
	S0_corresponding_to_pp_owned_C = rp_have_C.apply(reverse_Access_C)
	r_need_A = S0_corresponding_to_p_owned_C.apply(Access_A)
	r_need_B = S0_corresponding_to_p_owned_C.apply(Access_B)
	rp_need_A = S0_corresponding_to_pp_owned_C.apply(Access_A)
	rp_need_B = S0_corresponding_to_pp_owned_C.apply(Access_B)
	print "r_need_A = ", r_need_A;
	print "r_need_B = ", r_need_B
	print "rp_need_A = ", rp_need_A;
	print "rp_need_B = ", rp_need_B
	print "\n"


	# III- Compute the missing part for r. We interpret r as the receiver in this section
	# (i.e., the part that r has to receive)
	#################################################
	# what needs to be received for A = r_missing_A
	r_missing_A = r_need_A - r_have_A
	r_missing_B = r_need_B - r_have_B
	r_receive_A = r_missing_A.intersect(rp_have_A)
	r_receive_B = r_missing_B.intersect(rp_have_B)
	print "r_missing_A = r_need_A - r_have_A = ", r_missing_A
	print "r_missing_B = r_need_B - r_have_B = ", r_missing_B
	print "r_receive_A = r_missing_A.intersect(rp_have_A) = ", r_receive_A
	print "r_receive_B = r_missing_B.intersect(rp_have_B) = ", r_receive_B
	print "\n\n"


	# IV- Compute the missing part for the senders.
	#################################################
	rp_missing_A = rp_need_A - rp_have_A
	rp_missing_B = rp_need_B - rp_have_B
	r_send_A = rp_missing_A.intersect(r_have_A)
	r_send_B = rp_missing_B.intersect(r_have_B)
	print "rp_missing_A = rp_need_A - rp_have_A = ", rp_missing_A
	print "rp_missing_B = rp_need_B - rp_have_B = ", rp_missing_B
	print "r_send_A = rp_missing_A.intersect(r_have_A) = ", r_send_A
	print "r_send_B = rp_missing_B.intersect(r_have_B) = ", r_send_B
	print "\n\n"


	# V- Code generation
	##################################################
	if (r_send_A.is_empty() == False):
		r_send_A = augment_set_with_processor(r_send_A, r_partitioner_A.dim(isl.dim_type.param))
	if (r_send_B.is_empty() == False):
		r_send_B = augment_set_with_processor(r_send_B, r_partitioner_B.dim(isl.dim_type.param))
	if (r_receive_A.is_empty() == False):
		r_receive_A = augment_set_with_processor(r_receive_A, r_partitioner_A.dim(isl.dim_type.param))
	if (r_receive_B.is_empty() == False):
		r_receive_B = augment_set_with_processor(r_receive_B, r_partitioner_B.dim(isl.dim_type.param))
	print "\n\n"
	r_send_A = r_send_A.set_tuple_name("Send_A")
	r_send_B = r_send_B.set_tuple_name("Send_B")
	r_receive_A = r_receive_A.set_tuple_name("Receive_A")
	r_receive_B = r_receive_B.set_tuple_name("Receive_B")
	print_set("Send_A   ", r_send_A)
	print "\n"
	print_set("Send_B   ", r_send_B)
	print "\n"
	print_set("Receive_A", r_receive_A)
	print "\n"
        print_set("Receive_B", r_receive_B)
	print "\n"

	uA_to_send_p = isl.UnionSet.from_set(r_send_A)
	uB_to_send_p = isl.UnionSet.from_set(r_send_B)
	ur_receive_A = isl.UnionSet.from_set(r_receive_A)
	ur_receive_B = isl.UnionSet.from_set(r_receive_B)
	uS0_dom = isl.UnionSet.from_set(S0_dom)
	domain = uA_to_send_p.union(uB_to_send_p)
	domain = domain.union(ur_receive_A)
	domain = domain.union(ur_receive_B)
	domain = domain.union(uS0_dom)
	schedule = isl.UnionMap(str_schedule);
	schedule = schedule.intersect_domain(domain)
	print "Schedule intersect domain = ", schedule
	build = isl.AstBuild.alloc(schedule.get_ctx())
	build = build.restrict(context)
	node = build.node_from_schedule_map(schedule)
	print node.to_C_str()

		

