#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string>

namespace coli
{

extern std::map<std::string, coli::computation *> computations_list;
extern int id_counter;

/**
  * Get the computation associated with a node.
  */
coli::computation *get_computation_name(isl_ast_node *node)
{
	isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
	isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
	isl_id *id = isl_ast_expr_get_id(arg);
	isl_ast_expr_free(arg);
	std::string computation_name(isl_id_get_name(id));
	isl_id_free(id);
	coli::computation *comp = computations_list.find(computation_name)->second;

	assert((comp != NULL) && "Computation not found for this node.");

	return comp;
}


/**
 * Retrieve the access function of the ISL AST leaf node (which represents a
 * computation).  Store the access in computation->access.
 */
isl_ast_node *stmt_code_generator(isl_ast_node *node, isl_ast_build *build, void *user)
{
	assert(build != NULL);
	assert(node != NULL);

	/* Retrieve the iterator map and store it in computations_list.  */
	isl_union_map *schedule = isl_ast_build_get_schedule(build);
	isl_map *map = isl_map_reverse(isl_map_from_union_map(schedule));
	isl_pw_multi_aff *iterator_map = isl_pw_multi_aff_from_map(map);

	if (DEBUG2)
	{
		coli::str_dump("The iterator map of an AST leaf (after scheduling):\n");
		isl_pw_multi_aff_dump(iterator_map);
		coli::str_dump("\n");
	}

	// Find the name of the computation associated to this AST leaf node.
	coli::computation *comp = get_computation_name(node);

	// Check that the access domain is compatible with the time-processor representation.
	// The must have the same tuple name and same number of dimensions.
	int diff = strcmp(isl_set_get_tuple_name(comp->get_time_processor_representation()),
			  isl_map_get_tuple_name(comp->get_access(), isl_dim_in));
	assert((diff == 0) && "Name of space in time-processor space and the domain of the access function must be identical");
	int n_dim_tpr = isl_space_dim(isl_set_get_space(comp->get_time_processor_representation()), isl_dim_set);
	int n_dim_a  = isl_space_dim(isl_map_get_space(comp->get_access()), isl_dim_in);
	assert((n_dim_a == n_dim_tpr) && "Number of dimensions in the time-processor space and the domain of the access function must be identical");

	isl_pw_multi_aff *index_aff = isl_pw_multi_aff_from_map(isl_map_copy(comp->access));
	iterator_map = isl_pw_multi_aff_pullback_pw_multi_aff(index_aff, iterator_map);

	isl_ast_expr *index_expr = isl_ast_build_access_from_pw_multi_aff(build, isl_pw_multi_aff_copy(iterator_map));
	comp->index_expr = index_expr;

	if (DEBUG2)
	{
		coli::str_dump("Index expression (for an AST leaf):\n");
		isl_ast_expr_dump(index_expr);
		coli::str_dump("\n");
	}

	return node;
}

Halide::Expr create_halide_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
	Halide::Expr result;

	if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
	{
		isl_val *init_val = isl_ast_expr_get_val(isl_expr);
		result = Halide::Expr((int32_t)isl_val_get_num_si(init_val));
	}
	else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
	{
		isl_id *identifier = isl_ast_expr_get_id(isl_expr);
		std::string name_str(isl_id_get_name(identifier));
		result = Halide::Internal::Variable::make(Halide::Int(32), name_str);
	}
	else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op)
	{
		Halide::Expr op0, op1;

		op0 = create_halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 0));

		if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
			op1 = create_halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 1));

		switch(isl_ast_expr_get_op_type(isl_expr))
		{
			case isl_ast_op_and:
				result = Halide::Internal::And::make(op0, op1);
				break;
			case isl_ast_op_and_then:
				result = Halide::Internal::And::make(op0, op1);
				coli::error("isl_ast_op_and_then operator found in the AST. This operator is not well supported.", 0);
				break;
			case isl_ast_op_or:
				result = Halide::Internal::Or::make(op0, op1);
				break;
			case isl_ast_op_or_else:
				result = Halide::Internal::Or::make(op0, op1);
				coli::error("isl_ast_op_or_then operator found in the AST. This operator is not well supported.", 0);
				break;
			case isl_ast_op_max:
				result = Halide::Internal::Max::make(op0, op1);
				break;
			case isl_ast_op_min:
				result = Halide::Internal::Min::make(op0, op1);
				break;
			case isl_ast_op_minus:
				result = Halide::Internal::Sub::make(Halide::Expr(0), op0);
				break;
			case isl_ast_op_add:
				result = Halide::Internal::Add::make(op0, op1);
				break;
			case isl_ast_op_sub:
				result = Halide::Internal::Sub::make(op0, op1);
				break;
			case isl_ast_op_mul:
				result = Halide::Internal::Mul::make(op0, op1);
				break;
			case isl_ast_op_div:
				result = Halide::Internal::Div::make(op0, op1);
				break;
			case isl_ast_op_le:
				result = Halide::Internal::LE::make(op0, op1);
				break;
			case isl_ast_op_lt:
				result = Halide::Internal::LT::make(op0, op1);
				break;
			case isl_ast_op_ge:
				result = Halide::Internal::GE::make(op0, op1);
				break;
			case isl_ast_op_gt:
				result = Halide::Internal::GT::make(op0, op1);
				break;
			case isl_ast_op_eq:
				result = Halide::Internal::EQ::make(op0, op1);
				break;
			default:
				coli::error("Translating an unsupported ISL expression in a Halide expression.", 1);
		}
	}
	else
		coli::error("Translating an unsupported ISL expression in a Halide expression.", 1);

	return result;
}

isl_ast_node *for_code_generator_after_for(isl_ast_node *node, isl_ast_build *build, void *user)
{

	return node;
}

// Level represents the level of the node in the schedule.  0 means root.
Halide::Internal::Stmt *generate_Halide_stmt_from_isl_node(coli::library lib, isl_ast_node *node,
		int level, std::vector<std::string> &generated_stmts, std::vector<std::string> &iterators)
{
	Halide::Internal::Stmt *result = new Halide::Internal::Stmt();
	int i;

	if (isl_ast_node_get_type(node) == isl_ast_node_block)
	{
		IF_DEBUG2(coli::str_dump("Generating code for a block\n"));

		isl_ast_node_list *list = isl_ast_node_block_get_children(node);
		isl_ast_node *child, *child2;

		if (isl_ast_node_list_n_ast_node(list) >= 2)
		{
			child = isl_ast_node_list_get_ast_node(list, 0);
			child2 = isl_ast_node_list_get_ast_node(list, 1);

			*result = Halide::Internal::Block::make(*coli::generate_Halide_stmt_from_isl_node(lib, child, level+1, generated_stmts, iterators),
					*coli::generate_Halide_stmt_from_isl_node(lib, child2, level+1, generated_stmts, iterators));

			for (i = 2; i < isl_ast_node_list_n_ast_node(list); i++)
			{
				child = isl_ast_node_list_get_ast_node(list, i);
				*result = Halide::Internal::Block::make(*result, *coli::generate_Halide_stmt_from_isl_node(lib, child, level+1, generated_stmts, iterators));
			}
		}
		else
			// The above code expects the isl ast block to have at least two statemenets so that the
			// Halide::Internal::Block::make works (because that function expects its two inputs to be define).
			coli::error("Expecting the block to have at least 2 statements but it does not.", true);
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_for)
	{
		IF_DEBUG2(coli::str_dump("Generating code for Halide::For\n"));

		isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
		char *iterator_str = isl_ast_expr_to_C_str(iter);

		// Add this iterator to the list of iterators.
		// This list is used later when generating access of inner
		// statements.
		iterators.push_back(std::string(iterator_str));

		isl_ast_expr *init = isl_ast_node_for_get_init(node);
		isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
		isl_ast_expr *inc  = isl_ast_node_for_get_inc(node);

		if (!isl_val_is_one(isl_ast_expr_get_val(inc)))
			coli::error("The increment in one of the loops is not +1."
			      "This is not supported by Halide", 1);

		isl_ast_node *body = isl_ast_node_for_get_body(node);
		isl_ast_expr *cond_upper_bound_isl_format = NULL;

		/*
		   Halide expects the loop bound to be of the form
			iter < bound
		   where as ISL can generated loop bounds of the forms
			ite < bound
		   and
			iter <= bound
		   We need to transform the two ISL loop bounds into the Halide
		   format.
		   */
		if (isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
			cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
		else if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le)
		{
			// Create an expression of "1".
			isl_val *one = isl_val_one(isl_ast_node_get_ctx(node));
			// Add 1 to the ISL ast upper bound to transform it into a strinct bound.
			cond_upper_bound_isl_format = isl_ast_expr_add(
							isl_ast_expr_get_op_arg(cond, 1),
							isl_ast_expr_from_val(one));
		}
		else
			coli::error("The for loop upper bound is not an isl_est_expr of type le or lt" ,1);

		assert(cond_upper_bound_isl_format != NULL);
		Halide::Expr init_expr = create_halide_expr_from_isl_ast_expr(init);
		Halide::Expr cond_upper_bound_halide_format =  create_halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format);
		Halide::Internal::Stmt *halide_body = coli::generate_Halide_stmt_from_isl_node(lib, body, level+1, generated_stmts, iterators);
		Halide::Internal::ForType fortype = Halide::Internal::ForType::Serial;

		// Change the type from Serial to parallel or vector if the
		// current level was marked as such.
		for (auto generated_stmt: generated_stmts)
			if (lib.parallelize(generated_stmt, level))
				fortype = Halide::Internal::ForType::Parallel;
			else if (lib.vectorize(generated_stmt, level))
				fortype = Halide::Internal::ForType::Vectorized;

		*result = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format, fortype,
				Halide::DeviceAPI::Host, *halide_body);
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_user)
	{
		IF_DEBUG2(coli::str_dump("Generating code for user node\n"));

		isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
		isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
		isl_id *id = isl_ast_expr_get_id(arg);
		isl_ast_expr_free(arg);
		std::string computation_name(isl_id_get_name(id));
		isl_id_free(id);
		generated_stmts.push_back(computation_name);

		coli::computation *comp = computations_list.find(computation_name)->second;

		comp->create_halide_assignement(iterators);

		*result = comp->stmt;
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_if)
	{
		IF_DEBUG2(coli::str_dump("Generating code for conditional\n"));

		isl_ast_expr *cond = isl_ast_node_if_get_cond(node);
		isl_ast_node *if_stmt = isl_ast_node_if_get_then(node);
		isl_ast_node *else_stmt = isl_ast_node_if_get_else(node);

		*result = Halide::Internal::IfThenElse::make(create_halide_expr_from_isl_ast_expr(cond),
				*coli::generate_Halide_stmt_from_isl_node(lib, if_stmt,
					level+1, generated_stmts, iterators),
				*coli::generate_Halide_stmt_from_isl_node(lib, else_stmt,
					level+1, generated_stmts, iterators));
	}

	return result;
}

/**
  * Linearize a multidimensional access to a Halide buffer.
  * Supposing that we have buf[N1][N2][N3], transform buf[i][j][k]
  * into buf[k + j*N3 + i*N3*N2].
  * Note that the first arg in index_expr is the buffer name.  The other args
  * are the indices for each dimension of the buffer.
  */
Halide::Expr linearize_access(Halide::Buffer *buffer,
		isl_ast_expr *index_expr)
{
	assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);

	int buf_dims = buffer->dimensions();

	// Get the rightmost access index: in A[i][j], this will return j
	isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, buf_dims);
	Halide::Expr index = create_halide_expr_from_isl_ast_expr(operand);

	Halide::Expr extents;

	if (buf_dims > 1)
		extents = Halide::Expr(buffer->extent(buf_dims - 1));

	for (int i = buf_dims - 1; i >= 1; i--)
	{
		operand = isl_ast_expr_get_op_arg(index_expr, i);
		Halide::Expr operand_h = create_halide_expr_from_isl_ast_expr(operand);
		Halide::Expr mul = Halide::Internal::Mul::make(operand_h, extents);

		index = Halide::Internal::Add::make(index, mul);

		extents = Halide::Internal::Mul::make(extents, Halide::Expr(buffer->extent(i - 1)));
	}

	return index;
}

/*
 * Create a Halide assign statement from a computation.
 * The statement will assign the computations to a memory buffer based on the
 * access function provided in access.
 */
void computation::create_halide_assignement(std::vector<std::string> &iterators)
{
	   assert(this->access != NULL);

	   const char *buffer_name = isl_space_get_tuple_name(
					isl_map_get_space(this->access), isl_dim_out);
	   assert(buffer_name != NULL);

	   isl_map *access = this->access;
	   isl_space *space = isl_map_get_space(access);
	   // Get the number of dimensions of the ISL map representing
	   // the access.
	   int access_dims = isl_space_dim(space, isl_dim_out);

	   // Fetch the actual buffer.
	   auto buffer_entry = this->function->buffers_list.find(buffer_name);
	   assert(buffer_entry != this->function->buffers_list.end());
	   Halide::Buffer *buffer = buffer_entry->second;
	   int buf_dims = buffer->dimensions();

	   // The number of dimensions in the Halide buffer should be equal to
	   // the number of dimensions of the access function.
	   assert(buf_dims == access_dims);

	   auto index_expr = this->index_expr;
	   assert(index_expr != NULL);

	   //TODO: Currently the names of the iterators in the ISL AST and
	   //the names that are generated when creating the Halide IR are
	   //equivalent by chance.  They should be made always equivalent.

	   if (DEBUG2)
	   {
		   std::cout << "List of iterators: ";
		   for (auto iter: iterators)
			   std::cout << iter << ", ";
		   std::cout << std::endl;
	   }

	   Halide::Expr index = coli::linearize_access(buffer, index_expr);

	   Halide::Internal::Parameter param(buffer->type(), true,
			buffer->dimensions(), buffer->name());
	   param.set_buffer(*buffer);
	   this->stmt = Halide::Internal::Store::make(buffer_name, this->expression, index, param);
}

void library::gen_halide_obj(std::string obj_file_name,
		Halide::Target::OS os,
		Halide::Target::Arch arch, int bits)
{
	Halide::Target target;
	target.os = os;
	target.arch = arch;
	target.bits = bits;
	std::vector<Halide::Target::Feature> x86_features;
	x86_features.push_back(Halide::Target::AVX);
	x86_features.push_back(Halide::Target::SSE41);
	target.set_features(x86_features);

	Halide::Module m(obj_file_name, target);

	for (auto func: this->get_functions())
	{
		m.append(Halide::Internal::LoweredFunc(func->get_name(), func->get_arguments(), func->get_halide_stmt(), Halide::Internal::LoweredFunc::External));
	}

	Halide::Outputs output = Halide::Outputs().object(obj_file_name);
	m.compile(output);
}

}
