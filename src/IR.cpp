#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <DebugIR.h>
#include <IR.h>

#include <string>

std::map<std::string, Halide::Internal::Stmt> stmts_list;
int id_counter = 0;

isl_ast_node *stmt_halide_code_generator(isl_ast_node *node, isl_ast_build *build, void *user)
{

	return node;
}

Halide::Expr create_halide_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
	Halide::Expr result;

	if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
	{
		isl_val *init_val = isl_ast_expr_get_val(isl_expr);
		result = Halide::Expr((int64_t)isl_val_get_num_si(init_val));
	}
	else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
	{
		isl_id *identifier = isl_ast_expr_get_id(isl_expr);
		std::string name_str(isl_id_get_name(identifier));
		result = Halide::Internal::Variable::make(Halide::Int(64), name_str);
	}
	else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op)
	{
		Halide::Expr op0, op1;

		int nb_args = isl_ast_expr_get_op_n_arg(isl_expr);
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
				Error("isl_ast_op_and_then operator found in the AST. This operator is not well supported.", 0);
				break;
			case isl_ast_op_or:
				result = Halide::Internal::Or::make(op0, op1);
				break;
			case isl_ast_op_or_else:
				result = Halide::Internal::Or::make(op0, op1);
				Error("isl_ast_op_or_then operator found in the AST. This operator is not well supported.", 0);
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
				Error("Translating an unsupported ISL expression in a Halide expression.", 1);
		}
	}
	else
		Error("Translating an unsupported ISL expression in a Halide expression.", 1);

	return result;
}

isl_ast_node *for_halide_code_generator_after_for(isl_ast_node *node, isl_ast_build *build, void *user)
{

	return node;
}

isl_schedule *create_schedule_tree(isl_ctx *ctx,
		   isl_union_set *udom,
		   isl_union_map *sched_map)
{
	isl_union_set *scheduled_domain = isl_union_set_apply(udom, sched_map);
	IF_DEBUG2(str_dump("[ir.c] Scheduled domain: "));
	IF_DEBUG2(isl_union_set_dump(scheduled_domain));

	isl_schedule *sched_tree = isl_schedule_from_domain(scheduled_domain);

	return sched_tree;
}

// Level represents the level of the node in the schedule.  0 means root.
Halide::Internal::Stmt generate_Halide_stmt_from_isl_node(IRProgram pgm, isl_ast_node *node,
		int level, std::vector<std::string> &generated_stmts)
{
	Halide::Internal::Stmt result;
	int i;

	if (isl_ast_node_get_type(node) == isl_ast_node_block)
	{
		isl_ast_node_list *list = isl_ast_node_block_get_children(node);
		isl_ast_node *child;
		
		if (isl_ast_node_list_n_ast_node(list) >= 1)
		{
			child = isl_ast_node_list_get_ast_node(list, 0);
			result = Halide::Internal::Block::make(generate_Halide_stmt_from_isl_node(pgm, child, level+1, generated_stmts), Halide::Internal::Stmt());
		
			for (i = 1; i < isl_ast_node_list_n_ast_node(list); i++)
			{
				child = isl_ast_node_list_get_ast_node(list, i);
				result = Halide::Internal::Block::make(result, generate_Halide_stmt_from_isl_node(pgm, child, level+1, generated_stmts));
			}
		}
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_for)
	{
		isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
		char *iterator_str = isl_ast_expr_to_str(iter);

		isl_ast_expr *init = isl_ast_node_for_get_init(node);
		isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
		isl_ast_expr *inc  = isl_ast_node_for_get_inc(node);

		if (!isl_val_is_one(isl_ast_expr_get_val(inc)))
			Error("The increment in one of the loops is not +1."
			      "This is not supported by Halide", 1);

		isl_ast_node *body = isl_ast_node_for_get_body(node);
		isl_ast_expr *cond_upper_bound_isl_format;
		if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le || isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
			cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
		else
			Error("The for loop upper bound is not an isl_est_expr of type le or lt" ,1);

		Halide::Expr init_expr = create_halide_expr_from_isl_ast_expr(init);
		Halide::Expr cond_upper_bound_halide_format =  create_halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format);
		Halide::Internal::Stmt halide_body = generate_Halide_stmt_from_isl_node(pgm, body, level+1, generated_stmts);
		Halide::Internal::ForType fortype = Halide::Internal::ForType::Serial;

		// Change the type from Serial to parallel or vector if the
		// current level was marked as such.
		for (auto generated_stmt: generated_stmts)
			if (pgm.parallel_dimensions.find(generated_stmt)->second == level)
				fortype = Halide::Internal::ForType::Parallel;
			else if (pgm.vector_dimensions.find(generated_stmt)->second == level)
				fortype = Halide::Internal::ForType::Vectorized;

		result = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format, fortype,
				Halide::DeviceAPI::Host, halide_body);
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_user)
	{
		isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
		isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
		isl_id *id = isl_ast_expr_get_id(arg);
		isl_ast_expr_free(arg);
		std::string computation_name(isl_id_get_name(id));
		isl_id_free(id);
		generated_stmts.push_back(computation_name);

		result = stmts_list.find(computation_name)->second; 
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_if)
	{
		isl_ast_expr *cond = isl_ast_node_if_get_cond(node);
		isl_ast_node *if_stmt = isl_ast_node_if_get_then(node);
		isl_ast_node *else_stmt = isl_ast_node_if_get_else(node);

		result = Halide::Internal::IfThenElse::make(create_halide_expr_from_isl_ast_expr(cond),
				generate_Halide_stmt_from_isl_node(pgm, if_stmt,
					level+1, generated_stmts),
				generate_Halide_stmt_from_isl_node(pgm, else_stmt,
					level+1, generated_stmts));
	}

	return result;
}

/* Schedule the iteration space.  */
isl_union_set *create_time_space(__isl_take isl_union_set *set, __isl_take isl_union_map *umap)
{
	return isl_union_set_apply(set, umap);
}

isl_ast_node *generate_code(isl_ctx *ctx,
		   isl_schedule *sched_tree)
{
	isl_ast_build *ast = isl_ast_build_alloc(ctx);
 	isl_ast_node *program = isl_ast_build_node_from_schedule(ast, sched_tree);
	isl_ast_build_free(ast);

	return program;
}

void isl_ast_node_dump_c_code(isl_ctx *ctx, isl_ast_node *root_node)
{
	if (DEBUG)
	{
		str_dump("\n\n");
		str_dump("\nC like code:\n");
		isl_printer *p;
		p = isl_printer_to_file(ctx, stdout);
		p = isl_printer_set_output_format(p, ISL_FORMAT_C);
		p = isl_printer_print_ast_node(p, root_node);
		isl_printer_free(p);
		str_dump("\n\n");
	}
}


// Computation related methods

void Computation::dump_ISIR()
{
	if (DEBUG)
	{
		isl_set_dump(this->iter_space);
	}
}

void Computation::dump_schedule()
{
	if (DEBUG)
	{
		isl_map_dump(this->schedule);
	}
}

void Computation::dump()
{
	if (DEBUG)
	{
		std::cout << "Computation \"" << this->name << "\"" << std::endl;
		isl_set_dump(this->iter_space);
		std::cout << "Schedule " << std::endl;
		isl_map_dump(this->schedule);
		str_dump("Halide statement:\n");
		Halide::Internal::IRPrinter pr(std::cout);
	    	pr.print(this->stmt);
		str_dump("\n");

	}
}

void Computation::Schedule(std::string map_str)
{
	isl_map *map = isl_map_read_from_str(this->ctx,
			map_str.c_str());

	this->schedule = map;
}

void Computation::Tile(int inDim0, int inDim1,
			int sizeX, int sizeY)
{
	assert((inDim0 == inDim1+1) || (inDim1 == inDim0+1));

	this->Split(inDim0, sizeX);
	this->Split(inDim1+1, sizeY);
	this->Interchange(inDim0+1, inDim1+1);
}

void split_string(std::string str, std::string delimiter,
		  std::vector<std::string> &vector)
{
	size_t pos = 0;
	std::string token;
	while ((pos = str.find(delimiter)) != std::string::npos) {
		token = str.substr(0, pos);
		vector.push_back(token);
		str.erase(0, pos + delimiter.length());
	}
	token = str.substr(0, pos);
	vector.push_back(token);
}

void isl_space_tokens::Parse(std::string space)
{
	split_string(space, ",", this->dimensions);
}

std::string generate_new_variable_name()
{
	return "r" + std::to_string(id_counter++);
}

/**
 * Modify the schedule of this computation so that the two dimensions
 * inDim0 and inDime1 are interchanged (swaped).
 */
void Computation::Interchange(int inDim0, int inDim1)
{
	assert(inDim0 >= 0);
	assert(inDim0 < isl_space_dim(isl_map_get_space(this->schedule),
							isl_dim_out));
	assert(inDim1 >= 0);
	assert(inDim1 < isl_space_dim(isl_map_get_space(this->schedule),
				          		isl_dim_out));

	isl_map_tokens map(isl_map_to_str(this->schedule));

	std::iter_swap(map.range.dimensions.begin()+inDim0,
			map.range.dimensions.begin()+inDim1);

	this->schedule = isl_map_read_from_str(this->ctx, map.get_str().c_str());
}

/**
 * Modify the schedule of this computation so that it splits the
 * dimension inDim0 of the iteration space into two new dimensions.
 * The size of the inner dimension created is sizeX.
 */
void Computation::Split(int inDim0, int sizeX)
{
	assert(inDim0 >= 0);
	assert(inDim0 < isl_space_dim(isl_map_get_space(this->schedule),
				          isl_dim_out));
	assert(sizeX >= 1);


	isl_map_tokens map(isl_map_to_str(this->schedule));

	std::string inDim0_str = map.range.dimensions.at(inDim0);
	std::string outDim0 = generate_new_variable_name(); 
	std::string outDim1 = generate_new_variable_name();
	std::string outDimensions = outDim0 + "," + outDim1;

	map.range.replace(inDim0_str, outDim0, outDim1);

	// Add the relations
	std::string relation1 = outDim0 + "=floor(" + inDim0_str + "/" +
		std::to_string(sizeX) + ") ";
	std::string relation2 = outDim1 + "=" + inDim0_str + "%" +
	 	std::to_string(sizeX);

	map.constraints.add(relation1);
	map.constraints.add(relation2);

	this->schedule = isl_map_read_from_str(this->ctx, map.get_str().c_str());
}


// Function related methods

void IRFunction::add_computation_to_body(Computation *cpt)
{
	this->body.push_back(cpt);
}

void IRFunction::add_computation_to_signature(Computation *cpt)
{
	this->signature.push_back(cpt);
}

void IRFunction::dump()
{
	if (DEBUG)
	{
		std::cout << "Function \"" << this->name << "\"" << std::endl;
		std::cout << "Body " << std::endl;

		for (auto cpt : this->body)
		       cpt->dump();

		std::cout << "Signature:" << std::endl;

		for (auto cpt : this->signature)
		       cpt->dump();

		std::cout << std::endl;
	}
}

void IRFunction::dump_ISIR()
{
	if (DEBUG)
	{
		for (auto cpt : this->body)
		       cpt->dump_ISIR();
	}
}

void IRFunction::dump_schedule()
{
	if (DEBUG)
	{
		for (auto cpt : this->body)
		       cpt->dump_schedule();
	}
}


// Program related methods

void IRProgram::tag_parallel_dimension(std::string stmt_name,
				      int par_dim)
{
	if (par_dim >= 0)
		this->parallel_dimensions.insert(
				std::pair<std::string,int>(stmt_name,
							   par_dim));
}

void IRProgram::tag_vector_dimension(std::string stmt_name,
		int vec_dim)
{
	if (vec_dim >= 0)
		this->vector_dimensions.insert(
				std::pair<std::string,int>(stmt_name,
					                   vec_dim));
}

void IRProgram::dump_ISIR()
{
	if (DEBUG)
	{
		str_dump("\nIteration Space IR:\n");
		for (const auto &fct : this->functions)
		       fct->dump_ISIR();
		str_dump("\n");
	}
}

void IRProgram::dump_schedule()
{
	if (DEBUG)
	{
		str_dump("\nSchedule:\n");
		for (const auto &fct : this->functions)
		       fct->dump_schedule();

		std::cout << "Parallel dimensions: ";
		for (auto par_dim: parallel_dimensions)
			std::cout << par_dim.first << "(" << par_dim.second << ") ";

		std::cout << std::endl;

		std::cout << "Vector dimensions: ";
		for (auto vec_dim: vector_dimensions)
			std::cout << vec_dim.first << "(" << vec_dim.second << ") ";

		std::cout<< std::endl << std::endl << std::endl;
	}
}

void IRProgram::dump()
{
	if (DEBUG)
	{
		std::cout << "Program \"" << this->name << "\"" << std::endl
			  <<
			std::endl;

		for (const auto &fct : this->functions)
		       fct->dump();

		std::cout << "Parallel dimensions: ";
		for (auto par_dim: parallel_dimensions)
			std::cout << par_dim.first << "(" << par_dim.second << ") ";

		std::cout << std::endl;

		std::cout << "Vector dimensions: ";
		for (auto vec_dim: vector_dimensions)
			std::cout << vec_dim.first << "(" << vec_dim.second << ") ";

		std::cout<< std::endl << std::endl;
	}
}

void IRProgram::add_function(IRFunction *fct)
{
	this->functions.push_back(fct);
}

isl_union_set * IRProgram::get_iteration_spaces()
{
	isl_union_set *result;
	isl_space *space;

	if (this->functions.empty() == false)
	{
		if(this->functions[0]->body.empty() == false)
			space = isl_set_get_space(this->functions[0]->body[0]->iter_space);
	}
	else
		return NULL;

	result = isl_union_set_empty(isl_space_copy(space));

	for (const auto &fct : this->functions)
		for (const auto &cpt : fct->body)
		{
			isl_set *cpt_iter_space = isl_set_copy(cpt->iter_space);
			result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
		}

	return result;
}

isl_union_map * IRProgram::get_schedule_map()
{
	isl_union_map *result;
	isl_space *space;

	if (this->functions.empty() == false)
	{
		if(this->functions[0]->body.empty() == false)
			space = isl_map_get_space(this->functions[0]->body[0]->schedule);
	}
	else
		return NULL;

	result = isl_union_map_empty(isl_space_copy(space));

	for (const auto &fct : this->functions)
		for (const auto &cpt : fct->body)
		{
			isl_map *m = isl_map_copy(cpt->schedule);
			result = isl_union_map_union(isl_union_map_from_map(m), result);
		}

	return result;
}


// Halide IR related methods

void halide_IR_dump(Halide::Internal::Stmt s)
{
	if (DEBUG)
	{
		str_dump("\n\n");
		str_dump("\nGenerated Halide Low Level IR:\n");
		Halide::Internal::IRPrinter pr(std::cout);
	    	pr.print(s);
		str_dump("\n\n\n\n");
	}
}
