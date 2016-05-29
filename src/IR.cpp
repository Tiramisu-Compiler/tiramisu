#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <DebugIR.h>
#include <IR.h>

std::map<std::string, Halide::Internal::Stmt> stmts_list;

/* Schedule the iteration space.  */
isl_union_map *create_schedule_map(isl_ctx *ctx, std::string map)
{
	isl_union_map *schedule_map = isl_union_map_read_from_str(ctx,
			map.c_str());

	return schedule_map;
}

isl_ast_node *stmt_halide_code_generator(isl_ast_node *node, isl_ast_build *build, void *user)
{
#if 0
	isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
	isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
	isl_id *id = isl_ast_expr_get_id(arg);
	isl_ast_expr_free(arg);
	std::string computation_name(isl_id_get_name(id));
	isl_id_free(id);

	Halide::Internal::Stmt s = stmts_list.find(computation_name)->second; 
	user = (void *) &s;
#endif

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
			case isl_ast_op_min:
				result = Halide::Internal::Min::make(op0, op1);
				break;
			case isl_ast_op_max:
				result = Halide::Internal::Max::make(op0, op1);
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
			case isl_ast_op_and:
				result = Halide::Internal::And::make(op0, op1);
				break;
			case isl_ast_op_or:
				result = Halide::Internal::Or::make(op0, op1);
				break;
			case isl_ast_op_minus:
				result = Halide::Internal::Sub::make(Halide::Expr(0), op0);
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

#if 0
	Halide::Internal::Stmt *s = (Halide::Internal::Stmt *) user;

	isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
	char *iterator_str = isl_ast_expr_to_str(iter);

	isl_ast_expr *init = isl_ast_node_for_get_init(node);
	isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
	isl_ast_expr *cond_upper_bound_isl_format;
	if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le || isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
		cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
	else
		Error("The for loop upper bound is not an isl_est_expr of type le or lt" ,1);

        Halide::Expr init_expr = create_halide_expr_from_isl_ast_expr(init);
	Halide::Expr cond_upper_bound_halide_format =  create_halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format);
	*s = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format, Halide::Internal::ForType::Serial,
			Halide::DeviceAPI::Host, *s);
#endif 

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

Halide::Internal::Stmt generate_Halide_stmt_from_isl_node(isl_ast_node *node)
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
			result = Halide::Internal::Block::make(generate_Halide_stmt_from_isl_node(child), Halide::Internal::Stmt());
		
			for (i = 1; i < isl_ast_node_list_n_ast_node(list); i++)
			{
				child = isl_ast_node_list_get_ast_node(list, i);
				result = Halide::Internal::Block::make(result, generate_Halide_stmt_from_isl_node(child));
			}
		}
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_for)
	{
		isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
		char *iterator_str = isl_ast_expr_to_str(iter);

		isl_ast_expr *init = isl_ast_node_for_get_init(node);
		isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
		isl_ast_node *body = isl_ast_node_for_get_body(node);
		isl_ast_expr *cond_upper_bound_isl_format;
		if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le || isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
			cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
		else
			Error("The for loop upper bound is not an isl_est_expr of type le or lt" ,1);

		Halide::Expr init_expr = create_halide_expr_from_isl_ast_expr(init);
		Halide::Expr cond_upper_bound_halide_format =  create_halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format);
		result = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format, Halide::Internal::ForType::Serial,
				Halide::DeviceAPI::Host, generate_Halide_stmt_from_isl_node(body));
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_user)
	{
		isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
		isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
		isl_id *id = isl_ast_expr_get_id(arg);
		isl_ast_expr_free(arg);
		std::string computation_name(isl_id_get_name(id));
		isl_id_free(id);

		result = stmts_list.find(computation_name)->second; 
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

void Computation::dump()
{
	if (DEBUG)
	{
		std::cout << "Computation \"" << this->name << "\"" << std::endl;
		isl_set_dump(this->iter_space);
		str_dump("Halide statement:\n");
		Halide::Internal::IRPrinter pr(std::cout);
	    	pr.print(this->stmt);
		str_dump("\n");

	}
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


// Program related methods

void IRProgram::dump_ISIR()
{
	if (DEBUG)
	{
		str_dump("\nIteration Space IR:\n");
		for (const auto &fct : this->functions)
		       fct->dump_ISIR();
		str_dump("\n\n\n");
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

		std::cout << std::endl;
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


// Halide IR related methods

void halide_IR_dump(Halide::Internal::Stmt s)
{
	if (DEBUG)
	{
		str_dump("\n\n");
		str_dump("\nGenerated Halide Low Level IR:\n");
		Halide::Internal::IRPrinter pr(std::cout);
	    	pr.print(s);
		str_dump("\n\n");
	}
}
