#ifndef _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_
#define _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_

#include "auto_scheduler.h"
#include "utils.h"

namespace tiramisu::auto_scheduler
{

/**
  * An abstract class that represents an evaluation function.
  * Derive this class and implement the method "evaluate" to
  * create new evaluation functions.
  */
class evaluator
{
private:
    
protected:
    
public:
    virtual ~evaluator() {}
    
    /**
     * Takes as input an abstract syntax tree and returns
     * its evaluation.
     */
    virtual float evaluate(syntax_tree& ast) =0;
    
    /**
     * Indicates if the given ast should be transform by using ast.transform_ast().
     */
    virtual bool should_transform_ast(syntax_tree const& ast) = 0;
};

/**
 * Evaluate programs by compiling and executing them.
 */
class evaluate_by_execution : public evaluator
{
private:

protected:
    std::vector<Halide::Target::Feature> halide_features = {
        Halide::Target::AVX,
        Halide::Target::SSE41,
        //Halide::Target::AVX2,
	    //Halide::Target::FMA,
        Halide::Target::LargeBuffers
    };
    
    Halide::Target halide_target;
    std::vector<Halide::Argument> halide_arguments;

	tiramisu::function *fct;
    std::string obj_filename;
    std::string wrapper_cmd;

public:
    evaluate_by_execution(tiramisu::function *fct, 
						  std::vector<tiramisu::buffer*> const& arguments, 
						  std::string const& obj_filename, 
						  std::string const& wrapper_cmd);
    
	/**
	 * Apply the specified optimizations, compile the program
	 * and execute it.
	 */
    virtual float evaluate(syntax_tree& ast);
    
    /**
     * Indicates if the given ast should be transform by using ast.transform_ast().
     */
    virtual bool should_transform_ast(syntax_tree const& ast) { return true; }
};

class tree_lstm_evaluator : public evaluator
{
private:

protected:
    FILE *model_write;
    FILE *model_read;

public:
    tree_lstm_evaluator(std::string const& cmd_path, std::vector<std::string> const& cmd_args);
    
	/**
	 * Call the model and return its evaluation.
	 */
    virtual float evaluate(syntax_tree& ast);
    
    /**
     * Indicates if the given ast should be transform by using ast.transform_ast().
     */
    virtual bool should_transform_ast(syntax_tree const& ast) { return true; }
    
    /**
     * Return a JSON representation of the program represented by the AST.
     */
    std::string get_program_json(syntax_tree const& ast);
    
    /**
     *
     */
    static void represent_iterators_from_nodes(ast_node *node, std::string& iterators_json);
    
    /**
     *
     */
    static void represent_computations_from_nodes(ast_node *node, std::string& computations_json, int& comp_absolute_order);
    
    /**
     * Return a JSON representation of the schedule of the given AST.
     */
    std::string get_schedule_json(syntax_tree const& ast);
    
    /**
     *
     */
    static std::string get_tree_structure_json(syntax_tree const& ast);
    
    /**
     *
     */
    static std::string get_tree_structure_json(ast_node *node);
};

}

#endif
