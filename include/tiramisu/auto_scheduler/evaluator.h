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
class evaluation_function
{
private:
    
protected:
    
public:
    virtual ~evaluation_function() {}
    
    /**
     * Takes as input an abstract syntax tree and returns
     * its evaluation.
     */
    virtual float evaluate(syntax_tree& ast) =0;
};

/**
 * Evaluate programs by compiling and executing them.
 */
class evaluate_by_execution : public evaluation_function
{
private:

protected:
    /**
     * The following three attributes are parameters for Halide.
     */
    std::vector<Halide::Target::Feature> halide_features = {
        Halide::Target::AVX,
        Halide::Target::SSE41,
        //Halide::Target::AVX2,
	    //Halide::Target::FMA,
        Halide::Target::LargeBuffers
    };
    
    Halide::Target halide_target;
    std::vector<Halide::Argument> halide_arguments;

    /**
     * The program to compile and to execute.
     */
	tiramisu::function *fct;
	
	/**
	 * The name of the ".o" to generate (it will contain the compiled Tiramisu program).
	 */
    std::string obj_filename;
    
    /**
     * The command that will be used to execute the wrapper to measure execution time.
     */
    std::string wrapper_cmd;

public:
    /**
     * arguments : the input and output buffers of the program.
     */
    evaluate_by_execution(std::vector<tiramisu::buffer*> const& arguments, 
						  std::string const& obj_filename, 
						  std::string const& wrapper_cmd,
						  tiramisu::function *fct = tiramisu::global::get_implicit_function());
    
	/**
	 * Apply the specified optimizations, compile the program and execute it.
	 */
    virtual float evaluate(syntax_tree& ast);

    /**
     * Apply the specified optimizations, compile the program and execute it.
     * Returns a vector of measured execution times
     * If the timeout parameter is defined, it stops the execution after NB_EXEC*timeout seconds
     * If exit_on_timeout is set to true, it raises an error when the timeout is reached and termintes the program
     */
    std::vector<float> get_measurements(syntax_tree &ast,  bool exit_on_timeout = false, float timeout = 0);
};

/**
 * This evaluation function uses system pipes to communicate with an ML model
 * that will evaluate schedules.
 * JSON is used to transfer information about the schedule to evaluate.
 */
class evaluate_by_learning_model : public evaluation_function
{
private:

protected:
    /**
     * The pipe on which to write information about schedules to evaluate.
     */
    FILE *model_write;
    
    /**
     * The pipe on which to read the evaluation of a schedule.
     */
    FILE *model_read;

public:
    /**
     * cmd_path : path to the program containing the ML model.
     * cmd_args : arguments to pass to the program in cmd_path.
     */
    evaluate_by_learning_model(std::string const& cmd_path, std::vector<std::string> const& cmd_args);
    
	/**
	 * Call the model and return its evaluation.
	 */
    virtual float evaluate(syntax_tree& ast);
    
    /**
     * Return a JSON representation of the program represented by the AST.
     * Uses the function : represent_computations_from_nodes. 
     */
    static std::string get_program_json(syntax_tree const& ast);
    
    /**
     * A recursive subroutine that represents in JSON the computations of a given tree.
     */
    static void represent_computations_from_nodes(ast_node *node, std::string& computations_json, int& comp_absolute_order);
    
    /**
     * Return a JSON representation of the schedule of the given AST.
     */
    static std::string get_schedule_json(syntax_tree const& ast);
    
    // --------------------------------------------------------------------------------- //
    
    /**
     * A recursive subroutine that represents in JSON the loop structure of a given tree.
     * This function is not called by this class, but by the AST class.
     * The result of this function is stored in ast.iterators_json.
     * The function get_program_json uses directly ast.iterators_json to get the JSON of iterators.
     *
     * This is done for the following reason : the autoscheduler needs to change the structure
     * of the AST after each optimization. On the other hand, the structure of the original tree
     * must be passed to the model to get the evaluation of a schedule. So we store in
     * ast.iterators_json the JSON corresponding to the initial structure of the tree, in order
     * to retrieve it later, even though the structure of the AST has changed.
     */
    static void represent_iterators_from_nodes(ast_node *node, std::string& iterators_json);
    
    /**
     * Transform the structure of the given AST into JSON.
     * The model needs this information in the schedule.
     *
     * Like represent_iterators_from_nodes, this function is called by the AST class.
     * AST class calls this function every time the optimization UNFUSE is applied.
     * The result is retrieved in ast.tree_structure_json.
     */
    static std::string get_tree_structure_json(syntax_tree const& ast);
    
    /**
     * A recursive subroutine used by get_tree_structure_json(syntax_tree const& ast).
     */
    static std::string get_tree_structure_json(ast_node *node);
};

}

#endif
