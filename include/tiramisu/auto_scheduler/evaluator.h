#ifndef _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_
#define _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_

#include <torch/script.h>
#include "auto_scheduler.h"
#include "utils.h"

#define MAX_NB_ITERATORS 4
#define MAX_NB_ACCESSES 17

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
    virtual float evaluate(syntax_tree const& ast) =0;
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
	 * Apply the optimizations specified by the syntax tree
	 * using the Tiramisu API.
	 */
    void apply_optimizations(syntax_tree const& ast);
    
    void apply_fusions(syntax_tree const& ast);
    tiramisu::computation* apply_fusions(ast_node *node, tiramisu::computation *last_comp, int dimension);
    
	/**
	 * Apply the specified optimizations, compile the program
	 * and execute it.
	 */
    virtual float evaluate(syntax_tree const& ast);
};

}

#endif
