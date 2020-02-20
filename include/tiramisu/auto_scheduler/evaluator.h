#ifndef _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_
#define _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_

#include <torch/script.h>
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
	 * Apply the specified optimizations, compile the program
	 * and execute it.
	 */
    virtual float evaluate(syntax_tree const& ast);
};

/**
  * Implements an evaluation function that uses a simple
  * RNN model.
  *
  * We use LibTorch to handle DNN models in C++.
  */
class simple_rnn_evaluator : public evaluator
{
private:

protected:
    /**
     * The model to use as an evaluation function.
     */
    torch::jit::script::Module model;

public:
    /**
      * model_path is the path to the serialized PyTorch model.
      * The model must be serialized with TorchScript.
      */
    simple_rnn_evaluator(std::string const& model_path);
    
	/**
	 * Call the model and return its evaluation.
	 */
    virtual float evaluate(syntax_tree const& ast);
    
    /**
     * Transform the AST to a tensor that can be fed to the model.
     */
    at::Tensor get_computation_repr(dnn_schedule const& sched, std::vector<dnn_access_matrix> const& accesses);
    
    const int MAX_NB_ITERATORS = 4;
    const int MAX_NB_ACCESSES = 17;
    const int ITERATORS_REPR_SIZE = 5;
    const int ACCESS_REPR_SIZE = MAX_NB_ITERATORS * (MAX_NB_ITERATORS + 1) + 1;
    const int COMPUTATION_REPR_SIZE = 379;
};

}

#endif
