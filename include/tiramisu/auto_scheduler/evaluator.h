#ifndef _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_
#define _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_

#include <torch/script.h>
#include "core.h"

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
      * Takes as input a computation graph and returns
      * its evaluation.
      */
    virtual float evaluate(computation_graph const& cg, schedule_info const& sched) =0;
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
    
    at::Tensor build_computation_repr(cg_node *node, schedule_info const& sched);
    
public:
    /**
      * model_path is the path to the serialized PyTorch model.
      * The model must be serialized with TorchScript.
      */
    simple_rnn_evaluator(std::string const& model_path);
    virtual ~simple_rnn_evaluator() {}
        
    virtual float evaluate(computation_graph const& cg, schedule_info const& sched);
};

}

#endif
