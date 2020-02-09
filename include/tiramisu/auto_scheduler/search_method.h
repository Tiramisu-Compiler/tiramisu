#ifndef _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_
#define _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_

#include "auto_scheduler.h"
#include "states_generator.h"
#include "evaluator.h"
#include "utils.h"

namespace tiramisu::auto_scheduler
{

/**
  * An abstract class that represents a search method.
  * Derive this class and give an implementation of
  * the method "search" to implement a new search algorithm.
  */
class search_method
{
private:
    
protected:
    /**
      * The evaluation function used by the search method.
      */
    evaluator *eval_func;

    states_generator* states_gen;
    
public:
    search_method(evaluator *eval_func = nullptr, states_generator *states_gen = nullptr)
        : eval_func(eval_func), states_gen(states_gen) {}
            
    virtual ~search_method() {}
    
    /**
      * The method to call to start a search.
      * It takes as input a computation graph and returns a list of
      * code transformations.
      */
    virtual void search(computation_graph const& cg) =0;
        
    void set_eval_func(evaluator *eval_func) { this->eval_func = eval_func; }
};

/**
  * Implements the beam search algorithm.
  */
class beam_search : public search_method
{
private:
    
protected:
    int beam_size;
    
public:
    beam_search(int beam_size, evaluator *eval_func = nullptr, states_generator *states_gen = nullptr)
        : search_method(eval_func, states_gen), beam_size(beam_size) {}
        
    virtual ~beam_search() {}
    virtual void search(computation_graph const& cg);
};

}

#endif
