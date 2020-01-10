#ifndef _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_
#define _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_

#include "core.h"
#include "evaluator.h"

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
    
public:
    search_method(evaluator *eval_func = nullptr) 
        : eval_func(eval_func) {}
            
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
    
public:
    beam_search(evaluator *eval_func = nullptr)
        : search_method(eval_func) {}
        
    virtual ~beam_search() {}
    virtual void search(computation_graph const& cg);
};

}

#endif
