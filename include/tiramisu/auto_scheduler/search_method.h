#ifndef _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_
#define _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_

#include <climits>
#include <cfloat>

#include "auto_scheduler.h"
#include "states_generator.h"
#include "evaluator.h"
#include "utils.h"

namespace tiramisu::auto_scheduler
{

const std::vector<optimization_type> DEFAULT_OPTIMIZATIONS_ORDER = {FUSION, TILING, INTERCHANGE, UNROLLING};

const int DEFAULT_MAX_DEPTH = INT_MAX;

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

    /**
     * The search method use this attribute to generate new states.
     */
    states_generator* states_gen;
    
    int nb_explored_schedules = 0;
    
    float best_evaluation = FLT_MAX;
    syntax_tree *best_ast = nullptr;
    
public:
    search_method(evaluator *eval_func = nullptr, states_generator *states_gen = nullptr)
        : eval_func(eval_func), states_gen(states_gen) {}
            
    virtual ~search_method() {}

    int get_nb_explored_schedules() const { return nb_explored_schedules; }
    float get_best_evaluation() const { return best_evaluation; }
    syntax_tree* get_best_ast() const { return best_ast; }
    
    void set_eval_func(evaluator *eval_func) { this->eval_func = eval_func; }
    
    /**
      * The method to call to start a search.
      * It takes as input an AST and returns a list of
      * code transformations.
      */
    virtual void search(syntax_tree& ast) =0;
};

/**
  * Implements the beam search algorithm.
  */
class beam_search : public search_method
{
private:
    
protected:
    /**
     * The beam size used by beam search.
     */
    int beam_size;
    
    /**
     * The maximum depth of the search tree.
     */
    int max_depth;
    
public:
    beam_search(int beam_size, int max_depth = DEFAULT_MAX_DEPTH, evaluator *eval_func = nullptr, states_generator *states_gen = nullptr)
        : search_method(eval_func, states_gen), beam_size(beam_size), max_depth(max_depth) {}
        
    virtual ~beam_search() {}

    /**
      * The method to call to start a search.
      * It takes as input an AST and returns a list of
      * code transformations.
      */
    virtual void search(syntax_tree& ast);
};

}

#endif
