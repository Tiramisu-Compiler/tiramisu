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

const std::vector<optimization_type> DEFAULT_OPTIMIZATIONS_ORDER = {FUSION, INTERCHANGE, TILING, UNROLLING};

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

class beam_search_accuracy_evaluator : public beam_search
{
private:

protected:
    evaluator *exec_eval;
    
    std::vector<float> model_evals_list;
    std::vector<float> exec_evals_list;
    
public:
    beam_search_accuracy_evaluator(int beam_size, int max_depth = DEFAULT_MAX_DEPTH, evaluator *model_eval = nullptr, evaluator *exec_eval = nullptr, states_generator *states_gen = nullptr)
        : beam_search(beam_size, max_depth, model_eval, states_gen), exec_eval(exec_eval) {}
        
    virtual ~beam_search_accuracy_evaluator() {}
    
    void set_exec_eval(evaluator *exec_eval) { this->exec_eval = exec_eval; }

    /**
      * The method to call to start a search.
      * It takes as input an AST and returns a list of
      * code transformations.
      */
    virtual void search(syntax_tree& ast);
    
    /**
     * Print the evaluations given by the model and the evaluations
     * given by execution.
     */
    void print_evals_list() const
    {
        for (int i = 0; i < model_evals_list.size(); ++i)
            std::cout << model_evals_list[i] << " " << exec_evals_list[i] << std::endl;
    }
};

class simple_mcts : public search_method
{
private:

protected:
    /**
     * The number of times to sample schedules from the search tree.
     */
    int nb_samples;
    
    /**
     * The number of schedules to execute at the end of the search
     * to return the best schedule.
     */
    int topk;
    
    /**
     * The maximum depth of the search tree.
     */
    int max_depth;
    
    /**
     * An evaluator returning the execution time of a program.
     */
    evaluator *exec_eval;

public:
    simple_mcts(int nb_samples, int topk, int max_depth = DEFAULT_MAX_DEPTH, evaluator *eval_func = nullptr, evaluator *exec_eval = nullptr, states_generator *states_gen = nullptr)
        : search_method(eval_func, states_gen), nb_samples(nb_samples), topk(topk), max_depth(max_depth), exec_eval(exec_eval) {}
        
    virtual ~simple_mcts() {}
    
    void set_exec_eval(evaluator *exec_eval) { this->exec_eval = exec_eval; }

    /**
      * The method to call to start a search.
      * It takes as input an AST and returns a list of
      * code transformations.
      */
    virtual void search(syntax_tree& ast);
};

}

#endif
