#ifndef _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_
#define _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_

#include <climits>
#include <cfloat>

#include "auto_scheduler.h"
#include "schedules_generator.h"
#include "evaluator.h"
#include "utils.h"
// max matrices to be explored per computation
const int MAX_MAT_DEPTH = 4; 
namespace tiramisu::auto_scheduler
{

//const std::vector<optimization_type> DEFAULT_OPTIMIZATIONS_ORDER = {UNFUSE, INTERCHANGE, SKEWING, PARALLELIZE, TILING};
const std::vector<optimization_type> DEFAULT_OPTIMIZATIONS_ORDER = { PARALLELIZE, TILING, UNROLLING};
//const std::vector<optimization_type> DEFAULT_OPTIMIZATIONS_ORDER = {UNFUSE, INTERCHANGE, SKEWING, PARALLELIZE, TILING, UNROLLING, VECTORIZATION};
const int NB_OPTIMIZATIONS = DEFAULT_OPTIMIZATIONS_ORDER.size();
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
    evaluation_function *eval_func;

    /**
     * The search method use this attribute to generate new schedules.
     */
    schedules_generator* scheds_gen;
    
    /**
     * The number of schedules explored.
     */
    int nb_explored_schedules = 0;
    
    /**
     * The evaluation of the best schedule so far.
     * At the end of search, contains the evaluation of the best AST found.
     */
    float best_evaluation = FLT_MAX;
    
    /**
     * The best AST so far.
     * At the end of search, contains the best AST found.
     */
    syntax_tree *best_ast = nullptr;
    
    /**
     * An evaluator returning the execution time of a program.
     * Not mandatory, can be usefull for some search methods (like MCTS).
     */
    evaluate_by_execution *exec_eval = nullptr;
    
public:
    search_method(evaluation_function *eval_func = nullptr, schedules_generator *scheds_gen = nullptr)
        : eval_func(eval_func), scheds_gen(scheds_gen) {}
            
    virtual ~search_method() {}

    int get_nb_explored_schedules() const { return nb_explored_schedules; }
    float get_best_evaluation() const { return best_evaluation; }
    void set_best_evaluation(float best_eval) { best_evaluation = best_eval; }
    syntax_tree* get_best_ast() const { return best_ast; }
    void set_best_ast(syntax_tree* best_explored_ast)  { best_ast = best_explored_ast; }
    
    void set_eval_func(evaluation_function *eval_func) { this->eval_func = eval_func; }
    void set_exec_eval(evaluate_by_execution *exec_eval) { this->exec_eval = exec_eval; }
        
    /**
      * The method to call to start a search.
      * At the end of search, the best AST can be found in best_ast,
      * and its evaluation in best_evaluation.
      */
    virtual void search(syntax_tree& ast) =0;


    /**
      * The method to call to start a search.
      * The explored schedules annotation and their execution time are stored in schedules_annotations
      */
    virtual std::vector<syntax_tree*> search_save(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0) =0;
    /**
      * The method to call to start a search.
      * The explored schedules annotation and their execution time are stored in schedules_annotations
      */
    virtual std::vector<syntax_tree*> search_save_matrix(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0) =0;
    virtual void explore_schedules(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0)=0;

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
    
    
public:
    beam_search(int beam_size, evaluation_function *eval_func = nullptr, schedules_generator *scheds_gen = nullptr)
        : search_method(eval_func, scheds_gen), beam_size(beam_size) {}
        
    virtual ~beam_search() {}

    virtual void search(syntax_tree& ast);

    /**
     * Searches for the best schedule and saves the explored schedules and their execution time
     *
     */
    virtual std::vector<syntax_tree*> search_save(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);
    virtual void explore_schedules(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);
    virtual std::vector<syntax_tree*> search_save_matrix(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);

};

}

#endif
