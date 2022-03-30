#ifndef _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_
#define _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_

#include <climits>
#include <cfloat>

#include "auto_scheduler.h"
#include "schedules_generator.h"
#include "evaluator.h"
#include "utils.h"
// max matrices to be explored per computation
const int MAX_MAT_DEPTH = 1; 
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
    syntax_tree* get_best_ast() const { return best_ast; }
    
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
    virtual void search_save(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0) =0;
    /**
      * The method to call to start a search.
      * The explored schedules annotation and their execution time are stored in schedules_annotations
      */
    virtual void search_save_matrix(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0) =0;
    virtual void explore_fusion(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0)=0;
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
    beam_search(int beam_size, int max_depth = DEFAULT_MAX_DEPTH, evaluation_function *eval_func = nullptr, schedules_generator *scheds_gen = nullptr)
        : search_method(eval_func, scheds_gen), beam_size(beam_size), max_depth(max_depth) {}
        
    virtual ~beam_search() {}

    virtual void search(syntax_tree& ast);

    /**
     * Searches for the best schedule and saves the explored schedules and their execution time
     *
     */
    virtual void search_save(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);
    virtual void search_save_matrix(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);
    virtual void explore_fusion(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);
    virtual void explore_parallelization(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);

};

/**
 * Implements the MCTS search method.
 */
//class mcts : public search_method
//{
//private:
//
//protected:
//    /**
//     * The number of times to sample schedules from the search tree.
//     */
//    int nb_samples;
//
//    /**
//     * The number of schedules to execute at the end of the search
//     * to return the best schedule.
//     */
//    int topk;
//
//    /**
//     * The maximum depth of the search tree.
//     */
//    int max_depth;
//
//public:
//    mcts(int nb_samples, int topk, int max_depth = DEFAULT_MAX_DEPTH, evaluation_function *eval_func = nullptr, evaluate_by_execution *exec_eval = nullptr, schedules_generator *scheds_gen = nullptr)
//        : search_method(eval_func, scheds_gen), nb_samples(nb_samples),
//          topk(topk), max_depth(max_depth)
//    { set_exec_eval(exec_eval); }
//
//    virtual ~mcts() {}
//
//    virtual void search(syntax_tree& ast);
//
//    /**
//     * Searches for the best schedule and saves the explored schedules and their execution time
//     *
//     */
//    virtual void search_save(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);
//};

// ----------------------------------------------------------------------- //

/**
 * Same as beam search, but executes the topk schedules found, and return
 * the best of them.
 */
//class beam_search_topk : public search_method
//{
//private:
//
//protected:
//    /**
//     * The beam size used by beam search.
//     */
//    int beam_size;
//
//    /**
//     * The number of schedules to execute at the end of the search
//     * to return the best schedule.
//     */
//    int topk;
//
//    /**
//     * The maximum depth of the search tree.
//     */
//    int max_depth;
//
//    /**
//     * The list of schedules found.
//     * Used to return the TOPK schedules.
//     */
//    std::vector<syntax_tree*> schedules;
//
//public:
//    beam_search_topk(int beam_size, int topk, int max_depth = DEFAULT_MAX_DEPTH, evaluation_function *eval_func = nullptr, evaluate_by_execution *exec_eval = nullptr, schedules_generator *scheds_gen = nullptr)
//        : search_method(eval_func, scheds_gen), beam_size(beam_size),
//          topk(topk), max_depth(max_depth)
//    { set_exec_eval(exec_eval); }
//
//    virtual ~beam_search_topk() {}
//
//    virtual void search(syntax_tree& ast);
//
//    /**
//     * Searches for the best schedule and saves the explored schedules and their execution time
//     *
//     */
//    virtual void search_save(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout=0);
//
//    /**
//     * A subroutine used by search(syntax_tree& ast);
//     */
//    void beam_search_subroutine(syntax_tree& ast);
//};

/**
 * Use this class if you want to assess the accuracy of the model by using beam search.
 * This class performs beam search with the model, and also measures the execution time
 * of each schedule found. This can be used to compare the predicted and the measured speedups.
 */
//class beam_search_accuracy_evaluator : public beam_search
//{
//private:
//
//protected:
//    /**
//     * Stores the evaluation of each schedule with the model.
//     */
//    std::vector<float> model_evals_list;
//
//    /**
//     * Stores the evaluation of each schedule with execution.
//     */
//    std::vector<float> exec_evals_list;
//
//public:
//    beam_search_accuracy_evaluator(int beam_size, int max_depth = DEFAULT_MAX_DEPTH, evaluation_function *model_eval = nullptr, evaluate_by_execution *exec_eval = nullptr, schedules_generator *scheds_gen = nullptr)
//        : beam_search(beam_size, max_depth, model_eval, scheds_gen)
//    { set_exec_eval(exec_eval); }
//
//    virtual ~beam_search_accuracy_evaluator() {}
//
//    virtual void search(syntax_tree& ast);
//
//    /**
//     * Print the evaluations given by the model and the evaluations
//     * given by execution.
//     */
//    void print_evals_list() const
//    {
//        for (int i = 0; i < model_evals_list.size(); ++i)
//            std::cout << model_evals_list[i] << " " << exec_evals_list[i] << std::endl;
//    }
//};

}

#endif
