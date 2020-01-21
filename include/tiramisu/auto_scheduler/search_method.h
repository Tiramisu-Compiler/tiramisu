#ifndef _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_
#define _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_

#include "core.h"
#include "evaluator.h"

namespace tiramisu::auto_scheduler
{

const std::vector<int> TILING_FACTORS_DEFAULT_LIST = {32, 64, 128};
const std::vector<int> UNROLLING_FACTORS_DEFAULT_LIST = {4, 8, 16};

/**
  * An abstract class that represents a search method.
  * Derive this class and give an implementation of
  * the method "search" to implement a new search algorithm.
  */
class search_method
{
private:
    void add_tilings(std::vector<schedule_info>& schedules, 
                     std::vector<iterator> const& iterators,
                     int it_pos, int nb_tiles, schedule_info base_sched);
    
protected:
    /**
      * The evaluation function used by the search method.
      */
    evaluator *eval_func;
    
    std::vector<int> tiling_factors_list;
    std::vector<int> unrolling_factors_list;
    
public:
    search_method(evaluator *eval_func = nullptr, 
                  std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                  std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
                  
        : eval_func(eval_func), tiling_factors_list(tiling_factors_list),
          unrolling_factors_list(unrolling_factors_list) {}
            
    virtual ~search_method() {}
    
    /**
      * The method to call to start a search.
      * It takes as input a computation graph and returns a list of
      * code transformations.
      */
    virtual void search(computation_graph const& cg) =0;
    
    void add_interchanges(std::vector<schedule_info>& schedules, std::vector<iterator> const& iterators);
    void add_tilings(std::vector<schedule_info>& schedules, std::vector<iterator> const& iterators);
    void add_unrollings(std::vector<schedule_info>& schedules, std::vector<iterator> const& iterators);
        
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
    beam_search(evaluator *eval_func = nullptr, 
                std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        : search_method(eval_func, tiling_factors_list, unrolling_factors_list) {}
        
    virtual ~beam_search() {}
    virtual void search(computation_graph const& cg);
};

}

#endif
