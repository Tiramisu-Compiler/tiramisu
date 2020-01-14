#ifndef _H_TIRAMISU_AUTO_SCHEDULER_CORE_
#define _H_TIRAMISU_AUTO_SCHEDULER_CORE_

#include <vector>

#include <tiramisu/core.h>
#include <tiramisu/expr.h>

namespace tiramisu::auto_scheduler
{

class evaluator;
class search_method;

/**
  * An iterator is defined by its name, its lower
  * bound and its upper bound.
  */
class iterator
{
public:
    std::string name;
    int low_bound;
    int up_bound;
        
    iterator(std::string const& name, int low_bound, int up_bound)
        : name(name), low_bound(low_bound), up_bound(up_bound) {}
};

/**
  * An access matrix contains information
  * about the access pattern of a buffer.
  */
class access_matrix
{
public:
    int buffer_id;
    std::vector<std::vector<int>> matrix;
    
    access_matrix(int nb_iterators, int nb_dims)
        : matrix(nb_dims)
    {
        for (int i = 0; i < nb_dims; ++i)
            matrix[i] = std::vector<int>(nb_iterators + 1, 0);
    }
};

/**
  * Contains information about a schedule.
  *
  * - A pair of dimensions to interchange.
  * - The list of iterators to tile, with the tiling factors.
  * - The unrolling factor.
  */
class schedule_info
{
public:
    std::vector<int> interchanged;
    
    std::vector<int> tiled;
    std::vector<int> tiling_factors;
    
    int unrolling_factor;
    
    schedule_info(int nb_iterators)
        : interchanged(nb_iterators), tiled(nb_iterators), 
          tiling_factors(nb_iterators), unrolling_factor(0) {}
};

/**
  * Computation graph node.
  */
class cg_node
{
public:
    tiramisu::computation* comp;
    
    /**
      * Contains the iterators of the computation.
      */
    std::vector<iterator> iterators;
    
    /**
      * Contains all the access matrices
      */
    std::vector<access_matrix> accesses;
        
    std::vector<cg_node*> children;
        
    ~cg_node()
    {
        for (cg_node* child : children)
            delete child;
    }
    
    int get_iterator_pos_by_name(std::string const& name)
    {
        for (int i = 0; i < iterators.size(); ++i)
            if (iterators[i].name == name)
                return i;
                
        return -1;
    }
};

class computation_graph
{
public:
    /**
      * Computation graph root nodes.
      */
    std::vector<cg_node*> roots;
    
    /**
      * Contains the list of the accesses used in
      * the computation graph.
      */
    std::vector<std::string> accesses_names;
        
    computation_graph(tiramisu::function *fct);
    
    /**
      * Search for accesses in the expression e, and add
      * their access matrices to the given node.
      */
    void build_access_matrices(tiramisu::expr const& e, cg_node *node);
    
    /**
      * Add to node the access matrix of the given access.
      */
    void add_access_matrix(tiramisu::expr const& acc_expr, cg_node *node);
    
    /**
      * Fill acc_raw given the access expression e.
      */
    void fill_matrix_raw(tiramisu::expr const& e, cg_node *node, std::vector<int>& acc_raw, bool minus = false);
};

/**
  * The core class for the autoscheduler.
  * The user must provide the program to optimize, the evaluation
  * function and the search method.
  */
class auto_scheduler
{
private:
        
protected:
    computation_graph cg;
        
    search_method *searcher;
    evaluator *eval_func;
        
    tiramisu::function *fct;
        
public:
    auto_scheduler(search_method *searcher, evaluator *eval_func,
                   tiramisu::function *fct = tiramisu::global::get_implicit_function());
};

}

#endif
