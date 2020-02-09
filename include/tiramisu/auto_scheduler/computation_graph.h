#ifndef _H_TIRAMISU_AUTO_SCHEDULER_COMPUTATION_GRAPH_
#define _H_TIRAMISU_AUTO_SCHEDULER_COMPUTATION_GRAPH_

#include <tiramisu/core.h>

namespace tiramisu::auto_scheduler
{

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
        
    std::vector<cg_node*> children;
        
    ~cg_node()
    {
        for (cg_node* child : children)
            delete child;
    }
};

class computation_graph
{
public:
    /**
      * Computation graph root nodes.
      */
    std::vector<cg_node*> roots;

    float evaluation;
        
    computation_graph() {}
    computation_graph(tiramisu::function *fct);
};

}

#endif