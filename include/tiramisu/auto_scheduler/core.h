#ifndef _H_TIRAMISU_AUTO_SCHEDULER_CORE_
#define _H_TIRAMISU_AUTO_SCHEDULER_CORE_

#include <vector>

#include <tiramisu/core.h>
#include <tiramisu/expr.h>

namespace tiramisu::auto_scheduler
{

class evaluator;
class search_method;

class iterator
{
    public:
        std::string name;
        int low_bound;
        int up_bound;
        
        iterator(std::string const& name, int low_bound, int up_bound)
            : name(name), low_bound(low_bound), up_bound(up_bound) {}
};

class cg_node
{
    public:
        std::vector<iterator> iterators;
        tiramisu::computation* comp;
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
        std::vector<cg_node*> roots;
        
        computation_graph(tiramisu::function *fct);
};

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
