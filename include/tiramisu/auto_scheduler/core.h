#ifndef _H_TIRAMISU_AUTO_SCHEDULER_CORE_
#define _H_TIRAMISU_AUTO_SCHEDULER_CORE_

#include <vector>

#include <tiramisu/core.h>
#include <tiramisu/expr.h>

namespace tiramisu::auto_scheduler
{

class evaluator;
class search_method;

class loop_iterator
{
    private:
    
    protected:
        int low_bound;
        int up_bound;
        
        std::vector<loop_iterator*> children;
        std::vector<tiramisu::computation*> comps;
        
    public:
        loop_iterator(int low_bound, int up_bound)
            : low_bound(low_bound), up_bound(up_bound) {}
            
        ~loop_iterator()
        {
            for (loop_iterator* child : children)
                delete child;
        }
        
        void add_child(loop_iterator* child) { children.push_back(child); }
        void add_computation(tiramisu::computation* comp) { comps.push_back(comp); }
        
        void set_low_bound(int low_bound) { this->low_bound = low_bound; }
        void set_up_bound(int up_bound) { this->up_bound = up_bound; }
        
        std::vector<loop_iterator*> get_children() const { return children; }
        std::vector<tiramisu::computation*> get_computations() { return comps; }
};

class program_repr
{
    private:
    
    protected:
        std::vector<loop_iterator*> iterators_tree;
        
    public:
        program_repr(tiramisu::function *fct);
        
        std::vector<loop_iterator*> get_iterators() const { return iterators_tree; }
};

class auto_scheduler
{
    private:
        
    protected:
        program_repr prog_repr;
        
        search_method *searcher;
        evaluator *eval_func;
        
        tiramisu::function *fct;
        
    public:
        auto_scheduler(search_method *searcher, evaluator *eval_func,
                       tiramisu::function *fct = tiramisu::global::get_implicit_function());
};

}

#endif
