#ifndef _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_
#define _TIRAMISU_AUTO_SCHEDULER_SEARCH_METHOD_

#include "core.h"
#include "evaluator.h"

namespace tiramisu::auto_scheduler
{

class search_method
{
    private:
    
    protected:
        evaluator *eval_func;
    
    public:
        search_method(evaluator *eval_func = nullptr) 
            : eval_func(eval_func) {}
            
        virtual ~search_method() {}
        virtual void search(computation_graph const& cg) =0;
        
        void set_eval_func(evaluator *eval_func) { this->eval_func = eval_func; }
};

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
