#ifndef _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_
#define _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_

#include <torch/script.h>
#include "core.h"

namespace tiramisu::auto_scheduler
{

class evaluator
{
    private:
    
    protected:
    
    public:
        virtual ~evaluator() {}
        virtual float evaluate(program_repr const& prog_repr) =0;
};

class eval_dnn_model : public evaluator
{
    private:
    
    protected:
        torch::jit::script::Module model;
    
    public:
        eval_dnn_model(std::string const& model_path);
        virtual ~eval_dnn_model() {}
        
        virtual float evaluate(program_repr const& prog_repr);
};

}

#endif
