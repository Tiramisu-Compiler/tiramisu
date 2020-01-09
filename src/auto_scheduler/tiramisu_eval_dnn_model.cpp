#include <tiramisu/auto_scheduler/evaluator.h>

namespace tiramisu::auto_scheduler
{

eval_dnn_model::eval_dnn_model(std::string const& model_path) 
    : evaluator()
{
    model = torch::jit::load(model_path);
}

float eval_dnn_model::evaluate(computation_graph const& cg)
{

}

}
