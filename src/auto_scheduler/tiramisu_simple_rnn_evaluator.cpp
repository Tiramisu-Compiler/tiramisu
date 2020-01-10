#include <tiramisu/auto_scheduler/evaluator.h>

namespace tiramisu::auto_scheduler
{

simple_rnn_evaluator::simple_rnn_evaluator(std::string const& model_path) 
    : evaluator()
{
    model = torch::jit::load(model_path);
}

float simple_rnn_evaluator::evaluate(computation_graph const& cg)
{

}

}
