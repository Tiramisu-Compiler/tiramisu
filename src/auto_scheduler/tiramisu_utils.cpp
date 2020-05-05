#include <tiramisu/auto_scheduler/utils.h>
#include <tiramisu/auto_scheduler/ast.h>
#include <cmath>

namespace tiramisu::auto_scheduler
{

std::vector<double> compute_softmax(std::vector<syntax_tree*> const& ast_list)
{
    std::vector<double> ret(ast_list.size());
    double exp_sum = 0.f;
    
    for (int i = 0; i < ast_list.size(); ++i)
    {
        ret[i] = std::exp(ast_list[i]->evaluation);
        exp_sum += ret[i];
    }
    
    for (int i = 0; i < ast_list.size(); ++i)
        ret[i] /= exp_sum;
    
    return ret;
}

}
