#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

void beam_search::search(computation_graph const& cg)
{
    std::vector<schedule_info> successors;
    
    add_interchanges(successors, cg.roots[0]->iterators);
    add_tilings(successors, cg.roots[0]->iterators);
    add_unrollings(successors, cg.roots[0]->iterators);
    
    for (schedule_info& s : successors)
        s.eval = eval_func->evaluate(cg, s);
        
    std::sort(successors.begin(), successors.end(), [] (schedule_info const& a, schedule_info const& b) {
        return a.eval <= b.eval;
    });
    
    for (schedule_info s : successors)
    {
        for (int i : s.interchanged)
            std::cout << i << " ";
            
        std::cout << std::endl;
        
        for (int i : s.tiled)
            std::cout << i << " ";
            
        std::cout << std::endl;
        
        for (int i : s.tiling_factors)
            std::cout << i << " ";
            
        std::cout << std::endl;
        
        std::cout << s.unrolling_factor << std::endl << s.eval << std::endl << std::endl;
    }
}

}
