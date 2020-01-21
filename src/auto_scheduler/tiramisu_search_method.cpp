#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{
void search_method::add_interchanges(std::vector<schedule_info>& schedules, std::vector<iterator> const& iterators)
{
    if (schedules.empty())
        schedules.push_back(schedule_info(iterators.size()));
        
    int nb_scheds = schedules.size();
    
    for (int i = 0; i < nb_scheds; ++i)
        for (int it1 = 0; it1 < iterators.size(); ++it1)
            for (int it2 = it1 + 1; it2 < iterators.size(); ++it2)
            {
                schedule_info new_sched = schedules[i];
                
                new_sched.interchanged[it1] = 1;
                new_sched.interchanged[it2] = 1;
                
                schedules.push_back(new_sched);
            }
}

void search_method::add_tilings(std::vector<schedule_info>& schedules, std::vector<iterator> const& iterators)
{
    if (schedules.empty())
        schedules.push_back(schedule_info(iterators.size()));
        
    int nb_scheds = schedules.size();
    
    for (int i = 0; i < nb_scheds; ++i)
        for (int nb_tiles = 2; nb_tiles <= 3; ++nb_tiles)
            for (int it_beg = 0; it_beg <= iterators.size() - nb_tiles; ++it_beg)
                add_tilings(schedules, iterators, it_beg, nb_tiles - 1, schedules[i]);
}

void search_method::add_tilings(std::vector<schedule_info>& schedules, 
                                std::vector<iterator> const& iterators,
                                int it_pos, int nb_tiles, schedule_info base_sched)
{
    int extent = iterators[it_pos].up_bound - iterators[it_pos].low_bound + 1;
    
    for (int tiling_fact : tiling_factors_list)
    {
        if (extent % tiling_fact != 0 || extent / tiling_fact <= 1)
            continue;
            
        schedule_info new_sched = base_sched;
        
        new_sched.tiled[it_pos] = 1;
        new_sched.tiling_factors[it_pos] = tiling_fact;
        
        if (nb_tiles == 0)
            schedules.push_back(new_sched);

        else
            add_tilings(schedules, iterators, it_pos + 1, nb_tiles - 1, new_sched);
    }
}

void search_method::add_unrollings(std::vector<schedule_info>& schedules, std::vector<iterator> const& iterators)
{
    if (schedules.empty())
        schedules.push_back(schedule_info(iterators.size()));
        
    int nb_scheds = schedules.size();
    
    for (int i = 0; i < nb_scheds; ++i)
        for (int unrolling_fact : unrolling_factors_list)
        {
            schedule_info new_sched = schedules[i];
            
            new_sched.unrolling_factor = unrolling_fact;
            schedules.push_back(new_sched);
        }
}

}
