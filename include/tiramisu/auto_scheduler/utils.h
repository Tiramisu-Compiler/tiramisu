#ifndef _TIRAMISU_AUTO_SCHEDULER_UTILS_
#define _TIRAMISU_AUTO_SCHEDULER_UTILS_

namespace tiramisu::auto_scheduler
{

/**
 * Return true if an iterator having extent = it_extent can
 * be split perfectly by a factor = split_fact.
 */
inline bool can_split_iterator(int it_extent, int split_fact)
{
    return it_extent > split_fact && it_extent % split_fact == 0;
}

}

#endif
