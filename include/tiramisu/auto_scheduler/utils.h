#ifndef _TIRAMISU_AUTO_SCHEDULER_UTILS_
#define _TIRAMISU_AUTO_SCHEDULER_UTILS_

#include <vector>

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

/**
 * Returns true if the extent is bigger than split factor,
 * (.i.e more than one iteration would be produced after splitting) 
*/
inline bool can_split_iterator_sup(int it_extent, int split_fact)
{
    return it_extent > split_fact;
}

}

#endif
