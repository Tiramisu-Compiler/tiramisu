#ifndef _TIRAMISU_AUTO_SCHEDULER_UTILS_
#define _TIRAMISU_AUTO_SCHEDULER_UTILS_

#include <tiramisu/core.h>
#include <tiramisu/expr.h>

namespace tiramisu::auto_scheduler
{

/**
  * An access matrix contains information
  * about the access pattern of a buffer.
  */
class access_matrix
{
public:
    int buffer_id;
    std::vector<std::vector<int>> matrix;
    
    access_matrix(int nb_iterators, int nb_dims)
        : matrix(nb_dims)
    {
        for (int i = 0; i < nb_dims; ++i)
            matrix[i] = std::vector<int>(nb_iterators + 1, 0);
    }
};

}

#endif