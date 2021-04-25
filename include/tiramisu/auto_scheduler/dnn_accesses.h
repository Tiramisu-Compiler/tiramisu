#ifndef _TIRAMISU_AUTO_SCHEDULER_DNN_ACCESSES_
#define _TIRAMISU_AUTO_SCHEDULER_DNN_ACCESSES_

#include <tiramisu/core.h>
#include <tiramisu/expr.h>

#include "optimization_info.h"

namespace tiramisu::auto_scheduler
{

/**
 * Contains information about an iterator.
 * Just a convenient class to simplify working with the ML model.
 */
class dnn_iterator
{
public:
    std::string name;
    int low_bound;
    int up_bound;
    
    dnn_iterator(std::string const& name, int low_bound, int up_bound)
        : name(name), low_bound(low_bound), up_bound(up_bound) {}
        
    /**
     * Return a list of dnn_iterators from the iterators of the given computation.
     */
    static std::vector<dnn_iterator> get_iterators_from_computation(tiramisu::computation const& comp);
};

/**
 * Contains the access matrix for a given access.
 */
class dnn_access_matrix
{
protected:
    /**
     * A recursive subroutine used by the constructor :
     * dnn_access_matrix(int nb_iterators, tiramisu::expr const& e, tiramisu::computation *comp);
     */
    void fill_matrix_row(int i, tiramisu::expr const& e, bool minus = false);

public:
    int nb_iterators;
    int nb_dims;
    std::vector<std::vector<int>> matrix;
    
    /**
     * The buffer that this matrix accesses.
     */
    std::string buffer_name;
    int buffer_id;
    
    /**
     * The computation from which the access has been extracted.
     */
    tiramisu::computation *comp;
    
    /**
     * Create an empty access matrix (filled with zeros),
     * with the given number of iterators and the given number of dimensions.
     */
    dnn_access_matrix(int nb_iterators, int nb_dims);
    
    /**
     * Create an access matrix for the access represented by the given expression.
     * "comp" is the computation containing the expression "e".
     */
    dnn_access_matrix(int nb_iterators, tiramisu::expr const& e, tiramisu::computation *comp);

    /**
     * Copy constructor
    */
    //dnn_access_matrix(dnn_access_matrix const& reference);
    /**
     * Prints Matrix for debug in the format:
     * line1,line2,line3
    */
    void print_access_matrix() const;

    /**
     * transforms the matrix by skewing
    */
    void transforme_matrix_by_skewing(int first_node_depth,int alpha,int beta,int gamma,int sigma);

};

/**
 * Contains the list of access matrices for a given computation.
 */
class dnn_accesses
{
public:
    /**
     * The computation from which the accesses have been retrieved.
     */
    tiramisu::computation* comp;
    int nb_iterators;
    
    /**
     * A list of matrices, such as each matrix represents an access of "comp".
     */
    std::vector<dnn_access_matrix> accesses_list;
        
    /**
     * Create the list of accesses of the given computation.
     */
    dnn_accesses(tiramisu::computation *comp, int nb_iterators, tiramisu::function *fct);

    /**
     * copy constructor
    */
    //dnn_accesses(dnn_accesses const& reference);
        
    /**
     * Recursively retrieve accesses from expression "e".
     */
    void create_accesses(tiramisu::expr const& e);

    void print_all_access() const;

    void modify_accesses_by_skewing(int first_node_depth,int alpha,int beta,int gamma,int sigma);
};

}

#endif
