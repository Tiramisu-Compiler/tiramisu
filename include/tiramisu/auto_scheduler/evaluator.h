#ifndef _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_
#define _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_

#include <torch/script.h>
#include "auto_scheduler.h"
#include "utils.h"

#define MAX_NB_ITERATORS 4
#define MAX_NB_ACCESSES 17

namespace tiramisu::auto_scheduler
{

/**
  * An abstract class that represents an evaluation function.
  * Derive this class and implement the method "evaluate" to
  * create new evaluation functions.
  */
class evaluator
{
private:
    
protected:
    
public:
    virtual ~evaluator() {}
    
    /**
      * Takes as input a computation graph and returns
      * its evaluation.
      */
    virtual float evaluate(syntax_tree const& ast) =0;
};

}

#endif
