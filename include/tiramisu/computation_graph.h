#ifndef _H_TIRAMISU_COMPUTATION_GRAPH_
#define _H_TIRAMISU_COMPUTATION_GRAPH_

namespace tiramisu
{

/**
  * Computation graph node.
  */
class cg_node
{
	computation *node;
	std::vector<cg_node *> children;
};

class computation_graph
{
	/**
	  * Computation graph root nodes.
	  */
	std::vector<cg_node *> roots;
};

}

#endif
