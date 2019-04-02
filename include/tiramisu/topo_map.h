#ifndef _H_TIRAMISU_TOPOLOGY_
#define _H_TIRAMISU_TOPOLOGY_

#include <string>
#include <iostream>
#include <vector>

namespace tiramisu 
{

  typedef std::pair<uint64_t, uint64_t> proc_pair;

  struct Proc {
    bool defined = false;
    std::vector<proc_pair> phys_id_range; // pair._0 = start range, pair._1 = end range (inclusive)
    Proc();
    Proc(std::string name, std::vector<std::pair<uint64_t, uint64_t>> phys_id_range);
    std::string to_string();
  };

  struct Socket { // or call it NUMA node? Are they the same?
    bool defined = false;
    std::string name;
    int phys_id;
    std::vector<Proc> procs;
    Socket();
    Socket(std::string name, int phys_id, std::vector<Proc> procs);
    std::string to_string(bool with_socket_id = false);
  };

  struct Node {
    bool defined = false;
    std::string phys_name;
    std::vector<Socket> sockets;
    Node();
    Node(std::string phys_name, std::vector<Socket> sockets);
    bool contains_socket(Socket &s);
  };

  struct Rank {
    int linear_rank; // everything boils down to a linear rank
    std::vector<int> rank; // since could have different ranks in different communicators, these are basically mapping dimensions
    Node node;
    Socket socket;
    Proc proc;
    //explicit Rank(int rank);
    Rank(int rank, Node node);
    Rank(int rank, Node node, Socket socket);
    Rank(int rank, Node node, Socket socket, Proc proc);
    Rank(int linear_rank, std::vector<int> rank, Node node);
    Rank(int linear_rank, std::vector<int> rank, Node node, Socket socket);
    Rank(int linear_rank, std::vector<int> rank, Node node, Socket socket, Proc proc);
  };


  // TODO move implementations to cpp file
  struct GridRank {
    std::vector<int> rank; // since could have different ranks in different communicators, these are basically mapping dimensions
    Node node;
    Socket socket;
    Proc proc;
  GridRank(std::vector<int> rank, Node node) : rank(rank), node(node) { }
  GridRank(std::vector<int> rank, Node node, Socket socket) : rank(rank), node(node), socket(socket) { }
  GridRank(std::vector<int> rank, Node node, Socket socket, Proc proc) : rank(rank), node(node), socket(socket), proc(proc) { }
  };


  struct Topo {
    bool defined = false;
    std::vector<Rank> ranks;
    Topo() { }
    Topo(std::vector<Rank> ranks) : defined(true), ranks(ranks) { } 
  };

  struct GridTopo : public Topo {
    std::vector<int> dim_lengths;
    int coord_to_rank(GridRank gr) {
      int rank = 0;
      for (int outer_dim = 0; outer_dim < gr.rank.size(); outer_dim++) {
	int inner_rank = gr.rank[outer_dim];
	for (int inner_dim = 0; inner_dim < gr.rank.size() - outer_dim - 1; inner_dim++) {
	  inner_rank *= dim_lengths[inner_dim];
	}
	rank += inner_rank;
      }
      std::cerr << "Coords were ";
      for (auto coord : gr.rank) {
	std::cerr << coord << " ";
      }
      std::cerr << std::endl << " got linear idx " << rank << std::endl;
      return rank;
    }

    GridTopo();
    
  GridTopo(std::vector<int> dim_lengths, std::vector<GridRank> grid_ranks) : dim_lengths(dim_lengths) { 
      defined = true;
      // convert each GridRank to a Rank
      for (auto gr : grid_ranks) {
	auto r = Rank(coord_to_rank(gr), gr.rank, gr.node, gr.socket, gr.proc);
	this->ranks.push_back(r);
      }
    } 
  };

  // Defines a collection of topographies that will be used
  struct MultiTopo {
    bool defined = false;
    std::vector<Topo *> topos;
    MultiTopo();
    MultiTopo(std::vector<Topo *> topos);
    void print_mapping();
    void generate_run_script(std::string fn);
  };

}

#endif
