#ifndef _H_TIRAMISU_TOPOLOGY_
#define _H_TIRAMISU_TOPOLOGY_

#include <string>
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

  struct TopoMap {
    bool defined = false;
    std::vector<Rank> ranks;
    TopoMap();
    TopoMap(std::vector<Rank> ranks);
    void print_mapping();
    void generate_run_script(std::string fn);
  };

}

#endif
