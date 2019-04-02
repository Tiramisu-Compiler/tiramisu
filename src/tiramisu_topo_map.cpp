#include <tiramisu/topo_map.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

namespace tiramisu {
  
  Proc::Proc() { }

  Proc::Proc(std::string name, std::vector<proc_pair> phys_id_range) : defined(true), phys_id_range(phys_id_range) { }

  std::string Proc::to_string() {
    std::stringstream ss;
    int i = 0;
    for (std::pair<uint64_t, uint64_t> p : phys_id_range) {
      if (i != 0) {
	ss << ",";
      }
      if (p.first == p.second) {
	ss << p.first;
      } else {
	ss << p.first << "-" << p.second;
      }
      i++;
    }
    return ss.str();
  }
  
  Socket::Socket() : phys_id(-1) { }
  
  Socket::Socket(std::string name, int phys_id, std::vector<Proc> procs) : defined(true), name(name), phys_id(phys_id), procs(procs) { }

  std::string Socket::to_string(bool with_socket_id) {
    std::stringstream ss;
    if (with_socket_id) {
      ss << phys_id << ":";
    }
    int i = 0;
    for (auto p : procs) {
      if (i != 0) {
	ss << ",";
      }
      ss << p.to_string();
      i++;
    }
    return ss.str();
  }

  Node::Node() { }

  Node::Node(std::string phys_name, std::vector<Socket> sockets) : defined(true), phys_name(phys_name), sockets(sockets) { }

  bool Node::contains_socket(Socket &s) {
    for (auto socket : sockets) {
      if (socket.phys_id == s.phys_id) {
	return true;
      }
    }
    return false;
  }

  //  Rank::Rank(int rank) : rank(rank) { }

  Rank::Rank(int rank, Node node) : linear_rank(rank), node(node) { 
    this->rank.push_back(rank);
  }

  Rank::Rank(int rank, Node node, Socket socket) : linear_rank(rank), node(node), socket(socket) { 
    this->rank.push_back(rank);
  }
  
  Rank::Rank(int rank, Node node, Socket socket, Proc proc) : 
    linear_rank(rank), node(node), socket(socket), proc(proc) { 
    this->rank.push_back(rank);
  }

  Rank::Rank(int linear_rank, std::vector<int> rank, Node node) : linear_rank(linear_rank), rank(rank), node(node) { } 

  Rank::Rank(int linear_rank, std::vector<int> rank, Node node, Socket socket) :
    linear_rank(linear_rank), rank(rank), node(node), socket(socket) { } 

  Rank::Rank(int linear_rank, std::vector<int> rank, Node node, Socket socket, Proc proc) : 
    linear_rank(linear_rank), rank(rank), node(node), socket(socket), proc(proc) { }

  TopoMap::TopoMap() { }
  
  TopoMap::TopoMap(std::vector<Rank> ranks) : defined(true), ranks(ranks) { }

  void TopoMap::print_mapping() {

  }

  void TopoMap::generate_run_script(std::string fn) {
    // currently ignoring Socket id in rank file because they aren't supported in a physical rank file 
    std::ofstream rank_file;
    rank_file.open(fn + ".rank_file.txt");
    for (auto rank : ranks) {
      rank_file << "rank " << rank.linear_rank << "=" << rank.node.phys_name << " slot=";
      // get the proc ids
      if (rank.socket.defined) {
	// use just the CPUs defined in this socket
	// first, make sure this socket is actually part of this node though
	assert(rank.node.contains_socket(rank.socket));
	std::string procs = rank.socket.to_string(false);
	rank_file << procs << std::endl;
      } else {
	// use all procs in the node's sockets
	int i = 0;
	for (auto socket : rank.node.sockets) {
	  if (i != 0) {
	    rank_file << ",";
	  }
	  rank_file << socket.to_string(false);	  
	  i++;
	}
	rank_file << std::endl;
      }      
    }
    rank_file.close();
    
    std::ofstream script;
    script.open(fn + ".bash");
    script << "#!/bin/bash" << std::endl;
    
    std::string cmd = MPI_RUN_CMD" --nooversubscribe --mca btl_openib_allow_ib 1 --mca rmaps_rank_file_physical 1 --rankfile " + fn + ".rank_file.txt -np " + std::to_string(ranks.size()) + " $1";
    
    script << "echo " << cmd << std::endl;
    script << cmd << std::endl;    
    script.close();
  }

}
