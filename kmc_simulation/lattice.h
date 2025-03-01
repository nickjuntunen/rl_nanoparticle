#ifndef LAT_
#define LAT_

#include <unordered_map>
#include <iostream>
#include <variant>
#include <stdio.h>
#include <vector>
#include <random>
#include <cmath>
#include <array>
#include "site_info.h"


struct Cluster {
  int size;
  std::vector<int> sites;
};


class Lattice {
  public:
    Lattice();
    Lattice(const int, const int, const int);
    ~Lattice();
    int pos2idx(int, int, int);
    int select_interface_site();
    void get_neighboring_indices();
    void update_lattice_with_move(const std::pair<int, int>);
    SiteInfo get_site_info(const int);
    std::vector<Cluster> find_clusters();
    std::unordered_map<int, int> get_cluster_size_distribution();
    void reset();

    int volume;
    int n_side;
    int height;
    int num_np;
    int n_fluid;
    int np_lat_sides;
    std::random_device rd;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uni_rng;
    std::vector<int> sim_box;
    std::vector<double> orientations;
    std::vector<std::vector<int>> nl;
    std::unordered_map<int, double> np_list;

  private:
    void explore_cluster(int, std::vector<bool>&, Cluster&);
};

#endif