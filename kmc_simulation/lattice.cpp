#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <set>
#include <stdexcept>
#include <optional>
#include <algorithm>
#include "lattice.h"
#include "site_info.h"


Lattice::Lattice() {}


Lattice::Lattice(const int height, const int n_side, const int np_lat_sides):
  // constructor
  n_side(n_side),
  height(height),
  volume(n_side * n_side * height),
  np_lat_sides(np_lat_sides),
  num_np(0),
  n_fluid(n_side * n_side * (height - 1)),
  orientations(std::vector<double>{0.0, M_PI/32, M_PI/16, 3*M_PI/32, M_PI/8, 5*M_PI/32, 3*M_PI/16, 7*M_PI/32, M_PI/4}),
  nl(std::vector<std::vector<int>>(volume, std::vector<int>(np_lat_sides+2, -1))),
  np_list(std::unordered_map<int, double>()),
  sim_box(volume, FLUID),
  rng(rd()),
  uni_rng(0.0, 1.0)
{
  for (int i = 0; i < n_side*n_side; i++) {
    // initialize the lattice with surface (0) and fluid (1) sites
    sim_box[i] = SURFACE;
  }
  sim_box[volume] = FLUID;
  get_neighboring_indices();
}


Lattice::~Lattice()
  // destructor
{}


int Lattice::pos2idx(int x, int y, int z) {
  /* Get a nanoparticle index based on its position
  */
  return x + y * n_side + z * n_side * n_side;
}


void Lattice::get_neighboring_indices() {
  /* Get the indices for each site neighbor
  */
  int idx, left, right, front, back, top, bottom;
  // std::vector<std::vector<int>> neighbors(n_side + n_side*n_side + n_side*n_side*height, std::vector<int>(np_lat_sides+2, -1));
  // (bottom, left, front, back, right, top)
  // (0, 1, 2, 3, 4, 5)

  for (int i=0; i<n_side; i++) {
    for (int j=0; j<n_side; j++) {
      for (int k=0; k<height; k++) {
        idx = pos2idx(i, j, k);
        left = pos2idx((i-1+n_side) % n_side, j, k);
        right = pos2idx((i+1) % n_side, j, k);
        front = pos2idx(i, (j-1+n_side) % n_side, k);
        back = pos2idx(i, (j+1) % n_side, k);

        if (k == 0) bottom = SURFACE;
        else bottom = idx - n_side*n_side;
        if (k == height-1) top = volume;
        else top = idx + n_side*n_side;

        nl[idx] = {bottom, left, front, back, right, top};
      }
    }
  }
}


int Lattice::select_interface_site() {
  /* Select a random fluid/solid interface site
  */
  while (true) {
    int i = uni_rng(rng) * n_side;
    int j = uni_rng(rng) * n_side;
    int id = int(uni_rng(rng) > 0.5) + 1; // 1 for fluid, 2 for nanoparticle, never choose surface site
    for (int k=1; k<height; k++) {
      if (sim_box[pos2idx(i, j, k)] == id) {
        return pos2idx(i, j, k);
      }
    }
  }
}


SiteInfo Lattice::get_site_info(const int idx) {
  /* Get the site information for a given index
  */
  // std::cout << "idx: " << idx << std::endl;
  if (idx < 0 || idx >= volume) {
    throw std::invalid_argument("Index out of bounds.");
  }
  int central_type = sim_box[idx];
  int bottom_neighbor = sim_box[nl[idx][0]];
  int top_neighbor = sim_box[nl[idx][5]];
  int lateral_f_count = 0;
  int lateral_n_count = 0;
  for (int i=1; i<=np_lat_sides; i++) {
    if (sim_box[nl[idx][i]] == FLUID) {
      lateral_f_count++;
    } else if (sim_box[nl[idx][i]] == NANOPARTICLE) {
      lateral_n_count++;
    }
  }
  return {central_type, bottom_neighbor, lateral_f_count, lateral_n_count, top_neighbor};
}


void Lattice::update_lattice_with_move(const std::pair<int, int> move) {
  /* Update the lattice with the selected move
  Map:
    0: fluid->np with orientation 0
    1: np->fluid
    2: diffuse left
    3: diffuse front
    4: diffuse back
    5: diffuse right
    6: rotate 1
    7: rotate 2
    8: rotate 3
    9: rotate 4
    10: rotate 5
    11: rotate 6
    12: rotate 7
    13: rotate 8
  */
  int lattice_idx = move.first;
  int move_type = move.second;
  double old_angle;


  switch (move_type) {

    case 0:
      sim_box[lattice_idx] = NANOPARTICLE;
      np_list[lattice_idx] = orientations[move_type];
      num_np++;
      n_fluid--;
      break;

    case 1:
      sim_box[lattice_idx] = FLUID;
      np_list.erase(lattice_idx);
      num_np--;
      n_fluid++;
      break;

    case 2: case 3: case 4: case 5:
      sim_box[nl[lattice_idx][move_type - 5]] = NANOPARTICLE;
      np_list[nl[lattice_idx][move_type - 5]] = np_list[lattice_idx];
      sim_box[lattice_idx] = FLUID;
      np_list.erase(lattice_idx);
      break;

    case 6: case 7: case 8: case 9: case 10: case 11: case 12: case 13:
      old_angle = np_list[lattice_idx];
      auto it = std::find(orientations.begin(), orientations.end(), old_angle);
      int r = std::distance(orientations.begin(), it);
      int new_idx = fmod(move_type + r, orientations.size());
      np_list[lattice_idx] = orientations[new_idx];
      break;
  }

  if (num_np+n_fluid != volume-n_side*n_side) {
    std::cout << "Number of nanoparticles and fluid sites does not add up to the total number of sites." << std::endl;
    exit(0);
  }
}


void Lattice::explore_cluster(int idx, std::vector<bool>& visited, Cluster& cluster) {
  if (visited[idx] || get_site_info(idx).central_type!=NANOPARTICLE) return;
  visited[idx] = true;
  cluster.size++;
  cluster.sites.push_back(idx);
  for (int n_idx : nl[idx]) {
    if (n_idx < volume && n_idx >=0) explore_cluster(n_idx, visited, cluster);
  }
}


std::vector<Cluster> Lattice::find_clusters() {
  std::vector<Cluster> clusters;
  std::vector<bool> visited(volume, false);
  for (int idx=0; idx<volume; idx++) {
    if (!visited[idx] && get_site_info(idx).central_type==NANOPARTICLE) {
      Cluster cluster = {0, {}};
      explore_cluster(idx, visited, cluster);
      clusters.push_back(cluster);
    }
  }
  return clusters;
}


std::unordered_map<int, int> Lattice::get_cluster_size_distribution() {
  std::unordered_map<int, int> cluster_size_distribution;
  for (const Cluster& cluster : find_clusters()) {
    cluster_size_distribution[cluster.size]++;
  }
  return cluster_size_distribution;
}


void Lattice::reset() {
  /* Reset the lattice to its initialized state
  */
  for (int i = 0; i < n_side*n_side; i++) {
    sim_box[i] = SURFACE;
  }
  for (int i = n_side*n_side; i < volume; i++) {
    sim_box[i] = FLUID;
  }
  num_np = 0;
  n_fluid = n_side * n_side * (height - 1);
  np_list.clear();
}