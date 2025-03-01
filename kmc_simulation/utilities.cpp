#include "utilities.h"
#include "lattice.h"


Utilities::Utilities() {}
Utilities::~Utilities() {}


void Utilities::save_lattice_traj_xyz(Lattice& lat, const char* filename, const int n_side, const int height) {
  /* Write current state to .xyz file
  */
  double angle;
  FILE *f = fopen(filename, "a");
  if (lat.np_lat_sides != 4) throw std::invalid_argument("Only cubic lattices are supported for this function.");
  fprintf(f, "%d\n", n_side*n_side*height);
  fprintf(f, "Cubic Lattice Configuration\n");
  for (int i=0; i<lat.volume; i++) {
    int x = i % n_side;
    int y = (i / n_side) % n_side;
    int z = i / (n_side*n_side);
    if (lat.sim_box[i] == 0) {
      fprintf(f, "0 %d %d %d 0 0 0 0 1\n", x, y, z);
    } else if (lat.sim_box[i] == 1) {
      fprintf(f, "1 %d %d %d 0 0 0 0 1\n", x, y, z);
    } else if (lat.sim_box[i] == 2) {
      angle = lat.np_list[i];
      fprintf(f, "2 %d %d %d %f 0 0 %f %f\n", x, y, z, angle, sin(angle/2), cos(angle/2));
    }
  }
  fclose(f);
}


void Utilities::final_print(Lattice& lat, RateCalculator& rc, MC& mc) {
  /* Print final results of simulation
  */
  std::unordered_map<int, int> cluster_sizes = lat.get_cluster_size_distribution();
  save_lattice_traj_xyz(lat, "lattice_traj.xyz", lat.n_side, lat.height);
  fprintf(stdout, "\nSimulation complete\n\n");
  fprintf(stdout, "Total energy (final): %f\n", rc.total_energy);
  fprintf(stdout, "recomputed total energy: %f\n", rc.calculate_total_lattice_energy(lat));
  fprintf(stdout, "Final time: %f\n", mc.time);
  fprintf(stdout, "Number of nanoparticles: %d\n", lat.num_np);
  fprintf(stdout, "Cluster size distribution:\n");
  int total_np = 0;
  for (const auto& cluster : cluster_sizes) {
    fprintf(stdout, "\tCluster size: %d; Number of clusters: %d\n", cluster.first, cluster.second);
    total_np += cluster.first * cluster.second;
  }
  fprintf(stdout, "Total number of nanoparticles: %d\n\n", total_np);
}