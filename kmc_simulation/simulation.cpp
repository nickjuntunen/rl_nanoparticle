#include "simulation.h"


Simulation::Simulation(
  const int seed,
  double enn,
  double ens,
  double enf,
  double esf,
  double eff,
  double muf,
  double mun,
  double fug,
  double temp_set,
  double prefactor_diff,
  double prefactor_fluc,
  double prefactor_rot,
  const int height,
  const int n_side
) : 
  seed_value(seed),
  rng(seed),
  lat(height, n_side, 4),
  rc(lat, temp_set, prefactor_diff, prefactor_fluc, prefactor_rot, 
              enn, enf, esf, eff, muf, mun, fug),
  // vf(rc, lat, ens, amplitude, frequency, lower, delay),
  mc(),
  utils(),
  area(n_side * n_side),
  steps(0),
  site_idx(0),
  n_side(n_side),
  height(height),
  // time_dependent_rates(time_dependent_rates),
  ens_grid(rc.ens),
  num_np(lat.num_np),
  temperature(rc.temperature)
{
  // Move the initialization code here
  take_action(ens, false);
  rc.total_energy = rc.calculate_total_lattice_energy(lat);
  fprintf(stdout, "Total energy (intial): %f\n", rc.total_energy);
  fprintf(stdout, "System initialized\n");
}

Simulation::~Simulation() {
  // Move the cleanup code here
}


void Simulation::step(int num_steps) {
  std::pair<int, int> move;
  for (int i = 0; i < num_steps; i++) {
    beginning:
    site_idx = lat.select_interface_site();
    if (area > site_idx || site_idx > lat.volume) continue;
    
    mc.get_possible_moves(lat, rc, site_idx);
    // if (time_dependent_rates) {
    //   move = mc.sample_move(lat, rc, vf, site_idx);
    // } else {
    move = mc.sample_move(lat, rc, site_idx);
    // }

    if (move.first == -1 && move.second == 0) goto beginning;
    if (move.first == -1 && move.second == -1) {
      fprintf(stdout, "No valid moves found- end simulation.\n");
      break;
    }

    lat.update_lattice_with_move(move);

    if (steps % 10000 == 0) {
      fprintf(stdout, "Step: %d; Time: %f; Number of nanoparticles: %d\n",
          steps, mc.time, lat.num_np);
    }
    time = mc.time;
    steps++;
  }
}


void Simulation::take_action(const std::vector<double>& ens_update) {
  for (int i = 0; i < ens_update.size(); i++) {
    ens_grid[i] = ens_update[i];
  }
}


void Simulation::take_action(const double& update_value, bool update_temp) {
  if (update_temp) {
    temperature = update_value;
  } else {
    for (int i = 0; i < ens_grid.size(); i++) {
      ens_grid[i] = update_value;
    }
  }
}


void Simulation::reset() {
  rng.seed(seed_value);
  lat.reset();
  mc.reset();
  // vf.update();
  rc.reset(lat);
  steps = 0;
  site_idx = 0;
  area = n_side * n_side;
  time = 0.0;
  fprintf(stdout, "System reset\n");
}


std::vector<int> Simulation::get_state() {
  std::unordered_map<int, int> cluster_sizes = lat.get_cluster_size_distribution();
  int total_np = 0;
  std::vector<int> state(lat.n_side*lat.n_side/2);
  for (const auto& cluster : cluster_sizes) {
    state[cluster.first] = cluster.second;
    total_np += cluster.first * cluster.second;
  }
  return state;
}


std::vector<int> Simulation::get_box() {
  return lat.sim_box;
}


void Simulation::print_state() {
  std::unordered_map<int, int> cluster_sizes = lat.get_cluster_size_distribution();
  for (const auto& cluster : cluster_sizes) {
    fprintf(stdout, "Cluster size: %d; Number of clusters: %d\n", cluster.first, cluster.second);
  }
  fprintf(stdout, "Total number of nanoparticles: %d\n", lat.num_np);
  fprintf(stdout, "Total energy: %f\n", rc.total_energy);
  fprintf(stdout, "Total time: %f\n", mc.time);
}


void Simulation::save_traj(std::string filename) {
  utils.save_lattice_traj_xyz(lat, filename.c_str(), n_side, height);
}