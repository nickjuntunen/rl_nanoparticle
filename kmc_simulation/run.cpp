#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <array>
#include "lattice.h"
#include "utilities.h"
#include "monte_carlo.h"
#include "rate_calculator.h"
#include "variable_field.h"

int main(int argc, char **argv) {
  if (argc != 8) {
    printf("Usage: ./run <seed> <num_steps> <n_side> <enn> <ens> <lower> <time_delay>\n");
    exit(-1);
  }

  const int seed = atoi(argv[1]);
  const int num_steps = atoi(argv[2]);
  const int n_side = atoi(argv[3]);
  double enn = atof(argv[4]);
  double ens = atof(argv[5]);
  const double lower = atof(argv[6]);
  const double delay = atof(argv[7]);
  double enf = 1.0;
  double esf = 1.0;
  double eff = 1.0;
  double muf = 1.0;
  double mun = 5.0;
  double fug = 0.0; // check if this is np->fluid or fluid->np if it's not set to 0
  double temperature = 0.5;
  const double prefactor_diff = 1.0;
  const double prefactor_fluc = 1.0;
  const double prefactor_rot = 10.0;
  const int height = 10;
  const double amplitude = 0.5;
  const double frequency = 0.5;

  // set seed
  std::mt19937 rng(seed);

  // initialize classes
  Lattice lat(height, n_side, 4);
  RateCalculator rc(lat, temperature, prefactor_diff, prefactor_fluc, prefactor_rot, enn, enf, esf, eff, muf, mun, fug);
  MC mc;
  VariableField vf(rc, lat, ens, amplitude, frequency, lower, delay);
  Utilities utils;
  fprintf(stdout, "Total energy (intial): %f\n", rc.total_energy);

  // initialize variables
  std::pair<int, int> move;
  int area = n_side * n_side;
  int step = 0;
  int site_idx;
  bool time_dependent = true; // time-dependent or -independent rates
  std::string filename = "delay" + std::to_string(delay) + ".xyz";

  // carry out simulation
  fprintf(stdout, "\nStarting simulation...\n\n");
  while (step < num_steps && lat.num_np < lat.volume-n_side*n_side) {
    // return here if move is rejected
    beginning:

    // select site to update
    site_idx = lat.select_interface_site();
    if (area > site_idx || site_idx > lat.volume) continue;

    // get possible moves and sample move
    mc.get_possible_moves(lat, rc, site_idx);
    if (time_dependent) { move = mc.sample_move(lat, rc, vf, site_idx); }
    else { move = mc.sample_move(lat, rc, site_idx); }

    // check if move is valid
    if (move.first == -1 && move.second == 0) { goto beginning;} // move rejected (rate importance sampling)
    if (move.first == -1 && move.second == -1) { fprintf(stdout, "No valid moves found- end simulation.\n"); break; }

    // update lattice
    lat.update_lattice_with_move(move);

    // logging
    if (step % 100 == 0) {
      fprintf(stdout, "Step: %d; Time: %f; Number of nanoparticles: %d\n", step, mc.time, lat.num_np);
      utils.save_lattice_traj_xyz(lat, filename.c_str(), n_side, height);
    }
    step++;
  }
  // simulation complete

  utils.final_print(lat, rc, mc);
  return 0;
}