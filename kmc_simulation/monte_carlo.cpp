#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <variant>
#include <vector>
#include <random>
#include <cmath>
#include <array>
#include "monte_carlo.h"
#include "rate_calculator.h"
#include "lattice.h"
#include "site_info.h"

MC::MC():
max_move_rate(0.0)
  // constructor
{
  rng = std::mt19937(rd());
  uni_rng = std::uniform_real_distribution<double>(0.0, 1.0);
  time = 0.0;
  move_set.reserve(13);
  for (int i = 0; i < 13; i++) {
    move_rates.push_back({0.0, 0.0, i});
  }
}


MC::~MC()
  // destructor
{}


void MC::get_possible_moves(Lattice& lat, RateCalculator& rc, const int idx) {
  /* Get the rates for all possible transitions given the current state and chosen site
  Map:
    ### CHANGE: SHOULD BE FLUID->NP
    0: fluid->np (random orientation)
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
  SiteInfo from_site, to_site;
  const int box_id = lat.sim_box[idx];
  const int idx_left = lat.nl[idx][1];
  const int idx_front = lat.nl[idx][2];
  const int idx_back = lat.nl[idx][3];
  const int idx_right = lat.nl[idx][4];

  // Reset move set (move_rates cleared in get_possible_moves)
  for (int i=0; i<move_rates.size(); i++) { move_rates[i] = {0.0, 0.0, i}; }
  for (int move : move_set) { move_set.pop_back(); }

  switch (box_id) {

    case FLUID:
      // fluctuate fluid to nanoparticle (of certain orientation)
      move_set.push_back(0);
      break;

    case NANOPARTICLE:
      // rotate nanoparticle
      for (int opt : {6, 7, 8, 9, 10, 11, 12, 13}) {
        move_set.push_back(opt);
      }

      // don't allow fluctuations or diffusions if stacked below
      if (from_site.top_neighbor != FLUID) break;

      // fluctuate nanoparticle -> fluid
      move_set.push_back(1);
      
      // lateral diffusion
      if (from_site.lateral_f_count == 0) break;

      // diffuse left
      if (lat.sim_box[idx_left] == FLUID) {
        to_site = lat.get_site_info(idx_left);
        if (to_site.bottom_neighbor != FLUID) {
          move_set.push_back(2);
        }
      }
      // diffuse front
      if (lat.sim_box[idx_front] == FLUID) {
        to_site = lat.get_site_info(idx_front);
        if (to_site.bottom_neighbor != FLUID) {
          move_set.push_back(3);
        }
      }
      // diffuse back
      if (lat.sim_box[idx_back] == FLUID) {
        to_site = lat.get_site_info(idx_back);
        if (to_site.bottom_neighbor != FLUID) {
          move_set.push_back(4);
        }
      }
      // diffuse right
      if (lat.sim_box[idx_right] == FLUID) {
        to_site = lat.get_site_info(idx_right);
        if (to_site.bottom_neighbor != FLUID) {
          move_set.push_back(5);
        }
      }
      break;
  }
  return;
}


void MC::get_new_max_move_rate(Lattice& lat, RateCalculator& rc, const int idx) {
  const int box_id = lat.sim_box[idx];
  const std::vector<int>& nl = lat.nl[idx];
  double max_rate = 0.0;
  for (int i = 0; i < 5; i++) {
    get_possible_moves(lat, rc, nl[i]);
    for (int move : move_set) {
      move_rates[move] = rc.get_rates(lat, nl[i], move);
    }
    double total_rates = 0.0;
    for (const auto& sub : move_rates) { total_rates += std::get<double>(sub[0]); }
    if (total_rates > max_rate) { max_rate = total_rates; }
  }
  if (max_rate > max_move_rate) { max_move_rate = max_rate; }
  return;
}


std::pair<int, int> MC::sample_move(Lattice& lat, RateCalculator& rc, VariableField& vf, const int idx) {
  /* Sample a move given all possible transitions
    Method for time-dependent rates
    Input: lattice, rate calculator, variable field
    Return: index of nanoparticle to move, move type
  */
  // get rates for all possible moves, total rate
  for (int move : move_set) {
    move_rates[move] = rc.get_rates(lat, idx, move); // returns rate [0] and energy [1]
  }
  double total_rates = 0.0;
  for (const auto& sub : move_rates) { total_rates += std::get<double>(sub[0]); }
  if (total_rates == 0.0) { return {-1, -1}; } // No valid moves found

  // cutoff step for site choice
  double r;
  if (total_rates > max_move_rate) {
    max_move_rate = total_rates;
  } else {
    r = uni_rng(rng);
    if (r > (total_rates / max_move_rate)) {
      return {-1, 0}; // reject move
    }
  }

  // time delay for next event
  r = -log(uni_rng(rng));
  double dt = 2*M_PI / (8 * vf.frequency);
  double integral = 0.0;
  while (integral < r) {
    integral += dt * total_rates;
    time += dt;
    vf.update_rate_parameter(rc, lat, time);
    total_rates = 0.0;
    for (int move : move_set) { move_rates[move] = rc.get_rates(lat, idx, move); }
    for (const auto& sub : move_rates) { total_rates += std::get<double>(sub[0]); }
  }  

  // sort rates, get cumulative sum
  std::vector<double> cumsum_rates;
  cumsum_rates.push_back(std::get<double>(move_rates[0][0]));
  for (int i = 1; i < move_rates.size(); i++) {
    cumsum_rates.push_back(cumsum_rates[i-1] + std::get<double>(move_rates[i][0]));
  }

  // move selection
  double r2 = uni_rng(rng) * total_rates;
  auto it = std::upper_bound(cumsum_rates.begin(), cumsum_rates.end(), r2);
  int choose_idx = std::distance(cumsum_rates.begin(), it);
  if (choose_idx < move_rates.size()) {
    rc.total_energy += std::get<double>(move_rates[choose_idx][1]);
  } else {
    return {-1, -1}; // no valid moves found
  }

  // Return idx (selected lattice site) and move type
  return {idx, std::get<int>(move_rates[choose_idx][2])};
}


std::pair<int, int> MC::sample_move(Lattice& lat, RateCalculator& rc, int idx) {
  /* Sample a move given all possible transitions
    Method for time-independent rates
    Input: lattice, rate calculator
    Return: index of nanoparticle to move, move type
  */
  // get rates for all possible moves, total_rate
  for (int move : move_set) {
    move_rates[move] = rc.get_rates(lat, idx, move); // returns rate [0] and energy [1]
  }
  sort_rates(move_rates);
  double total_rates = 0.0;
  for (const auto& sub : move_rates) { total_rates += std::get<double>(sub[0]); }
  if (total_rates == 0.0) { return {-1, -1}; } // No valid moves found

  // cutoff step for site choice
  double r;
  if (total_rates > max_move_rate) {
    max_move_rate = total_rates;
  } else {
    r = uni_rng(rng);
    if (r > (total_rates / max_move_rate)) {
      return {-1, 0}; // reject move
    }
  }

  // move selection
  double r1 = uni_rng(rng);
  double r2 = uni_rng(rng) * total_rates;
  double total_rate = 0.0;
  for (int i = 0; i < move_rates.size(); i++) {
    total_rate += std::get<double>(move_rates[i][0]);
    if (total_rate > r2) {
      time += 1.0 / total_rates * (-log(r1));
      rc.total_energy += std::get<double>(move_rates[i][1]);
      int move = std::get<int>(move_rates[i][2]);
      get_new_max_move_rate(lat, rc, idx);
      return {idx, move};
    }
  }
  return {-1, -1}; // no valid moves found
}


void MC::sort_rates(std::vector<std::vector<std::variant<double,int>>>& v) {
  /* Sort rates in possible_move_rates in ascending order
    Here to clean up sample_move code
  */
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
    return std::get<double>(a[0]) < std::get<double>(b[0]);
  });
}


void MC::reset() {
  /* reset the Monte Carlo object
  */
  time = 0.0;
  max_move_rate = 0.0;
  for (int i=0; i<move_rates.size(); i++) { move_rates[i] = {0.0, 0.0, i}; }
  for (int move : move_set) { move_set.pop_back(); }
}