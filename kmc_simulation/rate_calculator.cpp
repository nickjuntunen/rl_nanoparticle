#include <string>
#include <cmath>
#include <variant>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "rate_calculator.h"
#include "lattice.h"
#include "site_info.h"


RateCalculator::RateCalculator():
  temperature(0.0),
  prefactor_diff(0.0),
  prefactor_fluc(0.0),
  prefactor_rot(0.0),
  enn(0.0),
  ens(std::vector<double>(0)),
  enf(0.0),
  esf(0.0),
  eff(0.0),
  muf(0.0),
  mun(0.0),
  fug(0.0),
  total_energy(0.0)
{
  // constructor
}


RateCalculator::RateCalculator(Lattice& lat, double temperature, double prefactor_diff, double prefactor_fluc, double prefactor_rot, double enn, double enf, double esf, double eff, double muf, double mun, double fug):
  temperature(temperature),
  prefactor_diff(prefactor_diff),
  prefactor_fluc(prefactor_fluc),
  prefactor_rot(prefactor_rot),
  enn(enn),
  ens(std::vector<double>(lat.n_side*lat.n_side, 0.0)),
  enf(enf),
  esf(esf),
  eff(eff),
  muf(muf),
  mun(mun),
  fug(fug),
  total_energy(0.0)
{
  // constructor
}


RateCalculator::~RateCalculator() {
  // destructor
}


double RateCalculator::calculate_total_lattice_energy(Lattice& lat) {
  /* Calculate the total energy of the lattice
  */
  int idx_left, idx_below, idx_front;
  double alignment_factor;
  double energy = 0.0;
  for (int i = lat.n_side*lat.n_side; i < lat.volume; i++) {
    idx_left = lat.nl[i][1];
    idx_below = lat.nl[i][0];
    idx_front = lat.nl[i][2];
    if (lat.sim_box[i] == FLUID) {
      for (int j : {idx_left, idx_below, idx_front}) {
        if (lat.sim_box[j] == NANOPARTICLE) {
          energy -= enf;
        } else if (lat.sim_box[j] == FLUID) {
          energy -= eff;
        } else {
          energy -= esf;
        }
      }
    }
    if (lat.sim_box[i] == NANOPARTICLE) {
      for (int j : {idx_left, idx_below, idx_front}) {
        if (lat.sim_box[j] == NANOPARTICLE) {
          alignment_factor = get_alignment_factor(lat, lat.np_list[i], j);
          energy -= enn * alignment_factor;
        } else if (lat.sim_box[j] == FLUID) {
          energy -= enf;
        } else {
          energy -= ens[j];
        }
      }
    }
  }
  for (int i = (lat.volume - lat.n_side*lat.n_side); i < lat.volume; i++) {
    if (lat.sim_box[i] == NANOPARTICLE) {
      energy -= enf;
    } else {
      energy -= eff;
    }
  }
  energy += lat.num_np * mun;
  energy += lat.n_fluid * muf;
  return energy;
}


double RateCalculator::get_alignment_factor(Lattice& lat, double angle1, int idx2) {
  double angle2 = lat.np_list[idx2];
  double cos_arg = 2 * abs(angle1 - angle2);
  double alignment_factor = 2.0 * (cos(cos_arg) * cos(cos_arg)) - 1.0;
  return alignment_factor;
}


double RateCalculator::calculate_diff_rate(const double delta_E) {
  /* Calculate the rate of a transition from one site to another
    - use canonical ensemble for diffusion
  */
  if (delta_E < 0.0) return prefactor_diff;
  else return prefactor_diff * exp(-delta_E / temperature);
}


double RateCalculator::calculate_rot_rate(const double delta_E) {
  /* Calculate the rate of a rotation transition
    - use canonical ensemble for rotation
  */
  if (delta_E < 0.0) return prefactor_rot;
  else return prefactor_rot * exp(-delta_E / temperature);
}


double RateCalculator::calculate_fluc_rate(Lattice& lat, const double delta_E, bool np_to_fluid) {
  /* Calculate the rate of a transition from one site to another
    - use grand canonical ensemble for fluctuation
  */
  double criterion, exp_term;
  if (np_to_fluid) {
    exp_term = (-fug - delta_E) / temperature;
    criterion = lat.num_np / (1+lat.n_fluid) * exp(exp_term);
    if (criterion > 1.0) return prefactor_fluc;
    else return prefactor_fluc * criterion;
  } else {
    exp_term = (fug - delta_E) / temperature;
    criterion = lat.n_fluid / (1+lat.num_np) * exp(exp_term);
    if (criterion > 1.0) return prefactor_fluc;
    else return prefactor_fluc * criterion;
  }
}


std::vector<std::variant<double,int>> RateCalculator::get_rates(Lattice& lat, int idx, int move) {
  /* Given a move, get the rate and energy
  Move map:
    0: fluid->np
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

  int idx_left = lat.nl[idx][1];
  int idx_front = lat.nl[idx][2];
  int idx_back = lat.nl[idx][3];
  int idx_right = lat.nl[idx][4];
  int idx_bottom = lat.nl[idx][0];
  int id_bottom = lat.sim_box[idx_bottom];
  double delta_energy = 0.0;
  double alignment_factor, angle, old_angle, new_angle;
  double rate = 0.0;
  int next_idx_bottom, next_idx_left, next_idx_front, next_idx_back, next_idx_right, r;

  switch (move) {
    case 0:
    if (id_bottom == SURFACE) {
      delta_energy += esf;
      delta_energy -= ens[idx_bottom];
    } else {
      delta_energy += enf;
      delta_energy -= enn;
    }
    r = static_cast<int>(std::floor(lat.orientations.size() * random() / (RAND_MAX + 1.0)));
    angle = lat.orientations[r];
    for (int i : {idx_left, idx_front, idx_back, idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        delta_energy += enf;
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy -= alignment_factor * enn;
      } else {
        delta_energy += eff;
        delta_energy -= enf;
      }
    }
    delta_energy += eff;
    delta_energy -= enf;
    delta_energy -= muf;
    delta_energy += mun;
    rate = calculate_fluc_rate(lat, delta_energy, false);
    break;

    case 1:
    if (id_bottom == SURFACE) {
      delta_energy += ens[idx_bottom];
      delta_energy -= esf;
    } else {
      delta_energy += enn;
      delta_energy -= enf;
    }
    angle = lat.np_list[idx];
    for (int i : {idx_left, idx_front, idx_back, idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy += alignment_factor * enn;
        delta_energy -= enf;
      } else {
        delta_energy += enf;
        delta_energy -= eff;
      }
    }
    delta_energy += enf;
    delta_energy -= eff;
    delta_energy += muf;
    delta_energy -= mun;
    rate = calculate_fluc_rate(lat, delta_energy, true);
    break;

    case 2:
    next_idx_bottom = lat.nl[idx_left][0];
    next_idx_left = lat.nl[idx_left][1];
    next_idx_front = lat.nl[idx_left][2];
    next_idx_back = lat.nl[idx_left][3];
    if (id_bottom == SURFACE) {
      delta_energy += ens[idx_bottom];
      delta_energy -= ens[next_idx_bottom];
    }
    angle = lat.np_list[idx];
    for (int i : {idx_front, idx_back, idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy += alignment_factor * enn;
        delta_energy -= enf;
      } else {
        delta_energy += enf;
        delta_energy -= eff;
      }
    }
    for (int i : {next_idx_left, next_idx_front, next_idx_back}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        delta_energy += enf;
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy -= alignment_factor * enn;
      } else {
        delta_energy += eff;
        delta_energy -= enf;
      }
    }
    rate = calculate_diff_rate(delta_energy);
    break;

    case 3:
    next_idx_bottom = lat.nl[idx_front][0];
    next_idx_left = lat.nl[idx_front][1];
    next_idx_front = lat.nl[idx_front][2];
    next_idx_right = lat.nl[idx_front][4];
    if (id_bottom == SURFACE) {
      delta_energy += ens[idx_bottom];
      delta_energy -= ens[next_idx_bottom];
    }
    angle = lat.np_list[idx];
    for (int i : {idx_left, idx_back, idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy += alignment_factor * enn;
        delta_energy -= enf;
      } else {
        delta_energy += enf;
        delta_energy -= eff;
      }
    }
    for (int i : {next_idx_left, next_idx_front, next_idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        delta_energy += enf;
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy -= alignment_factor * enn;
      } else {
        delta_energy += eff;
        delta_energy -= enf;
      }
    }
    rate = calculate_diff_rate(delta_energy);
    break;

    case 4:
    next_idx_bottom = lat.nl[idx_back][0];
    next_idx_left = lat.nl[idx_back][1];
    next_idx_back = lat.nl[idx_back][3];
    next_idx_right = lat.nl[idx_back][4];
    if (id_bottom == SURFACE) {
      delta_energy += ens[idx_bottom];
      delta_energy -= ens[next_idx_bottom];
    }
    angle = lat.np_list[idx];
    for (int i : {idx_left, idx_front, idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy += alignment_factor * enn;
        delta_energy -= enf;
      } else {
        delta_energy += enf;
        delta_energy -= eff;
      }
    }
    for (int i : {next_idx_left, next_idx_back, next_idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        delta_energy += enf;
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy -= alignment_factor * enn;
      } else {
        delta_energy += eff;
        delta_energy -= enf;
      }
    }
    rate = calculate_diff_rate(delta_energy);
    break;

    case 5:
    next_idx_bottom = lat.nl[idx_right][0];
    next_idx_front = lat.nl[idx_right][2];
    next_idx_back = lat.nl[idx_right][3];
    next_idx_right = lat.nl[idx_right][4];
    if (id_bottom == SURFACE) {
      delta_energy += ens[idx_bottom];
      delta_energy -= ens[next_idx_bottom];
    }
    angle = lat.np_list[idx];
    for (int i : {idx_left, idx_front, idx_back}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy += alignment_factor * enn;
        delta_energy -= enf;
      } else {
        delta_energy += enf;
        delta_energy -= eff;
      }
    }
    for (int i : {next_idx_front, next_idx_back, next_idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        delta_energy += enf;
        alignment_factor = get_alignment_factor(lat, angle, i);
        delta_energy -= alignment_factor * enn;
      } else {
        delta_energy += eff;
        delta_energy -= enf;
      }
    }
    rate = calculate_diff_rate(delta_energy);
    break;

    case 6: case 7: case 8: case 9: case 10: case 11: case 12: case 13:
    old_angle = lat.np_list[idx];
    auto it = std::find(lat.orientations.begin(), lat.orientations.end(), old_angle);
    int old_idx = std::distance(lat.orientations.begin(), it);
    int new_idx = fmod(move + old_idx, lat.orientations.size());
    new_angle = lat.orientations[new_idx];
    for (int i : {idx_left, idx_front, idx_back, idx_right}) {
      if (lat.sim_box[i] == NANOPARTICLE) {
        alignment_factor = get_alignment_factor(lat, old_angle, i);
        delta_energy += alignment_factor * enn;
        alignment_factor = get_alignment_factor(lat, new_angle, i);
        delta_energy -= alignment_factor * enn;
      }
    }
    rate = calculate_rot_rate(delta_energy);
    break;
  }
  std::vector<std::variant<double,int>> move_tuple = {rate, delta_energy, move};
  return move_tuple;
}


void RateCalculator::reset(Lattice& lat) {
  total_energy = calculate_total_lattice_energy(lat);
  return;
}