#include <cmath>
#include <variant>
#include "variable_field.h"
#include "rate_calculator.h"


VariableField::VariableField() {}


VariableField::VariableField(RateCalculator& rc, Lattice& lat, double starting_ens, double amplitude, double frequency, double lower, double delay):
  // cosine wave
  amplitude(amplitude),
  frequency(frequency),
  lower(lower),
  delay(delay)
{
  // constructor
  update_ens_(rc, 0.0, starting_ens);
  rc.total_energy = rc.calculate_total_lattice_energy(lat);
}


VariableField::~VariableField() {}


void VariableField::update_ens_(RateCalculator& rc, double t) {
  /* Update the field
  */
  double offset = 8.;
  double sharpness = 5.;
  double noise;
  double width = 0.2;
  for (int i = 0; i < rc.ens.size(); i++) {
    noise = ((double)rand() / RAND_MAX) * width + (1.0 - 0.5 * width);  // Random number between 0 and 0.1
    rc.ens[i] = offset / (1 + exp(-sharpness*(delay - t))) + amplitude * std::cos(2.0 * M_PI * frequency * t) + lower;
    rc.ens[i] *= noise;
  }
  return;
}


void VariableField::update_ens_(RateCalculator& rc, double t, double starting_ens) {
  /* Update the field at initialization
  */
  double offset = starting_ens - lower;
  double sharpness = 5.;
  double noise;
  double width = 0.2;
  for (int i = 0; i < rc.ens.size(); i++) {
    noise = ((double)rand() / RAND_MAX) * width + (1.0 - 0.5 * width);  // Random number between 0 and 0.1
    rc.ens[i] = offset / (1 + exp(-sharpness*(delay - t))) + amplitude * std::cos(2.0 * M_PI * frequency * t) + lower;
    rc.ens[i] *= noise;
  }
  return;
}


void VariableField::update_ens(RateCalculator& rc, Lattice& lat, double t) {
  /* Update the energy with the field
  */
  std::vector<double> ens_old = rc.ens;
  double area = lat.n_side * lat.n_side;
  update_ens_(rc, t);
  for (int i = 0; i < rc.ens.size(); i++) {
    if (lat.sim_box[i+area] == NANOPARTICLE) {
      rc.total_energy += ens_old[i] - rc.ens[i];
    }
  }
  return;
}
  


void VariableField::update_rate_parameter(RateCalculator& rc, Lattice& lat, double t) {
  /* Update the rate parameter
  */
  update_ens(rc, lat, t);
  return;
}


void VariableField::set_amplitude(double new_amplitude) { amplitude = new_amplitude; }


void VariableField::set_frequency(double new_frequency) { frequency = new_frequency; }
