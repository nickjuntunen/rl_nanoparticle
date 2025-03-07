#ifndef SIMULATION_
#define SIMULATION_

#include "lattice.h"
#include "rate_calculator.h"
#include "monte_carlo.h"
#include "variable_field.h"
#include "utilities.h"


class Simulation {
  private:
    std::mt19937 rng;
    Lattice lat;
    RateCalculator rc;
    MC mc;
    // VariableField vf;
    Utilities utils;
    int area;
    int steps;
    int site_idx;
    int n_side;
    int height;
  public:
    ~Simulation();
    Simulation(
      const int,
      double,
      double,
      double,
      double,
      double,
      double,
      double,
      double,
      double,
      double,
      double,
      double,
      const int,
      const int
    );
    void step(
      int
    );
    void reset();
    std::vector<int> get_state();
    std::vector<int> get_box();
    void print_state();
    void take_action(
      const std::vector<double>&
    );
    void take_action(
      const double&,
      bool
    );
    void save_traj(
      std::string
    );
    double time;
    int seed_value;
    int& num_np;
    double& temperature;
    std::vector<double>& ens_grid;
};
#endif
