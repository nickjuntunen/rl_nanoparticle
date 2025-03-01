#ifndef MC_
#define MC_

#include <unordered_map>
#include <stdio.h>
#include <vector>
#include <random>
#include <cmath>
#include <array>
#include <variant>
#include "lattice.h"
#include "rate_calculator.h"
#include "variable_field.h"

class MC {
  public:
    MC();
    ~MC();

    void get_possible_moves(Lattice&, RateCalculator&, const int);
    std::pair<int, int> sample_move(Lattice&, RateCalculator&, VariableField&, const int);
    std::pair<int, int> sample_move(Lattice&, RateCalculator&, const int);
    void reset();

    double time;
    double max_move_rate;
    std::mt19937 rng;
    std::random_device rd;
    std::vector<int> move_set;
    std::vector<std::vector<std::variant<double, int>>> move_rates;
    std::uniform_real_distribution<double> uni_rng;

  private:
    void sort_rates(std::vector<std::vector<std::variant<double,int>>>&);
};

#endif