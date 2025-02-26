#ifndef SM_
#define SM_

#include <string>
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include "lattice.h"
#include "site_info.h"

class RateCalculator {
  public:
    RateCalculator();
    RateCalculator(Lattice&, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double, const double);
    ~RateCalculator();
    double calculate_fluc_rate(Lattice&, const double, bool);
    double calculate_diff_rate(const double);
    double calculate_rot_rate(const double);
    // double calculate_energy(const std::string&);
    double calculate_total_lattice_energy(Lattice&);
    double get_alignment_factor(Lattice& lat, double, int);
    std::vector<std::variant<double,int>> get_rates(Lattice&, int, int);
    void reset(Lattice&);

    double temperature;
    double total_energy;
    std::vector<double> ens;
    double enn;

  private:
    const double enf;
    const double esf;
    const double eff;
    const double mun;
    const double muf;
    const double fug;
    const double prefactor_fluc;
    const double prefactor_diff;
    const double prefactor_rot;
};
#endif