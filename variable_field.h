#ifndef VF_
#define VF_

#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <variant>
#include "lattice.h"
#include "rate_calculator.h"

class VariableField {
  public:
    VariableField();
    VariableField(RateCalculator&, Lattice&, double, double, double, double, double);
    ~VariableField();
    void set_amplitude(double);
    void set_frequency(double);
    void update_ens_(RateCalculator&, double);
    void update_ens_(RateCalculator&, double, double);
    void update_ens(RateCalculator&, Lattice&, double);
    void update_rate_parameter(RateCalculator&, Lattice& lat, double);

    double lower;
    double delay;
    double amplitude;
    double frequency;
};
#endif