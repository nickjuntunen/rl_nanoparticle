#ifndef UTILITIES_
#define UTILITIES_

#include "lattice.h"
#include "rate_calculator.h"
#include "monte_carlo.h"

class Lattice;
class Utilities {
  public:
    Utilities();
    ~Utilities();

    void save_lattice_traj_xyz(Lattice&, const char* filename, const int n_side, const int height);
    void final_print(Lattice&, RateCalculator&, MC&);
};

#endif