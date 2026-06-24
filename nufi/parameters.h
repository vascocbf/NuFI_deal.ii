#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cmath>
#include <cstdlib>

namespace Parameters
{
  constexpr unsigned int DIMENSION = 1;

  constexpr double X_DOMAIN_LEFT  = 0.0;
  constexpr double X_DOMAIN_RIGHT = 4*M_PI;
  constexpr double LX = std::abs(X_DOMAIN_RIGHT- X_DOMAIN_LEFT);
  constexpr double LX_INV = 1/LX;

  constexpr size_t CALC_NX = 256;
  constexpr double CALC_DX = LX/CALC_NX;

  constexpr double V_DOMAIN_LEFT = -10.;
  constexpr double V_DOMAIN_RIGHT = 10.;

  constexpr unsigned int NV = 256;
  constexpr double DV = std::abs(V_DOMAIN_RIGHT - V_DOMAIN_LEFT)/NV;

  // deal.ii options
  constexpr unsigned int GLOBAL_REFINEMENT = 8;
  constexpr unsigned int FE_DEGREE = 3;
  constexpr unsigned int CONVERGENCE_ITERATIONS = 5000;
  constexpr double CONVERGENCE_LIMIT = 1e-8;

  constexpr double EPS = 0.01;
  constexpr double WAVE_NR = 0.5;
  constexpr double F0_FACTOR = 0.39894228040143267793994; // 1/sqrt(2pi)

  // NUFI options
  constexpr double DT=1./4.;
  constexpr unsigned int TMAX = 50;

  //Plotting options
  constexpr int PLOT_FREQUENCY = 20;
  constexpr size_t PLOT_NX = 256;
  constexpr double PLOT_DX = LX/PLOT_NX;
}

#endif
