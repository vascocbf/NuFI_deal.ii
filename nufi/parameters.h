#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cmath>
#include <cstdlib>
#include <string>

namespace Parameters {
constexpr unsigned int DIMENSION = 1;

constexpr double X_DOMAIN_LEFT = 0.0;
constexpr double X_DOMAIN_RIGHT = 4 * M_PI;
constexpr double LX = std::abs(X_DOMAIN_RIGHT - X_DOMAIN_LEFT);
constexpr double LX_INV = 1 / LX;

constexpr size_t CALC_NX = 256;
constexpr double CALC_DX = LX / CALC_NX;

constexpr double V_DOMAIN_LEFT_1 = -10.;
constexpr double V_DOMAIN_RIGHT_1 = 10.;

constexpr unsigned int NV_1 = 128;
constexpr double DV_1 = std::abs(V_DOMAIN_RIGHT_1 - V_DOMAIN_LEFT_1) / NV_1;

constexpr double V_DOMAIN_LEFT_2 = -10.;
constexpr double V_DOMAIN_RIGHT_2 = 10.;

constexpr unsigned int NV_2 = 128;
constexpr double DV_2 = std::abs(V_DOMAIN_RIGHT_2 - V_DOMAIN_LEFT_2) / NV_2;
// deal.ii options
constexpr unsigned int GLOBAL_REFINEMENT = 6;
constexpr unsigned int FE_DEGREE = 3;
constexpr unsigned int CONVERGENCE_ITERATIONS = 5000;
constexpr double CONVERGENCE_LIMIT = 1e-8;

// Gauge options
constexpr double GAUGE_DOMAIN_LEFT = 3.2;
constexpr double GAUGE_DOMAIN_RIGHT = 3.8;

constexpr double EPS = 0.01;
constexpr double WAVE_NR = 0.5;
constexpr double F0_FACTOR = 0.39894228040143267793994; // 1/sqrt(2pi)

// NUFI options
constexpr double DT = 1. / 10.;
constexpr unsigned int TMAX = 100;
constexpr unsigned int REFINE_FREQUENCY = 30;

// Plotting options
constexpr int PLOT_FREQUENCY = 10;
constexpr size_t PLOT_NX = CALC_NX;
constexpr double PLOT_DX = LX / PLOT_NX;
constexpr double PLOT_FIXED_V2 = 0.;
const std::string PLOT_DIR = "results/";
} // namespace Parameters

#endif
