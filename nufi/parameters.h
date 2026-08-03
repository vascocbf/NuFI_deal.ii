#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>

namespace Parameters {
constexpr unsigned int DIMENSION = 1;

constexpr double X_DOMAIN_LEFT = 0.0;
constexpr double X_DOMAIN_RIGHT = 4 * M_PI;
constexpr double LX = std::abs(X_DOMAIN_RIGHT - X_DOMAIN_LEFT);
constexpr double LX_INV = 1 / LX;

constexpr double V_DOMAIN_LEFT = -10.;
constexpr double V_DOMAIN_RIGHT = 10.;

constexpr unsigned int NV = 128;
constexpr double DV = std::abs(V_DOMAIN_RIGHT - V_DOMAIN_LEFT) / NV;

// f0_TYPE:
// 0 -> twos-stream
// 1 -> landau-damping
// 2 -> maxwellian
// 3 -> bump-on-tail
constexpr size_t f0_TYPE = 1;

// deal.ii options
constexpr unsigned int GLOBAL_REFINEMENT = 8;
constexpr unsigned int FE_DEGREE = 3;
constexpr unsigned int CONVERGENCE_ITERATIONS = 5000;
constexpr double CONVERGENCE_LIMIT = 1e-8;

// Adaptive refinement options
constexpr unsigned int REFINE_FREQUENCY = 10;
constexpr double REFINEMENT_TOP_FRACTION = 0.8;
constexpr double REFINEMENT_BOTTOM_FRACTION = 0.1;

// Gauge options
constexpr double GAUGE_DOMAIN_LEFT = 3.2;
constexpr double GAUGE_DOMAIN_RIGHT = 3.8;

constexpr double EPS = 0.01;
constexpr double WAVE_NR = 0.5;
constexpr double F0_FACTOR = 0.39894228040143267793994; // 1/sqrt(2pi)

// NUFI options
constexpr double DT = 1. / 10.;
constexpr unsigned int TMAX = 100;

// Plotting options
constexpr int PLOT_FREQUENCY = 5;
constexpr size_t PLOT_NX = 512;
constexpr double PLOT_DX = LX / PLOT_NX;
const std::string PLOT_DIR = "results/";
} // namespace Parameters

#endif
