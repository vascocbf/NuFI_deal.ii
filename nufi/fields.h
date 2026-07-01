#ifndef FIELDS_H
#define FIELDS_H

#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"
#include <cmath>
#include <cstddef>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <vector>

using namespace dealii;

// inline std::vector<int> Indices_of_points(const std::vector<double> &points,
// double x_min, double x_max, double dx, int grid_type=0)
// {
//     // grid type:
//     // 0 => uniform
//     // 1 => non uniform (TODO)
//
//     if (dx <= 0.0) {
//         throw std::invalid_argument("dx must be positive");
//     }
//     if (x_max <= x_min) {
//         throw std::invalid_argument("x_max must be > x_min");
//     }
//
//     std::vector<int> indices;
//     indices.reserve(points.size());
//
//     switch (grid_type) {
//         case 0:
//             {
//             const double L = x_max - x_min;
//             const int N = std::floor(L/dx);
//
//
//             for (double x : points) //GPT loop, to check
//             {
//                 x-= x_min;
//                 x = x - L * std::floor(x/L);
//
//                 int i = static_cast<int>(std::floor(x / dx));
//
//                 // safety: handle rare edge case due to floating precision
//                 if (i == N) i = 0;
//
//                 indices.push_back(i);
//             }
//             }
//         case 1:
//             {
//             throw std::invalid_argument("Case for non uniform grid is not
//             completed");
//             }
//         default:
//             throw std::invalid_argument("Invalid grid_type argument");
//
//     }
//     return indices;
// }
inline std::vector<double> make_x_eval(size_t Nx) {
  std::vector<double> x_eval(Nx);
  for (size_t i = 0; i < Nx; ++i)
    x_eval[i] = Parameters::X_DOMAIN_LEFT + i * Parameters::CALC_DX;
  return x_eval;
}

inline void reset_x_eval(std::vector<double> &x_vals) {
  const size_t Nx = x_vals.size();
  for (size_t i = 0; i < Nx; ++i)
    x_vals[i] = Parameters::X_DOMAIN_LEFT + i * Parameters::CALC_DX;
};

inline double f0(const double x, const double v,
                 const double eps = Parameters::EPS,
                 const double k = Parameters::WAVE_NR) {
  const double prefactor =
      Parameters::F0_FACTOR * (1.0 + eps * std::cos(k * x));
  const double gaussian = v * v * std::exp(-0.5 * v * v);

  return prefactor * gaussian;
}

// wrapper for eval_point() { VectorTools::point_values() }
inline std::vector<double> eval(std::vector<double> &X,
                                const PoissonProblem<1> &poisson,
                                const Vector<double> &solution) noexcept {
  size_t x_size = X.size();
  std::vector<double> evals(x_size);
  std::vector<Point<1>> Points(x_size);

  for (size_t i = 0; i < x_size; ++i) {
    X[i] = X[i] - Parameters::X_DOMAIN_LEFT;
    X[i] = X[i] - Parameters::LX * std::floor(X[i] * Parameters::LX_INV);

    Points[i][0] = X[i];
  }

  return eval_vector_grad(poisson.get_mapping(), poisson.get_dof_handler(),
                          solution, Points);
}

inline double integral_space_vector(const PoissonProblem<1> &poisson,
                                    const Vector<double> &solution,
                                    double dx = Parameters::PLOT_DX,
                                    size_t Nx = Parameters::PLOT_NX) {
  double integral = 0.0;
  double xmin = Parameters::X_DOMAIN_LEFT;
  std::vector<double> x_eval(Nx);
  for (size_t i = 0; i < Nx; ++i)
    x_eval[i] = xmin + i * dx;

  std::vector<double> tmp = eval(x_eval, poisson, solution);
  for (size_t i = 0; i < Nx; ++i)
    integral += tmp[i];
  return integral * dx;
};

inline double integral_space_vector_squared(const PoissonProblem<1> &poisson,
                                            const Vector<double> &solution,
                                            double dx = Parameters::PLOT_DX,
                                            size_t Nx = Parameters::PLOT_NX) {
  double integral = 0.0;
  double xmin = Parameters::X_DOMAIN_LEFT;
  std::vector<double> x_eval(Nx);
  for (size_t i = 0; i < Nx; ++i)
    x_eval[i] = xmin + i * dx;

  std::vector<double> tmp = eval(x_eval, poisson, solution);
  for (size_t i = 0; i < Nx; ++i)
    integral += tmp[i] * tmp[i];
  return integral * dx;
};

#endif
