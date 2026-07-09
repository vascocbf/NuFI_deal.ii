#ifndef FIELDS_H
#define FIELDS_H

#include "grids.h"
#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"
#include <cmath>
#include <cstddef>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <vector>

using namespace dealii;

inline std::vector<double> make_x_eval(size_t Nx) {
  std::vector<double> x_eval(Nx);
  const double dx = Parameters::LX / Nx;
  for (size_t i = 0; i < Nx; ++i)
    x_eval[i] = Parameters::X_DOMAIN_LEFT + i * dx;
  return x_eval;
}

inline void reset_x_eval(std::vector<double> &x_vals) {
  const size_t Nx = x_vals.size();
  const double dx = Parameters::LX / Nx;
  for (size_t i = 0; i < Nx; ++i)
    x_vals[i] = Parameters::X_DOMAIN_LEFT + i * dx;
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
                                const GridStructure<1> &grid,
                                const Vector<double> &solution) noexcept {
  size_t x_size = X.size();
  std::vector<double> evals(x_size);
  std::vector<Point<1>> Points(x_size);

  for (size_t i = 0; i < x_size; ++i) {
    X[i] = X[i] - Parameters::X_DOMAIN_LEFT;
    X[i] = X[i] - Parameters::LX * std::floor(X[i] * Parameters::LX_INV);

    Points[i][0] = X[i];
  }

  return grid.eval_vector_grad(solution, Points);
}

inline double integral_space_vector(const GridStructure<1> &grid,
                                    const Vector<double> &solution,
                                    double dx = Parameters::PLOT_DX,
                                    size_t Nx = Parameters::PLOT_NX) {
  double integral = 0.0;
  double xmin = Parameters::X_DOMAIN_LEFT;
  std::vector<double> x_eval(Nx);
  for (size_t i = 0; i < Nx; ++i)
    x_eval[i] = xmin + i * dx;

  std::vector<double> tmp = eval(x_eval, grid, solution);
  for (size_t i = 0; i < Nx; ++i)
    integral += tmp[i];
  return integral * dx;
};

inline double integral_space_vector_squared(const GridStructure<1> &grid,
                                            const Vector<double> &solution,
                                            double dx = Parameters::PLOT_DX,
                                            size_t Nx = Parameters::PLOT_NX) {
  double integral = 0.0;
  double xmin = Parameters::X_DOMAIN_LEFT;
  std::vector<double> x_eval(Nx);
  for (size_t i = 0; i < Nx; ++i)
    x_eval[i] = xmin + i * dx;

  std::vector<double> tmp = eval(x_eval, grid, solution);
  for (size_t i = 0; i < Nx; ++i)
    integral += tmp[i] * tmp[i];
  return integral * dx;
};

inline std::vector<double>
Point_vector_to_double_vector(const std::vector<Point<1>> &Points) {
  const size_t n_points = Points.size();
  std::vector<double> vector(n_points);

  for (size_t i = 0; i < n_points; ++i)
    vector[i] = Points[i][0];

  return vector;
}

#endif
