#ifndef NUFI_FIELDS_H_
#define NUFI_FIELDS_H_

#include "nufi/grids.h"
#include "nufi/parameters.h"

#include <cmath>
#include <cstddef>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <vector>

using namespace dealii;
inline std::vector<double> make_x_eval(size_t Nx) {
  std::vector<double> x_eval_E;

  double dx = (Parameters::X_DOMAIN_RIGHT - Parameters::X_DOMAIN_LEFT) / Nx;

  for (unsigned int i = 0; i < Nx; ++i)
    x_eval_E.push_back(Parameters::X_DOMAIN_LEFT + (i + 0.5) * dx);

  return x_eval_E;
}

inline void reset_x_eval(std::vector<double> &x_vals) {
  const size_t Nx = x_vals.size();
  const double dx = Parameters::LX / Nx;
  for (size_t i = 0; i < Nx; ++i)
    x_vals[i] = Parameters::X_DOMAIN_LEFT + i * dx;
};

inline double f0(const double x, const double v,
                 const size_t f0_type = Parameters::f0_TYPE) {
  const double eps = Parameters::EPS;
  const double k = Parameters::WAVE_NR;

  const double factor = Parameters::F0_FACTOR;

  auto maxwell = [factor](double u, double v = 0, double v_th = 1) {
    return 1 / v_th * factor *
           std::exp(-0.5 * (u - v) * (u - v) / (v_th * v_th));
  };

  double prefactor;
  double computed_max;
  double result;

  switch (f0_type) {
  case 0: // two-stream
  {
    computed_max = maxwell(v);
    prefactor = (1.0 + eps * std::cos(k * x)) * v * v;
    result = prefactor * computed_max;
    break;
  }
  case 1: // Landau-damping
  {
    computed_max = maxwell(v);
    prefactor = (1.0 + eps * std::cos(k * x));
    result = prefactor * computed_max;
    break;
  }
  case 2: // Maxwellian
  {
    result = maxwell(v);
    break;
  }
  case 3: // Bump-on tail
  {
    const double beam_density = 0.05;
    const double beam_v_th = 0.2;
    const double beam_v = 3;
    const double alpha = beam_density / (1 - beam_density);

    const double beam_max = maxwell(v, beam_v, beam_v_th);
    computed_max = maxwell(v);
    result = (1 - alpha) * computed_max + alpha * beam_max;
    break;
  }
  default:
    throw std::invalid_argument("Invalid f0_type");
  }

  return result;
}

inline double f0_ion(const double x, const double v) {
  const double Mr = Parameters::MASS_RATIO;
  const double eps = Parameters::ION_EPS;
  const double k = Parameters::WAVE_NR;
  const double norm = Parameters::F0_FACTOR *
                      std::sqrt(Mr); // 1/(v_th*sqrt(2pi)), v_th=1/sqrt(Mr)

  const double prefactor = 1.0 + eps * std::cos(k * x);
  return prefactor * norm * std::exp(-0.5 * Mr * v * v);
}

// wrapper for eval_point() { VectorTools::point_values() }
inline std::vector<double> eval(std::vector<double> &X,
                                const GridStructure<1> &grid,
                                const Vector<double> &solution) noexcept {

  AssertThrow(grid.dof_handler->n_dofs() == solution.size(),
              ExcMessage("@ eval(...) grid's number of DoFs doesn't correspond "
                         "to solution's size"));

  const size_t x_size = X.size();
  std::vector<Point<1>> points(x_size);

  for (size_t i = 0; i < x_size; ++i)
    points[i][0] = X[i];

  return grid.eval_vector_grad(solution, points);
}

inline double integral_space_vector(const GridStructure<1> &grid,
                                    const Vector<double> &solution,
                                    size_t Nx = Parameters::PLOT_NX) {
  double integral = 0.0;
  std::vector<double> x_eval = make_x_eval(Nx);
  const double dx = Parameters::LX / Nx;

  std::vector<double> tmp = eval(x_eval, grid, solution);
  for (size_t i = 0; i < Nx; ++i)
    integral += tmp[i];
  return integral * dx;
}

inline double integral_space_vector_squared(const GridStructure<1> &grid,
                                            const Vector<double> &solution,
                                            size_t Nx = Parameters::PLOT_NX) {
  double integral = 0.0;
  std::vector<double> x_eval = make_x_eval(Nx);
  const double dx = Parameters::LX / Nx;

  std::vector<double> tmp = eval(x_eval, grid, solution);
  for (size_t i = 0; i < Nx; ++i)
    integral += tmp[i] * tmp[i];
  return integral * dx;
}

inline std::vector<double>
Point_vector_to_double_vector(const std::vector<Point<1>> &Points) {
  const size_t n_points = Points.size();
  std::vector<double> vector(n_points);

  for (size_t i = 0; i < n_points; ++i)
    vector[i] = Points[i][0];

  return vector;
}
#endif // NUFI_FIELDS_H_
