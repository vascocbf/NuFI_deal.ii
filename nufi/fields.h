#ifndef NUFI_FIELDS_H_
#define NUFI_FIELDS_H_

#include "nufi/grids.h"
#include "nufi/parameters.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <vector>

using namespace dealii;
using std::array;

template <size_t X_DIM>
inline std::vector<array<double, X_DIM>>
make_x_eval(const array<size_t, X_DIM> &Nx) {

  array<double, X_DIM> dx = {};

  for (size_t d = 0; d < X_DIM; ++d)
    dx[d] = (Parameters::X_DOMAIN_RIGHT[d] - Parameters::X_DOMAIN_LEFT[d]) /
            static_cast<double>(Nx[d]);

  size_t n_points = 1;
  for (size_t d = 0; d < X_DIM; ++d)
    n_points *= Nx[d];

  std::vector<array<double, X_DIM>> x_eval(n_points);

#pragma omp parallel for
  for (size_t i = 0; i < n_points; ++i) {
    size_t index = i;
    array<double, X_DIM> x = {};

    for (size_t d = 0; d < X_DIM; ++d) {
      size_t j = index % Nx[d];
      index /= Nx[d];
      x[d] =
          Parameters::X_DOMAIN_LEFT[d] + (static_cast<double>(j) + .5) * dx[d];
    }
    x_eval[i] = x;
  }

  return x_eval;
}

template <size_t X_DIM>
inline void reset_x_eval(std::vector<array<double, X_DIM>> &x_vals,
                         array<size_t, X_DIM> &Nx) {

  array<double, X_DIM> dv = {};

  for (size_t d = 0; d < X_DIM; ++d)
    dv[d] = (Parameters::X_DOMAIN_RIGHT[d] - Parameters::X_DOMAIN_LEFT[d]) /
            static_cast<double>(Nx[d]);

  size_t n_points = 1;
  for (size_t d = 0; d < X_DIM; ++d)
    n_points *= Nx[d];

  if (x_vals.size() != n_points)
    x_vals.resize(n_points);

#pragma omp parallel for
  for (size_t i = 0; i < n_points; ++i) {
    size_t index = i;
    array<double, X_DIM> x = {};

    for (size_t d = 0; d < X_DIM; ++d) {
      size_t j = index % Nx[d];
      index /= Nx[d];
      x[d] =
          Parameters::X_DOMAIN_LEFT[d] + (static_cast<double>(j) + .5) * dv[d];
    }
    x_vals[i] = x;
  }
};

template <size_t V_DIM>
inline array<double, V_DIM> make_dv(const array<size_t, V_DIM> &Nv,
                                    bool is_electron = true) {
  array<double, V_DIM> dv = {};
  for (size_t d = 0; d < V_DIM; ++d) {
    const double v_min = is_electron ? Parameters::V_DOMAIN_LEFT[d]
                                     : Parameters::ION_V_DOMAIN_LEFT;
    const double v_max = is_electron ? Parameters::V_DOMAIN_RIGHT[d]
                                     : Parameters::ION_V_DOMAIN_RIGHT;
    dv[d] = (v_max - v_min) / static_cast<double>(Nv[d]);
  }
  return dv;
}

template <size_t V_DIM>
inline std::vector<array<double, V_DIM>>
make_v_eval(const array<size_t, V_DIM> &Nv, const bool is_electron = true) {

  const array<double, V_DIM> dv = make_dv<V_DIM>(Nv, is_electron);

  array<double, V_DIM> v_min = {};
  for (size_t d = 0; d < V_DIM; ++d)
    v_min[d] = is_electron ? Parameters::V_DOMAIN_LEFT[d]
                           : Parameters::ION_V_DOMAIN_LEFT;

  size_t n_points = 1;
  for (size_t d = 0; d < V_DIM; ++d)
    n_points *= Nv[d];

  std::vector<array<double, V_DIM>> x_eval(n_points);

#pragma omp parallel for
  for (size_t i = 0; i < n_points; ++i) {
    size_t index = i;
    array<double, V_DIM> x = {};

    for (size_t d = 0; d < V_DIM; ++d) {
      size_t j = index % Nv[d];
      index /= Nv[d];
      x[d] =
          Parameters::V_DOMAIN_LEFT[d] + (static_cast<double>(j) + .5) * dv[d];
    }
    x_eval[i] = x;
  }

  return x_eval;
}

template <size_t V_DIM>
inline void reset_v_eval(std::vector<array<double, V_DIM>> &x_vals,
                         array<size_t, V_DIM> &Nv,
                         const bool is_electron = true) {

  const array<double, V_DIM> dv = make_dv<V_DIM>(Nv, is_electron);

  array<double, V_DIM> v_min = {};
  for (size_t d = 0; d < V_DIM; ++d)
    v_min[d] = is_electron ? Parameters::V_DOMAIN_LEFT[d]
                           : Parameters::ION_V_DOMAIN_LEFT;

  size_t n_points = 1;
  for (size_t d = 0; d < V_DIM; ++d)
    n_points *= Nv[d];

  if (x_vals.size() != n_points)
    x_vals.resize(n_points);

#pragma omp parallel for
  for (size_t i = 0; i < n_points; ++i) {
    size_t index = i;
    array<double, V_DIM> x = {};

    for (size_t d = 0; d < V_DIM; ++d) {
      size_t j = index % Nv[d];
      index /= Nv[d];
      x[d] =
          Parameters::V_DOMAIN_LEFT[d] + (static_cast<double>(j) + .5) * dv[d];
    }
    x_vals[i] = x;
  }
};

template <size_t X_DIM>
inline std::vector<std::array<double, X_DIM>>
make_x_eval_slice(const size_t free_dim,
                  const std::array<double, X_DIM> &x_fixed,
                  const size_t Nx_free) {
  const double left = Parameters::X_DOMAIN_LEFT[free_dim];
  const double right = Parameters::X_DOMAIN_RIGHT[free_dim];
  const double dx = (right - left) / static_cast<double>(Nx_free);

  std::vector<std::array<double, X_DIM>> x_eval(Nx_free);
  for (size_t i = 0; i < Nx_free; ++i) {
    std::array<double, X_DIM> x = x_fixed;
    x[free_dim] = left + (static_cast<double>(i) + 0.5) * dx;
    x_eval[i] = x;
  }
  return x_eval;
}

template <size_t V_DIM>
inline std::vector<std::array<double, V_DIM>>
make_v_eval_slice(const size_t free_dim,
                  const std::array<double, V_DIM> &v_fixed,
                  const size_t Nv_free, const bool is_electron = true) {

  const double left = is_electron ? Parameters::V_DOMAIN_LEFT[free_dim]
                                  : Parameters::ION_V_DOMAIN_LEFT;
  const double right = is_electron ? Parameters::V_DOMAIN_RIGHT[free_dim]
                                   : Parameters::ION_V_DOMAIN_RIGHT;
  const double dv = (right - left) / static_cast<double>(Nv_free);

  std::vector<std::array<double, V_DIM>> v_eval(Nv_free);
  for (size_t j = 0; j < Nv_free; ++j) {
    std::array<double, V_DIM> v = v_fixed;
    v[free_dim] = left + (static_cast<double>(j) + 0.5) * dv;
    v_eval[j] = v;
  }
  return v_eval;
}

template <size_t X_DIM, size_t V_DIM>
inline double f0(const array<double, X_DIM> &x, const array<double, V_DIM> &v,
                 const size_t f0_type = Parameters::f0_TYPE) {

  const double eps = Parameters::EPS;
  const double k = Parameters::WAVE_NR;

  const double factor = Parameters::F0_FACTOR;

  auto maxwell = [factor](const array<double, V_DIM> &u,
                          const array<double, V_DIM> &v0 = {},
                          const double v_th = 1) {
    double v2 = 0.0;
    for (std::size_t d = 0; d < V_DIM; ++d) {
      const double dv = u[d] - v0[d];
      v2 += dv * dv;
    }

    return factor / std::pow(v_th, static_cast<double>(V_DIM)) *
           std::exp(-0.5 * v2 / (v_th * v_th));
  };

  double prefactor;
  double computed_max;
  double result;

  switch (f0_type) {
  case 0: // two-stream
  {
    computed_max = maxwell(v);

    // WARNING: For two-stream instability on higher dimensions
    //          in which direction is the perturbation? just x_1 or more
    //          same question on f0_ion
    prefactor = (1.0 + eps * std::cos(k * x[0]));

    double v2 = 0;
    for (size_t i = 0; i < V_DIM; ++i)
      v2 += v[i] * v[i];

    result = prefactor * v2 * computed_max;
    break;
  }
  case 1: // Landau-damping
  {
    computed_max = maxwell(v);
    prefactor = (1.0 + eps * std::cos(k * x[0]));
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

    // Beam velocity only along v_1
    array<double, V_DIM> beam_velocity = {};
    beam_velocity[0] = beam_v;

    const double beam_max = maxwell(v, beam_velocity, beam_v_th);
    computed_max = maxwell(v);

    result = (1 - alpha) * computed_max + alpha * beam_max;
    break;
  }
  default:
    throw std::invalid_argument("Invalid f0_type");
  }

  return result;
}

template <size_t X_DIM, size_t V_DIM>
inline double f0_ion(const array<double, X_DIM> &x,
                     const array<double, V_DIM> &v) {
  const double Mr = Parameters::MASS_RATIO;
  const double eps = Parameters::ION_EPS;
  const double k = Parameters::WAVE_NR;
  const double norm = Parameters::F0_FACTOR *
                      std::sqrt(Mr); // 1/(v_th*sqrt(2pi)), v_th=1/sqrt(Mr)

  // TODO: For two-stream instability on higher dimensions
  //       in which direction is the perturbation? just x_1 or more
  //       same question on f0_ion
  const double prefactor = 1.0 + eps * std::cos(k * x[0]);

  double v2 = 0;
  for (size_t i = 0; i < V_DIM; ++i)
    v2 += v[i] * v[i];

  return prefactor * norm * std::exp(-0.5 * Mr * v2);
}

// wrapper for eval_point() { VectorTools::point_values() }
template <size_t X_DIM>
inline std::vector<array<double, X_DIM>>
eval(const std::vector<array<double, X_DIM>> &X,
     const GridStructure<X_DIM> &grid, const Vector<double> &solution) {

  AssertThrow(grid.dof_handler->n_dofs() == solution.size(),
              ExcMessage("@ eval(...) grid's number of DoFs doesn't correspond "
                         "to solution's size"));

  const size_t x_size = X.size();
  std::vector<Point<X_DIM>> points(x_size);

  for (size_t i = 0; i < x_size; ++i) {
    for (size_t d = 0; d < X_DIM; ++d)
      points[i][d] = X[i][d];
  }

  return grid.eval_vector_grad(solution, points);
}

template <size_t X_DIM>
inline double
integral_space_vector(const GridStructure<X_DIM> &grid,
                      const Vector<double> &solution,
                      const array<size_t, X_DIM> Nx = Parameters::PLOT_NX) {

  const auto X = make_x_eval<X_DIM>(Nx);

  const std::vector<array<double, X_DIM>> tmp = eval<X_DIM>(X, grid, solution);

  double volume_element = 1.0;

  for (size_t d = 0; d < X_DIM; ++d) {
    const double dx =
        (Parameters::X_DOMAIN_RIGHT[d] - Parameters::X_DOMAIN_LEFT[d]) /
        static_cast<double>(Nx[d]);
    volume_element *= dx;
  }
  double integral = 0.0;
  for (const auto &E : tmp) {
    double magnitude_squared = 0.0;
    for (size_t d = 0; d < X_DIM; ++d)
      magnitude_squared += E[d] * E[d];
    integral += std::sqrt(magnitude_squared);
  }

  return integral * volume_element;
}

template <size_t X_DIM>
inline double integral_space_vector_squared(
    const GridStructure<X_DIM> &grid, const Vector<double> &solution,
    const array<size_t, X_DIM> Nx = Parameters::PLOT_NX) {

  const auto X = make_x_eval<X_DIM>(Nx);
  const std::vector<array<double, X_DIM>> tmp = eval<X_DIM>(X, grid, solution);

  double volume_element = 1.0;
  for (size_t d = 0; d < X_DIM; ++d) {
    const double dx =
        (Parameters::X_DOMAIN_RIGHT[d] - Parameters::X_DOMAIN_LEFT[d]) /
        static_cast<double>(Nx[d]);
    volume_element *= dx;
  }

  double integral = 0.0;
  for (const auto &E : tmp) {
    for (size_t d = 0; d < X_DIM; ++d)
      integral += E[d] * E[d];
  }

  return integral * volume_element;
}

template <size_t X_DIM>
inline std::vector<array<double, X_DIM>>
Point_vector_to_double_vector(const std::vector<Point<X_DIM>> &Points) {
  const size_t n_points = Points.size();
  std::vector<array<double, X_DIM>> result(n_points);

  for (size_t i = 0; i < n_points; ++i) {
    for (size_t d = 0; d < X_DIM; ++d)
      result[i][d] = Points[i][d];
  }

  return result;
}
#endif // NUFI_FIELDS_H_
