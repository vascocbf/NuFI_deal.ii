#include "nufi/nufi_solver.h"

#include <algorithm>
#include <boost/qvm/mat_access.hpp>
#include <cmath>
#include <cstdlib>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "nufi/fields.h"
#include "nufi/grids.h"
#include "nufi/parameters.h"
#include "nufi/save_results.h"

using namespace dealii;

std::vector<double>
NuFISolver::eval_f(unsigned int n, std::vector<double> X, double u,
                   const std::vector<GridStructure<1>> &grid_struct,
                   const std::vector<SolutionSnapshot<1>> &phi_history) const {

  size_t x_size = X.size();

  std::vector<double> U(x_size, u);
  std::vector<double> results(x_size);
  if (n == 0) {
    for (size_t i = 0; i < x_size; ++i)
      results[i] = f0(X[i], U[i]);
    return results;
  }

  std::vector<double> Ex(x_size);

  std::vector<double> tmp(x_size);
  // Initial half-step.
  tmp = eval(X, grid_struct[phi_history[n].grid_version],
             phi_history[n].solution); // call eval only once
  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U[i] = U[i] + 0.5 * Parameters::DT * Ex[i];
  }

  while (--n) {
    for (size_t i = 0; i < x_size; ++i)
      X[i] = X[i] - Parameters::DT * U[i];

    tmp = eval(X, grid_struct[phi_history[n].grid_version],
               phi_history[n].solution); // call eval only once
    for (size_t i = 0; i < x_size; ++i) {
      Ex[i] = -tmp[i];
      U[i] = U[i] + Parameters::DT * Ex[i];
    }
  }

  // The final half-step.
  for (size_t i = 0; i < x_size; ++i)
    X[i] = X[i] - Parameters::DT * U[i];

  tmp = eval(X, grid_struct[phi_history[n].grid_version],
             phi_history[n].solution); // call eval only once
  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U[i] = U[i] + 0.5 * Parameters::DT * Ex[i];
  }

  for (size_t i = 0; i < x_size; ++i)
    results[i] = f0(X[i], U[i]);
  return results;
}

std::vector<double> NuFISolver::eval_ftilda_batch(
    unsigned int n, std::vector<double> X, std::vector<double> U,
    const std::vector<GridStructure<1>> &grid_struct,
    const std::vector<SolutionSnapshot<1>> &phi_history) const {

  const size_t x_size = X.size();
  std::vector<double> results(x_size);

  if (n == 0) {
    for (size_t k = 0; k < x_size; ++k)
      results[k] = f0(X[k], U[k]);
    return results;
  }

  std::vector<double> Ex(x_size);
  std::vector<double> tmp(x_size);

  // We omit the initial half-step.
  while (--n) {
    for (size_t k = 0; k < x_size; ++k)
      X[k] = X[k] - Parameters::DT * U[k];

    AssertThrow(
        phi_history[n].solution.size() ==
            grid_struct[phi_history[n].grid_version].dof_handler->n_dofs(),
        ExcMessage("In eval_ftilda_batch: Solution size = " +
                   std::to_string(phi_history[n].solution.size()) +
                   ", expected by grid_struct = " +
                   std::to_string(grid_struct[phi_history[n].grid_version]
                                      .dof_handler->n_dofs())));
    AssertThrow(
        grid_struct[phi_history[n].grid_version].grid_version ==
            phi_history[n].grid_version,
        ExcMessage(
            "grid.grid_version not equal to phi_history[n].grid_version"));

    tmp = eval(X, grid_struct[phi_history[n].grid_version],
               phi_history[n].solution); // one call for the whole batch

    for (size_t k = 0; k < x_size; ++k) {
      Ex[k] = -tmp[k];
      U[k] = U[k] + Parameters::DT * Ex[k];
    }
  }

  // The final half-step.
  for (size_t k = 0; k < x_size; ++k)
    X[k] = X[k] - Parameters::DT * U[k];

  tmp = eval(X, grid_struct[phi_history[n].grid_version],
             phi_history[n].solution);

  for (size_t k = 0; k < x_size; ++k) {
    Ex[k] = -tmp[k];
    U[k] = U[k] + 0.5 * Parameters::DT * Ex[k];
  }
  for (size_t k = 0; k < x_size; ++k)
    results[k] = f0(X[k], U[k]);
  return results;
}

std::vector<double>
NuFISolver::eval_rho(unsigned int n, std::vector<double> &X,
                     const std::vector<GridStructure<1>> &grid_struct,
                     const std::vector<SolutionSnapshot<1>> &phi_history,
                     const unsigned int Nv) const {
  const size_t x_size = X.size();
  const size_t total = static_cast<size_t>(Nv) * x_size;

  std::vector<double> X_all(total);
  std::vector<double> U_all(total);

  const double dv =
      (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;
  const double v_min = Parameters::V_DOMAIN_LEFT + .5 * dv;

#pragma omp parallel for
  for (unsigned int i = 0; i < Nv; ++i) {
    const double v_i = v_min + i * dv;
    std::copy(X.begin(), X.end(),
              X_all.begin() + static_cast<size_t>(i) * x_size);
    std::fill(U_all.begin() + static_cast<size_t>(i) * x_size,
              U_all.begin() + static_cast<size_t>(i + 1) * x_size, v_i);
  }

  std::vector<double> ftilda_all = eval_ftilda_batch(
      n, std::move(X_all), std::move(U_all), grid_struct, phi_history);

  std::vector<double> integral(x_size, 0.0);
  for (unsigned int i = 0; i < Nv; ++i)
    for (size_t ii = 0; ii < x_size; ++ii)
      integral[ii] += ftilda_all[static_cast<size_t>(i) * x_size + ii];

  for (size_t i = 0; i < x_size; ++i)
    integral[i] = 1 - integral[i] * dv;
  return integral;
}

std::vector<double>
NuFISolver::eval_rho_points(unsigned int n, const std::vector<Point<1>> &points,
                            const std::vector<GridStructure<1>> &grid_struct,
                            const std::vector<SolutionSnapshot<1>> &phi_history,
                            const unsigned int Nv) const {
  std::vector<double> point_vector = Point_vector_to_double_vector(points);
  return NuFISolver::eval_rho(n, point_vector, grid_struct, phi_history, Nv);
}

NuFISolver::NuFISolver() = default;
