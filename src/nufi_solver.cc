#include "nufi/nufi_solver.h"

#include <algorithm>
#include <boost/qvm/mat_access.hpp>
#include <cstdlib>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstddef>
#include <string>
#include <vector>

#include "nufi/fields.h"
#include "nufi/grids.h"
#include "nufi/parameters.h"
#include "nufi/save_results.h"

using namespace dealii;

// TO-UPDATE for dim template =================
// here double u
std::vector<double>
NuFISolver::eval_f(unsigned int n, std::vector<double> X, double u,
                   const std::vector<GridStructure<1>> &grid_struct,
                   const std::vector<SolutionSnapshot<1>> &phi_history,
                   bool is_electron) const {
  const double dt = Parameters::DT;
  size_t x_size = X.size();

  const double charge_to_mass =
      is_electron ? 1.0 : -1.0 / Parameters::MASS_RATIO;

  std::vector<double> U(x_size, u);
  std::vector<double> results(x_size);
  if (n == 0) {
    for (size_t i = 0; i < x_size; ++i)
      results[i] = is_electron ? f0(X[i], U[i]) : f0_ion(X[i], U[i]);
    return results;
  }

  std::vector<double> Ex(x_size);

  std::vector<double> tmp(x_size);
  // Initial half-step.
  tmp = eval(X, grid_struct[phi_history[n].grid_version],
             phi_history[n].solution); // call eval only once
  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U[i] = U[i] + 0.5 * dt * charge_to_mass * Ex[i];
  }

  while (--n) {
    for (size_t i = 0; i < x_size; ++i)
      X[i] = X[i] - dt * U[i];

    tmp = eval(X, grid_struct[phi_history[n].grid_version],
               phi_history[n].solution); // call eval only once
    for (size_t i = 0; i < x_size; ++i) {
      Ex[i] = -tmp[i];
      U[i] = U[i] + dt * charge_to_mass * Ex[i];
    }
  }

  // The final half-step.
  for (size_t i = 0; i < x_size; ++i)
    X[i] = X[i] - dt * U[i];

  tmp = eval(X, grid_struct[phi_history[n].grid_version],
             phi_history[n].solution); // call eval only once
  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U[i] = U[i] + 0.5 * dt * charge_to_mass * Ex[i];
  }

  for (size_t i = 0; i < x_size; ++i)
    results[i] = is_electron ? f0(X[i], U[i]) : f0_ion(X[i], U[i]);
  return results;
}

// TO-UPDATE for dim template =================

std::vector<double> NuFISolver::eval_ftilda_batch(
    unsigned int n, std::vector<double> X, std::vector<double> U,
    const std::vector<GridStructure<1>> &grid_structures,
    const std::vector<SolutionSnapshot<1>> &phi_history,
    bool is_electron) const {

  const size_t x_size = X.size();
  std::vector<double> results(x_size);

  const double charge_to_mass =
      is_electron ? 1.0 : -1.0 / Parameters::MASS_RATIO;

  if (n == 0) {
    for (size_t k = 0; k < x_size; ++k)
      results[k] = is_electron ? f0(X[k], U[k]) : f0_ion(X[k], U[k]);
    return results;
  }

#pragma omp parallel
  {
    const int nthreads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const size_t chunk = (x_size + nthreads - 1) / nthreads;
    const size_t begin = std::min(x_size, chunk * static_cast<size_t>(tid));
    const size_t end = std::min(x_size, chunk * static_cast<size_t>(tid + 1));
    const size_t local_n = end - begin;

    if (local_n > 0) {
      std::vector<double> Xl(X.begin() + begin, X.begin() + end);
      std::vector<double> Ul(U.begin() + begin, U.begin() + end);
      std::vector<double> Exl(local_n);
      std::vector<double> tmpl(local_n);

      unsigned int nn = n;
      while (--nn) {
        const GridStructure<Parameters::DIMENSION> &grid_struct =
            grid_structures[phi_history[nn].grid_version];

        for (size_t k = 0; k < local_n; ++k)
          Xl[k] -= Parameters::DT * Ul[k];

        AssertThrow(
            phi_history[nn].solution.size() ==
                grid_struct.dof_handler->n_dofs(),
            ExcMessage("[NuFISolver::eval_ftilda_batch] Solution size = " +
                       std::to_string(phi_history[nn].solution.size()) +
                       ", expected by grid_struct = " +
                       std::to_string(grid_struct.dof_handler->n_dofs())));
        AssertThrow(
            grid_struct.grid_version == phi_history[nn].grid_version,
            ExcMessage("[NuFISolver::eval_ftilda_batch] grid.grid_version "
                       "not equal to phi_history[nn].grid_version"));

        tmpl = eval(Xl, grid_struct, phi_history[nn].solution);

        for (size_t k = 0; k < local_n; ++k) {
          Exl[k] = -tmpl[k];
          Ul[k] += Parameters::DT * charge_to_mass * Exl[k];
        }
      }

      // Final half-step, nn == 0 here.
      const GridStructure<Parameters::DIMENSION> &grid_struct =
          grid_structures[phi_history[nn].grid_version];

      for (size_t k = 0; k < local_n; ++k)
        Xl[k] -= Parameters::DT * Ul[k];

      tmpl = eval(Xl, grid_struct, phi_history[nn].solution);

      for (size_t k = 0; k < local_n; ++k) {
        Exl[k] = -tmpl[k];
        Ul[k] += .5 * Parameters::DT * charge_to_mass * Exl[k];
      }

      for (size_t k = 0; k < local_n; ++k)
        results[begin + k] =
            is_electron ? f0(Xl[k], Ul[k]) : f0_ion(Xl[k], Ul[k]);
    }
  }

  return results;
}

// TO-UPDATE for dim template =================

std::vector<double>
NuFISolver::eval_f_batch(unsigned int n, std::vector<double> X,
                         std::vector<double> U,
                         const std::vector<GridStructure<1>> &grid_structures,
                         const std::vector<SolutionSnapshot<1>> &phi_history,
                         bool is_electron) const {

  const size_t x_size = X.size();
  std::vector<double> results(x_size);

  const double charge_to_mass =
      is_electron ? 1.0 : -1.0 / Parameters::MASS_RATIO;

  if (n == 0) {
    for (size_t k = 0; k < x_size; ++k)
      results[k] = is_electron ? f0(X[k], U[k]) : f0_ion(X[k], U[k]);
    return results;
  }

#pragma omp parallel
  {
    const int nthreads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const size_t chunk = (x_size + nthreads - 1) / nthreads;
    const size_t begin = std::min(x_size, chunk * static_cast<size_t>(tid));
    const size_t end = std::min(x_size, chunk * static_cast<size_t>(tid + 1));
    const size_t local_n = end - begin;

    if (local_n > 0) {
      std::vector<double> Xl(X.begin() + begin, X.begin() + end);
      std::vector<double> Ul(U.begin() + begin, U.begin() + end);
      std::vector<double> Exl(local_n);
      std::vector<double> tmpl(local_n);

      // Initial half-step, using phi_history[n] (n not yet decremented).
      {
        const GridStructure<Parameters::DIMENSION> &grid_struct =
            grid_structures[phi_history[n].grid_version];

        AssertThrow(
            phi_history[n].solution.size() == grid_struct.dof_handler->n_dofs(),
            ExcMessage("[NuFISolver::eval_f_batch] Solution size = " +
                       std::to_string(phi_history[n].solution.size()) +
                       ", expected by grid_struct = " +
                       std::to_string(grid_struct.dof_handler->n_dofs())));
        AssertThrow(grid_struct.grid_version == phi_history[n].grid_version,
                    ExcMessage("[NuFISolver::eval_f_batch] grid.grid_version "
                               "not equal to phi_history[n].grid_version"));

        tmpl = eval(Xl, grid_struct, phi_history[n].solution);

        for (size_t k = 0; k < local_n; ++k) {
          Exl[k] = -tmpl[k];
          Ul[k] += 0.5 * Parameters::DT * charge_to_mass * Exl[k];
        }
      }

      unsigned int nn = n;
      while (--nn) {
        const GridStructure<Parameters::DIMENSION> &grid_struct =
            grid_structures[phi_history[nn].grid_version];

        for (size_t k = 0; k < local_n; ++k)
          Xl[k] -= Parameters::DT * Ul[k];

        AssertThrow(
            phi_history[nn].solution.size() ==
                grid_struct.dof_handler->n_dofs(),
            ExcMessage("[NuFISolver::eval_f_batch] Solution size = " +
                       std::to_string(phi_history[nn].solution.size()) +
                       ", expected by grid_struct = " +
                       std::to_string(grid_struct.dof_handler->n_dofs())));
        AssertThrow(grid_struct.grid_version == phi_history[nn].grid_version,
                    ExcMessage("[NuFISolver::eval_f_batch] grid.grid_version "
                               "not equal to phi_history[nn].grid_version"));

        tmpl = eval(Xl, grid_struct, phi_history[nn].solution);

        for (size_t k = 0; k < local_n; ++k) {
          Exl[k] = -tmpl[k];
          Ul[k] += Parameters::DT * charge_to_mass * Exl[k];
        }
      }

      // Final half-step, nn == 0 here.
      const GridStructure<Parameters::DIMENSION> &grid_struct =
          grid_structures[phi_history[nn].grid_version];

      for (size_t k = 0; k < local_n; ++k)
        Xl[k] -= Parameters::DT * Ul[k];

      tmpl = eval(Xl, grid_struct, phi_history[nn].solution);

      for (size_t k = 0; k < local_n; ++k) {
        Exl[k] = -tmpl[k];
        Ul[k] += 0.5 * Parameters::DT * charge_to_mass * Exl[k];
      }

      for (size_t k = 0; k < local_n; ++k)
        results[begin + k] =
            is_electron ? f0(Xl[k], Ul[k]) : f0_ion(Xl[k], Ul[k]);
    }
  }

  return results;
}

// TO-UPDATE for dim template =================

std::vector<double> NuFISolver::eval_species_density(
    unsigned int n, const std::vector<double> &X,
    const std::vector<GridStructure<1>> &grid_struct,
    const std::vector<SolutionSnapshot<1>> &phi_history, unsigned int Nv,
    double v_min_domain, double v_max_domain, bool is_electron) const {

  const size_t x_size = X.size();
  const size_t total = static_cast<size_t>(Nv) * x_size;

  std::vector<double> X_all(total);
  std::vector<double> U_all(total);

  const double dv = (v_max_domain - v_min_domain) / Nv;
  const double v_min = v_min_domain + .5 * dv;

#pragma omp parallel for
  for (unsigned int i = 0; i < Nv; ++i) {
    const double v_i = v_min + i * dv;
    std::copy(X.begin(), X.end(),
              X_all.begin() + static_cast<size_t>(i) * x_size);
    std::fill(U_all.begin() + static_cast<size_t>(i) * x_size,
              U_all.begin() + static_cast<size_t>(i + 1) * x_size, v_i);
  }

  std::vector<double> ftilda_all =
      eval_ftilda_batch(n, std::move(X_all), std::move(U_all), grid_struct,
                        phi_history, is_electron);

  std::vector<double> density(x_size, 0.0);
  for (unsigned int i = 0; i < Nv; ++i)
    for (size_t ii = 0; ii < x_size; ++ii)
      density[ii] += ftilda_all[static_cast<size_t>(i) * x_size + ii];

  for (size_t i = 0; i < x_size; ++i)
    density[i] *= dv;

  return density;
}

// TO-UPDATE for dim template =================

std::vector<double>
NuFISolver::eval_rho(unsigned int n, std::vector<double> &X,
                     const std::vector<GridStructure<1>> &grid_struct,
                     const std::vector<SolutionSnapshot<1>> &phi_history,
                     const unsigned int Nv) const {

  std::vector<double> n_electron = eval_species_density(
      n, X, grid_struct, phi_history, Nv, Parameters::V_DOMAIN_LEFT,
      Parameters::V_DOMAIN_RIGHT, /*is_electron=*/true);

  if (!Parameters::IONS_ENABLED) {
    std::vector<double> rho(X.size());
    for (size_t i = 0; i < X.size(); ++i)
      rho[i] = 1.0 - n_electron[i];
    return rho;
  }

  std::vector<double> n_ion = eval_species_density(
      n, X, grid_struct, phi_history, Parameters::NV_ION,
      Parameters::ION_V_DOMAIN_LEFT, Parameters::ION_V_DOMAIN_RIGHT,
      /*is_electron=*/false);

  std::vector<double> rho(X.size());
  for (size_t i = 0; i < X.size(); ++i)
    rho[i] = n_ion[i] - n_electron[i];

  return rho;
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
