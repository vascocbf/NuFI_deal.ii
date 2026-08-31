#include "nufi/nufi_solver.h"

#include <algorithm>
#include <array>
#include <boost/qvm/mat_access.hpp>
#include <cstdlib>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>
#include <omp.h>

#include <cstddef>
#include <string>
#include <vector>

#include "nufi/fields.h"
#include "nufi/grids.h"
#include "nufi/parameters.h"

using namespace dealii;
using std::array;

template <size_t X_DIM, size_t V_DIM>
NuFISolver<X_DIM, V_DIM>::NuFISolver() = default;

template <size_t X_DIM, size_t V_DIM>
std::vector<double> NuFISolver<X_DIM, V_DIM>::eval_f(
    unsigned int n, std::vector<array<double, X_DIM>> X, array<double, V_DIM> u,
    const std::vector<GridStructure<X_DIM>> &grid_struct,
    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
    bool is_electron) const {

  const double dt = Parameters::DT;
  const size_t x_size = X.size();

  const double charge_to_mass =
      is_electron ? 1.0 : -1.0 / Parameters::MASS_RATIO;

  std::vector<array<double, V_DIM>> U(x_size, u);
  std::vector<double> results(x_size);

  if (n == 0) {
    for (size_t i = 0; i < x_size; ++i)
      results[i] = is_electron ? f0<X_DIM, V_DIM>(X[i], U[i])
                               : f0_ion<X_DIM, V_DIM>(X[i], U[i]);
    return results;
  }

  std::vector<array<double, X_DIM>> Ex(x_size);
  std::vector<array<double, X_DIM>> tmp(x_size);

  // Initial half-step.
  tmp = eval<X_DIM>(X, grid_struct[phi_history[n].grid_version],
                    phi_history[n].solution);
  for (size_t i = 0; i < x_size; ++i)
    for (size_t d = 0; d < X_DIM; ++d) {
      Ex[i][d] = -tmp[i][d];
      U[i][d] += 0.5 * dt * charge_to_mass * Ex[i][d];
    }

  while (--n) {
    for (size_t i = 0; i < x_size; ++i)
      for (size_t d = 0; d < X_DIM; ++d)
        X[i][d] -= dt * U[i][d];

    tmp = eval<X_DIM>(X, grid_struct[phi_history[n].grid_version],
                      phi_history[n].solution);
    for (size_t i = 0; i < x_size; ++i)
      for (size_t d = 0; d < X_DIM; ++d) {
        Ex[i][d] = -tmp[i][d];
        U[i][d] += dt * charge_to_mass * Ex[i][d];
      }
  }

  // The final half-step.
  for (size_t i = 0; i < x_size; ++i)
    for (size_t d = 0; d < X_DIM; ++d)
      X[i][d] -= dt * U[i][d];

  tmp = eval<X_DIM>(X, grid_struct[phi_history[n].grid_version],
                    phi_history[n].solution);
  for (size_t i = 0; i < x_size; ++i)
    for (size_t d = 0; d < X_DIM; ++d) {
      Ex[i][d] = -tmp[i][d];
      U[i][d] += 0.5 * dt * charge_to_mass * Ex[i][d];
    }

  for (size_t i = 0; i < x_size; ++i)
    results[i] = is_electron ? f0<X_DIM, V_DIM>(X[i], U[i])
                             : f0_ion<X_DIM, V_DIM>(X[i], U[i]);
  return results;
}

template <size_t X_DIM, size_t V_DIM>
std::vector<double> NuFISolver<X_DIM, V_DIM>::eval_ftilda_batch(
    unsigned int n, std::vector<array<double, X_DIM>> X,
    std::vector<array<double, V_DIM>> U,
    const std::vector<GridStructure<X_DIM>> &grid_structures,
    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
    bool is_electron) const {

  const size_t x_size = X.size();
  std::vector<double> results(x_size);

  const double charge_to_mass =
      is_electron ? 1.0 : -1.0 / Parameters::MASS_RATIO;

  if (n == 0) {
    for (size_t k = 0; k < x_size; ++k)
      results[k] = is_electron ? f0<X_DIM, V_DIM>(X[k], U[k])
                               : f0_ion<X_DIM, V_DIM>(X[k], U[k]);
    return results;
  }

#pragma omp parallel if (!omp_in_parallel())
  {
    const int nthreads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const size_t chunk = (x_size + nthreads - 1) / nthreads;
    const size_t begin = std::min(x_size, chunk * static_cast<size_t>(tid));
    const size_t end = std::min(x_size, chunk * static_cast<size_t>(tid + 1));
    const size_t local_n = end - begin;

    if (local_n > 0) {
      std::vector<array<double, X_DIM>> Xl(X.begin() + begin, X.begin() + end);
      std::vector<array<double, V_DIM>> Ul(U.begin() + begin, U.begin() + end);
      std::vector<array<double, X_DIM>> Exl(local_n);
      std::vector<array<double, X_DIM>> tmpl(local_n);

      unsigned int nn = n;
      while (--nn) {
        const GridStructure<X_DIM> &grid_struct =
            grid_structures[phi_history[nn].grid_version];

        for (size_t k = 0; k < local_n; ++k)
          for (size_t d = 0; d < X_DIM; ++d)
            Xl[k][d] -= Parameters::DT * Ul[k][d];

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

        tmpl = eval<X_DIM>(Xl, grid_struct, phi_history[nn].solution);

        for (size_t k = 0; k < local_n; ++k)
          for (size_t d = 0; d < X_DIM; ++d) {
            Exl[k][d] = -tmpl[k][d];
            Ul[k][d] += Parameters::DT * charge_to_mass * Exl[k][d];
          }
      }

      // Final half-step, nn == 0 here.
      const GridStructure<X_DIM> &grid_struct =
          grid_structures[phi_history[nn].grid_version];

      for (size_t k = 0; k < local_n; ++k)
        for (size_t d = 0; d < X_DIM; ++d)
          Xl[k][d] -= Parameters::DT * Ul[k][d];

      tmpl = eval<X_DIM>(Xl, grid_struct, phi_history[nn].solution);

      for (size_t k = 0; k < local_n; ++k)
        for (size_t d = 0; d < X_DIM; ++d) {
          Exl[k][d] = -tmpl[k][d];
          Ul[k][d] += .5 * Parameters::DT * charge_to_mass * Exl[k][d];
        }

      for (size_t k = 0; k < local_n; ++k)
        results[begin + k] = is_electron ? f0<X_DIM, V_DIM>(Xl[k], Ul[k])
                                         : f0_ion<X_DIM, V_DIM>(Xl[k], Ul[k]);
    }
  }

  return results;
}

template <size_t X_DIM, size_t V_DIM>
std::vector<double> NuFISolver<X_DIM, V_DIM>::eval_f_batch(
    unsigned int n, std::vector<array<double, X_DIM>> X,
    std::vector<array<double, V_DIM>> U,
    const std::vector<GridStructure<X_DIM>> &grid_structures,
    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
    bool is_electron) const {

  const size_t x_size = X.size();
  std::vector<double> results(x_size);

  const double charge_to_mass =
      is_electron ? 1.0 : -1.0 / Parameters::MASS_RATIO;

  if (n == 0) {
    for (size_t k = 0; k < x_size; ++k)
      results[k] = is_electron ? f0<X_DIM, V_DIM>(X[k], U[k])
                               : f0_ion<X_DIM, V_DIM>(X[k], U[k]);
    return results;
  }

#pragma omp parallel if (!omp_in_parallel())
  {
    const int nthreads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const size_t chunk = (x_size + nthreads - 1) / nthreads;
    const size_t begin = std::min(x_size, chunk * static_cast<size_t>(tid));
    const size_t end = std::min(x_size, chunk * static_cast<size_t>(tid + 1));
    const size_t local_n = end - begin;

    if (local_n > 0) {
      std::vector<array<double, X_DIM>> Xl(X.begin() + begin, X.begin() + end);
      std::vector<array<double, V_DIM>> Ul(U.begin() + begin, U.begin() + end);
      std::vector<array<double, X_DIM>> Exl(local_n);
      std::vector<array<double, X_DIM>> tmpl(local_n);

      // Initial half-step, using phi_history[n] (n not yet decremented).
      {
        const GridStructure<X_DIM> &grid_struct =
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

        tmpl = eval<X_DIM>(Xl, grid_struct, phi_history[n].solution);

        for (size_t k = 0; k < local_n; ++k)
          for (size_t d = 0; d < X_DIM; ++d) {
            Exl[k][d] = -tmpl[k][d];
            Ul[k][d] += 0.5 * Parameters::DT * charge_to_mass * Exl[k][d];
          }
      }

      unsigned int nn = n;
      while (--nn) {
        const GridStructure<X_DIM> &grid_struct =
            grid_structures[phi_history[nn].grid_version];

        for (size_t k = 0; k < local_n; ++k)
          for (size_t d = 0; d < X_DIM; ++d)
            Xl[k][d] -= Parameters::DT * Ul[k][d];

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

        tmpl = eval<X_DIM>(Xl, grid_struct, phi_history[nn].solution);

        for (size_t k = 0; k < local_n; ++k)
          for (size_t d = 0; d < X_DIM; ++d) {
            Exl[k][d] = -tmpl[k][d];
            Ul[k][d] += Parameters::DT * charge_to_mass * Exl[k][d];
          }
      }

      // Final half-step, nn == 0 here.
      const GridStructure<X_DIM> &grid_struct =
          grid_structures[phi_history[nn].grid_version];

      for (size_t k = 0; k < local_n; ++k)
        for (size_t d = 0; d < X_DIM; ++d)
          Xl[k][d] -= Parameters::DT * Ul[k][d];

      tmpl = eval<X_DIM>(Xl, grid_struct, phi_history[nn].solution);

      for (size_t k = 0; k < local_n; ++k)
        for (size_t d = 0; d < X_DIM; ++d) {
          Exl[k][d] = -tmpl[k][d];
          Ul[k][d] += 0.5 * Parameters::DT * charge_to_mass * Exl[k][d];
        }

      for (size_t k = 0; k < local_n; ++k)
        results[begin + k] = is_electron ? f0<X_DIM, V_DIM>(Xl[k], Ul[k])
                                         : f0_ion<X_DIM, V_DIM>(Xl[k], Ul[k]);
    }
  }

  return results;
}

template <size_t X_DIM, size_t V_DIM>
double NuFISolver<X_DIM, V_DIM>::eval_density_at_x(
    unsigned int n, const array<double, X_DIM> &x_point,
    const std::vector<array<double, V_DIM>> &v_eval,
    const std::vector<GridStructure<X_DIM>> &grid_struct,
    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
    bool is_electron) const {

  const size_t n_v = v_eval.size();
  const size_t chunk_size = Parameters::V_CHUNK_SIZE;

  double rho_local = 0.0;

  for (size_t chunk_start = 0; chunk_start < n_v; chunk_start += chunk_size) {
    const size_t chunk_end = std::min(n_v, chunk_start + chunk_size);
    const size_t local_n = chunk_end - chunk_start;

    std::vector<array<double, X_DIM>> X_chunk(local_n, x_point);
    std::vector<array<double, V_DIM>> U_chunk(v_eval.begin() + chunk_start,
                                              v_eval.begin() + chunk_end);

    std::vector<double> ftilda_chunk =
        eval_ftilda_batch(n, std::move(X_chunk), std::move(U_chunk),
                          grid_struct, phi_history, is_electron);

    for (double f : ftilda_chunk)
      rho_local += f;
  }

  return rho_local;
}

template <size_t X_DIM, size_t V_DIM>
std::vector<double> NuFISolver<X_DIM, V_DIM>::eval_species_density(
    unsigned int n, const std::vector<array<double, X_DIM>> &X,
    const std::vector<GridStructure<X_DIM>> &grid_struct,
    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
    const array<size_t, V_DIM> &Nv, bool is_electron) const {

  const size_t x_size = X.size();

  const std::vector<array<double, V_DIM>> v_eval =
      make_v_eval<V_DIM>(Nv, is_electron);
  const array<double, V_DIM> dv = make_dv<V_DIM>(Nv, is_electron);

  double v_cell_volume = 1.0;
  for (size_t d = 0; d < V_DIM; ++d)
    v_cell_volume *= dv[d];

  std::vector<double> density(x_size, 0.0);

#pragma omp parallel for schedule(dynamic)
  for (size_t xi = 0; xi < x_size; ++xi) {
    const double rho_local = eval_density_at_x(n, X[xi], v_eval, grid_struct,
                                               phi_history, is_electron);
    density[xi] = rho_local * v_cell_volume;
  }

  return density;
}

template <size_t X_DIM, size_t V_DIM>
std::vector<double> NuFISolver<X_DIM, V_DIM>::eval_rho(
    unsigned int n, const std::vector<array<double, X_DIM>> &X,
    const std::vector<GridStructure<X_DIM>> &grid_struct,
    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
    const array<size_t, V_DIM> &Nv) const {

  std::vector<double> n_electron = eval_species_density(
      n, X, grid_struct, phi_history, Nv, /*is_electron=*/true);

  if (!Parameters::IONS_ENABLED) {
    std::vector<double> rho(X.size());
    for (size_t i = 0; i < X.size(); ++i)
      rho[i] = 1.0 - n_electron[i];
    return rho;
  }

  std::vector<double> n_ion =
      eval_species_density(n, X, grid_struct, phi_history, Parameters::NV_ION,
                           /*is_electron=*/false);

  std::vector<double> rho(X.size());
  for (size_t i = 0; i < X.size(); ++i)
    rho[i] = n_ion[i] - n_electron[i];

  return rho;
}

template <size_t X_DIM, size_t V_DIM>
std::vector<double> NuFISolver<X_DIM, V_DIM>::eval_rho_points(
    unsigned int n, const std::vector<Point<X_DIM>> &points,
    const std::vector<GridStructure<X_DIM>> &grid_struct,
    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
    const array<size_t, V_DIM> &Nv) const {
  std::vector<array<double, X_DIM>> X =
      Point_vector_to_double_vector<X_DIM>(points);
  return eval_rho(n, X, grid_struct, phi_history, Nv);
}

template class NuFISolver<Parameters::X_DIM, Parameters::V_DIM>;
