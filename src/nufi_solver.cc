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
#include "nufi/poisson_problem.h"
#include "nufi/save_results.h"
#include "nufi/stopwatch.h"

using namespace dealii;

std::vector<double> NuFISolver::eval_ftilda(
    unsigned int n, std::vector<double> X, double u,
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

  // We omit the initial half-step.
  while (--n) {
    for (size_t i = 0; i < x_size; ++i)
      X[i] = X[i] - Parameters::DT * U[i];

    AssertThrow(
        phi_history[n].solution.size() ==
            grid_struct[phi_history[n].grid_version].dof_handler->n_dofs(),
        ExcMessage("In eval_ftilda: Solution size = " +
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

std::vector<double>
NuFISolver::eval_rho(unsigned int n, std::vector<double> &X,
                     const std::vector<GridStructure<1>> &grid_struct,
                     const std::vector<SolutionSnapshot<1>> &phi_history,
                     const unsigned int Nv) const {
  size_t x_size = X.size();

  const double dv =
      (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;
  const double v_min = Parameters::V_DOMAIN_LEFT + 0.5 * dv;

  std::vector<double> partial(static_cast<size_t>(Nv) * x_size);

#pragma omp parallel for
  for (unsigned int i = 0; i < Nv; ++i) {
    std::vector<double> tmp_int =
        eval_ftilda(n, X, v_min + i * dv, grid_struct,
                    phi_history); // used eval_ftilda once per i
    std::copy(tmp_int.begin(), tmp_int.end(),
              partial.begin() + static_cast<size_t>(i) * x_size);
  }

  std::vector<double> integral(x_size, 0.0);
  for (unsigned int i = 0; i < Nv; ++i)
    for (size_t ii = 0; ii < x_size; ++ii)
      integral[ii] += partial[static_cast<size_t>(i) * x_size + ii];

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

void NuFISolver::run() {

  //====//====//
  // Run prep //
  //====//====//

  using std::abs;
  using std::max;

  std::vector<double> int_E_squared;
  int_E_squared.reserve(Nt);
  std::vector<double> int_E_squared_times;
  int_E_squared_times.reserve(Nt);

  std::vector<GridStructure<1>> grid_versions;
  std::vector<SolutionSnapshot<1>> phi_history;

  std::vector<double> x_eval(Parameters::CALC_NX);

  std::ofstream time_file(Parameters::PLOT_DIR + "simulation_time.dat");

  double total_time = 0;
  stopwatch<double> total_timer;

  time_file << "it "
            << "step_time "
            << "total_time "
            << "compute_time "
            << "refine_time "
            << "plot_time"
            << "\n";

  std::ofstream error_file(Parameters::PLOT_DIR + "error_estimate.dat");
  error_file << "# nufi Kelly l2 error estimate\n";
  error_file << "# it l2_error_estimate\n";

  [[maybe_unused]] const double x_min = Parameters::X_DOMAIN_LEFT;
  [[maybe_unused]] double dx = Parameters::CALC_DX;

  //====//====//
  // Time loop//
  //====//====//

  for (unsigned int it = 0; it < Nt; ++it) {
    stopwatch<double> timer;

    double time_elapsed_before = timer.elapsed();
    double compute_time = 0.0;
    double refine_time = 0.0;
    double plot_time = 0.0;

    std::cout << "Timestep " << it << " / " << Nt
              << " (simulation time = " << it * Parameters::DT << ")"
              << "\n";

    // START: diagnostics
    std::cout << "cells = " << poisson.get_triangulation().n_active_cells()
              << "\n"
              << " dofs = " << poisson.get_dof_handler().n_dofs() << "\n";
    // double min_h = 1e100;
    // double max_h = 0;
    //
    // for (auto cell : poisson.triangulation.active_cell_iterators()) {
    //   min_h = std::min(min_h, cell->diameter());
    //   max_h = std::max(max_h, cell->diameter());
    // }
    //
    // std::cout << "h ratio = " << max_h / min_h << std::endl;
    // std::vector<double> x = make_x_eval(Parameters::CALC_NX);
    // auto rho = eval_rho(it, x, grid_versions, phi_history, Parameters::NV);
    //
    // double mean = 0;
    //
    // for (auto r : rho)
    //   mean += r;
    //
    // mean /= rho.size();
    //
    // std::cout << "rho mean = " << mean << "\n";
    // std::cout << "rho min = " << *std::min_element(rho.begin(), rho.end())
    //           << "\n";
    // std::cout << "rho max = " << *std::max_element(rho.begin(), rho.end())
    //           << "\n";
    // END: diagnostics

    double compute_start = timer.elapsed();
    // compute rho
    //
    poisson.set_rhs_function([&](const std::vector<Point<1>> &points) {
      std::vector<double> x(points.size());

      for (size_t i = 0; i < points.size(); ++i)
        x[i] = points[i][0];

      return eval_rho(it, x, grid_versions, phi_history, Parameters::NV);
    });

    if (it % Parameters::REFINE_FREQUENCY == 0) {
      // if (it == 0) {
      refine_time = poisson.solve_step(it, grid_versions, true);
      compute_time = timer.elapsed() - compute_start - refine_time;
    } else {
      poisson.solve_step(it, grid_versions, false);
      compute_time = timer.elapsed() - compute_start;
    }
    update_solution_history(phi_history, poisson,
                            grid_versions.back().grid_version);

    error_file << it << " " << poisson.get_error_estimate() << "\n";
    error_file.flush();

    double timer_elapsed = timer.elapsed();
    double step_time = timer_elapsed - time_elapsed_before;

    std::cout << "step made in " << step_time << " seconds\n\n";

    //====//====//
    // Plotting //
    //====//====//

    if (it % Parameters::PLOT_FREQUENCY == 0) {
      double plot_start = timer.elapsed();

      std::cout << "Saving results...   ";
      DiagnosticsSnapshot snap =
          compute_diagnostics(*this, it, grid_versions, phi_history,
                              Parameters::PLOT_NX, Parameters::NV);

      save_f(snap, Parameters::PLOT_DIR + "f_" + std::to_string(it) + ".dat");
      save_rho(snap,
               Parameters::PLOT_DIR + "rho_" + std::to_string(it) + ".dat");
      save_Efield(snap,
                  Parameters::PLOT_DIR + "E_" + std::to_string(it) + ".dat");

      int_E_squared.push_back(compute_int_E_squared(snap));
      int_E_squared_times.push_back(it * Parameters::DT);
      save_time_series(int_E_squared_times, int_E_squared,
                       Parameters::PLOT_DIR + "int_E_sqr.dat");

      plot_time = timer.elapsed() - plot_start;
      std::cout << "Results saved in " << plot_start << "[s]" << "\n";
    }

    total_time = total_timer.elapsed();
    std::cout << "Time since start = " << total_time << "\n\n";

    time_file << it << " " << step_time << " " << total_time << " "
              << compute_time << " " << refine_time << " " << plot_time << "\n";
    time_file.flush();
  }

  std::cout << "NuFI simulation finished in " << total_time << " seconds.\n";
}

NuFISolver::NuFISolver() : order(Parameters::FE_DEGREE), poisson(order) {
  std::cout << "Initializing dealii Poisson Solver\n";
  poisson.initialize();
}
