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
    unsigned int n, std::vector<double> X, double u1, double u2,
    const std::vector<GridStructure<1>> &grid_struct,
    const std::vector<SolutionSnapshot<1>> &phi_history) const {

  size_t x_size = X.size();

  std::vector<double> U1(x_size, u1);
  std::vector<double> U2(x_size, u2);
  std::vector<double> results(x_size);
  if (n == 0) {
    for (size_t i = 0; i < x_size; ++i)
      results[i] = f0(X[i], U1[i], U2[i]);
    return results;
  }

  std::vector<double> Ex(x_size);
  std::vector<double> tmp(x_size);

  // We omit the initial half-step.
  while (--n) {
    for (size_t i = 0; i < x_size; ++i)
      X[i] = X[i] - Parameters::DT * U1[i];

    tmp = eval(X, grid_struct[phi_history[n].grid_version],
               phi_history[n].solution); // call eval only once

    for (size_t i = 0; i < x_size; ++i) {
      Ex[i] = -tmp[i];
      U1[i] = U1[i] + Parameters::DT * Ex[i];

      // IMPORTANT!! LOOK at this before 1x2v, and check with Paul
      //
      // No change to U2 because dv1/dt = E_x
      // but dv2/dt = E_y = 0 (because were in 1x not 2x)
      // U2[i] = U2[i] + Parameters::DT * Ey[i]; TO_BUILD
    }
  }

  // The final half-step.
  for (size_t i = 0; i < x_size; ++i)
    X[i] = X[i] - Parameters::DT * U1[i];

  tmp = eval(X, grid_struct[phi_history[n].grid_version],
             phi_history[n].solution); // call eval only once

  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U1[i] = U1[i] + 0.5 * Parameters::DT * Ex[i];
  }
  for (size_t i = 0; i < x_size; ++i)
    results[i] = f0(X[i], U1[i], U2[i]);
  return results;
}

std::vector<double>
NuFISolver::eval_f(unsigned int n, std::vector<double> X, double u1, double u2,
                   const std::vector<GridStructure<1>> &grid_struct,
                   const std::vector<SolutionSnapshot<1>> &phi_history) const {

  size_t x_size = X.size();

  std::vector<double> U1(x_size, u1);
  std::vector<double> U2(x_size, u2);

  std::vector<double> results(x_size);
  if (n == 0) {
    for (size_t i = 0; i < x_size; ++i)
      results[i] = f0(X[i], U1[i], U2[i]);
    return results;
  }

  std::vector<double> Ex(x_size);

  std::vector<double> tmp(x_size);
  // Initial half-step.
  tmp = eval(X, grid_struct[phi_history[n].grid_version],
             phi_history[n].solution); // call eval only once
  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U1[i] = U1[i] + 0.5 * Parameters::DT * Ex[i];
  }

  while (--n) {
    for (size_t i = 0; i < x_size; ++i)
      X[i] = X[i] - Parameters::DT * U1[i];

    tmp = eval(X, grid_struct[phi_history[n].grid_version],
               phi_history[n].solution); // call eval only once
    for (size_t i = 0; i < x_size; ++i) {
      Ex[i] = -tmp[i];
      U1[i] = U1[i] + Parameters::DT * Ex[i];
    }
  }

  // The final half-step.
  for (size_t i = 0; i < x_size; ++i)
    X[i] = X[i] - Parameters::DT * U1[i];

  tmp = eval(X, grid_struct[phi_history[n].grid_version],
             phi_history[n].solution); // call eval only once
  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U1[i] = U1[i] + 0.5 * Parameters::DT * Ex[i];
  }

  for (size_t i = 0; i < x_size; ++i)
    results[i] = f0(X[i], U1[i], U2[i]);
  return results;
}

std::vector<double>
NuFISolver::eval_rho(unsigned int n, std::vector<double> &X,
                     const std::vector<GridStructure<1>> &grid_struct,
                     const std::vector<SolutionSnapshot<1>> &phi_history,
                     const unsigned int Nv1, const unsigned int Nv2) const {
  size_t x_size = X.size();

  const double dv_0 = Parameters::DV_1;
  const double v_min_0 = Parameters::V_DOMAIN_LEFT_1 + 0.5 * dv_0;

  const double dv_1 = Parameters::DV_2;
  const double v_min_1 = Parameters::V_DOMAIN_LEFT_2 + 0.5 * dv_1;

  std::vector<double> integral(x_size, 0.0);
  std::vector<double> tmp_int(x_size);

  for (unsigned int i = 0; i < Nv1; ++i) {
    for (unsigned int j = 0; j < Nv2; ++j) {
      tmp_int =
          eval_ftilda(n, X, v_min_0 + i * dv_0, v_min_1 + j * dv_1, grid_struct,
                      phi_history); // used eval_ftilda once per i
      for (size_t ii = 0; ii < x_size; ++ii)
        integral[ii] += tmp_int[ii];
    }
  }

  for (size_t i = 0; i < x_size; ++i)
    integral[i] = 1 - integral[i] * dv_0;
  return integral;
}

std::vector<double>
NuFISolver::eval_rho_points(unsigned int n, const std::vector<Point<1>> &points,
                            const std::vector<GridStructure<1>> &grid_struct,
                            const std::vector<SolutionSnapshot<1>> &phi_history,
                            const unsigned int Nv1,
                            const unsigned int Nv2) const {
  std::vector<double> point_vector = Point_vector_to_double_vector(points);
  return NuFISolver::eval_rho(n, point_vector, grid_struct, phi_history, Nv1,
                              Nv2);
}

void NuFISolver::run() {

  //====//====//
  // Run prep //
  //====//====//

  using std::abs;
  using std::max;

  // std::unique_ptr<double, decltype(std::free) *> rho{
  //     reinterpret_cast<double *>(std::aligned_alloc(64, sizeof(double) *
  //     Nx)),
  // std::free};
  //
  // if (rho == nullptr)
  //   throw std::bad_alloc{};

  std::vector<double> int_E_squared;
  int_E_squared.reserve(Nt);

  std::vector<GridStructure<1>> grid_versions;
  std::vector<SolutionSnapshot<1>> phi_history;

  // update_grid_versions(grid_versions, poisson);
  // update_solution_history(phi_history, poisson,
  //                         grid_versions.back().grid_version);

  std::vector<double> x_eval(Parameters::CALC_NX);

  std::ofstream time_file("results/simulation_time.dat");

  double total_time = 0;

  time_file << "it "
            << "step_time "
            << "total_time "
            << "compute_time "
            << "refine_time "
            << "plot_time"
            << "\n";

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
              << std::endl;

    // START: diagnostics
    // std::cout << "cells = " << poisson.triangulation.n_active_cells() << "\n"
    //           << " dofs = " << poisson.dof_handler.n_dofs() << "\n";
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

      return eval_rho(it, x, grid_versions, phi_history, Parameters::NV_1,
                      Parameters::NV_2);
    });

    if (it % Parameters::REFINE_FREQUENCY == 0) {
      // if (it == 0) {
      poisson.solve_step(it, grid_versions, true);
      compute_time = timer.elapsed() - compute_start;
    } else {
      poisson.solve_step(it, grid_versions, false);
      compute_time = timer.elapsed() - compute_start;
    }
    update_solution_history(phi_history, poisson,
                            grid_versions.back().grid_version);

    double timer_elapsed = timer.elapsed();
    double step_time = timer_elapsed - time_elapsed_before;

    std::cout << "step made in " << step_time << " seconds\n\n";

    //====//====//
    // Plotting //
    //====//====//
    if (it % Parameters::PLOT_FREQUENCY == 0) {
      double plot_start = timer.elapsed();

      std::cout << "Saving results...   ";
      save_f(*this, it, Parameters::PLOT_FIXED_V2, grid_versions, phi_history,
             Parameters::PLOT_NX, Parameters::NV_1,
             "results/ftilda_" + std::to_string(it) + ".dat");
      save_rho(*this, it, grid_versions, phi_history, Parameters::PLOT_NX,
               "results/rho_" + std::to_string(it) + ".dat");

      save_Efield(it, grid_versions, phi_history);

      double int_val = 0.5 * integral_space_vector_squared(
                                 grid_versions[phi_history[it].grid_version],
                                 phi_history[it].solution);
      int_E_squared.push_back(int_val);
      save_space_vector(int_E_squared, "electricint", it);

      plot_time = timer.elapsed() - plot_start;
      std::cout << "Results saved in " << plot_start << "[s]" << "\n";
    }
    total_time = timer.elapsed();
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
