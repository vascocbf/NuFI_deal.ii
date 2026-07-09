#include "nufi/nufi_solver.h"

#include <algorithm>
#include <boost/qvm/mat_access.hpp>
#include <cmath>
#include <cstdlib>
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
#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"
#include "nufi/save_results.h"
#include "nufi/stopwatch.h"

using namespace dealii;

std::vector<double>
NuFISolver::eval_ftilda(unsigned int n, std::vector<double> &X, double u,
                        const PoissonProblem<1> &poisson,
                        const std::vector<Vector<double>> &phi_history) const {

  size_t x_size = X.size();

  std::vector<double> U(x_size, u);
  std::vector<double> results(x_size);
  if (n == 0) {
    for (size_t i = 0; i < x_size; ++i)
      results[i] = f0(X[i], U[i]);
    reset_x_eval(X);
    return results;
  }

  std::vector<double> Ex(x_size);
  std::vector<double> tmp(x_size);

  // We omit the initial half-step.
  while (--n) {
    for (size_t i = 0; i < x_size; ++i)
      X[i] = X[i] - Parameters::DT * U[i];

    tmp = eval(X, poisson, phi_history[n]); // call eval only once

    for (size_t i = 0; i < x_size; ++i) {
      Ex[i] = -tmp[i];
      U[i] = U[i] + Parameters::DT * Ex[i];
    }
  }

  // The final half-step.
  for (size_t i = 0; i < x_size; ++i)
    X[i] = X[i] - Parameters::DT * U[i];

  tmp = eval(X, poisson, phi_history[n]); // call eval only once

  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U[i] = U[i] + 0.5 * Parameters::DT * Ex[i];
  }
  for (size_t i = 0; i < x_size; ++i)
    results[i] = f0(X[i], U[i]);
  reset_x_eval(X);
  return results;
}

std::vector<double>
NuFISolver::eval_f(unsigned int n, std::vector<double> &X, double u,
                   const PoissonProblem<1> &poisson,
                   const std::vector<Vector<double>> &phi_history) const {

  size_t x_size = X.size();

  std::vector<double> U(x_size, u);
  std::vector<double> results(x_size);
  if (n == 0) {
    for (size_t i = 0; i < x_size; ++i)
      results[i] = f0(X[i], U[i]);
    reset_x_eval(X);
    return results;
  }

  std::vector<double> Ex(x_size);

  std::vector<double> tmp(x_size);
  // Initial half-step.
  tmp = eval(X, poisson, phi_history[n]); // call eval only once
  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U[i] = U[i] + 0.5 * Parameters::DT * Ex[i];
  }

  while (--n) {
    for (size_t i = 0; i < x_size; ++i)
      X[i] = X[i] - Parameters::DT * U[i];

    tmp = eval(X, poisson, phi_history[n]); // call eval only once
    for (size_t i = 0; i < x_size; ++i) {
      Ex[i] = -tmp[i];
      U[i] = U[i] + Parameters::DT * Ex[i];
    }
  }

  // The final half-step.
  for (size_t i = 0; i < x_size; ++i)
    X[i] = X[i] - Parameters::DT * U[i];

  tmp = eval(X, poisson, phi_history[n]); // call eval only once
  for (size_t i = 0; i < x_size; ++i) {
    Ex[i] = -tmp[i];
    U[i] = U[i] + 0.5 * Parameters::DT * Ex[i];
  }

  for (size_t i = 0; i < x_size; ++i)
    results[i] = f0(X[i], U[i]);
  reset_x_eval(X);
  return results;
}

std::vector<double>
NuFISolver::eval_rho(unsigned int n, std::vector<double> &X,
                     const PoissonProblem<1> &poisson,
                     const std::vector<Vector<double>> &phi_history,
                     const unsigned int Nv) const {
  size_t x_size = X.size();
  const double dv =
      (Parameters::V_DOMAIN_RIGHT - Parameters::V_DOMAIN_LEFT) / Nv;
  const double v_min = Parameters::V_DOMAIN_LEFT + 0.5 * dv;

  std::vector<double> integral(x_size, 0.0);

  std::vector<double> tmp_int(x_size);

  for (unsigned int i = 0; i < Nv; ++i) {
    tmp_int = eval_ftilda(n, X, v_min + i * dv, poisson,
                          phi_history); // used eval_ftilda once per i
    for (size_t ii = 0; ii < x_size; ++ii)
      integral[ii] += tmp_int[ii];
  }

  for (size_t i = 0; i < x_size; ++i)
    integral[i] = 1 - integral[i] * dv;
  return integral;
}

void NuFISolver::run() {

  //====//====//
  // Run prep //
  //====//====//

  using std::abs;
  using std::max;

  std::unique_ptr<double, decltype(std::free) *> rho{
      reinterpret_cast<double *>(std::aligned_alloc(64, sizeof(double) * Nx)),
      std::free};

  std::vector<double> int_E_squared;
  int_E_squared.reserve(Nt);

  std::vector<Vector<double>> phi_history;

  std::vector<double> x_eval(Parameters::CALC_NX);

  if (rho == nullptr)
    throw std::bad_alloc{};

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
    std::cout << "cells = " << poisson.triangulation.n_active_cells() << "\n"
              << " dofs = " << poisson.dof_handler.n_dofs() << "\n";
    // double min_h = 1e100;
    // double max_h = 0;
    //
    // for (auto cell : poisson.triangulation.active_cell_iterators()) {
    //   min_h = std::min(min_h, cell->diameter());
    //   max_h = std::max(max_h, cell->diameter());
    // }
    //
    // std::cout << "h ratio = " << max_h / min_h << std::endl;

    // END: diagnostics

    double compute_start = timer.elapsed();
    // compute rho
    std::vector<double> x_eval = make_x_eval(poisson.get_dof_size());
    std::vector<double> rho_values =
        eval_rho(it, x_eval, poisson, phi_history, Parameters::NV);

    Vector<double> rhs(x_eval.size());
    for (unsigned int i = 0; i < rhs.size(); ++i)
      rhs[i] = rho_values[i];

    poisson.set_rhs(rhs);

    poisson.solve_step();
    phi_history.push_back(poisson.get_solution());

    compute_time = timer.elapsed() - compute_start;

    if (it % Parameters::REFINE_FREQUENCY == 0 && it != 0) {
      double refine_start = timer.elapsed();
      poisson.coarse_and_refine_grid(it, phi_history);
      refine_time = timer.elapsed() - refine_start;
      std::cout << "Refinement step done in "
                << std::to_string(std::round(std::floor(refine_time))) << "[s]"
                << "\n";
    }

    double timer_elapsed = timer.elapsed();
    double step_time = timer_elapsed - time_elapsed_before;

    std::cout << "step made in " << step_time << " seconds\n\n";
    if (it % Parameters::PLOT_FREQUENCY == 0) {
      double plot_start = timer.elapsed();
      std::cout << "Saving results...   ";
      save_f(*this, it, poisson, phi_history, Parameters::PLOT_NX,
             Parameters::NV, "results/ftilda_" + std::to_string(it) + ".dat");
      save_rho(*this, it, poisson, phi_history, Parameters::PLOT_NX,
               "results/rho_" + std::to_string(it) + ".dat");
      // save_Efield(it, coeffs.get(), 128, "results/field_" +
      // std::to_string(it) + ".dat");

      std::vector<double> x_eval_Ex = make_x_eval(Parameters::PLOT_NX);
      std::vector<double> tmp_rho(x_eval_Ex.size());
      tmp_rho = eval(x_eval_Ex, poisson, phi_history[it]);

      std::vector<double> E_x(Parameters::PLOT_NX);
      for (size_t i = 0; i < Parameters::PLOT_NX; ++i)
        E_x[i] = -tmp_rho[i];
      save_space_vector(E_x, "field", it);

      double int_val =
          0.5 * integral_space_vector_squared(poisson, phi_history[it]);
      int_E_squared.push_back(int_val);
      save_space_vector(int_E_squared, "electricint", it);
      std::cout << "Time since start = " << total_time << "\n\n";

      plot_time = timer.elapsed() - plot_start;
    }

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
