#include <filesystem>
#include <iostream>
#include <nufi/nufi_solver.h>
#include <nufi/poisson_problem.h>
#include <nufi/save_results.h>
#include <omp.h>

void clear_results_directory(const std::string &dir) {
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir))
    return;

  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (std::filesystem::is_regular_file(entry)) {
      std::filesystem::remove(entry.path());
      std::cout << "Deleted: " << entry.path() << '\n';
    }
  }
};

template <int dim> void run() {
  unsigned int Nt = std::floor(Parameters::TMAX / Parameters::DT);

  PoissonProblem<dim> poisson(dim);
  NuFISolver solver;
  using std::abs;
  using std::max;

  std::cout << "Initializing dealii Poisson Solver\n";
  poisson.initialize();

  std::vector<double> int_E_squared;
  int_E_squared.reserve(Nt);
  std::vector<double> int_E_squared_times;
  int_E_squared_times.reserve(Nt);

  std::vector<GridStructure<1>> grid_versions;
  std::vector<SolutionSnapshot<1>> phi_history;

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
    // END: diagnostics

    double compute_start = timer.elapsed();
    // compute rho
    poisson.set_rhs_function([&](const std::vector<Point<1>> &points) {
      std::vector<double> x(points.size());

      for (size_t i = 0; i < points.size(); ++i)
        x[i] = points[i][0];

      return solver.eval_rho(it, x, grid_versions, phi_history, Parameters::NV);
    });

    if (it % Parameters::REFINE_FREQUENCY == 0) {
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

    if (it % Parameters::PLOT_FREQUENCY == 0) {
      double plot_start = timer.elapsed();

      std::cout << "Saving results...   ";
      DiagnosticsSnapshot snap =
          compute_diagnostics(solver, it, grid_versions, phi_history,
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
};

int main() {
  // omp_set_max_active_levels(1);
  std::cout << "Threads: " << omp_get_max_threads() << "\n";
  try {

    clear_results_directory("results");
    run<1>(); // 1 = space_dim

  } catch (const std::exception &exc) {
    std::cerr << "\nException:\n" << exc.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "\nUnknown exception!\n";
    return 1;
  }

  return 0;
}
