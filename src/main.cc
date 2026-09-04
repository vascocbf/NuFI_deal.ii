#include "nufi/stopwatch.h"
#include <array>
#include <filesystem>
#include <iostream>
#include <nufi/fields.h>
#include <nufi/nufi_solver.h>
#include <nufi/parameters.h>
#include <nufi/poisson_problem.h>
#include <nufi/save_results.h>
#include <omp.h>

using std::array;

template <size_t X_DIM, size_t V_DIM> void run() {
  unsigned int Nt = std::floor(Parameters::TMAX / Parameters::DT);

  array<size_t, V_DIM> Nv = {};
  for (size_t d = 0; d < V_DIM; ++d)
    Nv[d] = static_cast<size_t>(Parameters::NV[d]);

  PoissonProblem<X_DIM> poisson(Parameters::FE_DEGREE,
                                /*mapping_dregree = */ 1);
  NuFISolver<X_DIM, V_DIM> solver;
  using std::abs;
  using std::max;

  std::cout << "[run] Initializing dealii Poisson Solver\n";
  poisson.initialize();

  std::vector<double> int_E_squared;
  int_E_squared.reserve(Nt);
  std::vector<double> int_E_squared_times;
  int_E_squared_times.reserve(Nt);

  std::vector<GridStructure<X_DIM>> grid_versions;
  std::vector<SolutionSnapshot<X_DIM>> phi_history;

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

    std::cout << "[run: for(it<NT)] Timestep: " << it << " / " << Nt
              << " (simulation time = " << it * Parameters::DT << ")"
              << "\n";

    // START: diagnostics
    std::cout << "[run: for(it<NT)] cells = "
              << poisson.get_triangulation().n_active_cells() << "\n"
              << "[run: for(it<NT)] dofs = "
              << poisson.get_dof_handler().n_dofs() << "\n";
    // END: diagnostics

    double compute_start = timer.elapsed();

    // compute rho
    poisson.set_rhs_function([&](const std::vector<Point<X_DIM>> &points) {
      std::vector<array<double, X_DIM>> X =
          Point_vector_to_double_vector<X_DIM>(points);
      return solver.eval_rho(it, X, grid_versions, phi_history, Nv);
    });

    if (it == 0 ||
        (it % Parameters::REFINE_FREQUENCY == 0 &&
         poisson.get_dof_handler().n_dofs() < Parameters::MAX_DOFS)) {
      refine_time =
          poisson.solve_step(it, grid_versions, true); // Refine and Solve
      compute_time = timer.elapsed() - compute_start - refine_time;
    } else {
      poisson.solve_step(it, grid_versions, false); // Dont refine and Solve
      compute_time = timer.elapsed() - compute_start;
    }
    update_solution_history(phi_history, poisson,
                            grid_versions.back().grid_version);

    error_file << it << " " << poisson.get_error_estimate() << "\n";
    error_file.flush();

    if (Parameters::SAVE_INT_E_ALWAYS) {
      // Save |E|^2; cheap because uses just calculated solution
      // (does not to the entire backward characteristics)
      const double int_E_sq_now =
          0.5 * integral_space_vector_squared<X_DIM>(
                    grid_versions[phi_history.back().grid_version],
                    phi_history.back().solution);

      int_E_squared.push_back(int_E_sq_now);
      int_E_squared_times.push_back(it * Parameters::DT);
    }

    double timer_elapsed = timer.elapsed();
    double step_time = timer_elapsed - time_elapsed_before;

    std::cout << "[run: for(it<Nt)] step made in " << step_time
              << " seconds\n\n";

    if (it % Parameters::PLOT_FREQUENCY == 0) {
      double plot_start = timer.elapsed();

      std::cout << "Saving results...   ";
      // SAVE FULL SPACE
      //
      if (X_DIM == 1 && V_DIM == 1) {
        DiagnosticsSnapshot<X_DIM, V_DIM> snap =
            compute_diagnostics<X_DIM, V_DIM>(solver, it, grid_versions,
                                              phi_history, Parameters::PLOT_NX,
                                              Nv);

        save_f(snap, Parameters::PLOT_DIR + "f_" + std::to_string(it) + ".dat");
        save_rho(snap,
                 Parameters::PLOT_DIR + "rho_" + std::to_string(it) + ".dat");
        save_Efield(snap,
                    Parameters::PLOT_DIR + "E_" + std::to_string(it) + ".dat");

        if (!Parameters::SAVE_INT_E_ALWAYS) {
          // OLD; Not needed because this is done above
          int_E_squared.push_back(compute_int_E_squared(snap));
          int_E_squared_times.push_back(it * Parameters::DT);
        }
      } else {
        // SAVE SLICE EXAMPLE
        // fixed coordinate to make slice, free_x_dim direction ignored

        array<double, X_DIM> x_fixed = {};
        array<double, V_DIM> v_fixed = {};
        size_t free_x_dim = 0;
        size_t free_v_dim = 0;

        auto slice =
            compute_diagnostics_slice<Parameters::X_DIM, Parameters::V_DIM>(
                solver, it, grid_versions, phi_history, free_x_dim, free_v_dim,
                x_fixed, v_fixed, Parameters::PLOT_NX[free_x_dim],
                Parameters::NV[free_v_dim]);

        save_f_slice(slice, Parameters::PLOT_DIR + "f_slice_" +
                                std::to_string(it) + ".dat");
        save_rho_slice(slice, Parameters::PLOT_DIR + "rho_slice_" +
                                  std::to_string(it) + ".dat");
        save_Efield_slice(slice, Parameters::PLOT_DIR + "E_slice_" +
                                     std::to_string(it) + ".dat");
      }
      save_time_series(int_E_squared_times, int_E_squared,
                       Parameters::PLOT_DIR + "int_E_sqr.dat");

      plot_time = timer.elapsed() - plot_start;
      std::cout << "Results saved in " << plot_start << "[s]" << "\n";
    }

    total_time = total_timer.elapsed();
    std::cout << "[run: for(it<Nt)] Time since start = " << total_time
              << "\n\n";

    time_file << it << " " << step_time << " " << total_time << " "
              << compute_time << " " << refine_time << " " << plot_time << "\n";
    time_file.flush();
  }

  std::cout << "NuFI simulation finished in " << total_time << " seconds.\n";
};

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

int main() {
  omp_set_max_active_levels(1);
  std::cout << "[main] Threads: " << omp_get_max_threads() << "\n";
  try {

    std::cout << "[main] Loading parameters from lua config" << "\n";
    Parameters::load_lua_config("parameters.lua");

    clear_results_directory("results");
    std::filesystem::copy_file(
        "parameters.lua", "results/parameters.lua",
        std::filesystem::copy_options::overwrite_existing);
    run<Parameters::X_DIM, Parameters::V_DIM>();

  } catch (const std::exception &exc) {
    std::cerr << "\nException:\n" << exc.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "\nUnknown exception!\n";
    return 1;
  }

  return 0;
}
