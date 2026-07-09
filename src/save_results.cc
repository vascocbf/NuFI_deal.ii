#include "nufi/save_results.h"

#include "nufi/fields.h"
#include "nufi/grids.h"
#include "nufi/nufi_solver.h"
#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"
#include <cstddef>
#include <deal.II/numerics/solution_transfer.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void save_f(const NuFISolver &solver, unsigned int n,
            std::vector<GridStructure<1>> &grid_struct,
            std::vector<SolutionSnapshot<1>> &phi_history, unsigned int Nx_out,
            unsigned int Nv_out, const std::string &filename) {
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;

  double vmin = Parameters::V_DOMAIN_LEFT;
  double vmax = Parameters::V_DOMAIN_RIGHT;

  double dv = (vmax - vmin) / Nv_out;

  file << Nx_out << " " << Nv_out << "\n";
  file << xmin << " " << xmax << "\n";
  file << vmin << " " << vmax << "\n";

  std::vector<double> x_eval = make_x_eval(Nx_out);
  std::vector<double> val(Nx_out);

  for (unsigned int j = 0; j < Nv_out; ++j) {
    double v = vmin + (j + 0.5) * dv;
    val = solver.eval_f(n, x_eval, v, grid_struct, phi_history);

    for (unsigned int i = 0; i < Nx_out; ++i) {
      file << val[i];

      if (i < Nx_out - 1)
        file << " ";
    }

    file << "\n";
  }

  file.close();
}

void save_rho(const NuFISolver &solver, unsigned int n,
              std::vector<GridStructure<1>> &grid_struct,
              std::vector<SolutionSnapshot<1>> &phi_history,
              unsigned int Nx_out, const std::string &filename) {
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;

  std::vector<double> x_eval = make_x_eval(Nx_out);
  file << Nx_out << "\n";
  file << xmin << " " << xmax << "\n";

  std::vector<double> tmp =
      solver.eval_rho(n, x_eval, grid_struct, phi_history);
  for (size_t i = 0; i < Nx_out; ++i) {
    file << tmp[i];
    file << "\n";
  }
  file.close();
}

void save_Efield([[maybe_unused]] unsigned int n, GridStructure<1> &grid_struct,
                 std::vector<SolutionSnapshot<1>> &phi_history,
                 unsigned int Nx_out, const std::string &filename) {
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;

  std::vector<double> x_eval = make_x_eval(Nx_out);

  // select from E_coeffs

  file << Nx_out << "\n";
  file << xmin << " " << xmax << "\n";

  std::vector<double> tmp = eval(x_eval, grid_struct, phi_history[n].solution);
  for (size_t i = 0; i < Nx_out; ++i) {
    file << -tmp[i];
    file << "\n";
  }
  file.close();
}

void save_space_vector(const std::vector<double> &vals,
                       const std::string &filename, size_t it) {
  std::ofstream file("results/" + filename + "_" + std::to_string(it) + ".dat");

  if (!file)
    throw std::runtime_error("failed to start file in results/");

  file << vals.size() << "\n";
  file << Parameters::X_DOMAIN_LEFT << " " << Parameters::X_DOMAIN_RIGHT
       << "\n";
  file << std::fixed << std::setprecision(8);
  for (double val : vals)
    file << val << "\n";
}
