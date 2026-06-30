#include "nufi/save_results.h"

#include "nufi/fields.h"
#include "nufi/nufi_solver.h"
#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void save_f(const NuFISolver &solver, unsigned int n,
            const PoissonProblem<1> &poisson,
            const std::vector<Vector<double>> &phi_history, unsigned int Nx_out,
            unsigned int Nv_out, const std::string &filename) {
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;

  double vmin = Parameters::V_DOMAIN_LEFT;
  double vmax = Parameters::V_DOMAIN_RIGHT;

  double dx = (xmax - xmin) / Nx_out;
  double dv = (vmax - vmin) / Nv_out;

  file << Nx_out << " " << Nv_out << "\n";
  file << xmin << " " << xmax << "\n";
  file << vmin << " " << vmax << "\n";

  for (unsigned int i = 0; i < Nx_out; ++i) {
    double x = xmin + (i + 0.5) * dx;

    for (unsigned int j = 0; j < Nv_out; ++j) {
      double v = vmin + (j + 0.5) * dv;

      double val = solver.eval_f(n, x, v, poisson, phi_history);

      file << val;

      if (j < Nv_out - 1)
        file << " ";
    }

    file << "\n";
  }

  file.close();
}

void save_rho(const NuFISolver &solver, unsigned int n,
              const PoissonProblem<1> &poisson,
              const std::vector<Vector<double>> &phi_history,
              unsigned int Nx_out, const std::string &filename) {
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;
  double dx = (xmax - xmin) / Nx_out;

  file << Nx_out << "\n";
  file << xmin << " " << xmax << "\n";

  for (unsigned int i = 0; i < Nx_out; ++i, xmin += dx) {
    double val = solver.eval_rho(n, xmin, poisson, phi_history);
    file << val;
    file << "\n";
  }
  file.close();
}

void save_Efield([[maybe_unused]] unsigned int n,
                 const PoissonProblem<1> &poisson,
                 const std::vector<Vector<double>> &phi_history,
                 unsigned int Nx_out, const std::string &filename) {
  std::ofstream file(filename);

  double xmin = Parameters::X_DOMAIN_LEFT;
  double xmax = Parameters::X_DOMAIN_RIGHT;
  double dx = (xmax - xmin) / Nx_out;

  // select from E_coeffs

  file << Nx_out << "\n";
  file << xmin << " " << xmax << "\n";

  for (unsigned int i = 0; i < Nx_out; ++i, xmin += dx) {
    double val = -eval(xmin, poisson, phi_history[n]);
    file << val;
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
