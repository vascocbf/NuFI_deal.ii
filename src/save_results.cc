#include "nufi/save_results.h"

#include "nufi/fields.h"
#include "nufi/grids.h"
#include "nufi/nufi_solver.h"
#include "nufi/parameters.h"
#include "nufi/poisson_problem.h"
#include <cstddef>
#include <deal.II/numerics/solution_transfer.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void ensure_results_dir() { std::filesystem::create_directories("results"); }
} // namespace

void save_field_1d(const std::vector<double> &x,
                   const std::vector<double> &values,
                   const std::string &filepath) {
  if (x.size() != values.size())
    throw std::runtime_error(
        "save_field_1d: size mismatch (x=" + std::to_string(x.size()) +
        ", values=" + std::to_string(values.size()) + ") writing " + filepath);

  ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_field_1d: failed to open " + filepath);

  file << "# nufi 1d field\n";
  file << "# n = " << x.size() << "\n";
  file << "# columns: x value\n";
  file << std::setprecision(10);
  for (size_t i = 0; i < x.size(); ++i)
    file << x[i] << " " << values[i] << "\n";
}

void save_time_series(const std::vector<double> &t,
                      const std::vector<double> &values,
                      const std::string &filepath) {
  if (t.size() != values.size())
    throw std::runtime_error(
        "save_time_series: size mismatch (t=" + std::to_string(t.size()) +
        ", values=" + std::to_string(values.size()) + ") writing " + filepath);

  ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_time_series: failed to open " + filepath);

  file << "# nufi time series\n";
  file << "# columns: t value\n";
  file << std::setprecision(10);
  for (size_t i = 0; i < t.size(); ++i)
    file << t[i] << " " << values[i] << "\n";
}

void save_f_binary(const NuFISolver &solver, unsigned int n,
                   std::vector<GridStructure<1>> &grid_struct,
                   std::vector<SolutionSnapshot<1>> &phi_history,
                   unsigned int Nx_out, unsigned int Nv_out,
                   const std::string &filepath) {
  ensure_results_dir();
  std::ofstream file(filepath, std::ios::binary);
  if (!file)
    throw std::runtime_error("save_f_binary: failed to open " + filepath);

  const double vmin = Parameters::V_DOMAIN_LEFT;
  const double vmax = Parameters::V_DOMAIN_RIGHT;
  const double dv = (vmax - vmin) / Nv_out;

  std::vector<double> x_eval = make_x_eval(Nx_out);
  std::vector<double> v_eval(Nv_out);
  for (unsigned int j = 0; j < Nv_out; ++j)
    v_eval[j] = vmin + (j + 0.5) * dv;

  // simple fixed binary layout: magic, Nx, Nv, x[], v[], f[Nv*Nx]
  const uint32_t magic = 0x4E554649; // "NUFI"
  const uint64_t nx64 = Nx_out, nv64 = Nv_out;

  file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  file.write(reinterpret_cast<const char *>(&nx64), sizeof(nx64));
  file.write(reinterpret_cast<const char *>(&nv64), sizeof(nv64));
  file.write(reinterpret_cast<const char *>(x_eval.data()),
             x_eval.size() * sizeof(double));
  file.write(reinterpret_cast<const char *>(v_eval.data()),
             v_eval.size() * sizeof(double));

  std::vector<double> val(Nx_out);
  for (unsigned int j = 0; j < Nv_out; ++j) {
    val = solver.eval_f(n, x_eval, v_eval[j], grid_struct, phi_history);
    file.write(reinterpret_cast<const char *>(val.data()),
               val.size() * sizeof(double));
  }
}

void save_rho(const NuFISolver &solver, unsigned int n,
              std::vector<GridStructure<1>> &grid_struct,
              std::vector<SolutionSnapshot<1>> &phi_history,
              unsigned int Nx_out, const std::string &filename) {
  std::vector<double> x_eval = make_x_eval(Nx_out);
  std::vector<double> rho =
      solver.eval_rho(n, x_eval, grid_struct, phi_history);
  save_field_1d(x_eval, rho, filename);
}

void save_Efield(unsigned int it, std::vector<GridStructure<1>> &grid_versions,
                 std::vector<SolutionSnapshot<1>> &phi_history,
                 unsigned int Nx_out) {
  std::vector<double> x_eval_E = make_x_eval(Nx_out);

  auto grad_phi = eval(x_eval_E, grid_versions[phi_history[it].grid_version],
                       phi_history[it].solution);

  std::vector<double> E(x_eval_E.size());
  for (size_t i = 0; i < x_eval_E.size(); ++i)
    E[i] = -grad_phi[i];

  save_field_1d(x_eval_E, E, "results/E_" + std::to_string(it) + ".dat");

  std::cout << "Saving iteration " << it << " using grid version "
            << phi_history[it].grid_version << "\n\n";
}
