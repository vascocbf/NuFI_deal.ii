#include "nufi/save_results.h"

#include "nufi/fields.h"
#include "nufi/grids.h"
#include "nufi/nufi_solver.h"
#include "nufi/parameters.h"
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

RhoESnapshot
compute_rho_E_snapshot(const NuFISolver &solver, unsigned int n,
                       std::vector<GridStructure<1>> &grid_struct,
                       std::vector<SolutionSnapshot<1>> &phi_history,
                       unsigned int Nx_out, unsigned int Nv) {
  RhoESnapshot snap;
  snap.it = n;
  snap.time = n * Parameters::DT;
  snap.x_eval = make_x_eval(Nx_out);

  // rho at the plot grid (the expensive part — full characteristic trace)
  snap.rho = solver.eval_rho(n, snap.x_eval, grid_struct, phi_history, Nv);

  // E is just -grad(phi), cheap — reuse the already-solved phi for this step
  std::vector<double> x_copy = snap.x_eval;
  auto grad_phi = eval(x_copy, grid_struct[phi_history[n].grid_version],
                       phi_history[n].solution);
  snap.E.resize(Nx_out);
  for (unsigned int i = 0; i < Nx_out; ++i)
    snap.E[i] = -grad_phi[i];

  snap.int_E_sqr = compute_int_E_squared(snap.E, Parameters::LX);

  return snap;
}

void flush_rho_E_history(std::vector<RhoESnapshot> &history,
                         const std::string &dir) {
  ensure_results_dir();

  std::vector<double> t, int_E_sqr;
  t.reserve(history.size());
  int_E_sqr.reserve(history.size());

  for (const auto &snap : history) {
    save_field_1d(snap.x_eval, snap.rho,
                  dir + "rho_" + std::to_string(snap.it) + ".dat");
    save_field_1d(snap.x_eval, snap.E,
                  dir + "E_" + std::to_string(snap.it) + ".dat");
    t.push_back(snap.time);
    int_E_sqr.push_back(snap.int_E_sqr);
  }

  // Append rather than overwrite, since flush is called repeatedly.
  static bool wrote_header = false;
  std::ofstream file(dir + "int_E_sqr.dat",
                     wrote_header ? std::ios::app : std::ios::trunc);
  if (!wrote_header) {
    file << "# nufi time series\n";
    file << "# columns: t value\n";
    wrote_header = true;
  }
  file << std::setprecision(10);
  for (size_t i = 0; i < t.size(); ++i)
    file << t[i] << " " << int_E_sqr[i] << "\n";

  history.clear();
}

DiagnosticsSnapshot
compute_diagnostics(const NuFISolver &solver, unsigned int n,
                    std::vector<GridStructure<1>> &grid_struct,
                    std::vector<SolutionSnapshot<1>> &phi_history,
                    unsigned int Nx_out, unsigned int Nv_out) {
  DiagnosticsSnapshot snap;
  snap.Nx = Nx_out;
  snap.Nv = Nv_out;

  const double vmin = Parameters::V_DOMAIN_LEFT;
  const double vmax = Parameters::V_DOMAIN_RIGHT;
  const double dv = (vmax - vmin) / Nv_out;

  snap.x_eval = make_x_eval(Nx_out);
  snap.v_eval.resize(Nv_out);
  for (unsigned int j = 0; j < Nv_out; ++j)
    snap.v_eval[j] = vmin + (j + 0.5) * dv;

  snap.f.resize(static_cast<size_t>(Nx_out) * Nv_out);
#pragma omp parallel for
  for (unsigned int j = 0; j < Nv_out; ++j) {
    std::vector<double> val =
        solver.eval_f(n, snap.x_eval, snap.v_eval[j], grid_struct, phi_history);
    std::copy(val.begin(), val.end(), snap.f.begin() + j * Nx_out);
  }

  snap.rho.assign(Nx_out, 1.0);
  for (unsigned int j = 0; j < Nv_out; ++j)
    for (unsigned int i = 0; i < Nx_out; ++i)
      snap.rho[i] -= snap.f[j * Nx_out + i] * dv;

  std::vector<double> x_copy = snap.x_eval;
  auto grad_phi = eval(x_copy, grid_struct[phi_history[n].grid_version],
                       phi_history[n].solution);
  snap.E.resize(Nx_out);
  for (unsigned int i = 0; i < Nx_out; ++i)
    snap.E[i] = -grad_phi[i];

  return snap;
}

void save_f(const DiagnosticsSnapshot &snap, const std::string &filepath) {
  ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_f: failed to open " + filepath);

  file << "# nufi f(x,v) ascii dump\n";
  file << "# Nx = " << snap.Nx << " Nv = " << snap.Nv << "\n";
  file << "# columns: x v f\n";
  file << std::setprecision(10);
  for (unsigned int j = 0; j < snap.Nv; ++j)
    for (unsigned int i = 0; i < snap.Nx; ++i)
      file << snap.x_eval[i] << " " << snap.v_eval[j] << " "
           << snap.f[j * snap.Nx + i] << "\n";
}

void save_rho(const DiagnosticsSnapshot &snap, const std::string &filepath) {
  save_field_1d(snap.x_eval, snap.rho, filepath);
}

void save_Efield(const DiagnosticsSnapshot &snap, const std::string &filepath) {
  save_field_1d(snap.x_eval, snap.E, filepath);
}

double compute_int_E_squared(const std::vector<double> &E, double Lx) {
  const double dx = Lx / E.size();
  double integral = 0.0;
  for (double e : E)
    integral += e * e;
  return 0.5 * integral * dx;
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
