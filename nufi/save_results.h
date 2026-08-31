#ifndef SAVE_RESULTS_H
#define SAVE_RESULTS_H

#include "nufi/fields.h"
#include "nufi/grids.h"
#include "nufi/nufi_solver.h"
#include "nufi/parameters.h"

#include <array>
#include <cstddef>
#include <deal.II/lac/vector.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

template <size_t X_DIM, size_t V_DIM> struct DiagnosticsSnapshot {
  std::array<size_t, X_DIM> Nx = {};
  std::array<size_t, V_DIM> Nv = {};
  std::vector<std::array<double, X_DIM>> x_eval;
  std::vector<std::array<double, V_DIM>> v_eval;
  std::vector<double> f;
  std::vector<double> rho;
  std::vector<std::array<double, X_DIM>> E;
};

template <size_t X_DIM, size_t V_DIM> struct DiagnosticsSlice {
  size_t free_x_dim = 0;
  size_t free_v_dim = 0;
  std::array<double, X_DIM> x_fixed = {};
  std::array<double, V_DIM> v_fixed = {};
  std::vector<std::array<double, X_DIM>> x_eval;
  std::vector<std::array<double, V_DIM>> v_eval;
  std::vector<double> f;
  std::vector<double> rho;
  std::vector<std::array<double, X_DIM>> E;
};

namespace detail {
inline void ensure_results_dir() {
  std::filesystem::create_directories("results");
}

// unravel a flat index into a multi-index over dims Nv[]
template <size_t DIM>
inline std::array<size_t, DIM> unravel_index(size_t index,
                                             const std::array<size_t, DIM> &N) {
  std::array<size_t, DIM> idx = {};
  for (size_t d = 0; d < DIM; ++d) {
    idx[d] = index % N[d];
    index /= N[d];
  }
  return idx;
}
} // namespace detail

template <size_t X_DIM, size_t V_DIM>
DiagnosticsSnapshot<X_DIM, V_DIM>
compute_diagnostics(const NuFISolver<X_DIM, V_DIM> &solver, unsigned int n,
                    std::vector<GridStructure<X_DIM>> &grid_struct,
                    std::vector<SolutionSnapshot<X_DIM>> &phi_history,
                    const std::array<size_t, X_DIM> &Nx_out,
                    const std::array<size_t, V_DIM> &Nv_out) {
  DiagnosticsSnapshot<X_DIM, V_DIM> snap;
  snap.Nx = Nx_out;
  snap.Nv = Nv_out;

  size_t n_x = 1;
  for (size_t d = 0; d < X_DIM; ++d)
    n_x *= Nx_out[d];

  size_t n_v = 1;
  for (size_t d = 0; d < V_DIM; ++d)
    n_v *= Nv_out[d];

  snap.x_eval = make_x_eval<X_DIM>(Nx_out);

  std::array<double, V_DIM> dv = make_dv(Nv_out);

  snap.v_eval = make_v_eval(Nv_out);

  snap.f.resize(n_x * n_v);
#pragma omp parallel for
  for (size_t j = 0; j < n_v; ++j) {
    std::vector<double> val =
        solver.eval_f(n, snap.x_eval, snap.v_eval[j], grid_struct, phi_history);
    std::copy(val.begin(), val.end(), snap.f.begin() + j * n_x);
  }

  double v_cell_volume = 1.0;
  for (size_t d = 0; d < V_DIM; ++d)
    v_cell_volume *= dv[d];

  snap.rho.assign(n_x, 1.0);
  for (size_t j = 0; j < n_v; ++j)
    for (size_t i = 0; i < n_x; ++i)
      snap.rho[i] -= snap.f[j * n_x + i] * v_cell_volume;

  auto x_copy = snap.x_eval;
  snap.E = eval<X_DIM>(x_copy, grid_struct[phi_history[n].grid_version],
                       phi_history[n].solution);
  for (auto &e : snap.E)
    for (size_t d = 0; d < X_DIM; ++d)
      e[d] = -e[d];

  return snap;
}

template <size_t X_DIM, size_t V_DIM>
DiagnosticsSlice<X_DIM, V_DIM> compute_diagnostics_slice(
    const NuFISolver<X_DIM, V_DIM> &solver, unsigned int n,
    std::vector<GridStructure<X_DIM>> &grid_struct,
    std::vector<SolutionSnapshot<X_DIM>> &phi_history, size_t free_x_dim,
    size_t free_v_dim, const std::array<double, X_DIM> &x_fixed,
    const std::array<double, V_DIM> &v_fixed, size_t Nx_free, size_t Nv_free) {

  AssertThrow(
      free_x_dim < X_DIM,
      ExcMessage("[compute_diagnostics_slice] free_x_dim out of range"));
  AssertThrow(
      free_v_dim < V_DIM,
      ExcMessage("[compute_diagnostics_slice] free_v_dim out of range"));

  DiagnosticsSlice<X_DIM, V_DIM> slice;
  slice.free_x_dim = free_x_dim;
  slice.free_v_dim = free_v_dim;
  slice.x_fixed = x_fixed;
  slice.v_fixed = v_fixed;

  slice.x_eval = make_x_eval_slice<X_DIM>(free_x_dim, x_fixed, Nx_free);
  slice.v_eval = make_v_eval_slice<V_DIM>(free_v_dim, v_fixed, Nv_free);

  slice.f.resize(Nx_free * Nv_free);
#pragma omp parallel for
  for (size_t j = 0; j < Nv_free; ++j) {
    std::vector<double> val = solver.eval_f(n, slice.x_eval, slice.v_eval[j],
                                            grid_struct, phi_history);
    std::copy(val.begin(), val.end(), slice.f.begin() + j * Nx_free);
  }

  array<size_t, V_DIM> Nv = {};
  for (size_t d = 0; d < V_DIM; ++d)
    Nv[d] = static_cast<size_t>(Parameters::NV[d]);

  slice.rho = solver.eval_rho(n, slice.x_eval, grid_struct, phi_history, Nv);

  auto x_copy = slice.x_eval;
  slice.E = eval<X_DIM>(x_copy, grid_struct[phi_history[n].grid_version],
                        phi_history[n].solution);
  for (auto &e : slice.E)
    for (size_t d = 0; d < X_DIM; ++d)
      e[d] = -e[d];

  return slice;
}

template <size_t X_DIM, size_t V_DIM>
void save_f(const DiagnosticsSnapshot<X_DIM, V_DIM> &snap,
            const std::string &filepath) {
  detail::ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_f: failed to open " + filepath);

  size_t n_x = snap.x_eval.size();
  size_t n_v = snap.v_eval.size();

  file << "# f(x,v)\n";
  file << "# X_DIM = " << X_DIM << " V_DIM = " << V_DIM << "\n";
  file << "# columns: x0..x" << (X_DIM - 1) << " v0..v" << (V_DIM - 1)
       << " f\n";
  file << std::setprecision(10);

  for (size_t j = 0; j < n_v; ++j) {
    for (size_t i = 0; i < n_x; ++i) {
      for (size_t d = 0; d < X_DIM; ++d)
        file << snap.x_eval[i][d] << " ";
      for (size_t d = 0; d < V_DIM; ++d)
        file << snap.v_eval[j][d] << " ";
      file << snap.f[j * n_x + i] << "\n";
    }
  }
}

template <size_t X_DIM, size_t V_DIM>
void save_f_slice(const DiagnosticsSlice<X_DIM, V_DIM> &slice,
                  const std::string &filepath) {
  detail::ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_f_slice: failed to open " + filepath);

  const size_t n_x = slice.x_eval.size();
  const size_t n_v = slice.v_eval.size();

  file << "# f slice on plane (x" << slice.free_x_dim << ", v"
       << slice.free_v_dim << ")\n";
  file << "# fixed x:";
  for (size_t d = 0; d < X_DIM; ++d)
    if (d != slice.free_x_dim)
      file << " x" << d << "=" << slice.x_fixed[d];
  file << "\n# fixed v:";
  for (size_t d = 0; d < V_DIM; ++d)
    if (d != slice.free_v_dim)
      file << " v" << d << "=" << slice.v_fixed[d];
  file << "\n# columns: x_free v_free f\n";
  file << std::setprecision(10);

  for (size_t j = 0; j < n_v; ++j) {
    for (size_t i = 0; i < n_x; ++i) {
      file << slice.x_eval[i][slice.free_x_dim] << " "
           << slice.v_eval[j][slice.free_v_dim] << " " << slice.f[j * n_x + i]
           << "\n";
    }
  }
}

template <size_t X_DIM, size_t V_DIM>
void save_rho(const DiagnosticsSnapshot<X_DIM, V_DIM> &snap,
              const std::string &filepath) {
  detail::ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_rho: failed to open " + filepath);

  file << "# rho(x)\n";
  file << "# X_DIM = " << X_DIM << "\n";
  file << "# columns: x0.." << (X_DIM - 1) << " rho\n";
  file << std::setprecision(10);

  for (size_t i = 0; i < snap.x_eval.size(); ++i) {
    for (size_t d = 0; d < X_DIM; ++d)
      file << snap.x_eval[i][d] << " ";
    file << snap.rho[i] << "\n";
  }
}

template <size_t X_DIM, size_t V_DIM>
void save_rho_slice(const DiagnosticsSlice<X_DIM, V_DIM> &slice,
                    const std::string &filepath) {
  detail::ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_rho_slice: failed to open " + filepath);

  file << "# rho slice along x" << slice.free_x_dim << "\n";
  file << "# fixed x:";
  for (size_t d = 0; d < X_DIM; ++d)
    if (d != slice.free_x_dim)
      file << " x" << d << "=" << slice.x_fixed[d];
  file << "\n# columns: x_free rho\n";
  file << std::setprecision(10);

  for (size_t i = 0; i < slice.x_eval.size(); ++i)
    file << slice.x_eval[i][slice.free_x_dim] << " " << slice.rho[i] << "\n";
}

template <size_t X_DIM, size_t V_DIM>
void save_Efield(const DiagnosticsSnapshot<X_DIM, V_DIM> &snap,
                 const std::string &filepath) {
  detail::ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_Efield: failed to open " + filepath);

  file << "# E(x)\n";
  file << "# X_DIM = " << X_DIM << "\n";
  file << "# columns: x0.." << (X_DIM - 1) << " E0.." << (X_DIM - 1) << "\n";
  file << std::setprecision(10);

  for (size_t i = 0; i < snap.x_eval.size(); ++i) {
    for (size_t d = 0; d < X_DIM; ++d)
      file << snap.x_eval[i][d] << " ";
    for (size_t d = 0; d < X_DIM; ++d)
      file << snap.E[i][d] << (d + 1 < X_DIM ? " " : "\n");
  }
}

template <size_t X_DIM, size_t V_DIM>
void save_Efield_slice(const DiagnosticsSlice<X_DIM, V_DIM> &slice,
                       const std::string &filepath) {
  detail::ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_Efield_slice: failed to open " + filepath);

  file << "# E slice along x" << slice.free_x_dim << "\n";
  file << "# fixed x:";
  for (size_t d = 0; d < X_DIM; ++d)
    if (d != slice.free_x_dim)
      file << " x" << d << "=" << slice.x_fixed[d];
  file << "\n# columns: x_free E0.." << (X_DIM - 1) << "\n";
  file << std::setprecision(10);

  for (size_t i = 0; i < slice.x_eval.size(); ++i) {
    file << slice.x_eval[i][slice.free_x_dim] << " ";
    for (size_t d = 0; d < X_DIM; ++d)
      file << slice.E[i][d] << (d + 1 < X_DIM ? " " : "\n");
  }
}

template <size_t X_DIM, size_t V_DIM>
double compute_int_E_squared(const DiagnosticsSnapshot<X_DIM, V_DIM> &snap) {
  double volume_element = 1.0;
  for (size_t d = 0; d < X_DIM; ++d)
    volume_element *= Parameters::LX[d] / static_cast<double>(snap.Nx[d]);

  double integral = 0.0;
  for (const auto &E : snap.E)
    for (size_t d = 0; d < X_DIM; ++d)
      integral += E[d] * E[d];

  return 0.5 * integral * volume_element;
}

inline void save_time_series(const std::vector<double> &t,
                             const std::vector<double> &values,
                             const std::string &filepath) {
  if (t.size() != values.size())
    throw std::runtime_error(
        "save_time_series: size mismatch (t=" + std::to_string(t.size()) +
        ", values=" + std::to_string(values.size()) + ") writing " + filepath);

  detail::ensure_results_dir();
  std::ofstream file(filepath);
  if (!file)
    throw std::runtime_error("save_time_series: failed to open " + filepath);

  file << "# time series\n";
  file << "# columns: t value\n";
  file << std::setprecision(10);
  for (size_t i = 0; i < t.size(); ++i)
    file << t[i] << " " << values[i] << "\n";
}

#endif
