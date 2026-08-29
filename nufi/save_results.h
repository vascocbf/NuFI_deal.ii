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
  std::array<size_t, X_DIM> Nx{};
  std::array<size_t, V_DIM> Nv{};
  std::vector<std::array<double, X_DIM>> x_eval;
  std::vector<std::array<double, V_DIM>> v_eval;
  std::vector<double> f;                    // flattened, v-major then x
  std::vector<double> rho;                  // flattened over x multi-index
  std::vector<std::array<double, X_DIM>> E; // -grad(phi), per x point
};

namespace detail {
inline void ensure_results_dir() {
  std::filesystem::create_directories("results");
}

// unravel a flat index into a multi-index over dims Nv[]
template <size_t DIM>
inline std::array<size_t, DIM> unravel_index(size_t index,
                                             const std::array<size_t, DIM> &N) {
  std::array<size_t, DIM> idx{};
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

  std::array<double, V_DIM> dv{};
  for (size_t d = 0; d < V_DIM; ++d)
    dv[d] = (Parameters::V_DOMAIN_RIGHT[d] - Parameters::V_DOMAIN_LEFT[d]) /
            static_cast<double>(Nv_out[d]);

  snap.v_eval.resize(n_v);
  for (size_t j = 0; j < n_v; ++j) {
    auto idx = detail::unravel_index<V_DIM>(j, Nv_out);
    for (size_t d = 0; d < V_DIM; ++d)
      snap.v_eval[j][d] = Parameters::V_DOMAIN_LEFT[d] + (idx[d] + 0.5) * dv[d];
  }

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
