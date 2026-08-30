#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <deal.II/base/exceptions.h>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <string>

#include "lua_config.h"

using dealii::ExcMessage;

namespace Parameters {

// IF YOU CHANGE X_DIM or V_DIM: change them here AND in parameters.lua,
// then re-build. These must match at runtime (checked in load_lua_config).
inline constexpr unsigned int X_DIM = 3;
inline constexpr unsigned int V_DIM = 3;

inline std::array<double, X_DIM> X_DOMAIN_LEFT;
inline std::array<double, X_DIM> X_DOMAIN_RIGHT;
inline std::array<double, X_DIM> LX;
inline std::array<double, X_DIM> LX_INV;

inline std::array<double, V_DIM> V_DOMAIN_LEFT;
inline std::array<double, V_DIM> V_DOMAIN_RIGHT;

inline std::array<unsigned int, V_DIM> NV;
inline std::array<double, V_DIM> DV;

// f0_TYPE:
// 0 -> twos-stream
// 1 -> landau-damping
// 2 -> maxwellian
// 3 -> bump-on-tail
inline size_t f0_TYPE;

inline bool IONS_ENABLED;

// deal.ii options
inline unsigned int GLOBAL_REFINEMENT;
inline unsigned int FE_DEGREE;
inline unsigned int CONVERGENCE_ITERATIONS;
inline double CONVERGENCE_LIMIT;
inline unsigned int MAX_DOFS;

// Adaptive refinement options
inline unsigned int REFINE_FREQUENCY;
inline double REFINEMENT_TOP_FRACTION;
inline double REFINEMENT_BOTTOM_FRACTION;

inline double EPS;
inline double WAVE_NR;
inline double F0_FACTOR; // 1/sqrt(2pi)

// NUFI options
inline double DT;
inline unsigned int TMAX;

// Plotting options
inline int PLOT_FREQUENCY;
inline bool SAVE_INT_E_ALWAYS;
inline std::array<size_t, X_DIM> PLOT_NX;
inline std::array<double, X_DIM> PLOT_DX;
inline std::string PLOT_DIR;

inline double MASS_RATIO;
inline double ION_EPS;
inline double ION_V_DOMAIN_LEFT;
inline double ION_V_DOMAIN_RIGHT;
inline std::array<size_t, V_DIM> NV_ION;

inline size_t V_CHUNK_SIZE;
inline size_t V_CHUNK_MAX;

inline size_t read_available_memory_bytes() {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo)
    return 0;

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.rfind("MemAvailable:", 0) == 0) {
      std::istringstream iss(line.substr(13));
      size_t kb = 0;
      iss >> kb;
      return kb * 1024; // kb to b
    }
  }
  return 0; // MemAvailable not found (very old kernel)
}

inline size_t compute_auto_chunk_size() {
  constexpr size_t MIN_CHUNK = 256;
  const size_t MAX_CHUNK = V_CHUNK_MAX;

  const int n_threads = omp_get_max_threads();

  const size_t available_bytes = read_available_memory_bytes();

  constexpr size_t per_point_bytes =
      2 * X_DIM * sizeof(double) + 2 * V_DIM * sizeof(double);
  constexpr size_t safety_factor = 8;

  if (available_bytes == 0) {
    std::cout << "[Parameters] Could not read /proc/meminfo; "
              << "falling back to default V_CHUNK_SIZE = 4096\n";
    return 4096;
  }

  const size_t usable_bytes = available_bytes / 2;

  const size_t bytes_per_thread =
      usable_bytes / static_cast<size_t>(std::max(n_threads, 1));

  size_t chunk = bytes_per_thread / (per_point_bytes * safety_factor);

  chunk = std::max<size_t>(chunk, MIN_CHUNK);
  chunk = std::min<size_t>(chunk, MAX_CHUNK);

  std::cout << "[Parameters] Auto chunk size: threads=" << n_threads
            << " available_mem=" << (available_bytes / (1024 * 1024)) << "MB"
            << " -> V_CHUNK_SIZE=" << chunk << "\n";

  return static_cast<size_t>(chunk);
}

inline void load_lua_config(const std::string &luaFilePath) {
  LuaConfig config(luaFilePath, "parameters");

  const unsigned int lua_x_dim =
      static_cast<unsigned int>(config.get<int>("X_DIM"));
  AssertThrow(lua_x_dim == X_DIM,
              ExcMessage("parameters.lua X_DIM (" + std::to_string(lua_x_dim) +
                         ") does not match compiled Parameters::X_DIM (" +
                         std::to_string(X_DIM) + "). Rebuild required."));

  const unsigned int lua_v_dim =
      static_cast<unsigned int>(config.get<int>("V_DIM"));
  AssertThrow(lua_v_dim == V_DIM,
              ExcMessage("parameters.lua V_DIM (" + std::to_string(lua_v_dim) +
                         ") does not match compiled Parameters::V_DIM (" +
                         std::to_string(V_DIM) + "). Rebuild required."));

  {
    auto xl = config.getArray<double>("X_DOMAIN_LEFT", X_DIM);
    auto xr = config.getArray<double>("X_DOMAIN_RIGHT", X_DIM);
    auto lx = config.getArray<double>("LX", X_DIM);
    std::copy(xl.begin(), xl.end(), X_DOMAIN_LEFT.begin());
    std::copy(xr.begin(), xr.end(), X_DOMAIN_RIGHT.begin());
    std::copy(lx.begin(), lx.end(), LX.begin());
    for (unsigned int d = 0; d < X_DIM; ++d)
      LX_INV[d] = 1.0 / LX[d];
  }

  {
    auto vl = config.getArray<double>("V_DOMAIN_LEFT", V_DIM);
    auto vr = config.getArray<double>("V_DOMAIN_RIGHT", V_DIM);
    auto nv = config.getArray<int>("NV", V_DIM);
    std::copy(vl.begin(), vl.end(), V_DOMAIN_LEFT.begin());
    std::copy(vr.begin(), vr.end(), V_DOMAIN_RIGHT.begin());
    for (unsigned int d = 0; d < V_DIM; ++d) {
      NV[d] = static_cast<unsigned int>(nv[d]);
      DV[d] = std::abs(V_DOMAIN_RIGHT[d] - V_DOMAIN_LEFT[d]) /
              static_cast<double>(NV[d]);
    }
  }

  f0_TYPE = static_cast<size_t>(config.get<int>("f0_TYPE"));

  IONS_ENABLED = config.get<bool>("IONS_ENABLED");

  GLOBAL_REFINEMENT =
      static_cast<unsigned int>(config.get<int>("GLOBAL_REFINEMENT"));
  FE_DEGREE = static_cast<unsigned int>(config.get<int>("FE_DEGREE"));
  CONVERGENCE_ITERATIONS =
      static_cast<unsigned int>(config.get<int>("CONVERGENCE_ITERATIONS"));
  CONVERGENCE_LIMIT = config.get<double>("CONVERGENCE_LIMIT");
  MAX_DOFS = static_cast<unsigned int>(config.get<int>("MAX_DOFS"));

  REFINE_FREQUENCY =
      static_cast<unsigned int>(config.get<int>("REFINE_FREQUENCY"));
  REFINEMENT_TOP_FRACTION = config.get<double>("REFINEMENT_TOP_FRACTION");
  REFINEMENT_BOTTOM_FRACTION = config.get<double>("REFINEMENT_BOTTOM_FRACTION");

  EPS = config.get<double>("EPS");
  WAVE_NR = config.get<double>("WAVE_NR");
  F0_FACTOR = config.get<double>("F0_FACTOR");

  MASS_RATIO = config.get<double>("MASS_RATIO");
  ION_EPS = config.get<double>("ION_EPS");
  ION_V_DOMAIN_LEFT = config.get<double>("ION_V_DOMAIN_LEFT");
  ION_V_DOMAIN_RIGHT = config.get<double>("ION_V_DOMAIN_RIGHT");

  {
    auto nv_ion = config.getArray<int>("NV_ION", V_DIM);
    for (unsigned int d = 0; d < V_DIM; ++d)
      NV_ION[d] = static_cast<size_t>(nv_ion[d]);
  }

  DT = config.get<double>("DT");
  TMAX = static_cast<unsigned int>(config.get<int>("TMAX"));

  PLOT_FREQUENCY = config.get<int>("PLOT_FREQUENCY");
  SAVE_INT_E_ALWAYS = config.get<bool>("SAVE_INT_E_ALWAYS");

  {
    auto plot_nx = config.getArray<int>("PLOT_NX", X_DIM);
    for (unsigned int d = 0; d < X_DIM; ++d) {
      PLOT_NX[d] = static_cast<size_t>(plot_nx[d]);
      PLOT_DX[d] = LX[d] / static_cast<double>(PLOT_NX[d]);
    }
  }

  PLOT_DIR = config.get<std::string>("PLOT_DIR");

  V_CHUNK_MAX = static_cast<size_t>(config.get<int>("V_CHUNK_MAX"));
  V_CHUNK_SIZE = static_cast<size_t>(config.get<int>("V_CHUNK_SIZE"));
  if (V_CHUNK_SIZE == 0)
    V_CHUNK_SIZE = compute_auto_chunk_size();
}

} // namespace Parameters

#endif // PARAMETERS_H
