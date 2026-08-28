#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <deal.II/base/exceptions.h>
#include <string>

#include "lua_config.h"

namespace Parameters {

inline constexpr unsigned int DIMENSION = 1;

// IF YOU CHANGE X_DIM or V_DIM: change them here AND in parameters.lua,
// then re-build. These must match at runtime (checked in load_lua_config).
inline constexpr unsigned int X_DIM = 1;
inline constexpr unsigned int V_DIM = 1;

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
inline std::array<unsigned int, V_DIM> NV_ION;

inline void load_lua_config(const std::string &luaFilePath) {
  LuaConfig config(luaFilePath, "parameters");

  // --- dimension sanity check ---
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

  // --- spatial domain (X_DIM entries) ---
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

  // --- velocity domain (V_DIM entries) ---
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
      NV_ION[d] = static_cast<unsigned int>(nv_ion[d]);
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
}

} // namespace Parameters

#endif // PARAMETERS_H
