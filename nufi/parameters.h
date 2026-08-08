#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cstddef>
#include <string>

#include "lua_config.h"

namespace Parameters {

inline unsigned int DIMENSION;

inline double X_DOMAIN_LEFT;
inline double X_DOMAIN_RIGHT;
inline double LX;
inline double LX_INV;

inline double V_DOMAIN_LEFT;
inline double V_DOMAIN_RIGHT;

inline unsigned int NV;
inline double DV;

// f0_TYPE:
// 0 -> twos-stream
// 1 -> landau-damping
// 2 -> maxwellian
// 3 -> bump-on-tail
inline size_t f0_TYPE;

// deal.ii options
inline unsigned int GLOBAL_REFINEMENT;
inline unsigned int FE_DEGREE;
inline unsigned int CONVERGENCE_ITERATIONS;
inline double CONVERGENCE_LIMIT;

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
inline size_t PLOT_NX;
inline double PLOT_DX;
inline std::string PLOT_DIR;

inline void load_lua_config(const std::string &luaFilePath) {
  LuaConfig config(luaFilePath, "parameters");

  DIMENSION = static_cast<unsigned int>(config.get<int>("DIMENSION"));

  X_DOMAIN_LEFT = config.get<double>("X_DOMAIN_LEFT");
  X_DOMAIN_RIGHT = config.get<double>("X_DOMAIN_RIGHT");
  LX = config.get<double>("LX");
  LX_INV = config.get<double>("LX_INV");

  V_DOMAIN_LEFT = config.get<double>("V_DOMAIN_LEFT");
  V_DOMAIN_RIGHT = config.get<double>("V_DOMAIN_RIGHT");

  NV = static_cast<unsigned int>(config.get<int>("NV"));
  DV = config.get<double>("DV");

  f0_TYPE = static_cast<size_t>(config.get<int>("f0_TYPE"));

  GLOBAL_REFINEMENT =
      static_cast<unsigned int>(config.get<int>("GLOBAL_REFINEMENT"));
  FE_DEGREE = static_cast<unsigned int>(config.get<int>("FE_DEGREE"));
  CONVERGENCE_ITERATIONS =
      static_cast<unsigned int>(config.get<int>("CONVERGENCE_ITERATIONS"));
  CONVERGENCE_LIMIT = config.get<double>("CONVERGENCE_LIMIT");

  REFINE_FREQUENCY =
      static_cast<unsigned int>(config.get<int>("REFINE_FREQUENCY"));
  REFINEMENT_TOP_FRACTION = config.get<double>("REFINEMENT_TOP_FRACTION");
  REFINEMENT_BOTTOM_FRACTION = config.get<double>("REFINEMENT_BOTTOM_FRACTION");

  EPS = config.get<double>("EPS");
  WAVE_NR = config.get<double>("WAVE_NR");
  F0_FACTOR = config.get<double>("F0_FACTOR");

  DT = config.get<double>("DT");
  TMAX = static_cast<unsigned int>(config.get<int>("TMAX"));

  PLOT_FREQUENCY = config.get<int>("PLOT_FREQUENCY");
  PLOT_NX = static_cast<size_t>(config.get<int>("PLOT_NX"));
  PLOT_DX = config.get<double>("PLOT_DX");
  PLOT_DIR = config.get<std::string>("PLOT_DIR");
}

} // namespace Parameters

#endif // PARAMETERS_H
