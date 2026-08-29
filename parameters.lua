--
-- IF YOU CHANGE X or V DIMS CHANGE THEM ON `nufi/parameters.h` AND THEN RE-BUILD
--
local X_DIM          = 3
local V_DIM          = 3

local X_DOMAIN_LEFT  = { 0.0, 0.0, 0.0 }
local X_DOMAIN_RIGHT = { 4 * math.pi, 4 * math.pi, 4 * math.pi }

local V_DOMAIN_LEFT  = { -10.0, -10.0, -10.0 }
local V_DOMAIN_RIGHT = { 10.0, 10.0, 10.0 }
local NV_BY_VDIM     = {
  [1] = { 512, 512, 512 },
  [2] = { 64, 64, 64 },
  [3] = { 8, 8, 8 },
}
local NV             = NV_BY_VDIM[V_DIM]


local LX                 =
{ math.abs(X_DOMAIN_RIGHT[1] - X_DOMAIN_LEFT[1]),
  math.abs(X_DOMAIN_RIGHT[2] - X_DOMAIN_LEFT[2]),
  math.abs(X_DOMAIN_RIGHT[3] - X_DOMAIN_LEFT[3]) }

local DV                 =
{ math.abs(V_DOMAIN_RIGHT[1] - V_DOMAIN_LEFT[1]) / NV[1],
  math.abs(V_DOMAIN_RIGHT[2] - V_DOMAIN_LEFT[2]) / NV[2],
  math.abs(V_DOMAIN_RIGHT[3] - V_DOMAIN_LEFT[3]) / NV[3] }

-- f0_TYPE:
-- 0 -> twos-stream
-- 1 -> landau-damping
-- 2 -> maxwellian
-- 3 -> bump-on-tail
local f0_TYPE            = 0

-- Ion options
local IONS_ENABLED       = false
local MASS_RATIO         = 1000
local ION_EPS            = 0.01
local ION_V_DOMAIN_LEFT  = -0.4
local ION_V_DOMAIN_RIGHT = 0.4
local NV_ION_BY_VDIM     = {
  [1] = { 128, 128, 128 },
  [2] = { 32, 32, 32 },
  [3] = { 8, 8, 8 },
}
local NV_ION             = NV_ION_BY_VDIM[V_DIM]


-- deal.ii options
local GLOBAL_REFINEMENT = 3
local FE_DEGREE = 3
local CONVERGENCE_ITERATIONS = 5000
local CONVERGENCE_LIMIT = 1e-7
local MAX_DOFS = 2000


-- Adaptive refinement options
local REFINE_FREQUENCY = 50
local REFINEMENT_TOP_FRACTION = .8
local REFINEMENT_BOTTOM_FRACTION = .1

local EPS = .01
local WAVE_NR = .5
local F0_FACTOR = 0.39894228040143267793994


-- NUFI options
local DT = 1. / 16.
local TMAX = 100

-- Plotting options
local PLOT_FREQUENCY = 160
local SAVE_INT_E_ALWAYS = true
local PLOT_NX_BY_XDIM = {
  [1] = { 512, 512, 512 },
  [2] = { 128, 128, 128 },
  [3] = { 32, 32, 32 },
}
local PLOT_NX = PLOT_NX_BY_XDIM[X_DIM]

local PLOT_DX =
{ LX[1] / PLOT_NX[1],
  LX[2] / PLOT_NX[2],
  LX[3] / PLOT_NX[3] }

parameters = {
  X_DIM = X_DIM,
  V_DIM = V_DIM,

  X_DOMAIN_LEFT = X_DOMAIN_LEFT,
  X_DOMAIN_RIGHT = X_DOMAIN_RIGHT,
  LX = LX,
  LX_INV = { 1.0 / LX[1], 1.0 / LX[2], 1.0 / LX[3] },

  V_DOMAIN_LEFT = V_DOMAIN_LEFT,
  V_DOMAIN_RIGHT = V_DOMAIN_RIGHT,

  NV = NV,
  DV = DV,

  f0_TYPE = f0_TYPE,

  IONS_ENABLED = IONS_ENABLED,

  -- deal.ii options
  GLOBAL_REFINEMENT = GLOBAL_REFINEMENT,
  FE_DEGREE = FE_DEGREE,
  CONVERGENCE_ITERATIONS = CONVERGENCE_ITERATIONS,
  CONVERGENCE_LIMIT = CONVERGENCE_LIMIT,
  MAX_DOFS = MAX_DOFS,

  -- Adaptive refinement options
  REFINE_FREQUENCY = REFINE_FREQUENCY,
  REFINEMENT_TOP_FRACTION = REFINEMENT_TOP_FRACTION,
  REFINEMENT_BOTTOM_FRACTION = REFINEMENT_BOTTOM_FRACTION,

  EPS = EPS,
  WAVE_NR = WAVE_NR,
  F0_FACTOR = F0_FACTOR, -- 1/sqrt(2pi)

  MASS_RATIO = MASS_RATIO,
  ION_EPS = ION_EPS,
  ION_V_DOMAIN_LEFT = ION_V_DOMAIN_LEFT,
  ION_V_DOMAIN_RIGHT = ION_V_DOMAIN_RIGHT,
  NV_ION = NV_ION,


  -- NUFI options
  DT = DT,
  TMAX = TMAX,

  -- Plotting options
  PLOT_FREQUENCY = PLOT_FREQUENCY,
  SAVE_INT_E_ALWAYS = SAVE_INT_E_ALWAYS,
  PLOT_NX = PLOT_NX,
  PLOT_DX = PLOT_DX,
  PLOT_DIR = "results/",
}
