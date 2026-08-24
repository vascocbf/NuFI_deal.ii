--
-- DIMENSION not changable here! you need to change it on `nufi/parameters.h` and re-build binary
--
local DIMENSION = 1

local X_DOMAIN_LEFT = 0.0
local X_DOMAIN_RIGHT = 4 * math.pi

local V_DOMAIN_LEFT = -10.0
local V_DOMAIN_RIGHT = 10.0

local NV = 256

-- f0_TYPE:
-- 0 -> twos-stream
-- 1 -> landau-damping
-- 2 -> maxwellian
-- 3 -> bump-on-tail
local f0_TYPE = 0


-- deal.ii options
local GLOBAL_REFINEMENT = 8
local FE_DEGREE = 3
local CONVERGENCE_ITERATIONS = 5000
local CONVERGENCE_LIMIT = 1e-7


-- Adaptive refinement options
local REFINE_FREQUENCY = 50
local REFINEMENT_TOP_FRACTION = .8
local REFINEMENT_BOTTOM_FRACTION = .1

local EPS = .01
local WAVE_NR = .5
local F0_FACTOR = 0.39894228040143267793994


-- NUFI options
local DT = 1. / 10.
local TMAX = 100

-- Plotting options
local PLOT_FREQUENCY = 20

local LX = math.abs(X_DOMAIN_RIGHT - X_DOMAIN_LEFT)
local PLOT_NX = 512

parameters = {
  DIMENSION = DIMENSION,

  X_DOMAIN_LEFT = X_DOMAIN_LEFT,
  X_DOMAIN_RIGHT = X_DOMAIN_RIGHT,
  LX = LX,
  LX_INV = 1.0 / LX,

  V_DOMAIN_LEFT = V_DOMAIN_LEFT,
  V_DOMAIN_RIGHT = V_DOMAIN_RIGHT,

  NV = NV,
  DV = math.abs(V_DOMAIN_RIGHT - V_DOMAIN_LEFT) / NV,

  f0_TYPE = f0_TYPE,

  -- deal.ii options
  GLOBAL_REFINEMENT = GLOBAL_REFINEMENT,
  FE_DEGREE = FE_DEGREE,
  CONVERGENCE_ITERATIONS = CONVERGENCE_ITERATIONS,
  CONVERGENCE_LIMIT = CONVERGENCE_LIMIT,

  -- Adaptive refinement options
  REFINE_FREQUENCY = REFINE_FREQUENCY,
  REFINEMENT_TOP_FRACTION = REFINEMENT_TOP_FRACTION,
  REFINEMENT_BOTTOM_FRACTION = REFINEMENT_BOTTOM_FRACTION,

  EPS = EPS,
  WAVE_NR = WAVE_NR,
  F0_FACTOR = F0_FACTOR, -- 1/sqrt(2pi)

  -- NUFI options
  DT = DT,
  TMAX = TMAX,

  -- Plotting options
  PLOT_FREQUENCY = PLOT_FREQUENCY,
  PLOT_NX = PLOT_NX,
  PLOT_DX = LX / PLOT_NX,
  PLOT_DIR = "results/",
}
