#ifndef NUFI_SOLVER_H
#define NUFI_SOLVER_H

#include <array>
#include <boost/qvm/mat_access.hpp>
#include <cmath>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>
#include <vector>

#include "nufi/fields.h" // dont remove
#include "nufi/grids.h"
#include "nufi/parameters.h"

using namespace dealii;
using std::array;

template <size_t X_DIM, size_t V_DIM> class NuFISolver {
  static_assert(
      X_DIM <= V_DIM,
      "NuFISolver assumes X_DIM <= V_DIM: E has X_DIM components and only "
      "pushes velocity components 0..X_DIM-1. Extra velocity components "
      "(e.g. v2 in 1x2v) are non-spatial and stay force-free until magnetic "
      "coupling (Boris pusher) is added.");

public:
  NuFISolver();

  std::vector<double>
  eval_rho(unsigned int n, const std::vector<array<double, X_DIM>> &X,
           const std::vector<GridStructure<X_DIM>> &grid_struct,
           const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
           const array<size_t, V_DIM> &Nv = Parameters::NV) const;

  std::vector<double>
  eval_ftilda_batch(unsigned int n, std::vector<array<double, X_DIM>> X,
                    std::vector<array<double, V_DIM>> U,
                    const std::vector<GridStructure<X_DIM>> &grid_structures,
                    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
                    bool is_electron = true) const;

  std::vector<double>
  eval_f_batch(unsigned int n, std::vector<array<double, X_DIM>> X,
               std::vector<array<double, V_DIM>> U,
               const std::vector<GridStructure<X_DIM>> &grid_structures,
               const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
               bool is_electron = true) const;

  std::vector<double>
  eval_f(unsigned int n, std::vector<array<double, X_DIM>> X,
         array<double, V_DIM> u,
         const std::vector<GridStructure<X_DIM>> &grid_struct,
         const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
         bool is_electron = true) const;

  std::vector<double>
  eval_rho_points(unsigned int n, const std::vector<Point<X_DIM>> &points,
                  const std::vector<GridStructure<X_DIM>> &grid_struct,
                  const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
                  const array<size_t, V_DIM> &Nv) const;

private:
  double
  eval_density_at_x(unsigned int n, const array<double, X_DIM> &x_point,
                    const std::vector<array<double, V_DIM>> &v_eval,
                    const std::vector<GridStructure<X_DIM>> &grid_struct,
                    const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
                    bool is_electron) const;

  std::vector<double>
  eval_species_density(unsigned int n,
                       const std::vector<array<double, X_DIM>> &X,
                       const std::vector<GridStructure<X_DIM>> &grid_struct,
                       const std::vector<SolutionSnapshot<X_DIM>> &phi_history,
                       const array<size_t, V_DIM> &Nv, bool is_electron) const;

  unsigned int Nt = std::floor(Parameters::TMAX / Parameters::DT);
};
#endif
